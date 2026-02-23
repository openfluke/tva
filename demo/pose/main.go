package main

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/openfluke/loom/gpu"
	"github.com/openfluke/loom/nn"
)

// ─────────────────────────────────────────────
// Model catalog
// ─────────────────────────────────────────────

type PoseModel struct {
	Name    string
	Repo    string
	File    string
	LocalID string
	License string
}

var poseModels = []PoseModel{
	{
		Name:    "ViTPose Base Simple",
		Repo:    "usyd-community/vitpose-base-simple",
		File:    "model.safetensors",
		LocalID: "vitpose-base-simple",
		License: "Apache-2.0",
	},
}

const (
	imagesDir = "images"
	modelsDir = "models"
	inputH    = 256
	inputW    = 192
)

func main() {
	fmt.Println("🦴 Loom Pose Demo")
	fmt.Println("══════════════════════════════════════════")

	imageFiles := checkImages(imagesDir)
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		log.Fatalf("Cannot create models dir: %v", err)
	}
	localPaths := ensureModels(poseModels)

	if len(imageFiles) > 0 {
		m := poseModels[0]
		path, ok := localPaths[m.LocalID]
		if ok {
			fmt.Printf("\n🚀 Inference: %s -> %s\n", m.Name, imageFiles[0])
			runInference(m.Name, path, imageFiles)
		}
	}
}

// ─────────────────────────────────────────────
// Inference Engine
// ─────────────────────────────────────────────

func runInference(modelName, modelPath string, imageFiles []string) {
	data, err := os.ReadFile(modelPath)
	if err != nil {
		return
	}

	tensors, _ := nn.LoadSafetensorsWithShapes(data)

	// VitPose Base Simple config
	const dModel = 768
	net, _, err := nn.LoadGenericFromBytes(data, []byte(`{"patch_size": 16, "hidden_size": 768}`))
	if err != nil {
		fmt.Println("Err:", err)
		return
	}

	// Fix CPU dimensions
	net.InputSize = inputH * inputW * 3
	net.BatchSize = 1
	fmt.Printf("     🔧 Net: %dx%dx%d layers=%d batch=%d\n", net.GridRows, net.GridCols, net.LayersPerCell, len(net.Layers), net.BatchSize)
	for i := range net.Layers {
		l := &net.Layers[i]
		l.SeqLength = 192 // 16x12 tokens
		if l.Type == nn.LayerConv2D {
			if i == 0 {
				l.InputHeight = inputH
				l.InputWidth = inputW
				l.InputChannels = 3
				// PatchEmbed: 256/16=16, 192/16=12
				l.OutputHeight = 16
				l.OutputWidth = 12
			} else {
				// Likely a head conv or similar
				l.InputHeight = 16
				l.InputWidth = 12
				l.OutputHeight = 16
				l.OutputWidth = 12
			}
			fmt.Printf("        [%d] Conv2D: %dx%d -> %dx%d filters=%d\n", i, l.InputHeight, l.InputWidth, l.OutputHeight, l.OutputWidth, l.Filters)
		}
	}

	net.GPU = true
	gpu.SetAdapterPreference("nvidia")
	// gpu.Debug = true
	if net.GPU {
		if err := safeWeightsToGPU(net); err != nil {
			fmt.Printf("     ⚠️ GPU Error: %v, falling back to CPU\n", err)
			net.GPU = false
		}
	}

	for _, imgName := range imageFiles {
		imgPath := filepath.Join(imagesDir, imgName)
		input, origW, origH, err := preprocessImage(imgPath, inputW, inputH)
		if err != nil {
			continue
		}

		// Backbone forward
		start := time.Now()
		output, _ := net.Forward(input)
		fmt.Printf("     🧠 Backbone: %v\n", time.Since(start))

		if len(output) == 0 {
			fmt.Println("     ⚠️  Zero output from network")
			continue
		}

		// Check for NaN
		nanCount := 0
		for _, v := range output {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				nanCount++
			}
		}
		if nanCount > 0 {
			fmt.Printf("     ⚠️  Network produced %d NaNs/Infs!\n", nanCount)
		} else {
			minO, maxO := float32(1e30), float32(-1e30)
			for _, v := range output {
				if v < minO {
					minO = v
				}
				if v > maxO {
					maxO = v
				}
			}
			fmt.Printf("     📊 Heatmap Range: [%.3f, %.3f] size=%d\n", minO, maxO, len(output))
		}

		// Use output directly if it's the expected heatmap size
		var heatmaps []float32
		if len(output) == 17*16*12 {
			heatmaps = output
		} else if len(output) == 192*dModel {
			patches := transposePatches(output, 16, 12, dModel)
			heatmaps = applyConv3x3(patches, tensors["head.conv.weight"].Values, tensors["head.conv.bias"].Values, 16, 12, 768, 17)
		} else {
			fmt.Printf("     ⚠️  Unexpected output size: %d\n", len(output))
			continue
		}

		// Decode
		keypoints := decodeHeatmaps(heatmaps, 16, 12, 17)
		fmt.Printf("     ✨ Detected keypoints:\n")
		printKeypoints(keypoints, origW, origH)
	}
}

// ─────────────────────────────────────────────
// Manual Head Math
// ─────────────────────────────────────────────

func transposePatches(in []float32, h, w, c int) []float32 {
	out := make([]float32, c*h*w)
	for i := 0; i < h*w; i++ {
		for j := 0; j < c; j++ {
			out[j*h*w+i] = in[i*c+j]
		}
	}
	return out
}

func applyConv3x3(in []float32, weights, bias []float32, h, w, iC, oC int) []float32 {
	out := make([]float32, oC*h*w)
	kSize := 3
	pad := 1
	for oc := 0; oc < oC; oc++ {
		b := bias[oc]
		for oh := 0; oh < h; oh++ {
			for ow := 0; ow < w; ow++ {
				sum := b
				for ic := 0; ic < iC; ic++ {
					for kh := 0; kh < kSize; kh++ {
						for kw := 0; kw < kSize; kw++ {
							ih := oh + kh - pad
							iw := ow + kw - pad
							if ih >= 0 && ih < h && iw >= 0 && iw < w {
								wIdx := oc*iC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
								sum += in[ic*h*w+ih*w+iw] * weights[wIdx]
							}
						}
					}
				}
				out[oc*h*w+oh*w+ow] = sum
			}
		}
	}
	return out
}

func decodeHeatmaps(heatmaps []float32, h, w, c int) []Keypoint {
	kps := make([]Keypoint, c)
	for i := 0; i < c; i++ {
		maxV := float32(-1e30)
		maxIdx := 0
		found := false
		for j := 0; j < h*w; j++ {
			v := heatmaps[i*h*w+j]
			if v > maxV {
				maxV = v
				maxIdx = j
				found = true
			}
		}
		if !found {
			maxV = 0
		} // Handle all NaN
		kps[i] = Keypoint{
			ID:    i,
			X:     (float32(maxIdx%w) + 0.5) / float32(w),
			Y:     (float32(maxIdx/w) + 0.5) / float32(h),
			Score: maxV,
		}
	}
	return kps
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

func preprocessImage(path string, tw, th int) ([]float32, int, int, error) {
	f, _ := os.Open(path)
	defer f.Close()
	img, _, _ := image.Decode(f)
	bounds := img.Bounds()
	origW, origH := bounds.Dx(), bounds.Dy()
	data := make([]float32, tw*th*3)
	for y := 0; y < th; y++ {
		for x := 0; x < tw; x++ {
			sx, sy := x*origW/tw, y*origH/th
			r, g, b, _ := img.At(sx, sy).RGBA()
			mean, std := []float32{0.485, 0.456, 0.406}, []float32{0.229, 0.224, 0.225}
			data[(0*th+y)*tw+x] = (float32(r>>8)/255.0 - mean[0]) / std[0]
			data[(1*th+y)*tw+x] = (float32(g>>8)/255.0 - mean[1]) / std[1]
			data[(2*th+y)*tw+x] = (float32(b>>8)/255.0 - mean[2]) / std[2]
		}
	}
	return data, origW, origH, nil
}

type Keypoint struct {
	ID          int
	X, Y, Score float32
}

func printKeypoints(kps []Keypoint, w, h int) {
	names := []string{"Nose", "L-Eye", "R-Eye", "L-Ear", "R-Ear", "L-Shldr", "R-Shldr", "L-Elb", "R-Elb", "L-Wri", "R-Wri", "L-Hip", "R-Hip", "L-Knee", "R-Knee", "L-Ank", "R-Ank"}
	for i, kp := range kps {
		if i >= len(names) {
			break
		}
		fmt.Printf("        %-15s  %-10.1f  %-10.1f  %.2f\n", names[i], kp.X*float32(w), kp.Y*float32(h), kp.Score)
	}
}

func checkImages(dir string) []string {
	entries, _ := os.ReadDir(dir)
	var files []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(strings.ToLower(e.Name()), ".jpg") {
			files = append(files, e.Name())
		}
	}
	return files
}

func ensureModels(models []PoseModel) map[string]string {
	paths := make(map[string]string)
	for _, m := range models {
		local := filepath.Join(modelsDir, m.LocalID+".safetensors")
		if _, err := os.Stat(local); err == nil {
			paths[m.LocalID] = local
			continue
		}
		url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", m.Repo, m.File)
		resp, _ := http.Get(url)
		defer resp.Body.Close()
		out, _ := os.Create(local)
		defer out.Close()
		io.Copy(out, resp.Body)
		paths[m.LocalID] = local
	}
	return paths
}

func safeWeightsToGPU(net *nn.Network) (gpuErr error) {
	defer func() {
		if r := recover(); r != nil {
			gpuErr = fmt.Errorf("GPU panic: %v", r)
		}
	}()
	return net.WeightsToGPU()
}
