package main

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/openfluke/loom/nn"
)

func main() {
	path := "models/vitpose-plus-small.safetensors"
	if len(os.Args) > 1 {
		path = os.Args[1]
	}

	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read: %v\n", err)
		os.Exit(1)
	}

	tensors, err := nn.LoadSafetensorsWithShapes(data)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}

	names := make([]string, 0, len(tensors))
	for k := range tensors {
		names = append(names, k)
	}
	sort.Strings(names)

	// Print all tensors grouped by top-level prefix
	var lastPfx string
	for _, name := range names {
		t := tensors[name]
		pfx := strings.SplitN(name, ".", 2)[0]
		if pfx != lastPfx {
			fmt.Printf("\n── %s ──\n", pfx)
			lastPfx = pfx
		}
		shape := make([]string, len(t.Shape))
		for i, v := range t.Shape {
			shape[i] = fmt.Sprintf("%d", v)
		}
		params := 1
		for _, v := range t.Shape {
			params *= v
		}
		fmt.Printf("  %-70s  [%s]  (%d params)\n", name, strings.Join(shape, "×"), params)
	}
	fmt.Printf("\nTotal tensors: %d\n", len(names))
}
