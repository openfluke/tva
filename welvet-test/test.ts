/**
 * test.ts
 * Combined Loom WASM C-ABI Diagnostic and Benchmark
 */

import { init } from "@openfluke/welvet";

const EXPECTED_SYMBOLS = [
  'createLoomNetwork', 'loadLoomNetwork',
  'compareLoomDNA',
  'getDefaultTargetPropConfig', 'defaultSpliceConfig', 'defaultNEATConfig',
  'createLoomNEATPopulation',
  'setupWebGPU',
];

const EXPECTED_NET_METHODS = [
  'sequentialForward', 'extractDNA', 'extractBlueprint', 'getLayerCount',
  'getLayerSpec', 'morphLayer', 'spliceDNA', 'neatMutate',
  'createSystolicState', 'createTargetPropState', 'initGPU', 'syncToGPU',
  'syncToCPU', 'train', 'free', '_id',
];

const EXPECTED_POP_METHODS = [
  '_id', 'size', 'getNetwork', 'evolveWithFitnesses',
  'best', 'bestFitness', 'summary', 'free',
];

const DENSE_3L = JSON.stringify({
  depth: 3, rows: 1, cols: 1, layers_per_cell: 1,
  layers: [
    { z: 0, y: 0, x: 0, l: 0, type: "Dense", input_height: 16, output_height: 16, activation: "ReLU", dtype: "F32" },
    { z: 1, y: 0, x: 0, l: 0, type: "Dense", input_height: 16, output_height: 16, activation: "ReLU", dtype: "F32" },
    { z: 2, y: 0, x: 0, l: 0, type: "Dense", input_height: 16, output_height: 4, activation: "Linear", dtype: "F32" },
  ]
});

const SWIGLU_NET = JSON.stringify({
  depth: 2, rows: 1, cols: 1, layers_per_cell: 1,
  layers: [
    { z: 0, y: 0, x: 0, l: 0, type: "SwiGLU", input_height: 16, output_height: 32, dtype: "F32" },
    { z: 1, y: 0, x: 0, l: 0, type: "Dense", input_height: 32, output_height: 4, activation: "Linear", dtype: "F32" },
  ]
});

export async function runVerify() {
  console.log("=== Loom WASM C-ABI Diagnostic Report ===");
  
  let totalPass = 0;
  let totalFail = 0;

  // 1. Global symbol check
  console.log("\n[1] Checking global WASM exports...");
  for (const sym of EXPECTED_SYMBOLS) {
    // @ts-ignore
    if (typeof globalThis[sym] === 'function') {
      console.log(`  [PASS] ${sym}`);
      totalPass++;
    } else {
      console.error(`  [FAIL] ${sym} (missing)`);
      totalFail++;
    }
  }

  // 2. Network method check
  console.log("\n[2] Checking network wrapper methods...");
  let net: any = null;
  try {
    // @ts-ignore
    net = globalThis.createLoomNetwork(DENSE_3L);
    if (net) {
      for (const m of EXPECTED_NET_METHODS) {
        if (net[m] !== undefined) {
          console.log(`  [PASS] ${m}`);
          totalPass++;
        } else {
          console.error(`  [FAIL] ${m} (missing)`);
          totalFail++;
        }
      }
    }
  } catch (e) {
    console.error("  [FAIL] createLoomNetwork failed:", e);
    totalFail++;
  }

  // 3. Population method check
  if (net) {
    console.log("\n[3] Checking NEAT population wrapper methods...");
    try {
      // @ts-ignore
      const cfg = globalThis.defaultNEATConfig(16);
      // @ts-ignore
      const pop = globalThis.createLoomNEATPopulation(net._id, 4, cfg);
      if (pop) {
        for (const m of EXPECTED_POP_METHODS) {
          if ((pop as any)[m] !== undefined) {
            console.log(`  [PASS] ${m}`);
            totalPass++;
          } else {
            console.error(`  [FAIL] ${m} (missing)`);
            totalFail++;
          }
        }
        pop.free();
      }
    } catch (e) {
      console.error("  [FAIL] Population creation failed:", e);
      totalFail++;
    }
  }

  // 4. Functional smoke tests
  console.log("\n[4] Running functional smoke tests...");

  const smokeTest = (name: string, fn: () => any) => {
    try {
      const result = fn();
      console.log(`  [PASS] ${name}${result ? " → " + result : ""}`);
      totalPass++;
    } catch (e: any) {
      console.error(`  [FAIL] ${name} → ${e.message}`);
      totalFail++;
    }
  };

  smokeTest("sequentialForward", () => {
    // @ts-ignore
    const n = globalThis.createLoomNetwork(DENSE_3L);
    const input = new Float32Array(16).fill(0.5);
    const out = n.sequentialForward(input);
    n.free();
    if (!out || out.length === 0) throw new Error("empty output");
    return "out[0]=" + out[0].toFixed(4);
  });

  smokeTest("extractDNA", () => {
    // @ts-ignore
    const n = globalThis.createLoomNetwork(DENSE_3L);
    const dna = n.extractDNA();
    n.free();
    const parsed = JSON.parse(dna);
    return "sigs=" + parsed.length;
  });

  smokeTest("compareLoomDNA", () => {
    // @ts-ignore
    const n1 = globalThis.createLoomNetwork(DENSE_3L);
    // @ts-ignore
    const n2 = globalThis.createLoomNetwork(DENSE_3L);
    const dna1 = n1.extractDNA();
    const dna2 = n2.extractDNA();
    // @ts-ignore
    const result = JSON.parse(globalThis.compareLoomDNA(dna1, dna2));
    n1.free(); n2.free();
    return "overall_overlap=" + (result.overall_overlap || result.OverallOverlap || "?");
  });

  smokeTest("createLoomNetwork (SwiGLU)", () => {
    // @ts-ignore
    const n = globalThis.createLoomNetwork(SWIGLU_NET);
    const c = n.getLayerCount();
    n.free();
    return "layers=" + c;
  });

  console.log("\n[5] Final Summary");
  console.log("================================");
  if (totalFail === 0) {
    console.log(`SUCCESS: All ${totalPass} checks passed.`);
  } else {
    console.warn(`PARTIAL: ${totalPass} passed, ${totalFail} FAILED.`);
    process.exit(1);
  }
  console.log("================================\n");

  if (net) net.free();
}

/**
 * Benchmark logic
 */

const TRAINING_CASES = [
  {
    name: 'Dense (Linear)', iters: 5, inDim: 512, outDim: 512,
    cfg: JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1, layers: [
      { z: 0, y: 0, x: 0, l: 0, type: "Dense", input_height: 512, output_height: 512, activation: "Linear", dtype: "F32" }
    ]})
  },
  {
    name: 'RMSNorm', iters: 5, inDim: 512, outDim: 512,
    cfg: JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1, layers: [
      { z: 0, y: 0, x: 0, l: 0, type: "RMSNorm", input_height: 512, output_height: 512, dtype: "F32" }
    ]})
  },
  {
    name: 'SwiGLU (MLP)', iters: 5, inDim: 512, outDim: 1024,
    cfg: JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1, layers: [
      { z: 0, y: 0, x: 0, l: 0, type: "SwiGLU", input_height: 512, output_height: 1024, dtype: "F32" }
    ]})
  },
  {
    name: 'Embedding', iters: 5, inDim: 16, outDim: 2048, isEmbedding: true,
    cfg: JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1, layers: [
      { z: 0, y: 0, x: 0, l: 0, type: "Embedding", vocab_size: 1024, embedding_dim: 128, dtype: "F32" }
    ]})
  },
  {
    name: 'Residual Add', iters: 5, inDim: 512, outDim: 512,
    cfg: JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1, layers: [
      { z: 0, y: 0, x: 0, l: 0, type: "Residual", input_height: 512, output_height: 512, dtype: "F32" }
    ]})
  },
  {
    name: 'MHA (Fused)', iters: 5, inDim: 128, outDim: 128,
    cfg: JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1, layers: [
      { z: 0, y: 0, x: 0, l: 0, type: "MHA", input_height: 128, output_height: 128, num_heads: 4, d_model: 128, dtype: "F32" }
    ]})
  }
];

function makeTrainBatches(inDim: number, outDim: number, nBatches: number, batchSize: number, isEmbedding?: boolean) {
  const batches: any[] = [];
  for (let b = 0; b < nBatches; b++) {
    const inp = new Float32Array(batchSize * inDim);
    const tgt = new Float32Array(batchSize * outDim);
    if (isEmbedding) {
      for (let i = 0; i < inp.length; i++) inp[i] = i % 1024;
    } else {
      for (let i = 0; i < inp.length; i++) inp[i] = (Math.random() * 2 - 1) * 0.5;
    }
    for (let i = 0; i < tgt.length; i++) tgt[i] = Math.random() * 0.1;
    batches.push({
      input: { shape: [batchSize, inDim], data: Array.from(inp) },
      target: { shape: [batchSize, outDim], data: Array.from(tgt) }
    });
  }
  return batches;
}

async function runCase(tc: any) {
  // @ts-ignore
  const net = globalThis.createLoomNetwork(tc.cfg);
  const batchSize = 4;
  const nBatches = 4;
  const epochs = 3;

  const batches = makeTrainBatches(tc.inDim, tc.outDim, nBatches, batchSize, tc.isEmbedding);
  const batchesJSON = JSON.stringify(batches);

  const input = new Float32Array(tc.inDim);
  input.fill(0.5);
  if (tc.isEmbedding) for (let i = 0; i < input.length; i++) input[i] = i % 1024;

  // warm-up
  net.sequentialForward(input);

  const t0 = performance.now();
  let lastOut: any;
  for (let i = 0; i < tc.iters; i++) {
    lastOut = net.sequentialForward(input);
  }
  const fwdMs = (performance.now() - t0) / tc.iters;

  let trainMs = -1;
  let initialLoss: number | null = null, finalLoss: number | null = null;
  try {
    const t1 = performance.now();
    const trainResult = await net.train(batchesJSON, epochs, 0.001);
    trainMs = performance.now() - t1;
    if (typeof trainResult === 'string') {
      try {
        const r = JSON.parse(trainResult);
        if (r.loss_history && r.loss_history.length > 0) {
          initialLoss = r.loss_history[0];
          finalLoss = r.loss_history[r.loss_history.length - 1];
        }
      } catch (e) {}
    }
  } catch (e) {
    trainMs = -1;
  }

  const sample = lastOut ? [lastOut[0] || 0, lastOut[1] || 0, lastOut[2] || 0] : null;
  const sanity = sample && sample.some((v: number) => Math.abs(v) > 1e-9);
  net.free();
  return { fwdMs, trainMs, sample, sanity, initialLoss, finalLoss };
}

export async function runBenchmark() {
  console.log("=== M-POLY-VTD Training Showdown Benchmark ===");
  
  console.log("Layer".padEnd(15) + " | " + "Fwd ms/it".padEnd(11) + " | " + "Train ms".padEnd(10) +
    " | " + "Init Loss".padEnd(11) + " | " + "Final Loss".padEnd(11) + " | Sanity");
  console.log("-".repeat(85));

  for (const tc of TRAINING_CASES) {
    const res = await runCase(tc);

    const fwdStr = res.fwdMs >= 0 ? res.fwdMs.toFixed(3).padEnd(10) : 'N/A'.padEnd(10);
    const trainStr = res.trainMs >= 0 ? res.trainMs.toFixed(1).padEnd(9) : 'N/A'.padEnd(9);
    const iLoss = res.initialLoss != null ? res.initialLoss.toFixed(4).padEnd(10) : 'N/A'.padEnd(10);
    const fLoss = res.finalLoss != null ? res.finalLoss.toFixed(4).padEnd(10) : 'N/A'.padEnd(10);
    const sanStr = res.sanity ? 'REAL' : 'ZERO';

    console.log(`${tc.name.padEnd(15)} | ${fwdStr} | ${trainStr} | ${iLoss} | ${fLoss} | ${sanStr}`);
  }
}

async function main() {
  console.log("Initializing WASM...");
  await init();
  
  await runVerify();
  await runBenchmark();
}

main().catch(err => {
  console.error("Fatal error:", err);
  process.exit(1);
});
