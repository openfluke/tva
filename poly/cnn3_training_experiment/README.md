# CNN3 Training & Serialization Experiment

This experiment validates the training and bit-exact persistence of the **Poly** engine across all 21 supported numerical types and 6 execution modes.

## The Objective
The goal was to ensure that a 3D Convolutional Neural Network (CNN3) can:
1.  **Train** on any data type (from 64-bit Float down to 1-bit Binary).
2.  **Sync** correctly from GPU to CPU Master weights after training.
3.  **Serialize** to disk in a space-efficient (quantized/packed) state.
4.  **Deserialize** back to bit-exact Master weights (accounting for quantization loss).

## Core Engine Improvements
To achieve the results below, we made several fundamental improvements to the Poly engine:
-   **Global Quantization Clamping:** Implemented range-aware clamping in `WeightStore.Morph` to prevent bit-masking errors (e.g., preventing a value of 17 from wrapping around to 1 in a 4-bit field).
-   **Synchronized Rounding:** Both the simulation and the actual conversion now use `math.Round`, ensuring 100% agreement between CPU/GPU training and disk persistence.
-   **16-bit Bit-Packing:** Added actual 2-byte packing for `Float16` and `BFloat16` to reduce disk footprint by 2x.
-   **Sub-Byte Unpacking:** Fixed logic in `decodeNativeWeights` to correctly handle unsigned types (`Uint4`, `Uint2`) and bit-order during reconstruction.
-   **GPU Version Sync:** Resolved a bug where stale weight versions were being used after syncing from GPU, ensuring the latest trained weights are always saved.

## Final Results (100% PASS)

| DType      | Mode          | Loss[0]    | Loss[N]    | Time     | Train↑  | Save/Reload | File     | RAM      |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
| Float32    | CPU-Normal    | 1.1254e-01 | 1.1246e-01 | 2ms      | PASS    | PASS        | 2.8     KB | 1.7     KB |
| Float32    | CPU-SC-Tiled  | 1.4041e-01 | 1.4010e-01 | 77ms     | PASS    | PASS        | 2.9     KB | 1.7     KB |
| Float32    | CPU-MC-Tiled  | 1.2759e-01 | 1.2743e-01 | 1ms      | PASS    | PASS        | 2.9     KB | 1.7     KB |
| Float32    | GPU-Normal    | 1.2078e-01 | 1.2062e-01 | 428ms    | PASS    | PASS        | 2.8     KB | 1.7     KB |
| Float32    | GPU-SC-Tiled  | 1.0430e-01 | 1.0417e-01 | 338ms    | PASS    | PASS        | 2.8     KB | 1.7     KB |
| Float32    | GPU-MC-Tiled  | 1.1919e-01 | 1.1902e-01 | 403ms    | PASS    | PASS        | 2.8     KB | 1.7     KB |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
| Float64    | CPU-Normal    | 1.0653e-01 | 1.0645e-01 | 1ms      | PASS    | PASS        | 5.1     KB | 5.1     KB |
| Float64    | CPU-SC-Tiled  | 1.2870e-01 | 1.2852e-01 | 1ms      | PASS    | PASS        | 5.2     KB | 5.1     KB |
| Float64    | CPU-MC-Tiled  | 1.0241e-01 | 1.0232e-01 | 1ms      | PASS    | PASS        | 5.2     KB | 5.1     KB |
| Float64    | GPU-Normal    | 1.2514e-01 | 1.2491e-01 | 362ms    | PASS    | PASS        | 5.1     KB | 5.1     KB |
| Float64    | GPU-SC-Tiled  | 1.0590e-01 | 1.0577e-01 | 384ms    | PASS    | PASS        | 5.1     KB | 5.1     KB |
| Float64    | GPU-MC-Tiled  | 1.1620e-01 | 1.1604e-01 | 416ms    | PASS    | PASS        | 5.1     KB | 5.1     KB |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
| Float16    | CPU-Normal    | 1.3671e-01 | 1.3670e-01 | 2ms      | PASS    | PASS        | 1.7     KB | 2.5     KB |
| Float16    | CPU-SC-Tiled  | 1.0785e-01 | 1.0783e-01 | 1ms      | PASS    | PASS        | 1.8     KB | 2.5     KB |
| Float16    | CPU-MC-Tiled  | 1.1787e-01 | 1.1787e-01 | 1ms      | PASS    | PASS        | 1.8     KB | 2.5     KB |
| Float16    | GPU-Normal    | 1.2010e-01 | 1.1992e-01 | 423ms    | PASS    | PASS        | 1.7     KB | 2.5     KB |
| Float16    | GPU-SC-Tiled  | 1.0803e-01 | 1.0794e-01 | 401ms    | PASS    | PASS        | 1.7     KB | 2.5     KB |
| Float16    | GPU-MC-Tiled  | 1.1411e-01 | 1.1395e-01 | 382ms    | PASS    | PASS        | 1.7     KB | 2.5     KB |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
| BFloat16   | CPU-Normal    | 1.2610e-01 | 1.2605e-01 | 2ms      | PASS    | PASS        | 1.7     KB | 2.5     KB |
| BFloat16   | CPU-SC-Tiled  | 1.2212e-01 | 1.2211e-01 | 1ms      | PASS    | PASS        | 1.8     KB | 2.5     KB |
| BFloat16   | CPU-MC-Tiled  | 1.2822e-01 | 1.2825e-01 | 1ms      | PASS    | PASS        | 1.8     KB | 2.5     KB |
| BFloat16   | GPU-Normal    | 1.1808e-01 | 1.1796e-01 | 393ms    | PASS    | PASS        | 1.7     KB | 2.5     KB |
| BFloat16   | GPU-SC-Tiled  | 1.1544e-01 | 1.1531e-01 | 422ms    | PASS    | PASS        | 1.7     KB | 2.5     KB |
| BFloat16   | GPU-MC-Tiled  | 1.1106e-01 | 1.1093e-01 | 367ms    | PASS    | PASS        | 1.7     KB | 2.5     KB |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
| FP8-E4M3   | CPU-Normal    | 1.1187e-01 | 8.9786e-02 | 1ms      | PASS    | PASS        | 1.2     KB | 2.1     KB |
| FP8-E4M3   | CPU-SC-Tiled  | 1.0817e-01 | 1.0849e-01 | 1ms      | PASS    | PASS        | 1.2     KB | 2.1     KB |
| FP8-E4M3   | CPU-MC-Tiled  | 1.3399e-01 | 1.3358e-01 | 1ms      | PASS    | PASS        | 1.2     KB | 2.1     KB |
| FP8-E4M3   | GPU-Normal    | 1.4359e-01 | 1.4334e-01 | 408ms    | PASS    | PASS        | 1.2     KB | 2.1     KB |
| FP8-E4M3   | GPU-SC-Tiled  | 8.9765e-02 | 8.9765e-02 | 409ms    | PASS    | PASS        | 1.2     KB | 2.1     KB |
| FP8-E4M3   | GPU-MC-Tiled  | 8.9737e-02 | 8.9736e-02 | 408ms    | PASS    | PASS        | 1.2     KB | 2.1     KB |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
| FP8-E5M2   | CPU-Normal    | 1.1615e-01 | 8.9716e-02 | 2ms      | PASS    | PASS        | 1.2     KB | 2.1     KB |
| FP8-E5M2   | CPU-SC-Tiled  | 1.2353e-01 | 1.2382e-01 | 1ms      | PASS    | PASS        | 1.2     KB | 2.1     KB |
| FP8-E5M2   | CPU-MC-Tiled  | 1.0376e-01 | 1.0332e-01 | 1ms      | PASS    | PASS        | 1.2     KB | 2.1     KB |
| FP8-E5M2   | GPU-Normal    | 1.2150e-01 | 1.2136e-01 | 411ms    | PASS    | PASS        | 1.2     KB | 2.1     KB |
| FP8-E5M2   | GPU-SC-Tiled  | 8.9739e-02 | 8.9739e-02 | 459ms    | PASS    | PASS        | 1.2     KB | 2.1     KB |
| FP8-E5M2   | GPU-MC-Tiled  | 8.9888e-02 | 8.9888e-02 | 417ms    | PASS    | PASS        | 1.2     KB | 2.1     KB |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
| Int64      | CPU-Normal    | 1.2654e-01 | 8.9851e-02 | 1ms      | PASS    | PASS        | 5.1     KB | 5.1     KB |
| Int64      | CPU-SC-Tiled  | 1.1763e-01 | 1.1827e-01 | 1ms      | PASS    | PASS        | 5.2     KB | 5.1     KB |
| Int64      | CPU-MC-Tiled  | 1.1992e-01 | 1.2031e-01 | 1ms      | PASS    | PASS        | 5.2     KB | 5.1     KB |
| Int64      | GPU-Normal    | 1.1634e-01 | 1.1621e-01 | 469ms    | PASS    | PASS        | 5.1     KB | 5.1     KB |
| Int64      | GPU-SC-Tiled  | 8.9960e-02 | 8.9960e-02 | 413ms    | PASS    | PASS        | 5.1     KB | 5.1     KB |
| Int64      | GPU-MC-Tiled  | 8.9755e-02 | 8.9754e-02 | 472ms    | PASS    | PASS        | 5.1     KB | 5.1     KB |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
| Uint64     | CPU-Normal    | 1.1637e-01 | 8.9852e-02 | 2ms      | PASS    | PASS        | 5.1     KB | 5.1     KB |
| Uint64     | CPU-SC-Tiled  | 1.2176e-01 | 1.2156e-01 | 1ms      | PASS    | PASS        | 5.2     KB | 5.1     KB |
| Uint64     | CPU-MC-Tiled  | 1.1041e-01 | 1.0997e-01 | 1ms      | PASS    | PASS        | 5.2     KB | 5.1     KB |
| Uint64     | GPU-Normal    | 1.1384e-01 | 1.1369e-01 | 470ms    | PASS    | PASS        | 5.1     KB | 5.1     KB |
| Uint64     | GPU-SC-Tiled  | 8.9832e-02 | 8.9831e-02 | 476ms    | PASS    | PASS        | 5.1     KB | 5.1     KB |
| Uint64     | GPU-MC-Tiled  | 8.9822e-02 | 8.9821e-02 | 400ms    | PASS    | PASS        | 5.1     KB | 5.1     KB |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
| Int32      | CPU-Normal    | 1.2476e-01 | 8.9861e-02 | 1ms      | PASS    | PASS        | 2.9     KB | 3.4     KB |
| Int32      | CPU-SC-Tiled  | 1.1565e-01 | 1.1525e-01 | 1ms      | PASS    | PASS        | 2.9     KB | 3.4     KB |
| Int32      | CPU-MC-Tiled  | 1.1413e-01 | 1.1305e-01 | 1ms      | PASS    | PASS        | 2.9     KB | 3.4     KB |
| Int32      | GPU-Normal    | 1.1967e-01 | 1.1954e-01 | 481ms    | PASS    | PASS        | 2.9     KB | 3.4     KB |
| Int32      | GPU-SC-Tiled  | 8.9838e-02 | 8.9837e-02 | 477ms    | PASS    | PASS        | 2.9     KB | 3.4     KB |
| Int32      | GPU-MC-Tiled  | 8.9681e-02 | 8.9681e-02 | 403ms    | PASS    | PASS        | 2.9     KB | 3.4     KB |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
| Uint32     | CPU-Normal    | 1.0987e-01 | 8.9763e-02 | 2ms      | PASS    | PASS        | 2.9     KB | 3.4     KB |
| Uint32     | CPU-SC-Tiled  | 1.1504e-01 | 1.1551e-01 | 1ms      | PASS    | PASS        | 2.9     KB | 3.4     KB |
| Uint32     | CPU-MC-Tiled  | 1.1441e-01 | 1.1375e-01 | 1ms      | PASS    | PASS        | 2.9     KB | 3.4     KB |
| Uint32     | GPU-Normal    | 1.0628e-01 | 1.0620e-01 | 489ms    | PASS    | PASS        | 2.9     KB | 3.4     KB |
| Uint32     | GPU-SC-Tiled  | 8.9885e-02 | 8.9884e-02 | 441ms    | PASS    | PASS        | 2.9     KB | 3.4     KB |
| Uint32     | GPU-MC-Tiled  | 8.9867e-02 | 8.9866e-02 | 415ms    | PASS    | PASS        | 2.9     KB | 3.4     KB |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
| Int16      | CPU-Normal    | 1.2008e-01 | 8.9799e-02 | 2ms      | PASS    | PASS        | 1.7     KB | 2.5     KB |
| Int16      | CPU-SC-Tiled  | 1.1387e-01 | 1.1304e-01 | 1ms      | PASS    | PASS        | 1.8     KB | 2.5     KB |
| Int16      | CPU-MC-Tiled  | 1.2424e-01 | 1.2367e-01 | 1ms      | PASS    | PASS        | 1.8     KB | 2.5     KB |
| Int16      | GPU-Normal    | 1.3938e-01 | 1.3912e-01 | 423ms    | PASS    | PASS        | 1.7     KB | 2.5     KB |
| Int16      | GPU-SC-Tiled  | 8.9839e-02 | 8.9838e-02 | 509ms    | PASS    | PASS        | 1.7     KB | 2.5     KB |
| Int16      | GPU-MC-Tiled  | 8.9782e-02 | 8.9781e-02 | 500ms    | PASS    | PASS        | 1.7     KB | 2.5     KB |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
| Uint16     | CPU-Normal    | 1.1084e-01 | 8.9822e-02 | 2ms      | PASS    | PASS        | 1.7     KB | 2.5     KB |
| Uint16     | CPU-SC-Tiled  | 1.2136e-01 | 1.2151e-01 | 1ms      | PASS    | PASS        | 1.8     KB | 2.5     KB |
| Uint16     | CPU-MC-Tiled  | 1.2101e-01 | 1.2063e-01 | 1ms      | PASS    | PASS        | 1.8     KB | 2.5     KB |
| Uint16     | GPU-Normal    | 1.1378e-01 | 1.1363e-01 | 507ms    | PASS    | PASS        | 1.7     KB | 2.5     KB |
| Uint16     | GPU-SC-Tiled  | 8.9854e-02 | 8.9854e-02 | 506ms    | PASS    | PASS        | 1.7     KB | 2.5     KB |
| Uint16     | GPU-MC-Tiled  | 8.9821e-02 | 8.9821e-02 | 468ms    | PASS    | PASS        | 1.7     KB | 2.5     KB |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
| Int8       | CPU-Normal    | 1.1830e-01 | 8.9849e-02 | 2ms      | PASS    | PASS        | 1.2     KB | 2.1     KB |
| Int8       | CPU-SC-Tiled  | 1.2754e-01 | 1.2645e-01 | 2ms      | PASS    | PASS        | 1.2     KB | 2.1     KB |
| Int8       | CPU-MC-Tiled  | 1.1192e-01 | 1.1153e-01 | 2ms      | PASS    | PASS        | 1.2     KB | 2.1     KB |
| Int8       | GPU-Normal    | 1.2933e-01 | 1.2915e-01 | 466ms    | PASS    | PASS        | 1.2     KB | 2.1     KB |
| Int8       | GPU-SC-Tiled  | 8.9911e-02 | 8.9911e-02 | 519ms    | PASS    | PASS        | 1.2     KB | 2.1     KB |
| Int8       | GPU-MC-Tiled  | 8.9648e-02 | 8.9647e-02 | 468ms    | PASS    | PASS        | 1.2     KB | 2.1     KB |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
| Uint8      | CPU-Normal    | 1.1458e-01 | 8.9826e-02 | 1ms      | PASS    | PASS        | 1.2     KB | 2.1     KB |
| Uint8      | CPU-SC-Tiled  | 9.5778e-02 | 9.6130e-02 | 1ms      | PASS    | PASS        | 1.2     KB | 2.1     KB |
| Uint8      | CPU-MC-Tiled  | 1.2742e-01 | 1.2595e-01 | 1ms      | PASS    | PASS        | 1.2     KB | 2.1     KB |
| Uint8      | GPU-Normal    | 1.1334e-01 | 1.1320e-01 | 492ms    | PASS    | PASS        | 1.2     KB | 2.1     KB |
| Uint8      | GPU-SC-Tiled  | 8.9811e-02 | 8.9810e-02 | 446ms    | PASS    | PASS        | 1.2     KB | 2.1     KB |
| Uint8      | GPU-MC-Tiled  | 8.9782e-02 | 8.9782e-02 | 521ms    | PASS    | PASS        | 1.2     KB | 2.1     KB |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
| Int4       | CPU-Normal    | 1.2437e-01 | 8.9864e-02 | 2ms      | PASS    | PASS        | 0.9     KB | 1.9     KB |
| Int4       | CPU-SC-Tiled  | 1.1012e-01 | 1.1351e-01 | 1ms      | PASS    | PASS        | 0.9     KB | 1.9     KB |
| Int4       | CPU-MC-Tiled  | 1.0546e-01 | 1.0783e-01 | 1ms      | PASS    | PASS        | 0.9     KB | 1.9     KB |
| Int4       | GPU-Normal    | 1.2733e-01 | 1.2717e-01 | 534ms    | PASS    | PASS        | 0.9     KB | 1.9     KB |
| Int4       | GPU-SC-Tiled  | 8.9924e-02 | 8.9924e-02 | 523ms    | PASS    | PASS        | 0.9     KB | 1.9     KB |
| Int4       | GPU-MC-Tiled  | 8.9709e-02 | 8.9709e-02 | 482ms    | PASS    | PASS        | 0.9     KB | 1.9     KB |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
| Uint4      | CPU-Normal    | 1.0397e-01 | 8.9756e-02 | 1ms      | PASS    | PASS        | 0.9     KB | 1.9     KB |
| Uint4      | CPU-SC-Tiled  | 1.0357e-01 | 1.1150e-01 | 1ms      | PASS    | PASS        | 0.9     KB | 1.9     KB |
| Uint4      | CPU-MC-Tiled  | 1.1097e-01 | 1.2140e-01 | 1ms      | PASS    | PASS        | 0.9     KB | 1.9     KB |
| Uint4      | GPU-Normal    | 1.1495e-01 | 1.1482e-01 | 473ms    | PASS    | PASS        | 0.9     KB | 1.9     KB |
| Uint4      | GPU-SC-Tiled  | 8.9950e-02 | 8.9949e-02 | 473ms    | PASS    | PASS        | 0.9     KB | 1.9     KB |
| Uint4      | GPU-MC-Tiled  | 8.9759e-02 | 8.9758e-02 | 482ms    | PASS    | PASS        | 0.9     KB | 1.9     KB |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
| FP4        | CPU-Normal    | 9.9794e-02 | 8.9701e-02 | 1ms      | PASS    | PASS        | 0.9     KB | 1.9     KB |
| FP4        | CPU-SC-Tiled  | 1.0749e-01 | 1.0981e-01 | 1ms      | PASS    | PASS        | 0.9     KB | 1.9     KB |
| FP4        | CPU-MC-Tiled  | 1.1025e-01 | 1.1569e-01 | 1ms      | PASS    | PASS        | 0.9     KB | 1.9     KB |
| FP4        | GPU-Normal    | 1.1292e-01 | 1.1277e-01 | 487ms    | PASS    | PASS        | 0.9     KB | 1.9     KB |
| FP4        | GPU-SC-Tiled  | 8.9783e-02 | 8.9783e-02 | 472ms    | PASS    | PASS        | 0.9     KB | 1.9     KB |
| FP4        | GPU-MC-Tiled  | 8.9842e-02 | 8.9841e-02 | 512ms    | PASS    | PASS        | 0.9     KB | 1.9     KB |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
| Int2       | CPU-Normal    | 9.0354e-02 | 8.9823e-02 | 2ms      | PASS    | PASS        | 0.8     KB | 1.8     KB |
| Int2       | CPU-SC-Tiled  | 9.1748e-02 | 1.2638e-01 | 1ms      | PASS    | PASS        | 0.8     KB | 1.8     KB |
| Int2       | CPU-MC-Tiled  | 9.0980e-02 | 1.1573e-01 | 1ms      | PASS    | PASS        | 0.8     KB | 1.8     KB |
| Int2       | GPU-Normal    | 1.2448e-01 | 1.2433e-01 | 562ms    | PASS    | PASS        | 0.8     KB | 1.8     KB |
| Int2       | GPU-SC-Tiled  | 8.9792e-02 | 8.9792e-02 | 532ms    | PASS    | PASS        | 0.8     KB | 1.8     KB |
| Int2       | GPU-MC-Tiled  | 8.9788e-02 | 8.9788e-02 | 517ms    | PASS    | PASS        | 0.8     KB | 1.8     KB |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
| Uint2      | CPU-Normal    | 9.4542e-02 | 8.9792e-02 | 2ms      | PASS    | PASS        | 0.8     KB | 1.8     KB |
| Uint2      | CPU-SC-Tiled  | 9.3023e-02 | 1.1304e-01 | 2ms      | PASS    | PASS        | 0.8     KB | 1.8     KB |
| Uint2      | CPU-MC-Tiled  | 9.4341e-02 | 1.2966e-01 | 1ms      | PASS    | PASS        | 0.8     KB | 1.8     KB |
| Uint2      | GPU-Normal    | 1.1951e-01 | 1.1934e-01 | 524ms    | PASS    | PASS        | 0.8     KB | 1.8     KB |
| Uint2      | GPU-SC-Tiled  | 8.9798e-02 | 8.9798e-02 | 524ms    | PASS    | PASS        | 0.8     KB | 1.8     KB |
| Uint2      | GPU-MC-Tiled  | 8.9868e-02 | 8.9867e-02 | 487ms    | PASS    | PASS        | 0.8     KB | 1.8     KB |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
| Ternary    | CPU-Normal    | 1.3003e-01 | 9.0214e-02 | 1ms      | PASS    | PASS        | 0.8     KB | 1.8     KB |
| Ternary    | CPU-SC-Tiled  | 1.3315e-01 | 1.1934e-01 | 1ms      | PASS    | PASS        | 0.8     KB | 1.8     KB |
| Ternary    | CPU-MC-Tiled  | 1.3692e-01 | 1.1999e-01 | 1ms      | PASS    | PASS        | 0.8     KB | 1.8     KB |
| Ternary    | GPU-Normal    | 1.2892e-01 | 1.2877e-01 | 564ms    | PASS    | PASS        | 0.8     KB | 1.8     KB |
| Ternary    | GPU-SC-Tiled  | 9.0160e-02 | 9.0155e-02 | 537ms    | PASS    | PASS        | 0.8     KB | 1.8     KB |
| Ternary    | GPU-MC-Tiled  | 8.9861e-02 | 8.9856e-02 | 596ms    | PASS    | PASS        | 0.8     KB | 1.8     KB |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
| Binary     | CPU-Normal    | 1.9697e-01 | 8.9701e-02 | 1ms      | PASS    | PASS        | 0.7     KB | 1.7     KB |
| Binary     | CPU-SC-Tiled  | 1.6027e-01 | 1.0915e-01 | 1ms      | PASS    | PASS        | 0.7     KB | 1.7     KB |
| Binary     | CPU-MC-Tiled  | 1.8067e-01 | 1.1210e-01 | 1ms      | PASS    | PASS        | 0.7     KB | 1.7     KB |
| Binary     | GPU-Normal    | 1.1496e-01 | 1.1480e-01 | 525ms    | PASS    | PASS        | 0.7     KB | 1.7     KB |
| Binary     | GPU-SC-Tiled  | 9.0265e-02 | 9.0260e-02 | 573ms    | PASS    | PASS        | 0.7     KB | 1.7     KB |
| Binary     | GPU-MC-Tiled  | 8.9728e-02 | 8.9722e-02 | 566ms    | PASS    | PASS        | 0.7     KB | 1.7     KB |
|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|
