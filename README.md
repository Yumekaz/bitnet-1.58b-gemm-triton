# BitNet 1.58b Fused Triton Kernel & Weight Packing

This repository is a prototype GPU kernel path for BitNet-style 1.58-bit ternary
weights. It includes CPU-side 2-bit weight packing, a Triton kernel that fuses
RMSNorm, dynamic activation quantization, packed weight unpacking, and tiled dot
product accumulation, plus a benchmark harness for correctness and latency tests.

The current kernel uses packed ternary weights and quantized activations, but the
dot-product path is still expressed through Triton `tl.dot` with `float16` input
tiles. Treat it as a fused packed-weight prototype, not yet as a finished
integer-GEMM implementation.

## The Systems Engineering Problem

BitNet b1.58-style models restrict weights to `{-1, 0, 1}`. That creates an
opportunity to reduce weight bandwidth dramatically, but a naive PyTorch
implementation still pays for several expensive memory movements:

1. Activations are read to compute RMSNorm.
2. Normalized activations are materialized or recomputed for quantization.
3. Ternary weights are often stored in byte or floating-point formats instead of
   compact 2-bit form.

This project explores how much of that work can be moved into one GPU kernel
while keeping weights packed in memory.

## Current Approach

1. **Fused activation handling**: the Triton kernel computes RMSNorm statistics,
   quantizes activations, and performs GEMM inside one kernel launch. The current
   implementation reads activations once for row statistics and again for the
   tiled dot-product pass, avoiding intermediate HBM writes.
2. **2-bit weight packing**: four ternary weights are stored in one `int8` byte.
   This is up to 8x smaller than FP16 weight storage and 16x smaller than FP32.
3. **On-the-fly unpacking**: packed bytes are loaded and unpacked with bit shifts
   immediately before dot-product accumulation.
4. **Same-math reference benchmark**: `benchmark.py` compares the custom kernel
   against a PyTorch reference that performs the same RMSNorm, quantization,
   ternary GEMM, and dequantization math.

## Weight Packing Layout

Ternary weights are mapped to 2-bit values:

```text
-1 -> 00
 0 -> 01
 1 -> 10
```

A single `int8` byte stores four packed weights:

```text
[ weight 3 | weight 2 | weight 1 | weight 0 ]
```

If `K` is not divisible by 4, packing pads the final byte with zero-weight lanes
(`01`). The kernel expects packed weights with shape `(N, ceil(K / 4))`.

## File Structure

- `bitnet_packing.py`: CPU/GPU PyTorch utility for packing ternary weights into
  2-bit byte storage and unpacking them for validation.
- `bitnet_kernel.py`: fused Triton kernel and Python wrapper.
- `benchmark.py`: CPU packing validation, GPU correctness checks, and benchmark
  chart generation.
- `tests/test_packing.py`: fast pytest coverage for packing invariants.

## Local CPU Validation

CPU validation checks the packing path only:

```bash
pip install -r requirements.txt
pytest
python benchmark.py
```

On a machine without CUDA, `benchmark.py` runs the CPU packing validation and
prints GPU benchmark instructions.

## GPU Benchmark Workflow

Use Google Colab or a Linux environment with an NVIDIA GPU:

```bash
pip install -r requirements-gpu.txt
python benchmark.py
```

The script runs:

1. CPU pack/unpack validation.
2. GPU correctness checks for standard and padded `K` shapes.
3. Latency benchmarks across sequence lengths.
4. A chart saved as `benchmark_results.png`.

To sweep Triton tile and launch parameters on one representative shape:

```bash
BITNET_TUNE=1 PYTHONPATH=. python benchmark.py
```

By default, this tunes at `M=512, N=4096, K=4096`. You can change the sequence
length with:

```bash
BITNET_TUNE=1 BITNET_TUNE_M=1024 PYTHONPATH=. python benchmark.py
```

The tuning sweep reports drift against the PyTorch reference and against the
default kernel output, then ranks finite-output configs by latency. The large
tuning shape may have larger FP16 accumulation drift than the smaller correctness
suite, so use the printed drift metrics when deciding whether to promote a
configuration.

## Current Benchmark Status

Google Colab Tesla T4 validation passes for both the main benchmark shape and a
non-multiple-of-4 hidden dimension:

```text
M=128, N=1024, K=2048: max diff 8.6578e-02, rtol=1e-2, atol=1e-1
M=17,  N=129,  K=513:  max diff 2.4048e-02, rtol=1e-2, atol=1e-1
```

The current kernel is correctness-valid, but not yet performance-competitive.
The full benchmark table below was captured on Tesla T4 with the original
`BLOCK_M=64, BLOCK_N=64, BLOCK_K=64` default. A later tuning sweep promoted
`BLOCK_M=32, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=3` as the current
default after improving the `M=512, N=4096, K=4096` kernel latency from
`16.514 ms` to `9.155 ms` with identical output to the old default. The full
chart should be regenerated for the promoted default.

On a Tesla T4 with `N=4096, K=4096`, the original default was slower than the
PyTorch quantized reference:

| M | Dense FP16 (ms) | Quant Ref (ms) | Fused Triton (ms) | Speedup vs Quant Ref |
|---:|---:|---:|---:|---:|
| 16 | 0.284 | 0.742 | 2.160 | 0.34x |
| 32 | 0.270 | 0.579 | 2.125 | 0.27x |
| 64 | 0.223 | 0.740 | 2.403 | 0.31x |
| 128 | 0.259 | 1.112 | 4.765 | 0.23x |
| 256 | 0.573 | 2.345 | 8.311 | 0.28x |
| 512 | 0.915 | 4.635 | 15.378 | 0.30x |
| 1024 | 1.753 | 8.394 | 30.708 | 0.27x |
| 2048 | 3.512 | 17.030 | 63.215 | 0.27x |

![Tesla T4 benchmark chart](assets/benchmark_results_t4.png)

## Next Engineering Targets

- Run the kernel config sweep on Tesla T4 and promote the fastest correct
  `BLOCK_M/BLOCK_N/BLOCK_K/num_warps/num_stages` configuration.
- Optimize the packed-weight unpack layout without redundantly reloading packed
  bytes across logical weight lanes.
- Reduce activation bandwidth by avoiding the current two-pass activation read
  for RMS/max statistics and GEMM input loading.
- Replace the current `float16` `tl.dot` path with a true integer or ternary
  accumulation implementation if the goal is to claim integer GEMM.
- Add CI for CPU packing tests.
- Add kernel-level tests for more shapes, dtypes, and edge cases once a CUDA
  test environment is available.
