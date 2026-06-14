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
- `bitnet_kernel.py`: fused Triton kernel, packed-GEMM diagnostic kernel,
  experimental wide-dot packed kernels, pre-unpacked-weight control kernel, and
  Python wrappers.
- `benchmark.py`: CPU packing validation, GPU correctness checks, and benchmark
  chart generation. It reports the full fused kernel, a packed-GEMM-only path,
  a same-input cuBLAS FP16 control, and a pre-unpacked-weight Triton control.
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
3. Latency benchmarks across sequence lengths for the PyTorch references, the
   full fused Triton kernel, the packed-GEMM-only diagnostic path, a same-input
   cuBLAS FP16 control, and the pre-unpacked-weight Triton control.
4. A chart saved as `benchmark_results.png`.

To reproduce the rejected wide-dot packed experiment:

```bash
BITNET_WIDE=1 PYTHONPATH=. python benchmark.py
```

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
After a tuning sweep, the default Triton launch config is
`BLOCK_M=32, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=3`. This improves
the full benchmark substantially over the original `64x64x64` default, but it
is still slower than the PyTorch quantized reference for most sequence lengths.

The packed-GEMM-only diagnostic closely tracks the full fused kernel. That means
the current bottleneck is not primarily RMSNorm/activation quantization; it is
the packed ternary GEMM path itself, especially unpacking plus the four fp16
`tl.dot` accumulations.

The pre-unpacked-weight control is a deliberately simple Triton kernel using one
fp16 `tl.dot`. Weight conversion happens once before timing. The packed path is
8.5x-14.6x faster than this control on T4:

| M | Packed GEMM (ms) | Naive unpacked GEMM (ms) | Packed speedup |
|---:|---:|---:|---:|
| 16 | 0.711 | 6.026 | 8.47x |
| 32 | 0.673 | 6.027 | 8.95x |
| 64 | 1.234 | 13.742 | 11.14x |
| 128 | 2.448 | 28.342 | 11.58x |
| 256 | 4.263 | 58.753 | 13.78x |
| 512 | 7.996 | 116.688 | 14.59x |
| 1024 | 16.254 | 233.878 | 14.39x |
| 2048 | 32.602 | 468.625 | 14.37x |

This does not isolate bit extraction alone: the unpacked control reads 8x more
weight data than the 2-bit path and has not been independently tuned. It does
show that compressed weight traffic can outweigh the runtime unpacking cost in
this custom-kernel comparison.

The same-input cuBLAS FP16 control is the fair dense baseline for the GEMM
portion: it receives the same pre-quantized activation matrix and the same
pre-unpacked FP16 ternary weights as the packed diagnostic path. On T4, the
packed Triton GEMM is still 3.6x-13.8x slower than this cuBLAS control:

| M | Packed GEMM (ms) | Same-input cuBLAS (ms) | Packed slowdown |
|---:|---:|---:|---:|
| 16 | 0.711 | 0.172 | 4.14x |
| 32 | 0.673 | 0.187 | 3.60x |
| 64 | 1.234 | 0.190 | 6.50x |
| 128 | 2.448 | 0.206 | 11.91x |
| 256 | 4.263 | 0.505 | 8.43x |
| 512 | 7.996 | 0.703 | 11.37x |
| 1024 | 16.254 | 1.175 | 13.83x |
| 2048 | 32.602 | 2.574 | 12.67x |

The wide-dot packed experiment compiled and passed correctness, but was slower
than the legacy packed kernel on T4. It trades four smaller dot products for one
larger dot over an unpacked K tile, but the extra packed-byte reloads and wider
temporary weight tile outweighed that benefit:

| M | Legacy packed GEMM (ms) | Wide-dot packed GEMM (ms) | Wide/legacy |
|---:|---:|---:|---:|
| 16 | 0.636 | 4.514 | 7.10x |
| 32 | 0.708 | 4.549 | 6.42x |
| 64 | 1.257 | 10.037 | 7.98x |
| 128 | 2.482 | 20.494 | 8.26x |
| 256 | 4.374 | 42.254 | 9.66x |
| 512 | 8.170 | 84.090 | 10.29x |
| 1024 | 16.721 | 168.632 | 10.08x |
| 2048 | 33.427 | 337.345 | 10.09x |

Tesla T4 results with `N=4096, K=4096`:

| M | Dense FP16 (ms) | Quant Ref (ms) | Same-input cuBLAS (ms) | Packed GEMM (ms) | Fused Triton (ms) | Fused/Packed |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.288 | 0.743 | 0.172 | 0.711 | 0.634 | 0.89x |
| 32 | 0.213 | 0.429 | 0.187 | 0.673 | 0.727 | 1.08x |
| 64 | 0.224 | 0.684 | 0.190 | 1.234 | 1.265 | 1.03x |
| 128 | 0.258 | 1.147 | 0.206 | 2.448 | 2.434 | 0.99x |
| 256 | 0.569 | 2.426 | 0.505 | 4.263 | 4.330 | 1.02x |
| 512 | 0.924 | 4.673 | 0.703 | 7.996 | 8.238 | 1.03x |
| 1024 | 1.793 | 8.705 | 1.175 | 16.254 | 17.228 | 1.06x |
| 2048 | 3.654 | 18.287 | 2.574 | 32.602 | 35.327 | 1.08x |

The latest benchmark script also writes `benchmark_results.png`; regenerate the
chart from a fresh T4 run when publishing updated visuals.

## Next Engineering Targets

- Optimize the packed-weight GEMM path against the same-input cuBLAS control.
  This is now the main performance gap: the custom packed path is still much
  slower than optimized dense FP16 GEMM despite moving less weight data.
- Optimize the packed-weight unpack layout without redundantly reloading packed
  bytes across logical weight lanes.
- Replace the current `float16` `tl.dot` path with a true integer or ternary
  accumulation implementation if the goal is to claim integer GEMM.
- Revisit activation bandwidth only after the packed GEMM path improves, since
  the diagnostic run shows fused overhead is currently small.
- Add CI for CPU packing tests.
- Add kernel-level tests for more shapes, dtypes, and edge cases once a CUDA
  test environment is available.
