# BitNet 1.58b Triton Kernel Prototype

Prototype Triton kernels and benchmarks for packed ternary matrix
multiplication in BitNet-style 1.58-bit models.

This project explores an end-to-end GPU path for weights in `{-1, 0, 1}`:
packing four ternary weights into one byte, loading the packed representation on
GPU, unpacking with bit operations inside Triton, and combining the GEMM path
with RMSNorm and dynamic activation quantization.

The implementation is correctness-valid and useful as a systems prototype and
benchmark study. It is not currently faster than optimized PyTorch/cuBLAS FP16
GEMM.

## Highlights

- Packs ternary weights into a 2-bit layout: 4 weights per `int8` byte.
- Implements a fused Triton kernel for RMSNorm, activation quantization,
  packed-weight unpacking, matrix multiplication, and row-wise dequantization.
- Includes diagnostic kernels to separate fused activation overhead from packed
  GEMM overhead.
- Benchmarks against PyTorch quantized reference, `torch.compile`, same-input
  cuBLAS FP16 GEMM, and Triton control kernels.
- Documents both successful and rejected optimization attempts.

## Current Status

| Area | Status |
|---|---|
| Weight packing | Working and tested on CPU |
| Non-multiple-of-4 K padding | Supported |
| Fused Triton correctness | Passing on Tesla T4 |
| Packed GEMM vs naive unpacked Triton | Faster by 8.5x-14.6x |
| Packed GEMM vs same-input cuBLAS FP16 | Slower by 3.6x-13.8x |
| Wide-dot optimization attempt | Correct but slower, kept as optional experiment |

## Why This Exists

BitNet b1.58-style models use ternary weights, which can be stored much more
compactly than FP16 or FP32 weights. In principle, packed ternary weights should
reduce memory bandwidth pressure during inference.

A straightforward PyTorch implementation does not automatically get those
benefits:

1. Weights are commonly materialized as byte or floating-point tensors.
2. Activation normalization and quantization can create intermediate memory
   traffic.
3. A dense GEMM backend is highly optimized, but it is not operating directly on
   a compact 2-bit ternary representation.

This repository tests how much of that path can be moved into custom Triton
kernels while keeping weights packed in memory.

## Implementation

### Weight Packing

Ternary weights are encoded as:

```text
-1 -> 00
 0 -> 01
 1 -> 10
```

Each byte stores four weights:

```text
[ weight 3 | weight 2 | weight 1 | weight 0 ]
```

For a weight matrix with shape `(N, K)`, the packed representation has shape
`(N, ceil(K / 4))`. If `K` is not divisible by 4, the final byte is padded with
zero-weight lanes.

### Kernel Path

The main fused kernel:

1. Computes row-wise RMS statistics for activations.
2. Quantizes normalized activations into an int8-range value represented as
   `float16` tiles for Triton `tl.dot`.
3. Loads 2-bit packed ternary weights.
4. Unpacks weights on the fly with shifts and masks.
5. Accumulates the matrix product and applies row dequantization.

The current dot-product path still uses `tl.dot` with `float16` input tiles. It
should be treated as a packed-weight Triton prototype, not as a finished integer
or ternary GEMM implementation.

## Repository Layout

| File | Purpose |
|---|---|
| `bitnet_packing.py` | Pack/unpack utilities for ternary weights |
| `bitnet_kernel.py` | Fused kernel, packed GEMM diagnostics, controls, and experiments |
| `benchmark.py` | Correctness tests, benchmark harness, tuning sweep |
| `tests/test_packing.py` | CPU packing tests |
| `requirements.txt` | CPU validation dependencies |
| `requirements-gpu.txt` | GPU benchmark dependencies |

## Quickstart

CPU validation checks packing behavior only:

```bash
pip install -r requirements.txt
pytest
python benchmark.py
```

On a machine without CUDA, `benchmark.py` runs the CPU validation and prints GPU
benchmark instructions.

## GPU Benchmark

Use Google Colab or a Linux machine with an NVIDIA GPU:

```bash
pip install -r requirements-gpu.txt
PYTHONPATH=. python benchmark.py
```

The benchmark runs:

1. CPU pack/unpack validation.
2. GPU correctness tests for a standard shape and an odd-`K` padded shape.
3. Latency measurements for PyTorch references, same-input cuBLAS, packed
   Triton GEMM, naive unpacked Triton control, and the fused packed Triton
   kernel.
4. A chart saved as `benchmark_results.png`.

To run the rejected wide-dot experiment:

```bash
BITNET_WIDE=1 PYTHONPATH=. python benchmark.py
```

To sweep Triton tile and launch configurations:

```bash
BITNET_TUNE=1 PYTHONPATH=. python benchmark.py
```

The tuning sweep defaults to `M=512, N=4096, K=4096`. Override the sequence
length with:

```bash
BITNET_TUNE=1 BITNET_TUNE_M=1024 PYTHONPATH=. python benchmark.py
```

## Tesla T4 Results

Benchmarks below were run on Google Colab with a Tesla T4 at `N=4096, K=4096`.

Correctness checks:

```text
M=128, N=1024, K=2048: max diff 8.6578e-02, rtol=1e-2, atol=1e-1
M=17,  N=129,  K=513:  max diff 2.4048e-02, rtol=1e-2, atol=1e-1
```

Main latency comparison:

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

![Tesla T4 benchmark chart](assets/benchmark_results_t4.png)

## Interpretation

The packed Triton path validates the memory-compression idea against a naive
Triton control. Compared with a pre-unpacked-weight Triton kernel that reads
FP16 ternary weights, packed GEMM is significantly faster:

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

However, the fair dense baseline is same-input cuBLAS FP16 GEMM. It receives the
same pre-quantized activation matrix and the same pre-unpacked FP16 ternary
weights as the packed diagnostic path. Against this baseline, the custom packed
kernel is still slower:

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

The full fused kernel closely tracks the packed-GEMM-only diagnostic. This
suggests the current bottleneck is the packed GEMM path itself, not RMSNorm or
activation quantization overhead.

## Rejected Optimization: Wide-Dot Packing

An experimental wide-dot variant was tested. It expands packed weights across a
full K tile and performs one larger `tl.dot`, instead of four smaller dot
products over packed lanes.

The experiment compiled and passed correctness, but was much slower on T4:

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

The extra packed-byte reloads and wider temporary unpacked weight tile outweighed
the benefit of reducing four `tl.dot` calls to one. The implementation remains
available behind `BITNET_WIDE=1` for reproducibility, but it is not the default
path.

## Limitations

- The current kernels use `float16` `tl.dot` inputs after unpacking. They do not
  implement a true integer or ternary accumulation path.
- The packed Triton kernel is not performance-competitive with cuBLAS on T4.
- The naive unpacked Triton control is useful for studying weight traffic, but
  it is not an optimized dense GEMM baseline.
- CUDA tests are not part of CI; GPU correctness is validated manually in Colab.

## Future Work

- Replace the float16 dot path with a real integer or ternary accumulation
  strategy.
- Redesign the packed-weight layout to reduce redundant unpack/reload work.
- Add CUDA CI or a reproducible GPU benchmark workflow.
- Expand kernel tests across more shapes, dtypes, and GPU architectures.
