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

## Next Engineering Targets

- Replace the current `float16` `tl.dot` path with a true integer dot-product
  implementation if the goal is to claim integer GEMM.
- Add captured benchmark results from a known GPU target such as T4, L4, A100,
  or RTX 4090.
- Add CI for CPU packing tests.
- Add kernel-level tests for more shapes, dtypes, and edge cases once a CUDA
  test environment is available.
