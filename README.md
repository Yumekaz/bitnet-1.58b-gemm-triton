# BitNet 1.58b Fused Triton Kernel & Weight Packing

This repository contains a high-performance custom GPU Triton kernel that implements fused RMSNorm, dynamic 2-bit weight unpacking, and quantized integer matrix multiplication (GEMM) for 1.58-bit (ternary) neural networks.

## The Systems Engineering Problem
1.58-bit models (like BitNet b1.58) restrict their weights to $\{-1, 0, 1\}$, replacing expensive floating-point multiplications with integer additions. 

However, implementing this in standard PyTorch creates a massive memory-bandwidth bottleneck:
1. Activations are loaded from High-Bandwidth Memory (HBM) into SRAM to calculate and apply RMSNorm, and written back to HBM.
2. Normalized activations are loaded again to apply quantization, and written back.
3. Unquantized float weights are loaded from HBM to perform floating-point GEMM.

This process involves multiple round-trips to HBM, which is the slowest memory layer on a GPU.

## Our Solution
We bypass PyTorch's memory abstractions and write a custom Triton kernel that achieves three major optimizations:

1. **SRAM Kernel Fusion**: Activations are loaded from global memory into SRAM exactly once. RMSNorm and 8-bit quantization are computed on the fly in registers, completely avoiding intermediate writes to HBM.
2. **2-bit Memory Packing**: Weights are stored as packed 2-bit values in a single `int8` byte (4 weights per byte), reducing weight memory bandwidth requirements by **4x** compared to standard FP16.
3. **On-the-Fly Bit-Shift Unpacking**: Packed weight bytes are loaded into SRAM and unpacked using fast, single-cycle GPU bitwise operations (`>>` and `&`) right before executing Tensor Core integer dot products.

---

## Weight Packing Layout
Ternary weights are mapped to 2-bit values:
*   `-1` $\to$ `00`
*   `0` $\to$ `01`
*   `1` $\to$ `10`
*   `Padding` $\to$ `11` (unused)

A single `int8` byte stores 4 packed weights:
```
[ Weight 3 (2-bits) | Weight 2 (2-bits) | Weight 1 (2-bits) | Weight 0 (2-bits) ]
```

---

## File Structure
*   `bitnet_packing.py`: CPU-side PyTorch utility for packing weights into 2-bit representations and unpacking baseline logic.
*   `bitnet_kernel.py`: Fused GPU kernel written in OpenAI's Triton language.
*   `benchmark.py`: Test suite validating mathematical correctness and benchmarking latency speedups.

---

## How to Run (Google Colab / Cloud GPU Workflow)

Since executing Triton requires an NVIDIA GPU (which is unsupported on AMD Radeon + Windows WSL setup), you can run and benchmark this code for free in Google Colab:

1.  **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.
2.  **Change Runtime Type**: In the top menu, go to **Runtime > Change runtime type** and select **T2 GPU** (or A100/L4 if available).
3.  **Upload the Code**:
    *   Click on the **Folder icon** on the left panel.
    *   Upload `bitnet_packing.py`, `bitnet_kernel.py`, and `benchmark.py` directly into the environment.
4.  **Install Triton & Matplotlib**:
    In a code cell, run:
    ```bash
    !pip install triton matplotlib
    ```
5.  **Run Correctness & Benchmarks**:
    In another code cell, execute:
    ```bash
    !python benchmark.py
    ```

This will run the validation check (verifying our kernel matches PyTorch's output down to $10^{-2}$ precision) and execute latency benchmarks across different sequence lengths ($M$), generating a chart saved as `benchmark_results.png`.
