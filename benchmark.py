import torch
import numpy as np
import time
from bitnet_packing import pack_weights, unpack_weights_cpu

# Import the Triton kernel wrapper if CUDA is available, otherwise define a stub
if torch.cuda.is_available():
    from bitnet_kernel import bitnet_fused_gemm
else:
    bitnet_fused_gemm = None


def run_correctness_test(M=128, N=1024, K=2048, eps=1e-5):
    """
    Verifies that the fused Triton kernel matches the PyTorch reference computation.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available on this system. Skipping correctness test.")
        return False
        
    print(f"Running correctness test for M={M}, N={N}, K={K}...")
    
    device = torch.device("cuda")
    
    # 1. Generate inputs
    X = torch.randn((M, K), device=device, dtype=torch.float16)
    # Generate weights in {-1, 0, 1}
    W_cpu = torch.randint(-1, 2, (N, K), dtype=torch.float32)
    W = W_cpu.to(device)
    
    # Pack weights
    packed_W = pack_weights(W_cpu).to(device)
    
    # 2. PyTorch reference execution (Fused RMSNorm + Quantization + GEMM + Dequantization)
    rms = torch.sqrt(torch.mean(X.to(torch.float32)**2, dim=1, keepdim=True) + eps)
    X_norm = X.to(torch.float32) / rms
    row_max = torch.max(torch.abs(X_norm), dim=1, keepdim=True).values
    quant_scale = 127.0 / torch.clamp(row_max, min=eps)
    X_quant = torch.round(X_norm * quant_scale)
    
    # GEMM
    Y_ref = (X_quant @ W.to(torch.float32).T) * (rms / quant_scale)
    
    # 3. Triton execution
    Y_triton = bitnet_fused_gemm(X, packed_W, eps=eps)
    
    # 4. Compare results
    # Use reasonable tolerances for quantized differences
    is_close = torch.allclose(Y_triton, Y_ref, rtol=1e-2, atol=1e-2)
    max_diff = torch.max(torch.abs(Y_triton - Y_ref)).item()
    
    if is_close:
        print(f"Correctness validation SUCCESS! (Max diff: {max_diff:.4e})")
    else:
        print(f"Correctness validation FAILED! (Max diff: {max_diff:.4e})")
        
    return is_close


def run_benchmark(N=4096, K=4096):
    """
    Benchmarks standard PyTorch FP16 MatMul, torch.compile FP16 MatMul,
    and our Custom Fused Packed Triton kernel.
    """
    if not torch.cuda.is_available():
        print("\n" + "="*70)
        print("BENCHMARK INFORMATION")
        print("="*70)
        print("To run the benchmark and generate the speedup charts, copy this folder")
        print("to a Google Colab notebook containing an NVIDIA GPU (e.g. Tesla T4).")
        print("Install triton via: pip install triton matplotlib")
        print("="*70 + "\n")
        return
        
    import matplotlib.pyplot as plt
    import triton
    
    print("\nStarting benchmarks...")
    device = torch.device("cuda")
    
    # Sizes of M to sweep (batch sizes / token lengths)
    M_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
    
    # Storage for latency metrics
    latencies_pytorch = []
    latencies_compiled = []
    latencies_triton = []
    
    # Create fixed weights
    W_cpu = torch.randint(-1, 2, (N, K), dtype=torch.float32)
    W_fp16 = W_cpu.to(device).half()
    packed_W = pack_weights(W_cpu).to(device)
    
    # Define functions to benchmark
    def benchmark_pytorch(X):
        # We also compute the RMSNorm in PyTorch to make the benchmark fair
        rms = torch.sqrt(torch.mean(X.to(torch.float32)**2, dim=1, keepdim=True) + 1e-5)
        X_norm = (X / rms).half()
        return X_norm @ W_fp16.T
        
    # Compile the PyTorch implementation for comparison
    benchmark_compiled = torch.compile(benchmark_pytorch)
    
    # Warm up compiled function
    X_warmup = torch.randn((128, K), device=device, dtype=torch.float16)
    benchmark_compiled(X_warmup)
    
    for M in M_sizes:
        X = torch.randn((M, K), device=device, dtype=torch.float16)
        
        # Benchmark Standard PyTorch (FP16)
        ms_py = triton.testing.do_bench(lambda: benchmark_pytorch(X))
        latencies_pytorch.append(ms_py)
        
        # Benchmark Compiled PyTorch (FP16)
        ms_comp = triton.testing.do_bench(lambda: benchmark_compiled(X))
        latencies_compiled.append(ms_comp)
        
        # Benchmark Fused Triton Kernel
        ms_triton = triton.testing.do_bench(lambda: bitnet_fused_gemm(X, packed_W))
        latencies_triton.append(ms_triton)
        
        print(f"M={M:4d} | PyTorch FP16: {ms_py:6.3f} ms | Compiled: {ms_comp:6.3f} ms | Fused Triton: {ms_triton:6.3f} ms | Speedup: {ms_py/ms_triton:.2f}x")
        
    # Save the chart
    plt.figure(figsize=(10, 6))
    plt.plot(M_sizes, latencies_pytorch, label='PyTorch FP16 (Standard)', marker='o', linewidth=2)
    plt.plot(M_sizes, latencies_compiled, label='PyTorch FP16 (torch.compile)', marker='s', linewidth=2)
    plt.plot(M_sizes, latencies_triton, label='Custom Fused Triton Kernel', marker='x', linewidth=2)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Token Sequence Length (M)', fontsize=12)
    plt.ylabel('Execution Latency (ms)', fontsize=12)
    plt.title(f'BitNet 1.58b GEMM Latency Comparison (N={N}, K={K})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="-")
    
    # Save image
    plt.savefig('benchmark_results.png', dpi=300)
    print("\nBenchmark chart saved as 'benchmark_results.png'")


if __name__ == "__main__":
    # If CUDA is available, run correctness check and benchmark
    if torch.cuda.is_available():
        correct = run_correctness_test()
        if correct:
            run_benchmark()
    else:
        # Run CPU bit-packing validation anyway to verify the utility works
        print("CUDA unavailable. Verifying bit-packing on CPU...")
        W_mock = torch.randint(-1, 2, (128, 512), dtype=torch.float32)
        packed = pack_weights(W_mock)
        unpacked = unpack_weights_cpu(packed, original_shape=W_mock.shape)
        assert torch.allclose(W_mock, unpacked)
        print("CPU local verification passed successfully!")
        run_benchmark()
