import torch

from bitnet_packing import pack_weights, unpack_weights_cpu


# Import the Triton kernel wrapper only when CUDA is available. This keeps CPU
# packing validation runnable on machines without Triton/CUDA.
if torch.cuda.is_available():
    from bitnet_kernel import bitnet_fused_gemm
else:
    bitnet_fused_gemm = None


def quantized_reference(X: torch.Tensor, W: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    PyTorch reference for the same math as the Triton kernel:
    RMSNorm -> dynamic int8 activation quantization -> ternary GEMM -> dequant.
    """
    X_float = X.to(torch.float32)
    rms = torch.sqrt(torch.mean(X_float * X_float, dim=1, keepdim=True) + eps)
    X_norm = X_float / rms
    row_max = torch.max(torch.abs(X_norm), dim=1, keepdim=True).values
    quant_scale = 127.0 / torch.clamp(row_max, min=eps)
    X_quant = torch.round(X_norm * quant_scale)

    return (X_quant @ W.to(torch.float32).T) * (rms / quant_scale)


def run_cpu_packing_validation():
    """
    Verifies CPU-side packing and unpacking, including non-multiple-of-4 K.
    """
    print("Running CPU bit-packing validation...")
    torch.manual_seed(0)

    for N, K in [(128, 512), (128, 513), (7, 3)]:
        W_mock = torch.randint(-1, 2, (N, K), dtype=torch.float32)
        packed = pack_weights(W_mock)
        unpacked = unpack_weights_cpu(packed, original_shape=W_mock.shape)

        assert packed.shape == (N, (K + 3) // 4)
        assert torch.allclose(W_mock, unpacked)
        print(f"  OK N={N}, K={K} -> packed K={packed.shape[1]}")

    print("CPU local verification passed successfully!")


def run_correctness_test(M=128, N=1024, K=2048, eps=1e-5):
    """
    Verifies that the fused Triton kernel matches the PyTorch quantized reference.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available on this system. Skipping GPU correctness test.")
        return False

    print(f"Running GPU correctness test for M={M}, N={N}, K={K}...")
    torch.manual_seed(0)
    device = torch.device("cuda")

    X = torch.randn((M, K), device=device, dtype=torch.float16)
    W_cpu = torch.randint(-1, 2, (N, K), dtype=torch.float32)
    W = W_cpu.to(device)
    packed_W = pack_weights(W_cpu).to(device)

    Y_ref = quantized_reference(X, W, eps=eps)
    Y_triton = bitnet_fused_gemm(X, packed_W, eps=eps)

    is_close = torch.allclose(Y_triton, Y_ref, rtol=1e-2, atol=1e-2)
    max_diff = torch.max(torch.abs(Y_triton - Y_ref)).item()

    if is_close:
        print(f"Correctness validation SUCCESS! (Max diff: {max_diff:.4e})")
    else:
        print(f"Correctness validation FAILED! (Max diff: {max_diff:.4e})")

    return is_close


def run_correctness_suite():
    """
    Covers the standard benchmark shape and an odd K shape that exercises padding.
    """
    cases = [
        (128, 1024, 2048),
        (17, 129, 513),
    ]
    return all(run_correctness_test(M=M, N=N, K=K) for M, N, K in cases)


def run_benchmark(N=4096, K=4096):
    """
    Benchmarks:
      1. Dense FP16 RMSNorm + GEMM as a context baseline.
      2. PyTorch quantized reference for the same math as the Triton kernel.
      3. torch.compile quantized reference when available.
      4. Custom fused packed Triton kernel.
    """
    if not torch.cuda.is_available():
        print("\n" + "=" * 70)
        print("BENCHMARK INFORMATION")
        print("=" * 70)
        print("CUDA is unavailable on this system.")
        print("Run GPU benchmarks in Colab or a Linux environment with an NVIDIA GPU.")
        print("Install GPU dependencies with: pip install triton matplotlib")
        print("=" * 70 + "\n")
        return

    import matplotlib.pyplot as plt
    import triton

    print("\nStarting benchmarks...")
    torch.manual_seed(0)
    device = torch.device("cuda")

    M_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
    latencies_dense_fp16 = []
    latencies_quantized_ref = []
    latencies_compiled_quantized = []
    latencies_triton = []

    W_cpu = torch.randint(-1, 2, (N, K), dtype=torch.float32)
    W = W_cpu.to(device)
    W_fp16 = W.half()
    packed_W = pack_weights(W_cpu).to(device)

    def dense_fp16_reference(X):
        rms = torch.sqrt(torch.mean(X.to(torch.float32) ** 2, dim=1, keepdim=True) + 1e-5)
        X_norm = (X.to(torch.float32) / rms).half()
        return X_norm @ W_fp16.T

    def quantized_reference_for_bench(X):
        return quantized_reference(X, W)

    compiled_quantized_reference = None
    if hasattr(torch, "compile"):
        compiled_quantized_reference = torch.compile(quantized_reference_for_bench)

    X_warmup = torch.randn((128, K), device=device, dtype=torch.float16)
    dense_fp16_reference(X_warmup)
    quantized_reference_for_bench(X_warmup)
    if compiled_quantized_reference is not None:
        compiled_quantized_reference(X_warmup)
    bitnet_fused_gemm(X_warmup, packed_W)
    torch.cuda.synchronize()

    for M in M_sizes:
        X = torch.randn((M, K), device=device, dtype=torch.float16)

        ms_dense = triton.testing.do_bench(lambda: dense_fp16_reference(X))
        ms_quantized = triton.testing.do_bench(lambda: quantized_reference_for_bench(X))
        ms_triton = triton.testing.do_bench(lambda: bitnet_fused_gemm(X, packed_W))

        latencies_dense_fp16.append(ms_dense)
        latencies_quantized_ref.append(ms_quantized)
        latencies_triton.append(ms_triton)

        if compiled_quantized_reference is not None:
            ms_compiled = triton.testing.do_bench(lambda: compiled_quantized_reference(X))
            latencies_compiled_quantized.append(ms_compiled)
            compiled_msg = f" | Compiled Quant Ref: {ms_compiled:6.3f} ms"
        else:
            compiled_msg = ""

        print(
            f"M={M:4d} | Dense FP16: {ms_dense:6.3f} ms "
            f"| Quant Ref: {ms_quantized:6.3f} ms"
            f"{compiled_msg} | Fused Triton: {ms_triton:6.3f} ms "
            f"| Speedup vs Quant Ref: {ms_quantized / ms_triton:.2f}x"
        )

    plt.figure(figsize=(10, 6))
    plt.plot(M_sizes, latencies_dense_fp16, label="Dense FP16 RMSNorm+GEMM", marker="o", linewidth=2)
    plt.plot(M_sizes, latencies_quantized_ref, label="PyTorch Quantized Reference", marker="s", linewidth=2)
    if latencies_compiled_quantized:
        plt.plot(
            M_sizes,
            latencies_compiled_quantized,
            label="torch.compile Quantized Reference",
            marker="^",
            linewidth=2,
        )
    plt.plot(M_sizes, latencies_triton, label="Custom Fused Packed Triton", marker="x", linewidth=2)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Token Sequence Length (M)", fontsize=12)
    plt.ylabel("Execution Latency (ms)", fontsize=12)
    plt.title(f"BitNet 1.58b GEMM Latency Comparison (N={N}, K={K})", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="-")
    plt.savefig("benchmark_results.png", dpi=300)
    print("\nBenchmark chart saved as 'benchmark_results.png'")


if __name__ == "__main__":
    run_cpu_packing_validation()

    if torch.cuda.is_available():
        if run_correctness_suite():
            run_benchmark()
    else:
        run_benchmark()
