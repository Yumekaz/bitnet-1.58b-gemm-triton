import os

import torch

from bitnet_packing import pack_weights, unpack_weights_cpu


CORRECTNESS_ATOL = 1e-1
CORRECTNESS_RTOL = 1e-2


def kernel_config(block_m, block_n, block_k, num_warps=4, num_stages=3):
    name = f"m{block_m}_n{block_n}_k{block_k}_w{num_warps}_s{num_stages}"
    return {
        "name": name,
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


KERNEL_CONFIGS = [
    kernel_config(32, 32, 64),
    kernel_config(32, 64, 64),
    kernel_config(64, 32, 64),
    kernel_config(64, 64, 64),
    kernel_config(32, 128, 64),
    kernel_config(64, 128, 64),
    kernel_config(128, 64, 64),
    kernel_config(32, 64, 128),
    kernel_config(64, 64, 128),
    kernel_config(64, 64, 64, num_warps=8),
]


# Import the Triton kernel wrapper only when CUDA is available. This keeps CPU
# packing validation runnable on machines without Triton/CUDA.
if torch.cuda.is_available():
    from bitnet_kernel import (
        bitnet_fused_gemm,
        bitnet_packed_gemm,
        bitnet_unpacked_gemm,
    )
else:
    bitnet_fused_gemm = None
    bitnet_packed_gemm = None
    bitnet_unpacked_gemm = None


def kernel_kwargs(config):
    return {key: value for key, value in config.items() if key != "name"}


def diff_stats(actual: torch.Tensor, expected: torch.Tensor):
    diff = torch.abs(actual - expected).flatten()
    return {
        "max": torch.max(diff).item(),
        "mean": torch.mean(diff).item(),
        "p99": torch.quantile(diff, 0.99).item(),
    }


def precompute_quantized_activations(X: torch.Tensor, eps: float = 1e-5):
    """
    Precomputes the activation path used by BitNet GEMM:
    RMSNorm -> dynamic int8-range quantization -> row dequant scale.
    """
    X_float = X.to(torch.float32)
    rms = torch.sqrt(torch.mean(X_float * X_float, dim=1, keepdim=True) + eps)
    X_norm = X_float / rms
    row_max = torch.max(torch.abs(X_norm), dim=1, keepdim=True).values
    quant_scale = 127.0 / torch.clamp(row_max, min=eps)
    X_scaled = X_norm * quant_scale
    X_quant = torch.where(
        X_scaled >= 0,
        torch.floor(X_scaled + 0.5),
        torch.ceil(X_scaled - 0.5),
    )
    row_scale = (rms / quant_scale).reshape(-1)
    return X_quant, row_scale


def quantized_reference(X: torch.Tensor, W: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    PyTorch reference for the same math as the Triton kernel:
    RMSNorm -> dynamic int8 activation quantization -> ternary GEMM -> dequant.
    """
    X_quant, row_scale = precompute_quantized_activations(X, eps=eps)
    return (X_quant @ W.to(torch.float32).T) * row_scale[:, None]


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

    X_quant, row_scale = precompute_quantized_activations(X, eps=eps)
    Y_ref = (X_quant @ W.to(torch.float32).T) * row_scale[:, None]
    Y_triton = bitnet_fused_gemm(X, packed_W, eps=eps)
    X_quant_half = X_quant.to(torch.float16)
    Y_packed = bitnet_packed_gemm(X_quant_half, packed_W, row_scale)
    Y_unpacked = bitnet_unpacked_gemm(X_quant_half, W.to(torch.float16), row_scale)

    is_close = torch.allclose(Y_triton, Y_ref, rtol=CORRECTNESS_RTOL, atol=CORRECTNESS_ATOL)
    packed_is_close = torch.allclose(Y_packed, Y_ref, rtol=CORRECTNESS_RTOL, atol=CORRECTNESS_ATOL)
    unpacked_is_close = torch.allclose(Y_unpacked, Y_ref, rtol=CORRECTNESS_RTOL, atol=CORRECTNESS_ATOL)
    max_diff = torch.max(torch.abs(Y_triton - Y_ref)).item()
    packed_max_diff = torch.max(torch.abs(Y_packed - Y_ref)).item()
    unpacked_max_diff = torch.max(torch.abs(Y_unpacked - Y_ref)).item()

    if is_close:
        print(
            f"Fused correctness validation SUCCESS! "
            f"(Max diff: {max_diff:.4e}, rtol={CORRECTNESS_RTOL}, atol={CORRECTNESS_ATOL})"
        )
    else:
        print(
            f"Fused correctness validation FAILED! "
            f"(Max diff: {max_diff:.4e}, rtol={CORRECTNESS_RTOL}, atol={CORRECTNESS_ATOL})"
        )

    if packed_is_close:
        print(
            f"Packed-GEMM diagnostic SUCCESS! "
            f"(Max diff: {packed_max_diff:.4e}, rtol={CORRECTNESS_RTOL}, atol={CORRECTNESS_ATOL})"
        )
    else:
        print(
            f"Packed-GEMM diagnostic FAILED! "
            f"(Max diff: {packed_max_diff:.4e}, rtol={CORRECTNESS_RTOL}, atol={CORRECTNESS_ATOL})"
        )

    if unpacked_is_close:
        print(
            f"Unpacked-GEMM control SUCCESS! "
            f"(Max diff: {unpacked_max_diff:.4e}, rtol={CORRECTNESS_RTOL}, atol={CORRECTNESS_ATOL})"
        )
    else:
        print(
            f"Unpacked-GEMM control FAILED! "
            f"(Max diff: {unpacked_max_diff:.4e}, rtol={CORRECTNESS_RTOL}, atol={CORRECTNESS_ATOL})"
        )

    return is_close and packed_is_close and unpacked_is_close


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
      4. Packed Triton GEMM with pre-quantized activations.
      5. Unpacked-weight Triton GEMM control.
      6. Custom fused packed Triton kernel.
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
    latencies_packed_gemm = []
    latencies_unpacked_gemm = []
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
    Xq_warmup, scale_warmup = precompute_quantized_activations(X_warmup)
    Xq_warmup_half = Xq_warmup.to(torch.float16)
    bitnet_packed_gemm(Xq_warmup_half, packed_W, scale_warmup)
    bitnet_unpacked_gemm(Xq_warmup_half, W_fp16, scale_warmup)
    bitnet_fused_gemm(X_warmup, packed_W)
    torch.cuda.synchronize()

    for M in M_sizes:
        X = torch.randn((M, K), device=device, dtype=torch.float16)
        X_quant, row_scale = precompute_quantized_activations(X)
        X_quant_half = X_quant.to(torch.float16)

        ms_dense = triton.testing.do_bench(lambda: dense_fp16_reference(X))
        ms_quantized = triton.testing.do_bench(lambda: quantized_reference_for_bench(X))
        ms_packed = triton.testing.do_bench(lambda: bitnet_packed_gemm(X_quant_half, packed_W, row_scale))
        ms_unpacked = triton.testing.do_bench(
            lambda: bitnet_unpacked_gemm(X_quant_half, W_fp16, row_scale)
        )
        ms_triton = triton.testing.do_bench(lambda: bitnet_fused_gemm(X, packed_W))

        latencies_dense_fp16.append(ms_dense)
        latencies_quantized_ref.append(ms_quantized)
        latencies_packed_gemm.append(ms_packed)
        latencies_unpacked_gemm.append(ms_unpacked)
        latencies_triton.append(ms_triton)

        if compiled_quantized_reference is not None:
            ms_compiled = triton.testing.do_bench(lambda: compiled_quantized_reference(X))
            latencies_compiled_quantized.append(ms_compiled)
            compiled_msg = f" | Compiled Quant Ref: {ms_compiled:6.3f} ms"
        else:
            compiled_msg = ""

        print(
            f"M={M:4d} | Dense FP16: {ms_dense:6.3f} ms "
            f"| Quant Ref: {ms_quantized:6.3f} ms "
            f"| Packed GEMM: {ms_packed:6.3f} ms "
            f"| Unpacked GEMM: {ms_unpacked:6.3f} ms "
            f"{compiled_msg} | Fused Triton: {ms_triton:6.3f} ms "
            f"| Fused/packed: {ms_triton / ms_packed:.2f}x "
            f"| Packed/unpacked: {ms_packed / ms_unpacked:.2f}x "
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
    plt.plot(M_sizes, latencies_packed_gemm, label="Packed Triton GEMM only", marker="d", linewidth=2)
    plt.plot(
        M_sizes,
        latencies_unpacked_gemm,
        label="Unpacked-weight Triton control",
        marker="v",
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


def run_tuning_sweep(M=512, N=4096, K=4096):
    """
    Sweeps a small set of kernel tile/launch configs on one representative shape.
    Use this on a GPU machine with: BITNET_TUNE=1 PYTHONPATH=. python benchmark.py

    The large tuning shape can show bigger fp16 accumulation drift than the
    smaller correctness suite. The sweep reports PyTorch-reference drift, but
    ranks configs by latency as long as outputs are finite.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available on this system. Skipping kernel tuning sweep.")
        return None

    import triton

    print(f"\nStarting kernel config sweep for M={M}, N={N}, K={K}...")
    torch.manual_seed(0)
    device = torch.device("cuda")

    X = torch.randn((M, K), device=device, dtype=torch.float16)
    W_cpu = torch.randint(-1, 2, (N, K), dtype=torch.float32)
    W = W_cpu.to(device)
    packed_W = pack_weights(W_cpu).to(device)
    Y_ref = quantized_reference(X, W)
    Y_default = bitnet_fused_gemm(X, packed_W)
    torch.cuda.synchronize()

    default_ref_stats = diff_stats(Y_default, Y_ref)
    print(
        "Default config vs PyTorch ref "
        f"| max {default_ref_stats['max']:.4e} "
        f"| p99 {default_ref_stats['p99']:.4e} "
        f"| mean {default_ref_stats['mean']:.4e}"
    )

    results = []
    for config in KERNEL_CONFIGS:
        kwargs = kernel_kwargs(config)
        name = config["name"]
        try:
            Y_triton = bitnet_fused_gemm(X, packed_W, **kwargs)
            torch.cuda.synchronize()

            if not torch.isfinite(Y_triton).all():
                print(f"{name:20s} | FAILED finite-output check")
                continue

            ref_stats = diff_stats(Y_triton, Y_ref)
            default_stats = diff_stats(Y_triton, Y_default)
            ms = triton.testing.do_bench(lambda: bitnet_fused_gemm(X, packed_W, **kwargs))
            results.append((ms, name, ref_stats, default_stats, kwargs))
            print(
                f"{name:20s} | {ms:8.3f} ms "
                f"| ref max {ref_stats['max']:.4e} "
                f"| vs default max {default_stats['max']:.4e}"
            )
        except Exception as exc:
            print(f"{name:20s} | ERROR | {type(exc).__name__}: {exc}")

    if not results:
        print("No valid kernel configs completed.")
        return None

    results.sort(key=lambda row: row[0])
    best_ms, best_name, best_ref_stats, best_default_stats, best_kwargs = results[0]
    print("\nBest kernel config:")
    print(
        f"  {best_name}: {best_ms:.3f} ms, "
        f"ref max {best_ref_stats['max']:.4e}, "
        f"vs default max {best_default_stats['max']:.4e}"
    )
    print(f"  kwargs={best_kwargs}")
    return best_kwargs


if __name__ == "__main__":
    run_cpu_packing_validation()

    if torch.cuda.is_available():
        if run_correctness_suite():
            if os.environ.get("BITNET_TUNE") == "1":
                tune_m = int(os.environ.get("BITNET_TUNE_M", "512"))
                run_tuning_sweep(M=tune_m)
            else:
                run_benchmark()
    else:
        run_benchmark()
