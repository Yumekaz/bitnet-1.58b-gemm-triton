import os

import torch

from bitnet_packing import pack_weights, unpack_weights_cpu


CORRECTNESS_ATOL = 1e-1
CORRECTNESS_RTOL = 1e-2
_MM_SUPPORTS_OUT_DTYPE = None
ENABLE_WIDE_EXPERIMENTS = os.getenv("BITNET_WIDE", "0") == "1"


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
        bitnet_fused_gemm_wide,
        bitnet_packed_gemm,
        bitnet_packed_gemm_wide,
        bitnet_unpacked_gemm,
    )
else:
    bitnet_fused_gemm = None
    bitnet_fused_gemm_wide = None
    bitnet_packed_gemm = None
    bitnet_packed_gemm_wide = None
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


def same_input_cublas_reference(
    X_quant_half: torch.Tensor,
    W_fp16: torch.Tensor,
    row_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Optimized dense control using the same pre-quantized activations and row
    scale as the packed Triton GEMM. On CUDA, torch.mm dispatches to cuBLAS.
    """
    global _MM_SUPPORTS_OUT_DTYPE

    if _MM_SUPPORTS_OUT_DTYPE is not False:
        try:
            gemm = torch.mm(X_quant_half, W_fp16.T, out_dtype=torch.float32)
            _MM_SUPPORTS_OUT_DTYPE = True
            return gemm * row_scale[:, None]
        except TypeError:
            _MM_SUPPORTS_OUT_DTYPE = False

    return torch.mm(X_quant_half, W_fp16.T).to(torch.float32) * row_scale[:, None]


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
    Y_cublas = same_input_cublas_reference(X_quant_half, W.to(torch.float16), row_scale)
    Y_packed_wide = None
    Y_fused_wide = None
    packed_wide_error = None
    fused_wide_error = None

    if ENABLE_WIDE_EXPERIMENTS:
        try:
            Y_packed_wide = bitnet_packed_gemm_wide(X_quant_half, packed_W, row_scale)
        except Exception as exc:
            packed_wide_error = exc

        try:
            Y_fused_wide = bitnet_fused_gemm_wide(X, packed_W, eps=eps)
        except Exception as exc:
            fused_wide_error = exc

    is_close = torch.allclose(Y_triton, Y_ref, rtol=CORRECTNESS_RTOL, atol=CORRECTNESS_ATOL)
    packed_is_close = torch.allclose(Y_packed, Y_ref, rtol=CORRECTNESS_RTOL, atol=CORRECTNESS_ATOL)
    unpacked_is_close = torch.allclose(Y_unpacked, Y_ref, rtol=CORRECTNESS_RTOL, atol=CORRECTNESS_ATOL)
    cublas_is_close = torch.allclose(Y_cublas, Y_ref, rtol=CORRECTNESS_RTOL, atol=CORRECTNESS_ATOL)
    packed_wide_is_close = False
    fused_wide_is_close = False
    max_diff = torch.max(torch.abs(Y_triton - Y_ref)).item()
    packed_max_diff = torch.max(torch.abs(Y_packed - Y_ref)).item()
    unpacked_max_diff = torch.max(torch.abs(Y_unpacked - Y_ref)).item()
    cublas_max_diff = torch.max(torch.abs(Y_cublas - Y_ref)).item()
    packed_wide_max_diff = None
    fused_wide_max_diff = None

    if Y_packed_wide is not None:
        packed_wide_is_close = torch.allclose(
            Y_packed_wide,
            Y_ref,
            rtol=CORRECTNESS_RTOL,
            atol=CORRECTNESS_ATOL,
        )
        packed_wide_max_diff = torch.max(torch.abs(Y_packed_wide - Y_ref)).item()

    if Y_fused_wide is not None:
        fused_wide_is_close = torch.allclose(
            Y_fused_wide,
            Y_ref,
            rtol=CORRECTNESS_RTOL,
            atol=CORRECTNESS_ATOL,
        )
        fused_wide_max_diff = torch.max(torch.abs(Y_fused_wide - Y_ref)).item()

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

    if cublas_is_close:
        print(
            f"Same-input cuBLAS control SUCCESS! "
            f"(Max diff: {cublas_max_diff:.4e}, rtol={CORRECTNESS_RTOL}, atol={CORRECTNESS_ATOL})"
        )
    else:
        print(
            f"Same-input cuBLAS control FAILED! "
            f"(Max diff: {cublas_max_diff:.4e}, rtol={CORRECTNESS_RTOL}, atol={CORRECTNESS_ATOL})"
        )

    if not ENABLE_WIDE_EXPERIMENTS:
        print("Wide-dot experiments disabled. Set BITNET_WIDE=1 to reproduce them.")
    elif packed_wide_error is not None:
        print(
            f"Wide-dot packed diagnostic SKIPPED! "
            f"({type(packed_wide_error).__name__}: {packed_wide_error})"
        )
    else:
        if packed_wide_is_close:
            print(
                f"Wide-dot packed diagnostic SUCCESS! "
                f"(Max diff: {packed_wide_max_diff:.4e}, rtol={CORRECTNESS_RTOL}, atol={CORRECTNESS_ATOL})"
            )
        else:
            print(
                f"Wide-dot packed diagnostic FAILED! "
                f"(Max diff: {packed_wide_max_diff:.4e}, rtol={CORRECTNESS_RTOL}, atol={CORRECTNESS_ATOL})"
            )

    if not ENABLE_WIDE_EXPERIMENTS:
        pass
    elif fused_wide_error is not None:
        print(
            f"Wide-dot fused experiment SKIPPED! "
            f"({type(fused_wide_error).__name__}: {fused_wide_error})"
        )
    else:
        if fused_wide_is_close:
            print(
                f"Wide-dot fused experiment SUCCESS! "
                f"(Max diff: {fused_wide_max_diff:.4e}, rtol={CORRECTNESS_RTOL}, atol={CORRECTNESS_ATOL})"
            )
        else:
            print(
                f"Wide-dot fused experiment FAILED! "
                f"(Max diff: {fused_wide_max_diff:.4e}, rtol={CORRECTNESS_RTOL}, atol={CORRECTNESS_ATOL})"
            )

    return is_close and packed_is_close and unpacked_is_close and cublas_is_close


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
      5. Same-input dense fp16 GEMM dispatched to cuBLAS by PyTorch.
      6. Optional experimental wide-dot packed Triton GEMM with BITNET_WIDE=1.
      7. Naive unpacked-weight Triton GEMM control.
      8. Custom fused packed Triton kernel.
      9. Optional experimental wide-dot fused Triton kernel with BITNET_WIDE=1.
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
    latencies_packed_wide_gemm = []
    latencies_cublas_gemm = []
    latencies_unpacked_gemm = []
    latencies_triton = []
    latencies_triton_wide = []

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
    Y_packed_warmup = bitnet_packed_gemm(Xq_warmup_half, packed_W, scale_warmup)
    same_input_cublas_reference(Xq_warmup_half, W_fp16, scale_warmup)
    bitnet_unpacked_gemm(Xq_warmup_half, W_fp16, scale_warmup)
    Y_fused_warmup = bitnet_fused_gemm(X_warmup, packed_W)
    packed_wide_available = ENABLE_WIDE_EXPERIMENTS
    fused_wide_available = ENABLE_WIDE_EXPERIMENTS
    if ENABLE_WIDE_EXPERIMENTS:
        try:
            Y_packed_wide_warmup = bitnet_packed_gemm_wide(Xq_warmup_half, packed_W, scale_warmup)
            if not torch.allclose(
                Y_packed_wide_warmup,
                Y_packed_warmup,
                rtol=CORRECTNESS_RTOL,
                atol=CORRECTNESS_ATOL,
            ):
                packed_wide_available = False
                wide_diff = torch.max(torch.abs(Y_packed_wide_warmup - Y_packed_warmup)).item()
                print(f"Wide-dot packed GEMM disabled after warmup correctness diff: {wide_diff:.4e}")
        except Exception as exc:
            packed_wide_available = False
            print(f"Wide-dot packed GEMM disabled after warmup: {type(exc).__name__}: {exc}")
        try:
            Y_fused_wide_warmup = bitnet_fused_gemm_wide(X_warmup, packed_W)
            if not torch.allclose(
                Y_fused_wide_warmup,
                Y_fused_warmup,
                rtol=CORRECTNESS_RTOL,
                atol=CORRECTNESS_ATOL,
            ):
                fused_wide_available = False
                wide_diff = torch.max(torch.abs(Y_fused_wide_warmup - Y_fused_warmup)).item()
                print(f"Wide-dot fused GEMM disabled after warmup correctness diff: {wide_diff:.4e}")
        except Exception as exc:
            fused_wide_available = False
            print(f"Wide-dot fused GEMM disabled after warmup: {type(exc).__name__}: {exc}")
    torch.cuda.synchronize()

    for M in M_sizes:
        X = torch.randn((M, K), device=device, dtype=torch.float16)
        X_quant, row_scale = precompute_quantized_activations(X)
        X_quant_half = X_quant.to(torch.float16)

        ms_dense = triton.testing.do_bench(lambda: dense_fp16_reference(X))
        ms_quantized = triton.testing.do_bench(lambda: quantized_reference_for_bench(X))
        ms_packed = triton.testing.do_bench(lambda: bitnet_packed_gemm(X_quant_half, packed_W, row_scale))
        ms_cublas = triton.testing.do_bench(
            lambda: same_input_cublas_reference(X_quant_half, W_fp16, row_scale)
        )
        ms_packed_wide = None
        if packed_wide_available:
            try:
                ms_packed_wide = triton.testing.do_bench(
                    lambda: bitnet_packed_gemm_wide(X_quant_half, packed_W, row_scale)
                )
            except Exception as exc:
                packed_wide_available = False
                print(f"Wide-dot packed GEMM disabled at M={M}: {type(exc).__name__}: {exc}")
        ms_unpacked = triton.testing.do_bench(
            lambda: bitnet_unpacked_gemm(X_quant_half, W_fp16, row_scale)
        )
        ms_triton = triton.testing.do_bench(lambda: bitnet_fused_gemm(X, packed_W))
        ms_triton_wide = None
        if fused_wide_available:
            try:
                ms_triton_wide = triton.testing.do_bench(lambda: bitnet_fused_gemm_wide(X, packed_W))
            except Exception as exc:
                fused_wide_available = False
                print(f"Wide-dot fused GEMM disabled at M={M}: {type(exc).__name__}: {exc}")

        latencies_dense_fp16.append(ms_dense)
        latencies_quantized_ref.append(ms_quantized)
        latencies_packed_gemm.append(ms_packed)
        if ms_packed_wide is not None:
            latencies_packed_wide_gemm.append(ms_packed_wide)
        latencies_cublas_gemm.append(ms_cublas)
        latencies_unpacked_gemm.append(ms_unpacked)
        latencies_triton.append(ms_triton)
        if ms_triton_wide is not None:
            latencies_triton_wide.append(ms_triton_wide)

        if compiled_quantized_reference is not None:
            ms_compiled = triton.testing.do_bench(lambda: compiled_quantized_reference(X))
            latencies_compiled_quantized.append(ms_compiled)
            compiled_msg = f" | Compiled Quant Ref: {ms_compiled:6.3f} ms"
        else:
            compiled_msg = ""
        packed_wide_msg = ""
        if ms_packed_wide is not None:
            packed_wide_msg = (
                f"| Wide Packed: {ms_packed_wide:6.3f} ms "
                f"| Wide/legacy packed: {ms_packed_wide / ms_packed:.2f}x "
            )
        fused_wide_msg = ""
        if ms_triton_wide is not None:
            fused_wide_msg = (
                f"| Wide Fused: {ms_triton_wide:6.3f} ms "
                f"| Wide/legacy fused: {ms_triton_wide / ms_triton:.2f}x "
            )

        print(
            f"M={M:4d} | Dense FP16: {ms_dense:6.3f} ms "
            f"| Quant Ref: {ms_quantized:6.3f} ms "
            f"| Packed GEMM: {ms_packed:6.3f} ms "
            f"{packed_wide_msg}"
            f"| Same-input cuBLAS: {ms_cublas:6.3f} ms "
            f"| Unpacked GEMM: {ms_unpacked:6.3f} ms "
            f"{compiled_msg} | Fused Triton: {ms_triton:6.3f} ms "
            f"{fused_wide_msg}"
            f"| Fused/packed: {ms_triton / ms_packed:.2f}x "
            f"| Packed slowdown vs cuBLAS: {ms_packed / ms_cublas:.2f}x "
            f"| Packed speedup vs unpacked: {ms_unpacked / ms_packed:.2f}x "
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
        latencies_cublas_gemm,
        label="Same-input cuBLAS FP16 GEMM",
        marker="P",
        linewidth=2,
    )
    if len(latencies_packed_wide_gemm) == len(M_sizes):
        plt.plot(
            M_sizes,
            latencies_packed_wide_gemm,
            label="Wide-dot packed Triton GEMM",
            marker="*",
            linewidth=2,
        )
    plt.plot(
        M_sizes,
        latencies_unpacked_gemm,
        label="Naive unpacked-weight Triton control",
        marker="v",
        linewidth=2,
    )
    plt.plot(M_sizes, latencies_triton, label="Custom Fused Packed Triton", marker="x", linewidth=2)
    if len(latencies_triton_wide) == len(M_sizes):
        plt.plot(M_sizes, latencies_triton_wide, label="Wide-dot fused packed Triton", marker="h", linewidth=2)
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
