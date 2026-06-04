import torch
import triton
import triton.language as tl

@triton.jit
def _bitnet_fused_gemm_kernel(
    # Pointers to matrices
    x_ptr, w_ptr, y_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides (for row-major format)
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    # Hyperparameters
    eps,
    # Block sizes (passed as constants)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
):
    """
    Fused Triton Kernel for BitNet 1.58b:
    1. Computes RMSNorm for each row block of X on the fly.
    2. Quantizes normalized activations into the int8 value range.
    3. Loads 2-bit packed weights from HBM to SRAM.
    4. Unpacks weights on the fly using bit shifts.
    5. Computes high-performance GEMM and writes results to Y.
    
    Weight Packing Details:
      W contains values packed as 2-bit weights in int8.
      Inner dimension K is packed by a factor of 4, rounded up.
      Stride of packed K dimension is stride_wk.
    """
    # Identify the row and column of the output tile this program block computes
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # -------------------------------------------------------------
    # 1. Compute offsets and masks
    # -------------------------------------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Mask for output bounds
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # -------------------------------------------------------------
    # 2. First Pass: Compute RMS and max absolute value per row of X
    # -------------------------------------------------------------
    # We need RMS to normalize X, and max absolute value to scale/quantize into int8 range.
    # Accumulate sum of squares and max values across the K dimension.
    row_sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    row_max_val = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        x_offsets = offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        
        # Load X tile
        x_tile = tl.load(x_ptr + x_offsets, mask=mask_x, other=0.0).to(tl.float32)
        
        # Accumulate squares and absolute max
        row_sum_sq += tl.sum(x_tile * x_tile, axis=1)
        row_max_val = tl.maximum(row_max_val, tl.max(tl.abs(x_tile), axis=1))
        
    # Calculate RMS and quantization scale for each row
    # RMS = sqrt(sum(x^2) / K + eps)
    rms = tl.sqrt(row_sum_sq / K + eps)
    
    # Quantization Scale = 127.0 / max(abs(x / rms))
    # We clip max value to avoid division by zero
    norm_max = row_max_val / (rms + eps)
    quant_scale = 127.0 / tl.maximum(norm_max, eps)
    
    # -------------------------------------------------------------
    # 3. Main Loop: GEMM computation with on-the-fly weight unpacking
    # -------------------------------------------------------------
    # Accumulator tile for GEMM (stored in registers as float32 for high precision)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Weight K dimension is packed by 4
    K_PACKED = tl.cdiv(K, 4)
    
    for k_idx in range(0, tl.cdiv(K_PACKED, BLOCK_K_PACKED)):
        # Compute packed K offsets
        offs_k_packed = k_idx * BLOCK_K_PACKED + tl.arange(0, BLOCK_K_PACKED)
        
        # ---------------------------------------------------------
        # A. Load X tiles and perform Fused RMSNorm & Quantization
        # ---------------------------------------------------------
        # We load 4 sub-tiles of X corresponding to the 4 packed channels in W.
        # This keeps the GEMM memory layout linear.
        k_base = k_idx * BLOCK_K_PACKED * 4
        cols = tl.arange(0, BLOCK_K_PACKED)
        
        x0_offs = offs_m[:, None] * stride_xm + (k_base + cols[None, :] * 4 + 0) * stride_xk
        x1_offs = offs_m[:, None] * stride_xm + (k_base + cols[None, :] * 4 + 1) * stride_xk
        x2_offs = offs_m[:, None] * stride_xm + (k_base + cols[None, :] * 4 + 2) * stride_xk
        x3_offs = offs_m[:, None] * stride_xm + (k_base + cols[None, :] * 4 + 3) * stride_xk
        
        mask_x0 = (offs_m[:, None] < M) & ((k_base + cols[None, :] * 4 + 0) < K)
        mask_x1 = (offs_m[:, None] < M) & ((k_base + cols[None, :] * 4 + 1) < K)
        mask_x2 = (offs_m[:, None] < M) & ((k_base + cols[None, :] * 4 + 2) < K)
        mask_x3 = (offs_m[:, None] < M) & ((k_base + cols[None, :] * 4 + 3) < K)
        
        x0 = tl.load(x_ptr + x0_offs, mask=mask_x0, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + x1_offs, mask=mask_x1, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + x2_offs, mask=mask_x2, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + x3_offs, mask=mask_x3, other=0.0).to(tl.float32)
        
        # Apply RMSNorm and quantize into integer-valued fp16 tiles:
        # x_quant = round( (x / rms) * quant_scale )
        # Divide by rms and multiply by scale using row broadcasting
        x0_scaled = (x0 / rms[:, None]) * quant_scale[:, None]
        x1_scaled = (x1 / rms[:, None]) * quant_scale[:, None]
        x2_scaled = (x2 / rms[:, None]) * quant_scale[:, None]
        x3_scaled = (x3 / rms[:, None]) * quant_scale[:, None]

        x0_q = tl.where(
            x0_scaled >= 0.0,
            tl.floor(x0_scaled + 0.5),
            tl.ceil(x0_scaled - 0.5),
        ).to(tl.float16)
        x1_q = tl.where(
            x1_scaled >= 0.0,
            tl.floor(x1_scaled + 0.5),
            tl.ceil(x1_scaled - 0.5),
        ).to(tl.float16)
        x2_q = tl.where(
            x2_scaled >= 0.0,
            tl.floor(x2_scaled + 0.5),
            tl.ceil(x2_scaled - 0.5),
        ).to(tl.float16)
        x3_q = tl.where(
            x3_scaled >= 0.0,
            tl.floor(x3_scaled + 0.5),
            tl.ceil(x3_scaled - 0.5),
        ).to(tl.float16)
        
        # ---------------------------------------------------------
        # B. Load Packed Weights & Unpack in SRAM
        # ---------------------------------------------------------
        # W has shape (N, K_PACKED)
        w_offs = offs_n[:, None] * stride_wn + offs_k_packed[None, :] * stride_wk
        mask_w = (offs_n[:, None] < N) & (offs_k_packed[None, :] < K_PACKED)
        
        # Load int8 packed weights
        packed_w = tl.load(w_ptr + w_offs, mask=mask_w, other=0)
        
        # Unpack 2-bit weights to {-1, 0, 1}
        # w = ((packed_w >> shift) & 0b11) - 1
        w0 = (((packed_w >> 0) & 3) - 1).to(tl.float16)
        w1 = (((packed_w >> 2) & 3) - 1).to(tl.float16)
        w2 = (((packed_w >> 4) & 3) - 1).to(tl.float16)
        w3 = (((packed_w >> 6) & 3) - 1).to(tl.float16)
        
        # ---------------------------------------------------------
        # C. Compute Matrix Multiplication (GEMM)
        # ---------------------------------------------------------
        # Accumulate: acc += X_quant @ W_unpacked.T
        # Transpose w0, w1, w2, w3 since W shape is (N, K_packed) but we multiply by (K_packed, N)
        acc += tl.dot(x0_q, tl.trans(w0))
        acc += tl.dot(x1_q, tl.trans(w1))
        acc += tl.dot(x2_q, tl.trans(w2))
        acc += tl.dot(x3_q, tl.trans(w3))
        
    # -------------------------------------------------------------
    # 4. Epilogue: Scale GEMM accumulator back to float and write
    # -------------------------------------------------------------
    # Since we scaled activations by quant_scale, the result of the accumulation
    # must be divided by quant_scale to restore original range.
    # Output = Accumulator * (rms / quant_scale)
    dequant_scale = rms / (quant_scale + eps)
    
    # Broadcast dequant_scale along column dimension to match (BLOCK_M, BLOCK_N)
    y_val = acc * dequant_scale[:, None]
    
    # Write output to global memory
    y_offsets = offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    mask_y = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptr + y_offsets, y_val, mask=mask_y)


def bitnet_fused_gemm(X: torch.Tensor, packed_W: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Python wrapper for the fused BitNet GEMM Triton kernel.
    
    Arguments:
      X: Input activation tensor of shape (M, K), dtype float16 or float32.
      packed_W: Packed 2-bit weight tensor of shape (N, ceil(K / 4)), dtype int8.
      eps: Epsilon for numerical stability in RMSNorm.
      
    Returns:
      Y: Output tensor of shape (M, N), dtype float32.
    """
    assert X.is_cuda, "Activations must be on CUDA"
    assert packed_W.is_cuda, "Weights must be on CUDA"
    assert X.dim() == 2, "X must be 2D"
    assert packed_W.dim() == 2, "W must be 2D"
    
    M, K = X.shape
    N, K_packed = packed_W.shape
    expected_k_packed = (K + 3) // 4
    assert expected_k_packed == K_packed, (
        f"Weight K packing mismatch: X K={K}, expected packed K={expected_k_packed}, "
        f"packed_W K_packed={K_packed}"
    )
    
    # Output tensor allocation
    Y = torch.empty((M, N), device=X.device, dtype=torch.float32)
    
    # Block size configuration
    # These sizes are optimized to fit GPU shared memory limit (SRAM limits)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64  # Must be a multiple of 4 since we pack 4 elements per byte
    
    # Grid size definition (2D Grid over M and N dimensions)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    # Launch the compiled kernel
    _bitnet_fused_gemm_kernel[grid](
        X, packed_W, Y,
        M, N, K,
        X.stride(0), X.stride(1),
        packed_W.stride(0), packed_W.stride(1),
        Y.stride(0), Y.stride(1),
        eps,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BLOCK_K_PACKED=BLOCK_K // 4,
    )
    
    return Y
