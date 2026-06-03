import torch

def pack_weights(W: torch.Tensor) -> torch.Tensor:
    """
    Packs a 2D ternary weight matrix W of shape (N, K) containing values in {-1, 0, 1}
    into a packed int8 tensor of shape (N, K // 4) along the K dimension.
    If K is not a multiple of 4, it pads K with 0s (which map to '0' weight).
    
    Mapping rule:
      -1 -> 0 (binary 00)
       0 -> 1 (binary 01)
       1 -> 2 (binary 10)
    """
    assert W.dim() == 2, "Weight matrix must be 2D"
    N, K = W.shape
    
    # Pad K if not a multiple of 4
    remainder = K % 4
    if remainder != 0:
        padding = 4 - remainder
        W = torch.nn.functional.pad(W, (0, padding), value=0)
        K = K + padding
        
    # Ensure weight values are valid ternary values
    # In case there are minor float inaccuracies, round them
    W_clamped = torch.round(W).to(torch.int8)
    if not torch.all((W_clamped == -1) | (W_clamped == 0) | (W_clamped == 1)):
        raise ValueError("Weights must only contain -1, 0, or 1.")
        
    # Map {-1, 0, 1} to {0, 1, 2}
    W_mapped = W_clamped + 1
    
    # Vectorized bit-packing: pack 4 values into a single int8 byte
    # We slice W_mapped along the K dimension with stride 4
    w0 = W_mapped[:, 0::4]
    w1 = W_mapped[:, 1::4]
    w2 = W_mapped[:, 2::4]
    w3 = W_mapped[:, 3::4]
    
    packed = (w0 << 0) | (w1 << 2) | (w2 << 4) | (w3 << 6)
    
    # Return as signed int8
    return packed.to(torch.int8)


def unpack_weights_cpu(packed_W: torch.Tensor, original_shape=None) -> torch.Tensor:
    """
    Unpacks a packed int8 tensor of shape (N, K_packed) back to a float32 tensor
    of shape (N, K_packed * 4) using CPU vectorized operations.
    
    If original_shape (N, K) is provided, the output is sliced to match the original shape,
    removing any padding.
    """
    assert packed_W.dim() == 2, "Packed weight matrix must be 2D"
    N, K_packed = packed_W.shape
    
    # Extract the 2-bit values
    # Use bitwise shifts and mask with 0b11 (3) to prevent sign extension issues
    w0 = (packed_W >> 0) & 3
    w1 = (packed_W >> 2) & 3
    w2 = (packed_W >> 4) & 3
    w3 = (packed_W >> 6) & 3
    
    # Interleave the extracted channels back into the sequence
    # Shape: (N, K_packed, 4)
    stacked = torch.stack([w0, w1, w2, w3], dim=-1)
    
    # Flatten last two dimensions to get (N, K_packed * 4)
    unpacked = stacked.view(N, -1)
    
    # Map {0, 1, 2} back to {-1, 0, 1}
    W_unpacked = (unpacked - 1).to(torch.float32)
    
    # Slice to original shape if provided to remove padding
    if original_shape is not None:
        W_unpacked = W_unpacked[:, :original_shape[1]]
        
    return W_unpacked


if __name__ == "__main__":
    # Self-test validation on CPU
    print("Running CPU Bit-Packing verification...")
    
    # Create mock ternary weight matrix
    N, K = 128, 513  # Not a multiple of 4 to test padding
    W_mock = torch.randint(-1, 2, (N, K), dtype=torch.float32)
    
    # Pack
    W_packed = pack_weights(W_mock)
    print(f"Original shape: {W_mock.shape}, Packed shape: {W_packed.shape}")
    assert W_packed.shape == (N, (K + 3) // 4)
    
    # Unpack
    W_unpacked = unpack_weights_cpu(W_packed, original_shape=W_mock.shape)
    
    # Verify exact match
    all_close = torch.allclose(W_mock, W_unpacked)
    print(f"Correctness validation: {'SUCCESS' if all_close else 'FAILED'}")
    
    # Verify memory compression ratio
    orig_size = W_mock.nelement() * 2 # assuming FP16 (2 bytes)
    packed_size = W_packed.nelement() # signed int8 (1 byte)
    print(f"Original FP16 size: {orig_size} bytes")
    print(f"Packed 2-bit size: {packed_size} bytes")
    print(f"Compression ratio: {orig_size / packed_size:.2f}x vs FP16 ({orig_size*2/packed_size:.2f}x vs FP32)")
