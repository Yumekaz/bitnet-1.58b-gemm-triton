import pytest
import torch

from bitnet_packing import pack_weights, unpack_weights_cpu


@pytest.mark.parametrize(
    ("shape", "expected_packed_k"),
    [
        ((4, 4), 1),
        ((4, 5), 2),
        ((7, 3), 1),
        ((128, 513), 129),
    ],
)
def test_pack_unpack_round_trip_with_padding(shape, expected_packed_k):
    torch.manual_seed(0)
    weights = torch.randint(-1, 2, shape, dtype=torch.float32)

    packed = pack_weights(weights)
    unpacked = unpack_weights_cpu(packed, original_shape=weights.shape)

    assert packed.dtype == torch.int8
    assert packed.shape == (shape[0], expected_packed_k)
    assert torch.equal(weights, unpacked)


def test_pack_rejects_non_ternary_values():
    weights = torch.tensor([[0.0, 1.0, -1.0, 2.0]])

    with pytest.raises(ValueError, match="Weights must only contain"):
        pack_weights(weights)


def test_unpack_validates_original_shape():
    packed = torch.zeros((2, 1), dtype=torch.int8)

    with pytest.raises(ValueError, match="row count"):
        unpack_weights_cpu(packed, original_shape=(3, 4))

    with pytest.raises(ValueError, match="larger than the packed capacity"):
        unpack_weights_cpu(packed, original_shape=(2, 5))


def test_unpack_requires_int8_input():
    packed = torch.zeros((2, 1), dtype=torch.int16)

    with pytest.raises(TypeError, match="torch.int8"):
        unpack_weights_cpu(packed)
