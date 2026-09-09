from itertools import permutations
from unittest.mock import patch
import numpy as np

from pvc.genotype.private.role_separated import split_arithmetic_secret, split_binary_secret, reconstruct_arithmetic
from pvc.genotype.private.runtime import spawn_multiparty_rank_args


def _kernels(raw_values, masks, public, precision, pairwise_pads):
    import os
    import torch
    from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor, converters, beaver
    from pvc.genotype.private.mixed_protocols import mixed_mul_no_truncation, binary_times_public_no_truncation, mixed_mul_scalar
    import crypten
    os.environ['PVC_OT_PAIRWISE_PRF_PADS'] = str(pairwise_pads)
    arithmetic = ArithmeticSharedTensor.from_shares(torch.tensor(raw_values), precision=precision)
    binary = BinarySharedTensor.from_shares(torch.tensor(masks), precision=0)
    # The preserved kernels must not fall back to upstream B2A, Beaver
    # multiplication, or any fixed-point division/truncation path.
    forbidden = AssertionError('unexpected conversion, Beaver multiplication, or truncation')
    with patch.object(ArithmeticSharedTensor, 'div', side_effect=forbidden), \
         patch.object(ArithmeticSharedTensor, 'div_', side_effect=forbidden), \
         patch.object(converters, '_B2A', side_effect=forbidden), \
         patch.object(beaver, 'mul', side_effect=forbidden):
        private = mixed_mul_no_truncation(arithmetic, binary)
        public_result = binary_times_public_no_truncation(torch.tensor(public), binary, precision=precision)
        rotation_results = []
        encoded = (torch.tensor(public) * 2**precision).long().expand(binary.size()).contiguous()
        for roles in permutations((0, 1, 2)):
            sender_values = encoded if crypten.communicator.get().get_rank() == roles[0] else None
            result = mixed_mul_scalar(sender_values, binary, roles=roles)
            rotation_results.append(result.share.tolist())
    assert private.encoder._precision_bits == public_result.encoder._precision_bits == precision
    from pvc.genotype.private.runtime import first_winner_one_hot
    # Exercise the resulting encoder in normal downstream addition/comparison.
    downstream = private + torch.tensor([0.5, -0.25, 1.0]) if precision else private + torch.tensor([2, -1, 1])
    winner = first_winner_one_hot(downstream, dim=1)
    return private.share.tolist(), public_result.share.tolist(), rotation_results, winner.share.tolist()


def test_preserved_ot_products_no_truncation_broadcast_and_fractional_scales():
    # Odd sizes exercise packed-bit padding. Arithmetic (1, 3) broadcasts
    # against binary (3, 1); public values use that same broadcasting rule.
    for precision, pairwise_pads in [(0, 0), (16, 1)]:
        values = np.array([[-3.25, 1.5, 4.0]]) if precision else np.array([[-7, 3, 2**40]], dtype=np.int64)
        mask = np.array([[1], [0], [1]], dtype=np.int64)
        arithmetic_shares = split_arithmetic_secret((values * 2**precision).astype(np.int64))
        # Test scalar OT separately on a shape that includes all public values.
        mask = np.broadcast_to(mask, (3, 3)).copy()
        binary_shares = split_binary_secret(mask)
        outputs = spawn_multiparty_rank_args(_kernels, [
            (arithmetic_shares[r], binary_shares[r], values, precision, pairwise_pads) for r in range(3)
        ], timeout=90)
        expected = (values * 2**precision).astype(np.int64) * mask
        np.testing.assert_array_equal(reconstruct_arithmetic([o[0] for o in outputs]), expected)
        np.testing.assert_array_equal(reconstruct_arithmetic([o[1] for o in outputs]), expected)
        bias = np.array([0.5, -0.25, 1.0]) if precision else np.array([2, -1, 1])
        scores = expected / 2**precision + bias
        winners = np.zeros_like(expected)
        winners[np.arange(3), np.argmax(scores, axis=1)] = 1
        np.testing.assert_array_equal(reconstruct_arithmetic([o[3] for o in outputs]), winners)
        for permutation in range(6):
            np.testing.assert_array_equal(reconstruct_arithmetic([o[2][permutation] for o in outputs]), expected)


def test_packed_bits_round_trip():
    import torch
    from pvc.genotype.private.mixed_protocols import pack_bits, unpack_bits
    for shape in [(), (1,), (7,), (2, 9), (0,)]:
        values = torch.randint(0, 2, shape)
        assert torch.equal(unpack_bits(pack_bits(values), shape), values)
