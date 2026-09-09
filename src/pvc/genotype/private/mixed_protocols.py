"""PVC's three-party OT mixed products on original CrypTen tensor shares.

Ported from the PVC additions to CrypTen-research's replicated.py. The binary
selector is converted with the existing sender/receiver/helper OT, not upstream
Beaver B2A. Arithmetic products use the replicated cross-term formula directly.
No division, fixed-point rescaling, or truncation is performed by these kernels.

Upstream CrypTen is left unmodified. These protocols require exactly three
computing parties, and assume semi-honest parties with at most one corrupted
party. The upstream runtime supplies communication and pairwise PRG streams.

Copyright (c) Facebook, Inc. and its affiliates.
Derived protocol code is distributed under the MIT license in
src/crypto/CrypTen/LICENSE.
"""
import os
import torch
import crypten
from crypten.encoder import FixedPointEncoder
from crypten.mpc import MPCTensor
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor
from crypten.common.rng import generate_kbit_random_tensor


def _communicator():
    if not crypten.is_initialized():
        raise RuntimeError("mixed protocols require an initialized CrypTen communicator")
    communicator = crypten.communicator.get()
    if communicator.get_world_size() != 3:
        raise RuntimeError("mixed protocols require exactly three computing parties")
    return communicator


def _pairwise_generator(peer, device):
    rank = _communicator().get_rank()
    if peer == (rank - 1) % 3:
        return crypten.generators['prev'][device]
    if peer == (rank + 1) % 3:
        return crypten.generators['next'][device]
    raise ValueError("peer must be a different computing party")


def pack_bits(tensor):
    bits = (tensor.reshape(-1) & 1).to(torch.uint8)
    padding = (-bits.numel()) % 8
    bits = torch.cat([bits, bits.new_zeros(padding)]).reshape(8, -1)
    packed = bits.new_zeros(bits.size(1))
    for bit in range(8):
        packed |= bits[bit] << bit
    return packed


def unpack_bits(packed, size):
    bits = torch.stack([(packed >> bit) & 1 for bit in range(8)]).reshape(-1)
    count = 1
    for width in size:
        count *= width
    return bits[:count].reshape(size).long()


def _replicate(share, send_to, receive_from, *, packed=False):
    communicator = _communicator()
    send = pack_bits(share) if packed else share.contiguous()
    receive = torch.empty_like(send)
    requests = [communicator.isend(send, dst=send_to),
                communicator.irecv(receive, src=receive_from)]
    for request in requests:
        request.wait()
    return unpack_bits(receive, share.size()) if packed else receive


def mixed_mul_scalar(xs, binary, bits=1, roles=(0, 1, 2)):
    """OT product of sender-owned raw integer values and XOR-shared selector bits.

    Returns an arithmetic primitive at precision 0. Only the sender supplies xs;
    all three parties supply their own BinarySharedTensor. Public values are a
    special case. Payloads must have the same public shape as the selector.
    """
    communicator = _communicator()
    if not isinstance(binary, BinarySharedTensor) or bits != 1:
        raise ValueError("mixed_mul_scalar requires a binary tensor of one-bit selectors")
    if sorted(roles) != [0, 1, 2]:
        raise ValueError("roles must be a permutation of the three computing parties")
    sender, receiver, helper = roles
    rank = communicator.get_rank()
    if rank == sender and (not torch.is_tensor(xs) or xs.dtype != torch.long or xs.shape != binary.size()):
        raise ValueError("sender values must be int64 with the selector's shape")
    zero = ArithmeticSharedTensor.PRZS(binary.size(), device=binary.device).share
    if binary.share.numel() == 0:
        return ArithmeticSharedTensor.from_shares(zero, precision=0)
    # Replicate in OT-role order, preserving the original packed-bit exchange.
    position = roles.index(rank)
    local = binary.share & 1
    previous = _replicate(local, roles[(position + 1) % 3],
                          roles[(position - 1) % 3], packed=True)
    pairwise_pads = os.environ.get('PVC_OT_PAIRWISE_PRF_PADS', '1').lower() in {'1', 'true', 'yes', 'on'}
    shape = (2, *binary.size())
    if rank == sender:
        mask = generate_kbit_random_tensor(binary.size(), bitlength=64, device=binary.device)
        messages = torch.stack([(local ^ previous) * xs - mask,
                                (local ^ previous ^ 1) * xs - mask])
        pads = generate_kbit_random_tensor(shape, bitlength=64, device=binary.device,
                                          generator=_pairwise_generator(helper, binary.device) if pairwise_pads else None)
        masked = messages ^ pads
        requests = [communicator.isend(masked, dst=receiver)]
        if not pairwise_pads:
            requests.append(communicator.isend(pads, dst=helper))
        for request in requests:
            request.wait()
        share = zero + mask
    elif rank == receiver:
        masked = torch.empty(shape, dtype=torch.long, device=binary.device)
        selected_pad = torch.empty_like(local)
        requests = [communicator.irecv(masked, src=sender),
                    communicator.irecv(selected_pad, src=helper)]
        for request in requests:
            request.wait()
        selected = masked.reshape(2, -1)[local.reshape(-1), torch.arange(local.numel(), device=binary.device)]
        share = zero + (selected.reshape(local.size()) ^ selected_pad)
    else:
        if pairwise_pads:
            pads = generate_kbit_random_tensor(shape, bitlength=64, device=binary.device,
                                              generator=_pairwise_generator(sender, binary.device))
        else:
            pads = torch.empty(shape, dtype=torch.long, device=binary.device)
            communicator.irecv(pads, src=sender).wait()
        selected = pads.reshape(2, -1)[previous.reshape(-1), torch.arange(previous.numel(), device=binary.device)]
        selected = selected.reshape(previous.size()).contiguous()
        communicator.isend(selected, dst=receiver).wait()
        share = zero
    return ArithmeticSharedTensor.from_shares(share, precision=0)


def mul_no_truncation(left, right):
    """Replicated arithmetic product; output precision is the sum of inputs'."""
    communicator = _communicator()
    if not isinstance(left, ArithmeticSharedTensor) or not isinstance(right, ArithmeticSharedTensor):
        raise TypeError("mul_no_truncation requires two arithmetic primitives")
    rank = communicator.get_rank()
    x, y = left.share, right.share
    x_previous = _replicate(x, (rank + 1) % 3, (rank - 1) % 3)
    y_previous = _replicate(y, (rank + 1) % 3, (rank - 1) % 3)
    product = x * y + x_previous * y + x * y_previous
    precision = left.encoder._precision_bits + right.encoder._precision_bits
    return ArithmeticSharedTensor.from_shares(product, precision=precision)


def mixed_mul_no_truncation(arithmetic, binary, *, bits=1):
    """Binary × arithmetic using the original OT + replicated multiplication."""
    if isinstance(arithmetic, MPCTensor):
        arithmetic = arithmetic._tensor
    if not isinstance(arithmetic, ArithmeticSharedTensor):
        raise TypeError("arithmetic operand must contain arithmetic shares")
    sender_values = torch.ones_like(binary.share) if _communicator().get_rank() == 0 else None
    selector = mixed_mul_scalar(sender_values, binary, bits=bits)
    product = mul_no_truncation(arithmetic, selector)
    return MPCTensor.from_shares(product.share, precision=product.encoder._precision_bits)


def binary_times_public_no_truncation(public, binary, *, precision=0):
    """Binary × public directly through OT; retains the public value's scale."""
    if not isinstance(binary, BinarySharedTensor):
        raise TypeError("selector must be a BinarySharedTensor")
    encoded = FixedPointEncoder(precision_bits=precision).encode(public, device=binary.device)
    values, shares = torch.broadcast_tensors(encoded, binary.share)
    selector = BinarySharedTensor.from_shares(shares.contiguous(), precision=0)
    product = mixed_mul_scalar(values.contiguous() if _communicator().get_rank() == 0 else None, selector)
    return MPCTensor.from_shares(product.share, precision=precision)
