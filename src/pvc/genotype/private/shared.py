"""Three-party upstream CrypTen scoring kernel for PVC haplotype scoring.

This module intentionally protects only read-derived emissions and scores.
Haplotypes, LD blocks, candidate pairs, and public indices are treated as
public inputs.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import json
import os
import tempfile
import time

import numpy as np

from pvc.genotype.private.mixed_protocols import mixed_mul_no_truncation, binary_times_public_no_truncation
from pvc.genotype.private.runtime import (
    from_arithmetic, secure_maximum, first_winner_one_hot,
    require_three_party_runtime,
)

from pvc.genotype.genotyping import (
    assign_variants_to_blocks,
    block_assignments_from_ir_blocks,
    compute_allele_frequency_vectors,
    compute_emission_for_genotype,
    compute_emission_matrix,
    compute_frequency_matrix,
    normalize_log_matrix,
    precompute_haplotype_pair_table,
    run_plink_blocks,
)
from pvc.index.ir import GenotypingPlan
from pvc.pangenome.genotyping_result import GenotypingResult
from pvc.genotype.private.workflow_profile import (
    account_client_transfer,
    profile_phase,
)

DEFAULT_SCALE = 1_000_000
NEG_INF_FIXED = -(1 << 50)

# ---------------------------------------------------------------------------
# Lightweight phase profiler (opt-in via PVC_PROFILE=1). Accumulates wall-time
# and call counts per named phase and prints a summary at the end of a run, so
# we can see where the per-block scoring time actually goes (emission prep vs
# cryptensor encryption vs gather vs reveal). Zero-overhead when disabled.
# ---------------------------------------------------------------------------
_PROFILE_ENABLED = os.environ.get("PVC_PROFILE", "0") != "0"
_PROFILE_TIMES: dict[str, float] = {}
_PROFILE_COUNTS: dict[str, int] = {}


@contextmanager
def _profile(name):
    if not _PROFILE_ENABLED:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - start
        _PROFILE_TIMES[name] = _PROFILE_TIMES.get(name, 0.0) + dt
        _PROFILE_COUNTS[name] = _PROFILE_COUNTS.get(name, 0) + 1


# Comm-byte attribution profiler (opt-in via PVC_PROFILE_COMM=1). Reads the
# CrypTen communicator's cumulative comm_bytes and legacy comm_rounds operation
# proxy before/after a named region. This shows which regions (cov lookup,
# count lookup, logaddexp-max, ...) move bytes and invoke low-level operations;
# it does not measure MPC dependency-round depth. Zero-overhead when disabled.
_COMM_PROFILE_ENABLED = os.environ.get("PVC_PROFILE_COMM", "0") != "0"
_COMM_BYTES: dict[str, int] = {}
_COMM_OPERATIONS: dict[str, int] = {}
_COMM_CALLS: dict[str, int] = {}

# Client -> server secret-share UPLOAD model. This CrypTen shares src=0 inputs via
# local PRZS (zero network), so the client's real upload is invisible to the byte
# counters and is modeled as the information-theoretic cost: one independent int64
# share (8 B/elt) per non-source party. Counts only the client's PRIVATE inputs
# (emission matrices for light/medium; the one-hot count/coverage selectors that
# encode the private read counts/coverage for heavy) -- NOT the public probability
# table. Accumulated on rank 0 for the legacy total and also attributed to the
# default-on workflow profiler. Detailed legacy region reports remain gated by
# PVC_PROFILE_COMM.
_CLIENT_UPLOAD_BYTES = 0


def _account_client_upload(
    nelement,
    element_size=8,
    *,
    phase="emission_computation",
    rounds=1,
    model=(
        "one independent int64 share per private value per non-source party; "
        "one modeled upload call per share-construction call"
    ),
):
    global _CLIENT_UPLOAD_BYTES
    try:
        import crypten

        comm = crypten.communicator.get()
        if comm.get_rank() != 0:
            return
        world = comm.get_world_size()
    except Exception:
        world = 3
    nbytes = int(nelement) * int(element_size) * (world - 1)
    _CLIENT_UPLOAD_BYTES += nbytes
    account_client_transfer(
        phase,
        "upload",
        nbytes,
        rounds=rounds,
        model=model,
    )


def _account_hybrid_selector_upload(
    arithmetic_elements,
    binary_elements,
    *,
    phase="emission_input_encoding",
    rounds=1,
):
    """Model arithmetic coverage shares plus bit-packed binary count shares."""
    global _CLIENT_UPLOAD_BYTES
    try:
        import crypten

        communicator = crypten.communicator.get()
        if communicator.get_rank() != 0:
            return
        non_source_parties = communicator.get_world_size() - 1
    except Exception:
        non_source_parties = 2
    arithmetic_bytes = int(arithmetic_elements) * 8 * non_source_parties
    # Pack 64 binary values into each transmitted 64-bit share word.
    packed_words = (int(binary_elements) + 63) // 64
    binary_bytes = packed_words * 8 * non_source_parties
    nbytes = arithmetic_bytes + binary_bytes
    _CLIENT_UPLOAD_BYTES += nbytes
    account_client_transfer(
        phase,
        "upload",
        nbytes,
        rounds=rounds,
        model=(
            "one batched upload containing int64 arithmetic coverage shares "
            "and bit-packed binary count shares (64 selectors per word)"
        ),
    )


def get_client_upload_bytes():
    return _CLIENT_UPLOAD_BYTES


def reset_client_upload_bytes():
    global _CLIENT_UPLOAD_BYTES
    _CLIENT_UPLOAD_BYTES = 0


def reset_diagnostic_profiles():
    """Reset legacy stdout diagnostic accumulators for one independent run."""
    _PROFILE_TIMES.clear()
    _PROFILE_COUNTS.clear()
    _COMM_BYTES.clear()
    _COMM_OPERATIONS.clear()
    _COMM_CALLS.clear()
    reset_client_upload_bytes()


@contextmanager
def _comm_profile(name):
    if not _COMM_PROFILE_ENABLED:
        yield
        return
    import crypten

    comm = crypten.communicator.get()
    b0 = getattr(comm, "comm_bytes", 0)
    r0 = getattr(comm, "comm_rounds", 0)
    try:
        yield
    finally:
        comm = crypten.communicator.get()
        _COMM_BYTES[name] = _COMM_BYTES.get(name, 0) + (getattr(comm, "comm_bytes", 0) - b0)
        _COMM_OPERATIONS[name] = _COMM_OPERATIONS.get(name, 0) + (
            getattr(comm, "comm_rounds", 0) - r0
        )
        _COMM_CALLS[name] = _COMM_CALLS.get(name, 0) + 1


def _comm_profile_report():
    if not _COMM_PROFILE_ENABLED or not _COMM_BYTES:
        return
    try:
        import crypten

        rank = crypten.communicator.get().get_rank()
    except Exception:
        rank = int(os.environ.get("RANK", "0"))
    total = sum(_COMM_BYTES.values())
    print(f"[pvc-comm-profile rank {rank}] comm bytes by region (attributed total={total}):", flush=True)
    for name, nbytes in sorted(_COMM_BYTES.items(), key=lambda kv: -kv[1]):
        pct = (100.0 * nbytes / total) if total else 0.0
        print(
            f"[pvc-comm-profile rank {rank}]   {name:26s} bytes={nbytes:14d} "
            f"{pct:5.1f}%  legacy_operations="
            f"{_COMM_OPERATIONS.get(name, 0):9d}  diagnostic_regions="
            f"{_COMM_CALLS.get(name, 0)}",
            flush=True,
        )


def _profile_report():
    if not _PROFILE_ENABLED or not _PROFILE_TIMES:
        return
    try:
        import crypten

        rank = crypten.communicator.get().get_rank()
    except Exception:
        rank = int(os.environ.get("RANK", "0"))
    total = sum(_PROFILE_TIMES.values())
    print(f"[pvc-profile rank {rank}] cumulative phase timings (total={total:.1f}s):", flush=True)
    for name, seconds in sorted(_PROFILE_TIMES.items(), key=lambda kv: -kv[1]):
        calls = _PROFILE_COUNTS.get(name, 0)
        pct = (100.0 * seconds / total) if total else 0.0
        print(
            f"[pvc-profile rank {rank}]   {name:32s} {seconds:9.2f}s "
            f"{pct:5.1f}%  calls={calls}",
            flush=True,
        )


def fixed_point_emission_matrix(emission, scale=DEFAULT_SCALE):
    """Convert log-emission floats to fixed-point integers for MPC scoring."""
    values = np.asarray(emission, dtype=np.float64)
    scaled = np.full(values.shape, NEG_INF_FIXED, dtype=np.float64)
    np.multiply(values, scale, out=scaled, where=np.isfinite(values))
    return np.rint(scaled).astype(np.int64)


def _matmul_public_integer_no_truncation(encrypted, public_coefficients):
    """Apply exact public integer coefficients without a truncation protocol.

    Heavy-mode probability lookups already return fixed-point values at the
    correct encoder scale. Its exact logaddexp emission coefficient matrices are
    strictly 0/1, so a share-local integer matmul preserves that scale exactly.
    """
    from crypten.mpc import MPCTensor

    # Multiplication by public integers is linear in the additive ring shares.
    return MPCTensor.from_shares(
        encrypted.share.matmul(public_coefficients.to(encrypted.share.device)),
        precision=encrypted.encoder._precision_bits,
    )


def score_haplotype_block_private(
    variant_indices,
    pair_list,
    fixed_emission_matrices,
    scale=DEFAULT_SCALE,
    reveal_scores=True,
):
    """Score public haplotype pairs with the required CrypTen backend."""
    return score_haplotype_block_crypten(
        variant_indices,
        pair_list,
        fixed_emission_matrices,
        scale=scale,
        reveal_scores=reveal_scores,
    )


def _resolve_genotype_device(torch):
    """Device for local encrypted-tensor compute (gather/sum).

    Controlled by the ``PVC_GENOTYPE_DEVICE`` env var (set per tool: the
    ``*-gpu`` tiers set it to ``"cuda"``). Communication (share creation and
    reveal) always happens on CPU, so only the local data-movement kernels run
    on this device. Defaults to CPU and never raises when CUDA is unavailable in
    the default (CPU) configuration.
    """
    name = os.environ.get("PVC_GENOTYPE_DEVICE", "cpu").strip().lower()
    if name in ("", "cpu"):
        return torch.device("cpu")
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"PVC_GENOTYPE_DEVICE={name!r} but torch.cuda.is_available() is "
                "False. Run on a GPU node with a CUDA-enabled torch build."
            )
        return torch.device(name)
    return torch.device(name)


def _score_pairs_vectorized(
    torch, encrypted_emissions, variant_indices, pair_list, device=None
):
    """Sum gathered encrypted emission cells across variants for all pairs.

    Equivalent to selecting ``emission[vi][h1[bi], h2[bi]]`` for every candidate
    pair and summing over the variants in the block, but batches each variant's
    per-pair selection into a single encrypted gather (public advanced index)
    instead of issuing one CrypTen index+add per (pair, variant). This collapses
    the inner loop from O(pairs * variants) encrypted ops to O(variants) gathers,
    which dominates wall-clock for large LD blocks. The gather is pure data
    movement (no arithmetic), so the revealed scores are bit-identical to the
    scalar reference (and device-independent: GPU gathers reorder the same
    encrypted shares, with no floating-point arithmetic involved).

    Returns a length-``len(pair_list)`` encrypted score vector, or ``None`` when
    there are no variants to score.
    """
    total = None
    for block_idx, vi in enumerate(variant_indices):
        emission = encrypted_emissions[vi]
        size = int(emission.size(-1))
        flat = emission.reshape(-1)
        flat_idx = torch.tensor(
            [int(h1[block_idx]) * size + int(h2[block_idx]) for h1, h2 in pair_list],
            dtype=torch.long,
            device=device,
        )
        gathered = flat[flat_idx]
        total = gathered if total is None else total + gathered
    return total


def _encrypt_and_gather_block(
    crypten, torch, variant_indices, pair_list, fixed_emission_matrices, device
):
    """Encrypt a whole block's emissions in ONE cryptensor call and score every
    candidate pair with ONE vectorized gather + sum.

    Replaces the per-variant pattern (one ``crypten.cryptensor`` and one gather
    per variant) which dominated wall-clock at scale (~355k tiny encryptions and
    per-variant Python index lists across a chromosome). All of the block's
    emission matrices are concatenated into one flat public array and encrypted
    once; the per-(pair, variant) cell indices are built with numpy and gathered
    in a single advanced index, then summed over the variant axis.

    Bit-identical to :func:`_score_pairs_vectorized`: the same encrypted cells
    are gathered and summed (integer ring adds, order-independent). Returns the
    encrypted ``(n_pairs,)`` score vector, or ``None`` if nothing to score.
    """
    if not variant_indices or not pair_list:
        return None

    n_pairs = len(pair_list)
    n_var = len(variant_indices)

    # Light/medium emissions are client plaintext.  Keep their packing, tensor
    # conversion, and one batched share-construction call out of the
    # computing-party scoring timer.  This matters for the paper breakdown:
    # those tiers do real client-side emission work in addition to k-mer
    # counting and output postprocessing.
    with profile_phase("emission_input_encoding"):
        # Concatenate this block's emission matrices into one flat public array
        # and record per-variant (offset, size) so cells can be addressed
        # globally.
        flats = []
        offsets = np.empty(n_var, dtype=np.int64)
        sizes = np.empty(n_var, dtype=np.int64)
        off = 0
        for i, vi in enumerate(variant_indices):
            m = np.asarray(fixed_emission_matrices[vi])
            s = int(m.shape[0])
            flats.append(m.reshape(-1).astype(np.float64))
            offsets[i] = off
            sizes[i] = s
            off += s * s
        flat_all = np.concatenate(flats)
        with _profile("cryptensor_encrypt"):
            enc_flat = crypten.cryptensor(
                torch.tensor(flat_all, dtype=torch.float64), src=0
            )
            _account_client_upload(
                enc_flat.nelement(),
                phase="emission_input_encoding",
                model=(
                    "one batched client upload containing the int64 fixed-point "
                    "emission shares for this haplotype block"
                ),
            )

    with profile_phase("haplotype_scoring"), _profile("gather_score"):
        if device.type != "cpu":
            enc_flat = enc_flat.to(device)
        # h1[p][i], h2[p][i] = allele indices of pair p at the i-th block variant.
        h1 = np.array([pair[0] for pair in pair_list], dtype=np.int64)  # (n_pairs, n_var)
        h2 = np.array([pair[1] for pair in pair_list], dtype=np.int64)  # (n_pairs, n_var)
        # global flat index of emission cell (h1, h2) for each (pair, variant)
        idx = offsets[None, :] + h1 * sizes[None, :] + h2  # (n_pairs, n_var)
        idx_t = torch.tensor(idx.reshape(-1), dtype=torch.long, device=device)
        gathered = enc_flat[idx_t]  # encrypted (n_pairs * n_var,)
        score_vector = gathered.reshape(n_pairs, n_var).sum(dim=1)  # (n_pairs,)
    return score_vector


def score_haplotype_block_crypten(
    variant_indices,
    pair_list,
    fixed_emission_matrices,
    scale=DEFAULT_SCALE,
    reveal_scores=True,
    crypten_root=None,
    return_encrypted=False,
):
    """Score public haplotype pairs with CrypTen arithmetic MPC tensors.

    This backend uses the vendored CrypTen source tree under ``crypto``. It
    expects PyTorch and CrypTen dependencies to be available in the Python
    environment. In the current algorithm, candidate haplotype pairs and allele
    indices remain public; read-derived emission matrices are encrypted.

    If reveal_scores is false, scores remain encrypted through argmax and only
    the winning candidate-pair index is revealed.

    If ``return_encrypted`` is true, the still-encrypted per-pair score vector
    (on CPU) is returned without any reveal, so the caller can concatenate many
    blocks' vectors and reveal them in one logical reduce call (see
    :func:`_reveal_encrypted_vectors_batched`). Returns ``None`` for empty
    blocks.
    """
    crypten, torch = _load_crypten(crypten_root)
    _init_crypten_if_needed(crypten)
    _progress_print(
        crypten,
        f"score block start: variants={len(variant_indices)}, pairs={len(pair_list)}, "
        f"reveal_scores={reveal_scores}",
    )

    # Shares are created via comm (broadcast from src=0) on CPU, then moved to
    # the compute device. The gloo backend only handles CPU tensors, so all
    # communication stays on CPU; only the local gather/sum runs on `device`.
    device = _resolve_genotype_device(torch)

    if not pair_list:
        return None if return_encrypted else (([], []) if reveal_scores else (0, None))

    t0 = time.perf_counter()
    score_vector = _encrypt_and_gather_block(
        crypten, torch, variant_indices, pair_list, fixed_emission_matrices, device
    )
    _progress_print(
        crypten,
        f"score block pairs {len(pair_list)}/{len(pair_list)} "
        f"elapsed={time.perf_counter() - t0:.1f}s device={device}",
    )

    if score_vector is None:
        return None if return_encrypted else (([], []) if reveal_scores else (0, None))

    # Move back to CPU for the communication-bound reveal/argmax (gloo).
    if device.type != "cpu":
        with profile_phase("haplotype_scoring"):
            score_vector = score_vector.to("cpu")

    if return_encrypted:
        # Defer reveal: caller batches many blocks into one reduce.
        return score_vector

    if not reveal_scores:
        best_idx = _private_argmax_index_vector(
            crypten, torch, score_vector, len(pair_list)
        )
        return best_idx, None

    _progress_print(crypten, f"revealing {len(pair_list)} scores to rank 0")
    scores_fixed = _reveal_score_vector_to_rank0(score_vector)
    account_client_transfer(
        "reveal",
        "download",
        len(pair_list) * 8,
        rounds=1,
        model="one int64 revealed score per candidate pair",
    )
    with profile_phase("reveal_postprocessing"):
        scores_float = [score / scale for score in scores_fixed]
    _progress_print(crypten, f"score block done: revealed_scores={len(scores_fixed)}")
    return scores_fixed, scores_float


def score_haplotype_block_encrypted_vector(
    variant_indices,
    pair_list,
    fixed_emission_matrices,
    scale=DEFAULT_SCALE,
    crypten_root=None,
):
    """Return the still-encrypted per-pair score vector for a block (no reveal).

    Thin wrapper around :func:`score_haplotype_block_crypten` used by the
    batched-reveal path. Returns ``None`` for empty blocks.
    """
    return score_haplotype_block_crypten(
        variant_indices,
        pair_list,
        fixed_emission_matrices,
        scale=scale,
        crypten_root=crypten_root,
        return_encrypted=True,
    )


def _reveal_encrypted_vectors_batched(vectors):
    """Reveal score vectors with one top-level reduce call.

    Concatenates the per-block encrypted vectors (a local share concatenation,
    no comm) and reveals the whole thing with one ``reduce`` to rank 0, instead
    of one reveal per block. Returns a list aligned with ``vectors``: each entry
    is the block's list of revealed fixed-point ints on rank 0, or ``None`` on
    non-output ranks. Entries that were ``None`` on input (empty blocks) stay
    ``None``.
    """
    import crypten

    with profile_phase("reveal"):
        present = [(i, v) for i, v in enumerate(vectors) if v is not None]
        if not present:
            return [None] * len(vectors)
        sizes = [int(v.size(0)) for _, v in present]
        combined = crypten.cat([v for _, v in present], dim=0)
        if combined._tensor.encoder.scale == 1:
            # Preserve exact integer scores. FixedPointEncoder.decode casts to
            # float32 and loses low bits above 2**24.
            revealed = combined.reveal(dst=0)
        else:
            revealed = combined.get_plain_text(dst=0)
    # Either reveal is one top-level reduce call for all blocks. This is not a
    # claim about transport-level or MPC dependency-round depth.
    if revealed is None:
        return [None] * len(vectors)  # non-output rank

    account_client_transfer(
        "reveal",
        "download",
        sum(sizes) * 8,
        rounds=1,
        model="one int64 revealed score per candidate pair in one chromosome batch",
    )

    with profile_phase("reveal_postprocessing"):
        # Vectorized round + a single C-level .tolist() instead of a Python loop
        # over every revealed score (millions of items per chromosome).
        import torch

        out = [None] * len(vectors)
        revealed = torch.round(revealed).to(torch.int64)
        offset = 0
        for (idx, _), size in zip(present, sizes):
            out[idx] = revealed[offset : offset + size].tolist()
            offset += size
    return out


def _compute_block_emissions_max_batched(
    crypten, torch, unique_kmers, variant_indices, sizes, probability_table, scale
):
    """Batched complete-mode exact logaddexp emissions for a whole block.

    Equivalent to calling :func:`_compute_encrypted_emission_scores_for_variant`
    with ``logaddexp_mode="max"`` once per variant, but constructs one batched set
    of private coverage/count selectors across every variant in the block. Those
    selectors are applied without truncation to both the main probability table
    and its precomputed exact logaddexp supplement. Concatenating variants' k-mers
    yields per-row-identical results to the per-variant path while amortizing the
    dominant selector-construction overhead.

    Returns ``{vi: encrypted (size, size) emission matrix}``.
    """
    from crypten.mpc.primitives import (
        ArithmeticSharedTensor,
        BinarySharedTensor,
    )

    with profile_phase("emission_computation"):
        cov_min = probability_table.cov_min
        cov_max = probability_table.cov_max
        count_max = probability_table.count_max
        cov_width = cov_max - cov_min
        table = np.rint(probability_table.probabilities * scale).astype(np.int64)
        lae_table = np.rint(probability_table.probabilities_lae * scale).astype(
            np.int64
        )
        lookup_method = os.environ.get(
            "PVC_HEAVY_LOOKUP_METHOD", "arithmetic-coverage-binary-count"
        ).strip().lower()
        if lookup_method not in {
            "binary-mixed-separate",
            "arithmetic-onehot-coverage-first",
            "arithmetic-coverage-binary-count",
        }:
            raise ValueError(
                f"Unsupported PVC_HEAVY_LOOKUP_METHOD={lookup_method!r}"
            )

    # The selectors are the heavy tier's client-private input.  Time their
    # plaintext encoding independently from the secure lookup that consumes
    # their shares.
    with profile_phase("emission_input_encoding"):
        count_sel_rows = []
        cov_sel_rows = []
        kmer_variant_indices = []
        var_info = []  # (vi, size, n_kmers, offset)
        offset = 0
        for block_idx, vi in enumerate(variant_indices):
            uk = unique_kmers[vi]
            size = sizes[block_idx]
            n_kmers = len(uk.kmer_to_count)
            var_info.append((vi, size, n_kmers, offset))
            cov = max(cov_min, min(int(uk.local_coverage), cov_max - 1))
            vs = torch.zeros((cov_width,), dtype=torch.long)
            vs[cov - cov_min] = 1
            cov_sel_rows.append(vs)
            if n_kmers == 0:
                continue
            cs = torch.zeros((n_kmers, count_max), dtype=torch.long)
            for k in range(n_kmers):
                c = max(0, min(int(uk.kmer_to_count[k]), count_max - 1))
                cs[k, c] = 1
            count_sel_rows.append(cs)
            kmer_variant_indices.extend([block_idx] * n_kmers)
            offset += n_kmers
        total = offset
        if total:
            count_selectors = torch.cat(count_sel_rows, dim=0)
            cov_selectors = torch.stack(cov_sel_rows, dim=0)
            kmer_variant_indices = torch.tensor(
                kmer_variant_indices, dtype=torch.long
            )
            if lookup_method == "arithmetic-coverage-binary-count":
                _account_hybrid_selector_upload(
                    cov_selectors.nelement(),
                    count_selectors.nelement(),
                )
            else:
                _account_client_upload(
                    count_selectors.nelement() + cov_selectors.nelement(),
                    phase="emission_input_encoding",
                    rounds=1,
                    model=(
                        "one batched client upload containing the arithmetic "
                        "count- and coverage-selector shares for this emission block"
                        if lookup_method == "arithmetic-onehot-coverage-first"
                        else "one batched client upload containing the int64 count- and "
                        "coverage-selector shares for this emission block"
                    ),
                )

    with profile_phase("emission_computation"):
        emissions = {}
        if total == 0:
            for vi, size, n_kmers, off in var_info:
                emissions[vi] = crypten.cryptensor(
                    torch.zeros(size * size, dtype=torch.float64), src=0
                ).view(size, size)
            return emissions

        if lookup_method in {
            "arithmetic-onehot-coverage-first",
            "arithmetic-coverage-binary-count",
        }:
            from crypten.mpc import MPCTensor

            cov_one_hot = ArithmeticSharedTensor(cov_selectors, src=0)
            combined_table = torch.tensor(
                np.concatenate([table, lae_table], axis=-1), dtype=torch.long
            )
            # Coverage selectors are private but the probability table is
            # public, so this selection is share-local and communication-free.
            per_variant = (
                cov_one_hot.view(len(variant_indices), 1, cov_width, 1)
                * combined_table.view(1, count_max, cov_width, 5)
            ).sum(dim=2)
            per_kmer = per_variant.index_select(0, kmer_variant_indices)
            if lookup_method == "arithmetic-onehot-coverage-first":
                count_one_hot = ArithmeticSharedTensor(count_selectors, src=0)
                # One upstream arithmetic multiplication performs count
                # selection for all five outputs in one exchange.
                selected = (
                    per_kmer * count_one_hot.view(total, count_max, 1)
                ).sum(dim=1)
            else:
                count_one_hot = BinarySharedTensor(count_selectors, src=0)
                selected = mixed_mul_no_truncation(
                    per_kmer,
                    count_one_hot.view(total, count_max, 1),
                    bits=1,
                ).sum(dim=1)
            if lookup_method == "arithmetic-onehot-coverage-first":
                probs_all = from_arithmetic(selected[..., :3])
                lae_all = from_arithmetic(selected[..., 3:])
            else:
                # mixed_mul_no_truncation already returns an MPCTensor; do not
                # wrap it a second time or downstream primitive access sees a
                # nested MPCTensor instead of an ArithmeticSharedTensor.
                probs_all = selected[..., :3]
                lae_all = selected[..., 3:]
        else:
            count_one_hot = BinarySharedTensor(count_selectors, src=0)
            cov_one_hot = BinarySharedTensor(cov_selectors, src=0)

            def batched_two_stage_lookup(table_values, n_cols):
                table_tensor = torch.tensor(
                    table_values.reshape(1, count_max, cov_width, n_cols),
                    dtype=torch.float64,
                )
                cov_selected = binary_times_public_no_truncation(
                    table_tensor,
                    cov_one_hot.view(len(variant_indices), 1, cov_width, 1),
                )
                per_variant = cov_selected.sum(dim=2)._tensor
                per_kmer = per_variant.index_select(0, kmer_variant_indices)
                count_selected = mixed_mul_no_truncation(
                    per_kmer,
                    count_one_hot.view(total, count_max, 1),
                    bits=1,
                )
                return count_selected.sum(dim=1)

            probs_all = batched_two_stage_lookup(table, 3)
            lae_all = batched_two_stage_lookup(lae_table, 2)

        for vi, size, n_kmers, off in var_info:
            if n_kmers == 0:
                emissions[vi] = crypten.cryptensor(
                    torch.zeros(size * size, dtype=torch.float64), src=0
                ).view(size, size)
                continue
            uk = unique_kmers[vi]
            probs = probs_all[off : off + n_kmers]
            coefficients, max_coefficients, constants = (
                _build_emission_max_coefficient_matrices(uk, size, scale)
            )
            coeff_tensor = torch.tensor(
                coefficients.reshape(size * size, n_kmers * 3).T,
                dtype=torch.long,
            )
            scores = _matmul_public_integer_no_truncation(
                probs.reshape(1, -1), coeff_tensor
            ).view(-1)
            if np.any(max_coefficients):
                lae_terms = lae_all[off : off + n_kmers]
                max_coeff_tensor = torch.tensor(
                    max_coefficients.reshape(size * size, n_kmers * 2).T,
                    dtype=torch.long,
                )
                scores = scores + _matmul_public_integer_no_truncation(
                    lae_terms.reshape(1, -1), max_coeff_tensor
                ).view(-1)
            scores = scores + torch.tensor(constants, dtype=torch.float64)
            emissions[vi] = scores.view(size, size)
        return emissions


def score_haplotype_block_private_complete(
    unique_kmers,
    probability_table,
    variant_indices,
    pair_list,
    scale=DEFAULT_SCALE,
    crypten_root=None,
    logaddexp_mode="max",
    return_encrypted=False,
):
    """Score a haplotype block without revealing intermediate emissions.

    This is the complete privacy level: private table lookup produces encrypted
    genotype-emission cells, public haplotype pairs select encrypted cells, and
    the candidate-pair argmax is computed privately. Rank 0 only receives the
    final best pair index.

    With ``return_encrypted=True`` the per-block argmax is skipped and the
    still-encrypted length-P score vector is returned instead (``None`` for an
    empty block), so that the run loop can batch every block of the chromosome
    into the bucketed secure argmax (:func:`_private_argmax_index_vector_batch`),
    exactly as PVC medium does. The score vector is identical either way.
    """
    crypten, torch = _load_crypten(crypten_root)
    _init_crypten_if_needed(crypten)
    _require_three_party(crypten)
    _progress_print(
        crypten,
        f"complete score block start: variants={len(variant_indices)}, "
        f"pairs={len(pair_list)}, logaddexp_mode={logaddexp_mode}",
    )

    if not pair_list:
        return None if return_encrypted else (0, None)

    t0 = time.perf_counter()
    with profile_phase("emission_computation"):
        # Per-variant emission sizes depend only on public allele/block data.
        sizes = []
        for block_idx, vi in enumerate(variant_indices):
            unique_kmer = unique_kmers[vi]
            max_allele = max(
                max(int(h1[block_idx]), int(h2[block_idx]))
                for h1, h2 in pair_list
            )
            sizes.append(max(len(unique_kmer.alleles), max_allele + 1, 1))

    # The per-variant path uses the fast two-stage oblivious lookup; cross-
    # variant batching only amortized framework overhead (~1.1x), so it is opt-in.
    use_batched = (
        logaddexp_mode == "max"
        and os.environ.get("PVC_BATCH_EMISSION", "0") != "0"
    )
    if use_batched:
        encrypted_emissions = _compute_block_emissions_max_batched(
            crypten, torch, unique_kmers, variant_indices, sizes, probability_table, scale
        )
        _progress_print(
            crypten,
            f"complete score block emissions batched variants={len(variant_indices)} "
            f"elapsed={time.perf_counter() - t0:.1f}s",
        )
    else:
        encrypted_emissions = {}
        variant_interval = max(1, len(variant_indices) // 10) if variant_indices else 1
        for block_idx, vi in enumerate(variant_indices):
            unique_kmer = unique_kmers[vi]
            size = sizes[block_idx]
            encrypted_scores = _compute_encrypted_emission_scores_for_variant(
                crypten,
                torch,
                unique_kmer,
                probability_table,
                size,
                scale,
                logaddexp_mode=logaddexp_mode,
            )
            with profile_phase("emission_computation"):
                encrypted_emissions[vi] = encrypted_scores.view(size, size)
            if (block_idx + 1) % variant_interval == 0 or block_idx + 1 == len(variant_indices):
                _progress_print(
                    crypten,
                    f"complete score block encrypted variants {block_idx + 1}/{len(variant_indices)} "
                    f"elapsed={time.perf_counter() - t0:.1f}s",
                )

    with profile_phase("haplotype_scoring"):
        score_vector = _score_pairs_vectorized(
            torch, encrypted_emissions, variant_indices, pair_list
        )
    _progress_print(
        crypten,
        f"complete score block pairs {len(pair_list)}/{len(pair_list)} "
        f"elapsed={time.perf_counter() - t0:.1f}s",
    )

    if return_encrypted:
        return score_vector
    return _private_argmax_index_vector(crypten, torch, score_vector, len(pair_list)), None


def score_haplotype_block_private_complete_encrypted_vector(
    unique_kmers,
    probability_table,
    variant_indices,
    pair_list,
    scale=DEFAULT_SCALE,
    crypten_root=None,
    logaddexp_mode="max",
):
    """Heavy-tier counterpart of :func:`score_haplotype_block_encrypted_vector`.

    Returns the block's still-encrypted score vector (``None`` for an empty
    block) without performing the per-block argmax.
    """
    return score_haplotype_block_private_complete(
        unique_kmers,
        probability_table,
        variant_indices,
        pair_list,
        scale=scale,
        crypten_root=crypten_root,
        logaddexp_mode=logaddexp_mode,
        return_encrypted=True,
    )


def _private_argmax_index(crypten, torch, encrypted_scores):
    if not encrypted_scores:
        return 0
    score_vector = crypten.stack(encrypted_scores)
    return _private_argmax_index_vector(
        crypten, torch, score_vector, len(encrypted_scores)
    )


def _private_argmax_index_vector(crypten, torch, score_vector, n_scores):
    """Reveal the first winning index using integer-only private comparisons.

    A comparison tournament finds the maximum; an encrypted prefix sum breaks
    ties without multiplying the scores by the candidate count.
    """
    if score_vector is None or n_scores == 0:
        return 0

    _require_three_party(crypten)
    _progress_print(crypten, f"private argmax start: scores={n_scores}")
    with _comm_profile("argmax_per_block"):
        # Avoid upstream random tie sampling at encoder precision 0.
        with profile_phase("argmax"):
            one_hot = first_winner_one_hot(score_vector)
        with profile_phase("reveal"):
            revealed = one_hot.get_plain_text(dst=0)
    account_client_transfer(
        "reveal",
        "download",
        n_scores * 8,
        rounds=1,
        model=(
            "implemented secure argmax reveals one int64 one-hot value per "
            "candidate score, then rank 0 decodes the winning index"
        ),
    )
    if revealed is None:
        return 0
    with profile_phase("reveal_postprocessing"):
        best_idx = int(torch.argmax(revealed).item())
    _progress_print(crypten, f"private argmax done: best_pair_index={best_idx}")
    return best_idx


def _private_argmax_index_vector_batch(crypten, torch, score_vectors):
    """Group score vectors by width and reveal a first-winner mask per row.

    Short rows are padded with their first value. A private prefix sum selects
    only the first maximum, so a padded copy never displaces a real candidate.
    """
    out = [0] * len(score_vectors)
    present = [
        (i, v)
        for i, v in enumerate(score_vectors)
        if v is not None and int(v.size(0)) > 0
    ]
    if not present:
        return out
    _require_three_party(crypten)

    buckets = {}
    for i, v in present:
        n = int(v.size(0))
        width = 1 << max(1, (n - 1).bit_length())  # next power of two, >= 2
        buckets.setdefault(width, []).append((i, v))

    _progress_print(
        crypten,
        f"batched argmax: {len(present)} blocks in {len(buckets)} width buckets",
    )
    for width, items in sorted(buckets.items()):
        with profile_phase("argmax"):
            rows = []
            real_lens = []
            for _, v in items:
                n = int(v.size(0))
                real_lens.append(n)
                if n < width:
                    # Pad with an in-range copy of the row's own first element.
                    v = crypten.cat([v, v[0:1].expand(width - n)], dim=0)
                rows.append(v.unsqueeze(0))
            mat = crypten.cat(rows, dim=0)  # (n_blk, width) encrypted
            one_hot = first_winner_one_hot(mat, dim=1)
        with profile_phase("reveal"):
            revealed = one_hot.get_plain_text(dst=0)  # ONE reveal for the bucket
        account_client_transfer(
            "reveal",
            "download",
            len(items) * width * 8,
            rounds=1,
            model=(
                "implemented batched secure argmax reveals a padded int64 "
                "one-hot row per block in the width bucket, then rank 0 "
                "decodes winning indices"
            ),
        )
        if revealed is None:
            continue
        with profile_phase("reveal_postprocessing"):
            # Zero the pad columns (free public mask): in-range padding keeps
            # row_max == the block's real max; this is a defensive public mask.
            col = torch.arange(width)
            pad_mask = (col[None, :] < torch.tensor(real_lens)[:, None]).to(
                revealed.dtype
            )
            masked = revealed * pad_mask  # may contain multiple 1s on exact ties
            # Tie-break among equal-max columns (env PVC_TIEBREAK; default first).
            _tb = os.environ.get("PVC_TIEBREAK", "first")
            if _tb == "last":
                best = width - 1 - torch.argmax(masked.flip(1), dim=1)
            elif _tb == "random":
                noise = torch.rand(masked.shape) * (masked > 0).to(masked.dtype)
                best = torch.argmax(masked + 0.5 * noise, dim=1)
            else:  # "first"
                best = torch.argmax(masked, dim=1)
            for row, (i, _) in enumerate(items):
                out[i] = int(best[row].item())
    return out

def _reveal_score_vector_to_rank0(score_vector):
    """Reveal a length-P encrypted score vector to the client (rank 0)."""
    # ``FixedPointEncoder.decode`` converts reconstructed int64 shares to
    # float32. At precision 0, chromosome-scale scores can exceed 2**24, so
    # that conversion discards low integer bits and can change the winner of a
    # close plaintext argmax. Reconstruct the ring integers directly whenever
    # the encoder scale is one.
    encoder = score_vector._tensor.encoder
    with profile_phase("reveal"):
        if encoder.scale == 1:
            revealed = score_vector.reveal(dst=0)
        else:
            revealed = score_vector.get_plain_text(dst=0)
    # For real multi-party use, non-client parties receive None here.
    if revealed is None:
        return []
    with profile_phase("reveal_postprocessing"):
        if encoder.scale == 1:
            return [int(value) for value in revealed.reshape(-1).tolist()]
        return [
            int(round(float(revealed[i].item())))
            for i in range(revealed.numel())
        ]


def _reveal_scores_to_rank0(encrypted_scores, torch):
    scores_fixed = []
    for score in encrypted_scores:
        encoder = score._tensor.encoder
        if encoder.scale == 1:
            revealed = score.reveal(dst=0)
            if revealed is None:
                continue
            scores_fixed.append(int(revealed.item()))
            continue
        revealed = score.get_plain_text(dst=0)
        # For real multi-party use, non-client parties receive None here.
        if revealed is None:
            continue
        scores_fixed.append(int(round(float(revealed.item()))))
    return scores_fixed


def _is_output_rank():
    return os.environ.get("RANK", "0") == "0"


def _compute_fixed_emission_matrices(unique_kmers, chrom_probs, scale):
    """Client-side conversion of read-derived emissions to fixed-point ints."""
    fixed_matrices = []
    for i, unique_kmer in enumerate(unique_kmers):
        n_alleles = len(unique_kmer.alleles)
        if n_alleles == 0 or i >= len(chrom_probs) or len(chrom_probs[i]) == 0:
            shape = (max(n_alleles, 1), max(n_alleles, 1))
            fixed_matrices.append(np.zeros(shape, dtype=np.int64))
            continue

        emission = compute_emission_matrix(unique_kmer, chrom_probs[i], n_alleles)
        fixed_matrices.append(fixed_point_emission_matrix(emission, scale=scale))

    return fixed_matrices


def _ensure_fixed_emissions_cover_pairs(
    unique_kmers,
    chrom_probs,
    fixed_matrices,
    variant_indices,
    pair_list,
    scale,
):
    """Expand fixed-point emissions for public haplotype alleles if needed."""
    for block_idx, vi in enumerate(variant_indices):
        max_allele = -1
        for h1, h2 in pair_list:
            max_allele = max(max_allele, int(h1[block_idx]), int(h2[block_idx]))

        if max_allele < 0 or max_allele < fixed_matrices[vi].shape[0]:
            continue

        size = max_allele + 1
        prob_vector = chrom_probs[vi] if vi < len(chrom_probs) else []
        expanded = np.empty((size, size), dtype=np.float64)
        for a1 in range(size):
            for a2 in range(size):
                expanded[a1, a2] = compute_emission_for_genotype(
                    unique_kmers[vi],
                    prob_vector,
                    a1,
                    a2,
                )
        fixed_matrices[vi] = fixed_point_emission_matrix(expanded, scale=scale)


def prepare_public_fixed_emissions(unique_kmers, chrom_probs, private_scale):
    with profile_phase("emission_computation"):
        return _compute_fixed_emission_matrices(
            unique_kmers,
            chrom_probs,
            private_scale,
        )


def ensure_fixed_emissions_cover_pairs(
    unique_kmers,
    chrom_probs,
    fixed_matrices,
    variant_indices,
    pair_list,
    private_scale,
):
    with profile_phase("emission_computation"), _profile("emission_prep"):
        _ensure_fixed_emissions_cover_pairs(
            unique_kmers,
            chrom_probs,
            fixed_matrices,
            variant_indices,
            pair_list,
            private_scale,
        )


def private_block_score(best_pair_idx=None, scores_fixed=None, scores_float=None):
    return {
        "best_pair_idx": best_pair_idx,
        "scores_fixed": scores_fixed,
        "scores_float": scores_float,
    }


def run_private_haplotype_genotyping(
    unique_kmers_map,
    prob_vectors,
    panel_vcf,
    private_method,
    prepare_context,
    score_block,
    verbose=False,
    blocks_file=None,
    ir: GenotypingPlan | None = None,
    private_scale=DEFAULT_SCALE,
    probability_table=None,
    private_scores_output=None,
    score_block_encrypted=None,
    batched_argmax=False,
):
    """Shared haplotype-genotyping loop for private PVC methods.

    When ``score_block_encrypted`` is provided (the revealed-score "fast"/light
    tiers) and ``PVC_BATCH_REVEAL`` is not "0", the per-block reveals are batched:
    every block's encrypted score vector is computed first (no comm), then the
    whole chromosome is revealed in one top-level ``reduce`` call instead of
    one call per block. Results are bit-identical (reveal just sums shares).
    """
    results = {}
    score_records = []
    with profile_phase("frequency_vector_setup"):
        freq_vectors = compute_allele_frequency_vectors(unique_kmers_map)
    # ``batched_argmax`` marks a hidden-score (argmax) tier (medium/heavy): it
    # may use the batched argmax, and when that is disabled it falls back to the
    # PER-BLOCK private argmax -- never to batched reveal (which would leak the
    # scores). Reveal-score tiers (light) use the batched reveal instead.
    is_argmax_tier = batched_argmax
    use_batched_argmax = (
        score_block_encrypted is not None
        and is_argmax_tier
        and os.environ.get("PVC_BATCH_ARGMAX", "1") != "0"
    )
    use_batched_reveal = (
        score_block_encrypted is not None
        and not is_argmax_tier
        and os.environ.get("PVC_BATCH_REVEAL", "1") != "0"
    )

    for chrom in unique_kmers_map.unique_kmers:
        unique_kmers = unique_kmers_map.unique_kmers[chrom]
        chrom_probs = prob_vectors.get(chrom, [])
        chrom_freqs = freq_vectors.get(chrom, [])
        n_variants = len(unique_kmers)

        with profile_phase("result_buffer_setup"):
            chrom_results = [GenotypingResult() for _ in range(n_variants)]

            if ir is not None:
                block_assignments = []
                for superblock in ir.superblocks_by_chrom.get(chrom, []):
                    block_assignments.extend(
                        block_assignments_from_ir_blocks(ir.blocks_for_superblock(superblock))
                    )
                singleton_indices = []
                print(f"  IR private blocks: {len(block_assignments)}")
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    blocks = run_plink_blocks(panel_vcf, chrom, tmpdir, blocks_file=blocks_file)

                block_assignments, singleton_indices = assign_variants_to_blocks(
                    blocks,
                    unique_kmers,
                )

        n_in_blocks = sum(len(indices) for _, indices in block_assignments)
        print(f"  Variants in {len(block_assignments)} private-score blocks: {n_in_blocks}")
        print(f"  Singleton variants: {len(singleton_indices)}")

        print("  Precomputing public full-panel haplotype pair tables...")
        with profile_phase("pair_table_setup"), _profile("precompute_pair_tables"):
            block_tables, n_precomputed = precompute_haplotype_pair_table(
                unique_kmers,
                block_assignments,
            )
        print(f"  Precomputed tables for {n_precomputed} blocks")

        print(
            f"  Preparing fixed-point emissions "
            f"(method={private_method}, scale={private_scale})..."
        )
        with _profile("prepare_context"):
            context = prepare_context(
                unique_kmers=unique_kmers,
                chrom_probs=chrom_probs,
                block_assignments=block_assignments,
                n_variants=n_variants,
                probability_table=probability_table,
                private_scale=private_scale,
            )

        blocks_genotyped = 0
        _tie_rng = np.random.default_rng(0)
        # Tie-break strategy for the light (revealed-score) path only. On exact
        # ties at the max score, PVC_TIEBREAK selects: current/first index,
        # last (alternative deterministic), random among tied, or missing
        # (leave the block's variants no-call). Diagnostic to see how much of the
        # accuracy is decided by an arbitrary tie choice (~30% of blocks tie).
        _light_tiebreak = os.environ.get("PVC_TIEBREAK", "current")

        def _assign_block(
            block_id, bp1, bp2, variant_indices, pair_list,
            scores_fixed, scores_float, best_pair_idx,
        ):
            """Argmax + genotype assignment from a block's revealed scores."""
            nonlocal blocks_genotyped
            if _is_output_rank():
                if best_pair_idx is None:
                    if not scores_fixed:
                        return  # no revealed scores (e.g. variantless block)
                    with profile_phase("argmax"):
                        scores = np.asarray(scores_fixed)
                        tied = np.flatnonzero(scores == scores.max())
                        if tied.size == 1 or _light_tiebreak in ("current", "first"):
                            best_pair_idx = int(tied[0])
                        elif _light_tiebreak in ("alternative", "last"):
                            best_pair_idx = int(tied[-1])
                        elif _light_tiebreak == "random":
                            best_pair_idx = int(_tie_rng.choice(tied))
                        elif _light_tiebreak == "missing":
                            blocks_genotyped += 1  # ambiguous -> leave variants no-call
                            return
                        else:
                            best_pair_idx = int(tied[0])
                else:
                    best_pair_idx = int(best_pair_idx)

                with profile_phase("genotype_reconstruction"):
                    best_h1, best_h2 = pair_list[best_pair_idx]
                    for block_idx, vi in enumerate(variant_indices):
                        a1, a2 = best_h1[block_idx], best_h2[block_idx]
                        result = chrom_results[vi]
                        result.set_coverage(int(unique_kmers[vi].local_coverage))
                        result.set_unique_kmers(
                            len(chrom_probs[vi]) if vi < len(chrom_probs) else 0
                        )
                        result.add_to_likelihood(a1, a2, 1.0)
                        result.normalize()
            else:
                best_pair_idx = None

            blocks_genotyped += 1

            if private_scores_output and _is_output_rank():
                with profile_phase("private_score_artifact_output"):
                    score_records.append({
                        "chrom": chrom,
                        "block_id": block_id,
                        "start": bp1,
                        "end": bp2,
                        "variant_indices": list(map(int, variant_indices)),
                        "scores_fixed": (
                            list(map(int, scores_fixed))
                            if scores_fixed is not None and best_pair_idx is not None
                            else None
                        ),
                        "scores": scores_float,
                        "best_pair_index": best_pair_idx,
                    })

            if verbose and blocks_genotyped <= 3:
                print(
                    f"    Private block {bp1}-{bp2}: {len(variant_indices)} vars, "
                    f"{len(pair_list)} pairs, best_pair_index={best_pair_idx}"
                )

        if use_batched_reveal or use_batched_argmax:
            # PASS 1: compute every block's encrypted score vector (no reveal).
            pending = []  # (block_id, bp1, bp2, variant_indices, pair_list)
            enc_vectors = []
            for block_id, (((bp1, bp2), variant_indices), (_, pair_list)) in enumerate(
                zip(block_assignments, block_tables)
            ):
                if not pair_list:
                    singleton_indices.extend(variant_indices)
                    continue
                enc_vectors.append(
                    score_block_encrypted(
                        unique_kmers=unique_kmers,
                        chrom_probs=chrom_probs,
                        context=context,
                        variant_indices=variant_indices,
                        pair_list=pair_list,
                        private_scale=private_scale,
                        probability_table=probability_table,
                    )
                )
                pending.append((block_id, bp1, bp2, variant_indices, pair_list))

            if use_batched_argmax:
                # Hidden-score tiers (medium/heavy): reveal ONLY the per-block
                # argmax index, batched into ~O(width-buckets) exchange groups
                # for the chromosome instead of one operation set per block.
                import crypten
                import torch

                print(
                    f"  Batched argmax: {len(enc_vectors)} blocks "
                    f"(bucketed, ~O(buckets) exchange groups)"
                )
                with _profile("batched_argmax"), _comm_profile("batched_argmax"):
                    best_indices = _private_argmax_index_vector_batch(
                        crypten, torch, enc_vectors
                    )
                with _profile("assign_genotypes"):
                    for (block_id, bp1, bp2, variant_indices, pair_list), best_idx in zip(
                        pending, best_indices
                    ):
                        _assign_block(
                            block_id, bp1, bp2, variant_indices, pair_list,
                            None, None, best_idx,
                        )
            else:
                # One top-level reduce call for the whole chromosome.
                print(
                    f"  Batched reveal: {len(enc_vectors)} blocks in one "
                    "logical reduce call"
                )
                with _profile("batched_reveal"):
                    revealed = _reveal_encrypted_vectors_batched(enc_vectors)

                # PASS 2: argmax + genotype assignment from the revealed scores.
                with _profile("assign_genotypes"):
                    for (block_id, bp1, bp2, variant_indices, pair_list), scores_fixed in zip(
                        pending, revealed
                    ):
                        with profile_phase("reveal_postprocessing"):
                            scores_float = (
                                [s / private_scale for s in scores_fixed]
                                if scores_fixed is not None
                                else None
                            )
                        _assign_block(
                            block_id, bp1, bp2, variant_indices, pair_list,
                            scores_fixed, scores_float, None,
                        )
        else:
            for block_id, (((bp1, bp2), variant_indices), (_, pair_list)) in enumerate(
                zip(block_assignments, block_tables)
            ):
                if not pair_list:
                    singleton_indices.extend(variant_indices)
                    continue

                block_score = score_block(
                    unique_kmers=unique_kmers,
                    chrom_probs=chrom_probs,
                    context=context,
                    variant_indices=variant_indices,
                    pair_list=pair_list,
                    private_scale=private_scale,
                    probability_table=probability_table,
                )
                _assign_block(
                    block_id, bp1, bp2, variant_indices, pair_list,
                    block_score.get("scores_fixed"),
                    block_score.get("scores_float"),
                    block_score.get("best_pair_idx"),
                )

        print(f"  Private-score blocks genotyped: {blocks_genotyped}")

        n_singleton_called = 0
        with profile_phase("singleton_calling"), _profile("singletons"):
          for i in singleton_indices:
            if not _is_output_rank():
                continue

            unique_kmer = unique_kmers[i]
            result = chrom_results[i]
            n_alleles = len(unique_kmer.alleles)

            if n_alleles == 0 or i >= len(chrom_probs) or len(chrom_probs[i]) == 0:
                continue

            prob_vector = chrom_probs[i]
            freq_vector = chrom_freqs[i] if i < len(chrom_freqs) else []

            emission = compute_emission_matrix(unique_kmer, prob_vector, n_alleles)
            frequency = compute_frequency_matrix(freq_vector, n_alleles)
            log_posterior = emission + frequency
            posterior = normalize_log_matrix(log_posterior)

            result.set_coverage(int(unique_kmer.local_coverage))
            result.set_unique_kmers(len(prob_vector))

            for a1 in range(n_alleles):
                for a2 in range(n_alleles):
                    prob = posterior[a1, a2]
                    if prob > 1e-10:
                        result.add_to_likelihood(a1, a2, prob)

            result.normalize()
            n_singleton_called += 1

        n_total_called = sum(1 for r in chrom_results if not r.contains_no_likelihoods())
        print(f"  Singletons genotyped locally: {n_singleton_called}")
        print(f"  Total genotyped: {n_total_called}/{n_variants}")
        results[chrom] = chrom_results

    if private_scores_output and _is_output_rank():
        with profile_phase("private_score_artifact_output"):
            with open(private_scores_output, "w", encoding="utf-8") as score_file:
                json.dump(
                    {
                        "scale": private_scale,
                        "private_method": private_method,
                        "records": score_records,
                    },
                    score_file,
                )
        print(f"  Wrote client-side private scores to {private_scores_output}")

    _profile_report()
    return results


def compute_fixed_emission_matrices_private(
    unique_kmers,
    probability_table,
    scale=DEFAULT_SCALE,
    crypten_root=None,
    variant_indices_to_compute=None,
    logaddexp_mode="linear",
):
    """Build fixed-point emission matrices with private probability lookup.

    The client provides one-hot encoded private selectors for read count and
    average coverage. CrypTen then performs mixed multiplication against the
    public probability table. This avoids private equality circuits for every
    lookup.

    The fixed emission matrices are revealed at the end of this phase for
    compatibility with the existing block-scoring pipeline. In ``full`` mode,
    the downstream candidate-pair argmax remains private and only the best index
    is revealed.
    """
    crypten, torch = _load_crypten(crypten_root)
    _init_crypten_if_needed(crypten)

    n_variants = len(unique_kmers)
    if variant_indices_to_compute is None:
        variant_indices = list(range(n_variants))
    else:
        variant_indices = sorted({int(i) for i in variant_indices_to_compute})

    total_kmers = sum(len(unique_kmers[i].kmer_to_count) for i in variant_indices)
    total_cells = sum(max(len(unique_kmers[i].alleles), 1) ** 2 for i in variant_indices)
    total_lookup_ops = sum(
        max(len(unique_kmers[i].alleles), 1) ** 2 * len(unique_kmers[i].kmer_to_count)
        for i in variant_indices
    )
    _progress_print(
        crypten,
        f"private emission start: variants_to_compute={len(variant_indices)}/{n_variants}, "
        f"kmers={total_kmers}, "
        f"genotype_cells={total_cells}, genotype_kmer_lookups={total_lookup_ops}, "
        f"logaddexp_mode={logaddexp_mode}",
    )

    fixed_matrices = [
        np.zeros((max(len(unique_kmer.alleles), 1), max(len(unique_kmer.alleles), 1)), dtype=np.int64)
        for unique_kmer in unique_kmers
    ]
    progress_interval = 1 if len(variant_indices) <= 200 else max(1, len(variant_indices) // 100)
    t0 = time.perf_counter()
    for progress_idx, variant_idx in enumerate(variant_indices):
        unique_kmer = unique_kmers[variant_idx]
        n_alleles = len(unique_kmer.alleles)
        size = max(n_alleles, 1)
        n_kmers = len(unique_kmer.kmer_to_count)
        lookup_ops = size * size * n_kmers
        should_log = (
            progress_idx == 0
            or progress_idx + 1 == len(variant_indices)
            or progress_idx % progress_interval == 0
            or lookup_ops >= 5000
        )
        if should_log:
            _progress_print(
                crypten,
                f"private emission variant {progress_idx + 1}/{len(variant_indices)} "
                f"(global_index={variant_idx}) start: "
                f"alleles={n_alleles}, kmers={n_kmers}, genotype_cells={size * size}, "
                f"lookup_ops={lookup_ops}",
            )
        if n_alleles == 0 or len(unique_kmer.kmer_to_count) == 0:
            if should_log:
                _progress_print(
                    crypten,
                    f"private emission variant {progress_idx + 1}/{len(variant_indices)} "
                    f"(global_index={variant_idx}) skipped empty",
                )
            continue

        variant_t0 = time.perf_counter()
        matrix = _compute_encrypted_emission_matrix_for_variant(
            crypten,
            torch,
            unique_kmer,
            probability_table,
            size,
            scale,
            logaddexp_mode=logaddexp_mode,
        )
        fixed_matrices[variant_idx] = matrix
        if should_log:
            elapsed = time.perf_counter() - t0
            variant_elapsed = time.perf_counter() - variant_t0
            _progress_print(
                crypten,
                f"private emission variant {progress_idx + 1}/{len(variant_indices)} "
                f"(global_index={variant_idx}) done: "
                f"variant_elapsed={variant_elapsed:.1f}s total_elapsed={elapsed:.1f}s",
            )

    _progress_print(
        crypten,
        f"private emission done: variants={len(fixed_matrices)}, "
        f"elapsed={time.perf_counter() - t0:.1f}s",
    )
    return fixed_matrices


def _compute_encrypted_emission_matrix_for_variant(
    crypten,
    torch,
    unique_kmer,
    probability_table,
    size,
    scale,
    logaddexp_mode="linear",
):
    """Compute and reveal all genotype-cell emissions for one variant."""
    encrypted_scores = _compute_encrypted_emission_scores_for_variant(
        crypten,
        torch,
        unique_kmer,
        probability_table,
        size,
        scale,
        logaddexp_mode=logaddexp_mode,
    )

    revealed = encrypted_scores.get_plain_text(dst=0)
    if revealed is None:
        return np.zeros((size, size), dtype=np.int64)
    return np.rint(revealed.detach().cpu().numpy()).astype(np.int64).reshape(size, size)


def _compute_encrypted_emission_scores_for_variant(
    crypten,
    torch,
    unique_kmer,
    probability_table,
    size,
    scale,
    logaddexp_mode="linear",
):
    """Compute encrypted genotype-cell emissions for one variant.

    The private table lookup is batched across all k-mers once. Public allele
    membership is then represented as coefficients over CN=0/1/2, so every
    genotype cell can be scored with one encrypted-public matrix multiply.
    """
    n_kmers = len(unique_kmer.kmer_to_count)
    if n_kmers == 0:
        with profile_phase("emission_computation"):
            return crypten.cryptensor(
                torch.zeros(size * size, dtype=torch.float64), src=0
            )

    encrypted_probs = _private_probability_lookup_batch_fixed(
        crypten,
        torch,
        probability_table,
        int(unique_kmer.local_coverage),
        unique_kmer.kmer_to_count,
        scale,
    )

    if logaddexp_mode == "linear":
        with profile_phase("emission_computation"):
            coefficients, constants = _build_emission_coefficient_matrix(
                unique_kmer, size, scale
            )
            coeff_tensor = torch.tensor(
                coefficients.reshape(size * size, n_kmers * 3).T,
                dtype=torch.float64,
            )
            encrypted_scores = encrypted_probs.view(1, -1).matmul(
                coeff_tensor
            ).view(-1)
            encrypted_scores = encrypted_scores + torch.tensor(
                constants, dtype=torch.float64
            )
    elif logaddexp_mode == "max":
        encrypted_scores = _compute_encrypted_emission_scores_max_logaddexp(
            crypten,
            torch,
            encrypted_probs,
            unique_kmer,
            size,
            scale,
            probability_table,
        )
    else:
        raise ValueError(f"Unsupported private logaddexp mode: {logaddexp_mode}")

    return encrypted_scores


def _compute_encrypted_emission_scores_max_logaddexp(
    crypten,
    torch,
    encrypted_probs,
    unique_kmer,
    size,
    scale,
    probability_table,
):
    """Build emissions for undefined-allele genotypes using EXACT logaddexp.

    For undefined-allele genotypes, plaintext PVC uses
    ``logaddexp(log_p_low, log_p_high) + log(0.5)``. This used to be approximated
    privately with a secure ``max`` (cheap in plaintext, but a comparison in MPC
    that dominated communication at scale). Instead we look up the EXACT
    logaddexp term from ``probability_table.probabilities_lae`` (precomputed in
    plaintext at table-build time) via the same cheap two-stage oblivious
    selection used for the main table, keyed on the same private (count,
    coverage). The 01-vs-12 choice per genotype cell is public
    (``max_coefficients``), so this leaks nothing new -- and it is both more
    accurate (true logaddexp, not max) and far less communication (a selection,
    not a secure comparison).
    """
    with profile_phase("emission_computation"):
        n_kmers = len(unique_kmer.kmer_to_count)
        coefficients, max_coefficients, constants = (
            _build_emission_max_coefficient_matrices(
                unique_kmer,
                size,
                scale,
            )
        )

        coeff_tensor = torch.tensor(
            coefficients.reshape(size * size, n_kmers * 3).T,
            dtype=torch.long,
        )
        encrypted_scores = _matmul_public_integer_no_truncation(
            encrypted_probs.view(1, -1), coeff_tensor
        ).view(-1)

    if np.any(max_coefficients):
        # Oblivious lookup of [lae01, lae12] per k-mer -- replaces the two secure
        # max ops with one two-stage selection into the supplementary table.
        encrypted_lae_terms = _private_probability_lookup_batch_fixed(
            crypten,
            torch,
            probability_table,
            int(unique_kmer.local_coverage),
            unique_kmer.kmer_to_count,
            scale,
            table_values=probability_table.probabilities_lae,
            n_cols=2,
            profile_label="logaddexp_lookup",
        )  # (n_kmers, 2) = [lae01, lae12], matching the max_coefficients columns
        with profile_phase("emission_computation"):
            max_coeff_tensor = torch.tensor(
                max_coefficients.reshape(size * size, n_kmers * 2).T,
                dtype=torch.long,
            )
            encrypted_scores = (
                encrypted_scores
                + _matmul_public_integer_no_truncation(
                    encrypted_lae_terms.view(1, -1), max_coeff_tensor
                ).view(-1)
            )

    with profile_phase("emission_computation"):
        return encrypted_scores + torch.tensor(constants, dtype=torch.float64)


def _build_emission_max_coefficient_matrices(unique_kmer, size, scale):
    n_kmers = len(unique_kmer.kmer_to_count)
    coefficients = np.zeros((size * size, n_kmers, 3), dtype=np.float64)
    max_coefficients = np.zeros((size * size, n_kmers, 2), dtype=np.float64)
    constants = np.zeros(size * size, dtype=np.float64)

    allele_ids = set(unique_kmer.alleles.keys())
    fallback = round(np.log(1.0 / 3.0) * scale)
    log_half = round(np.log(0.5) * scale)

    for a1 in range(size):
        a1_valid = a1 in allele_ids
        a1_undef = unique_kmer.is_undefined_allele(a1) if a1_valid else False
        for a2 in range(size):
            cell_idx = a1 * size + a2
            a2_valid = a2 in allele_ids
            a2_undef = unique_kmer.is_undefined_allele(a2) if a2_valid else False

            for kmer_idx in range(n_kmers):
                if not a1_valid or not a2_valid or (a1_undef and a2_undef):
                    constants[cell_idx] += fallback
                    continue

                cn_from_a1 = unique_kmer.kmer_on_allele(kmer_idx, a1) if not a1_undef else 0
                cn_from_a2 = unique_kmer.kmer_on_allele(kmer_idx, a2) if not a2_undef else 0
                if a1_undef or a2_undef:
                    cn_known = cn_from_a1 if not a1_undef else cn_from_a2
                    cn_low = min(cn_known, 2)
                    cn_high = min(cn_known + 1, 2)
                    if cn_low == cn_high:
                        coefficients[cell_idx, kmer_idx, cn_low] += 1.0
                    else:
                        max_idx = 0 if cn_low == 0 else 1
                        max_coefficients[cell_idx, kmer_idx, max_idx] += 1.0
                        constants[cell_idx] += log_half
                else:
                    expected_cn = min(cn_from_a1 + cn_from_a2, 2)
                    coefficients[cell_idx, kmer_idx, expected_cn] += 1.0

    return coefficients, max_coefficients, constants


def _build_emission_coefficient_matrix(unique_kmer, size, scale):
    n_kmers = len(unique_kmer.kmer_to_count)
    coefficients = np.zeros((size * size, n_kmers, 3), dtype=np.float64)
    constants = np.zeros(size * size, dtype=np.float64)

    allele_ids = set(unique_kmer.alleles.keys())
    fallback = round(np.log(1.0 / 3.0) * scale)
    log_half = round(np.log(0.5) * scale)

    for a1 in range(size):
        a1_valid = a1 in allele_ids
        a1_undef = unique_kmer.is_undefined_allele(a1) if a1_valid else False
        for a2 in range(size):
            cell_idx = a1 * size + a2
            a2_valid = a2 in allele_ids
            a2_undef = unique_kmer.is_undefined_allele(a2) if a2_valid else False

            for kmer_idx in range(n_kmers):
                if not a1_valid or not a2_valid or (a1_undef and a2_undef):
                    constants[cell_idx] += fallback
                    continue

                cn_from_a1 = unique_kmer.kmer_on_allele(kmer_idx, a1) if not a1_undef else 0
                cn_from_a2 = unique_kmer.kmer_on_allele(kmer_idx, a2) if not a2_undef else 0
                if a1_undef or a2_undef:
                    cn_known = cn_from_a1 if not a1_undef else cn_from_a2
                    cn_low = min(cn_known, 2)
                    cn_high = min(cn_known + 1, 2)
                    coefficients[cell_idx, kmer_idx, cn_low] += 0.5
                    coefficients[cell_idx, kmer_idx, cn_high] += 0.5
                    constants[cell_idx] += log_half
                else:
                    expected_cn = min(cn_from_a1 + cn_from_a2, 2)
                    coefficients[cell_idx, kmer_idx, expected_cn] += 1.0

    return coefficients, constants


def _compute_encrypted_emission_for_genotype(
    crypten,
    torch,
    unique_kmer,
    probability_table,
    allele1,
    allele2,
    scale,
):
    allele_ids = list(unique_kmer.alleles.keys())
    a1_valid = allele1 in allele_ids
    a2_valid = allele2 in allele_ids
    a1_undef = unique_kmer.is_undefined_allele(allele1) if a1_valid else False
    a2_undef = unique_kmer.is_undefined_allele(allele2) if a2_valid else False

    total = None
    fallback = crypten.cryptensor(torch.tensor(round(np.log(1.0 / 3.0) * scale), dtype=torch.float64), src=0)
    log_half = round(np.log(0.5) * scale)

    for kmer_idx, read_count in enumerate(unique_kmer.kmer_to_count):
        if not a1_valid or not a2_valid or (a1_undef and a2_undef):
            term = fallback
        else:
            probs = _private_probability_lookup_fixed(
                crypten,
                torch,
                probability_table,
                int(unique_kmer.local_coverage),
                int(read_count),
                scale,
            )

            cn_from_a1 = unique_kmer.kmer_on_allele(kmer_idx, allele1) if not a1_undef else 0
            cn_from_a2 = unique_kmer.kmer_on_allele(kmer_idx, allele2) if not a2_undef else 0
            if a1_undef or a2_undef:
                cn_known = cn_from_a1 if not a1_undef else cn_from_a2
                cn_low = min(cn_known, 2)
                cn_high = min(cn_known + 1, 2)
                # Fixed-point approximation: choose the average of the two log
                # probabilities plus log(0.5). This keeps the operation linear
                # for CrypTen and avoids private logaddexp.
                term = (probs[cn_low] + probs[cn_high]) * 0.5 + log_half
            else:
                expected_cn = min(cn_from_a1 + cn_from_a2, 2)
                term = probs[expected_cn]

        total = term if total is None else total + term

    return total if total is not None else crypten.cryptensor(torch.tensor(0.0), src=0)


def _private_probability_lookup_fixed(
    crypten,
    torch,
    probability_table,
    kmer_coverage,
    read_kmer_count,
    scale,
):
    with profile_phase("emission_input_encoding"):
        cov = max(
            probability_table.cov_min,
            min(int(kmer_coverage), probability_table.cov_max - 1),
        )
        count = max(0, min(int(read_kmer_count), probability_table.count_max - 1))
        cov_width = probability_table.cov_max - probability_table.cov_min
        count_selector = torch.zeros(
            probability_table.count_max, dtype=torch.long
        )
        cov_selector = torch.zeros(cov_width, dtype=torch.long)
        count_selector[count] = 1
        cov_selector[cov - probability_table.cov_min] = 1
        _account_client_upload(
            count_selector.nelement() + cov_selector.nelement(),
            phase="emission_input_encoding",
        )

    with profile_phase("emission_computation"):
        n_rows = probability_table.count_max * cov_width
        table = np.rint(probability_table.probabilities * scale).astype(np.int64)
        flat_table = torch.tensor(table.reshape(n_rows, 3), dtype=torch.float64)

        from crypten.mpc.primitives import (
            ArithmeticSharedTensor,
            BinarySharedTensor,
            )

        count_one_hot = BinarySharedTensor(count_selector, src=0)
        cov_one_hot = BinarySharedTensor(cov_selector, src=0)
        selector = (count_one_hot.view(-1, 1) & cov_one_hot.view(1, -1)).view(
            n_rows, 1
        )
        selected_values = binary_times_public_no_truncation(flat_table, selector)
        return selected_values.sum(dim=0).view(3)


def _private_probability_lookup_batch_fixed(
    crypten,
    torch,
    probability_table,
    kmer_coverage,
    read_kmer_counts,
    scale,
    table_values=None,
    n_cols=3,
    profile_label="lookup",
):
    with profile_phase("emission_input_encoding"):
        cov_min = probability_table.cov_min
        count_max = probability_table.count_max
        cov = max(
            cov_min, min(int(kmer_coverage), probability_table.cov_max - 1)
        )
        counts = [
            max(0, min(int(read_count), count_max - 1))
            for read_count in read_kmer_counts
        ]
        n_kmers = len(counts)
        cov_width = probability_table.cov_max - cov_min
        cov_selector = torch.zeros((cov_width,), dtype=torch.long)
        cov_selector[cov - cov_min] = 1
        count_selectors = torch.zeros((n_kmers, count_max), dtype=torch.long)
        for kmer_idx, count in enumerate(counts):
            count_selectors[kmer_idx, count] = 1
        _account_client_upload(
            cov_selector.nelement(),
            phase="emission_input_encoding",
        )
        if n_kmers:
            _account_client_upload(
                count_selectors.nelement(),
                phase="emission_input_encoding",
            )

    with profile_phase("emission_computation"):
        # The public table is a computing-party input.  Selectors are shared
        # only after the client's encoding timer has ended.
        if table_values is None:
            table_values = probability_table.probabilities
        table = np.rint(table_values * scale).astype(np.int64)
        table_t = torch.tensor(
            table.reshape(1, count_max, cov_width, n_cols), dtype=torch.float64
        )

        from crypten.mpc.primitives import (
            ArithmeticSharedTensor,
            BinarySharedTensor,
            )

        # Two-stage oblivious lookup. Coverage is shared once per variant,
        # followed by one count-selector row per k-mer.
        cov_one_hot = BinarySharedTensor(cov_selector, src=0)
        with _comm_profile(f"{profile_label}_stage1_cov"):
            cov_selected = binary_times_public_no_truncation(
                table_t, cov_one_hot.view(1, 1, cov_width, 1),
            )
        p_cov = cov_selected.sum(dim=2)._tensor

        if n_kmers == 0:
            return p_cov.view(count_max, n_cols)[:0]

        count_one_hot = BinarySharedTensor(count_selectors, src=0)
        with _comm_profile(f"{profile_label}_stage2_count"):
            selected_values = mixed_mul_no_truncation(
                p_cov.view(1, count_max, n_cols),
                count_one_hot.view(n_kmers, count_max, 1),
                bits=1,
            )
        return selected_values.sum(dim=1)


def _load_crypten(crypten_root=None):
    """Import CrypTen and PyTorch with a clear dependency error."""
    try:
        import torch
        import crypten
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CrypTen backend requires PyTorch and CrypTen dependencies. "
            "Run `python3 -m pip install -e .` from the repository root so "
            "the upstream CrypTen package is importable, and make sure its "
            "Python dependencies are installed in the active environment."
        ) from exc

    from crypten.config import cfg
    cfg.encoder.precision_bits = 0
    return crypten, torch


def _init_crypten_if_needed(crypten):
    """Initialize CrypTen for local single-process backend execution if needed.

    The PVC launcher initializes upstream CrypTen before entering a party.
    Direct plaintext-owner scoring may initialize a single-process harness;
    private argmax still requires the three-party launcher and TTP provider.
    """
    if crypten.is_initialized():
        return
    crypten.init()


def _require_three_party(crypten):
    require_three_party_runtime()


def _progress_print(crypten, message, rank0_only=True):
    progress = os.environ.get("PVC_PROGRESS_LOG", "1")
    if progress not in {"0", "1"}:
        raise ValueError(
            "PVC_PROGRESS_LOG must be exactly 0 or 1; "
            f"got {progress!r}"
        )
    if progress == "0":
        return
    try:
        comm = crypten.communicator.get()
        rank = comm.get_rank()
        world_size = comm.get_world_size()
    except Exception:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if rank0_only and rank != 0:
        return
    print(f"[pvc-private rank {rank}/{world_size}] {message}", flush=True)
