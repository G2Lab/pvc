"""Role-separated three-party PVC execution for research deployments.

The parent process is the client.  It alone reads read-derived evidence,
creates independent additive/XOR input shares, and reconstructs output
shares.  Child processes are computing parties: each receives the public
genotyping structure and exactly one local share of every private value.

Process separation on one host is useful for tests and benchmarks, but is not
an administrative trust boundary.  A real deployment must run the same party
worker interface under three independent principals and transport the rank
payloads over authenticated confidential channels.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from pvc.genotype.genotyping import (
    block_assignments_from_ir_blocks,
    compute_allele_frequency_vectors,
    compute_emission_matrix,
    compute_frequency_matrix,
    normalize_log_matrix,
    precompute_haplotype_pair_table,
)
from pvc.genotype.private.shared import (
    _build_emission_max_coefficient_matrices,
    _matmul_public_integer_no_truncation,
    _score_pairs_vectorized,
    fixed_point_emission_matrix,
)
from pvc.pangenome.genotyping_result import GenotypingResult


@dataclass
class PartyPrivateInput:
    """The only private payload made available to one computing party."""

    mode: str
    emissions: dict[str, list[np.ndarray]] | None = None
    coverage_selectors: dict[str, list[np.ndarray]] | None = None
    count_selectors: dict[str, list[np.ndarray]] | None = None


@dataclass
class PublicProtocolInput:
    """Sample-independent structure shared identically with all parties."""

    unique_kmers_map: Any
    blocks_by_chrom: dict[str, list[Any]]
    probability_table: Any
    scale: int


def redact_read_evidence(unique_kmers_map):
    """Return a server-safe copy retaining only public shape and panel data."""
    public_map = deepcopy(unique_kmers_map)
    # Profiling fields can depend on the private read input as well.  They are
    # not needed by the computing parties.
    if hasattr(public_map, "runtimes"):
        public_map.runtimes = []
    if hasattr(public_map, "sampling_runtimes"):
        public_map.sampling_runtimes = []
    for unique_kmers in public_map.unique_kmers.values():
        for unique_kmer in unique_kmers:
            unique_kmer.local_coverage = 0
            unique_kmer.kmer_to_count = [0] * len(unique_kmer.kmer_to_count)
    return public_map


def _random_int64(shape) -> np.ndarray:
    count = int(np.prod(shape, dtype=np.int64))
    return (
        np.frombuffer(os.urandom(count * 8), dtype=np.uint64)
        .reshape(shape)
        .view(np.int64)
        .copy()
    )


def _random_bits(shape) -> np.ndarray:
    count = int(np.prod(shape, dtype=np.int64))
    return (
        (np.frombuffer(os.urandom(count), dtype=np.uint8) & 1)
        .astype(np.int64)
        .reshape(shape)
    )


def split_arithmetic_secret(values) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split signed ring values into three fresh additive shares."""
    secret = np.asarray(values, dtype=np.int64)
    first = _random_int64(secret.shape)
    second = _random_int64(secret.shape)
    third_u = secret.view(np.uint64) - first.view(np.uint64) - second.view(np.uint64)
    return first, second, third_u.view(np.int64).copy()


def split_binary_secret(values) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split integer bit tensors into three fresh XOR shares."""
    secret = np.asarray(values, dtype=np.int64)
    if np.any((secret != 0) & (secret != 1)):
        raise ValueError("binary secrets must contain only zero and one")
    first = _random_bits(secret.shape)
    second = _random_bits(secret.shape)
    third = secret ^ first ^ second
    return first, second, third


def reconstruct_arithmetic(shares) -> np.ndarray:
    if len(shares) != 3:
        raise ValueError("three arithmetic shares are required")
    result = np.asarray(shares[0], dtype=np.int64).view(np.uint64).copy()
    result += np.asarray(shares[1], dtype=np.int64).view(np.uint64)
    result += np.asarray(shares[2], dtype=np.int64).view(np.uint64)
    return result.view(np.int64)


def _public_blocks(genotyping_plan, public_map):
    blocks_by_chrom = {}
    for chrom, unique_kmers in public_map.unique_kmers.items():
        assignments = []
        for superblock in genotyping_plan.superblocks_by_chrom.get(chrom, []):
            assignments.extend(
                block_assignments_from_ir_blocks(
                    genotyping_plan.blocks_for_superblock(superblock)
                )
            )
        tables, _ = precompute_haplotype_pair_table(unique_kmers, assignments)
        blocks_by_chrom[chrom] = [
            (assignment[0], assignment[1], table[1])
            for assignment, table in zip(assignments, tables)
        ]
    return blocks_by_chrom


def prepare_role_separated_inputs(
    tool,
    unique_kmers_map,
    prob_vectors,
    probability_table,
    genotyping_plan,
    *,
    scale=1_000_000,
):
    """Client-side preparation of public structure and per-rank shares."""
    mode = {
        "pvc-light": "light",
        "pvc-light-gpu": "light",
        "pvc-medium": "medium",
        "pvc-heavy": "heavy",
    }[tool]
    public_map = redact_read_evidence(unique_kmers_map)
    public = PublicProtocolInput(
        unique_kmers_map=public_map,
        blocks_by_chrom=_public_blocks(genotyping_plan, public_map),
        probability_table=probability_table,
        scale=int(scale),
    )
    parties = [PartyPrivateInput(mode=mode) for _ in range(3)]

    if mode in {"light", "medium"}:
        required_sizes = {
            chrom: [max(len(item.alleles), 1) for item in unique_kmers]
            for chrom, unique_kmers in public_map.unique_kmers.items()
        }
        for chrom, blocks in public.blocks_by_chrom.items():
            for _, variant_indices, pair_list in blocks:
                for block_index, variant_index in enumerate(variant_indices):
                    if pair_list:
                        required_sizes[chrom][variant_index] = max(
                            required_sizes[chrom][variant_index],
                            1
                            + max(
                                max(int(h1[block_index]), int(h2[block_index]))
                                for h1, h2 in pair_list
                            ),
                        )
        per_rank = [dict(), dict(), dict()]
        for chrom, unique_kmers in unique_kmers_map.unique_kmers.items():
            matrices_by_rank = [[], [], []]
            chrom_probs = prob_vectors.get(chrom, [])
            for index, unique_kmer in enumerate(unique_kmers):
                size = required_sizes[chrom][index]
                if index >= len(chrom_probs) or not chrom_probs[index]:
                    fixed = np.zeros((size, size), dtype=np.int64)
                else:
                    fixed = fixed_point_emission_matrix(
                        compute_emission_matrix(unique_kmer, chrom_probs[index], size),
                        scale=scale,
                    )
                shares = split_arithmetic_secret(fixed)
                for rank in range(3):
                    matrices_by_rank[rank].append(shares[rank])
            for rank in range(3):
                per_rank[rank][chrom] = matrices_by_rank[rank]
        for rank in range(3):
            parties[rank].emissions = per_rank[rank]
    else:
        cov_by_rank = [dict(), dict(), dict()]
        count_by_rank = [dict(), dict(), dict()]
        cov_min = probability_table.cov_min
        cov_width = probability_table.cov_max - cov_min
        count_max = probability_table.count_max
        for chrom, unique_kmers in unique_kmers_map.unique_kmers.items():
            rank_cov = [[], [], []]
            rank_count = [[], [], []]
            for unique_kmer in unique_kmers:
                cov = max(cov_min, min(int(unique_kmer.local_coverage), probability_table.cov_max - 1))
                cov_selector = np.zeros(cov_width, dtype=np.int64)
                cov_selector[cov - cov_min] = 1
                counts = np.zeros((len(unique_kmer.kmer_to_count), count_max), dtype=np.int64)
                for row, value in enumerate(unique_kmer.kmer_to_count):
                    counts[row, max(0, min(int(value), count_max - 1))] = 1
                cov_shares = split_arithmetic_secret(cov_selector)
                count_shares = split_binary_secret(counts)
                for rank in range(3):
                    rank_cov[rank].append(cov_shares[rank])
                    rank_count[rank].append(count_shares[rank])
            for rank in range(3):
                cov_by_rank[rank][chrom] = rank_cov[rank]
                count_by_rank[rank][chrom] = rank_count[rank]
        for rank in range(3):
            parties[rank].coverage_selectors = cov_by_rank[rank]
            parties[rank].count_selectors = count_by_rank[rank]
    return public, parties


def _score_shared_emissions(torch, matrices, variant_indices, pair_list):
    from crypten.mpc import MPCTensor

    if not variant_indices:
        return MPCTensor.from_shares(
            torch.zeros(len(pair_list), dtype=torch.int64), precision=0
        )
    offsets = []
    sizes = []
    flat = []
    offset = 0
    for vi in variant_indices:
        matrix = np.asarray(matrices[vi], dtype=np.int64)
        size = int(matrix.shape[0])
        offsets.append(offset)
        sizes.append(size)
        flat.append(matrix.reshape(-1))
        offset += size * size
    encrypted = MPCTensor.from_shares(
        torch.tensor(np.concatenate(flat), dtype=torch.int64), precision=0
    )
    h1 = np.asarray([pair[0] for pair in pair_list], dtype=np.int64)
    h2 = np.asarray([pair[1] for pair in pair_list], dtype=np.int64)
    indices = (
        np.asarray(offsets)[None, :]
        + h1 * np.asarray(sizes)[None, :]
        + h2
    )
    return encrypted[torch.tensor(indices.reshape(-1), dtype=torch.long)].reshape(
        len(pair_list), len(variant_indices)
    ).sum(dim=1)


def _heavy_emission(
    torch, unique_kmer, cov_share, count_share, probability_table, size, scale
):
    from crypten.mpc import MPCTensor
    from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor
    from pvc.genotype.private.mixed_protocols import mixed_mul_no_truncation

    n_kmers = len(unique_kmer.kmer_to_count)
    if n_kmers == 0:
        return MPCTensor.from_shares(
            torch.zeros(size * size, dtype=torch.int64), precision=0
        )
    cov_width = probability_table.cov_max - probability_table.cov_min
    count_max = probability_table.count_max
    combined = torch.tensor(
        np.rint(
            np.concatenate(
                [probability_table.probabilities, probability_table.probabilities_lae],
                axis=-1,
            )
            * scale
        ).astype(np.int64),
        dtype=torch.long,
    )
    cov = ArithmeticSharedTensor.from_shares(
        torch.tensor(cov_share, dtype=torch.int64), precision=0
    )
    per_count = (
        cov.view(1, cov_width, 1) * combined.view(count_max, cov_width, 5)
    ).sum(dim=1)
    counts = BinarySharedTensor.from_shares(
        torch.tensor(count_share, dtype=torch.int64), precision=0
    )
    selected = mixed_mul_no_truncation(
        per_count.view(1, count_max, 5),
        counts.view(n_kmers, count_max, 1),
        bits=1,
    ).sum(dim=1)
    coefficients, lae_coefficients, constants = _build_emission_max_coefficient_matrices(
        unique_kmer, size, scale
    )
    coeff = torch.tensor(
        coefficients.reshape(size * size, n_kmers * 3).T, dtype=torch.long
    )
    scores = _matmul_public_integer_no_truncation(
        selected[..., :3].reshape(1, -1), coeff
    ).view(-1)
    if np.any(lae_coefficients):
        lae_coeff = torch.tensor(
            lae_coefficients.reshape(size * size, n_kmers * 2).T,
            dtype=torch.long,
        )
        scores = scores + _matmul_public_integer_no_truncation(
            selected[..., 3:].reshape(1, -1), lae_coeff
        ).view(-1)
    return scores + torch.tensor(constants, dtype=torch.float64)


def _score_heavy_block(torch, public_variants, party, chrom, variant_indices, pair_list, table, scale):
    encrypted = {}
    for block_index, vi in enumerate(variant_indices):
        max_allele = max(
            max(int(h1[block_index]), int(h2[block_index])) for h1, h2 in pair_list
        )
        size = max(len(public_variants[vi].alleles), max_allele + 1, 1)
        encrypted[vi] = _heavy_emission(
            torch,
            public_variants[vi],
            party.coverage_selectors[chrom][vi],
            party.count_selectors[chrom][vi],
            table,
            size,
            scale,
        ).view(size, size)
    return _score_pairs_vectorized(torch, encrypted, variant_indices, pair_list)


def _private_winner_one_hot(score_vector, width, torch):
    from pvc.genotype.private.runtime import first_winner_one_hot

    if score_vector.size(0) != width or width < 1:
        raise ValueError("winner width must match a nonempty score vector")
    return first_winner_one_hot(score_vector)


def run_role_separated_party(public, private):
    """Computing-party entry point; returns only this rank's output shares."""
    import torch
    from pvc.genotype.private.runtime import require_three_party_runtime

    require_three_party_runtime()
    if private.mode not in {"light", "medium", "heavy"}:
        raise ValueError(f"Unsupported PVC private mode: {private.mode}")
    records = []
    for chrom, blocks in public.blocks_by_chrom.items():
        variants = public.unique_kmers_map.unique_kmers[chrom]
        for coordinates, variant_indices, pair_list in blocks:
            if not pair_list:
                continue
            if private.mode in {"light", "medium"}:
                scores = _score_shared_emissions(
                    torch, private.emissions[chrom], variant_indices, pair_list
                )
            else:
                scores = _score_heavy_block(
                    torch,
                    variants,
                    private,
                    chrom,
                    variant_indices,
                    pair_list,
                    public.probability_table,
                    public.scale,
                )
            output = scores if private.mode == "light" else _private_winner_one_hot(
                scores, len(pair_list), torch
            )
            local_share = output._tensor.share
            records.append(local_share.detach().cpu().numpy().astype(np.int64).tolist())
    return records


def reconstruct_role_separated_results(
    tool, public, party_outputs, private_map, prob_vectors
):
    """Client-only reconstruction and genotype assignment."""
    if len(party_outputs) != 3:
        raise ValueError("exactly three party outputs are required")
    results = {}
    output_index = 0
    frequency_vectors = compute_allele_frequency_vectors(private_map)
    for chrom, blocks in public.blocks_by_chrom.items():
        variants = private_map.unique_kmers[chrom]
        chrom_probs = prob_vectors.get(chrom, [])
        chrom_results = [GenotypingResult() for _ in variants]
        singleton_indices = []
        for _, variant_indices, pair_list in blocks:
            if not pair_list:
                singleton_indices.extend(variant_indices)
                continue
            reconstructed = reconstruct_arithmetic(
                [party_outputs[rank][output_index] for rank in range(3)]
            )
            output_index += 1
            best = int(np.argmax(reconstructed))
            best_h1, best_h2 = pair_list[best]
            for block_index, vi in enumerate(variant_indices):
                result = chrom_results[vi]
                result.set_coverage(int(variants[vi].local_coverage))
                result.set_unique_kmers(len(chrom_probs[vi]))
                result.add_to_likelihood(best_h1[block_index], best_h2[block_index], 1.0)
                result.normalize()
        chrom_freqs = frequency_vectors.get(chrom, [])
        for vi in singleton_indices:
            unique_kmer = variants[vi]
            if not unique_kmer.alleles or vi >= len(chrom_probs) or not chrom_probs[vi]:
                continue
            emission = compute_emission_matrix(
                unique_kmer, chrom_probs[vi], len(unique_kmer.alleles)
            )
            frequency = compute_frequency_matrix(
                chrom_freqs[vi] if vi < len(chrom_freqs) else [],
                len(unique_kmer.alleles),
            )
            posterior = normalize_log_matrix(emission + frequency)
            result = chrom_results[vi]
            result.set_coverage(int(unique_kmer.local_coverage))
            result.set_unique_kmers(len(chrom_probs[vi]))
            for a1 in range(len(unique_kmer.alleles)):
                for a2 in range(len(unique_kmer.alleles)):
                    if posterior[a1, a2] > 1e-10:
                        result.add_to_likelihood(a1, a2, posterior[a1, a2])
            result.normalize()
        results[chrom] = chrom_results
    return results
