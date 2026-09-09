from __future__ import annotations

from pvc.genotype.private.shared import (
    ensure_fixed_emissions_cover_pairs,
    prepare_public_fixed_emissions,
    private_block_score,
    run_private_haplotype_genotyping,
    score_haplotype_block_encrypted_vector,
    score_haplotype_block_private,
)

PRIVATE_METHOD = "max-only"


def _prepare_context(
    unique_kmers,
    chrom_probs,
    block_assignments,
    n_variants,
    probability_table,
    private_scale,
):
    return prepare_public_fixed_emissions(unique_kmers, chrom_probs, private_scale)


def _score_block(
    unique_kmers,
    chrom_probs,
    context,
    variant_indices,
    pair_list,
    private_scale,
    probability_table,
):
    ensure_fixed_emissions_cover_pairs(
        unique_kmers,
        chrom_probs,
        context,
        variant_indices,
        pair_list,
        private_scale,
    )
    best_pair_idx, _ = score_haplotype_block_private(
        variant_indices,
        pair_list,
        context,
        scale=private_scale,
        reveal_scores=False,
    )
    return private_block_score(best_pair_idx=best_pair_idx)


def _score_block_encrypted(
    unique_kmers,
    chrom_probs,
    context,
    variant_indices,
    pair_list,
    private_scale,
    probability_table,
):
    """Return the block's still-encrypted score vector (no per-block argmax).

    Identical score vector to :func:`_score_block`; the run loop batches every
    block's vector into a single chromosome-wide bucketed argmax (one grouped
    secure-argmax exchange per width bucket instead of per block).
    """
    ensure_fixed_emissions_cover_pairs(
        unique_kmers,
        chrom_probs,
        context,
        variant_indices,
        pair_list,
        private_scale,
    )
    return score_haplotype_block_encrypted_vector(
        variant_indices,
        pair_list,
        context,
        scale=private_scale,
    )


def run_genotyping_haplotype_private_medium(
    unique_kmers_map,
    prob_vectors,
    panel_vcf,
    verbose=False,
    blocks_file=None,
    ir=None,
    private_scale=1_000_000,
    probability_table=None,
    private_scores_output=None,
):
    return run_private_haplotype_genotyping(
        unique_kmers_map,
        prob_vectors,
        panel_vcf,
        PRIVATE_METHOD,
        _prepare_context,
        _score_block,
        verbose=verbose,
        blocks_file=blocks_file,
        ir=ir,
        private_scale=private_scale,
        probability_table=probability_table,
        private_scores_output=private_scores_output,
        # Medium is a hidden-score (argmax) tier: mark it as such so the run
        # loop uses the batched argmax (env PVC_BATCH_ARGMAX, default on) and,
        # when disabled, falls back to the per-block private argmax -- never to
        # batched reveal. The env gate is applied inside the run loop.
        score_block_encrypted=_score_block_encrypted,
        batched_argmax=True,
    )
