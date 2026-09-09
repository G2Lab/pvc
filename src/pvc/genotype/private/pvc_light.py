from __future__ import annotations

from pvc.genotype.private.shared import (
    ensure_fixed_emissions_cover_pairs,
    prepare_public_fixed_emissions,
    private_block_score,
    run_private_haplotype_genotyping,
    score_haplotype_block_encrypted_vector,
    score_haplotype_block_private,
)

PRIVATE_METHOD = "fast"


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
    scores_fixed, scores_float = score_haplotype_block_private(
        variant_indices,
        pair_list,
        context,
        scale=private_scale,
        reveal_scores=True,
    )
    return private_block_score(scores_fixed=scores_fixed, scores_float=scores_float)


def _score_block_encrypted(
    unique_kmers,
    chrom_probs,
    context,
    variant_indices,
    pair_list,
    private_scale,
    probability_table,
):
    """Deferred-reveal scorer: return the block's encrypted score vector.

    Same scoring as :func:`_score_block` but without the per-block reveal, so the
    run loop can batch every block's reveal into one logical reduce call.
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


def run_genotyping_haplotype_private_light(
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
        score_block_encrypted=_score_block_encrypted,
    )
