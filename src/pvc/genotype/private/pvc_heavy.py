from __future__ import annotations

import os

from pvc.genotype.private.shared import (
    private_block_score,
    run_private_haplotype_genotyping,
    score_haplotype_block_private_complete,
    score_haplotype_block_private_complete_encrypted_vector,
)

PRIVATE_METHOD = "complete"


def _prepare_context(
    unique_kmers,
    chrom_probs,
    block_assignments,
    n_variants,
    probability_table,
    private_scale,
):
    if probability_table is None:
        raise ValueError("probability_table is required for pvc-heavy")
    if os.environ.get("PVC_PROGRESS_LOG", "1") != "0":
        print(
            "  Complete mode keeps private lookup outputs encrypted through "
            "block scoring",
            flush=True,
        )
    return None


def _score_block(
    unique_kmers,
    chrom_probs,
    context,
    variant_indices,
    pair_list,
    private_scale,
    probability_table,
):
    best_pair_idx, _ = score_haplotype_block_private_complete(
        unique_kmers,
        probability_table,
        variant_indices,
        pair_list,
        scale=private_scale,
        logaddexp_mode="max",
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

    Same score vector as :func:`_score_block`; the run loop batches every
    block's vector into a single chromosome-wide bucketed argmax (one grouped
    secure-argmax exchange per width bucket instead of per block), exactly as
    PVC medium does.
    """
    return score_haplotype_block_private_complete_encrypted_vector(
        unique_kmers,
        probability_table,
        variant_indices,
        pair_list,
        scale=private_scale,
        logaddexp_mode="max",
    )


def run_genotyping_haplotype_private_heavy(
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
        # Heavy is a hidden-score (argmax) tier like medium: use the batched
        # chromosome-wide argmax (env PVC_BATCH_ARGMAX, default on) and fall
        # back to the per-block private argmax when it is disabled.
        score_block_encrypted=_score_block_encrypted,
        batched_argmax=True,
    )
