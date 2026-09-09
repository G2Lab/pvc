"""
Genotyping algorithms for PVC.
"""

import os
import subprocess
import tempfile

from pvc.pangenome.genotyping_result import GenotypingResult

import numpy as np
from pvc.config import BCFTOOLS_BIN, PLINK_BIN, PLINK_BLOCKS_MAX_KB, PLINK_BLOCKS_MIN_MAF

from pvc.index.ir import GenotypingPlan

# ---------------------------------------------------------------------------
# Per-variant emission computation (shared by all modes)
# ---------------------------------------------------------------------------
def compute_emission_for_genotype(unique_kmer, probability_vector, allele1, allele2):
    """Log-probability of observed k-mer counts given genotype (allele1, allele2)."""
    n_kmers = len(probability_vector)
    if n_kmers == 0:
        return 0.0

    allele_ids = list(unique_kmer.alleles.keys())
    a1_valid = allele1 in allele_ids
    a2_valid = allele2 in allele_ids
    a1_undef = unique_kmer.is_undefined_allele(allele1) if a1_valid else False
    a2_undef = unique_kmer.is_undefined_allele(allele2) if a2_valid else False

    log_prob_sum = 0.0
    log_half = np.log(0.5)

    for kmer_idx in range(n_kmers):
        log_probs = probability_vector[kmer_idx]

        if not a1_valid or not a2_valid:
            log_prob_sum += np.log(1.0 / 3.0)
            continue

        if a1_undef and a2_undef:
            log_prob_sum += np.log(1.0 / 3.0)
            continue

        cn_from_a1 = unique_kmer.kmer_on_allele(kmer_idx, allele1) if not a1_undef else 0
        cn_from_a2 = unique_kmer.kmer_on_allele(kmer_idx, allele2) if not a2_undef else 0

        if a1_undef or a2_undef:
            cn_known = cn_from_a1 if not a1_undef else cn_from_a2
            cn_low = min(cn_known, 2)
            cn_high = min(cn_known + 1, 2)
            log_prob_sum += np.logaddexp(log_probs[cn_low], log_probs[cn_high]) + log_half
        else:
            expected_cn = min(cn_from_a1 + cn_from_a2, 2)
            log_prob_sum += log_probs[expected_cn]

    return log_prob_sum


def compute_emission_matrix(unique_kmer, probability_vector, n_alleles):
    """Emission matrix: log P(k-mer counts | genotype) for all genotype pairs."""
    emission = np.full((n_alleles, n_alleles), -np.inf)
    for a1 in range(n_alleles):
        for a2 in range(n_alleles):
            emission[a1, a2] = compute_emission_for_genotype(
                unique_kmer, probability_vector, a1, a2
            )
    return emission


def compute_frequency_matrix(allele_frequencies, n_alleles):
    """Hardy-Weinberg frequency matrix (prior) in log-space."""
    freqs = np.array(allele_frequencies[:n_alleles])
    if len(freqs) < n_alleles:
        freqs = np.pad(freqs, (0, n_alleles - len(freqs)), constant_values=1e-10)
    freq_sum = freqs.sum()
    if freq_sum > 0:
        freqs = freqs / freq_sum
    else:
        freqs = np.ones(n_alleles) / n_alleles
    freqs = np.maximum(freqs, 1e-10)
    log_freqs = np.log(freqs)

    freq_matrix = np.zeros((n_alleles, n_alleles))
    log_2 = np.log(2.0)
    for a1 in range(n_alleles):
        for a2 in range(n_alleles):
            if a1 == a2:
                freq_matrix[a1, a2] = 2 * log_freqs[a1]
            else:
                freq_matrix[a1, a2] = log_2 + log_freqs[a1] + log_freqs[a2]
    return freq_matrix


def normalize_log_matrix(log_matrix):
    """Normalize log-probability matrix to probabilities summing to 1."""
    max_val = np.max(log_matrix)
    if np.isinf(max_val) and max_val < 0:
        return np.ones_like(log_matrix) / log_matrix.size
    log_sum = max_val + np.log(np.sum(np.exp(log_matrix - max_val)))
    return np.exp(log_matrix - log_sum)


def compute_allele_frequency_vectors(unique_kmers_map):
    """Estimate per-variant allele frequencies from pangenome path assignments."""
    frequency_vectors = {}

    for chrom, unique_kmers in unique_kmers_map.unique_kmers.items():
        chrom_freqs = []
        for unique_kmer in unique_kmers:
            allele_ids = [int(allele_id) for allele_id in unique_kmer.alleles.keys()]
            path_alleles = [
                int(allele_id)
                for allele_id in getattr(unique_kmer, "path_to_allele", [])
                if int(allele_id) in unique_kmer.alleles
            ]

            max_allele = max(allele_ids + path_alleles, default=-1)
            if max_allele < 0:
                chrom_freqs.append([])
                continue

            counts = np.zeros(max_allele + 1, dtype=float)
            for allele_id in path_alleles:
                counts[allele_id] += 1.0

            total = counts.sum()
            if total > 0:
                chrom_freqs.append((counts / total).tolist())
                continue

            freqs = np.zeros(max_allele + 1, dtype=float)
            if allele_ids:
                uniform = 1.0 / len(allele_ids)
                for allele_id in allele_ids:
                    freqs[allele_id] = uniform
            chrom_freqs.append(freqs.tolist())

        frequency_vectors[chrom] = chrom_freqs

    return frequency_vectors


# ===========================================================================
# MODE 2: Haplotype-block genotyping
# ===========================================================================
#
# Approach: precompute a table of valid haplotype pairs per LD block from
# pangenome paths (the reference panel IS the pangenome). At query time,
# pick the pair whose implied genotypes have the highest dot product against
# observed emission likelihoods — pure argmax, no frequency prior.
#
# This mimics an HMM's path-constraint effect by construction: the valid
# diploid paths are baked into the precomputed table.
# ===========================================================================

def parse_plink_blocks_file(blocks_file):
    """Parse a PLINK .blocks.det file and return list of (start, end, n_snps) tuples."""
    blocks = []
    if os.path.exists(blocks_file):
        with open(blocks_file) as f:
            f.readline()  # skip header
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    blocks.append((int(parts[1]), int(parts[2]), int(parts[4])))
    return blocks


def run_plink_blocks(panel_vcf, chrom, tmpdir, blocks_file=None):
    """Get LD blocks, either from a pre-computed file or by running PLINK."""
    if blocks_file:
        print(f"  Loading pre-computed blocks from {blocks_file}")
        blocks = parse_plink_blocks_file(blocks_file)
        print(f"  Found {len(blocks)} LD blocks")
        return blocks

    print("  Running PLINK --blocks for LD block detection...")

    biallelic_vcf = os.path.join(tmpdir, "biallelic.vcf.gz")
    subprocess.run([
        str(BCFTOOLS_BIN), "view", "-m2", "-M2", "-v", "snps",
        panel_vcf, "-Oz", "-o", biallelic_vcf
    ], check=True, capture_output=True)
    subprocess.run([str(BCFTOOLS_BIN), "index", "-t", biallelic_vcf],
                   check=True, capture_output=True)

    plink_prefix = os.path.join(tmpdir, "plink_blocks")
    subprocess.run([
        str(PLINK_BIN), "--vcf", biallelic_vcf,
        "--vcf-half-call", "missing",
        "--blocks", "no-pheno-req",
        "--blocks-max-kb", str(PLINK_BLOCKS_MAX_KB),
        "--blocks-min-maf", str(PLINK_BLOCKS_MIN_MAF),
        "--allow-extra-chr",
        "--out", plink_prefix
    ], check=True, capture_output=True)

    blocks = parse_plink_blocks_file(plink_prefix + ".blocks.det")
    print(f"  Found {len(blocks)} LD blocks")
    return blocks


def assign_variants_to_blocks(blocks, unique_kmers):
    """Map variants to LD blocks by position. Returns (block_assignments, singleton_indices)."""
    pos_to_idx = {uk.variant_pos: i for i, uk in enumerate(unique_kmers)}
    assigned = set()
    block_assignments = []

    for bp1, bp2, _ in blocks:
        indices = [idx for pos, idx in pos_to_idx.items() if bp1 <= pos <= bp2]
        if indices:
            indices.sort(key=lambda i: unique_kmers[i].variant_pos)
            block_assignments.append(((bp1, bp2), indices))
            assigned.update(indices)

    singleton_indices = [i for i in range(len(unique_kmers)) if i not in assigned]
    return block_assignments, singleton_indices


def evaluate_block_assignments(
    unique_kmers,
    chrom_probs,
    chrom_freqs,
    chrom_results,
    block_assignments,
    verbose=False,
    label=None,
):
    """Evaluate one superblock-sized group of block assignments."""
    singleton_indices = []
    n_structured_blocks = sum(1 for _, indices in block_assignments if len(indices) > 1)
    n_in_structured_blocks = sum(
        len(indices) for _, indices in block_assignments if len(indices) > 1
    )
    n_singleton_blocks = sum(1 for _, indices in block_assignments if len(indices) == 1)

    prefix = f"  {label}: " if label else "  "
    print(
        f"{prefix}Blocks: {n_structured_blocks} structured, "
        f"{n_singleton_blocks} singleton"
    )
    print(f"{prefix}Variants in structured blocks: {n_in_structured_blocks}")

    print(f"{prefix}Precomputing full-panel haplotype pair tables...")
    block_tables, n_precomputed = precompute_haplotype_pair_table(
        unique_kmers, block_assignments
    )
    print(f"{prefix}Precomputed tables for {n_precomputed} blocks")

    blocks_genotyped = 0
    for (_, variant_indices), (_, pair_list) in zip(block_assignments, block_tables):
        if not pair_list:
            singleton_indices.extend(variant_indices)
            continue

        best_score = -np.inf
        best_h1, best_h2 = None, None

        for h1, h2 in pair_list:
            score = score_haplotype_pair(
                h1, h2, variant_indices, unique_kmers, chrom_probs
            )
            if score > best_score:
                best_score = score
                best_h1, best_h2 = h1, h2

        if best_h1 is not None:
            for block_idx, vi in enumerate(variant_indices):
                a1, a2 = best_h1[block_idx], best_h2[block_idx]
                result = chrom_results[vi]
                result.set_coverage(int(unique_kmers[vi].local_coverage))
                result.set_unique_kmers(len(chrom_probs[vi]) if vi < len(chrom_probs) else 0)
                result.add_to_likelihood(a1, a2, 1.0)
                result.normalize()
            blocks_genotyped += 1

        if verbose and blocks_genotyped <= 3:
            bp1 = unique_kmers[variant_indices[0]].variant_pos
            bp2 = unique_kmers[variant_indices[-1]].variant_pos
            print(f"    Block {bp1}-{bp2}: {len(variant_indices)} vars, "
                  f"{len(pair_list)} pairs, "
                  f"best=({best_h1[:3]}..., {best_h2[:3]}...)")

    n_singleton_called = 0
    for i in singleton_indices:
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

    print(f"{prefix}Blocks genotyped: {blocks_genotyped}")
    print(f"{prefix}Singletons genotyped: {n_singleton_called}")
    return blocks_genotyped, n_singleton_called


def block_assignments_from_ir_blocks(blocks):
    return [
        ((block.start, block.end), list(block.bubble_indices))
        for block in blocks
        if block.bubble_indices
    ]


def precompute_haplotype_pair_table(unique_kmers, block_assignments):
    """
    Precompute ALL distinct haplotype pairs per block from pangenome paths.

    For each block, extracts all distinct haplotypes from pangenome paths,
    then enumerates every possible diploid pair (n*(n+1)/2 combinations).
    The query phase scores all pairs and picks the best.

    Returns:
        block_tables: list of (variant_indices, [(h1, h2), ...]) per block
        n_precomputed: number of blocks with valid tables
    """
    block_tables = []
    n_precomputed = 0
    total_pairs = 0

    for (bp1, bp2), variant_indices in block_assignments:
        if len(variant_indices) < 2:
            block_tables.append((variant_indices, []))
            continue

        # Extract per-path haplotypes across this block
        first_var = unique_kmers[variant_indices[0]]
        n_paths = len(first_var.path_to_allele)

        haplotypes = []
        for path_id in range(n_paths):
            hap = tuple(
                unique_kmers[vi].path_to_allele[path_id]
                if path_id < len(unique_kmers[vi].path_to_allele) else 0
                for vi in variant_indices
            )
            haplotypes.append(hap)

        # Get all distinct haplotypes
        # Canonical ordering makes candidate-pair indices and first-index
        # tie-breaking reproducible across Python versions and hash seeds.
        distinct_haps = sorted(set(haplotypes))

        # Enumerate ALL n*(n+1)/2 diploid pairs
        pair_list = []
        for i in range(len(distinct_haps)):
            for j in range(i, len(distinct_haps)):
                pair_list.append((distinct_haps[i], distinct_haps[j]))

        total_pairs += len(pair_list)
        block_tables.append((variant_indices, pair_list))
        if pair_list:
            n_precomputed += 1

    print(f"  Total haplotype pairs to score: {total_pairs} "
          f"(avg {total_pairs/max(n_precomputed,1):.0f}/block)")
    return block_tables, n_precomputed


def score_haplotype_pair(h1, h2, variant_indices, unique_kmers, prob_vectors):
    """
    Dot product of a haplotype pair against observed emission likelihoods.

    score = sum_v log P(observed k-mers at v | genotype = (h1[v], h2[v]))

    Pure likelihood — no frequency prior.
    """
    log_score = 0.0
    for block_idx, vi in enumerate(variant_indices):
        if vi >= len(prob_vectors):
            continue
        pv = prob_vectors[vi]
        if len(pv) == 0:
            continue
        log_score += compute_emission_for_genotype(
            unique_kmers[vi], pv, h1[block_idx], h2[block_idx]
        )
    return log_score


def run_genotyping(
    unique_kmers_map, prob_vectors,
    panel_vcf, verbose=False,
    blocks_file=None,
    ir: GenotypingPlan | None = None,
):
    """
    Haplotype-block genotyping via precomputed pair tables.

    1. The IR supplies superblocks containing one or more LD/singleton blocks.
       If no IR is provided, PLINK blocks are loaded directly for compatibility.
    2. For each block, precompute all distinct diploid haplotype pairs from
       pangenome paths — these are the only valid genotype configurations.
    3. At query time, argmax: pick the pair with highest emission dot product.
    4. Singletons (variants outside any block) fall back to independent mode.
    """
    results = {}
    freq_vectors = compute_allele_frequency_vectors(unique_kmers_map)

    for chrom in unique_kmers_map.unique_kmers:
        unique_kmers = unique_kmers_map.unique_kmers[chrom]
        chrom_probs = prob_vectors.get(chrom, [])
        chrom_freqs = freq_vectors.get(chrom, [])
        n_variants = len(unique_kmers)

        chrom_results = [GenotypingResult() for _ in range(n_variants)]

        if ir is not None:
            superblocks = ir.superblocks_by_chrom.get(chrom, [])
            print(f"  IR superblocks: {len(superblocks)}")
            total_blocks_genotyped = 0
            total_singletons_called = 0
            for superblock in superblocks:
                ir_blocks = ir.blocks_for_superblock(superblock)
                block_assignments = block_assignments_from_ir_blocks(ir_blocks)
                blocks_genotyped, singletons_called = evaluate_block_assignments(
                    unique_kmers,
                    chrom_probs,
                    chrom_freqs,
                    chrom_results,
                    block_assignments,
                    verbose=verbose,
                    label=f"Superblock {superblock.superblock_id}",
                )
                total_blocks_genotyped += blocks_genotyped
                total_singletons_called += singletons_called
            print(f"  IR blocks genotyped: {total_blocks_genotyped}")
            print(f"  IR singletons genotyped: {total_singletons_called}")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                blocks = run_plink_blocks(panel_vcf, chrom, tmpdir, blocks_file=blocks_file)

            block_assignments, singleton_indices = assign_variants_to_blocks(
                blocks, unique_kmers
            )
            block_assignments.extend(
                ((unique_kmers[i].variant_pos, unique_kmers[i].variant_pos), [i])
                for i in singleton_indices
            )
            evaluate_block_assignments(
                unique_kmers,
                chrom_probs,
                chrom_freqs,
                chrom_results,
                block_assignments,
                verbose=verbose,
            )

        n_total_called = sum(1 for r in chrom_results if not r.contains_no_likelihoods())
        print(f"  Total genotyped: {n_total_called}/{n_variants}")
        results[chrom] = chrom_results

    return results
