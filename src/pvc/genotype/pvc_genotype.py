#!/usr/bin/env python3
"""
PVC — Variant calling with haplotype-block-aware genotyping.

The haplotype mode:
  1. Runs PLINK --blocks on the panel VCF to find LD blocks
  2. Extracts haplotypes from pangenome paths within each block
  3. Enumerates all distinct haplotype pairs per block
  4. Scores each pair by summing per-variant emission log-probs
  5. Picks the pair with highest score (argmax)

Usage:
    python -m pvc --cereal <read_UniqueKmersMap.json> \\
                  --graph <Graph.json> \\
                  --panel-vcf <panel.vcf.gz> \\
                  --vcf <truth.vcf> \\
                  --mean-kmer-abundance 30 \\
                  --mode haplotype \\
                  --output <output.vcf>
"""

import os
import time
import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from pvc.pangenome.unique_kmers_map import read_unique_kmers_map_from_cereal
from pvc.pangenome.graph import read_graph
from pvc.pangenome.probability_table import ProbabilityTable

from pvc.genotype.genotyping import run_genotyping
from pvc.index.ir import (
    build_genotyping_plan,
    write_bubble_likelihoods,
)

from pvc.index.pangenie_process import manifest_chromosome
from pvc.index.pvc_index import (
    find_completed_pvc_index,
    pvc_index_complete_path,
    pvc_index_lookup_order,
)
from pvc.config import PVC_GENOTYPE_MEAN_KMER_ABUNDANCE

def is_output_rank():
    """Return true for the single process allowed to write outputs."""
    return os.environ.get("RANK", "0") == "0"


# ---------------------------------------------------------------------------
# Client: compute probability vectors from k-mer counts
# ---------------------------------------------------------------------------
def compute_probability_vectors(unique_kmers_map, probability_table):
    """For each variant, compute log P(count | CN=0,1,2) for each k-mer."""
    probability_vectors = {}
    for chrom, unique_kmers in unique_kmers_map.unique_kmers.items():
        chrom_vectors = []
        for unique_kmer in unique_kmers:
            kmer_coverage = int(unique_kmer.local_coverage)
            prob_vector = []
            for i in range(len(unique_kmer.kmer_to_count)):
                count = max(0, min(int(unique_kmer.kmer_to_count[i]),
                                   probability_table.count_max - 1))
                log_probs = probability_table.get_probability(kmer_coverage, count)
                prob_vector.append(log_probs)
            chrom_vectors.append(prob_vector)
        probability_vectors[chrom] = chrom_vectors
    return probability_vectors


# ===========================================================================
# Verification & VCF output (shared)
# ===========================================================================
def read_vcf_genotypes(vcf_file):
    """Read genotypes from a VCF file."""
    genotypes = {}
    opener = gzip.open if str(vcf_file).endswith(".gz") else open
    with opener(vcf_file, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            pos = int(parts[1])
            gt_str = parts[9].split(':')[0]
            gt_map = {
                '0/0': (0, 0), '0|0': (0, 0),
                '0/1': (0, 1), '0|1': (0, 1),
                '1/0': (0, 1), '1|0': (0, 1),
                '1/1': (1, 1), '1|1': (1, 1),
            }
            genotypes[pos] = gt_map.get(gt_str, (-1, -1))
    return genotypes


def verify_results(genotype_results, graph, vcf_file):
    """Compare predicted genotypes against ground truth VCF."""
    true_genotypes = read_vcf_genotypes(vcf_file)
    chrom = graph.chromosome

    if chrom not in genotype_results:
        print(f"No results for chromosome {chrom}")
        return 0.0, 0, 0

    results_arr = genotype_results[chrom]
    correct = 0
    total = 0
    errors = {}

    for i, result in enumerate(results_arr):
        try:
            variant = graph.variants[i]
            if variant is None:
                continue

            if variant.is_combined():
                sv, sr = variant.separate_variants(result, True)
            else:
                sv, sr = [variant], [result]

            for s_var, s_res in zip(sv, sr):
                pos = s_var.start_position + 1
                if pos not in true_genotypes:
                    continue
                true_gt = true_genotypes[pos]
                if true_gt == (-1, -1):
                    continue

                total += 1

                nr_alleles = s_var.nr_of_alleles()
                defined_alleles = [0]
                for a in range(1, nr_alleles):
                    if not s_var.is_undefined_allele(a):
                        defined_alleles.append(a)

                if s_res.contains_no_likelihoods():
                    s_res.add_to_likelihood(0, 0, 1.0)

                if len(defined_alleles) < nr_alleles:
                    s_res = s_res.get_specific_likelihoods(defined_alleles)

                pred_gt = s_res.get_likeliest_genotype()

                if pred_gt == true_gt or pred_gt == (true_gt[1], true_gt[0]):
                    correct += 1
                else:
                    key = f"{true_gt} -> {pred_gt}"
                    errors[key] = errors.get(key, 0) + 1
        except Exception:
            continue

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nAccuracy: {accuracy:.2%} ({correct}/{total})")

    if errors:
        print(f"\nError summary ({total - correct} incorrect):")
        for err, count in sorted(errors.items(), key=lambda x: -x[1])[:10]:
            print(f"  {err}: {count}")

    return accuracy, correct, total


def write_output_vcf(genotype_results, graph, unique_kmers_map, output_path, sample_name="SAMPLE"):
    """
    Write genotyping results to VCF format.

    Combined (multiallelic) variants are split into biallelic records using
    variant.separate_variants(), and allele indices are remapped via
    get_specific_likelihoods() so output genotypes use 0/0, 0/1, 1/1 encoding.
    This matches PanGenie's output format.
    """
    chrom = graph.chromosome
    if chrom not in genotype_results:
        return

    results_arr = genotype_results[chrom]

    with open(output_path, 'w') as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write(f"##contig=<ID={chrom}>\n")
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        f.write('##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">\n')
        f.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_name}\n")

        for i, (result, variant) in enumerate(zip(results_arr, graph.variants)):
            if variant is None:
                continue

            # Split combined variants into singletons (biallelic)
            if variant.is_combined():
                singleton_variants, singleton_results = variant.separate_variants(result, True)
            else:
                singleton_variants = [variant]
                singleton_results = [result]

            for s_var, s_res in zip(singleton_variants, singleton_results):
                pos = s_var.start_position + 1
                nr_alleles = s_var.nr_of_alleles()

                # Get REF and ALT sequences
                allele_seqs = s_var.allele_sequences[0] if s_var.allele_sequences else []
                ref = allele_seqs[0].to_string() if len(allele_seqs) > 0 else "N"
                alts = [s.to_string() for s in allele_seqs[1:]] if len(allele_seqs) > 1 else ["."]

                # Remap to defined alleles only (removes undefined alleles)
                defined_alleles = [0]
                for a in range(1, nr_alleles):
                    if not s_var.is_undefined_allele(a):
                        defined_alleles.append(a)

                if s_res.contains_no_likelihoods():
                    s_res.add_to_likelihood(0, 0, 1.0)

                if len(defined_alleles) < nr_alleles:
                    s_res = s_res.get_specific_likelihoods(defined_alleles)

                gt = s_res.get_likeliest_genotype()
                if gt == (-1, -1):
                    gt_str = "./."
                    gq = 0
                else:
                    gt_str = f"{gt[0]}/{gt[1]}"
                    try:
                        gq = s_res.get_genotype_quality(gt[0], gt[1])
                    except Exception:
                        gq = 0

                f.write(f"{chrom}\t{pos}\t.\t{ref}\t{','.join(alts)}\t.\tPASS\t.\tGT:GQ\t{gt_str}:{gq}\n")


def pvc_genotype_dir(run_dir: str | Path, tool: str) -> Path:
    return Path(run_dir) / tool / "genotype"


def pvc_genotype_prefix(run_dir: str | Path, tool: str) -> Path:
    return pvc_genotype_dir(run_dir, tool) / tool


def pvc_genotype_complete_path(run_dir: str | Path, tool: str) -> Path:
    return pvc_genotype_dir(run_dir, tool) / "genotype_complete.json"


def pvc_genotype_vcf_path(run_dir: str | Path, tool: str) -> Path:
    return pvc_genotype_prefix(run_dir, tool).with_suffix(".vcf")


def load_pvc_index_metadata(
    run_dir: str | Path,
    tool: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    completed_index = find_completed_pvc_index(run_dir, tool, manifest)
    if completed_index is None:
        searched = "\n  ".join(
            str(pvc_index_complete_path(run_dir, candidate_tool))
            for candidate_tool in pvc_index_lookup_order(tool)
        )
        raise FileNotFoundError(
            f"{tool} genotype requires one completed PVC index in this run. "
            f"Searched:\n  {searched}\n"
            f"Run any PVC index stage first, for example --tools pvc."
        )

    return completed_index[2]


def manifest_testcase_dir(manifest: dict[str, Any]) -> Path | None:
    for key in ("fastq", "panel_subset", "truth", "reference_fasta"):
        if key not in manifest:
            continue
        path = Path(manifest[key])
        if path.parent.name == "inputs":
            return path.parent.parent
    return None


def candidate_read_unique_kmers_paths(
    manifest: dict[str, Any],
    index_prefix: Path,
    tool: str,
) -> list[Path]:
    candidates = []
    for key in (
        "read_unique_kmers_json",
        "read_unique_kmers_map",
        "pvc_read_unique_kmers_json",
        "pvc_read_unique_kmers_map",
        "unique_kmers_json",
    ):
        if manifest.get(key):
            candidates.append(Path(manifest[key]))

    candidates.extend(
        [
            Path(f"{index_prefix}_read_UniqueKmersMap.json"),
            Path(f"{index_prefix}_UniqueKmersMap.json"),
        ]
    )

    testcase_dir = manifest_testcase_dir(manifest)
    if testcase_dir is not None:
        candidates.extend(
            [
                testcase_dir / "pvc_haplotype" / "prep" / "pvc_haplotype_read_UniqueKmersMap.json",
                testcase_dir / "pvc_private" / "pvc_subset_read_UniqueKmersMap.json",
                testcase_dir / tool / f"{tool}_read_UniqueKmersMap.json",
                testcase_dir / tool / "prep" / f"{tool}_read_UniqueKmersMap.json",
            ]
        )

    return list(dict.fromkeys(candidates))


def resolve_read_unique_kmers_path(
    manifest: dict[str, Any],
    index_prefix: Path,
    tool: str,
) -> Path:
    candidates = candidate_read_unique_kmers_paths(manifest, index_prefix, tool)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "PVC genotype currently requires a read UniqueKmersMap JSON file. "
        "The index-produced UniqueKmersMap.cereal is binary and cannot be read "
        f"by this Python genotyper. Searched:\n  {searched}"
    )


def pvc_index_input_paths(
    tool: str,
    manifest: dict[str, Any],
    index_metadata: dict[str, Any],
) -> tuple[Path, Path, Path]:
    try:
        index_prefix = Path(index_metadata["index_prefix"])
        blocks_file = Path(index_metadata["blocks_file"])
    except KeyError as exc:
        raise ValueError(f"PVC index marker is missing required key: {exc.args[0]}") from exc

    chromosome = manifest_chromosome(manifest)
    unique_kmers_path = resolve_read_unique_kmers_path(manifest, index_prefix, tool)
    # Fresh read-map markers keep the sample output prefix (for the private
    # read map) separate from the reusable public prefix (for the graph).
    graph_prefix = Path(index_metadata.get("public_index_prefix", index_prefix))
    graph_path = Path(f"{graph_prefix}_{chromosome}_Graph.json")
    return unique_kmers_path, graph_path, blocks_file


def validate_pvc_genotype_inputs(
    manifest: dict[str, Any],
    unique_kmers_path: Path,
    graph_path: Path,
    blocks_file: Path,
) -> None:
    missing_keys = [key for key in ("panel_subset",) if key not in manifest]
    if missing_keys:
        raise ValueError(f"Manifest is missing required PVC genotype keys: {missing_keys}")

    missing_paths = [
        path for path in (unique_kmers_path, graph_path, blocks_file) if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "PVC genotype requires completed index artifacts. Missing: "
            f"{[str(path) for path in missing_paths]}"
        )


def write_pvc_genotype_complete(
    run_dir: str | Path,
    tool: str,
    output_prefix: str | Path,
    output_vcf: str | Path,
    ir_dir: str | Path,
    index_metadata: dict[str, Any],
    accuracy: tuple[float, int, int] | None,
) -> Path:
    marker_path = pvc_genotype_complete_path(run_dir, tool)
    payload: dict[str, Any] = {
        "genotype_prefix": str(output_prefix),
        "output_vcf": str(output_vcf),
        "ir_dir": str(ir_dir),
        "index_prefix": str(index_metadata.get("index_prefix")),
        "public_index_prefix": str(
            index_metadata.get("public_index_prefix", index_metadata.get("index_prefix"))
        ),
        "blocks_file": str(index_metadata.get("blocks_file")),
        "index_tool": str(index_metadata.get("index_tool")),
        "index_marker": str(index_metadata.get("index_marker")),
    }
    if accuracy is not None:
        payload["accuracy"] = {
            "rate": accuracy[0],
            "correct": accuracy[1],
            "total": accuracy[2],
        }

    marker_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return marker_path


def run_pvc_genotype(
    tool: str,
    manifest: dict[str, Any],
    run_dir: str | Path,
    mean_kmer_abundance: int | None = None,
    verbose: bool = False,
) -> Path:
    index_metadata = load_pvc_index_metadata(run_dir, tool, manifest)
    unique_kmers_path, graph_path, blocks_file = pvc_index_input_paths(
        tool,
        manifest,
        index_metadata,
    )
    validate_pvc_genotype_inputs(manifest, unique_kmers_path, graph_path, blocks_file)

    output_dir = pvc_genotype_dir(run_dir, tool)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = pvc_genotype_prefix(run_dir, tool)
    output_vcf = pvc_genotype_vcf_path(run_dir, tool)

    mka = int(
        mean_kmer_abundance
        if mean_kmer_abundance is not None
        else manifest.get("mean_kmer_abundance", PVC_GENOTYPE_MEAN_KMER_ABUNDANCE)
    )

    if verbose:
        print(f"{tool} genotype output directory: {output_dir}")
        print(
            f"{tool} genotype index source: "
            f"{index_metadata.get('index_tool')} ({index_metadata.get('index_marker')})"
        )
        print(f"{tool} genotype unique kmers: {unique_kmers_path}")
        print(f"{tool} genotype graph: {graph_path}")
        print(f"{tool} genotype blocks: {blocks_file}")
        print(f"{tool} genotype mean k-mer abundance: {mka}")

    t = time.perf_counter()
    graph = read_graph(graph_path)
    if verbose:
        print(f"{tool} loaded graph in {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    unique_kmers_map = read_unique_kmers_map_from_cereal(unique_kmers_path)
    if verbose:
        print(f"{tool} loaded unique kmers map in {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    probability_table = ProbabilityTable(0, mka * 10, mka * 2)
    genotyping_plan = build_genotyping_plan(
        unique_kmers_map,
        blocks_file,
        output_dir / "ir",
        sample=manifest.get("sample", "sample"),
        source={
            "read_unique_kmers": str(unique_kmers_path),
            "graph": str(graph_path),
            "blocks_file": str(blocks_file),
        },
        verbose=verbose,
    )
    ir_dir = genotyping_plan.root if genotyping_plan.root is not None else output_dir / "ir"
    prob_vectors = compute_probability_vectors(unique_kmers_map, probability_table)
    if verbose:
        print(
            f"{tool} built control IR and computed probability vectors in "
            f"{time.perf_counter() - t:.1f}s"
        )

    results = run_genotyping(
        unique_kmers_map,
        prob_vectors,
        manifest["panel_subset"],
        verbose=verbose,
        blocks_file=blocks_file,
        ir=genotyping_plan,
    )
    likelihoods_path = write_bubble_likelihoods(
        results,
        Path(ir_dir) / "bubble_likelihoods.tsv",
        unique_kmers_map=unique_kmers_map,
    )

    write_output_vcf(
        results,
        graph,
        unique_kmers_map,
        output_vcf,
        sample_name=manifest.get("sample", "SAMPLE"),
    )

    accuracy = None
    if manifest.get("truth") and is_output_rank():
        accuracy = verify_results(results, graph, manifest["truth"])

    marker_path = write_pvc_genotype_complete(
        run_dir,
        tool,
        output_prefix,
        output_vcf,
        ir_dir,
        index_metadata,
        accuracy,
    )

    if verbose:
        print(f"{tool} genotype output VCF: {output_vcf}")
        print(f"{tool} genotype IR: {ir_dir}")
        print(f"{tool} genotype likelihood IR: {likelihoods_path}")
        print(f"{tool} genotype marker: {marker_path}")

    return output_prefix


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PVC genotyping directly.")
    parser.add_argument("--mode", choices=("plaintext", "light", "medium", "heavy"), default="plaintext")
    parser.add_argument("--cereal", required=True, help="UniqueKmersMap cereal JSON")
    parser.add_argument("--graph", required=True, help="Graph JSON")
    parser.add_argument("--panel-vcf", required=True, help="Panel VCF used for blocks")
    parser.add_argument("--vcf", default=None, help="Optional truth VCF for verification")
    parser.add_argument("--blocks-file", default=None, help="Precomputed PLINK .blocks.det")
    parser.add_argument("--mean-kmer-abundance", type=int, default=PVC_GENOTYPE_MEAN_KMER_ABUNDANCE)
    parser.add_argument("--output", required=True, help="Output VCF path")
    parser.add_argument("--sample-name", default="SAMPLE")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None):
    args = parse_args(argv)
    if args.mode != "plaintext":
        from pvc.genotype.private.pvc_private_genotype import run_private_from_files
        return run_private_from_files(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    print("=" * 60)
    print("PVC Genotyping")
    print("=" * 60)

    # Load data
    print(f"\nLoading graph from {args.graph}...")
    t = time.perf_counter()
    graph = read_graph(args.graph)
    print(f"  Loaded in {time.perf_counter() - t:.1f}s")

    print(f"Loading unique kmers map from {args.cereal}...")
    t = time.perf_counter()
    unique_kmers_map = read_unique_kmers_map_from_cereal(args.cereal)
    print(f"  Loaded in {time.perf_counter() - t:.1f}s")

    # Build probability table
    mka = args.mean_kmer_abundance
    print(f"Building probability table (coverage range 0-{mka*10}, count max {mka*2})...")
    t = time.perf_counter()
    probability_table = ProbabilityTable(0, mka * 10, mka * 2)
    print(f"  Built in {time.perf_counter() - t:.1f}s")

    # Preprocessing
    print("\nComputing probability vectors (client side)...")
    t = time.perf_counter()
    prob_vectors = compute_probability_vectors(unique_kmers_map, probability_table)
    print(f"  Done in {time.perf_counter() - t:.1f}s")

    # Genotyping
    print("\n" + "=" * 60)
    print("Running genotyping...")
    print("=" * 60)
    t = time.perf_counter()

    results = run_genotyping(
        unique_kmers_map, prob_vectors,
        args.panel_vcf, verbose=args.verbose,
        blocks_file=getattr(args, 'blocks_file', None)
    )

    genotyping_time = time.perf_counter() - t
    print(f"\nGenotyping completed in {genotyping_time:.1f}s")

    # Output VCF
    if args.output and is_output_rank():
        print(f"\nWriting output VCF to {args.output}...")
        write_output_vcf(results, graph, unique_kmers_map, args.output, sample_name=args.sample_name)

    if args.vcf and is_output_rank():
        print("\n" + "=" * 60)
        print("Verification against ground truth")
        print("=" * 60)
        verify_results(results, graph, args.vcf)
    
    total_time = time.perf_counter() - t_start
    print(f"\nTotal time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
