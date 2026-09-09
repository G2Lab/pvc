from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from pvc.config import PVC_GENOTYPE_MEAN_KMER_ABUNDANCE
from pvc.pangenome.unique_kmers_map import read_unique_kmers_map_from_cereal
from pvc.pangenome.graph import read_graph
from pvc.pangenome.probability_table import ProbabilityTable
from pvc.index.ir import build_genotyping_plan, write_bubble_likelihoods
from pvc.genotype.pvc_genotype import (
    compute_probability_vectors, load_pvc_index_metadata, pvc_index_input_paths,
    validate_pvc_genotype_inputs, verify_results, write_output_vcf,
    write_pvc_genotype_complete,
)
from pvc.genotype.private.workflow_profile import mark_fastq_to_vcf_complete
from pvc.genotype.private.pvc_light import run_genotyping_haplotype_private_light
from pvc.genotype.private.pvc_medium import run_genotyping_haplotype_private_medium
from pvc.genotype.private.pvc_heavy import run_genotyping_haplotype_private_heavy

def _diagnostic_outputs_enabled() -> bool:
    """Resolve the explicit legacy-compatible pre-VCF diagnostic switch."""
    raw = os.environ.get("PVC_WRITE_DIAGNOSTIC_OUTPUTS", "1")
    if raw not in {"0", "1"}:
        raise ValueError(
            "PVC_WRITE_DIAGNOSTIC_OUTPUTS must be exactly 0 or 1; "
            f"got {raw!r}"
        )
    return raw == "1"


PVC_PRIVATE_RUNNERS = {
    "pvc-light": run_genotyping_haplotype_private_light,
    # pvc-light-gpu reuses the exact light runner; the only difference is that
    # local encrypted-tensor compute runs on CUDA (see PVC_GENOTYPE_DEVICE set
    # in _run_replicated_private_genotype). Communication still happens on CPU.
    "pvc-light-gpu": run_genotyping_haplotype_private_light,
    "pvc-medium": run_genotyping_haplotype_private_medium,
    "pvc-heavy": run_genotyping_haplotype_private_heavy,
}

PVC_PRIVATE_METHODS = {
    "pvc-light": "fast",
    "pvc-light-gpu": "fast",
    "pvc-medium": "max-only",
    "pvc-heavy": "complete",
}

def pvc_genotype_dir(run_dir: str | Path, tool: str) -> Path:
    return Path(run_dir) / tool / "genotype"


def pvc_genotype_prefix(run_dir: str | Path, tool: str) -> Path:
    return pvc_genotype_dir(run_dir, tool) / tool


def pvc_genotype_complete_path(run_dir: str | Path, tool: str) -> Path:
    return pvc_genotype_dir(run_dir, tool) / "genotype_complete.json"


def pvc_genotype_vcf_path(run_dir: str | Path, tool: str) -> Path:
    return pvc_genotype_prefix(run_dir, tool).with_suffix(".vcf")


def pvc_private_scores_path(run_dir: str | Path, tool: str) -> Path:
    return pvc_genotype_prefix(run_dir, tool).with_name(f"{tool}_scores.json")


def run_pvc_private_genotype(
    tool: str,
    manifest: dict[str, Any],
    run_dir: str | Path,
    mean_kmer_abundance: int | None = None,
    verbose: bool = False,
) -> Path:
    if tool not in PVC_PRIVATE_RUNNERS:
        raise ValueError(f"No PVC private implementation is configured for tool: {tool}")

    _run_role_separated_private_genotype(
        tool,
        manifest,
        Path(run_dir),
        mean_kmer_abundance,
        verbose,
    )
    return pvc_genotype_prefix(run_dir, tool)


def _run_role_separated_private_genotype(
    tool: str,
    manifest: dict[str, Any],
    run_dir: Path,
    mean_kmer_abundance: int | None,
    verbose: bool,
) -> None:
    """Run with a distinct parent/client and three share-only workers.

    This is the mandatory research execution path.
    """
    from pvc.genotype.private.runtime import spawn_multiparty_rank_args
    from pvc.genotype.private.role_separated import (
        prepare_role_separated_inputs,
        reconstruct_role_separated_results,
        redact_read_evidence,
        run_role_separated_party,
    )

    index_metadata = load_pvc_index_metadata(run_dir, tool, manifest)
    unique_kmers_path, graph_path, blocks_file = pvc_index_input_paths(
        tool, manifest, index_metadata
    )
    validate_pvc_genotype_inputs(
        manifest, unique_kmers_path, graph_path, blocks_file
    )

    output_dir = pvc_genotype_dir(run_dir, tool)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = pvc_genotype_prefix(run_dir, tool)
    output_vcf = pvc_genotype_vcf_path(run_dir, tool)
    mka = int(
        mean_kmer_abundance
        if mean_kmer_abundance is not None
        else manifest.get("mean_kmer_abundance", PVC_GENOTYPE_MEAN_KMER_ABUNDANCE)
    )
    cov_max_mult = int(os.environ.get("PVC_COV_MAX_MULT", "10"))
    count_max_mult = int(os.environ.get("PVC_COUNT_MAX_MULT", "2"))

    # Client-only plaintext operations.  Neither the path nor the resulting
    # object is included in any worker argument tuple.
    graph = read_graph(graph_path)
    private_map = read_unique_kmers_map_from_cereal(unique_kmers_path)
    probability_table = ProbabilityTable(
        0, mka * cov_max_mult, mka * count_max_mult
    )
    prob_vectors = compute_probability_vectors(private_map, probability_table)

    # The IR contains public panel structure. Build it from a redacted map so
    # its disk-backed arrays cannot accidentally persist read evidence.
    public_map = redact_read_evidence(private_map)
    genotyping_plan = build_genotyping_plan(
        public_map,
        blocks_file,
        output_dir / "ir",
        sample=manifest.get("sample", "sample"),
        source={
            "read_unique_kmers": "client-private:not-disclosed-to-parties",
            "graph": str(graph_path),
            "blocks_file": str(blocks_file),
        },
        verbose=verbose,
    )
    public_input, party_inputs = prepare_role_separated_inputs(
        tool,
        private_map,
        prob_vectors,
        probability_table,
        genotyping_plan,
    )

    if verbose:
        print(
            f"{tool}: parent is client; launching three share-only MPC parties and an upstream TTP server",
            flush=True,
        )
    party_outputs = spawn_multiparty_rank_args(
        run_role_separated_party,
        [(public_input, party_inputs[rank]) for rank in range(3)],
    )
    results = reconstruct_role_separated_results(
        tool, public_input, party_outputs, private_map, prob_vectors
    )

    if not _diagnostic_outputs_enabled():
        likelihoods_path = None
    else:
        likelihoods_path = write_bubble_likelihoods(
            results,
            Path(genotyping_plan.root) / "bubble_likelihoods.tsv",
            unique_kmers_map=private_map,
        )
    write_output_vcf(
        results,
        graph,
        private_map,
        output_vcf,
        sample_name=manifest.get("sample", "SAMPLE"),
    )
    mark_fastq_to_vcf_complete(output_vcf)
    accuracy = (
        verify_results(results, graph, manifest["truth"])
        if manifest.get("truth")
        else None
    )
    write_pvc_genotype_complete(
        run_dir,
        tool,
        output_prefix,
        output_vcf,
        genotyping_plan.root,
        index_metadata,
        accuracy,
    )
    if verbose and likelihoods_path is not None:
        print(f"{tool} genotype likelihood IR: {likelihoods_path}")



def run_private_from_files(args):
    """Core file-based CLI for Light, Medium, and Heavy genotyping."""
    import tempfile
    from pvc.genotype.genotyping import run_plink_blocks
    from pvc.genotype.private.role_separated import (
        prepare_role_separated_inputs, reconstruct_role_separated_results,
        redact_read_evidence, run_role_separated_party,
    )
    from pvc.genotype.private.runtime import spawn_multiparty_rank_args

    tool = f"pvc-{args.mode}"
    graph = read_graph(args.graph)
    private_map = read_unique_kmers_map_from_cereal(args.cereal)
    mean = args.mean_kmer_abundance
    if mean <= 0:
        raise ValueError("mean k-mer abundance must be positive")
    table = ProbabilityTable(0, mean * 10, mean * 2)
    probabilities = compute_probability_vectors(private_map, table)
    with tempfile.TemporaryDirectory(prefix="pvc-genotype-") as temporary:
        blocks = args.blocks_file
        if not blocks:
            run_plink_blocks(args.panel_vcf, graph.chromosome, temporary)
            blocks = Path(temporary) / "plink_blocks.blocks.det"
        plan = build_genotyping_plan(redact_read_evidence(private_map), blocks)
        public, parties = prepare_role_separated_inputs(
            tool, private_map, probabilities, table, plan,
        )
        outputs = spawn_multiparty_rank_args(
            run_role_separated_party, [(public, party) for party in parties],
        )
        results = reconstruct_role_separated_results(
            tool, public, outputs, private_map, probabilities,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_output_vcf(results, graph, private_map, output, sample_name=args.sample_name)
    if args.vcf:
        verify_results(results, graph, args.vcf)
    return output
