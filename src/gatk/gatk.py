from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pvc.config import (
    BCFTOOLS_BIN,
    GATK_BIN_CANDIDATES,
    GATK_READ_ROOT_CANDIDATES,
    PROJECT_ROOT,
    SAMTOOLS_BIN_CANDIDATES,
)


GATK_DEDUP_ALIGNMENT_KEYS = (
    "dedup_bam",
    "dedup_cram",
    "gatk_dedup_bam",
    "gatk_dedup_cram",
)

GATK_SOURCE_ALIGNMENT_KEYS = (
    "gatk_bam",
    "gatk_cram",
    "input_bam",
    "input_cram",
    "bam",
    "cram",
    "alignment",
)

GATK_ALIGNMENT_KEYS = GATK_DEDUP_ALIGNMENT_KEYS + GATK_SOURCE_ALIGNMENT_KEYS


def gatk_index_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "gatk" / "index"


def gatk_prep_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "gatk" / "prep"


def gatk_genotype_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "gatk" / "genotype"


def gatk_discovery_index_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "gatk-discovery" / "index"


def gatk_discovery_genotype_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "gatk-discovery" / "genotype"


def gatk_prep_complete_path(run_dir: str | Path) -> Path:
    return gatk_prep_dir(run_dir) / "prep_complete.json"


def gatk_index_complete_path(run_dir: str | Path) -> Path:
    return gatk_index_dir(run_dir) / "index_complete.json"


def gatk_genotype_complete_path(run_dir: str | Path) -> Path:
    return gatk_genotype_dir(run_dir) / "genotype_complete.json"


def gatk_discovery_index_complete_path(run_dir: str | Path) -> Path:
    return gatk_discovery_index_dir(run_dir) / "index_complete.json"


def gatk_discovery_genotype_complete_path(run_dir: str | Path) -> Path:
    return gatk_discovery_genotype_dir(run_dir) / "genotype_complete.json"


def gatk_alleles_vcf_path(run_dir: str | Path) -> Path:
    return gatk_index_dir(run_dir) / "alleles_no_large.vcf.gz"


def gatk_prepared_bam_path(run_dir: str | Path, manifest: dict[str, Any]) -> Path:
    sample = str(manifest.get("sample", "sample"))
    return gatk_prep_dir(run_dir) / f"{sample}.dedup.bam"


def gatk_genotype_vcf_gz_path(run_dir: str | Path) -> Path:
    return gatk_genotype_dir(run_dir) / "gatk_regenotyped.vcf.gz"


def gatk_genotype_vcf_path(run_dir: str | Path) -> Path:
    return gatk_genotype_dir(run_dir) / "gatk_regenotyped.vcf"


def gatk_discovery_vcf_gz_path(run_dir: str | Path) -> Path:
    return gatk_discovery_genotype_dir(run_dir) / "gatk_discovery.vcf.gz"


def gatk_discovery_vcf_path(run_dir: str | Path) -> Path:
    return gatk_discovery_genotype_dir(run_dir) / "gatk_discovery.vcf"


def gatk_bin() -> Path:
    env_bin = os.environ.get("GATK_BIN")
    candidates = [Path(env_bin)] if env_bin else []
    candidates.extend(GATK_BIN_CANDIDATES)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"GATK binary not found. Searched:\n  {searched}")


def samtools_bin() -> str:
    env_bin = os.environ.get("SAMTOOLS_BIN")
    candidates: list[str] = []
    if env_bin:
        candidates.append(env_bin)
    path_bin = shutil.which("samtools")
    if path_bin:
        candidates.append(path_bin)
    candidates.extend(SAMTOOLS_BIN_CANDIDATES)

    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    searched = "\n  ".join(candidates)
    raise FileNotFoundError(f"samtools binary not found. Searched:\n  {searched}")


def read_roots() -> list[Path]:
    env_roots = os.environ.get("PVC_GATK_READ_ROOTS")
    if env_roots:
        return [Path(root) for root in env_roots.split(":") if root]
    return list(GATK_READ_ROOT_CANDIDATES)


def manifest_interval(manifest: dict[str, Any]) -> str:
    interval = manifest.get("interval") or manifest.get("chromosome")
    if not isinstance(interval, str) or not interval:
        raise ValueError("Manifest is missing required GATK interval/chromosome")
    return interval


def manifest_chromosome(manifest: dict[str, Any]) -> str:
    return manifest_interval(manifest).split(":", 1)[0]


def reference_dict_path(reference_fasta: str | Path) -> Path:
    return Path(reference_fasta).with_suffix(".dict")


def default_threads() -> int:
    raw_threads = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    try:
        return max(1, int(raw_threads))
    except ValueError:
        return 1


def alignment_index_candidates(alignment: str | Path) -> list[Path]:
    path = Path(alignment)
    if path.suffix == ".bam":
        return [path.with_suffix(".bai"), Path(str(path) + ".bai")]
    if path.suffix == ".cram":
        return [path.with_suffix(".crai"), Path(str(path) + ".crai")]
    return []


def alignment_has_index(alignment: str | Path) -> bool:
    candidates = alignment_index_candidates(alignment)
    return not candidates or any(candidate.exists() for candidate in candidates)


def manifest_path_for_keys(
    manifest: dict[str, Any],
    keys: tuple[str, ...],
) -> Path | None:
    for key in keys:
        value = manifest.get(key)
        if not value:
            continue

        path = Path(str(value))
        if not path.exists():
            raise FileNotFoundError(f"Manifest GATK alignment key {key} not found: {path}")
        return path

    return None


def shared_sample_dedup_bam(manifest: dict[str, Any]) -> Path | None:
    if manifest_chromosome(manifest) != "chr20":
        return None

    sample = str(manifest.get("sample", "sample"))
    path = (
        PROJECT_ROOT.parent
        / "pvc_l1o"
        / "data"
        / "pangenome"
        / "PanGenie-Experiments"
        / sample
        / "gatk"
        / f"{sample}.dedup.bam"
    )
    return path if path.exists() else None


def manifest_dedup_alignment_path(manifest: dict[str, Any]) -> Path | None:
    explicit_path = manifest_path_for_keys(manifest, GATK_DEDUP_ALIGNMENT_KEYS)
    if explicit_path is not None:
        return explicit_path
    # A manifest-specified source alignment is condition-specific (for example,
    # a coverage-downsampled BAM) and must take precedence over the historical
    # chr20 full-depth duplicate-marked fallback.
    if manifest_path_for_keys(manifest, GATK_SOURCE_ALIGNMENT_KEYS) is not None:
        return None
    return shared_sample_dedup_bam(manifest)


def manifest_source_alignment_path(manifest: dict[str, Any]) -> Path:
    explicit_path = manifest_path_for_keys(manifest, GATK_SOURCE_ALIGNMENT_KEYS)
    if explicit_path is not None:
        return explicit_path

    sample = str(manifest.get("sample", "sample"))
    chromosome = manifest_chromosome(manifest)
    candidates: list[Path] = []
    for root in read_roots():
        sample_dir = root / sample
        candidates.extend(
            [
                sample_dir / f"{sample}.final.{chromosome}.bam",
                sample_dir / f"{sample}.{chromosome}.bam",
                sample_dir / f"{sample}.final.bam",
                sample_dir / f"{sample}.bam",
                sample_dir / f"{sample}.final.cram",
                sample_dir / f"{sample}.cram",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not find a GATK input BAM/CRAM. Add one of "
        f"{GATK_ALIGNMENT_KEYS} to the manifest, or place reads under a known root. "
        f"Searched:\n  {searched}"
    )


def build_gatk_markduplicates_command(
    manifest: dict[str, Any],
    source_alignment: str | Path,
    output_bam: str | Path,
    metrics_file: str | Path,
) -> list[str]:
    return [
        str(gatk_bin()),
        "MarkDuplicates",
        "-I",
        str(source_alignment),
        "-O",
        str(output_bam),
        "-M",
        str(metrics_file),
        "--CREATE_INDEX",
        "true",
        "-R",
        str(manifest["reference_fasta"]),
    ]


def gatk_extracted_bam_path(run_dir: str | Path, manifest: dict[str, Any]) -> Path:
    sample = str(manifest.get("sample", "sample"))
    chromosome = manifest_chromosome(manifest)
    return gatk_prep_dir(run_dir) / f"{sample}.{chromosome}.bam"


def build_samtools_view_command(
    manifest: dict[str, Any],
    source_alignment: str | Path,
    chromosome: str,
) -> list[str]:
    command = [
        samtools_bin(),
        "view",
        "-@",
        str(default_threads()),
        "-b",
    ]
    if Path(source_alignment).suffix == ".cram":
        command.extend(["-T", str(manifest["reference_fasta"])])
    command.extend([str(source_alignment), chromosome])
    return command


def build_samtools_sort_command(output_bam: str | Path) -> list[str]:
    return [
        samtools_bin(),
        "sort",
        "-@",
        str(default_threads()),
        "-o",
        str(output_bam),
        "-",
    ]


def build_samtools_index_command(bam: str | Path) -> list[str]:
    return [
        samtools_bin(),
        "index",
        "-@",
        str(default_threads()),
        str(bam),
    ]


def run_samtools_extract_chromosome(
    manifest: dict[str, Any],
    source_alignment: str | Path,
    output_bam: str | Path,
    output_dir: Path,
    verbose: bool = False,
) -> None:
    chromosome = manifest_chromosome(manifest)
    view_command = build_samtools_view_command(manifest, source_alignment, chromosome)
    sort_command = build_samtools_sort_command(output_bam)
    metadata_path = output_dir / "samtools_extract_chromosome_command.json"
    metadata_path.write_text(
        json.dumps(
            {
                "view_command": view_command,
                "sort_command": sort_command,
                "cwd": str(PROJECT_ROOT),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if verbose:
        print(f"Saved samtools_extract_chromosome command metadata: {metadata_path}")
        print(f"samtools view command: {' '.join(view_command)}")
        print(f"samtools sort command: {' '.join(sort_command)}")

    stdout_path = output_dir / "samtools_extract_chromosome_stdout.log"
    stderr_path = output_dir / "samtools_extract_chromosome_stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout_file:
        with stderr_path.open("w", encoding="utf-8") as stderr_file:
            view = subprocess.Popen(
                view_command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=False,
            )
            assert view.stdout is not None
            sort = subprocess.Popen(
                sort_command,
                cwd=PROJECT_ROOT,
                stdin=view.stdout,
                stdout=stdout_file,
                stderr=stderr_file,
                text=False,
            )
            view.stdout.close()
            sort_returncode = sort.wait()
            view_returncode = view.wait()

    if view_returncode != 0 or sort_returncode != 0:
        raise RuntimeError(
            "samtools chromosome extraction failed with "
            f"view={view_returncode}, sort={sort_returncode}; see {stderr_path}"
        )

    result = run_gatk_aux_command(
        build_samtools_index_command(output_bam),
        output_dir,
        "samtools_index_chromosome_bam",
        verbose=verbose,
    )
    require_success(result, output_dir, "samtools_index_chromosome_bam")


def write_gatk_prep_complete(
    run_dir: str | Path,
    alignment: str | Path,
    source_alignment: str | Path,
    reference_dict: str | Path,
    duplicate_marked: bool,
) -> Path:
    marker_path = gatk_prep_complete_path(run_dir)
    marker_path.write_text(
        json.dumps(
            {
                "alignment": str(alignment),
                "source_alignment": str(source_alignment),
                "reference_dict": str(reference_dict),
                "duplicate_marked": duplicate_marked,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return marker_path


def read_gatk_prep_complete(run_dir: str | Path) -> dict[str, Any]:
    marker_path = gatk_prep_complete_path(run_dir)
    if not marker_path.exists():
        raise FileNotFoundError(
            "GATK genotype requires a completed duplicate-marking prep stage. "
            f"Missing {marker_path}. Run the gatk index or gatk-discovery index stage first."
        )
    with marker_path.open("r", encoding="utf-8") as marker_file:
        metadata = json.load(marker_file)
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected JSON object in {marker_path}")
    return metadata


def run_gatk_prepare_alignment(
    manifest: dict[str, Any],
    run_dir: str | Path,
    verbose: bool = False,
) -> Path:
    if "reference_fasta" not in manifest:
        raise ValueError("Manifest is missing required GATK prep key: reference_fasta")

    output_dir = gatk_prep_dir(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dict = ensure_reference_dictionary(manifest, output_dir, verbose=verbose)

    existing_dedup = manifest_dedup_alignment_path(manifest)
    if existing_dedup is not None:
        if not alignment_has_index(existing_dedup):
            searched = ", ".join(str(path) for path in alignment_index_candidates(existing_dedup))
            raise FileNotFoundError(
                f"Duplicate-marked GATK alignment has no index: {existing_dedup}. "
                f"Searched: {searched}"
            )
        marker_path = write_gatk_prep_complete(
            run_dir,
            existing_dedup,
            existing_dedup,
            reference_dict,
            duplicate_marked=True,
        )
        if verbose:
            print(f"GATK prep using existing duplicate-marked alignment: {existing_dedup}")
            print(f"GATK prep marker: {marker_path}")
        return existing_dedup

    output_bam = gatk_prepared_bam_path(run_dir, manifest)
    metrics_file = output_dir / "mark_duplicates_metrics.txt"
    if output_bam.exists() and alignment_has_index(output_bam):
        source_alignment = manifest_source_alignment_path(manifest)
        marker_path = write_gatk_prep_complete(
            run_dir,
            output_bam,
            source_alignment,
            reference_dict,
            duplicate_marked=True,
        )
        if verbose:
            print(f"GATK prep duplicate-marked BAM already exists: {output_bam}")
            print(f"GATK prep marker: {marker_path}")
        return output_bam

    source_alignment = manifest_source_alignment_path(manifest)
    markduplicates_input = source_alignment
    if Path(source_alignment).suffix == ".cram":
        extracted_bam = gatk_extracted_bam_path(run_dir, manifest)
        if not (extracted_bam.exists() and alignment_has_index(extracted_bam)):
            run_samtools_extract_chromosome(
                manifest,
                source_alignment,
                extracted_bam,
                output_dir,
                verbose=verbose,
            )
        markduplicates_input = extracted_bam

    command = build_gatk_markduplicates_command(
        manifest,
        markduplicates_input,
        output_bam,
        metrics_file,
    )
    result = run_gatk_aux_command(
        command,
        output_dir,
        "gatk_mark_duplicates",
        verbose=verbose,
    )
    require_success(result, output_dir, "gatk_mark_duplicates")
    if not output_bam.exists():
        raise FileNotFoundError(
            f"GATK MarkDuplicates completed but did not create {output_bam}"
        )
    if not alignment_has_index(output_bam):
        searched = ", ".join(str(path) for path in alignment_index_candidates(output_bam))
        raise FileNotFoundError(
            f"GATK MarkDuplicates completed but did not create a BAM index. "
            f"Searched: {searched}"
        )

    marker_path = write_gatk_prep_complete(
        run_dir,
        output_bam,
        source_alignment,
        reference_dict,
        duplicate_marked=True,
    )
    if verbose:
        print(f"GATK prep source alignment: {source_alignment}")
        print(f"GATK prep duplicate-marked BAM: {output_bam}")
        print(f"GATK prep marker: {marker_path}")

    return output_bam


def run_gatk_aux_command(
    command: list[str],
    output_dir: Path,
    name: str,
    verbose: bool = False,
) -> subprocess.CompletedProcess[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}_command.json").write_text(
        json.dumps({"command": command, "cwd": str(PROJECT_ROOT)}, indent=2) + "\n",
        encoding="utf-8",
    )
    if verbose:
        print(f"Saved {name} command metadata: {output_dir / f'{name}_command.json'}")
        print(f"{name} command: {' '.join(command)}")

    try:
        with (output_dir / f"{name}_stdout.log").open("w", encoding="utf-8") as stdout_file:
            with (output_dir / f"{name}_stderr.log").open(
                "w",
                encoding="utf-8",
            ) as stderr_file:
                return subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    check=False,
                )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Required command not found while running {name}: {command[0]}"
        ) from exc


def require_success(
    result: subprocess.CompletedProcess[str],
    output_dir: Path,
    name: str,
) -> None:
    if result.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {result.returncode}; "
            f"see {output_dir / f'{name}_stderr.log'}"
        )


def ensure_reference_dictionary(
    manifest: dict[str, Any],
    output_dir: Path,
    verbose: bool = False,
) -> Path:
    reference_fasta = Path(manifest["reference_fasta"])
    dict_path = reference_dict_path(reference_fasta)
    if dict_path.exists():
        if verbose:
            print(f"GATK reference dictionary exists: {dict_path}")
        return dict_path

    command = [
        str(gatk_bin()),
        "CreateSequenceDictionary",
        "-R",
        str(reference_fasta),
    ]
    result = run_gatk_aux_command(
        command,
        output_dir,
        "gatk_create_sequence_dictionary",
        verbose=verbose,
    )
    if result.returncode != 0 and dict_path.exists():
        if verbose:
            print(
                "GATK CreateSequenceDictionary exited non-zero, but the "
                f"reference dictionary now exists: {dict_path}"
            )
        return dict_path
    require_success(result, output_dir, "gatk_create_sequence_dictionary")
    if not dict_path.exists():
        raise FileNotFoundError(
            f"GATK CreateSequenceDictionary completed but did not create {dict_path}"
        )
    return dict_path


def build_bcftools_view_biallelic_command(
    manifest: dict[str, Any],
    output_vcf: str | Path,
) -> list[str]:
    return [
        str(BCFTOOLS_BIN),
        "view",
        "-m2",
        "-M2",
        str(manifest["panel_subset"]),
        "-o",
        str(output_vcf),
    ]


def build_bcftools_annotate_command(
    input_vcf: str | Path,
    output_vcf: str | Path,
) -> list[str]:
    return [
        str(BCFTOOLS_BIN),
        "annotate",
        "-x",
        "INFO/AT,INFO/LV,INFO/PS,INFO/ID,INFO/CONFLICT",
        str(input_vcf),
        "-o",
        str(output_vcf),
    ]


def filter_variants_under_50bp(input_vcf: str | Path, output_vcf: str | Path) -> int:
    kept = 0
    with Path(input_vcf).open("r", encoding="utf-8") as src:
        with Path(output_vcf).open("w", encoding="utf-8") as dst:
            for line in src:
                if line.startswith("#"):
                    dst.write(line)
                    continue

                cols = line.rstrip("\n").split("\t")
                if len(cols) < 5:
                    continue

                ref = cols[3]
                alt = cols[4]
                diff = abs(len(alt) - len(ref))
                if diff < 50:
                    dst.write(line)
                    kept += 1

    return kept


def build_bcftools_compress_command(
    input_vcf: str | Path,
    output_vcf_gz: str | Path,
) -> list[str]:
    return [
        str(BCFTOOLS_BIN),
        "view",
        str(input_vcf),
        "-Oz",
        "-o",
        str(output_vcf_gz),
    ]


def build_bcftools_index_command(vcf_gz: str | Path) -> list[str]:
    return [str(BCFTOOLS_BIN), "index", "-t", str(vcf_gz)]


def write_gatk_index_complete(
    run_dir: str | Path,
    alleles_vcf: str | Path,
    reference_dict: str | Path,
    alignment: str | Path,
    kept_variants: int,
) -> Path:
    marker_path = gatk_index_complete_path(run_dir)
    marker_path.write_text(
        json.dumps(
            {
                "alleles_vcf": str(alleles_vcf),
                "reference_dict": str(reference_dict),
                "alignment": str(alignment),
                "kept_variants": kept_variants,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return marker_path


def read_gatk_index_complete(run_dir: str | Path) -> dict[str, Any]:
    marker_path = gatk_index_complete_path(run_dir)
    if not marker_path.exists():
        raise FileNotFoundError(
            "GATK genotype requires a completed GATK index in this run. "
            f"Missing {marker_path}. Run the gatk index stage first."
        )
    with marker_path.open("r", encoding="utf-8") as marker_file:
        metadata = json.load(marker_file)
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected JSON object in {marker_path}")
    return metadata


def run_gatk_index(
    manifest: dict[str, Any],
    run_dir: str | Path,
    verbose: bool = False,
) -> Path:
    missing_keys = [
        key for key in ("reference_fasta", "panel_subset", "interval") if key not in manifest
    ]
    if missing_keys:
        raise ValueError(f"Manifest is missing required GATK index keys: {missing_keys}")

    output_dir = gatk_index_dir(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alignment = run_gatk_prepare_alignment(manifest, run_dir, verbose=verbose)
    alleles_vcf_gz = gatk_alleles_vcf_path(run_dir)
    alleles_tbi = alleles_vcf_gz.with_suffix(alleles_vcf_gz.suffix + ".tbi")
    reference_dict = ensure_reference_dictionary(manifest, output_dir, verbose=verbose)

    if alleles_vcf_gz.exists() and alleles_tbi.exists():
        marker_path = write_gatk_index_complete(
            run_dir,
            alleles_vcf_gz,
            reference_dict,
            alignment,
            kept_variants=-1,
        )
        if verbose:
            print(f"GATK filtered alleles already exist: {alleles_vcf_gz}")
            print(f"GATK index marker: {marker_path}")
        return alleles_vcf_gz

    biallelic_vcf = output_dir / "alleles_biallelic.vcf"
    annotated_vcf = output_dir / "alleles_biallelic_annotated.vcf"
    filtered_vcf = output_dir / "alleles_no_large.vcf"

    view_result = run_gatk_aux_command(
        build_bcftools_view_biallelic_command(manifest, biallelic_vcf),
        output_dir,
        "bcftools_view_biallelic",
        verbose=verbose,
    )
    require_success(view_result, output_dir, "bcftools_view_biallelic")

    annotate_result = run_gatk_aux_command(
        build_bcftools_annotate_command(biallelic_vcf, annotated_vcf),
        output_dir,
        "bcftools_annotate_strip_graph_info",
        verbose=verbose,
    )
    require_success(annotate_result, output_dir, "bcftools_annotate_strip_graph_info")

    kept_variants = filter_variants_under_50bp(annotated_vcf, filtered_vcf)

    compress_result = run_gatk_aux_command(
        build_bcftools_compress_command(filtered_vcf, alleles_vcf_gz),
        output_dir,
        "bcftools_compress_alleles",
        verbose=verbose,
    )
    require_success(compress_result, output_dir, "bcftools_compress_alleles")

    index_result = run_gatk_aux_command(
        build_bcftools_index_command(alleles_vcf_gz),
        output_dir,
        "bcftools_index_alleles",
        verbose=verbose,
    )
    require_success(index_result, output_dir, "bcftools_index_alleles")

    marker_path = write_gatk_index_complete(
        run_dir,
        alleles_vcf_gz,
        reference_dict,
        alignment,
        kept_variants,
    )
    if verbose:
        print(f"GATK filtered alleles: {alleles_vcf_gz}")
        print(f"GATK kept variants after <50bp filter: {kept_variants}")
        print(f"GATK index marker: {marker_path}")

    return alleles_vcf_gz


def manifest_alignment_path(manifest: dict[str, Any]) -> Path:
    dedup_alignment = manifest_dedup_alignment_path(manifest)
    if dedup_alignment is not None:
        return dedup_alignment
    return manifest_source_alignment_path(manifest)


def build_gatk_haplotypecaller_command(
    manifest: dict[str, Any],
    alignment: str | Path,
    alleles_vcf: str | Path,
    output_vcf_gz: str | Path,
    threads: int,
) -> list[str]:
    return [
        str(gatk_bin()),
        "HaplotypeCaller",
        "-R",
        str(manifest["reference_fasta"]),
        "-I",
        str(alignment),
        "-O",
        str(output_vcf_gz),
        "-L",
        manifest_interval(manifest),
        "--minimum-mapping-quality",
        "20",
        "--genotyping-mode",
        "GENOTYPE_GIVEN_ALLELES",
        "--alleles",
        str(alleles_vcf),
        "--output-mode",
        "EMIT_ALL_SITES",
        "-stand-call-conf",
        "0",
        "--native-pair-hmm-threads",
        str(threads),
    ]


def gunzip_vcf(input_vcf_gz: str | Path, output_vcf: str | Path) -> None:
    with gzip.open(input_vcf_gz, "rt", encoding="utf-8") as src:
        with Path(output_vcf).open("w", encoding="utf-8") as dst:
            for line in src:
                dst.write(line)


def write_gatk_genotype_complete(
    run_dir: str | Path,
    genotype_vcf: str | Path,
    genotype_vcf_gz: str | Path,
    alignment: str | Path,
    alleles_vcf: str | Path,
) -> Path:
    marker_path = gatk_genotype_complete_path(run_dir)
    marker_path.write_text(
        json.dumps(
            {
                "genotype_vcf": str(genotype_vcf),
                "genotype_vcf_gz": str(genotype_vcf_gz),
                "alignment": str(alignment),
                "alleles_vcf": str(alleles_vcf),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return marker_path


def run_gatk_genotype(
    manifest: dict[str, Any],
    run_dir: str | Path,
    threads: int | None = None,
    verbose: bool = False,
) -> Path:
    missing_keys = [
        key for key in ("reference_fasta", "panel_subset", "interval") if key not in manifest
    ]
    if missing_keys:
        raise ValueError(f"Manifest is missing required GATK genotype keys: {missing_keys}")

    index_metadata = read_gatk_index_complete(run_dir)
    alleles_vcf = Path(index_metadata["alleles_vcf"])
    if not alleles_vcf.exists():
        raise FileNotFoundError(f"GATK filtered alleles VCF not found: {alleles_vcf}")

    prep_metadata = read_gatk_prep_complete(run_dir)
    alignment = Path(prep_metadata["alignment"])
    if not alignment.exists():
        raise FileNotFoundError(f"GATK duplicate-marked alignment not found: {alignment}")

    output_dir = gatk_genotype_dir(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_vcf_gz = gatk_genotype_vcf_gz_path(run_dir)
    output_vcf = gatk_genotype_vcf_path(run_dir)
    thread_count = threads if threads is not None else default_threads()

    if output_vcf.exists() and output_vcf_gz.exists():
        marker_path = write_gatk_genotype_complete(
            run_dir,
            output_vcf,
            output_vcf_gz,
            alignment,
            alleles_vcf,
        )
        if verbose:
            print(f"GATK genotype result already exists: {output_vcf}")
            print(f"GATK genotype marker: {marker_path}")
        return output_vcf

    ensure_reference_dictionary(manifest, output_dir, verbose=verbose)

    command = build_gatk_haplotypecaller_command(
        manifest,
        alignment,
        alleles_vcf,
        output_vcf_gz,
        thread_count,
    )
    if verbose:
        print(f"GATK genotype output directory: {output_dir}")
        print(f"GATK alignment: {alignment}")
        print(f"GATK alleles: {alleles_vcf}")

    result = run_gatk_aux_command(
        command,
        output_dir,
        "gatk_haplotypecaller",
        verbose=verbose,
    )
    require_success(result, output_dir, "gatk_haplotypecaller")

    gunzip_vcf(output_vcf_gz, output_vcf)
    marker_path = write_gatk_genotype_complete(
        run_dir,
        output_vcf,
        output_vcf_gz,
        alignment,
        alleles_vcf,
    )
    if verbose:
        print(f"GATK genotype VCF: {output_vcf}")
        print(f"GATK genotype marker: {marker_path}")

    return output_vcf


def write_gatk_discovery_index_complete(
    run_dir: str | Path,
    alignment: str | Path,
    reference_dict: str | Path,
) -> Path:
    marker_path = gatk_discovery_index_complete_path(run_dir)
    marker_path.write_text(
        json.dumps(
            {
                "alignment": str(alignment),
                "reference_dict": str(reference_dict),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return marker_path


def read_gatk_discovery_index_complete(run_dir: str | Path) -> dict[str, Any]:
    marker_path = gatk_discovery_index_complete_path(run_dir)
    if not marker_path.exists():
        raise FileNotFoundError(
            "GATK discovery genotype requires a completed discovery index/prep stage. "
            f"Missing {marker_path}. Run the gatk-discovery index stage first."
        )
    with marker_path.open("r", encoding="utf-8") as marker_file:
        metadata = json.load(marker_file)
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected JSON object in {marker_path}")
    return metadata


def run_gatk_discovery_index(
    manifest: dict[str, Any],
    run_dir: str | Path,
    verbose: bool = False,
) -> Path:
    missing_keys = [key for key in ("reference_fasta", "interval") if key not in manifest]
    if missing_keys:
        raise ValueError(f"Manifest is missing required GATK discovery index keys: {missing_keys}")

    output_dir = gatk_discovery_index_dir(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alignment = run_gatk_prepare_alignment(manifest, run_dir, verbose=verbose)
    reference_dict = ensure_reference_dictionary(manifest, output_dir, verbose=verbose)
    marker_path = write_gatk_discovery_index_complete(run_dir, alignment, reference_dict)
    if verbose:
        print(f"GATK discovery alignment: {alignment}")
        print(f"GATK discovery index marker: {marker_path}")
    return alignment


def build_gatk_discovery_command(
    manifest: dict[str, Any],
    alignment: str | Path,
    output_vcf_gz: str | Path,
    threads: int,
) -> list[str]:
    return [
        str(gatk_bin()),
        "HaplotypeCaller",
        "-R",
        str(manifest["reference_fasta"]),
        "-I",
        str(alignment),
        "-O",
        str(output_vcf_gz),
        "-L",
        manifest_interval(manifest),
        "--minimum-mapping-quality",
        "20",
        "--genotyping-mode",
        "DISCOVERY",
        "--native-pair-hmm-threads",
        str(threads),
    ]


def write_gatk_discovery_genotype_complete(
    run_dir: str | Path,
    genotype_vcf: str | Path,
    genotype_vcf_gz: str | Path,
    alignment: str | Path,
) -> Path:
    marker_path = gatk_discovery_genotype_complete_path(run_dir)
    marker_path.write_text(
        json.dumps(
            {
                "genotype_vcf": str(genotype_vcf),
                "genotype_vcf_gz": str(genotype_vcf_gz),
                "alignment": str(alignment),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return marker_path


def run_gatk_discovery_genotype(
    manifest: dict[str, Any],
    run_dir: str | Path,
    threads: int | None = None,
    verbose: bool = False,
) -> Path:
    missing_keys = [key for key in ("reference_fasta", "interval") if key not in manifest]
    if missing_keys:
        raise ValueError(f"Manifest is missing required GATK discovery genotype keys: {missing_keys}")

    index_metadata = read_gatk_discovery_index_complete(run_dir)
    alignment = Path(index_metadata["alignment"])
    if not alignment.exists():
        raise FileNotFoundError(f"GATK discovery alignment not found: {alignment}")

    output_dir = gatk_discovery_genotype_dir(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_vcf_gz = gatk_discovery_vcf_gz_path(run_dir)
    output_vcf = gatk_discovery_vcf_path(run_dir)
    thread_count = threads if threads is not None else default_threads()

    if output_vcf.exists() and output_vcf_gz.exists():
        marker_path = write_gatk_discovery_genotype_complete(
            run_dir,
            output_vcf,
            output_vcf_gz,
            alignment,
        )
        if verbose:
            print(f"GATK discovery result already exists: {output_vcf}")
            print(f"GATK discovery genotype marker: {marker_path}")
        return output_vcf

    ensure_reference_dictionary(manifest, output_dir, verbose=verbose)
    command = build_gatk_discovery_command(
        manifest,
        alignment,
        output_vcf_gz,
        thread_count,
    )
    if verbose:
        print(f"GATK discovery output directory: {output_dir}")
        print(f"GATK discovery alignment: {alignment}")

    result = run_gatk_aux_command(
        command,
        output_dir,
        "gatk_haplotypecaller_discovery",
        verbose=verbose,
    )
    require_success(result, output_dir, "gatk_haplotypecaller_discovery")

    gunzip_vcf(output_vcf_gz, output_vcf)
    marker_path = write_gatk_discovery_genotype_complete(
        run_dir,
        output_vcf,
        output_vcf_gz,
        alignment,
    )
    if verbose:
        print(f"GATK discovery VCF: {output_vcf}")
        print(f"GATK discovery genotype marker: {marker_path}")

    return output_vcf
