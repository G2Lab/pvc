from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from pvc.config import PANGENIE_BIN, PANGENIE_INDEX_BIN, PROJECT_ROOT


def pangenie_index_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "pangenie" / "index"


def pangenie_index_prefix(run_dir: str | Path) -> Path:
    return pangenie_index_dir(run_dir) / "pangenie"


def pangenie_index_complete_path(run_dir: str | Path) -> Path:
    return pangenie_index_dir(run_dir) / "index_complete.json"


def pangenie_genotype_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "pangenie" / "genotype"


def pangenie_genotype_prefix(run_dir: str | Path) -> Path:
    return pangenie_genotype_dir(run_dir) / "pangenie"


def pangenie_genotype_vcf_path(run_dir: str | Path) -> Path:
    return pangenie_genotype_dir(run_dir) / "pangenie_genotyping.vcf"


def pangenie_genotype_complete_path(run_dir: str | Path) -> Path:
    return pangenie_genotype_dir(run_dir) / "genotype_complete.json"


def manifest_reused_pangenie_index_prefix(manifest: dict[str, Any]) -> Path | None:
    for key in ("pangenie_index_prefix", "pangenie_reuse_index_prefix"):
        value = manifest.get(key)
        if not value:
            continue
        prefix = Path(str(value))
        required = [
            Path(f"{prefix}_UniqueKmersMap.cereal"),
            Path(f"{prefix}_path_segments.fasta"),
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"PanGenie reuse index prefix {prefix} is missing required artifacts: "
                f"{[str(path) for path in missing]}"
            )
        return prefix
    return None


def build_pangenie_index_command(
    manifest: dict[str, Any],
    output_prefix: str | Path,
    threads: int = 1,
) -> list[str]:
    return [
        str(PANGENIE_INDEX_BIN),
        "-r",
        str(manifest["reference_fasta"]),
        "-v",
        str(manifest["panel_subset"]),
        "-o",
        str(output_prefix),
        "-t",
        str(threads),
    ]


def build_pangenie_genotype_command(
    manifest: dict[str, Any],
    index_prefix: str | Path,
    output_prefix: str | Path,
    threads: int = 1,
) -> list[str]:
    genotype_input = manifest.get("pangenie_input", manifest.get("fastq"))
    if not genotype_input:
        raise ValueError("Manifest requires fastq or pangenie_input")
    return [
        str(PANGENIE_BIN),
        "-f",
        str(index_prefix),
        "-i",
        str(genotype_input),
        "-o",
        str(output_prefix),
        "-s",
        str(manifest.get("sample", "sample")),
        "-g",
        "-j",
        str(threads),
        "-t",
        str(threads),
    ]


def run_command(
    command: list[str],
    output_dir: Path,
    verbose: bool = False,
) -> subprocess.CompletedProcess[str]:
    (output_dir / "command.json").write_text(
        json.dumps({"command": command, "cwd": str(PROJECT_ROOT)}, indent=2) + "\n",
        encoding="utf-8",
    )
    if verbose:
        print(f"Saved command metadata: {output_dir / 'command.json'}")

    with (output_dir / "stdout.log").open("w", encoding="utf-8") as stdout_file:
        with (output_dir / "stderr.log").open("w", encoding="utf-8") as stderr_file:
            return subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                check=False,
            )


def run_pangenie_index(
    manifest: dict[str, Any],
    run_dir: str | Path,
    threads: int = 1,
    verbose: bool = False,
) -> Path:
    missing_keys = [key for key in ("reference_fasta", "panel_subset") if key not in manifest]
    if missing_keys:
        raise ValueError(f"Manifest is missing required PanGenie index keys: {missing_keys}")

    reused_prefix = manifest_reused_pangenie_index_prefix(manifest)
    if reused_prefix is not None:
        output_dir = pangenie_index_dir(run_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pangenie_index_complete_path(run_dir).write_text(
            json.dumps(
                {
                    "index_prefix": str(reused_prefix),
                    "reused_from_manifest": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if verbose:
            print(f"PanGenie reusing index prefix from manifest: {reused_prefix}")
            print(f"PanGenie index marker: {pangenie_index_complete_path(run_dir)}")
        return reused_prefix

    if not PANGENIE_INDEX_BIN.exists():
        raise FileNotFoundError(f"PanGenie-index binary not found: {PANGENIE_INDEX_BIN}")

    output_dir = pangenie_index_dir(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = pangenie_index_prefix(run_dir)
    command = build_pangenie_index_command(manifest, output_prefix, threads=threads)
    if verbose:
        print(f"PanGenie index output directory: {output_dir}")
        print(f"PanGenie index command: {' '.join(command)}")

    result = run_command(command, output_dir, verbose=verbose)

    if result.returncode != 0:
        raise RuntimeError(
            "PanGenie-index failed with exit code "
            f"{result.returncode}; see {output_dir / 'stderr.log'}"
        )

    pangenie_index_complete_path(run_dir).write_text(
        json.dumps({"index_prefix": str(output_prefix)}, indent=2) + "\n",
        encoding="utf-8",
    )
    if verbose:
        print(f"PanGenie index prefix: {output_prefix}")
        print(f"PanGenie index marker: {pangenie_index_complete_path(run_dir)}")
        print(f"PanGenie logs: {output_dir / 'stdout.log'}, {output_dir / 'stderr.log'}")

    return output_prefix


def run_pangenie_genotype(
    manifest: dict[str, Any],
    run_dir: str | Path,
    threads: int = 1,
    verbose: bool = False,
) -> Path:
    missing_keys = [key for key in ("fastq",) if key not in manifest]
    if missing_keys:
        raise ValueError(f"Manifest is missing required PanGenie genotype keys: {missing_keys}")

    if not PANGENIE_BIN.exists():
        raise FileNotFoundError(f"PanGenie binary not found: {PANGENIE_BIN}")

    marker_path = pangenie_index_complete_path(run_dir)
    if not marker_path.exists():
        raise FileNotFoundError(
            "PanGenie genotype requires a completed index in this run. "
            f"Missing {marker_path}. Run the pangenie index stage first."
        )
    with marker_path.open("r", encoding="utf-8") as marker_file:
        marker = json.load(marker_file)
    if not isinstance(marker, dict):
        raise ValueError(f"Expected JSON object in {marker_path}")

    output_dir = pangenie_genotype_dir(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = pangenie_genotype_prefix(run_dir)
    index_prefix = Path(str(marker.get("index_prefix", pangenie_index_prefix(run_dir))))
    command = build_pangenie_genotype_command(
        manifest,
        index_prefix,
        output_prefix,
        threads=threads,
    )
    if verbose:
        print(f"PanGenie genotype output directory: {output_dir}")
        print(f"PanGenie genotype command: {' '.join(command)}")

    result = run_command(command, output_dir, verbose=verbose)

    if result.returncode != 0:
        raise RuntimeError(
            "PanGenie failed with exit code "
            f"{result.returncode}; see {output_dir / 'stderr.log'}"
        )

    pangenie_genotype_complete_path(run_dir).write_text(
        json.dumps(
            {
                "genotype_prefix": str(output_prefix),
                "genotype_vcf": str(pangenie_genotype_vcf_path(run_dir)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if verbose:
        print(f"PanGenie genotype output prefix: {output_prefix}")
        print(f"PanGenie genotype marker: {pangenie_genotype_complete_path(run_dir)}")
        print(f"PanGenie logs: {output_dir / 'stdout.log'}, {output_dir / 'stderr.log'}")

    return output_prefix
