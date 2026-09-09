from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from pvc.config import (
    BCFTOOLS_BIN,
    PLINK_BIN,
    PLINK_BLOCKS_MAX_KB,
    PLINK_BLOCKS_MIN_MAF,
)


def pvc_biallelic_snps_path(index_dir: str | Path) -> Path:
    return Path(index_dir) / "biallelic_snps.vcf.gz"


def pvc_plink_blocks_prefix(index_dir: str | Path) -> Path:
    return Path(index_dir) / "plink_blocks"


def pvc_plink_blocks_file(index_dir: str | Path) -> Path:
    return pvc_plink_blocks_prefix(index_dir).with_suffix(".blocks.det")


def build_bcftools_view_command(manifest: dict[str, Any], output_vcf: str | Path) -> list[str]:
    return [
        str(BCFTOOLS_BIN),
        "view",
        "-m2",
        "-M2",
        "-v",
        "snps",
        str(manifest["panel_subset"]),
        "-Oz",
        "-o",
        str(output_vcf),
    ]


def build_bcftools_index_command(vcf_path: str | Path) -> list[str]:
    return [str(BCFTOOLS_BIN), "index", "-t", str(vcf_path)]


def build_plink_blocks_command(
    biallelic_vcf: str | Path,
    output_prefix: str | Path,
) -> list[str]:
    return [
        str(PLINK_BIN),
        "--vcf",
        str(biallelic_vcf),
        "--blocks",
        "no-pheno-req",
        "--blocks-max-kb",
        str(PLINK_BLOCKS_MAX_KB),
        "--blocks-min-maf",
        str(PLINK_BLOCKS_MIN_MAF),
        "--allow-extra-chr",
        "--vcf-half-call",
        "missing",
        "--out",
        str(output_prefix),
    ]


def run_pvc_aux_command(
    command: list[str],
    output_dir: Path,
    name: str,
    verbose: bool = False,
) -> subprocess.CompletedProcess[str]:
    (output_dir / f"{name}_command.json").write_text(
        json.dumps({"command": command}, indent=2) + "\n",
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
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    check=False,
                )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Required command not found while running {name}: {command[0]}"
        ) from exc


def run_plink_blocks(
    manifest: dict[str, Any],
    index_dir: str | Path,
    verbose: bool = False,
) -> Path:
    output_dir = Path(index_dir)
    biallelic_vcf = pvc_biallelic_snps_path(output_dir)
    blocks_prefix = pvc_plink_blocks_prefix(output_dir)
    blocks_file = pvc_plink_blocks_file(output_dir)

    view_result = run_pvc_aux_command(
        build_bcftools_view_command(manifest, biallelic_vcf),
        output_dir,
        "bcftools_view_biallelic_snps",
        verbose=verbose,
    )
    if view_result.returncode != 0:
        raise RuntimeError(
            "bcftools view failed while preparing PVC PLINK blocks; "
            f"see {output_dir / 'bcftools_view_biallelic_snps_stderr.log'}"
        )

    index_result = run_pvc_aux_command(
        build_bcftools_index_command(biallelic_vcf),
        output_dir,
        "bcftools_index_biallelic_snps",
        verbose=verbose,
    )
    if index_result.returncode != 0:
        raise RuntimeError(
            "bcftools index failed while preparing PVC PLINK blocks; "
            f"see {output_dir / 'bcftools_index_biallelic_snps_stderr.log'}"
        )

    plink_result = run_pvc_aux_command(
        build_plink_blocks_command(biallelic_vcf, blocks_prefix),
        output_dir,
        "plink_blocks",
        verbose=verbose,
    )
    if plink_result.returncode != 0:
        raise RuntimeError(
            f"PLINK --blocks failed with exit code {plink_result.returncode}; "
            f"see {output_dir / 'plink_blocks_stderr.log'}"
        )

    if not blocks_file.exists():
        raise FileNotFoundError(
            f"PLINK --blocks completed but did not create expected file: {blocks_file}"
        )

    if verbose:
        print(f"PVC PLINK blocks file: {blocks_file}")

    return blocks_file
