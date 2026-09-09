"""Fresh PVC sample read-map launcher against a reusable public PanGenie index.

This module is intentionally separate from :mod:`pangenie_process`.  The
historical launcher rebuilt the public graph/index and continued into the
PanGenie HMM, so its direct process wall was not a valid PVC client boundary.
Fresh paper runs use the frozen ``PanGenie-readmap`` helper and stop after the
sample-specific read ``UniqueKmersMap`` JSON has been serialized.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pangenie.pangenie import PROJECT_ROOT as PANGENIE_PROJECT_ROOT, run_command
from pvc.genotype.private.workflow_profile import INDEX_PROFILE_SCHEMA
from pvc.index.pangenie_process import (
    INDEX_ARTIFACT_POLICY,
    _canonical_sha256,
    _infer_run_dir,
    _sha256,
    _sha256_artifact,
    _stat_identity,
)
from pvc.config import PVC_READMAP_BIN


READMAP_LAUNCH_SCHEMA = "pvc-readmap-launch-v1"
READMAP_NATIVE_PROFILE_SCHEMA = "pvc-pangenie-readmap-profile-v2"
READMAP_COUNT_MODES = ("all", "graph", "targets", "jf")
READMAP_TARGET_CERTIFICATE_SCHEMA = "pvc-public-target-fasta-certificate-v1"
PUBLIC_BLOCKS_MANIFEST_KEY = "pvc_public_blocks_file"
READMAP_PHASES = (
    "public_index_load",
    "kmer_counting",
    "read_map_probability_setup",
    "read_map_fill",
    "read_map_serialization",
)
READMAP_PHASE_SCOPES = {
    "public_index_load": "load the reusable public UniqueKmersMap",
    "kmer_counting": (
        "count sample read k-mers in the explicitly selected population and "
        "compute its abundance histogram"
    ),
    "read_map_probability_setup": (
        "construct the sample read-map probability table from the measured "
        "k-mer abundance peak"
    ),
    "read_map_fill": (
        "map observed sample read-k-mer counts onto public panel-unique k-mers"
    ),
    "read_map_serialization": (
        "serialize the sample-specific read UniqueKmersMap JSON consumed by PVC"
    ),
}
READMAP_INPUT_ARTIFACT_POLICY = {
    "large_inputs": "resolved path plus exact POSIX stat before/after execution",
    "public_blocks": "full SHA-256 content digest before/after execution",
    "target_certificate": "full SHA-256 content digest",
    "content_digest_for_large_inputs": "not_computed",
}
READMAP_OUTPUT_ARTIFACT_POLICY = (
    "resolved path plus exact POSIX stat immediately after execution"
)


def manifest_chromosome(manifest: dict[str, Any]) -> str:
    chromosome = manifest.get("chromosome")
    if isinstance(chromosome, str) and chromosome:
        return chromosome
    interval = manifest.get("interval", "")
    if isinstance(interval, str) and ":" in interval:
        return interval.split(":", 1)[0]
    if isinstance(interval, str) and interval:
        return interval
    return "chr20"


def resolve_read_count_mode(manifest: dict[str, Any]) -> str:
    """Return the explicitly selected population; never infer a default."""
    mode = manifest.get("pvc_read_count_mode")
    if not isinstance(mode, str) or not mode:
        raise ValueError(
            "Fresh PVC read-map generation requires explicit manifest key "
            f"pvc_read_count_mode, one of {list(READMAP_COUNT_MODES)}"
        )
    if mode not in READMAP_COUNT_MODES:
        raise ValueError(
            f"Invalid pvc_read_count_mode {mode!r}; expected one of "
            f"{list(READMAP_COUNT_MODES)}"
        )
    return mode


def read_count_input(manifest: dict[str, Any]) -> Path:
    """Return the reads or reusable Jellyfish database used for counting.

    ``fastq`` remains the provenance key for the sample's whole-genome read
    set.  Campaigns that count those reads once may provide
    ``pvc_read_count_input`` so every chromosome reuses the same ``.jf``
    database instead of recounting (or, worse, chromosome-filtering) reads.
    """
    value = manifest.get("pvc_read_count_input", manifest.get("fastq"))
    if not isinstance(value, str) or not value:
        raise ValueError(
            "Manifest requires fastq or pvc_read_count_input for PVC read-map generation"
        )
    return Path(value).expanduser().resolve()


def public_index_prefix(manifest: dict[str, Any]) -> Path:
    value = manifest.get("pangenie_index_prefix")
    if not isinstance(value, str) or not value:
        raise ValueError(
            "Fresh PVC read-map generation requires reusable public index key "
            "pangenie_index_prefix"
        )
    return Path(value).expanduser().resolve()


def public_blocks_file(manifest: dict[str, Any]) -> Path:
    """Resolve the frozen, public, chromosome-level LD-block artifact."""
    value = manifest.get(PUBLIC_BLOCKS_MANIFEST_KEY)
    if not isinstance(value, str) or not value:
        raise ValueError(
            "Fresh PVC read-map generation requires frozen public LD-block key "
            f"{PUBLIC_BLOCKS_MANIFEST_KEY}"
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frozen public LD-block artifact is missing: {path}")
    return path


def public_blocks_artifact(manifest: dict[str, Any]) -> dict[str, Any]:
    return _sha256_artifact(public_blocks_file(manifest))


def public_readmap_input_paths(manifest: dict[str, Any]) -> list[Path]:
    prefix = public_index_prefix(manifest)
    chromosome = manifest_chromosome(manifest)
    paths = [
        Path(f"{prefix}_UniqueKmersMap.cereal"),
        Path(f"{prefix}_{chromosome}_kmers.tsv.gz"),
    ]
    if resolve_read_count_mode(manifest) == "graph":
        paths.append(Path(f"{prefix}_path_segments.fasta"))
    return paths


def public_genotype_artifact_paths(manifest: dict[str, Any]) -> list[Path]:
    """Public artifacts bound into the completed index marker.

    PVC's Python genotyper consumes the graph JSON directly.  The public
    UniqueKmersMap and per-chromosome k-mer table are included as provenance
    for the sample read-map construction that produced its other input.
    """
    prefix = public_index_prefix(manifest)
    chromosome = manifest_chromosome(manifest)
    return [
        Path(f"{prefix}_UniqueKmersMap.cereal"),
        Path(f"{prefix}_{chromosome}_kmers.tsv.gz"),
        Path(f"{prefix}_{chromosome}_Graph.json"),
    ]


def expected_pvc_readmap_outputs(output_prefix: str | Path) -> list[Path]:
    prefix = Path(output_prefix).expanduser().resolve()
    return [
        Path(f"{prefix}_read_UniqueKmersMap.json"),
        Path(f"{prefix}_readmap_profile.tsv"),
    ]


def expected_pvc_readmap_index_artifacts(
    manifest: dict[str, Any], output_prefix: str | Path
) -> list[Path]:
    """Artifacts certified by the marker (native profile is evidence, not input)."""
    return [
        expected_pvc_readmap_outputs(output_prefix)[0],
        *public_genotype_artifact_paths(manifest),
    ]


def missing_pvc_readmap_outputs(output_prefix: str | Path) -> list[Path]:
    return [path for path in expected_pvc_readmap_outputs(output_prefix) if not path.is_file()]


def _target_paths(manifest: dict[str, Any]) -> tuple[Path, Path]:
    target = manifest.get("pvc_read_target_fasta")
    certificate = manifest.get("pvc_read_target_certificate")
    if not isinstance(target, str) or not target:
        raise ValueError("pvc_read_count_mode='targets' requires pvc_read_target_fasta")
    if not isinstance(certificate, str) or not certificate:
        raise ValueError(
            "pvc_read_count_mode='targets' requires pvc_read_target_certificate"
        )
    return Path(target).expanduser().resolve(), Path(certificate).expanduser().resolve()


def _count_fasta_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    return count


def _target_sequence_identity(path: Path) -> tuple[int, str]:
    """Return record count and a digest of normalized FASTA sequence lines."""
    digest = hashlib.sha256()
    count = 0
    sequence_parts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sequence_parts:
                    digest.update("".join(sequence_parts).encode("ascii"))
                    digest.update(b"\n")
                    count += 1
                    sequence_parts = []
                continue
            sequence_parts.append(line)
    if sequence_parts:
        digest.update("".join(sequence_parts).encode("ascii"))
        digest.update(b"\n")
        count += 1
    return count, digest.hexdigest()


def validate_target_certificate(
    certificate_path: str | Path,
    *,
    expected_target: str | Path,
    expected_public_index_prefix: str | Path,
) -> dict[str, Any]:
    path = Path(certificate_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid public target certificate {path}: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != READMAP_TARGET_CERTIFICATE_SCHEMA
        or payload.get("complete") is not True
    ):
        raise ValueError(f"Incomplete or incompatible public target certificate: {path}")
    target = Path(str(payload.get("target_fasta", {}).get("path", ""))).resolve()
    expected = Path(expected_target).expanduser().resolve()
    if target != expected or not target.is_file():
        raise ValueError("Public target certificate is bound to a different target FASTA")
    target_record = payload.get("target_fasta")
    if not isinstance(target_record, dict):
        raise ValueError("Public target certificate lacks target_fasta identity")
    stat = target.stat()
    current_target = {
        "path": str(target),
        "sha256": _sha256(target),
        "size_bytes": int(stat.st_size),
        "record_count": _count_fasta_records(target),
    }
    if target_record != current_target or current_target["record_count"] <= 0:
        raise ValueError("Certified public target FASTA changed or is empty")
    prefix = Path(str(payload.get("public_index_prefix", ""))).resolve()
    if prefix != Path(expected_public_index_prefix).expanduser().resolve():
        raise ValueError("Public target certificate is bound to a different index prefix")
    sources = payload.get("source_kmer_tables")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Public target certificate lacks source k-mer-table identities")
    for source in sources:
        if not isinstance(source, dict) or source != _sha256_artifact(source.get("path", "")):
            raise ValueError("A target-certificate source k-mer table changed")
    extraction = payload.get("extraction")
    record_count, sequence_sha256 = _target_sequence_identity(target)
    required_extraction = {
        "columns": [4, 5],
        "comma_separated_within_columns": True,
        "kmer_length": 31,
        "alphabet": "ACGT",
        "deduplicated": True,
        "complete_columns_4_and_5": True,
        "completeness_verified": True,
    }
    if (
        not isinstance(extraction, dict)
        or any(
            extraction.get(key) != value
            for key, value in required_extraction.items()
        )
        or extraction.get("distinct_target_kmers") != record_count
        or extraction.get("source_sorted_unique_sha256") != sequence_sha256
        or extraction.get("target_sequence_sha256") != sequence_sha256
        or isinstance(extraction.get("source_valid_kmer_occurrences"), bool)
        or not isinstance(extraction.get("source_valid_kmer_occurrences"), int)
        or extraction.get("source_valid_kmer_occurrences", -1) < record_count
    ):
        raise ValueError("Target certificate does not prove complete columns 4 and 5 extraction")
    return payload


def _positive_int(manifest: dict[str, Any], key: str, default: int) -> int:
    value = manifest.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return result


def _nonnegative_int(manifest: dict[str, Any], key: str, default: int) -> int:
    value = manifest.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a nonnegative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a nonnegative integer") from exc
    if result < 0:
        raise ValueError(f"{key} must be a nonnegative integer")
    return result


def _positive_number(manifest: dict[str, Any], key: str, default: float) -> float:
    value = manifest.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{key} must be a positive finite number")
    return result


def build_pvc_readmap_command(
    manifest: dict[str, Any],
    output_prefix: str | Path,
    *,
    threads: int = 1,
) -> list[str]:
    mode = resolve_read_count_mode(manifest)
    if int(threads) != 1:
        raise ValueError("Fresh paper PVC read-map profiling requires threads=1")
    command = [
        str(PVC_READMAP_BIN),
        "-f",
        str(public_index_prefix(manifest)),
        "-i",
        str(read_count_input(manifest)),
        "-o",
        str(Path(output_prefix).expanduser().resolve()),
        "-m",
        mode,
        "-t",
        str(threads),
        "-j",
        str(threads),
        "-e",
        str(_positive_int(manifest, "pvc_readmap_hash_size", 3_000_000_000)),
        "-p",
        str(_nonnegative_int(manifest, "pvc_readmap_panel_size", 0)),
        "-n",
        format(_positive_number(manifest, "pvc_readmap_sampling_effective_n", 0.01), ".17g"),
        "-c",
        format(_positive_number(manifest, "pvc_readmap_recombination_rate", 1.26), ".17g"),
        "-u",
        format(_positive_number(manifest, "pvc_readmap_regularization", 0.001), ".17g"),
    ]
    if mode == "targets":
        target, certificate = _target_paths(manifest)
        validate_target_certificate(
            certificate,
            expected_target=target,
            expected_public_index_prefix=public_index_prefix(manifest),
        )
        command.extend(["-x", str(target)])
    # -B/-d/-l are deliberately absent.  Optional validation/diagnostic
    # outputs are outside the paper's direct T14 process boundary.
    return command


def _build_input_artifacts(manifest: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "reads": _stat_identity(read_count_input(manifest)),
        "public_index": [
            _stat_identity(path) for path in public_readmap_input_paths(manifest)
        ],
        "public_blocks": _sha256_artifact(public_blocks_file(manifest)),
    }
    if resolve_read_count_mode(manifest) == "targets":
        target, certificate = _target_paths(manifest)
        result["target_fasta"] = _stat_identity(target)
        result["target_certificate"] = _sha256_artifact(certificate)
    return result


def _command_option(command: list[str], option: str) -> str:
    positions = [index for index, value in enumerate(command) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"Expected exactly one {option} option in read-map command")
    return str(command[positions[0] + 1])


def _parse_native_profile(path: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    metadata: dict[str, str] = {}
    data_lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                key, separator, value = line[1:].rstrip("\n").partition("\t")
                if not separator or not key or key in metadata:
                    raise ValueError(f"Malformed native read-map profile metadata: {path}")
                metadata[key] = value
            else:
                data_lines.append(line)
    if metadata.get("schema") != READMAP_NATIVE_PROFILE_SCHEMA:
        raise ValueError(f"Incompatible native read-map profile: {path}")
    reader = csv.DictReader(data_lines, delimiter="\t")
    if reader.fieldnames != [
        "phase",
        "wall_seconds",
        "cpu_seconds",
        "peak_rss_kib",
    ]:
        raise ValueError(f"Malformed native read-map phase header: {path}")
    native_rows = list(reader)
    if [row.get("phase") for row in native_rows] != list(READMAP_PHASES):
        raise ValueError(
            "Final read-map profile must contain exactly the five required phases "
            f"in order; got {[row.get('phase') for row in native_rows]}"
        )
    rows: list[dict[str, Any]] = []
    for order, native in enumerate(native_rows):
        phase = str(native["phase"])
        numbers: dict[str, float] = {}
        for field in ("wall_seconds", "cpu_seconds"):
            try:
                value = float(str(native[field]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid {field} for {phase} in {path}") from exc
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {field} for {phase} in {path}")
            numbers[field] = value
        peak_rss_text = str(native.get("peak_rss_kib", ""))
        if not peak_rss_text.isdigit():
            raise ValueError(f"Invalid peak_rss_kib for {phase} in {path}")
        peak_rss_kib = int(peak_rss_text)
        rows.append(
            {
                "phase": phase,
                "order": order,
                "actor": "client",
                "scope": READMAP_PHASE_SCOPES[phase],
                "included_fastq_to_vcf": True,
                "required_for_vcf": True,
                "measurement_status": "measured",
                "wall_seconds": numbers["wall_seconds"],
                "cpu_seconds": numbers["cpu_seconds"],
                "calls": 1,
                "process_peak_rss_kib": peak_rss_kib,
                "process_peak_rss_gib": peak_rss_kib / float(1024**2),
                "process_peak_rss_definition": (
                    "Linux getrusage(RUSAGE_SELF).ru_maxrss integer KiB: "
                    "cumulative process-lifetime high-water mark observed at "
                    "phase end, not a phase-local memory peak"
                ),
                "communicator_endpoint_payload_bytes": 0,
                "communicator_endpoint_payload_bytes_max": 0,
                "communicator_endpoint_payload_bytes_sum": 0,
                "communicator_payload_bytes": 0,
                "communicator_operations": 0,
                "logical_exchange_calls": 0,
                "client_upload_bytes": 0,
                "client_upload_rounds": 0,
                "client_download_bytes": 0,
                "client_download_rounds": 0,
                "client_transfer_model": [],
                "reported_threads": int(
                    metadata["count_threads"]
                    if phase == "kmer_counting"
                    else metadata["fill_threads"]
                    if phase == "read_map_fill"
                    else 1
                ),
                "cpu_seconds_source": "getrusage(RUSAGE_SELF) process CPU delta",
                "cpu_seconds_is_proxy": False,
                "cpu_seconds_model": (
                    "true process user+system CPU delta, including worker threads"
                ),
            }
        )
    return metadata, rows


def _validate_native_metadata(
    metadata: dict[str, str],
    manifest: dict[str, Any],
    output_prefix: Path,
    *,
    command: list[str],
) -> None:
    expected_command = build_pvc_readmap_command(manifest, output_prefix, threads=1)
    if command != expected_command:
        raise ValueError(
            "Native read-map evidence is not bound to the exact manifest command"
        )
    expected = {
        "count_mode": resolve_read_count_mode(manifest),
        "public_index_prefix": str(public_index_prefix(manifest)),
        "reads": str(read_count_input(manifest)),
        "output_prefix": str(output_prefix.resolve()),
        "target_fasta": (
            str(_target_paths(manifest)[0])
            if resolve_read_count_mode(manifest) == "targets"
            else ""
        ),
        "count_threads": "1",
        "fill_threads": "1",
        "hash_size": _command_option(command, "-e"),
        "requested_panel_size": _command_option(command, "-p"),
        "write_binary_cereal": "0",
        "write_read_tsv": "0",
        "write_histogram": "0",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Native read-map profile {key} mismatch: "
                f"{metadata.get(key)!r} != {value!r}"
            )
    for key in (
        "kmer_size",
        "histogram_peak",
        "panel_size",
        "recombination_rate",
        "sampling_effective_N",
        "regularization",
        "allele_penalty",
    ):
        if key not in metadata:
            raise ValueError(f"Native read-map profile lacks metadata field {key}")
    for key, option in {
        "recombination_rate": "-c",
        "sampling_effective_N": "-n",
        "regularization": "-u",
    }.items():
        try:
            observed = float(metadata[key])
            expected_number = float(_command_option(command, option))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Native read-map profile has invalid {key}") from exc
        if not math.isclose(observed, expected_number, rel_tol=1e-15, abs_tol=0.0):
            raise ValueError(
                f"Native read-map profile {key} mismatch: "
                f"{observed!r} != {expected_number!r}"
            )
    if metadata.get("allele_penalty") != "5":
        raise ValueError("Native read-map profile allele_penalty mismatch")
    if not Path(f"{output_prefix}_read_UniqueKmersMap.json").is_file():
        raise ValueError("Native read-map profile lacks its required JSON output")


def _validate_launch(
    directory: Path,
    *,
    expected_input_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = directory / "process_launch_provenance.json"
    try:
        launch = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid read-map launch provenance {path}: {exc}") from exc
    if (
        not isinstance(launch, dict)
        or launch.get("schema") != READMAP_LAUNCH_SCHEMA
        or launch.get("schema_version") != 1
        or launch.get("returncode") != 0
    ):
        raise ValueError(f"Incomplete or incompatible read-map launch: {path}")
    command_artifact = launch.get("command_artifact")
    if not isinstance(command_artifact, dict) or command_artifact != _sha256_artifact(
        directory / "command.json"
    ):
        raise ValueError("Read-map command artifact changed")
    command_payload = json.loads((directory / "command.json").read_text(encoding="utf-8"))
    command = command_payload.get("command")
    if (
        not isinstance(command, list)
        or not all(isinstance(value, str) for value in command)
        or command != launch.get("command")
        or _canonical_sha256(command) != launch.get("command_identity_sha256")
    ):
        raise ValueError("Read-map launch is not bound to the exact command")
    cwd = Path(str(command_payload.get("cwd", ""))).resolve()
    if cwd != Path(str(launch.get("command_cwd", ""))).resolve():
        raise ValueError("Read-map launch working-directory binding changed")
    executable = Path(command[0]).expanduser().resolve()
    current_executable_sha = _sha256(executable)
    if not (
        Path(str(launch.get("executable", ""))).resolve() == executable
        and launch.get("executable_unchanged_during_run") is True
        and launch.get("executable_sha256_before") == current_executable_sha
        and launch.get("executable_sha256_after") == current_executable_sha
    ):
        raise ValueError("Read-map executable provenance changed")
    wall = launch.get("process_wall_seconds")
    if (
        isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not math.isfinite(float(wall))
        or float(wall) <= 0
    ):
        raise ValueError("Read-map launch lacks a positive direct process wall")
    binding = launch.get("input_binding")
    if not isinstance(binding, dict) or not binding:
        raise ValueError("Read-map launch lacks the workflow input binding")
    if expected_input_binding is not None and binding != expected_input_binding:
        raise ValueError("Read-map input binding differs from the requested workflow")
    manifest = binding.get("canonical_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Read-map launch binding lacks its canonical manifest")
    output_prefix = Path(_command_option(command, "-o")).resolve()
    expected_command = build_pvc_readmap_command(manifest, output_prefix, threads=1)
    if command != expected_command:
        raise ValueError(
            "Read-map launch command differs from the exact canonical manifest command"
        )
    if launch.get("input_artifact_policy") != READMAP_INPUT_ARTIFACT_POLICY:
        raise ValueError("Read-map input artifact policy changed")
    current_inputs = _build_input_artifacts(manifest)
    if not (
        launch.get("input_artifacts_before") == current_inputs
        and launch.get("input_artifacts_after") == current_inputs
        and launch.get("inputs_unchanged_during_run") is True
    ):
        raise ValueError("Read-map inputs changed during or after execution")
    logs = launch.get("timing_logs")
    expected_logs = {
        "stdout": _sha256_artifact(directory / "stdout.log"),
        "stderr": _sha256_artifact(directory / "stderr.log"),
        "native_profile": _sha256_artifact(Path(f"{output_prefix}_readmap_profile.tsv")),
    }
    if logs != expected_logs:
        raise ValueError("Read-map timing evidence changed")
    outputs = launch.get("process_output_artifacts")
    expected_outputs = [
        _stat_identity(path)
        for path in expected_pvc_readmap_index_artifacts(manifest, output_prefix)
    ]
    if (
        launch.get("process_output_artifact_policy") != READMAP_OUTPUT_ARTIFACT_POLICY
        or outputs != expected_outputs
    ):
        raise ValueError("Read-map output/public-artifact identity changed")
    return launch


def run_pangenie_readmap(
    tool: str,
    manifest: dict[str, Any],
    output_dir: str | Path,
    output_prefix: str | Path,
    *,
    threads: int = 1,
    verbose: bool = False,
    run_dir: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = build_pvc_readmap_command(manifest, output_prefix, threads=threads)
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if not PVC_READMAP_BIN.is_file():
        raise FileNotFoundError(f"Frozen PanGenie-readmap binary not found: {PVC_READMAP_BIN}")
    for input_path in public_readmap_input_paths(manifest):
        if not input_path.is_file():
            raise FileNotFoundError(f"Reusable public index artifact is missing: {input_path}")
    for input_path in public_genotype_artifact_paths(manifest):
        if not input_path.is_file():
            raise FileNotFoundError(f"Reusable public genotype artifact is missing: {input_path}")
    effective_run_dir = (
        Path(run_dir).expanduser().resolve()
        if run_dir is not None
        else _infer_run_dir(directory)
    )
    from pvc.genotype.private.workflow_profile import build_workflow_input_binding

    input_binding = build_workflow_input_binding(manifest, effective_run_dir)
    input_before = _build_input_artifacts(manifest)
    executable = PVC_READMAP_BIN.resolve()
    executable_sha_before = _sha256(executable)
    started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    process_wall_started = time.perf_counter()
    result = run_command(command, directory, verbose=verbose)
    process_wall_seconds = time.perf_counter() - process_wall_started
    finished_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    executable_sha_after = _sha256(executable)
    input_after = _build_input_artifacts(manifest)
    command_path = directory / "command.json"
    command_payload = json.loads(command_path.read_text(encoding="utf-8"))
    if (
        command_payload.get("command") != command
        or Path(str(command_payload.get("cwd", ""))).resolve()
        != Path(PANGENIE_PROJECT_ROOT).resolve()
    ):
        raise RuntimeError("PanGenie-readmap command metadata does not match its launch")
    output_prefix_path = Path(output_prefix).expanduser().resolve()
    outputs = expected_pvc_readmap_index_artifacts(manifest, output_prefix_path)
    process_output_artifacts = [
        _stat_identity(path) if path.is_file() else {"path": str(path), "missing": True}
        for path in outputs
    ]
    timing_logs: dict[str, Any] = {
        "stdout": _sha256_artifact(directory / "stdout.log"),
        "stderr": _sha256_artifact(directory / "stderr.log"),
    }
    native_profile = Path(f"{output_prefix_path}_readmap_profile.tsv")
    timing_logs["native_profile"] = (
        _sha256_artifact(native_profile)
        if native_profile.is_file()
        else {"path": str(native_profile), "missing": True}
    )
    launch = {
        "schema": READMAP_LAUNCH_SCHEMA,
        "schema_version": 1,
        "tool": tool,
        "requested_threads": int(threads),
        "read_count_mode": resolve_read_count_mode(manifest),
        "command": command,
        "command_cwd": str(Path(PANGENIE_PROJECT_ROOT).resolve()),
        "command_identity_sha256": _canonical_sha256(command),
        "command_artifact": _sha256_artifact(command_path),
        "executable": str(executable),
        "executable_sha256_before": executable_sha_before,
        "executable_sha256_after": executable_sha_after,
        "executable_unchanged_during_run": (
            executable_sha_before == executable_sha_after
        ),
        "timing_logs": timing_logs,
        "input_binding": input_binding,
        "input_artifact_policy": READMAP_INPUT_ARTIFACT_POLICY,
        "input_artifacts_before": input_before,
        "input_artifacts_after": input_after,
        "inputs_unchanged_during_run": input_before == input_after,
        "process_output_artifact_policy": READMAP_OUTPUT_ARTIFACT_POLICY,
        "process_output_artifacts": process_output_artifacts,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "process_wall_seconds": process_wall_seconds,
        "direct_wall_boundary": {
            "start": "immediately before subprocess launch",
            "end": "subprocess return after native profile/log flush",
            "includes": [
                "process startup and shutdown",
                *READMAP_PHASES,
                "native read-map profile write",
            ],
            "excludes": [
                "public graph/index construction",
                "PanGenie HMM or VCF output",
                "PLINK LD-block preparation",
                "optional binary cereal, read TSV, and histogram outputs",
            ],
        },
        "returncode": int(result.returncode),
    }
    launch_path = directory / "process_launch_provenance.json"
    temporary = launch_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(launch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(launch_path)
    if executable_sha_before != executable_sha_after:
        raise RuntimeError(f"PanGenie-readmap executable changed during run: {executable}")
    if input_before != input_after:
        raise RuntimeError("PanGenie-readmap inputs changed during execution")
    return result


def _build_index_identity(
    marker_path: str | Path,
    *,
    expected_profile_path: Path,
    expected_launch: dict[str, Any],
) -> dict[str, Any]:
    marker = Path(marker_path).expanduser().resolve()
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("exit_code") != 0:
        raise ValueError("Read-map completion marker is invalid")
    if Path(str(payload.get("client_read_profile", ""))).resolve() != expected_profile_path:
        raise ValueError("Read-map marker/profile binding changed")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Read-map completion marker lacks artifacts")
    artifact_identities = [_stat_identity(value) for value in artifacts]
    if artifact_identities != expected_launch.get("process_output_artifacts"):
        raise ValueError("Read-map marker artifacts differ from launch-bound artifacts")
    manifest = expected_launch.get("input_binding", {}).get("canonical_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Read-map launch lacks a canonical manifest")
    expected_blocks_path = public_blocks_file(manifest)
    expected_blocks = _sha256_artifact(expected_blocks_path)
    if (
        Path(str(payload.get("blocks_file", ""))).resolve()
        != expected_blocks_path
        or payload.get("public_blocks_artifact") != expected_blocks
        or payload.get("blocks_source") != "frozen_public_manifest_artifact"
    ):
        raise ValueError(
            "Read-map marker is not bound to the frozen public blocks artifact"
        )
    body = {
        "schema": "pvc-index-artifact-identity-v1",
        "policy": INDEX_ARTIFACT_POLICY,
        "tool": payload.get("tool"),
        "output_prefix": str(Path(str(payload.get("index_prefix", ""))).resolve()),
        "public_index_prefix": str(
            Path(str(payload.get("public_index_prefix", ""))).resolve()
        ),
        "read_count_mode": payload.get("read_count_mode"),
        "marker": _sha256_artifact(marker),
        "blocks_file": expected_blocks,
        "index_artifacts": artifact_identities,
    }
    body["index_identity_sha256"] = _canonical_sha256(body)
    return body


def write_pvc_readmap_client_profile(
    output_dir: str | Path,
    *,
    tool: str,
    manifest: dict[str, Any],
    run_dir: str | Path,
    index_marker_path: str | Path,
) -> tuple[Path, Path]:
    directory = Path(output_dir).expanduser().resolve()
    from pvc.genotype.private.workflow_profile import build_workflow_input_binding

    binding = build_workflow_input_binding(manifest, run_dir)
    launch = _validate_launch(directory, expected_input_binding=binding)
    command = launch["command"]
    output_prefix = Path(_command_option(command, "-o")).resolve()
    native_path = Path(f"{output_prefix}_readmap_profile.tsv")
    metadata, rows = _parse_native_profile(native_path)
    _validate_native_metadata(
        metadata, manifest, output_prefix, command=command
    )
    profile_path = directory / "client_read_profile.json"
    identity = _build_index_identity(
        index_marker_path,
        expected_profile_path=profile_path,
        expected_launch=launch,
    )
    launch_artifact = _sha256_artifact(directory / "process_launch_provenance.json")
    evidence = {
        "launch_artifact_sha256": launch_artifact["sha256"],
        "native_profile_sha256": _sha256(native_path),
        "input_identity_sha256": binding["input_identity_sha256"],
        "index_identity_sha256": identity["index_identity_sha256"],
        "direct_process_wall_seconds": launch["process_wall_seconds"],
    }
    payload = {
        "schema": INDEX_PROFILE_SCHEMA,
        "schema_version": 1,
        "tool": tool,
        "source": "PanGenie-readmap native machine-readable phase profile",
        "source_profile": str(native_path),
        "source_profile_sha256": _sha256(native_path),
        "timing_log_artifacts": launch["timing_logs"],
        "command_artifact": launch["command_artifact"],
        "executable_artifact": {
            "path": launch["executable"],
            "sha256": launch["executable_sha256_after"],
            "role": "executed_binary",
            "verification": "before/after/current SHA-256 equality",
        },
        "execution_provenance": {
            "status": "fresh_verified",
            "certified": True,
            "reason": None,
            "validation": "complete read-map execution chain re-derived from bound artifacts",
            "launch_artifact_sha256": launch_artifact["sha256"],
        },
        "input_binding": binding,
        "index_identity": identity,
        "read_count_mode": resolve_read_count_mode(manifest),
        "native_metadata": metadata,
        "direct_process_wall_seconds": float(launch["process_wall_seconds"]),
        "direct_process_wall_definition": (
            "external time.perf_counter elapsed immediately around the complete "
            "PanGenie-readmap subprocess, including process startup/shutdown and "
            "native profile write; excluding PLINK and optional diagnostics"
        ),
        "thread_policy": {
            "paper_profile_requires_one_thread": True,
            "count_threads": 1,
            "fill_threads": 1,
            "one_thread_verified": True,
        },
        "profile_evidence_sha256": _canonical_sha256(evidence),
        "process_launch_provenance": launch,
        "process_launch_provenance_artifact": launch_artifact,
        "requested_threads": 1,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runtime_provenance": {
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "slurm": {
                key: os.environ.get(key, "")
                for key in (
                    "SLURM_JOB_ID",
                    "SLURM_ARRAY_JOB_ID",
                    "SLURM_ARRAY_TASK_ID",
                    "SLURM_JOB_PARTITION",
                    "SLURM_NODELIST",
                    "SLURM_CPUS_PER_TASK",
                )
            },
        },
        "canonical_phases_non_overlapping": True,
        "boundary_note": (
            "Only sample-specific read-map work against a reusable public index "
            "is included; public construction, PanGenie HMM/VCF, PLINK, and "
            "optional diagnostics are excluded."
        ),
        "phases": rows,
    }
    temporary = profile_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(profile_path)
    csv_path = directory / "client_read_profile.csv"
    fields = list(rows[0])
    csv_temporary = csv_path.with_suffix(".csv.tmp")
    with csv_temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["client_transfer_model"] = ""
            writer.writerow(output)
    csv_temporary.replace(csv_path)
    validate_pvc_readmap_client_profile(
        profile_path,
        expected_input_binding=binding,
    )
    return profile_path, csv_path


def validate_pvc_readmap_client_profile(
    profile_path: str | Path,
    *,
    expected_input_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(profile_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid PVC read-map client profile {path}: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != INDEX_PROFILE_SCHEMA
        or payload.get("schema_version") != 1
        or payload.get("source")
        != "PanGenie-readmap native machine-readable phase profile"
    ):
        raise ValueError(f"Incompatible PVC read-map client profile: {path}")
    launch = _validate_launch(path.parent, expected_input_binding=expected_input_binding)
    if payload.get("process_launch_provenance") != launch:
        raise ValueError("Read-map client profile embeds stale launch provenance")
    launch_artifact = _sha256_artifact(path.parent / "process_launch_provenance.json")
    if payload.get("process_launch_provenance_artifact") != launch_artifact:
        raise ValueError("Read-map client profile launch binding changed")
    binding = launch["input_binding"]
    if payload.get("input_binding") != binding:
        raise ValueError("Read-map client profile input binding changed")
    manifest = binding.get("canonical_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Read-map client profile lacks canonical manifest")
    output_prefix = Path(_command_option(launch["command"], "-o")).resolve()
    native_path = Path(f"{output_prefix}_readmap_profile.tsv")
    metadata, rows = _parse_native_profile(native_path)
    _validate_native_metadata(
        metadata, manifest, output_prefix, command=launch["command"]
    )
    if payload.get("source_profile") != str(native_path):
        raise ValueError("Read-map native profile path changed")
    if payload.get("source_profile_sha256") != _sha256(native_path):
        raise ValueError("Read-map native profile content changed")
    if payload.get("native_metadata") != metadata or payload.get("phases") != rows:
        raise ValueError("Read-map native phase parsing changed")
    if payload.get("read_count_mode") != resolve_read_count_mode(manifest):
        raise ValueError("Read-map profile count-mode binding changed")
    marker_record = payload.get("index_identity", {}).get("marker")
    if not isinstance(marker_record, dict):
        raise ValueError("Read-map profile lacks marker identity")
    identity = _build_index_identity(
        marker_record.get("path", ""),
        expected_profile_path=path,
        expected_launch=launch,
    )
    if payload.get("index_identity") != identity:
        raise ValueError("Read-map index artifact identity changed")
    executable = payload.get("executable_artifact")
    if not isinstance(executable, dict) or executable.get("sha256") != _sha256(
        Path(str(executable.get("path", "")))
    ):
        raise ValueError("Read-map profile executable identity changed")
    expected_execution = {
        "status": "fresh_verified",
        "certified": True,
        "reason": None,
        "validation": "complete read-map execution chain re-derived from bound artifacts",
        "launch_artifact_sha256": launch_artifact["sha256"],
    }
    if payload.get("execution_provenance") != expected_execution:
        raise ValueError("Read-map execution certification changed")
    if payload.get("direct_process_wall_seconds") != float(launch["process_wall_seconds"]):
        raise ValueError("Read-map direct process wall binding changed")
    evidence = {
        "launch_artifact_sha256": launch_artifact["sha256"],
        "native_profile_sha256": _sha256(native_path),
        "input_identity_sha256": binding["input_identity_sha256"],
        "index_identity_sha256": identity["index_identity_sha256"],
        "direct_process_wall_seconds": launch["process_wall_seconds"],
    }
    if payload.get("profile_evidence_sha256") != _canonical_sha256(evidence):
        raise ValueError("Read-map profile evidence digest changed")
    return payload
