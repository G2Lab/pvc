from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pangenie.pangenie import PROJECT_ROOT as PANGENIE_PROJECT_ROOT, run_command
from pvc.genotype.private.workflow_profile import (
    LEGACY_INDEX_PROFILE_SCHEMA as INDEX_PROFILE_SCHEMA,
)
from pvc.config import PVC_INDEX_BINS


_READ_PHASE_PATTERNS = {
    "kmer_counting": re.compile(
        r"time spent counting kmers in reads \((?P<threads>\d+) thread\(s\)\):\s*"
        r"(?P<wall>[0-9.eE+-]+) sec"
    ),
    "read_map_fill": re.compile(
        r"time spent filling read kmer counts \((?P<threads>\d+) thread\(s\)"
        r"\s*/\s*single thread\):\s*(?P<wall>[0-9.eE+-]+)"
        r"/(?P<cpu>[0-9.eE+-]+) sec"
    ),
    "read_map_serialization": re.compile(
        r"time spent writing read UniqueKmersMap to disk \(single thread\):\s*"
        r"(?P<wall>[0-9.eE+-]+) sec"
    ),
}

PROCESS_LAUNCH_SCHEMA = "pvc-index-process-launch-v2"
INPUT_ARTIFACT_POLICY = {
    "identity": "resolved path plus POSIX stat before/after execution",
    "stat_fields": ["size_bytes", "mtime_ns", "device", "inode"],
    "content_digest": "not_computed",
    "reason": (
        "Reference, panel, and FASTQ inputs may be very large; their content is "
        "identified by stable resolved paths in input_binding and guarded "
        "against in-place drift by exact stat snapshots."
    ),
}
INDEX_ARTIFACT_POLICY = {
    "marker": "full SHA-256 content digest",
    "blocks_file": "full SHA-256 content digest",
    "large_index_artifacts": "resolved path plus exact POSIX stat",
    "large_index_content_digest": "not_computed",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_artifact(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"Required provenance artifact is missing: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": int(stat.st_size),
    }


def _stat_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"Required identity input is missing: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }


def _optional_stat_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "missing": True}
    return _stat_identity(resolved)


def _validate_sha256_artifact(
    artifact: Any,
    *,
    label: str,
    expected_path: str | Path | None = None,
) -> Path:
    if not isinstance(artifact, dict):
        raise ValueError(f"{label} provenance is missing or malformed")
    raw_path = artifact.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} provenance lacks a path")
    path = Path(raw_path).resolve()
    if expected_path is not None and path != Path(expected_path).resolve():
        raise ValueError(f"{label} path mismatch: {path} != {Path(expected_path).resolve()}")
    current = _sha256_artifact(path)
    if artifact != current:
        raise ValueError(f"{label} changed after it was bound: {path}")
    return path


def _command_option(command: list[str], option: str) -> str:
    positions = [index for index, value in enumerate(command) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"Expected exactly one {option} option in PVC index command")
    return str(command[positions[0] + 1])


def _resolve_command_path(command_cwd: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = command_cwd / path
    return path.resolve()


def _infer_run_dir(output_dir: Path) -> Path:
    directory = output_dir.resolve()
    if directory.name == "index" and directory.parent.name in PVC_INDEX_BINS:
        return directory.parent.parent
    return directory.parent


def _binding_input_path(binding: dict[str, Any], key: str) -> Path:
    input_paths = binding.get("input_paths")
    value = input_paths.get(key) if isinstance(input_paths, dict) else None
    if value is None:
        value = binding.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Workflow input binding lacks {key}")
    return Path(value).resolve()


def _build_input_artifacts(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: _stat_identity(manifest[key])
        for key in ("reference_fasta", "panel_subset", "fastq")
    }


def _validate_input_artifacts(
    launch: dict[str, Any], binding: dict[str, Any]
) -> None:
    if launch.get("input_artifact_policy") != INPUT_ARTIFACT_POLICY:
        raise ValueError("PVC index input artifact policy is missing or changed")
    before = launch.get("input_artifacts_before")
    after = launch.get("input_artifacts_after")
    if (
        not isinstance(before, dict)
        or not isinstance(after, dict)
        or before != after
        or launch.get("inputs_unchanged_during_run") is not True
    ):
        raise ValueError("PVC index inputs changed during execution")
    for key in ("reference_fasta", "panel_subset", "fastq"):
        expected_path = _binding_input_path(binding, key)
        recorded = after.get(key)
        if not isinstance(recorded, dict) or recorded.get("path") != str(expected_path):
            raise ValueError(f"PVC index {key} is not bound to the workflow input")
        if _stat_identity(expected_path) != recorded:
            raise ValueError(f"PVC index {key} changed after execution")


def _validate_process_output_artifacts(launch: dict[str, Any]) -> None:
    if launch.get("process_output_artifact_policy") != INDEX_ARTIFACT_POLICY[
        "large_index_artifacts"
    ]:
        raise ValueError("PVC index process output artifact policy changed")
    artifacts = launch.get("process_output_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("PVC index launch lacks process output artifact identities")
    for artifact in artifacts:
        if not isinstance(artifact, dict) or _stat_identity(artifact.get("path", "")) != artifact:
            raise ValueError("PVC index process output artifact changed after execution")


def _validate_fresh_launch(
    directory: Path,
    *,
    tool: str | None = None,
    expected_input_binding: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Re-derive fresh certification from bound artifacts, never a flag."""
    launch_path = directory / "process_launch_provenance.json"
    try:
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid process launch provenance {launch_path}: {exc}") from exc
    if (
        not isinstance(launch, dict)
        or launch.get("schema") != PROCESS_LAUNCH_SCHEMA
        or launch.get("schema_version") != 2
    ):
        raise ValueError(f"Incompatible process launch provenance {launch_path}")
    launch_tool = launch.get("tool")
    if not isinstance(launch_tool, str) or (tool is not None and launch_tool != tool):
        raise ValueError(f"PVC index launch tool mismatch in {launch_path}")

    command_path = _validate_sha256_artifact(
        launch.get("command_artifact"),
        label="PVC index command artifact",
        expected_path=directory / "command.json",
    )
    try:
        command_payload = json.loads(command_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid command artifact {command_path}: {exc}") from exc
    command = command_payload.get("command") if isinstance(command_payload, dict) else None
    command_cwd_value = (
        command_payload.get("cwd") if isinstance(command_payload, dict) else None
    )
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(value, str) for value in command)
        or not isinstance(command_cwd_value, str)
        or not command_cwd_value
        or command != launch.get("command")
        or _canonical_sha256(command) != launch.get("command_identity_sha256")
    ):
        raise ValueError("PVC index launch is not bound to the exact command")
    command_cwd = Path(command_cwd_value).expanduser().resolve()
    if (
        not command_cwd.is_dir()
        or launch.get("command_cwd") != str(command_cwd)
    ):
        raise ValueError("PVC index launch is not bound to its execution directory")

    executable = _resolve_command_path(command_cwd, command[0])
    executable_sha = _sha256(executable)
    if (
        Path(str(launch.get("executable", ""))).resolve() != executable
        or launch.get("executable_unchanged_during_run") is not True
        or launch.get("executable_sha256_before") != executable_sha
        or launch.get("executable_sha256_after") != executable_sha
    ):
        raise ValueError(f"PanGenie-process executable provenance failed: {executable}")

    returncode = launch.get("returncode")
    if isinstance(returncode, bool) or not isinstance(returncode, int) or returncode != 0:
        raise ValueError(f"PVC index launch return code is not successful: {returncode!r}")
    process_wall_seconds = launch.get("process_wall_seconds")
    if (
        isinstance(process_wall_seconds, bool)
        or not isinstance(process_wall_seconds, (int, float))
        or not math.isfinite(float(process_wall_seconds))
        or float(process_wall_seconds) <= 0
    ):
        raise ValueError(
            "PVC index launch lacks a positive direct process wall duration"
        )
    requested_threads = launch.get("requested_threads")
    if requested_threads != 1:
        raise ValueError(
            f"Fresh paper index profile requires one requested thread; got {requested_threads!r}"
        )
    for option in ("-j", "-t"):
        if _command_option(command, option) != "1":
            raise ValueError(f"Fresh paper index profile requires {option} 1")

    logs = launch.get("timing_logs")
    if not isinstance(logs, dict) or set(logs) != {"stdout", "stderr"}:
        raise ValueError("PVC index launch lacks exact stdout/stderr timing-log bindings")
    _validate_sha256_artifact(
        logs["stdout"], label="PVC index stdout log", expected_path=directory / "stdout.log"
    )
    _validate_sha256_artifact(
        logs["stderr"], label="PVC index stderr log", expected_path=directory / "stderr.log"
    )

    binding = launch.get("input_binding")
    if not isinstance(binding, dict) or not binding:
        raise ValueError("Fresh PVC index launch lacks a workflow input binding")
    if expected_input_binding is not None and binding != expected_input_binding:
        raise ValueError("PVC index workflow input binding does not match the requested run")
    if _resolve_command_path(
        command_cwd, _command_option(command, "-r")
    ) != _binding_input_path(binding, "reference_fasta"):
        raise ValueError("PVC index command/reference binding mismatch")
    if _resolve_command_path(
        command_cwd, _command_option(command, "-v")
    ) != _binding_input_path(binding, "panel_subset"):
        raise ValueError("PVC index command/panel binding mismatch")
    if _resolve_command_path(
        command_cwd, _command_option(command, "-i")
    ) != _binding_input_path(binding, "fastq"):
        raise ValueError("PVC index command/FASTQ binding mismatch")
    canonical_manifest = binding.get("canonical_manifest")
    expected_sample = (
        canonical_manifest.get("sample", "sample")
        if isinstance(canonical_manifest, dict)
        else None
    )
    if expected_sample is None or _command_option(command, "-s") != str(expected_sample):
        raise ValueError("PVC index command/sample binding mismatch")
    _validate_input_artifacts(launch, binding)
    _validate_process_output_artifacts(launch)
    return launch, launch_path


def _build_index_identity(
    marker_path: str | Path,
    *,
    expected_profile_path: str | Path,
    expected_returncode: int,
) -> dict[str, Any]:
    marker = Path(marker_path).resolve()
    marker_artifact = _sha256_artifact(marker)
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid PVC index marker {marker}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid PVC index marker {marker}")
    if payload.get("exit_code") != expected_returncode:
        raise ValueError("PVC index marker return code does not match process launch")
    profile_value = payload.get("client_read_profile")
    if (
        not isinstance(profile_value, str)
        or Path(profile_value).resolve() != Path(expected_profile_path).resolve()
    ):
        raise ValueError("PVC index marker is not bound to the client profile path")
    prefix_value = payload.get("index_prefix")
    blocks_value = payload.get("blocks_file")
    artifact_values = payload.get("artifacts")
    if (
        not isinstance(prefix_value, str)
        or not isinstance(blocks_value, str)
        or not isinstance(artifact_values, list)
        or not artifact_values
        or not all(isinstance(value, str) for value in artifact_values)
    ):
        raise ValueError("PVC index marker lacks complete artifact identity fields")
    marker_tool = payload.get("tool")
    if not isinstance(marker_tool, str) or not marker_tool:
        raise ValueError("PVC index marker lacks tool identity")
    identity = {
        "schema": "pvc-index-artifact-identity-v1",
        "policy": INDEX_ARTIFACT_POLICY,
        "tool": marker_tool,
        "output_prefix": str(Path(prefix_value).resolve()),
        "marker": marker_artifact,
        "blocks_file": _sha256_artifact(blocks_value),
        "index_artifacts": [_stat_identity(value) for value in artifact_values],
    }
    identity["index_identity_sha256"] = _canonical_sha256(identity)
    return identity


def _validate_index_identity(
    identity: Any,
    *,
    expected_profile_path: Path,
    expected_tool: str,
) -> None:
    if not isinstance(identity, dict) or identity.get("schema") != "pvc-index-artifact-identity-v1":
        raise ValueError("Fresh PVC index profile lacks an index artifact identity")
    digest = identity.get("index_identity_sha256")
    unsigned = dict(identity)
    unsigned.pop("index_identity_sha256", None)
    if digest != _canonical_sha256(unsigned):
        raise ValueError("PVC index artifact identity digest is invalid")
    if identity.get("policy") != INDEX_ARTIFACT_POLICY:
        raise ValueError("PVC index artifact identity policy changed")
    marker_path = _validate_sha256_artifact(
        identity.get("marker"), label="PVC index marker"
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(marker, dict):
        raise ValueError("PVC index marker is not a JSON object")
    if Path(str(marker.get("client_read_profile", ""))).resolve() != expected_profile_path:
        raise ValueError("PVC index marker no longer names this client profile")
    if identity.get("tool") != expected_tool or marker.get("tool") != expected_tool:
        raise ValueError("PVC index marker/profile tool identity mismatch")
    if marker.get("exit_code") != 0:
        raise ValueError("PVC index marker records an unsuccessful process")
    if str(Path(str(marker.get("index_prefix", ""))).resolve()) != identity.get(
        "output_prefix"
    ):
        raise ValueError("PVC index marker/output-prefix identity mismatch")
    blocks_path = _validate_sha256_artifact(
        identity.get("blocks_file"), label="PVC LD-block artifact"
    )
    if Path(str(marker.get("blocks_file", ""))).resolve() != blocks_path:
        raise ValueError("PVC index marker/blocks identity mismatch")
    artifacts = identity.get("index_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("PVC index artifact identity lacks index artifacts")
    for artifact in artifacts:
        if not isinstance(artifact, dict) or _stat_identity(artifact.get("path", "")) != artifact:
            raise ValueError("PVC index artifact changed after profile materialization")
    marker_artifacts = marker.get("artifacts")
    if not isinstance(marker_artifacts, list) or [
        str(Path(str(value)).resolve()) for value in marker_artifacts
    ] != [str(artifact["path"]) for artifact in artifacts]:
        raise ValueError("PVC index marker/artifact-list identity mismatch")


def _parse_timing_rows(text: str, log_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes = {
        "kmer_counting": "count sample FASTQ k-mers used by the PVC read map",
        "read_map_fill": "map observed read-k-mer counts onto panel-unique k-mers",
        "read_map_serialization": "serialize the sample-specific read UniqueKmersMap",
    }
    for order, (phase, pattern) in enumerate(_READ_PHASE_PATTERNS.items()):
        matches = list(pattern.finditer(text))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one {phase} timing in {log_path}, found {len(matches)}"
            )
        match = matches[0]
        wall = float(match.group("wall"))
        cpu_group = match.groupdict().get("cpu")
        cpu = float(cpu_group) if cpu_group is not None else wall
        reported_threads = int(match.groupdict().get("threads") or 1)
        if wall < 0 or cpu < 0:
            raise ValueError(f"Negative native phase timing for {phase} in {log_path}")
        rows.append(
            {
                "phase": phase,
                "order": order,
                "actor": "client",
                "scope": scopes[phase],
                "included_fastq_to_vcf": True,
                "required_for_vcf": True,
                "wall_seconds": wall,
                "cpu_seconds": cpu,
                "calls": 1,
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
                "reported_threads": reported_threads,
                "cpu_seconds_source": (
                    "PanGenie-process single-thread component used as CPU proxy"
                    if cpu_group is not None
                    else (
                        "native one-thread wall duration used as CPU proxy"
                        if reported_threads == 1
                        else (
                            "native multithread wall duration retained only as "
                            "an uncertified historical CPU proxy"
                        )
                    )
                ),
                "cpu_seconds_is_proxy": True,
                "cpu_seconds_model": (
                    "native single-thread duration used as CPU-time proxy"
                    if cpu_group is not None or reported_threads == 1
                    else "multithread wall duration; not a certified CPU-time proxy"
                ),
            }
        )
    return rows


def write_pvc_index_client_profile(
    output_dir: str | Path,
    *,
    tool: str,
    threads: int,
    manifest: dict[str, Any] | None = None,
    run_dir: str | Path | None = None,
    index_marker_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Persist direct client read-preparation phases from PanGenie-process.

    PanGenie-process reports phase wall times in ``stderr.log`` and, for the
    map-fill phase, an explicit single-thread CPU duration.  The production
    PVC launcher uses one thread.  For the other two single-thread phases the
    direct duration is consequently also used as the phase CPU duration and is
    labelled as such in the profile metadata.
    """
    directory = Path(output_dir).resolve()
    log_path = directory / "stderr.log"
    text = log_path.read_text(errors="replace")
    rows = _parse_timing_rows(text, log_path)
    launch_path = directory / "process_launch_provenance.json"
    launch_provenance: dict[str, Any] | None = None
    input_binding: dict[str, Any] | None = None
    index_identity: dict[str, Any] | None = None
    profile_path = directory / "client_read_profile.json"
    reported_threads = sorted({int(row["reported_threads"]) for row in rows})
    one_thread_verified = int(threads) == 1 and reported_threads == [1]
    if launch_path.is_file():
        expected_binding = None
        if manifest is not None and run_dir is not None:
            from pvc.genotype.private.workflow_profile import (
                build_workflow_input_binding,
            )

            expected_binding = build_workflow_input_binding(manifest, run_dir)
        launch_provenance, launch_path = _validate_fresh_launch(
            directory,
            tool=tool,
            expected_input_binding=expected_binding,
        )
        input_binding = dict(launch_provenance["input_binding"])
        if not one_thread_verified:
            raise ValueError(
                "Fresh paper index profile requires requested and reported one-thread timings"
            )
        marker = (
            Path(index_marker_path).resolve()
            if index_marker_path is not None
            else directory / "index_complete.json"
        )
        index_identity = _build_index_identity(
            marker,
            expected_profile_path=profile_path,
            expected_returncode=int(launch_provenance["returncode"]),
        )
        launch_artifact = _sha256_artifact(launch_path)
        execution_provenance = {
            "status": "fresh_verified",
            "certified": True,
            "reason": None,
            "validation": "complete execution chain re-derived from bound artifacts",
            "launch_artifact_sha256": launch_artifact["sha256"],
        }
        command_artifact = dict(launch_provenance["command_artifact"])
        executable_artifact = {
            "path": str(Path(launch_provenance["executable"]).resolve()),
            "sha256": launch_provenance["executable_sha256_after"],
            "role": "executed_binary",
            "verification": "before/after/current SHA-256 equality",
        }
        log_artifacts = dict(launch_provenance["timing_logs"])
        direct_process_wall_seconds: float | None = float(
            launch_provenance["process_wall_seconds"]
        )
    else:
        if manifest is not None and run_dir is not None:
            from pvc.genotype.private.workflow_profile import (
                build_workflow_input_binding,
            )

            input_binding = build_workflow_input_binding(manifest, run_dir)
        execution_provenance = {
            "status": "historical_unverified",
            "certified": False,
            "reason": (
                "No launch-time executable hash exists; any current binary "
                "hash is a reference only and does not certify the historical run."
            ),
        }
        command_path = directory / "command.json"
        command_artifact = (
            _sha256_artifact(command_path) if command_path.is_file() else None
        )
        log_artifacts = {
            "stderr": _sha256_artifact(log_path),
            "stdout": (
                _sha256_artifact(directory / "stdout.log")
                if (directory / "stdout.log").is_file()
                else None
            ),
        }
        executable_artifact = None
        command: list[str] = []
        if command_path.is_file():
            try:
                command_payload = json.loads(command_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                command_payload = None
            if isinstance(command_payload, dict) and isinstance(
                command_payload.get("command"), list
            ):
                command = [str(value) for value in command_payload["command"]]
        try:
            executable = Path(command[0]).resolve() if command else None
            if executable is not None and executable.is_file():
                executable_artifact = {
                    "path": str(executable),
                    "sha256": _sha256(executable),
                    "role": "current_binary_reference_only",
                    "verification": "historical_unverified",
                    "certifies_executed_binary": False,
                }
        except (OSError, TypeError):
            executable_artifact = None
        launch_artifact = None
        direct_process_wall_seconds = None

    thread_policy = {
        "paper_profile_requires_one_thread": True,
        "requested_threads": int(threads),
        "reported_threads": reported_threads,
        "one_thread_verified": one_thread_verified,
        "historical_behavior": (
            "Historical timings remain descriptive and uncertified when the "
            "one-thread condition cannot be proven."
        ),
    }
    evidence = {
        "execution_status": execution_provenance["status"],
        "command_artifact": command_artifact,
        "timing_logs": log_artifacts,
        "input_identity_sha256": (
            input_binding.get("input_identity_sha256")
            if isinstance(input_binding, dict)
            else None
        ),
        "index_identity_sha256": (
            index_identity.get("index_identity_sha256")
            if isinstance(index_identity, dict)
            else None
        ),
        "direct_process_wall_seconds": direct_process_wall_seconds,
        "thread_policy": thread_policy,
    }
    payload = {
        "schema": INDEX_PROFILE_SCHEMA,
        "schema_version": 2,
        "tool": tool,
        "source": "PanGenie-process native phase summary",
        "source_log": str(log_path.resolve()),
        "source_log_sha256": _sha256(log_path),
        "timing_log_artifacts": log_artifacts,
        "command_artifact": command_artifact,
        "executable_artifact": executable_artifact,
        "execution_provenance": execution_provenance,
        "input_binding": input_binding,
        "index_identity": index_identity,
        "direct_process_wall_seconds": direct_process_wall_seconds,
        "direct_process_wall_definition": (
            "time.perf_counter elapsed immediately around the complete "
            "PanGenie-process command, including process startup and gaps "
            "between native phase summaries"
            if direct_process_wall_seconds is not None
            else "unavailable for historical logs"
        ),
        "thread_policy": thread_policy,
        "profile_evidence_sha256": _canonical_sha256(evidence),
        "process_launch_provenance": launch_provenance,
        "process_launch_provenance_artifact": launch_artifact,
        "requested_threads": int(threads),
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
            "Only sample-specific FASTQ/read-map preparation is included; "
            "public graph/index construction and PanGenie HMM work are excluded."
        ),
        "phases": rows,
    }
    json_path = profile_path
    csv_path = directory / "client_read_profile.csv"
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json_tmp.replace(json_path)
    fields = [
        "phase", "order", "actor", "scope", "included_fastq_to_vcf",
        "required_for_vcf",
        "wall_seconds", "cpu_seconds", "calls",
        "communicator_endpoint_payload_bytes",
        "communicator_endpoint_payload_bytes_max",
        "communicator_endpoint_payload_bytes_sum", "communicator_payload_bytes",
        "communicator_operations", "logical_exchange_calls",
        "client_upload_bytes", "client_upload_rounds",
        "client_download_bytes", "client_download_rounds",
        "client_transfer_model", "reported_threads", "cpu_seconds_source",
        "cpu_seconds_is_proxy", "cpu_seconds_model",
    ]
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["included_fastq_to_vcf"] = 1
            output["required_for_vcf"] = 1
            output["client_transfer_model"] = ""
            writer.writerow(output)
    csv_tmp.replace(csv_path)
    validate_pvc_index_client_profile(
        json_path,
        expected_input_binding=input_binding if launch_provenance is not None else None,
    )
    return json_path, csv_path


def validate_pvc_index_client_profile(
    profile_path: str | Path,
    *,
    expected_input_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a materialized profile by re-checking its underlying evidence."""
    path = Path(profile_path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid PVC index client profile {path}: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != INDEX_PROFILE_SCHEMA
        or payload.get("schema_version") != 2
        or not isinstance(payload.get("tool"), str)
    ):
        raise ValueError(f"Incompatible PVC index client profile {path}")

    source_log = Path(str(payload.get("source_log", ""))).resolve()
    if source_log != path.parent / "stderr.log" or not source_log.is_file():
        raise ValueError("PVC index profile source log path is invalid")
    if payload.get("source_log_sha256") != _sha256(source_log):
        raise ValueError("PVC index profile source timing log changed")
    rows = _parse_timing_rows(source_log.read_text(errors="replace"), source_log)
    if payload.get("phases") != rows:
        raise ValueError("PVC index profile timing rows do not match the bound log")

    requested_threads = payload.get("requested_threads")
    if isinstance(requested_threads, bool) or not isinstance(requested_threads, int):
        raise ValueError("PVC index profile requested_threads is invalid")
    reported_threads = sorted({int(row["reported_threads"]) for row in rows})
    one_thread_verified = requested_threads == 1 and reported_threads == [1]
    expected_thread_policy = {
        "paper_profile_requires_one_thread": True,
        "requested_threads": requested_threads,
        "reported_threads": reported_threads,
        "one_thread_verified": one_thread_verified,
        "historical_behavior": (
            "Historical timings remain descriptive and uncertified when the "
            "one-thread condition cannot be proven."
        ),
    }
    if payload.get("thread_policy") != expected_thread_policy:
        raise ValueError("PVC index profile thread policy is stale or malformed")

    launch_path = path.parent / "process_launch_provenance.json"
    if launch_path.is_file():
        launch, validated_launch_path = _validate_fresh_launch(
            path.parent,
            tool=str(payload["tool"]),
            expected_input_binding=expected_input_binding,
        )
        if not one_thread_verified:
            raise ValueError("Fresh PVC index profile lacks one-thread proof")
        if payload.get("process_launch_provenance") != launch:
            raise ValueError("PVC index profile embeds stale launch provenance")
        launch_artifact = _sha256_artifact(validated_launch_path)
        if payload.get("process_launch_provenance_artifact") != launch_artifact:
            raise ValueError("PVC index profile launch artifact binding is stale")
        if payload.get("input_binding") != launch.get("input_binding"):
            raise ValueError("PVC index profile input binding differs from its launch")
        if expected_input_binding is not None and payload.get("input_binding") != expected_input_binding:
            raise ValueError("PVC index profile input binding differs from the requested run")
        if payload.get("command_artifact") != launch.get("command_artifact"):
            raise ValueError("PVC index profile command artifact binding is stale")
        if payload.get("timing_log_artifacts") != launch.get("timing_logs"):
            raise ValueError("PVC index profile timing-log bindings are stale")
        executable_artifact = payload.get("executable_artifact")
        if (
            not isinstance(executable_artifact, dict)
            or executable_artifact.get("path")
            != str(Path(launch["executable"]).resolve())
            or executable_artifact.get("sha256")
            != launch["executable_sha256_after"]
            or executable_artifact.get("role") != "executed_binary"
        ):
            raise ValueError("PVC index profile executable binding is stale")
        expected_execution = {
            "status": "fresh_verified",
            "certified": True,
            "reason": None,
            "validation": "complete execution chain re-derived from bound artifacts",
            "launch_artifact_sha256": launch_artifact["sha256"],
        }
        if payload.get("execution_provenance") != expected_execution:
            raise ValueError("PVC index profile certification is not evidence-derived")
        identity = payload.get("index_identity")
        _validate_index_identity(
            identity,
            expected_profile_path=path,
            expected_tool=str(payload["tool"]),
        )
        if identity.get("index_artifacts") != launch.get("process_output_artifacts"):
            raise ValueError("PVC index marker artifacts differ from process outputs")
        command = launch["command"]
        command_cwd = Path(str(launch["command_cwd"])).resolve()
        if identity.get("output_prefix") != str(
            _resolve_command_path(command_cwd, _command_option(command, "-o"))
        ):
            raise ValueError("PVC index output-prefix identity differs from command")
        input_binding = payload["input_binding"]
        log_artifacts = launch["timing_logs"]
        command_artifact = launch["command_artifact"]
        direct_process_wall_seconds = float(launch["process_wall_seconds"])
        if payload.get("direct_process_wall_seconds") != direct_process_wall_seconds:
            raise ValueError("PVC index profile direct process wall binding is stale")
        expected_direct_definition = (
            "time.perf_counter elapsed immediately around the complete "
            "PanGenie-process command, including process startup and gaps "
            "between native phase summaries"
        )
    else:
        expected_execution = {
            "status": "historical_unverified",
            "certified": False,
            "reason": (
                "No launch-time executable hash exists; any current binary "
                "hash is a reference only and does not certify the historical run."
            ),
        }
        if payload.get("execution_provenance") != expected_execution:
            raise ValueError("Historical PVC index profile must remain uncertified")
        if payload.get("process_launch_provenance") is not None or payload.get(
            "process_launch_provenance_artifact"
        ) is not None:
            raise ValueError("Historical PVC index profile has false launch evidence")
        if payload.get("index_identity") is not None:
            raise ValueError("Historical PVC index profile must not claim certified index identity")
        command_artifact = payload.get("command_artifact")
        if command_artifact is not None:
            _validate_sha256_artifact(
                command_artifact,
                label="historical PVC index command reference",
                expected_path=path.parent / "command.json",
            )
        log_artifacts = payload.get("timing_log_artifacts")
        if not isinstance(log_artifacts, dict):
            raise ValueError("Historical PVC index profile lacks log references")
        _validate_sha256_artifact(
            log_artifacts.get("stderr"),
            label="historical PVC index stderr reference",
            expected_path=source_log,
        )
        stdout_artifact = log_artifacts.get("stdout")
        if stdout_artifact is not None:
            _validate_sha256_artifact(
                stdout_artifact,
                label="historical PVC index stdout reference",
                expected_path=path.parent / "stdout.log",
            )
        executable_artifact = payload.get("executable_artifact")
        if executable_artifact is not None:
            executable = Path(str(executable_artifact.get("path", ""))).resolve()
            if (
                executable_artifact.get("role") != "current_binary_reference_only"
                or executable_artifact.get("certifies_executed_binary") is not False
                or executable_artifact.get("sha256") != _sha256(executable)
            ):
                raise ValueError("Historical executable reference is stale")
        input_binding = payload.get("input_binding")
        if expected_input_binding is not None and input_binding != expected_input_binding:
            raise ValueError("Historical PVC index profile input binding is stale")
        direct_process_wall_seconds = None
        if payload.get("direct_process_wall_seconds") is not None:
            raise ValueError("Historical PVC index profile claims a direct process wall")
        expected_direct_definition = "unavailable for historical logs"

    if payload.get("direct_process_wall_definition") != expected_direct_definition:
        raise ValueError("PVC index profile direct process-wall definition is stale")

    evidence = {
        "execution_status": expected_execution["status"],
        "command_artifact": command_artifact,
        "timing_logs": log_artifacts,
        "input_identity_sha256": (
            input_binding.get("input_identity_sha256")
            if isinstance(input_binding, dict)
            else None
        ),
        "index_identity_sha256": (
            payload["index_identity"].get("index_identity_sha256")
            if isinstance(payload.get("index_identity"), dict)
            else None
        ),
        "direct_process_wall_seconds": direct_process_wall_seconds,
        "thread_policy": expected_thread_policy,
    }
    if payload.get("profile_evidence_sha256") != _canonical_sha256(evidence):
        raise ValueError("PVC index profile evidence digest is stale")
    return payload


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


def expected_pvc_index_artifacts(
    manifest: dict[str, Any],
    output_prefix: str | Path,
) -> list[Path]:
    prefix = Path(output_prefix)
    chromosome = manifest_chromosome(manifest)
    return [
        prefix.with_name(f"{prefix.name}_path_segments.fasta"),
        prefix.with_name(f"{prefix.name}_UniqueKmersMap.cereal"),
        prefix.with_name(f"{prefix.name}_read_UniqueKmersMap.json"),
        prefix.with_name(f"{prefix.name}_{chromosome}_Graph.cereal"),
        prefix.with_name(f"{prefix.name}_{chromosome}_Graph.json"),
        prefix.with_name(f"{prefix.name}_{chromosome}_kmers.tsv.gz"),
    ]


def missing_pvc_index_artifacts(
    manifest: dict[str, Any],
    output_prefix: str | Path,
) -> list[Path]:
    return [
        artifact
        for artifact in expected_pvc_index_artifacts(manifest, output_prefix)
        if not artifact.exists()
    ]


def build_pvc_index_command(
    tool: str,
    manifest: dict[str, Any],
    output_prefix: str | Path,
    threads: int = 1,
) -> list[str]:
    pvc_index_bin = Path(PVC_INDEX_BINS[tool]).expanduser().resolve()
    return [
        str(pvc_index_bin),
        "-r",
        str(Path(manifest["reference_fasta"]).expanduser().resolve()),
        "-v",
        str(Path(manifest["panel_subset"]).expanduser().resolve()),
        "-i",
        str(Path(manifest["fastq"]).expanduser().resolve()),
        "-o",
        str(Path(output_prefix).expanduser().resolve()),
        "-s",
        str(manifest.get("sample", "sample")),
        "-j",
        str(threads),
        "-t",
        str(threads),
    ]


def run_pangenie_process_index(
    tool: str,
    manifest: dict[str, Any],
    output_dir: str | Path,
    output_prefix: str | Path,
    threads: int = 1,
    verbose: bool = False,
    run_dir: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    if int(threads) != 1:
        raise ValueError(
            "Fresh paper PVC index profiling requires threads=1 for a valid CPU proxy"
        )
    command = build_pvc_index_command(tool, manifest, output_prefix, threads=threads)
    if verbose:
        print(f"{tool} pvc-index output directory: {output_dir}")
        print(f"{tool} pvc-index command: {' '.join(command)}")
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    effective_run_dir = (
        Path(run_dir).resolve() if run_dir is not None else _infer_run_dir(directory)
    )
    from pvc.genotype.private.workflow_profile import build_workflow_input_binding

    input_binding = build_workflow_input_binding(manifest, effective_run_dir)
    input_artifacts_before = _build_input_artifacts(manifest)
    command_cwd = Path(PANGENIE_PROJECT_ROOT).expanduser().resolve()
    executable = _resolve_command_path(command_cwd, command[0])
    executable_sha_before = _sha256(executable)
    started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    process_wall_started = time.perf_counter()
    result = run_command(command, directory, verbose=verbose)
    process_wall_seconds = time.perf_counter() - process_wall_started
    # Capture the exact timing logs immediately on process return, before any
    # index postprocessing or profile materialization can touch them.
    finished_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    timing_logs = {
        "stdout": _sha256_artifact(directory / "stdout.log"),
        "stderr": _sha256_artifact(directory / "stderr.log"),
    }
    executable_sha_after = _sha256(executable)
    unchanged = executable_sha_before == executable_sha_after
    input_artifacts_after = _build_input_artifacts(manifest)
    command_path = directory / "command.json"
    if not command_path.is_file():
        raise RuntimeError(
            f"PanGenie-process launch did not persist command artifact: {command_path}"
        )
    try:
        command_payload = json.loads(command_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"PanGenie-process wrote malformed command metadata: {command_path}"
        ) from exc
    recorded_cwd = (
        command_payload.get("cwd") if isinstance(command_payload, dict) else None
    )
    if (
        not isinstance(recorded_cwd, str)
        or Path(recorded_cwd).expanduser().resolve() != command_cwd
        or command_payload.get("command") != command
    ):
        raise RuntimeError(
            "PanGenie-process command metadata does not match the executed command/cwd"
        )
    process_output_artifacts = [
        _optional_stat_identity(path)
        for path in expected_pvc_index_artifacts(manifest, output_prefix)
    ]
    launch_payload = {
        "schema": PROCESS_LAUNCH_SCHEMA,
        "schema_version": 2,
        "tool": tool,
        "requested_threads": int(threads),
        "command": command,
        "command_cwd": str(command_cwd),
        "command_identity_sha256": _canonical_sha256(command),
        "command_artifact": _sha256_artifact(command_path),
        "executable": str(executable),
        "executable_sha256_before": executable_sha_before,
        "executable_sha256_after": executable_sha_after,
        "executable_unchanged_during_run": unchanged,
        "timing_logs": timing_logs,
        "input_binding": input_binding,
        "input_artifact_policy": INPUT_ARTIFACT_POLICY,
        "input_artifacts_before": input_artifacts_before,
        "input_artifacts_after": input_artifacts_after,
        "inputs_unchanged_during_run": (
            input_artifacts_before == input_artifacts_after
        ),
        "process_output_artifact_policy": INDEX_ARTIFACT_POLICY[
            "large_index_artifacts"
        ],
        "process_output_artifacts": process_output_artifacts,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "process_wall_seconds": process_wall_seconds,
        "returncode": int(result.returncode),
    }
    launch_path = directory / "process_launch_provenance.json"
    launch_tmp = launch_path.with_suffix(".json.tmp")
    launch_tmp.write_text(
        json.dumps(launch_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    launch_tmp.replace(launch_path)
    if not unchanged:
        raise RuntimeError(
            f"PanGenie-process executable changed during the run: {executable}"
        )
    if input_artifacts_before != input_artifacts_after:
        raise RuntimeError("PVC index reference/panel/FASTQ inputs changed during the run")
    return result
