"""Stable, phase-native profiling for private PVC workflows.

The paper profiler deliberately keeps two communication domains separate:

* ``communicator_*payload_bytes`` values are modeled application-payload
  deltas from CrypTen's communicator, not physical wire measurements.
* ``client_*`` values are explicit protocol models because the prototype runs
  the logical client as rank 0 and CrypTen's counters cannot observe an
  external client's upload/download.

Canonical phase contexts must not overlap.  This makes the aggregate CSV safe
to sum by actor without reconstructing a component as an algebraic residual.
Legacy nested diagnostic timers in :mod:`pvc.genotype.private.shared` remain
available, but are not mixed into these canonical rows.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import numbers
import os
import platform
import resource
import socket
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from pvc.genotype.private.dependency_rounds import (
    CERTIFICATE_SCHEMA,
    TRACE_SCHEMA,
    build_application_dependency_round_certificate,
    build_modeled_client_rounds,
)

SCHEMA = "pvc-private-workflow-profile-v3"
AGGREGATE_SCHEMA = "pvc-private-workflow-profile-aggregate-v4"
FASTQ_TO_VCF_BOUNDARY_CONTRACT_SCHEMA = "pvc-fastq-to-vcf-boundary-contract-v1"
LEGACY_INDEX_PROFILE_SCHEMA = "pvc-index-client-profile-v2"
# Fresh paper profiles use a distinct schema because their measured boundary is
# scientifically different: sample-only PanGenie-readmap against a reusable
# public index, rather than a full PanGenie-process invocation.
INDEX_PROFILE_SCHEMA = "pvc-readmap-client-profile-v1"
INPUT_BINDING_SCHEMA = "pvc-workflow-input-binding-v1"
GENOTYPE_INPUT_ARTIFACT_SCHEMA = "pvc-genotype-consumed-index-artifacts-v1"
EFFECTIVE_CONFIGURATION_SCHEMA = "pvc-private-effective-configuration-v1"
PROCESS_MEMORY_SCHEMA = "pvc-rank-process-memory-v1"
PARENT_LAUNCH_BOUNDARY_SCHEMA = "pvc-private-parent-launch-boundary-v1"
PROFILE_ENABLED = os.environ.get("PVC_WORKFLOW_PROFILE", "1") != "0"

_MANIFEST_SUMMARY_FIELDS = (
    "sample",
    "chromosome",
    "interval",
    "coverage",
    "coverage_x",
    "replicate",
    "mean_kmer_abundance",
)
_MANIFEST_INPUT_PATH_FIELDS = (
    "reference_fasta",
    "panel_subset",
    "fastq",
    "truth",
    "source_cram",
    "source_crai",
    "input_bam",
    "bam",
    "pangenie_index_prefix",
)
_REQUIRED_COMM_COUNTERS = {
    "comm_bytes": "modeled application tensor-payload bytes at this communicator endpoint",
    "comm_rounds": "legacy low-level communicator-operation counter",
    "comm_logical_exchanges": "top-level collective or batched point-to-point call counter",
}
_REQUIRED_COMM_EVENT_COUNTERS = {
    "comm_event_bytes": "per-event endpoint application tensor-payload bytes",
    "comm_event_rounds": "per-event legacy communicator operations",
    "comm_event_logical_exchanges": "per-event top-level logical exchange calls",
}

# Order is part of the stable paper-facing schema.  Index phases are supplied
# by the direct PanGenie-readmap summary; the rest are observed in each party.
PHASE_ORDER = (
    "public_index_load",
    "kmer_counting",
    "read_map_probability_setup",
    "read_map_fill",
    "read_map_serialization",
    "input_loading",
    "private_read_map_loading",
    "probability_table_setup",
    "control_plan_setup",
    "probability_vector_construction",
    "frequency_vector_setup",
    "result_buffer_setup",
    "pair_table_setup",
    "emission_input_encoding",
    "emission_computation",
    "haplotype_scoring",
    "argmax",
    "reveal",
    "reveal_postprocessing",
    "genotype_reconstruction",
    "singleton_calling",
    "private_score_artifact_output",
    "likelihood_output",
    "vcf_output",
    "truth_verification",
    "completion_marker",
)

_INDEX_PHASES = {
    "public_index_load",
    "kmer_counting",
    "read_map_probability_setup",
    "read_map_fill",
    "read_map_serialization",
}
_POST_VCF_PHASES = {
    "truth_verification",
    "completion_marker",
}
_NONESSENTIAL_OUTPUT_PHASES = {
    "private_score_artifact_output",
    "likelihood_output",
    "truth_verification",
    "completion_marker",
}

# Exhaustive, non-overlapping reporting groups used directly by the paper
# plotting code.  Keeping this mapping beside the canonical order makes it
# impossible for downstream code to manufacture a component by subtraction.
PHASE_GROUPS = {
    "kmer_read_preparation": (
        "public_index_load",
        "kmer_counting",
        "read_map_probability_setup",
        "read_map_fill",
        "read_map_serialization",
        "private_read_map_loading",
        "probability_vector_construction",
    ),
    "public_protocol_setup": (
        "input_loading",
        "probability_table_setup",
        "control_plan_setup",
        "frequency_vector_setup",
        "result_buffer_setup",
        "pair_table_setup",
    ),
    "emissions": (
        "emission_input_encoding",
        "emission_computation",
    ),
    "haplotype_scoring": ("haplotype_scoring",),
    "argmax": ("argmax",),
    "reveal": ("reveal",),
    "client_postprocessing_reconstruction": (
        "reveal_postprocessing",
        "genotype_reconstruction",
        "singleton_calling",
    ),
    "vcf_output": ("vcf_output",),
    "diagnostic_output": (
        "private_score_artifact_output",
        "likelihood_output",
    ),
    "post_vcf_verification": (
        "truth_verification",
        "completion_marker",
    ),
}

_GROUPED_PHASES = tuple(phase for phases in PHASE_GROUPS.values() for phase in phases)
if len(_GROUPED_PHASES) != len(set(_GROUPED_PHASES)) or set(_GROUPED_PHASES) != set(
    PHASE_ORDER
):
    raise RuntimeError("PVC workflow phase groups must partition PHASE_ORDER exactly")


def phase_actor(name: str, tool: str) -> str:
    """Return the logical protocol actor for a canonical phase."""
    if name in _INDEX_PHASES or name in {
        "private_read_map_loading",
        "probability_vector_construction",
        "emission_input_encoding",
        "reveal_postprocessing",
        "genotype_reconstruction",
        "singleton_calling",
        "private_score_artifact_output",
        "likelihood_output",
        "vcf_output",
        "truth_verification",
        "completion_marker",
    }:
        return "client"
    if name == "emission_computation":
        return "computing_parties" if tool == "pvc-heavy" else "client"
    if name == "argmax":
        return "client" if tool in {"pvc-light", "pvc-light-gpu"} else "computing_parties"
    return "computing_parties"


def phase_scope(name: str, tool: str) -> str:
    scopes = {
        "public_index_load": "load the reusable public UniqueKmersMap",
        "kmer_counting": "count sample FASTQ k-mers used by the PVC read map",
        "read_map_probability_setup": (
            "construct the sample read-map probability table from the measured "
            "k-mer abundance peak"
        ),
        "read_map_fill": "map observed read-k-mer counts onto panel-unique k-mers",
        "read_map_serialization": "serialize the sample-specific read UniqueKmersMap",
        "input_loading": "load and validate public graph, panel, and protocol inputs",
        "private_read_map_loading": "client load of the sample-specific read-k-mer map",
        "probability_table_setup": "construct the public probability lookup table",
        "control_plan_setup": "construct/load the public block control plan",
        "probability_vector_construction": (
            "construct per-variant probability vectors from read counts"
        ),
        "frequency_vector_setup": "construct public allele-frequency vectors",
        "result_buffer_setup": "allocate per-variant genotype result buffers and block assignments",
        "pair_table_setup": "precompute public candidate haplotype-pair tables",
        "emission_input_encoding": (
            "client construction of private count/coverage selectors and modeled upload"
        ),
        "emission_computation": (
            "secure private count/coverage lookup and encrypted emissions"
            if tool == "pvc-heavy"
            else "client plaintext fixed-point genotype-emission construction"
        ),
        "haplotype_scoring": "score public candidate haplotype pairs from genotype emissions",
        "argmax": (
            "client local first-index argmax of revealed scores"
            if tool in {"pvc-light", "pvc-light-gpu"}
            else "secure first-index argmax of encrypted candidate scores"
        ),
        "reveal": "network reconstruction of permitted scores or winning indices",
        "reveal_postprocessing": (
            "client decode, split, and materialize reconstructed scores or winning indices"
        ),
        "genotype_reconstruction": "client assignment of winning haplotype pairs to genotypes",
        "singleton_calling": "client local genotype calling for variants outside scored blocks",
        "private_score_artifact_output": "write optional private-score diagnostic JSON",
        "likelihood_output": "write diagnostic bubble-likelihood table",
        "vcf_output": "write the requested genotype VCF",
        "truth_verification": "post-VCF comparison with truth data",
        "completion_marker": "write workflow completion/provenance marker",
    }
    return scopes[name]


def phase_in_fastq_to_vcf(name: str) -> bool:
    """Whether a phase executes before the prototype's VCF boundary."""
    return name not in _POST_VCF_PHASES


def phase_required_for_vcf(name: str) -> bool:
    """Whether a phase is necessary to produce the VCF rather than diagnostic."""
    return name not in _NONESSENTIAL_OUTPUT_PHASES


def _empty_record(name: str, tool: str) -> dict[str, Any]:
    return {
        "phase": name,
        "order": PHASE_ORDER.index(name),
        "actor": phase_actor(name, tool),
        "scope": phase_scope(name, tool),
        "included_fastq_to_vcf": phase_in_fastq_to_vcf(name),
        "required_for_vcf": phase_required_for_vcf(name),
        "measurement_status": "measured",
        "wall_seconds": 0.0,
        "cpu_seconds": 0.0,
        "cpu_seconds_model": "observed process_time delta",
        "calls": 0,
        # CrypTen's communicator accounts serialized application tensor
        # payload at each endpoint.  It does not measure transport headers,
        # retransmission, or physical wire traffic.  The aggregate retains the
        # per-rank values and applies an explicitly labelled half-sum payload
        # accounting convention.
        "communicator_endpoint_payload_bytes": 0,
        "communicator_endpoint_payload_bytes_max": None,
        "communicator_endpoint_payload_bytes_sum": None,
        "communicator_payload_bytes": None,
        # CrypTen historically calls this counter ``comm_rounds``, but it is a
        # low-level communicator-operation proxy (and can count every tensor in
        # one batched invocation).  It must not be reported as protocol rounds.
        "communicator_operations": 0,
        "logical_exchange_calls": 0,
        "communicator_events": {},
        "modeled_computing_party_payload_bytes": None,
        "modeled_client_reconstruction_payload_bytes": None,
        "observed_computing_party_exchange_calls": None,
        "observed_client_reconstruction_exchange_calls": None,
        "classified_computing_party_payload_bytes": None,
        "classified_client_payload_bytes": None,
        "classified_computing_party_exchange_calls": None,
        "classified_client_exchange_calls": None,
        "client_upload_bytes": 0,
        "client_upload_rounds": 0,
        "client_download_bytes": 0,
        "client_download_rounds": 0,
        "client_transfer_model": [],
        "client_transfer_events": [],
    }


_tool = ""
_run_dir: Path | None = None
_records: dict[str, dict[str, Any]] = {}
_active_canonical: list[str] = []
_started_wall = 0.0
_started_cpu = 0.0
_fastq_to_vcf_wall: float | None = None
_fastq_to_vcf_cpu: float | None = None
_started_utc = ""
_source_start: dict[str, Any] = {}
_runtime_start: dict[str, Any] = {}
_execution_id = ""
_input_binding: dict[str, Any] = {}
_output_vcf_artifact: dict[str, Any] | None = None
_communication_counter_capability: dict[str, Any] = {}
_communication_run_start: dict[str, Any] | None = None
_communication_run_start_error: str | None = None
_genotype_input_artifacts: dict[str, Any] | None = None
_effective_configuration: dict[str, Any] | None = None
_process_identity_start: dict[str, Any] | None = None
_client_transfer_ordinal = 0


def _canonical_json_value(value: Any) -> Any:
    """Return a deterministic JSON value or reject an ambiguous manifest."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            raise ValueError("Workflow manifests cannot contain NaN or infinity")
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    raise TypeError(
        "Workflow manifest values must be JSON-compatible; "
        f"got {type(value).__name__}"
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _canonical_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_workflow_input_binding(
    manifest: dict[str, Any], run_dir: str | Path
) -> dict[str, Any]:
    """Build the canonical sample/input identity shared by index and genotype.

    File contents are deliberately not re-hashed here: FASTQ and reference
    inputs can be hundreds of gigabytes.  The binding instead commits to the
    complete canonical manifest and resolved input path strings.  Individual
    index/output artifacts are bound separately by their own SHA-256 records.
    """
    canonical_manifest = _canonical_json_value(manifest)
    if not isinstance(canonical_manifest, dict):  # defensive typing guard
        raise TypeError("PVC workflow manifest must be a JSON object")
    input_paths = {
        key: str(Path(str(canonical_manifest[key])).expanduser().resolve())
        for key in _MANIFEST_INPUT_PATH_FIELDS
        if canonical_manifest.get(key) not in (None, "")
    }
    summary = {
        key: canonical_manifest.get(key)
        for key in _MANIFEST_SUMMARY_FIELDS
        if key in canonical_manifest
    }
    binding = {
        "schema": INPUT_BINDING_SCHEMA,
        "canonicalization": (
            "UTF-8 JSON, recursively normalized JSON values, sorted object "
            "keys, compact separators; input paths resolved without content hashing"
        ),
        "run_dir": str(Path(run_dir).expanduser().resolve()),
        "manifest_sha256": _canonical_sha256(canonical_manifest),
        "canonical_manifest": canonical_manifest,
        "manifest_summary": summary,
        "input_paths": input_paths,
        "input_paths_sha256": _canonical_sha256(input_paths),
    }
    binding["input_identity_sha256"] = _canonical_sha256(binding)
    return binding


def _validate_input_binding(binding: Any) -> dict[str, Any]:
    if not isinstance(binding, dict) or binding.get("schema") != INPUT_BINDING_SCHEMA:
        raise ValueError("Workflow profile lacks a canonical input binding")
    required = {
        "run_dir",
        "manifest_sha256",
        "canonical_manifest",
        "manifest_summary",
        "input_paths",
        "input_paths_sha256",
        "input_identity_sha256",
    }
    missing = sorted(required - set(binding))
    if missing:
        raise ValueError(f"Workflow input binding lacks fields: {missing}")
    if binding["manifest_sha256"] != _canonical_sha256(binding["canonical_manifest"]):
        raise ValueError("Workflow input binding manifest digest is invalid")
    if binding["input_paths_sha256"] != _canonical_sha256(binding["input_paths"]):
        raise ValueError("Workflow input binding path digest is invalid")
    body = dict(binding)
    claimed_identity = body.pop("input_identity_sha256")
    if claimed_identity != _canonical_sha256(body):
        raise ValueError("Workflow input binding identity digest is invalid")
    return dict(binding)


def new_workflow_execution_id() -> str:
    """Create a parent-side identifier that cannot collide with stale ranks."""
    import uuid

    return f"pvc-genotype-{uuid.uuid4().hex}"


def _communication_counter_capability_metadata() -> dict[str, Any]:
    """Describe the live communicator implementation without hiding failures."""
    metadata: dict[str, Any] = {
        "crypten_initialized": False,
        "required_counters": dict(_REQUIRED_COMM_COUNTERS),
        "required_event_counters": dict(_REQUIRED_COMM_EVENT_COUNTERS),
        "counters": {
            name: {"available": False, "value_type": None}
            for name in (*_REQUIRED_COMM_COUNTERS, *_REQUIRED_COMM_EVENT_COUNTERS)
        },
        "complete": False,
        "status": "crypten_unavailable",
        "implementation": None,
        "dependency_trace": {
            "schema": TRACE_SCHEMA,
            "enabled": False,
            "snapshot_api_available": False,
            "phase_api_available": False,
            "complete": False,
        },
    }
    try:
        import crypten
    except Exception as exc:
        metadata["detail"] = f"{type(exc).__name__}: {exc}"
        return metadata
    is_initialized = getattr(crypten, "is_initialized", None)
    if not callable(is_initialized):
        metadata["status"] = "initialization_api_unavailable"
        return metadata
    try:
        initialized = bool(is_initialized())
    except Exception as exc:
        metadata["status"] = "initialization_check_failed"
        metadata["detail"] = f"{type(exc).__name__}: {exc}"
        return metadata
    metadata["crypten_initialized"] = initialized
    if not initialized:
        metadata["status"] = "not_initialized"
        return metadata
    try:
        communicator = crypten.communicator.get()
    except Exception as exc:
        metadata["status"] = "communicator_unavailable"
        metadata["detail"] = f"{type(exc).__name__}: {exc}"
        return metadata

    communicator_type = type(communicator)
    module = inspect.getmodule(communicator_type)
    module_path = None
    module_sha256 = None
    if module is not None and getattr(module, "__file__", None):
        try:
            resolved_module = Path(str(module.__file__)).resolve()
            module_path = str(resolved_module)
            if resolved_module.is_file():
                module_sha256 = _sha256_file(resolved_module)
        except OSError:
            module_path = str(getattr(module, "__file__", ""))
    metadata["implementation"] = {
        "class": f"{communicator_type.__module__}.{communicator_type.__qualname__}",
        "module_path": module_path,
        "module_sha256": module_sha256,
        "crypten_version": str(getattr(crypten, "__version__", "unknown")),
        "crypten_module_path": (
            str(Path(crypten.__file__).resolve())
            if getattr(crypten, "__file__", None)
            else None
        ),
    }
    trace_snapshot = getattr(communicator, "dependency_trace_snapshot", None)
    trace_phase = getattr(communicator, "set_dependency_trace_phase", None)
    trace_enabled = getattr(communicator, "_dependency_trace_enabled", False)
    metadata["dependency_trace"] = {
        "schema": TRACE_SCHEMA,
        "enabled": trace_enabled is True,
        "snapshot_api_available": callable(trace_snapshot),
        "phase_api_available": callable(trace_phase),
        "complete": bool(
            trace_enabled is True
            and callable(trace_snapshot)
            and callable(trace_phase)
        ),
    }
    missing = []
    for name in _REQUIRED_COMM_COUNTERS:
        try:
            value = getattr(communicator, name)
        except Exception:
            missing.append(name)
            continue
        valid = isinstance(value, numbers.Integral) and not isinstance(value, bool)
        metadata["counters"][name] = {
            "available": bool(valid),
            "value_type": type(value).__name__,
        }
        if not valid:
            missing.append(name)
    for name in _REQUIRED_COMM_EVENT_COUNTERS:
        try:
            value = getattr(communicator, name)
        except Exception:
            missing.append(name)
            continue
        valid = isinstance(value, dict)
        metadata["counters"][name] = {
            "available": bool(valid),
            "value_type": type(value).__name__,
        }
        if not valid:
            missing.append(name)
    metadata["complete"] = not missing
    metadata["status"] = "available" if not missing else "missing_required_counters"
    if missing:
        metadata["missing_counters"] = missing
    return metadata


def _linux_process_identity() -> dict[str, Any]:
    """Bind this rank to a PID identity that remains safe after PID reuse."""
    if platform.system() != "Linux":
        raise RuntimeError("PVC rank memory profiling requires Linux procfs")
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
        stat = Path("/proc/self/stat").read_text(encoding="utf-8")
        after = stat[stat.rindex(")") + 1 :].split()
        start_time_ticks = int(after[19])  # field 22; after[0] is field 3
    except (OSError, ValueError, IndexError) as exc:
        raise RuntimeError("Could not bind rank process identity from procfs") from exc
    if not boot_id or start_time_ticks <= 0:
        raise RuntimeError("Rank process identity from procfs is incomplete")
    return {
        "boot_id": boot_id,
        "pid": os.getpid(),
        "start_time_ticks": start_time_ticks,
        "ppid_at_profile_start": os.getppid(),
    }


def _read_self_status_memory_kib() -> dict[str, int]:
    """Read current and lifetime-high-water RSS from Linux procfs once."""
    values: dict[str, int] = {}
    try:
        with Path("/proc/self/status").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    fields = line.split()
                    if len(fields) < 3 or fields[2] != "kB":
                        raise RuntimeError("Unexpected VmRSS unit in /proc/self/status")
                    values["vmrss_kib"] = int(fields[1])
                elif line.startswith("VmHWM:"):
                    fields = line.split()
                    if len(fields) < 3 or fields[2] != "kB":
                        raise RuntimeError("Unexpected VmHWM unit in /proc/self/status")
                    values["vmhwm_kib"] = int(fields[1])
    except (OSError, ValueError) as exc:
        raise RuntimeError("Could not read rank RSS from /proc/self/status") from exc
    if set(values) != {"vmrss_kib", "vmhwm_kib"} or any(
        value < 0 for value in values.values()
    ):
        raise RuntimeError("Rank RSS fields are missing or invalid in procfs")
    return values


def _capture_process_memory(
    process_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    """Capture low-overhead, process-lifetime rank memory at workflow end."""
    current_identity = _linux_process_identity()
    if process_identity is None or any(
        process_identity.get(field) != current_identity.get(field)
        for field in ("boot_id", "pid", "start_time_ticks")
    ):
        raise RuntimeError("Rank process identity changed during profiling")
    proc_status = _read_self_status_memory_kib()
    raw_ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if (
        isinstance(raw_ru_maxrss, bool)
        or not isinstance(raw_ru_maxrss, numbers.Real)
        or not math.isfinite(float(raw_ru_maxrss))
        or float(raw_ru_maxrss) < 0
    ):
        raise RuntimeError("resource.getrusage(RUSAGE_SELF).ru_maxrss is invalid")
    # Linux reports ru_maxrss in KiB.  procfs is required above, so silently
    # applying macOS's byte convention would be an error rather than fallback.
    ru_maxrss_kib = int(raw_ru_maxrss)
    return {
        "schema": PROCESS_MEMORY_SCHEMA,
        "schema_version": 1,
        "measurement_status": "measured",
        "process_identity": dict(process_identity),
        "observation_boundary": (
            "once immediately after workflow/whole-run communication completion "
            "and before end-of-run source hashing or profile serialization"
        ),
        "proc_self_status": {
            "source": "/proc/self/status",
            "unit": "KiB (1024 bytes)",
            **proc_status,
        },
        "resource_getrusage_self": {
            "source": "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss",
            "platform": "Linux",
            "raw_value": raw_ru_maxrss,
            "raw_unit": "KiB (1024 bytes) on Linux",
            "ru_maxrss_kib": ru_maxrss_kib,
        },
        "semantics": {
            "vmhwm_kib": (
                "kernel-maintained high-water resident set of this one rank "
                "process from process launch through the observation boundary"
            ),
            "ru_maxrss_kib": (
                "getrusage high-water resident set of this one rank process "
                "from process launch through the observation boundary"
            ),
            "not_phase_specific": True,
            "rank_peaks_are_not_time_aligned_or_additive": True,
            "rank0_role_caveat": (
                "Rank 0 hosts both a computing party and prototype logical-client "
                "work; its process RSS cannot be separated into client-only and "
                "computing-party-only memory."
            ),
        },
    }


def _validate_process_memory(value: Any) -> dict[str, Any]:
    """Validate a native rank-memory record without re-reading an exited PID."""
    if (
        not isinstance(value, dict)
        or value.get("schema") != PROCESS_MEMORY_SCHEMA
        or value.get("schema_version") != 1
        or value.get("measurement_status") != "measured"
    ):
        raise ValueError("Rank profile lacks measured native process memory")
    identity = value.get("process_identity")
    if not isinstance(identity, dict):
        raise ValueError("Rank process memory lacks a stable process identity")
    for field in ("pid", "start_time_ticks"):
        item = identity.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"Rank process identity has invalid {field}: {item!r}")
    if not isinstance(identity.get("boot_id"), str) or not identity["boot_id"]:
        raise ValueError("Rank process identity lacks a Linux boot ID")
    proc_status = value.get("proc_self_status")
    usage = value.get("resource_getrusage_self")
    if not isinstance(proc_status, dict) or not isinstance(usage, dict):
        raise ValueError("Rank process memory lacks both native measurement sources")
    if (
        proc_status.get("source") != "/proc/self/status"
        or proc_status.get("unit") != "KiB (1024 bytes)"
        or usage.get("source")
        != "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss"
        or usage.get("platform") != "Linux"
        or usage.get("raw_unit") != "KiB (1024 bytes) on Linux"
    ):
        raise ValueError("Rank process memory has incompatible source/unit metadata")
    for owner, field in (
        (proc_status, "vmrss_kib"),
        (proc_status, "vmhwm_kib"),
        (usage, "ru_maxrss_kib"),
    ):
        item = owner.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"Rank process memory has invalid {field}: {item!r}")
    if int(proc_status["vmhwm_kib"]) < int(proc_status["vmrss_kib"]):
        raise ValueError("Rank VmHWM is smaller than current VmRSS")
    semantics = value.get("semantics")
    if not isinstance(semantics, dict) or not (
        semantics.get("not_phase_specific") is True
        and semantics.get("rank_peaks_are_not_time_aligned_or_additive") is True
        and isinstance(semantics.get("rank0_role_caveat"), str)
    ):
        raise ValueError("Rank process memory lacks required interpretation caveats")
    return dict(value)


def reset_workflow_profile(
    tool: str,
    run_dir: str | Path,
    *,
    execution_id: str | None = None,
    input_binding: dict[str, Any] | None = None,
    source_provenance_start: dict[str, Any] | None = None,
) -> None:
    """Reset process-local records at the beginning of one CrypTen party."""
    global _tool, _run_dir, _records, _active_canonical
    global _started_wall, _started_cpu, _fastq_to_vcf_wall, _fastq_to_vcf_cpu
    global _started_utc
    global _source_start, _runtime_start, _execution_id, _input_binding
    global _output_vcf_artifact, _communication_counter_capability
    global _communication_run_start, _communication_run_start_error
    global _genotype_input_artifacts, _effective_configuration
    global _process_identity_start
    global _client_transfer_ordinal
    _tool = tool
    _run_dir = Path(run_dir)
    _records = {}
    _active_canonical = []
    _execution_id = str(execution_id or new_workflow_execution_id())
    if not _execution_id.strip():
        raise ValueError("PVC workflow execution ID must be non-empty")
    _input_binding = _validate_input_binding(
        input_binding
        if input_binding is not None
        else build_workflow_input_binding({}, run_dir)
    )
    _output_vcf_artifact = None
    _genotype_input_artifacts = None
    _effective_configuration = None
    _process_identity_start = _linux_process_identity() if PROFILE_ENABLED else None
    _client_transfer_ordinal = 0
    # Production captures this source manifest once in the parent before the
    # direct parent-launch timer. Passing the small certified record into each
    # spawned rank avoids charging three shared-filesystem source-tree scans to
    # T15. Standalone/test callers retain the local fallback.
    _source_start = (
        _validate_source_provenance(source_provenance_start)
        if PROFILE_ENABLED and source_provenance_start is not None
        else _source_provenance()
        if PROFILE_ENABLED
        else {}
    )
    _runtime_start = _runtime_provenance() if PROFILE_ENABLED else {}
    _communication_counter_capability = (
        _communication_counter_capability_metadata() if PROFILE_ENABLED else {}
    )
    _communication_run_start = None
    _communication_run_start_error = None
    if PROFILE_ENABLED:
        try:
            # The production caller resets CrypTen's counters immediately
            # before this function.  Capture the accounting boundary before
            # starting the workflow clocks so profiler snapshots themselves
            # are not charged to a paper phase or the direct genotype timer.
            _communication_run_start = _communication_accounting_snapshot()
        except Exception as exc:
            # Preserve an auditable rank artifact and fail it closed at write
            # time instead of losing the evidence for a counter-capability
            # failure before the workflow starts.
            _communication_run_start_error = f"{type(exc).__name__}: {exc}"
    _started_wall = time.perf_counter()
    _started_cpu = time.process_time()
    _fastq_to_vcf_wall = None
    _fastq_to_vcf_cpu = None
    _started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Required genotype input artifact is missing: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }


def _sha256_file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Required genotype input artifact is missing: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": int(stat.st_size),
    }


def _build_genotype_input_artifacts(
    read_map: str | Path,
    graph: str | Path,
    blocks_file: str | Path,
) -> dict[str, Any]:
    body = {
        "schema": GENOTYPE_INPUT_ARTIFACT_SCHEMA,
        "policy": {
            "read_map": "resolved path plus exact POSIX stat",
            "graph": "resolved path plus exact POSIX stat",
            "blocks_file": "resolved path plus size and full SHA-256",
        },
        "read_map": _stat_file_identity(read_map),
        "graph": _stat_file_identity(graph),
        "blocks_file": _sha256_file_identity(blocks_file),
    }
    body["identity_sha256"] = _canonical_sha256(body)
    return body


def bind_genotype_input_artifacts(
    read_map: str | Path,
    graph: str | Path,
    blocks_file: str | Path,
) -> dict[str, Any]:
    """Bind the exact index artifacts this rank will consume.

    The read map and graph use the same large-artifact identity convention as
    the certified index profile.  The much smaller LD-block file is content
    hashed.  Rebinding to a different artifact set in one rank fails closed.
    """
    global _genotype_input_artifacts
    value = _build_genotype_input_artifacts(read_map, graph, blocks_file)
    if _genotype_input_artifacts is not None and value != _genotype_input_artifacts:
        raise RuntimeError("Genotype input artifacts changed within one rank execution")
    _genotype_input_artifacts = value
    return dict(value)


def _validate_genotype_input_artifacts(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("schema") != GENOTYPE_INPUT_ARTIFACT_SCHEMA
    ):
        raise ValueError("Rank profile lacks bound genotype input artifacts")
    unsigned = dict(value)
    claimed = unsigned.pop("identity_sha256", None)
    if claimed != _canonical_sha256(unsigned):
        raise ValueError("Rank genotype input artifact identity digest is invalid")
    try:
        current = _build_genotype_input_artifacts(
            value["read_map"]["path"],
            value["graph"]["path"],
            value["blocks_file"]["path"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("Rank genotype input artifact identity is malformed") from exc
    if current != value:
        raise ValueError("A genotype-consumed index artifact changed after binding")
    return dict(value)


def bind_effective_configuration(configuration: dict[str, Any]) -> dict[str, Any]:
    """Bind the effective protocol configuration used by this rank."""
    global _effective_configuration
    canonical = _canonical_json_value(configuration)
    if not isinstance(canonical, dict):
        raise TypeError("Effective PVC configuration must be a JSON object")
    value = {
        "schema": EFFECTIVE_CONFIGURATION_SCHEMA,
        "values": canonical,
    }
    value["configuration_sha256"] = _canonical_sha256(value)
    if _effective_configuration is not None and value != _effective_configuration:
        raise RuntimeError("Effective PVC configuration changed within one rank execution")
    _effective_configuration = value
    return dict(value)


def _validate_effective_configuration(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("schema") != EFFECTIVE_CONFIGURATION_SCHEMA
        or not isinstance(value.get("values"), dict)
    ):
        raise ValueError("Rank profile lacks an effective PVC configuration")
    unsigned = dict(value)
    claimed = unsigned.pop("configuration_sha256", None)
    if claimed != _canonical_sha256(unsigned):
        raise ValueError("Rank effective PVC configuration digest is invalid")
    return dict(value)


def _validate_tool_effective_configuration(value: Any, tool: str) -> dict[str, Any]:
    """Apply protocol-specific fail-closed checks after digest validation."""
    validated = _validate_effective_configuration(value)
    configuration = validated["values"]
    if tool == "pvc-heavy" and configuration.get("private_lookup") != "no_truncation":
        raise ValueError(
            "pvc-heavy requires effective_configuration.private_lookup="
            "'no_truncation'"
        )
    exact_trace = configuration.get("exact_communication_trace")
    if not (
        isinstance(exact_trace, dict)
        and exact_trace.get("enabled") is True
        and exact_trace.get("environment_variable") == "PVC_EXACT_COMM_TRACE"
        and exact_trace.get("environment_value") == "1"
        and exact_trace.get("scope") == "CrypTen API application payloads"
    ):
        raise ValueError(
            "PVC paper profiles require effective exact application "
            "communication tracing with PVC_EXACT_COMM_TRACE=1"
        )
    return validated


def _source_provenance() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[4]
    paths: set[Path] = set()
    for relative in (
        "src/pvc",
        "src/crypto/CrypTen/crypten",
    ):
        tree = project_root / relative
        if tree.is_dir():
            paths.update(path for path in tree.rglob("*.py") if path.is_file())
    manifest = {
        path.relative_to(project_root).as_posix(): _sha256_file(path)
        for path in sorted(paths)
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return {
        "project_root": str(project_root),
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_manifest": manifest,
    }


def _validate_source_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Workflow source provenance must be a JSON object")
    project_root = value.get("project_root")
    manifest = value.get("source_manifest")
    claimed = value.get("manifest_sha256")
    if (
        not isinstance(project_root, str)
        or not project_root
        or not isinstance(manifest, dict)
        or not all(
            isinstance(path, str)
            and path
            and isinstance(digest, str)
            and len(digest) == 64
            for path, digest in manifest.items()
        )
    ):
        raise ValueError("Workflow source provenance is malformed")
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    if claimed != hashlib.sha256(encoded).hexdigest():
        raise ValueError("Workflow source provenance digest is invalid")
    return {
        "project_root": project_root,
        "manifest_sha256": claimed,
        "source_manifest": dict(manifest),
    }


def capture_workflow_source_provenance() -> dict[str, Any]:
    """Capture and validate the source manifest before a direct run timer."""
    return _validate_source_provenance(_source_provenance())


def _runtime_provenance() -> dict[str, Any]:
    module_versions: dict[str, Any] = {}
    for name in ("numpy", "torch", "crypten"):
        try:
            module = __import__(name)
            module_versions[name] = {
                "version": str(getattr(module, "__version__", "unknown")),
                "path": str(Path(module.__file__).resolve())
                if getattr(module, "__file__", None)
                else "",
            }
        except Exception as exc:
            module_versions[name] = {"unavailable": f"{type(exc).__name__}: {exc}"}
    configuration_defaults = {
        "PVC_BATCH_ARGMAX": "1",
        "PVC_BATCH_REVEAL": "1",
        "PVC_BATCH_EMISSION": "0",
        "PVC_COV_MAX_MULT": "10",
        "PVC_COUNT_MAX_MULT": "2",
        "PVC_PRECISION_BITS": "0",
        "PVC_TIEBREAK": "tier-path default (current/first)",
        "PVC_GENOTYPE_DEVICE": "cpu",
        "PVC_WRITE_DIAGNOSTIC_OUTPUTS": "1",
        "PVC_PROGRESS_LOG": "1",
        "PVC_WORKFLOW_PROFILE": "1",
        "PVC_EXACT_COMM_TRACE": "1",
        "PVC_HEAVY_LOOKUP_METHOD": "arithmetic-coverage-binary-count",
        "PVC_OT_PAIRWISE_PRF_PADS": "0",
        "PVC_PROFILE": "0",
        "PVC_PROFILE_COMM": "0",
    }
    try:
        affinity_cpus = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
        affinity_error = None
    except (AttributeError, OSError) as exc:
        affinity_cpus = []
        affinity_error = f"{type(exc).__name__}: {exc}"
    try:
        import torch

        torch_num_threads = int(torch.get_num_threads())
        torch_num_interop_threads = int(torch.get_num_interop_threads())
        torch_thread_error = None
    except Exception as exc:
        torch_num_threads = None
        torch_num_interop_threads = None
        torch_thread_error = f"{type(exc).__name__}: {exc}"
    return {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "cpu_model": _linux_cpu_model(),
        "pid": os.getpid(),
        "modules": module_versions,
        "protocol_configuration": {
            key: {
                "raw": os.environ.get(key),
                "effective": os.environ.get(key, default),
            }
            for key, default in configuration_defaults.items()
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "slurm": {
            key: os.environ.get(key, "")
            for key in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_JOB_PARTITION",
                "SLURM_NODELIST",
                "SLURM_CPUS_PER_TASK",
                "SLURM_CPUS_ON_NODE",
                "SLURM_JOB_CPUS_PER_NODE",
                "SLURM_MEM_PER_NODE",
                "SLURM_NTASKS",
            )
        },
        "thread_environment": {
            key: os.environ.get(key, "")
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "thread_runtime": {
            "os_cpu_count": os.cpu_count(),
            "sched_getaffinity_cpu_count": len(affinity_cpus),
            "sched_getaffinity_cpus": affinity_cpus,
            "sched_getaffinity_error": affinity_error,
            "torch_num_threads": torch_num_threads,
            "torch_num_interop_threads": torch_num_interop_threads,
            "torch_thread_error": torch_thread_error,
        },
    }


def _linux_cpu_model() -> dict[str, Any]:
    """Describe the Linux CPU model using a stable, command-free source."""
    path = Path("/proc/cpuinfo")
    try:
        models: list[str] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in {"model name", "Processor"}:
                model = value.strip()
                if model and model not in models:
                    models.append(model)
        return {
            "source": str(path),
            "model_names": models,
            "architecture": platform.machine(),
            "error": None if models else "No model name or Processor field was present",
        }
    except OSError as exc:
        return {
            "source": str(path),
            "model_names": [],
            "architecture": platform.machine(),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _record(name: str) -> dict[str, Any]:
    if name not in PHASE_ORDER:
        raise ValueError(f"Unknown canonical PVC profile phase: {name}")
    if not _tool:
        raise RuntimeError("reset_workflow_profile() must be called before profiling")
    return _records.setdefault(name, _empty_record(name, _tool))


def _set_dependency_trace_phase(name: str | None) -> None:
    """Assign a canonical phase to subsequently issued CrypTen messages."""
    try:
        import crypten
    except Exception:
        return
    is_initialized = getattr(crypten, "is_initialized", None)
    if not callable(is_initialized) or not bool(is_initialized()):
        return
    communicator = crypten.communicator.get()
    setter = getattr(communicator, "set_dependency_trace_phase", None)
    if callable(setter):
        setter(name)


def _dependency_trace_snapshot() -> dict[str, Any] | None:
    """Materialize the rank trace only after the measured VCF workflow."""
    try:
        import crypten
    except Exception:
        return None
    is_initialized = getattr(crypten, "is_initialized", None)
    if not callable(is_initialized) or not bool(is_initialized()):
        return None
    communicator = crypten.communicator.get()
    snapshot = getattr(communicator, "dependency_trace_snapshot", None)
    return snapshot() if callable(snapshot) else None


def _comm_snapshot() -> tuple[int, int, int]:
    """Read required counters, failing closed for an initialized communicator."""
    try:
        import crypten
    except Exception:
        # Standalone kernels intentionally remain usable without CrypTen.
        return 0, 0, 0
    is_initialized = getattr(crypten, "is_initialized", None)
    if not callable(is_initialized):
        return 0, 0, 0
    try:
        initialized = bool(is_initialized())
    except Exception as exc:
        raise RuntimeError(
            "Could not determine whether CrypTen communication counters are active"
        ) from exc
    if not initialized:
        return 0, 0, 0
    try:
        comm = crypten.communicator.get()
    except Exception as exc:
        raise RuntimeError(
            "Initialized CrypTen communicator is unavailable to the workflow profiler"
        ) from exc
    values: list[int] = []
    missing: list[str] = []
    invalid: list[str] = []
    for name in _REQUIRED_COMM_COUNTERS:
        try:
            value = getattr(comm, name)
        except Exception:
            missing.append(name)
            continue
        if not isinstance(value, numbers.Integral) or isinstance(value, bool):
            invalid.append(f"{name}={value!r}")
            continue
        if int(value) < 0:
            invalid.append(f"{name}={value!r}")
            continue
        values.append(int(value))
    if missing or invalid:
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if invalid:
            detail.append(f"invalid {invalid}")
        raise RuntimeError(
            "Initialized CrypTen communicator lacks required workflow-profile "
            f"counters ({'; '.join(detail)})"
        )
    return values[0], values[1], values[2]


def _comm_event_snapshot() -> dict[str, dict[str, int]]:
    """Read auditable per-primitive counters from an initialized communicator."""
    try:
        import crypten
    except Exception:
        return {}
    is_initialized = getattr(crypten, "is_initialized", None)
    if not callable(is_initialized) or not bool(is_initialized()):
        return {}
    try:
        communicator = crypten.communicator.get()
    except Exception as exc:
        raise RuntimeError(
            "Initialized CrypTen communicator is unavailable to the workflow profiler"
        ) from exc
    raw_maps: dict[str, dict[Any, Any]] = {}
    for attribute in _REQUIRED_COMM_EVENT_COUNTERS:
        try:
            value = getattr(communicator, attribute)
        except Exception as exc:
            raise RuntimeError(
                "Initialized CrypTen communicator lacks required workflow-profile "
                f"event counter {attribute}"
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError(
                f"CrypTen workflow-profile event counter {attribute} is not a dictionary"
            )
        raw_maps[attribute] = value
    result: dict[str, dict[str, int]] = {}
    names = sorted(set().union(*(set(values) for values in raw_maps.values())))
    for raw_name in names:
        name = str(raw_name)
        values: dict[str, int] = {}
        for attribute, field in (
            ("comm_event_bytes", "communicator_endpoint_payload_bytes"),
            ("comm_event_rounds", "communicator_operations"),
            ("comm_event_logical_exchanges", "logical_exchange_calls"),
        ):
            value = raw_maps[attribute].get(raw_name, 0)
            if (
                not isinstance(value, numbers.Integral)
                or isinstance(value, bool)
                or int(value) < 0
            ):
                raise RuntimeError(
                    f"CrypTen event counter {attribute}[{raw_name!r}] is invalid: "
                    f"{value!r}"
                )
            values[field] = int(value)
        result[name] = values
    return result


def _communication_accounting_snapshot() -> dict[str, Any]:
    """Capture one read-only snapshot of every required communicator counter."""
    payload_bytes, operations, exchanges = _comm_snapshot()
    return {
        "communicator_endpoint_payload_bytes": payload_bytes,
        "communicator_operations": operations,
        "logical_exchange_calls": exchanges,
        "communicator_events": _comm_event_snapshot(),
    }


def _event_delta(
    before: dict[str, dict[str, int]], after: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for event in sorted(set(before) | set(after)):
        values = {}
        for field in _RANK_EVENT_COUNTER_FIELDS:
            start = int(before.get(event, {}).get(field, 0))
            end = int(after.get(event, {}).get(field, 0))
            if end < start:
                raise RuntimeError(
                    f"CrypTen event counter {event!r}/{field} decreased "
                    f"during profiling: {start} -> {end}"
                )
            values[field] = end - start
        if any(values.values()):
            result[event] = values
    return result


def _merge_event_delta(
    target: dict[str, dict[str, int]], delta: dict[str, dict[str, int]]
) -> None:
    for event, values in delta.items():
        target_values = target.setdefault(
            event,
            {
                "communicator_endpoint_payload_bytes": 0,
                "communicator_operations": 0,
                "logical_exchange_calls": 0,
            },
        )
        for field, value in values.items():
            target_values[field] += int(value)


def _validate_communication_accounting_snapshot(value: Any) -> dict[str, Any]:
    """Validate a persisted whole-run communicator snapshot."""
    if not isinstance(value, dict):
        raise ValueError("Communication accounting snapshot is not an object")
    expected = {*_RANK_EVENT_COUNTER_FIELDS, "communicator_events"}
    if set(value) != expected:
        raise ValueError(
            "Communication accounting snapshot has an incompatible schema"
        )
    validated: dict[str, Any] = {}
    for field in _RANK_EVENT_COUNTER_FIELDS:
        counter = value[field]
        if (
            isinstance(counter, bool)
            or not isinstance(counter, numbers.Integral)
            or int(counter) < 0
        ):
            raise ValueError(
                f"Communication accounting snapshot has invalid {field}: "
                f"{counter!r}"
            )
        validated[field] = int(counter)
    events = value["communicator_events"]
    if not isinstance(events, dict):
        raise ValueError("Communication accounting event snapshot is not an object")
    validated_events: dict[str, dict[str, int]] = {}
    for event, counters in events.items():
        if not isinstance(event, str) or not isinstance(counters, dict):
            raise ValueError("Communication accounting snapshot has a malformed event")
        if set(counters) != set(_RANK_EVENT_COUNTER_FIELDS):
            raise ValueError(
                f"Communication accounting event {event!r} has an incompatible schema"
            )
        validated_counters: dict[str, int] = {}
        for field in _RANK_EVENT_COUNTER_FIELDS:
            counter = counters[field]
            if (
                isinstance(counter, bool)
                or not isinstance(counter, numbers.Integral)
                or int(counter) < 0
            ):
                raise ValueError(
                    f"Communication accounting event {event!r} has invalid "
                    f"{field}: {counter!r}"
                )
            validated_counters[field] = int(counter)
        validated_events[event] = validated_counters
    validated["communicator_events"] = validated_events
    return validated


def _communication_snapshot_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    """Compute a strict whole-run delta, rejecting a counter reset or decrease."""
    start = _validate_communication_accounting_snapshot(before)
    end = _validate_communication_accounting_snapshot(after)
    result: dict[str, Any] = {}
    for field in _RANK_EVENT_COUNTER_FIELDS:
        if end[field] < start[field]:
            raise ValueError(
                f"CrypTen whole-run counter {field} decreased during profiling: "
                f"{start[field]} -> {end[field]}"
            )
        result[field] = end[field] - start[field]
    try:
        result["communicator_events"] = _event_delta(
            start["communicator_events"], end["communicator_events"]
        )
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    return result


def _sum_phase_communication(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum all canonical genotype phase counters for one rank exactly."""
    result: dict[str, Any] = {
        field: sum(int(row[field]) for row in rows)
        for field in _RANK_EVENT_COUNTER_FIELDS
    }
    events: dict[str, dict[str, int]] = {}
    for row in rows:
        _merge_event_delta(events, row["communicator_events"])
    result["communicator_events"] = events
    return result


def _event_counter_totals(events: dict[str, dict[str, int]]) -> dict[str, int]:
    return {
        field: sum(int(counters[field]) for counters in events.values())
        for field in _RANK_EVENT_COUNTER_FIELDS
    }


def _signed_communication_difference(
    observed: dict[str, Any], attributed: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        field: int(observed[field]) - int(attributed[field])
        for field in _RANK_EVENT_COUNTER_FIELDS
    }
    events: dict[str, dict[str, int]] = {}
    observed_events = observed["communicator_events"]
    attributed_events = attributed["communicator_events"]
    for event in sorted(set(observed_events) | set(attributed_events)):
        counters = {
            field: int(observed_events.get(event, {}).get(field, 0))
            - int(attributed_events.get(event, {}).get(field, 0))
            for field in _RANK_EVENT_COUNTER_FIELDS
        }
        if any(counters.values()):
            events[event] = counters
    result["communicator_events"] = events
    return result


def _build_rank_communication_reconciliation(
    rows: list[dict[str, Any]],
    start_snapshot: dict[str, Any] | None,
    end_snapshot: dict[str, Any] | None,
    *,
    start_error: str | None = None,
    end_error: str | None = None,
) -> dict[str, Any]:
    """Prove that every communicator delta belongs to one canonical phase."""
    phase_sum = _sum_phase_communication(rows)
    base: dict[str, Any] = {
        "schema": "pvc-rank-communication-reconciliation-v1",
        "measurement_window": {
            "start": (
                "read-only counter snapshot in reset_workflow_profile, called "
                "immediately after the production CrypTen counter reset"
            ),
            "end": "read-only counter snapshot at entry to write_rank_profile",
            "canonical_scope": "all canonical genotype phases; native index phases excluded",
            "counter_snapshot_overhead_generates_communication": False,
        },
        "start_snapshot": start_snapshot,
        "end_snapshot": end_snapshot,
        "whole_run_delta": None,
        "canonical_phase_sum": phase_sum,
        "unattributed_or_overlapping_delta": None,
        "whole_run_event_totals_match_counters": None,
        "canonical_phase_event_totals_match_counters": (
            _event_counter_totals(phase_sum["communicator_events"])
            == {field: phase_sum[field] for field in _RANK_EVENT_COUNTER_FIELDS}
        ),
        "exact_match": False,
        "status": "unavailable",
        "error": start_error or end_error,
        "communication_time": {
            "measurement_status": "not_measured",
            "seconds": None,
            "legacy_comm_time_counter_used": False,
            "definition": (
                "CrypTen's legacy comm_time remains zero in this instrumented "
                "communicator; no communication-time measurement is claimed"
            ),
        },
    }
    if (
        start_error is not None
        or end_error is not None
        or start_snapshot is None
        or end_snapshot is None
    ):
        return base
    try:
        whole_delta = _communication_snapshot_delta(
            start_snapshot, end_snapshot
        )
    except ValueError as exc:
        base["status"] = "invalid_counter_delta"
        base["error"] = str(exc)
        return base
    whole_event_match = _event_counter_totals(
        whole_delta["communicator_events"]
    ) == {field: whole_delta[field] for field in _RANK_EVENT_COUNTER_FIELDS}
    phase_event_match = bool(base["canonical_phase_event_totals_match_counters"])
    exact = whole_delta == phase_sum and whole_event_match and phase_event_match
    base.update(
        {
            "whole_run_delta": whole_delta,
            "unattributed_or_overlapping_delta": _signed_communication_difference(
                whole_delta, phase_sum
            ),
            "whole_run_event_totals_match_counters": whole_event_match,
            "exact_match": exact,
            "status": "reconciled" if exact else "mismatch",
            "error": (
                None
                if exact
                else (
                    "Whole-run communicator deltas do not exactly equal the "
                    "sum of canonical genotype phase deltas"
                )
            ),
        }
    )
    return base


def current_rank() -> int:
    try:
        import crypten

        if crypten.is_initialized():
            return int(crypten.communicator.get().get_rank())
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def current_world_size() -> int:
    try:
        import crypten

        if crypten.is_initialized():
            return int(crypten.communicator.get().get_world_size())
    except Exception:
        pass
    return int(os.environ.get("WORLD_SIZE", "1"))


@contextmanager
def profile_phase(name: str) -> Iterator[None]:
    """Measure one non-overlapping canonical phase in the current process."""
    # Scoring kernels are also used by standalone validators and microbenchmarks
    # that intentionally do not run the workflow wrapper.  Profiling must stay
    # transparent in those callers rather than requiring synthetic run state.
    if not PROFILE_ENABLED or not _tool:
        yield
        return
    if _active_canonical:
        raise RuntimeError(
            f"Canonical PVC profile phases may not overlap: {name!r} entered "
            f"inside {_active_canonical[-1]!r}"
        )
    record = _record(name)
    # Snapshot before mutating the active stack so a missing required counter
    # cannot leave profile state corrupted.
    bytes0, operations0, exchanges0 = _comm_snapshot()
    events0 = _comm_event_snapshot()
    _set_dependency_trace_phase(name)
    _active_canonical.append(name)
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    try:
        yield
    finally:
        try:
            # Stop the phase clocks before reading the profiler's own counter
            # dictionaries.  Counter-snapshot overhead is instrumentation, not
            # workflow work, and must not be charged once per phase call.
            wall1 = time.perf_counter()
            cpu1 = time.process_time()
            _set_dependency_trace_phase(None)
            bytes1, operations1, exchanges1 = _comm_snapshot()
            events1 = _comm_event_snapshot()
            record["wall_seconds"] += wall1 - wall0
            record["cpu_seconds"] += cpu1 - cpu0
            record["calls"] += 1
            for field, start, end in (
                ("communicator_endpoint_payload_bytes", bytes0, bytes1),
                ("communicator_operations", operations0, operations1),
                ("logical_exchange_calls", exchanges0, exchanges1),
            ):
                if end < start:
                    raise RuntimeError(
                        f"CrypTen counter {field} decreased within canonical "
                        f"phase {name!r}: {start} -> {end}"
                    )
                record[field] += end - start
            _merge_event_delta(
                record["communicator_events"], _event_delta(events0, events1)
            )
        finally:
            _set_dependency_trace_phase(None)
            popped = _active_canonical.pop()
            if popped != name:
                raise RuntimeError("PVC canonical profile stack was corrupted")


def account_client_transfer(
    phase: str,
    direction: str,
    nbytes: int,
    *,
    rounds: int = 1,
    model: str,
) -> None:
    """Add a modeled external-client transfer to a named protocol phase.

    Only rank 0 records logical client traffic; otherwise the same modeled
    transfer would be duplicated in every party file.
    """
    if not PROFILE_ENABLED or not _tool or current_rank() != 0:
        return
    if direction not in {"upload", "download"}:
        raise ValueError(f"Unsupported client transfer direction: {direction}")
    if nbytes < 0 or rounds < 0:
        raise ValueError("Client transfer bytes and rounds must be nonnegative")
    if not isinstance(model, str) or not model:
        raise ValueError("Client transfer model must be a non-empty string")
    global _client_transfer_ordinal
    record = _record(phase)
    record[f"client_{direction}_bytes"] += int(nbytes)
    record[f"client_{direction}_rounds"] += int(rounds)
    if model not in record["client_transfer_model"]:
        record["client_transfer_model"].append(model)
    record["client_transfer_events"].append(
        {
            "ordinal": _client_transfer_ordinal,
            "phase": phase,
            "direction": direction,
            "payload_bytes": int(nbytes),
            "protocol_rounds": int(rounds),
            "model": model,
        }
    )
    _client_transfer_ordinal += 1


def mark_fastq_to_vcf_complete(output_vcf: str | Path | None = None) -> None:
    """Capture the direct boundary and bind the successfully written VCF."""
    global _fastq_to_vcf_wall, _fastq_to_vcf_cpu, _output_vcf_artifact
    if not PROFILE_ENABLED or not _started_wall:
        return
    if current_rank() != 0:
        raise RuntimeError("Only rank 0 may mark the FASTQ-to-VCF boundary complete")
    # Capture the inclusive write boundary before hashing the output artifact;
    # provenance materialization is not part of FASTQ-to-VCF runtime.
    _fastq_to_vcf_wall = time.perf_counter() - _started_wall
    _fastq_to_vcf_cpu = time.process_time() - _started_cpu
    if output_vcf is None:
        _output_vcf_artifact = None
        return
    path = Path(output_vcf).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Successful VCF boundary output is missing: {path}")
    stat = path.stat()
    _output_vcf_artifact = {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": int(stat.st_size),
        "role": "rank0_fastq_to_vcf_output",
        "bound_immediately_after_write": True,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "phase",
        "order",
        "actor",
        "scope",
        "included_fastq_to_vcf",
        "required_for_vcf",
        "measurement_status",
        "wall_seconds",
        "cpu_seconds",
        "cpu_seconds_model",
        "calls",
        "communicator_endpoint_payload_bytes",
        "communicator_endpoint_payload_bytes_max",
        "communicator_endpoint_payload_bytes_sum",
        "communicator_payload_bytes",
        "communicator_operations",
        "logical_exchange_calls",
        "communicator_events",
        "modeled_computing_party_payload_bytes",
        "modeled_client_reconstruction_payload_bytes",
        "observed_computing_party_exchange_calls",
        "observed_client_reconstruction_exchange_calls",
        "classified_computing_party_payload_bytes",
        "classified_client_payload_bytes",
        "classified_computing_party_exchange_calls",
        "classified_client_exchange_calls",
        "client_upload_bytes",
        "client_upload_rounds",
        "client_download_bytes",
        "client_download_rounds",
        "client_transfer_model",
        "client_transfer_events",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["included_fastq_to_vcf"] = int(bool(row["included_fastq_to_vcf"]))
            csv_row["required_for_vcf"] = int(bool(row["required_for_vcf"]))
            csv_row["client_transfer_model"] = "; ".join(row["client_transfer_model"])
            csv_row["client_transfer_events"] = json.dumps(
                row["client_transfer_events"], sort_keys=True, separators=(",", ":")
            )
            csv_row["communicator_events"] = json.dumps(
                row["communicator_events"], sort_keys=True, separators=(",", ":")
            )
            writer.writerow({field: csv_row[field] for field in fields})
    temporary.replace(path)


def write_rank_profile(
    genotype_dir: str | Path,
    *,
    status: str,
    error: str | None = None,
) -> tuple[Path, Path] | None:
    """Write a stable JSON/CSV phase table for the current party."""
    if not PROFILE_ENABLED:
        return None
    # Close the whole-run communication window before provenance hashing or
    # profile serialization.  These read-only snapshots do not call any
    # communicator primitive and therefore cannot add communication counters.
    communication_run_end: dict[str, Any] | None = None
    communication_run_end_error: str | None = None
    try:
        communication_run_end = _communication_accounting_snapshot()
    except Exception as exc:
        communication_run_end_error = f"{type(exc).__name__}: {exc}"
    # Capture process high-water memory before provenance hashing and profile
    # serialization can perturb it. This is one O(1) procfs/getrusage read.
    try:
        process_memory = _capture_process_memory(_process_identity_start)
    except Exception as exc:
        process_memory = {
            "schema": PROCESS_MEMORY_SCHEMA,
            "schema_version": 1,
            "measurement_status": "unavailable",
            "process_identity": _process_identity_start,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if status == "complete":
            status = "invalid_process_memory"
            error = str(process_memory["error"])
    # Materialize the compact event arrays only after both the VCF boundary and
    # native process-memory observation.  This keeps JSON construction out of
    # the reported runtime and peak-memory domains.
    try:
        application_dependency_trace = _dependency_trace_snapshot()
    except Exception as exc:
        application_dependency_trace = None
        if status == "complete":
            status = "invalid_application_dependency_trace"
            error = f"{type(exc).__name__}: {exc}"
    source_end = _source_provenance()
    source_valid = (
        bool(_source_start)
        and _source_start.get("manifest_sha256") == source_end.get("manifest_sha256")
    )
    if not source_valid and status == "complete":
        status = "invalid_source_drift"
        error = (
            "source manifest changed during run: "
            f"{_source_start.get('manifest_sha256')} -> "
            f"{source_end.get('manifest_sha256')}"
        )
    world_size = current_world_size()
    implementation = _communication_counter_capability.get("implementation")
    communication_capability_valid = (
        _communication_counter_capability.get("crypten_initialized") is True
        and _communication_counter_capability.get("complete") is True
        and isinstance(implementation, dict)
        and bool(implementation)
    ) if world_size == 3 else not (
        _communication_counter_capability.get("crypten_initialized") is True
        and _communication_counter_capability.get("complete") is not True
    )
    if not communication_capability_valid and status == "complete":
        status = "invalid_communication_counter_capability"
        missing = _communication_counter_capability.get("missing_counters", [])
        error = (
            "three-party PVC profiling requires an initialized communicator, "
            "all required counters, and a bound implementation identity; "
            f"status={_communication_counter_capability.get('status')!r}, "
            f"missing={missing}"
        )
    trace_capability = _communication_counter_capability.get(
        "dependency_trace", {}
    )
    trace_valid = bool(
        isinstance(application_dependency_trace, dict)
        and application_dependency_trace.get("schema") == TRACE_SCHEMA
        and application_dependency_trace.get("enabled") is True
        and application_dependency_trace.get("complete") is True
        and application_dependency_trace.get("errors") == []
        and isinstance(trace_capability, dict)
        and trace_capability.get("complete") is True
    )
    if world_size == 3 and not trace_valid and status == "complete":
        status = "invalid_application_dependency_trace"
        error = (
            "three-party PVC profiling requires a complete exact application "
            "dependency trace on every rank"
        )
    try:
        bound_artifacts = _validate_genotype_input_artifacts(
            _genotype_input_artifacts
        )
    except ValueError as exc:
        bound_artifacts = _genotype_input_artifacts
        if status == "complete":
            status = "invalid_genotype_input_artifacts"
            error = str(exc)
    try:
        effective_configuration = _validate_effective_configuration(
            _effective_configuration
        )
    except ValueError as exc:
        effective_configuration = _effective_configuration
        if status == "complete":
            status = "invalid_effective_configuration"
            error = str(exc)
    effective_values = (
        effective_configuration.get("values", {})
        if isinstance(effective_configuration, dict)
        else {}
    )
    diagnostic_values = (
        effective_values.get("diagnostic_outputs", {})
        if isinstance(effective_values, dict)
        else {}
    )
    diagnostic_outputs_enabled = bool(
        isinstance(diagnostic_values, dict)
        and diagnostic_values.get("enabled") is True
    )
    directory = Path(genotype_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rank = current_rank()
    rows = []
    for name in PHASE_ORDER:
        if name in _INDEX_PHASES:
            continue
        record = dict(_records.get(name, _empty_record(name, _tool)))
        if record["calls"] == 0 and not any(
            record[field]
            for field in (
                "client_upload_bytes",
                "client_download_bytes",
                "communicator_endpoint_payload_bytes",
                "communicator_operations",
                "logical_exchange_calls",
            )
        ):
            record["measurement_status"] = "zero_calls"
        record["wall_seconds"] = round(float(record["wall_seconds"]), 9)
        record["cpu_seconds"] = round(float(record["cpu_seconds"]), 9)
        rows.append(record)
    communication_reconciliation = _build_rank_communication_reconciliation(
        rows,
        _communication_run_start,
        communication_run_end,
        start_error=_communication_run_start_error,
        end_error=communication_run_end_error,
    )
    if not communication_reconciliation["exact_match"] and status == "complete":
        status = "invalid_communication_reconciliation"
        error = str(communication_reconciliation["error"])
    payload = {
        "schema": SCHEMA,
        "schema_version": 3,
        "execution_id": _execution_id,
        "input_binding": _input_binding,
        "genotype_input_artifacts": bound_artifacts,
        "effective_configuration": effective_configuration,
        "tool": _tool,
        "rank": rank,
        "world_size": world_size,
        "status": status,
        "error": error,
        "profile_started_utc": _started_utc,
        "profile_written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_provenance": {
            **_source_start,
            "end_manifest_sha256": source_end.get("manifest_sha256"),
            "unchanged_during_run": source_valid,
        },
        "runtime_provenance": _runtime_start,
        "process_memory": process_memory,
        "communication_counter_capability": _communication_counter_capability,
        "communication_reconciliation": communication_reconciliation,
        "application_dependency_trace": application_dependency_trace,
        "timing_semantics": {
            "wall_seconds": "time.perf_counter delta in this rank",
            "cpu_seconds": "time.process_time delta in this rank",
            "canonical_phases_non_overlapping": True,
            "counter_snapshot_overhead_excluded": True,
        },
        "communication_semantics": {
            "communicator_endpoint_payload_bytes": (
                "CrypTen-modeled serialized application tensor-payload delta "
                "at this rank endpoint; excludes transport overhead and is not "
                "a physical wire-byte measurement"
            ),
            "communicator_operations": (
                "CrypTen comm_rounds delta; a low-level operation proxy, not "
                "protocol dependency-round depth"
            ),
            "logical_exchange_calls": (
                "top-level collective or batched point-to-point invocation delta"
            ),
            "communicator_events": (
                "per-phase deltas of CrypTen event dictionaries, retaining "
                "primitive/event labels for payload, legacy-operation, and "
                "logical-exchange auditing; not MPC dependency-round depth"
            ),
            "application_dependency_trace": (
                "complete CrypTen-API application-message issue/completion trace; "
                "serialized after the VCF and process-memory boundaries and later "
                "matched across all three ranks to derive causal rounds"
            ),
            "whole_run_reconciliation": (
                "exact equality is required between the reset-to-write whole-run "
                "counter delta and the sum of all canonical genotype phase deltas, "
                "including exact per-event dictionaries"
            ),
            "communication_time": (
                "not measured; the legacy CrypTen comm_time field remains zero "
                "and is not used or reported as elapsed communication time"
            ),
            "client_upload": (
                "modeled external-client protocol transfers, recorded on rank 0 only"
            ),
            "reveal_reconstruction": (
                "CrypTen communicator payload reclassified as the protocol's "
                "logical-client reconstruction model; not independently "
                "observed external-client traffic"
            ),
        },
        "boundary": {
            "name": "FASTQ-to-VCF",
            "inclusive_end": "successful return from write_output_vcf",
            "observed_elapsed_includes": (
                ["private_score_artifact_output", "likelihood_output"]
                if diagnostic_outputs_enabled
                else []
            ),
            "observed_elapsed_excludes_post_vcf": [
                "truth_verification",
                "completion_marker",
            ],
            "genotype_wall_seconds": _fastq_to_vcf_wall,
            "genotype_cpu_seconds": _fastq_to_vcf_cpu,
            "output_vcf_artifact": _output_vcf_artifact,
        },
        "process_total": {
            "wall_seconds": time.perf_counter() - _started_wall,
            "cpu_seconds": time.process_time() - _started_cpu,
        },
        "phases": rows,
    }
    json_path = directory / f"workflow_profile_rank_{rank}.json"
    csv_path = directory / f"workflow_profile_rank_{rank}.csv"
    _atomic_json(json_path, payload)
    _atomic_csv(csv_path, rows)
    return json_path, csv_path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


_RANK_EVENT_COUNTER_FIELDS = (
    "communicator_endpoint_payload_bytes",
    "communicator_operations",
    "logical_exchange_calls",
)
_RANK_NONNEGATIVE_INT_FIELDS = (
    "calls",
    *_RANK_EVENT_COUNTER_FIELDS,
    "client_upload_bytes",
    "client_upload_rounds",
    "client_download_bytes",
    "client_download_rounds",
)


def _validate_rank_phase_row(row: Any, tool: str) -> dict[str, Any]:
    if not isinstance(row, dict) or row.get("phase") not in PHASE_ORDER:
        raise ValueError("Rank profile contains a malformed phase row")
    name = str(row["phase"])
    if name in _INDEX_PHASES:
        raise ValueError(f"Rank profile must not embed native index phase {name}")
    expected_metadata = {
        "order": PHASE_ORDER.index(name),
        "actor": phase_actor(name, tool),
        "scope": phase_scope(name, tool),
        "included_fastq_to_vcf": phase_in_fastq_to_vcf(name),
        "required_for_vcf": phase_required_for_vcf(name),
    }
    for field, expected in expected_metadata.items():
        if row.get(field) != expected:
            raise ValueError(
                f"Rank phase {name} has incompatible {field}: "
                f"{row.get(field)!r} != {expected!r}"
            )
    for field in ("wall_seconds", "cpu_seconds"):
        value = row.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"Rank phase {name} has invalid {field}: {value!r}")
    for field in _RANK_NONNEGATIVE_INT_FIELDS:
        value = row.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Integral)
            or int(value) < 0
        ):
            raise ValueError(f"Rank phase {name} has invalid {field}: {value!r}")
    models = row.get("client_transfer_model")
    if not isinstance(models, list) or not all(
        isinstance(value, str) for value in models
    ):
        raise ValueError(f"Rank phase {name} has malformed client transfer models")
    client_events = row.get("client_transfer_events")
    if not isinstance(client_events, list):
        raise ValueError(f"Rank phase {name} has malformed client transfer events")
    for client_event in client_events:
        if not isinstance(client_event, dict) or set(client_event) != {
            "ordinal",
            "phase",
            "direction",
            "payload_bytes",
            "protocol_rounds",
            "model",
        }:
            raise ValueError(f"Rank phase {name} has malformed client transfer event")
        if (
            client_event.get("phase") != name
            or client_event.get("direction") not in {"upload", "download"}
            or isinstance(client_event.get("ordinal"), bool)
            or not isinstance(client_event.get("ordinal"), numbers.Integral)
            or int(client_event["ordinal"]) < 0
            or isinstance(client_event.get("payload_bytes"), bool)
            or not isinstance(client_event.get("payload_bytes"), numbers.Integral)
            or int(client_event["payload_bytes"]) < 0
            or isinstance(client_event.get("protocol_rounds"), bool)
            or not isinstance(client_event.get("protocol_rounds"), numbers.Integral)
            or int(client_event["protocol_rounds"]) < 0
            or not isinstance(client_event.get("model"), str)
            or not client_event["model"]
        ):
            raise ValueError(f"Rank phase {name} has inconsistent client transfer event")
    events = row.get("communicator_events")
    if not isinstance(events, dict):
        raise ValueError(f"Rank phase {name} has malformed communicator events")
    event_totals = {field: 0 for field in _RANK_EVENT_COUNTER_FIELDS}
    for event_name, counters in events.items():
        if not isinstance(event_name, str) or not isinstance(counters, dict):
            raise ValueError(f"Rank phase {name} has a malformed communicator event")
        if set(counters) != set(_RANK_EVENT_COUNTER_FIELDS):
            raise ValueError(
                f"Rank phase {name} event {event_name!r} has an incompatible schema"
            )
        for field in _RANK_EVENT_COUNTER_FIELDS:
            value = counters[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or int(value) < 0
            ):
                raise ValueError(
                    f"Rank phase {name} event {event_name!r} has invalid "
                    f"{field}: {value!r}"
                )
            event_totals[field] += int(value)
    for field, total in event_totals.items():
        if total != int(row[field]):
            raise ValueError(
                f"Rank phase {name} event {field} total {total} does not "
                f"match phase total {row[field]}"
            )
    return row


def _rank_phase_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    phases = payload.get("phases")
    if not isinstance(phases, list):
        raise ValueError("Rank profile lacks a phase list")
    tool = payload.get("tool")
    if not isinstance(tool, str):
        raise ValueError("Rank profile lacks its tool identity")
    result: dict[str, dict[str, Any]] = {}
    for row in phases:
        validated = _validate_rank_phase_row(row, tool)
        name = str(validated["phase"])
        if name in result:
            raise ValueError(f"Rank profile contains duplicate phase {name}")
        result[name] = validated
    expected = set(PHASE_ORDER) - _INDEX_PHASES
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise ValueError(
            f"Rank profile phase set is incomplete or incompatible; "
            f"missing={missing}, extra={extra}"
        )
    return result


def _validate_rank_communication_reconciliation(
    payload: dict[str, Any], phase_map: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Re-derive a rank's fail-closed whole-run communication proof."""
    claimed = payload.get("communication_reconciliation")
    if not isinstance(claimed, dict):
        raise ValueError("Rank profile lacks whole-run communication reconciliation")
    rows = [
        phase_map[name]
        for name in PHASE_ORDER
        if name not in _INDEX_PHASES
    ]
    expected = _build_rank_communication_reconciliation(
        rows,
        claimed.get("start_snapshot"),
        claimed.get("end_snapshot"),
    )
    if claimed != expected:
        raise ValueError(
            "Rank profile whole-run communication reconciliation is invalid"
        )
    if claimed.get("status") != "reconciled" or claimed.get("exact_match") is not True:
        raise ValueError(
            "Rank profile does not prove exact canonical-phase communication attribution"
        )
    return claimed


def _aggregate_rank_communication_reconciliation(
    reconciliations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Preserve rank proofs and summarize their whole-run endpoint deltas."""
    if not reconciliations:
        raise ValueError("No rank communication reconciliations were supplied")
    whole_run_values = [value["whole_run_delta"] for value in reconciliations]
    endpoint_values = [
        int(value["communicator_endpoint_payload_bytes"])
        for value in whole_run_values
    ]
    endpoint_sum = sum(endpoint_values)
    if endpoint_sum % 2:
        raise ValueError(
            "Whole-run reconciliation has an odd endpoint-payload-byte sum; "
            "cannot apply the half-sum accounting convention"
        )
    event_values = [
        {"communicator_events": value["communicator_events"]}
        for value in whole_run_values
    ]
    return {
        "schema": "pvc-aggregate-communication-reconciliation-v1",
        "status": "reconciled",
        "all_ranks_exact_match": True,
        "rank_count": len(reconciliations),
        "communicator_endpoint_payload_bytes_max": max(endpoint_values),
        "communicator_endpoint_payload_bytes_sum": endpoint_sum,
        "communicator_payload_bytes": endpoint_sum // 2,
        "communicator_operations_max": max(
            int(value["communicator_operations"]) for value in whole_run_values
        ),
        "logical_exchange_calls_max": max(
            int(value["logical_exchange_calls"]) for value in whole_run_values
        ),
        "communicator_events": _aggregate_communicator_events(event_values),
        "rank_values": [
            {
                "rank": rank,
                "start_snapshot": value["start_snapshot"],
                "end_snapshot": value["end_snapshot"],
                "whole_run_delta": value["whole_run_delta"],
                "canonical_phase_sum": value["canonical_phase_sum"],
                "unattributed_or_overlapping_delta": value[
                    "unattributed_or_overlapping_delta"
                ],
                "exact_match": value["exact_match"],
            }
            for rank, value in enumerate(reconciliations)
        ],
        "aggregation_note": (
            "Exact reconciliation is established independently per rank. "
            "Payload uses the same half-sum endpoint accounting convention; "
            "operation and logical-exchange summaries use the maximum rank."
        ),
        "communication_time": reconciliations[0]["communication_time"],
    }


_TOTAL_VARIANTS = {
    "required_for_vcf": lambda row: bool(row["required_for_vcf"]),
    "included_fastq_to_vcf": lambda row: bool(row["included_fastq_to_vcf"]),
    "all_profiled": lambda row: True,
}


def _timing_total(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum direct canonical rows, returning null rather than a partial total."""
    wall_values = [row["wall_seconds"] for row in rows]
    cpu_values = [row["cpu_seconds"] for row in rows]
    return {
        "phase_count": len(rows),
        "phases": [str(row["phase"]) for row in rows],
        "wall_seconds": (
            round(sum(float(value) for value in wall_values), 9)
            if all(value is not None for value in wall_values)
            else None
        ),
        "cpu_seconds": (
            round(sum(float(value) for value in cpu_values), 9)
            if all(value is not None for value in cpu_values)
            else None
        ),
    }


def _build_direct_timing_totals(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build actor and paper-bar totals solely by summing canonical rows."""
    actor_totals: dict[str, Any] = {}
    for actor in ("client", "computing_parties"):
        actor_rows = [row for row in rows if row["actor"] == actor]
        actor_totals[actor] = {
            variant: _timing_total([row for row in actor_rows if include(row)])
            for variant, include in _TOTAL_VARIANTS.items()
        }

    phase_group_totals: dict[str, Any] = {}
    for group, phases in PHASE_GROUPS.items():
        group_rows = [row for row in rows if row["phase"] in phases]
        phase_group_totals[group] = {
            "phases": list(phases),
            **{
                variant: {
                    "client": _timing_total(
                        [
                            row
                            for row in group_rows
                            if row["actor"] == "client" and include(row)
                        ]
                    ),
                    "computing_parties": _timing_total(
                        [
                            row
                            for row in group_rows
                            if row["actor"] == "computing_parties" and include(row)
                        ]
                    ),
                    "combined": _timing_total(
                        [row for row in group_rows if include(row)]
                    ),
                }
                for variant, include in _TOTAL_VARIANTS.items()
            },
        }
    return actor_totals, phase_group_totals


def _sum_nullable_int(rows: list[dict[str, Any]], field: str) -> int | None:
    values = [row[field] for row in rows]
    if not all(value is not None for value in values):
        return None
    return sum(int(value) for value in values)


def _build_communication_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Emit an additive modeled payload stack without double counting."""
    return {
        "computing_parties": {
            "payload_bytes": _sum_nullable_int(
                rows, "classified_computing_party_payload_bytes"
            ),
            "logical_exchange_calls": _sum_nullable_int(
                rows, "classified_computing_party_exchange_calls"
            ),
            "definition": (
                "half-sum CrypTen communicator application-payload accounting "
                "and top-level logical exchange calls in non-reveal phases; "
                "not physical wire traffic"
            ),
        },
        "client": {
            "payload_bytes": _sum_nullable_int(rows, "classified_client_payload_bytes"),
            "logical_exchange_calls": _sum_nullable_int(
                rows, "classified_client_exchange_calls"
            ),
            "modeled_upload_rounds": _sum_nullable_int(rows, "client_upload_rounds"),
            "modeled_upload_payload_bytes": _sum_nullable_int(
                rows, "client_upload_bytes"
            ),
            "observed_reveal_reconstruction_exchange_calls": _sum_nullable_int(
                rows, "observed_client_reconstruction_exchange_calls"
            ),
            "overlapping_download_payload_bytes": _sum_nullable_int(
                rows, "client_download_bytes"
            ),
            "overlapping_download_payload_rounds": _sum_nullable_int(
                rows, "client_download_rounds"
            ),
            "definition": (
                "modeled external-client upload plus CrypTen communicator "
                "payload reclassified as logical-client reveal reconstruction; "
                "the latter is not independently observed external-client "
                "traffic, and download payload fields are overlapping metadata"
            ),
        },
        "byte_metric": (
            "modeled application tensor payload; excludes transport headers, "
            "retransmissions, and other physical wire overhead"
        ),
        "exact_application_dependency_rounds_available": True,
        "round_metric": (
            "application_dependency_rounds derives exact observed CrypTen-API "
            "causal depth; logical_exchange_calls is retained only as a deprecated "
            "top-level invocation diagnostic"
        ),
    }


def _validate_bound_vcf_artifact(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Rank 0 profile lacks its bound FASTQ-to-VCF output artifact")
    path_value = value.get("path")
    sha_value = value.get("sha256")
    size_value = value.get("size_bytes")
    if (
        not isinstance(path_value, str)
        or not path_value
        or not isinstance(sha_value, str)
        or len(sha_value) != 64
        or isinstance(size_value, bool)
        or not isinstance(size_value, int)
        or size_value < 0
    ):
        raise ValueError("Rank 0 profile has a malformed bound VCF artifact")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise ValueError(f"Rank 0 bound VCF artifact is missing: {path}")
    actual_size = path.stat().st_size
    actual_sha = _sha256_file(path)
    if actual_size != size_value or actual_sha != sha_value:
        raise ValueError(
            "Rank 0 bound VCF artifact changed after profiling: "
            f"{path} (expected {sha_value}/{size_value}, got {actual_sha}/{actual_size})"
        )
    return dict(value)


def _aggregate_communicator_events(
    values: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Retain per-rank event deltas and auditable max/sum summaries."""
    event_names: set[str] = set()
    for value in values:
        events = value.get("communicator_events", {})
        if not isinstance(events, dict):
            raise ValueError("Rank phase has malformed communicator event counters")
        event_names.update(str(name) for name in events)
    result: dict[str, dict[str, Any]] = {}
    for event in sorted(event_names):
        rank_values = []
        for rank, value in enumerate(values):
            events = value.get("communicator_events", {})
            raw = events.get(event, {})
            if not isinstance(raw, dict):
                raise ValueError(
                    f"Rank {rank} event {event!r} has malformed counter values"
                )
            counters: dict[str, int] = {}
            for field in (
                "communicator_endpoint_payload_bytes",
                "communicator_operations",
                "logical_exchange_calls",
            ):
                counter = raw.get(field, 0)
                if (
                    not isinstance(counter, int)
                    or isinstance(counter, bool)
                    or counter < 0
                ):
                    raise ValueError(
                        f"Rank {rank} event {event!r} has invalid {field}: "
                        f"{counter!r}"
                    )
                counters[field] = counter
            rank_values.append({"rank": rank, **counters})
        endpoint_values = [
            row["communicator_endpoint_payload_bytes"] for row in rank_values
        ]
        result[event] = {
            "communicator_endpoint_payload_bytes_max": max(endpoint_values),
            "communicator_endpoint_payload_bytes_sum": sum(endpoint_values),
            "communicator_operations_max": max(
                row["communicator_operations"] for row in rank_values
            ),
            "logical_exchange_calls_max": max(
                row["logical_exchange_calls"] for row in rank_values
            ),
            "rank_values": rank_values,
        }
    return result


def _timing_reconciliation(
    rows: list[dict[str, Any]], observed_wall_seconds: float | None
) -> dict[str, Any]:
    included_values = [
        row["wall_seconds"] for row in rows if row["included_fastq_to_vcf"]
    ]
    summed = (
        round(sum(float(value) for value in included_values), 9)
        if all(value is not None for value in included_values)
        else None
    )
    delta = (
        None
        if summed is None or observed_wall_seconds is None
        else float(summed) - float(observed_wall_seconds)
    )
    relative = (
        None
        if delta is None or observed_wall_seconds in (None, 0)
        else delta / float(observed_wall_seconds)
    )
    absolute_tolerance = 0.001
    relative_tolerance = 0.001
    allowed = (
        None
        if observed_wall_seconds is None
        else max(absolute_tolerance, abs(float(observed_wall_seconds)) * relative_tolerance)
    )
    reconciled = (
        None if delta is None or allowed is None else abs(delta) <= allowed
    )
    return {
        "direct_observed_fastq_to_vcf_wall_seconds": observed_wall_seconds,
        "summed_included_phase_wall_seconds": summed,
        "summed_phases_minus_direct_observed_seconds": delta,
        "relative_delta": relative,
        "strict_tolerance": {
            "absolute_seconds": absolute_tolerance,
            "relative_fraction": relative_tolerance,
            "allowed_seconds": allowed,
        },
        "reconciled_within_strict_tolerance": reconciled,
        "actor_phase_costs_are_exact_elapsed_partition": False,
        "interpretation": (
            "Canonical actor/group rows are direct phase costs, not an exact "
            "partition of elapsed FASTQ-to-VCF wall time. Computing-party "
            "phases use per-phase maximum rank time, client phases use rank 0, "
            "and orchestration gaps are intentionally not assigned to an actor. "
            "No residual is manufactured as a component."
        ),
    }


def _load_validated_index_profile(
    path: Path, expected_input_binding: dict[str, Any]
) -> dict[str, Any]:
    """Load an index profile and re-derive claims under its exact schema.

    Legacy full-PanGenie profiles remain readable for audit/migration, but the
    aggregate final gate accepts only ``INDEX_PROFILE_SCHEMA`` below.
    """
    payload = _load_json(path)
    if (
        payload.get("schema") == INDEX_PROFILE_SCHEMA
        and payload.get("schema_version") == 1
    ):
        from pvc.index.pangenie_readmap import validate_pvc_readmap_client_profile

        return validate_pvc_readmap_client_profile(
            path,
            expected_input_binding=expected_input_binding,
        )
    if (
        payload.get("schema") == LEGACY_INDEX_PROFILE_SCHEMA
        and payload.get("schema_version") == 2
    ):
        # Local import avoids the existing index-profiler -> schema-constant
        # import cycle during module initialization.
        from pvc.index.pangenie_process import validate_pvc_index_client_profile

        return validate_pvc_index_client_profile(
            path,
            expected_input_binding=expected_input_binding,
        )
    return payload


def _match_consumed_artifacts_to_index_identity(
    consumed: dict[str, Any], index_identity: Any
) -> tuple[bool, str | None]:
    """Match what genotype loaded to the artifacts certified by indexing."""
    if not isinstance(index_identity, dict):
        return False, "Certified index profile lacks an index artifact identity."
    if consumed.get("blocks_file") != index_identity.get("blocks_file"):
        return False, "Genotype-consumed LD blocks differ from the certified index."
    artifacts = index_identity.get("index_artifacts")
    if not isinstance(artifacts, list):
        return False, "Certified index identity lacks its large artifacts."
    by_path: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            return False, "Certified index identity has a malformed artifact row."
        if artifact["path"] in by_path:
            return False, "Certified index identity contains duplicate artifact paths."
        by_path[artifact["path"]] = artifact
    for role in ("read_map", "graph"):
        value = consumed.get(role)
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            return False, f"Genotype-consumed {role} identity is malformed."
        if by_path.get(value["path"]) != value:
            return False, (
                f"Genotype-consumed {role} is not the exact artifact certified "
                "by the index profile."
            )
    return True, None


def _validate_parent_launch_boundary(
    value: Any,
    *,
    execution_id: str,
    expected_output_vcf: str | Path,
    expected_world_size: int,
) -> dict[str, Any]:
    """Revalidate the cross-process Linux monotonic timing certificate."""
    if (
        not isinstance(value, dict)
        or value.get("schema") != PARENT_LAUNCH_BOUNDARY_SCHEMA
        or value.get("measurement_status") != "measured_and_certified"
        or value.get("execution_id") != execution_id
    ):
        raise ValueError("Parent-launch-to-VCF boundary is missing or incompatible")
    clock = value.get("clock")
    scope = value.get("clock_scope")
    if not (
        isinstance(clock, dict)
        and clock.get("name") == "monotonic"
        and clock.get("monotonic") is True
        and clock.get("adjustable") is False
        and isinstance(clock.get("implementation"), str)
        and bool(clock["implementation"])
        and isinstance(clock.get("resolution_seconds"), (int, float))
        and not isinstance(clock.get("resolution_seconds"), bool)
        and math.isfinite(float(clock["resolution_seconds"]))
        and float(clock["resolution_seconds"]) > 0
        and isinstance(scope, dict)
        and isinstance(scope.get("hostname"), str)
        and bool(scope["hostname"])
        and isinstance(scope.get("boot_id"), str)
        and bool(scope["boot_id"])
        and scope.get("linux_cross_process_clock_domain") is True
    ):
        raise ValueError("Parent-launch boundary lacks a stable Linux clock identity")

    integer_fields = (
        "parent_start_monotonic_ns",
        "rank0_vcf_boundary_monotonic_ns",
        "parent_after_join_monotonic_ns",
    )
    timestamps: dict[str, int] = {}
    for field in integer_fields:
        item = value.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"Parent-launch boundary has invalid {field}: {item!r}")
        timestamps[field] = item
    start = timestamps["parent_start_monotonic_ns"]
    boundary = timestamps["rank0_vcf_boundary_monotonic_ns"]
    joined = timestamps["parent_after_join_monotonic_ns"]
    if not start <= boundary <= joined:
        raise ValueError("Parent-launch boundary timestamps are not ordered")
    expected_wall = (boundary - start) / 1e9
    expected_post = (joined - boundary) / 1e9
    for field, expected in (
        ("wall_seconds", expected_wall),
        ("post_vcf_until_join_seconds", expected_post),
    ):
        observed = value.get(field)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or not math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(f"Parent-launch boundary has inconsistent {field}")

    rank_results = value.get("rank_results")
    if not isinstance(rank_results, list) or len(rank_results) != expected_world_size:
        raise ValueError("Parent-launch boundary lacks one result per rank")
    expected_output = str(Path(expected_output_vcf).expanduser().resolve())
    by_rank: dict[int, dict[str, Any]] = {}
    for result in rank_results:
        if (
            not isinstance(result, dict)
            or result.get("schema") != PARENT_LAUNCH_BOUNDARY_SCHEMA
            or result.get("execution_id") != execution_id
            or result.get("clock") != clock
            or result.get("hostname") != scope["hostname"]
            or result.get("boot_id") != scope["boot_id"]
        ):
            raise ValueError("Parent-launch boundary contains a mismatched rank result")
        rank = result.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank in by_rank:
            raise ValueError("Parent-launch boundary contains an invalid/duplicate rank")
        by_rank[rank] = result
    if set(by_rank) != set(range(expected_world_size)):
        raise ValueError("Parent-launch boundary rank set is incomplete")
    if (
        by_rank[0].get("vcf_boundary_monotonic_ns") != boundary
        or by_rank[0].get("output_vcf") != expected_output
    ):
        raise ValueError("Rank 0 parent-launch boundary does not bind the expected VCF")
    for rank in range(1, expected_world_size):
        if (
            by_rank[rank].get("vcf_boundary_monotonic_ns") is not None
            or by_rank[rank].get("output_vcf") is not None
        ):
            raise ValueError(f"Non-output rank {rank} claims the VCF boundary")
    for field in ("inclusive_start", "inclusive_end"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"Parent-launch boundary lacks {field} semantics")
    for field in ("includes", "excludes_after_vcf"):
        items = value.get(field)
        if not isinstance(items, list) or not items or not all(
            isinstance(item, str) and item for item in items
        ):
            raise ValueError(f"Parent-launch boundary lacks {field} semantics")
    return dict(value)


def aggregate_workflow_profiles(
    genotype_dir: str | Path,
    tool: str,
    *,
    index_profile_path: str | Path | None,
    expected_world_size: int = 3,
    expected_execution_id: str | None = None,
    expected_input_binding: dict[str, Any] | None = None,
    expected_output_vcf: str | Path | None = None,
    parent_launch_boundary: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Aggregate complete rank files and the direct index-phase profile.

    Client phase values use rank 0. Computing-party wall time and logical
    exchange-call counts use the maximum rank; communicator payload uses an
    exact half-sum endpoint accounting convention, and computing-party CPU is
    summed.  Communicator payload is not a physical wire-byte measurement.
    All rank values remain embedded in the JSON for auditability.
    """
    directory = Path(genotype_dir)
    rank_payloads = []
    rank_artifacts = []
    for rank in range(expected_world_size):
        path = directory / f"workflow_profile_rank_{rank}.json"
        payload = _load_json(path)
        if (
            payload.get("schema") != SCHEMA
            or payload.get("schema_version") != 3
            or payload.get("tool") != tool
            or payload.get("rank") != rank
            or payload.get("world_size") != expected_world_size
            or payload.get("status") != "complete"
        ):
            raise ValueError(f"Incomplete or incompatible rank profile {path}")
        rank_payloads.append(payload)
        rank_artifacts.append(
            {
                "rank": rank,
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
            }
        )

    rank_process_memories = [
        _validate_process_memory(payload.get("process_memory"))
        for payload in rank_payloads
    ]
    memory_boot_ids = {
        memory["process_identity"]["boot_id"] for memory in rank_process_memories
    }
    if len(memory_boot_ids) != 1:
        raise ValueError("Rank process-memory records do not share one Linux boot ID")
    runtime_pids: list[int | None] = []
    for rank, (payload, memory) in enumerate(
        zip(rank_payloads, rank_process_memories)
    ):
        runtime_pid = payload.get("runtime_provenance", {}).get("pid")
        runtime_pids.append(runtime_pid)
        if runtime_pid is not None and runtime_pid != memory["process_identity"]["pid"]:
            raise ValueError(
                f"Rank {rank} runtime PID does not match its process-memory identity"
            )
    identity_tuples = [
        (
            memory["process_identity"]["boot_id"],
            memory["process_identity"]["pid"],
            memory["process_identity"]["start_time_ticks"],
        )
        for memory in rank_process_memories
    ]
    identities_unique = len(set(identity_tuples)) == len(identity_tuples)
    if (
        expected_world_size == 3
        and all(pid is not None for pid in runtime_pids)
        and not identities_unique
    ):
        raise ValueError("Three-party rank process-memory identities are not unique")
    rank_memory_summary = {
        "schema": "pvc-aggregate-rank-process-memory-v1",
        "measurement_status": "measured",
        "unit": "KiB (1024 bytes)",
        "per_rank": [
            {
                "rank": rank,
                "process_identity": memory["process_identity"],
                "proc_self_status_vmhwm_kib": memory["proc_self_status"][
                    "vmhwm_kib"
                ],
                "proc_self_status_vmrss_at_observation_kib": memory[
                    "proc_self_status"
                ]["vmrss_kib"],
                "resource_ru_maxrss_kib": memory["resource_getrusage_self"][
                    "ru_maxrss_kib"
                ],
            }
            for rank, memory in enumerate(rank_process_memories)
        ],
        "largest_single_rank_process_vmhwm_kib": max(
            memory["proc_self_status"]["vmhwm_kib"]
            for memory in rank_process_memories
        ),
        "largest_single_rank_process_ru_maxrss_kib": max(
            memory["resource_getrusage_self"]["ru_maxrss_kib"]
            for memory in rank_process_memories
        ),
        "rank_process_identities_unique": identities_unique,
        "time_aligned_sum_ranks_0_2_kib": None,
        "time_aligned_sum_status": (
            "requires post-hoc exact PID-identity join to the optional "
            "measure_peak_rss JSON sidecar; never sum per-rank high-water marks"
        ),
        "semantics": {
            "per_rank_peak_scope": (
                "one rank process from its process launch through the native "
                "profile observation boundary"
            ),
            "rank_peaks_are_not_time_aligned_or_additive": True,
            "not_phase_specific": True,
            "rank0_role_caveat": (
                "Rank 0 hosts both a computing party and prototype logical-client "
                "work; its process RSS cannot be separated into client-only and "
                "computing-party-only memory."
            ),
        },
    }

    source_shas = {
        str(payload.get("source_provenance", {}).get("manifest_sha256", ""))
        for payload in rank_payloads
    }
    if len(source_shas) != 1 or "" in source_shas:
        raise ValueError("Rank profiles do not share one source manifest")

    execution_ids = {str(payload.get("execution_id", "")) for payload in rank_payloads}
    if len(execution_ids) != 1 or "" in execution_ids:
        raise ValueError(
            "Rank profiles do not share one non-empty parent execution ID; "
            "stale rank files may be mixed into this run"
        )
    execution_id = next(iter(execution_ids))
    if expected_execution_id is not None and execution_id != expected_execution_id:
        raise ValueError(
            "Rank profile execution ID does not match the parent execution: "
            f"{execution_id!r} != {expected_execution_id!r}"
        )

    validated_rank_bindings = [
        _validate_input_binding(payload.get("input_binding"))
        for payload in rank_payloads
    ]
    binding_identity_shas = {
        str(binding["input_identity_sha256"])
        for binding in validated_rank_bindings
    }
    if len(binding_identity_shas) != 1:
        raise ValueError(
            "Rank profiles do not share one canonical manifest/input identity"
        )
    input_binding = validated_rank_bindings[0]
    if any(binding != input_binding for binding in validated_rank_bindings[1:]):
        raise ValueError("Rank profile manifest/input bindings are not identical")
    if expected_input_binding is not None:
        expected_binding = _validate_input_binding(expected_input_binding)
        if input_binding != expected_binding:
            raise ValueError(
                "Rank profile manifest/input identity does not match the parent input"
            )

    validated_genotype_artifacts = [
        _validate_genotype_input_artifacts(
            payload.get("genotype_input_artifacts")
        )
        for payload in rank_payloads
    ]
    genotype_input_artifacts = validated_genotype_artifacts[0]
    if any(
        value != genotype_input_artifacts
        for value in validated_genotype_artifacts[1:]
    ):
        raise ValueError(
            "Rank profiles did not consume one identical read-map/graph/blocks identity"
        )

    validated_configurations = [
        _validate_tool_effective_configuration(
            payload.get("effective_configuration"), tool
        )
        for payload in rank_payloads
    ]
    effective_configuration = validated_configurations[0]
    if any(
        value != effective_configuration for value in validated_configurations[1:]
    ):
        raise ValueError("Rank profiles used different effective PVC configurations")
    diagnostic_configuration = effective_configuration["values"].get(
        "diagnostic_outputs", {}
    )
    diagnostic_outputs_enabled = bool(
        isinstance(diagnostic_configuration, dict)
        and diagnostic_configuration.get("enabled") is True
    )

    communication_capabilities: list[dict[str, Any]] = []
    for rank, payload in enumerate(rank_payloads):
        capability = payload.get("communication_counter_capability")
        if not isinstance(capability, dict):
            raise ValueError(
                f"Rank {rank} profile lacks communication-counter capability metadata"
            )
        implementation = capability.get("implementation")
        if expected_world_size == 3 and not (
            capability.get("crypten_initialized") is True
            and capability.get("complete") is True
            and isinstance(implementation, dict)
            and bool(implementation)
        ):
            raise ValueError(
                f"Rank {rank} lacks a complete initialized three-party "
                "communication-counter capability and implementation identity"
            )
        if (
            expected_world_size != 3
            and capability.get("crypten_initialized") is True
            and capability.get("complete") is not True
        ):
            raise ValueError(
                f"Rank {rank} initialized CrypTen without all required "
                "communication counters"
            )
        communication_capabilities.append(capability)

    initialized_implementations = [
        capability.get("implementation")
        for capability in communication_capabilities
        if capability.get("crypten_initialized") is True
    ]
    if expected_world_size == 3 and (
        len(initialized_implementations) != expected_world_size
        or any(
            implementation != initialized_implementations[0]
            for implementation in initialized_implementations[1:]
        )
    ):
        raise ValueError(
            "Rank profiles used different communicator counter implementations"
        )

    rank0_boundary = rank_payloads[0].get("boundary")
    if not isinstance(rank0_boundary, dict):
        raise ValueError("Rank 0 profile lacks FASTQ-to-VCF boundary metadata")
    output_vcf_artifact = _validate_bound_vcf_artifact(
        rank0_boundary.get("output_vcf_artifact")
    )
    if expected_output_vcf is not None and Path(
        output_vcf_artifact["path"]
    ).resolve() != Path(expected_output_vcf).resolve():
        raise ValueError(
            "Rank 0 bound VCF path does not match the parent-requested output: "
            f"{output_vcf_artifact['path']} != {Path(expected_output_vcf).resolve()}"
        )
    if rank0_boundary.get("genotype_wall_seconds") is None:
        raise ValueError("Rank 0 did not record a completed FASTQ-to-VCF boundary")
    for rank, payload in enumerate(rank_payloads[1:], start=1):
        boundary = payload.get("boundary", {})
        if not isinstance(boundary, dict):
            raise ValueError(f"Rank {rank} profile has malformed boundary metadata")
        if (
            boundary.get("genotype_wall_seconds") is not None
            or boundary.get("genotype_cpu_seconds") is not None
            or boundary.get("output_vcf_artifact") is not None
        ):
            raise ValueError(
                f"Non-output rank {rank} claims the rank-0 VCF boundary"
            )

    validated_parent_launch_boundary = (
        _validate_parent_launch_boundary(
            parent_launch_boundary,
            execution_id=execution_id,
            expected_output_vcf=(
                expected_output_vcf
                if expected_output_vcf is not None
                else output_vcf_artifact["path"]
            ),
            expected_world_size=expected_world_size,
        )
        if parent_launch_boundary is not None
        else None
    )

    index_payload: dict[str, Any] | None = None
    index_rows: dict[str, dict[str, Any]] = {}
    if index_profile_path is not None:
        index_path = Path(index_profile_path)
        index_payload = _load_validated_index_profile(index_path, input_binding)
        if index_payload.get("schema") != INDEX_PROFILE_SCHEMA:
            raise ValueError(f"Incompatible PVC index profile {index_path}")
        phases = index_payload.get("phases")
        if not isinstance(phases, list):
            raise ValueError(f"PVC index profile {index_path} lacks phases")
        index_rows = {str(row["phase"]): row for row in phases if isinstance(row, dict)}
    index_complete = set(index_rows) >= _INDEX_PHASES
    index_execution_provenance = (
        index_payload.get("execution_provenance", {})
        if isinstance(index_payload, dict)
        else {}
    )
    if not isinstance(index_execution_provenance, dict):
        index_execution_provenance = {}
    index_claims_fresh_certification = (
        isinstance(index_payload, dict)
        and index_payload.get("schema_version") == 1
        and index_payload.get("schema") == INDEX_PROFILE_SCHEMA
        and index_execution_provenance.get("certified") is True
        and index_execution_provenance.get("status") == "fresh_verified"
    )
    index_input_binding: dict[str, Any] | None = None
    index_input_binding_matches = False
    index_input_binding_reason: str | None = None
    if isinstance(index_payload, dict) and index_payload.get("input_binding") is not None:
        try:
            index_input_binding = _validate_input_binding(
                index_payload.get("input_binding")
            )
        except ValueError as exc:
            index_input_binding_reason = str(exc)
            if index_claims_fresh_certification:
                raise ValueError(
                    "Fresh PVC index profile has an invalid input binding"
                ) from exc
        else:
            index_input_binding_matches = index_input_binding == input_binding
            if not index_input_binding_matches:
                index_input_binding_reason = (
                    "Index profile canonical manifest/input identity differs "
                    "from the genotype execution"
                )
                if index_claims_fresh_certification:
                    raise ValueError(index_input_binding_reason)
    else:
        index_input_binding_reason = "Index profile lacks a canonical input binding."
        if index_claims_fresh_certification:
            raise ValueError(
                "Fresh PVC index profile lacks the canonical input binding "
                "required by the genotype execution"
            )
    index_artifact_binding_matches, index_artifact_binding_reason = (
        _match_consumed_artifacts_to_index_identity(
            genotype_input_artifacts,
            index_payload.get("index_identity")
            if isinstance(index_payload, dict)
            else None,
        )
    )
    if index_claims_fresh_certification and not index_artifact_binding_matches:
        raise ValueError(str(index_artifact_binding_reason))
    index_execution_certified = (
        index_complete
        and index_claims_fresh_certification
        and index_input_binding_matches
        and index_artifact_binding_matches
    )
    if not index_complete:
        index_certification_reason = "Index phase timing rows are missing or incomplete."
    elif index_execution_certified:
        index_certification_reason = None
    elif not index_claims_fresh_certification:
        index_certification_reason = str(
            index_execution_provenance.get("reason")
            or "Index execution provenance is not certified as fresh_verified."
        )
        if index_input_binding_reason:
            index_certification_reason += f" {index_input_binding_reason}"
        if index_artifact_binding_reason:
            index_certification_reason += f" {index_artifact_binding_reason}"
    else:
        index_certification_reason = (
            index_input_binding_reason or index_artifact_binding_reason
        )

    by_rank = [_rank_phase_map(payload) for payload in rank_payloads]
    rank_communication_reconciliations = [
        _validate_rank_communication_reconciliation(payload, phase_map)
        for payload, phase_map in zip(rank_payloads, by_rank)
    ]
    aggregate_communication_reconciliation = (
        _aggregate_rank_communication_reconciliation(
            rank_communication_reconciliations
        )
    )
    aggregate_rows: list[dict[str, Any]] = []
    for name in PHASE_ORDER:
        actor = phase_actor(name, tool)
        if name in _INDEX_PHASES:
            row = dict(_empty_record(name, tool))
            source = index_rows.get(name)
            if source is None:
                for field in (
                    "wall_seconds",
                    "cpu_seconds",
                    "calls",
                    "communicator_endpoint_payload_bytes",
                    "communicator_endpoint_payload_bytes_max",
                    "communicator_endpoint_payload_bytes_sum",
                    "communicator_payload_bytes",
                    "communicator_operations",
                    "logical_exchange_calls",
                    "modeled_computing_party_payload_bytes",
                    "modeled_client_reconstruction_payload_bytes",
                    "observed_computing_party_exchange_calls",
                    "observed_client_reconstruction_exchange_calls",
                    "classified_computing_party_payload_bytes",
                    "classified_client_payload_bytes",
                    "classified_computing_party_exchange_calls",
                    "classified_client_exchange_calls",
                    "client_upload_bytes",
                    "client_upload_rounds",
                    "client_download_bytes",
                    "client_download_rounds",
                ):
                    row[field] = None
                row["measurement_status"] = "missing_index_profile"
                row["cpu_seconds_model"] = "unavailable"
                row["client_transfer_model"] = []
                row["client_transfer_events"] = []
            else:
                for field in (
                    "wall_seconds",
                    "cpu_seconds",
                    "calls",
                    "communicator_endpoint_payload_bytes",
                    "communicator_operations",
                    "logical_exchange_calls",
                    "client_upload_bytes",
                    "client_upload_rounds",
                    "client_download_bytes",
                    "client_download_rounds",
                ):
                    row[field] = source.get(field, 0)
                row["communicator_endpoint_payload_bytes_max"] = 0
                row["communicator_endpoint_payload_bytes_sum"] = 0
                row["communicator_payload_bytes"] = 0
                row["modeled_computing_party_payload_bytes"] = 0
                row["modeled_client_reconstruction_payload_bytes"] = 0
                row["observed_computing_party_exchange_calls"] = 0
                row["observed_client_reconstruction_exchange_calls"] = 0
                row["classified_computing_party_payload_bytes"] = 0
                row["classified_client_payload_bytes"] = 0
                row["classified_computing_party_exchange_calls"] = 0
                row["classified_client_exchange_calls"] = 0
                row["communicator_events"] = source.get("communicator_events", {})
                row["client_transfer_model"] = source.get("client_transfer_model", [])
                row["client_transfer_events"] = source.get(
                    "client_transfer_events", []
                )
                row["cpu_seconds_model"] = source.get(
                    "cpu_seconds_model", "native one-thread wall-time proxy"
                )
                row["measurement_status"] = "measured"
            row["rank_rule"] = "direct native index phase; no rank aggregation"
            row["rank_values"] = []
        else:
            values = [rank_map[name] for rank_map in by_rank]
            chosen = values[0] if actor == "client" else None
            row = dict(_empty_record(name, tool))
            row["wall_seconds"] = (
                float(chosen["wall_seconds"])
                if chosen is not None
                else max(float(value["wall_seconds"]) for value in values)
            )
            row["cpu_seconds"] = (
                float(chosen["cpu_seconds"])
                if chosen is not None
                else sum(float(value["cpu_seconds"]) for value in values)
            )
            row["calls"] = (
                int(chosen["calls"])
                if chosen is not None
                else max(int(value["calls"]) for value in values)
            )
            endpoint_bytes = []
            for rank, value in enumerate(values):
                endpoint_value = value["communicator_endpoint_payload_bytes"]
                if (
                    isinstance(endpoint_value, bool)
                    or not isinstance(endpoint_value, int)
                    or endpoint_value < 0
                ):
                    raise ValueError(
                        f"Phase {name} rank {rank} has non-integer or negative "
                        f"endpoint payload bytes: {endpoint_value!r}"
                    )
                endpoint_bytes.append(endpoint_value)
            endpoint_sum = sum(endpoint_bytes)
            if endpoint_sum % 2:
                raise ValueError(
                    f"Phase {name} has odd endpoint-payload-byte sum {endpoint_sum}; "
                    "cannot apply the half-sum communicator-payload convention"
                )
            row["communicator_endpoint_payload_bytes"] = None
            row["communicator_endpoint_payload_bytes_max"] = max(endpoint_bytes)
            row["communicator_endpoint_payload_bytes_sum"] = endpoint_sum
            row["communicator_payload_bytes"] = endpoint_sum // 2
            row["communicator_operations"] = max(
                int(value["communicator_operations"]) for value in values
            )
            row["logical_exchange_calls"] = max(
                int(value["logical_exchange_calls"]) for value in values
            )
            row["communicator_events"] = _aggregate_communicator_events(values)
            for field in (
                "client_upload_bytes",
                "client_upload_rounds",
                "client_download_bytes",
                "client_download_rounds",
            ):
                row[field] = int(values[0][field])
            if name == "reveal":
                row["modeled_computing_party_payload_bytes"] = 0
                row["modeled_client_reconstruction_payload_bytes"] = row[
                    "communicator_payload_bytes"
                ]
                row["observed_computing_party_exchange_calls"] = 0
                row["observed_client_reconstruction_exchange_calls"] = row[
                    "logical_exchange_calls"
                ]
            else:
                row["modeled_computing_party_payload_bytes"] = row[
                    "communicator_payload_bytes"
                ]
                row["modeled_client_reconstruction_payload_bytes"] = 0
                row["observed_computing_party_exchange_calls"] = row[
                    "logical_exchange_calls"
                ]
                row["observed_client_reconstruction_exchange_calls"] = 0
            row["classified_computing_party_payload_bytes"] = row[
                "modeled_computing_party_payload_bytes"
            ]
            row["classified_client_payload_bytes"] = (
                row["client_upload_bytes"]
                + row["modeled_client_reconstruction_payload_bytes"]
            )
            row["classified_computing_party_exchange_calls"] = row[
                "observed_computing_party_exchange_calls"
            ]
            row["classified_client_exchange_calls"] = (
                row["client_upload_rounds"]
                + row["observed_client_reconstruction_exchange_calls"]
            )
            row["client_transfer_model"] = list(values[0]["client_transfer_model"])
            row["client_transfer_events"] = list(values[0]["client_transfer_events"])
            row["measurement_status"] = (
                "measured"
                if any(int(value["calls"]) > 0 for value in values)
                else "zero_calls"
            )
            row["rank_rule"] = (
                "rank 0 logical-client observation"
                if actor == "client"
                else (
                    "max-rank wall/logical-exchange count, half-sum endpoint "
                    "communicator payload, and sum-rank CPU"
                )
            )
            row["rank_values"] = [
                {
                    "rank": rank,
                    **{
                        field: value[field]
                        for field in (
                            "wall_seconds",
                            "cpu_seconds",
                            "calls",
                            "communicator_endpoint_payload_bytes",
                            "communicator_operations",
                            "logical_exchange_calls",
                            "communicator_events",
                            "client_upload_bytes",
                            "client_upload_rounds",
                            "client_download_bytes",
                            "client_download_rounds",
                            "client_transfer_events",
                        )
                    },
                }
                for rank, value in enumerate(values)
            ]
        if row["wall_seconds"] is not None:
            row["wall_seconds"] = round(float(row["wall_seconds"]), 9)
        if row["cpu_seconds"] is not None:
            row["cpu_seconds"] = round(float(row["cpu_seconds"]), 9)
        aggregate_rows.append(row)

    application_dependency_rounds = build_application_dependency_round_certificate(
        [payload.get("application_dependency_trace") for payload in rank_payloads],
        [
            int(reconciliation["whole_run_delta"][
                "communicator_endpoint_payload_bytes"
            ])
            for reconciliation in rank_communication_reconciliations
        ],
        [phase for phase in PHASE_ORDER if phase not in _INDEX_PHASES],
    )
    actor_totals, phase_group_totals = _build_direct_timing_totals(aggregate_rows)
    communication_totals = _build_communication_totals(aggregate_rows)
    modeled_client_rounds = build_modeled_client_rounds(
        [by_rank[0][phase] for phase in PHASE_ORDER if phase not in _INDEX_PHASES],
        PHASE_ORDER,
    )
    modeled_reconciliation = modeled_client_rounds["reconciliation"]
    client_totals = communication_totals["client"]
    modeled_totals_comparable = (
        client_totals["modeled_upload_payload_bytes"] is not None
        and client_totals["overlapping_download_payload_bytes"] is not None
    )
    if modeled_totals_comparable and (
        modeled_reconciliation["upload_payload_bytes"]
        != client_totals["modeled_upload_payload_bytes"]
        or modeled_reconciliation["download_payload_bytes"]
        != client_totals["overlapping_download_payload_bytes"]
    ):
        raise ValueError(
            "Ordered modeled-client rounds do not reconcile to aggregate communication totals"
        )
    modeled_reconciliation["communication_totals_exact_match"] = (
        True if modeled_totals_comparable else None
    )
    modeled_reconciliation["communication_totals_comparison_status"] = (
        "exact" if modeled_totals_comparable else "unavailable_incomplete_index_profile"
    )
    communication_totals["exact_application_dependency_rounds_available"] = True
    communication_totals["dependency_round_depth"] = application_dependency_rounds[
        "dependency_depth"
    ]
    communication_totals["dependency_round_source_schema"] = CERTIFICATE_SCHEMA

    rank0_boundary = rank_payloads[0].get("boundary", {})
    genotype_wall = rank0_boundary.get("genotype_wall_seconds")
    genotype_cpu = rank0_boundary.get("genotype_cpu_seconds")
    index_profiled_phase_wall = (
        sum(float(index_rows[name]["wall_seconds"]) for name in _INDEX_PHASES)
        if index_complete
        else None
    )
    index_cpu = (
        sum(float(index_rows[name]["cpu_seconds"]) for name in _INDEX_PHASES)
        if index_complete
        else None
    )
    index_direct_process_wall = (
        float(index_payload["direct_process_wall_seconds"])
        if index_execution_certified
        and isinstance(index_payload, dict)
        and isinstance(index_payload.get("direct_process_wall_seconds"), (int, float))
        and not isinstance(index_payload.get("direct_process_wall_seconds"), bool)
        and math.isfinite(float(index_payload["direct_process_wall_seconds"]))
        and float(index_payload["direct_process_wall_seconds"]) > 0
        else None
    )
    genotype_parent_launch_wall = (
        float(validated_parent_launch_boundary["wall_seconds"])
        if validated_parent_launch_boundary is not None
        else None
    )
    observed_workflow_wall = (
        None
        if genotype_parent_launch_wall is None or index_direct_process_wall is None
        else index_direct_process_wall + genotype_parent_launch_wall
    )
    profiled_phase_plus_genotype_wall = (
        None
        if genotype_wall is None or index_profiled_phase_wall is None
        else index_profiled_phase_wall + float(genotype_wall)
    )
    workflow_cpu_rank0 = (
        None
        if genotype_cpu is None or index_cpu is None
        else index_cpu + float(genotype_cpu)
    )
    necessary_wall = (
        sum(
            float(row["wall_seconds"])
            for row in aggregate_rows
            if row["required_for_vcf"]
        )
        if index_complete
        and all(
            row["wall_seconds"] is not None
            for row in aggregate_rows
            if row["required_for_vcf"]
        )
        else None
    )
    necessary_cpu = (
        sum(
            float(row["cpu_seconds"])
            for row in aggregate_rows
            if row["required_for_vcf"]
        )
        if index_complete
        and all(
            row["cpu_seconds"] is not None
            for row in aggregate_rows
            if row["required_for_vcf"]
        )
        else None
    )
    timing_reconciliation = _timing_reconciliation(
        aggregate_rows, observed_workflow_wall
    )
    boundary_complete = bool(
        index_execution_certified
        and index_direct_process_wall is not None
        and validated_parent_launch_boundary is not None
    )
    if not index_execution_certified:
        boundary_certification_reason = index_certification_reason
    elif index_direct_process_wall is None:
        boundary_certification_reason = (
            "Fresh read-map profile lacks a direct PanGenie-readmap wall duration."
        )
    elif validated_parent_launch_boundary is None:
        boundary_certification_reason = (
            "Parent launch-to-rank-0 VCF monotonic boundary was not supplied."
        )
    else:
        boundary_certification_reason = None
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "schema_version": 4,
        "execution_id": execution_id,
        "input_binding": input_binding,
        "genotype_input_artifacts": genotype_input_artifacts,
        "effective_configuration": effective_configuration,
        "tool": tool,
        "world_size": expected_world_size,
        "profile_enabled_by_default": True,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_manifest_sha256": next(iter(source_shas)),
        "output_vcf_artifact": output_vcf_artifact,
        "parent_launch_boundary": validated_parent_launch_boundary,
        "process_memory": rank_memory_summary,
        "communication_counter_capabilities": communication_capabilities,
        "communication_counter_implementation": (
            initialized_implementations[0] if initialized_implementations else None
        ),
        "communication_reconciliation": aggregate_communication_reconciliation,
        "application_dependency_rounds": application_dependency_rounds,
        "modeled_client_rounds": modeled_client_rounds,
        "boundary_contract": {
            "schema": FASTQ_TO_VCF_BOUNDARY_CONTRACT_SCHEMA,
            "certified": boundary_complete,
            "direct_total_field": "observed_fastq_to_vcf_wall_seconds",
            "direct_total_components": [
                "index_client_wall_seconds",
                "genotype_parent_launch_to_vcf_wall_seconds",
            ],
            "direct_total_is_sum_of_components": True,
            "legacy_rank0_timer_field": "genotype_rank0_wall_seconds",
            "legacy_rank0_timer_eligible": False,
        },
        "rank_profile_artifacts": rank_artifacts,
        "index_profile_artifact": (
            {
                "path": str(Path(index_profile_path).resolve()),
                "sha256": _sha256_file(Path(index_profile_path)),
            }
            if index_profile_path is not None
            else None
        ),
        "aggregation_rules": {
            "client": "rank 0 logical-client value",
            "process_memory": (
                "retain each rank's independent process-lifetime VmHWM and "
                "ru_maxrss; report the maximum single rank but never sum rank "
                "high-water marks; a concurrent rank sum requires the sampled "
                "process-tree sidecar PID-identity join"
            ),
            "computing_party_wall_seconds": "maximum across ranks",
            "computing_party_cpu_seconds": "sum across ranks",
            "communicator_endpoint_payload_bytes_max": (
                "maximum single-rank CrypTen modeled application-payload count"
            ),
            "communicator_payload_bytes": (
                "half the sum of rank endpoint application-payload counts as "
                "an accounting convention; not a physical link/wire measurement"
            ),
            "communicator_operations": (
                "maximum per-rank CrypTen comm_rounds counter: a legacy "
                "low-level communicator-operation proxy that may count each "
                "tensor in one batched invocation; this is not MPC round depth"
            ),
            "logical_exchange_calls": (
                "deprecated diagnostic maximum of top-level collective or batched "
                "point-to-point invocations; the exact causal metric is in "
                "application_dependency_rounds"
            ),
            "application_dependency_rounds": (
                "fail-closed cross-rank matching of CrypTen API payload messages; "
                "round IDs are contiguous and one-based, every round balances sent "
                "and received bytes, and modeled client calls remain separate"
            ),
            "whole_run_communication_reconciliation": (
                "required exact equality, independently on every rank, between "
                "the post-reset-to-rank-write communicator delta and the sum "
                "of all canonical genotype phase deltas including event dictionaries"
            ),
            "communication_time": (
                "not measured; the legacy CrypTen comm_time field remains zero "
                "and is not used or reported as elapsed communication time"
            ),
            "client_transfers": "rank 0 modeled external-client values",
            "nonoverlapping_paper_stack": (
                "classified_client_payload_bytes + "
                "classified_computing_party_payload_bytes; CrypTen reveal "
                "payload is assigned to the logical-client reconstruction "
                "model and excluded from computing-party payload"
            ),
            "client_download_overlap": (
                "client_download_bytes describes the revealed plaintext payload; "
                "it overlaps the modeled logical-client reveal reconstruction "
                "and must not be added to classified_client_payload_bytes"
            ),
            "classified_exchange_stack": (
                "classified_client_exchange_calls is modeled client upload "
                "rounds plus logical-client reveal reconstruction exchanges; "
                "classified_computing_party_exchange_calls contains logical "
                "exchanges from every non-reveal phase"
            ),
            "actor_and_phase_group_totals": (
                "direct sums of the non-overlapping canonical phase rows; "
                "no component is inferred as a residual or by subtraction; "
                "these phase costs are not asserted to partition elapsed wall time"
            ),
            "parent_launch_to_vcf_wall_seconds": (
                "one direct Linux CLOCK_MONOTONIC interval from the parent "
                "immediately before spawn_multiparty to rank 0 immediately "
                "after write_output_vcf returns; no rank aggregation, phase "
                "sum, or residual subtraction"
            ),
        },
        "boundary": {
            "name": "FASTQ-to-VCF",
            "complete": boundary_complete,
            "certification_reason": boundary_certification_reason,
            "index_execution_provenance_status": index_execution_provenance.get(
                "status", "missing"
            ),
            "index_execution_provenance_certified": bool(index_execution_certified),
            "index_input_binding_matches": bool(index_input_binding_matches),
            "index_input_binding_reason": index_input_binding_reason,
            "index_artifact_binding_matches": bool(
                index_artifact_binding_matches
            ),
            "index_artifact_binding_reason": index_artifact_binding_reason,
            "inclusive_start": (
                "immediately before invoking the sample-specific "
                "PanGenie-readmap command against the reusable public index"
            ),
            "inclusive_end": "successful return from write_output_vcf",
            "observed_elapsed_includes": (
                ["private_score_artifact_output", "likelihood_output"]
                if diagnostic_outputs_enabled
                else []
            ),
            "observed_elapsed_excludes": [
                "public panel/index construction",
                "truth_verification",
                "completion_marker",
            ],
            "observed_prototype_wall_seconds": observed_workflow_wall,
            "observed_fastq_to_vcf_wall_seconds": observed_workflow_wall,
            "observed_prototype_rank0_cpu_seconds": None,
            "profiled_phase_plus_genotype_wall_seconds": (
                profiled_phase_plus_genotype_wall
            ),
            "profiled_phase_plus_genotype_rank0_cpu_seconds": workflow_cpu_rank0,
            "necessary_profiled_phase_wall_seconds": necessary_wall,
            "necessary_profiled_phase_cpu_seconds": necessary_cpu,
            "index_client_wall_seconds": index_direct_process_wall,
            "index_client_profiled_phase_sum_wall_seconds": (
                index_profiled_phase_wall
            ),
            "index_client_cpu_seconds": index_cpu,
            "genotype_rank0_wall_seconds": genotype_wall,
            "genotype_rank0_cpu_seconds": genotype_cpu,
            "genotype_parent_launch_to_vcf_wall_seconds": genotype_parent_launch_wall,
            "genotype_parent_launch_boundary_status": (
                validated_parent_launch_boundary["measurement_status"]
                if validated_parent_launch_boundary is not None
                else "not_supplied"
            ),
            "output_vcf_artifact": output_vcf_artifact,
            "note": (
                "The direct observed wall adds the complete PanGenie-readmap "
                "command timer (including startup and inter-phase gaps) to the "
                "certified parent-immediately-before-spawn through successful "
                "VCF-writer-return timer. The older rank-0 post-initialization "
                "timer is retained only as diagnostic source data and is never "
                "used in the paper total or timing reconciliation. The separate "
                "profiled-phase sum retains the five native read-map summaries "
                "for attribution; no component is obtained by subtraction."
            ),
        },
        "index_profile": index_payload,
        "index_binding_validation": {
            "expected_input_identity_sha256": input_binding[
                "input_identity_sha256"
            ],
            "index_input_identity_sha256": (
                index_input_binding.get("input_identity_sha256")
                if index_input_binding is not None
                else None
            ),
            "matches": bool(index_input_binding_matches),
            "reason": index_input_binding_reason,
            "consumed_artifacts_match": bool(index_artifact_binding_matches),
            "consumed_artifacts_reason": index_artifact_binding_reason,
            "consumed_artifacts": genotype_input_artifacts,
        },
        "rank_profiles": rank_payloads,
        "actor_totals": actor_totals,
        "phase_group_totals": phase_group_totals,
        "timing_reconciliation": timing_reconciliation,
        "communication_totals": communication_totals,
        "phases": aggregate_rows,
    }
    json_path = directory / "workflow_profile.json"
    csv_path = directory / "workflow_profile.csv"
    _atomic_json(json_path, payload)
    _atomic_csv(csv_path, aggregate_rows)
    return json_path, csv_path
