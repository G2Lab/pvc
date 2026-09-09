from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pvc.index.blocks import run_plink_blocks
from pvc.index.pangenie_process import (
    expected_pvc_index_artifacts,
    missing_pvc_index_artifacts,
    run_pangenie_process_index,
    validate_pvc_index_client_profile,
    write_pvc_index_client_profile,
)
from pvc.index.pangenie_readmap import (
    PUBLIC_BLOCKS_MANIFEST_KEY,
    READMAP_LAUNCH_SCHEMA,
    expected_pvc_readmap_index_artifacts,
    missing_pvc_readmap_outputs,
    public_blocks_artifact,
    public_blocks_file,
    public_index_prefix,
    resolve_read_count_mode,
    run_pangenie_readmap,
    validate_pvc_readmap_client_profile,
    write_pvc_readmap_client_profile,
)
from pvc.config import PVC_INDEX_BINS, PVC_READMAP_BIN


def pvc_index_dir(run_dir: str | Path, tool: str) -> Path:
    return Path(run_dir) / tool / "index"


def pvc_index_prefix(run_dir: str | Path, tool: str) -> Path:
    return pvc_index_dir(run_dir, tool) / tool


def pvc_index_complete_path(run_dir: str | Path, tool: str) -> Path:
    return pvc_index_dir(run_dir, tool) / "index_complete.json"


def pvc_index_client_profile_path(run_dir: str | Path, tool: str) -> Path:
    return pvc_index_dir(run_dir, tool) / "client_read_profile.json"


def ensure_pvc_index_client_profile(
    metadata: dict[str, Any],
    *,
    tool: str,
    threads: int = 1,
    manifest: dict[str, Any] | None = None,
    run_dir: str | Path | None = None,
) -> Path | None:
    """Resolve/backfill the native client profile through index aliases.

    Historical alias markers may not carry ``client_read_profile`` and their
    own directory has neither ``stderr.log`` nor index artifacts.  Follow the
    actual index prefix and reused marker chain before considering the alias
    marker's directory.
    """
    # Keep the marker directory that supplied each payload so relative paths
    # in historical markers retain their intended meaning.
    pending: list[tuple[dict[str, Any], Path | None]] = [(dict(metadata), None)]
    seen_markers: set[Path] = set()
    candidate_dirs: list[Path] = []
    candidate_profiles: list[Path] = []

    def resolve_metadata_path(value: Any, base: Path | None) -> Path:
        path = Path(str(value))
        if not path.is_absolute() and base is not None:
            path = base / path
        return path

    while pending:
        current, base = pending.pop(0)
        profile_value = current.get("client_read_profile")
        if profile_value:
            candidate_profiles.append(resolve_metadata_path(profile_value, base))
        prefix_value = current.get("index_prefix")
        if prefix_value:
            candidate_dirs.append(resolve_metadata_path(prefix_value, base).parent)

        # ``index_marker`` identifies the current marker, whereas
        # ``reused_from_marker`` points to its predecessor.  Following only
        # the first truthy value stops at the first alias, so traverse both.
        for marker_key in ("index_marker", "reused_from_marker"):
            marker_value = current.get(marker_key)
            if not marker_value:
                continue
            marker = resolve_metadata_path(marker_value, base).resolve()
            candidate_dirs.append(marker.parent)
            if marker.is_file() and marker not in seen_markers:
                seen_markers.add(marker)
                try:
                    nested = read_pvc_index_complete(marker)
                except (OSError, ValueError, json.JSONDecodeError):
                    nested = None
                if isinstance(nested, dict):
                    pending.append((dict(nested), marker.parent))

    expected_binding = None
    if manifest is not None and run_dir is not None:
        from pvc.genotype.private.workflow_profile import build_workflow_input_binding

        expected_binding = build_workflow_input_binding(manifest, run_dir)

    def validate_or_migrate_historical(path: Path) -> Path:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if (
            isinstance(payload, dict)
            and payload.get("schema") == "pvc-readmap-client-profile-v1"
        ):
            validate_pvc_readmap_client_profile(
                path,
                expected_input_binding=expected_binding,
            )
            return path
        try:
            validate_pvc_index_client_profile(
                path,
                expected_input_binding=expected_binding,
            )
            return path
        except ValueError:
            # Profiles written before schema v2 have no complete evidence
            # chain.  They may be re-derived from an untouched historical log,
            # but are never promoted to fresh/certified.  Any profile already
            # claiming v2, or any directory with a launch artifact, fails
            # closed instead of being silently rewritten.
            if (path.parent / "process_launch_provenance.json").exists():
                raise
            try:
                old_payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raise
            if (
                not isinstance(old_payload, dict)
                or old_payload.get("schema") != "pvc-index-client-profile-v1"
                or old_payload.get("schema_version") != 1
                or not (path.parent / "stderr.log").is_file()
            ):
                raise
            historical_tool = old_payload.get("tool")
            if not isinstance(historical_tool, str):
                historical_tool = tool
            migrated, _ = write_pvc_index_client_profile(
                path.parent,
                tool=historical_tool,
                threads=threads,
                manifest=manifest,
                run_dir=run_dir,
            )
            validate_pvc_index_client_profile(
                migrated,
                expected_input_binding=expected_binding,
            )
            return migrated

    for path in candidate_profiles:
        if path.is_file():
            return validate_or_migrate_historical(path)
    unique_dirs = list(dict.fromkeys(path.resolve() for path in candidate_dirs))
    for directory in unique_dirs:
        path = directory / "client_read_profile.json"
        if path.is_file():
            return validate_or_migrate_historical(path)
    incomplete_reasons: list[dict[str, str]] = []
    for directory in unique_dirs:
        if (directory / "stderr.log").is_file():
            profile_tool = tool
            launch_path = directory / "process_launch_provenance.json"
            if launch_path.is_file():
                try:
                    launch_payload = json.loads(launch_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    # Let the strict profile writer provide the canonical
                    # fail-closed error for a malformed fresh launch artifact.
                    pass
                else:
                    if isinstance(launch_payload, dict) and isinstance(
                        launch_payload.get("tool"), str
                    ):
                        profile_tool = launch_payload["tool"]
            try:
                marker = (
                    directory / "index_complete.json"
                    if (directory / "index_complete.json").is_file()
                    else None
                )
                launch_schema = None
                if launch_path.is_file():
                    try:
                        launch_schema = json.loads(
                            launch_path.read_text(encoding="utf-8")
                        ).get("schema")
                    except (OSError, json.JSONDecodeError, AttributeError):
                        launch_schema = None
                if launch_schema == READMAP_LAUNCH_SCHEMA:
                    if manifest is None or run_dir is None or marker is None:
                        raise ValueError(
                            "Fresh read-map profile materialization requires the "
                            "manifest, run directory, and completion marker"
                        )
                    path, _ = write_pvc_readmap_client_profile(
                        directory,
                        tool=profile_tool,
                        manifest=manifest,
                        run_dir=run_dir,
                        index_marker_path=marker,
                    )
                else:
                    path, _ = write_pvc_index_client_profile(
                        directory,
                        tool=profile_tool,
                        threads=threads,
                        manifest=manifest,
                        run_dir=run_dir,
                        index_marker_path=marker,
                    )
            except ValueError as exc:
                # A launch artifact marks a fresh profiled execution.  Never
                # turn its malformed timing/provenance into a silently missing
                # measurement.  Historical logs have no such artifact and are
                # allowed to remain explicitly incomplete so genotype output
                # is not lost merely because an old diagnostic summary differs.
                if (directory / "process_launch_provenance.json").exists():
                    raise
                incomplete_reasons.append(
                    {"directory": str(directory), "reason": str(exc)}
                )
                continue
            return path

    if incomplete_reasons:
        # Best-effort sidecar for operators; the aggregate workflow profile
        # receives ``None`` and marks all index phases missing/incomplete.
        diagnostic_dir = unique_dirs[0]
        diagnostic_path = diagnostic_dir / "client_read_profile_incomplete.json"
        payload = {
            "schema": "pvc-index-client-profile-incomplete-v1",
            "status": "historical_profile_incomplete",
            "execution_provenance_verified": False,
            "reasons": incomplete_reasons,
        }
        try:
            diagnostic_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
    return None


def pvc_index_lookup_order(tool: str) -> list[str]:
    tools = list(PVC_INDEX_BINS)
    if tool not in tools:
        return tools
    return [tool, *(candidate for candidate in tools if candidate != tool)]


def read_pvc_index_complete(marker_path: str | Path) -> dict[str, Any]:
    path = Path(marker_path)
    with path.open("r", encoding="utf-8") as marker_file:
        metadata = json.load(marker_file)

    if not isinstance(metadata, dict):
        raise ValueError(f"Expected JSON object in {path}")

    return metadata


def pvc_index_metadata_paths(
    manifest: dict[str, Any],
    metadata: dict[str, Any],
) -> list[Path]:
    try:
        output_prefix = Path(metadata["index_prefix"])
        blocks_file = Path(metadata["blocks_file"])
    except KeyError as exc:
        raise ValueError(f"PVC index marker is missing required key: {exc.args[0]}") from exc

    artifact_paths = metadata.get("artifacts")
    if artifact_paths:
        artifacts = [Path(path) for path in artifact_paths]
    else:
        artifacts = expected_pvc_index_artifacts(manifest, output_prefix)

    return [blocks_file, *artifacts]


def pvc_index_metadata_is_complete(
    manifest: dict[str, Any],
    metadata: dict[str, Any],
) -> bool:
    return all(path.exists() for path in pvc_index_metadata_paths(manifest, metadata))


def pvc_index_metadata_matches_fresh_readmap(
    manifest: dict[str, Any], metadata: dict[str, Any]
) -> bool:
    """Prevent an old full-process marker from satisfying a fresh campaign."""
    if "pvc_read_count_mode" not in manifest:
        return metadata.get("readmap_boundary_schema") != READMAP_LAUNCH_SCHEMA
    try:
        expected_mode = resolve_read_count_mode(manifest)
        expected_public = public_index_prefix(manifest)
        expected_blocks = public_blocks_file(manifest)
        expected_blocks_artifact = public_blocks_artifact(manifest)
    except (ValueError, OSError):
        return False
    return (
        metadata.get("readmap_boundary_schema") == READMAP_LAUNCH_SCHEMA
        and metadata.get("read_count_mode") == expected_mode
        and Path(str(metadata.get("public_index_prefix", ""))).resolve()
        == expected_public
        and Path(str(metadata.get("blocks_file", ""))).resolve()
        == expected_blocks
        and metadata.get("blocks_source") == "frozen_public_manifest_artifact"
        and metadata.get("public_blocks_artifact") == expected_blocks_artifact
    )


def find_completed_pvc_index(
    run_dir: str | Path,
    tool: str,
    manifest: dict[str, Any],
) -> tuple[str, Path, dict[str, Any]] | None:
    for candidate_tool in pvc_index_lookup_order(tool):
        marker_path = pvc_index_complete_path(run_dir, candidate_tool)
        if not marker_path.exists():
            continue

        metadata = read_pvc_index_complete(marker_path)
        if pvc_index_metadata_is_complete(
            manifest, metadata
        ) and pvc_index_metadata_matches_fresh_readmap(manifest, metadata):
            metadata = dict(metadata)
            metadata["index_tool"] = candidate_tool
            metadata["index_marker"] = str(marker_path)
            return candidate_tool, marker_path, metadata

    return None


def validate_pvc_index_inputs(tool: str, manifest: dict[str, Any]) -> None:
    fresh_readmap = "pvc_read_count_mode" in manifest
    required = ["reference_fasta", "panel_subset", "fastq"]
    if fresh_readmap:
        required.extend(("pangenie_index_prefix", PUBLIC_BLOCKS_MANIFEST_KEY))
    missing_keys = [key for key in required if key not in manifest]
    if missing_keys:
        raise ValueError(f"Manifest is missing required PVC index keys: {missing_keys}")

    if tool not in PVC_INDEX_BINS:
        raise ValueError(f"Unknown PVC tool: {tool}")

    if fresh_readmap:
        resolve_read_count_mode(manifest)
        public_blocks_file(manifest)
        if not PVC_READMAP_BIN.exists():
            raise FileNotFoundError(
                f"Frozen PVC read-map binary not found for {tool}: {PVC_READMAP_BIN}"
            )
    else:
        pvc_index_bin = PVC_INDEX_BINS[tool]
        if not pvc_index_bin.exists():
            raise FileNotFoundError(
                f"PVC index binary not found for {tool}: {pvc_index_bin}"
            )


def write_pvc_index_complete(
    run_dir: str | Path,
    tool: str,
    output_prefix: str | Path,
    blocks_file: str | Path,
    exit_code: int,
    artifacts: list[Path],
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    marker_path = pvc_index_complete_path(run_dir, tool)
    payload: dict[str, Any] = {
        "schema": "pvc-index-complete-v2",
        "tool": tool,
        "index_prefix": str(Path(output_prefix).resolve()),
        "blocks_file": str(Path(blocks_file).resolve()),
        "exit_code": int(exit_code),
        "artifacts": [str(path.resolve()) for path in artifacts],
    }
    if extra_metadata:
        payload.update(extra_metadata)

    marker_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return marker_path


def run_pvc_index(
    tool: str,
    manifest: dict[str, Any],
    run_dir: str | Path,
    threads: int = 1,
    verbose: bool = False,
) -> Path:
    validate_pvc_index_inputs(tool, manifest)

    completed_index = find_completed_pvc_index(run_dir, tool, manifest)
    if completed_index is not None:
        source_tool, marker_path, metadata = completed_index
        source_profile = ensure_pvc_index_client_profile(
            metadata,
            tool=source_tool,
            threads=threads,
            manifest=manifest,
            run_dir=run_dir,
        )
        if source_tool != tool:
            output_dir = pvc_index_dir(run_dir, tool)
            output_dir.mkdir(parents=True, exist_ok=True)
            alias_marker = write_pvc_index_complete(
                run_dir,
                tool,
                metadata["index_prefix"],
                metadata["blocks_file"],
                int(metadata.get("exit_code", 0)),
                [Path(path) for path in metadata.get("artifacts", [])],
                extra_metadata={
                    "reused_from_tool": source_tool,
                    "reused_from_marker": str(marker_path),
                    "client_read_profile": (
                        str(source_profile) if source_profile is not None else None
                    ),
                    "public_index_prefix": metadata.get("public_index_prefix"),
                    "read_count_mode": metadata.get("read_count_mode"),
                    "readmap_boundary_schema": metadata.get(
                        "readmap_boundary_schema"
                    ),
                    "blocks_source": metadata.get("blocks_source"),
                    "public_blocks_artifact": metadata.get(
                        "public_blocks_artifact"
                    ),
                },
            )
            if verbose:
                print(
                    f"{tool} pvc-index reusing completed {source_tool} PVC index: "
                    f"{marker_path}"
                )
                print(f"{tool} pvc-index alias marker: {alias_marker}")
        elif verbose:
            print(f"{tool} pvc-index already complete: {marker_path}")

        return Path(metadata["index_prefix"])

    output_dir = pvc_index_dir(run_dir, tool)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = pvc_index_prefix(run_dir, tool)

    fresh_readmap = "pvc_read_count_mode" in manifest
    if fresh_readmap:
        result = run_pangenie_readmap(
            tool,
            manifest,
            output_dir,
            output_prefix,
            threads=threads,
            verbose=verbose,
            run_dir=run_dir,
        )
        missing_artifacts = missing_pvc_readmap_outputs(output_prefix)
        if result.returncode != 0:
            raise RuntimeError(
                f"{tool} pvc-index failed with exit code {result.returncode}; "
                f"see {output_dir / 'stderr.log'}"
            )
        if missing_artifacts:
            raise RuntimeError(
                f"{tool} pvc-index returned success but is missing expected artifacts: "
                f"{[str(path) for path in missing_artifacts]}"
            )

        # The frozen chromosome-level file is public campaign input.  Never
        # execute bcftools/PLINK in an 88-sample fresh preparation array.
        blocks_file = public_blocks_file(manifest)
        blocks_artifact = public_blocks_artifact(manifest)
        artifacts = expected_pvc_readmap_index_artifacts(manifest, output_prefix)
        planned_client_profile = output_dir.resolve() / "client_read_profile.json"
        marker_path = write_pvc_index_complete(
            run_dir,
            tool,
            output_prefix,
            blocks_file,
            result.returncode,
            artifacts,
            extra_metadata={
                "client_read_profile": str(planned_client_profile),
                "public_index_prefix": str(public_index_prefix(manifest)),
                "read_count_mode": resolve_read_count_mode(manifest),
                "readmap_boundary_schema": READMAP_LAUNCH_SCHEMA,
                "blocks_source": "frozen_public_manifest_artifact",
                "public_blocks_artifact": blocks_artifact,
            },
        )
        write_pvc_readmap_client_profile(
            output_dir,
            tool=tool,
            manifest=manifest,
            run_dir=run_dir,
            index_marker_path=marker_path,
        )
    else:
        # Historical manifests retain the original full PanGenie-process plus
        # per-run PLINK behavior for reproducibility.  They can never satisfy
        # the fresh read-map marker contract above.
        result = run_pangenie_process_index(
            tool,
            manifest,
            output_dir,
            output_prefix,
            threads=threads,
            verbose=verbose,
            run_dir=run_dir,
        )
        missing_artifacts = missing_pvc_index_artifacts(manifest, output_prefix)
        if result.returncode != 0 and missing_artifacts:
            raise RuntimeError(
                f"{tool} pvc-index failed with exit code {result.returncode}; "
                f"missing expected artifacts: {[str(path) for path in missing_artifacts]}; "
                f"see {output_dir / 'stderr.log'}"
            )
        blocks_file = run_plink_blocks(manifest, output_dir, verbose=verbose)
        artifacts = expected_pvc_index_artifacts(manifest, output_prefix)
        planned_client_profile = output_dir.resolve() / "client_read_profile.json"
        marker_path = write_pvc_index_complete(
            run_dir,
            tool,
            output_prefix,
            blocks_file,
            result.returncode,
            artifacts,
            extra_metadata={"client_read_profile": str(planned_client_profile)},
        )
        if result.returncode == 0:
            write_pvc_index_client_profile(
                output_dir,
                tool=tool,
                threads=threads,
                manifest=manifest,
                run_dir=run_dir,
                index_marker_path=marker_path,
            )

    if verbose:
        print(f"{tool} pvc-index prefix: {output_prefix}")
        print(f"{tool} pvc-index marker: {marker_path}")
        print(f"{tool} logs: {output_dir / 'stdout.log'}, {output_dir / 'stderr.log'}")

    return output_prefix
