"""Exact application-level communication dependency-round certification.

This module deliberately certifies the CrypTen API boundary.  It does not
claim to observe Gloo/NCCL packets, chunks, headers, retransmissions, or the
algorithm selected internally by a collective backend.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


TRACE_SCHEMA = "crypten-application-dependency-trace-v1"
CERTIFICATE_SCHEMA = "pvc-application-dependency-round-certificate-v1"
MODELED_CLIENT_SCHEMA = "pvc-modeled-client-rounds-v1"

_TRACE_COLUMNS = {
    "kind",
    "src",
    "dst",
    "sequence",
    "participants_mask",
    "phase",
    "operation",
    "sent_bytes",
    "received_bytes",
    "application_messages",
    "predecessor_offset",
    "predecessor_count",
}
_ROOTED_SOURCE_COLLECTIVES = {
    "broadcast",
    "broadcast_object_size",
    "broadcast_object_payload",
    "scatter",
}
_ROOTED_DESTINATION_COLLECTIVES = {"reduce", "gather"}
_ROOTLESS_COLLECTIVES = {"all_reduce", "all_gather", "barrier"}


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}, got {value}")
    return value


def _participants(mask: int, world_size: int) -> list[int]:
    if mask <= 0 or mask >> world_size:
        raise ValueError(
            f"Dependency trace has invalid participant mask {mask:#x} "
            f"for world size {world_size}"
        )
    return [rank for rank in range(world_size) if mask & (1 << rank)]


def _message_id(key: tuple[int, int, int, int]) -> str:
    kind, first, second, sequence = key
    if kind == 0:
        return f"p2p:{first}->{second}:{sequence}"
    return f"collective:{first:#x}:{sequence}"


def _validate_rank_trace(
    value: Any,
    *,
    expected_rank: int,
    expected_world_size: int,
    canonical_phases: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or value.get("schema") != TRACE_SCHEMA:
        raise ValueError(f"Rank {expected_rank} lacks a {TRACE_SCHEMA} trace")
    if value.get("enabled") is not True or value.get("complete") is not True:
        raise ValueError(f"Rank {expected_rank} dependency trace is not complete")
    if value.get("errors") != []:
        raise ValueError(f"Rank {expected_rank} dependency trace reports errors")
    rank = _integer(value.get("rank"), "trace rank", minimum=0)
    world_size = _integer(value.get("world_size"), "trace world_size", minimum=1)
    if rank != expected_rank or world_size != expected_world_size:
        raise ValueError(
            f"Rank trace identity mismatch: expected {expected_rank}/{expected_world_size}, "
            f"got {rank}/{world_size}"
        )
    event_count = _integer(value.get("event_count"), "trace event_count", minimum=0)
    phase_table = value.get("phase_table")
    operation_table = value.get("operation_table")
    columns = value.get("columns")
    predecessors = value.get("predecessors")
    completed = value.get("completed")
    if (
        not isinstance(phase_table, list)
        or not phase_table
        or phase_table[0] is not None
        or any(not isinstance(item, str) for item in phase_table[1:])
        or len(set(phase_table[1:])) != len(phase_table[1:])
    ):
        raise ValueError(f"Rank {rank} dependency trace has an invalid phase table")
    if (
        not isinstance(operation_table, list)
        or any(not isinstance(item, str) or not item for item in operation_table)
        or len(set(operation_table)) != len(operation_table)
    ):
        raise ValueError(f"Rank {rank} dependency trace has an invalid operation table")
    if not isinstance(columns, dict) or set(columns) != _TRACE_COLUMNS:
        raise ValueError(f"Rank {rank} dependency trace has incompatible columns")
    if any(not isinstance(column, list) or len(column) != event_count for column in columns.values()):
        raise ValueError(f"Rank {rank} dependency trace column lengths disagree")
    if not isinstance(predecessors, list) or not isinstance(completed, list):
        raise ValueError(f"Rank {rank} dependency trace arrays are malformed")
    if len(completed) != event_count or any(item != 1 for item in completed):
        raise ValueError(f"Rank {rank} dependency trace contains incomplete requests")

    events: list[dict[str, Any]] = []
    expected_predecessor_offset = 0
    for local_id in range(event_count):
        kind = _integer(columns["kind"][local_id], "event kind", minimum=0)
        if kind not in (0, 1):
            raise ValueError(f"Rank {rank} event {local_id} has unknown kind {kind}")
        src = _integer(columns["src"][local_id], "event src")
        dst = _integer(columns["dst"][local_id], "event dst")
        sequence = _integer(columns["sequence"][local_id], "event sequence", minimum=0)
        mask = _integer(
            columns["participants_mask"][local_id],
            "event participants mask",
            minimum=1,
        )
        participant_ranks = _participants(mask, world_size)
        if rank not in participant_ranks:
            raise ValueError(f"Rank {rank} event {local_id} omits its own rank")
        phase_code = _integer(columns["phase"][local_id], "event phase code", minimum=0)
        operation_code = _integer(
            columns["operation"][local_id], "event operation code", minimum=0
        )
        if phase_code == 0 or phase_code >= len(phase_table):
            raise ValueError(
                f"Rank {rank} event {local_id} lacks a canonical phase assignment"
            )
        if operation_code >= len(operation_table):
            raise ValueError(f"Rank {rank} event {local_id} has an invalid operation code")
        phase = phase_table[phase_code]
        if phase not in canonical_phases:
            raise ValueError(
                f"Rank {rank} event {local_id} uses noncanonical phase {phase!r}"
            )
        sent_bytes = _integer(
            columns["sent_bytes"][local_id], "event sent_bytes", minimum=0
        )
        received_bytes = _integer(
            columns["received_bytes"][local_id], "event received_bytes", minimum=0
        )
        application_messages = _integer(
            columns["application_messages"][local_id],
            "event application_messages",
            minimum=0,
        )
        predecessor_offset = _integer(
            columns["predecessor_offset"][local_id],
            "event predecessor_offset",
            minimum=0,
        )
        predecessor_count = _integer(
            columns["predecessor_count"][local_id],
            "event predecessor_count",
            minimum=0,
        )
        if predecessor_offset != expected_predecessor_offset:
            raise ValueError(f"Rank {rank} dependency predecessor array is not contiguous")
        predecessor_end = predecessor_offset + predecessor_count
        if predecessor_end > len(predecessors):
            raise ValueError(f"Rank {rank} event {local_id} predecessor range is invalid")
        local_predecessors = [
            _integer(item, "event predecessor", minimum=0)
            for item in predecessors[predecessor_offset:predecessor_end]
        ]
        if (
            len(set(local_predecessors)) != len(local_predecessors)
            or any(item >= local_id for item in local_predecessors)
        ):
            raise ValueError(
                f"Rank {rank} event {local_id} has duplicate or non-prior predecessors"
            )
        expected_predecessor_offset = predecessor_end
        if kind == 0:
            if (
                src < 0
                or dst < 0
                or src == dst
                or sorted(participant_ranks) != sorted((src, dst))
                or rank not in (src, dst)
                or application_messages != 1
            ):
                raise ValueError(f"Rank {rank} event {local_id} has invalid P2P metadata")
            if rank == src and (received_bytes != 0) or rank == dst and sent_bytes != 0:
                raise ValueError(f"Rank {rank} event {local_id} has invalid P2P byte direction")
            key = (0, src, dst, sequence)
        else:
            if len(participant_ranks) < 2:
                raise ValueError(f"Rank {rank} event {local_id} has a singleton collective")
            key = (1, mask, -1, sequence)
        events.append(
            {
                "rank": rank,
                "local_id": local_id,
                "key": key,
                "phase": phase,
                "operation": operation_table[operation_code],
                "participants": participant_ranks,
                "src": src,
                "dst": dst,
                "sent_bytes": sent_bytes,
                "received_bytes": received_bytes,
                "application_messages": application_messages,
                "local_predecessors": local_predecessors,
            }
        )
    if expected_predecessor_offset != len(predecessors):
        raise ValueError(f"Rank {rank} dependency trace has unused predecessor values")
    return events


def _require_same(records: list[dict[str, Any]], field: str, message_id: str) -> Any:
    values = [record[field] for record in records]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"Matched message {message_id} disagrees on {field}")
    return values[0]


def _merge_message_node(
    key: tuple[int, int, int, int], records: list[dict[str, Any]], world_size: int
) -> dict[str, Any]:
    message_id = _message_id(key)
    phase = _require_same(records, "phase", message_id)
    participants = _require_same(records, "participants", message_id)
    src = _require_same(records, "src", message_id)
    dst = _require_same(records, "dst", message_id)
    application_messages = _require_same(records, "application_messages", message_id)
    if len(records) != len(participants) or sorted(record["rank"] for record in records) != participants:
        raise ValueError(f"Matched message {message_id} has missing or duplicate endpoints")
    per_rank = {
        rank: {"sent_bytes": 0, "received_bytes": 0}
        for rank in range(world_size)
    }
    for record in records:
        per_rank[record["rank"]] = {
            "sent_bytes": record["sent_bytes"],
            "received_bytes": record["received_bytes"],
        }
    total_sent = sum(value["sent_bytes"] for value in per_rank.values())
    total_received = sum(value["received_bytes"] for value in per_rank.values())
    if total_sent != total_received:
        raise ValueError(
            f"Matched message {message_id} sent/received mismatch: "
            f"{total_sent} != {total_received}"
        )
    operations = sorted({record["operation"] for record in records})
    if key[0] == 0:
        if len(records) != 2 or sorted(record["rank"] for record in records) != sorted((src, dst)):
            raise ValueError(f"P2P message {message_id} does not have exactly two endpoints")
        sender = next(record for record in records if record["rank"] == src)
        receiver = next(record for record in records if record["rank"] == dst)
        if sender["sent_bytes"] != receiver["received_bytes"]:
            raise ValueError(f"P2P message {message_id} payload sizes do not match")
    else:
        if len(operations) != 1:
            raise ValueError(f"Collective {message_id} endpoints disagree on operation")
        operation = operations[0]
        if operation in _ROOTED_SOURCE_COLLECTIVES:
            if src not in participants or dst != -1:
                raise ValueError(f"Collective {message_id} has invalid source metadata")
            for record in records:
                if record["rank"] == src:
                    if record["received_bytes"] != 0:
                        raise ValueError(f"Collective {message_id} source received bytes")
                elif record["sent_bytes"] != 0:
                    raise ValueError(f"Collective {message_id} receiver sent bytes")
        elif operation in _ROOTED_DESTINATION_COLLECTIVES:
            if dst not in participants or src != -1:
                raise ValueError(f"Collective {message_id} has invalid destination metadata")
            for record in records:
                if record["rank"] == dst:
                    if record["sent_bytes"] != 0:
                        raise ValueError(f"Collective {message_id} destination sent bytes")
                elif record["received_bytes"] != 0:
                    raise ValueError(f"Collective {message_id} sender received bytes")
        elif operation in _ROOTLESS_COLLECTIVES:
            if src != -1 or dst != -1:
                raise ValueError(f"Collective {message_id} unexpectedly has a root")
            if operation == "barrier":
                if total_sent or total_received or application_messages:
                    raise ValueError(f"Barrier {message_id} contains a payload")
            else:
                endpoint_values = {
                    (record["sent_bytes"], record["received_bytes"])
                    for record in records
                }
                if len(endpoint_values) != 1 or any(sent != received for sent, received in endpoint_values):
                    raise ValueError(f"Collective {message_id} endpoint payload contracts disagree")
        else:
            raise ValueError(
                f"Collective {message_id} uses unsupported operation {operation!r}"
            )
    return {
        "key": key,
        "message_id": message_id,
        "phase": phase,
        "operations": operations,
        "participants": participants,
        "application_messages": application_messages,
        "per_rank": per_rank,
        "total_sent_bytes": total_sent,
        "total_received_bytes": total_received,
        "predecessors": set(),
    }


def build_application_dependency_round_certificate(
    rank_traces: list[dict[str, Any]],
    rank_endpoint_payload_bytes: Iterable[int],
    canonical_phases: Iterable[str],
) -> dict[str, Any]:
    """Validate rank traces, merge messages, and derive exact causal rounds."""
    if not isinstance(rank_traces, list) or not rank_traces:
        raise ValueError("No rank dependency traces were supplied")
    world_size = len(rank_traces)
    phases = set(canonical_phases)
    if not phases:
        raise ValueError("Canonical dependency-trace phase set is empty")
    expected_endpoint_bytes = [
        _integer(value, "rank endpoint payload bytes", minimum=0)
        for value in rank_endpoint_payload_bytes
    ]
    if len(expected_endpoint_bytes) != world_size:
        raise ValueError("Rank endpoint payload-byte totals have the wrong length")

    rank_events = [
        _validate_rank_trace(
            trace,
            expected_rank=rank,
            expected_world_size=world_size,
            canonical_phases=phases,
        )
        for rank, trace in enumerate(rank_traces)
    ]
    records_by_key: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    local_to_key: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for events in rank_events:
        for event in events:
            records_by_key[event["key"]].append(event)
            local_to_key[(event["rank"], event["local_id"])] = event["key"]
    nodes = {
        key: _merge_message_node(key, records, world_size)
        for key, records in records_by_key.items()
    }
    for events in rank_events:
        for event in events:
            node = nodes[event["key"]]
            for local_predecessor in event["local_predecessors"]:
                predecessor_key = local_to_key[(event["rank"], local_predecessor)]
                if predecessor_key == event["key"]:
                    raise ValueError(
                        f"Message {node['message_id']} contains a self dependency"
                    )
                node["predecessors"].add(predecessor_key)

    successors: dict[tuple[int, int, int, int], set[tuple[int, int, int, int]]] = {
        key: set() for key in nodes
    }
    indegree = {key: len(node["predecessors"]) for key, node in nodes.items()}
    for key, node in nodes.items():
        for predecessor in node["predecessors"]:
            if predecessor not in nodes:
                raise ValueError(f"Message {node['message_id']} has an unknown predecessor")
            successors[predecessor].add(key)
    ready = sorted((key for key, degree in indegree.items() if degree == 0), key=_message_id)
    round_by_key: dict[tuple[int, int, int, int], int] = {}
    while ready:
        key = ready.pop(0)
        predecessor_rounds = [round_by_key[item] for item in nodes[key]["predecessors"]]
        round_by_key[key] = 1 + max(predecessor_rounds, default=0)
        for successor in sorted(successors[key], key=_message_id):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
        ready.sort(key=_message_id)
    if len(round_by_key) != len(nodes):
        raise ValueError("Application dependency trace contains a causal cycle")

    keys_by_round: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for key, round_number in round_by_key.items():
        keys_by_round[round_number].append(key)
    computing_party_rounds = []
    for round_number in sorted(keys_by_round):
        keys = sorted(keys_by_round[round_number], key=_message_id)
        round_nodes = [nodes[key] for key in keys]
        per_rank = []
        for rank in range(world_size):
            per_rank.append(
                {
                    "rank": rank,
                    "sent_bytes": sum(node["per_rank"][rank]["sent_bytes"] for node in round_nodes),
                    "received_bytes": sum(node["per_rank"][rank]["received_bytes"] for node in round_nodes),
                }
            )
        total_sent = sum(item["sent_bytes"] for item in per_rank)
        total_received = sum(item["received_bytes"] for item in per_rank)
        if total_sent != total_received:
            raise ValueError(f"Dependency round {round_number} does not balance bytes")
        participant_groups = sorted(
            {tuple(node["participants"]) for node in round_nodes}
        )
        round_phases = sorted({node["phase"] for node in round_nodes})
        if len(round_phases) != 1:
            raise ValueError(
                f"Dependency round {round_number} unexpectedly mixes canonical "
                f"phases {round_phases}"
            )
        computing_party_rounds.append(
            {
                "round": round_number,
                "phases": round_phases,
                "operations": sorted(
                    {operation for node in round_nodes for operation in node["operations"]}
                ),
                "participant_groups": [list(group) for group in participant_groups],
                "message_ids": [node["message_id"] for node in round_nodes],
                "application_messages": sum(
                    node["application_messages"] for node in round_nodes
                ),
                "event_count": len(round_nodes),
                "per_rank": per_rank,
                "total_sent_bytes": total_sent,
                "total_received_bytes": total_received,
                "unique_payload_bytes": total_sent,
            }
        )

    phase_summary: dict[str, dict[str, Any]] = {}
    for phase in sorted(phases):
        phase_rounds = [
            value for value in computing_party_rounds if value["phases"] == [phase]
        ]
        phase_summary[phase] = {
            "round_count": len(phase_rounds),
            "first_round": phase_rounds[0]["round"] if phase_rounds else None,
            "last_round": phase_rounds[-1]["round"] if phase_rounds else None,
            "event_count": sum(value["event_count"] for value in phase_rounds),
            "application_messages": sum(
                value["application_messages"] for value in phase_rounds
            ),
            "unique_payload_bytes": sum(
                value["unique_payload_bytes"] for value in phase_rounds
            ),
        }

    trace_endpoint_bytes = [
        sum(event["sent_bytes"] + event["received_bytes"] for event in events)
        for events in rank_events
    ]
    endpoint_reconciliation = [
        {
            "rank": rank,
            "trace_endpoint_payload_bytes": trace_endpoint_bytes[rank],
            "profile_endpoint_payload_bytes": expected_endpoint_bytes[rank],
            "exact_match": trace_endpoint_bytes[rank] == expected_endpoint_bytes[rank],
        }
        for rank in range(world_size)
    ]
    if not all(item["exact_match"] for item in endpoint_reconciliation):
        raise ValueError(
            "Application dependency trace bytes do not match whole-run communicator counters"
        )
    total_sent = sum(node["total_sent_bytes"] for node in nodes.values())
    total_received = sum(node["total_received_bytes"] for node in nodes.values())
    if total_sent != total_received or sum(trace_endpoint_bytes) != 2 * total_sent:
        raise ValueError("Application dependency trace global byte reconciliation failed")
    return {
        "schema": CERTIFICATE_SCHEMA,
        "schema_version": 1,
        "trace_schema": TRACE_SCHEMA,
        "exact_application_dependency_rounds_available": True,
        "scope": "observed_computing_party_application_payloads",
        "dependency_depth": max(round_by_key.values(), default=0),
        "computing_party_rounds": computing_party_rounds,
        "phase_summary": phase_summary,
        "reconciliation": {
            "status": "exact",
            "matched_event_count": len(nodes),
            "rank_event_counts": [len(events) for events in rank_events],
            "application_messages": sum(
                node["application_messages"] for node in nodes.values()
            ),
            "total_sent_bytes": total_sent,
            "total_received_bytes": total_received,
            "unique_payload_bytes": total_sent,
            "rank_endpoint_payload_bytes": endpoint_reconciliation,
            "endpoint_sum_equals_twice_unique_payload": True,
            "all_messages_matched": True,
            "all_rounds_byte_balanced": True,
        },
        "semantics": {
            "round_definition": (
                "one plus the maximum causal predecessor round, where a local "
                "request completion before a later API message issue creates a dependency"
            ),
            "byte_definition": (
                "CrypTen API application tensor/object payload bytes; excludes "
                "backend packets, chunks, headers, retransmissions, and collective internals"
            ),
            "unique_payload_bytes": (
                "sum of matched application sends; every byte is reconciled to exactly "
                "one receive and is not endpoint-double-counted"
            ),
            "collective_contract": (
                "collectives are one application dependency epoch per API invocation; "
                "batched tensors share that epoch"
            ),
            "modeled_client_rounds_included": False,
        },
    }


def build_modeled_client_rounds(
    phase_rows: list[dict[str, Any]], canonical_phase_order: Iterable[str]
) -> dict[str, Any]:
    """Validate and retain ordered, explicitly non-observed client calls."""
    order = {phase: index for index, phase in enumerate(canonical_phase_order)}
    events: list[dict[str, Any]] = []
    phase_upload = 0
    phase_download = 0
    phase_rounds = 0
    for row in phase_rows:
        phase = row.get("phase")
        if phase not in order:
            continue
        phase_upload += _integer(row.get("client_upload_bytes", 0), "client upload bytes", minimum=0)
        phase_download += _integer(row.get("client_download_bytes", 0), "client download bytes", minimum=0)
        phase_rounds += _integer(row.get("client_upload_rounds", 0), "client upload rounds", minimum=0)
        phase_rounds += _integer(row.get("client_download_rounds", 0), "client download rounds", minimum=0)
        raw_events = row.get("client_transfer_events", [])
        if not isinstance(raw_events, list):
            raise ValueError(f"Phase {phase!r} has malformed modeled client events")
        for raw in raw_events:
            if not isinstance(raw, dict):
                raise ValueError(f"Phase {phase!r} has malformed modeled client event")
            ordinal = _integer(raw.get("ordinal"), "modeled client ordinal", minimum=0)
            direction = raw.get("direction")
            payload_bytes = _integer(raw.get("payload_bytes"), "modeled client payload", minimum=0)
            protocol_rounds = _integer(raw.get("protocol_rounds"), "modeled client rounds", minimum=0)
            model = raw.get("model")
            if (
                raw.get("phase") != phase
                or direction not in {"upload", "download"}
                or not isinstance(model, str)
                or not model
            ):
                raise ValueError(f"Phase {phase!r} has inconsistent modeled client event")
            events.append(
                {
                    "ordinal": ordinal,
                    "phase": phase,
                    "direction": direction,
                    "payload_bytes": payload_bytes,
                    "protocol_rounds": protocol_rounds,
                    "model": model,
                    "classification": (
                        "modeled_external_client_upload"
                        if direction == "upload"
                        else "overlapping_modeled_client_download_metadata"
                    ),
                }
            )
    events.sort(key=lambda item: item["ordinal"])
    if [event["ordinal"] for event in events] != list(range(len(events))):
        raise ValueError("Modeled client event ordinals are not contiguous and unique")
    event_upload = sum(event["payload_bytes"] for event in events if event["direction"] == "upload")
    event_download = sum(event["payload_bytes"] for event in events if event["direction"] == "download")
    event_rounds = sum(event["protocol_rounds"] for event in events)
    if (event_upload, event_download, event_rounds) != (
        phase_upload,
        phase_download,
        phase_rounds,
    ):
        raise ValueError("Modeled client event totals do not reconcile to phase totals")
    return {
        "schema": MODELED_CLIENT_SCHEMA,
        "schema_version": 1,
        "observed": False,
        "included_in_computing_party_dependency_depth": False,
        "rounds": events,
        "reconciliation": {
            "status": "exact_model_reconciliation",
            "call_count": len(events),
            "modeled_protocol_rounds": event_rounds,
            "upload_payload_bytes": event_upload,
            "download_payload_bytes": event_download,
            "modeled_external_client_upload_payload_bytes": event_upload,
            "overlapping_download_metadata_payload_bytes": event_download,
            "total_directional_payload_bytes": event_upload + event_download,
            "phase_totals_exact_match": True,
        },
        "semantics": (
            "Ordered protocol-model calls recorded on rank 0; these are not "
            "observed network telemetry and are not merged into computing-party causal depth"
        ),
    }
