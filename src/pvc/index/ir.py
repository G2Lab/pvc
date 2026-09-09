from __future__ import annotations

"""Disk-backed PVC index IR dataset definitions."""

import gzip
import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO

from pvc.config import (
    max_blocks_per_superblock as DEFAULT_MAX_BLOCKS_PER_SUPERBLOCK,
    max_genotype_state_bytes as DEFAULT_MAX_GENOTYPE_STATE_BYTES,
    max_variants_per_superblock as DEFAULT_MAX_VARIANTS_PER_SUPERBLOCK,
)


IR_VERSION = "0.1.0"
GENOTYPE_SCORE_BYTES = 8


@dataclass(frozen=True)
class IrBuildConfig:
    max_variants_per_superblock: int = DEFAULT_MAX_VARIANTS_PER_SUPERBLOCK
    max_blocks_per_superblock: int = DEFAULT_MAX_BLOCKS_PER_SUPERBLOCK
    max_genotype_state_bytes: int = DEFAULT_MAX_GENOTYPE_STATE_BYTES


@dataclass(frozen=True)
class VariantRecord:
    index: int
    chrom: str
    pos: int
    end: int
    record_id: str
    ref_len: int
    alt_count: int
    variant_type: str


@dataclass(frozen=True)
class KmerWindow:
    index: int
    chrom: str
    start: int
    end: int
    unique_kmers: int
    unique_kmers_overhang: int


@dataclass(frozen=True)
class PlinkBlock:
    index: int
    chrom: str
    start: int
    end: int
    kb: float
    nsnps: int
    snps: str


@dataclass(frozen=True)
class Superblock:
    index: int
    chrom: str
    start: int
    end: int
    variant_start: int
    variant_end: int
    block_start: int
    block_end: int
    kmer_window_start: int
    kmer_window_end: int
    estimated_genotype_state_bytes: int

    @property
    def n_variants(self) -> int:
        return self.variant_end - self.variant_start

    @property
    def n_blocks(self) -> int:
        return self.block_end - self.block_start

    @property
    def n_kmer_windows(self) -> int:
        return self.kmer_window_end - self.kmer_window_start


@dataclass(frozen=True)
class IntervalIndex:
    records_by_chrom: dict[str, list[PlinkBlock | KmerWindow]]
    starts_by_chrom: dict[str, list[int]]
    ends_by_chrom: dict[str, list[int]]


def build_index_metadata_ir(
    *,
    tool: str,
    manifest: dict[str, Any],
    output_dir: str | Path,
    output_prefix: str | Path,
    blocks_file: str | Path,
    source_index_exit_code: int | None = None,
    config: IrBuildConfig | None = None,
    verbose: bool = False,
) -> Path:
    """Build a disk-backed PVC index IR from existing index artifacts.

    The IR is deliberately metadata-heavy and tensor-light at index time. It
    records public/reusable panel and block structure without materializing a
    giant chromosome-sized in-memory object.
    """

    build_config = config or IrBuildConfig()
    output_dir = Path(output_dir)
    output_prefix = Path(output_prefix)
    blocks_file = Path(blocks_file)
    ir_dir = output_dir / "ir"
    superblocks_dir = ir_dir / "superblocks"
    superblocks_dir.mkdir(parents=True, exist_ok=True)
    clear_generated_superblock_metadata(superblocks_dir)

    panel_vcf = Path(manifest["panel_subset"])
    kmer_tsv = output_prefix.with_name(f"{output_prefix.name}_{manifest_chromosome(manifest)}_kmers.tsv.gz")

    variants = read_variants(panel_vcf)
    kmer_windows = read_kmer_windows(kmer_tsv)
    blocks = read_plink_blocks(blocks_file)
    superblocks = make_superblocks(variants, blocks, kmer_windows, build_config)

    write_variants_tsv(ir_dir / "variants.tsv", variants)
    write_kmer_windows_tsv(ir_dir / "kmer_windows.tsv", kmer_windows)
    write_blocks_tsv(ir_dir / "blocks.tsv", blocks)
    write_superblocks_tsv(ir_dir / "superblocks.tsv", superblocks)
    write_superblock_metadata(superblocks_dir, superblocks)

    artifacts = collect_artifacts(
        manifest=manifest,
        output_prefix=output_prefix,
        blocks_file=blocks_file,
        kmer_tsv=kmer_tsv,
    )
    write_json(ir_dir / "artifacts.json", artifacts)
    write_json(
        ir_dir / "metadata.json",
        {
            "ir_version": IR_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "sample": manifest.get("sample", "sample"),
            "interval": manifest.get("interval"),
            "chromosome": manifest_chromosome(manifest),
            "source_index_exit_code": source_index_exit_code,
            "config": {
                "max_variants_per_superblock": build_config.max_variants_per_superblock,
                "max_blocks_per_superblock": build_config.max_blocks_per_superblock,
                "max_genotype_state_bytes": build_config.max_genotype_state_bytes,
            },
            "counts": {
                "variants": len(variants),
                "kmer_windows": len(kmer_windows),
                "plink_blocks": len(blocks),
                "superblocks": len(superblocks),
            },
            "files": {
                "variants": "variants.tsv",
                "kmer_windows": "kmer_windows.tsv",
                "blocks": "blocks.tsv",
                "superblocks": "superblocks.tsv",
                "superblock_metadata_dir": "superblocks",
                "artifacts": "artifacts.json",
            },
        },
    )

    if verbose:
        print(f"PVC IR directory: {ir_dir}")
        print(
            "PVC IR counts: "
            f"{len(variants)} variants, {len(kmer_windows)} kmer windows, "
            f"{len(blocks)} blocks, {len(superblocks)} superblocks"
        )

    return ir_dir


def manifest_chromosome(manifest: dict[str, Any]) -> str:
    interval = manifest.get("interval", "")
    if isinstance(interval, str) and ":" in interval:
        return interval.split(":", 1)[0]
    if isinstance(interval, str) and interval.startswith("chr"):
        return interval
    return "chr20"


def read_variants(panel_vcf: str | Path) -> list[VariantRecord]:
    records: list[VariantRecord] = []
    with open_text(panel_vcf) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                continue
            chrom = fields[0]
            pos = int(fields[1])
            record_id = fields[2]
            ref = fields[3]
            alts = [] if fields[4] in {"", "."} else fields[4].split(",")
            end = pos + len(ref) - 1
            records.append(
                VariantRecord(
                    index=len(records),
                    chrom=chrom,
                    pos=pos,
                    end=end,
                    record_id=record_id if record_id != "." else f"{chrom}:{pos}:{ref}>{fields[4]}",
                    ref_len=len(ref),
                    alt_count=len(alts),
                    variant_type=classify_variant(ref, alts),
                )
            )
    if not records:
        raise ValueError(f"No variants found in panel VCF: {panel_vcf}")
    return records


def classify_variant(ref: str, alts: list[str]) -> str:
    if not alts:
        return "reference"
    allele_lengths = [len(ref), *(len(alt) for alt in alts)]
    if len(ref) == 1 and all(len(alt) == 1 for alt in alts):
        return "snp"
    if all(len(alt) > len(ref) for alt in alts):
        kind = "insertion"
    elif all(len(alt) < len(ref) for alt in alts):
        kind = "deletion"
    else:
        kind = "complex"
    max_delta = max(abs(len(alt) - len(ref)) for alt in alts)
    if max(allele_lengths) < 20 and max_delta < 20:
        return f"small-{kind}"
    if max(allele_lengths) < 50 and max_delta < 50:
        return f"midsize-{kind}"
    return f"large-{kind}"


def read_kmer_windows(kmer_tsv_gz: str | Path) -> list[KmerWindow]:
    path = Path(kmer_tsv_gz)
    if not path.exists():
        raise FileNotFoundError(f"PVC kmer TSV not found: {path}")

    windows: list[KmerWindow] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        expected = [
            "#chromosome",
            "start",
            "end",
            "unique_kmers",
            "unique_kmers_overhang",
        ]
        if header[:5] != expected:
            raise ValueError(f"Unexpected kmer TSV header in {path}: {header}")

        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                continue
            windows.append(
                KmerWindow(
                    index=len(windows),
                    chrom=fields[0],
                    start=int(fields[1]),
                    end=int(fields[2]),
                    unique_kmers=count_comma_values(fields[3]),
                    unique_kmers_overhang=count_comma_values(fields[4]),
                )
            )
    return windows


def count_comma_values(value: str) -> int:
    if value in {"", "."}:
        return 0
    return value.count(",") + 1


def read_plink_blocks(blocks_file: str | Path) -> list[PlinkBlock]:
    path = Path(blocks_file)
    if not path.exists():
        raise FileNotFoundError(f"PLINK blocks file not found: {path}")

    blocks: list[PlinkBlock] = []
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().split()
        expected = ["CHR", "BP1", "BP2", "KB", "NSNPS", "SNPS"]
        if header[:6] != expected:
            raise ValueError(f"Unexpected PLINK blocks header in {path}: {header}")

        for line in handle:
            fields = line.split(maxsplit=5)
            if len(fields) < 6:
                continue
            chrom = fields[0]
            if not chrom.startswith("chr"):
                chrom = f"chr{chrom}"
            blocks.append(
                PlinkBlock(
                    index=len(blocks),
                    chrom=chrom,
                    start=int(fields[1]),
                    end=int(fields[2]),
                    kb=float(fields[3]),
                    nsnps=int(fields[4]),
                    snps=fields[5].strip(),
                )
            )
    return blocks


def make_superblocks(
    variants: list[VariantRecord],
    blocks: list[PlinkBlock],
    kmer_windows: list[KmerWindow],
    config: IrBuildConfig,
) -> list[Superblock]:
    superblocks: list[Superblock] = []
    block_index = build_interval_index(blocks)
    kmer_window_index = build_interval_index(kmer_windows)
    genotype_state_prefix = cumulative_genotype_states(variants)
    offset = 0

    while offset < len(variants):
        chrom = variants[offset].chrom
        start = offset
        end = offset
        interval_start = variants[offset].pos
        interval_end = variants[offset].end

        while end < len(variants):
            variant = variants[end]
            if variant.chrom != chrom:
                break

            candidate_end = max(interval_end, variant.end)
            candidate_variant_count = end - start + 1
            candidate_block_count = count_overlapping_records(
                block_index,
                chrom,
                interval_start,
                candidate_end,
            )
            candidate_bytes = estimate_genotype_state_bytes(
                genotype_state_prefix[end + 1] - genotype_state_prefix[start],
                candidate_block_count,
            )

            if (
                candidate_variant_count > config.max_variants_per_superblock
                or candidate_block_count > config.max_blocks_per_superblock
                or candidate_bytes > config.max_genotype_state_bytes
            ):
                if end == start:
                    end += 1
                    interval_end = candidate_end
                break

            end += 1
            interval_end = candidate_end

        if end == start:
            end += 1

        interval_end = max(variant.end for variant in variants[start:end])
        block_indices = overlapping_indices(block_index, chrom, interval_start, interval_end)
        kmer_indices = overlapping_indices(kmer_window_index, chrom, interval_start, interval_end)
        superblocks.append(
            Superblock(
                index=len(superblocks),
                chrom=chrom,
                start=interval_start,
                end=interval_end,
                variant_start=start,
                variant_end=end,
                block_start=min(block_indices) if block_indices else 0,
                block_end=max(block_indices) + 1 if block_indices else 0,
                kmer_window_start=min(kmer_indices) if kmer_indices else 0,
                kmer_window_end=max(kmer_indices) + 1 if kmer_indices else 0,
                estimated_genotype_state_bytes=estimate_genotype_state_bytes(
                    genotype_state_prefix[end] - genotype_state_prefix[start],
                    len(block_indices),
                ),
            )
        )
        offset = end

    return superblocks


def build_interval_index(records: Iterable[PlinkBlock | KmerWindow]) -> IntervalIndex:
    records_by_chrom: dict[str, list[PlinkBlock | KmerWindow]] = {}
    for record in records:
        records_by_chrom.setdefault(record.chrom, []).append(record)

    starts_by_chrom: dict[str, list[int]] = {}
    ends_by_chrom: dict[str, list[int]] = {}
    for chrom, chrom_records in records_by_chrom.items():
        chrom_records.sort(key=lambda record: (record.start, record.end, record.index))
        starts_by_chrom[chrom] = [record.start for record in chrom_records]
        ends_by_chrom[chrom] = sorted(record.end for record in chrom_records)

    return IntervalIndex(
        records_by_chrom=records_by_chrom,
        starts_by_chrom=starts_by_chrom,
        ends_by_chrom=ends_by_chrom,
    )


def count_overlapping_records(
    index: IntervalIndex,
    chrom: str,
    start: int,
    end: int,
) -> int:
    starts = index.starts_by_chrom.get(chrom, [])
    ends = index.ends_by_chrom.get(chrom, [])
    return bisect_right(starts, end) - bisect_left(ends, start)


def overlapping_indices(
    index: IntervalIndex,
    chrom: str,
    start: int,
    end: int,
) -> list[int]:
    starts = index.starts_by_chrom.get(chrom, [])
    records = index.records_by_chrom.get(chrom, [])
    candidate_stop = bisect_right(starts, end)
    return [
        record.index
        for record in records[:candidate_stop]
        if overlaps(record.start, record.end, start, end)
    ]


def overlaps(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start <= right_end and right_start <= left_end


def cumulative_genotype_states(variants: list[VariantRecord]) -> list[int]:
    prefix = [0]
    for variant in variants:
        prefix.append(prefix[-1] + diploid_genotype_count(variant.alt_count))
    return prefix


def diploid_genotype_count(alt_count: int) -> int:
    n_alleles = alt_count + 1
    return n_alleles * (n_alleles + 1) // 2


def estimate_genotype_state_bytes(n_genotype_states: int, n_blocks: int) -> int:
    return max(1, n_genotype_states) * max(1, n_blocks) * GENOTYPE_SCORE_BYTES


def clear_generated_superblock_metadata(superblocks_dir: Path) -> None:
    # The 3 private-genotyping parties run build_genotyping_plan concurrently on
    # the same run dir, so two can race to unlink the same file. missing_ok makes
    # the clear idempotent (the goal is "file gone") instead of crashing on the
    # loser of the race.
    for path in superblocks_dir.glob("sb_*.json"):
        path.unlink(missing_ok=True)


def collect_artifacts(
    *,
    manifest: dict[str, Any],
    output_prefix: Path,
    blocks_file: Path,
    kmer_tsv: Path,
) -> dict[str, Any]:
    chromosome = manifest_chromosome(manifest)
    artifacts = {
        "manifest": manifest,
        "panel_subset": file_metadata(Path(manifest["panel_subset"])),
        "reference_fasta": file_metadata(Path(manifest["reference_fasta"])),
        "fastq": file_metadata(Path(manifest["fastq"])),
        "truth": file_metadata(Path(manifest["truth"])) if "truth" in manifest else None,
        "pangenie_process": {
            "output_prefix": str(output_prefix),
            "path_segments_fasta": file_metadata(output_prefix.with_name(f"{output_prefix.name}_path_segments.fasta")),
            "unique_kmers_map_cereal": file_metadata(output_prefix.with_name(f"{output_prefix.name}_UniqueKmersMap.cereal")),
            "graph_cereal": file_metadata(output_prefix.with_name(f"{output_prefix.name}_{chromosome}_Graph.cereal")),
            "graph_json": file_metadata(output_prefix.with_name(f"{output_prefix.name}_{chromosome}_Graph.json")),
            "kmers_tsv_gz": file_metadata(kmer_tsv),
        },
        "plink_blocks": file_metadata(blocks_file),
    }
    return artifacts


def file_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def write_variants_tsv(path: Path, variants: list[VariantRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("variant_index\tchrom\tpos\tend\tid\tref_len\talt_count\tvariant_type\n")
        for variant in variants:
            handle.write(
                f"{variant.index}\t{variant.chrom}\t{variant.pos}\t{variant.end}\t"
                f"{variant.record_id}\t{variant.ref_len}\t{variant.alt_count}\t"
                f"{variant.variant_type}\n"
            )


def write_kmer_windows_tsv(path: Path, windows: list[KmerWindow]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "kmer_window_index\tchrom\tstart\tend\tunique_kmers\tunique_kmers_overhang\n"
        )
        for window in windows:
            handle.write(
                f"{window.index}\t{window.chrom}\t{window.start}\t{window.end}\t"
                f"{window.unique_kmers}\t{window.unique_kmers_overhang}\n"
            )


def write_blocks_tsv(path: Path, blocks: list[PlinkBlock]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("block_index\tchrom\tstart\tend\tkb\tnsnps\tsnps\n")
        for block in blocks:
            handle.write(
                f"{block.index}\t{block.chrom}\t{block.start}\t{block.end}\t"
                f"{block.kb}\t{block.nsnps}\t{block.snps}\n"
            )


def write_superblocks_tsv(path: Path, superblocks: list[Superblock]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "superblock_index\tchrom\tstart\tend\tn_variants\tvariant_start\tvariant_end\t"
            "n_blocks\tblock_start\tblock_end\tn_kmer_windows\tkmer_window_start\t"
            "kmer_window_end\testimated_genotype_state_bytes\n"
        )
        for superblock in superblocks:
            handle.write(
                f"{superblock.index}\t{superblock.chrom}\t{superblock.start}\t"
                f"{superblock.end}\t{superblock.n_variants}\t"
                f"{superblock.variant_start}\t{superblock.variant_end}\t"
                f"{superblock.n_blocks}\t{superblock.block_start}\t{superblock.block_end}\t"
                f"{superblock.n_kmer_windows}\t{superblock.kmer_window_start}\t"
                f"{superblock.kmer_window_end}\t{superblock.estimated_genotype_state_bytes}\n"
            )


def write_superblock_metadata(superblocks_dir: Path, superblocks: list[Superblock]) -> None:
    for superblock in superblocks:
        path = superblocks_dir / f"sb_{superblock.index:06d}.json"
        write_json(
            path,
            {
                "superblock_index": superblock.index,
                "chrom": superblock.chrom,
                "start": superblock.start,
                "end": superblock.end,
                "variant_start": superblock.variant_start,
                "variant_end": superblock.variant_end,
                "block_start": superblock.block_start,
                "block_end": superblock.block_end,
                "kmer_window_start": superblock.kmer_window_start,
                "kmer_window_end": superblock.kmer_window_end,
                "estimated_genotype_state_bytes": superblock.estimated_genotype_state_bytes,
            },
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def open_text(path: str | Path) -> TextIO:
    raw_path = Path(path)
    if raw_path.suffix == ".gz":
        return gzip.open(raw_path, "rt", encoding="utf-8")
    return raw_path.open("r", encoding="utf-8")


# ---------------------------------------------------------------------------
# Bubble-genotyping IR
# ---------------------------------------------------------------------------
#
# The adapter above was an index/provenance summary. The standalone PVC
# genotyper needs a narrower IR: one row per variant bubble plus compact arrays
# for the read-counted UniqueKmers object used by PVC genotyping.

BUBBLE_IR_VERSION = "0.2.0"


@dataclass(frozen=True)
class BubbleIrRecord:
    bubble_id: int
    chrom: str
    variant_index: int
    position: int
    local_coverage: float
    n_kmers: int
    n_alleles: int
    max_allele_id: int
    n_paths: int
    kmer_start: int
    kmer_end: int
    path_start: int
    path_end: int
    undefined_start: int
    undefined_end: int
    kmer_allele_start: int
    kmer_allele_end: int


@dataclass(frozen=True)
class GenotypingIrBlock:
    block_id: int
    chrom: str
    start: int
    end: int
    source: str
    plink_block_index: int
    bubble_indices: tuple[int, ...]

    @property
    def n_bubbles(self) -> int:
        return len(self.bubble_indices)


@dataclass(frozen=True)
class GenotypingIrSuperblock:
    superblock_id: int
    chrom: str
    start: int
    end: int
    block_start: int
    block_end: int
    block_indices: tuple[int, ...]
    n_bubbles: int
    estimated_genotype_state_bytes: int

    @property
    def n_blocks(self) -> int:
        return len(self.block_indices)


@dataclass(frozen=True)
class IrAllele:
    is_undefined: bool = False


class IrUniqueKmer:
    """IR-backed bubble view with the API used by the genotype scorer."""

    def __init__(self, dataset: "GenotypingIrDataset", record: BubbleIrRecord):
        self._dataset = dataset
        self._record = record
        self.variant_pos = record.position
        self.local_coverage = record.local_coverage
        self.current_index = record.n_kmers
        self.path_to_allele = [
            int(value)
            for value in dataset.path_alleles[record.path_start:record.path_end]
        ]
        undefined = {
            int(value)
            for value in dataset.undefined_alleles[
                record.undefined_start:record.undefined_end
            ]
        }
        self.alleles = {
            allele_id: IrAllele(is_undefined=allele_id in undefined)
            for allele_id in range(record.max_allele_id + 1)
        }
        self.kmer_to_count = dataset.kmer_counts[record.kmer_start:record.kmer_end]

    def size(self) -> int:
        return self.current_index

    def get_coverage(self) -> float:
        return self.local_coverage

    def get_readcount_of(self, kmer_index: int) -> int:
        return int(self.kmer_to_count[kmer_index])

    def get_variant_position(self) -> int:
        return self.variant_pos

    def is_undefined_allele(self, allele_id: int) -> bool:
        allele = self.alleles.get(int(allele_id))
        return bool(allele and allele.is_undefined)

    def kmer_on_allele(self, kmer_index: int, allele_id: int) -> bool:
        global_kmer_index = self._record.kmer_start + int(kmer_index)
        start = int(self._dataset.kmer_allele_offsets[global_kmer_index])
        end = int(self._dataset.kmer_allele_offsets[global_kmer_index + 1])
        if start == end:
            return False
        memberships = self._dataset.kmer_alleles[start:end]
        return bool((memberships == int(allele_id)).any())


@dataclass
class GenotypingIrDataset:
    root: Path
    bubbles: list[BubbleIrRecord]
    blocks: list[GenotypingIrBlock]
    superblocks: list[GenotypingIrSuperblock]
    kmer_counts: Any
    bubble_kmer_offsets: Any
    kmer_allele_offsets: Any
    kmer_alleles: Any
    path_allele_offsets: Any
    path_alleles: Any
    undefined_allele_offsets: Any
    undefined_alleles: Any

    def __post_init__(self) -> None:
        self.blocks_by_chrom: dict[str, list[GenotypingIrBlock]] = {}
        for block in self.blocks:
            self.blocks_by_chrom.setdefault(block.chrom, []).append(block)

        self.superblocks_by_chrom: dict[str, list[GenotypingIrSuperblock]] = {}
        for superblock in self.superblocks:
            self.superblocks_by_chrom.setdefault(superblock.chrom, []).append(superblock)

        unique_kmers: dict[str, list[IrUniqueKmer | None]] = {}
        for record in self.bubbles:
            chrom_unique_kmers = unique_kmers.setdefault(record.chrom, [])
            while len(chrom_unique_kmers) <= record.variant_index:
                chrom_unique_kmers.append(None)
            chrom_unique_kmers[record.variant_index] = IrUniqueKmer(self, record)

        self.unique_kmers = {
            chrom: [unique_kmer for unique_kmer in chrom_unique_kmers if unique_kmer is not None]
            for chrom, chrom_unique_kmers in unique_kmers.items()
        }

    def blocks_for_superblock(
        self,
        superblock: GenotypingIrSuperblock,
    ) -> list[GenotypingIrBlock]:
        return [self.blocks[block_id] for block_id in superblock.block_indices]


@dataclass
class IrUniqueKmersMap:
    unique_kmers: dict[str, list[IrUniqueKmer]]


@dataclass
class GenotypingPlan:
    """Lightweight control-plane IR for PVC genotyping.

    This standardizes superblock -> block -> bubble traversal without copying
    k-mer counts, allele memberships, or path data out of UniqueKmersMap.
    """

    root: Path | None
    bubbles: list[BubbleIrRecord]
    blocks: list[GenotypingIrBlock]
    superblocks: list[GenotypingIrSuperblock]

    def __post_init__(self) -> None:
        self.blocks_by_chrom: dict[str, list[GenotypingIrBlock]] = {}
        for block in self.blocks:
            self.blocks_by_chrom.setdefault(block.chrom, []).append(block)

        self.superblocks_by_chrom: dict[str, list[GenotypingIrSuperblock]] = {}
        for superblock in self.superblocks:
            self.superblocks_by_chrom.setdefault(superblock.chrom, []).append(superblock)

    def blocks_for_superblock(
        self,
        superblock: GenotypingIrSuperblock,
    ) -> list[GenotypingIrBlock]:
        return [self.blocks[block_id] for block_id in superblock.block_indices]


def build_genotyping_plan(
    unique_kmers_map: Any,
    blocks_file: str | Path,
    output_dir: str | Path | None = None,
    *,
    sample: str | None = None,
    source: Mapping[str, Any] | None = None,
    config: IrBuildConfig | None = None,
    verbose: bool = False,
) -> GenotypingPlan:
    """Build a lightweight IR plan for block/superblock traversal."""

    build_config = config or IrBuildConfig()
    bubbles = make_bubble_records(unique_kmers_map)
    plink_blocks = read_plink_blocks(blocks_file)
    blocks = make_genotyping_blocks(bubbles, plink_blocks)
    superblocks = make_genotyping_superblocks(bubbles, blocks, build_config)
    root = Path(output_dir) if output_dir is not None else None

    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        clear_generated_superblock_metadata(root / "superblocks")
        clear_stale_genotyping_arrays(root)
        (root / "superblocks").mkdir(parents=True, exist_ok=True)
        _write_bubble_records(root / "bubbles.tsv", bubbles)
        write_genotyping_blocks_tsv(root / "blocks.tsv", blocks)
        write_genotyping_superblocks_tsv(root / "superblocks.tsv", superblocks)
        write_genotyping_superblock_metadata(root / "superblocks", superblocks)
        write_json(
            root / "metadata.json",
            {
                "ir_version": BUBBLE_IR_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "kind": "control-plane-genotyping",
                "sample": sample,
                "source": {
                    **dict(source or {}),
                    "blocks_file": str(blocks_file),
                },
                "counts": {
                    "bubbles": len(bubbles),
                    "blocks": len(blocks),
                    "plink_blocks": sum(1 for block in blocks if block.source == "plink"),
                    "singleton_blocks": sum(1 for block in blocks if block.source == "singleton"),
                    "superblocks": len(superblocks),
                },
                "files": {
                    "bubbles": "bubbles.tsv",
                    "blocks": "blocks.tsv",
                    "superblocks": "superblocks.tsv",
                    "superblock_metadata_dir": "superblocks",
                },
            },
        )

    if verbose:
        print(
            "PVC genotyping control IR counts: "
            f"{len(bubbles)} bubbles, {len(blocks)} blocks, "
            f"{len(superblocks)} superblocks"
        )

    return GenotypingPlan(
        root=root,
        bubbles=bubbles,
        blocks=blocks,
        superblocks=superblocks,
    )


def make_bubble_records(unique_kmers_map: Any) -> list[BubbleIrRecord]:
    records: list[BubbleIrRecord] = []
    unique_kmers_by_chrom = getattr(unique_kmers_map, "unique_kmers")
    for chrom, chrom_unique_kmers in unique_kmers_by_chrom.items():
        for variant_index, unique_kmer in enumerate(chrom_unique_kmers):
            allele_ids = _unique_allele_ids(unique_kmer)
            records.append(
                BubbleIrRecord(
                    bubble_id=len(records),
                    chrom=str(chrom),
                    variant_index=variant_index,
                    position=_variant_position(unique_kmer),
                    local_coverage=float(_local_coverage(unique_kmer)),
                    n_kmers=_unique_kmer_count(unique_kmer),
                    n_alleles=len(allele_ids),
                    max_allele_id=max(allele_ids) if allele_ids else -1,
                    n_paths=len(_path_to_allele(unique_kmer)),
                    kmer_start=0,
                    kmer_end=0,
                    path_start=0,
                    path_end=0,
                    undefined_start=0,
                    undefined_end=0,
                    kmer_allele_start=0,
                    kmer_allele_end=0,
                )
            )
    return records


def clear_stale_genotyping_arrays(ir_dir: Path) -> None:
    stale_probability_table = ir_dir / "probability_table.json"
    if stale_probability_table.exists():
        stale_probability_table.unlink()

    arrays_dir = ir_dir / "arrays"
    if not arrays_dir.exists():
        return
    for filename in (
        "kmer_counts.npy",
        "bubble_kmer_offsets.npy",
        "kmer_allele_offsets.npy",
        "kmer_alleles.npy",
        "path_allele_offsets.npy",
        "path_alleles.npy",
        "undefined_allele_offsets.npy",
        "undefined_alleles.npy",
    ):
        path = arrays_dir / filename
        if path.exists():
            path.unlink()
    try:
        arrays_dir.rmdir()
    except OSError:
        pass


def build_genotyping_ir(
    unique_kmers_map: Any,
    blocks_file: str | Path,
    output_dir: str | Path,
    *,
    probability_table: Any | None = None,
    sample: str | None = None,
    source: Mapping[str, Any] | None = None,
    config: IrBuildConfig | None = None,
    verbose: bool = False,
) -> Path:
    """Build the disk-backed IR used by PVC genotyping.

    The IR is organized as:
      - bubble: one variant bubble from the read-counted UniqueKmersMap
      - block: one PLINK LD block containing one or more bubbles, plus
        singleton blocks for bubbles outside any PLINK block
      - superblock: a bounded group of one or more blocks
    """

    build_config = config or IrBuildConfig()
    ir_dir = write_bubble_genotyping_ir(
        unique_kmers_map,
        output_dir,
        probability_table=probability_table,
        sample=sample,
        source=source,
    )
    bubbles = read_bubble_records(ir_dir / "bubbles.tsv")
    plink_blocks = read_plink_blocks(blocks_file)
    blocks = make_genotyping_blocks(bubbles, plink_blocks)
    superblocks = make_genotyping_superblocks(bubbles, blocks, build_config)
    superblocks_dir = ir_dir / "superblocks"
    superblocks_dir.mkdir(parents=True, exist_ok=True)
    clear_generated_superblock_metadata(superblocks_dir)

    write_genotyping_blocks_tsv(ir_dir / "blocks.tsv", blocks)
    write_genotyping_superblocks_tsv(ir_dir / "superblocks.tsv", superblocks)
    write_genotyping_superblock_metadata(superblocks_dir, superblocks)
    update_genotyping_ir_metadata(ir_dir, blocks, superblocks, blocks_file)

    if verbose:
        print(
            "PVC genotyping IR counts: "
            f"{len(bubbles)} bubbles, {len(blocks)} blocks, "
            f"{len(superblocks)} superblocks"
        )

    return ir_dir


def write_bubble_genotyping_ir(
    unique_kmers_map: Any,
    output_dir: str | Path,
    *,
    probability_table: Any | None = None,
    sample: str | None = None,
    source: Mapping[str, Any] | None = None,
) -> Path:
    """Write the compact per-bubble genotyping IR.

    This stores only what the plaintext genotype scoring kernels need:
    read counts for each unique kmer, the alleles each kmer maps to, path
    haplotypes as path-to-allele vectors, undefined allele flags, local
    coverage, and probability table parameters.
    """

    import numpy as np

    ir_dir = Path(output_dir)
    arrays_dir = ir_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    records: list[BubbleIrRecord] = []
    kmer_counts: list[int] = []
    bubble_kmer_offsets: list[int] = [0]
    kmer_allele_offsets: list[int] = [0]
    kmer_alleles: list[int] = []
    path_allele_offsets: list[int] = [0]
    path_alleles: list[int] = []
    undefined_allele_offsets: list[int] = [0]
    undefined_alleles: list[int] = []

    bubble_id = 0
    unique_kmers_by_chrom = getattr(unique_kmers_map, "unique_kmers")
    for chrom, chrom_unique_kmers in unique_kmers_by_chrom.items():
        for variant_index, unique_kmer in enumerate(chrom_unique_kmers):
            allele_ids = _unique_allele_ids(unique_kmer)
            n_kmers = _unique_kmer_count(unique_kmer)
            kmer_start = len(kmer_counts)
            kmer_allele_start = len(kmer_alleles)

            for kmer_index in range(n_kmers):
                kmer_counts.append(_kmer_read_count(unique_kmer, kmer_index))
                for allele_id in allele_ids:
                    if _kmer_on_allele(unique_kmer, kmer_index, allele_id):
                        kmer_alleles.append(allele_id)
                kmer_allele_offsets.append(len(kmer_alleles))

            bubble_kmer_offsets.append(len(kmer_counts))

            path_start = len(path_alleles)
            path_alleles.extend(_path_to_allele(unique_kmer))
            path_allele_offsets.append(len(path_alleles))

            undefined_start = len(undefined_alleles)
            undefined_alleles.extend(
                allele_id
                for allele_id in allele_ids
                if _is_undefined_allele(unique_kmer, allele_id)
            )
            undefined_allele_offsets.append(len(undefined_alleles))

            records.append(
                BubbleIrRecord(
                    bubble_id=bubble_id,
                    chrom=str(chrom),
                    variant_index=variant_index,
                    position=_variant_position(unique_kmer),
                    local_coverage=float(_local_coverage(unique_kmer)),
                    n_kmers=n_kmers,
                    n_alleles=len(allele_ids),
                    max_allele_id=max(allele_ids) if allele_ids else -1,
                    n_paths=len(_path_to_allele(unique_kmer)),
                    kmer_start=kmer_start,
                    kmer_end=len(kmer_counts),
                    path_start=path_start,
                    path_end=len(path_alleles),
                    undefined_start=undefined_start,
                    undefined_end=len(undefined_alleles),
                    kmer_allele_start=kmer_allele_start,
                    kmer_allele_end=len(kmer_alleles),
                )
            )
            bubble_id += 1

    _write_bubble_records(ir_dir / "bubbles.tsv", records)
    _save_npy(arrays_dir / "kmer_counts.npy", kmer_counts, np.uint32)
    _save_npy(arrays_dir / "bubble_kmer_offsets.npy", bubble_kmer_offsets, np.uint64)
    _save_npy(arrays_dir / "kmer_allele_offsets.npy", kmer_allele_offsets, np.uint64)
    _save_npy(arrays_dir / "kmer_alleles.npy", kmer_alleles, np.uint16)
    _save_npy(arrays_dir / "path_allele_offsets.npy", path_allele_offsets, np.uint64)
    _save_npy(arrays_dir / "path_alleles.npy", path_alleles, np.uint16)
    _save_npy(arrays_dir / "undefined_allele_offsets.npy", undefined_allele_offsets, np.uint64)
    _save_npy(arrays_dir / "undefined_alleles.npy", undefined_alleles, np.uint16)

    probability_spec = _probability_table_spec(probability_table)
    write_json(ir_dir / "probability_table.json", probability_spec)
    write_json(
        ir_dir / "metadata.json",
        {
            "ir_version": BUBBLE_IR_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "kind": "bubble-genotyping",
            "sample": sample,
            "source": dict(source or {}),
            "counts": {
                "bubbles": len(records),
                "kmers": len(kmer_counts),
                "kmer_allele_memberships": len(kmer_alleles),
                "path_alleles": len(path_alleles),
                "undefined_alleles": len(undefined_alleles),
            },
            "files": {
                "bubbles": "bubbles.tsv",
                "probability_table": "probability_table.json",
                "arrays": {
                    "kmer_counts": "arrays/kmer_counts.npy",
                    "bubble_kmer_offsets": "arrays/bubble_kmer_offsets.npy",
                    "kmer_allele_offsets": "arrays/kmer_allele_offsets.npy",
                    "kmer_alleles": "arrays/kmer_alleles.npy",
                    "path_allele_offsets": "arrays/path_allele_offsets.npy",
                    "path_alleles": "arrays/path_alleles.npy",
                    "undefined_allele_offsets": "arrays/undefined_allele_offsets.npy",
                    "undefined_alleles": "arrays/undefined_alleles.npy",
                },
            },
        },
    )
    return ir_dir


def load_genotyping_ir(ir_dir: str | Path) -> GenotypingIrDataset:
    """Load a disk-backed genotyping IR with mmap-backed arrays."""

    import numpy as np

    root = Path(ir_dir)
    arrays_dir = root / "arrays"
    return GenotypingIrDataset(
        root=root,
        bubbles=read_bubble_records(root / "bubbles.tsv"),
        blocks=read_genotyping_blocks_tsv(root / "blocks.tsv"),
        superblocks=read_genotyping_superblocks_tsv(root / "superblocks.tsv"),
        kmer_counts=np.load(arrays_dir / "kmer_counts.npy", mmap_mode="r"),
        bubble_kmer_offsets=np.load(arrays_dir / "bubble_kmer_offsets.npy", mmap_mode="r"),
        kmer_allele_offsets=np.load(arrays_dir / "kmer_allele_offsets.npy", mmap_mode="r"),
        kmer_alleles=np.load(arrays_dir / "kmer_alleles.npy", mmap_mode="r"),
        path_allele_offsets=np.load(arrays_dir / "path_allele_offsets.npy", mmap_mode="r"),
        path_alleles=np.load(arrays_dir / "path_alleles.npy", mmap_mode="r"),
        undefined_allele_offsets=np.load(arrays_dir / "undefined_allele_offsets.npy", mmap_mode="r"),
        undefined_alleles=np.load(arrays_dir / "undefined_alleles.npy", mmap_mode="r"),
    )


def load_ir_unique_kmers_map(ir_dir: str | Path) -> IrUniqueKmersMap:
    dataset = load_genotyping_ir(ir_dir)
    return IrUniqueKmersMap(unique_kmers=dataset.unique_kmers)


def make_genotyping_blocks(
    bubbles: list[BubbleIrRecord],
    plink_blocks: list[PlinkBlock],
) -> list[GenotypingIrBlock]:
    bubbles_by_chrom: dict[str, list[BubbleIrRecord]] = {}
    for bubble in bubbles:
        bubbles_by_chrom.setdefault(bubble.chrom, []).append(bubble)

    for chrom_bubbles in bubbles_by_chrom.values():
        chrom_bubbles.sort(key=lambda bubble: (bubble.position, bubble.variant_index))

    blocks: list[GenotypingIrBlock] = []
    assigned: set[tuple[str, int]] = set()

    for plink_block in plink_blocks:
        chrom_bubbles = bubbles_by_chrom.get(plink_block.chrom, [])
        bubble_indices = tuple(
            bubble.variant_index
            for bubble in chrom_bubbles
            if plink_block.start <= bubble.position <= plink_block.end
        )
        if not bubble_indices:
            continue

        assigned.update((plink_block.chrom, index) for index in bubble_indices)
        blocks.append(
            GenotypingIrBlock(
                block_id=-1,
                chrom=plink_block.chrom,
                start=plink_block.start,
                end=plink_block.end,
                source="plink",
                plink_block_index=plink_block.index,
                bubble_indices=bubble_indices,
            )
        )

    for chrom, chrom_bubbles in bubbles_by_chrom.items():
        for bubble in chrom_bubbles:
            if (chrom, bubble.variant_index) in assigned:
                continue
            blocks.append(
                GenotypingIrBlock(
                    block_id=-1,
                    chrom=chrom,
                    start=bubble.position,
                    end=bubble.position,
                    source="singleton",
                    plink_block_index=-1,
                    bubble_indices=(bubble.variant_index,),
                )
            )

    blocks.sort(key=lambda block: (block.chrom, block.start, block.end, block.source))
    return [
        GenotypingIrBlock(
            block_id=index,
            chrom=block.chrom,
            start=block.start,
            end=block.end,
            source=block.source,
            plink_block_index=block.plink_block_index,
            bubble_indices=block.bubble_indices,
        )
        for index, block in enumerate(blocks)
    ]


def make_genotyping_superblocks(
    bubbles: list[BubbleIrRecord],
    blocks: list[GenotypingIrBlock],
    config: IrBuildConfig,
) -> list[GenotypingIrSuperblock]:
    bubble_lookup = {
        (bubble.chrom, bubble.variant_index): bubble
        for bubble in bubbles
    }
    blocks_by_chrom: dict[str, list[GenotypingIrBlock]] = {}
    for block in blocks:
        blocks_by_chrom.setdefault(block.chrom, []).append(block)

    superblocks: list[GenotypingIrSuperblock] = []
    for chrom, chrom_blocks in blocks_by_chrom.items():
        chrom_blocks.sort(key=lambda block: block.block_id)
        offset = 0
        while offset < len(chrom_blocks):
            selected: list[GenotypingIrBlock] = []
            n_bubbles = 0
            estimated_bytes = 0

            while offset + len(selected) < len(chrom_blocks):
                candidate = chrom_blocks[offset + len(selected)]
                candidate_bubbles = len(candidate.bubble_indices)
                candidate_bytes = estimate_ir_block_state_bytes(candidate, bubble_lookup)

                if selected and (
                    len(selected) + 1 > config.max_blocks_per_superblock
                    or n_bubbles + candidate_bubbles > config.max_variants_per_superblock
                    or estimated_bytes + candidate_bytes > config.max_genotype_state_bytes
                ):
                    break

                selected.append(candidate)
                n_bubbles += candidate_bubbles
                estimated_bytes += candidate_bytes

                if not selected:
                    break

                if (
                    len(selected) >= config.max_blocks_per_superblock
                    or n_bubbles >= config.max_variants_per_superblock
                    or estimated_bytes >= config.max_genotype_state_bytes
                ):
                    break

            if not selected:
                selected.append(chrom_blocks[offset])
                n_bubbles = len(selected[0].bubble_indices)
                estimated_bytes = estimate_ir_block_state_bytes(selected[0], bubble_lookup)

            block_indices = tuple(block.block_id for block in selected)
            superblocks.append(
                GenotypingIrSuperblock(
                    superblock_id=len(superblocks),
                    chrom=chrom,
                    start=min(block.start for block in selected),
                    end=max(block.end for block in selected),
                    block_start=min(block_indices),
                    block_end=max(block_indices) + 1,
                    block_indices=block_indices,
                    n_bubbles=n_bubbles,
                    estimated_genotype_state_bytes=estimated_bytes,
                )
            )
            offset += len(selected)

    return superblocks


def estimate_ir_block_state_bytes(
    block: GenotypingIrBlock,
    bubble_lookup: Mapping[tuple[str, int], BubbleIrRecord],
) -> int:
    n_genotype_states = 0
    for bubble_index in block.bubble_indices:
        bubble = bubble_lookup[(block.chrom, bubble_index)]
        n_genotype_states += diploid_genotype_count(max(0, bubble.max_allele_id))
    return max(1, n_genotype_states) * GENOTYPE_SCORE_BYTES


def read_bubble_records(path: str | Path) -> list[BubbleIrRecord]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        expected = list(BubbleIrRecord.__dataclass_fields__.keys())
        if header != expected:
            raise ValueError(f"Unexpected bubble IR header in {path}: {header}")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(expected):
                continue
            values = dict(zip(expected, fields))
            records.append(
                BubbleIrRecord(
                    bubble_id=int(values["bubble_id"]),
                    chrom=values["chrom"],
                    variant_index=int(values["variant_index"]),
                    position=int(values["position"]),
                    local_coverage=float(values["local_coverage"]),
                    n_kmers=int(values["n_kmers"]),
                    n_alleles=int(values["n_alleles"]),
                    max_allele_id=int(values["max_allele_id"]),
                    n_paths=int(values["n_paths"]),
                    kmer_start=int(values["kmer_start"]),
                    kmer_end=int(values["kmer_end"]),
                    path_start=int(values["path_start"]),
                    path_end=int(values["path_end"]),
                    undefined_start=int(values["undefined_start"]),
                    undefined_end=int(values["undefined_end"]),
                    kmer_allele_start=int(values["kmer_allele_start"]),
                    kmer_allele_end=int(values["kmer_allele_end"]),
                )
            )
    return records


def write_genotyping_blocks_tsv(path: Path, blocks: list[GenotypingIrBlock]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "block_id\tchrom\tstart\tend\tsource\tplink_block_index\t"
            "n_bubbles\tbubble_indices\n"
        )
        for block in blocks:
            handle.write(
                f"{block.block_id}\t{block.chrom}\t{block.start}\t{block.end}\t"
                f"{block.source}\t{block.plink_block_index}\t{block.n_bubbles}\t"
                f"{format_int_tuple(block.bubble_indices)}\n"
            )


def read_genotyping_blocks_tsv(path: str | Path) -> list[GenotypingIrBlock]:
    blocks = []
    with Path(path).open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        expected = [
            "block_id",
            "chrom",
            "start",
            "end",
            "source",
            "plink_block_index",
            "n_bubbles",
            "bubble_indices",
        ]
        if header != expected:
            raise ValueError(f"Unexpected genotyping block IR header in {path}: {header}")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(expected):
                continue
            values = dict(zip(expected, fields))
            blocks.append(
                GenotypingIrBlock(
                    block_id=int(values["block_id"]),
                    chrom=values["chrom"],
                    start=int(values["start"]),
                    end=int(values["end"]),
                    source=values["source"],
                    plink_block_index=int(values["plink_block_index"]),
                    bubble_indices=parse_int_tuple(values["bubble_indices"]),
                )
            )
    return blocks


def write_genotyping_superblocks_tsv(
    path: Path,
    superblocks: list[GenotypingIrSuperblock],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "superblock_id\tchrom\tstart\tend\tblock_start\tblock_end\t"
            "n_blocks\tn_bubbles\testimated_genotype_state_bytes\tblock_indices\n"
        )
        for superblock in superblocks:
            handle.write(
                f"{superblock.superblock_id}\t{superblock.chrom}\t"
                f"{superblock.start}\t{superblock.end}\t"
                f"{superblock.block_start}\t{superblock.block_end}\t"
                f"{superblock.n_blocks}\t{superblock.n_bubbles}\t"
                f"{superblock.estimated_genotype_state_bytes}\t"
                f"{format_int_tuple(superblock.block_indices)}\n"
            )


def read_genotyping_superblocks_tsv(path: str | Path) -> list[GenotypingIrSuperblock]:
    superblocks = []
    with Path(path).open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        expected = [
            "superblock_id",
            "chrom",
            "start",
            "end",
            "block_start",
            "block_end",
            "n_blocks",
            "n_bubbles",
            "estimated_genotype_state_bytes",
            "block_indices",
        ]
        if header != expected:
            raise ValueError(f"Unexpected genotyping superblock IR header in {path}: {header}")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(expected):
                continue
            values = dict(zip(expected, fields))
            superblocks.append(
                GenotypingIrSuperblock(
                    superblock_id=int(values["superblock_id"]),
                    chrom=values["chrom"],
                    start=int(values["start"]),
                    end=int(values["end"]),
                    block_start=int(values["block_start"]),
                    block_end=int(values["block_end"]),
                    block_indices=parse_int_tuple(values["block_indices"]),
                    n_bubbles=int(values["n_bubbles"]),
                    estimated_genotype_state_bytes=int(values["estimated_genotype_state_bytes"]),
                )
            )
    return superblocks


def write_genotyping_superblock_metadata(
    superblocks_dir: Path,
    superblocks: list[GenotypingIrSuperblock],
) -> None:
    for superblock in superblocks:
        write_json(
            superblocks_dir / f"sb_{superblock.superblock_id:06d}.json",
            {
                "superblock_id": superblock.superblock_id,
                "chrom": superblock.chrom,
                "start": superblock.start,
                "end": superblock.end,
                "block_start": superblock.block_start,
                "block_end": superblock.block_end,
                "block_indices": list(superblock.block_indices),
                "n_blocks": superblock.n_blocks,
                "n_bubbles": superblock.n_bubbles,
                "estimated_genotype_state_bytes": superblock.estimated_genotype_state_bytes,
            },
        )


def update_genotyping_ir_metadata(
    ir_dir: Path,
    blocks: list[GenotypingIrBlock],
    superblocks: list[GenotypingIrSuperblock],
    blocks_file: str | Path,
) -> None:
    metadata_path = ir_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["kind"] = "block-genotyping"
    metadata["counts"]["blocks"] = len(blocks)
    metadata["counts"]["plink_blocks"] = sum(1 for block in blocks if block.source == "plink")
    metadata["counts"]["singleton_blocks"] = sum(
        1 for block in blocks if block.source == "singleton"
    )
    metadata["counts"]["superblocks"] = len(superblocks)
    metadata["files"]["blocks"] = "blocks.tsv"
    metadata["files"]["superblocks"] = "superblocks.tsv"
    metadata["files"]["superblock_metadata_dir"] = "superblocks"
    metadata["source"] = {
        **dict(metadata.get("source") or {}),
        "blocks_file": str(blocks_file),
    }
    write_json(metadata_path, metadata)


def format_int_tuple(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in value.split(",") if part)


def write_bubble_likelihoods(
    genotype_results: Mapping[str, list[Any]],
    output_path: str | Path,
    *,
    unique_kmers_map: Any | None = None,
    value_domain: str = "probability",
) -> Path:
    """Write genotype likelihoods per variant bubble before VCF splitting."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    unique_kmers_by_chrom = (
        getattr(unique_kmers_map, "unique_kmers", {}) if unique_kmers_map is not None else {}
    )

    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "chrom\tbubble_id\tvariant_index\tposition\tlocal_coverage\t"
            "n_unique_kmers\tallele1\tallele2\tlikelihood\n"
        )
        bubble_id = 0
        for chrom, chrom_results in genotype_results.items():
            chrom_unique_kmers = unique_kmers_by_chrom.get(chrom, [])
            for variant_index, result in enumerate(chrom_results):
                unique_kmer = (
                    chrom_unique_kmers[variant_index]
                    if variant_index < len(chrom_unique_kmers)
                    else None
                )
                position = _variant_position(unique_kmer) if unique_kmer is not None else -1
                coverage = (
                    float(_local_coverage(unique_kmer))
                    if unique_kmer is not None
                    else float(_result_coverage(result))
                )
                n_unique_kmers = (
                    _unique_kmer_count(unique_kmer)
                    if unique_kmer is not None
                    else _result_unique_kmers(result)
                )
                for (allele1, allele2), likelihood in _stored_likelihoods(result).items():
                    handle.write(
                        f"{chrom}\t{bubble_id}\t{variant_index}\t{position}\t"
                        f"{coverage}\t{n_unique_kmers}\t{int(allele1)}\t"
                        f"{int(allele2)}\t{float(likelihood):.17g}\n"
                    )
                bubble_id += 1

    write_json(
        path.with_suffix(path.suffix + ".metadata.json"),
        {
            "ir_version": BUBBLE_IR_VERSION,
            "kind": "bubble-likelihoods",
            "value_domain": value_domain,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "file": str(path),
        },
    )
    return path


def _write_bubble_records(path: Path, records: list[BubbleIrRecord]) -> None:
    fields = list(BubbleIrRecord.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(fields) + "\n")
        for record in records:
            handle.write("\t".join(str(getattr(record, field)) for field in fields) + "\n")


def _save_npy(path: Path, values: list[int], dtype: Any) -> None:
    import numpy as np

    np.save(path, np.asarray(values, dtype=dtype))


def _unique_allele_ids(unique_kmer: Any) -> list[int]:
    alleles = getattr(unique_kmer, "alleles", {})
    if isinstance(alleles, Mapping):
        return sorted(int(allele_id) for allele_id in alleles.keys())
    allele_ids = unique_kmer.get_allele_ids()
    if isinstance(allele_ids, Mapping):
        return sorted(int(allele_id) for allele_id in allele_ids.keys())
    return sorted(int(allele_id) for allele_id in allele_ids)


def _unique_kmer_count(unique_kmer: Any) -> int:
    if hasattr(unique_kmer, "size"):
        return int(unique_kmer.size())
    return int(getattr(unique_kmer, "current_index"))


def _kmer_read_count(unique_kmer: Any, kmer_index: int) -> int:
    if hasattr(unique_kmer, "get_readcount_of"):
        return int(unique_kmer.get_readcount_of(kmer_index))
    return int(unique_kmer.kmer_to_count[kmer_index])


def _kmer_on_allele(unique_kmer: Any, kmer_index: int, allele_id: int) -> bool:
    try:
        return bool(unique_kmer.kmer_on_allele(kmer_index, allele_id))
    except (KeyError, IndexError):
        return False


def _path_to_allele(unique_kmer: Any) -> list[int]:
    return [int(allele_id) for allele_id in getattr(unique_kmer, "path_to_allele", [])]


def _is_undefined_allele(unique_kmer: Any, allele_id: int) -> bool:
    if hasattr(unique_kmer, "is_undefined_allele"):
        return bool(unique_kmer.is_undefined_allele(allele_id))
    allele = getattr(unique_kmer, "alleles", {}).get(allele_id)
    return bool(getattr(allele, "is_undefined", False))


def _variant_position(unique_kmer: Any) -> int:
    if hasattr(unique_kmer, "get_variant_position"):
        return int(unique_kmer.get_variant_position())
    return int(getattr(unique_kmer, "variant_pos"))


def _local_coverage(unique_kmer: Any) -> float:
    if hasattr(unique_kmer, "get_coverage"):
        return float(unique_kmer.get_coverage())
    return float(getattr(unique_kmer, "local_coverage"))


def _probability_table_spec(probability_table: Any | None) -> dict[str, Any]:
    if probability_table is None:
        return {
            "stored": "parameters-only",
            "note": "Probability table is deterministic from these parameters; full table is not duplicated in IR.",
        }
    return {
        "stored": "parameters-only",
        "cov_min": int(getattr(probability_table, "cov_min")),
        "cov_max": int(getattr(probability_table, "cov_max")),
        "count_max": int(getattr(probability_table, "count_max")),
        "regularization_const": float(
            getattr(probability_table, "regularization_const", 0.0)
        ),
        "value_domain": "log P(read_count | copy_number)",
    }


def _stored_likelihoods(result: Any) -> Mapping[tuple[int, int], float]:
    if hasattr(result, "get_stored_likelihoods"):
        return result.get_stored_likelihoods()
    return getattr(result, "_genotype_to_likelihood", {})


def _result_coverage(result: Any) -> int:
    if hasattr(result, "coverage"):
        return int(result.coverage())
    return int(getattr(result, "local_coverage", 0))


def _result_unique_kmers(result: Any) -> int:
    if hasattr(result, "nr_unique_kmers"):
        return int(result.nr_unique_kmers())
    return int(getattr(result, "unique_kmers", 0))
