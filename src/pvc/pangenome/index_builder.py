"""
Build PanGenie-compatible index from VCF, reference FASTA, and reads.

This replaces the need for pre-computed cereal/JSON files by building the
index directly from raw input files.
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

from pvc.pangenome.vcf_reader import VCFReader
from pvc.pangenome.fasta_reader import FastaReader
from pvc.pangenome.kmer_counter import KmerCounter, count_unique_kmers_jellyfish
from pvc.pangenome.unique_kmer_computer import (
    ComputedUniqueKmers,
    build_unique_kmers_map,
    count_unique_kmers_from_reads
)
from pvc.pangenome.variant import Variant


@dataclass
class IndexBuildResult:
    """Result of index building."""
    variants_per_chrom: Dict[str, List[Variant]]
    unique_kmers_per_chrom: Dict[str, List[ComputedUniqueKmers]]
    kmer_size: int
    kmer_coverage: int
    nr_paths: int
    sample_names: List[str]
    build_time_seconds: float

    def count_kmers_from_reads(
        self,
        reads_path: str,
        max_reads: Optional[int] = None,
        use_jellyfish: bool = True,
        threads: int = 4,
        verbose: bool = True
    ) -> None:
        """
        Count unique k-mers from reads and update the index.

        This is useful when the index was built without reads, or when
        you want to count k-mers from a different reads file.

        Args:
            reads_path: Path to reads file (FASTA or FASTQ)
            max_reads: Maximum reads to process (for testing, Python only)
            use_jellyfish: Use Jellyfish for faster counting (recommended)
            threads: Number of threads for Jellyfish
            verbose: Print progress messages
        """
        if use_jellyfish and max_reads is None:
            # Use Jellyfish for full file processing
            count_unique_kmers_jellyfish(
                unique_kmers_per_chrom=self.unique_kmers_per_chrom,
                reads_path=reads_path,
                kmer_size=self.kmer_size,
                threads=threads,
                verbose=verbose
            )
        else:
            # Use Python implementation (supports max_reads)
            count_unique_kmers_from_reads(
                unique_kmers_per_chrom=self.unique_kmers_per_chrom,
                reads_path=reads_path,
                kmer_size=self.kmer_size,
                max_reads=max_reads,
                verbose=verbose
            )

    def get_total_unique_kmers(self) -> int:
        """Get total number of unique k-mers across all variants."""
        return sum(
            sum(uk.size() for uk in uks)
            for uks in self.unique_kmers_per_chrom.values()
        )

    def get_total_variants(self) -> int:
        """Get total number of variants."""
        return sum(len(v) for v in self.variants_per_chrom.values())

    def save(self, output_path: str) -> None:
        """
        Save the index to a JSON file.

        Args:
            output_path: Path to save the index
        """
        data = {
            "kmer_size": self.kmer_size,
            "kmer_coverage": self.kmer_coverage,
            "nr_paths": self.nr_paths,
            "sample_names": self.sample_names,
            "build_time_seconds": self.build_time_seconds,
            "variants_per_chrom": {
                chrom: [_variant_to_dict(v) for v in variants]
                for chrom, variants in self.variants_per_chrom.items()
            },
            "unique_kmers_per_chrom": {
                chrom: [_unique_kmers_to_dict(uk) for uk in uks]
                for chrom, uks in self.unique_kmers_per_chrom.items()
            }
        }
        with open(output_path, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def load(index_path: str) -> 'IndexBuildResult':
        """
        Load an index from a JSON file.

        Args:
            index_path: Path to the saved index

        Returns:
            IndexBuildResult
        """
        with open(index_path, 'r') as f:
            data = json.load(f)

        variants_per_chrom = {
            chrom: [_dict_to_variant(vd) for vd in variants]
            for chrom, variants in data["variants_per_chrom"].items()
        }
        unique_kmers_per_chrom = {
            chrom: [_dict_to_unique_kmers(ukd) for ukd in uks]
            for chrom, uks in data["unique_kmers_per_chrom"].items()
        }

        return IndexBuildResult(
            variants_per_chrom=variants_per_chrom,
            unique_kmers_per_chrom=unique_kmers_per_chrom,
            kmer_size=data["kmer_size"],
            kmer_coverage=data["kmer_coverage"],
            nr_paths=data["nr_paths"],
            sample_names=data["sample_names"],
            build_time_seconds=data["build_time_seconds"]
        )


class IndexBuilder:
    """
    Build PanGenie-compatible index from raw input files.

    Usage:
        builder = IndexBuilder(
            vcf_path="pangenome.vcf",
            reference_path="ref.fa",
            reads_path="reads.fastq",
            kmer_size=31
        )
        result = builder.build()
    """

    def __init__(
        self,
        vcf_path: str,
        reference_path: str,
        reads_path: Optional[str] = None,
        kmer_size: int = 31,
        add_reference: bool = True,
        max_reads: Optional[int] = None
    ):
        """
        Args:
            vcf_path: Path to pangenome VCF
            reference_path: Path to reference FASTA
            reads_path: Path to reads (FASTA or FASTQ)
            kmer_size: K-mer size (default 31)
            add_reference: Add reference as a path
            max_reads: Maximum reads to process (for testing)
        """
        self.vcf_path = vcf_path
        self.reference_path = reference_path
        self.reads_path = reads_path
        self.kmer_size = kmer_size
        self.add_reference = add_reference
        self.max_reads = max_reads

    def build(self, verbose: bool = True) -> IndexBuildResult:
        """
        Build the index.

        Args:
            verbose: Print progress messages

        Returns:
            IndexBuildResult containing all computed data
        """
        start_time = time.time()

        # Step 1: Parse VCF and build variants
        if verbose:
            print(f"Step 1: Parsing VCF...")
        vcf_reader = VCFReader(
            vcf_path=self.vcf_path,
            reference_path=self.reference_path,
            kmer_size=self.kmer_size,
            add_reference=self.add_reference
        )

        variants_per_chrom = vcf_reader.variants_per_chromosome

        # Step 2: Count k-mers in pangenome
        # The pangenome consists of:
        # 1. Reference segments BETWEEN variants (not the variant regions themselves)
        # 2. Allele sequences for each variant (including flanks)
        # This ensures that reference allele k-mers that span variants can be unique.
        if verbose:
            print(f"Step 2: Counting k-mers in pangenome...")
        pangenome_counter = KmerCounter(k=self.kmer_size)

        # Count k-mers from reference segments between variants and allele sequences
        total_ref_segments = 0
        total_alleles = 0

        for chrom_name in vcf_reader.fasta_reader.get_sequence_names():
            prev_end = 0
            chrom_variants = variants_per_chrom.get(chrom_name, [])

            for variant in chrom_variants:
                start_pos = variant.start_position

                # Reference segment before this variant
                if start_pos > prev_end:
                    ref_seg = vcf_reader.fasta_reader.get_subsequence(
                        chrom_name, prev_end, start_pos
                    )
                    pangenome_counter.add_sequence(ref_seg.to_string())
                    total_ref_segments += 1

                # Add VCF-observed allele combinations (PanGenie behavior)
                for i in range(variant.nr_of_alleles()):
                    pangenome_counter.add_sequence(variant.get_allele_on_path(i))
                    total_alleles += 1

                prev_end = variant.get_end_position()

            # Reference segment after last variant
            chrom_size = vcf_reader.fasta_reader.get_size_of(chrom_name)
            if chrom_size > prev_end:
                ref_seg = vcf_reader.fasta_reader.get_subsequence(
                    chrom_name, prev_end, chrom_size
                )
                pangenome_counter.add_sequence(ref_seg.to_string())
                total_ref_segments += 1

        if verbose:
            print(f"  Reference segments: {total_ref_segments:,}")
            print(f"  Allele sequences: {total_alleles:,}")
            print(f"  Total distinct pangenome k-mers: {len(pangenome_counter):,}")

        # Step 3: Count k-mers in reads (if provided)
        read_counter = KmerCounter(k=self.kmer_size)
        kmer_coverage = 0

        if self.reads_path:
            if verbose:
                print(f"Step 3: Counting k-mers in reads...")

            if self.reads_path.endswith('.fa') or self.reads_path.endswith('.fasta') or \
               self.reads_path.endswith('.fa.gz') or self.reads_path.endswith('.fasta.gz'):
                read_counter.count_from_fasta(self.reads_path)
            else:
                read_counter.count_from_fastq(self.reads_path, max_reads=self.max_reads)

            if verbose:
                print(f"  Distinct read k-mers: {len(read_counter):,}")

            # Compute k-mer coverage from histogram peak (same as PanGenie)
            kmer_coverage = read_counter.compute_histogram_peak_coverage()
            if verbose:
                print(f"  K-mer coverage (histogram peak): {kmer_coverage}")
        else:
            if verbose:
                print(f"Step 3: Skipping read counting (no reads provided)")
            kmer_coverage = 30  # Default

        # Step 4: Compute unique k-mers
        if verbose:
            print(f"Step 4: Computing unique k-mers...")

        unique_kmers_per_chrom = build_unique_kmers_map(
            variants_per_chrom=variants_per_chrom,
            pangenome_counter=pangenome_counter,
            read_counter=read_counter,
            kmer_size=self.kmer_size,
            kmer_coverage=kmer_coverage
        )

        elapsed = time.time() - start_time

        if verbose:
            total_variants = sum(len(v) for v in variants_per_chrom.values())
            total_unique = sum(
                sum(uk.size() for uk in uks)
                for uks in unique_kmers_per_chrom.values()
            )
            print()
            print(f"Index build complete in {elapsed:.1f}s:")
            print(f"  Chromosomes: {list(variants_per_chrom.keys())}")
            print(f"  Total variants: {total_variants:,}")
            print(f"  Total unique k-mers: {total_unique:,}")
            print(f"  Paths: {vcf_reader.nr_paths}")

        return IndexBuildResult(
            variants_per_chrom=variants_per_chrom,
            unique_kmers_per_chrom=unique_kmers_per_chrom,
            kmer_size=self.kmer_size,
            kmer_coverage=kmer_coverage,
            nr_paths=vcf_reader.nr_paths,
            sample_names=vcf_reader.sample_names,
            build_time_seconds=elapsed
        )


def build_index(
    vcf_path: str,
    reference_path: str,
    reads_path: Optional[str] = None,
    kmer_size: int = 31,
    add_reference: bool = True,
    max_reads: Optional[int] = None,
    verbose: bool = True
) -> IndexBuildResult:
    """
    Convenience function to build index.

    Args:
        vcf_path: Path to pangenome VCF
        reference_path: Path to reference FASTA
        reads_path: Path to reads (FASTA or FASTQ), optional
        kmer_size: K-mer size
        add_reference: Add reference as a path
        max_reads: Maximum reads to process
        verbose: Print progress

    Returns:
        IndexBuildResult
    """
    builder = IndexBuilder(
        vcf_path=vcf_path,
        reference_path=reference_path,
        reads_path=reads_path,
        kmer_size=kmer_size,
        add_reference=add_reference,
        max_reads=max_reads
    )
    return builder.build(verbose=verbose)


# --- Serialization helpers ---

def _variant_to_dict(v: Variant) -> Dict[str, Any]:
    """Serialize a Variant to a dictionary."""
    return {
        "chromosome": v.chromosome,
        "start_position": v.start_position,
        "left_flank": v.left_flank.to_string() if hasattr(v.left_flank, 'to_string') else str(v.left_flank),
        "right_flank": v.right_flank.to_string() if hasattr(v.right_flank, 'to_string') else str(v.right_flank),
        "inner_flanks": [f.to_string() if hasattr(f, 'to_string') else str(f) for f in v.inner_flanks],
        "allele_sequences": [
            [seq.to_string() if hasattr(seq, 'to_string') else str(seq) for seq in bubble]
            for bubble in v.allele_sequences
        ],
        "allele_combinations": v.allele_combinations,
        "uncovered_alleles": v.uncovered_alleles,
        "paths": v.paths,
        "flanks_added": v.flanks_added,
    }


def _dict_to_variant(d: Dict[str, Any]) -> Variant:
    """Deserialize a Variant from a dictionary."""
    from pvc.pangenome.DNA_sequence import DnaSequence

    return Variant(
        chromosome=d["chromosome"],
        start_position=d["start_position"],
        left_flank=DnaSequence.from_string(d["left_flank"]),
        right_flank=DnaSequence.from_string(d["right_flank"]),
        inner_flanks=[DnaSequence.from_string(f) for f in d["inner_flanks"]],
        allele_sequences=[
            [DnaSequence.from_string(seq) for seq in bubble]
            for bubble in d["allele_sequences"]
        ],
        allele_combinations=d["allele_combinations"],
        uncovered_alleles=d.get("uncovered_alleles", [[]]),
        paths=d["paths"],
        flanks_added=d["flanks_added"],
    )


def _unique_kmers_to_dict(uk: ComputedUniqueKmers) -> Dict[str, Any]:
    """Serialize ComputedUniqueKmers to a dictionary."""
    return {
        "variant_pos": uk.variant_pos,
        "local_coverage": uk.local_coverage,
        "kmer_to_count": uk.kmer_to_count,
        "kmers": uk.kmers,
        "allele_to_kmers": {str(k): v for k, v in uk.allele_to_kmers.items()},
        "path_to_allele": uk.path_to_allele,
        "is_biallelic": uk.is_biallelic,
    }


def _dict_to_unique_kmers(d: Dict[str, Any]) -> ComputedUniqueKmers:
    """Deserialize ComputedUniqueKmers from a dictionary."""
    return ComputedUniqueKmers(
        variant_pos=d["variant_pos"],
        local_coverage=d["local_coverage"],
        kmer_to_count=d["kmer_to_count"],
        kmers=d["kmers"],
        allele_to_kmers={int(k): v for k, v in d["allele_to_kmers"].items()},
        path_to_allele=d["path_to_allele"],
        is_biallelic=d["is_biallelic"],
    )
