"""
VCF Reader - Parse VCF files and create Variant objects.

This is a Python port of PanGenie's VariantReader class.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Iterator
from pathlib import Path

from pvc.pangenome.DNA_sequence import DnaSequence
from pvc.pangenome.fasta_reader import FastaReader
from pvc.pangenome.variant import Variant


@dataclass
class VCFRecord:
    """A single VCF record (line)."""
    chrom: str
    pos: int  # 0-based position
    id: str
    ref: DnaSequence
    alts: List[DnaSequence]
    qual: str
    filter: str
    info: Dict[str, str]
    format: List[str]
    samples: List[str]  # raw genotype strings like "0|1"

    @property
    def alleles(self) -> List[DnaSequence]:
        """All alleles including ref."""
        return [self.ref] + self.alts

    @property
    def end_pos(self) -> int:
        """End position (0-based, exclusive)."""
        return self.pos + self.ref.size()


def parse_info_field(info_str: str) -> Dict[str, str]:
    """Parse INFO field into dictionary."""
    if info_str == ".":
        return {}
    result = {}
    for field in info_str.split(";"):
        if "=" in field:
            key, value = field.split("=", 1)
            result[key] = value
        else:
            result[field] = "True"
    return result


def parse_vcf_line(line: str, num_samples: int) -> Optional[VCFRecord]:
    """Parse a single VCF data line."""
    tokens = line.strip().split("\t")
    if len(tokens) < 9 + num_samples:
        return None

    chrom = tokens[0]
    pos = int(tokens[1]) - 1  # Convert to 0-based
    vid = tokens[2]
    ref = DnaSequence.from_string(tokens[3])

    # Parse ALT alleles
    alt_str = tokens[4]
    if alt_str == ".":
        alts = []
    else:
        alts = [DnaSequence.from_string(a) for a in alt_str.split(",")]

    qual = tokens[5]
    filt = tokens[6]
    info = parse_info_field(tokens[7])
    fmt = tokens[8].split(":")
    samples = tokens[9:9 + num_samples]

    return VCFRecord(
        chrom=chrom,
        pos=pos,
        id=vid,
        ref=ref,
        alts=alts,
        qual=qual,
        filter=filt,
        info=info,
        format=fmt,
        samples=samples
    )


class VCFReader:
    """
    Read VCF files and create Variant objects.

    This handles:
    - Reading and parsing VCF files
    - Validating REF alleles against reference FASTA
    - Creating Variant objects with flanking sequences
    - Clustering nearby variants (within kmer_size distance)
    - Merging clusters into combined variants
    """

    def __init__(
        self,
        vcf_path: str,
        reference_path: str,
        kmer_size: int = 31,
        add_reference: bool = True
    ):
        self.vcf_path = vcf_path
        self.kmer_size = kmer_size
        self.add_reference = add_reference

        # Load reference
        self.fasta_reader = FastaReader.from_file(reference_path)

        # Storage
        self.variants_per_chromosome: Dict[str, List[Variant]] = {}
        self.variant_ids: Dict[str, List[List[str]]] = {}
        self.sample_names: List[str] = []
        self.nr_paths = 0
        self.nr_variants = 0

        # Parse VCF
        self._parse_vcf()

    def _parse_vcf(self):
        """Parse the VCF file."""
        with open(self.vcf_path, "r") as f:
            previous_chrom = ""
            previous_end_pos = 0
            variant_cluster: List[Variant] = []

            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Skip meta-info lines
                if line.startswith("##"):
                    continue

                # Parse header line
                if line.startswith("#CHROM"):
                    tokens = line.split("\t")
                    if len(tokens) < 10:
                        raise ValueError("VCF must have at least one sample")
                    self.sample_names = tokens[9:]
                    self.nr_paths = len(self.sample_names) * 2
                    if self.add_reference:
                        self.nr_paths += 1
                    continue

                # Parse data line
                record = parse_vcf_line(line, len(self.sample_names))
                if record is None:
                    continue

                # Skip if contained in previous variant
                if (previous_chrom == record.chrom and
                    record.pos < previous_end_pos):
                    print(f"VCFReader: skip variant at {record.chrom}:{record.pos} "
                          f"(contained in previous)")
                    continue

                # Validate ALT alleles (must be explicit nucleotides)
                if not self._valid_alleles(record):
                    print(f"VCFReader: skip variant at {record.chrom}:{record.pos} "
                          f"(invalid alleles)")
                    continue

                # Validate REF matches reference
                if not self._validate_ref(record):
                    print(f"VCFReader: skip variant at {record.chrom}:{record.pos} "
                          f"(REF doesn't match reference)")
                    continue

                # Skip variants too close to chromosome ends
                chrom_size = self.fasta_reader.get_size_of(record.chrom)
                if (record.pos < self.kmer_size * 2 or
                    record.end_pos + self.kmer_size * 2 > chrom_size):
                    print(f"VCFReader: skip variant at {record.chrom}:{record.pos} "
                          f"(too close to chromosome end)")
                    continue

                # Check if we need to start a new cluster
                if (previous_chrom != record.chrom or
                    record.pos - previous_end_pos >= self.kmer_size - 1):
                    # Save current cluster
                    self._add_variant_cluster(previous_chrom, variant_cluster)
                    variant_cluster = []

                # Create Variant and add to cluster
                variant = self._create_variant(record)
                if variant is not None:
                    variant_cluster.append(variant)
                    previous_chrom = record.chrom
                    previous_end_pos = record.end_pos

            # Add final cluster
            self._add_variant_cluster(previous_chrom, variant_cluster)

        print(f"VCFReader: Identified {self.nr_variants} variants from VCF")

    def _valid_alleles(self, record: VCFRecord) -> bool:
        """Check if all alleles contain only valid nucleotides."""
        pattern = re.compile(r"^[ACGTacgt]+$")
        for allele in record.alleles:
            if not pattern.match(allele.to_string()):
                return False
        return True

    def _validate_ref(self, record: VCFRecord) -> bool:
        """Check if REF allele matches reference sequence."""
        ref_seq = self.fasta_reader.get_subsequence(
            record.chrom, record.pos, record.end_pos
        )
        return record.ref.to_string().upper() == ref_seq.to_string().upper()

    def _create_variant(self, record: VCFRecord) -> Optional[Variant]:
        """Create a Variant object from a VCF record."""
        # Build allele list (ref + alts) - ONLY from VCF, no synthetic N alleles
        alleles = [record.ref]
        alleles.extend(record.alts)

        # Parse genotypes to build paths
        # Paths with "." (undefined) get their own unique undefined allele (like PanGenie)
        paths: List[int] = []
        undefined_idx = len(alleles)  # Next available index for undefined alleles

        if self.add_reference:
            paths.append(0)  # Reference path uses allele 0

        for sample_gt in record.samples:
            # Handle phased genotypes (|) and unphased (/)
            if "/" in sample_gt:
                raise ValueError(f"Unphased genotype found: {sample_gt}")

            gt_field = sample_gt.split(":")[0]  # GT is first field
            haplotypes = gt_field.split("|")

            if len(haplotypes) != 2:
                raise ValueError(f"Invalid genotype (must be diploid): {sample_gt}")

            for hap in haplotypes:
                if hap == ".":
                    # Unknown allele - create a NEW undefined allele for this path
                    alleles.append(DnaSequence.from_string("N"))
                    paths.append(undefined_idx)
                    undefined_idx += 1
                else:
                    allele_idx = int(hap)
                    if allele_idx >= len(record.alleles):
                        raise ValueError(f"Invalid allele index {allele_idx}")
                    paths.append(allele_idx)

        # Get flanking sequences
        left_flank = self.fasta_reader.get_subsequence(
            record.chrom,
            record.pos - self.kmer_size + 1,
            record.pos
        )
        right_flank = self.fasta_reader.get_subsequence(
            record.chrom,
            record.end_pos,
            record.end_pos + self.kmer_size - 1
        )

        # Convert alleles to strings for Variant constructor
        allele_strings = [a.to_string() for a in alleles]

        return Variant.from_strings(
            left_flank=left_flank.to_string(),
            right_flank=right_flank.to_string(),
            chromosome=record.chrom,
            start_position=record.pos,
            end_position=record.end_pos,
            alleles=allele_strings,
            paths=paths
        )

    def _add_variant_cluster(self, chrom: str, cluster: List[Variant]):
        """Merge variants in a cluster and add to storage."""
        if not cluster or not chrom:
            return

        # Merge all variants in cluster
        combined = cluster[0]
        for i in range(1, len(cluster)):
            combined = self._combine_variants(combined, cluster[i])

        # Add flanking sequences
        combined.flanks_added = True

        # Store
        if chrom not in self.variants_per_chromosome:
            self.variants_per_chromosome[chrom] = []
        self.variants_per_chromosome[chrom].append(combined)
        self.nr_variants += 1

    def _combine_variants(self, v1: Variant, v2: Variant) -> Variant:
        """
        Combine two adjacent variants into one.

        This creates a multi-bubble variant where each original variant
        becomes a separate "bubble" with its own alleles.

        Args:
            v1: First variant (will be modified in place)
            v2: Second variant (must start after v1 ends, within kmer_size)

        Returns:
            The combined variant (v1 modified in place)
        """
        v1.combine_variants(v2)
        return v1

    def get_chromosomes(self) -> List[str]:
        """Get list of chromosomes with variants."""
        return list(self.variants_per_chromosome.keys())

    def get_variants(self, chrom: str) -> List[Variant]:
        """Get variants for a chromosome."""
        return self.variants_per_chromosome.get(chrom, [])

    def size_of(self, chrom: str) -> int:
        """Get number of variants for a chromosome."""
        return len(self.variants_per_chromosome.get(chrom, []))

    def write_path_segments(self, filename: str):
        """
        Write all path segments (allele sequences + reference between) to FASTA.

        This is used for k-mer counting.
        """
        with open(filename, "w") as f:
            for chrom in self.fasta_reader.get_sequence_names():
                prev_end = 0

                if chrom in self.variants_per_chromosome:
                    for variant in self.variants_per_chromosome[chrom]:
                        start_pos = variant.start_position

                        # Write reference segment before this variant
                        f.write(f">{chrom}_reference_{start_pos}\n")
                        ref_seg = self.fasta_reader.get_subsequence(chrom, prev_end, start_pos)
                        f.write(f"{ref_seg.to_string()}\n")

                        # Write each allele
                        for allele_idx in range(variant.nr_of_alleles()):
                            f.write(f">{chrom}_{start_pos}_{allele_idx}\n")
                            f.write(f"{variant.get_allele_on_path(allele_idx)}\n")

                        prev_end = variant.get_end_position()

                # Write reference segment after last variant
                f.write(f">{chrom}_reference_end\n")
                chrom_len = self.fasta_reader.get_size_of(chrom)
                ref_seg = self.fasta_reader.get_subsequence(chrom, prev_end, chrom_len)
                f.write(f"{ref_seg.to_string()}\n")
