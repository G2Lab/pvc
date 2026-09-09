"""
Compute unique k-mers for each variant.

This finds k-mers that uniquely identify alleles at each variant position,
then counts how often these k-mers appear in the reads.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from pvc.pangenome.kmer_counter import KmerCounter, canonical_kmer, iter_kmers
from pvc.pangenome.variant import Variant


@dataclass
class AlleleKmers:
    """K-mers associated with an allele."""
    allele_id: int
    kmers: List[str] = field(default_factory=list)
    is_undefined: bool = False


@dataclass
class ComputedUniqueKmers:
    """
    Unique k-mers for a variant position.

    This mirrors the UniqueKmers class but is computed from scratch.
    Compatible with the genotyping code interface.
    """
    variant_pos: int
    local_coverage: float
    kmer_to_count: List[int]  # Read counts for each unique k-mer
    kmers: List[str]  # The actual k-mer strings (canonical form)
    allele_to_kmers: Dict[int, List[int]]  # Allele ID -> indices into kmer_to_count
    path_to_allele: List[int]  # Path index -> allele ID
    is_biallelic: bool

    def size(self) -> int:
        """Number of unique k-mers."""
        return len(self.kmer_to_count)

    def get_readcount_of(self, kmer_index: int) -> int:
        """Get read count for a k-mer by index."""
        return self.kmer_to_count[kmer_index]

    def get_coverage(self) -> float:
        return self.local_coverage

    def get_kmer(self, kmer_index: int) -> str:
        """Get k-mer string by index."""
        return self.kmers[kmer_index]

    def update_counts(self, read_counter: 'KmerCounter') -> None:
        """
        Update k-mer counts from a read counter.

        Args:
            read_counter: KmerCounter with read k-mer counts
        """
        self.kmer_to_count = [read_counter.get_count(k) for k in self.kmers]
        # Update local coverage estimate
        non_zero = [c for c in self.kmer_to_count if c > 0]
        if non_zero:
            non_zero.sort()
            mid = len(non_zero) // 2
            if len(non_zero) % 2 == 0:
                self.local_coverage = (non_zero[mid - 1] + non_zero[mid]) / 2.0
            else:
                self.local_coverage = float(non_zero[mid])

    # --- Compatibility interface for genotyping code ---

    @property
    def alleles(self) -> Dict[int, 'AlleleInfo']:
        """
        Return allele info dict compatible with genotyping code.
        Maps allele_id -> AlleleInfo with kmer indices.
        """
        return {
            allele_id: AlleleInfo(kmer_indices=indices, is_undefined=False)
            for allele_id, indices in self.allele_to_kmers.items()
        }

    def is_undefined_allele(self, allele_id: int) -> bool:
        """Check if an allele is undefined (contains N's)."""
        # For computed unique kmers, we don't track undefined status
        # so we return False by default
        return False

    def kmer_on_allele(self, kmer_index: int, allele_id: int) -> int:
        """
        Check if k-mer at kmer_index is on allele_id.
        Returns 1 if yes, 0 if no.
        """
        if allele_id not in self.allele_to_kmers:
            return 0
        return 1 if kmer_index in self.allele_to_kmers[allele_id] else 0

    def get_variant_position(self) -> int:
        """Get variant position."""
        return self.variant_pos


@dataclass
class AlleleInfo:
    """Compatibility class for allele information."""
    kmer_indices: List[int]
    is_undefined: bool = False


class UniqueKmerComputer:
    """
    Compute unique k-mers for variants.

    For each variant:
    1. Extract all k-mers from each allele (including flanking sequences)
    2. Find k-mers that uniquely identify each allele
    3. Count those k-mers in the reads
    """

    def __init__(
        self,
        pangenome_counter: KmerCounter,
        read_counter: KmerCounter,
        kmer_size: int,
        kmer_coverage: int
    ):
        """
        Args:
            pangenome_counter: K-mer counts from pangenome (path segments)
            read_counter: K-mer counts from reads
            kmer_size: K-mer size
            kmer_coverage: Expected k-mer coverage
        """
        self.pangenome_counter = pangenome_counter
        self.read_counter = read_counter
        self.kmer_size = kmer_size
        self.kmer_coverage = kmer_coverage

    def compute_for_variant(
        self,
        variant: Variant,
        max_kmers_per_allele: Optional[int] = None
    ) -> ComputedUniqueKmers:
        """
        Compute unique k-mers for a single variant.

        This follows PanGenie's algorithm:
        1. Find k-mers unique to each allele (appear exactly once on that allele)
        2. Filter to k-mers that appear exactly once in the pangenome
        3. Select limited number of k-mers per allele (16 biallelic, 32 multiallelic)

        Args:
            variant: The variant to process
            max_kmers_per_allele: Override max k-mers per allele (default: 16 biallelic, 32 multiallelic)

        Returns:
            ComputedUniqueKmers object
        """
        n_paths = len(variant.paths)

        # Use VCF-observed allele combinations (PanGenie behavior)
        n_alleles = variant.nr_of_alleles()
        allele_seqs = []
        for i in range(n_alleles):
            allele_seqs.append(variant.get_allele_on_path(i))
        path_to_allele = variant.paths

        # Check if biallelic
        is_biallelic = n_alleles == 2

        # Set k-mer limits per PanGenie: 16 for biallelic, 32 for multiallelic
        if max_kmers_per_allele is None:
            max_kmers_per_allele = 16 if is_biallelic else 32

        # Total k-mer limit: max(nr_paths, 301)
        max_total_kmers = max(n_paths, 301)

        # Find which k-mers appear on which alleles and count occurrences
        # Key difference from before: track how many times k-mer appears on each allele
        kmer_to_allele_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for allele_idx, seq in enumerate(allele_seqs):
            for kmer in iter_kmers(seq, self.kmer_size):
                kmer_to_allele_counts[canonical_kmer(kmer)][allele_idx] += 1

        # Find unique k-mers for each allele
        # A k-mer is unique to an allele if:
        # 1. It appears on exactly one allele
        # 2. It appears exactly once on that allele (within the allele sequence)
        # 3. It appears exactly once in the pangenome (genomic_count == local_count)
        allele_unique_kmers: Dict[int, List[str]] = defaultdict(list)

        for kmer, allele_counts in kmer_to_allele_counts.items():
            # Must appear on exactly one allele
            if len(allele_counts) != 1:
                continue

            allele_idx = next(iter(allele_counts.keys()))
            local_count = allele_counts[allele_idx]

            # Must appear exactly once within the allele
            if local_count != 1:
                continue

            # Check genomic count - k-mer should only appear at this location
            genomic_count = self.pangenome_counter.get_count(kmer)

            # Per PanGenie: skip if (genomic_count - local_count) != 0
            if genomic_count != local_count:
                continue

            allele_unique_kmers[allele_idx].append(kmer)

        # Select k-mers per allele with limits
        # PanGenie uses a round-robin selection across alleles
        selected_kmers: List[str] = []
        allele_to_kmer_indices: Dict[int, List[int]] = defaultdict(list)

        # Build queues of k-mers for each allele
        allele_queues: Dict[int, List[str]] = {}
        for allele_idx in range(n_alleles):
            allele_queues[allele_idx] = list(allele_unique_kmers.get(allele_idx, []))

        # Round-robin selection
        while len(selected_kmers) < max_total_kmers:
            added_any = False
            for allele_idx in range(n_alleles):
                if len(selected_kmers) >= max_total_kmers:
                    break
                if len(allele_to_kmer_indices[allele_idx]) >= max_kmers_per_allele:
                    continue
                if not allele_queues[allele_idx]:
                    continue

                kmer = allele_queues[allele_idx].pop(0)
                kmer_idx = len(selected_kmers)
                selected_kmers.append(kmer)
                allele_to_kmer_indices[allele_idx].append(kmer_idx)
                added_any = True

            if not added_any:
                break

        # Get read counts for selected k-mers
        kmer_to_count = [self.read_counter.get_count(k) for k in selected_kmers]

        # Compute local coverage from flanking region unique k-mers (per PanGenie)
        local_coverage = self._compute_local_coverage_from_flanks(variant)

        return ComputedUniqueKmers(
            variant_pos=variant.start_position,
            local_coverage=local_coverage,
            kmer_to_count=kmer_to_count,
            kmers=selected_kmers,
            allele_to_kmers=dict(allele_to_kmer_indices),
            path_to_allele=path_to_allele,
            is_biallelic=is_biallelic
        )

    def _compute_local_coverage_from_flanks(self, variant: Variant) -> float:
        """
        Compute local coverage from flanking region unique k-mers.

        Per PanGenie algorithm:
        1. Get left and right flanking sequences
        2. Find unique k-mers in flanks that appear exactly once in pangenome
        3. Count those k-mers in reads
        4. Filter extreme counts (< kmer_coverage/4 or > kmer_coverage*4)
        5. Return average, or default kmer_coverage if no valid k-mers
        """
        # Get flanking sequences
        left_flank = variant.left_flank.to_string() if hasattr(variant.left_flank, 'to_string') else str(variant.left_flank)
        right_flank = variant.right_flank.to_string() if hasattr(variant.right_flank, 'to_string') else str(variant.right_flank)

        min_cov = self.kmer_coverage // 4
        max_cov = self.kmer_coverage * 4
        max_kmers_per_side = 12

        total_coverage = 0
        total_kmers = 0

        # Process left flank
        left_kmer_counts: Dict[str, int] = defaultdict(int)
        for kmer in iter_kmers(left_flank, self.kmer_size):
            left_kmer_counts[canonical_kmer(kmer)] += 1

        selected = 0
        for kmer, count in left_kmer_counts.items():
            if selected >= max_kmers_per_side:
                break
            if count != 1:  # Must appear exactly once in flank
                continue
            genomic_count = self.pangenome_counter.get_count(kmer)
            if genomic_count != 1:  # Must be unique in pangenome
                continue
            selected += 1
            read_count = self.read_counter.get_count(kmer)
            if read_count < min_cov or read_count > max_cov:
                continue
            total_coverage += read_count
            total_kmers += 1

        # Process right flank
        right_kmer_counts: Dict[str, int] = defaultdict(int)
        for kmer in iter_kmers(right_flank, self.kmer_size):
            right_kmer_counts[canonical_kmer(kmer)] += 1

        selected = 0
        for kmer, count in right_kmer_counts.items():
            if selected >= max_kmers_per_side:
                break
            if count != 1:
                continue
            genomic_count = self.pangenome_counter.get_count(kmer)
            if genomic_count != 1:
                continue
            selected += 1
            read_count = self.read_counter.get_count(kmer)
            if read_count < min_cov or read_count > max_cov:
                continue
            total_coverage += read_count
            total_kmers += 1

        # Return average or default
        if total_kmers > 0 and total_coverage > 0:
            return total_coverage / total_kmers
        return float(self.kmer_coverage)

    def _compute_local_coverage(self, kmers: List[str]) -> float:
        """Estimate local coverage from k-mer counts (legacy method)."""
        if not kmers:
            return float(self.kmer_coverage)

        # Use median of non-zero counts
        counts = [self.read_counter.get_count(k) for k in kmers]
        non_zero = [c for c in counts if c > 0]

        if not non_zero:
            return float(self.kmer_coverage)

        non_zero.sort()
        mid = len(non_zero) // 2
        if len(non_zero) % 2 == 0:
            return (non_zero[mid - 1] + non_zero[mid]) / 2.0
        return float(non_zero[mid])

    def compute_all(
        self,
        variants: List[Variant],
        max_kmers_per_allele: Optional[int] = None
    ) -> List[ComputedUniqueKmers]:
        """
        Compute unique k-mers for all variants.

        Args:
            variants: List of variants
            max_kmers_per_allele: Maximum k-mers per allele (None = use PanGenie defaults: 16 biallelic, 32 multiallelic)

        Returns:
            List of ComputedUniqueKmers, one per variant
        """
        results = []
        for i, variant in enumerate(variants):
            if i % 1000 == 0 and i > 0:
                print(f"  Processed {i:,} / {len(variants):,} variants...")

            results.append(self.compute_for_variant(variant, max_kmers_per_allele))

        return results


def build_unique_kmers_map(
    variants_per_chrom: Dict[str, List[Variant]],
    pangenome_counter: KmerCounter,
    read_counter: KmerCounter,
    kmer_size: int,
    kmer_coverage: int
) -> Dict[str, List[ComputedUniqueKmers]]:
    """
    Build unique k-mers map for all chromosomes.

    Args:
        variants_per_chrom: Dict mapping chromosome to variants
        pangenome_counter: K-mer counts from pangenome
        read_counter: K-mer counts from reads
        kmer_size: K-mer size
        kmer_coverage: Expected k-mer coverage

    Returns:
        Dict mapping chromosome to list of ComputedUniqueKmers
    """
    computer = UniqueKmerComputer(
        pangenome_counter=pangenome_counter,
        read_counter=read_counter,
        kmer_size=kmer_size,
        kmer_coverage=kmer_coverage
    )

    result = {}
    for chrom, variants in variants_per_chrom.items():
        print(f"Computing unique k-mers for {chrom} ({len(variants)} variants)...")
        result[chrom] = computer.compute_all(variants)
        print(f"  Done: {sum(uk.size() for uk in result[chrom]):,} total unique k-mers")

    return result


def count_unique_kmers_from_reads(
    unique_kmers_per_chrom: Dict[str, List[ComputedUniqueKmers]],
    reads_path: str,
    kmer_size: int,
    max_reads: Optional[int] = None,
    verbose: bool = True
) -> KmerCounter:
    """
    Count unique k-mers from reads and update the unique k-mers map.

    This function:
    1. Collects all unique k-mers from the index
    2. Counts only those k-mers in the reads (memory efficient)
    3. Updates the kmer_to_count and local_coverage fields

    Args:
        unique_kmers_per_chrom: Dict mapping chromosome to list of ComputedUniqueKmers
        reads_path: Path to reads file (FASTA or FASTQ)
        kmer_size: K-mer size
        max_reads: Maximum reads to process (for testing)
        verbose: Print progress messages

    Returns:
        KmerCounter with read k-mer counts (for the unique k-mers only)
    """
    import time
    start_time = time.time()

    # Step 1: Collect all unique k-mers we need to count
    if verbose:
        print("Collecting unique k-mers from index...")

    target_kmers = set()
    for chrom, uks in unique_kmers_per_chrom.items():
        for uk in uks:
            target_kmers.update(uk.kmers)

    if verbose:
        print(f"  Found {len(target_kmers):,} unique k-mers to count")

    # Step 2: Count only target k-mers in reads (memory efficient)
    if verbose:
        print(f"Counting k-mers in reads: {reads_path}")

    read_counter = count_target_kmers_in_reads(
        target_kmers=target_kmers,
        reads_path=reads_path,
        kmer_size=kmer_size,
        max_reads=max_reads,
        verbose=verbose
    )

    # Step 3: Update counts in unique k-mers map
    if verbose:
        print("Updating unique k-mer counts...")

    total_updated = 0
    for chrom, uks in unique_kmers_per_chrom.items():
        for uk in uks:
            uk.update_counts(read_counter)
            total_updated += uk.size()

    elapsed = time.time() - start_time
    if verbose:
        print(f"Updated {total_updated:,} k-mer counts in {elapsed:.1f}s")

    return read_counter


def count_target_kmers_in_reads(
    target_kmers: set,
    reads_path: str,
    kmer_size: int,
    max_reads: Optional[int] = None,
    verbose: bool = True
) -> KmerCounter:
    """
    Count only specific k-mers in reads (memory efficient).

    Instead of counting all k-mers in reads, this only counts the k-mers
    that are in the target set. This is much more memory efficient for
    large read sets.

    Args:
        target_kmers: Set of k-mers to count (in canonical form)
        reads_path: Path to reads file
        kmer_size: K-mer size
        max_reads: Maximum reads to process
        verbose: Print progress

    Returns:
        KmerCounter with counts for target k-mers only
    """
    import gzip

    counter = KmerCounter(k=kmer_size)
    read_count = 0
    kmer_hits = 0

    # Determine file type
    is_fastq = reads_path.endswith('.fastq') or reads_path.endswith('.fastq.gz') or \
               reads_path.endswith('.fq') or reads_path.endswith('.fq.gz')
    is_gzipped = reads_path.endswith('.gz')

    opener = gzip.open if is_gzipped else open

    with opener(reads_path, 'rt') as f:
        if is_fastq:
            # FASTQ format: 4 lines per read
            while True:
                header = f.readline()
                if not header:
                    break
                seq = f.readline().strip()
                plus = f.readline()
                qual = f.readline()

                # Count only target k-mers
                for kmer in iter_kmers(seq, kmer_size):
                    canon = canonical_kmer(kmer)
                    if canon in target_kmers:
                        counter.counts[canon] += 1
                        kmer_hits += 1

                read_count += 1
                if max_reads and read_count >= max_reads:
                    break

                if verbose and read_count % 1000000 == 0:
                    print(f"  Processed {read_count:,} reads, {kmer_hits:,} k-mer hits...")
        else:
            # FASTA format
            current_seq_parts = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    # Process previous sequence
                    if current_seq_parts:
                        seq = ''.join(current_seq_parts)
                        for kmer in iter_kmers(seq, kmer_size):
                            canon = canonical_kmer(kmer)
                            if canon in target_kmers:
                                counter.counts[canon] += 1
                                kmer_hits += 1
                        read_count += 1

                        if max_reads and read_count >= max_reads:
                            break

                        if verbose and read_count % 100000 == 0:
                            print(f"  Processed {read_count:,} sequences, {kmer_hits:,} k-mer hits...")

                    current_seq_parts = []
                else:
                    current_seq_parts.append(line)

            # Process last sequence
            if current_seq_parts and (not max_reads or read_count < max_reads):
                seq = ''.join(current_seq_parts)
                for kmer in iter_kmers(seq, kmer_size):
                    canon = canonical_kmer(kmer)
                    if canon in target_kmers:
                        counter.counts[canon] += 1
                        kmer_hits += 1
                read_count += 1

    if verbose:
        print(f"  Processed {read_count:,} reads/sequences")
        print(f"  Found {kmer_hits:,} k-mer hits")
        print(f"  Unique k-mers with counts > 0: {sum(1 for c in counter.counts.values() if c > 0):,}")

    return counter
