"""
K-mer counter for counting k-mers in sequences.

This module provides both Python-based and Jellyfish-based k-mer counting.
"""

from typing import Dict, Iterator, Tuple, Optional, Set
from collections import defaultdict
import gzip
import subprocess
import tempfile
import os


def reverse_complement(seq: str) -> str:
    """Get the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N',
                  'a': 't', 't': 'a', 'g': 'c', 'c': 'g', 'n': 'n'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq))


def canonical_kmer(kmer: str) -> str:
    """Get canonical form of k-mer (lexicographically smaller of kmer and its RC)."""
    rc = reverse_complement(kmer)
    return min(kmer.upper(), rc.upper())


def iter_kmers(sequence: str, k: int) -> Iterator[str]:
    """Iterate over all k-mers in a sequence."""
    seq_upper = sequence.upper()
    for i in range(len(seq_upper) - k + 1):
        kmer = seq_upper[i:i + k]
        # Skip k-mers with N
        if 'N' not in kmer:
            yield kmer


class KmerCounter:
    """
    Count k-mers in sequences.

    Stores counts in canonical form (lexicographically smaller of kmer and its RC).
    """

    def __init__(self, k: int):
        self.k = k
        self.counts: Dict[str, int] = defaultdict(int)
        self.total_kmers = 0

    def add_sequence(self, sequence: str):
        """Add all k-mers from a sequence to the counter."""
        for kmer in iter_kmers(sequence, self.k):
            self.counts[canonical_kmer(kmer)] += 1
            self.total_kmers += 1

    def get_count(self, kmer: str) -> int:
        """Get the count of a k-mer (in canonical form)."""
        return self.counts.get(canonical_kmer(kmer), 0)

    def __len__(self) -> int:
        """Number of distinct k-mers."""
        return len(self.counts)

    def count_from_fasta(self, fasta_path: str):
        """Count k-mers from a FASTA file."""
        opener = gzip.open if fasta_path.endswith('.gz') else open
        with opener(fasta_path, 'rt') as f:
            current_seq_parts = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    # Process previous sequence
                    if current_seq_parts:
                        self.add_sequence(''.join(current_seq_parts))
                    current_seq_parts = []
                else:
                    current_seq_parts.append(line)
            # Process last sequence
            if current_seq_parts:
                self.add_sequence(''.join(current_seq_parts))

    def count_from_fastq(self, fastq_path: str, max_reads: Optional[int] = None):
        """Count k-mers from a FASTQ file."""
        opener = gzip.open if fastq_path.endswith('.gz') else open
        read_count = 0
        with opener(fastq_path, 'rt') as f:
            while True:
                # Read 4 lines per read
                header = f.readline()
                if not header:
                    break
                seq = f.readline().strip()
                plus = f.readline()
                qual = f.readline()

                self.add_sequence(seq)
                read_count += 1

                if max_reads and read_count >= max_reads:
                    break

                if read_count % 1000000 == 0:
                    print(f"  Processed {read_count:,} reads...")

        print(f"Counted k-mers from {read_count:,} reads")

    def compute_coverage(self, genome_kmers: int) -> int:
        """Compute k-mer coverage relative to genome k-mer count."""
        if genome_kmers == 0:
            return 0
        return self.total_kmers // genome_kmers

    def compute_histogram_peak_coverage(self, max_count: int = 1000) -> int:
        """
        Compute k-mer coverage by finding the peak in the k-mer count histogram.

        This is the same method PanGenie uses - find the count value where
        the most k-mers occur (excluding very low counts that represent errors).

        Args:
            max_count: Maximum count to consider

        Returns:
            Estimated coverage from histogram peak
        """
        # Build histogram: count -> number of k-mers with that count
        histogram = defaultdict(int)
        for count in self.counts.values():
            if 0 < count <= max_count:
                histogram[count] += 1

        if not histogram:
            return 30  # Default fallback

        # Find peaks in the histogram (local maxima)
        # Skip count=1 as it's often error k-mers
        sorted_counts = sorted(histogram.keys())

        # Simple approach: find the mode (most common count) excluding count=1
        best_count = 1
        best_freq = 0
        for count in sorted_counts:
            if count > 1:  # Skip error peak at count=1
                if histogram[count] > best_freq:
                    best_freq = histogram[count]
                    best_count = count

        return best_count


def count_variant_kmers(allele_sequences: list, k: int) -> Dict[str, set]:
    """
    Count which k-mers appear in which alleles.

    Args:
        allele_sequences: List of allele sequence strings
        k: k-mer size

    Returns:
        Dict mapping k-mer to set of allele indices where it appears
    """
    kmer_to_alleles: Dict[str, set] = defaultdict(set)

    for allele_idx, seq in enumerate(allele_sequences):
        for kmer in iter_kmers(seq, k):
            kmer_to_alleles[canonical_kmer(kmer)].add(allele_idx)

    return kmer_to_alleles


def find_unique_kmers(allele_sequences: list, k: int) -> Dict[int, list]:
    """
    Find k-mers that are unique to each allele.

    Args:
        allele_sequences: List of allele sequence strings
        k: k-mer size

    Returns:
        Dict mapping allele index to list of unique k-mers for that allele
    """
    kmer_to_alleles = count_variant_kmers(allele_sequences, k)

    # Find k-mers unique to each allele
    allele_unique_kmers: Dict[int, list] = defaultdict(list)

    for kmer, alleles in kmer_to_alleles.items():
        if len(alleles) == 1:
            allele_idx = next(iter(alleles))
            allele_unique_kmers[allele_idx].append(kmer)

    return dict(allele_unique_kmers)


class JellyfishCounter:
    """
    Count k-mers using Jellyfish for better performance.

    This class wraps the Jellyfish CLI to count k-mers in reads and
    query counts for specific k-mers.
    """

    def __init__(self, k: int, hash_size: str = "100M", threads: int = 4):
        """
        Args:
            k: K-mer size
            hash_size: Jellyfish hash size (e.g., "100M", "1G")
            threads: Number of threads for Jellyfish
        """
        self.k = k
        self.hash_size = hash_size
        self.threads = threads
        self.jf_file: Optional[str] = None
        self._temp_dir: Optional[str] = None

    def count_reads(self, reads_path: str, verbose: bool = True) -> str:
        """
        Count all k-mers in reads using Jellyfish.

        Args:
            reads_path: Path to reads file (FASTA or FASTQ)
            verbose: Print progress

        Returns:
            Path to Jellyfish database file
        """
        import time
        start_time = time.time()

        # Create temp directory for output
        self._temp_dir = tempfile.mkdtemp(prefix="jellyfish_")
        self.jf_file = os.path.join(self._temp_dir, "mer_counts.jf")

        # Determine if input is gzipped
        if reads_path.endswith('.gz'):
            # Jellyfish can read from stdin, pipe through zcat
            cmd = f"zcat {reads_path} | jellyfish count -m {self.k} -s {self.hash_size} -t {self.threads} -C -o {self.jf_file} /dev/fd/0"
        else:
            cmd = f"jellyfish count -m {self.k} -s {self.hash_size} -t {self.threads} -C -o {self.jf_file} {reads_path}"

        if verbose:
            print(f"Running Jellyfish count on {reads_path}...")

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Jellyfish count failed: {result.stderr}")

        elapsed = time.time() - start_time
        if verbose:
            # Get stats
            stats_cmd = f"jellyfish stats {self.jf_file}"
            stats_result = subprocess.run(stats_cmd, shell=True, capture_output=True, text=True)
            print(f"  Completed in {elapsed:.1f}s")
            if stats_result.returncode == 0:
                for line in stats_result.stdout.strip().split('\n'):
                    print(f"  {line}")

        return self.jf_file

    def load_database(self, jf_path: str):
        """Load an existing Jellyfish database."""
        if not os.path.exists(jf_path):
            raise FileNotFoundError(f"Jellyfish database not found: {jf_path}")
        self.jf_file = jf_path

    def query_kmers(self, kmers: Set[str], verbose: bool = True) -> Dict[str, int]:
        """
        Query counts for specific k-mers.

        Args:
            kmers: Set of k-mers to query (in canonical form)
            verbose: Print progress

        Returns:
            Dict mapping k-mer to count
        """
        if self.jf_file is None:
            raise RuntimeError("No Jellyfish database loaded. Call count_reads() or load_database() first.")

        import time
        start_time = time.time()

        # Write k-mers to temp file
        kmer_file = tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False)
        try:
            for i, kmer in enumerate(kmers):
                kmer_file.write(f">{i}\n{kmer}\n")
            kmer_file.close()

            if verbose:
                print(f"Querying {len(kmers):,} k-mers from Jellyfish database...")

            # Query using jellyfish query
            cmd = f"jellyfish query {self.jf_file} -s {kmer_file.name}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Jellyfish query failed: {result.stderr}")

            # Parse results
            counts: Dict[str, int] = {}
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        kmer = parts[0]
                        count = int(parts[1])
                        counts[kmer] = count

            elapsed = time.time() - start_time
            if verbose:
                non_zero = sum(1 for c in counts.values() if c > 0)
                print(f"  Completed in {elapsed:.1f}s")
                print(f"  Found {non_zero:,} k-mers with count > 0")

            return counts

        finally:
            os.unlink(kmer_file.name)

    def get_count(self, kmer: str) -> int:
        """
        Get count for a single k-mer.

        Note: For many queries, use query_kmers() which is more efficient.
        """
        counts = self.query_kmers({canonical_kmer(kmer)}, verbose=False)
        return counts.get(canonical_kmer(kmer), 0)

    def cleanup(self):
        """Remove temporary files."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            import shutil
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None
            self.jf_file = None

    def __del__(self):
        self.cleanup()


def count_unique_kmers_jellyfish(
    unique_kmers_per_chrom: Dict[str, list],
    reads_path: str,
    kmer_size: int,
    threads: int = 4,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Count unique k-mers from reads using Jellyfish and update the unique k-mers map.

    This is much faster than the pure Python implementation for large read sets.

    Args:
        unique_kmers_per_chrom: Dict mapping chromosome to list of ComputedUniqueKmers
        reads_path: Path to reads file (FASTA or FASTQ)
        kmer_size: K-mer size
        threads: Number of threads for Jellyfish
        verbose: Print progress messages

    Returns:
        Dict mapping k-mer to count
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

    # Step 2: Count k-mers in reads using Jellyfish
    jf = JellyfishCounter(k=kmer_size, threads=threads)
    jf.count_reads(reads_path, verbose=verbose)

    # Step 3: Query our target k-mers
    counts = jf.query_kmers(target_kmers, verbose=verbose)

    # Step 4: Update counts in unique k-mers map
    if verbose:
        print("Updating unique k-mer counts...")

    total_updated = 0
    for chrom, uks in unique_kmers_per_chrom.items():
        for uk in uks:
            # Update kmer_to_count
            uk.kmer_to_count = [counts.get(k, 0) for k in uk.kmers]
            total_updated += uk.size()

            # Update local coverage estimate
            non_zero = [c for c in uk.kmer_to_count if c > 0]
            if non_zero:
                non_zero.sort()
                mid = len(non_zero) // 2
                if len(non_zero) % 2 == 0:
                    uk.local_coverage = (non_zero[mid - 1] + non_zero[mid]) / 2.0
                else:
                    uk.local_coverage = float(non_zero[mid])

    # Cleanup
    jf.cleanup()

    elapsed = time.time() - start_time
    if verbose:
        print(f"Updated {total_updated:,} k-mer counts in {elapsed:.1f}s")

    return counts
