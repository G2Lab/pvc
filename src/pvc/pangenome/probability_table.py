import math
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


LOGGING_HITS = 2

# 0: nothing
# 1: cache hits / misses
# 2: store all misses


def get_error_param(kmer_coverage: float) -> float:
    """Port of the free function get_error_param."""
    if kmer_coverage < 10.0:
        return 0.99
    elif kmer_coverage < 20:
        return 0.95
    elif kmer_coverage < 40:
        return 0.9
    else:
        return 0.8



class ProbabilityTable:
    def __init__(self,
                 cov_min: int = 0,
                 cov_max: int = 0,
                 count_max: int = 0,
                 regularization_const: float = 0.01):
        self.cov_min = int(cov_min)
        self.cov_max = int(cov_max)
        self.count_max = int(count_max)
        self.regularization_const = float(regularization_const)
        self.probabilities: np.ndarray

        # Initialize and precompute if a valid range is provided
        if self.count_max > 0 and self.cov_max > self.cov_min:
            # rows: read_kmer_count in [0, count_max)
            # cols: kmer_coverage in [cov_min, cov_max)
            width = self.cov_max - self.cov_min
            self.probabilities = np.zeros((self.count_max, width, 3), dtype=float)

            for i in range(self.count_max):
                for j in range(width):
                    cov = j + self.cov_min
                    self.probabilities[i][j] = self.compute_probability(cov, i)

            # Supplementary log-domain tables for the undefined-allele case. When
            # one allele of a genotype cell is undefined, the emission uses
            # logaddexp(log_p_cn_low, log_p_cn_high) + log(0.5). Precomputing the
            # EXACT logaddexp here (plaintext) lets the private genotyper do a
            # cheap oblivious table lookup instead of a secure max approximation:
            # more accurate (true logaddexp, not max) and far less communication.
            #   probabilities_lae[..., 0] = logaddexp(log P(CN=0), log P(CN=1))  (the "01" term)
            #   probabilities_lae[..., 1] = logaddexp(log P(CN=1), log P(CN=2))  (the "12" term)
            self.probabilities_lae = np.stack(
                [
                    np.logaddexp(self.probabilities[..., 0], self.probabilities[..., 1]),
                    np.logaddexp(self.probabilities[..., 1], self.probabilities[..., 2]),
                ],
                axis=-1,
            )


        # Logging info
        if LOGGING_HITS >= 1:
            self._hits = 0
            self._misses = 0
        if LOGGING_HITS >= 2:
            self._missed_params = {}

    def get_probability(self, kmer_coverage: int, read_kmer_count: int) -> np.ndarray:
        """Return precomputed probability if in range; otherwise compute on the fly."""
        if (self.cov_min <= kmer_coverage < self.cov_max) and (0 <= read_kmer_count < self.count_max):
            if LOGGING_HITS >= 1:
                self._hits += 1
            return self.probabilities[read_kmer_count,int(kmer_coverage) - self.cov_min]
        else:
            if LOGGING_HITS >= 1:
                self._misses += 1
            if LOGGING_HITS >= 2:
                self._missed_params[(kmer_coverage, read_kmer_count)] = self._missed_params.get((kmer_coverage, read_kmer_count), 0) + 1
            return self.compute_probability(kmer_coverage, read_kmer_count)
    
    def compute_probability(self, kmer_coverage: int, read_kmer_count: int) -> np.ndarray:
        p_cn0 = self._geometric(get_error_param(float(kmer_coverage)), read_kmer_count)
        p_cn1 = self._poisson(float(kmer_coverage) / 2.0, read_kmer_count)
        p_cn2 = self._poisson(float(kmer_coverage), read_kmer_count)

        p = np.array([p_cn0, p_cn1, p_cn2])

        if self.regularization_const > 0.0:
            p += self.regularization_const

        probs = p / p.sum()
        log_probs = np.log(probs)
        return log_probs
        
        # Return raw probabilities; CopyNumber.__post_init__ handles normalization and log conversion
        return p

    def modify_probability(self, kmer_coverage: int, read_kmer_count: int, prob: np.ndarray) -> None:
        if (self.cov_min <= kmer_coverage < self.cov_max) and (0 <= read_kmer_count < self.count_max):
            self.probabilities[read_kmer_count][kmer_coverage - self.cov_min] = prob
        else:
            raise RuntimeError("ProbabilityTable.modify_probability: no precomputed values for these parameters.")

    @staticmethod
    def _poisson(mean: float, value: int) -> float:
        """
        Poisson PMF: P(X=value) = e^{-mean} * mean^{value} / value!
        (log-space via lgamma for stability)
        """
        if value < 0:
            return 0.0
        if mean < 0.0:
            return 0.0
        if mean == 0.0:
            return 1.0 if value == 0 else 0.0
        log_p = -mean + value * math.log(mean) - math.lgamma(value + 1)
        return math.exp(log_p)

    @staticmethod
    def _geometric(p: float, value: int) -> float:
        """
        Geometric PMF (support k = 0,1,2,...):
        P(X=value) = (1 - p)^{value} * p
        """
        if not (0.0 < p < 1.0) or value < 0:
            return 0.0
        return (1.0 - p) ** value * p

    def __str__(self) -> str:
        """
        Pretty-print like the C++ operator<<:
            header row: coverage values
            each subsequent row: read_kmer_count followed by triplets (p0 p1 p2)
        """
        lines = []
        # Header
        header_cells = ["\t"]
        for cov in range(self.cov_min, self.cov_max):
            if cov > self.cov_min:
                header_cells.append("\t\t\t")
            header_cells.append(str(cov))
        lines.append("".join(header_cells))

        # Body
        for i in range(self.count_max):
            row = [f"{i}\t"]
            for j in range(self.cov_max - self.cov_min):
                if j > 0:
                    row.append("\t")
                cn = self.probabilities[i][j]
                row.append(f"{cn.get_probability_of(0)}\t{cn.get_probability_of(1)}\t{cn.get_probability_of(2)}")
            lines.append("".join(row))
        return "\n".join(lines)
    
    def print_logging_stats(self):
        if LOGGING_HITS >= 1:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100.0) if total > 0 else 0.0
            print(f"ProbabilityTable logging: Hits={self._hits}, Misses={self._misses}, Hit rate={hit_rate:.2f}%")
        if LOGGING_HITS >= 2:
            print("Missed parameters (kmer_coverage, read_kmer_count):")
            sorted_params = sorted(self._missed_params.items(), key=lambda x: x[1], reverse=True)

            for (kmer_cov, read_kmer_cnt), count in sorted_params:
                print(f"  ({kmer_cov}, {read_kmer_cnt}): {count} misses")
