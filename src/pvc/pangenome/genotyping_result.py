from __future__ import annotations
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple
import math

def genotype_from_alleles(allele1: int, allele2: int) -> Tuple[int, int]:
    """Return (min, max) so genotypes are stored in canonical order."""
    return (allele1, allele2) if allele1 < allele2 else (allele2, allele1)


class GenotypingResult:
    def __init__(self) -> None:
        self.haplotype_1: int = 0
        self.haplotype_2: int = 0
        self.local_coverage: int = 0
        self.unique_kmers: int = 0
        # Map[(allele1, allele2)] -> likelihood (float)
        self._genotype_to_likelihood: Dict[Tuple[int, int], float] = {}

    # --- Mutators / Adders ---

    def add_to_likelihood(self, allele1: int, allele2: int, value: float) -> None:
        g = genotype_from_alleles(allele1, allele2)
        self._genotype_to_likelihood[g] = self._genotype_to_likelihood.get(g, 0.0) + float(value)

    def add_first_haplotype_allele(self, allele: int) -> None:
        self.haplotype_1 = int(allele)

    def add_second_haplotype_allele(self, allele: int) -> None:
        self.haplotype_2 = int(allele)

    # --- Accessors ---

    def get_genotype_likelihood(self, allele1: int, allele2: int) -> float:
        g = genotype_from_alleles(allele1, allele2)
        return float(self._genotype_to_likelihood.get(g, 0.0))

    def get_all_likelihoods(self, nr_alleles: int) -> List[float]:
        """
        Return a dense vector of genotype likelihoods in VCF order:
          index = (j * (j + 1)) // 2 + i for genotype (i/j) with i<=j, i,j in [0..nr_alleles-1].
        """
        assert 0 <= nr_alleles < 65536
        nr_genotypes = (nr_alleles * (nr_alleles + 1)) // 2
        result: List[float] = [0.0] * nr_genotypes

        for (a1, a2), val in self._genotype_to_likelihood.items():
            # a1 <= a2 must hold by construction
            index = (a2 * (a2 + 1)) // 2 + a1
            if not (0 <= index < nr_genotypes):
                raise RuntimeError("GenotypeResult::get_all_likelihoods: genotype does not match number of alleles.")
            result[index] = float(val)

        return result

    def get_specific_likelihoods(self, alleles: List[int]) -> "GenotypingResult":
        """
        Restrict to a subset of alleles. Re-index alleles into [0..len(alleles)-1],
        keep haplotype alleles if they map, and (if sum>0) normalize the new result.
        """
        res = GenotypingResult()
        assert len(alleles) < 65536

        allowed = set(int(a) for a in alleles)
        index_map = {alleles[i]: i for i in range(len(alleles))}

        total = 0.0
        for (a1, a2), like in self._genotype_to_likelihood.items():
            if a1 not in allowed or a2 not in allowed:
                continue
            i = index_map[a1]
            j = index_map[a2]
            if self.haplotype_1 == a1:
                res.haplotype_1 = i
            if self.haplotype_2 == a2:
                res.haplotype_2 = j
            res.add_to_likelihood(i, j, like)
            total += like

        if total > 0.0:
            res.divide_likelihoods_by(total)

        return res

    def get_genotype_quality(self, allele1: int, allele2: int) -> int:
        """
        Phred-scaled genotype quality (GQ) = -10 * log10(1 - P(best genotype)).
        Requires normalized likelihoods (sum == 1).
        """
        total = sum(self._genotype_to_likelihood.values())
        if abs(total - 1.0) > 1e-10:
            raise RuntimeError(
                "GenotypingResult::get_genotype_quality: genotype quality can only be computed from normalized likelihoods."
            )

        prob_wrong = 1.0 - self.get_genotype_likelihood(allele1, allele2)
        if prob_wrong > 0.0:
            return int(-10.0 * math.log10(prob_wrong))
        else:
            # Default large value as in C++
            return 10000

    def get_haplotype(self) -> Tuple[int, int]:
        return (self.haplotype_1, self.haplotype_2)

    def divide_likelihoods_by(self, value: float) -> None:
        if value == 0:
            return
        for g in list(self._genotype_to_likelihood.keys()):
            self._genotype_to_likelihood[g] = self._genotype_to_likelihood[g] / float(value)

    def get_likeliest_genotype(self) -> Tuple[int, int]:
        """
        Return (allele1, allele2) of the unique maximum-likelihood genotype,
        or (-1, -1) if there is a tie or if all likelihoods are zero/empty.
        """
        if not self._genotype_to_likelihood:
            return (-1, -1)

        # Find max value (prefer later entries only if >= as in C++)
        best_value = -1.0
        best_genotype: Tuple[int, int] | None = None
        for g, v in self._genotype_to_likelihood.items():
            if v >= best_value:
                best_value = float(v)
                best_genotype = g

        if best_genotype is None:
            return (-1, -1)

        # Ensure uniqueness (no other genotype within 1e-10 of best_value)
        for g, v in self._genotype_to_likelihood.items():
            if g != best_genotype and abs(v - best_value) < 1e-300:
                return (-1, -1)

        if best_value > 0.0:
            return best_genotype
        else:
            return (-1, -1)

    # --- Combination / normalization ---

    def combine(self, other: "GenotypingResult") -> None:
        """Add likelihoods from another result (non-normalized sum)."""
        for g, v in other._genotype_to_likelihood.items():
            self._genotype_to_likelihood[g] = self._genotype_to_likelihood.get(g, 0.0) + float(v)

    def normalize(self) -> None:
        total = sum(self._genotype_to_likelihood.values())
        if total > 0.0:
            self.divide_likelihoods_by(total)

    # --- Coverage / unique k-mers ---

    def set_unique_kmers(self, nr_unique_kmers: int) -> None:
        assert 0 <= nr_unique_kmers < 65536
        self.unique_kmers = int(nr_unique_kmers)

    def set_coverage(self, coverage: int) -> None:
        assert 0 <= coverage < 65536
        self.local_coverage = int(coverage)

    def nr_unique_kmers(self) -> int:
        return self.unique_kmers

    def coverage(self) -> int:
        return self.local_coverage

    # --- Queries ---

    def contains_no_likelihoods(self) -> bool:
        return len(self._genotype_to_likelihood) == 0

    def get_stored_likelihoods(self) -> Mapping[Tuple[int, int], float]:
        # Return a read-only view (in spirit); callers can copy if needed.
        return dict(self._genotype_to_likelihood)

    # --- Pretty printing (akin to operator<<) ---

    def __str__(self) -> str:
        lines = [
            f"haplotype allele 1: {self.haplotype_1}",
            f"haplotype allele 2: {self.haplotype_2}",
            f"local coverage: {self.local_coverage}",
            f"nr of unique kmers: {self.unique_kmers}",
        ]

        indices = list(self._genotype_to_likelihood.keys())
        indices.sort(key=lambda x: (int(x[0]), int(x[1])))

        # Use the sorted keys to print in logical order
        for a1, a2 in indices:
            v = self._genotype_to_likelihood[(a1, a2)]
            lines.append(f"{int(a1)}/{int(a2)}: {v}")

        return "\n".join(lines)