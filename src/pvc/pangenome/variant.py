from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

from pvc.pangenome.DNA_sequence import DnaSequence
from pvc.util.utils import _expect_keys
from pvc.pangenome.genotyping_result import GenotypingResult


@dataclass
class VariantStats:
    nr_unique_kmers: int
    kmer_counts: Dict[int, int]  # C++ map<unsigned short,int>
    coverage: int                # C++ unsigned short

@dataclass
class Variant:
    # cereal order: left_flank, right_flank, inner_flanks, chromosome, start_position,
    # allele_sequences, allele_combinations, uncovered_alleles, paths, flanks_added
    left_flank: DnaSequence
    right_flank: DnaSequence
    inner_flanks: List[DnaSequence]
    chromosome: str
    start_position: int
    allele_sequences: List[List[DnaSequence]]
    allele_combinations: List[List[int]]   # C++ unsigned short IDs
    uncovered_alleles: List[List[int]]     # C++ unsigned short IDs
    paths: List[int]                       # C++ unsigned short IDs
    flanks_added: bool

    @staticmethod
    def from_cereal_data(d: Dict) -> "Variant":

        _expect_keys(d, [
            "value0", "value1", "value2", "value3", "value4",
            "value5", "value6", "value7", "value8", "value9"
        ], "Variant")

        left_flank = DnaSequence.from_cereal_data(d["value0"])
        right_flank = DnaSequence.from_cereal_data(d["value1"])
        inner_flanks = [DnaSequence.from_cereal_data(f) for f in d["value2"]]
        chromosome = str(d["value3"])
        start_position = int(d["value4"])
        allele_sequences = [
            [DnaSequence.from_cereal_data(seq) for seq in allele_list]
            for allele_list in d["value5"]
        ]
        allele_combinations = [
            [int(aid) for aid in comb]
            for comb in d["value6"]
        ]
        uncovered_alleles = [
            [int(aid) for aid in ulist]
            for ulist in d["value7"]
        ]
        paths = [int(pid) for pid in d["value8"]]
        flanks_added = bool(d["value9"])

        return Variant(
            left_flank=left_flank,
            right_flank=right_flank,
            inner_flanks=inner_flanks,
            chromosome=chromosome,
            start_position=start_position,
            allele_sequences=allele_sequences,
            allele_combinations=allele_combinations,
            uncovered_alleles=uncovered_alleles,
            paths=paths,
            flanks_added=flanks_added
        )

    @classmethod
    def from_strings(
        cls,
        left_flank: str,
        right_flank: str,
        chromosome: str,
        start_position: int,
        end_position: int,          # kept for API parity; not stored explicitly
        alleles: List[str],
        paths: List[int],
    ) -> "Variant":
        """
        Python port of:
        Variant::Variant(string left_flank, string right_flank, string chromosome,
                         size_t start_position, size_t end_position,
                         vector<string> alleles, vector<unsigned short> paths)
        """

        if len(alleles) > 65535:
            raise RuntimeError(
                "Variant::Variant: number of alleles per variant exceeds 65536. "
                "Current implementation does not support higher numbers."
            )

        if len(paths) > 65535:
            raise RuntimeError(
                "Variant::Variant: number of paths exceeds 65536. "
                "Current implementation does not support higher numbers."
            )

        left = DnaSequence.from_string(left_flank)
        right = DnaSequence.from_string(right_flank)

        # Single bubble: allele_sequences[0] is the list of alleles for this variant
        allele_sequences = [[DnaSequence.from_string(a) for a in alleles]]

        # Identity allele combinations: combined allele index -> [allele_id]
        allele_combinations = [[i] for i in range(len(alleles))]

        # No uncovered alleles info here: initialize as one empty list (like C++ set_values would)
        uncovered_alleles: List[List[int]] = [[]]

        return cls(
            left_flank=left,
            right_flank=right,
            inner_flanks=[],               # single variant: no inner flanks
            chromosome=chromosome,
            start_position=int(start_position),
            allele_sequences=allele_sequences,
            allele_combinations=allele_combinations,
            uncovered_alleles=uncovered_alleles,
            paths=list(paths),
            flanks_added=False,
        )

    @classmethod
    def from_dna_sequences(
        cls,
        left_flank: DnaSequence,
        right_flank: DnaSequence,
        chromosome: str,
        start_position: int,
        end_position: int,          # kept for API parity; not stored explicitly
        alleles: List[DnaSequence],
        paths: List[int],
    ) -> "Variant":
        """
        Python port of:
        Variant::Variant(DnaSequence& left_flank, DnaSequence& right_flank,
                         string chromosome, size_t start_position, size_t end_position,
                         vector<DnaSequence>& alleles, vector<unsigned short>& paths)
        """

        if len(alleles) > 65535:
            raise RuntimeError(
                "Variant::Variant: number of alleles per variant exceeds 65536. "
                "Current implementation does not support higher numbers."
            )

        if len(paths) > 65535:
            raise RuntimeError(
                "Variant::Variant: number of paths exceeds 65536. "
                "Current implementation does not support higher numbers."
            )

        # Single bubble: allele_sequences[0] is the list of alleles for this variant
        allele_sequences = [list(alleles)]

        # Identity allele combinations: combined allele index -> [allele_id]
        allele_combinations = [[i] for i in range(len(alleles))]

        uncovered_alleles: List[List[int]] = [[]]

        return cls(
            left_flank=left_flank,
            right_flank=right_flank,
            inner_flanks=[],               # single variant: no inner flanks
            chromosome=chromosome,
            start_position=int(start_position),
            allele_sequences=allele_sequences,
            allele_combinations=allele_combinations,
            uncovered_alleles=uncovered_alleles,
            paths=list(paths),
            flanks_added=False,
        )
    
    def is_combined(self) -> bool:
        """Returns True if this variant represents multiple combined variants."""
        return len(self.allele_sequences) > 1
    
    def get_allele_on_path(self, index: int) -> str:
        """
        Python port of:
        string Variant::get_allele_string(size_t index) const
        """
        if index >= len(self.allele_combinations):
            raise RuntimeError("Variant::get_allele_string: Index out of bounds.")

        # Start with left flank if flanks have been added
        if self.flanks_added:
            result = self.left_flank.to_string()
        else:
            result = ""

        nr_alleles = len(self.allele_combinations[index])

        for i in range(nr_alleles):
            allele_id = self.allele_combinations[index][i]
            # append allele sequence at this bubble
            result += self.allele_sequences[i][allele_id].to_string()

            # append inner flank between bubbles
            if i < nr_alleles - 1:
                result += self.inner_flanks[i].to_string()

        # append right flank if present
        if self.flanks_added:
            result += self.right_flank.to_string()

        return result
    
    def nr_of_alleles(self) -> int:
        return len(self.allele_combinations)
    
    def is_undefined_allele(self, allele_id: int) -> bool:
        """
        Python port of:
        bool Variant::is_undefined_allele(size_t allele_id) const
        """
        if allele_id >= len(self.allele_combinations):
            raise RuntimeError("Variant::is_undefined_allele: allele_id out of range")

        for i in range(len(self.allele_combinations[allele_id])):
            allele = self.allele_combinations[allele_id][i]
            if self.allele_sequences[i][allele].contains_undefined():
                return True

        return False

    def separate_variants(
        self,
        input_genotyping: Optional[GenotypingResult] = None,
        skip_flanks: bool = False
    ) -> Tuple[List[Any], Optional[List[GenotypingResult]]]:
        """
        Python port of:
        void Variant::separate_variants(vector<Variant>* res_vars, const GenotypingResult* in_gt,
                                        vector<GenotypingResult>* res_gt, bool skip_flanks) const;

        Returns:
        (resulting_variants, resulting_g enotyping or None)
        """
        nr_variants = len(self.allele_sequences)
        assert len(self.uncovered_alleles) == nr_variants, f"uncovered_alleles size mismatch ({len(self.uncovered_alleles)} vs {nr_variants})"

        # --- Construct per-variant paths (paths_per_variant[v][path_idx] = allele_id at variant v for that path)
        paths_per_variant: List[List[int]] = [[] for _ in range(nr_variants)]
        for path_idx in range(len(self.paths)):
            a = self.paths[path_idx]
            comb = self.allele_combinations[a]
            assert len(comb) == nr_variants, "allele_combinations[a] size mismatch"
            for v in range(nr_variants):
                allele_id = comb[v]  # allele id of variant v carried on this path
                paths_per_variant[v].append(allele_id)

        # --- Build reference_allele pieces (used to reconstruct flanks), unless skipping
        reference_allele: List[DnaSequence] = []
        if not skip_flanks:
            # take the "reference" (allele index from combined allele 0) at each position
            for i in range(nr_variants):
                allele_id = self.allele_combinations[0][i]
                reference_allele.append(self.allele_sequences[i][allele_id])
                if i < nr_variants - 1:
                    reference_allele.append(self.inner_flanks[i])

            # add flanking ends
            reference_allele.insert(0, self.left_flank)
            reference_allele.append(self.right_flank)

        # --- Iterate variants, build per-variant objects (and per-variant genotyping if provided)
        resulting_variants: List[Variant] = []
        resulting_genotyping: Optional[List[GenotypingResult]] = [] if input_genotyping is not None else None

        current_start = int(self.start_position)

        for i in range(nr_variants):
            if not skip_flanks:
                # center index in reference_allele for variant i is i*2 + 1
                center_idx = i * 2 + 1
                left = _concat_left_flank(reference_allele, center_idx, want_len=self.left_flank.size())
                right = _concat_right_flank(reference_allele, center_idx, want_len=self.right_flank.size())
            else:
                # create empty flanks
                left = DnaSequence.from_string("")
                right = DnaSequence.from_string("")

            alleles_i: List[DnaSequence] = list(self.allele_sequences[i])
            # end = start + length of reference allele (alleles_i[0])
            current_end = current_start + alleles_i[0].size()

            # New single-variant object
            v_single = Variant.from_dna_sequences(
                left_flank=left,
                right_flank=right,
                chromosome=self.chromosome,
                start_position=current_start,
                end_position=current_end,
                alleles=alleles_i,
                paths=paths_per_variant[i]
            )

            resulting_variants.append(v_single)

            if input_genotyping is not None and resulting_genotyping is not None:
                # Precompute mapping from combined-allele index → single-variant allele for this position i
                precomputed_ids = [self.allele_combinations[a0][i] for a0 in range(len(self.allele_combinations))]

                g = GenotypingResult()

                if not input_genotyping.contains_no_likelihoods():
                    # For every stored genotype (a0, a1) on the combined variant,
                    # map to (single_allele0, single_allele1) at position i, accumulate likelihood.
                    for genotype, like in input_genotyping._genotype_to_likelihood.items():
                
                        a0, a1 = genotype  # tuple[int,int]
                        single_allele0 = precomputed_ids[a0]
                        single_allele1 = precomputed_ids[a1]
                        g.add_to_likelihood(single_allele0, single_allele1, like)

                # Map phased haplotype alleles as well
                h0, h1 = input_genotyping.get_haplotype()
                g.add_first_haplotype_allele(precomputed_ids[h0])
                g.add_second_haplotype_allele(precomputed_ids[h1])

                resulting_genotyping.append(g)

            # advance start for next variant
            current_start = current_end
            if i < nr_variants - 1:
                current_start += self.inner_flanks[i].size()

        return resulting_variants, resulting_genotyping

    def get_end_position(self) -> int:
        end_position = self.start_position
        for i in range(len(self.allele_sequences)):
            end_position += self.allele_sequences[i][0].size()
            if i < (len(self.allele_sequences) - 1):
                end_position += self.inner_flanks[i].size()
        return end_position

    def combine_variants(self, v2: "Variant") -> None:
        """
        Combine this variant with another variant v2 that follows it.

        This merges two nearby variants into a single multi-bubble variant.
        Each unique combination of (allele_from_v1, allele_from_v2) becomes
        a new combined allele.

        Python port of:
        void Variant::combine_variants(Variant const &v2)

        Args:
            v2: The variant to merge (must start after this variant ends)
        """
        end_position = self.get_end_position()

        if v2.start_position < end_position:
            raise RuntimeError("Variant::combine_variants: Variants are overlapping.")

        if self.flanks_added or v2.flanks_added:
            raise RuntimeError(
                "Variant::combine_variants: Variant objects can only be combined "
                "if no flanks were added."
            )

        kmersize_v1 = self.left_flank.size()
        kmersize_v2 = v2.left_flank.size()
        if kmersize_v1 != kmersize_v2:
            raise RuntimeError("Variant::combine_variants: kmersizes are not the same.")

        dist = v2.start_position - end_position
        if dist > kmersize_v1 or self.chromosome != v2.chromosome:
            raise RuntimeError(
                "Variant::combine_variants: Variant objects are more than kmersize bases apart."
            )

        if len(self.paths) != len(v2.paths):
            raise RuntimeError(
                "Variant::combine_variants: Variant objects not covered by the same paths."
            )

        # Consider all combinations of alleles defined by paths (= new alleles)
        # index_to_path: path_index -> (left_allele, right_allele)
        # path_to_index: (left_allele, right_allele) -> list of path indices
        index_to_path: Dict[int, Tuple[int, int]] = {}
        path_to_index: Dict[Tuple[int, int], List[int]] = {}

        for p in range(len(self.paths)):
            left_allele = self.paths[p]
            right_allele = v2.paths[p]
            index_to_path[p] = (left_allele, right_allele)
            key = (left_allele, right_allele)
            if key not in path_to_index:
                path_to_index[key] = []
            path_to_index[key].append(p)

        # Add REF-REF allele (0, 0) if not already present
        ref_path = (0, 0)
        if ref_path not in path_to_index:
            path_to_index[ref_path] = []

        new_paths = [0] * len(self.paths)
        new_alleles: List[List[int]] = []
        allele_index = 0

        assert len(path_to_index) < 65536, "Too many allele combinations"

        # Construct new allele sequences
        # Sort keys to ensure deterministic ordering (REF-REF first)
        sorted_keys = sorted(path_to_index.keys())

        for key in sorted_keys:
            path_indices = path_to_index[key]
            for p in path_indices:
                new_paths[p] = allele_index

            left_allele_combo = list(self.allele_combinations[key[0]])
            right_allele_combo = list(v2.allele_combinations[key[1]])
            combined = left_allele_combo + right_allele_combo
            new_alleles.append(combined)
            allele_index += 1

        # Construct sequence between variants (inner flank)
        # This is the portion of self.right_flank that spans from end of v1 to start of v2
        flank_str = self.right_flank.to_string()[:dist]
        flank = DnaSequence.from_string(flank_str)

        # Update inner_flanks: add the connecting flank, then v2's inner flanks
        self.inner_flanks.append(flank)
        self.inner_flanks.extend(v2.inner_flanks)

        # Update variant
        self.right_flank = v2.right_flank
        self.allele_combinations = new_alleles
        self.allele_sequences.extend(v2.allele_sequences)
        self.uncovered_alleles.extend(v2.uncovered_alleles)
        self.paths = new_paths

