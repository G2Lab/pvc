from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class UniqueKmers:
    variant_pos: int
    local_coverage: float
    current_index: int
    kmer_to_count: List[int]
    alleles: Dict[Any, Any]
    path_to_allele: List[Any]

    def get_allele_ids(self) -> List[int]:
        return self.alleles

    def is_undefined_allele(self, allele_id: int) -> bool: 
        if allele_id in self.alleles:
            return self.alleles[allele_id].is_undefined
        return False
 
    def size(self) -> int: 
        return self.current_index

    def kmer_on_allele(self, kmer_index: int, allele_id: int):
        return self.alleles[allele_id].kmer_path.get_position(kmer_index) # might be probablematic

    def get_coverage(self):
        return self.local_coverage

    def get_readcount_of(self, kmer_index: int) -> int: 
        return self.kmer_to_count[kmer_index]

    def get_variant_position(self) -> int:
        return self.variant_pos
