from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from pvc.pangenome.unique_kmers import UniqueKmers
from pvc.pangenome.allele_info16 import AlleleInfo16

def _expect_keys(d: Dict[str, Any], keys: List[str], ctx: str = ""):
    for k in keys:
        if k not in d:
            raise KeyError(f"Missing key '{k}' in {ctx or 'object'}; got keys={list(d.keys())}")

def _decode_cereal_map_bool_keys(seq: List[Dict[str, Any]]) -> Dict[bool, Any]:
    """
    cereal JSON for std::map<K,V> is a list of {"key": <K>, "value": <V>}.
    This variant expects K=bool (true/false).
    """
    out: Dict[int, Any] = {}
    for kv in seq:
        _expect_keys(kv, ["key", "value"], "map<bool,*> entry")
        out[int(kv["key"])] = kv["value"]
    return out

@dataclass
class BiallelicUniqueKmers(UniqueKmers):
    @staticmethod
    def from_cereal_data(d: Dict[str, Any]) -> "BiallelicUniqueKmers":
        """
        Expects keys value0..value5 corresponding to:
          0: variant_pos
          1: local_coverage
          2: current_index
          3: kmer_to_count (vector<unsigned short>)
          4: alleles (map<bool, AlleleInfo16>) as [{"key": <bool>, "value": {"value0":<KmerPath16>, "value1":<bool>}}, ...]
          5: path_to_allele (vector<bool>)
        """
        _expect_keys(d, ["value0", "value1", "value2", "value3", "value4", "value5"], "BiallelicUniqueKmers")

        variant_pos = int(d["value0"])
        local_cov = float(d["value1"])
        current_idx = int(d["value2"])

        # vector<unsigned short> → list[int]
        kmer_to_count = [int(x) for x in d["value3"]]

        # map<bool, AlleleInfo16>
        raw_map_seq = d["value4"]
        m_bool = _decode_cereal_map_bool_keys(raw_map_seq)
        alleles = {int(k): AlleleInfo16.from_cereal_data(v) for k, v in m_bool.items()}

        # vector<bool> → list[bool]
        path_to_allele = [int(x) for x in d["value5"]]

        return BiallelicUniqueKmers(
            variant_pos=variant_pos,
            local_coverage=local_cov,
            current_index=current_idx,
            kmer_to_count=kmer_to_count,
            alleles=alleles,
            path_to_allele=path_to_allele,
        )
    

    def get_path_ids(self, only_include: Optional[List[int]] = None) -> Tuple[List[int], List[int]]:
        """
        Python port of:
          void BiallelicUniqueKmers::get_path_ids(vector<unsigned short>& p,
                                                  vector<unsigned short>& a,
                                                  vector<unsigned short>* only_include)

        Returns:
          (paths, alleles) where:
            - paths[i] is a path id
            - alleles[i] is the allele id covered by that path at this variant
        """
        paths: List[int] = []
        alleles_out: List[int] = []

        if only_include is not None:
            # only return paths contained in only_include and valid for this object
            for pid in only_include:
                if 0 <= pid < len(self.path_to_allele):
                    paths.append(int(pid))
                    alleles_out.append(int(self.path_to_allele[pid]))
        else:
            # return all paths and corresponding alleles
            for pid, allele in enumerate(self.path_to_allele):
                paths.append(int(pid))
                alleles_out.append(int(allele))

        return paths, alleles_out

    def get_allele(self, path_id: int) -> int:
        """
        Get the allele id covered by the given path at this variant.
        """
        if 0 <= path_id < len(self.path_to_allele):
            return int(self.path_to_allele[path_id])
        else:
            raise IndexError(f"Path ID {path_id} out of bounds for path_to_allele of size {len(self.path_to_allele)}")

def print_biallelic_unique_kmers(uk):
    """
    Pretty-print a BiallelicUniqueKmers object (like operator<< in C++).
    """

    print(f"BiallelicUniqueKmers for variant: {uk.variant_pos}")

    # Kmer counts
    for i, count in enumerate(uk.kmer_to_count):
        print(f"  {i}: {count}")

    # Alleles
    print("alleles:")
    for allele_id, info in uk.alleles.items():
        # convert_to_string() analog — if KmerPath16.raw has a readable form
        if hasattr(info.kmer_path, "convert_to_string"):
            kmer_path_str = info.kmer_path.convert_to_string()
        else:
            # fallback: serialize raw dict
            kmer_path_str = str(info.kmer_path.raw)
        print(f"  {int(allele_id)}\t{kmer_path_str}")

    # Undefined alleles
    print("undefined alleles:")
    for allele_id, info in uk.alleles.items():
        if info.is_undefined:
            print(f"  {int(allele_id)}")

    # Paths
    print("paths:")
    for i, allele in enumerate(uk.path_to_allele):
        print(f"  {i} covers allele {int(allele)}")

    # Local coverage
    print(f"local coverage: {uk.local_coverage}")