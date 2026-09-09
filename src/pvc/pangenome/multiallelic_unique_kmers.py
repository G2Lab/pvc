from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from pvc.util.utils import _expect_keys
from pvc.pangenome.unique_kmers import UniqueKmers
from pvc.pangenome.allele_info import AlleleInfo

def _decode_cereal_map(seq: List[Dict[str, Any]], key_name="key", val_name="value") -> Dict[Any, Any]:
    """
    cereal JSON for std::map is a list of {"key": <K>, "value": <V>}.
    """
    out = {}
    for kv in seq:
        _expect_keys(kv, [key_name, val_name], "map entry")
        out[kv[key_name]] = kv[val_name]
    return out

@dataclass
class MultiallelicUniqueKmers(UniqueKmers):
    @staticmethod
    def from_cereal_data(d: Dict[str, Any]) -> "MultiallelicUniqueKmers":
        """
        Expects a dict with keys: value0..value5 corresponding to:
          0: variant_pos (size_t -> int)
          1: local_coverage (float)
          2: current_index (size_t -> int)
          3: kmer_to_count (vector<unsigned short>)
          4: alleles (map<unsigned short, AlleleInfo>) as [{"key": <u16>, "value": {"value0":<KmerPath>, "value1":<bool>}}, ...]
          5: path_to_allele (vector<unsigned short>)
        """
        _expect_keys(d, ["value0", "value1", "value2", "value3", "value4", "value5"], "MultiallelicUniqueKmers")

        variant_pos = int(d["value0"])
        local_cov = float(d["value1"])
        current_idx = int(d["value2"])

        # vector<unsigned short> -> list[int]
        kmer_to_count = [int(x) for x in d["value3"]]

        # map<unsigned short, AlleleInfo>
        # cereal emits as list of {"key": u16, "value": <AlleleInfo-tuple>}
        raw_map_seq = d["value4"]
        m = _decode_cereal_map(raw_map_seq)  # keys stay ints
        alleles: Dict[int, AlleleInfo] = {int(k): AlleleInfo.from_cereal_data(v) for k, v in m.items()}

        # vector<unsigned short> -> list[int]
        path_to_allele = [int(x) for x in d["value5"]]

        return MultiallelicUniqueKmers(
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
        if 0 <= path_id < len(self.path_to_allele):
            return int(self.path_to_allele[path_id])
        else:
            raise IndexError(f"Path ID {path_id} out of range for MultiallelicUniqueKmers with {len(self.path_to_allele)} paths.")


def print_multiallelic_unique_kmers(uk):
    """
    Pretty-print a MultiallelicUniqueKmers object, mirroring the C++ operator<<.
    """

    print(f"MultiallelicUniqueKmers for variant: {uk.variant_pos}")

    # Kmer counts
    for i, count in enumerate(uk.kmer_to_count):
        print(f"  {i}: {count}")

    # Alleles
    print("alleles:")
    for allele_id, info in uk.alleles.items():
        # Use convert_to_string() if available, else fall back to dict string
        if hasattr(info.kmer_path, "convert_to_string"):
            kmer_path_str = info.kmer_path.convert_to_string()
        else:
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