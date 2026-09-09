from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import json

from pvc.pangenome.biallelic_unique_kmers import BiallelicUniqueKmers
from pvc.pangenome.multiallelic_unique_kmers import MultiallelicUniqueKmers
from pvc.util.type_registry import TypeRegistry
from pvc.util.utils import _expect_keys

POLY_HIGH_BIT = 1 << 31



def _decode_cereal_map(seq: List[Dict[str, Any]], key_name="key", val_name="value") -> Dict[Any, Any]:
    """
    cereal JSON for std::map is a list of {"key": <K>, "value": <V>}.
    """
    out = {}
    for kv in seq:
        _expect_keys(kv, [key_name, val_name], "map entry")
        out[kv[key_name]] = kv[val_name]
    return out

def _decode_cereal_map_bool_keys(seq: List[Dict[str, Any]]) -> Dict[bool, Any]:
    """
    cereal JSON for std::map<K,V> is a list of {"key": <K>, "value": <V>}.
    This variant expects K=bool (true/false).
    """
    out: Dict[bool, Any] = {}
    for kv in seq:
        _expect_keys(kv, ["key", "value"], "map<bool,*> entry")
        out[bool(kv["key"])] = kv["value"]
    return out

@dataclass
class UniqueKmersMap:
    def __init__(self, kmer_size, unique_kmers, runtimes, sampling_runtimes):
        self.kmer_size = kmer_size
        self.unique_kmers = {}
        self.type_registry = TypeRegistry()

        for kv in unique_kmers:
            chrom = kv["key"]
            unique_kmers_per_chrom = kv["value"]

            self.unique_kmers[chrom] = []

            for variant in unique_kmers_per_chrom:
                poly_id = variant['polymorphic_id']
                poly_maybe_name = variant.get('polymorphic_name', None)
                poly_name = self.type_registry.process_polymorphic_header(poly_id, poly_maybe_name)
                if poly_name == "BiallelicUniqueKmers":
                    self.unique_kmers[chrom].append(BiallelicUniqueKmers.from_cereal_data(variant["ptr_wrapper"]["data"]))
                elif poly_name == "MultiallelicUniqueKmers":
                    self.unique_kmers[chrom].append(MultiallelicUniqueKmers.from_cereal_data(variant["ptr_wrapper"]["data"]))
                else:
                    assert f"Could not find {poly_name}"
        self.runtimes = runtimes
        self.sampling_runtimes = sampling_runtimes

def read_unique_kmers_map_from_cereal(filename):
    with open(filename, "r") as f:
        raw_json = json.load(f)

    unique_kmers_map_json = raw_json['value0']
    unique_kmers_map = UniqueKmersMap(unique_kmers_map_json["value0"], unique_kmers_map_json["value1"], unique_kmers_map_json["value2"], unique_kmers_map_json["value3"])

    return unique_kmers_map