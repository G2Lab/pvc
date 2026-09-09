from dataclasses import dataclass
from typing import List, Optional, Union, IO, Dict, Any
import json

from pvc.pangenome.fasta_reader import FastaReader
from pvc.pangenome.DNA_sequence import DnaSequence
from pvc.pangenome.variant import Variant
from pvc.util.utils import _expect_keys

@dataclass
class Graph:
    # cereal save/load: archive(fasta_reader, chromosome, kmer_size, add_reference,
    #                          variants_deleted, variants, variant_ids)
    fasta_reader: FastaReader
    chromosome: str
    kmer_size: int
    add_reference: bool
    variants_deleted: bool
    variants: List[Optional[Variant]]            # shared_ptr<Variant> can be null
    variant_ids: List[List[str]]


    @staticmethod
    def from_cereal_data(d: Dict[str, Any]) -> "Graph":
        _expect_keys(d, [
            "value0", "value1", "value2", "value3",
            "value4", "value5", "value6"
        ], "Graph")

        fasta_reader = FastaReader.from_cereal_data(d["value0"])
        chromosome = str(d["value1"])
        kmer_size = int(d["value2"])
        add_reference = bool(d["value3"])
        variants_deleted = bool(d["value4"])
        variants: List[Optional[Variant]] = []
        for var in d["value5"]:
            if var is None:
                variants.append(None)
            else:
                variants.append(Variant.from_cereal_data(var["ptr_wrapper"]["data"]))
        variant_ids: List[List[str]] = []
        for vid_list in d["value6"]:
            variant_ids.append([str(vid) for vid in vid_list])
        
        return Graph(
            fasta_reader=fasta_reader,
            chromosome=chromosome,
            kmer_size=kmer_size,
            add_reference=add_reference,
            variants_deleted=variants_deleted,
            variants=variants,
            variant_ids=variant_ids
        )

def read_graph(filename):
    with open(filename, "r") as f:
        raw_json = json.load(f)

    return Graph.from_cereal_data(raw_json['value0'])