from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from pvc.pangenome.kmer_path import KmerPath
from pvc.util.utils import _expect_keys

@dataclass
class AlleleInfo:
    kmer_path: KmerPath
    is_undefined: bool

    @staticmethod
    def from_cereal_data(d: Dict[str, Any]) -> "AlleleInfo":
        _expect_keys(d, ["value0", "value1"], "AlleleInfo")
        kp = KmerPath.from_cereal_data(d["value0"])
        is_undef = bool(d["value1"])
        return AlleleInfo(kmer_path=kp, is_undefined=is_undef)