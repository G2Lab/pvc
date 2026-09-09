from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from pvc.pangenome.kmer_path16 import KmerPath16
from pvc.util.utils import _expect_keys

@dataclass
class AlleleInfo16:
    kmer_path: KmerPath16
    is_undefined: bool

    @staticmethod
    def from_cereal_data(d: Dict[str, Any]) -> "AlleleInfo16":
        _expect_keys(d, ["value0", "value1"], "AlleleInfo16")
        kp = KmerPath16.from_cereal_data(d["value0"])
        is_undef = bool(d["value1"])
        return AlleleInfo16(kmer_path=kp, is_undefined=is_undef)