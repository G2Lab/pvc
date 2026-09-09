from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

@dataclass
class KmerPath:
    offset: int
    kmers: int

    @staticmethod
    def from_cereal_data(d: Dict[str, Any]) -> "KmerPath":
        # If you later know the exact fields, replace with a structured decoder.
        return KmerPath(d["value0"], d["value1"])

    def get_position(self, index: int) -> int:
        upper_limit = self.offset + 32
        lower_limit = self.offset
        if  (index < lower_limit) or (index >= upper_limit):
            return 0
        position = index - self.offset

        if self.kmers & (1 << position):
            return 1
        else:
            return 0
