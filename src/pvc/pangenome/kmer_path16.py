from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

@dataclass
class KmerPath16:
    offset: int
    kmers: int

    @staticmethod
    def from_cereal_data(d: Dict[str, Any]) -> "KmerPath16":
        # If you have the concrete fields later, decode them here.
        return KmerPath16(offset=d["value0"], kmers=d["value1"])

    def get_position(self, index: int) -> int:
        upper_limit = self.offset + 16
        lower_limit = self.offset
        if  (index < lower_limit) or (index >= upper_limit):
            return 0
        position = index - self.offset

        if self.kmers & (1 << position):
            return 1
        else:
            return 0