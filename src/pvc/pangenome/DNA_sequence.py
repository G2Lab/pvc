from dataclasses import dataclass
from typing import List, Dict, Any

from pvc.util.utils import _expect_keys

# --- helpers to mirror C++ encode/decode/complement assumptions ---
# C++ encode() returns: A=0, C=1, G=2, T=3, undefined=4
# decode() maps back to 'A','C','G','T','N'
_DECODE = {0: "A", 1: "C", 2: "G", 3: "T"}
_DEF_CHAR = "N"  # for undefined value 4 (or anything else)

def _decode_nibble(n: int) -> str:
    return _DECODE.get(n, _DEF_CHAR)

@dataclass
class DnaSequence:
    # cereal fields (matches: sequence, even_length, is_undefined)
    # C++ std::vector<unsigned char> → List[int] with values 0–255
    sequence: List[int]
    even_length: bool
    is_undefined: bool

    @staticmethod
    def from_cereal_data(d: Dict[str, Any]) -> "DnaSequence":
        """
        cereal packs the std::vector<uint8_t> + two bools as value0, value1, value2.
        value0 may arrive as a bytes/bytearray or a list[int]; normalize to List[int].
        """
        _expect_keys(d, ["value0", "value1", "value2"], "DnaSequence")
        v0 = d["value0"]
        if isinstance(v0, (bytes, bytearray)):
            seq = list(v0)
        else:
            # assume iterable of ints
            seq = list(v0)

        even_length = bool(d["value1"])
        is_undef = bool(d["value2"])

        return DnaSequence(
            sequence=seq,
            even_length=even_length,
            is_undefined=is_undef
        )

    @staticmethod
    def from_string(s: str) -> "DnaSequence":
        """
        Create a DnaSequence from a string of bases (A,C,G,T,N).
        Mirrors C++ encode() logic.
        """
        # Encoding: A=0, C=1, G=2, T=3, N/other=4
        _ENCODE = {"A": 0, "a": 0, "C": 1, "c": 1, "G": 2, "g": 2, "T": 3, "t": 3}

        seq_bytes: List[int] = []
        n = len(s)
        even_length = (n % 2 == 0)
        is_undef = False

        # Process two chars at a time into one byte
        i = 0
        while i < n:
            # High nibble
            high_char = s[i]
            high_nib = _ENCODE.get(high_char, 4)
            if high_nib == 4:
                is_undef = True

            # Low nibble (or 0 if odd length and this is last char)
            if i + 1 < n:
                low_char = s[i + 1]
                low_nib = _ENCODE.get(low_char, 4)
                if low_nib == 4:
                    is_undef = True
            else:
                low_nib = 0

            byte = (high_nib << 4) | low_nib
            seq_bytes.append(byte)
            i += 2

        return DnaSequence(
            sequence=seq_bytes,
            even_length=even_length,
            is_undefined=is_undef
        )

    def size(self) -> int:
        """
        Number of bases, mirroring the C++ size():
        result = sequence.size() * 2; if not even_length -> subtract 1.
        """
        n = len(self.sequence) * 2
        if not self.even_length:
            n -= 1
        return n

    def __len__(self) -> int:
        return self.size()

    def __getitem__(self, position: int) -> str:
        """
        Mirror C++ operator[]: return the decoded base char at zero-based index.
        """
        if position < 0 or position >= self.size():
            raise IndexError("DnaSequence.__getitem__: index out of bounds")

        byte = self.sequence[position // 2]
        if position % 2 == 0:
            # even index -> high nibble
            nib = (byte >> 4) & 0x0F
        else:
            # odd index -> low nibble
            nib = byte & 0x0F
        return _decode_nibble(nib)

    def base_at(self, position: int) -> "DnaSequence":
        """
        Mirror C++ base_at(): return a new DnaSequence containing only the base at 'position'.
        The single-base sequence is stored as one nibble (so even_length=False).
        """
        if position < 0 or position >= self.size():
            raise IndexError("DnaSequence.base_at: index out of bounds")

        byte = self.sequence[position // 2]
        if position % 2 == 0:
            # keep high nibble, zero low nibble
            only = byte & 0xF0
        else:
            # move low nibble into high nibble, zero low nibble
            only = (byte & 0x0F) << 4

        # is_undefined can be recomputed from the nibble (bit 0x4 set in either nibble).
        # Since we keep only the high nibble, check bit 0x40 on the byte we store.
        is_undef = (only & 0x40) != 0

        return DnaSequence(sequence=[only], even_length=False, is_undefined=is_undef)

    def to_string(self) -> str:
        """
        Mirror C++ to_string(): iterate over logical length and decode each nibble.
        """
        # Fast path: iterate by bytes, but respect odd trailing nibble.
        out_chars: List[str] = []
        total = self.size()
        full_bytes, has_trailing = divmod(total, 2)

        # process all full bytes (2 bases each)
        for i in range(full_bytes):
            b = self.sequence[i]
            out_chars.append(_decode_nibble((b >> 4) & 0x0F))  # high
            out_chars.append(_decode_nibble(b & 0x0F))         # low

        # trailing high nibble if odd length
        if has_trailing:
            b = self.sequence[full_bytes]
            out_chars.append(_decode_nibble((b >> 4) & 0x0F))

        return "".join(out_chars)

    def contains_undefined(self) -> bool:
        """
        Matches C++ contains_undefined(): return the stored flag.
        (In C++ it’s tracked while appending and sometimes recomputed for substrings.)
        """
        return bool(self.is_undefined)