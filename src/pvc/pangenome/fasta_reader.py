from typing import Any, Dict, List
from dataclasses import dataclass, field
from pathlib import Path

from pvc.pangenome.DNA_sequence import DnaSequence


@dataclass
class FastaReader:
    """
    Read and index FASTA files for random access to sequences.
    """
    name_to_sequence: Dict[str, DnaSequence] = field(default_factory=dict)
    # Cache for string representations (for fast substring access)
    _string_cache: Dict[str, str] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_cereal_data(d: Dict[str, Any]) -> "FastaReader":
        """Load from cereal JSON data."""
        if "value0" not in d:
            raise KeyError("Missing key 'name_to_sequence' in FastaReader data")

        return FastaReader(
            name_to_sequence={
                str(kv["key"]): DnaSequence.from_cereal_data(kv["value"]["ptr_wrapper"]["data"])
                for kv in d["value0"]
            }
        )

    @staticmethod
    def from_file(fasta_path: str) -> "FastaReader":
        """
        Load sequences from a FASTA file.

        Args:
            fasta_path: Path to uncompressed FASTA file

        Returns:
            FastaReader with all sequences loaded
        """
        name_to_sequence: Dict[str, DnaSequence] = {}
        current_name = None
        current_seq_parts: List[str] = []

        with open(fasta_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith(">"):
                    # Save previous sequence
                    if current_name is not None:
                        seq_str = "".join(current_seq_parts)
                        name_to_sequence[current_name] = DnaSequence.from_string(seq_str)

                    # Start new sequence
                    # Extract name (everything after > until first whitespace)
                    header = line[1:]
                    current_name = header.split()[0] if header else header
                    current_seq_parts = []
                else:
                    current_seq_parts.append(line)

            # Save last sequence
            if current_name is not None:
                seq_str = "".join(current_seq_parts)
                name_to_sequence[current_name] = DnaSequence.from_string(seq_str)

        return FastaReader(name_to_sequence=name_to_sequence)

    def get_sequence_names(self) -> List[str]:
        """Get list of all sequence/chromosome names."""
        return list(self.name_to_sequence.keys())

    def get_size_of(self, name: str) -> int:
        """Get the length of a sequence."""
        if name not in self.name_to_sequence:
            raise KeyError(f"Sequence '{name}' not found in FASTA")
        # Use cached string if available for O(1) length
        if name in self._string_cache:
            return len(self._string_cache[name])
        return self.name_to_sequence[name].size()

    def _get_string(self, name: str) -> str:
        """Get cached string representation of a sequence."""
        if name not in self._string_cache:
            self._string_cache[name] = self.name_to_sequence[name].to_string()
        return self._string_cache[name]

    def get_subsequence(self, name: str, start: int, end: int) -> DnaSequence:
        """
        Get a subsequence from a chromosome.

        Args:
            name: Chromosome/sequence name
            start: Start position (0-based, inclusive)
            end: End position (0-based, exclusive)

        Returns:
            DnaSequence containing the subsequence
        """
        if name not in self.name_to_sequence:
            raise KeyError(f"Sequence '{name}' not found in FASTA")

        seq_str = self._get_string(name)

        # Clamp to valid range
        start = max(0, start)
        end = min(len(seq_str), end)

        return DnaSequence.from_string(seq_str[start:end])

    def get_full_sequence(self, name: str) -> DnaSequence:
        """Get the full sequence for a chromosome."""
        if name not in self.name_to_sequence:
            raise KeyError(f"Sequence '{name}' not found in FASTA")
        return self.name_to_sequence[name]
