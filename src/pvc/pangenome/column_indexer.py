from typing import List, Optional, Tuple

class ColumnIndexer:
    """
    Python port of the C++ ColumnIndexer.

    Assumptions about `UniqueKmers` API for each entry in `unique_kmers`:
      - get_path_ids(only_paths: Optional[List[int]] = None) -> (paths: List[int], alleles: List[int])
        Returns (current_paths, current_alleles), both aligned lists.
      - is_undefined_allele(allele_id: int) -> bool
      - get_allele(path_id: int) -> int

    Behavior:
      - Scans columns (variants) and collects those with at least one non-reference,
        non-undefined allele (allele != 0 and not undefined) into `variant_positions`.
      - Stores the first column's path list as the canonical `paths`.
      - Provides helpers to map between variant index and original column index,
        fetch path/allele info, and iterate over diploid path-pair indices.
    """

    def __init__(
        self,
        unique_kmers: List,                      # List[UniqueKmers]
        only_paths: Optional[List[int]] = None,
    ) -> None:
        if not unique_kmers:
            raise RuntimeError("ColumnIndexer: unique_kmers is empty.")

        self._unique_kmers: List = unique_kmers
        self._variant_positions: List[int] = []
        self._paths: List[int] = []

        column_count = len(unique_kmers)

        for column_index in range(column_count):
            # Expect a tuple (current_paths, current_alleles)
            current_paths, current_alleles = unique_kmers[column_index].get_path_ids(only_paths)

            nr_paths = len(current_paths)
            if nr_paths == 0:
                raise RuntimeError(f"HMM::index_columns: column {column_index} is not covered by any paths.")

            # Initialize canonical path list from first column
            if column_index == 0:
                self._paths = list(current_paths)

            # Check if there exists any non-reference (allele != 0) and defined allele
            all_absent = True
            for i in range(nr_paths):
                allele = current_alleles[i]
                if allele != 0 and not unique_kmers[column_index].is_undefined_allele(allele):
                    all_absent = False
                    break

            if not all_absent:
                self._variant_positions.append(column_index)
        
        self._get_allele_cache = [[] for _ in range(column_count)]
        for column_index in range(column_count):
            self._get_allele_cache[column_index] = [
                unique_kmers[column_index].get_allele(path_id)
                for path_id in self._paths
            ]



    # --- API mirroring the C++ methods ---

    def get_variant_id(self, column_index: int) -> int:
        """
        Given a variant *index* (0..size()-1), return the original column index.
        """
        if column_index < 0 or column_index >= len(self._variant_positions):
            raise RuntimeError("ColumnIndexer::get_variant_id: column index does not exist.")
        return self._variant_positions[column_index]

    def size(self) -> int:
        """Number of retained variant positions."""
        return len(self._variant_positions)

    def nr_paths(self) -> int:
        """Number of paths."""
        return len(self._paths)

    def get_path(self, path_index: int) -> int:
        """Return the path ID at a given index."""
        if path_index < 0 or path_index >= len(self._paths):
            raise RuntimeError("ColumnIndexer::get_path: path_index does not exist.")
        return self._paths[path_index]

    def get_allele_cached(self, path_index: int, column_index: int) -> int:
        """
        Cached version of get_allele to improve performance.
        For a given path_index and *variant-index* (0..size()-1),
        return the allele covered by that path at the corresponding original column.
        """
        if column_index < 0 or column_index >= len(self._variant_positions):
            raise RuntimeError("ColumnIndex::get_allele_cached: column_index does not exist.")
        return self._get_allele_cache[column_index][path_index]

    def get_allele(self, path_index: int, column_index: int) -> int:
        """
        For a given path_index and *variant-index* (0..size()-1),
        return the allele covered by that path at the corresponding original column.
        """
        path_id = self.get_path(path_index)
        if column_index < 0 or column_index >= len(self._variant_positions):
            raise RuntimeError("ColumnIndex::get_allele: column_index does not exist.")
        variant_col = self._variant_positions[column_index]
        return self._unique_kmers[variant_col].get_allele(path_id)

    def get_path_ids_at(self, position: int) -> Tuple[int, int]:
        """
        Map a flattened index in [0, nr_paths^2) to a pair of path IDs (p2, p1),
        matching the C++ modulo arithmetic:
            p_id1 = position % nr_paths
            p_id2 = (position / nr_paths) % nr_paths
        Returns (p_id2, p_id1).
        """
        n = self.nr_paths()
        if position < 0 or position >= n * n:
            raise RuntimeError("ColumnIndexer::get_path_ids_at: index out of bounds.")
        p_id1 = position % n
        p_id2 = (position // n) % n
        return (self.get_path(p_id2), self.get_path(p_id1))