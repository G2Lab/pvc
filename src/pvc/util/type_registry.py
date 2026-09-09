POLY_HIGH_BIT = 1 << 31

class TypeRegistry:
    """Maintains id <-> name mapping while scanning the archive."""
    def __init__(self):
        self.id_to_name: Dict[int, str] = {}

    def process_polymorphic_header(self, poly_id: int, maybe_name: str) -> str:
        """
        Returns the base_id (without high bit). Registers mapping if the high bit is set.
        """
        if poly_id > POLY_HIGH_BIT:
            base_id = poly_id & ~POLY_HIGH_BIT
            if not maybe_name:
                raise ValueError(f"First-time polymorphic id {poly_id} missing polymorphic_name")
            # First occurrence: define the mapping
            self.id_to_name[base_id] = maybe_name
            return maybe_name
        else:
            base_id = poly_id
            if base_id not in self.id_to_name:
                raise ValueError(
                    f"Polymorphic id {base_id} seen before any mapping was defined."
                )
            return self.id_to_name[base_id]

    def name_for_id(self, base_id: int) -> str:
        return self.id_to_name[base_id]
