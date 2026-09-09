"""Core PVC defaults; executable locations can be overridden by environment."""
from pathlib import Path
import os
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PVC_TOOLS = ("pvc", "pvc-light", "pvc-medium", "pvc-heavy", "pvc-light-gpu")


def _executable(variable, name, bundled=None):
    if os.environ.get(variable):
        candidate = os.path.expanduser(os.environ[variable])
        return Path(shutil.which(candidate) or candidate)
    if bundled is not None and bundled.is_file():
        return bundled
    return Path(shutil.which(name) or name)


PANGENIE_BIN = _executable("PANGENIE_BIN", "PanGenie", PROJECT_ROOT / "src/pangenie/PanGenie")
PANGENIE_INDEX_BIN = _executable("PANGENIE_INDEX_BIN", "PanGenie-index", PROJECT_ROOT / "src/pangenie/PanGenie-index")
PVC_INDEX_BINS = {tool: _executable("PVC_INDEX_BIN", "PanGenie-process", PROJECT_ROOT / "src/pangenie/PanGenie-process") for tool in PVC_TOOLS}
PVC_READMAP_BIN = _executable("PVC_READMAP_BIN", "PanGenie-readmap")
PLINK_BIN = _executable("PLINK_BIN", "plink", PROJECT_ROOT / "tools/plink_bin/plink")
BCFTOOLS_BIN = _executable("BCFTOOLS_BIN", "bcftools")
GATK_BIN_CANDIDATES = (_executable("GATK_BIN", "gatk"),)
SAMTOOLS_BIN_CANDIDATES = (_executable("SAMTOOLS_BIN", "samtools"),)
GATK_READ_ROOT_CANDIDATES = (Path(os.environ.get("PVC_READ_ROOT", PROJECT_ROOT / "data/reads")),)
PLINK_BLOCKS_MAX_KB = 200
PLINK_BLOCKS_MIN_MAF = 0.05
PVC_GENOTYPE_MEAN_KMER_ABUNDANCE = 30
max_variants_per_superblock = 10_000
max_blocks_per_superblock = 5_000
max_genotype_state_bytes = 512 * 1024 * 1024
