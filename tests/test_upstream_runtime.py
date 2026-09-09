import hashlib
import importlib
import json
from pathlib import Path


def test_vendored_runtime_matches_upstream_manifest():
    root = Path(__file__).resolve().parents[1] / 'src/crypto/CrypTen'
    provenance = json.loads((root / 'UPSTREAM.json').read_text())
    assert provenance['repository'] == 'https://github.com/facebookresearch/CrypTen'
    assert provenance['commit'] == '775868a02d6dac50774ce376a55b01fbd8bd85b6'
    for name, expected in provenance['files_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    actual = {str(path.relative_to(root)) for path in root.rglob('*.py')}
    expected = {name for name in provenance['files_sha256'] if name.endswith('.py')}
    assert actual == expected


def test_upstream_import_and_core_integrations():
    import crypten
    from crypten.config import cfg
    assert crypten.__version__ == '0.4.0'
    assert not hasattr(crypten, 'tensors')
    assert not hasattr(cfg.mpc, 'csprng')
    for module in ['pvc.config', 'pvc.genotype.pvc_genotype',
                   'pvc.genotype.private.pvc_private_genotype', 'pvc.util.protocols',
                   'pvc.index.pvc_index', 'pangenie.pangenie', 'gatk.gatk']:
        importlib.import_module(module)
    from pvc.config import PVC_READMAP_BIN, PANGENIE_BIN, PANGENIE_INDEX_BIN
    assert all(isinstance(value, Path) for value in (PVC_READMAP_BIN, PANGENIE_BIN, PANGENIE_INDEX_BIN))
