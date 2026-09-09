"""Run all four core CLI modes through real cereal parsing and VCF output."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pvc.pangenome.DNA_sequence import DnaSequence


def _dna(text):
    sequence = DnaSequence.from_string(text)
    return {'value0': sequence.sequence, 'value1': sequence.even_length, 'value2': sequence.is_undefined}


@pytest.fixture
def tiny_inputs(tmp_path):
    variants, maps = [], []
    for index, position in enumerate([10, 20]):
        variants.append({'ptr_wrapper': {'data': {
            'value0': _dna(''), 'value1': _dna(''), 'value2': [],
            'value3': 'chr1', 'value4': position - 1,
            'value5': [[_dna('A'), _dna('C')]], 'value6': [[0], [1]],
            'value7': [[]], 'value8': [0, 1], 'value9': False,
        }}})
        maps.append({
            'polymorphic_id': (1 << 31) + 1 if index == 0 else 1,
            **({'polymorphic_name': 'BiallelicUniqueKmers'} if index == 0 else {}),
            'ptr_wrapper': {'data': {
                'value0': position, 'value1': 2, 'value2': 2,
                'value3': [0, 3],
                'value4': [
                    {'key': False, 'value': {'value0': {'value0': 0, 'value1': 1}, 'value1': False}},
                    {'key': True, 'value': {'value0': {'value0': 0, 'value1': 2}, 'value1': False}},
                ],
                'value5': [0, 1],
            }},
        })
    graph = tmp_path / 'graph.json'
    graph.write_text(json.dumps({'value0': {'value0': {'value0': []}, 'value1': 'chr1',
        'value2': 31, 'value3': True, 'value4': False, 'value5': variants, 'value6': [[], []]}}))
    cereal = tmp_path / 'read_map.json'
    cereal.write_text(json.dumps({'value0': {'value0': 31, 'value1': [{'key': 'chr1', 'value': maps}],
                                            'value2': [], 'value3': []}}))
    panel = tmp_path / 'panel.vcf'
    panel.write_text('##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tPANEL\n'
                     'chr1\t10\tv1\tA\tC\t.\tPASS\t.\tGT\t0|1\n'
                     'chr1\t20\tv2\tA\tC\t.\tPASS\t.\tGT\t0|1\n')
    blocks = tmp_path / 'panel.blocks.det'
    blocks.write_text('CHR BP1 BP2 KB NSNPS SNPS\nchr1 10 20 0.01 2 v1|v2\n')
    return cereal, graph, panel, blocks


def test_all_modes_write_matching_genotypes(tiny_inputs, tmp_path):
    cereal, graph, panel, blocks = tiny_inputs
    calls = {}
    for mode in ['plaintext', 'light', 'medium', 'heavy']:
        output = tmp_path / mode / 'calls.vcf'
        command = [sys.executable, '-m', 'pvc', '--mode', mode, '--cereal', str(cereal),
                   '--graph', str(graph), '--panel-vcf', str(panel), '--blocks-file', str(blocks),
                   '--mean-kmer-abundance', '2', '--sample-name', 'TINY', '--output', str(output)]
        result = subprocess.run(command, text=True, capture_output=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        text = output.read_text()
        assert '#CHROM' in text and '\tTINY\n' in text
        rows = [row.split('\t') for row in text.splitlines() if not row.startswith('#')]
        assert len(rows) == 2
        assert [int(row[1]) for row in rows] == [10, 20]
        calls[mode] = [row[-1].split(':')[0] for row in rows]
        assert calls[mode] == ['1/1', '1/1']
    assert all(value == calls['plaintext'] for value in calls.values())


@pytest.mark.parametrize('mode', ['light', 'medium', 'heavy'])
def test_manifest_api_uses_upstream_launcher_and_writes_completion(tiny_inputs, tmp_path, mode):
    from pvc.genotype.private.pvc_private_genotype import run_pvc_private_genotype
    cereal, graph, panel, blocks = tiny_inputs
    tool = 'pvc-' + mode
    run_dir = tmp_path / 'run'
    index_dir = run_dir / tool / 'index'
    index_dir.mkdir(parents=True)
    prefix = index_dir / 'index'
    graph_path = Path(str(prefix) + '_chr1_Graph.json')
    graph_path.write_bytes(graph.read_bytes())
    (index_dir / 'index_complete.json').write_text(json.dumps({
        'index_prefix': str(prefix), 'blocks_file': str(blocks),
        'artifacts': [str(cereal), str(graph_path)],
    }))
    manifest = {'sample': 'TINY', 'chromosome': 'chr1', 'panel_subset': str(panel),
                'read_unique_kmers_json': str(cereal), 'mean_kmer_abundance': 2}
    output_prefix = run_pvc_private_genotype(tool, manifest, run_dir)
    output = output_prefix.with_suffix('.vcf')
    calls = [row.split('\t')[-1].split(':')[0] for row in output.read_text().splitlines()
             if not row.startswith('#')]
    assert calls == ['1/1', '1/1']
    marker = json.loads((output.parent / 'genotype_complete.json').read_text())
    assert marker['index_tool'] == tool
    assert Path(marker['genotype_prefix']) == output_prefix
    assert Path(marker['output_vcf']) == output
    # Public on-disk IR must contain only redacted coverage.
    import csv
    with (output.parent / 'ir/bubbles.tsv').open() as handle:
        bubbles = list(csv.DictReader(handle, delimiter='\t'))
    assert all(float(row['local_coverage']) == 0 for row in bubbles)
