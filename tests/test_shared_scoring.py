from types import SimpleNamespace
import numpy as np
from pvc.genotype.private.runtime import spawn_multiparty_rank_args


def _lookup_paths(variant, table):
    import os
    import torch
    import crypten
    from pvc.genotype.private.shared import (
        _compute_block_emissions_max_batched, _private_probability_lookup_batch_fixed,
        _private_probability_lookup_fixed, _private_argmax_index_vector_batch,
    )
    results = {}
    for method in ['binary-mixed-separate', 'arithmetic-onehot-coverage-first', 'arithmetic-coverage-binary-count']:
        os.environ['PVC_HEAVY_LOOKUP_METHOD'] = method
        emissions = _compute_block_emissions_max_batched(crypten, torch, [variant], [0], [3], table, 1_000_000)
        results[method] = emissions[0].get_plain_text().tolist()
    results['batch_lookup'] = _private_probability_lookup_batch_fixed(
        crypten, torch, table, 2, [0, 1, 3], 1_000_000).get_plain_text().tolist()
    results['scalar_lookup'] = _private_probability_lookup_fixed(
        crypten, torch, table, 2, 1, 1_000_000).get_plain_text().tolist()
    vectors = [crypten.cryptensor(torch.tensor(row), precision=0, src=0)
               for row in [[4, 4, -2], [-8], [-9, -1, -1, -8, -3]]]
    results['winners'] = _private_argmax_index_vector_batch(crypten, torch, vectors)
    return results


def test_alternate_lookup_paths_and_batched_argmax():
    from pvc.pangenome.unique_kmers import UniqueKmers
    from pvc.pangenome.allele_info import AlleleInfo
    from pvc.pangenome.kmer_path import KmerPath
    from pvc.pangenome.probability_table import ProbabilityTable
    from pvc.genotype.genotyping import compute_emission_matrix
    variant = UniqueKmers(10, 2, 3, [0, 1, 3],
        {0: AlleleInfo(KmerPath(0, 0b101), False),
         1: AlleleInfo(KmerPath(0, 0b010), False),
         2: AlleleInfo(KmerPath(0, 0), True)}, [0, 1, 2])
    table = ProbabilityTable(0, 4, 4)
    probabilities = [table.get_probability(2, c) for c in variant.kmer_to_count]
    expected = compute_emission_matrix(variant, probabilities, 3)
    outputs = spawn_multiparty_rank_args(_lookup_paths, [(variant, table)] * 3, timeout=120)
    for result in outputs:
        for method in ['binary-mixed-separate', 'arithmetic-onehot-coverage-first', 'arithmetic-coverage-binary-count']:
            np.testing.assert_allclose(np.array(result[method]) / 1_000_000, expected, atol=5e-6, rtol=0)
        np.testing.assert_array_equal(result['batch_lookup'], np.rint(np.array(probabilities) * 1_000_000).astype(np.int64))
        np.testing.assert_array_equal(result['scalar_lookup'], np.rint(probabilities[1] * 1_000_000).astype(np.int64))
    assert outputs[0]['winners'] == [0, 0, 1]
