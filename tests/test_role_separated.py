from types import SimpleNamespace

import numpy as np
import pytest

from pvc.genotype.private.role_separated import (
    PartyPrivateInput, PublicProtocolInput, reconstruct_arithmetic,
    redact_read_evidence, run_role_separated_party,
    split_arithmetic_secret, split_binary_secret,
)
from pvc.genotype.private.runtime import spawn_multiparty_rank_args


def _echo_rank_payload(value):
    import crypten
    from pvc.genotype.private.runtime import require_three_party_runtime
    require_three_party_runtime()
    return crypten.communicator.get().get_rank(), value


def _winner_shares(rows):
    import torch
    from crypten.mpc import MPCTensor
    from pvc.genotype.private.runtime import first_winner_one_hot
    return [first_winner_one_hot(MPCTensor.from_shares(torch.tensor(row), precision=0)).share.tolist() for row in rows]


def _heavy_matrix(variant, coverage, counts, table, size, scale):
    import torch
    from pvc.genotype.private.role_separated import _heavy_emission
    return _heavy_emission(torch, variant, coverage, counts, table, size, scale).share.tolist()


def _fail(_):
    raise ValueError("intentional worker failure")


def test_arithmetic_and_binary_client_sharing_round_trip():
    secret = np.array([[0, 1, -1], [2**62, -(2**62), 17]], dtype=np.int64)
    arithmetic = split_arithmetic_secret(secret)
    assert np.array_equal(reconstruct_arithmetic(arithmetic), secret)
    assert not np.array_equal(arithmetic[0], split_arithmetic_secret(secret)[0])
    bits = np.array([[0, 1, 1], [1, 0, 0]], dtype=np.int64)
    binary = split_binary_secret(bits)
    assert all(np.isin(share, [0, 1]).all() for share in binary)
    assert np.array_equal(binary[0] ^ binary[1] ^ binary[2], bits)
    with pytest.raises(ValueError):
        split_binary_secret(np.array([2]))


def test_redaction_removes_read_evidence_without_mutating_client_copy():
    private_variant = SimpleNamespace(local_coverage=37, kmer_to_count=[0, 4, 19],
                                      alleles={0: object(), 1: object()}, path_to_allele=[0, 1])
    private_map = SimpleNamespace(unique_kmers={"chr1": [private_variant]}, runtimes=[37])
    public_map = redact_read_evidence(private_map)
    assert public_map.unique_kmers["chr1"][0].local_coverage == 0
    assert public_map.unique_kmers["chr1"][0].kmer_to_count == [0, 0, 0]
    assert public_map.runtimes == []
    assert private_variant.local_coverage == 37
    assert private_variant.kmer_to_count == [0, 4, 19]


def test_runtime_rejects_uninitialized_execution():
    from pvc.genotype.private.runtime import require_three_party_runtime
    with pytest.raises(RuntimeError, match="three initialized"):
        require_three_party_runtime()


def test_rank_specific_launcher_handles_large_payloads():
    payloads = [(f"only-{rank}-" + "x" * 100_000,) for rank in range(3)]
    assert spawn_multiparty_rank_args(_echo_rank_payload, payloads, timeout=60) == [
        (rank, payloads[rank][0]) for rank in range(3)
    ]


@pytest.mark.parametrize("mode", ["light", "medium"])
def test_role_separated_score_and_winner_reconstruction(mode):
    emission = np.array([[11, 2], [2, 29]], dtype=np.int64)
    shares = split_arithmetic_secret(emission)
    public = PublicProtocolInput(
        unique_kmers_map=SimpleNamespace(unique_kmers={"chr1": [SimpleNamespace()]}),
        blocks_by_chrom={"chr1": [((10, 10), [0], [((0,), (0,)), ((1,), (1,))])]},
        probability_table=None, scale=1_000_000,
    )
    outputs = spawn_multiparty_rank_args(run_role_separated_party, [
        (public, PartyPrivateInput(mode=mode, emissions={"chr1": [shares[rank]]}))
        for rank in range(3)
    ], timeout=90)
    result = reconstruct_arithmetic([outputs[rank][0] for rank in range(3)])
    assert result.tolist() == ([11, 29] if mode == "light" else [0, 1])


def test_integer_argmax_ties_negative_and_wide_vectors():
    rng = np.random.default_rng(20260909)
    rows = [np.array([7]), np.array([-9, -2, -2, -7]),
            np.array([0, 0, 0]), np.array([-2**50, -2**50 + 1, -2**50]),
            rng.integers(-100_000_000, 0, size=513, dtype=np.int64)]
    shares = [split_arithmetic_secret(row) for row in rows]
    outputs = spawn_multiparty_rank_args(_winner_shares, [
        ([item[rank] for item in shares],) for rank in range(3)
    ], timeout=120)
    for i, row in enumerate(rows):
        winner = reconstruct_arithmetic([outputs[rank][i] for rank in range(3)])
        expected = np.zeros(len(row), dtype=np.int64)
        expected[int(np.argmax(row))] = 1
        np.testing.assert_array_equal(winner, expected)


@pytest.mark.parametrize("n_kmers", [0, 3])
def test_heavy_emissions_match_plaintext_with_undefined_allele(n_kmers):
    from pvc.pangenome.unique_kmers import UniqueKmers
    from pvc.pangenome.allele_info import AlleleInfo
    from pvc.pangenome.kmer_path import KmerPath
    from pvc.pangenome.probability_table import ProbabilityTable
    from pvc.genotype.genotyping import compute_emission_matrix
    variant = UniqueKmers(10, 2, n_kmers, [0, 1, 3][:n_kmers],
                          {0: AlleleInfo(KmerPath(0, 0b101), False),
                           1: AlleleInfo(KmerPath(0, 0b010), False),
                           2: AlleleInfo(KmerPath(0, 0), True)}, [0, 1, 2])
    table = ProbabilityTable(0, 4, 4)
    cov = np.array([0, 0, 1, 0], dtype=np.int64)
    counts = np.eye(4, dtype=np.int64)[variant.kmer_to_count]
    cov_shares, count_shares = split_arithmetic_secret(cov), split_binary_secret(counts)
    public = redact_read_evidence(SimpleNamespace(unique_kmers={"chr1": [variant]}))
    outputs = spawn_multiparty_rank_args(_heavy_matrix, [
        (public.unique_kmers['chr1'][0], cov_shares[r], count_shares[r], table, 3, 1_000_000)
        for r in range(3)
    ], timeout=90)
    actual = reconstruct_arithmetic(outputs).reshape(3, 3) / 1_000_000
    probs = [table.get_probability(2, c) for c in variant.kmer_to_count]
    expected = compute_emission_matrix(variant, probs, 3)
    np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=0)


def test_launcher_propagates_worker_failure_and_reaps_children():
    import multiprocessing
    before = {p.pid for p in multiprocessing.active_children()}
    with pytest.raises(RuntimeError, match="worker .* (failed|exited)"):
        spawn_multiparty_rank_args(_fail, [(None,)] * 3, timeout=60)
    assert {p.pid for p in multiprocessing.active_children()} == before
