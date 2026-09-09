"""PVC orchestration and integer operations using unmodified upstream CrypTen.

Three computing parties use upstream Beaver protocols and a separate TTP
preprocessing server. Only the client owns all input/output shares. This local
spawn harness is for research, not a multi-host transport or trust boundary.
"""
from multiprocessing.connection import wait
from pathlib import Path
import multiprocessing as mp
import os
import tempfile
import time
import traceback


def require_three_party_runtime():
    import crypten
    from crypten.config import cfg

    if not crypten.is_initialized() or crypten.communicator.get().get_world_size() != 3:
        raise RuntimeError("PVC private computation requires three initialized CrypTen parties")
    if cfg.mpc.protocol != "beaver" or cfg.mpc.provider != "TTP":
        raise RuntimeError("PVC requires upstream CrypTen Beaver protocols with a separate TTP provider")
    if cfg.encoder.precision_bits != 0:
        raise RuntimeError("PVC integer scores require encoder.precision_bits=0")


def _worker(rank, rendezvous, function, args, connection, threads):
    try:
        os.environ.update(RANK=str(rank), WORLD_SIZE="3", RENDEZVOUS=rendezvous,
                          DISTRIBUTED_BACKEND="gloo")
        import torch
        import crypten
        from crypten.config import cfg

        torch.set_num_threads(threads)
        cfg.mpc.provider = "TTP"
        cfg.mpc.protocol = "beaver"
        cfg.mpc.active_security = False
        cfg.encoder.precision_bits = 0
        cfg.communicator.verbose = True
        crypten.init()
        if rank == 3:
            crypten.mpc.provider.TTPServer()
            result = None
        else:
            result = function(*args)
        crypten.uninit()
        connection.send((True, result))
    except BaseException:
        connection.send((False, traceback.format_exc()))
        raise
    finally:
        connection.close()


def spawn_multiparty_rank_args(function, rank_args, *, timeout=None):
    """Run a callable with exactly one payload and return channel per party.

    Uses spawn so children cannot inherit the client's other plaintext inputs.
    Errors and timeouts terminate/reap all workers, including the TTP server.
    Drain pipes before joining to support results larger than a pipe buffer.
    """
    if len(rank_args) != 3:
        raise ValueError("exactly three rank-specific argument tuples are required")
    timeout = float(timeout if timeout is not None else os.environ.get("PVC_MPC_TIMEOUT_SECONDS", "3600"))
    threads = int(os.environ.get("PVC_THREADS_PER_ROLE", "1"))
    if timeout <= 0 or threads <= 0:
        raise ValueError("timeout and threads per role must be positive")
    context = mp.get_context("spawn")
    processes, receivers = [], []
    results = {}
    with tempfile.TemporaryDirectory(prefix="pvc-mpc-") as temporary:
        rendezvous = Path(temporary, "rendezvous").as_uri()
        try:
            for rank in range(4):
                receive, send = context.Pipe(duplex=False)
                receivers.append(receive)
                process = context.Process(
                    target=_worker,
                    args=(rank, rendezvous, function if rank < 3 else None,
                          rank_args[rank] if rank < 3 else (), send, threads),
                    name=f"pvc-party-{rank}" if rank < 3 else "pvc-ttp",
                )
                process.start()
                processes.append(process)
                send.close()
            pending = {connection: rank for rank, connection in enumerate(receivers)}
            deadline = time.monotonic() + timeout
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"PVC MPC workers exceeded {timeout:g} seconds")
                for connection in wait(list(pending), timeout=min(remaining, 0.2)):
                    rank = pending.pop(connection)
                    try:
                        success, result = connection.recv()
                    except EOFError as exc:
                        raise RuntimeError(f"PVC worker {rank} exited without returning a result") from exc
                    if not success:
                        raise RuntimeError(f"PVC worker {rank} failed:\n{result}")
                    results[rank] = result
                for rank, process in enumerate(processes):
                    if process.exitcode not in (None, 0):
                        raise RuntimeError(f"PVC worker {rank} exited with code {process.exitcode}")
            for process in processes:
                process.join(timeout=max(0, deadline - time.monotonic()))
                if process.is_alive():
                    raise TimeoutError("PVC worker failed to shut down")
                if process.exitcode != 0:
                    raise RuntimeError(f"{process.name} exited with code {process.exitcode}")
            return [results[rank] for rank in range(3)]
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
            for connection in receivers:
                connection.close()


def from_arithmetic(primitive):
    """Wrap an existing upstream arithmetic share without re-sharing it."""
    from crypten.mpc import MPCTensor
    return MPCTensor.from_shares(primitive.share, precision=primitive.encoder._precision_bits)


def secure_maximum(values, dim=0):
    """Tournament maximum with no random tie selection or score rescaling."""
    import crypten
    if values.size(dim) == 0:
        raise ValueError("maximum requires a nonempty dimension")
    reduced = values
    while reduced.size(dim) > 1:
        size = reduced.size(dim)
        left, right, extra = reduced.split([size // 2, size // 2, size % 2], dim=dim)
        # Preserve PVC's sign-bit + binary/arithmetic no-truncation selection.
        # Only the A2B conversion comes from upstream; no arithmetic-bit
        # re-encoding or generic fixed-point where() is needed here.
        from pvc.genotype.private.mixed_protocols import mixed_mul_no_truncation
        difference = left - right
        binary_difference = difference.to(crypten.mpc.binary)
        choose_left = ((binary_difference._tensor >> 63) & 1) ^ 1
        maximum = mixed_mul_no_truncation(difference, choose_left) + right
        reduced = crypten.cat([maximum, extra], dim=dim)
    return reduced


def first_winner_one_hot(scores, dim=0):
    """Keep only the first maximum, including ties, using integer MPC operations."""
    from crypten.mpc import MPCTensor
    # Compare the raw encoded integers at precision 0. This is a metadata-only
    # view: no shares are rescaled or truncated. Upstream equality otherwise
    # depends on the global default precision during its B2A conversion.
    integers = MPCTensor.from_shares(scores.share, precision=0)
    winners = integers.eq(secure_maximum(integers, dim=dim))
    return winners * winners.cumsum(dim).eq(1)
