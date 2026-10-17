# Thin distributed-training helpers used by the CLI and the training stages.
# world() reports rank, world size and local rank from torch.distributed or the torchrun
# environment variables, and is_main() tells whether this process should write shared files.
# barrier() and gather_objects() are safe no-ops in a single process.
# relaunch_with_torchrun re-executes the CLI under torch.distributed.run when --gpus asks for
# more than one process, and require_single_process guards stages that must not run under DDP.
# No module here initialises a process group; the HF Trainer does that itself.

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

import torch.distributed as tdist


@dataclass(frozen=True)
class World:
    # Process placement: global rank, world size and local (per-node) rank.
    rank: int
    size: int
    local_rank: int


def _initialized() -> bool:
    # Output: whether a torch.distributed process group is available and initialised.
    return tdist.is_available() and tdist.is_initialized()


def world() -> World:
    # Output: the current World, from the initialised process group if any, otherwise from the
    # RANK/WORLD_SIZE/LOCAL_RANK environment set by torchrun (defaults 0/1/0).
    local = int(os.environ.get("LOCAL_RANK", 0))
    if _initialized():
        return World(tdist.get_rank(), tdist.get_world_size(), local)
    return World(int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), local)


def is_main() -> bool:
    # Output: True on global rank 0 (the process that writes shared artifacts and prints).
    return world().rank == 0


def under_torchrun() -> bool:
    # Output: whether this process was started by torchrun (LOCAL_RANK and WORLD_SIZE set).
    return "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ


def barrier() -> None:
    # Blocks until every rank reaches this point; no-op without an initialised process group.
    if _initialized():
        tdist.barrier()


def gather_objects(obj) -> list:
    # Input: any picklable object. Output: the list of every rank's object in rank order
    # ([obj] in a single process).
    if not _initialized():
        return [obj]
    out = [None] * tdist.get_world_size()
    tdist.all_gather_object(out, obj)
    return out


def torchrun_command(argv: list, gpus: int) -> list:
    # Inputs: CLI arguments (without the program name) and process count. Output: the command
    # line that runs `python -m ByteResearch.run argv...` under torch.distributed.run.
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={int(gpus)}", "-m", "ByteResearch.run", *argv]


def relaunch_with_torchrun(argv: list, gpus: int):
    # Inputs: CLI arguments and the requested GPU count. Output: None when no relaunch is needed
    # (gpus <= 1 or already under torchrun); otherwise runs the torchrun subprocess and returns its
    # exit code.
    if int(gpus) <= 1 or under_torchrun():
        return None
    return subprocess.run(torchrun_command(argv, gpus), check=False).returncode


def require_single_process(stage: str) -> None:
    # Input: stage name. Raises RuntimeError when called with world size > 1 (for stages that
    # write unsharded results and must not run under DDP).
    if world().size > 1:
        raise RuntimeError(f"stage {stage!r} must run in a single process (world size {world().size})")
