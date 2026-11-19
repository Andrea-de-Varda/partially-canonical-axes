"""
Atomic writes for activation-cache files.

A plain `np.save(path, obj)` leaves a truncated, still-loadable file if the
process is killed mid-write (SLURM walltime, OOM killer) or if two jobs write
the same cache concurrently -- several pipelines extract the same model's
activations from different array tasks. A cache in that state loads without
error and silently produces garbage. Writing to a unique temp file in the same directory and then
os.replace()-ing it makes the visible path flip atomically: readers see either
the old file or a complete new one, never a partial one.
"""
import os
import numpy as np


def atomic_save_npy(path, obj):
    """np.save(path, obj) that is atomic and safe under concurrent writers."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    # unique per process so concurrent writers never share a temp file
    tmp = os.path.join(d, ".%s.tmp%d" % (os.path.basename(path), os.getpid()))
    try:
        np.save(tmp, obj, allow_pickle=True)
        # np.save appends .npy when the name lacks it
        written = tmp if os.path.exists(tmp) else tmp + ".npy"
        os.replace(written, path)
    finally:
        for leftover in (tmp, tmp + ".npy"):
            if os.path.exists(leftover):
                try:
                    os.remove(leftover)
                except OSError:
                    pass
