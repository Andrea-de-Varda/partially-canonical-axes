"""
Free disk for one model: delete its cached activations and its downloaded weights.

Used at the end of a per-model pipeline task (see pipeline_stage.sh, job type
MODEL). Activations are cheap to recompute and weights are re-downloadable, so
neither is worth keeping once every analysis for that model has finished.

SAFETY. Only ever call this when NOTHING else still needs the model:
  * The MM (model-to-model) reference EleutherAI/pythia-160m and its seed twin
    are read by every other model's task and by the RSA jobs, so they are on a
    hard keep-list and are never deleted here regardless of arguments.
  * Pythia training-checkpoint revisions used by CKPT/RSA are evicted by those
    jobs themselves (_checkpoint_helpers.evict_revision); this script only
    touches whole repos it is explicitly given.
  * Shared, tiny caches (brain_expt*.npy, rotation matrices) never match a model
    alias and are left alone.

Usage:
  python other/cleanup_model.py --model gpt2-large            # activations + weights
  python other/cleanup_model.py --model gpt2 --dry-run        # report only
  python other/cleanup_model.py --model gpt2 --activations-only
"""
import argparse
import os
import shutil
import sys

# Never delete these: MM reference + RSA partner, needed by every other task.
KEEP = {"EleutherAI/pythia-160m", "EleutherAI/pythia-160m-seed2"}


def _human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024


def _dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def clean_activations(model, cache_dir="cached_activations", dry_run=False):
    """Delete cached activation files whose name starts with the model's alias."""
    if model in KEEP:
        # e.g. pythia-160m's *_pereira_combined_* cache is read by every other
        # model's MM step, so its activations must outlive its own task too.
        print(f"  [KEEP] {model} is on the keep-list - activations retained", flush=True)
        return 0
    alias = model.replace("/", "_")
    if not os.path.isdir(cache_dir):
        return 0
    freed = 0
    for fname in sorted(os.listdir(cache_dir)):
        # alias-prefixed only: never matches brain_expt*.npy or other shared files
        if not fname.startswith(alias + "_"):
            continue
        path = os.path.join(cache_dir, fname)
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        print(f"  [{'would delete' if dry_run else 'deleted'}] {path}  ({_human(size)})", flush=True)
        if not dry_run:
            try:
                os.remove(path)
            except OSError as e:
                print(f"  [WARN] could not delete {path}: {e}", flush=True)
                continue
        freed += size
    return freed


def clean_weights(model, dry_run=False):
    """Delete the model's HF hub repo (all revisions)."""
    if model in KEEP:
        print(f"  [KEEP] {model} is on the keep-list (MM reference / RSA partner) - weights retained",
              flush=True)
        return 0
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    repo_dir = os.path.join(hub, "models--" + model.replace("/", "--"))
    if not os.path.isdir(repo_dir):
        print(f"  [skip] no weights cached at {repo_dir}", flush=True)
        return 0
    size = _dir_size(repo_dir)
    print(f"  [{'would delete' if dry_run else 'deleted'}] {repo_dir}  ({_human(size)})", flush=True)
    if dry_run:
        return size
    # Prefer the official API so hub bookkeeping stays consistent; fall back to rmtree.
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir(hub)
        hashes = [rev.commit_hash for repo in info.repos
                  if repo.repo_id == model for rev in repo.revisions]
        if hashes:
            info.delete_revisions(*hashes).execute()
            return size
    except Exception as e:
        print(f"  [INFO] hub API delete unavailable ({e}); using rmtree", flush=True)
    try:
        shutil.rmtree(repo_dir)
    except OSError as e:
        print(f"  [WARN] could not remove {repo_dir}: {e}", flush=True)
        return 0
    return size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id, e.g. meta-llama/Llama-3.2-3B")
    ap.add_argument("--cache-dir", default="cached_activations")
    ap.add_argument("--activations-only", action="store_true")
    ap.add_argument("--weights-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"[CLEANUP] {args.model}{' (dry run)' if args.dry_run else ''}", flush=True)
    freed = 0
    if not args.weights_only:
        freed += clean_activations(args.model, args.cache_dir, args.dry_run)
    if not args.activations_only:
        freed += clean_weights(args.model, args.dry_run)
    print(f"[CLEANUP] {args.model}: {'would free' if args.dry_run else 'freed'} {_human(freed)}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
