#!/bin/bash
# =============================================================================
# Basis-alignment: submit the whole analysis pipeline, in dependency order,
# never asking for more than MAX_GPUS GPUs at any moment.
#
#   ./run_all_analyses.sh                 # default: no pre/post-MLP streams
#   ./run_all_analyses.sh --test-mlp      # include pre/post-MLP (much slower)
#   ./run_all_analyses.sh --dry-run       # print the sbatch commands only
#
# This script issues sbatch calls and nothing else -- no python, no data access,
# no computation. All real work happens inside the submitted jobs.
#
# GPU BUDGET
#   Stages run strictly one after another (SLURM --dependency), and each stage
#   is a SINGLE job array throttled with %MAX_GPUS. Because "%N" caps
#   concurrency *within one array*, one-array-per-stage plus sequential stages
#   is what makes the cap global: at no instant are more than MAX_GPUS GPU
#   tasks running. (Submitting several arrays at once would give N per array.)
#   The final audit stage is CPU-only and requests no GPU.
#
# STAGES
#   1 all    22 jobs. 12 of them are one-model-does-everything tasks:
#              LANG -> Pereira expt2 -> Pereira expt3 -> MM -> [SEM] -> SPARSITY
#              -> JOINT -> delete that model's activations AND weights.
#            The other 10 are the Pythia seed-pair jobs (5 CKPT + 5 RSA), which
#            evict their own checkpoint revisions as they go.
#   2 audit  CPU-only: Pereira checkpoint recovery + completeness report.
#
# DISK
#   Grouping a model's whole pipeline into one task is what makes cleanup safe:
#   SPARSITY and JOINT read LANG's activation cache, so it can only be freed
#   once they are done - guaranteed here because they are the same task. Peak
#   disk is therefore about MAX_GPUS models' worth of weights + activations
#   rather than all twelve. EleutherAI/pythia-160m and its seed twin are on a
#   keep-list (other/cleanup_model.py): they are the model-to-model reference
#   and an RSA partner, so every other task reads them. Use --no-cleanup to
#   keep everything on disk.
#
# Everything downstream has checkpoint/resume, so re-running this script after
# a failure recomputes only what is missing.
# =============================================================================
set -euo pipefail

MAX_GPUS=6
TEST_MLP=0
DRY=0
WITH_SEM=1
NO_AUDIT=0
NO_CLEANUP=0
STEPS=""
ONLY=""
WALLTIME=""
MANIFEST_OVERRIDE=""

usage() {
    sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Options:
  --test-mlp        Include Pereira pre/post-activation (MLP) streams. These live
                    in d_mlp (3072-6400) rather than d_model (768-1600) and are the
                    dominant cost, which is why they are off by default.
  --max-gpus N      Global concurrent GPU cap (default 6).
  --no-sem          Skip SEM alignment (on by default; --with-sem is accepted as a no-op).
  --no-audit        Skip the final CPU-only audit stage.
  --no-cleanup      Keep activations and weights on disk (default: each model's
                    are deleted as soon as all of its analyses finish).
  --steps LIST      Comma-separated subset of a MODEL task's steps to run:
                    LANG,PEREIRA,MM,SEM,SPARSITY,JOINT (default: all). Use to
                    redo only part of the pipeline, e.g. --steps LANG,JOINT.
                    Cleanup still runs after the selected steps, so include
                    every step that needs a model's caches in the same call.
  --manifest FILE   Use FILE as the stage-1 manifest instead of pipeline/stage_all.txt,
                    e.g. pipeline/stage_pereira.txt (the 12 MODEL lines only) with
                    --steps PEREIRA for a brain-only rerun.
  --only STAGE      Submit a single stage: all | audit.
                    Its dependency on earlier stages is NOT applied - use only
                    when the earlier stages are already complete.
  --time T          Walltime for the main array (default 2-00:00:00, or
                    4-00:00:00 with --test-mlp).
  --sbatch-args S   Extra sbatch options for every job (GPU array and CPU audit),
                    e.g. partition/account: --sbatch-args "-p gpu -A mylab".
                    The GPU request itself is in pipeline_stage.sh's header.
                    Default: $BA_SBATCH_ARGS (an optional, unshipped
                    pipeline/submit_defaults.sh may set it).
  --dry-run         Print what would be submitted and exit.
  -h, --help        This message.
USAGE
}

# Site defaults (e.g. partition) for the sbatch calls: optional local file, not shipped.
[ -f pipeline/submit_defaults.sh ] && source pipeline/submit_defaults.sh
SBATCH_EXTRA="${BA_SBATCH_ARGS:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        --test-mlp)  TEST_MLP=1; shift ;;
        --max-gpus)  MAX_GPUS="${2:?--max-gpus needs a value}"; shift 2 ;;
        --with-sem)  WITH_SEM=1; shift ;;
        --no-sem)    WITH_SEM=0; shift ;;
        --no-audit)  NO_AUDIT=1; shift ;;
        --no-cleanup) NO_CLEANUP=1; shift ;;
        --steps)     STEPS="${2:?--steps needs a comma-separated list}"; shift 2 ;;
        --only)      ONLY="${2:?--only needs a stage name}"; shift 2 ;;
        --time)      WALLTIME="${2:?--time needs a value}"; shift 2 ;;
        --manifest)  MANIFEST_OVERRIDE="${2:?--manifest needs a file}"; shift 2 ;;
        --sbatch-args) SBATCH_EXTRA="${2?--sbatch-args needs a value}"; shift 2 ;;
        --dry-run)   DRY=1; shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "Unknown option: $1"; echo; usage; exit 1 ;;
    esac
done

case "$MAX_GPUS" in ''|*[!0-9]*) echo "ERROR: --max-gpus must be an integer"; exit 1 ;; esac
[ "$MAX_GPUS" -ge 1 ] || { echo "ERROR: --max-gpus must be >= 1"; exit 1; }
if [ -n "$ONLY" ]; then
    case "$ONLY" in all|audit) ;; *) echo "ERROR: --only must be all|audit"; exit 1 ;; esac
fi

cd "$(dirname "$0")"
mkdir -p logs

if [ "$TEST_MLP" -eq 1 ]; then
    STAGE1=pipeline/stage_all_mlp.txt
    : "${WALLTIME:=4-00:00:00}"
else
    STAGE1=pipeline/stage_all.txt
    : "${WALLTIME:=2-00:00:00}"
fi

[ -n "$MANIFEST_OVERRIDE" ] && STAGE1="$MANIFEST_OVERRIDE"

njobs() { grep -v '^[[:space:]]*#' "$1" | grep -vc '^[[:space:]]*$'; }

for f in "$STAGE1" pipeline_stage.sh other/cleanup_model.py; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done

N1=$(njobs "$STAGE1")

echo "=============================================================="
echo " Basis-alignment pipeline"
echo "   pre/post-MLP streams : $([ $TEST_MLP -eq 1 ] && echo 'INCLUDED (--test-mlp)' || echo 'skipped (default)')"
echo "   max concurrent GPUs  : $MAX_GPUS"
echo "   main-array walltime  : $WALLTIME"
echo "   SEM alignment        : $([ $WITH_SEM -eq 1 ] && echo 'included (default)' || echo 'skipped (--no-sem)')"
echo "   matching metric      : MATCH_ABS=$(grep -oE '^MATCH_ABS *= *[A-Za-z]+' metric_config.py | head -1 | tr -d ' ' | cut -d= -f2)  (metric_config.py)"
echo "   per-model cleanup    : $([ $NO_CLEANUP -eq 1 ] && echo 'DISABLED (--no-cleanup)' || echo 'on (activations + weights freed after each model)')"
[ -n "$STEPS" ] && echo "   steps per model      : $STEPS (subset)"
echo "   stage 1 (all)        : $N1 jobs   [$STAGE1]"
[ "$NO_AUDIT" -eq 0 ] && echo "   stage 2 (audit)      : 1 job, CPU-only"
[ -n "$ONLY" ] && echo "   ONLY stage           : $ONLY (dependencies NOT applied)"
echo "=============================================================="

submit() {  # submit <name> <extra sbatch args...>
    local name="$1"; shift
    if [ "$DRY" -eq 1 ]; then
        echo "[dry-run] $name: sbatch $SBATCH_EXTRA $*" >&2
        echo "DRYRUN_${name}"
    else
        local jid
        # shellcheck disable=SC2086
        jid=$(sbatch --parsable $SBATCH_EXTRA "$@")
        echo "  submitted $name -> job $jid" >&2
        echo "$jid"
    fi
}

want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

JID1=""
DEP=""   # dependency carried to the audit stage

# Exported (not passed via --export=VAR=...) because the value contains commas,
# which sbatch --export would parse as variable separators. --export=ALL carries
# the submitting environment, so this reaches the job intact.
[ -n "$STEPS" ] && export PIPELINE_STEPS="$STEPS"

if want all; then
    JID1=$(submit stage1_all \
        --job-name=ba_all --array=0-$((N1-1))%"$MAX_GPUS" --time="$WALLTIME" \
        --export=ALL,MANIFEST="$STAGE1",WITH_SEM="$WITH_SEM",NO_CLEANUP="$NO_CLEANUP" \
        pipeline_stage.sh)
    [ -z "$ONLY" ] && DEP="--dependency=afterany:$JID1"
fi

if [ "$NO_AUDIT" -eq 0 ] && want audit; then
    submit stage2_audit --job-name=ba_audit ${DEP:+$DEP} other/pereira_maintenance.sh >/dev/null
fi

echo
if [ "$DRY" -eq 1 ]; then
    echo "Dry run - nothing was submitted."
else
    echo "Queued. The array is throttled to $MAX_GPUS concurrent tasks and the audit"
    echo "stage requests no GPU, so at most $MAX_GPUS GPUs are in use at any moment."
    echo
    echo "  squeue -u \$USER                     # watch progress"
    echo "  tail -f logs/ba_all_<jobid>_0.out    # a running task ([HEARTBEAT] every 10 min)"
    echo "  grep CLEANUP logs/ba_all_*.out       # disk freed per model"
    echo "  scancel --name=ba_all --name=ba_audit   # stop everything"
fi
