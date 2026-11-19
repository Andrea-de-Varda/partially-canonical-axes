# Copy to pipeline/env_setup.sh and edit for your cluster. Sourced by
# pipeline_stage.sh on the compute node before any Python runs.
#
# Must activate an environment built from environment.yml. May also set:
#   HF_HOME         Hugging Face cache (default: ~/.cache/huggingface). Use a
#                   dedicated directory: after each model finishes, the pipeline
#                   deletes that model's weights from $HF_HOME/hub (unless
#                   run_all_analyses.sh --no-cleanup).
#   HF_OFFLOAD_DIR  offload dir for large models (default: $HF_HOME/offload)
#   PROJECT_DIR     repo root (default: the directory you submitted from)
# Partition and account go on the sbatch command line, e.g.
#   ./run_all_analyses.sh ... --sbatch-args "-p gpu -A mylab"
# (they also reach the CPU-only audit job). The GPU request (and its type) is in
# the #SBATCH lines at the top of pipeline_stage.sh.

# module load anaconda3            # if your cluster uses environment modules
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate basis-alignment
