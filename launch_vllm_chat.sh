#!/bin/bash
#SBATCH --job-name=vllm_chat
#SBATCH --account=project_462000963
#SBATCH --partition=dev-g
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euxo pipefail

### config — edit these for your model
MODEL="${1:-LumiOpen/Llama-Poro-2-8B-Math-Reasoning-SFT-rc1}"
GPUS="${2:-1}"
### end config

echo "======================================="
echo "vLLM Chat Interface"
echo "======================================="
echo "Started: $(date)"
echo "Model:   $MODEL"
echo "GPUs:    $GPUS"
echo ""

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$SCRIPT_DIR/lumi_launcher.sh"

echo "Installing gradio (container)..."
run_sing_pip_install "gradio>=4" brotli "huggingface-hub<1.0" "numpy<2.3"
echo "Gradio installed."

srun -l bash -c '
    source '"$SCRIPT_DIR"'/lumi_launcher.sh

    run_sing_bash '\''
        set -euxo pipefail

        # Redirect HF_HOME to a job-local dir so Gradio downloads frpc
        # with correct permissions. Model weights still use the shared
        # cache via HUGGINGFACE_HUB_CACHE which remains unchanged.
        export HF_HOME=/tmp/hf_local
        mkdir -p "$HF_HOME"

        LOCALID=${SLURM_LOCALID:-0}

        GPU_IDS=""
        for (( i=0; i<'"$GPUS"'; i++ )); do
            if [ -z "$GPU_IDS" ]; then
                GPU_IDS="$i"
            else
                GPU_IDS="${GPU_IDS},$i"
            fi
        done
        export ROCR_VISIBLE_DEVICES="$GPU_IDS"
        export HIP_VISIBLE_DEVICES="$GPU_IDS"

        export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
        export MASTER_PORT=$(( 7000 + LOCALID ))

        echo "Launching vLLM chat on GPUs $GPU_IDS"
        run_python -u /workspace/vllm_chat.py '"$GPUS"':'"$MODEL"'
    '\''
'

echo ""
echo "Completed: $(date)"
