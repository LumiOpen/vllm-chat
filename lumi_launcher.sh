#!/bin/bash
# LUMI AI Factory Container Launcher for SLURM jobs
#
# Usage:
#   source lumi_launcher.sh
#   run_sing_bash 'run_python -m vllm.entrypoints.openai.api_server ...'
#
# For srun workers:
#   srun -l bash -c "run_sing_bash 'your_command_here'"
#
# The launcher handles:
#   - LUMI AI Factory container setup (Ubuntu 24.04, ROCm 6.4, vLLM)
#   - Container bind mounts (LUMI filesystems, CXI, Cray)
#   - Per-rank Triton/XDG cache isolation
#   - ROCm/vLLM environment variables for MI250X (gfx90a)
#   - SLURM variable translation for container
#   - NCCL network configuration for Slingshot interconnect

###############################################################################
# Configuration Variables (override before sourcing)
###############################################################################

: "${LAUNCHER_IMG:=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20251209_134408/lumi-multitorch-full-u24r64f21m43t29-20251209_134408.sif}"
: "${LAUNCHER_PYTHON_VERSION:=3.12}"
: "${LAUNCHER_STORAGE_PRJ:=/scratch/project_462000963}"
: "${LAUNCHER_HF_CACHE:=${LAUNCHER_STORAGE_PRJ}/cache}"

export LAUNCHER_IMG
export LAUNCHER_PYTHON_VERSION
export LAUNCHER_STORAGE_PRJ
export LAUNCHER_HF_CACHE

# Will be set during setup and exported
export IMG=""

# pip --target directory for extra packages (ephemeral, inside container /tmp)
export LAUNCHER_PIP_TARGET="/tmp/pip-packages"

###############################################################################
# get_binds() - Returns bind mount arguments (one per line)
# Exported so it's available in srun workers
###############################################################################
get_binds() {
    local binds=(
        -B "${PWD:-$(pwd)}:/workspace"
        -B "${LAUNCHER_STORAGE_PRJ}:/project"
        -B /pfs,/scratch,/projappl,/flash,/appl
        -B /var/spool/slurmd
        -B /opt/cray/
        -B /usr/lib64/libcxi.so.1
        -B /usr/share/libdrm:/usr/share/libdrm
    )
    printf '%s\n' "${binds[@]}"
}
export -f get_binds

###############################################################################
# translate_slurm_vars() - Translate SLURM_* to SINGULARITYENV_SLURM_*
# Must be called before singularity exec so SLURM vars pass through --cleanenv
###############################################################################
translate_slurm_vars() {
    local var
    for var in SLURM_PROCID SLURM_LOCALID SLURM_STEP_ID SLURM_STEP_TASK_ID \
               SLURM_JOB_ID SLURM_NODEID SLURM_NTASKS SLURM_NNODES; do
        if [ -n "${!var:-}" ]; then
            export "SINGULARITYENV_${var}=${!var}"
        fi
    done
}
export -f translate_slurm_vars

###############################################################################
# setup_singularity_environment() - Set SINGULARITYENV_* variables
# These pass through --cleanenv and become regular env vars inside container
# Note: APPTAINERENV_* does not work with --cleanenv on LUMI
###############################################################################
setup_singularity_environment() {
    # HuggingFace
    export SINGULARITYENV_HF_HOME="/project/cache"
    export SINGULARITYENV_HUGGINGFACE_HUB_CACHE="/project/cache/hub"
    export SINGULARITYENV_HF_HUB_DISABLE_XET=1
    if [ -n "${HF_TOKEN:-}" ]; then
        { set +x; } 2>/dev/null
        export SINGULARITYENV_HF_TOKEN="$HF_TOKEN"
        set -x
    fi

    # vLLM settings
    export SINGULARITYENV_VLLM_USE_V1="${VLLM_USE_V1:-1}"
    export SINGULARITYENV_VLLM_TARGET_DEVICE="rocm"
    export SINGULARITYENV_VLLM_WORKER_MULTIPROC_METHOD="spawn"

    # ROCm architecture (MI250X)
    export SINGULARITYENV_PYTORCH_ROCM_ARCH="gfx90a"

    # NCCL for Slingshot interconnect
    export SINGULARITYENV_NCCL_SOCKET_IFNAME="hsn0,hsn1,hsn2,hsn3"
    export SINGULARITYENV_NCCL_NET_GDR_LEVEL="3"

    # Dispatcher server (if set)
    [ -n "${DISPATCHER_SERVER:-}" ] && export SINGULARITYENV_DISPATCHER_SERVER="$DISPATCHER_SERVER"
    [ -n "${DISPATCHER_PORT:-}" ] && export SINGULARITYENV_DISPATCHER_PORT="$DISPATCHER_PORT"

    # Pass through launcher config
    export SINGULARITYENV_LAUNCHER_STORAGE_PRJ="$LAUNCHER_STORAGE_PRJ"
    export SINGULARITYENV_LAUNCHER_PYTHON_VERSION="$LAUNCHER_PYTHON_VERSION"
    export SINGULARITYENV_USER="$USER"
}

###############################################################################
# setup_launcher_environment() - Main setup (call once from launcher script)
###############################################################################
setup_launcher_environment() {
    IMG="$LAUNCHER_IMG"
    export IMG

    echo "[launcher] Using container: $IMG"
    echo "[launcher] Storage project: $LAUNCHER_STORAGE_PRJ"

    # Create directories
    mkdir -p logs
    mkdir -p "${LAUNCHER_HF_CACHE}/hub"
    mkdir -p "${LAUNCHER_HF_CACHE}/hub/.locks"

    # Set up SINGULARITYENV_* variables
    setup_singularity_environment

    echo "[launcher] Setup complete"
}

###############################################################################
# run_sing_bash "command" - Run bash commands inside container
# This function is exported and works inside srun workers
###############################################################################
run_sing_bash() {
    [ -n "${IMG:-}" ] || {
        echo "[launcher] ERROR: run_sing_bash called before setup_launcher_environment" >&2
        return 1
    }

    if [ $# -eq 0 ]; then
        echo "[launcher] ERROR: no command provided" >&2
        return 1
    fi

    # Get bind mounts fresh
    local binds_array
    mapfile -t binds_array < <(get_binds)

    # Pass SLURM variables explicitly via --env
    local slurm_env_args=()
    local var
    for var in SLURM_PROCID SLURM_LOCALID SLURM_STEP_ID SLURM_STEP_TASK_ID \
               SLURM_JOB_ID SLURM_NODEID SLURM_NTASKS SLURM_NNODES; do
        if [ -n "${!var:-}" ]; then
            slurm_env_args+=(--env "${var}=${!var}")
        fi
    done

    # Build inline environment setup that runs inside the container
    # This handles per-rank cache isolation and defines run_python helper
    local env_setup="
    # Group-writable files for shared project cache (locks, downloaded weights)
    umask 002

    export HOME=/tmp

    # HuggingFace cache (explicit - ensure vLLM downloads to project scratch)
    export HF_HOME=/project/cache
    export HUGGINGFACE_HUB_CACHE=/project/cache/hub

    # Activate container venv + set up pip --target dir for extra packages
    source /opt/venv/bin/activate
    export PIP_TARGET=\"/tmp/pip-packages\"
    mkdir -p \"\$PIP_TARGET\"
    export PYTHONPATH=\"\$PIP_TARGET:\${PYTHONPATH:-}\"
    export PATH=\"\$PIP_TARGET/bin:\$PATH\"

    # Per-rank Triton cache isolation (avoids multi-rank races on shared filesystems)
    export TRITON_CACHE_DIR=\"/tmp/triton_cache/\${SLURM_JOB_ID:-nojob}/\${SLURM_PROCID:-\${SLURM_LOCALID:-0}}\"
    export XDG_CACHE_HOME=\"/tmp/xdg_cache/\${SLURM_JOB_ID:-nojob}/\${SLURM_PROCID:-\${SLURM_LOCALID:-0}}\"
    mkdir -p \"\$TRITON_CACHE_DIR\" \"\$XDG_CACHE_HOME\" 2>/dev/null || true

    # Torch extensions in shared memory
    export TORCH_EXTENSIONS_DIR=\"\${TORCH_EXTENSIONS_DIR:-/dev/shm/torch_ext}\"
    mkdir -p \"\$TORCH_EXTENSIONS_DIR\" 2>/dev/null || true

    # Remove stale HF cache lock files (multi-user fix)
    find /project/cache/hub/.locks -name \"*.lock\" -delete 2>/dev/null || true

    # Define run_python helper
    run_python() {
        /opt/venv/bin/python \"\$@\"
    }
"

    local full_command="${env_setup}
${*}"

    # Execute with --cleanenv (only APPTAINERENV_* vars pass through)
    singularity exec --rocm --cleanenv "${binds_array[@]}" "${slurm_env_args[@]}" "$IMG" \
        bash --noprofile --norc -c "$full_command"
}
export -f run_sing_bash

###############################################################################
# run_sing_python [args] - Run Python directly in container
###############################################################################
run_sing_python() {
    [ -n "${IMG:-}" ] || {
        echo "[launcher] ERROR: run_sing_python called before setup_launcher_environment" >&2
        return 1
    }

    local binds_array
    mapfile -t binds_array < <(get_binds)

    local full_cmd="
export HOME=/tmp
source /opt/venv/bin/activate
mkdir -p \"$LAUNCHER_PIP_TARGET\"
export PYTHONPATH=\"$LAUNCHER_PIP_TARGET:\${PYTHONPATH:-}\"
export PATH=\"$LAUNCHER_PIP_TARGET/bin:\$PATH\"
/opt/venv/bin/python \"\$@\"
"

    singularity exec --rocm --cleanenv "${binds_array[@]}" "$IMG" \
        bash --noprofile --norc -c "$full_cmd" bash "$@"
}
export -f run_sing_python

###############################################################################
# run_sing_pip_install [args] - pip install inside container
# Uses explicit --target to avoid PIP_TARGET leaking to build isolation
###############################################################################
run_sing_pip_install() {
    run_sing_python -m pip install --target "$LAUNCHER_PIP_TARGET" "$@"
}
export -f run_sing_pip_install

###############################################################################
# Cleanup trap
###############################################################################
_cleanup() {
    kill "${srv_pid:-0}" 2>/dev/null || true
}
trap _cleanup EXIT

###############################################################################
# Auto-setup when sourced
###############################################################################
setup_launcher_environment
