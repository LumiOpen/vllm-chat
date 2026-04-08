# vLLM Chat Interface on LUMI

A Gradio-based chat UI powered by [vLLM](https://github.com/vllm-project/vllm) running on LUMI's AMD MI250X GPUs. Supports any HuggingFace model with a chat template, with single-model and side-by-side dual-model comparison modes in a single unified interface.

Uses the [LUMI AI Factory container](https://lumi-supercomputer.github.io/LUMI-AI-Guide/) (Ubuntu 24.04, ROCm 6.4, PyTorch, vLLM) via the shared `lumi_launcher.sh` environment.

## Quick Start

```bash
# Single model (default: LumiOpen/Llama-Poro-2-8B-Math-Reasoning-SFT-rc1, 1 GPU)
sbatch launch_vllm_chat.sh

# Custom model and GPU count
sbatch launch_vllm_chat.sh meta-llama/Llama-3.1-8B-Instruct 2

# Dual model comparison
sbatch launch_vllm_chat.sh meta-llama/Llama-3.1-8B-Instruct 1 Qwen/Qwen2.5-7B-Instruct 1
```

Once the job is running, a **public Gradio URL** (e.g. `https://xxxxx.gradio.live`) will appear in the log. Open it in any browser to start chatting.

## Deploying a Custom Model

### Step 1: Pick a model

Any HuggingFace model compatible with vLLM can be used. It should have a tokenizer with a `chat_template` for proper conversation formatting. Examples:

| Model | HuggingFace ID | GPUs |
|---|---|---|
| Llama 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` | 1-2 |
| Mistral 7B Instruct | `mistralai/Mistral-7B-Instruct-v0.3` | 1 |
| Qwen 2.5 72B Instruct | `Qwen/Qwen2.5-72B-Instruct` | 8 |
| Gemma 2 9B IT | `google/gemma-2-9b-it` | 1-2 |

**GPU guidelines:**

| Model size | Recommended GPUs |
|---|---|
| up to ~8B params | 1 |
| 8B-20B params | 2 |
| 20B-40B params | 4 |
| 40B+ params | 8 |

### Step 2: Submit the job

The launcher accepts positional arguments for single or dual model:

```bash
# Single model
sbatch launch_vllm_chat.sh <MODEL_ID> <GPU_COUNT>

# Dual model comparison
sbatch launch_vllm_chat.sh <MODEL1_ID> <GPU_COUNT1> <MODEL2_ID> <GPU_COUNT2>
```

Or edit the defaults in the `### config` section of `launch_vllm_chat.sh`:

```bash
MODEL="${1:-your-org/your-model}"
GPUS="${2:-2}"
```

### Step 3: (Optional) Gated models

For gated HuggingFace models (e.g. Llama), set your token before submitting:

```bash
export HF_TOKEN="hf_your_token_here"
sbatch launch_vllm_chat.sh meta-llama/Llama-3.1-8B-Instruct 2
```

The `lumi_launcher.sh` environment automatically passes `HF_TOKEN` into the container.

### Step 4: Connect to the UI

```bash
# Check job status
squeue -u $USER

# Tail the output log
tail -f logs/<JOB_ID>.out
```

Look for:

```
Running on public URL: https://xxxxxxxxxx.gradio.live
```

Open this URL in your browser. The link is valid for 72 hours or until the job ends.

## Project Structure

| File | Purpose |
|---|---|
| `launch_vllm_chat.sh` | SLURM job script — accepts 1 or 2 models, sources `lumi_launcher.sh`, installs Gradio, runs `vllm_chat.py` |
| `vllm_chat.py` | Unified Gradio chat app — single-model and dual-model comparison in one interface |
| `logs/` | SLURM stdout/stderr (created at runtime) |

### Launcher

`lumi_launcher.sh` (included in this directory) handles container setup, bind mounts, environment variables, and provides `run_sing_bash` / `run_python` helpers. Originally from the [dispatcher](../dispatcher/) project.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  LUMI compute node (Singularity/Apptainer)          │
│  LUMI AI Factory container (ROCm 6.4, vLLM)        │
│                                                     │
│  ┌─────────────┐       ┌──────────────────────────┐ │
│  │  Gradio UI  │──HTTP──│  vLLM Server(s)         │ │
│  │ (share=True)│       │  :8001 [+ :8002]         │ │
│  └──────┬──────┘       └────────┬─────────────────┘ │
│         │                       │                    │
│    Public URL             AMD MI250X GPUs            │
│  (gradio.live)          (tensor parallelism)         │
└─────────────────────────────────────────────────────┘
         │
    ┌────┴────┐
    │ Browser │
    └─────────┘
```

1. SLURM allocates a node with GPUs
2. `lumi_launcher.sh` sets up the container environment (bind mounts, caches, env vars)
3. Gradio is pip-installed into the container's `/tmp/pip-packages`
4. `vllm_chat.py` starts 1 or 2 vLLM OpenAI-compatible API servers as subprocesses
5. After health checks pass, Gradio launches with `share=True` creating a public tunnel
6. Messages are formatted with the model's chat template and streamed via `/v1/completions`

## UI Features

- **Single & dual mode** — chat with one model or compare two side by side
- **Mode switching** — toggle between single and dual when two models are deployed
- **Model selector** — in single mode with two models, choose which one to chat with
- **Generation parameters** — adjust temperature, top-p, top-k, and max tokens via the Settings accordion
- **Streaming responses** — tokens appear as they are generated
- **Regenerate** — re-run the last response with a new random seed
- **Delete last** — remove the most recent message
- **Edit last** — pop the last user message back into the input box for editing
- **Clear** — reset the conversation
- **Auto context trimming** — long conversations are truncated to fit the context window
- **Concurrent users** — up to 16 simultaneous sessions via Gradio queue

## Troubleshooting

### Job stays pending (PD)
The `dev-g` partition has limited slots. Check with `squeue -p dev-g`. Consider `small-g` for longer runs.

### vLLM fails to load the model
- **OOM**: increase `GPUS` or use a smaller model
- **Not found**: verify the HuggingFace ID; check that the HF cache at `/scratch/project_462000963/cache` is accessible
- **Gated model**: ensure `HF_TOKEN` is set and has model access

### No Gradio URL
Check `logs/<JOB_ID>.err`. Common cause: Gradio couldn't reach its relay server. The `run_sing_pip_install "gradio>=4"` step in the launcher ensures a fresh install.

### Slow first response
First inference triggers ROCm kernel JIT compilation. Subsequent responses are faster.
