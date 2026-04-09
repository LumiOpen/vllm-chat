import gradio as gr
import random
import requests
import subprocess
import threading
import time
import atexit
import sys
import json
import os
from typing import List, Dict
from transformers import AutoTokenizer

###############################################################################
# Configuration & Constants
###############################################################################

MAX_MODEL_LENGTH = 131072
BUFFER_TOKENS = 32

FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}"
    "{% elif message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}"
)


def _log(msg):
    print(f"[DEBUG] {msg}", flush=True)


###############################################################################
# GPU & Argument Parsing
###############################################################################

try:
    import torch
    total_gpus_available = torch.cuda.device_count()
except ImportError:
    total_gpus_available = 1


def parse_model_arg(arg: str):
    if ":" in arg:
        prefix, model = arg.split(":", 1)
        return int(prefix), model
    return 1, arg


default_model = "meta-llama/Llama-3.2-3B-Instruct"
args = sys.argv[1:]

if len(args) == 0:
    model_specs = [(1, default_model)]
elif len(args) == 1:
    model_specs = [parse_model_arg(args[0])]
elif len(args) == 2:
    model_specs = [parse_model_arg(args[0]), parse_model_arg(args[1])]
else:
    raise ValueError("Usage: vllm_chat.py [GPUS1:MODEL1] [GPUS2:MODEL2]")

num_models = len(model_specs)
gpu_count_1, model_name1 = model_specs[0]
gpu_count_2, model_name2 = model_specs[1] if num_models == 2 else (0, None)

total_gpus_needed = gpu_count_1 + gpu_count_2
if total_gpus_needed > total_gpus_available:
    raise ValueError(
        f"Requested {total_gpus_needed} GPUs but only {total_gpus_available} available."
    )

###############################################################################
# Tokenizers
###############################################################################

tokenizer1 = AutoTokenizer.from_pretrained(model_name1)
if not getattr(tokenizer1, "chat_template", None):
    tokenizer1.chat_template = FALLBACK_CHAT_TEMPLATE

tokenizer2 = None
if num_models == 2:
    tokenizer2 = AutoTokenizer.from_pretrained(model_name2)
    if not getattr(tokenizer2, "chat_template", None):
        tokenizer2.chat_template = FALLBACK_CHAT_TEMPLATE

tokenizers = [tokenizer1, tokenizer2]
model_names = [model_name1, model_name2]

###############################################################################
# GPU Assignment & Environment Setup
###############################################################################

gpu_ids_1 = list(range(gpu_count_1))

if num_models == 2:
    start_gpu_2 = gpu_count_1
    gpu_ids_2 = list(range(start_gpu_2, start_gpu_2 + gpu_count_2))
    all_gpu_ids = ",".join(str(i) for i in range(total_gpus_needed))
else:
    gpu_ids_2 = []
    all_gpu_ids = ",".join(str(i) for i in range(gpu_count_1))

os.environ["CUDA_VISIBLE_DEVICES"] = all_gpu_ids

###############################################################################
# Launch vLLM Servers
###############################################################################

print("Launching vLLM server(s)...")

def make_server_env(gpu_ids: list, server_idx: int) -> dict:
    env = os.environ.copy()
    for var in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        env.pop(var, None)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    env["VLLM_HOST_IP"] = "127.0.0.1"
    env["VLLM_PORT"] = str(29600 + server_idx * 10)
    env["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_server{server_idx + 1}"
    env["XDG_CACHE_HOME"] = f"/tmp/xdg_cache_server{server_idx + 1}"
    env["TORCH_EXTENSIONS_DIR"] = f"/dev/shm/torch_ext_server{server_idx + 1}"
    return env


def launch_server(model: str, port: int, tp_size: int, env: dict) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model, "--port", str(port),
        "--tensor-parallel-size", str(tp_size),
    ]
    return subprocess.Popen(cmd, env=env)


ports = [8001]
procs = [launch_server(model_name1, 8001, gpu_count_1, make_server_env(gpu_ids_1, 0))]

if num_models == 2:
    ports.append(8002)
    procs.append(launch_server(model_name2, 8002, gpu_count_2, make_server_env(gpu_ids_2, 1)))

print("Server process(es) started.")
atexit.register(lambda: [p.terminate() for p in procs])

###############################################################################
# Wait for /health
###############################################################################

def wait_for_health(port, tries=600, delay=2):
    url = f"http://localhost:{port}/health"
    for _ in range(tries):
        try:
            if requests.get(url).status_code == 200:
                print(f"Server on port {port} is healthy.")
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(delay)
    raise RuntimeError(f"vLLM server on port {port} failed to start in time.")

print("Waiting for server(s) to become healthy...")
for port in ports:
    wait_for_health(port)
print("All health checks passed.")

###############################################################################
# Helpers
###############################################################################

def _normalize_content(content) -> str:
    """Extract plain text from Gradio 6 structured content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _normalize_history(history: list) -> List[Dict[str, str]]:
    """Convert Gradio 6 structured messages to plain {role, content} dicts."""
    return [
        {"role": (msg.get("role", "user") if isinstance(msg, dict) else "user"),
         "content": _normalize_content(msg.get("content", "") if isinstance(msg, dict) else str(msg))}
        for msg in history
    ]


def parse_vllm_sse_line(line: str):
    for chunk in line.split("data:"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk == "[DONE]":
            yield None
        else:
            try:
                yield json.loads(chunk)
            except json.JSONDecodeError:
                continue


def format_chat_prompt(msgs, tokenizer):
    if msgs and msgs[-1]["role"] == "assistant" and not msgs[-1]["content"]:
        msgs = msgs[:-1]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))


def truncate_history(msgs, tokenizer, max_tokens):
    budget = MAX_MODEL_LENGTH - (max_tokens + BUFFER_TOKENS)
    if count_tokens(format_chat_prompt(msgs, tokenizer), tokenizer) <= budget:
        return msgs
    trimmed = msgs.copy()
    while count_tokens(format_chat_prompt(trimmed, tokenizer), tokenizer) > budget and len(trimmed) > 1:
        for i, m in enumerate(trimmed):
            if m["role"] != "system":
                trimmed.pop(i)
                break
    return trimmed


def _build_prompt(history, model_idx, max_tokens):
    """Normalise chatbot history and build a vLLM prompt string."""
    tok = tokenizers[model_idx]
    normalised = _normalize_history(history)
    normalised = truncate_history(normalised, tok, max_tokens)
    return format_chat_prompt(normalised, tok)


def _make_payload(model_idx, prompt, temp, top_p, top_k, max_tokens):
    return {
        "model": model_names[model_idx],
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": temp,
        "top_p": top_p,
        "top_k": top_k,
        "seed": random.randint(0, 2**32),
    }


###############################################################################
# Dual-mode background streamer (thread-based, needed for 2 concurrent HTTP)
###############################################################################

def _stream_to_buffer(url, payload, buffer, lock, done):
    """Thread target: POST to vLLM and accumulate tokens in buffer[0]."""
    try:
        with requests.post(url, json=payload, stream=True) as resp:
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                for data in parse_vllm_sse_line(raw):
                    if data is None:
                        done[0] = True
                        return
                    token = data.get("choices", [{}])[0].get("text", "")
                    if token:
                        with lock:
                            buffer[0] += token
        done[0] = True
    except Exception as e:
        _log(f"_stream_to_buffer error: {e}")
        done[0] = True


###############################################################################
# Core: mode resolution
###############################################################################

def _resolve_model_idx(mode, model_choice):
    if mode == "Dual Model":
        return -1
    if num_models == 2 and model_choice != model_name1:
        return 1
    return 0


###############################################################################
# UI dispatch
###############################################################################

def user_submit(user_msg, mode, model_choice, hist1, hist2):
    """Add user message and clear textbox.  Runs with queue=False."""
    if not user_msg or not user_msg.strip():
        return user_msg, hist1, hist2

    midx = _resolve_model_idx(mode, model_choice)
    if midx == -1:
        hist1 = list(hist1) + [{"role": "user", "content": user_msg}]
        hist2 = list(hist2) + [{"role": "user", "content": user_msg}]
    else:
        hist1 = list(hist1) + [{"role": "user", "content": user_msg}]
    return "", hist1, hist2


def bot_respond(mode, model_choice, hist1, hist2,
                temp, top_p, top_k, max_tokens):
    """Stream assistant response.  Modifies hist1/hist2 IN PLACE and yields
    the *same* objects back — this is critical for Gradio's streaming to work.
    """
    midx = _resolve_model_idx(mode, model_choice)
    _log(f"bot_respond: midx={midx}, hist1_len={len(hist1)}, hist2_len={len(hist2)}")

    if midx == -1:
        # ── Dual mode ──────────────────────────────────────────────
        prompt1 = _build_prompt(hist1, 0, max_tokens)
        prompt2 = _build_prompt(hist2, 1, max_tokens)

        hist1.append({"role": "assistant", "content": ""})
        hist2.append({"role": "assistant", "content": ""})

        buf1, lock1, done1 = [""], threading.Lock(), [False]
        buf2, lock2, done2 = [""], threading.Lock(), [False]
        pay1 = _make_payload(0, prompt1, temp, top_p, top_k, max_tokens)
        pay2 = _make_payload(1, prompt2, temp, top_p, top_k, max_tokens)
        threading.Thread(target=_stream_to_buffer, daemon=True,
                         args=(f"http://localhost:{ports[0]}/v1/completions", pay1, buf1, lock1, done1)).start()
        threading.Thread(target=_stream_to_buffer, daemon=True,
                         args=(f"http://localhost:{ports[1]}/v1/completions", pay2, buf2, lock2, done2)).start()

        while not (done1[0] and done2[0]):
            with lock1:
                hist1[-1]["content"] = buf1[0]
            with lock2:
                hist2[-1]["content"] = buf2[0]
            yield hist1, hist2
            time.sleep(0.1)

        hist1[-1]["content"] = buf1[0]
        hist2[-1]["content"] = buf2[0]
        _log(f"bot_respond dual done: buf1={len(buf1[0])}, buf2={len(buf2[0])}")
        yield hist1, hist2
    else:
        # ── Single mode ────────────────────────────────────────────
        prompt = _build_prompt(hist1, midx, max_tokens)
        hist1.append({"role": "assistant", "content": ""})

        payload = _make_payload(midx, prompt, temp, top_p, top_k, max_tokens)
        url = f"http://localhost:{ports[midx]}/v1/completions"

        _log(f"bot_respond single: POST {url}")
        last_yield = time.monotonic()
        token_count = 0
        try:
            with requests.post(url, json=payload, stream=True) as resp:
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    for data in parse_vllm_sse_line(raw):
                        if data is None:
                            _log(f"bot_respond single done: {token_count} tokens")
                            yield hist1, hist2
                            return
                        token = data.get("choices", [{}])[0].get("text", "")
                        if token:
                            token_count += 1
                            hist1[-1]["content"] += token
                            now = time.monotonic()
                            if now - last_yield >= 0.05:
                                yield hist1, hist2
                                last_yield = now
        except Exception as e:
            _log(f"bot_respond single error: {e}")
        yield hist1, hist2


def _strip_last_turn(hist):
    """Remove last user+assistant pair.  Returns (trimmed, user_text)."""
    if not hist:
        return hist, None
    if (hist[-1].get("role") == "assistant"
            and len(hist) >= 2
            and hist[-2].get("role") == "user"):
        return hist[:-2], _normalize_content(hist[-2].get("content", ""))
    if hist[-1].get("role") == "user":
        return hist[:-1], _normalize_content(hist[-1].get("content", ""))
    return hist, None


def delete_last(mode, model_choice, hist1, hist2):
    midx = _resolve_model_idx(mode, model_choice)
    if midx == -1:
        h1, _ = _strip_last_turn(hist1)
        h2, _ = _strip_last_turn(hist2)
        return h1, h2
    else:
        h1, _ = _strip_last_turn(hist1)
        return h1, hist2


def regenerate(mode, model_choice, hist1, hist2,
               temp, top_p, top_k, max_tokens):
    midx = _resolve_model_idx(mode, model_choice)

    if midx == -1:
        hist1, user_msg = _strip_last_turn(hist1)
        if user_msg is None:
            yield hist1, hist2
            return
        hist2, _ = _strip_last_turn(hist2)
        hist1 = list(hist1) + [{"role": "user", "content": user_msg}]
        hist2 = list(hist2) + [{"role": "user", "content": user_msg}]
        yield from bot_respond(mode, model_choice, hist1, hist2,
                               temp, top_p, top_k, max_tokens)
    else:
        hist1, user_msg = _strip_last_turn(hist1)
        if user_msg is None:
            yield hist1, hist2
            return
        hist1 = list(hist1) + [{"role": "user", "content": user_msg}]
        yield from bot_respond(mode, model_choice, hist1, hist2,
                               temp, top_p, top_k, max_tokens)


def edit_last(mode, model_choice, hist1, hist2):
    midx = _resolve_model_idx(mode, model_choice)
    if midx == -1:
        hist1, user_msg = _strip_last_turn(hist1)
        hist2, _ = _strip_last_turn(hist2)
        return hist1, hist2, user_msg or ""
    else:
        hist1, user_msg = _strip_last_turn(hist1)
        return hist1, hist2, user_msg or ""


def clear_all():
    return [], []


def switch_mode(mode):
    is_dual = mode == "Dual Model"
    return (
        [],
        gr.Chatbot(value=[], visible=is_dual),
        gr.Dropdown(visible=not is_dual),
    )


def on_model_change(model_choice):
    """Clear chatbot1 when the selected model changes in single mode."""
    return []


###############################################################################
# Gradio UI
###############################################################################

_log(f"Gradio version: {gr.__version__}")

with gr.Blocks() as demo:
    gr.Markdown("## vLLM Chat Interface")

    if num_models == 2:
        gr.Markdown(
            f"**Model A**: {model_name1} ({gpu_count_1} GPU(s)) &nbsp;&nbsp; "
            f"**Model B**: {model_name2} ({gpu_count_2} GPU(s))"
        )
    else:
        gr.Markdown(f"**Model**: {model_name1} ({gpu_count_1} GPU(s))")

    with gr.Accordion("Settings", open=True):
        with gr.Row():
            if num_models == 2:
                mode_radio = gr.Radio(
                    choices=["Single Model", "Dual Model"],
                    value="Single Model", label="Mode",
                )
                model_dropdown = gr.Dropdown(
                    choices=[model_name1, model_name2],
                    value=model_name1, label="Model",
                )
            else:
                mode_radio = gr.Radio(
                    choices=["Single Model"], value="Single Model",
                    label="Mode", visible=False,
                )
                model_dropdown = gr.Dropdown(
                    choices=[model_name1], value=model_name1,
                    label="Model", visible=False,
                )
        with gr.Row():
            temp_slider = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Temperature")
            top_p_slider = gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="Top-p")
        with gr.Row():
            top_k_slider = gr.Slider(-1, 100, value=40, step=1, label="Top-k")
            max_tokens_slider = gr.Slider(2048, 65536, value=32768, step=1, label="Max tokens")

    with gr.Row():
        chatbot1 = gr.Chatbot(label=model_name1, height=600)
        chatbot2 = gr.Chatbot(
            label=model_name2 if model_name2 else "",
            height=600, visible=False,
        )

    with gr.Row():
        regen_btn = gr.Button("🔄 Regenerate", variant="secondary")
        delete_btn = gr.Button("🗑️ Delete last", variant="secondary")
        edit_btn = gr.Button("✏️ Edit last", variant="secondary")
        clear_btn = gr.Button("🧹 Clear", variant="secondary")

    with gr.Row():
        txt = gr.Textbox(show_label=False,
                         placeholder="Enter your message and press Enter",
                         scale=8)
        send_btn = gr.Button("Send", variant="primary", scale=1, elem_id="send-btn")

    # --- Wiring ---
    chatbots = [chatbot1, chatbot2]
    user_inputs = [txt, mode_radio, model_dropdown, chatbot1, chatbot2]
    user_outputs = [txt, chatbot1, chatbot2]
    bot_inputs = [mode_radio, model_dropdown, chatbot1, chatbot2,
                  temp_slider, top_p_slider, top_k_slider, max_tokens_slider]

    txt.submit(
        user_submit, user_inputs, user_outputs, queue=False
    ).then(bot_respond, bot_inputs, chatbots)

    send_btn.click(
        user_submit, user_inputs, user_outputs, queue=False
    ).then(bot_respond, bot_inputs, chatbots)

    regen_btn.click(regenerate, bot_inputs, chatbots)

    delete_inputs = [mode_radio, model_dropdown, chatbot1, chatbot2]
    delete_btn.click(delete_last, delete_inputs, chatbots)
    edit_btn.click(edit_last, delete_inputs, chatbots + [txt])
    clear_btn.click(clear_all, None, chatbots)

    if num_models == 2:
        mode_radio.change(
            switch_mode, [mode_radio],
            [chatbot1, chatbot2, model_dropdown],
            queue=False,
        )
        model_dropdown.change(
            on_model_change, [model_dropdown], [chatbot1],
            queue=False,
        )

###############################################################################
# Launch
###############################################################################

if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()

    print("Launching chat interface...")
    demo.queue(default_concurrency_limit=16)
    _app, _local, _share = demo.launch(
        share=True, server_name="0.0.0.0", prevent_thread_lock=True,
    )

    if not _share:
        print()
        print("=" * 64)
        print(f"  SSH tunnel:  ssh -L 7860:{hostname}:7860 <user>@lumi.csc.fi")
        print(f"  Then open:   http://localhost:7860")
        print("=" * 64)
        print()

    demo.block_thread()
