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
    """Debug logger — prints with flush so output appears immediately in logs."""
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


# Accept 1 or 2 model arguments
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
    print("⚠️  Tokenizer 1 has no chat_template – using a generic fallback.")
    tokenizer1.chat_template = FALLBACK_CHAT_TEMPLATE

tokenizer2 = None
if num_models == 2:
    tokenizer2 = AutoTokenizer.from_pretrained(model_name2)
    if not getattr(tokenizer2, "chat_template", None):
        print("⚠️  Tokenizer 2 has no chat_template – using a generic fallback.")
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
# Launch vLLM Servers (sub-processes)
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
        "--model", model,
        "--port", str(port),
        "--tensor-parallel-size", str(tp_size),
    ]
    return subprocess.Popen(cmd, env=env)


ports = [8001]
procs = [launch_server(model_name1, 8001, gpu_count_1, make_server_env(gpu_ids_1, 0))]

if num_models == 2:
    ports.append(8002)
    procs.append(launch_server(model_name2, 8002, gpu_count_2, make_server_env(gpu_ids_2, 1)))

print("Server process(es) started.")


def cleanup():
    for p in procs:
        p.terminate()


atexit.register(cleanup)
print("Exit handler registered.")

###############################################################################
# Wait for /health
###############################################################################


def wait_for_health(port: int, tries: int = 600, delay: int = 2):
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
# Helpers – Gradio 6 content normalisation, prompt trimming, SSE parsing
###############################################################################


def _normalize_content(content) -> str:
    """Extract plain text from Gradio 6 structured content.

    Gradio 6 wraps every message content as a list of typed blocks, e.g.
        [{'text': 'hello', 'type': 'text'}]
    This converts back to a plain string for the tokenizer / textbox.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def _normalize_history(history: list) -> List[Dict[str, str]]:
    """Convert Gradio 6 structured messages to plain {role, content} dicts.

    Gradio 6 returns chatbot values with extra keys (metadata, options)
    and structured content blocks.  The tokenizer and vLLM expect plain
    role/content dicts with string content.
    """
    out = []
    for msg in history:
        role = msg.get("role", "user") if isinstance(msg, dict) else "user"
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        out.append({"role": role, "content": _normalize_content(content)})
    return out


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


def format_chat_prompt(msgs: List[Dict[str, str]], tokenizer) -> str:
    # Strip trailing empty assistant placeholder so the template only adds
    # one generation prompt (not an empty turn + another prompt).
    if msgs and msgs[-1]["role"] == "assistant" and not msgs[-1]["content"]:
        msgs = msgs[:-1]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text))


def truncate_history(msgs: List[Dict[str, str]], tokenizer, max_tokens: int) -> List[Dict[str, str]]:
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


###############################################################################
# Streaming generator (used by dual mode)
###############################################################################


def stream_from_server(url: str, payload: dict, buffer: list, lock: threading.Lock, done: list):
    _log(f"stream_from_server: POST {url}")
    try:
        with requests.post(url, json=payload, stream=True) as resp:
            _log(f"stream_from_server: response status={resp.status_code}")
            line_count = 0
            token_count = 0
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line_count += 1
                if line_count <= 3:
                    _log(f"stream_from_server: raw line {line_count}: {raw[:200]}")
                for data in parse_vllm_sse_line(raw):
                    if data is None:
                        done[0] = True
                        break
                    token = data.get("choices", [{}])[0].get("text", "")
                    if token:
                        token_count += 1
                        with lock:
                            buffer[0] += token
                if done[0]:
                    break
            _log(f"stream_from_server: finished. lines={line_count}, tokens={token_count}, buf_len={len(buffer[0])}")
        done[0] = True
    except Exception as e:
        _log(f"stream_from_server: EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        done[0] = True


###############################################################################
# Core chat logic
###############################################################################


def _resolve_model_idx(mode, model_choice):
    if mode == "Dual Model":
        return -1
    if num_models == 2 and model_choice != model_name1:
        return 1
    return 0


def _stream_single(history: List[Dict], model_idx: int,
                   temp: float, top_p: float, top_k: int, max_tokens: int):
    """Add an assistant placeholder and stream tokens. Yields history.

    *history* is already normalised (plain role/content dicts).
    """
    _log(f"_stream_single: ENTER model_idx={model_idx}, history has {len(history)} msgs")

    tok = tokenizers[model_idx]
    port = ports[model_idx]

    history = list(history) + [{"role": "assistant", "content": ""}]
    history = truncate_history(history, tok, max_tokens)

    prompt = format_chat_prompt(history, tok)
    _log(f"_stream_single: prompt length={len(prompt)} chars, first 200: {prompt[:200]}")

    payload = {
        "model": model_names[model_idx],
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": temp,
        "top_p": top_p,
        "top_k": top_k,
        "seed": random.randint(0, 2**32),
    }

    url = f"http://localhost:{port}/v1/completions"
    _log(f"_stream_single: POST {url}")
    last_yield = time.monotonic()
    yield_count = 0
    token_count = 0
    try:
        with requests.post(url, json=payload, stream=True) as resp:
            _log(f"_stream_single: response status={resp.status_code}")
            line_count = 0
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line_count += 1
                if line_count <= 3:
                    _log(f"_stream_single: SSE line {line_count}: {raw[:200]}")
                for data in parse_vllm_sse_line(raw):
                    if data is None:
                        _log(f"_stream_single: got [DONE] after {line_count} lines, {token_count} tokens, {yield_count} yields")
                        yield history
                        return
                    token = data.get("choices", [{}])[0].get("text", "")
                    if token:
                        token_count += 1
                        history[-1]["content"] += token
                        now = time.monotonic()
                        if now - last_yield >= 0.05:
                            yield_count += 1
                            yield history
                            last_yield = now
            _log(f"_stream_single: iter_lines exhausted. lines={line_count}, tokens={token_count}, yields={yield_count}")
    except Exception as e:
        _log(f"_stream_single: EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
    _log(f"_stream_single: EXIT. total yields={yield_count}, final content_len={len(history[-1]['content'])}")
    yield history


def _stream_dual(hist1: List[Dict], hist2: List[Dict],
                 temp: float, top_p: float, top_k: int, max_tokens: int):
    """Add assistant placeholders to both histories and stream. Yields (h1, h2).

    Both histories are already normalised.
    """
    _log(f"_stream_dual: ENTER hist1={len(hist1)} msgs, hist2={len(hist2)} msgs")

    hist1 = list(hist1) + [{"role": "assistant", "content": ""}]
    hist2 = list(hist2) + [{"role": "assistant", "content": ""}]

    hist1 = truncate_history(hist1, tokenizer1, max_tokens)
    hist2 = truncate_history(hist2, tokenizer2, max_tokens)

    def make_payload(model_name, hist, tok):
        return {
            "model": model_name,
            "prompt": format_chat_prompt(hist, tok),
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "seed": random.randint(0, 2**32),
        }

    buf1, lock1, done1 = [""], threading.Lock(), [False]
    buf2, lock2, done2 = [""], threading.Lock(), [False]

    threading.Thread(
        target=stream_from_server,
        args=(f"http://localhost:{ports[0]}/v1/completions",
              make_payload(model_name1, hist1, tokenizer1), buf1, lock1, done1),
        daemon=True,
    ).start()
    threading.Thread(
        target=stream_from_server,
        args=(f"http://localhost:{ports[1]}/v1/completions",
              make_payload(model_name2, hist2, tokenizer2), buf2, lock2, done2),
        daemon=True,
    ).start()

    yield_count = 0
    while not (done1[0] and done2[0]):
        with lock1:
            hist1[-1]["content"] = buf1[0]
        with lock2:
            hist2[-1]["content"] = buf2[0]
        yield_count += 1
        if yield_count <= 5:
            _log(f"_stream_dual: yield #{yield_count}, buf1_len={len(buf1[0])}, buf2_len={len(buf2[0])}, done1={done1[0]}, done2={done2[0]}")
        yield hist1, hist2
        time.sleep(0.1)

    hist1[-1]["content"] = buf1[0]
    hist2[-1]["content"] = buf2[0]
    _log(f"_stream_dual: EXIT. yields={yield_count}, final buf1_len={len(buf1[0])}, buf2_len={len(buf2[0])}")
    yield hist1, hist2


###############################################################################
# UI dispatch: route actions through mode
###############################################################################


def user_submit(user_msg, mode, model_choice, hist1, hist2):
    """Add the user message to the correct history and clear the textbox."""
    _log(f"user_submit: ENTER msg={user_msg!r}, mode={mode!r}, model_choice={model_choice!r}")

    if not user_msg or not user_msg.strip():
        _log("user_submit: empty msg, returning unchanged")
        return user_msg, hist1, hist2

    midx = _resolve_model_idx(mode, model_choice)
    _log(f"user_submit: midx={midx}")
    if midx == -1:
        # Dual mode: add to both
        hist1 = list(hist1) + [{"role": "user", "content": user_msg}]
        hist2 = list(hist2) + [{"role": "user", "content": user_msg}]
    else:
        # Single mode: always display in chatbot1 regardless of model
        hist1 = list(hist1) + [{"role": "user", "content": user_msg}]

    _log(f"user_submit: EXIT returning hist1_len={len(hist1)}, hist2_len={len(hist2)}")
    return "", hist1, hist2


def bot_respond(mode, model_choice, hist1, hist2,
                temp, top_p, top_k, max_tokens):
    """Stream the assistant response. Histories already contain the user message."""
    _log(f"bot_respond: ENTER mode={mode!r}, model_choice={model_choice!r}, hist1_len={len(hist1)}, hist2_len={len(hist2)}")

    midx = _resolve_model_idx(mode, model_choice)
    _log(f"bot_respond: midx={midx}")

    if midx == -1:
        # Dual: normalise both, stream both
        h1 = _normalize_history(hist1)
        h2 = _normalize_history(hist2)
        for h1, h2 in _stream_dual(h1, h2, temp, top_p, top_k, max_tokens):
            yield h1, h2
    else:
        # Single: always use chatbot1 for display, pick model by midx
        h1 = _normalize_history(hist1)
        for h1 in _stream_single(h1, midx, temp, top_p, top_k, max_tokens):
            yield h1, hist2

    _log("bot_respond: EXIT")


def _strip_last_turn(hist):
    """Remove the last user+assistant pair (or lone user message).
    Returns (trimmed_history, user_message_text) or (hist, None)."""
    if not hist:
        return hist, None
    if hist[-1]["role"] == "assistant" and len(hist) >= 2 and hist[-2]["role"] == "user":
        user_content = hist[-2].get("content", "")
        return hist[:-2], _normalize_content(user_content)
    if hist[-1]["role"] == "user":
        user_content = hist[-1].get("content", "")
        return hist[:-1], _normalize_content(user_content)
    return hist, None


def delete_last(mode, model_choice, hist1, hist2):
    """Delete the last user+assistant turn."""
    midx = _resolve_model_idx(mode, model_choice)
    if midx == -1:
        h1, _ = _strip_last_turn(hist1)
        h2, _ = _strip_last_turn(hist2)
        return h1, h2
    else:
        # Single mode: history is always in chatbot1
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
        h1 = _normalize_history(hist1) + [{"role": "user", "content": user_msg}]
        h2 = _normalize_history(hist2) + [{"role": "user", "content": user_msg}]
        for h1, h2 in _stream_dual(h1, h2, temp, top_p, top_k, max_tokens):
            yield h1, h2
    else:
        # Single mode: history is in chatbot1
        hist1, user_msg = _strip_last_turn(hist1)
        if user_msg is None:
            yield hist1, hist2
            return
        h1 = _normalize_history(hist1) + [{"role": "user", "content": user_msg}]
        for h1 in _stream_single(h1, midx, temp, top_p, top_k, max_tokens):
            yield h1, hist2


def edit_last(mode, model_choice, hist1, hist2):
    """Remove the last user+assistant turn and return the user text to the textbox."""
    midx = _resolve_model_idx(mode, model_choice)

    if midx == -1:
        hist1, user_msg = _strip_last_turn(hist1)
        hist2, _ = _strip_last_turn(hist2)
        return hist1, hist2, user_msg or ""
    else:
        # Single mode: history is in chatbot1
        hist1, user_msg = _strip_last_turn(hist1)
        return hist1, hist2, user_msg or ""


def clear_all():
    _log("clear_all: called")
    return [], []


def switch_mode(mode):
    """When mode changes, clear history and toggle visibility."""
    _log(f"switch_mode: mode={mode!r}")
    is_dual = mode == "Dual Model"
    return (
        [],
        gr.Chatbot(value=[], visible=is_dual),
        gr.Dropdown(visible=not is_dual),
    )


###############################################################################
# Gradio UI
###############################################################################

_log(f"Gradio version: {gr.__version__}")

with gr.Blocks() as demo:
    gr.Markdown("## vLLM Chat Interface")

    # Header showing deployed models
    if num_models == 2:
        gr.Markdown(
            f"**Model A**: {model_name1} ({gpu_count_1} GPU(s)) &nbsp;&nbsp; "
            f"**Model B**: {model_name2} ({gpu_count_2} GPU(s))"
        )
    else:
        gr.Markdown(f"**Model**: {model_name1} ({gpu_count_1} GPU(s))")

    # Settings accordion
    with gr.Accordion("Settings", open=True):
        with gr.Row():
            if num_models == 2:
                mode_radio = gr.Radio(
                    choices=["Single Model", "Dual Model"],
                    value="Single Model",
                    label="Mode",
                )
                model_dropdown = gr.Dropdown(
                    choices=[model_name1, model_name2],
                    value=model_name1,
                    label="Model",
                )
            else:
                mode_radio = gr.Radio(
                    choices=["Single Model"],
                    value="Single Model",
                    label="Mode",
                    visible=False,
                )
                model_dropdown = gr.Dropdown(
                    choices=[model_name1],
                    value=model_name1,
                    label="Model",
                    visible=False,
                )
        with gr.Row():
            temp_slider = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Temperature")
            top_p_slider = gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="Top-p")
        with gr.Row():
            top_k_slider = gr.Slider(-1, 100, value=40, step=1, label="Top-k")
            max_tokens_slider = gr.Slider(2048, 65536, value=32768, step=1, label="Max tokens")

    # Chat area
    with gr.Row():
        chatbot1 = gr.Chatbot(
            label=model_name1,
            height=600,
        )
        chatbot2 = gr.Chatbot(
            label=model_name2 if model_name2 else "",
            height=600,
            visible=False,
        )

    # Control buttons
    with gr.Row():
        regen_btn = gr.Button("🔄 Regenerate", variant="secondary")
        delete_btn = gr.Button("🗑️ Delete last", variant="secondary")
        edit_btn = gr.Button("✏️ Edit last", variant="secondary")
        clear_btn = gr.Button("🧹 Clear", variant="secondary")

    # Input row
    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="Enter your message and press Enter",
            scale=8,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1, elem_id="send-btn")

    # --- Wiring ---
    # Chatbot components serve as both display and state (no separate gr.State).
    chatbots = [chatbot1, chatbot2]
    user_inputs = [txt, mode_radio, model_dropdown, chatbot1, chatbot2]
    user_outputs = [txt, chatbot1, chatbot2]
    bot_inputs = [mode_radio, model_dropdown, chatbot1, chatbot2,
                  temp_slider, top_p_slider, top_k_slider, max_tokens_slider]

    # Send: instantly show user message, then stream the bot response
    txt.submit(
        user_submit, user_inputs, user_outputs, queue=False
    ).then(
        bot_respond, bot_inputs, chatbots, show_progress="hidden"
    )
    send_btn.click(
        user_submit, user_inputs, user_outputs, queue=False
    ).then(
        bot_respond, bot_inputs, chatbots, show_progress="hidden"
    )

    # Regenerate
    regen_btn.click(
        regenerate, bot_inputs, chatbots, show_progress="hidden"
    )

    # Delete last
    delete_inputs = [mode_radio, model_dropdown, chatbot1, chatbot2]
    delete_btn.click(delete_last, delete_inputs, chatbots)

    # Edit last
    edit_btn.click(edit_last, delete_inputs, chatbots + [txt])

    # Clear
    clear_btn.click(clear_all, None, chatbots)

    # Mode switch
    if num_models == 2:
        mode_radio.change(
            switch_mode,
            [mode_radio],
            [chatbot1, chatbot2, model_dropdown],
            show_progress="hidden",
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
        print("  Gradio share link unavailable (compute node cannot")
        print("  reach Gradio servers).  Access the UI via SSH tunnel:")
        print()
        print(f"    ssh -L 7860:{hostname}:7860 <user>@lumi.csc.fi")
        print()
        print("  Then open:  http://localhost:7860")
        print("=" * 64)
        print()

    demo.block_thread()
