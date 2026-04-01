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

MAX_MODEL_LENGTH = 131072  # tokens (Llama‑3 8 K)
GENERATION_TOKENS = 2048
BUFFER_TOKENS = 32  # cushion so the prompt never overflows

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
gpu_count, model_name = parse_model_arg(sys.argv[1] if len(sys.argv) > 1 else default_model)
if gpu_count > total_gpus_available:
    raise ValueError(f"Requested {gpu_count} GPUs but only {total_gpus_available} available.")

###############################################################################
# Tokenizer
###############################################################################

tokenizer = AutoTokenizer.from_pretrained(model_name)
if not getattr(tokenizer, "chat_template", None):
    print("⚠️  Tokenizer has no chat_template – using a generic fallback.")
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}"
        "{% elif message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}"
    )

###############################################################################
# Launch vLLM Server (sub‑process)
###############################################################################

print("Launching vllm")
gpu_ids = ",".join(map(str, range(gpu_count)))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
server_cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", model_name,
    "--port", "8001",
    "--tensor-parallel-size", str(gpu_count),
]
proc = subprocess.Popen(server_cmd)
print("popen returned")

aTexit = atexit.register(lambda: proc.terminate())  # ensure cleanup
print("exit handler registered")

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
    raise RuntimeError("vLLM server failed to start in time.")

print("Waiting for healthy...")
wait_for_health(8001)

print("Health check passed, continuing..")

###############################################################################
# Helpers – prompt trimming & SSE parsing
###############################################################################

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

def format_chat_prompt(msgs: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def truncate_history(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    budget = MAX_MODEL_LENGTH - (GENERATION_TOKENS + BUFFER_TOKENS)
    if count_tokens(format_chat_prompt(msgs)) <= budget:
        return msgs
    trimmed = msgs.copy()
    while count_tokens(format_chat_prompt(trimmed)) > budget and len(trimmed) > 1:
        for i, m in enumerate(trimmed):
            if m["role"] != "system":
                trimmed.pop(i)
                break
    return trimmed

###############################################################################
# Streaming generator
###############################################################################

def stream_from_server(url: str, payload: dict, buffer: list, lock: threading.Lock, done: list):
    try:
        with requests.post(url, json=payload, stream=True) as resp:
            for raw in resp.iter_lines(decode_unicode=True, delimiter="\n"):
                if not raw:
                    continue
                for data in parse_vllm_sse_line(raw):
                    if data is None:
                        done[0] = True
                        break
                    token = data.get("choices", [{}])[0].get("text", "")
                    with lock:
                        buffer[0] += token
                if done[0]:
                    break
        done[0] = True
    except Exception as e:
        print("Streaming error:", e)
        done[0] = True

###############################################################################
# Core chat logic
###############################################################################

def generate_assistant(user_msg: str, history: List[Dict[str, str]]):
    history = history + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": ""}]
    history = truncate_history(history)

    payload = {
        "model": model_name,
        "prompt": format_chat_prompt(history),
        "max_tokens": GENERATION_TOKENS,
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "seed": random.randint(0, 2**32),
    }

    url = "http://localhost:8001/v1/completions"
    buffer, lock, done = [""], threading.Lock(), [False]
    threading.Thread(target=stream_from_server, args=(url, payload, buffer, lock, done), daemon=True).start()

    while not done[0]:
        with lock:
            cur = buffer[0]
        if history[-1]["content"] != cur:
            history[-1]["content"] = cur
            yield history, history
        time.sleep(0.1)

    history[-1]["content"] = buffer[0]
    yield history, history

###############################################################################
# Helper ops – delete / regenerate
###############################################################################

def delete_last(hist):
    return (hist[:-1], hist[:-1]) if hist else (hist, hist)

def regenerate(hist):
    if not hist:
        yield hist, hist; return
    if hist[-1]["role"] == "assistant" and len(hist) >= 2 and hist[-2]["role"] == "user":
        user_msg, prefix = hist[-2]["content"], hist[:-1]
    elif hist[-1]["role"] == "user":
        user_msg, prefix = hist[-1]["content"], hist[:-1]
    else:
        yield hist, hist; return
    yield from generate_assistant(user_msg, prefix)

###############################################################################
# Gradio UI
###############################################################################

with gr.Blocks() as demo:
    gr.Markdown("## Single‑Model Chat Interface (vLLM Backend)")
    gr.Markdown(f"**Model**: {model_name} using {gpu_count} GPU(s)")

    # Taller chat log
    chatbot = gr.Chatbot(label=model_name, height=600)
    state = gr.State([])

    # Control row (regenerate / delete / clear) — placed directly UNDER chat log, ABOVE input
    with gr.Row():
        regen_btn = gr.Button("🔄 Regenerate last", variant="secondary")
        delete_btn = gr.Button("🗑️ Delete last", variant="secondary")
        clear_btn = gr.ClearButton(value="Clear", components=[chatbot, state])

    # Input row
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Enter your message and press Enter", scale=8)
        send_btn = gr.Button("Send", variant="primary", scale=1, elem_id="send-btn")

    # Wiring
    txt.submit(generate_assistant, [txt, state], [chatbot, state])
    send_btn.click(generate_assistant, [txt, state], [chatbot, state])

    txt.submit(lambda _: gr.update(value=""), txt, txt)
    send_btn.click(lambda _: gr.update(value=""), txt, txt)

    regen_btn.click(regenerate, [state], [chatbot, state])
    delete_btn.click(delete_last, [state], [chatbot, state])

# Enable up to 16 simultaneous tasks (threads) so multiple users aren’t stuck waiting
# Adjust as high as your hardware comfortably supports.
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
