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
GENERATION_TOKENS = 2048
BUFFER_TOKENS = 32

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

default_model_1 = "meta-llama/Llama-3.2-3B-Instruct"
default_model_2 = "meta-llama/Llama-3.2-3B-Instruct"

arg1 = sys.argv[1] if len(sys.argv) >= 2 else default_model_1
arg2 = sys.argv[2] if len(sys.argv) >= 3 else default_model_2

gpu_count_1, model_name1 = parse_model_arg(arg1)
gpu_count_2, model_name2 = parse_model_arg(arg2)

if gpu_count_1 + gpu_count_2 > total_gpus_available:
    raise ValueError(
        f"Requested {gpu_count_1 + gpu_count_2} GPUs but only {total_gpus_available} are available."
    )

###############################################################################
# Tokenizers
###############################################################################

FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}"
    "{% elif message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}"
)

tokenizer1 = AutoTokenizer.from_pretrained(model_name1)
tokenizer2 = AutoTokenizer.from_pretrained(model_name2)

if not getattr(tokenizer1, "chat_template", None):
    print("⚠️  Tokenizer 1 has no chat_template – using a generic fallback.")
    tokenizer1.chat_template = FALLBACK_CHAT_TEMPLATE
if not getattr(tokenizer2, "chat_template", None):
    print("⚠️  Tokenizer 2 has no chat_template – using a generic fallback.")
    tokenizer2.chat_template = FALLBACK_CHAT_TEMPLATE

###############################################################################
# GPU Assignment & Environment Setup
###############################################################################

gpu_ids_1 = list(range(gpu_count_1))
start_gpu_2 = gpu_count_1
gpu_ids_2 = list(range(start_gpu_2, start_gpu_2 + gpu_count_2))

all_gpu_ids = ",".join(str(i) for i in range(gpu_count_1 + gpu_count_2))
os.environ["CUDA_VISIBLE_DEVICES"] = all_gpu_ids

env_gpus_1 = ",".join(str(g) for g in gpu_ids_1)
env_gpus_2 = ",".join(str(g) for g in gpu_ids_2)

###############################################################################
# Launch vLLM Servers (sub-processes)
###############################################################################

print("Launching vllm servers")

server1_cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", model_name1,
    "--port", "8001",
    "--tensor-parallel-size", str(gpu_count_1),
]
server2_cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", model_name2,
    "--port", "8002",
    "--tensor-parallel-size", str(gpu_count_2),
]

env1 = os.environ.copy()
env2 = os.environ.copy()

# Remove any inherited ROCm device restrictions so only
# CUDA_VISIBLE_DEVICES controls GPU visibility per server.
# vLLM 0.12+ asserts ROCR and CUDA device counts match;
# leaving ROCR/HIP unset avoids this mismatch.
for var in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
    env1.pop(var, None)
    env2.pop(var, None)

env1["CUDA_VISIBLE_DEVICES"] = env_gpus_1
env1["VLLM_HOST_IP"] = "127.0.0.1"
env1["VLLM_PORT"] = "29600"
env1["TRITON_CACHE_DIR"] = "/tmp/triton_cache_server1"
env1["XDG_CACHE_HOME"] = "/tmp/xdg_cache_server1"
env1["TORCH_EXTENSIONS_DIR"] = "/dev/shm/torch_ext_server1"

env2["CUDA_VISIBLE_DEVICES"] = env_gpus_2
env2["VLLM_HOST_IP"] = "127.0.0.1"
env2["VLLM_PORT"] = "29610"
env2["TRITON_CACHE_DIR"] = "/tmp/triton_cache_server2"
env2["XDG_CACHE_HOME"] = "/tmp/xdg_cache_server2"
env2["TORCH_EXTENSIONS_DIR"] = "/dev/shm/torch_ext_server2"

proc1 = subprocess.Popen(server1_cmd, env=env1)
proc2 = subprocess.Popen(server2_cmd, env=env2)
print("popen returned")

def cleanup():
    proc1.terminate()
    proc2.terminate()

atexit.register(cleanup)
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
wait_for_health(8002)
print("Health checks passed, continuing..")

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

def format_chat_prompt(msgs: List[Dict[str, str]], tokenizer: AutoTokenizer) -> str:
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def count_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    return len(tokenizer.encode(text))

def truncate_history(msgs: List[Dict[str, str]], tokenizer: AutoTokenizer) -> List[Dict[str, str]]:
    budget = MAX_MODEL_LENGTH - (GENERATION_TOKENS + BUFFER_TOKENS)
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

def respond(user_message, chat_history1, chat_history2):
    chat_history1 = chat_history1 or []
    chat_history2 = chat_history2 or []

    chat_history1 = chat_history1 + [{"role": "user", "content": user_message}, {"role": "assistant", "content": ""}]
    chat_history2 = chat_history2 + [{"role": "user", "content": user_message}, {"role": "assistant", "content": ""}]

    chat_history1 = truncate_history(chat_history1, tokenizer1)
    chat_history2 = truncate_history(chat_history2, tokenizer2)

    payload1 = {
        "model": model_name1,
        "prompt": format_chat_prompt(chat_history1, tokenizer1),
        "max_tokens": GENERATION_TOKENS,
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "seed": random.randint(0, 2**32),
    }
    payload2 = {
        "model": model_name2,
        "prompt": format_chat_prompt(chat_history2, tokenizer2),
        "max_tokens": GENERATION_TOKENS,
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "seed": random.randint(0, 2**32),
    }

    url1 = "http://localhost:8001/v1/completions"
    url2 = "http://localhost:8002/v1/completions"

    buffer1, lock1, done1 = [""], threading.Lock(), [False]
    buffer2, lock2, done2 = [""], threading.Lock(), [False]

    threading.Thread(target=stream_from_server, args=(url1, payload1, buffer1, lock1, done1), daemon=True).start()
    threading.Thread(target=stream_from_server, args=(url2, payload2, buffer2, lock2, done2), daemon=True).start()

    while not (done1[0] and done2[0]):
        with lock1:
            cur1 = buffer1[0]
        with lock2:
            cur2 = buffer2[0]

        updated = False
        if chat_history1[-1]["content"] != cur1:
            chat_history1[-1]["content"] = cur1
            updated = True
        if chat_history2[-1]["content"] != cur2:
            chat_history2[-1]["content"] = cur2
            updated = True

        if updated:
            yield chat_history1, chat_history2, chat_history1, chat_history2
        time.sleep(0.1)

    chat_history1[-1]["content"] = buffer1[0]
    chat_history2[-1]["content"] = buffer2[0]
    yield chat_history1, chat_history2, chat_history1, chat_history2

###############################################################################
# Helper ops – delete / regenerate
###############################################################################

def delete_last(hist1, hist2):
    h1 = hist1[:-1] if hist1 else hist1
    h2 = hist2[:-1] if hist2 else hist2
    return h1, h2, h1, h2

def regenerate(hist1, hist2):
    if not hist1 or not hist2:
        yield hist1, hist2, hist1, hist2
        return
    # Remove last assistant messages from both
    if hist1[-1]["role"] == "assistant" and len(hist1) >= 2 and hist1[-2]["role"] == "user":
        user_msg = hist1[-2]["content"]
        hist1 = hist1[:-2]
        hist2 = hist2[:-2]
    elif hist1[-1]["role"] == "user":
        user_msg = hist1[-1]["content"]
        hist1 = hist1[:-1]
        hist2 = hist2[:-1]
    else:
        yield hist1, hist2, hist1, hist2
        return
    yield from respond(user_msg, hist1, hist2)

###############################################################################
# Gradio UI
###############################################################################

with gr.Blocks() as demo:
    gr.Markdown("## Dual‑Model Chat Interface (vLLM Backend)")
    gr.Markdown(
        f"**Model 1**: {model_name1} ({gpu_count_1} GPU(s)) &nbsp;&nbsp;"
        f"**Model 2**: {model_name2} ({gpu_count_2} GPU(s))"
    )

    with gr.Row():
        chatbot1 = gr.Chatbot(label=model_name1, height=600)
        chatbot2 = gr.Chatbot(label=model_name2, height=600)

    state1 = gr.State([])
    state2 = gr.State([])

    with gr.Row():
        regen_btn = gr.Button("🔄 Regenerate last", variant="secondary")
        delete_btn = gr.Button("🗑️ Delete last", variant="secondary")
        clear_btn = gr.ClearButton(value="Clear", components=[chatbot1, chatbot2, state1, state2])

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Enter your message and press Enter", scale=8)
        send_btn = gr.Button("Send", variant="primary", scale=1, elem_id="send-btn")

    txt.submit(respond, [txt, state1, state2], [chatbot1, chatbot2, state1, state2])
    send_btn.click(respond, [txt, state1, state2], [chatbot1, chatbot2, state1, state2])

    txt.submit(lambda _: gr.update(value=""), txt, txt)
    send_btn.click(lambda _: gr.update(value=""), txt, txt)

    regen_btn.click(regenerate, [state1, state2], [chatbot1, chatbot2, state1, state2])
    delete_btn.click(delete_last, [state1, state2], [chatbot1, chatbot2, state1, state2])

if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()

    print("Launching dual chat interface...")
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
