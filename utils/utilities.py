from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import requests
import shutil
import os
import time
import json
from datetime import datetime
from huggingface_hub import login, HfApi, snapshot_download


load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
bot_token = "7539480564:AAGo4J-snrjQ8p45KoqVWqF_PJhp9A0zz6k"
chat_id = os.getenv("TELEGRAM_CHAT_ID")
login(token=hf_token)
hf_api = HfApi()

def send_telegram_message(message):
    if not bot_token or not chat_id:
        print("⚠️ Telegram credentials missing.")
        return
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}
        requests.post(url, data=payload)
    except Exception as e:
        print(f"⚠️ Telegram error: {e}")

def send_long_telegram_message(message, chunk_size=4000):
    for i in range(0, len(message), chunk_size):
        send_telegram_message(message[i:i+chunk_size])

def cleanup_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"🧹 Removed folder: {folder_path}")

def load_model_from_hub_or_pretrained(repo_id, base_dir, ratio, generation, model_name, device="cpu", unique_setting=False):
    subfolder = f"{base_dir}/ratio_{int(ratio * 100)}_gen_{generation}"

    current_dir = os.getcwd()
    local_cache_dir = os.path.join(current_dir, ".hf_model_cache")

    if generation >= 0:
        print(f"🔁 Downloading model files from HF Hub: {repo_id}/{subfolder}")
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=[
                f"{subfolder}/config.json",
                f"{subfolder}/generation_config.json", 
                f"{subfolder}/tokenizer_config.json",
                f"{subfolder}/special_tokens_map.json",
                f"{subfolder}/tokenizer.json",
                f"{subfolder}/tokenizer.model",
                f"{subfolder}/model.safetensors*",
                f"{subfolder}/chat_template.jinja"
            ],
            local_dir=local_cache_dir,
            force_download=True
        )

        local_subdir = os.path.join(local_dir, subfolder)
        model = AutoModelForCausalLM.from_pretrained(local_subdir, attn_implementation="eager", device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(local_subdir, use_fast=False)
        print(f"✅ Loaded model from local cache: {local_subdir}")

    else:
        print(f"⚠️ Falling back to model hub: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    if unique_setting:
        if tokenizer.pad_token is None:
            print("🔧 Adding pad_token to tokenizer (fallback to eos_token).")
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        # Set padding side to left for decoder-only models
        if tokenizer.padding_side is None:
            print("🔧 Setting padding side to left for tokenizer.")
            tokenizer.padding_side = "left"

    # Add fallback chat template if not present (especially for Gemma models)
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is None:
        print("🔧 Adding fallback chat template for Gemma.")
        # Default Gemma chat template
        tokenizer.chat_template = """{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"""

    return model.eval(), tokenizer

def find_latest_generation(base_dir, ratio):
    prefix = f"ratio_{int(ratio * 100)}_gen_"
    generations = []
    for folder in os.listdir(base_dir):
        if folder.startswith(prefix):
            try:
                generations.append(int(folder.replace(prefix, "")))
            except ValueError:
                continue
    return max(generations) if generations else -1

def run_with_retry(func, *args, retries=3, wait=60, **kwargs):
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"❌ Error: {e}. Retrying in {wait} seconds...")
            send_telegram_message(f"❌ Error in `{func.__name__}`: {e}. Retrying in {wait} sec...")
            time.sleep(wait)
    send_telegram_message(f"❌ Function `{func.__name__}` failed after {retries} retries.")
    raise RuntimeError(f"Function {func.__name__} failed after {retries} retries")

def upload_to_hub(model_dir, repo_id):
    try:
        # Upload selected top-level files
        mmlu_dir = os.path.join(model_dir, "mmlu")
        for filename in os.listdir(model_dir):
            if filename not in ["mmlu"]:
                file_path = os.path.join(model_dir, filename)
                hf_api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_path,
                    repo_id=repo_id,
                    repo_type="model"
                )

        # Upload all JSON logs inside ./mmlu/ folder
        mmlu_dir = os.path.join(model_dir, "mmlu")
        if os.path.exists(mmlu_dir):
            for mmlu_file in os.listdir(mmlu_dir):
                if mmlu_file.endswith(".json"):
                    full_path = os.path.join(mmlu_dir, mmlu_file)
                    hf_api.upload_file(
                        path_or_fileobj=full_path,
                        path_in_repo=full_path,
                        repo_id=repo_id,
                        repo_type="model"
                    )

        print(f"📤 Selected files (including mmlu/*.json) pushed to HF Hub: {repo_id}/{os.path.basename(model_dir)}")

    except Exception as e:
        print(f"⚠️ Failed to upload to hub: {e}")

def log_entropy_result(results, output_dir):
    # Append results to log_entropy_results.json with current time as key
    log_file = os.path.join(output_dir, "log_entropy_results.json")
    now_key = datetime.now().strftime("%Y%m%d%H%M")
    try:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                log_data = json.load(f)
        else:
            log_data = {}
    except Exception:
        log_data = {}
    log_data[now_key] = results
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)