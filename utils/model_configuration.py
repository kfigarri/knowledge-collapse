# Constant variables
import torch
import os
from dotenv import load_dotenv

load_dotenv()
REPO_ID = os.getenv("HUGGINGFACE_REPO_ID")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("use device:", DEVICE)
SYNTHETIC_RATIOS = [0.5]
SAMPLES_PER_GENERATION = 8000
GENERATIONS = 16

# Model configuration mapping for different models
model_mapping = {
    "gemma31b": {
        "output_dir":"gemma31b",
        "model_name":"google/gemma-3-1b-it",
        "is_chat_model":True,
        "unique_setting": False
    },
    "gemma22b": {
        "output_dir":"gemma22b",
        "model_name":"google/gemma-2-2b-it",
        "is_chat_model":True,
        "unique_setting": True
    },
    "gpt2": {
        "output_dir":"gpt2",
        "model_name":"openai-community/gpt2",
        "is_chat_model":False,
        "unique_setting": False
    },
    "gpt2-medium": {
        "output_dir":"gpt2-medium",
        "model_name":"openai-community/gpt2-medium",
        "is_chat_model":False,
        "unique_setting": False
    },
    "gpt2-large": {
        "output_dir":"gpt2-large",
        "model_name":"openai-community/gpt2-large",
        "is_chat_model":False,
        "unique_setting": False
    },
    "gpt2-xl": {
        "output_dir":"gpt2-xl",
        "model_name":"openai-community/gpt2-xl",
        "is_chat_model":False,
        "unique_setting": False
    },
     "phi15": {
        "output_dir":"phi15",
        "model_name":"microsoft/phi-1_5",
        "is_chat_model":False,
        "unique_setting": False
    },
    "smollm17": {
        "output_dir":"smollm17",
        "model_name":"HuggingFaceTB/SmolLM-1.7B",
        "is_chat_model":False,
        "unique_setting": False
    },
}