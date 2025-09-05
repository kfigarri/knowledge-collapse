import os
import torch
import json
from datasets import load_dataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModelForCausalLM
from tqdm import tqdm

def get_real_data(tokenizer, block_size=64, mmlu_topic=None):
    if mmlu_topic:
        # Load MMLU data for the specified topic
        with open(os.path.join("training_data", f"wikitext_{mmlu_topic}.txt"), "r", encoding="utf-8") as f:
            texts = f.readlines()
            texts = [t for t in texts if len(t.strip()) > 50]
    else:
        # Load WikiText-2
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        texts = [t for t in dataset["train"]["text"] if len(t.strip()) > 50]
        
    # Tokenize and chunk
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    tokenized = list(map(tokenize_function, [{"text": t} for t in texts]))
    # Flatten tokens
    all_ids = [tid for ex in tokenized for tid in ex["input_ids"]]
    # Chunk into blocks
    blocks = [all_ids[i:i+block_size] for i in range(0, len(all_ids)-block_size+1, block_size)]
    return blocks

def save_texts_to_file(texts, file_path):
    with open(file_path+".txt", "w") as f:
        for line in texts:
            f.write(line.strip() + "\n")
    with open(file_path+".json", "w") as fp:
        json.dump(texts, fp)

def blockwise_generate_batched(model, tokenizer, real_blocks, gen_length=64, is_chat_model=False, batch_size=4):
    synthetic_outputs = []

    def prepare_input(block):
        prompt_text = tokenizer.decode(block, skip_special_tokens=True).strip()

        if is_chat_model:
            prompt_system = "You complete historical texts without any preamble or commentary. Only continue the input text naturally."
            messages = [
            {"role": "system", "content": [{"type": "text", "text":prompt_system}]},
            {"role": "user", "content": [{"type": "text", "text":prompt_text}]}
            ]
            full_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(model.device)
        else:
            input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)

        return input_ids.squeeze(0)  # remove batch dim

    # Prepare inputs for generation
    blocks_prepared = [prepare_input(block) for block in real_blocks]

    # Ensure pad_token_id is set (GPT-2 has None by default)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    # Ensure left padding for decoder-only models (e.g., GPT-2)
    if hasattr(tokenizer, 'padding_side'):
        tokenizer.padding_side = 'left'

    # Process in batches
    for i in tqdm(range(0, len(blocks_prepared), batch_size), desc="Generating batched"):
        batch = blocks_prepared[i:i+batch_size]
        batch_padded = pad_sequence(
            batch,
            batch_first=True,
            padding_value=pad_token_id
        )

        attention_mask = batch_padded.ne(pad_token_id).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=batch_padded.to(model.device),
                attention_mask=attention_mask,
                max_new_tokens=gen_length,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        for out, inp in zip(outputs, batch):
            generated_part = out[len(inp):]
            decoded = tokenizer.decode(generated_part, skip_special_tokens=True).strip()
            synthetic_outputs.append(decoded)

    return synthetic_outputs

def fine_tune(model, tokenizer, train_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(train_file+".txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Tokenize full training text
    tokenized = tokenizer(raw_text, return_tensors="pt", truncation=False)["input_ids"][0]

    # Break into fixed-length chunks (128 tokens)
    block_size = 512
    chunks = [tokenized[i:i+block_size] for i in range(0, len(tokenized) - block_size, block_size)]
    texts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

    if len(chunks) == 0:
        print(f"⚠️ Training skipped: only {len(tokenized)} tokens available.")
        return model

    dataset = Dataset.from_dict({"text": texts})

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, max_length=block_size)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    model.gradient_checkpointing_enable()

    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=0.5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        save_strategy="no",
        logging_steps=100,
        learning_rate=1e-5,  # Learning rate
        weight_decay=0.01,  # Weight decay
        report_to="none"
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized_dataset, data_collator=data_collator)
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved at {output_dir}")

    model = AutoModelForCausalLM.from_pretrained(output_dir, attn_implementation="eager", device_map=model.device)

    return model.eval()