import os
import re
import json
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import yaml

################################################
######## MMLU Evaluation Utilities #############
################################################

MMLU_SUBJECT_GROUPS = {
    "factual": ["global_facts", "world_religions", "high_school_geography"],
    "reasoning": ["formal_logic", "elementary_mathematics", "high_school_mathematics"], 
}

INSTRUCTIONS = {
    "multiple_choice": "The following are multiple choice questions (with answers) about {subject_name}.\n\n{question}",
    "short_answer": "Give a short answer to the following question about {subject_name}.\n\n{question}",
    "no_instruction": "{question}"
}

def load_fewshot_examples(subject_name):
    """Load few-shot examples from the corresponding mmlu_{subject_name}.yaml file."""
    harness_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(harness_dir, '..', 'lm-evaluation-harness', 'lm_eval', 'tasks', 'mmlu', 'flan_cot_fewshot', f'mmlu_{subject_name}.yaml')
    if not os.path.exists(yaml_path):
        return []
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    fewshot = data.get('fewshot_config', {}).get('samples', [])
    return fewshot

def extract_choice(question_text: str, option: str) -> str:
    # Match all options using regex with capturing groups
    pattern = r"\((A)\)\s(.*?)\s(?=\(B\)|$)|" \
              r"\((B)\)\s(.*?)\s(?=\(C\)|$)|" \
              r"\((C)\)\s(.*?)\s(?=\(D\)|$)|" \
              r"\((D)\)\s(.*)"
    
    matches = re.findall(pattern, question_text, re.DOTALL)

    # Convert matches into a dict: {'A': ..., 'B': ..., ...}
    option_map = {}
    for match in matches:
        if match[0]: option_map["A"] = match[1].strip()
        if match[2]: option_map["B"] = match[3].strip()
        if match[4]: option_map["C"] = match[5].strip()
        if match[6]: option_map["D"] = match[7].strip()

    return option_map.get(option.upper(), "")

def format_fewshot_block(sample):
    # Format a single few-shot example block as in the YAML, but only show the answer letter (no brackets)
    q = sample['question'].strip()
    # Extract the answer letter from the target string (e.g., 'The answer is (A).')
    match = re.search(r'\(\s*([A-D])\s*\)', sample['target'])
    answer_letter = match.group(1) if match else ""
    answer_text = extract_choice(q, answer_letter)
    return f"Q: {q}\nAnswer: {answer_letter}. {answer_text}"

def get_fewshot_blocks(fewshot_samples, num_shots):
    # Return a list of formatted few-shot blocks up to num_shots
    return [format_fewshot_block(s) for s in fewshot_samples[:num_shots]]

def format_question_with_choices(example, short_answer=False):
    letters = ['A', 'B', 'C', 'D']
    question = f"{example['question']}\n"
    if short_answer:
        question += " / ".join([f"{choice}" for i, choice in enumerate(example['choices'])]) + "\n"
    else:
        for i, choice in enumerate(example['choices']):
            question += f"{letters[i]}. {choice}\n"
    question += "Answer:"
    return question

def prettify_subject_name(subject_name):
    return subject_name.replace('_', ' ').lower()

def generate_prompt(example, instruction_type, tokenizer=None, is_chat_model=False, subject_name=None, num_fewshot=None):
    pretty_subject = prettify_subject_name(subject_name) if subject_name else ""
    if instruction_type == "multiple_choice_few_shot" and subject_name is not None:
        fewshot_samples = load_fewshot_examples(subject_name)
        if num_fewshot is not None:
            fewshot_blocks = get_fewshot_blocks(fewshot_samples, num_fewshot)
        else:
            fewshot_blocks = [format_fewshot_block(s) for s in fewshot_samples]
        prompt = f"The following are multiple choice questions (with answers) about {pretty_subject}.\n\n"
        prompt += "\n\n".join(fewshot_blocks)
        prompt += f"\n\nQ: {example['question']}\n"
        for i, choice in enumerate(example['choices']):
            prompt += f"({chr(65+i)}) {choice}  "
        prompt = prompt.strip() + "\n"
        prompt += "Answer:"
    else:
        if instruction_type == "short_answer":
            question_block = format_question_with_choices(example, short_answer=True)
        else:
            question_block = format_question_with_choices(example)
        prompt_template = INSTRUCTIONS[instruction_type]
        prompt = prompt_template.format(question=question_block, subject_name=pretty_subject)

    if is_chat_model and tokenizer is not None:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        return prompt

def evaluate_single_question(model, tokenizer, prompt, letters, example, instruction_type="multiple_choice"):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    input_len = input_ids.shape[1]

    option_probs = []
    option_loglikelihoods = []
    option_isgreedy = []

    if instruction_type == "short_answer" and example is not None:
        for ans_text in example["choices"]:
            answer_ids = tokenizer(ans_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            if answer_ids.shape[1] < 1:
                option_probs.append(0.0)
                option_loglikelihoods.append(float('-inf'))
                option_isgreedy.append(False)
                continue
            combined_ids = torch.cat([input_ids, answer_ids], dim=1)
            attention_mask = torch.ones_like(combined_ids)
            with torch.no_grad():
                outputs = model(input_ids=combined_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, input_len - 1 : input_len + answer_ids.shape[1] - 1, :]
                probs = F.softmax(logits, dim=-1)
                prob_product = 1.0
                is_greedy = True
                for i in range(answer_ids.shape[1]):
                    token_id = answer_ids[0, i].item()
                    prob_product *= probs[0, i, token_id].item()
                    if probs[0, i].argmax().item() != token_id:
                        is_greedy = False
                avg_prob = prob_product ** (1.0 / answer_ids.shape[1])
                option_loglikelihoods.append(avg_prob)
                option_isgreedy.append(is_greedy)
                option_probs.append(avg_prob)
        predicted_index = int(torch.tensor(option_probs).argmax())
        return option_probs, option_loglikelihoods, option_isgreedy, predicted_index
    else:
        for i, letter in enumerate(letters):
            choice_text = f"{letter}.{example['choices'][i]}"
            choice_ids = tokenizer(choice_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            if choice_ids.shape[1] < 1:
                option_probs.append(0.0)
                option_loglikelihoods.append(float('-inf'))
                option_isgreedy.append(False)
                continue

            # For first token probability, keep using the letter as before
            letter_ids = tokenizer(letter, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            scored_token = letter_ids[0][0].item()
            combined_ids = torch.cat([input_ids, letter_ids], dim=1)
            attention_mask = torch.ones_like(combined_ids)
            with torch.no_grad():
                outputs = model(input_ids=combined_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, input_len - 1, :]
                probs = F.softmax(logits, dim=-1)
                token_prob = probs[0, scored_token].item()
                option_probs.append(token_prob)
                # Calculate is_greedy based only on the letter token
                is_greedy = probs[0].argmax().item() == scored_token
                option_isgreedy.append(is_greedy)

            # Now compute loglikelihood for the full option text (with letter)
            combined_ids_full = torch.cat([input_ids, choice_ids], dim=1)
            attention_mask_full = torch.ones_like(combined_ids_full)
            with torch.no_grad():
                outputs_full = model(input_ids=combined_ids_full, attention_mask=attention_mask_full)
                logits_full = outputs_full.logits[:, input_len - 1 : input_len + choice_ids.shape[1] - 1, :]
                probs_full = F.softmax(logits_full, dim=-1)
                prob_product = 1.0
                for j in range(choice_ids.shape[1]):
                    token_id = choice_ids[0, j].item()
                    prob_product *= probs_full[0, j, token_id].item()
                avg_prob = prob_product ** (1.0 / choice_ids.shape[1])
                option_loglikelihoods.append(avg_prob)
                
        predicted_index = int(torch.tensor(option_probs).argmax())
        return option_probs, option_loglikelihoods, option_isgreedy, predicted_index

def evaluate_mmlu_model(model, tokenizer, subject_name, generation, log_dir, instruction_type="multiple_choice", sample_limit=100, is_chat_model=False, num_fewshot=5):
    model.eval()
    letters = ['A', 'B', 'C', 'D']
    correct = 0
    total = 0
    greedy_correct = 0
    log_data = []

    ds = load_dataset("tasksource/mmlu", subject_name)["test"]
    if sample_limit:
        ds = ds.select(range(sample_limit))

    for ex in tqdm(ds, desc=f"Evaluating {subject_name} ({instruction_type})"):
        prompt = generate_prompt(ex, instruction_type, tokenizer=tokenizer, is_chat_model=is_chat_model, subject_name=subject_name, num_fewshot=num_fewshot)
        option_probs, option_loglikelihoods, option_isgreedy, predicted_index = evaluate_single_question(
            model, tokenizer, prompt, letters, example=ex, instruction_type=instruction_type
        )

        is_correct = predicted_index == ex["answer"]
        if is_correct:
            correct += 1
            if option_isgreedy[predicted_index]:
                greedy_correct += 1
        total += 1

        with torch.inference_mode():
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            attention_mask = torch.ones_like(input_ids)
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            full_output = tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        log_data.append({
            "generation": generation,
            "subject": subject_name,
            "question": ex["question"],
            "choices": ex["choices"],
            "correct_answer_index": ex["answer"],
            "probabilities": option_probs,
            "loglikelihoods": option_loglikelihoods,
            "is_greedy": option_isgreedy,
            "predicted_index": predicted_index,
            "is_correct": is_correct,
            "actual_answer": full_output,
        })

    os.makedirs(os.path.join(log_dir, "mmlu"), exist_ok=True)
    log_path = os.path.join(log_dir, "mmlu", f"{subject_name}_{instruction_type}_gen{generation}.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    accuracy = correct / total if total > 0 else 0.0
    greedy_prop = greedy_correct / correct if correct > 0 else 0.0
    return accuracy, greedy_prop

def evaluate_mmlu_by_subject(model, tokenizer, generation, log_dir, instruction_types=["multiple_choice", "multiple_choice_few_shot", "short_answer", "no_instruction"], is_chat_model=False, num_fewshot=3):
    subject_scores = {}
    greedy_props = {}
    MMLU_SUBJECTS = [subject for subjects in MMLU_SUBJECT_GROUPS.values() for subject in subjects]
    for instruction_type in instruction_types:
        subject_scores[instruction_type] = {}
        greedy_props[instruction_type] = {}
        for subject in MMLU_SUBJECTS:
            print(f"Evaluating on MMLU subject: {subject} at generation {generation} ({instruction_type})")
            accuracy, greedy_prop = evaluate_mmlu_model(
                model, tokenizer, subject, generation, log_dir,
                instruction_type=instruction_type, sample_limit=100, is_chat_model=is_chat_model, num_fewshot=num_fewshot
            )
            subject_scores[instruction_type][subject] = accuracy
            greedy_props[instruction_type][subject] = greedy_prop
    return subject_scores, greedy_props

######################################################
######## Train Text Evaluation Utilities #############
######################################################
import numpy as np
from collections import Counter
import math
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from .model_configuration import DEVICE

# Load Gibberish Detector
gibberish_tokenizer = AutoTokenizer.from_pretrained("madhurjindal/autonlp-Gibberish-Detector-492513457")
gibberish_model = AutoModelForSequenceClassification.from_pretrained("madhurjindal/autonlp-Gibberish-Detector-492513457").eval().to(DEVICE)

def shannon_index(texts):
    entropies = []

    for text in texts:
        token_counts = Counter(text.split())  # whitespace-based tokens
        total = sum(token_counts.values())
        if total == 0:
            continue
        entropy = -sum((count / total) * math.log(count / total) for count in token_counts.values())
        entropies.append(entropy)

    if not entropies:
        return 0.0, 0.0
    mean_entropy = sum(entropies) / len(entropies)
    std_entropy = np.std(entropies)
    return mean_entropy, float(std_entropy)

def compute_gibberish_score(texts):
    label_scores = {"noise": 0, "word salad": 1, "mild gibberish": 2, "clean": 3}
    scores = []

    for text in texts:
        inputs = gibberish_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            logits = gibberish_model(**inputs).logits
            pred = logits.argmax(dim=-1).item()
        scores.append(label_scores[gibberish_model.config.id2label[pred]])

    if not scores:
        return 0.0, 0.0
    return sum(scores) / len(scores), float(np.std(scores))

def get_wikitext2_valtest_texts():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    texts = []
    for split in ["validation", "test"]:
        texts.extend([t for t in dataset[split]["text"] if t.strip()])
    return texts

def compute_perplexity(text: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, stride: int = 512, max_length: int = 1024, num_chunks: int = 5, overlap: int = 20):
    """
    https://huggingface.co/docs/transformers/perplexity
    Compute autoregressive perplexity on a long text split into multiple overlapping chunks.
    
    Args:
        text (str): The input long text.
        model (PreTrainedModel): HuggingFace model.
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.
        stride (int): Stride length for autoregressive evaluation.
        max_length (int): Context window for the model.
        num_chunks (int): Number of chunks to divide the input text into.
        overlap (int): Number of overlapping tokens between chunks.

    Returns:
        Tuple[float, float]: Mean and std of perplexity across chunks.
    """
    model.eval()
    device = model.device
    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings.input_ids[0]

    total_len = input_ids.size(0)
    chunk_size = (total_len + (num_chunks - 1) * overlap) // num_chunks  # distribute with overlap
    all_ppls = []

    print(f"Total length of input text: {total_len}")
    for i in range(num_chunks):
        start = max(0, i * (chunk_size - overlap))
        end = min(total_len, start + chunk_size)
        chunk_ids = input_ids[start:end].unsqueeze(0).to(device)
        seq_len = chunk_ids.size(1)

        nll_total = 0.0
        token_total = 0
        prev_end_loc = 0

        for begin_loc in tqdm(range(0, seq_len, stride), desc="Processing chunks"):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_chunk = chunk_ids[:, begin_loc:end_loc]
            target_chunk = input_chunk.clone()
            target_chunk[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_chunk, labels=target_chunk)
                loss = outputs.loss

            num_valid_tokens = (target_chunk != -100).sum().item()
            num_loss_tokens = max(0, num_valid_tokens - input_chunk.size(0))  # due to shifting
            nll_total += loss.item() * num_loss_tokens
            token_total += num_loss_tokens
            prev_end_loc = end_loc

            if end_loc == seq_len:
                break

        if token_total > 0:
            avg_nll = nll_total / token_total
            ppl = math.exp(avg_nll)
            all_ppls.append(ppl)

    if not all_ppls:
        return float("inf"), float("inf")
    
    return float(np.mean(all_ppls)), float(np.std(all_ppls))
