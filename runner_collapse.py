import os
import json
import random

from utils.model_configuration import model_mapping, DEVICE, SYNTHETIC_RATIOS, GENERATIONS, REPO_ID, SAMPLES_PER_GENERATION
from utils.finetuning import get_real_data, save_texts_to_file, blockwise_generate_batched, fine_tune
from utils.evaluation import shannon_index, compute_gibberish_score, compute_perplexity, evaluate_mmlu_by_subject, get_wikitext2_valtest_texts
from utils.utilities import send_telegram_message, cleanup_folder, load_model_from_hub_or_pretrained, upload_to_hub, find_latest_generation, run_with_retry, log_entropy_result

# ========== Main Training Loop ==========
def main(mmlu_topic=None, reset_generation=False, model_name="gemma31b", start_generation=0):
    results = {}
    mmlu_results = {}
    valtest_texts = get_wikitext2_valtest_texts()
    custom_cache_dir = os.path.join(os.getcwd(), ".hf_model_cache")
    MODEL_NAME = model_mapping[model_name]["model_name"]
    OUTPUT_DIR = model_mapping[model_name]["output_dir"]
    is_chat_model = model_mapping[model_name]["is_chat_model"]
    unique_setting = model_mapping[model_name]["unique_setting"]

    for ratio in SYNTHETIC_RATIOS:
        print(f"\n=== Running for synthetic data ratio: {int(ratio * 100)}% ===")
        if start_generation > 0:
            latest_gen = start_generation - 1
            start_gen = start_generation
        else:
            latest_gen = -1 if reset_generation else find_latest_generation(OUTPUT_DIR, ratio)
            start_gen = latest_gen + 1

        current_model, tokenizer = load_model_from_hub_or_pretrained(
            repo_id=REPO_ID,
            base_dir=OUTPUT_DIR,
            ratio=ratio,
            generation=latest_gen,
            model_name=MODEL_NAME,
            device=DEVICE,
            unique_setting=unique_setting
        )

        entropy_scores = []
        gibberish_scores = []
        perplexity_scores = []
        text_perplexity_scores = []
        mmlu_scores_by_group = {}
        all_token_blocks = get_real_data(tokenizer, mmlu_topic=mmlu_topic)

        for gen in range(start_gen, GENERATIONS):
            gen_dir = os.path.join(OUTPUT_DIR, f"ratio_{int(ratio*100)}_gen_{gen}")
            os.makedirs(gen_dir, exist_ok=True)
            send_telegram_message(f"🚀 Starting Gen {gen} with {int(ratio*100)}% synthetic data using {model_name}...")

            try:
                # Step 1: Generate synthetic data
                # Ensure you have enough total blocks
                print(f"Total blocks available: {len(all_token_blocks)}")
                assert len(all_token_blocks) >= SAMPLES_PER_GENERATION, "Not enough blocks to sample from"
                # SAMPLES_PER_GENERATION = len(all_token_blocks)  # Adjust as needed

                # Step 1: Determine number of blocks for real and synthetic parts
                n_real = int((1 - ratio) * SAMPLES_PER_GENERATION)
                n_synth = SAMPLES_PER_GENERATION - n_real

                # Step 2: Sample real and synthetic blocks without replacement
                # sampled_blocks = random.sample(all_token_blocks, n_real + n_synth)
                flat_real_blocks = all_token_blocks[:n_real]
                flat_synth_blocks = all_token_blocks[n_real:n_real + n_synth]

                ## Decode original tokens for real part
                current_real = [tokenizer.decode(block, skip_special_tokens=True).strip() for block in flat_real_blocks]

                ## Generate synthetic continuation from sampled blocks
                synthetic_data = blockwise_generate_batched(current_model, tokenizer, flat_synth_blocks, is_chat_model=is_chat_model) if flat_synth_blocks else []

                ## Combine and write
                combined_data = current_real + synthetic_data
                train_file = os.path.join(gen_dir, "train")
                save_texts_to_file(combined_data, train_file)

                # Step 2: Evaluate generated data
                ## Entropy
                entropy_mean, entropy_std = shannon_index(combined_data)
                entropy_scores.append((entropy_mean, entropy_std))
                ## Gibberish Score
                gibberish_mean, gibberish_std = compute_gibberish_score(combined_data)
                gibberish_scores.append((gibberish_mean, gibberish_std))
                ## Perplexity
                long_valtest_texts = " ".join(valtest_texts)
                ppl_mean, ppl_std = compute_perplexity(long_valtest_texts, current_model, tokenizer)  # uses external frozen GPT-2
                perplexity_scores.append((ppl_mean, ppl_std))
                ## Text perplexity
                long_combined_data = " ".join(combined_data)
                text_ppl_mean, text_ppl_std = compute_perplexity(long_combined_data, current_model, tokenizer)  # uses external frozen GPT-2
                text_perplexity_scores.append((text_ppl_mean, text_ppl_std))

                # Step 3: MMLU Evaluation
                mmlu_group_scores, mmlu_group_greedy  = evaluate_mmlu_by_subject(current_model, tokenizer, gen, gen_dir, is_chat_model=is_chat_model)
                mmlu_scores_by_group[gen] = (mmlu_group_scores, mmlu_group_greedy)
                list_mmlu_group_score = []
                for group, scores in mmlu_group_scores.items():
                    first_subject = list(scores.keys())[0]
                    list_mmlu_group_score.append(f"{group.replace('_', ' ')}: {scores[first_subject]:.4f}")
                mmlu_group_scores_print = f"📝 {first_subject.replace('_', ' ')}\n"+"\n".join(list_mmlu_group_score)

                results[f"{int(ratio * 100)}%"] = [
                    {
                        "generation": i,
                        "entropy_mean": entropy_scores[i][0],
                        "entropy_std": entropy_scores[i][1],
                        "gibberish_score_mean": gibberish_scores[i][0],
                        "gibberish_score_std": gibberish_scores[i][1],
                        "perplexity_mean": perplexity_scores[i][0],
                        "perplexity_std": perplexity_scores[i][1],
                        "text_perplexity_mean": text_perplexity_scores[i][0],
                        "text_perplexity_std": text_perplexity_scores[i][1],
                    }
                    for i in range(len(entropy_scores))
                ]
                mmlu_results[f"{int(ratio * 100)}%"] = mmlu_scores_by_group
                
                # Final save
                with open(os.path.join(OUTPUT_DIR, "entropy_results.json"), "w") as f:
                    json.dump(results, f, indent=2)

                with open(os.path.join(OUTPUT_DIR,"mmlu_results.json"), "w") as f:
                    json.dump(mmlu_results, f, indent=2)

                # Step 4: Fine-tune
                current_model = run_with_retry(fine_tune, current_model, tokenizer, train_file, gen_dir)
                upload_to_hub(gen_dir, REPO_ID)
                cleanup_folder(gen_dir)

                send_telegram_message(f"✅ Gen {gen} complete. Entropy: {entropy_mean:.2f}, Gibberish Score: {gibberish_mean:.2f}, Perplexity: {ppl_mean:.2f}")
                send_telegram_message(mmlu_group_scores_print)

                cleanup_folder(os.path.join(custom_cache_dir, gen_dir))

            except Exception as e:
                send_telegram_message(f"❌ Gen {gen} failed: {str(e)}")
                #cleanup_folder(gen_dir)
                raise
    
    log_entropy_result(results, OUTPUT_DIR)
    send_telegram_message("🎉 All experiments finished successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run model collapse experiment.")
    parser.add_argument("-mt", "--mmlu_topic", type=str, help="MMLU topic to use (e.g., world_religions)")
    parser.add_argument("-rg", "--reset_generation", action="store_true", help="Start from scratch, ignoring previous generations")
    parser.add_argument("-mn", "--model_name", type=str, help="Model name to use (e.g., gemma31b, gpt2)", default="gemma31b")
    parser.add_argument("-sg", "--start_generation", type=int, help="Start from nth generation", default=0)
    args = parser.parse_args()

    main(mmlu_topic=args.mmlu_topic, reset_generation=args.reset_generation, model_name=args.model_name, start_generation=args.start_generation)