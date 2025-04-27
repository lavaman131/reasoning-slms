from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from typing import Optional, Any, List, Dict, Iterable
import re
from trl import GRPOConfig, GRPOTrainer, RewardConfig
from trl.trainer.grpo_trainer import RewardFunc
from functools import partial, update_wrapper
import os

import torch

# For peft
from peft import LoraConfig


def extract_hash_answer(text: str) -> Optional[str]:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def match_format_exactly(
    completions: List[List[Dict[str, Any]]], **kwargs: Any
) -> List[float]:
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if kwargs["match_format"].search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores


def match_format_approximately(
    completions: List[List[Dict[str, Any]]], **kwargs: Any
) -> List[float]:
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!
        score += 0.5 if response.count(kwargs["reasoning_start"]) == 1 else -0.5
        score += 0.5 if response.count(kwargs["reasoning_end"]) == 1 else -0.5
        score += 0.5 if response.count(kwargs["solution_start"]) == 1 else -0.5
        score += 0.5 if response.count(kwargs["solution_end"]) == 1 else -0.5
        scores.append(score)
    return scores


def check_answer(
    prompts: List[List[Dict[str, Any]]],
    completions: List[List[Dict[str, Any]]],
    answer: List[str],
    **kwargs: Any,
) -> List[float]:
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := kwargs["match_format"].search(r)) is not None
        else None
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        # Correct answer gets 3 points!
        if guess == true_answer:
            score += 3.0
        # Match if spaces are seen
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess) / float(true_answer)
                if ratio >= 0.9 and ratio <= 1.1:
                    score += 0.5
                elif ratio >= 0.8 and ratio <= 1.2:
                    score += 0.25
                else:
                    score -= 1.0  # Penalize wrong answers
            except Exception:
                score -= 0.5  # Penalize
        scores.append(score)
    return scores


def check_numbers(
    prompts: List[List[Dict[str, Any]]],
    completions: List[List[Dict[str, Any]]],
    answer: List[str],
    **kwargs: Any,
) -> List[float]:
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := kwargs["match_numbers"].search(r)) is not None
        else None
        for r in responses
    ]

    scores = []
    print(
        "*" * 20,
        f"Question:\n{question}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{extracted_responses[0]}",
    )
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            guess = float(guess.strip())
            scores.append(1.5 if guess == true_answer else 0.0)
        except Exception:
            scores.append(0)
            continue
    return scores


def main() -> None:
    reasoning_start = "<start_working_out>"
    reasoning_end = "<end_working_out>"
    solution_start = "<SOLUTION>"
    solution_end = "</SOLUTION>"
    match_format = re.compile(
        rf"^[\s]{{0,}}"
        rf"{reasoning_start}.+?{reasoning_end}.*?"
        rf"{solution_start}(.+?){solution_end}"
        rf"[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL,
    )

    system_prompt = f"""You are given a problem.
    Think about the problem and provide your working out.
    Place it between {reasoning_start} and {reasoning_end}.
    Then, provide your solution between {solution_start}{solution_end}"""

    match_numbers = re.compile(
        rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
    )

    train_dataset = load_dataset("openai/gsm8k", "main", split="train")

    train_dataset = train_dataset.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )

    max_seq_length = 1024
    max_prompt_length = 256
    experiment_name = "gemma-3-1b-it-grpo-baseline"
    project_name = "dapo"

    os.environ["WANDB_PROJECT"] = project_name

    training_args = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        logging_steps=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,  # Increase to 4 for smoother training
        num_generations=4,  # Decrease if out of memory
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps=50,
        save_steps=50,
        max_grad_norm=0.1,
        bf16=True,
        report_to="wandb",  # Can use Weights & Biases
        run_name=experiment_name,
        output_dir="outputs",
        seed=3407
    )

    model_id = "google/gemma-3-1b-it"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="eager",
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
        
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=True
    )

    lora_config = LoraConfig(
        r = 8,           # Larger = higher accuracy, but might overfit
        lora_alpha = 8,  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        peft_config = lora_config,
        reward_funcs=[
            update_wrapper(
                partial(match_format_exactly, match_format=match_format),
                match_format_exactly,
            ),
            update_wrapper(
                partial(
                    match_format_approximately,
                    reasoning_start=reasoning_start,
                    reasoning_end=reasoning_end,
                    solution_start=solution_start,
                    solution_end=solution_end,
                ),
                match_format_approximately,
            ),
            update_wrapper(
                partial(check_answer, match_format=match_format), check_answer
            ),
            update_wrapper(
                partial(check_numbers, match_numbers=match_numbers), check_numbers
            ),
        ],
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    model.save_pretrained(experiment_name)  # Local saving
    tokenizer.save_pretrained(experiment_name)  # Local saving


if __name__ == "__main__":
    main()
