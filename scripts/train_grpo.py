from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from typing import Optional, Any, List, Dict, Iterable
import re
from trl import GRPOConfig, GRPOTrainer, RewardConfig, TrlParser
from functools import partial, update_wrapper
import os
import torch
from peft import LoraConfig, TaskType, get_peft_model
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelArguments:
    model_id: str
    attn_implementation: str = "eager"
    torch_dtype: str = "auto"


@dataclass
class DatasetArguments:
    dataset_name: str


@dataclass
class ExperimentArguments:
    experiment_name: str
    wandb_project: str
    wandb_entity: str


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
    parser = TrlParser(
        (ModelArguments, GRPOConfig, DatasetArguments, ExperimentArguments)
    )
    model_args, training_args, dataset_args, experiment_args = (
        parser.parse_args_and_config()
    )
    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_PROJECT"] = experiment_args.wandb_project
    os.environ["WANDB_ENTITY"] = experiment_args.wandb_entity

    assert "HF_TOKEN" in os.environ, (
        "HF_TOKEN is not set, set it in environment variables"
    )
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

    train_dataset = load_dataset(dataset_args.dataset_name, "main", split="train")

    train_dataset = train_dataset.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_id,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_id,
        use_fast=True,
        trust_remote_code=True,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
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

    # Local saving
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
