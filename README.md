# Reasoning SLMs

Recent advances in Large Language Model fine-tuning have demonstrated how reinforcement learning techniques can dramatically enhance reasoning capabilities, with methods like DeepSeek’s Group Relative Policy Optimization (GRPO) eliminating the need for separate critic models. Building on this foundation, Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) has achieved impressive results on complex reasoning benchmarks with large models through its innovative clip-higher strategy, dynamic sampling mechanism, token-level policy gradient loss, and overlong reward shaping. While these techniques have proven effective for models with tens of billions of parameters, their applicability to smaller, more accessible models remains unexplored. This project aims to implement DAPO for fine-tuning the compact Qwen2.5-0.5B model, systematically experimenting with modifications to the algorithm’s core components to determine whether sophisticated reasoning capabilities can be induced in resource-constrained language models without requiring massive computational resources.

## Installation

```bash
uv sync --extra build
# for flash-attn support
uv sync --extra build --extra compile
```

## Training

Reference: 

Example with 8 GPUs (4 GPUs for training and 4 for vllm inference server):

```bash
MODEL="unsloth/Qwen2.5-0.5B-Instruct"
DTYPE="bfloat16"
```

### GRPO Baseline

```bash
# https://huggingface.co/docs/trl/main/en/speeding_up_training?vllm+examples=GRPO
CUDA_VISIBLE_DEVICES=4,5,6,7 trl vllm-serve --model $MODEL --dtype $DTYPE
accelerate launch --config_file config/accelerate.yaml scripts/train_grpo.py --config config/grpo.yaml
```

### Best DAPO Experiment

```bash
# https://huggingface.co/docs/trl/main/en/speeding_up_training?vllm+examples=GRPO
CUDA_VISIBLE_DEVICES=4,5,6,7 trl vllm-serve --model $MODEL --dtype $DTYPE
accelerate launch --config_file config/accelerate.yaml scripts/train_dapo.py --config config/dapo_best.yaml
```