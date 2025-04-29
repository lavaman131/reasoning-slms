# Reasoning SLMs

## Training

Reference: https://huggingface.co/docs/trl/main/en/speeding_up_training?vllm+examples=GRPO

### GRPO Baseline


```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 trl vllm-serve --model google/gemma-3-1b-it --dtype bfloat16
accelerate launch --config_file config/accelerate.yaml scripts/train_grpo.py --config config/grpo.yaml
```
