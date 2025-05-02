# Reasoning SLMs

## Training

Reference: https://huggingface.co/docs/trl/main/en/speeding_up_training?vllm+examples=GRPO

### GRPO Baseline


```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 trl vllm-serve --model "unsloth/Qwen2.5-0.5B-Instruct" --dtype bfloat16
accelerate launch --config_file config/accelerate.yaml scripts/train_dapo.py --config config/dapo.yaml
```
