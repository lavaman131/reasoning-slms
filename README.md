# Reasoning SLMs

## Training

### GRPO Baseline

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch scripts/train_grpo.py --use_peft
```
