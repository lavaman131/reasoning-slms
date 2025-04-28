#!/bin/bash -l

#$ -P ds543
#$ -l h_rt=24:00:00
#$ -N gemma-3-1b-it-grpo-4bit-baseline
#$ -pe omp 8
#$ -j y # Merge the error and output streams into a single file
#$ -l gpus=4
#$ -l gpu_memory=48G
#$ -l gpu_c=7.0

module load cmake gcc/10.2.0 llvm/9.0.1 miniconda openmpi cuda/12.5

conda activate trl

accelerate launch scripts/train_grpo.py --use_peft