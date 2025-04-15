# Reasoning SLMs

## Getting Started

```bash
git clone --recursive git@github.com:lavaman131/reasoning-slms.git
```

```bash
# from https://verl.readthedocs.io/en/latest/start/install.html#install-from-custom-environment

conda create -n verl python==3.10
conda activate verl

# install verl together with some lightweight dependencies in setup.py
conda install -c nvidia nccl
conda install numpy ninja
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install flash-attn --no-build-isolation
cd verl
python -m pip install -e .
```
