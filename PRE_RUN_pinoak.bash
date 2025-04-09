#!/bin/bash
salloc -p griz512 --gres=gpu:1 --time=01:00:00
srun --pty /bin/bash
module purge
module load cray-python/3.11.7
module load PrgEnv-nvidia/8.6.0
module load nvidia/24.11
pip install --upgrade pip
pip install requests
pip install -r requirements.txt
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
python offloading_distributed_script.py
