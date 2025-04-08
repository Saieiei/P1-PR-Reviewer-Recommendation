#!/bin/bash
module purge
module load cray-python/3.11.7
module load PrgEnv-nvidia/8.6.0
module swap nvidia/20.11 nvidia/24.7
pip install --upgrade pip
pip install requests
pip install -r requirements2.txt
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
python offloading_distributed_script.py
