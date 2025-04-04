#!/bin/bash
module purge
module load cray-python/3.11.7
module load PrgEnv-nvidia/8.6.0
module load nvidia/20.11
pip install --upgrade pip
pip install requests
pip install -r requirements2.txt
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
python offloading-&&-distributed_script.py
