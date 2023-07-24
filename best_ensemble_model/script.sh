#!/bin/bash

# Set the WandB API key
export WANDB_API_KEY="" # deleted key for security reasons.

cd program/

# CONDA
conda create -n mypython3 python=3.11 2> /dev/null > /dev/null
source activate mypython3 #2> /dev/null > /dev/null
conda update -n base -c defaults conda
conda install numpy  #2> /dev/null > /dev/null
conda install tensorflow #2> /dev/null > /dev/null
conda install keras #2> /dev/null > /dev/null
conda install pillow #2> /dev/null > /dev/null
conda install h5py #2> /dev/null > /dev/null
conda install transformers #2> /dev/null > /dev/null

# Install or upgrade transformers package
pip install --upgrade transformers

# Install wget package
pip install wget

# Install or upgrade transformers from GitHub
pip install -U git+https://github.com/huggingface/transformers.git

# Install or upgrade accelerate from GitHub
pip install -U git+https://github.com/huggingface/accelerate.git

pip install torchvision==0.15.1

pip install torch==2.0.0

pip install wandb

python3 deakin_ai_challenge_submission.py $1 $2


