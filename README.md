# KinMethyl

**KinMethyl** is a deep learning framework for detecting methylation especially 5-methylcytosine (5mC) from PacBio SMRT sequencing data.  It integrates regression-based unmethylated kinetics modeling of with classifier to improve detection performance.

## Features

- Sequence-to-kinetics regression from WGA data
- Fusion of raw and predicted IPD/PW signals with sequence
- Support for multiple methylation types: 5mC, 6mA, 4mC

# Usage
---

## Installation

### 1. Install PyTorch
Please install PyTorch first, according to your system and GPU environment:  
👉 [PyTorch official instructions](https://pytorch.org/get-started/locally/)

Examples:
```bash
# CPU only
pip install torch torchvision torchaudio

# With NVIDIA GPU (CUDA 11.7)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# With NVIDIA GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# With NVIDIA GPU (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
### 2. Clone this repository
git clone https://github.com/ZhangBio/KinMethyl.git
cd KinMethyl
###3. Install dependencies
pip install -r requirements.txt
pip install -e .

## Quick Start
```bash
###Train a classification model
kinmethyl-train --train_file example_data/P6C4_5mC/example_train.tsv --valid_file example_data/P6C4_5mC/example_dev.tsv --model_dir examples/model_out  --model_type combined --seq_model models/regression_models/P6C4_regression.ckpt --batch_size 32

###Predict on test data
kinmethyl-test -data_file  example_data/P6C4_5mC/example_test.tsv --model_file examples/model_out/combined.b21_epoch1.ckpt  --model_type combined

# Acknowledgements
We adopted feature extraction pipeline from ccsmeth project. https://github.com/PengNi/ccsmeth
# KinMethyl
Kinetic Modeling-based Methylation Detector
