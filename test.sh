#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -P gch51598

cd $PBS_O_WORKDIR
source ~/miniconda3/bin/activate
conda activate ccsmethenv

kinmethyl-train --train_file example_data/P6C4_5mC/example_train.tsv --valid_file example_data/P6C4_5mC/example_dev.tsv --model_dir examples/model_out  --model_type combined --seq_model models/regression_models/P6C4_regression.ckpt --batch_size 32
kinmethyl-test -data_file  example_data/P6C4_5mC/example_test.tsv --model_file examples/model_out/combined.b21_epoch1.ckpt  --model_type combined 