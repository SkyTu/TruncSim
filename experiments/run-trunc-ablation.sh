#!/bin/bash
# Truncation-protocol ablation: faithful (tt=0) vs stochastic (tt=1) vs probabilistic (tt=3)
# 6 models × 3 fl ∈ {16,20,24} × 3 trunc × 3 reps = 162 runs
# softmax = plaintext (sm=3); loss = mean (lr=0.015625)
# MNIST: 2 epoch; CIFAR10: 3 epoch.
# Parallel across models: 6 background subshells; each does its 27 runs sequentially.

export PATH=/home/student.unimelb.edu.au/xinyutu/.local/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
cd "$(dirname "$0")"

LR=0.015625
SM=3

mkdir -p /tmp/trunc_ablation

run_model() {
    local model=$1 ep=$2
    local logf=/tmp/trunc_ablation/${model}.log
    : > "$logf"
    for fl in 16 20 24 ; do
        for tt in 0 1 3 ; do
            ttname=$(case $tt in 0) echo faithful;; 1) echo stochastic;; 3) echo probabilistic;; esac)
            for time in 1 2 3 ; do
                echo "=== $model fl=$fl tt=$tt($ttname) rep=$time ===" >> "$logf"
                python3 truncation_test.py \
                    --model "$model" --fl "$fl" --epoch "$ep" \
                    --loss_type mean --softmax_type "$SM" --trunc_type "$tt" \
                    --lr "$LR" --time "$time" >> "$logf" 2>&1
            done
        done
    done
    echo "=== $model DONE ===" >> "$logf"
}

# MNIST 2 epochs
run_model MLP3   2 &
run_model CNN2   2 &
run_model LeNet  2 &
# CIFAR10 3 epochs
run_model CNN3   3 &
run_model AlexNet 3 &
run_model VGG    3 &

wait
echo "=== ALL MODELS DONE ===" > /tmp/trunc_ablation/_summary.log
