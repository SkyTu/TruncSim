#!/bin/bash
# fl=24 only; 3 softmax protocols × 6 models. MNIST=2ep, CIFAR10=3ep.
export PATH=/home/student.unimelb.edu.au/xinyutu/.local/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda

mnist_models="MLP3 CNN2 LeNet"
cifar_models="CNN3 AlexNet VGG"

for model in $mnist_models $cifar_models
do
    case " $mnist_models " in
        *" $model "*) ep=2 ;;
        *)            ep=3 ;;
    esac
    for sm in 0 1 2
    do
        echo "=== $model softmax=$sm fl=24 epoch=$ep ==="
        python3 truncation_test.py --model "$model" --fl 24 --epoch "$ep" --loss_type "mean" --softmax_type "$sm" --time 1 2>&1 | tail -50
    done
done
