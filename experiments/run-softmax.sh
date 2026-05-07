#!/bin/bash
# Sweep over (model, fl, softmax_type) for softmax-protocol error comparison.
# softmax_type: 0=piranha-submax, 1=piranha-relu, 2=sigma
# epochs: MNIST(MLP3/CNN2/LeNet)=2, CIFAR10(CNN3/AlexNet/VGG)=3

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
        for fl in 12 16 20 24
        do
            (
                for i in 1 2 3
                do
                    echo "Running $model softmax=$sm fl=$fl epoch=$ep time=$i"
                    python3 truncation_test.py --model "$model" --fl "$fl" --epoch "$ep" --loss_type "mean" --softmax_type "$sm" --time "$i"
                done
            ) &
        done
        wait
    done
done
