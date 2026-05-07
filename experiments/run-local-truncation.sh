#!/bin/bash

for model in MLP3 CNN2 AlexNet CNN3 LeNet VGG
do
    for wl in 64 74 84 94 104
    do
        (
            for i in 1
            do
                echo "Running $model with wl $wl, Time $i"
                python3 truncation_test.py --model "$model" --wl "$wl" --fl 24 --epoch 1 --loss_type "mean" --time "$i" --trunc_type "2"
            done
        ) &
    done
    wait
done