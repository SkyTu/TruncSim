for model in VGG #AlexNet CNN3 LeNet VGG
do
    for fl in 12 16 20 24
    do
        (
            for i in 2 3
            do
                echo "Running $model with fl $fl, Time $i"
                python3 truncation_test.py --model "$model" --fl "$fl" --epoch 10 --loss_type "rent" --rent 7 --time "$i"
            done
        ) &
    done
    wait
done