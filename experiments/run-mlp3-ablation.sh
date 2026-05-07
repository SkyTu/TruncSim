#!/bin/bash
# MLP3 ablation @ fl=24, all using PSecureMlNoRelu.dat init (deterministic).
#
# Layer A: plaintext baseline (float32, separate script)
# Layer B: fixed-point + plaintext-exact softmax, vary trunc_type:
#          faithful (tt=0) vs probabilistic-with-(-1)LSB-error (tt=3)
# Layer C: fixed-point + plaintext-exact softmax + tt=3, vary loss_type:
#          mean (lr=0.015625) vs rent (lr=0.000122)
# Layer D: fixed-point + tt=3 + mean loss, vary softmax:
#          plaintext / piranha-submax / piranha-relu / sigma

export PATH=/home/student.unimelb.edu.au/xinyutu/.local/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
cd "$(dirname "$0")"

EP=2
FL=24
TIME=1
LR_MEAN=0.015625
LR_RENT=0.000122   # 0.015625/128

# --- Layer A: plaintext baseline ---
echo "=== A: plaintext baseline (10 epochs as in plaintext/MLP3.py) ==="
mkdir -p plaintext/output/MLP3
( cd plaintext && python3 MLP3.py 2>&1 | tail -10 )

# --- Layer B: trunc_type comparison, sm=plaintext, loss=mean ---
for tt in 0 3 ; do
    name=$([ "$tt" = 0 ] && echo faithful || echo TruncXpert)
    echo "=== B: trunc_type=$tt ($name), sm=plaintext, loss=mean ==="
    python3 truncation_test.py --model MLP3 --fl $FL --epoch $EP --loss_type mean \
        --softmax_type 3 --trunc_type $tt --lr $LR_MEAN --time $TIME 2>&1 | tail -8
done

# --- Layer C: rent vs mean, sm=plaintext, tt=3 (TruncXpert) ---
echo "=== C: trunc_type=3 (TruncXpert), sm=plaintext, loss=rent (lr=$LR_RENT) ==="
python3 truncation_test.py --model MLP3 --fl $FL --epoch $EP --loss_type rent \
    --softmax_type 3 --trunc_type 3 --lr $LR_RENT --time $TIME 2>&1 | tail -8

# --- Layer D: vary softmax under tt=3 + mean loss ---
# (sm=3 plaintext already covered by Layer B's tt=3 run; re-run for self-consistency)
for sm in 0 1 2 3 ; do
    smname=$(case $sm in 0) echo piranha-submax;; 1) echo piranha-relu;; 2) echo sigma;; 3) echo plaintext;; esac)
    echo "=== D: trunc_type=3, sm=$sm ($smname), loss=mean ==="
    python3 truncation_test.py --model MLP3 --fl $FL --epoch $EP --loss_type mean \
        --softmax_type $sm --trunc_type 3 --lr $LR_MEAN --time $TIME 2>&1 | tail -8
done

echo "=== DONE: all MLP3 ablation runs finished ==="
