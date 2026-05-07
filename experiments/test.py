import os
import sys
curPath = os.path.relpath("../QPyTorch")
print(curPath)
quantPath = os.path.relpath("../FixedTorch")
sys.path.append(curPath)

sys.path.append(quantPath)
import torch
from torch import Tensor
import torch.nn as nn
# import torch.optim as optim
from collections import Counter
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from FixedTensor import FixedTensor
from Parameter import Parameter

# from torch.autograd import Variable 
import nn as qnn
import optim as optim
import MultiSpline
import math
import torchvision
import torchvision.transforms as transforms

Parameter.setfl(16)
Parameter.settrunc_type(2)
x = torch.randn(1, 1)
y = torch.randn(1, 1)
x_fix = FixedTensor(x).round()
y_fix = FixedTensor(y).round()

print("x:", x)
print("y:", y)
print("x_fix:", x_fix)
print("y_fix:", y_fix)

z_fix = x_fix * y_fix
z_fix = z_fix.round()
print("z_fix:", z_fix)