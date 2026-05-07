import os
import sys
curPath = os.path.relpath("../QPyTorch")

quantPath = os.path.relpath("../FixedTorch")
sys.path.append(curPath)

sys.path.append(quantPath)
import torch
from torch import Tensor
import torch.nn as nn
# import torch.optim as optim
from collections import Counter
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
import numpy as np
import torch.nn.functional as F
import optim as optim
from torchmetrics import Accuracy

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # param = Parameter.Parameter()

mnist_transforms = transforms.Compose([transforms.ToTensor()])#,
                                    #    transforms.Normalize(mean=mean, std=std)])
train_dataset = torchvision.datasets.MNIST(root="./dataset/", train=True, download=True, transform=mnist_transforms)
test_dataset = torchvision.datasets.MNIST(root="./dataset/", train=False, download=True, transform=mnist_transforms)

truncation_type_name_list = ["stochastic", "faithful", "local", "TruncXpert"]

# train_dataset, val_dataset = torch.utils.data.random_split(dataset=train_val_dataset, lengths=[train_size, val_size])
from torch.utils.data import DataLoader


import nn as qnn

class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = qnn.Sequential(
            qnn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=0),
            qnn.ReLU(),
            qnn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            qnn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=0),
            qnn.ReLU(),
            qnn.MaxPool2d(kernel_size=2, stride=2, padding=0)    
        )
        self.classifier = qnn.Sequential(
            qnn.Flatten(),
            qnn.Linear(256, 128),  # Assuming the output of the flatten layer is 256
            qnn.ReLU(),
            qnn.Linear(128, 10)  # 10 output classes for classification
        )
    def forward(self, x):
        return self.classifier(self.feature(x))

def read_file(filename):
    with open(filename, 'rb') as f:
        file_data = f.read()
    numbers = np.frombuffer(file_data, dtype=np.int64)  # dtype根据数据类型调整
    print(len(numbers))
    return numbers

def init_weights(model, weight_list):
    # 获取权重列表中的索引
    idx = 0

    # 遍历模型的所有参数
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            # 获取该参数的形状
            shape = param.shape
            # 计算当前参数需要的权重数量
            num_weights = np.prod(shape)

            # 获取对应的权重列表的部分，并调整为相应的形状
            weights = np.array(weight_list[idx: idx + num_weights]/(2**24)).reshape(shape)
            # 将权重赋值到模型参数
            param.data.copy_(torch.tensor(weights, dtype=param.dtype))
            print("-"*50)
            print(name)
            print(param.data)
            # 更新索引
            idx += num_weights
    print(f"Total number of weights set: {idx}")
   
def evaluate(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print("Validate acc:", accuracy)
    return accuracy
 
for _ in range(3):
    for fl in range(12, 25, 4):
        for trunc_type in [3]:
            Parameter.setfl(fl)
            Parameter.settrunc_type(trunc_type)

            BATCH_SIZE = 128

            train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
            # val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            model_cnn2 = CNN2()
            loss_fn = qnn.CrossEntropyLoss(reduction='mean').to(device)
            optimizer = optim.SGD(params=model_cnn2.parameters(), momentum =0.90625,lr=0.015625, batch_size =BATCH_SIZE)
            accuracy = Accuracy(task='multiclass', num_classes=10)

            # device-agnostic setup

            accuracy = accuracy.to(device)


            filename = 'weights/CNN2.dat'
            weights = read_file(filename)
            init_weights(model_cnn2, weights)
            EPOCHS = 10
            model_cnn2 = model_cnn2.to(device)

            # 检查设备状态
            print(f"Using device: {device}")

            # 开始训练
            for epoch in tqdm(range(EPOCHS)):
                # Training loop
                highest_acc = 0
                cnt = 0
                for X, y in train_dataloader:
                    X, y = X.to(device), y.to(device)
                    X, y = FixedTensor(X).round(), FixedTensor(y)
                    model_cnn2.train()
                    y_pred = model_cnn2(X)
                    loss = loss_fn(y_pred, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if cnt % 10 == 0:
                        accuracy = evaluate(model_cnn2, test_dataloader)
                        highest_acc = max(highest_acc, accuracy)
                        print(f"Epoch {epoch + 1}, Batch: {cnt}, Loss: {loss.item()}, Accuracy: {accuracy:.2f}")
                        with open(f'output/CNN2/'+truncation_type_name_list[trunc_type] + '-' + str(fl) +'.txt', 'a') as f:
                            f.write(f'{accuracy:.2f}\n')
                    cnt += 1
            with open(f'output/CNN2/highest_acc-'+truncation_type_name_list[trunc_type] + '-' + str(fl) +'.txt', 'a') as f:
                f.write(f'{highest_acc:.2f}\n')

            
