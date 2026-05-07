import os
import sys

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from torchmetrics import Accuracy

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import os

from torch.utils.data import DataLoader

class CNN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),   
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),   
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 10)  # 10 output classes for classification
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

            if("bias" in name):     
            # 获取对应的权重列表的部分，并调整为相应的形状
                weights = np.array(weight_list[idx: idx + num_weights]/(2**48)).reshape(shape)
            else:
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # param = Parameter.Parameter()

cifar_transforms = transforms.Compose([transforms.ToTensor()])#,
                                    #    transforms.Normalize(mean=mean, std=std)])

train_dataset = torchvision.datasets.CIFAR10(root="../dataset/", train=True, download=True, transform=cifar_transforms)
test_dataset = torchvision.datasets.CIFAR10(root="../dataset/", train=False, download=True, transform=cifar_transforms)

BATCH_SIZE = 128

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model_cnn3 = CNN3()
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(params=model_cnn3.parameters(), momentum =0.90625,lr=0.015625)
accuracy = Accuracy(task='multiclass', num_classes=10)

# device-agnostic setup

accuracy = accuracy.to(device)


filename = '../weights/CNN3.dat'
weights = read_file(filename)
init_weights(model_cnn3, weights)
EPOCHS = 10
model_cnn3 = model_cnn3.to(device)

# 检查设备状态
print(f"Using device: {device}")

# 开始训练
for epoch in tqdm(range(EPOCHS)):
    # Training loop
    highest_acc = 0
    cnt = 0
    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)
        model_cnn3.train()
        y_pred = model_cnn3(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if cnt % 10 == 0:
            accuracy = evaluate(model_cnn3, test_dataloader)
            highest_acc = max(highest_acc, accuracy)
            print(f"Epoch {epoch + 1}, Batch: {cnt}, Loss: {loss.item()}, Accuracy: {accuracy:.2f}")
            with open(f'output/CNN3/cleartext.txt', 'a') as f:
                f.write(f'{accuracy:.2f}\n')
        cnt += 1
with open(f'output/CNN3/highest_acc-cleartext.txt', 'a') as f:
    f.write(f'{highest_acc:.2f}\n')