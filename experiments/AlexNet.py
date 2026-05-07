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

cifar_transforms = transforms.Compose([transforms.ToTensor()])#,
                                    #    transforms.Normalize(mean=mean, std=std)])

train_dataset = torchvision.datasets.CIFAR10(root="./dataset/", train=True, download=True, transform=cifar_transforms)
test_dataset = torchvision.datasets.CIFAR10(root="./dataset/", train=False, download=True, transform=cifar_transforms)

truncation_type_name_list = ["stochastic", "faithful", "local", "TruncXpert"]

# train_dataset, val_dataset = torch.utils.data.random_split(dataset=train_val_dataset, lengths=[train_size, val_size])
from torch.utils.data import DataLoader


import nn as qnn
NUM_CLASSES = 10
class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        self.features = qnn.Sequential(
            qnn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            qnn.ReLU(inplace=True),
            qnn.MaxPool2d(kernel_size=2),
            qnn.Conv2d(64, 192, kernel_size=3, padding=1),
            qnn.ReLU(inplace=True),
            qnn.MaxPool2d(kernel_size=2),
            qnn.Conv2d(192, 384, kernel_size=3, padding=1),
            qnn.ReLU(inplace=True),
            qnn.Conv2d(384, 256, kernel_size=3, padding=1),
            qnn.ReLU(inplace=True),
            qnn.Conv2d(256, 256, kernel_size=3, padding=1),
            qnn.ReLU(inplace=True),
            qnn.MaxPool2d(kernel_size=2),
        )
        self.classifier = qnn.Sequential(
            qnn.Linear(256 * 2 * 2, 4096),
            qnn.ReLU(inplace=True),
            qnn.Linear(4096, 4096),
            qnn.ReLU(inplace=True),
            qnn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

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

EPSILON = 1
for _ in range(3):
    for fl in range(20, 24, 4):
        for trunc_type in [3]:
            Parameter.setfl(fl)
            Parameter.settrunc_type(trunc_type)

            BATCH_SIZE = 128

            train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            model_alexnet = AlexNet()
            loss_fn = qnn.CrossEntropyLoss(reduction='mean').to(device)
            optimizer = optim.SGD(params=model_alexnet.parameters(), momentum =0.90625,lr=0.015625, batch_size =BATCH_SIZE)
            accuracy = Accuracy(task='multiclass', num_classes=10)

            # device-agnostic setup

            accuracy = accuracy.to(device)
            EPOCHS = 10
            model_alexnet = model_alexnet.to(device)

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
                    model_alexnet.train()
                    y_pred = model_alexnet(X)
                    loss = loss_fn(y_pred, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if cnt % 10 == 0:
                        accuracy = evaluate(model_alexnet, test_dataloader)
                        highest_acc = max(highest_acc, accuracy)
                        print(f"Epoch {epoch + 1}, Batch: {cnt}, Loss: {loss.item()}, Accuracy: {accuracy:.2f}")
                        with open(f'output/AlexNet/'+truncation_type_name_list[trunc_type] + '-' + str(fl) +'.txt', 'a') as f:
                            f.write(f'{accuracy:.2f}\n')
                    cnt += 1
            with open(f'output/AlexNet/highest_acc-'+truncation_type_name_list[trunc_type] + '-' + str(fl) +'.txt', 'a') as f:
                f.write(f'{highest_acc:.2f}\n')

                
