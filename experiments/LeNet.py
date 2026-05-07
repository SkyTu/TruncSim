import os
import sys
curPath = os.path.relpath("../QPyTorch")

quantPath = os.path.relpath("../FixedTorch")
sys.path.append(curPath)

sys.path.append(quantPath)
import torch
from torch import Tensor
# import torch.optim as optim
from collections import Counter
import matplotlib.pyplot as plt
from FixedTensor import FixedTensor
from Parameter import Parameter
import nn as qnn
# from torch.autograd import Variable 

import optim as optim
import MultiSpline
import math
import torchvision
import torchvision.transforms as transforms
import numpy as np
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

# train_dataset, val_dataset = torch.utils.data.random_split(dataset=train_val_dataset, lengths=[train_size, val_size])
from torch.utils.data import DataLoader


class LeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = qnn.Sequential(
            qnn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            qnn.ReLU(),
            qnn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            qnn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            qnn.ReLU(),
            qnn.MaxPool2d(kernel_size=2, stride=2, padding=0)    
        )
        self.classifier = qnn.Sequential(
            qnn.Flatten(),
            qnn.Linear(256, 120),  # Assuming the output of the flatten layer is 256
            qnn.ReLU(),
            qnn.Linear(120, 84),
            qnn.ReLU(),
            qnn.Linear(84, 10)  # 10 output classes for classification
        )
    def forward(self, x):
        return self.classifier(self.feature(x))
    
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
 
BATCH_SIZE = 128

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
# val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

for _ in range(3):
    for fl in range(12, 25, 4):
        for trunc_type in [3]:
            Parameter.setfl(fl)
            Parameter.settrunc_type(trunc_type)

            BATCH_SIZE = 128

            train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            model_lenet = LeNet()
            loss_fn = qnn.CrossEntropyLoss(reduction='mean').to(device)
            optimizer = optim.SGD(params=model_lenet.parameters(), momentum =0.90625,lr=0.015625, batch_size =BATCH_SIZE)
            accuracy = Accuracy(task='multiclass', num_classes=10)

            # device-agnostic setup

            accuracy = accuracy.to(device)


            EPOCHS = 10
            model_lenet = model_lenet.to(device)

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
                    model_lenet.train()
                    y_pred = model_lenet(X)
                    loss = loss_fn(y_pred, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if cnt % 10 == 0:
                        accuracy = evaluate(model_lenet, test_dataloader)
                        highest_acc = max(highest_acc, accuracy)
                        print(f"Epoch {epoch + 1}, Batch: {cnt}, Loss: {loss.item()}, Accuracy: {accuracy:.2f}")
                        with open(f'output/lenet/'+truncation_type_name_list[trunc_type] + '-' + str(fl) +'.txt', 'a') as f:
                            f.write(f'{accuracy:.2f}\n')
                    cnt += 1
            with open(f'output/lenet/highest_acc-'+truncation_type_name_list[trunc_type] + '-' + str(fl) +'.txt', 'a') as f:
                f.write(f'{highest_acc:.2f}\n')
