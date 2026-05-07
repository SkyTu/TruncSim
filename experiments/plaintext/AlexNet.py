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
NUM_CLASSES=10

class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

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

model_alexnet = AlexNet()
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(params=model_alexnet.parameters(), momentum =0.90625,lr=0.015625)
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
            with open(f'output/AlexNet/cleartext.txt', 'a') as f:
                f.write(f'{accuracy:.2f}\n')
        cnt += 1
with open(f'output/AlexNet/highest_acc-cleartext.txt', 'a') as f:
    f.write(f'{highest_acc:.2f}\n')