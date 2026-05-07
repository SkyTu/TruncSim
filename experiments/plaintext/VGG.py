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
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGGModel(version=11):
    print(f"VGG version: {version}")
    if version == 11:
        return VGG('VGG11')
    elif version == 13:
        return VGG('VGG13')
    elif version == 16:
        return VGG('VGG16')
    elif version == 19:
        return VGG('VGG19')
    else:
        raise ValueError("Unsupported VGG version. Choose from 11, 13, 16, or 19.")

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
version=11
model_vgg = VGGModel(version)
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(params=model_vgg.parameters(), momentum =0.90625,lr=0.015625)
accuracy = Accuracy(task='multiclass', num_classes=10)

# device-agnostic setup

accuracy = accuracy.to(device)

EPOCHS = 10
model_vgg = model_vgg.to(device)

# 检查设备状态
print(f"Using device: {device}")

# 开始训练
for epoch in tqdm(range(EPOCHS)):
    # Training loop
    highest_acc = 0
    cnt = 0
    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)
        model_vgg.train()
        y_pred = model_vgg(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if cnt % 10 == 0:
            accuracy = evaluate(model_vgg, test_dataloader)
            highest_acc = max(highest_acc, accuracy)
            print(f"Epoch {epoch + 1}, Batch: {cnt}, Loss: {loss.item()}, Accuracy: {accuracy:.2f}")
            with open(f'output/VGG/'+str(version)+'/cleartext.txt', 'a') as f:
                f.write(f'{accuracy:.2f}\n')
        cnt += 1
with open(f'output/VGG/'+str(version)+'/highest_acc-cleartext.txt', 'a') as f:
    f.write(f'{highest_acc:.2f}\n')