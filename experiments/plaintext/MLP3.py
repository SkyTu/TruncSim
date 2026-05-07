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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # param = Parameter.Parameter()

mnist_transforms = transforms.Compose([transforms.ToTensor()])#,
                                    #    transforms.Normalize(mean=mean, std=std)])
train_dataset = torchvision.datasets.MNIST(root="../dataset/", train=True, download=True, transform=mnist_transforms)
test_dataset = torchvision.datasets.MNIST(root="../dataset/", train=False, download=True, transform=mnist_transforms)

from torch.utils.data import DataLoader

class MLP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.seq(x)

def read_file(filename):
    with open(filename, 'rb') as f:
        file_data = f.read()
    numbers = np.frombuffer(file_data, dtype=np.int64)  # dtype根据数据类型调整
    print(len(numbers))
    return numbers

def init_weights(model, weight_list):
    idx = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            shape = param.shape
            num_weights = np.prod(shape)
            if idx + num_weights > len(weight_list):
                raise ValueError(f"Not enough weights in weight_list to initialize parameter {name}")
            weights = np.array(weight_list[idx: idx + num_weights] / (2**24)).reshape(shape)
            param.data.copy_(torch.tensor(weights, dtype=param.dtype))
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
 
BATCH_SIZE = 128

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
# val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = MLP3()
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(params=model.parameters(), momentum =0.90625,lr=0.015625)
accuracy = Accuracy(task='multiclass', num_classes=10)

# device-agnostic setup

accuracy = accuracy.to(device)


filename = '../weights/PSecureMlNoRelu.dat'
weights = read_file(filename)
init_weights(model, weights)
EPOCHS = 10
model = model.to(device)

# 检查设备状态
print(f"Using device: {device}")

# 开始训练
for epoch in tqdm(range(EPOCHS)):
    # Training loop
    highest_acc = 0
    cnt = 0
    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)
        model.train()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if cnt % 10 == 0:
            accuracy = evaluate(model, test_dataloader)
            highest_acc = max(highest_acc, accuracy)
            print(f"Epoch {epoch + 1}, Batch: {cnt}, Loss: {loss.item()}, Accuracy: {accuracy:.2f}")
            with open(f'output/MLP3/plaintext.txt', 'a') as f:
                f.write(f'{accuracy:.2f}\n')
        cnt += 1
with open(f'output/MLP3/highest_plaintext.txt', 'a') as f:
    f.write(f'{highest_acc:.2f}\n')

            
