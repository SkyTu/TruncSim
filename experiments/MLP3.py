import os
import sys
curPath = os.path.relpath("../QPyTorch")

quantPath = os.path.relpath("../")
sys.path.append(curPath)

sys.path.append(quantPath)
import torch
from torch import Tensor
# import torch.optim as optim
from collections import Counter
import matplotlib.pyplot as plt
from FixedTorch.FixedTensor import FixedTensor
from FixedTorch.Parameter import Parameter
import FixedTorch.nn as qnn
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

truncation_type_name_list = ["faithful", "stochastic"]

# train_dataset, val_dataset = torch.utils.data.random_split(dataset=train_val_dataset, lengths=[train_size, val_size])
from torch.utils.data import DataLoader


import nn as qnn

class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = qnn.Sequential(
            qnn.Flatten(),
            qnn.Linear(784, 128),
            qnn.ReLU(),
            qnn.Linear(128, 128),
            qnn.ReLU(),
            qnn.Linear(128, 10),
        )
    def forward(self, x):
        return self.seq(x)

    def initialize_weight(self):
        from functools import reduce
        import operator
        for m in self.modules():
            if isinstance(m, qnn.Linear):
                bound = math.sqrt(6.0 / (m.weight.shape[0]+m.weight.shape[1])) 
                bound = FixedTensor([bound]).round()
                m.weight.data.uniform_(-bound[0], bound[0])
                # m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

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
 
for fl in range(12, 24, 4):
    for trunc_type in [1, 3]:
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


        model_cnn2.initialize_weight()
        EPOCHS = 2
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

            
