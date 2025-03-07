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
mean = 0.1307
std = 0.3081
# # param = Parameter.Parameter()
Parameter.setfl(16)
Parameter.settrunc_type(1)

mnist_transforms = transforms.Compose([transforms.ToTensor()])#,
                                    #    transforms.Normalize(mean=mean, std=std)])
train_dataset = torchvision.datasets.MNIST(root="./data/", train=True, download=True, transform=mnist_transforms)
test_dataset = torchvision.datasets.MNIST(root="./data/", train=False, download=True, transform=mnist_transforms)
# train_size = int(0.9 * len(train_val_dataset))
# val_size = len(train_val_dataset) - train_size

# train_dataset, val_dataset = torch.utils.data.random_split(dataset=train_val_dataset, lengths=[train_size, val_size])
from torch.utils.data import DataLoader

BATCH_SIZE = 512

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
# val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

import nn as qnn

class LeNet5V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = qnn.Sequential(
            #1
            qnn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1),   # 28*28->32*32-->28*28
            # qnn.BatchNorm2d(6),
            qnn.ReLU(),
            qnn.MaxPool2d(kernel_size=2, stride=2),  # 14*14
            
            #2
            qnn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1),  # 10*10
            # qnn.BatchNorm2d(16),
            qnn.ReLU(),
            qnn.MaxPool2d(kernel_size=2, stride=2),  # 5*5
            
        )
        self.classifier = qnn.Sequential(
            qnn.Flatten(),
            qnn.Linear(in_features=50*4*4, out_features=500),
            qnn.ReLU(),
            qnn.Linear(in_features=500, out_features=10),
            # qnn.ReLU(),
            # qnn.Linear(in_features=84, out_features=10),
        )
    def initialize_weight(self):
        from functools import reduce
        import operator
        for m in self.modules():
            if isinstance(m, qnn.Conv2d):
                bound = math.sqrt(6 / (m.weight.shape[0]+reduce(operator.mul, m.weight.shape[1:]))) 
                m.weight.data.uniform_(-bound, bound)
                # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, qnn.Linear):
                bound = math.sqrt(6.0 / (m.weight.shape[0]+m.weight.shape[1])) 
                bound = FixedTensor([bound]).round()
                m.weight.data.uniform_(-bound[0], bound[0])
                # m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()
    def forward(self, x):
        return self.classifier(self.feature(x))
import optim as optim
from torchmetrics import Accuracy
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_lenet5v1 = LeNet5V1()
loss_fn = qnn.CrossEntropyLoss(reduction='sum').to(device)
optimizer = optim.SGD(params=model_lenet5v1.parameters(), momentum =0.9,lr=0.01, batch_size =BATCH_SIZE)
accuracy = Accuracy(task='multiclass', num_classes=10)

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import os

# Experiment tracking
timestamp = datetime.now().strftime("%Y-%m-%d")
experiment_name = "MNIST"
model_name = "LeNet5V1"
log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
writer = SummaryWriter(log_dir)

# device-agnostic setup

accuracy = accuracy.to(device)
model_lenet5v1 = model_lenet5v1.to(device)
model_lenet5v1.initialize_weight()
EPOCHS = 1

for epoch in tqdm(range(EPOCHS)):
    # Training loop
    train_loss, train_acc = 0.0, 0.0
    for X, y in train_dataloader:
        X, y = X.to(device)*(255.0/256), y.to(device)
        # print(y)
        # print(X[0])
        X, y = FixedTensor(X).round(), FixedTensor(y)
        
        model_lenet5v1.train()
        
        y_pred = model_lenet5v1(X)
        
        loss = loss_fn(y_pred, y)
        
        # print(y_pred)
        train_loss += loss.item()
        acc = accuracy(y_pred, y)
        # print(loss.item())
        print((torch.div(loss,BATCH_SIZE)).item(), acc.item())
        # exit()

        train_acc += acc
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
        
    # # Validation loop
    # val_loss, val_acc = 0.0, 0.0
    # model_lenet5v1.eval()
    # with torch.inference_mode():
    #     for X, y in test_dataloader:
    #         X, y = X.to(device), y.to(device)
    #         # print(y)
            
    #         X, y = FixedTensor(X).round(), FixedTensor(y)
            
    #         y_pred = model_lenet5v1(X)
            
    #         loss = loss_fn(y_pred, y)
    #         val_loss += loss.item()
            
    #         acc = accuracy(y_pred, y)
    #         val_acc += acc
            
    #     val_loss /= len(test_dataloader)
    #     val_acc /= len(test_dataloader)
        
    # writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss}, global_step=epoch)
    # writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc}, global_step=epoch)
    
    # print(train_acc)
    # print(val_acc)
    # print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}")