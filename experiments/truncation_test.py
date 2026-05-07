import argparse
parser = argparse.ArgumentParser()
import os
import sys
curPath = os.path.relpath("../QPyTorch")
quantPath = os.path.relpath("../FixedTorch")
sys.path.append(curPath)
sys.path.append(quantPath)
from FixedTensor import FixedTensor
from Parameter import Parameter
import nn as qnn
import optim as optim
import numpy as np
from torchmetrics import Accuracy
import torch
import tqdm

parser.add_argument('-trunc',
                    '--trunc_type',
                    dest='trunc_type',
                    default=3,
                    type=int,
                    help='截断类别：0:faithful，1:stochastic，2:local，3:probabilistic in TruncXpert')

parser.add_argument('-sm',
                    '--softmax_type',
                    dest='softmax_type',
                    default=0,
                    type=int,
                    help='softmax协议：0:piranha-submax, 1:piranha-relu, 2:sigma, 3:plaintext-exact')

parser.add_argument('-model',
                    '--model_name',
                    dest='model_name', 
                    default='CNN3',
                    type=str)

parser.add_argument('-f',
                    '--fl',
                    dest='fl',
                    type=int, 
                    default=24,
                    help='fl: 小数位长')

parser.add_argument('-wl',
                    '--wl',
                    dest='wl',
                    type=int, 
                    default=64,
                    help='wl: 总位长')

parser.add_argument('-B',
                    '--batch_size',
                    dest='batch_size',
                    type=int,
                    default=128, 
                    help='batch_size: 批大小')

parser.add_argument('-e',
                    '--epoch',
                    dest='epoch',
                    type=int, 
                    default=10,
                    help='epoch: 训练轮数')

parser.add_argument('-l',
                    '--lr',
                    dest='lr',
                    default=0.015625,
                    type=float, 
                    help='lr: 学习率')

parser.add_argument('-m',
                    '--momentum',
                    dest='momentum',
                    default = 0.90625,
                    type=float, 
                    help='momentum: 动量')

parser.add_argument('-ltype',
                    '--loss_type',
                    dest='loss_type',
                    default='mean',
                    type=str,
                    help='loss_type: 损失函数类型：mean, sum, rent')
                    

parser.add_argument('-r',
                    '--rent_bit', 
                    dest='rent', 
                    default=1,
                    type=int, 
                    help='rent_bit: 借位数')

parser.add_argument('-t',
                    '--time',
                    dest='time',
                    default=1,
                    type=int,
                    help='第t次实验')

args = parser.parse_args()

# Seed by --time so within the same `time` index, different trunc/softmax
# settings start from the same random init (fair comparison); across times
# we get 3 different inits for variance estimates.
torch.manual_seed(args.time)
np.random.seed(args.time)

class AlexNet(torch.nn.Module):
    def __init__(self, num_classes=10):
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

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(torch.nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = qnn.Linear(512, 10)

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
                layers += [qnn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [qnn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           qnn.BatchNorm2d(x),
                           qnn.ReLU(inplace=True)]
                in_channels = x
        layers += [qnn.AvgPool2d(kernel_size=1, stride=1)]
        return qnn.Sequential(*layers)

class CNN2(torch.nn.Module):
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

class CNN3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = qnn.Sequential(
            qnn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=1),
            qnn.ReLU(),
            qnn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            qnn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=1),
            qnn.ReLU(),
            qnn.MaxPool2d(kernel_size=3, stride=2, padding=0),   
            
            qnn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=1),
            qnn.ReLU(),
            qnn.MaxPool2d(kernel_size=3, stride=2, padding=0),   
        )
        self.classifier = qnn.Sequential(
            qnn.Flatten(),
            qnn.Linear(64, 10)  # 10 output classes for classification
        )
    def forward(self, x):
        return self.classifier(self.feature(x))

class MLP3(torch.nn.Module):
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

def get_model(model_name):
    if model_name == "CNN2":
        model = CNN2()
    elif model_name == "CNN3":
        model = CNN3()
    elif model_name == "LeNet":
        model = LeNet()
    elif model_name == "AlexNet":
        model = AlexNet()
    elif model_name == "MLP3":
        model = MLP3()
    elif model_name == "VGG":
        model = VGGModel(11)
    return model

def get_dataset(dataset_name):
    if dataset_name == "mnist":
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor
        train_dataset = MNIST(root='./dataset', train=True, download=True, transform=ToTensor())
        test_dataset = MNIST(root='./dataset', train=False, download=True, transform=ToTensor())
    elif dataset_name == "cifar10":
        from torchvision.datasets import CIFAR10
        from torchvision.transforms import ToTensor
        train_dataset = CIFAR10(root='./dataset', train=True, download=True, transform=ToTensor())
        test_dataset = CIFAR10(root='./dataset', train=False, download=True, transform=ToTensor())
    return train_dataset, test_dataset

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
                # 获取对应的权重列表的部分，并调整为相应的形状
                weights = np.array(weight_list[idx: idx + num_weights]/(2**24)).reshape(shape)
            # 将权重赋值到模型参数
            param.data.copy_(torch.tensor(weights, dtype=param.dtype))
            # 更新索引
            idx += num_weights
    print(f"Total number of weights set: {idx}")


def init_weights_uniform(model, weight_list, scale_bits=24):
    """All parameters (weight + bias) loaded with the same scale = 2^scale_bits.
       Matches plaintext baseline convention (PSecureMlNoRelu.dat is stored as
       fl=24 fixed-point integers; both weight and bias are recovered via /2^24).
    """
    idx = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            shape = param.shape
            num_weights = int(np.prod(shape))
            weights = np.array(weight_list[idx: idx + num_weights] / (2**scale_bits)).reshape(shape)
            param.data.copy_(torch.tensor(weights, dtype=param.dtype))
            idx += num_weights
    print(f"Total number of weights set (uniform /2^{scale_bits}): {idx}")

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

truncation_type_name_list = ["faithful", "stochastic", "local", "TruncXpert"]
softmax_type_name_list = ["piranha-submax", "piranha-relu", "sigma", "plaintext"]

model = get_model(args.model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.model_name in ["CNN2", "LeNet", "MLP3"]:
    dataset = 'mnist'
else:
    dataset = 'cifar10'

train_dataset, test_dataset = get_dataset(dataset)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
EPSILON = (1<<args.rent)
Parameter.setfl(args.fl)
Parameter.setwl(args.wl)
Parameter.settrunc_type(args.trunc_type)
Parameter.setsoftmax_type(args.softmax_type)
BATCH_SIZE = args.batch_size
EPOCHS = args.epoch
# CrossEntropyLoss/SGD don't accept epsilon/loss_type kwargs.  Behavior differences:
#   mean:  NLL divides loss by N → small per-sample gradient.
#   rent:  NLL does not divide → sum-gradient retained (more bits survive truncation)
#          and lr is expected to be reduced by ~N (set via --lr).
loss_fn = qnn.CrossEntropyLoss(reduction=args.loss_type).to(device)
optimizer = optim.SGD(params=model.parameters(), momentum=args.momentum, lr=args.lr, batch_size=BATCH_SIZE)
accuracy = Accuracy(task='multiclass', num_classes=10)
accuracy = accuracy.to(device)

if args.model_name == "CNN2":
    filename = 'weights/CNN2.dat'
    weights = read_file(filename)
    init_weights(model, weights)
if args.model_name == "CNN3":
    filename = 'weights/CNN3.dat'
    weights = read_file(filename)
    init_weights(model, weights)
if args.model_name == "MLP3":
    filename = 'weights/PSecureMlNoRelu.dat'
    weights = read_file(filename)
    # PSecureMlNoRelu.dat: both weight and bias stored as round(value * 2^24);
    # recover via uniform /2^24 to match the plaintext baseline.
    init_weights_uniform(model, weights, scale_bits=24)

model = model.to(device)

    
SM_NAME = softmax_type_name_list[args.softmax_type]
PATH = "output/"+args.model_name+"/" + SM_NAME + "/" + args.loss_type + "/"

if args.trunc_type == 2:
    PATH = "output/"+args.model_name+"/" + SM_NAME + "/local/" + args.loss_type + "/"

os.makedirs(PATH, exist_ok=True)

for epoch in range(EPOCHS):
    highest_acc = 0
    cnt = 0
    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)
        X, y = FixedTensor(X).round(), FixedTensor(y)
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
            if args.trunc_type == 2:
                with open(PATH+truncation_type_name_list[args.trunc_type] + '-' + str(args.wl) +'-'+str(args.time)+ '.txt', 'a') as f:
                    f.write(f'{accuracy:.2f}\n')
            else:    
                with open(PATH+truncation_type_name_list[args.trunc_type] + '-' + str(args.fl) +'-'+str(args.time)+ '.txt', 'a') as f:
                    f.write(f'{accuracy:.2f}\n')
        cnt += 1
    if args.trunc_type == 2:
        with open(PATH+'highest_acc-'+truncation_type_name_list[args.trunc_type] + '-' + str(args.wl) +'-'+str(args.time) +'.txt', 'a') as f:
            f.write(f'{highest_acc:.2f}\n')
    else:    
        with open(PATH+'highest_acc-'+truncation_type_name_list[args.trunc_type] + '-' + str(args.fl) +'-'+str(args.time) +'.txt', 'a') as f:
            f.write(f'{highest_acc:.2f}\n')