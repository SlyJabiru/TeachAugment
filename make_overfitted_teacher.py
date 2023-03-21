import torchvision
from torchvision import utils
from torchvision import datasets
from torchvision import models
import torchvision.transforms as T
import torchvision.datasets as D

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
# from torch.utils.tensorboard import SummaryWriter
import argparse
import datetime
import os

import wandb

from lib.models import build_model


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='MNIST', choices=['MNIST', 'CIFAR10', 'CIFAR100'])
parser.add_argument('--root', default='./data', type=str, help='/path/to/dataset')
parser.add_argument('--log_dir', default='./log', type=str)
parser.add_argument('--wandb_project', default='safe-aug', type=str)
parser.add_argument('--data_fraction', default=1, type=int)
args = parser.parse_args()

if args.dataset == 'MNIST':
    transform = T.Compose([ T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST(root=args.root, train=True, download=True, transform=transform)
    testset = datasets.MNIST(root=args.root, train=False, download=True, transform=transform)
elif 'CIFAR' in args.dataset:
    transform = T.Compose([ T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = D.__dict__[args.dataset](args.root, download=True, transform=transform)
    testset = D.__dict__[args.dataset](args.root, train=False, transform=transform)
else:
    raise Exception(f'dataset is weird. args.dataset: {args.dataset}')

subset_indices = list(range(0, len(trainset), args.data_fraction))
subset_train = torch.utils.data.Subset(trainset, subset_indices)
print(f'len(subset_train): {len(subset_train)}')

batch_size = 128
train_dataloader = torch.utils.data.DataLoader(subset_train, batch_size=batch_size, shuffle=True, num_workers=8)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

n_classes = 100 if args.dataset == 'CIFAR100' else 10
n_channel = 1 if args.dataset == 'MNIST' else 3
model = build_model('wrn-28-10', n_classes, n_channel).to(device)

lr = 0.005
num_epochs = 200
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss().to(device)

params = {
    'num_epochs': num_epochs,
    'optimizer': optimizer,
    'loss_function': loss_function,
    'train_dataloader': train_dataloader,
    'test_dataloader': test_dataloader,
    'device': device
}

now = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
args.log_dir = os.path.join(args.log_dir, now)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir) 

wandb_name = f'{args.dataset}-fraction{args.data_fraction}-epoch{num_epochs}'
wandb.init(
    project=args.wandb_project,
    name=wandb_name,
    # sync_tensorboard=True
)
wandb.config.update(args)
print(args)

# loss_function = params["loss_function"]
# train_dataloader = params["train_dataloader"]
# test_dataloader = params["test_dataloader"]
# device = params["device"]

for epoch in range(0, num_epochs):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0

    for i, data in enumerate(train_dataloader, 0):
        # train dataloader 로 불러온 데이터에서 이미지와 라벨을 분리
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 이전 batch에서 계산된 가중치를 초기화
        optimizer.zero_grad() 

        # forward + back propagation 연산
        outputs = model(inputs)
        train_loss = loss_function(outputs, labels)
        train_loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        train_loss_sum += train_loss.item() * len(inputs)

    final_train_loss = train_loss_sum / len(subset_train)

    total = 0
    correct = 0
    test_loss_sum = 0.0

    model.eval()
    for i, data in enumerate(test_dataloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss = loss_function(outputs, labels)
            test_loss_sum += test_loss.item() * len(inputs)

    final_test_loss = test_loss_sum / len(testset)

    # 학습 결과 출력
    print('Epoch: %d, Train loss: %.6f, Test loss: %.6f, Accuracy: %.2f' %(epoch, final_train_loss, final_test_loss, 100*correct/total))
    wandb.log({
        'train/loss': final_train_loss,
        'train/acc': train_correct/len(subset_train),
        'eval/loss': final_test_loss,
        'eval/acc': correct/total,
    })

    if epoch % 20 == 0:
        torch.save(model.state_dict(), os.path.join(args.log_dir, f'withoutDA-wrn-28-10-epoch{epoch:02d}.pt'))
