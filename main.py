'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

# Our device here is GPU called cuda, if it is not avaliable then use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data, We are using CIFAR10 dataset. We are trasforming these data. those data, we are storing in root. we traing only in traing dataset, so train = False in test dataset 
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), # data augumentation = RandomCrop() = crop the image at random location
    transforms.RandomHorizontalFlip(), # data augumentation = RandomHorizontalFlip() = Horizontally flip the given image randomly with a given probability. 
    transforms.ToTensor(), # samples are coverted into tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # Normalizing data that means shift data where it is easy to accessible by model
])

#we are not augumenting data on test dataset, because we are evaluating our model on test dataset
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

net = ResNet18()           # ResNet18() is neural network with 18 layers

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()  # MLE(maximum likehood estimation)
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4) # training/updating function
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)  # consine function as learning rate


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()     # after running this line, the parameters of model will be trained/updated.
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):  # We are gonna split datasets in batches if dataset is too large using trainloader.
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()  # performing sophistic gradiant decent
        outputs = net(inputs)  # conditional prbability = p (outputs | inputs) , outputs = NN(images)
        loss = criterion(outputs, targets) # loss function
        loss.backward()   # calculating gradiants
        optimizer.step()   # updating parameters, loss will be minimized

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()     # after running this line, the parameters of model will not be trained.
    test_loss = 0
    correct = 0
    total = 0      
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)  # conditional prbability = p (outputs | inputs) , outputs = NN(images)
            loss = criterion(outputs, targets)   # loss function = loss(theta, inputs, outputs)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
        print('Accuracy after epoch ',epoch,': ',acc,'%')

# Here epoch means a complete pass of the training data set. We are training and testing 200 times.
for epoch in range(start_epoch, start_epoch+200):
    train(epoch)       # train model
    test(epoch)        # test model
    scheduler.step()   # update parameter for training process
