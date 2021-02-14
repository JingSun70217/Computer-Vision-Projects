from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io as spio
import os
import utils
import numpy as np
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 128)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        # self.batchnorm4 = nn.BatchNorm1d(10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.batchnorm1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.batchnorm2(self.conv2(x))), 2))
        # print(x.size)
        x = x.view(-1, 512)
        x = F.relu(self.batchnorm3(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # lambda_lr = lambda epoch: 0.95** epoch 
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    # dataTypes = ['digits-normal.mat', 'digits-scaled.mat', 'digits-jitter.mat']
    dataTypes = 'digits-jitter.mat'
    path = os.path.join('../data', dataTypes)
    # data = todict(spio.loadmat(path, struct_as_record=False, squeeze_me=True)['data'])
    data = utils.loadmat(path)
    print(data['x'].shape)
    data_x = np.transpose(data['x'], (2, 0, 1))
    print(data_x.shape)
    data_x = data_x[:,np.newaxis,:,:]
    train_x = data_x[data['set']==1]
    train_y = data['y'][data['set']==1]
    print(train_x.shape)
    tensor_trx = torch.from_numpy(train_x)
    tensor_trx = tensor_trx.float()
    tensor_try = torch.from_numpy(train_y)
    tensor_try = tensor_try.long()
    tr_loader = torch.utils.data.TensorDataset(tensor_trx, tensor_try)
    train_loader = torch.utils.data.DataLoader(tr_loader, batch_size=args.batch_size, shuffle=True)
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    # print(train_loader)
    
    test_x = data_x[data['set']==3]
    test_y = data['y'][data['set']==3]
    print(test_x.shape)
    tensor_tex = torch.from_numpy(test_x)
    tensor_tex = tensor_tex.float()
    tensor_tey = torch.from_numpy(test_y)
    tensor_tey = tensor_tey.long()
    te_loader = torch.utils.data.TensorDataset(tensor_tex, tensor_tey)
    test_loader = torch.utils.data.DataLoader(te_loader, batch_size=args.test_batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=False, transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70, 85, 90], gamma=0.1)
    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)


if __name__ == '__main__':
    main()
