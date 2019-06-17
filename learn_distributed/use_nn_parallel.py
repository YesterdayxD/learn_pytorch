#!/usr/bin/env python
import os
import time
from math import ceil
from random import Random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

os.environ["CUDA_VISIBLE_DEVICES"]="1,2"



class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class Net(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        #self.count=0

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # self.count+=1
        # print(self.count)
        return F.log_softmax(x)


def partition_dataset():
    """ Partitioning MNIST """
    dataset = datasets.MNIST(
        '/home/limk/pytorch_distributed/mnist',
        train=True,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    size = 2
    bsz = 128 / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(0)
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=int(bsz), shuffle=True)
    return train_set, bsz


#
# def average_gradients(model):
#     """ Gradient averaging. """
#     size = float(dist.get_world_size())
#     for param in model.parameters():
#         # print(param.grad.data)
#         # print(type(param.grad.data))
#         dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
#
#         param.grad.data /= size


def run():
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net().cuda()
    model = torch.nn.DataParallel(model, device_ids=[0, 1], output_device=0)
    print(model)
    #    model = model.cuda(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            data, target = Variable(
                data.cuda()), Variable(
                target.cuda())
            # data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            # print(' computing... on decive cuda: ',torch.cuda.current_device())
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            # average_gradients(model)
            optimizer.step()
        print('epoch ', epoch, ': ',
              epoch_loss / num_batches,
              ', cuda: ', torch.cuda.current_device())
        # 文件夹必须存在 路径最好用绝对路径
        torch.save(model.module.state_dict(),"/home/limk/pytorch_distributed/model/model{}.pth".format(epoch))


if __name__ == "__main__":
    start_time = time.time()
    run()
    end_time = start_time - time.time()
    print('It costs {}s in total.'.format(end_time))
