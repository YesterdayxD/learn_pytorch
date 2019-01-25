import sys
import argparse

import torch
# import visdom
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data as Data
import torchvision

print(torch.cuda.is_available())
print(torch.cuda.device_count())

torch.manual_seed(1)
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')



parser = argparse.ArgumentParser()
# parser.add_argument('echo')  # add_argument()
parser.add_argument('--rank', '-r', required=True, type=int)
parser.add_argument('--local_rank', '-loc_r', required=True, type=int)
parser.add_argument('--init_method', '-im', required=True, type=str)
parser.add_argument('--epoch', '-e', required=False, default=10, type=str)
args = parser.parse_args()  # parse_args()
# print(args)
# print(args.rank)
# print(args.init_method)

# master node's ip and port
EPOCH = args.epcoh
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False
dist_url = "tcp://172.31.22.234:23456"

dist_backend = 'nccl'
world_size = 2
print("Initialize Process Group...")

# Running the Code
# ----------------
#
# Unlike most of the other PyTorch tutorials, this code may not be run
# directly out of this notebook. To run, download the .py version of this
# file (or convert it using
# `this <https://gist.github.com/chsasank/7218ca16f8d022e02a9c0deb94a310fe>`__)
# and upload a copy to both nodes. The astute reader would have noticed
# that we hardcoded the **node0-privateIP** and :math:`world\_size=4` but
# input the *rank* and *local\_rank* inputs as arg[1] and arg[2] command
# line arguments, respectively. Once uploaded, open two ssh terminals into
# each node.
#
# -  On the first terminal for node0, run ``$ python main.py 0 0``
#
# -  On the second terminal for node0 run ``$ python main.py 1 1``
#
# -  On the first terminal for node1, run ``$ python main.py 2 0``
#
# -  On the second terminal for node1 run ``$ python main.py 3 1``
#

# dist.init_process_group(backend=dist_backend, init_method=sys.argv[3],
#                         rank=int(sys.argv[1]), world_size=world_size)

dist.init_process_group(backend=dist_backend, init_method=args.init_method,
                        rank=int(args.rank), world_size=world_size)

local_rank = int(args.local_rank)
dp_device_ids = [local_rank]
torch.cuda.set_device(local_rank)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


cnn = CNN().cuda()

cnn = torch.nn.parallel.DistributedDataParallel(cnn, device_ids=dp_device_ids,
                                                output_device=local_rank)

# cnn = nn.DataParallel(cnn, device_ids=[0, 1], output_device=0)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss().cuda()
print(cnn)

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
test_data = torchvision.datasets.MNIST(
    root='./mnist', train=False
)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)

train_loader = Data.DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE,
    shuffle=(train_sampler is None),
    sampler=train_sampler

)

train_x = torch.unsqueeze(train_data.train_data, dim=1).type(
    torch.FloatTensor)[:2000] / 255
train_y = train_data.train_labels[:2000]

test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[
         :2000] / 255
test_y = test_data.test_labels[:2000]

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x, b_y = torch.autograd.Variable(
            b_x.cuda()), torch.autograd.Variable(b_y.cuda())
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        # print('train_loader', train_loader)
        # print('acc', acc)
        train_x = torch.autograd.Variable(train_x.cuda())
        current_output = cnn(train_x)
        current_pred = torch.max(current_output, 1)[
            1].data.cpu().numpy().squeeze()
        real_labels = train_y.numpy()
        acc = sum(current_pred == real_labels) / 2000

        if step % 100 == 0:
            print('epoch', epoch, '|loss', loss.data.cpu().numpy(), '|acc:',
                  acc)
            print(loss.data)
        # vis.line(X=torch.FloatTensor([step]),
        #          Y=loss.data.view([1]),
        #          win='loss', update='append' if step > 0 else None)
        # vis.line(X=torch.FloatTensor([step]),
        #          Y=[acc],
        #          win='acc', update='append' if step > 0 else None)

        # y=[[step, step]]
        # print
        # y=np.array([[loss.data.numpy(),acc]])
        # vis.line(Y=y,X=np.array([[step,step]]),
        #          win='acc', update='append', opts=dict(legend=['Sine', 'Cosine']))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
print("finished!!!!!!!")
