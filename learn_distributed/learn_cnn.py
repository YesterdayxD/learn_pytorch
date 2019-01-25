import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())

import torch.nn as nn
import torch.utils.data as Data
import torchvision
# import visdom
import numpy as np
torch.manual_seed(1)

EPOCH = 10000
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
test_data = torchvision.datasets.MNIST(
    root='./mnist', train=False
)

train_loader = Data.DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE,
    shuffle=True,
)

train_x = torch.unsqueeze(train_data.train_data, dim=1).type(torch.FloatTensor)[:2000] / 255
train_y = train_data.train_labels[:2000]

test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000] / 255
test_y = test_data.test_labels[:2000]


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


cnn=CNN().cuda()
cnn = nn.DataParallel(cnn,device_ids=[0,1],output_device=0)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
print(cnn)

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x, b_y = torch.autograd.Variable(b_x.cuda()), torch.autograd.Variable(b_y.cuda())
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        # print('train_loader', train_loader)
        # print('acc', acc)
        train_x = torch.autograd.Variable(train_x.cuda())
        current_output = cnn(train_x)
        current_pred = torch.max(current_output, 1)[1].data.cpu().numpy().squeeze()
        real_labels = train_y.numpy()
        acc = sum(current_pred == real_labels) / 2000

        if step%100==0:
            print('epoch',epoch,'|loss', loss.data.cpu().numpy(), '|acc:', acc)
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