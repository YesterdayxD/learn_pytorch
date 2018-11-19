import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

torch.manual_seed(1)

EPOCH = 1
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


cnn = CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_func(output, b_y)

        # print('train_loader', train_loader)
        # print('acc', acc)
        current_output = cnn(train_x)
        current_pred = torch.max(current_output, 1)[1].data.numpy().squeeze()
        real_labels = train_y.numpy()
        acc = sum(current_pred == real_labels) / 2000

        print('loss', loss.data.numpy(), '|acc:', acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

test_output = cnn(test_x[:2000])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
real_out = test_y[:2000].numpy()
print(pred_y, 'prediction number')
print(real_out, 'real number')
print('test_set_acc:', sum(pred_y == real_out) / 2000)
