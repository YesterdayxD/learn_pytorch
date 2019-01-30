import torch
import numpy as np
from torch.autograd import Variable

# np_data = np.arange(6).reshape((2, 3))
# torch_data = torch.from_numpy(np_data)
# tensor2array = torch_data.numpy()
#
# print(
#     '\nnumpy', np_data,
#     '\ntorch', torch_data,
#     '\nnumpy', tensor2array
# )
#
# data = [[1, 2], [3, 4]]
# tensor = torch.FloatTensor(data)
# print(
#     '\nnumpy: ', np.matmul(data, data),
#     '\ntorch: ', torch.mm(tensor, tensor)
# )

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 200)
x = Variable(x)
x_np = x.data.numpy()

y_relu = F.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()

# plt.figure(1, figsize=(8, 6))
# plt.subplot(221)
# plt.plot(x_np, y_relu, c='red', label='relu')
# plt.ylim((-1, 5))
# plt.legend(loc='best')
#
# plt.subplot(222)
# plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
# plt.ylim((-0.2, 1.2))
# plt.legend(loc='best')
#
# plt.subplot(223)
# plt.plot(x_np, y_tanh, c='red', label='tanh')
# plt.ylim((-1.2, 1.2))
# plt.legend(loc='best')
#
# plt.subplot(224)
# plt.plot(x_np, y_softplus, c='red', label='softplus')
# plt.ylim((-0.2, 6))
# plt.legend(loc='best')


# plt.show()

x = torch.Tensor.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
print(x.size())
y = x.pow(2) + 0.2 * torch.rand(x.size())

x, y = Variable(x), Variable(y)

plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()

plt.ion()
plt.show()

for t in range(100):
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(),
                 fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.5)
        
