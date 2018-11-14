import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)
print(
    '\nx size:', x.size(),
    '\ny size:', y.size(),
)
plt.ion()
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1])
plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.prediction = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.prediction(x)
        return x


net = Net(n_feature=2, n_hidden=10, n_output=2)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()
plt.show()

for t in range(100):
    out = net(x)

    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 2 == 0:
        plt.cla()
        pred = torch.max(F.softmax(out, 1), 1)[1]  # [0]返回最大值[1]返回最大值的行索引
        pred_y = pred.data.numpy().squeeze()  # 去掉为1的维度
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1],
                    s=100, c=pred_y, cmap="RdYlGn")
        acc = sum(pred_y == target_y) / 200
        plt.text(1.5, -4, "acc=%.2f" % acc,
                 fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.5)
plt.ioff()
plt.show()
