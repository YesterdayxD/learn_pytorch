"""

"""
"""
Warmstarting Model Using Parameters from a Different Model
SAVE:
torch.save(model.state_dict(), PATH)

LOAD:
modelB = TheModelBClass(*args, **kwargs)
modelB.load_state_dict(torch.load(PATH), strict=False)

Whether you are loading from a partial state_dict, which is missing some keys,
or loading a state_dict with more keys than the model that
you are loading into,you can set the strict argument to
False in the load_state_dict() function to ignore non-matching keys.
"""


"""
Saving torch.nn.DataParallel Models

SAVE:
torch.save(model.module.state_dict(), PATH)

LOAD:
# Load to whatever device you want
"""

import matplotlib.pyplot as plt
import torch

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

print(x.size(), y.size())


def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    optimizer = torch.optim.SGD(net1.parameters(), lr=0.2)
    loss_func = torch.nn.MSELoss()
    for i in range(500):
        pred = net1(x)
        loss = loss_func(pred, y)
        print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), pred.data.numpy(), 'r-')

    torch.save(net1, 'net.pkl')
    torch.save(net1.state_dict(), 'net_params.pkl')


def restore_model():
    net2 = torch.load('net.pkl')
    pred = net2(x)

    plt.subplot(132)
    plt.title('net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), pred.data.numpy(),
             'r-', )


def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    net3.load_state_dict(torch.load('net_params.pkl'))
    pred = net3(x)

    plt.subplot(133)
    plt.title('net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), pred.data.numpy(),
             'r-', )


plt.ion()
save()
restore_model()
restore_params()
plt.pause(5)
plt.show()

