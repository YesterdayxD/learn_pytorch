import torch
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2)+0.2*torch.rand(x.size())

print(x.size(),y.size())

def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )

    optimizer = torch.optim.SGD(net1.parameters(),lr=0.2)
    loss_func=torch.nn.MSELoss()
    for i in range(500):
        pred=net1(x)
        loss = loss_func(pred,y)
        print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    plt.figure(1,figsize=(10,3))
    plt.subplot(131)
    plt.title('net1')
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),pred.data.numpy(),'r-')
    plt.show()
save()

