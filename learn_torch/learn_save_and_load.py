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


"""
在GPU训练，在CPU推理时，保存与加载模型需要注意的事项
model = nn.DataParallel(model).cuda()

执行这个代码之后，model就不在是我们原来的模型，而是相当于在我们原来的模型外面加了
一层支持GPU运行的外壳，这时候真正的模型对象为：real_model = model.module, 
所以我们在保存模型的时候注意，如果保存的时候是否带有这层加的外壳，
如果保存的时候带有的话，加载的时候也是带有的，如果保存的是真实的模型，
加载的也是真是的模型。这里我建议保存真是的模型，因为加了module壳的模型在CPU上是不能运行的。
--------------------- 
SAVE:
real_model = model.module
torch.save(real_model, os.path.join(args.save_path,"cos_mnist_"+str(epoch+1)+"_whole.pth"))

LOAD:
args.weight=checkpoint/cos_mnist_10_whole.pth
map_location = lambda storage, loc: storage
model = torch.load(args.weight,map_location=map_location)

作者：Lavi_qq_2910138025 
来源：CSDN 
原文：https://blog.csdn.net/liuweiyuxiang/article/details/82224374 
版权声明：本文为博主原创文章，转载请附上博文链接！
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
