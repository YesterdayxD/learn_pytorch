import torch
import os
import torch.distributed as dist
import sys
from torch.multiprocessing import Process

def run(rank,size):
    tensor_list=[torch.FloatTensor([1]).cuda(rank),torch.FloatTensor([1]).cuda(rank),torch.FloatTensor([1]).cuda(rank)]
    tensor=torch.FloatTensor([rank]).cuda(rank)
    print(tensor)
    dist.all_gather(tensor_list,tensor)
    print(tensor_list)

def run1(rank,size):
    tensor_list=[]
    tensor=torch.FloatTensor([rank]).cuda(rank)
    print(tensor)
    dist.all_reduce(tensor)
    print(tensor)

def init_process(rank,size,fn):
    dist.init_process_group(backend='nccl',init_method="tcp://172.17.0.2:2222",world_size=size,rank=rank)
    fn(rank,size)
if __name__=="__main__":
    size=3
    processes=[]
    for rank in range(size):
        p=Process(target=init_process,args=(rank,size,run1))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

# tensor_list=[]
# print(torch.cuda.device_count())
# for dev_idx in range(3):
# # for dev_idx in range(torch.cuda.device_count()):
#
#     print(dev_idx)
#     tensor_list.append(torch.FloatTensor([1]).cuda(dev_idx))
#     print(tensor_list)
#
# dist.reduce_multigpu(tensor_list,1,dst_tensor=1)
# print(tensor_list)
#
# print("dist.all_reduce_multigpu(tensor_list)")

# tensor_list=[]
# T_tensor_list=[]
# print(torch.cuda.device_count())
# for dev_idx in range(3):
# # for dev_idx in range(torch.cuda.device_count()):
#
#     tensor=torch.FloatTensor([1]).cuda(dev_idx)
#     print(tensor)
#     # print(tensor_list)

# tensor=torch.FloatTensor([1]).cuda(int(sys.argv[1]))
# print(tensor)
# dist.all_reduce(tensor)
# print(tensor)
#
# print("dist.all_reduce_multigpu(tensor_list)")
