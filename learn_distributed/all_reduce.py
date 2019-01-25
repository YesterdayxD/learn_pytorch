import torch
import torch.distributed as dist
import sys
dist.init_process_group(backend='nccl',init_method="tcp://172.17.0.2:2222",world_size=2,rank=sys.argv[1])
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

tensor=torch.FloatTensor([1]).cuda(int(sys.argv[1]))
print(tensor)
dist.all_reduce(tensor)
print(tensor)

print("dist.all_reduce_multigpu(tensor_list)")
