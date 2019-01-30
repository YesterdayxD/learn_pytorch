import torch

x = torch.Tensor([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]])
y = torch.Tensor([[[5, 2, 3, ], [6, 7, 8, ], [6, 7, 8, ]]])
z = torch.Tensor([0])

# torch.Size([1, 2, 5])
# (10, 5, 1)
# torch.Size([1, 3, 3])
# (9, 3, 1)
# torch.Size([1])
# (1,)

print(x.shape)
print(x.stride())

print(y.shape)
print(y.stride())

print(z.shape)
print(z.stride())

import numpy as np

y = np.reshape(np.arange(2 * 3 * 4), (2, 3, 4))

print(y.strides)
offset = sum(y.strides * np.array((1, 1, 1)))
print(offset)
print(y.itemsize)


# This array is stored in memory as 40 bytes, one after the other
# (known as a contiguous block of memory). The strides of an array tell us
# how many bytes we have to skip in memory
# to move to the next position along a certain axis.
#  For example, we have to skip 4 bytes (1 value) to move to the next column,
# but 20 bytes (5 values) to get to the same position in the next row.
#  As such, the strides for the array x will be (20, 4).




