import numpy as np
import torch

# check device to see if we have a cuda gpu if not use cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

# tensors are the primary data structurs in Neural Networks3
# tensors are multi dimensional arrays
my_tensor = torch.tensor(
    [[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True
)

# print the tensor (2d array)
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# other initializtion methods

# empty tensor 3x3
a = torch.empty(size=(3, 3))
print(a)

# tensor 3x3 with zeroes as placholders
b = torch.zeros(size=(3, 3))
print(b)

# tensor 3x3 with random numbers as values
c = torch.rand(size=(3, 3))
print(c)

# tensor 3x3 with random numbers as values
d = torch.ones(size=(3, 3))
print(d)

# tensor 5x5 identity matrix 1's diagonnally the rest are 0's
e = torch.eye(5, 5)
print(e)

# tensor steping by one up to 5
f = torch.arange(start=0, end=5, step=1)
print(f)

# tensor steping by one up to 1 by 10 steps
g = torch.arange(start=1, end=30, step=10)
print(g)

# empty tensor 1x5 normalized with a mean of 0 with starndardization of 1
h = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
print(h)

# empty tensor 1x5 uniform with a mean of 0 with starndardization of 1
i = torch.empty(size=(1, 5)).uniform_(0, 1)
print(i)

# empty tensor 1x5 uniform with a mean of 0 with starndardization of 1
j = torch.diag(torch.ones(3))
print(j)


# how to initalize and convert tensors into other types()int, float, double)
k = torch.arange(4)
print(k.bool())
print(k.short())
print(k.long())
print(k.half())
print(k.float())
print(k.double())


# array to tensor and vice-versa
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()
print(np_array)
print(tensor)
print(np_array_back)
