import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim)
print(scalar.item())

vector = torch.tensor([7, 7])
print(vector)
print(vector.shape)

MATRIX = torch.tensor([[7,8],[9,10]])
print(MATRIX)
print(MATRIX.shape)

TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print(TENSOR)
print(TENSOR.ndim)
print(TENSOR.shape)

# Create a random tensor of size (3, 4)
random_tensor = torch.rand(size=(3, 4,4))
print(random_tensor, random_tensor.dtype)

# Create a range of values 0 to 10
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)

loat_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16) # torch.half would also work
print(loat_16_tensor.device)

tensor_mult = torch.rand(size=(2,4,1),
                         dtype = torch.float32)
print(tensor_mult)
print(tensor_mult*10)
#print(datasets.MNIST)

model = nn.Sequential(
    nn.Linear(3, 4),
    nn.Sigmoid(),
    nn.Linear(4, 2),
    nn.Sigmoid()
)

