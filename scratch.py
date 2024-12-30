import torch

a = torch.arange(5)
print(a)
test = 5
a[0] = test
print(a)
test = 10
print(a)