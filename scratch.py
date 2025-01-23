import torch as th
import torch.nn.functional as F

tens = th.tensor([[ 0.0157,  0.1372, -0.1021, -0.1756]])
mask = th.tensor([[1., 0., 1., 0.]])

print(mask *tens)
# Method 1: masked_fill
masked_tens = tens.masked_fill(mask == 0, float('-inf'))
mts = F.softmax(masked_tens, dim=-1)
print(mts)