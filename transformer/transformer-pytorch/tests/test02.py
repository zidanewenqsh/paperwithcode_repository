import torch
a = torch.randn(2,2,3,4)
b = torch.randn(2,2,4,5)
c = torch.matmul(a, b)
print(c.shape)
print(c)

