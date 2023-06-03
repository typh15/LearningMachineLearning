import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.complex64)

shape = (2, 3, )
rand_tensor = torch.rand(shape, dtype=torch.complex64)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Tensor Shape: \n {rand_tensor.shape} \n")
print(f"Tensor Type: \n {rand_tensor.dtype} \n")
print(f"Tensor Location: \n {rand_tensor.device} \n")

m = 4
n = 6
shape2 = (m, n, )
tensor = torch.rand(shape2, dtype=torch.complex64)
print(torch.svd(torch.matmul(tensor, tensor.adjoint())))

if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f"Device tensor is stored on:\n{tensor.device}")

x = torch.full(shape2, 4, dtype=torch.complex64, requires_grad=True)

y = 3*x - 2

print(x)
print(y)

z = 2*x**2 + 3

print(z)

z.backward(torch.ones_like(x))
print(x.grad)

torch.reshape(x, (4, 3, 2,))

x.view((4, 6))













