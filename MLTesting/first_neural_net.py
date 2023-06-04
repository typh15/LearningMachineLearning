import torch
import numpy as np
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# This defines a transformation on the grid of training/test images to tensors
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, )), ])

batch_size = 64

train_set = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)


class FashionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.hidden3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)
        self.activation = nn.ReLU()
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.hidden3(x)
        x = self.activation(x)
        x = self.output(x)
        output = self.log_softmax(x)
        return output


#   Instantiating the Network:
model = FashionNetwork()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

# Training the model:

epoch = 6   # How many rounds of training?

for i in range(epoch):
    running_loss = 0
    for image, label in train_loader:
        optimizer.zero_grad()
        image = image.view(image.shape[0], -1)
        pred = model(image)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Training loss: {running_loss / len(train_loader):.4f}')


#  An example of looking at the loaded data:
#    look at first n-training images...
def peek(n):
    for j in range(n):
        tensor_image = train_set.data[i]
        plt.imshow(tensor_image)
        plt.show()







