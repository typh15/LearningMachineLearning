import torch
import numpy as np
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys

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
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)
        self.activation = nn.ReLU()
        self.log_softmax = nn.LogSoftmax()
        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.output(x)
        output = self.log_softmax(x)
        return output


#   Instantiating the Network:
model = FashionNetwork()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

# Training the model:

epoch = int(str(sys.argv[1]))   # How many rounds of training?

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


#  An example of looking at the first n-training images...
def peek(n):
    for j in range(n):
        tensor_image = train_set.data[j]
        plt.imshow(tensor_image)
        plt.show()

# Testing the model
model.eval()

test_run_count = 20
fail_count = 0
pass_count = 0
# Testing the trained model on the first image-label pair
for i in range(test_run_count):
    test_image = test_set[i][0].view(test_set[i][0].shape[0], -1)   # We need to "unwrap" the image, so it matches the dimension
    prediction = model(test_image)                                  # of the first layer.
    if torch.argmax(prediction).item() == test_set[i][1]:
        pass_count = pass_count + 1
    else:
        fail_count = fail_count + 1
        print("Failed prediction: ", prediction)
        print("Expected result: ", test_set[i][1])

print("Tests: ", test_run_count)
print("Pass: ", pass_count)
print("Fail: ", fail_count)


# What is it trying to classify?
tensor_image0 = test_set.data[0]
plt.imshow(tensor_image0)
plt.show()  # It's a shoe lol




