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

# The train_set and test_set objects are tuples (data, label)


class FashionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.hidden3 = nn.Linear(128, 128)
        self.hidden4 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.hidden3(x)
        x = self.activation(x)
        x = self.hidden4(x)
        x = self.activation(x)
        x = self.output(x)
        output = self.softmax(x)
        return output


# Training the model:
def training(epochs):   # How many rounds of training? epochs
    for i in range(epochs):
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


#  Looking at the first n-training images...
def peek(n):
    for j in range(n):
        tensor_image = train_set.data[j]
        plt.imshow(tensor_image)
        plt.show()


# Testing the trained model on the kth image-label pair
def test_the_model(k):
    test_image_tensor = test_set[k][0]                                          # This is a 28 by 28 tensor. It needs reshaped before going into the model
    test_image_vector = test_image_tensor.view(test_image_tensor.shape[0], -1)  # We "unwrap" the image, so it matches the dimension of the first layer.
    prediction = model(test_image_vector)

    print("Model Prediction: ", test_set.classes[int(prediction.argmax())])
    print("Actual Label: ", test_set.classes[test_set[k][1]])

    # What image it trying to classify?
    tensor_image0 = test_set.data[k]
    plt.imshow(tensor_image0)
    plt.show()


#   Instantiating the Network:
model = FashionNetwork()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

training(3)
test_the_model(1)


