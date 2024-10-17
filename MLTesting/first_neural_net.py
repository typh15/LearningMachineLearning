import torch
import numpy as np
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from random import sample
import torch.nn.functional as F

# This defines a transformation on the grid of training/test images to tensors
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, )), ])

batch_size = 64

train_set = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True,  transform=transform)
test_set  = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)

print(train_set)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=batch_size, shuffle=True)

# The train_set and test_set objects are tuples (data, label)

class FashionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.hidden3 = nn.Linear(128, 64)
        self.output  = nn.Linear(64, 10)

    def forward(self, x):
        x = F.dropout(F.relu(self.hidden1(x)), p=0.25)
        x = F.dropout(F.relu(self.hidden2(x)), p=0.25)
        x = F.dropout(F.relu(self.hidden3(x)), p=0.25)
        x = F.softmax(self.output(x))
        return x


def model_testing(n, display=True):
    correct = 0
    samples = sample(range(10000), n)
    for k in samples:
        test_image_tensor = test_set[k][0]                                          # This is a 28 by 28 tensor. It needs reshaped before going into the model
        test_image_vector = test_image_tensor.view(test_image_tensor.shape[0], -1)  # We "unwrap" the image, so it matches the dimension of the first layer.
        prediction = model(test_image_vector)
        if int(prediction.argmax()) == test_set[k][1]:
            if display:
                print("Correct: Image", k)
            correct += 1
        if int(prediction.argmax()) != test_set[k][1]:
            if display:
                print("Incorrect: Image", k)
    return correct/n


# Training the model:
def training(epochs, sample_test_num):   # How many rounds of training? epochs
    total_accuracy = [0]
    for i in range(epochs):
        model.train()
        running_loss = 0
        for image, label in train_loader:
            optimizer.zero_grad()
            image = image.view(image.shape[0], -1)
            pred = model(image)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        model.eval()
        accuracy = model_testing(sample_test_num, False)
        print("Epoch: ", i+1)
        print(f'Training loss: {running_loss / len(train_loader):.4f}')
        print("Accuracy out of ", sample_test_num, ":   ", accuracy)
        print("")
        total_accuracy.append(accuracy)
    return total_accuracy


#  Looking at the first n-training images...
def peek(a, b, train = 1):
    if a < b:
        for j in range(a, b + 1):
            if train:
                tensor_image = train_set.data[j]
            else:
                tensor_image = test_set.data[j]
            plt.imshow(tensor_image)
            plt.show()
    if a == b:
        if train:
            tensor_image = train_set.data[a]
        else:
            tensor_image = test_set.data[a]
        plt.imshow(tensor_image)
        plt.show()


# Testing the trained model on the kth image-label pair
def model_sample(k):
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

#print(training(5, 200))


#model_sample(1)

peek(1, 1)

print(train_set.data[1])

plt.imshow(train_set.data[1])
plt.show()