# Import necessary Python and PyTorch libraries
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

"""
    TO RUN FOR THE SECOND NETWORK ARCHITECTURE. TOGGLE THE FOLLOWING VARIABLE TO 'TRUE'
    <second = True>
"""
batch_size = 64
transform = transforms.Compose(
    [transforms.ToTensor()])

train_data = torchvision.datasets.FashionMNIST(
    root='./',
    train=True,
    download=True,
    transform=transform
)

# Load the data
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

num_train = len(train_data)


class Network1(nn.Module):
    def __init__(self, hidden1, in_dim, out_dim):
        """
        :param hidden1: hidden layer size
        :param in_dim: input layer size
        :param out_dim: output layer size
        """
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden1)  # Input layer
        self.fc2 = nn.Linear(hidden1, out_dim)  # Hidden layer


    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)  # Linear output of the network

        return out


def plot_weight_histogram(weights):
    """
    :param weights: optimal weights of the training model
    :return:
    """
    plt.hist(weights)


hidden_dim1_1 = 128
input_dim = 784
output_dim = 10
learning_rate = 0.001
epochs = 40
batch_size = 64
lambda_reg = 0.0001


model = Network1(hidden1=hidden_dim1_1, in_dim=input_dim, out_dim=output_dim)

criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_reg)


for epoch in range(epochs):
    train_correct = 0
    epoch_loss = 0
    for i, data in tqdm(enumerate(trainloader)):
        X, y = data

        optimizer.zero_grad()

        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        predicted_class = y_pred.argmax(1)
        train_correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()

        epoch_loss += loss

    print(f'Model: {model}, Optimizer: ADAM, Epoch: {epoch+1}, Epoch loss: {epoch_loss/batch_size:.3f}, Training accuracy: {train_correct/num_train:.3f}')
print(f'Finished Training!')

# save our model properties
model_path1 = './Fashion_model1.pth'
torch.save(model.state_dict(), model_path1)

model_1 = Network1(hidden1=hidden_dim1_1, in_dim=input_dim, out_dim=output_dim)
model_1.load_state_dict(torch.load(model_path1))

input_layer_weights1 = model_1.fc1.weight
hidden_layer1_weights1 = model_1.fc2.weight

plt.figure()
plot_weight_histogram(input_layer_weights1.detach().numpy().reshape(-1))
plt.title(f'input layer for model1')
plt.savefig(f'input_layer_weights_model1.png')
plt.show()

plt.figure()
plot_weight_histogram(hidden_layer1_weights1.detach().numpy().reshape(-1))
plt.title(f'hidden layer for model1')
plt.savefig(f'hidden_layer_weights_model1.png')
plt.show()

