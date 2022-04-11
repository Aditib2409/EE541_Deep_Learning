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

batch_size = 500
transform = transforms.Compose(
    [transforms.ToTensor()])

train_data = torchvision.datasets.CIFAR10(
    root='./',
    train=True,
    download=True,
    transform=transform
)

# Load the data
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

test_data = torchvision.datasets.CIFAR10(
    root='./',
    train=False,
    download=False,
    transform=transform
)

testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

num_train = len(train_data)
num_test = len(test_data)

# get the image and the label
data_iterator = iter(trainloader)
images, labels = data_iterator.next()  # 4D tensor (batch_size, #channels, (size))


def image_show(img):
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))  # put the channels in the end
    plt.show()


classes = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# to view the image
img = torchvision.utils.make_grid(images)
# image_show(img)
# print(', '.join(f'{classes[labels[j]]}' for j in range(batch_size)))


class Network(nn.Module):
    def __init__(self, d, hidden1, hidden2, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, out_dim)
        self.dropout = nn.Dropout(d)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)  # Linear output of the network
        out = self.sigmoid(x)

        return out


dropout = 0.3
hidden_dim1 = 256
hidden_dim2 = 128
input_dim = 3072
output_dim = 10
learning_rate = 0.1
epochs = 100
batch_size = 500
lambda_reg = 0.0001

model = Network(d=dropout, hidden1=hidden_dim1, hidden2=hidden_dim2, in_dim=input_dim, out_dim=output_dim)

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=lambda_reg)
optimizer2 = optim.Adam(model.parameters(), lr=learning_rate
                        , weight_decay=lambda_reg)

EpochLoss = []

for epoch in range(epochs):
    epoch_loss = 0
    for i, data in tqdm(enumerate(trainloader)):
        X, y = data

        optimizer1.zero_grad()

        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer1.step()

        epoch_loss += loss.item()
    EpochLoss.append(epoch_loss)
    print(f'Optimizer: SGD, Epoch: {epoch+1}, loss: {epoch_loss/batch_size:.3f}, Training accuracy: {1-epoch_loss/batch_size:.3f}')
print(f'Finished Training!')

# save our model properties
model_path = './model.pth'
torch.save(model.state_dict(), model_path)


test_correct = 0
test_total = 0

# Loading the trained parameters into the model
model = Network(d=dropout, hidden1=hidden_dim1, hidden2=hidden_dim2, in_dim=input_dim, out_dim=output_dim)
model.load_state_dict(torch.load(model_path))

# Since we only have to do a forward propagation
confusion_matrix = torch.zeros(len(classes), len(classes))
with torch.no_grad():
    for i, testdata in enumerate(testloader):
        images, label = testdata
        test_y_pred = model(images)

        test_total += 1

        print(label.shape, test_y_pred.shape, label[i].shape, test_y_pred[i].shape)
        if torch.max(test_y_pred[i], 0) == label[i]:
            test_correct += 1
        _, test_preds = torch.max(test_y_pred, 1)
        for t, p in zip(label.view(-1), test_preds.view(-1)):
            confusion_matrix[t, p] += 1

accuracy_test = test_correct/(test_total+1) * 100
print(f'Test Accuracy: {accuracy_test:.3f}%')
print(confusion_matrix)


