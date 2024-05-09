# -*- coding: utf-8 -*-
"""pytorch_basics.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rRB8urQeJyDmayvfLSRwUcN65iELUCTN
"""

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'digit-recognizer:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-competitions-data%2Fkaggle-v2%2F3004%2F861823%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240508%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240508T130936Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D4bef2da599225ed7316c471b5977377be00450ba464703e7b15357bb73e2c027e1e4c09c39b6919bde49bb1f4f4a8e95ebab9b77266472ae63c98064f3419389bc56125d39061c17939286f218b4ced62f5a030e6551abd3ddc3158323198e573054654789dab8c82aa21c86863220779b3d31cf1e674aec768aea3a70bcf1d8b1d18045851857f71af6189966e5f4f511d020dd9c757280f93da5f064c2f333a9724f4257523d941945a457890e2037ee5e2d5a0fe8ef8603a693dd4bc5cd14635b33223ab017ece8bc54509b90becda9c85374d000664b783216d36e683026afff6851894add6c89a4ebf3bdefcb149c0bbea2b00b8248759ae480c0da2235'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

"""# PyTorch Basics

- Notebook for demonstrating key attributes of PyTorch
- Sources:

https://www.kaggle.com/code/kanncaa1/pytorch-tutorial-for-deep-learning-lovers
https://pytorch.org/docs/stable/index.html
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/digit-recognizer"))

# Any results you write to the current directory are saved as output.

import torch

"""### Matrices in PyTorch

- Attributes (type and shape)
- Creating special matrices
- Conversion with numpy
- Matrix operations

"""

tensor = torch.Tensor([[1,2,4], [4,5,6]])
print(f"Type: {tensor.type}")
print(f"Shape: {tensor.shape}")
print(tensor)

print(torch.ones((2,3)))
print(torch.rand(2,3))

arr = np.random.rand(2,3)
from_np_to_tensor = torch.from_numpy(arr)
from_tensor_to_np = from_np_to_tensor.numpy()

print(from_np_to_tensor)
print(from_tensor_to_np)

# create tensor
tensor = torch.rand(3,3)
print("\n",tensor)

# Resize
print("{}{}\n".format(tensor.view(9).shape,tensor.view(9)))

# Addition
print("Addition: {}\n".format(torch.add(tensor,tensor)))

# Subtraction
print("Subtraction: {}\n".format(tensor.sub(tensor)))

# Element wise multiplication
print("Element wise multiplication: {}\n".format(torch.mul(tensor,tensor)))

# Element wise division
print("Element wise division: {}\n".format(torch.div(tensor,tensor)))

# Mean
print("Mean: {}".format(tensor.mean()))

# Standard deviation (std)
print("std: {}".format(tensor.std()))

"""### Gradients in PyTorch

- Tensors can accumulate gradients from backpropagation
"""

x = torch.tensor([2., 3.], requires_grad=True)
y = x**3
print("y = ", y)

s = sum(y)
s.backward()
print("gradients: ", x.grad)
print("3x squared: ", 3*x**2)

"""### Linear Regression

- Use tensors, loss function, gradients and optimizer to fit data
"""

car_prices = np.linspace(3, 9, 7)
car_prices = torch.tensor(car_prices, dtype=torch.float32).view(-1, 1)
print(car_prices)

car_sales = np.linspace(7.5, 4.5, 7)
car_sales = torch.tensor(car_sales, dtype=torch.float32).view(-1, 1)
print(car_sales)

plt.scatter(car_prices, car_sales)
plt.xlabel("Car Price ($)")
plt.ylabel("Cars Sold")
plt.title("Car Demand")
plt.show()

import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, input_size, output_dim):
        super().__init__()
        self.lin = nn.Linear(input_size, output_dim)

    def forward(self, x):
        return self.lin(x)

input_dim = 1
output_dim = 1
model = LinearRegression(input_dim, output_dim)

mse = nn.MSELoss()

learning_rate = 2e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training
loss_list = []
preds = []
max_iters = 1001
for i in range(max_iters):
    optimizer.zero_grad()
    res = model(car_prices)
    loss = mse(res, car_sales)
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())

    # print loss
    if(i % 200 == 0):
        print(f"epoch {i}, loss {loss.item()}")
        preds.append(model(car_prices).detach().numpy())

plt.plot(range(max_iters),loss_list)
plt.xlabel("Number of Iterations")
plt.ylabel("Loss")
plt.show()

plt.scatter(car_prices, car_sales,label = "original data",color ="red")
for i, pred in enumerate(preds):
    plt.scatter(car_prices, pred,label = "predicted data " + str(i))

plt.legend()
plt.xlabel("Car Price ($)")
plt.ylabel("Car Sales")
plt.title("Car Demand")
plt.show()

"""### Logistic Regression

- MNIST dataset
- Multiclass classification with linear layer and softmax output function
"""

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

train = pd.read_csv(r"../input/digit-recognizer/train.csv", dtype=np.float32)
targets = train.label.values
features = train.loc[:, train.columns != "label"].values / 255 # Normalized

features_train, features_test, targets_train, targets_test = train_test_split(features,
                                                                              targets,
                                                                              test_size=0.2,
                                                                              random_state=42)

batch_size = 100
max_iters = 10000
num_epochs = int(max_iters / (len(features_train) / batch_size))

train = torch.utils.data.TensorDataset(torch.tensor(features_train), torch.tensor(targets_train).type(torch.LongTensor))
test = torch.utils.data.TensorDataset(torch.tensor(features_test), torch.tensor(targets_test).type(torch.LongTensor))

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

# Visualize an example
plt.imshow(features[10].reshape(28,28))
plt.axis('off')
plt.title(str(targets[10]))
plt.savefig('graph.png')
plt.show()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def train_model(model, learning_rate, input_size):
    print("Running on device: ", device)
    m = model.to(device)

    error = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            x_train = images.view(input_size)
            x_train, labels = x_train.to(device), labels.to(device)
            optim.zero_grad()
            out = model(x_train)
            loss = error(out, labels)
            loss.backward()
            optim.step()
            count +=1

            if count % 50 == 0:
                correct = 0
                total = 0
                with torch.no_grad():
                    model.eval()
                    for images, labels in test_loader:
                        x_test = images.view(input_size)
                        x_test, labels = x_test.to(device), labels.to(device)
                        out = model(x_test)
                        preds = torch.max(out, 1)[1]
                        total += len(labels)
                        correct += torch.sum(preds == labels)
                        accuracy = 100 * correct / total
                loss_list.append(loss.item())
                iteration_list.append(count)
                accuracy_list.append(accuracy.item())
                model.train()

            if count % 500 == 0:
                print(f"Count: {count}, Loss: {loss.item()}, Accuracy: {accuracy}")

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(iteration_list, loss_list, 'r-')
    ax2.plot(iteration_list, accuracy_list, 'b-')

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    plt.show()

class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_dim):
        super().__init__()
        # Note, logistic/softmax function is in the loss calculation
        self.lin = nn.Linear(input_size, output_dim)

    def forward(self, x):
        return self.lin(x)

input_dim = 28 * 28
input_size = (-1, input_dim) # batch x 28 x 28 pixels
output_dim = 10 # Digits 0-9
learning_rate = 2e-2

model = LogisticRegression(input_dim, output_dim)
train_model(model, learning_rate, input_size)

"""### Artificial Neural Network

- Three hidden layers with different activation functions
"""

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

input_dim = 28 * 28
input_size = (-1, input_dim) # batch x 28 x 28 pixels
hidden_dim = 150
output_dim = 10
learning_rate = 2e-2

model = NeuralNetwork(input_dim, hidden_dim, output_dim)
train_model(model, learning_rate, input_size)

"""### Convolutional Neural Network

- CNN with convolution, max pooling, and output layers
"""

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net = nn.Sequential(
            # 28 x 28
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0),
            # 24 x 24
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 12 x 12
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0),
            # 8 x 8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 4 x 4
        )
        self.lin_out = nn.Linear(4 * 4 * 32, 10)

    def forward(self, x):
        out = self.conv_net(x)
        out = out.view(out.size(0), -1) # Keep the batch dimension in axis 0
        out = self.lin_out(out)
        return out

input_size = (batch_size, 1, 28, 28) # batch x 1 channel x 28 x 28 pixels
learning_rate = 2e-2
model = ConvolutionalNeuralNetwork()
train_model(model, learning_rate, input_size)

# 3 min on CPU CoLab. 1 min per 1000 iters
# 35 sec on TPU CoLab
# 15 sec on T4 GPU CoLab

"""### Recurrent Neural Network

- Build the RNN with basic PyTorch
- The RNN sequence is each row of pixels in the image
"""

class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_state_dim, output_dim):
        super().__init__()
        self.hidden_net = nn.Sequential(
            nn.Linear(input_dim + hidden_state_dim, hidden_state_dim),
            nn.ReLU(),
        )
        self.out_net = nn.Sequential(
            nn.Linear(hidden_state_dim, output_dim),
        )

    def forward(self, x):
        batch_dim, seq_len, _ = x.size()
        x = x.transpose(0,1)
        self.hidden_state = torch.zeros(batch_dim, hidden_state_dim).to(device)
        for s in range(seq_len):
            combined_in = torch.cat((x[s], self.hidden_state), 1)
            self.hidden_state = self.hidden_net(combined_in)
        out = self.out_net(self.hidden_state)
        return out

input_dim = 28
seq_dim = 28
input_size = (-1, seq_dim, input_dim)
hidden_dim = 20
hidden_state_dim = 100
output_dim = 10
learning_rate = 2e-2

model = RecurrentNeuralNetwork(input_dim, hidden_dim, hidden_state_dim, output_dim)
train_model(model, learning_rate, input_size)

"""### PyTorch Built-in RNN

- Use the existing PyTorch RNN component with 1 layer
"""

class RNNTorch(nn.Module):
    def __init__(self, input_dim, hidden_state_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_state_dim = hidden_state_dim
        self.layer_dim = layer_dim

        self.rnn = nn.RNN(input_dim, hidden_state_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_state_dim, output_dim)

    def forward(self, x):
        batch_dim, seq_dim, _ = x.size()
        h0 = torch.zeros(self.layer_dim, batch_dim, self.hidden_state_dim).to(device)
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1])
        return out

input_dim = 28
seq_dim = 28
input_size = (-1, seq_dim, input_dim)
hidden_state_dim = 100
layer_dim = 1
output_dim = 10
learning_rate = 2e-2

model = RNNTorch(input_dim, hidden_state_dim, layer_dim, output_dim)
train_model(model, learning_rate, input_size)

"""### LSTM with PyTorch

- Use basic PyTorch components to build LSTM
"""

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_state_dim, cell_state_dim, output_dim):
        super().__init__()
        self. hidden_state_dim = hidden_state_dim
        self.cell_state_dim = cell_state_dim

        self.forget_gate_net = nn.Sequential(
            nn.Linear(input_dim + hidden_state_dim, cell_state_dim),
            nn.Sigmoid(),
        )
        self.in_gate_net1 = nn.Sequential(
            nn.Linear(input_dim + hidden_state_dim, cell_state_dim),
            nn.Tanh(),
        )
        self.in_gate_net2 = nn.Sequential(
            nn.Linear(input_dim + hidden_state_dim, cell_state_dim),
            nn.Sigmoid(),
        )
        self.out_gate_net = nn.Sequential(
            nn.Linear(input_dim + hidden_state_dim, hidden_state_dim),
            nn.Sigmoid(),
        )
        self.out_net = nn.Sequential(
            nn.Linear(hidden_state_dim, output_dim),
        )

    def forward(self, x):
        batch_dim, seq_len, _ = x.size()
        x = x.transpose(0,1)
        self.hidden_state = torch.zeros(batch_dim, self.hidden_state_dim).to(device)
        self.cell_state = torch.zeros(batch_dim, self.cell_state_dim).to(device)
        for s in range(seq_len):
            combined_in = torch.cat((x[s], self.hidden_state), 1)

            forget_gate = torch.mul(self.cell_state, self.forget_gate_net(combined_in))
            input_gate = torch.mul(self.in_gate_net1(combined_in), self.in_gate_net2(combined_in))
            self.cell_state = torch.add(forget_gate, input_gate)

            output_gate = self.out_gate_net(combined_in)
            self.hidden_state = torch.mul(output_gate, torch.tanh(self.cell_state))

        out = self.out_net(self.hidden_state)
        return out

input_dim = 28
seq_dim = 28
input_size = (-1, seq_dim, input_dim)
hidden_dim = 20
hidden_state_dim = 100
cell_state_dim = 100
output_dim = 10
learning_rate = 2e-2

model = LSTM(input_dim, hidden_state_dim, cell_state_dim, output_dim)
train_model(model, learning_rate, input_size)

"""### PyTorch Built-in LSTM

- Use the existing PyTorch LSTM component with 1 layer
"""

class LSTMTorch(nn.Module):
    def __init__(self, input_dim, hidden_state_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_state_dim = hidden_state_dim
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(input_dim, hidden_state_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_state_dim, output_dim)

    def forward(self, x):
        batch_dim, seq_dim, _ = x.size()
        h0 = torch.zeros(self.layer_dim, batch_dim, self.hidden_state_dim).to(device)
        c0 = torch.zeros(self.layer_dim, batch_dim, self.hidden_state_dim).to(device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1])
        return out

input_dim = 28
seq_dim = 28
input_size = (-1, seq_dim, input_dim)
hidden_state_dim = 100
layer_dim = 1
output_dim = 10
learning_rate = 2e-2

model = RNNTorch(input_dim, hidden_state_dim, layer_dim, output_dim)
train_model(model, learning_rate, input_size)

