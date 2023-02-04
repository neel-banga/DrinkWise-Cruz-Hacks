import data
import torch
import torch.nn as nn
import random
import torchvision.transforms as transforms
import os
from PIL import Image

# Fetch datasets from data.py file
set_x, set_y = data.get_datasets()

# Create the model

class Net(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size) 

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.output(x)

        return torch.sigmoid(x)

random.shuffle(set_x)
random.shuffle(set_y)

input_size = 451632
hidden_size = 64
output_size = 2

EPOCHS = 5

def train_model():
    model = Net(input_size, hidden_size, output_size)
    #loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), 0.00000001)

    model = Net(input_size, hidden_size, output_size)

    loss_fn = nn.BCELoss()
    #optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(EPOCHS):

        for i in range(len(set_x)):

            optim.zero_grad()

            outputs = model(set_x[i])
            target = torch.tensor(set_y[i])
            loss = loss_fn(outputs, target)

            loss.backward()
            optim.step()
            print(loss)

    torch.save(model.state_dict(), 'model.pth')

#train_model()

intox = Net(input_size, hidden_size, output_size)
intox.load_state_dict(torch.load('model.pth'))
intox.eval()


def to_tensor(file_path):
    transform = transforms.ToTensor()
    img = Image.open(os.path.join(os.getcwd(), file_path))
    tensor = transform(img)
    flattened_tensor = tensor.flatten()
    target_size = 451632
    tensor_size = flattened_tensor.size(0)
    padding = target_size - tensor_size
    padded_tensor = torch.nn.functional.pad(flattened_tensor, (0, padding), "constant", 0)

    return padded_tensor

def check_intoxicated(file_path):
    output = intox(to_tensor(file_path))
    output = output.tolist()

    print(output)

    if output[0]>output[1]:
        return 0
    else:
        return 1

res = check_intoxicated('normal.jpg')

if res == 0:
    print('NOT INTOXICATED')
else:
    print('INTOXICATED')