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
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size) 
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.output(x)
        
        return torch.sigmoid(x)

random.shuffle(set_x)
random.shuffle(set_y)

input_size = 451632
hidden_size =  64
output_size = 2

EPOCHS = 5

def train_model():
    model = Net(input_size, hidden_size, output_size)
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()

    model = Net(input_size, hidden_size, output_size)

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
    intox = Net(input_size, hidden_size, output_size)
    intox.load_state_dict(torch.load('model.pth'))
    intox.eval()
    with torch.no_grad():
        output = intox(to_tensor(file_path))
    
        output = output.tolist()
        
    print(output)

    if output[0]>output[1]:
        return 0
    else:
        return 1

y = 0
correct = 0
incorrect = 0
for filename in os.listdir(os.path.join(os.getcwd(), 'pieces')):
    res = check_intoxicated(os.path.join('pieces', filename))
    
    if y == 0:
        y += 1
    else:
        y -= 1

    if res == y:
        correct += 1
    else:
        incorrect += 1

total = correct + incorrect
print(correct/incorrect)
