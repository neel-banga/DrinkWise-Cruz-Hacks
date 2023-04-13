import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
import os
import shutil

directory_path = 'pieces'

even_folder_path = 'sober'
odd_folder_path = 'drunk'

os.makedirs(even_folder_path, exist_ok=True)
os.makedirs(odd_folder_path, exist_ok=True)

for filename in os.listdir(directory_path):
    if filename.endswith('.jpg'):
        number = int(filename.replace('piece', '').replace('.jpg', ''))
        
        if number % 2 == 0:
            shutil.move(os.path.join(directory_path, filename), os.path.join(even_folder_path, filename))
        else:
            shutil.move(os.path.join(directory_path, filename), os.path.join(odd_folder_path, filename))


img_transforms = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

training = datasets.ImageFolder('IMAGES', transform=img_transforms)
train_loader = torch.utils.data.DataLoader(training, batch_size=3, shuffle=True)

class Model(nn.Module):
    def __init__(self):

        super().__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(1, 32, (5, 5)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (5, 5)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(112896, 2)
        )

    def forward(self, x):
        return self.convolution(x)

def train_model():
    model = Model()

    opt = optim.Adam(model.parameters(), lr = 0.0001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(20):
        for batch in train_loader:
            X, y = batch
            output = model(X)
            
            loss = loss_fn(output, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f'LOSS: {loss}')
        torch.save(model.state_dict(), 'model.pth')

#train_model()

def check_intoxicated(path):

    model = Model()
    model.load_state_dict(torch.load(path))
    img = Image.open('IMAGES/sober/piece72.jpg')
    img_transformed = img_transforms(img)
    img_batch = img_transformed.unsqueeze(0)  # Add a batch dimension

    output = model(img_batch)
    predicted_class = torch.argmax(output)
    print("Predicted class:", predicted_class.item())
