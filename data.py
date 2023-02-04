from PIL import Image
import os
import numpy as np
import shutil
import torch
import torchvision.transforms as transforms

x = 0
y = 0

set_y = []

# Seperate big file into four seperate peices

piece_path = os.path.join(os.getcwd(), 'pieces')

for filename in os.listdir(os.path.join(os.getcwd(), 'drunkImages')):
    if not filename.startswith('.'):
        f = os.path.join(os.getcwd(), 'drunkImages', filename)
        if os.path.isfile(f):
            im = Image.open(f)

            # Calculate the size of each piece
            width, height = im.size
            piece_width = width // 2
            piece_height = height // 2

            # Create the pieces by cropping the image
            pieces = [im.crop((i * piece_width, j * piece_height, (i + 1) * piece_width, (j + 1) * piece_height)) for i in range(2) for j in range(2)]

            # Save the pieces
            if not os.path.exists('pieces'):
                os.makedirs('pieces')

            for i, piece in enumerate(pieces):
                x += 1
                if x %4 == 0 or x %4 == 1 or x == 1:
                    piece.save(os.path.join('pieces', "piece{}.jpg".format(x)))
                    #set_y.append(y) # 0 or 1
                    
                    if y == 0:
                        set_y.append(torch.tensor([1, 0], dtype=torch.float32).clone().detach())
                        y+=1
                    else:
                        set_y.append(torch.tensor([0, 1], dtype=torch.float32).clone().detach())
                        y-=1

#set_y = torch.tensor(set_y)

piece_path = os.path.join(os.getcwd(), 'pieces')

tensors = []

transform = transforms.ToTensor()

for filename in os.listdir(piece_path):
    img = Image.open(os.path.join(piece_path, filename))
    tensors.append(transform(img))

set_x = []

for i in tensors:
    set_x.append(torch.flatten(i))

set_x = torch.nn.utils.rnn.pad_sequence(set_x, batch_first=True)

def get_datasets():
    return set_x, set_y

# Delete old drunk img dataset - comment for now

'''
path = os.path.join(os.getcwd(), 'drunkImages')

if os.path.exists(path):
    shutil.rmtree(path)
'''