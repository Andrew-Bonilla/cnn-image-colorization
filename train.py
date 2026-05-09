import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import glob
import os
from colorizer import Colorizer

torch.set_default_dtype(torch.float32)
device = torch.device("cuda")

img_dir = "/DATA/andrewbonilla/face_images/*.jpg"
files = sorted(glob.glob(img_dir))
data = []

for f1 in files:
    img = cv2.imread(f1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img, dtype=torch.float32) / 255.0
    img = img.permute(2,0,1)
    data.append(img)

data_tensor = torch.stack(data)
torch.manual_seed(42)
perm = torch.randperm(data_tensor.shape[0])
data_tensor = data_tensor[perm]

n_total = data_tensor.shape[0]
n_train = int(0.9*n_total)
n_test = n_total - n_train

train_data = data_tensor[:n_train]
test_data = data_tensor[n_train:]

augmented_train = torch.zeros(n_train * 10, 3, 128, 128)

for i in range(n_train * 10):
    img = train_data[i%n_train].clone()
    img_np = img.permute(1,2,0).numpy()

    if torch.rand(1).item() > 0.5:
        img_np = cv2.flip(img_np,1)
    
    crop_size = torch.randint(90,128,(1,)).item()
    x = torch.randint(0, 128 - crop_size, (1,)).item()
    y = torch.randint(0, 128 - crop_size, (1,)).item()
    img_np = img_np[y:y+crop_size, x:x+crop_size]
    img_np = cv2.resize(img_np, (128,128))

    scalar = 0.6 + torch.rand(1).item() * 0.4
    img_np = np.clip(img_np * scalar, 0, 1)

    img_out = torch.tensor(img_np, dtype=torch.float32).permute(2,0,1)
    augmented_train[i] = img_out

def to_lab(tensor):
    lab = torch.zeros_like(tensor)
    for i in range(len(tensor)):
        img_np = (tensor[i].permute(1,2,0).numpy() * 255).astype(np.uint8)
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        lab[i] = torch.from_numpy(img_lab).float().permute(2,0,1) / 255.0

    return lab

print("Converting training to LAB")

train_lab = to_lab(augmented_train)

train_inputs = train_lab[:, 0:1, :, :].to(device)
train_targets = train_lab[:, 1:3, :, :].to(device)
train_targets = train_targets * 2 - 1

model = Colorizer().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 10
epochs = 100
n_samples = train_inputs.shape[0]

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    
    for i in range(0,n_samples,batch_size):
        end = i + batch_size
        inputs_batch = train_inputs[i:end]
        targets_batch = train_targets[i:end]
        
        optimizer.zero_grad()
        outputs = model(inputs_batch)
        loss = criterion(outputs, targets_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch: [{epoch+1}/{epochs}], Loss: {(epoch_loss / (n_samples // batch_size)):.6f}')
        
torch.save(model.state_dict(), '/DATA/andrewbonilla/colorizer.pth')
print("Training finished and model is saved.")
