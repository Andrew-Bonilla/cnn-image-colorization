import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import glob
import os
from colorizer import Colorizer

torch.set_default_dtype(torch.float32)
torch.manual_seed(42)
device = torch.device("cuda")

model = Colorizer().to(device)
model.load_state_dict(torch.load('/DATA/andrewbonilla/colorizer.pth'))
print("Loaded pretrained model from faces dataset")

gray_paths = sorted(glob.glob('/DATA/andrewbonilla/NCDataset/Gray/*/*.jpg'))
color_paths = sorted(glob.glob('/DATA/andrewbonilla/NCColorful/ColorfulOriginal/*/*.jpg'))

inputs_list = []
targets_list = []

for gray_path, color_path in zip(gray_paths, color_paths):
    gray_img = cv2.imread(gray_path)
    gray_img = cv2.resize(gray_img, (128, 128))
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB)
    gray_lab = cv2.cvtColor(gray_img, cv2.COLOR_RGB2LAB)
    L = torch.from_numpy(gray_lab[:, :, 0]).float().unsqueeze(0) / 255.0  
    inputs_list.append(L)

    color_img = cv2.imread(color_path)
    color_img = cv2.resize(color_img, (128, 128))
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    color_lab = cv2.cvtColor(color_img, cv2.COLOR_RGB2LAB)
    ab = torch.from_numpy(color_lab[:, :, 1:3]).float().permute(2, 0, 1) / 255.0 
    targets_list.append(ab)

inputs_tensor = torch.stack(inputs_list)  
targets_tensor = torch.stack(targets_list) 

print(f"Inputs shape: {inputs_tensor.shape}")
print(f"Targets shape: {targets_tensor.shape}")

n_total = inputs_tensor.shape[0]
n_train = int(0.9 * n_total)

train_inputs = inputs_tensor[:n_train].to(device)
train_targets = targets_tensor[:n_train].to(device)
test_inputs = inputs_tensor[n_train:].to(device)
test_targets = targets_tensor[n_train:].to(device)

for param in model.encoder.parameters():
    param.requires_grad = False

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
criterion = nn.MSELoss()

batch_size = 10
epochs = 50
n_samples = train_inputs.shape[0]

print("Fine-tuning on NCD dataset.")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for i in range(0, n_samples, batch_size):
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
        avg_loss = epoch_loss / (n_samples // batch_size)
        print(f'Epoch: [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')

print("Evaluating on NCD test set.")
model.eval()
total_loss = 0.0
os.makedirs('/DATA/andrewbonilla/ncd_results', exist_ok=True)

with torch.no_grad():
    for i in range(0, test_inputs.shape[0], batch_size):
        end = i + batch_size
        inputs_batch = test_inputs[i:end]
        targets_batch = test_targets[i:end]

        outputs = model(inputs_batch)
        loss = criterion(outputs, targets_batch)
        total_loss += loss.item()

        for j in range(outputs.shape[0]):
            L = (test_inputs[i+j, 0].cpu().numpy() * 255).astype(np.uint8)
            a = (outputs[j, 0].cpu().numpy() * 255).astype(np.uint8)
            b = (outputs[j, 1].cpu().numpy() * 255).astype(np.uint8)

            lab_img = cv2.merge([L, a, b])
            rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'/DATA/andrewbonilla/ncd_results/image_{i+j:04d}.jpg', bgr_img)

avg_mse = total_loss / len(range(0, test_inputs.shape[0], batch_size))
print(f"NCD Test MSE: {avg_mse:.6f}")
print(f"Faces Test MSE: 0.000211")
print("Colorized NCD images saved to /DATA/andrewbonilla/ncd_results/")