import torch
import torch.nn as nn
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
    img = img.permute(2, 0, 1)
    data.append(img)

data_tensor = torch.stack(data)
torch.manual_seed(42)
n_total = data_tensor.shape[0]
perm = torch.randperm(data_tensor.shape[0])
n_train = int(0.9 * n_total)
test_data = data_tensor[perm][n_train:]

def to_lab(tensor):
    lab = torch.zeros_like(tensor)
    for i in range(len(tensor)):
        img_np = (tensor[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        lab[i] = torch.from_numpy(img_lab).float().permute(2, 0, 1) / 255.0
    return lab

print("Converting test data to LAB")
test_lab = to_lab(test_data)

test_inputs = test_lab[:, 0:1, :, :].to(device)
test_targets = test_lab[:, 1:3, :, :].to(device)

model = Colorizer().to(device)
model.load_state_dict(torch.load('/DATA/andrewbonilla/colorizer.pth'))
model.eval()

batch_size = 10
criterion = nn.MSELoss()
total_loss = 0.0
n_samples = test_inputs.shape[0]

os.makedirs('/DATA/andrewbonilla/colorized_results', exist_ok=True)

with torch.no_grad():
    for i in range(0, n_samples, batch_size):
        end = i + batch_size
        inputs_batch = test_inputs[i:end]
        targets_batch = test_targets[i:end]

        outputs = model(inputs_batch)
        outputs = (outputs + 1) / 2
        loss = criterion(outputs, targets_batch)
        total_loss += loss.item()

        for j in range(outputs.shape[0]):
            L = (test_inputs[i+j, 0].cpu().numpy() * 255).astype(np.uint8)
            a = (outputs[j, 0].cpu().numpy() * 255).astype(np.uint8)
            b = (outputs[j, 1].cpu().numpy() * 255).astype(np.uint8)

            lab_img = cv2.merge([L, a, b])
            rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'/DATA/andrewbonilla/colorized_results/image_{i+j:04d}.jpg', bgr_img)

print(f"Test MSE: {(total_loss / (n_samples // batch_size)):.6f}")
print("Colorized images saved to /DATA/andrewbonilla/colorized_results/")

print("Saving colorized training examples.")
os.makedirs('/DATA/andrewbonilla/colorized_train_results', exist_ok=True)

train_sample = to_lab(data_tensor[:10])
train_sample_inputs = train_sample[:, 0:1, :, :].to(device)

with torch.no_grad():
    train_outputs = model(train_sample_inputs)
    train_outputs = (train_outputs + 1) / 2

for j in range(10):
    L = (train_sample_inputs[j, 0].cpu().numpy() * 255).astype(np.uint8)
    a = (train_outputs[j, 0].cpu().numpy() * 255).astype(np.uint8)
    b = (train_outputs[j, 1].cpu().numpy() * 255).astype(np.uint8)

    lab_img = cv2.merge([L, a, b])
    rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'/DATA/andrewbonilla/colorized_train_results/image_{j:04d}.jpg', bgr_img)

print("Training colorization examples saved to /DATA/andrewbonilla/colorized_train_results/")

total_a_var = 0.0
total_b_var = 0.0

model.eval()
with torch.no_grad():
    for i in range(0, n_samples, batch_size):
        batch = test_inputs[i:i+batch_size]
        outputs = model(batch)
        outputs = (outputs + 1) / 2
        total_a_var += outputs[:, 0, :, :].var().item()
        total_b_var += outputs[:, 1, :, :].var().item()

avg_a_var = total_a_var / (n_samples // batch_size)
avg_b_var = total_b_var / (n_samples // batch_size)
print(f"Average a* variance (faces): {avg_a_var:.6f}")
print(f"Average b* variance (faces): {avg_b_var:.6f}")