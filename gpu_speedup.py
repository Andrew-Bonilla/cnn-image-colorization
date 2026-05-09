import torch
import torch.nn as nn
import time
import cv2
import glob
import numpy as np
from colorizer import Colorizer

torch.set_default_dtype(torch.float32)

img_dir = "/DATA/andrewbonilla/face_images/*.jpg"
files = sorted(glob.glob(img_dir))[:100]  

data = []
for f1 in files:
    img = cv2.imread(f1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img, dtype=torch.float32) / 255.0
    img = img.permute(2, 0, 1)
    data.append(img)

data_tensor = torch.stack(data)

inputs_list = []
for i in range(len(data_tensor)):
    img_np = (data_tensor[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    L = torch.from_numpy(img_lab[:, :, 0]).float().unsqueeze(0) / 255.0
    inputs_list.append(L)

inputs = torch.stack(inputs_list) 

print("Running CPU.")
model_cpu = Colorizer()
model_cpu.eval()

start = time.time()
with torch.no_grad():
    for i in range(0, len(inputs), 10):
        _ = model_cpu(inputs[i:i+10])
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.4f} seconds")

print("Running GPU.")
model_gpu = Colorizer().to('cuda')
model_gpu.eval()
inputs_gpu = inputs.to('cuda')

with torch.no_grad():
    _ = model_gpu(inputs_gpu[:10])

torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for i in range(0, len(inputs_gpu), 10):
        _ = model_gpu(inputs_gpu[i:i+10])
torch.cuda.synchronize()
gpu_time = time.time() - start
print(f"GPU time: {gpu_time:.4f} seconds")

speedup = cpu_time / gpu_time
print(f"Speedup: {speedup:.2f}x")