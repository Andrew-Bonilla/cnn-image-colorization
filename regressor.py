import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
import glob

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
        )
        self.final = nn.Linear(3, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.final(x)
        return x

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    print("Beginning the regression program.")

    img_dir = "/DATA/andrewbonilla/face_images/*.jpg"  
    files = glob.glob(img_dir)
    data = []

    for f1 in files:
        img = cv2.imread(f1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img, dtype=torch.float32) / 255.0
        img = img.permute(2, 0, 1)
        data.append(img)

    data_tensor = torch.stack(data)

    perm = torch.randperm(data_tensor.shape[0])
    data_tensor = data_tensor[perm]

    nimages = data_tensor.shape[0]
    augmented_tensor = torch.zeros(nimages * 10, 3, 128, 128)

    for i in range(nimages * 10):
        img = data_tensor[i%nimages].clone()
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
        augmented_tensor[i] = img_out

    os.makedirs('/DATA/andrewbonilla/augmented', exist_ok=True)
    os.makedirs('/DATA/andrewbonilla/L', exist_ok=True)
    os.makedirs('/DATA/andrewbonilla/a', exist_ok=True)
    os.makedirs('/DATA/andrewbonilla/b', exist_ok=True)

    lab_tensor = torch.zeros_like(augmented_tensor)

    for i in range(len(augmented_tensor)):
        img_np = (augmented_tensor[i].permute(1,2,0).numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'/DATA/andrewbonilla/augmented/image_{i:04d}.jpg', img_bgr)

        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        lab_tensor[i] = torch.from_numpy(img_lab).float().permute(2,0,1) / 255.0
        L,a,b = cv2.split(img_lab)
        a_magma = cv2.applyColorMap(a, cv2.COLORMAP_MAGMA)
        b_viridis = cv2.applyColorMap(b, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(f'/DATA/andrewbonilla/L/image_{i:04d}.jpg', L)
        cv2.imwrite(f'/DATA/andrewbonilla/a/image_{i:04d}.jpg', a_magma)
        cv2.imwrite(f'/DATA/andrewbonilla/b/image_{i:04d}.jpg', b_viridis)
    targets = lab_tensor[:, 1:3, :, :].mean(dim=(2,3))
    inputs = lab_tensor[:, 0:1, :, :]

    model = Regressor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch: [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    print("Training finished.")
