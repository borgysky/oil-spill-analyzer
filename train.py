import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from model import UNet
from tqdm import tqdm

class dataset(Dataset):
    def __init__(self, image_dir, threshold=0.5):
        self.image_dir = image_dir
        self.images    = os.listdir(image_dir)
        self.threshold = threshold

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, self.images[idx])
        img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img  = cv2.resize(img, (624, 320))
        mask = (img < self.threshold).astype(np.float32)
        img  = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)
        return torch.tensor(img), torch.tensor(mask)

IMAGE_DIR  = r"C:\Users\konev\Desktop\dataset\train"
BATCH_SIZE = 4
EPOCHS     = 25
LR         = 1e-4
THRESHOLD  = 0.5

dataset    = dataset(IMAGE_DIR, threshold=THRESHOLD)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = UNet().to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

epoch_losses = []

for epoch in range(EPOCHS):
    total_loss = 0
    loader     = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds       = model(imgs)
        loss        = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loader.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(dataloader)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} completed. Avg Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), 'model.pth')
