import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from model import UNet
from tqdm import tqdm
import argparse

class dataset(Dataset):
    def __init__(self, image_dir, threshold=0.5):
        self.image_dir = image_dir
        self.images = [
            img for img in os.listdir(image_dir) 
            if img.lower().endswith(('.jpg', '.jpeg'))
        ]
        self.threshold = threshold

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, self.images[idx])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Ошибка загрузки изображения: {path}")    
        img = img.astype(np.float32) / 255.0
        img = cv2.resize(img, (624, 320))
        mask = (img < self.threshold).astype(np.float32)
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)
        return torch.tensor(img), torch.tensor(mask)

def main():
    parser = argparse.ArgumentParser(description="Обучение нейросети UNet")
    parser.add_argument('--data', required=True, help="Путь к папке с изображениями")
    parser.add_argument('--output', required=True, help="Путь для сохранения модели (model.pth)")
    args = parser.parse_args()

    IMAGE_DIR  = args.data
    SAVE_PATH  = args.output
    BATCH_SIZE = 4
    EPOCHS     = 25
    LR         = 1e-4
    THRESHOLD  = 0.5

    print(f"Загрузка датасета из: {IMAGE_DIR}")
    train_dataset = dataset(IMAGE_DIR, threshold=THRESHOLD)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

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

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Модель сохранена в: {SAVE_PATH}")


if __name__ == "__main__":
    main()
