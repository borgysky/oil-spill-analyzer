import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from model import UNet
import argparse
import os

def dice_coefficient(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def load_image(path, size=(624, 320)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    img = cv2.resize(img, size)
    return img

def main():
    parser = argparse.ArgumentParser(description="Evaluate UNet on one image")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--weights", type=str, default="unet_pseudo_oilspill.pth",
                        help="Path to the saved .pth weights")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for pseudo-mask and for binarizing prediction")
    args = parser.parse_args()

    # Проверка файла
    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"Image file '{args.image_path}' not found")

    # Загружаем модель
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # Загружаем и подготавливаем изображение
    img_np = load_image(args.image_path)
    img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).to(device)  # shape (1,1,H,W)

    # Генерируем псевдомаску как «эталон»
    gt_mask_np = (img_np < args.threshold).astype(np.float32)
    gt_mask_t = torch.tensor(gt_mask_np).unsqueeze(0).unsqueeze(0).to(device)

    # Предсказание
    with torch.no_grad():
        pred = model(img_tensor)
    pred_mask_t = (pred > args.threshold).float()

    # Считаем метрики
    dice = dice_coefficient(pred_mask_t, gt_mask_t).item()
    iou  = iou_score(pred_mask_t, gt_mask_t).item()
    print(f"Dice coefficient: {dice:.4f}")
    print(f"IoU score       : {iou:.4f}")

    # Визуализация
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(img_np, cmap='gray')
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(gt_mask_np, cmap='gray')
    plt.title("Pseudo-Ground-Truth")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(pred_mask_t.squeeze().cpu().numpy(), cmap='gray')
    plt.title("Model Prediction")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
