import torch
import numpy as np
from PIL import Image
import os
from model import UNet

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
    img = Image.open(path).convert("L")
    img = img.resize(size)
    img_np = np.array(img).astype(np.float32) / 255.0
    return img_np

def run_evaluation(image_path, weights, threshold=0.5):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Изображение '{image_path}' не найдено")

    if not os.path.isfile(weights):
        raise FileNotFoundError(f"Параметры модели '{weights}' не найдены")

    original_img = Image.open(image_path)
    original_img_np = np.array(original_img).astype(np.uint8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    img_np = load_image(image_path)
    img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).to(device)  


    gt_mask_np = (img_np < threshold).astype(np.float32)
    gt_mask_t = torch.tensor(gt_mask_np).unsqueeze(0).unsqueeze(0).to(device)


    with torch.no_grad():
        pred = model(img_tensor)
    pred_mask_t = (pred > threshold).float()


    dice = dice_coefficient(pred_mask_t, gt_mask_t).item()
    iou = iou_score(pred_mask_t, gt_mask_t).item()

    return {
        "dice": dice,
        "iou": iou,
        "input_image": original_img_np,  
        "gt_mask": (gt_mask_np * 255).astype(np.uint8),
        "pred_mask": (pred_mask_t.squeeze().cpu().numpy() * 255).astype(np.uint8)
    }