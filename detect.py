import torch
import numpy as np
import cv2
from model import UNet
from PIL import Image
import io

MODEL_PATH = 'model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

def analyze_save(image_path, threshold=0.5):
    with open(image_path, "rb") as f:
        pil_img = Image.open(io.BytesIO(f.read()))
        pil_img = pil_img.convert("RGB")
        original_img = np.array(pil_img)[:, :, ::-1]  # RGB → BGR для OpenCV
        img_gray = np.array(pil_img.convert("L"))

    orig_h, original_w = img_gray.shape
    img_norm = img_gray.astype(np.float32) / 255.0
    img_resized = cv2.resize(img_norm, (624, 320))
    img_tensor = torch.tensor(np.expand_dims(img_resized, axis=(0, 1))).to(DEVICE)

    with torch.no_grad():
        pred = model(img_tensor)

    mask = (pred.squeeze().cpu().numpy() > threshold).astype(np.uint8)
    mask = cv2.resize(mask, (original_w, orig_h))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = original_img.copy()
    cv2.drawContours(result, contours, -1, (0, 0, 255), 2)

    return result
