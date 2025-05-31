import torch
import lpips
from torchvision import transforms
from PIL import Image

# 設定 LPIPS 裡使用的裝置：如果有 GPU 就用 GPU，否則用 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 載入 LPIPS model（這裡以 'alex' 為 backbone）
lpips_model = lpips.LPIPS(net='alex').to(device)

# 定義從 PIL Image 到張量的 Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),           # 根據需求調整大小
    transforms.ToTensor(),                   # 轉為 [0,1]
    transforms.Normalize((0.5, 0.5, 0.5),    # LPIPS 要求輸入範圍 [-1, 1]
                         (0.5, 0.5, 0.5))
])

def compute_lpips(pil_img1: Image.Image, pil_img2: Image.Image) -> float:
    # 將 PIL Image 轉成 normalized 張量，並送到指定 device
    img1_tensor = preprocess(pil_img1).unsqueeze(0).to(device)  # shape: [1, 3, 256, 256]
    img2_tensor = preprocess(pil_img2).unsqueeze(0).to(device)  # shape: [1, 3, 256, 256]

    # 計算 LPIPS 距離
    with torch.no_grad():
        dist = lpips_model(img1_tensor, img2_tensor)
    return dist.item()

# 範例：如何使用 compute_lpips 函式
if __name__ == "__main__":
    # 假設有兩張影像檔案 "imageA.jpg" 與 "imageB.jpg"
    imgA = Image.open("imageA.jpg").convert("RGB")
    imgB = Image.open("imageB.jpg").convert("RGB")

    lpips_distance = compute_lpips(imgA, imgB)
    print(f"LPIPS Distance: {lpips_distance:.4f}")