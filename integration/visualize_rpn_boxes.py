

import os
import sys
import argparse
from PIL import Image, ImageDraw
import torch
from torchvision import transforms

# --- 경로 설정 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'VIOLA'))

# --- 모듈 임포트 ---
from integration.verify_end2end_from_images import _rpn_viola_boxes, load_episode_frames

def visualize_boxes(image_path: str, output_path: str):
    """
    주어진 이미지에 _rpn_viola_boxes 함수로 추출된 바운딩 박스를 시각화합니다.
    """
    print(f"\n--- Processing image: {image_path} ---")

    # 1. 이미지 로드 및 전처리
    img = Image.open(image_path).convert("RGB")
    original_size = img.size

    preprocess = transforms.Compose([
        transforms.Resize((128, 128)), # VIOLA 모델 입력 크기에 맞춤
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(img).unsqueeze(0).unsqueeze(0) # [B=1, T=1, C, H, W]

    # 2. _rpn_viola_boxes 함수를 사용하여 바운딩 박스 추출
    # 이 함수는 CPU 텐서를 기대합니다.
    print("[PROOF] ==> Calling _rpn_viola_boxes (VIOLA's RPN/Object Detector).")
    boxes_tensor = _rpn_viola_boxes(img_tensor.cpu(), topk=5) # top 5 boxes
    
    # 결과는 [B, T, K, 4] 형태이므로, [K, 4]로 줄입니다.
    boxes = boxes_tensor.squeeze(0).squeeze(0).numpy()

    print(f"Detected {len(boxes)} bounding box proposals.")

    # 3. 원본 이미지에 바운딩 박스 그리기
    draw = ImageDraw.Draw(img)
    for box in boxes:
        x1, y1, x2, y2 = box
        # 모델 입력 크기(128x128)에서 원본 이미지 크기로 스케일링
        x1_scaled = x1 * original_size[0] / 128
        y1_scaled = y1 * original_size[1] / 128
        x2_scaled = x2 * original_size[0] / 128
        y2_scaled = y2 * original_size[1] / 128
        draw.rectangle([x1_scaled, y1_scaled, x2_scaled, y2_scaled], outline="red", width=3)

    # 4. 결과 이미지 저장
    img.save(output_path)
    print(f"Result image saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RPN bounding boxes from VIOLA.")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--output-path", type=str, default="rpn_boxes.png", help="Path to save the output image with bounding boxes.")
    args = parser.parse_args()
    visualize_boxes(args.image_path, args.output_path)

