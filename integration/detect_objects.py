
import os
import sys
import argparse
import requests
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from torchvision import transforms

# --- 경로 설정 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'VIOLA'))

# --- 모듈 임포트 ---
# build_viola 함수와 shape_meta를 가져옵니다.
from integration.verify_end2end_from_images import build_viola, _shape_meta

def decode_centernet_output(hmap, regs, w_h_, top_k=5, score_thresh=0.3):
    """
    CenterNet 출력을 디코딩하여 바운딩 박스 리스트를 반환하는 간단한 함수.
    """
    b, c, h, w = hmap.shape
    hmap = torch.sigmoid(hmap)

    # NMS와 유사하게 max_pool2d를 사용하여 피크 찾기
    hmap_peak = F.max_pool2d(hmap, 3, stride=1, padding=1)
    keep = (hmap_peak == hmap).float()
    hmap = hmap * keep

    scores, indices = torch.topk(hmap.view(b, c, -1), k=top_k)
    
    # 임계값 이하 점수 필터링
    scores = scores.squeeze(0).squeeze(0)
    indices = indices.squeeze(0).squeeze(0)
    valid = scores > score_thresh
    scores = scores[valid]
    indices = indices[valid]
    if scores.size(0) == 0:
        return []

    # 인덱스를 좌표로 변환
    ys = (indices / w).int().float()
    xs = (indices % w).int().float()

    # regs와 w_h_에서 오프셋과 크기 가져오기
    regs = regs.squeeze(0).permute(1, 0).reshape(-1, 2)
    w_h_ = w_h_.squeeze(0).permute(1, 0).reshape(-1, 2)
    
    regs = regs[indices]
    w_h_ = w_h_[indices]

    xs += regs[:, 0]
    ys += regs[:, 1]

    x1s = (xs - w_h_[:, 0] / 2).clamp(min=0, max=w)
    y1s = (ys - w_h_[:, 1] / 2).clamp(min=0, max=h)
    x2s = (xs + w_h_[:, 0] / 2).clamp(min=0, max=w)
    y2s = (ys + w_h_[:, 1] / 2).clamp(min=0, max=h)

    # 원본 이미지 크기(128x128)에 맞게 스케일링
    # CenterNet의 출력은 다운샘플링되어 있으므로 (기본 4배) 다시 스케일업
    scale = 4
    boxes = torch.stack([x1s, y1s, x2s, y2s], dim=1) * scale
    return boxes.detach().cpu().numpy().tolist()

def run_detection(args):
    """
    단일 이미지에 대해 VIOLA 객체 탐지를 실행하고 결과를 시각화합니다.
    """
    # 1. VIOLA 모델 로드 (CPU 사용)
    device = torch.device("cpu")
    viola_model = build_viola(device)
    viola_model.eval()
    print("VIOLA model loaded on CPU.")

    # 2. 이미지 다운로드 및 전처리
    try:
        response = requests.get(args.url, stream=True)
        response.raise_for_status()
        img = Image.open(response.raw).convert("RGB")
    except Exception as e:
        print(f"Failed to download or open image: {e}")
        return

    original_size = img.size
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(img).unsqueeze(0) # [1, 3, 128, 128]
    
    # 3. 모델의 Perception 부분 실행
    # VisualCore 백본과 BBox 헤드를 직접 호출
    encoder = viola_model.encoder
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        # time_distributed를 모방하기 위해 배치 차원만 있는 텐서로 전달
        features = encoder._backbone(img_tensor)
        bbox_output = encoder._bbox_head(features)
        hmap, regs, w_h_ = bbox_output

    # 4. 출력 디코딩
    import torch.nn.functional as F # 디코딩 함수 내에서 사용
    boxes = decode_centernet_output(hmap, regs, w_h_, top_k=args.top_k, score_thresh=args.threshold)
    print(f"Detected {len(boxes)} boxes.")

    # 5. 결과 시각화
    draw = ImageDraw.Draw(img)
    for box in boxes:
        x1, y1, x2, y2 = box
        # 원본 이미지 크기에 맞게 좌표 변환
        x1 = x1 * original_size[0] / 128
        y1 = y1 * original_size[1] / 128
        x2 = x2 * original_size[0] / 128
        y2 = y2 * original_size[1] / 128
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    img.save(args.output_path)
    print(f"Result saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VIOLA object detection on a single image.")
    parser.add_argument("--url", type=str, required=True, help="URL of the image to process.")
    parser.add_argument("--output-path", type=str, default="bbox_result.png", help="Path to save the output image.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions to consider.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Score threshold for detected boxes.")
    args = parser.parse_args()
    run_detection(args)
