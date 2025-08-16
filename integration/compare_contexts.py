

import os
import sys
import argparse
import requests
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

# --- 경로 설정 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'VIOLA'))

# --- 모듈 임포트 ---
from integration.verify_end2end_from_images import build_viola, _shape_meta
import robomimic.utils.tensor_utils as TensorUtils

def get_context_vector_for_image(viola_model, image_url: str, device: torch.device):
    """
    단일 이미지를 다운로드하여 전처리하고, VIOLA의 encode_fn을 통해 컨텍스트 벡터를 추출합니다.
    """
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        img = Image.open(response.raw).convert("RGB")
    except Exception as e:
        print(f"Failed to download or open image url {image_url}: {e}")
        return None

    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(img).unsqueeze(0).unsqueeze(0) # [B=1, T=1, C, H, W]

    # 더미 배치 생성. bbox는 중앙의 유효한 값으로 고정.
    batch = {
        "obs": {
            "agentview_rgb": img_tensor.to(device),
            "eye_in_hand_rgb": img_tensor.clone().to(device),
            "joint_states": torch.zeros(1, 1, 7, device=device),
            "gripper_states": torch.zeros(1, 1, 2, device=device),
            "gripper_history": torch.zeros(1, 1, 10, device=device),
            "centernet_bbox_20": torch.tensor([32, 32, 96, 96], device=device).float().view(1, 1, 1, 4).repeat(1, 1, 20, 1),
        },
        "actions": torch.zeros(1, 1, 7, device=device),
    }

    with torch.no_grad():
        # time_distributed는 내부적으로 루프를 돌며 encode_fn을 호출합니다.
        # 입력 텐서의 시간 차원(T=1)이 있으므로 그대로 사용합니다.
        context_vector = TensorUtils.time_distributed(batch, viola_model.encode_fn)
    
    # 최종 컨텍스트 벡터는 보통 (B, T, D) 형태이므로, 차원을 줄여 반환합니다.
    return context_vector.squeeze(0).squeeze(0)

def run_comparison(args):
    """
    두 이미지의 컨텍스트 벡터를 비교합니다.
    """
    device = torch.device("cpu")
    viola_model = build_viola(device)
    viola_model.eval()
    print("--- VIOLA model loaded on CPU. ---")

    print(f"\n--- Processing Image 1: {args.url1} ---")
    ctx1 = get_context_vector_for_image(viola_model, args.url1, device)
    if ctx1 is None:
        return
    print(f"Context 1 shape: {ctx1.shape}")

    print(f"\n--- Processing Image 2: {args.url2} ---")
    ctx2 = get_context_vector_for_image(viola_model, args.url2, device)
    if ctx2 is None:
        return
    print(f"Context 2 shape: {ctx2.shape}")

    # 코사인 유사도 계산
    # 벡터 정규화 후 내적
    similarity = F.cosine_similarity(ctx1, ctx2, dim=0)

    print("\n--- Comparison Result ---")
    print(f"Cosine Similarity between the two context vectors: {similarity.item():.4f}")

    if similarity.item() < 0.9:
        print("\n[SUCCESS] The context vectors are significantly different, as expected.")
        print("This proves that the VIOLA perception pipeline generates unique, image-specific context.")
    else:
        print("\n[FAILURE] The context vectors are too similar. The model may not be processing the images correctly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare VIOLA context vectors for two images.")
    parser.add_argument("--url1", type=str, required=True, help="URL of the first image.")
    parser.add_argument("--url2", type=str, required=True, help="URL of the second image.")
    args = parser.parse_args()
    run_comparison(args)
