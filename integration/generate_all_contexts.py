

import os
import sys
import argparse
from tqdm import tqdm
import torch
import numpy as np

# 1. 경로 문제 해결
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# 2. 필요한 함수 및 클래스 import
# 기존 검증 스크립트에서 사용했던 함수들을 그대로 가져옵니다.
from integration.verify_end2end_from_images import (
    build_viola,
    list_episodes,
    load_episode_frames,
    build_batch_from_images,
    extract_context_seq,
)

def generate_and_save_contexts(args):
    """
    모든 에피소드에 대한 컨텍스트 벡터를 생성하고 .npy 파일로 저장합니다.
    """
    print("--- Starting Context Generation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. VIOLA+CLIP 모델 로드
    print("Loading VIOLA-CLIP model...")
    viola_model = build_viola(device)
    print("Model loaded.")

    # 2. 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Context files will be saved to: {args.output_dir}")

    # 3. 모든 에피소드 경로 탐색
    all_episodes = list_episodes(args.episode_root)
    print(f"Found {len(all_episodes)} episodes in {args.episode_root}")

    # 4. 각 에피소드를 순회하며 컨텍스트 생성 및 저장
    for i, ep_files in enumerate(tqdm(all_episodes, desc="Processing Episodes")):
        # 에피소드의 모든 프레임을 로드
        # (B=1, T=전체 프레임 수)
        batch = build_batch_from_images(
            root=args.episode_root,
            B=1, # 한 번에 하나의 에피소드만 처리
            T=len(ep_files),
            device=device
        )
        
        # VIOLA 모델을 통해 컨텍스트 시퀀스 추출
        # 결과 shape: [1, T, D_context]
        ctx_seq = extract_context_seq(viola_model, batch)
        
        # 저장할 numpy 배열 준비 (batch 차원 제거)
        ctx_to_save = ctx_seq.squeeze(0).cpu().numpy()
        
        # .npy 파일로 저장
        output_path = os.path.join(args.output_dir, f"ctx_ep_{i:04d}.npy")
        np.save(output_path, ctx_to_save)

    print(f"\n--- Context Generation Complete ---")
    print(f"Successfully saved {len(all_episodes)} context files in {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save context vectors for all episodes.")
    parser.add_argument('--episode-root', type=str, required=True, help='Root directory containing episode image folders.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the output .npy context files.')
    
    args = parser.parse_args()
    generate_and_save_contexts(args)

