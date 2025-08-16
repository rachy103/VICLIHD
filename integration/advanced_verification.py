

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 경로 설정 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'VIOLA'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'hdp'))

# --- 모듈 임포트 ---
from integration.verify_end2end_from_images import (
    build_viola, list_episodes_with_cams, load_episode_frames, extract_context_seq
)
from rk_diffuser_mod.context_aware_diffusion import ContextAwareGaussianDynDiffusion
from hdp.rk_diffuser.models.diffusion import GaussianDynDiffusion
import robomimic.utils.tensor_utils as TensorUtils

# --- 더미 데이터 생성 (개선된 버전) ---
def build_structured_batch(root: str, B: int, T: int, device: torch.device):
    eps = list_episodes_with_cams(root)
    if len(eps) < B:
        print(f"[WARN] Only {len(eps)} episodes found; reducing batch to match.")
        B = len(eps)
    overhead_list = []
    wrist_list = []
    for i in range(B):
        overhead_list.append(load_episode_frames(eps[i]['overhead'], T))
        wrist_list.append(load_episode_frames(eps[i]['wrist'], T))
    overhead = torch.stack(overhead_list, dim=0).to(device) if overhead_list else torch.zeros(B, T, 3, 128, 128, device=device)
    wrist = torch.stack(wrist_list, dim=0).to(device) if wrist_list else overhead
    # assign views: agentview=overhead, eye_in_hand=wrist
    agent = overhead
    eye = wrist

    # 2. 가상의 부드러운 궤적(action) 생성
    t_steps = torch.linspace(0, 2 * np.pi, T, device=device)
    x = 0.5 * torch.sin(t_steps)
    y = 0.5 * torch.cos(t_steps)
    z = torch.linspace(0, 0.2, T, device=device)
    gripper = torch.ones(T, device=device) # Gripper open
    
    # [T, 7] 형태의 action 생성
    actions = torch.zeros(T, 7, device=device)
    actions[:, 0] = x
    actions[:, 1] = y
    actions[:, 2] = z
    actions[:, 6] = gripper
    
    # 배치 차원 추가
    actions = actions.unsqueeze(0).repeat(B, 1, 1)
    
    # 3. 나머지 더미 데이터 생성
    batch = {
        "obs": {
            "agentview_rgb": agent,
            "eye_in_hand_rgb": eye,
            "joint_states": torch.randn(B, T, 7, device=device) * 0.1,
            "gripper_states": torch.randn(B, T, 2, device=device) * 0.1,
            "gripper_history": torch.randn(B, T, 10, device=device) * 0.1,
            "centernet_bbox_20": torch.tensor([32, 32, 96, 96], device=device).float().view(1, 1, 1, 4).repeat(B, T, 20, 1),
        },
        "actions": actions,
    }
    return batch

# --- HDP 모델 빌드 ---
def build_hdp(D: int, device: torch.device, horizon: int):
    base = GaussianDynDiffusion(
        horizon=horizon, observation_dim=7, dim_mults=[1, 2], action_dim=7,
        scene_bounds=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], joint_limits=[], n_timesteps=50, loss_type='l2',
        clip_denoised=True, predict_epsilon=True, hidden_dim=128,
        loss_discount=1.0, condition_guidance_w=1.0, reverse_train=False,
        conditions=['context_fused'], hard_conditions=[],
        noise_init_method='normal', loss_fn='l2', coverage_weight=1.0,
        detach_reverse=False, joint_weight=1.0, robot_offset=[0,0,0],
        trans_loss_scale=1.0, rot_loss_scale=1.0, diffusion_var='gripper_poses',
        joint_pred_pose_loss=False, joint_loss_scale=1.0, rank_bins=10,
        backbone='unet', num_encoder_layers=2, num_decoder_layers=2,
        n_head=2, causal_attn=True, depth_proc='pointnet', rgb_encoder='resnet18'
    ).to(device)
    wrapper = ContextAwareGaussianDynDiffusion(base, context_dim=D, hidden_dim=256).to(device)
    return wrapper

# --- 메인 실행 함수 ---

def extract_context_seq_frame_by_frame(viola_model, batch):
    """Process frames one by one to conserve memory."""
    B, T = batch["obs"]["agentview_rgb"].shape[:2]
    D_context = viola_model.projection.out_dim # Get context dim from model

    all_ctx = []
    for i in range(B):
        ep_ctx = []
        for j in range(T):
            # Create a batch for a single frame
            single_frame_batch = {
                "obs": {k: v[i:i+1, j:j+1] for k, v in batch["obs"].items()},
                "actions": batch["actions"][i:i+1, j:j+1]
            }
            with torch.no_grad():
                # Pass the single-timestep batch to the model. It has shape [B=1, T=1, ...]
                enc = viola_model.encode_fn(single_frame_batch)
            
            ctx = enc.mean(dim=1) if enc.dim() == 3 else enc # [1, D] or [1, N, D]
            if ctx.dim() == 3:
                ctx = ctx.mean(dim=1)

            ep_ctx.append(ctx)
        all_ctx.append(torch.cat(ep_ctx, dim=0))
    
    return torch.stack(all_ctx, dim=0) # [B, T, D]

def run_advanced_verification(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Advanced Verification using device: {device} ---")

    # 1. VIOLA 모델 로드
    viola = build_viola(device)

    # 2. 구조화된 데이터 배치 생성
    batch = build_structured_batch(args.root, args.batch_size, args.frames, device)
    print(f"Structured batch created. Action shape: {batch['actions'].shape}")

    # 3. VIOLA로 컨텍스트 추출 (검증된 함수 사용)
    ctx_seq = extract_context_seq(viola, batch)
    B, T, D = ctx_seq.shape
    print(f"Context extracted from VIOLA. Shape: ({B}, {T}, {D})")

    # 4. HDP 모델 빌드 및 학습 준비
    hdp = build_hdp(D, device, horizon=args.frames)
    optimizer = torch.optim.AdamW(hdp.parameters(), lr=1e-4)
    
    print("\n--- Starting short training loop ---")
    for step in tqdm(range(args.steps), desc="Training"):
        optimizer.zero_grad()
        loss, _ = hdp.loss(obs=batch["obs"], action=batch["actions"], cond={"context": ctx_seq})
        loss.backward()
        optimizer.step()
    print("Training finished.")

    # 5. 학습된 모델로 궤적 생성 (추론)
    print("\n--- Generating trajectory from trained model ---")
    hdp.eval()
    with torch.no_grad():
        # 첫 번째 데이터 샘플의 시작/끝점을 조건으로 사용
        conditions = {
            0: batch["actions"][0, 0, :].unsqueeze(0),
            -1: batch["actions"][0, -1, :].unsqueeze(0),
            "context": ctx_seq[0, :, :].unsqueeze(0)
        }
        result = hdp.conditional_sample(cond=conditions, horizon=T)
        generated_traj = result['gripper_poses']['traj'].squeeze(0).cpu().numpy()

    # 6. 결과 평가 및 시각화
    print("\n--- Evaluating and visualizing results ---")
    ground_truth_traj = batch["actions"][0].cpu().numpy()
    
    # 정량적 평가 (MSE)
    mse = np.mean((ground_truth_traj[:, :3] - generated_traj[:, :3])**2)
    print(f"Mean Squared Error (Position): {mse:.6f}")

    # 정성적 평가 (3D 플롯)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Ground Truth 궤적
    ax.plot(ground_truth_traj[:, 0], ground_truth_traj[:, 1], ground_truth_traj[:, 2], 'b-', label='Ground Truth Trajectory')
    ax.scatter(ground_truth_traj[0, 0], ground_truth_traj[0, 1], ground_truth_traj[0, 2], c='green', s=100, marker='o', label='Start')
    ax.scatter(ground_truth_traj[-1, 0], ground_truth_traj[-1, 1], ground_truth_traj[-1, 2], c='red', s=100, marker='x', label='End')

    # 생성된 궤적
    ax.plot(generated_traj[:, 0], generated_traj[:, 1], generated_traj[:, 2], 'r--', label='Generated Trajectory')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Trajectory Comparison')
    
    plot_path = os.path.join(os.path.dirname(args.root), 'trajectory_comparison.png')
    plt.savefig(plot_path)
    print(f"Saved trajectory comparison plot to: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--frames', type=int, default=32)
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()
    run_advanced_verification(args)
