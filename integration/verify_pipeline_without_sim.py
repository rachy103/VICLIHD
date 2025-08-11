

import sys
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 1. 경로 문제 해결: 프로젝트 루트(VICLIHD)를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 2. 필요한 모든 모듈 import
# 에러 없이 import가 성공하는지 확인하는 것만으로도 경로 문제 해결을 검증할 수 있습니다.
from hdp.rk_diffuser.models.diffusion import GaussianDynDiffusion
from rk_diffuser_mod.context_aware_diffusion import ContextAwareGaussianDynDiffusion
from rk_diffuser_mod.context_dataset import ContextRLBenchDataset, context_collate_fn

print("Successfully imported all modules from hdp, VIOLA, and VICLIHD.")

# 3. 시뮬레이터와 실제 데이터셋을 대체할 더미 데이터셋 클래스 정의
class DummyRLBenchDataset(Dataset):
    """RLBenchDataset의 출력을 모방하는 가짜 데이터셋입니다."""
    def __init__(self, num_episodes=10, episode_length=100):
        self.num_episodes = num_episodes
        self.episode_length = episode_length

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        # 실제 데이터셋과 동일한 key와 tensor shape를 갖는 더미 데이터 생성
        return {
            'obs': {
                'agentview_rgb': torch.randn(3, 128, 128),
                # ... 다른 관측 데이터가 필요하다면 여기에 추가
            },
            'action': torch.randn(self.episode_length, 7), # (H, action_dim)
        }

# 4. VIOLA+CLIP의 출력을 시뮬레이션하는 컨텍스트 제공자
def get_viola_clip_context_provider(dataset, context_dim):
    """VIOLA+CLIP의 컨텍스트 벡터 출력을 시뮬레이션합니다."""
    def context_provider():
        print(f"--- Simulating context from VIOLA-CLIP model (Output Dimension: {context_dim}) ---")
        rng = np.random.default_rng(0)
        for _ in range(len(dataset)):
            # (Horizon, Context_Dimension) 형태의 더미 컨텍스트 생성
            yield rng.standard_normal((100, context_dim)).astype(np.float32)
    return context_provider


def main():
    """시뮬레이터 없이 전체 파이프라인의 호환성과 작동을 검증합니다."""
    
    # --- 파라미터 설정 ---
    # VIOLA+CLIP의 예상 출력 차원 (예: 512)
    VIOLA_CLIP_OUTPUT_DIM = 512
    
    # --- 데이터 파이프라인 검증 ---
    print("\n--- 1. Initializing Data Pipeline ---")
    base_dataset = DummyRLBenchDataset(num_episodes=10)
    context_provider_fn = get_viola_clip_context_provider(base_dataset, VIOLA_CLIP_OUTPUT_DIM)
    
    # 컨텍스트가 주입된 최종 데이터셋
    train_ds = ContextRLBenchDataset(base_dataset, context_provider_fn(), agg_fn=None) # agg_fn=None, 이미 (H,D) 형태이므로
    
    # DataLoader가 배치(batch)를 정상적으로 만드는지 확인
    loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=context_collate_fn)
    batch = next(iter(loader))
    print("Successfully created a batch from DataLoader.")
    print(f"Batch keys: {batch.keys()}")
    print(f"Action shape: {batch['action'].shape}")
    print(f"Context shape: {batch['context'].shape}")
    assert batch['context'].shape[-1] == VIOLA_CLIP_OUTPUT_DIM, "Context dimension mismatch!"
    print("Context dimension is compatible.")

    # --- 모델 호환성 검증 ---
    print("\n--- 2. Initializing Models ---")
    # HDP의 기본 확산 모델 (더미 인자로 초기화)
    base_diffusion_model = GaussianDynDiffusion(
        horizon=10, observation_dim=7, dim_mults=[1,2,4], action_dim=7, scene_bounds=[], joint_limits=[],
        n_timesteps=100, loss_type='l2', clip_denoised=True, predict_epsilon=False, hidden_dim=128,
        loss_discount=1.0, condition_guidance_w=1.2, reverse_train=False, conditions=['context_fused'], hard_conditions=[],
        noise_init_method='normal', loss_fn='state_l2', coverage_weight=1.0, detach_reverse=False,
        joint_weight=1.0, robot_offset=[0,0,0], trans_loss_scale=1.0, rot_loss_scale=1.0, diffusion_var='gripper_poses',
        joint_pred_pose_loss=False, joint_loss_scale=1.0, rank_bins=10, backbone='unet', num_encoder_layers=4,
        num_decoder_layers=4, n_head=4, causal_attn=True, depth_proc='pointnet', rgb_encoder='resnet18'
    )
    
    # **핵심 검증**: HDP 래퍼의 context_dim을 VIOLA+CLIP의 출력 차원과 동일하게 설정
    context_aware_model = ContextAwareGaussianDynDiffusion(
        base_diffusion_model, context_dim=VIOLA_CLIP_OUTPUT_DIM, hidden_dim=256)
    print("Successfully initialized HDP model wrapped with context-aware layer.")
    print(f"Wrapper's expected context_dim: {VIOLA_CLIP_OUTPUT_DIM}")

    # --- 최종 파이프라인 검증 (학습 스텝) ---
    print("\n--- 3. Verifying Training Step ---")
    optimizer = torch.optim.AdamW(context_aware_model.parameters(), lr=1e-4)
    
    # DataLoader에서 받은 배치로 손실 계산
    loss, _ = context_aware_model.loss(obs=batch["obs"], action=batch["action"],
                                       cond={"context": batch["context"]})
    print(f"Successfully computed loss: {loss.item():.4f}")
    
    # 역전파 실행
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Successfully completed a full training step (forward + backward pass).")
    
    print("\n[SUCCESS] The VICLIHD pipeline is correctly wired and compatible with the simulated VIOLA+CLIP context dimension.")

if __name__ == "__main__":
    main()
