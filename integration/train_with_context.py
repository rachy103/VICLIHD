# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# --- 경로 문제 해결 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'hdp'))


from rk_diffuser_mod.context_dataset import (
    ContextRLBenchDataset, context_collate_fn
)
from rk_diffuser_mod.context_aware_diffusion import ContextAwareGaussianDynDiffusion

# HDP (원본)
from rk_diffuser.models.diffusion import GaussianDynDiffusion
from rk_diffuser.dataset.rl_bench_dataset import RLBenchDataset

# =========================
# 토글: RLBench 없이 배선 확인
# =========================
USE_DUMMY_BASE = True  # RLBench 세팅 전까지 True로 두세요.
USE_MINI_DIFFUSION = False

# --- 중요 설정값 ---
CONTEXT_DIM = 32  # ★ VIOLA 컨텍스트 벡터의 실제 마지막 차원과 일치해야 합니다.
OBS_DIM = 12      # 더미 데이터셋의 obs 차원
ACT_DIM = 7       # 확산 모델이 예측할 action의 차원
HORIZON = 100


if USE_MINI_DIFFUSION:
    from rk_diffuser_mod.minidiffusion import MiniGaussianDynDiffusion
    model = MiniGaussianDynDiffusion(
        observation_dim=OBS_DIM, action_dim=ACT_DIM,
        context_dim=CONTEXT_DIM, hidden=256
    )
else:
    from rk_diffuser.models.diffusion import GaussianDynDiffusion
    from rk_diffuser_mod.context_aware_diffusion import ContextAwareGaussianDynDiffusion

    # ------ Preset A: Dummy 베이스용 (바로 실행용) ------
    dd_cfg = dict(
        horizon=HORIZON,
        observation_dim=ACT_DIM,
        dim_mults=[1,2,4],
        action_dim=ACT_DIM,
        scene_bounds=[[-1.0,-1.0,-1.0],[1.0,1.0,1.0]],
        joint_limits=[[-3.14,3.14]]*ACT_DIM,
        n_timesteps=50,
        loss_type="l2",
        loss_fn="l2",
        clip_denoised=True,
        predict_epsilon=True,
        hidden_dim=256,
        loss_discount=1.0,
        condition_guidance_w=0.0,
        reverse_train=False,
        conditions=['context_fused'],  # ★ 수정된 부분: 컨텍스트 사용 명시
        hard_conditions=[],
        noise_init_method="normal",
        coverage_weight=0.0,
        detach_reverse=False,
        joint_weight=1.0,
        robot_offset=[0.0,0.0,0.0],
        trans_loss_scale=1.0,
        rot_loss_scale=1.0,
        diffusion_var="gripper_poses",
        joint_pred_pose_loss=False,
        joint_loss_scale=1.0,
        rank_bins=0,
        backbone="transformer",
        num_decoder_layers=2,
        num_encoder_layers=2,
        n_head=4,
        causal_attn=False,
        depth_proc="none",
        rgb_encoder="none",
    )

    # (F) 모델 구성
    base_diff = GaussianDynDiffusion(**dd_cfg)
    model = ContextAwareGaussianDynDiffusion(
        base_model=base_diff, context_dim=CONTEXT_DIM, hidden_dim=256
    )

# ---------- 더미 베이스 ----------
class DummyBaseDataset(Dataset):
    def __init__(self, num_eps=32, H=100, obs_dim=12, act_dim=7, seed=0):
        self.num_eps, self.H, self.obs_dim, self.act_dim = num_eps, H, obs_dim, act_dim
        self.rng = np.random.default_rng(seed)

    def __len__(self): return self.num_eps

    def __getitem__(self, i):
        obs = self.rng.standard_normal((self.H, self.obs_dim)).astype(np.float32)
        act = self.rng.standard_normal((self.H, self.act_dim)).astype(np.float32)
        return {"obs": torch.from_numpy(obs), "action": torch.from_numpy(act)}

# ---------- RLBench → (obs, action) 어댑터 ----------
class RLBenchToSeqAdapter(Dataset):
    def __init__(self, base: RLBenchDataset, use_joints=True):
        self.base = base
        self.use_joints = use_joints

    def __len__(self): return len(self.base)

    def __getitem__(self, i: int):
        s = self.base[i]
        obs_parts = []
        if "proprios" in s:
            obs_parts.append(np.asarray(s["proprios"], dtype=np.float32))
        if self.use_joints and "joint_positions" in s:
            obs_parts.append(np.asarray(s["joint_positions"], dtype=np.float32))
        if len(obs_parts) == 0:
            raise KeyError("No proprio/joint_positions found in RLBench sample.")
        obs = np.concatenate(obs_parts, axis=-1)

        if "gripper_poses" not in s:
            raise KeyError("Expected 'gripper_poses' in RLBench sample for actions.")
        action = np.asarray(s["gripper_poses"], dtype=np.float32)
        return {"obs": torch.from_numpy(obs), "action": torch.from_numpy(action)}

# ---------- VIOLA 컨텍스트 공급자 (차원 검증 추가) ----------
def npy_context_provider(pattern: str, expected_T: int = None):
    paths = sorted(glob.glob(pattern))
    if len(paths) == 0:
        raise FileNotFoundError(f"컨텍스트 파일을 찾을 수 없습니다. 경로를 확인하세요: {pattern}")

    # 첫 번째 파일로 차원 검증
    print(f"컨텍스트 파일 로딩 시작: {paths[0]}")
    first_ctx = np.load(paths[0])
    actual_dim = first_ctx.shape[-1]
    print(f"실제 컨텍스트 파일의 차원: {first_ctx.shape} (마지막 차원: {actual_dim})")
    print(f"스크립트에 설정된 CONTEXT_DIM: {CONTEXT_DIM}")

    if actual_dim != CONTEXT_DIM:
        print("\n!!!!!!!!!! [오류] !!!!!!!!!!!")
        print(f"컨텍스트 차원이 일치하지 않습니다!")
        print(f"스크립트 상단의 CONTEXT_DIM 값을 {actual_dim}(으)로 수정해주세요.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        # 에러를 발생시켜 실행을 중지합니다.
        raise ValueError("Context dimension mismatch")

    for p in paths:
        ctx = np.load(p)
        if expected_T is not None and ctx.shape[0] != expected_T:
            T = ctx.shape[0]
            m = min(expected_T, T)
            ctx = ctx[:m]
            if m < expected_T:
                pad = np.repeat(ctx[-1:], expected_T - m, axis=0)
                ctx = np.concatenate([ctx, pad], axis=0)
        yield ctx.astype(np.float32)

def main():
    # 공통 설정
    DATA_ROOT = "/home/yun/Documents/rlbench_data"
    CTX_PATTERN = "/home/yun/Documents/rlbench_data/contexts/ctx_ep_*.npy"
    TRAJ_LEN = 100

    # (D) VIOLA 컨텍스트 공급자
    ctx_iter = npy_context_provider(CTX_PATTERN, expected_T=TRAJ_LEN)

    # (B,C) 데이터셋 분기
    if USE_DUMMY_BASE:
        base = DummyBaseDataset(num_eps=32, H=TRAJ_LEN, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    else:
        TASKS = ["close_drawer"]
        TASKS_RATIO = [1.0]
        CAMS = ["front"]
        base_raw = RLBenchDataset(
            tasks=TASKS, tasks_ratio=TASKS_RATIO, camera_names=CAMS,
            num_episodes=32, data_raw_path=DATA_ROOT, traj_len=TRAJ_LEN,
            output_img_size=64, observation_dim=7, frame_skips=1,
            rank_bins=10, robot=None, diffusion_var="gripper_poses",
            training=True, demo_aug_ratio=0.0, demo_aug_min_len=20,
            use_cached=True, ds_img_size=128,
        )
        base = RLBenchToSeqAdapter(base_raw, use_joints=True)

    # (E) 컨텍스트 포함 데이터셋으로 감싸기
    train_ds = ContextRLBenchDataset(base_dataset=base, context_provider=ctx_iter, agg_fn="mean")
    loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=context_collate_fn)

    # (G) 스모크 테스트 3 step
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    for step, batch in enumerate(loader):
        loss, _ = model.loss(obs=batch["obs"], action=batch["action"],
                             cond={"context": batch["context"]})
        opt.zero_grad(); loss.backward(); opt.step()
        print(f"step {step}  loss {loss.item():.4f}")
        if step >= 2: # 3번만 실행하고 종료
            break

if __name__ == "__main__":
    main()