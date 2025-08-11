# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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

CONTEXT_DIM = 32
OBS_DIM = 7       # ★ 여기 12 → 7
ACT_DIM = 7
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
        observation_dim=ACT_DIM,   # ★ 핵심: 12가 아니라 7로! (OBS_DIM가 아니라 ACT_DIM)
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
        conditions=[],
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
    base_diff = GaussianDynDiffusion(**dd_cfg)  # <<< config={} 쓰지 말고 dd_cfg로!
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

# ---------- VIOLA 컨텍스트 공급자 ----------
def npy_context_provider(pattern: str, expected_T: int = None):
    paths = sorted(glob.glob(pattern))
    if len(paths) == 0:
        raise FileNotFoundError(f"No context npy files matched: {pattern}")
    for p in paths:
        ctx = np.load(p)  # (T,N,D) or (T,D)
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
    CONTEXT_DIM = 32  # (T,D)에서 D

    # (D) VIOLA 컨텍스트 공급자
    ctx_iter = npy_context_provider(CTX_PATTERN, expected_T=TRAJ_LEN)

    # (B,C) 데이터셋 분기
    if USE_DUMMY_BASE:
        base = DummyBaseDataset(num_eps=32, H=TRAJ_LEN, obs_dim=12, act_dim=7)
    else:
        # RLBenchDataset은 데모가 실제로 있어야 합니다.
        TASKS = ["close_drawer"]
        TASKS_RATIO = [1.0]
        CAMS = ["front"]
        base_raw = RLBenchDataset(
            tasks=TASKS,
            tasks_ratio=TASKS_RATIO,
            camera_names=CAMS,
            num_episodes=32,
            data_raw_path=DATA_ROOT,
            traj_len=TRAJ_LEN,
            output_img_size=64,
            observation_dim=7,
            frame_skips=1,
            rank_bins=10,
            robot=None,
            diffusion_var="gripper_poses",
            training=True,
            demo_aug_ratio=0.0,
            demo_aug_min_len=20,
            use_cached=True,
            ds_img_size=128,
        )
        base = RLBenchToSeqAdapter(base_raw, use_joints=True)

    # (E) 컨텍스트 포함 데이터셋으로 감싸기
    train_ds = ContextRLBenchDataset(base_dataset=base, context_provider=ctx_iter, agg_fn="mean")
    loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=context_collate_fn)

    # (F) 모델 구성


    # (G) 스모크 테스트 3 step
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    for step, batch in enumerate(loader):
        loss, _ = model.loss(obs=batch["obs"], action=batch["action"],
                             cond={"context": batch["context"]})
        opt.zero_grad(); loss.backward(); opt.step()
        print(f"step {step}  loss {loss.item():.4f}")

if __name__ == "__main__":
    main()
