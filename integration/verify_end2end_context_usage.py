import os
import sys
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'VIOLA'))

from hydra import compose, initialize
from omegaconf import OmegaConf

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from viola_bc.policy import CenterNetSpatialTemporalPolicy
from rk_diffuser_mod.context_aware_diffusion import ContextAwareGaussianDynDiffusion
from hdp.rk_diffuser.models.diffusion import GaussianDynDiffusion


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_viola_cpu():
    device = TorchUtils.get_torch_device(try_to_use_cuda=False)
    with initialize(config_path="../VIOLA/viola_bc/configs", job_name="context_usage_test"):
        cfg = compose(config_name="config", overrides=[
            "experiment=stack_viola",
            "algo/model/encoder=clip_img_embedding",
        ])
    # small tweak to keep it light
    cfg.algo.model.transformer.network_kwargs.num_layers = 2
    model = CenterNetSpatialTemporalPolicy(cfg.algo.model, _dummy_shape_meta()).to(device)
    model.eval()
    return model, device


def _dummy_shape_meta():
    return {
        "all_shapes": {
            "agentview_rgb": (3, 128, 128),
            "eye_in_hand_rgb": (3, 128, 128),
            "joint_states": (7,),
            "gripper_states": (2,),
            "gripper_history": (10,),
            "centernet_bbox_20": (20, 4),
        },
        "ac_dim": 7,
        "all_obs_keys": [
            "agentview_rgb", "eye_in_hand_rgb", "joint_states",
            "gripper_states", "gripper_history", "centernet_bbox_20",
        ],
    }


def make_dummy_batch(B=2, T=8, device='cpu'):
    batch = {
        "obs": {
            "agentview_rgb": torch.randn(B, T, 3, 128, 128, device=device),
            "eye_in_hand_rgb": torch.randn(B, T, 3, 128, 128, device=device),
            "joint_states": torch.randn(B, T, 7, device=device),
            "gripper_states": torch.randn(B, T, 2, device=device),
            "gripper_history": torch.randn(B, T, 10, device=device),
            "centernet_bbox_20": torch.randn(B, T, 20, 4, device=device),
        },
        "actions": torch.randn(B, T, 7, device=device),
    }
    return batch


def extract_context_seq(viola_model, batch):
    # Use encode_fn via time_distributed to get position_embedding_out (B,T,*,D)
    with torch.no_grad():
        out = TensorUtils.time_distributed(batch, viola_model.encode_fn)
    # out shape could be (B,T,N,D). We reduce tokens to mean to get (B,T,D).
    if out.dim() == 4:
        ctx_seq = out.mean(dim=2)
    elif out.dim() == 3:
        ctx_seq = out
    else:
        raise RuntimeError(f"Unexpected encode output shape: {tuple(out.shape)}")
    return ctx_seq  # (B,T,D)


def build_hdp(context_dim, device='cpu'):
    base = GaussianDynDiffusion(
        horizon=10, observation_dim=7, dim_mults=[1, 2, 4], action_dim=7,
        scene_bounds=[], joint_limits=[], n_timesteps=50, loss_type='l2',
        clip_denoised=True, predict_epsilon=False, hidden_dim=128,
        loss_discount=1.0, condition_guidance_w=1.0, reverse_train=False,
        conditions=['context_fused'], hard_conditions=[],
        noise_init_method='normal', loss_fn='state_l2', coverage_weight=1.0,
        detach_reverse=False, joint_weight=1.0, robot_offset=[0,0,0],
        trans_loss_scale=1.0, rot_loss_scale=1.0, diffusion_var='gripper_poses',
        joint_pred_pose_loss=False, joint_loss_scale=1.0, rank_bins=10,
        backbone='unet', num_encoder_layers=2, num_decoder_layers=2,
        n_head=2, causal_attn=True, depth_proc='pointnet', rgb_encoder='resnet18'
    ).to(device)
    wrapper = ContextAwareGaussianDynDiffusion(base, context_dim=context_dim, hidden_dim=256).to(device)
    return wrapper


class TransformerFuserWrapper(ContextAwareGaussianDynDiffusion):
    def __init__(self, base_model, context_dim, hidden_dim=256, nhead=4, nlayers=2):
        super().__init__(base_model, context_dim=context_dim, hidden_dim=hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.tx = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    def _build_context_kwargs(self, cond, obs=None):
        cond_passthrough, extra = super()._build_context_kwargs(cond, obs)
        if not extra:
            return cond_passthrough, extra
        # context_seq: (B,H,C)
        ctx_seq = extra["context_seq"]
        # Transformer over time
        ctx_seq_tx = self.tx(ctx_seq)
        extra["context_seq"] = ctx_seq_tx
        extra["context_fused"] = ctx_seq_tx.mean(dim=1)
        return cond_passthrough, extra


def loss_with_context(model, obs, action, ctx_seq):
    cond = {"context": ctx_seq}
    loss, _ = model.loss(obs=obs, action=action, cond=cond, use_dropout=False)
    return loss


def run_verification():
    set_seed(0)
    viola, device = build_viola_cpu()
    batch = make_dummy_batch(B=2, T=8, device=device)

    # 1) 실제 VIOLA 출력에서 context_seq 추출 및 차원 확인
    ctx_seq = extract_context_seq(viola, batch)
    B, T, D = ctx_seq.shape
    print(f"[INFO] Derived context shape from VIOLA encode: (B,T,D)=({B},{T},{D})")

    # 2) HDP 래퍼 두 가지(MLP/Transformer) 구성
    hdp_mlp = build_hdp(D, device=device)
    hdp_tx = TransformerFuserWrapper(hdp_mlp.base, context_dim=D, hidden_dim=256, nhead=2, nlayers=2).to(device)

    # 3) 초기 민감도 점검: 올바른 컨텍스트 vs 무작위 순열/제로
    with torch.no_grad():
        loss_true_mlp = float(loss_with_context(hdp_mlp, batch["obs"], batch["actions"], ctx_seq))
        loss_true_tx = float(loss_with_context(hdp_tx, batch["obs"], batch["actions"], ctx_seq))
        perm = torch.randperm(B)
        loss_perm_mlp = float(loss_with_context(hdp_mlp, batch["obs"], batch["actions"], ctx_seq[perm]))
        loss_perm_tx = float(loss_with_context(hdp_tx, batch["obs"], batch["actions"], ctx_seq[perm]))
        loss_zero_mlp = float(loss_with_context(hdp_mlp, batch["obs"], batch["actions"], torch.zeros_like(ctx_seq)))
        loss_zero_tx = float(loss_with_context(hdp_tx, batch["obs"], batch["actions"], torch.zeros_like(ctx_seq)))
    print(f"[INIT] MLP  loss(true/perm/zero): {loss_true_mlp:.4f} / {loss_perm_mlp:.4f} / {loss_zero_mlp:.4f}")
    print(f"[INIT] TRX  loss(true/perm/zero): {loss_true_tx:.4f} / {loss_perm_tx:.4f} / {loss_zero_tx:.4f}")

    # 4) 그래디언트 경로 점검: 컨텍스트 입력/프로젝션 가중치의 grad 확인
    ctx_seq_req = ctx_seq.clone().detach().requires_grad_(True)
    loss = loss_with_context(hdp_mlp, batch["obs"], batch["actions"], ctx_seq_req)
    loss.backward()
    grad_ctx = ctx_seq_req.grad.detach().abs().mean().item()
    grad_w = hdp_mlp.ctx_proj.weight.grad.detach().abs().mean().item()
    print(f"[GRAD-MLP] dL/dContext: {grad_ctx:.6f}, dL/dW(ctx_proj): {grad_w:.6f}")
    hdp_mlp.ctx_proj.weight.grad.zero_()
    hdp_mlp.base.zero_grad(set_to_none=True)

    # 5) 짧은 학습으로 민감도 향상 확인 (각 50 스텝)
    def train_short(model, steps=50, freeze_base=True):
        if freeze_base:
            for p in model.base.parameters():
                p.requires_grad_(False)
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
        for _ in range(steps):
            opt.zero_grad()
            loss = loss_with_context(model, batch["obs"], batch["actions"], ctx_seq)
            loss.backward()
            opt.step()
        return model

    hdp_mlp = train_short(hdp_mlp, steps=100, freeze_base=True)
    hdp_tx = train_short(hdp_tx, steps=100, freeze_base=True)

    with torch.no_grad():
        loss_true_mlp = float(loss_with_context(hdp_mlp, batch["obs"], batch["actions"], ctx_seq))
        loss_perm_mlp = float(loss_with_context(hdp_mlp, batch["obs"], batch["actions"], ctx_seq[perm]))
        loss_zero_mlp = float(loss_with_context(hdp_mlp, batch["obs"], batch["actions"], torch.zeros_like(ctx_seq)))
        loss_true_tx = float(loss_with_context(hdp_tx, batch["obs"], batch["actions"], ctx_seq))
        loss_perm_tx = float(loss_with_context(hdp_tx, batch["obs"], batch["actions"], ctx_seq[perm]))
        loss_zero_tx = float(loss_with_context(hdp_tx, batch["obs"], batch["actions"], torch.zeros_like(ctx_seq)))
    print(f"[POST] MLP  loss(true/perm/zero): {loss_true_mlp:.4f} / {loss_perm_mlp:.4f} / {loss_zero_mlp:.4f}")
    print(f"[POST] TRX  loss(true/perm/zero): {loss_true_tx:.4f} / {loss_perm_tx:.4f} / {loss_zero_tx:.4f}")

    # 6) 간단한 판정: perm/zero 대비 true가 유의미하게 낮아야 함
    def assert_used(tag, true, perm, zero):
        # 여유 있는 기준(상대 차이 3% + 절대 1e-3)
        if not (true + max(1e-3, 0.03 * abs(true)) < perm and true + max(1e-3, 0.03 * abs(true)) < zero):
            print(f"[WARN] {tag} context sensitivity is weak: true={true:.4f}, perm={perm:.4f}, zero={zero:.4f}")
        else:
            print(f"[OK] {tag} context is used: true << perm/zero")

    assert_used('MLP', loss_true_mlp, loss_perm_mlp, loss_zero_mlp)
    assert_used('TRX', loss_true_tx, loss_perm_tx, loss_zero_tx)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    set_seed(0)
    run_verification()
