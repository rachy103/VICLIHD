import os
import sys
import glob
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'VIOLA'))

from hydra import compose, initialize
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from viola_bc.policy import CenterNetSpatialTemporalPolicy
from rk_diffuser_mod.context_aware_diffusion import ContextAwareGaussianDynDiffusion
from hdp.rk_diffuser.models.diffusion import GaussianDynDiffusion


def load_ep(ep_path):
    d = np.load(ep_path)
    data = {k: d[k] for k in d.files}
    return data


def to_batch(ep, device, T):
    def norm_img(x):
        x = torch.from_numpy(x[:T]).float() / 255.0
        return x.unsqueeze(0).permute(0,1,4,2,3).contiguous()  # [1,T,3,H,W]
    obs = {
        'agentview_rgb': norm_img(ep['agentview_rgb']).to(device),
        'eye_in_hand_rgb': norm_img(ep['eye_in_hand_rgb']).to(device),
        'joint_states': torch.from_numpy(ep['joint_states'][:T]).float().unsqueeze(0).to(device),
        'gripper_states': torch.from_numpy(ep['gripper_states'][:T]).float().unsqueeze(0).to(device),
        'gripper_history': torch.zeros(1, T, 10, device=device),
        'centernet_bbox_20': torch.from_numpy(ep['bboxes'][:T]).float().unsqueeze(0).to(device),
    }
    actions = torch.from_numpy(ep['actions'][:T]).float().unsqueeze(0).to(device)
    return {'obs': obs, 'actions': actions}


def build_viola(device):
    with initialize(config_path="../VIOLA/viola_bc/configs", job_name="rollout_eval"):
        cfg = compose(config_name="config", overrides=[
            "experiment=stack_viola",
            "algo/model/encoder=clip_img_embedding",
            "algo.model.grouping.network_kwargs.use_joint=false",
            "algo.model.grouping.network_kwargs.use_gripper=false",
            "algo.model.grouping.network_kwargs.use_gripper_history=false",
            "algo.model.grouping.network_kwargs.use_ee=false",
            "algo.model.pooling.network_kwargs.output_size=[6,6]",
            "algo.model.spatial_projection.network_kwargs.out_dim=64",
            "algo.model.projection.network_kwargs.out_dim=64",
        ])
    shape_meta = {
        "all_shapes": {
            "agentview_rgb": (3,128,128),
            "eye_in_hand_rgb": (3,128,128),
            "joint_states": (7,),
            "gripper_states": (2,),
            "gripper_history": (10,),
            "centernet_bbox_20": (20,4),
        },
        "ac_dim": 7,
        "all_obs_keys": [
            "agentview_rgb","eye_in_hand_rgb","joint_states","gripper_states","gripper_history","centernet_bbox_20",
        ],
    }
    model = CenterNetSpatialTemporalPolicy(cfg.algo.model, shape_meta).to(device)
    model.eval()
    return model


class TransformerFuserWrapper(ContextAwareGaussianDynDiffusion):
    def __init__(self, base_model, context_dim, hidden_dim=256, nhead=2, nlayers=2):
        super().__init__(base_model, context_dim=context_dim, hidden_dim=hidden_dim)
        enc_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.tx = torch.nn.TransformerEncoder(enc_layer, num_layers=nlayers)

    def _build_context_kwargs(self, cond, obs=None):
        cond_passthrough, extra = super()._build_context_kwargs(cond, obs)
        if not extra:
            return cond_passthrough, extra
        seq = extra["context_seq"]
        seq_tx = self.tx(seq)
        extra["context_seq"] = seq_tx
        extra["context_fused"] = seq_tx.mean(dim=1)
        return cond_passthrough, extra


def build_hdp(D, device, horizon):
    jl = [[-2.6]*7, [2.6]*7]
    base = GaussianDynDiffusion(
        horizon=horizon, observation_dim=7, dim_mults=[1,2,4], action_dim=7,
        scene_bounds=[], joint_limits=jl, n_timesteps=50, loss_type='l2',
        clip_denoised=True, predict_epsilon=False, hidden_dim=128,
        loss_discount=1.0, condition_guidance_w=1.0, reverse_train=False,
        conditions=['context_fused','context_seq'], hard_conditions=[],
        noise_init_method='normal', loss_fn='state_l2', coverage_weight=1.0,
        detach_reverse=False, joint_weight=1.0, robot_offset=[0,0,0],
        trans_loss_scale=1.0, rot_loss_scale=1.0, diffusion_var='joints',
        joint_pred_pose_loss=False, joint_loss_scale=1.0, rank_bins=10,
        backbone='unet', num_encoder_layers=2, num_decoder_layers=2,
        n_head=2, causal_attn=True, depth_proc='pointnet', rgb_encoder='resnet18'
    ).to(device)
    return TransformerFuserWrapper(base, context_dim=D, hidden_dim=256, nhead=2, nlayers=2).to(device)


def get_ctx(viola, batch):
    with torch.no_grad():
        enc = TensorUtils.time_distributed(batch, viola.encode_fn)
    return enc.mean(dim=2) if enc.dim() == 4 else enc


def integrate(init_joint: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return init_joint.unsqueeze(0) + actions.cumsum(dim=0)


def plot_joints(gt: np.ndarray, pred: np.ndarray, out_path: str, ylim=(-2.6, 2.6), title_info: str = ''):
    T, J = gt.shape
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(7, 1, figsize=(10, 12), sharex=True)
    t = np.arange(T)
    for j in range(7):
        axes[j].plot(t, gt[:,j], label='GT', color='#1f77b4', linewidth=2.0)
        axes[j].plot(t, pred[:,j], label='Pred', color='#d62728', linestyle='--', linewidth=1.8)
        axes[j].grid(True, alpha=0.3)
        axes[j].set_ylabel(f'j{j}')
        if ylim is not None:
            axes[j].set_ylim(ylim)
    axes[-1].set_xlabel('time (t)')
    axes[0].legend(loc='upper right')
    if title_info:
        fig.suptitle(title_info)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[PLOT] saved joints rollout: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', type=str, default='', help='Path to ep_xxxx/data.npz; if empty, picks the first under --root')
    ap.add_argument('--root', type=str, default='Documents/VICLIHD/rlbench_episodes')
    ap.add_argument('--frames', type=int, default=16)
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--outdir', type=str, default='Documents/VICLIHD/plots')
    args = ap.parse_args()

    device = TorchUtils.get_torch_device(try_to_use_cuda=False)
    if args.ep:
        ep_path = args.ep
    else:
        eps = sorted(glob.glob(os.path.join(args.root, 'ep_*', 'data.npz')))
        if not eps:
            raise FileNotFoundError('no ep data.npz found')
        ep_path = eps[0]
    ep = load_ep(ep_path)
    batch = to_batch(ep, device, args.frames)

    viola = build_viola(device)
    ctx = get_ctx(viola, batch)
    D = ctx.shape[-1]
    hdp = build_hdp(D, device, horizon=args.frames)
    state = torch.load(args.ckpt, map_location='cpu')
    hdp.load_state_dict(state)
    hdp.eval()

    with torch.no_grad():
        pred = hdp.conditional_sample(cond={'context': ctx}, horizon=args.frames)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        if isinstance(pred, dict):
            # unwrap joints branch
            inner = pred.get('joints', pred)
            if isinstance(inner, dict) and 'diffusion_hist' in inner:
                dh = inner['diffusion_hist']
                # select last denoising step
                if dh.ndim == 4:       # [B, S, T, D]
                    pred = dh[:, -1]
                elif dh.ndim == 3:     # [S, T, D]
                    pred = dh[-1]
                else:
                    raise RuntimeError('Unexpected diffusion_hist shape')
            elif isinstance(inner, torch.Tensor):
                pred = inner
            else:
                # fallback: first tensor [B,T,D]
                cand = None
                for v in pred.values():
                    if isinstance(v, torch.Tensor) and v.ndim >= 3:
                        cand = v; break
                if cand is None:
                    raise RuntimeError('Could not unwrap sample dict to tensor')
                pred = cand

    if isinstance(pred, dict):
        raise RuntimeError('Could not unwrap sample dict to tensor')

    if pred.ndim == 3 and pred.size(0) == 1:
        pred = pred.squeeze(0)
    pred = pred.cpu()  # [T,7]
    gt_actions = batch['actions'].squeeze(0).cpu()
    gt_joints = batch['obs']['joint_states'].squeeze(0).cpu()
    pred_joints = integrate(gt_joints[0], pred)

    rmse_t = torch.sqrt(((pred_joints - gt_joints)**2).mean(dim=0)).detach().cpu().view(-1)
    mae_t = (pred_joints - gt_joints).abs().mean(dim=0).detach().cpu().view(-1)
    rmse = rmse_t.numpy().tolist()
    mae = mae_t.numpy().tolist()
    print('[METRIC] per-joint RMSE:', ' '.join(f'{float(v):.4f}' for v in rmse))
    print('[METRIC] per-joint MAE :', ' '.join(f'{float(v):.4f}' for v in mae))
    print('[METRIC] overall  RMSE:', f'{float(np.mean(rmse)):.4f}')
    print('[METRIC] overall  MAE :', f'{float(np.mean(mae)):.4f}')

    base = os.path.splitext(os.path.basename(ep_path))[0]
    out_png = os.path.join(args.outdir, f'rollout_joints_{base}.png')
    # ensure same length for plotting
    L = min(gt_joints.shape[0], pred_joints.shape[0])
    title = f"RMSE={float(np.mean(rmse)):.3f}, MAE={float(np.mean(mae)):.3f} (T={L})"
    plot_joints(gt_joints[:L].numpy(), pred_joints[:L].numpy(), out_png, ylim=(-2.6,2.6), title_info=title)


if __name__ == '__main__':
    main()
