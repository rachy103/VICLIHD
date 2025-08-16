import os
import sys
import glob
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'VIOLA'))

from hydra import compose, initialize
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from viola_bc.policy import CenterNetSpatialTemporalPolicy
from rk_diffuser_mod.context_aware_diffusion import ContextAwareGaussianDynDiffusion
from hdp.rk_diffuser.models.diffusion import GaussianDynDiffusion
from hdp.rk_diffuser.robot import DiffRobot


def load_npz(path):
    d = np.load(path)
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
    with initialize(config_path="../VIOLA/viola_bc/configs", job_name="plot3d"):
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
    model = CenterNetSpatialTemporalPolicy(cfg.algo.model, {
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
    }).to(device)
    model.eval()
    return model


def build_hdp(D, device):
    base = GaussianDynDiffusion(
        horizon=10, observation_dim=7, dim_mults=[1,2,4], action_dim=7,
        scene_bounds=[], joint_limits=[], n_timesteps=50, loss_type='l2',
        clip_denoised=False, predict_epsilon=False, hidden_dim=128,
        loss_discount=1.0, condition_guidance_w=1.0, reverse_train=False,
        conditions=['context_fused','context_seq'], hard_conditions=[],
        noise_init_method='normal', loss_fn='state_l2', coverage_weight=1.0,
        detach_reverse=False, joint_weight=1.0, robot_offset=[0,0,0],
        trans_loss_scale=1.0, rot_loss_scale=1.0, diffusion_var='joints',
        joint_pred_pose_loss=False, joint_loss_scale=1.0, rank_bins=10,
        backbone='unet', num_encoder_layers=2, num_decoder_layers=2,
        n_head=2, causal_attn=True, depth_proc='pointnet', rgb_encoder='resnet18'
    ).to(device)
    return ContextAwareGaussianDynDiffusion(base, context_dim=D, hidden_dim=256).to(device)


def get_context_seq(viola, batch):
    with torch.no_grad():
        enc = TensorUtils.time_distributed(batch, viola.encode_fn)
    return enc.mean(dim=2) if enc.dim() == 4 else enc


def integrate_actions(init_joint: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    # init_joint: [7], actions: [T,7] (diffs)
    diffs = actions.cumsum(dim=0)
    traj = init_joint.unsqueeze(0) + diffs
    return traj  # [T,7]


def plot_traj_3d(gt: np.ndarray, pred: np.ndarray, out_path: str, dims=(0,1,2)):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gt[:,dims[0]], gt[:,dims[1]], gt[:,dims[2]], label='GT', color='blue')
    ax.plot(pred[:,dims[0]], pred[:,dims[1]], pred[:,dims[2]], label='Pred', color='red')
    ax.set_xlabel(f'joint[{dims[0]}]')
    ax.set_ylabel(f'joint[{dims[1]}]')
    ax.set_zlabel(f'joint[{dims[2]}]')
    ax.legend()
    ax.set_title('Trajectory in joint space (3D)')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[PLOT] Saved 3D trajectory to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', type=str, required=True, help='Path to ep_xxxx/data.npz')
    ap.add_argument('--frames', type=int, default=16)
    ap.add_argument('--ckpt', type=str, required=True, help='HDP wrapper checkpoint path')
    ap.add_argument('--out', type=str, default='Documents/VICLIHD/plots/traj3d_ep.png')
    ap.add_argument('--ee-out', type=str, default='', help='If set, also plot EE(xyz) to this path')
    ap.add_argument('--urdf', type=str, default='', help='URDF path for EE forward kinematics')
    args = ap.parse_args()

    device = TorchUtils.get_torch_device(try_to_use_cuda=False)
    ep = load_npz(args.ep)
    batch = to_batch(ep, device, T=args.frames)
    viola = build_viola(device)
    ctx_seq = get_context_seq(viola, batch)
    D = ctx_seq.shape[-1]
    hdp = build_hdp(D, device)
    state = torch.load(args.ckpt, map_location='cpu')
    hdp.load_state_dict(state)
    hdp.eval()

    # sample predicted actions
    with torch.no_grad():
        pred_actions = hdp.conditional_sample(cond={'context': ctx_seq}, horizon=args.frames)
        if isinstance(pred_actions, (list, tuple)):
            pred_actions = pred_actions[0]
        if isinstance(pred_actions, dict):
            # first unwrap by diffusion var if present
            key = 'joints' if 'joints' in pred_actions else ('gripper_poses' if 'gripper_poses' in pred_actions else None)
            inner = pred_actions[key] if key else pred_actions
            # sometimes inner is dict with 'diffusion_hist'
            if isinstance(inner, dict):
                if 'diffusion_hist' in inner and isinstance(inner['diffusion_hist'], torch.Tensor):
                    pred_actions = inner['diffusion_hist']
                else:
                    cand = None
                    for v in inner.values():
                        if isinstance(v, torch.Tensor) and v.ndim >= 3:
                            cand = v; break
                    if cand is None:
                        raise RuntimeError('No tensor sample found in conditional_sample return dict')
                    pred_actions = cand
            elif isinstance(inner, torch.Tensor):
                pred_actions = inner
    pred_actions = pred_actions.squeeze(0).cpu()  # [T,7]

    gt_actions = batch['actions'].squeeze(0).cpu()  # [T,7]
    gt_joints = batch['obs']['joint_states'].squeeze(0).cpu()  # [T,7]
    init_joint = gt_joints[0]
    pred_joints = integrate_actions(init_joint, pred_actions)

    plot_traj_3d(gt_joints.numpy(), pred_joints.numpy(), args.out, dims=(0,1,2))

    # optional EE plot
    if args.ee_out:
        if not args.urdf or not os.path.exists(args.urdf):
            print('[WARN] --ee-out set but --urdf missing or not found. Skipping EE plot.')
        else:
            try:
                robot = DiffRobot(args.urdf)
                # to device for fk
                robot.to(device)
                with torch.no_grad():
                    gt_ee = robot.forward_kinematics_batch(gt_joints.to(device)).cpu().numpy()  # [T,7]
                    pred_ee = robot.forward_kinematics_batch(pred_joints.to(device)).cpu().numpy()
                plot_traj_3d(gt_ee, pred_ee, args.ee_out, dims=(0,1,2))
            except Exception as e:
                print('[WARN] EE plotting failed:', e)


if __name__ == '__main__':
    main()
