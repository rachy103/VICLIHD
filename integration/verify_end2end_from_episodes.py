import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'VIOLA'))

from hydra import compose, initialize
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from viola_bc.policy import CenterNetSpatialTemporalPolicy
from hdp.rk_diffuser.models.diffusion import GaussianDynDiffusion
from rk_diffuser_mod.context_aware_diffusion import ContextAwareGaussianDynDiffusion


def load_episodes_npz(root, B, T):
    eps = sorted([p for p in glob.glob(os.path.join(root, 'ep_*', 'data.npz'))])
    if len(eps) == 0:
        raise FileNotFoundError(f"No data.npz under {root}")
    B = min(B, len(eps))
    data = []
    for i in range(B):
        d = np.load(eps[i])
        data.append({k: d[k] for k in d.files})
    # stack + trim
    def stack(key):
        arrs = [d[key] for d in data]
        arrs = [a[:T] for a in arrs]
        return np.stack(arrs, axis=0)
    return {
        'agentview_rgb': stack('agentview_rgb'),   # [B,T,H,W,3]
        'eye_in_hand_rgb': stack('eye_in_hand_rgb'),
        'joint_states': stack('joint_states'),     # [B,T,7]
        'gripper_states': stack('gripper_states'), # [B,T,2]
        'actions': stack('actions'),               # [B,T,7]
        'bboxes': stack('bboxes'),                 # [B,T,20,4]
    }


def to_batch_tensors(ep, device):
    B, T = ep['joint_states'].shape[:2]
    def norm_img(x):
        x = torch.from_numpy(x).float() / 255.0
        return x.permute(0,1,4,2,3).contiguous()  # [B,T,3,H,W]
    batch = {
        'obs': {
            'agentview_rgb': norm_img(ep['agentview_rgb']).to(device),
            'eye_in_hand_rgb': norm_img(ep['eye_in_hand_rgb']).to(device),
            'joint_states': torch.from_numpy(ep['joint_states']).float().to(device),
            'gripper_states': torch.from_numpy(ep['gripper_states']).float().to(device),
            'gripper_history': torch.zeros(B, T, 10, device=device),
            'centernet_bbox_20': torch.from_numpy(ep['bboxes']).float().to(device),
        },
        'actions': torch.from_numpy(ep['actions']).float().to(device),
    }
    return batch


def build_viola(device):
    with initialize(config_path="../VIOLA/viola_bc/configs", job_name="episodes_test"):
        cfg = compose(config_name="config", overrides=[
            "experiment=stack_viola",
            # CLIP 인코더 사용 명시
            "algo/model/encoder=clip_img_embedding",
            # 에피소드 텐서 정합성을 위해 grouping 모달리티 축소(차원 불일치 방지)
            "algo.model.grouping.network_kwargs.use_joint=false",
            "algo.model.grouping.network_kwargs.use_gripper=false",
            "algo.model.grouping.network_kwargs.use_gripper_history=false",
            "algo.model.grouping.network_kwargs.use_ee=false",
            # pooling 출력 크기 및 프로젝션 차원 재확인
            "algo.model.pooling.network_kwargs.output_size=[6,6]",
            "algo.model.spatial_projection.network_kwargs.out_dim=64",
            "algo.model.projection.network_kwargs.out_dim=64",
        ])
    model = CenterNetSpatialTemporalPolicy(cfg.algo.model, _shape_meta()).to(device)
    model.eval()
    # 증거 로그: 인코더 클래스명 및 설정 요약
    try:
        enc_name = model.encoder.__class__.__name__
    except Exception:
        enc_name = str(type(model.encoder))
    print(f"[VIOLA] Encoder class: {enc_name}")
    print(f"[VIOLA] Grouping uses eye_in_hand only; joint/gripper disabled for shape compatibility.")
    return model


def _shape_meta():
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


def extract_ctx_seq(viola, batch):
    with torch.no_grad():
        enc = TensorUtils.time_distributed(batch, viola.encode_fn)
    return enc.mean(dim=2) if enc.dim() == 4 else enc


def _rpn_viola_boxes_from_np(imgs_np: np.ndarray, topk: int = 20) -> torch.Tensor:
    """Use VIOLA's Detic/CenterNet2 ProposalNetwork to get proposals on numpy images.
    imgs_np: [B,T,H,W,3] uint8 or float(0..255). Returns torch [B,T,20,4] (xyxy), CPU.
    """
    from viola_bc.centernet_module import load_centernet_rpn
    if imgs_np.dtype != np.uint8:
        imgs_np = np.clip(imgs_np, 0, 255).astype('uint8')
    B, T, H, W, _ = imgs_np.shape
    predictor = load_centernet_rpn(nms=0.5)
    boxes_out = []
    for b in range(B):
        per_t = []
        for t in range(T):
            img = imgs_np[b, t][:,:,::-1]  # BGR
            outputs = predictor(img)
            box = None
            scores = None
            try:
                if isinstance(outputs, dict) and 'proposals' in outputs:
                    props = outputs['proposals']
                    box = props.proposal_boxes.tensor
                    if hasattr(props, 'objectness_logits'):
                        scores = props.objectness_logits
                elif hasattr(outputs, 'proposal_boxes'):
                    box = outputs.proposal_boxes.tensor
                elif isinstance(outputs, dict) and 'instances' in outputs:
                    inst = outputs['instances']
                    if hasattr(inst, 'pred_boxes'):
                        box = inst.pred_boxes.tensor
            except Exception:
                box = None
            if box is None or box.numel() == 0:
                box = torch.tensor([[0,0,W-1,H-1]], dtype=torch.float32)
            else:
                if scores is not None and scores.numel() == box.shape[0]:
                    idx = torch.argsort(scores, descending=True)[:topk]
                else:
                    areas = (box[:,2]-box[:,0]).clamp(min=0) * (box[:,3]-box[:,1]).clamp(min=0)
                    idx = torch.argsort(areas, descending=True)[:topk]
                box = box[idx]
            if box.shape[0] < topk:
                pad = box[-1:].repeat(max(0, topk - box.shape[0]), 1)
                box = torch.cat([box, pad], dim=0)
            per_t.append(box)
        boxes_out.append(torch.stack(per_t, dim=0))
    return torch.stack(boxes_out, dim=0)  # CPU tensor


class TransformerFuserWrapper(ContextAwareGaussianDynDiffusion):
    def __init__(self, base_model, context_dim, hidden_dim=256, nhead=2, nlayers=2):
        super().__init__(base_model, context_dim=context_dim, hidden_dim=hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.tx = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

    def _build_context_kwargs(self, cond, obs=None):
        cond_passthrough, extra = super()._build_context_kwargs(cond, obs)
        if not extra:
            return cond_passthrough, extra
        seq = extra["context_seq"]  # [B,H,C]
        seq_tx = self.tx(seq)
        extra["context_seq"] = seq_tx
        extra["context_fused"] = seq_tx.mean(dim=1)
        return cond_passthrough, extra


def build_hdp(D, device, horizon=10):
    # Set joint limits to reasonable range for 7-DoF
    jl_lo = [-2.6]*7
    jl_hi = [ 2.6]*7
    joint_limits = [jl_lo, jl_hi]
    base = GaussianDynDiffusion(
        horizon=horizon, observation_dim=7, dim_mults=[1,2,4], action_dim=7,
        scene_bounds=[], joint_limits=joint_limits, n_timesteps=50, loss_type='l2',
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
    # Use transformer fuser for context
    return TransformerFuserWrapper(base, context_dim=D, hidden_dim=256, nhead=2, nlayers=2).to(device)


def run(args):
    device = TorchUtils.get_torch_device(try_to_use_cuda=False)
    ep = load_episodes_npz(args.root, args.batch_size, args.frames)
    # 선택적으로 VIOLA RPN으로 wrist(agent: eye_in_hand)에서 bbox 재생성
    if args.rpn == 'viola':
        print('[RPN] Using VIOLA Detic/CenterNet2 ProposalNetwork on wrist (eye_in_hand)')
        wrist_np = ep['eye_in_hand_rgb']  # [B,T,H,W,3] uint8
        boxes_cpu = _rpn_viola_boxes_from_np(wrist_np, topk=20)
        ep['bboxes'] = boxes_cpu.numpy().astype(np.float32)
        # 오버레이 저장(증거): 각 에피소드 wrist 프레임 위에 top-1 박스 그리기
        if args.save_overlays:
            eps_dirs = sorted([d for d in glob.glob(os.path.join(args.root, 'ep_*')) if os.path.isdir(d)])[:args.batch_size]
            for i, ep_dir in enumerate(eps_dirs):
                outdir = os.path.join(ep_dir, 'wrist_rpn_overlays')
                os.makedirs(outdir, exist_ok=True)
                for t in range(min(args.frames, boxes_cpu.shape[1])):
                    fp = os.path.join(ep_dir, f'wrist_{t:04d}.png')
                    if not os.path.exists(fp):
                        continue
                    try:
                        im = Image.open(fp).convert('RGB').resize((ep['eye_in_hand_rgb'].shape[3], ep['eye_in_hand_rgb'].shape[2]))
                        draw = ImageDraw.Draw(im)
                        b = boxes_cpu[i, t, 0]
                        x1,y1,x2,y2 = [int(v.item()) for v in b]
                        draw.rectangle([x1,y1,x2,y2], outline=(255,255,0), width=2)
                        im.save(os.path.join(outdir, f'overlay_wrist_{t:04d}.png'))
                    except Exception:
                        pass
            print('[PROOF] Saved wrist RPN overlays under ep_*/wrist_rpn_overlays')
    batch = to_batch_tensors(ep, device)
    viola = build_viola(device)
    ctx_seq = extract_ctx_seq(viola, batch)
    B,T,D = ctx_seq.shape
    print(f"[INFO] episodes: B={B}, T={T}, D={D}")

    hdp = build_hdp(D, device, horizon=args.frames)
    # freeze base
    for p in hdp.base.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        loss_true, _ = hdp.loss(obs=batch['obs'], action=batch['actions'], cond={'context': ctx_seq}, use_dropout=False)
        loss_zero, _ = hdp.loss(obs=batch['obs'], action=batch['actions'], cond={'context': torch.zeros_like(ctx_seq)}, use_dropout=False)
        perm = torch.randperm(B)
        loss_perm, _ = hdp.loss(obs=batch['obs'], action=batch['actions'], cond={'context': ctx_seq[perm]}, use_dropout=False)
    print(f"[INIT] loss(true/perm/zero): {float(loss_true):.4f} / {float(loss_perm):.4f} / {float(loss_zero):.4f}")

    opt = torch.optim.AdamW([p for p in hdp.parameters() if p.requires_grad], lr=5e-4)
    for step in range(args.steps):
        opt.zero_grad()
        loss, _ = hdp.loss(obs=batch['obs'], action=batch['actions'], cond={'context': ctx_seq}, use_dropout=False)
        loss.backward()
        opt.step()
        if (step+1) % max(1, args.steps//5) == 0:
            print(f"[STEP {step+1}] loss: {float(loss):.4f}")

    with torch.no_grad():
        l_t, _ = hdp.loss(obs=batch['obs'], action=batch['actions'], cond={'context': ctx_seq}, use_dropout=False)
        l_z, _ = hdp.loss(obs=batch['obs'], action=batch['actions'], cond={'context': torch.zeros_like(ctx_seq)}, use_dropout=False)
        l_p, _ = hdp.loss(obs=batch['obs'], action=batch['actions'], cond={'context': ctx_seq[perm]}, use_dropout=False)
    print(f"[POST] loss(true/perm/zero): {float(l_t):.4f} / {float(l_p):.4f} / {float(l_z):.4f}")

    # gradient check on context
    ctx_req = ctx_seq.clone().detach().requires_grad_(True)
    l_g, _ = hdp.loss(obs=batch['obs'], action=batch['actions'], cond={'context': ctx_req}, use_dropout=False)
    l_g.backward()
    print(f"[GRAD] dL/dContext mean abs: {ctx_req.grad.detach().abs().mean().item():.6f}")

    # 선택적으로 체크포인트 저장
    if args.save_checkpoint:
        os.makedirs(os.path.dirname(args.save_checkpoint), exist_ok=True)
        torch.save(hdp.state_dict(), args.save_checkpoint)
        print(f"[CKPT] Saved HDP wrapper checkpoint: {args.save_checkpoint}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, required=True)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--frames', type=int, default=8)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--rpn', type=str, default='viola', choices=['none','viola'])
    ap.add_argument('--save-overlays', action='store_true')
    ap.add_argument('--save-checkpoint', type=str, default='Documents/VICLIHD/checkpoints/hdp_ctx.pth')
    args = ap.parse_args()
    run(args)
