import os
import sys
import glob
import argparse
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
import torchvision

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'VIOLA'))

from hydra import compose, initialize

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from viola_bc.policy import CenterNetSpatialTemporalPolicy
from rk_diffuser_mod.context_aware_diffusion import ContextAwareGaussianDynDiffusion
from hdp.rk_diffuser.models.diffusion import GaussianDynDiffusion


def list_episodes_with_cams(root: str) -> List[Dict[str, List[str]]]:
    """Return list of episodes with separate camera frame lists.
    Each item: { 'overhead': [...], 'wrist': [...] }
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Image root not found: {root}")
    episodes = []
    for ep_dir in sorted(os.listdir(root)):
        d = os.path.join(root, ep_dir)
        if not os.path.isdir(d):
            continue
        overhead = sorted(glob.glob(os.path.join(d, 'overhead_*.png')) + glob.glob(os.path.join(d, 'overhead_*.jpg')))
        wrist = sorted(glob.glob(os.path.join(d, 'wrist_*.png')) + glob.glob(os.path.join(d, 'wrist_*.jpg')))
        if overhead or wrist:
            episodes.append({'overhead': overhead, 'wrist': wrist})
    if not episodes:
        raise RuntimeError(f"No episodes with images found under {root}")
    return episodes


def load_episode_frames(files: List[str], T: int, resize: Tuple[int, int] = (128, 128)) -> torch.Tensor:
    """Load first T frames and return tensor [T, 3, H, W] in [0,1]."""
    tfm = transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    imgs = []
    for f in files[:T]:
        with Image.open(f) as im:
            if im.mode != 'RGB':
                im = im.convert('RGB')
            imgs.append(tfm(im))
    # pad if fewer than T
    if len(imgs) < T:
        if len(imgs) == 0:
            imgs = [torch.zeros(3, *resize)]
        while len(imgs) < T:
            imgs.append(imgs[-1].clone())
    return torch.stack(imgs, dim=0)  # [T,3,H,W]


def build_viola(device, encoder_name: str = "clip_img_embedding"):
    with initialize(config_path="../VIOLA/viola_bc/configs", job_name="from_images_test"):
        cfg = compose(config_name="config", overrides=[
            "experiment=stack_viola",
            f"algo/model/encoder={encoder_name}",
        ])
    model = CenterNetSpatialTemporalPolicy(cfg.algo.model, _shape_meta()).to(device)
    # 증거 로그: 실제 인코더 클래스명 출력
    try:
        enc_name = model.encoder.__class__.__name__
    except Exception:
        enc_name = str(type(model.encoder))
    print(f"[VIOLA] Encoder class: {enc_name}")
    model.eval()
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


def _rpn_torchvision_boxes(imgs_bt: torch.Tensor, topk: int = 20):
    """TorchVision FasterRCNN RPN proposals per frame.
    imgs_bt: [B,T,3,H,W] in [0,1]; returns [B,T,20,4] (xyxy).
    """
    B, T, C, H, W = imgs_bt.shape
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval()
    model = model.to(imgs_bt.device)
    out_bt = []
    with torch.no_grad():
        for b in range(B):
            per_t = []
            for t in range(T):
                img = imgs_bt[b, t]
                out = model([img])[0]
                boxes = out.get('boxes', torch.zeros(0, 4, device=img.device))
                scores = out.get('scores', torch.zeros(0, device=img.device))
                if boxes.numel() == 0:
                    box = torch.tensor([[0, 0, W - 1, H - 1]], dtype=torch.float32, device=img.device)
                else:
                    idx = torch.argsort(scores, descending=True)[:topk]
                    box = boxes[idx]
                if box.shape[0] < topk:
                    pad = box[-1:].repeat(topk - box.shape[0], 1)
                    box = torch.cat([box, pad], dim=0)
                per_t.append(box)
            out_bt.append(torch.stack(per_t, dim=0))  # [T,20,4]
    return torch.stack(out_bt, dim=0)  # [B,T,20,4]

def _rpn_viola_boxes(imgs_bt: torch.Tensor, topk: int = 20):
    """Use VIOLA's Detic/CenterNet2 ProposalNetwork to get proposals.
    imgs_bt: [B,T,3,H,W] in [0,1] CPU. Returns [B,T,20,4] (xyxy).
    """
    from viola_bc.centernet_module import load_centernet_rpn
    B, T, C, H, W = imgs_bt.shape
    predictor = load_centernet_rpn(nms=0.5)
    boxes_out = []
    for b in range(B):
        per_t = []
        for t in range(T):
            # convert to uint8 BGR for detectron2
            img = (imgs_bt[b, t].clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype('uint8')
            img = img[:, :, ::-1]
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
                box = torch.tensor([[0, 0, W - 1, H - 1]], dtype=torch.float32)
            else:
                if scores is not None and scores.numel() == box.shape[0]:
                    idx = torch.argsort(scores, descending=True)[:topk]
                else:
                    areas = (box[:, 2] - box[:, 0]).clamp(min=0) * (box[:, 3] - box[:, 1]).clamp(min=0)
                    idx = torch.argsort(areas, descending=True)[:topk]
                box = box[idx]
            if box.shape[0] < topk:
                pad = box[-1:].repeat(max(0, topk - box.shape[0]), 1)
                box = torch.cat([box, pad], dim=0)
            per_t.append(box)
        boxes_out.append(torch.stack(per_t, dim=0))
    return torch.stack(boxes_out, dim=0)


def build_batch_from_images(root: str, B: int, T: int, device: torch.device, bbox_mode: str = 'edge', corr_scale: float = 0.5, rpn: str = 'edge', roi_camera: str = 'wrist'):
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
    # assign views: 기본은 agentview=overhead, eye_in_hand=wrist
    agent = overhead
    eye = wrist
    # simple low-dim signals (작은 노이즈)
    joint = 0.05 * torch.randn(B, T, 7, device=device)
    grip = 0.05 * torch.randn(B, T, 2, device=device)
    grip_hist = 0.05 * torch.randn(B, T, 10, device=device)

    # choose ROI source image
    roi_src = eye if roi_camera == 'wrist' else agent
    # bbox 생성: RPN(VIOLA/TV) / edge / center (on roi_src)
    if rpn == 'viola':
        print(f"[RPN] Using VIOLA Detic/CenterNet2 ProposalNetwork on {roi_camera}")
        boxes = _rpn_viola_boxes(roi_src.cpu(), topk=20).to(device)
    elif rpn == 'torchvision':
        print(f"[RPN] Using TorchVision FasterRCNN proposals (top-20) on {roi_camera}")
        boxes = _rpn_torchvision_boxes(roi_src, topk=20)
    elif bbox_mode == 'center':
        cx, cy, w, h = 64, 64, 32, 32
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = cx + w // 2, cy + h // 2
        one_roi = torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=device)
        boxes = one_roi.view(1, 1, 1, 4).repeat(B, T, 20, 1)
    else:
        # 간단한 에지 기반: 그레이스케일 → 1차 미분 → magnitude → 최대값 근방 32x32 ROI
        gray = agent.mean(dim=2)  # [B,T,H,W]
        # padding and differences
        gx = torch.zeros_like(gray)
        gy = torch.zeros_like(gray)
        gx[:, :, :, 1:] = (gray[:, :, :, 1:] - gray[:, :, :, :-1])
        gy[:, :, 1:, :] = (gray[:, :, 1:, :] - gray[:, :, :-1, :])
        mag = gx.abs() + gy.abs()  # [B,T,H,W]
        boxes_list = []
        for b in range(B):
            per_t = []
            for t in range(T):
                m = mag[b, t]
                idx = torch.argmax(m)
                y = (idx // m.size(1)).item()
                x = (idx % m.size(1)).item()
                w = h = 32
                x1 = max(0, x - w // 2)
                y1 = max(0, y - h // 2)
                x2 = min(127, x1 + w)
                y2 = min(127, y1 + h)
                per_t.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=device))
            per_t = torch.stack(per_t, dim=0)
            boxes_list.append(per_t)
        boxes = torch.stack(boxes_list, dim=0)  # [B,T,4]
        boxes = boxes.unsqueeze(2).repeat(1, 1, 20, 1)

    # 이미지 밝기와 상관된 행동 생성 (ac_dim=7 중 첫 2차원에 상관 부여)
    # 정규화된 밝기 평균: [0,1]
    brightness = agent.mean(dim=(2, 3, 4))  # [B,T]
    brightness = (brightness - brightness.min()) / (brightness.max() - brightness.min() + 1e-6)
    # 1차원: 밝기 자체, 2차원: 프레임 차분
    diff = torch.zeros_like(brightness)
    diff[:, 1:] = brightness[:, 1:] - brightness[:, :-1]
    actions = 0.05 * torch.randn(B, T, 7, device=device)
    actions[:, :, 0] += corr_scale * brightness
    actions[:, :, 1] += corr_scale * diff
    batch = {
        "obs": {
            "agentview_rgb": agent,
            "eye_in_hand_rgb": eye,
            "joint_states": joint,
            "gripper_states": grip,
            "gripper_history": grip_hist,
            "centernet_bbox_20": boxes,
        },
        "actions": actions,
    }
    return batch


def extract_context_seq(viola_model, batch):
    with torch.no_grad():
        enc = TensorUtils.time_distributed(batch, viola_model.encode_fn)  # [B,T,N,D] or [B,T,D]
    ctx = enc.mean(dim=2) if enc.dim() == 4 else enc
    return ctx  # [B,T,D]


def build_hdp(D: int, device: torch.device):
    base = GaussianDynDiffusion(
        horizon=10, observation_dim=7, dim_mults=[1, 2, 4], action_dim=7,
        scene_bounds=[], joint_limits=[], n_timesteps=50, loss_type='l2',
        clip_denoised=True, predict_epsilon=False, hidden_dim=128,
        loss_discount=1.0, condition_guidance_w=1.0, reverse_train=False,
        conditions=['context_fused','context_seq'], hard_conditions=[],
        noise_init_method='normal', loss_fn='state_l2', coverage_weight=1.0,
        detach_reverse=False, joint_weight=1.0, robot_offset=[0,0,0],
        trans_loss_scale=1.0, rot_loss_scale=1.0, diffusion_var='gripper_poses',
        joint_pred_pose_loss=False, joint_loss_scale=1.0, rank_bins=10,
        backbone='unet', num_encoder_layers=2, num_decoder_layers=2,
        n_head=2, causal_attn=True, depth_proc='pointnet', rgb_encoder='resnet18'
    ).to(device)
    wrapper = ContextAwareGaussianDynDiffusion(base, context_dim=D, hidden_dim=256).to(device)
    return wrapper


class TransformerFuserWrapper(ContextAwareGaussianDynDiffusion):
    """시계열 컨텍스트를 간단한 Transformer로 가공 후 fused에 반영"""
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


def run(args):
    device = TorchUtils.get_torch_device(try_to_use_cuda=False)
    viola = build_viola(device, encoder_name=args.encoder)
    batch = build_batch_from_images(args.root, args.batch_size, args.frames, device, rpn=args.rpn, roi_camera=args.roi_camera)
    # wrist 카메라 오버레이 저장 (요청 증거)
    try:
        if args.save_overlays and args.roi_camera == 'wrist':
            eps = list_episodes_with_cams(args.root)
            B = min(args.batch_size, len(eps))
            boxes = batch['obs']['centernet_bbox_20'].cpu()  # [B,T,20,4]
            for i in range(B):
                wrist_files = eps[i]['wrist'][:args.frames]
                out_dir = os.path.join(os.path.dirname(wrist_files[0]) if wrist_files else args.root, 'wrist_rpn_overlays')
                os.makedirs(out_dir, exist_ok=True)
                for t, fp in enumerate(wrist_files):
                    try:
                        im = Image.open(fp).convert('RGB').resize((128,128))
                        draw = ImageDraw.Draw(im)
                        # top-1 박스만 증거로 그림 (원하면 상위 k 반복 가능)
                        b = boxes[i, t, 0]
                        x1, y1, x2, y2 = [int(v) for v in b.tolist()]
                        draw.rectangle([x1, y1, x2, y2], outline=(255,255,0), width=2)
                        im.save(os.path.join(out_dir, f'overlay_wrist_{t:04d}.png'))
                    except Exception:
                        pass
            print('[PROOF] Saved wrist RPN overlays under per-episode wrist_rpn_overlays')
    except Exception as e:
        print('[WARN] overlay saving failed:', e)
    ctx_seq = extract_context_seq(viola, batch)
    B, T, D = ctx_seq.shape
    print(f"[INFO] Context from VIOLA: (B,T,D)=({B},{T},{D})")

    hdp_base = build_hdp(D, device)
    # 시계열 정보 반영을 위한 Transformer 결합 래퍼
    hdp = TransformerFuserWrapper(hdp_base.base, context_dim=D, hidden_dim=256, nhead=2, nlayers=2).to(device)
    # 평가 시 드롭아웃 비활성화, 학습은 컨텍스트 경로만
    for p in hdp.base.parameters():
        p.requires_grad_(False)

    # 초기 손실/perm/zero 비교
    with torch.no_grad():
        loss_true, _ = hdp.loss(obs=batch["obs"], action=batch["actions"], cond={"context": ctx_seq}, use_dropout=False)
        loss_zero, _ = hdp.loss(obs=batch["obs"], action=batch["actions"], cond={"context": torch.zeros_like(ctx_seq)}, use_dropout=False)
        perm = torch.randperm(B)
        loss_perm, _ = hdp.loss(obs=batch["obs"], action=batch["actions"], cond={"context": ctx_seq[perm]}, use_dropout=False)
    print(f"[INIT] loss(true/perm/zero): {float(loss_true):.4f} / {float(loss_perm):.4f} / {float(loss_zero):.4f}")

    # short train on image-driven context
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, hdp.parameters()), lr=5e-4)
    for step in range(args.steps):
        opt.zero_grad()
        loss, _ = hdp.loss(obs=batch["obs"], action=batch["actions"], cond={"context": ctx_seq}, use_dropout=False)
        loss.backward()
        opt.step()
        if (step + 1) % max(1, args.steps // 5) == 0:
            print(f"[STEP {step+1}] loss: {float(loss):.4f}")

    with torch.no_grad():
        loss_true2, _ = hdp.loss(obs=batch["obs"], action=batch["actions"], cond={"context": ctx_seq}, use_dropout=False)
        loss_zero2, _ = hdp.loss(obs=batch["obs"], action=batch["actions"], cond={"context": torch.zeros_like(ctx_seq)}, use_dropout=False)
        loss_perm2, _ = hdp.loss(obs=batch["obs"], action=batch["actions"], cond={"context": ctx_seq[perm]}, use_dropout=False)
    print(f"[POST] loss(true/perm/zero): {float(loss_true2):.4f} / {float(loss_perm2):.4f} / {float(loss_zero2):.4f}")

    # 그래디언트 경로 점검
    ctx_req = ctx_seq.clone().detach().requires_grad_(True)
    loss_g, _ = hdp.loss(obs=batch["obs"], action=batch["actions"], cond={"context": ctx_req}, use_dropout=False)
    loss_g.backward()
    grad_ctx = ctx_req.grad.detach().abs().mean().item()
    print(f"[GRAD] dL/dContext mean abs: {grad_ctx:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end verification from real images (CPU)")
    parser.add_argument('--root', type=str, required=True, help='Root dir containing episodes as subfolders with images')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--encoder', type=str, default='clip_img_embedding', choices=['clip_img_embedding','resnet_conv','no_stride_resnet_conv','plain_conv'])
    parser.add_argument('--rpn', type=str, default='edge', choices=['edge','center','torchvision','viola'])
    parser.add_argument('--roi-camera', type=str, default='wrist', choices=['wrist','agent'], help='which camera to generate ROI boxes on')
    parser.add_argument('--save-overlays', action='store_true')
    args = parser.parse_args()
    run(args)
