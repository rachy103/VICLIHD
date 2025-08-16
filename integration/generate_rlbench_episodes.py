import os
import sys
import argparse
from pathlib import Path
import os
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='reach_target')
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--out', type=str, default='Documents/VICLIHD/rlbench_episodes')
    parser.add_argument('--camera', type=str, default='overhead', choices=['overhead','wrist','left_shoulder','right_shoulder'])
    parser.add_argument('--headless', action='store_true', default=True)
    args = parser.parse_args()

    os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
    from rlbench.environment import Environment
    from rlbench.action_modes import action_mode as am
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.observation_config import ObservationConfig

    # Resolve task class from name
    from rlbench.tasks import ReachTarget
    TASK_MAP = {
        'reach_target': ReachTarget,
    }
    if args.task not in TASK_MAP:
        print(f"Unsupported task: {args.task}")
        sys.exit(2)
    TaskCls = TASK_MAP[args.task]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    action_mode = am.ActionMode(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
    obs_conf = ObservationConfig()
    # 기본은 모두 꺼고 필요한 것만 켜기
    try:
        obs_conf.set_all(False)
    except Exception:
        pass
    # 카메라 RGB 활성화 (버전별 API 호환)
    if hasattr(obs_conf.overhead_camera, 'rgb'):
        obs_conf.overhead_camera.rgb = True
    else:
        try:
            obs_conf.overhead_camera.set_rgb(True)
        except Exception:
            pass
    if hasattr(obs_conf.wrist_camera, 'rgb'):
        obs_conf.wrist_camera.rgb = True
    else:
        try:
            obs_conf.wrist_camera.set_rgb(True)
        except Exception:
            pass
    # 관절/그리퍼 정보 활성화
    try:
        obs_conf.joint_positions = True
    except Exception:
        pass
    try:
        obs_conf.gripper_open = True
    except Exception:
        pass
    # 마스크/세그멘테이션 (있으면 사용)
    if hasattr(obs_conf.overhead_camera, 'mask'):
        obs_conf.overhead_camera.mask = True
    elif hasattr(obs_conf.overhead_camera, 'segmentation'):  # 일부 버전명
        obs_conf.overhead_camera.segmentation = True
    env = Environment(action_mode=action_mode, obs_config=obs_conf, headless=args.headless)
    env.launch()
    task = env.get_task(TaskCls)

    print(f"Generating {args.episodes} episodes for task: {args.task}")
    demos = task.get_demos(amount=args.episodes, live_demos=True)
    for i, demo in enumerate(demos):
        ep_dir = out_root / f'ep_{i:04d}'
        ep_dir.mkdir(parents=True, exist_ok=True)
        print(f"Episode {i}: {len(demo)} steps → saving frames + npz to {ep_dir}")
        agent_rgbs = []
        wrist_rgbs = []
        joint_states = []
        gripper_states = []
        overhead_masks = []
        for t, obs in enumerate(demo):
            over = obs.overhead_rgb
            wrist = obs.wrist_rgb
            Image.fromarray(over).save(ep_dir / f"overhead_{t:04d}.png")
            Image.fromarray(wrist).save(ep_dir / f"wrist_{t:04d}.png")
            agent_rgbs.append(over)
            wrist_rgbs.append(wrist)
            # robustly extract 7-dim joint vector
            jp = getattr(obs, 'joint_positions', None)
            try:
                arr = np.asarray(jp, dtype=np.float32)
                if arr.ndim == 0 or arr.size == 0:
                    arr = np.zeros((7,), dtype=np.float32)
                if arr.shape[0] < 7:
                    pad = np.zeros((7 - arr.shape[0],), dtype=np.float32)
                    arr = np.concatenate([arr, pad], axis=0)
                arr = arr[:7]
            except Exception:
                arr = np.zeros((7,), dtype=np.float32)
            joint_states.append(arr)
            # gripper_open이 None인 경우 대비
            try:
                go_raw = obs.gripper_open
            except Exception:
                go_raw = None
            try:
                go = float(0.0 if go_raw is None else go_raw)
            except Exception:
                go = 0.0
            gripper_states.append(np.asarray([go, 1.0 - go], dtype=np.float32))
            m = getattr(obs, 'overhead_mask', None)
            if m is not None:
                overhead_masks.append(m)
        agent_rgbs = np.stack(agent_rgbs, axis=0)
        wrist_rgbs = np.stack(wrist_rgbs, axis=0)
        joint_states = np.stack(joint_states, axis=0)
        gripper_states = np.stack(gripper_states, axis=0)
        actions = np.zeros_like(joint_states)
        actions[1:] = joint_states[1:] - joint_states[:-1]
        T = agent_rgbs.shape[0]
        boxes = np.zeros((T, 20, 4), dtype=np.float32)
        def center_box():
            cx, cy, w, h = 64, 64, 32, 32
            return np.array([cx-w//2, cy-h//2, cx+w//2, cy+h//2], dtype=np.float32)
        for t in range(T):
            if len(overhead_masks) == T:
                m = overhead_masks[t]
                ys, xs = np.where(m > 0)
                if ys.size > 0:
                    x1, y1 = xs.min(), ys.min()
                    x2, y2 = xs.max(), ys.max()
                    box = np.array([x1, y1, x2, y2], dtype=np.float32)
                else:
                    box = center_box()
            else:
                box = center_box()
            boxes[t, :, :] = box
        np.savez_compressed(
            ep_dir / 'data.npz',
            agentview_rgb=agent_rgbs,
            eye_in_hand_rgb=wrist_rgbs,
            joint_states=joint_states,
            gripper_states=gripper_states,
            actions=actions,
            bboxes=boxes,
        )

    env.shutdown()
    print(f"Saved episodes under: {out_root}")

if __name__ == '__main__':
    main()
