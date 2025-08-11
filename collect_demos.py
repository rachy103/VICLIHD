# ~/Documents/VICLIHD/collect_demos.py
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench import tasks as rlbench_tasks


def get_task_class(task_name: str):
    camel = "".join([p.capitalize() for p in task_name.split("_")])
    if hasattr(rlbench_tasks, camel):
        return getattr(rlbench_tasks, camel)
    raise ValueError(f"Unknown task: {task_name}")


def make_arm_mode(which: str):
    which = which.lower()
    if which == "joint_vel":
        return JointVelocity()

    # delta_ee 계열을 버전 호환으로 시도
    try:
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
        return EndEffectorPoseViaPlanning()
    except Exception:
        pass
    try:
        from rlbench.action_modes.arm_action_modes import EndEffectorPosePlanWorldFrame
        return EndEffectorPosePlanWorldFrame()
    except Exception:
        pass
    try:
        from rlbench.action_modes.arm_action_modes import EndEffectorPose
        return EndEffectorPose()
    except Exception:
        pass

    raise RuntimeError("delta_ee arm mode를 찾지 못했습니다. --arm_mode joint_vel 로 다시 시도하세요.")


def build_obs_config(rgb=True, depth=False, mask=False, wrist=True, overhead=True):
    oc = ObservationConfig()
    oc.set_all(False)

    # Proprio 항상 켜두자 (로봇 joint/ee state)
    oc.joint_forces = True
    oc.joint_positions = True
    oc.joint_velocities = True
    oc.gripper_open = True
    oc.gripper_pose = True
    oc.gripper_touch_forces = True
    oc.task_low_dim_state = True

    # 카메라 설정
    def set_cam(cam):
        cam.set_all(False)
        cam.rgb = rgb
        cam.depth = depth
        cam.mask = mask

    if wrist:
        set_cam(oc.wrist_camera)
    if overhead:
        set_cam(oc.overhead_camera)

    # 필요하면 추가 카메라 (front, left_shoulder 등)도 켤 수 있음
    return oc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="close_drawer")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="/home/yun/Documents/VICLIHD/data")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--qt_offscreen", action="store_true")
    parser.add_argument("--arm_mode", choices=["delta_ee", "joint_vel"], default="delta_ee")
    parser.add_argument("--variation", type=int, default=0, help="수집할 variation index")
    # 카메라/관측 토글
    parser.add_argument("--rgb", action="store_true", default=True)
    parser.add_argument("--depth", action="store_true", default=False)
    parser.add_argument("--mask", action="store_true", default=False)
    parser.add_argument("--wrist_cam", action="store_true", default=True)
    parser.add_argument("--overhead_cam", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Qt 플랫폼 (headless 충돌 방지)
    if args.qt_offscreen or args.headless:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # 액션 모드
    arm_mode = make_arm_mode("joint_vel" if args.arm_mode == "joint_vel" else "delta_ee")
    action_mode = MoveArmThenGripper(arm_action_mode=arm_mode, gripper_action_mode=Discrete())

    # 관측 구성
    obs_config = build_obs_config(
        rgb=args.rgb, depth=args.depth, mask=args.mask,
        wrist=args.wrist_cam, overhead=args.overhead_cam
    )

    # 환경 생성
    env = Environment(action_mode=action_mode, obs_config=obs_config, headless=args.headless)
    env.launch()

    TaskClass = get_task_class(args.task)
    task = env.get_task(TaskClass)

    # 저장 루트: <save_dir>/<TaskName>/variation_<id>/episodes/episode_<k>
    task_name = TaskClass.__name__
    var_dir = os.path.join(args.save_dir, task_name, f"variation_{args.variation}", "episodes")
    os.makedirs(var_dir, exist_ok=True)

    try:
        print(f"[Collect] task={task_name} variation={args.variation} episodes={args.episodes}")
        # variation 세팅 (가능한 경우)
        try:
            task.set_variation(args.variation)
        except Exception:
            print("  (알림) 이 RLBench 버전은 set_variation을 직접 노출하지 않을 수 있습니다. 기본 variation으로 진행합니다.")

        # 데모 생성 (라이브 GT)
        demos = task.get_demos(amount=args.episodes, live_demos=True)

        # 저장
        for idx, demo in enumerate(demos):
            ep_dir = os.path.join(var_dir, f"episode_{idx:03d}")
            os.makedirs(ep_dir, exist_ok=True)
            # Demonstration API가 제공하는 저장 메서드 사용
            # (현재 RLBench 버전은 demo.save(...) 지원)
            demo.save(ep_dir)
            print(f"  saved -> {ep_dir}")

        print("Done.")
    finally:
        env.shutdown()
        print("Env shutdown complete.")


if __name__ == "__main__":
    main()
