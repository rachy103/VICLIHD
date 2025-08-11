# rlbench_run.py
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

# 태스크 동적 로딩
from rlbench import tasks as rlbench_tasks
# 데모 저장 유틸
from rlbench.utils import save_demo


class Agent:
    def __init__(self, action_shape):
        self.action_shape = action_shape

    def ingest(self, demos):
        pass

    def act(self, obs):
        arm_dim = self.action_shape[0] - 1
        arm = np.random.normal(0.0, 0.1, size=(arm_dim,))
        gripper = [1.0]  # 항상 오픈
        return np.concatenate([arm, gripper], axis=-1)


def get_task_class(task_name: str):
    # "close_drawer" -> rlbench.tasks.CloseDrawer
    camel = "".join([p.capitalize() for p in task_name.split("_")])
    if hasattr(rlbench_tasks, camel):
        return getattr(rlbench_tasks, camel)
    raise ValueError(f"Unknown task: {task_name}")


def make_arm_mode(name: str):
    """
    RLBench 버전에 따라 delta EE 모드 클래스가 다를 수 있어 후보를 순차 시도.
    """
    name = name.lower()
    if name == "joint_vel":
        return JointVelocity()

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

    raise RuntimeError(
        "No suitable delta_ee arm mode found in this RLBench version. "
        "Try --arm_mode joint_vel"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["collect", "rollout"], default="collect")
    parser.add_argument("--task", type=str, default="close_drawer")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="/home/yun/Documents/VICLIHD/data")
    parser.add_argument("--headless", action="store_true", help="Headless rendering (권장)")
    parser.add_argument("--qt_offscreen", action="store_true", help="Force offscreen Qt platform")
    parser.add_argument("--arm_mode", choices=["delta_ee", "joint_vel"], default="delta_ee")
    parser.add_argument("--episode_length", type=int, default=40)   # rollout 모드에서만 사용
    parser.add_argument("--training_steps", type=int, default=120)  # rollout 모드에서만 사용
    args = parser.parse_args()

    # 저장 경로 준비
    os.makedirs(args.save_path, exist_ok=True)

    # Qt 플랫폼 설정 (Wayland 이슈 회피)
    if args.qt_offscreen or args.headless:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    else:
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    # 액션 모드
    try:
        arm_mode = make_arm_mode("joint_vel" if args.arm_mode == "joint_vel" else "delta_ee")
    except RuntimeError as e:
        print(f"[warn] {e}  Falling back to joint_vel.")
        arm_mode = make_arm_mode("joint_vel")

    action_mode = MoveArmThenGripper(
        arm_action_mode=arm_mode,
        gripper_action_mode=Discrete()
    )

    # 관측 구성
    obs_config = ObservationConfig()
    obs_config.set_all(True)

    # Environment 생성 (구버전 호환: data_path 인자 없음)
    env = Environment(
        action_mode,
        obs_config=obs_config,
        headless=args.headless,
    )
    env.launch()

    TaskClass = get_task_class(args.task)
    task = env.get_task(TaskClass)

    try:
        if args.mode == "collect":
            # 데모 수집 + 수동 저장
            task_root = os.path.join(args.save_path, args.task)
            os.makedirs(task_root, exist_ok=True)
            print(f"[Collect] task={args.task} episodes={args.episodes} save_dir={task_root}")

            # 보통 variation이 1개인 태스크가 많음. 안전하게 0으로 저장.
            variation_id = 0

            # GT 데모 생성
            demos = task.get_demos(amount=args.episodes, live_demos=True)

            # 저장
            for ep_idx, demo in enumerate(demos):
                # RLBench 디폴트 저장 구조와 유사하게 저장
                # save_demo(demo, path, variation_number, episode_number)
                save_demo(demo, task_root, variation_id, ep_idx)
                print(f" saved: var={variation_id} episode={ep_idx}")

            print("Done collecting demos.")

        else:
            # 랜덤 정책으로 롤아웃
            print(f"[Rollout] task={args.task} steps={args.training_steps}, ep_len={args.episode_length}")
            agent = Agent(env.action_shape)

            obs = None
            for i in range(args.training_steps):
                if i % args.episode_length == 0:
                    print("Reset Episode")
                    descriptions, obs = task.reset()
                    print(descriptions)
                action = agent.act(obs)
                print(action)
                obs, reward, terminate = task.step(action)
                if terminate:
                    print("Episode terminated early.")
                    descriptions, obs = task.reset()
            print("Rollout done.")
    finally:
        env.shutdown()
        print("Env shutdown complete.")


if __name__ == "__main__":
    main()
