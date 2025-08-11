#!/usr/bin/env python3

import sys
import h5py
import rospy
import numpy as np
from pynput import keyboard
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
import cv2
import time
from datetime import datetime
import os
import argparse

# Assuming these are the user's robot control libraries
# If they are not in the python path, this will fail.
# The user has provided a script with these imports, so I assume they are available.
try:
    sys.path.append('/home/vision/catkin_ws/src/robotory_rb10_rt/scripts')
    from teleop_data.msg import OnRobotRGOutput, OnRobotRGInput
    from api.cobot import *
    from rb import *
except ImportError as e:
    print(f"Warning: Could not import robot libraries. {e}")
    print("Using dummy robot functions.")
    # Dummy functions for development without robot hardware
    def GetCurrentSplitedJoint():
        return [0.0] * 6
    def ToCB(ip):
        pass
    def CobotInit():
        pass
    class RB10:
        def fkine(self, joints):
            return np.eye(4)
    # Dummy ROS messages if rospy is not fully configured
    if 'rospy' not in sys.modules:
        class OnRobotRGInput:
            def __init__(self):
                self.gGWD = 0
        class OnRobotRGOutput:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="VICLIHD Real Robot Data Collection Script")
    parser.add_argument('--task', type=str, required=True, help='Name of the task to collect data for.')
    parser.add_argument('--data_dir', type=str, default='/home/yun/Documents/VICLIHD/data', help='Directory to save the data.')
    parser.add_argument('--instruction', type=str, default='', help='Language instruction for the task.')
    return parser.parse_args()

def init_buffer():
    return {
        'obs': {
            'cam_high_rgb': [],
            'cam_low_rgb': [],
            'cam_high_depth': [],
            'cam_low_depth': [],
            'joint_positions': [],
            'joint_velocities': [],
            'gripper_pose': [], # [pos (3), quat_xyzw (4)]
            'gripper_open': [], # binary
            'gripper_joint_positions': [], # raw from gripper
        },
        'action': [],
        'lang_goal': ''
    }

def save_to_hdf5(buffer, data_dir, episode_idx=None):
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    if episode_idx is None:
        existing = [f for f in os.listdir(data_dir) if f.startswith(f'episode_') and f.endswith('.hdf5')]
        episode_idx = len(existing)

    filename = f'episode_{episode_idx}.hdf5'
    save_path = os.path.join(data_dir, filename)

    with h5py.File(save_path, 'w') as f:
        f.attrs['sim'] = False
        f.attrs['task_name'] = os.path.basename(os.path.dirname(data_dir))
        f.attrs['lang_goal'] = buffer['lang_goal']

        data_grp = f.create_group('data')
        demo_grp = data_grp.create_group(f'demo_0') # Store as a single demo per file
        
        obs_grp = demo_grp.create_group('obs')
        
        # Images
        obs_grp.create_dataset('cam_high_rgb', data=np.array(buffer['obs']['cam_high_rgb'], dtype=np.uint8))
        obs_grp.create_dataset('cam_low_rgb', data=np.array(buffer['obs']['cam_low_rgb'], dtype=np.uint8))
        obs_grp.create_dataset('cam_high_depth', data=np.array(buffer['obs']['cam_high_depth'], dtype=np.float32))
        obs_grp.create_dataset('cam_low_depth', data=np.array(buffer['obs']['cam_low_depth'], dtype=np.float32))

        # Proprioception
        obs_grp.create_dataset('joint_positions', data=np.array(buffer['obs']['joint_positions'], dtype=np.float64))
        obs_grp.create_dataset('joint_velocities', data=np.array(buffer['obs']['joint_velocities'], dtype=np.float64))
        obs_grp.create_dataset('gripper_pose', data=np.array(buffer['obs']['gripper_pose'], dtype=np.float64))
        obs_grp.create_dataset('gripper_open', data=np.array(buffer['obs']['gripper_open'], dtype=np.float64))
        obs_grp.create_dataset('gripper_joint_positions', data=np.array(buffer['obs']['gripper_joint_positions'], dtype=np.float64))

        # Actions
        demo_grp.create_dataset('actions', data=np.array(buffer['action'], dtype=np.float64))
        
        # Add other necessary attributes/datasets if needed by downstream processing
        # For example, robomimic might need 'dones'
        num_samples = len(buffer['action'])
        demo_grp.create_dataset('dones', data=np.zeros(num_samples, dtype=np.bool_))
        demo_grp['dones'][-1] = True
        
        demo_grp.attrs['num_samples'] = num_samples
        data_grp.attrs['num_demos'] = 1

        print(f"[HDF5] Saved {num_samples} timesteps to {save_path}")
        return episode_idx

def gripper_callback(msg):
    global latest_gripper_qpos
    latest_gripper_qpos = [msg.gGWD]

def on_press(key):
    global recording, terminal
    try:
        if key.char == 's':
            recording = True
            print("Start recording")
        elif key.char == 'q':
            recording = False
            print("Stop recording")
        elif key.char == 't':
            terminal = True
    except AttributeError:
        pass

def get_device_serials():
    ctx = rs.context()
    serials = []
    for device in ctx.query_devices():
        serials.append(device.get_info(rs.camera_info.serial_number))
    if len(serials) < 2:
        print("Warning: Less than 2 Realsense cameras detected. Using available ones.")
    if len(serials) == 0:
        raise RuntimeError("No Realsense camera connected.")
    print("Detected serials:", serials)
    return serials

def main():
    args = parse_args()
    
    # Data directory setup
    today = datetime.now().strftime('%Y%m%d')
    data_dir = os.path.join(args.data_dir, args.task, today)
    print(f"Data will be saved in: {data_dir}")

    # --- Hardware and ROS Setup ---
    serials = get_device_serials()
    pipelines = []
    configs = []
    
    # Align depth to color
    align = rs.align(rs.stream.color)

    for i, serial in enumerate(serials):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipelines.append(pipeline)
        configs.append(config)
        pipeline.start(config)
        print(f"Camera {i} (Serial: {serial}) started.")

    try:
        ToCB("192.168.111.50")
        robot = RB10()
        CobotInit()
    except NameError:
        robot = None
        print("Could not initialize robot. Running in dummy mode.")

    global terminal, recording, latest_gripper_qpos
    recording = False
    terminal = False
    latest_gripper_qpos = [0] # Default value

    rospy.init_node("viclihd_data_collection")
    try:
        rospy.Subscriber("/OnRobotRGInput", OnRobotRGInput, gripper_callback)
    except NameError:
        print("Could not subscribe to gripper topic. Using default gripper value.")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    buffer = init_buffer()
    buffer['lang_goal'] = args.instruction
    
    rate = rospy.Rate(20)

    # Gripper value normalization constants (from example)
    MAX_GRIP = 1100.0
    GRIPPER_OPEN_THRESHOLD = 0.9 # Assumes normalized value

    print('s: start, q: stop, t: terminate')

    last_joint_pos = np.zeros(6)

    while not rospy.is_shutdown():
        if terminal:
            print("Terminating.")
            break

        # --- Get Camera Data ---
        framesets = []
        try:
            for pipeline in pipelines:
                framesets.append(pipeline.wait_for_frames(timeout_ms=2000))
        except RuntimeError as e:
            rospy.logwarn(f"Realsense timeout: {e}")
            rate.sleep()
            continue
        
        aligned_framesets = [align.process(fs) for fs in framesets]
        
        color_images = [np.asanyarray(afs.get_color_frame().get_data()) for afs in aligned_framesets]
        depth_images = [np.asanyarray(afs.get_depth_frame().get_data()) for afs in aligned_framesets]

        # --- Display Images ---
        for i, (color_img, depth_img) in enumerate(zip(color_images, depth_images)):
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
            display_img = np.hstack((color_img, depth_colormap))
            cv2.imshow(f"Cam {i} (Serial: {serials[i]}) - RGB | Depth", display_img)
        cv2.waitKey(1)

        if recording:
            # --- Get Robot State ---
            current_jnt = np.array(GetCurrentSplitedJoint()) * np.pi / 180.0
            qpos_joint = current_jnt[:6]
            
            # Simple velocity calculation
            qvel_joint = (qpos_joint - last_joint_pos) * 20 # Assuming 20Hz rate
            last_joint_pos = qpos_joint

            # Gripper state
            gripper_raw = latest_gripper_qpos[0]
            gripper_normalized = gripper_raw / MAX_GRIP
            gripper_open = 1.0 if gripper_normalized > GRIPPER_OPEN_THRESHOLD else 0.0

            # EEF Pose
            if robot:
                T = np.array(robot.fkine(current_jnt))
                eef_pos = T[:3, 3]
                eef_quat_wxyz = R.from_matrix(T[:3, :3]).as_quat()
                eef_quat_xyzw = np.roll(eef_quat_wxyz, -1)
            else: # Dummy values
                eef_pos = np.zeros(3)
                eef_quat_xyzw = np.array([0,0,0,1])
            
            gripper_pose = np.concatenate([eef_pos, eef_quat_xyzw])

            # --- Store Data in Buffer ---
            # Observations
            buffer['obs']['cam_high_rgb'].append(color_images[0].copy())
            buffer['obs']['cam_low_rgb'].append(color_images[1].copy())
            buffer['obs']['cam_high_depth'].append(depth_images[0].copy().astype(np.float32))
            buffer['obs']['cam_low_depth'].append(depth_images[1].copy().astype(np.float32))
            buffer['obs']['joint_positions'].append(qpos_joint)
            buffer['obs']['joint_velocities'].append(qvel_joint)
            buffer['obs']['gripper_pose'].append(gripper_pose)
            buffer['obs']['gripper_open'].append(gripper_open)
            buffer['obs']['gripper_joint_positions'].append(np.array([gripper_normalized, gripper_normalized])) # Mimic robosuite format

            # Action (e.g., target end-effector pose and gripper state)
            action = np.concatenate([gripper_pose, [gripper_open]])
            buffer['action'].append(action)

        elif not recording and len(buffer['action']) > 0:
            while True:
                user_input = input("Store demo data? (y/n): ").strip().lower()
                if user_input == 'y':
                    save_to_hdf5(buffer, data_dir)
                    print(f"Data stored.")
                    break
                elif user_input == 'n':
                    print("Data discarded.")
                    break
                else:
                    print("Invalid input.")
            buffer = init_buffer()
            buffer['lang_goal'] = args.instruction

        rate.sleep()

    for pipeline in pipelines:
        pipeline.stop()
    cv2.destroyAllWindows()
    listener.stop()

if __name__ == "__main__":
    main()
