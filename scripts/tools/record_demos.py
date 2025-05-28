# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to record demonstrations with Isaac Lab environments using human teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Device for interacting with environment. (default: keyboard)
    --dataset_file            File path to export recorded demos. (default: "./datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 0)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful. (default: 10)
"""

"""Launch Isaac Sim Simulator first."""

# Standard library imports
import argparse
import contextlib

# Third-party imports
import gymnasium as gym
import numpy as np
import os
import time
import torch
import time
# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

if "handtracking" in args_cli.teleop_device.lower():
    from isaacsim.xr.openxr import OpenXRSpec

# Omniverse logger
import omni.log
import omni.ui as ui

# Additional Isaac Lab imports that can only be imported after the simulator is running
from isaaclab.devices import OpenXRDevice, Se3Keyboard, Se3SpaceMouse

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions

if args_cli.enable_pinocchio:
    from isaaclab.devices.openxr.retargeters.humanoid.fourier.gr1t2_retargeter import GR1T2Retargeter
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

from isaaclab.devices.openxr.retargeters.manipulator import GripperRetargeter, Se3AbsRetargeter, Se3RelRetargeter
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Import SAM related functions
from isaaclab_tasks.manager_based.manipulation.paint.mdp.detect_paintarea import initialize_sam

class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        """Initialize a RateLimiter with specified frequency.

        Args:
            hz: Frequency to enforce in Hertz.
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        """Attempt to sleep at the specified rate in hz.

        Args:
            env: Environment to render during sleep periods.
        """
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def pre_process_actions(
    teleop_data: tuple[np.ndarray, bool] | list[tuple[np.ndarray, np.ndarray, np.ndarray]], num_envs: int, device: str
) -> torch.Tensor:
    """Convert teleop data to the format expected by the environment action space.

    Args:
        teleop_data: Data from the teleoperation device.
        num_envs: Number of environments.
        device: Device to create tensors on.

    Returns:
        Processed actions as a tensor.
    """
    # compute actions based on environment
    if "Reach" in args_cli.task:
        delta_pose, gripper_command = teleop_data
        # convert to torch
        delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=device).repeat(num_envs, 1)
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    elif "PickPlace-GR1T2" in args_cli.task:
        (left_wrist_pose, right_wrist_pose, hand_joints) = teleop_data[0]
        # Reconstruct actions_arms tensor with converted positions and rotations
        actions = torch.tensor(
            np.concatenate([
                left_wrist_pose,  # left ee pose
                right_wrist_pose,  # right ee pose
                hand_joints,  # hand joint angles
            ]),
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)
        # Concatenate arm poses and hand joint angles
        return actions
    else:
        # resolve gripper command
        delta_pose, gripper_command = teleop_data
        # convert to torch
        delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=device).repeat(num_envs, 1)
        gripper_vel = torch.zeros((delta_pose.shape[0], 1), dtype=torch.float, device=device)
        gripper_vel[:] = -1 if gripper_command else 1
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)


def main():
    """Collect demonstrations from the environment using teleop interfaces."""

    # if handtracking is selected, rate limiting is achieved via OpenXR
    if "handtracking" in args_cli.teleop_device.lower():
        rate_limiter = None
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task

    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        omni.log.warn(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None

    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Flags for controlling the demonstration recording process
    should_reset_recording_instance = False
    running_recording_instance = True

    def reset_recording_instance():
        """Reset the current recording instance.

        This function is triggered when the user indicates the current demo attempt
        has failed and should be discarded. When called, it marks the environment
        for reset, which will start a fresh recording instance. This is useful when:
        - The robot gets into an unrecoverable state
        - The user makes a mistake during demonstration
        - The objects in the scene need to be reset to their initial positions
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    def start_recording_instance():
        """Start or resume recording the current demonstration.

        This function enables active recording of robot actions. It's used when:
        - Beginning a new demonstration after positioning the robot
        - Resuming recording after temporarily stopping to reposition
        - Continuing demonstration after pausing to adjust approach or strategy

        The user can toggle between stop/start to reposition the robot without
        recording those transitional movements in the final demonstration.
        """
        nonlocal running_recording_instance
        running_recording_instance = True

    def stop_recording_instance():
        """Temporarily stop recording the current demonstration.

        This function pauses the active recording of robot actions, allowing the user to:
        - Reposition the robot or hand tracking device without recording those movements
        - Take a break without terminating the entire demonstration
        - Adjust their approach before continuing with the task

        The environment will continue rendering but won't record actions or advance
        the simulation until recording is resumed with start_recording_instance().
        """
        nonlocal running_recording_instance
        running_recording_instance = False

    def create_teleop_device(device_name: str, env: gym.Env):
        """Create and configure teleoperation device for robot control.

        Args:
            device_name: Control device to use. Options include:
                - "keyboard": Use keyboard keys for simple discrete movements
                - "spacemouse": Use 3D mouse for precise 6-DOF control
                - "handtracking": Use VR hand tracking for intuitive manipulation
                - "handtracking_abs": Use VR hand tracking for intuitive manipulation with absolute EE pose

        Returns:
            DeviceBase: Configured teleoperation device ready for robot control
        """
        device_name = device_name.lower()
        nonlocal running_recording_instance
        if device_name == "keyboard":
            # Explicitly pass the environment instance to the Se3Keyboard constructor
            keyboard = Se3Keyboard(pos_sensitivity=0.2, rot_sensitivity=0.5, env=env)
            print(f"Created Se3Keyboard with environment reference: {keyboard._env}")
            return keyboard
        elif device_name == "spacemouse":
            return Se3SpaceMouse(pos_sensitivity=0.2, rot_sensitivity=0.5)
        elif "dualhandtracking_abs" in device_name and "GR1T2" in env.cfg.env_name:
            # Create GR1T2 retargeter with desired configuration
            gr1t2_retargeter = GR1T2Retargeter(
                enable_visualization=True,
                num_open_xr_hand_joints=2 * (int(OpenXRSpec.HandJointEXT.XR_HAND_JOINT_LITTLE_TIP_EXT) + 1),
                device=env.unwrapped.device,
                hand_joint_names=env.scene["robot"].data.joint_names[-22:],
            )

            # Create hand tracking device with retargeter
            device = OpenXRDevice(
                env_cfg.xr,
                retargeters=[gr1t2_retargeter],
            )
            device.add_callback("RESET", reset_recording_instance)
            device.add_callback("START", start_recording_instance)
            device.add_callback("STOP", stop_recording_instance)

            running_recording_instance = False
            return device
        elif "handtracking" in device_name:
            # Create Franka retargeter with desired configuration
            if "_abs" in device_name:
                retargeter_device = Se3AbsRetargeter(
                    bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT, zero_out_xy_rotation=True
                )
            else:
                retargeter_device = Se3RelRetargeter(
                    bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT, zero_out_xy_rotation=True
                )

            grip_retargeter = GripperRetargeter(bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT)

            # Create hand tracking device with retargeter (in a list)
            device = OpenXRDevice(
                env_cfg.xr,
                retargeters=[retargeter_device, grip_retargeter],
            )
            device.add_callback("RESET", reset_recording_instance)
            device.add_callback("START", start_recording_instance)
            device.add_callback("STOP", stop_recording_instance)

            running_recording_instance = False
            return device
        else:
            raise ValueError(
                f"Invalid device interface '{device_name}'. Supported: 'keyboard', 'spacemouse', 'handtracking',"
                " 'handtracking_abs', 'dualhandtracking_abs'."
            )

    teleop_interface = create_teleop_device(args_cli.teleop_device, env)
    teleop_interface.add_callback("R", reset_recording_instance)
    print(teleop_interface)

    # reset before starting
    env.sim.reset()
    env.reset()
    print(f"Number of environments: {env.num_envs}")
    if hasattr(env.scene, "asset_prim_paths"):
        print(f"env.scene.asset_prim_paths: {env.scene.asset_prim_paths}")
        if "robot" in env.scene.asset_prim_paths:
            print(f"Robot prim paths in scene: {env.scene.asset_prim_paths['robot']}")
        else:
            print("'robot' key not found in env.scene.asset_prim_paths")
    else:
        print("env.scene does not have asset_prim_paths attribute.")

    robot_asset = env.scene.articulations.get("robot") # 또는 env.scene.robot
    if robot_asset:
        if hasattr(robot_asset, 'prim_path_expr_resolved'):
            print(f"Robot asset 'prim_path_expr_resolved': {robot_asset.prim_path_expr_resolved}")
        else:
            print("Robot asset does not have 'prim_path_expr_resolved' attribute.")
    else:
        print("Could not get 'robot' asset from scene.")
        teleop_interface.reset()
    
    # --- 마지막 출력 시간 기록 변수 초기화 ---

    last_print_time = time.time()

    # simulate environment -- run everything in inference mode
    current_recorded_demo_count = 0
    success_step_count = 0

    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."

    instruction_display = InstructionDisplay(args_cli.teleop_device)
    if args_cli.teleop_device.lower() != "handtracking":
        window = EmptyWindow(env, "Instruction")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(label_text)
            subtask_label = ui.Label("")
            instruction_display.set_labels(subtask_label, demo_label)

    subtasks = {}

    # Initialize SAM mask generator
    initialize_sam()

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            # get data from teleop device
            teleop_data = teleop_interface.advance()

            # perform action on environment
            if running_recording_instance:
                # compute actions based on environment
                actions = pre_process_actions(teleop_data, env.num_envs, env.device)
                obv = env.step(actions)
                
                # Store the observation for the keyboard to use
                env.last_obv = obv
                
                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    # --- 1. 로봇의 EEF 월드 좌표 출력 ---
                    if isinstance(obv[0], dict) and "policy" in obv[0] and "eef_pos" in obv[0]["policy"]:
                        eef_world_pos = obv[0]["policy"]["eef_pos"][0].cpu().numpy()
                        print(f"1. EEF World Pos: [X={eef_world_pos[0]:.3f}, Y={eef_world_pos[1]:.3f}, Z={eef_world_pos[2]:.3f}]")
                    
                    # --- 2. 현재 Target의 World 좌표 출력 ---
                    # 현재 타겟 인덱스 확인
                    if hasattr(env, 'current_target_idx'):
                        current_target_idx = env.current_target_idx[0].item()
                        # 현재 타겟 로컬 좌표 가져오기
                        if hasattr(env, 'all_target_local_positions') and len(env.all_target_local_positions) > current_target_idx:
                            current_target_local_pos = env.all_target_local_positions[current_target_idx]
                            
                            # myblock 객체 가져오기
                            try:
                                # 다양한 방법으로 myblock 접근 시도
                                myblock = None
                                if hasattr(env.scene, "rigid_objects"):
                                    myblock = env.scene.rigid_objects.get("myblock")
                                
                                if myblock is None and hasattr(env.scene, "__getitem__"):
                                    try:
                                        myblock = env.scene["myblock"]
                                    except (KeyError, TypeError):
                                        pass
                                
                                if myblock is None:
                                    # 다른 방법으로 시도
                                    for possible_attr in ["rigid_objects", "assets"]:
                                        if hasattr(env.scene, possible_attr):
                                            scene_coll = getattr(env.scene, possible_attr)
                                            if hasattr(scene_coll, "get"):
                                                myblock = scene_coll.get("myblock")
                                                if myblock is not None:
                                                    break
                                
                                if myblock is not None:
                                    # myblock의 월드 포즈 가져오기
                                    myblock_pos_w = myblock.data.root_pos_w[0].cpu()
                                    myblock_quat_w = myblock.data.root_quat_w[0].cpu()
                                    
                                    # 타겟 로컬 좌표를 월드 좌표로 변환 (직접 계산)
                                    try:
                                        import isaaclab.utils.math as math_utils
                                        target_world_pos = math_utils.quat_apply(
                                            myblock_quat_w.to(env.device), 
                                            current_target_local_pos
                                        ).cpu() + myblock_pos_w
                                        
                                        # 월드 좌표 출력
                                        target_world_pos_np = target_world_pos.numpy()
                                        print(f"2. Current Target ({current_target_idx}) World Pos: [X={target_world_pos_np[0]:.3f}, Y={target_world_pos_np[1]:.3f}, Z={target_world_pos_np[2]:.3f}]")
                                    except Exception as e:
                                        print(f"Error calculating target world position: {e}")
                            except Exception as e:
                                print(f"Error accessing myblock: {e}")
                    
                    # --- 4. 서브태스크에 맞는 Target과 EEF 간의 거리 출력 ---
                    eef_to_current_target_dist_key = "eef_to_current_target_dist"
                    if isinstance(obv[0], dict) and "policy" in obv[0] and eef_to_current_target_dist_key in obv[0]["policy"]:
                        distance_tensor = obv[0]["policy"][eef_to_current_target_dist_key]
                        try:
                            distance = distance_tensor[0].item() if distance_tensor.dim() == 2 else distance_tensor[0, 0].item()
                            print(f"4. EEF to Current Target Dist: {distance:.4f}")
                        except Exception as e:
                            print(f"Error extracting distance value: {e}, tensor shape: {distance_tensor.shape}")
                    
                    # --- 3. 서브태스크 성공 여부 출력 ---
                    # 성공한 서브태스크를 추적하기 위한 변수 초기화
                    if not hasattr(env, '_completed_subtasks'):
                        env._completed_subtasks = set()
                    
                    # subtask_terms 확인
                    group_name = "subtask_terms"
                    if isinstance(obv[0], dict) and group_name in obv[0]:
                        # 모든 서브태스크 상태 확인
                        all_subtasks_status = {}
                        for subtask_key in obv[0][group_name]:
                            try:
                                subtask_tensor = obv[0][group_name][subtask_key]
                                is_completed = False
                                if subtask_tensor.numel() > 0:  # 비어있지 않은 텐서인지 확인
                                    is_completed = bool(subtask_tensor[0].item() if subtask_tensor.dim() == 2 else subtask_tensor[0, 0].item())
                                all_subtasks_status[subtask_key] = is_completed
                                
                                # 새로 완료된 서브태스크 확인
                                if is_completed and subtask_key not in env._completed_subtasks:
                                    env._completed_subtasks.add(subtask_key)
                                    print(f"3. 🎉 NEW SUCCESS: Subtask '{subtask_key}' completed!")
                            except (IndexError, AttributeError, RuntimeError) as e:
                                print(f"Error processing subtask {subtask_key}: {e}")
                        
                        # 모든 서브태스크 상태 출력
                        status_str = " | ".join([f"{key}: {'✅' if val else '❌'}" for key, val in all_subtasks_status.items()])
                        print(f"3. All Subtasks Status: {status_str}")
                    
                    # --- 5. Terminations 완료 시 성공 메시지 ---
                    if success_term is not None:
                        try:
                            success_result = success_term.func(env, **success_term.params)
                            if success_result.numel() > 0 and bool(success_result[0]):
                                if not hasattr(env, '_success_message_shown'):
                                    env._success_message_shown = False
                                
                                if not env._success_message_shown:
                                    print("\n" + "="*50)
                                    print("🎉🎉🎉 TASK COMPLETED SUCCESSFULLY! 🎉🎉🎉")
                                    print("="*50 + "\n")
                                    env._success_message_shown = True
                                
                                # 성공 단계 카운트 출력
                                print(f"5. Success Steps: {success_step_count}/{args_cli.num_success_steps}")
                            elif hasattr(env, '_success_message_shown'):
                                env._success_message_shown = False
                        except Exception as e:
                            print(f"Error checking success condition: {e}")
                    
                    # 마지막 출력 시간 갱신
                    last_print_time = current_time
                else:
                    env.sim.render()

                if success_term is not None:
                    if bool(success_term.func(env, **success_term.params)[0]):
                        success_step_count += 1
                        if success_step_count >= args_cli.num_success_steps:
                            print("SUCCESS!!!!!!!!!!!!!!!!!!!!!!!!")
                            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                            env.recorder_manager.set_success_to_episodes(
                                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                            )
                            env.recorder_manager.export_episodes([0])

                            should_reset_recording_instance = True
                else:
                    success_step_count = 0

            # print out the current demo count if it has changed
            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
                print(label_text)

            if should_reset_recording_instance:
                env.sim.reset()
                env.recorder_manager.reset()
                env.reset()
                should_reset_recording_instance = False
                success_step_count = 0
                instruction_display.show_demo(label_text)

            if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break

            # check that simulation is stopped or not
            if env.sim.is_stopped():
                break

            if rate_limiter:
                rate_limiter.sleep(env)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
