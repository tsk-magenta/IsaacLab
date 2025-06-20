# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# paint/mdp/observations.py

# paint/mdp/observations.py

from __future__ import annotations #PEP 563: Postponed Evaluation of Annotations

import torch
from typing import TYPE_CHECKING, List
import omni # For logging

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.sensors.frame_transformer.frame_transformer_data import FrameTransformerData
import isaaclab.utils.math as math_utils


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Global variable to track keyboard spray state - can be set from outside the environment
_KEYBOARD_IS_CREATING_PARTICLES = False

def set_keyboard_spray_state(state: bool):
    """Set the global keyboard spray state.
    This function can be called from outside the environment to update the state.
    
    Args:
        state: The new state (True for enabled, False for disabled).
    """
    global _KEYBOARD_IS_CREATING_PARTICLES
    _KEYBOARD_IS_CREATING_PARTICLES = state
    print(f"Updated global keyboard spray state to: {_KEYBOARD_IS_CREATING_PARTICLES}")

def spray(env: "ManagerBasedRLEnv", robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Determine if spray particles should be shown.
    
    Returns:
        torch.Tensor: Boolean tensor based on the keyboard's _is_creating_particles state.
        
    Note:
        This function returns a boolean tensor based on the global _KEYBOARD_IS_CREATING_PARTICLES
        variable, which needs to be updated from outside using set_keyboard_spray_state().
    """
    # Use the global keyboard state
    global _KEYBOARD_IS_CREATING_PARTICLES
    
    # Create a tensor filled with the global state
    spray_state = torch.full((env.num_envs, 1), _KEYBOARD_IS_CREATING_PARTICLES, 
                            dtype=torch.bool, device=env.device)
    
    return spray_state

def eef_to_myblock_current_target_dist(
    env: ManagerBasedRLEnv,
    eef_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    myblock_cfg: SceneEntityCfg = SceneEntityCfg("myblock")
) -> torch.Tensor:
    """
    엔드 이펙터와 현재 활성화된 타겟 위치 사이의 거리를 계산합니다.
    
    Args:
        env: 환경 인스턴스
        eef_frame_cfg: 엔드 이펙터 프레임 설정
        myblock_cfg: 대상 블록 설정
        
    Returns:
        엔드 이펙터와 현재 타겟 사이의 거리를 포함하는 텐서 (num_envs, 1)
    """
    # 객체 인스턴스 가져오기
    ee_frame: FrameTransformer = env.scene[eef_frame_cfg.name]
    myblock: RigidObject = env.scene[myblock_cfg.name]
    
    # 현재 타겟 위치 가져오기 (초기화 상태 확인)
    if not hasattr(env, 'all_target_local_positions') or not hasattr(env, 'get_current_target_local_pos'):
        # 초기화 상태에서는 기본값 반환 (초기화 중 호출될 때 사용)
        default_dist = torch.ones((env.num_envs, 1), device=env.device, dtype=torch.float32)
        return default_dist
    
    # 함수가 있다면 호출 시도
    try:
        current_target_local_pos = env.get_current_target_local_pos()
        if current_target_local_pos is None:
            return torch.ones((env.num_envs, 1), device=env.device, dtype=torch.float32)
    except Exception as e:
        omni.log.warn(f"Error calling get_current_target_local_pos: {str(e)}")
        return torch.ones((env.num_envs, 1), device=env.device, dtype=torch.float32)
    
    # 현재 월드 포즈 가져오기
    myblock_pos_w = myblock.data.root_pos_w
    myblock_quat_w = myblock.data.root_quat_w
    eef_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    
    # 로컬 좌표를 월드 좌표로 변환
    target_world_pos = math_utils.quat_apply(myblock_quat_w, current_target_local_pos) + myblock_pos_w
    
    # 거리 계산
    distance = torch.linalg.vector_norm(eef_pos_w - target_world_pos, dim=1, keepdim=True)
    
    return distance

def check_eef_near_myblock_target(
    env: ManagerBasedRLEnv,
    eef_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    myblock_cfg: SceneEntityCfg = SceneEntityCfg("myblock"),
    target_local_pos: List[float] = [0.0, 0.0, 0.0],
    threshold: float = 0.03
) -> torch.Tensor:
    """
    엔드 이펙터가 지정된 타겟 위치에 가까이 있는지 확인합니다.
    
    Args:
        env: 환경 인스턴스
        eef_frame_cfg: 엔드 이펙터 프레임 설정
        myblock_cfg: 대상 블록 설정
        target_local_pos: 블록 상의 타겟 위치 (로컬 좌표)
        threshold: 접근 성공으로 간주할 거리 임계값
        
    Returns:
        타겟에 접근한 환경에 대해 True를 포함하는 텐서 (num_envs, 1)
    """
    # 객체 인스턴스 가져오기
    ee_frame: FrameTransformer = env.scene[eef_frame_cfg.name]
    myblock: RigidObject = env.scene[myblock_cfg.name]
    
    # 현재 월드 포즈 가져오기
    myblock_pos_w = myblock.data.root_pos_w
    myblock_quat_w = myblock.data.root_quat_w
    eef_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    
    # 로컬 타겟 위치를 텐서로 변환하고 확장
    target_local_pos_tensor = torch.tensor(target_local_pos, device=env.device, dtype=torch.float32)
    target_local_pos_expanded = target_local_pos_tensor.unsqueeze(0).expand(env.num_envs, -1)
    
    # 로컬 좌표를 월드 좌표로 변환
    target_world_pos = math_utils.quat_apply(myblock_quat_w, target_local_pos_expanded) + myblock_pos_w
    
    # 거리 계산
    distance = torch.linalg.vector_norm(eef_pos_w - target_world_pos, dim=1)
    
    # 거리가 임계값보다 작은지 확인
    is_near = distance < threshold
    
    # 현재 타겟에 접근한 환경들에 대해 다음 타겟으로 진행 (선택적)
    if hasattr(env, 'advance_to_next_subtask'):
        env_ids_to_advance = torch.where(is_near)[0]
        if len(env_ids_to_advance) > 0:
            env.advance_to_next_subtask(env_ids_to_advance)
    
    return is_near.unsqueeze(-1)

def ee_frame_pos(
    env: ManagerBasedRLEnv, 
    eef_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """
    엔드 이펙터의 위치를 반환합니다.
    
    Args:
        env: 환경 인스턴스
        eef_frame_cfg: 엔드 이펙터 프레임 설정
        
    Returns:
        엔드 이펙터의 상대 위치 텐서 (num_envs, 3)
    """
    ee_frame_asset = env.scene.sensors.get(eef_frame_cfg.name)
    if not isinstance(ee_frame_asset, FrameTransformer):
        omni.log.error(f"FrameTransformer asset '{eef_frame_cfg.name}' not found or not a FrameTransformer for ee_frame_pos.")
        return torch.full((env.num_envs, 3), float('nan'), device=env.device, dtype=torch.float32)

    # 직접 target_pos_w에서 데이터 가져오기 (첫 번째 타겟 프레임 인덱스 0 사용)
    eef_pos_w = ee_frame_asset.data.target_pos_w[:, 0, :]
    
    # 환경 원점에 대한 상대 위치 계산
    ee_frame_pos_rel = eef_pos_w - env.scene.env_origins[:, 0:3]
    return ee_frame_pos_rel

def ee_frame_quat(
    env: ManagerBasedRLEnv, 
    eef_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """
    엔드 이펙터의 쿼터니언 방향을 반환합니다.
    
    Args:
        env: 환경 인스턴스
        eef_frame_cfg: 엔드 이펙터 프레임 설정
        
    Returns:
        엔드 이펙터의 쿼터니언 방향 텐서 (num_envs, 4)
    """
    ee_frame_asset = env.scene.sensors.get(eef_frame_cfg.name)
    if not isinstance(ee_frame_asset, FrameTransformer):
        omni.log.error(f"FrameTransformer asset '{eef_frame_cfg.name}' not found or not a FrameTransformer for ee_frame_quat.")
        return torch.full((env.num_envs, 4), float('nan'), device=env.device, dtype=torch.float32)
            
    # 직접 target_quat_w에서 데이터 가져오기 (첫 번째 타겟 프레임 인덱스 0 사용)
    eef_quat_w = ee_frame_asset.data.target_quat_w[:, 0, :]
    
    return eef_quat_w

# def spray_on_off_state(
#     env: ManagerBasedRLEnv,
#     spray_action_cfg_name: str = "spray_action"
# ) -> torch.Tensor:
#     processed_spray_action = None
#     if hasattr(env, "action_manager") and env.action_manager is not None:
#         action_term_instance = env.action_manager.get_term(spray_action_cfg_name)
#         if action_term_instance is not None and hasattr(action_term_instance, 'processed_actions'):
#             processed_spray_action = action_term_instance.processed_actions
#         else:
#             omni.log.warn(f"Action term '{spray_action_cfg_name}' not found in ActionManager or has no 'processed_actions' attribute for spray_on_off_state.")
#     else:
#         omni.log.warn("ActionManager not found in environment or not yet initialized for spray_on_off_state.")

#     if processed_spray_action is not None:
#         return processed_spray_action.float()
#     else:
#         omni.log.warn(f"Could not retrieve processed action for '{spray_action_cfg_name}' in spray_on_off_state. Returning zeros.")
#         return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)

##############################
def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("gripper")) -> torch.Tensor:
    """Update to use the gripper asset instead of robot asset."""
    gripper: Articulation = env.scene[robot_cfg.name]
    finger_joint_1 = gripper.data.joint_pos[:, 0].clone().unsqueeze(1)
    finger_joint_2 = -1 * gripper.data.joint_pos[:, 1].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)

# def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     robot: Articulation = env.scene[robot_cfg.name]
#     finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
#     finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)

#     return torch.cat((finger_joint_1, finger_joint_2), dim=1)

def cube_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """The position of the cubes in the world frame."""
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    return torch.cat((cube_1.data.root_pos_w, cube_2.data.root_pos_w, cube_3.data.root_pos_w), dim=1)


def instance_randomize_cube_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """The position of the cubes in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
    cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
    cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]

    cube_1_pos_w = []
    cube_2_pos_w = []
    cube_3_pos_w = []
    for env_id in range(env.num_envs):
        cube_1_pos_w.append(cube_1.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
        cube_2_pos_w.append(cube_2.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][1], :3])
        cube_3_pos_w.append(cube_3.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][2], :3])
    cube_1_pos_w = torch.stack(cube_1_pos_w)
    cube_2_pos_w = torch.stack(cube_2_pos_w)
    cube_3_pos_w = torch.stack(cube_3_pos_w)

    return torch.cat((cube_1_pos_w, cube_2_pos_w, cube_3_pos_w), dim=1)


def cube_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
):
    """The orientation of the cubes in the world frame."""
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    return torch.cat((cube_1.data.root_quat_w, cube_2.data.root_quat_w, cube_3.data.root_quat_w), dim=1)


def instance_randomize_cube_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """The orientation of the cubes in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
    cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
    cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]

    cube_1_quat_w = []
    cube_2_quat_w = []
    cube_3_quat_w = []
    for env_id in range(env.num_envs):
        cube_1_quat_w.append(cube_1.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
        cube_2_quat_w.append(cube_2.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][1], :4])
        cube_3_quat_w.append(cube_3.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][2], :4])
    cube_1_quat_w = torch.stack(cube_1_quat_w)
    cube_2_quat_w = torch.stack(cube_2_quat_w)
    cube_3_quat_w = torch.stack(cube_3_quat_w)

    return torch.cat((cube_1_quat_w, cube_2_quat_w, cube_3_quat_w), dim=1)


def object_obs(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Object observations (in world frame):
        cube_1 pos,
        cube_1 quat,
        cube_2 pos,
        cube_2 quat,
        cube_3 pos,
        cube_3 quat,
        gripper to cube_1,
        gripper to cube_2,
        gripper to cube_3,
        cube_1 to cube_2,
        cube_2 to cube_3,
        cube_1 to cube_3,
    """
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_1_pos_w = cube_1.data.root_pos_w
    cube_1_quat_w = cube_1.data.root_quat_w

    cube_2_pos_w = cube_2.data.root_pos_w
    cube_2_quat_w = cube_2.data.root_quat_w

    cube_3_pos_w = cube_3.data.root_pos_w
    cube_3_quat_w = cube_3.data.root_quat_w

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    gripper_to_cube_1 = cube_1_pos_w - ee_pos_w
    gripper_to_cube_2 = cube_2_pos_w - ee_pos_w
    gripper_to_cube_3 = cube_3_pos_w - ee_pos_w

    cube_1_to_2 = cube_1_pos_w - cube_2_pos_w
    cube_2_to_3 = cube_2_pos_w - cube_3_pos_w
    cube_1_to_3 = cube_1_pos_w - cube_3_pos_w

    return torch.cat(
        (
            cube_1_pos_w - env.scene.env_origins,
            cube_1_quat_w,
            cube_2_pos_w - env.scene.env_origins,
            cube_2_quat_w,
            cube_3_pos_w - env.scene.env_origins,
            cube_3_quat_w,
            gripper_to_cube_1,
            gripper_to_cube_2,
            gripper_to_cube_3,
            cube_1_to_2,
            cube_2_to_3,
            cube_1_to_3,
        ),
        dim=1,
    )


def instance_randomize_object_obs(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Object observations (in world frame):
        cube_1 pos,
        cube_1 quat,
        cube_2 pos,
        cube_2 quat,
        cube_3 pos,
        cube_3 quat,
        gripper to cube_1,
        gripper to cube_2,
        gripper to cube_3,
        cube_1 to cube_2,
        cube_2 to cube_3,
        cube_1 to cube_3,
    """
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
    cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
    cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_1_pos_w = []
    cube_2_pos_w = []
    cube_3_pos_w = []
    cube_1_quat_w = []
    cube_2_quat_w = []
    cube_3_quat_w = []
    for env_id in range(env.num_envs):
        cube_1_pos_w.append(cube_1.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
        cube_2_pos_w.append(cube_2.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][1], :3])
        cube_3_pos_w.append(cube_3.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][2], :3])
        cube_1_quat_w.append(cube_1.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
        cube_2_quat_w.append(cube_2.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][1], :4])
        cube_3_quat_w.append(cube_3.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][2], :4])
    cube_1_pos_w = torch.stack(cube_1_pos_w)
    cube_2_pos_w = torch.stack(cube_2_pos_w)
    cube_3_pos_w = torch.stack(cube_3_pos_w)
    cube_1_quat_w = torch.stack(cube_1_quat_w)
    cube_2_quat_w = torch.stack(cube_2_quat_w)
    cube_3_quat_w = torch.stack(cube_3_quat_w)

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    gripper_to_cube_1 = cube_1_pos_w - ee_pos_w
    gripper_to_cube_2 = cube_2_pos_w - ee_pos_w
    gripper_to_cube_3 = cube_3_pos_w - ee_pos_w

    cube_1_to_2 = cube_1_pos_w - cube_2_pos_w
    cube_2_to_3 = cube_2_pos_w - cube_3_pos_w
    cube_1_to_3 = cube_1_pos_w - cube_3_pos_w

    return torch.cat(
        (
            cube_1_pos_w - env.scene.env_origins,
            cube_1_quat_w,
            cube_2_pos_w - env.scene.env_origins,
            cube_2_quat_w,
            cube_3_pos_w - env.scene.env_origins,
            cube_3_quat_w,
            gripper_to_cube_1,
            gripper_to_cube_2,
            gripper_to_cube_3,
            cube_1_to_2,
            cube_2_to_3,
            cube_1_to_3,
        ),
        dim=1,
    )


# def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
#     ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

#     return ee_frame_pos


# def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
#     ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

#     return ee_frame_quat


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    gripper_threshold: float = 0.005,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    grasped = torch.logical_and(
        pose_diff < diff_threshold,
        torch.abs(robot.data.joint_pos[:, -1] - gripper_open_val.to(env.device)) > gripper_threshold,
    )
    grasped = torch.logical_and(
        grasped, torch.abs(robot.data.joint_pos[:, -2] - gripper_open_val.to(env.device)) > gripper_threshold
    )

    return grasped


def object_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    xy_threshold: float = 0.05,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
) -> torch.Tensor:
    """Check if an object is stacked by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    upper_object: RigidObject = env.scene[upper_object_cfg.name]
    lower_object: RigidObject = env.scene[lower_object_cfg.name]

    pos_diff = upper_object.data.root_pos_w - lower_object.data.root_pos_w
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    stacked = torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)

    stacked = torch.logical_and(
        torch.isclose(robot.data.joint_pos[:, -1], gripper_open_val.to(env.device), atol=1e-4, rtol=1e-4), stacked
    )
    stacked = torch.logical_and(
        torch.isclose(robot.data.joint_pos[:, -2], gripper_open_val.to(env.device), atol=1e-4, rtol=1e-4), stacked
    )

    return stacked
