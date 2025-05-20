# paint/mdp/rb_paint_events.py
from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import omni.log # 로그를 위해 추가

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# RB10 로봇 (6 DOF 암 관절)과 Panda 그리퍼 (2 DOF 핑거 관절)를 포함하는
# 통합된 로봇 시스템의 기본 관절 각도 (총 8 DOF).
# 이 값은 rb.py의 RB10_WITH_PANDA_GRIPPER_CFG.init_state.joint_pos 와 일관성 있게 설정해야 합니다.
# 순서: RB10의 6개 관절, 그 다음 Panda 그리퍼의 2개 관절.
# RB10 로봇의 기본 관절 각도 (6 DOF)
# RB10_DEFAULT_JOINT_POS = [0.0, -0.5, 0.0, -1.8, 0.0, 1.5]

def set_rb10_default_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """RB10 로봇의 기본 관절 위치 설정 (기존 함수와 동일한 매개변수)
    
    Args:
        env: 환경 인스턴스
        env_ids: 적용할 환경 ID
        default_pose: 기본 관절 위치
        asset_cfg: 로봇 설정
    """
    # 로봇 가져오기
    asset = env.scene[asset_cfg.name]
    asset.data.default_joint_pos = torch.tensor(default_pose, device=env.device).repeat(env.num_envs, 1)


def randomize_rb10_joints_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomizes the RB10 robot's joint positions by a Gaussian offset."""
    robot_asset: Articulation = env.scene[robot_cfg.name]

    # 현재 관절 상태 복사
    joint_pos = robot_asset.data.default_joint_pos[env_ids].clone()
    joint_vel = robot_asset.data.default_joint_vel[env_ids].clone()
    
    # 관절 이름 가져오기
    joint_names = robot_asset.data.joint_names
    
    # RB10 관절만 랜덤화 (그리퍼는 제외)
    rb10_joint_indices = []
    for i, name in enumerate(joint_names):
        if name in ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]:
            rb10_joint_indices.append(i)
    
    # 선택된 관절에만 노이즈 적용
    for joint_idx in rb10_joint_indices:
        noise = math_utils.sample_gaussian(mean, std, (len(env_ids), 1), joint_pos.device)
        joint_pos[:, joint_idx:joint_idx+1] += noise

    # 관절 제한 범위 내에서 조정
    joint_pos_limits = robot_asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # 물리 시뮬레이션에 적용
    robot_asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot_asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    robot_asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    
    # 디버깅
    print(f"Randomized {len(rb10_joint_indices)} RB10 joints")

# randomize_gripper_joints_by_gaussian_offset 함수는 이제 위 randomize_rb10_joints_by_gaussian_offset에 통합되므로 제거합니다.
# 만약 그리퍼 관절만 따로 무작위화해야 하는 특정 요구사항이 있다면, 별도로 함수를 만들고
# 해당 함수 내에서 그리퍼 관절의 인덱스를 찾아 노이즈를 적용하는 로직을 구현해야 합니다.
# 하지만 현재로서는 필요하지 않습니다.


# from __future__ import annotations

# import torch
# from typing import TYPE_CHECKING

# import isaaclab.utils.math as math_utils
# from isaaclab.assets import Articulation
# from isaaclab.managers import SceneEntityCfg

# if TYPE_CHECKING:
#     from isaaclab.envs import ManagerBasedEnv

# # RB10 로봇의 기본 관절 각도 (7 DOF) - 예시 값이며, 실제 로봇 및 작업에 맞게 조정 필요
# # (rb.py의 init_state.joint_pos 참고하여 일관성 있게 설정)
# # "root_joint": 0.0, "link0": 0.0, "link1": -0.5, "link2": 0.0, "link3": -1.8, "link4": 0.0, "link5": 1.5, "link6": 0.0
# # 관절 순서가 USD 파일 및 ArticulationCfg와 일치해야 함.
# # 보통 root_joint는 제외하고 실제 구동 관절만 포함. (여기서는 6 DOF 암 + 1 베이스로 가정)
# # 아래는 7개의 관절 값 예시 (link0 ~ link6 또는 root_joint ~ link5)
# # 로봇 USD 및 rb.py의 관절 순서 확인 필수.
# # rb.py에서는 "root_joint", "link0" ~ "link6"로 총 8개인데,
# # 일반적으로 "root_joint"는 베이스와 월드 간의 가상 조인트일 수 있고,
# # 실제 제어 대상은 link0~link6 또는 그 하위일 수 있음.
# # 여기서는 RB10의 주요 6개 암 관절 + 1개의 베이스 회전 관절로 가정하여 7개로 설정.
# # 실제 사용하는 관절에 맞춰 개수와 값을 조정하세요.
# # 만약 rb.py의 init_state 순서가 root_joint, link0, ..., link6 이고
# # articulation asset에서 이들이 모두 'robot' 에셋의 제어 대상이라면 8개가 됩니다.
# # 여기서는 예시로 7개 (주요 암 관절들)를 사용.
# RB10_DEFAULT_JOINT_POS = [0.0, -0.5, 0.0, -1.8, 0.0, 1.5, 0.0] # 예시 값 (7 DOF)

# # Panda 그리퍼 기본 관절 각도 (2 DOF) - 보통 양쪽 핑거가 대칭적으로 움직임
# # (rb.py의 PANDA_GRIPPER_CFG init_state 참고)
# PANDA_GRIPPER_DEFAULT_JOINT_POS = [0.04, 0.04] # 열린 상태

# def set_rb10_and_gripper_default_joint_pose(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     robot_default_pose: list[float],
#     gripper_default_pose: list[float],
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     gripper_cfg: SceneEntityCfg = SceneEntityCfg("gripper"),
# ):
#     """Sets the default joint poses for the RB10 robot and Panda gripper."""
#     # RB10 로봇 설정
#     robot_asset: Articulation = env.scene[robot_cfg.name]
#     # 로봇의 실제 DOF에 맞게 robot_default_pose 길이 확인 및 조정 필요
#     # 예를 들어, 로봇이 6 DOF 암이라면 robot_default_pose는 6개여야 함.
#     # rb.py의 RB10_1300E_CFG.init_state.joint_pos 를 보면 8개의 관절이 정의되어 있음.
#     # 어떤 관절이 'robot' 에셋의 제어 대상인지 명확히 해야 함.
#     # 여기서는 robot_default_pose가 'robot' 에셋의 모든 제어 가능한 관절을 포함한다고 가정.
#     if robot_asset.num_joints == len(robot_default_pose):
#         robot_asset.data.default_joint_pos = torch.tensor(robot_default_pose, device=env.device).repeat(env.num_envs, 1)
#     else:
#         print(f"[Warning] Mismatch in RB10 DOF for default_joint_pos. Expected {robot_asset.num_joints}, got {len(robot_default_pose)}")
#         # DOF 불일치 시, 일부만 설정하거나 에러 처리 (아래는 예시)
#         # 만약 robot_default_pose가 더 적은 수의 주요 관절만 포함한다면, 그에 맞게 처리.
#         # 여기서는 모든 환경에 대해 일단 채워넣음.
#         default_pos_tensor = torch.zeros(env.num_envs, robot_asset.num_joints, device=env.device)
#         num_to_fill = min(robot_asset.num_joints, len(robot_default_pose))
#         default_pos_tensor[:, :num_to_fill] = torch.tensor(robot_default_pose[:num_to_fill], device=env.device)
#         robot_asset.data.default_joint_pos = default_pos_tensor


#     # Panda 그리퍼 설정
#     gripper_asset: Articulation = env.scene[gripper_cfg.name]
#     if gripper_asset.num_joints == len(gripper_default_pose):
#         gripper_asset.data.default_joint_pos = torch.tensor(gripper_default_pose, device=env.device).repeat(env.num_envs, 1)
#     else:
#         print(f"[Warning] Mismatch in Gripper DOF for default_joint_pos. Expected {gripper_asset.num_joints}, got {len(gripper_default_pose)}")
#         default_pos_tensor = torch.zeros(env.num_envs, gripper_asset.num_joints, device=env.device)
#         num_to_fill = min(gripper_asset.num_joints, len(gripper_default_pose))
#         default_pos_tensor[:, :num_to_fill] = torch.tensor(gripper_default_pose[:num_to_fill], device=env.device)
#         gripper_asset.data.default_joint_pos = default_pos_tensor


# def randomize_rb10_joints_by_gaussian_offset(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     mean: float,
#     std: float,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ):
#     """Randomizes the RB10 robot's joint positions by a Gaussian offset.
#     Does not affect the gripper.
#     """
#     robot_asset: Articulation = env.scene[robot_cfg.name]

#     # Add gaussian noise to joint states
#     joint_pos = robot_asset.data.default_joint_pos[env_ids].clone()
#     joint_vel = robot_asset.data.default_joint_vel[env_ids].clone() # 보통 0으로 시작

#     # RB10 로봇 관절에만 노이즈 추가
#     # robot_asset.num_joints가 실제 로봇팔 관절 수와 일치해야 함
#     noise = math_utils.sample_gaussian(mean, std, (len(env_ids), robot_asset.num_joints), joint_pos.device)
#     joint_pos += noise

#     # Clamp joint pos to limits
#     joint_pos_limits = robot_asset.data.soft_joint_pos_limits[env_ids]
#     joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

#     # Set into the physics simulation
#     robot_asset.set_joint_position_target(joint_pos, env_ids=env_ids)
#     robot_asset.set_joint_velocity_target(joint_vel, env_ids=env_ids) # 속도는 0으로 설정
#     robot_asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


# def randomize_gripper_joints_by_gaussian_offset(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     mean: float,
#     std: float,
#     gripper_cfg: SceneEntityCfg = SceneEntityCfg("gripper"),
# ):
#     """Randomizes the Panda gripper's joint positions by a Gaussian offset."""
#     gripper_asset: Articulation = env.scene[gripper_cfg.name]

#     # Add gaussian noise to joint states
#     joint_pos = gripper_asset.data.default_joint_pos[env_ids].clone()
#     joint_vel = gripper_asset.data.default_joint_vel[env_ids].clone() # 보통 0으로 시작

#     # 그리퍼 관절에만 노이즈 추가 (보통 2 DOF)
#     noise = math_utils.sample_gaussian(mean, std, (len(env_ids), gripper_asset.num_joints), joint_pos.device)
#     joint_pos += noise

#     # Clamp joint pos to limits
#     joint_pos_limits = gripper_asset.data.soft_joint_pos_limits[env_ids]
#     joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

#     # Set into the physics simulation
#     gripper_asset.set_joint_position_target(joint_pos, env_ids=env_ids)
#     gripper_asset.set_joint_velocity_target(joint_vel, env_ids=env_ids) # 속도는 0으로 설정
#     gripper_asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

# # 참고: franka_paint_events.py 의 randomize_scene_lighting_domelight, sample_object_poses,
# # randomize_object_pose, randomize_rigid_objects_in_focus 등의 함수는
# # 로봇 종류와 직접적인 관련이 없으므로 필요하다면 그대로 가져오거나 임포트해서 사용할 수 있습니다.
# # 여기서는 로봇/그리퍼 관절 관련 이벤트만 새로 정의했습니다.