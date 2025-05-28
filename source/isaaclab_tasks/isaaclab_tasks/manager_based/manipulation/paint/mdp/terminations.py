# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from segment_anything import SamAutomaticMaskGenerator

from typing import TYPE_CHECKING, List
import isaaclab.utils.math as math_utils # math_utils 임포트 추가

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
import omni.log

from .detect_paintarea import calculate_painting_completion_rate

def check_paintarea_completed(
    env: ManagerBasedRLEnv,
    image_path: str,
    threshold: float = 90.0,
) -> torch.Tensor:
    completion_rate = calculate_painting_completion_rate(image_path=image_path)
    
    # Create a properly shaped tensor for the return value
    result_tensor = torch.zeros((env.num_envs, 1), dtype=torch.bool, device=env.device)
    
    if completion_rate is None:
        print("Failed to calculate completion rate: completion_rate is None")
        return result_tensor  # Return all False
    
    print(f"Completion rate: {completion_rate}%")
    
    if completion_rate >= threshold:
        # Set all values to True
        result_tensor.fill_(True)
    
    return result_tensor

# def cubes_stacked(
#     env: ManagerBasedRLEnv,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
#     cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
#     cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
#     xy_threshold: float = 0.05,
#     height_threshold: float = 0.005,
#     height_diff: float = 0.0468,
#     gripper_open_val: torch.tensor = torch.tensor([0.04]),
#     atol=0.0001,
#     rtol=0.0001,
# ):
#     robot: Articulation = env.scene[robot_cfg.name]
#     cube_1: RigidObject = env.scene[cube_1_cfg.name]
#     cube_2: RigidObject = env.scene[cube_2_cfg.name]
#     cube_3: RigidObject = env.scene[cube_3_cfg.name]

#     pos_diff_c12 = cube_1.data.root_pos_w - cube_2.data.root_pos_w
#     pos_diff_c23 = cube_2.data.root_pos_w - cube_3.data.root_pos_w

#     # Compute cube position difference in x-y plane
#     xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)
#     xy_dist_c23 = torch.norm(pos_diff_c23[:, :2], dim=1)

#     # Compute cube height difference
#     h_dist_c12 = torch.norm(pos_diff_c12[:, 2:], dim=1)
#     h_dist_c23 = torch.norm(pos_diff_c23[:, 2:], dim=1)

#     # Check cube positions
#     stacked = torch.logical_and(xy_dist_c12 < xy_threshold, xy_dist_c23 < xy_threshold)
#     stacked = torch.logical_and(h_dist_c12 - height_diff < height_threshold, stacked)
#     stacked = torch.logical_and(h_dist_c23 - height_diff < height_threshold, stacked)

#     # Check gripper positions
#     stacked = torch.logical_and(
#         torch.isclose(robot.data.joint_pos[:, -1], gripper_open_val.to(env.device), atol=atol, rtol=rtol), stacked
#     )
#     stacked = torch.logical_and(
#         torch.isclose(robot.data.joint_pos[:, -2], gripper_open_val.to(env.device), atol=atol, rtol=rtol), stacked
#     )

#     return stacked
#####################################################################################
def check_all_subtasks_completed(
    env: ManagerBasedRLEnv,
    eef_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    myblock_cfg: SceneEntityCfg = SceneEntityCfg("myblock"),
    threshold: float = 0.35
) -> torch.Tensor:
    """
    모든 서브태스크가 한 번이라도 완료되었는지 확인합니다.
    각 서브태스크의 완료 여부를 기억하고, 모든 서브태스크가 완료되면 True를 반환합니다.
    """
    # 환경에 completions_tracker 속성 초기화
    if not hasattr(env, '_subtask_completions_tracker'):
        env._subtask_completions_tracker = {
            "approach_1": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "approach_2": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        }
        # 리셋 감지를 위한 시간 추적기 추가
    
    # 여기서는 환경의 리셋 상태를 직접 확인하거나, 특정 속성을 통해 리셋을 감지할 수 있습니다
    if hasattr(env, 'recorder_manager') and hasattr(env.recorder_manager, 'exported_successful_episode_count'):
        # 환경에 리셋 추적 변수 추가
        if not hasattr(env, '_last_exported_count'):
            env._last_exported_count = 0
        
        # 성공적인 에피소드 수가 증가했다면 리셋이 발생한 것으로 간주
        current_exported_count = env.recorder_manager.exported_successful_episode_count
        if current_exported_count > env._last_exported_count:
            print(f"✨ Detected environment reset! Clearing subtask completion tracker.")
            env._subtask_completions_tracker = {
                "approach_1": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
                "approach_2": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            }
            env._last_exported_count = current_exported_count
    
    # 현재 서브태스크 상태 확인
    subtask1_current = eef_near_myblock_target(
        env, 
        eef_frame_cfg=eef_frame_cfg, 
        myblock_cfg=myblock_cfg,
        target_local_pos=env.all_target_local_positions[0] if hasattr(env, "all_target_local_positions") and len(env.all_target_local_positions) > 0 
                   else torch.tensor([-0.07 * 5.0, 0.0, 0.09 * 5.0], device=env.device),
        threshold=threshold
    )
    
    subtask2_current = eef_near_myblock_target(
        env, 
        eef_frame_cfg=eef_frame_cfg, 
        myblock_cfg=myblock_cfg,
        target_local_pos=env.all_target_local_positions[1] if hasattr(env, "all_target_local_positions") and len(env.all_target_local_positions) > 1
                   else torch.tensor([0.07 * 5.0, 0.0, 0.09 * 5.0], device=env.device),
        threshold=threshold
    )
    
    # 원본 텐서 형태를 bool로 조정
    subtask1_current_bool = subtask1_current.squeeze(-1)
    subtask2_current_bool = subtask2_current.squeeze(-1)
    
    # 현재 상태가 True인 경우 완료 상태를 True로 업데이트 (한 번 완료되면 계속 완료 상태 유지)
    env._subtask_completions_tracker["approach_1"] = env._subtask_completions_tracker["approach_1"] | subtask1_current_bool
    env._subtask_completions_tracker["approach_2"] = env._subtask_completions_tracker["approach_2"] | subtask2_current_bool
    
    # 모든 서브태스크가 완료되었는지 확인
    all_subtasks_completed = env._subtask_completions_tracker["approach_1"] & env._subtask_completions_tracker["approach_2"]
    
    # 디버깅 정보 출력
    completed_envs = torch.where(all_subtasks_completed)[0]
    if len(completed_envs) > 0:
        print(f"All subtasks completed for environments: {completed_envs.tolist()}")
    
    return all_subtasks_completed.unsqueeze(-1)  # shape을 맞추기 위해 차원 추가

def eef_near_myblock_target(
    env: "ManagerBasedRLEnv",
    eef_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    myblock_cfg: SceneEntityCfg = SceneEntityCfg("myblock"), # MyBlock 설정 추가
    target_local_pos = [0.0, 0.0, 0.0], # 로컬 오프셋을 파라미터로 받음
    threshold: float = 0.03,
) -> torch.Tensor:
    """Check if the EEF is within the threshold distance to a specified local point on MyBlock."""
    # 객체 인스턴스 가져오기
    ee_frame: FrameTransformer = env.scene[eef_frame_cfg.name]
    myblock: RigidObject = env.scene[myblock_cfg.name]
    num_envs = env.num_envs
    device = env.device

    # 현재 월드 포즈 가져오기
    myblock_pos_w = myblock.data.root_pos_w
    myblock_quat_w = myblock.data.root_quat_w
    eef_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    # 로컬 타겟 위치를 텐서로 변환하고 확장
    if isinstance(target_local_pos, torch.Tensor):
    # 이미 텐서인 경우 detach().clone() 사용
        target_local_pos_tensor = target_local_pos.detach().clone().to(device=device, dtype=torch.float32)
    else:
    # 텐서가 아닌 경우(리스트, 튜플 등) torch.tensor() 사용
        target_local_pos_tensor = torch.tensor(target_local_pos, device=device, dtype=torch.float32)
    target_local_pos_expanded = target_local_pos_tensor.unsqueeze(0).expand(num_envs, -1)

    # 로컬 좌표를 월드 좌표로 변환
    target_world_pos = math_utils.quat_apply(myblock_quat_w, target_local_pos_expanded) + myblock_pos_w

    # 거리 계산
    distance = torch.linalg.vector_norm(eef_pos_w - target_world_pos, dim=1) # 결과 shape: (num_envs,)

    # 거리가 임계값보다 작으면 True 반환
    return distance < threshold

# terminations.py 파일

