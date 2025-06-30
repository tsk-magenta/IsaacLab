# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Isaac Lab Mimic environment wrapper class for RB10 Paint Mimic env.
이 파일은 페인팅 작업을 위한 RB10 로봇용 Mimic 환경 래퍼 클래스입니다.
franka_stack_ik_rel_mimic_env.py의 구조를 따릅니다.
"""

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv
import omni.log


class MyRBPaintMimicEnv(ManagerBasedRLMimicEnv):
    """
    Isaac Lab Mimic environment wrapper class for RB10 Paint env.
    RB10 로봇을 사용한 페인팅 작업을 위한 Mimic 환경 래퍼 클래스입니다.
    """

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.
        현재 로봇 엔드 이펙터 포즈를 가져옵니다. 로봇 엔드 이펙터 컨트롤러에서 사용하는 것과 동일한 프레임이어야 합니다.

        Args:
            eef_name: Name of the end effector (예: "RB10").
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
        """
        if env_ids is None:
            env_ids = slice(None)

        # observations.py와 paint_env_cfg.py에서 정의된 관찰 버퍼에서 EEF 위치와 쿼터니언을 가져옵니다.
        # 관찰 버퍼의 키는 ObservationsCfg.PolicyCfg에서 정의되어 있습니다.
        if "policy" not in self.obs_buf:
            raise KeyError("Observation buffer does not contain 'policy' group.")
        if "eef_pos" not in self.obs_buf["policy"]:
            raise KeyError("Observation buffer's 'policy' group does not contain 'eef_pos'.")
        if "eef_quat" not in self.obs_buf["policy"]:
            raise KeyError("Observation buffer's 'policy' group does not contain 'eef_quat'.")

        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]  # Isaac Lab 기본 쿼터니언 형식은 (w, x, y, z) 입니다.

        # 위치와 쿼터니언을 사용하여 4x4 포즈 행렬 생성
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """
        Takes a target pose and gripper action for the end effector controller and returns an action
        (usually a normalized delta pose action) to try and achieve that target pose.
        Noise is added to the target pose action if specified.
        엔드 이펙터 컨트롤러의 목표 포즈와 스프레이 액션을 받아 해당 목표 포즈를 달성하기 위한 액션을 반환합니다.
        지정된 경우 액션에 노이즈가 추가됩니다.

        Args:
            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector (key should match eef_name).
            gripper_action_dict: Dictionary of spray actions for each end-effector (key should match eef_name).
            action_noise_dict: Optional noise dictionary (key should match eef_name).
            env_id: Environment index to get the action for.

        Returns:
            An action torch.Tensor that's compatible with env.step().
        """
        # 환경 설정에서 EEF 이름을 가져옵니다. 
        # self.cfg는 MyRbPaintMimicEnvCfg 인스턴스여야 합니다.
        if not hasattr(self, 'cfg') or not hasattr(self.cfg, 'subtask_configs') or not self.cfg.subtask_configs:
            raise ValueError("Environment configuration (self.cfg.subtask_configs) is not properly set.")
        
        # 첫 번째 키 값을 EEF 이름으로 사용합니다 (보통 "RB10" 또는 "franka")
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        # 지정된 eef에 대한 목표 위치 및 회전 
        target_eef_pose = target_eef_pose_dict.get(eef_name)
        if target_eef_pose is None:
            raise KeyError(f"Target pose for EEF '{eef_name}' not found in target_eef_pose_dict.")
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        # 현재 위치 및 회전
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # 상대적 위치 계산 (상대적 IK 가정)
        delta_position = target_pos - curr_pos

        # 상대적 회전 계산 (상대적 IK)
        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        # 지정된 eef에 대한 스프레이 액션 가져오기
        # 여기서는 gripper_action을 spray_action으로 사용합니다
        spray_action = gripper_action_dict.get(eef_name)
        if spray_action is None:
            raise KeyError(f"Spray action for EEF '{eef_name}' not found in gripper_action_dict.")

        # 포즈 액션 생성
        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        
        # 노이즈 적용 (옵션)
        if action_noise_dict is not None and eef_name in action_noise_dict:
            noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
            pose_action += noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        # 최종 액션 구성 (delta_pose, spray_action)
        return torch.cat([pose_action, spray_action], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_eef_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.
        액션(env.step과 호환 가능)을 엔드 이펙터 컨트롤러의 타겟 포즈로 변환합니다.
        @target_eef_pose_to_action의 역함수입니다. 일반적으로 기록된 액션을 사용하여 
        데모 궤적에서 타겟 컨트롤러 포즈 시퀀스를 추론하는 데 사용됩니다.

        Args:
            action: Environment action. Shape is (num_envs, action_dim)

        Returns:
            A dictionary of eef pose torch.Tensor that @action corresponds to
        """
        if not hasattr(self, 'cfg') or not hasattr(self.cfg, 'subtask_configs') or not self.cfg.subtask_configs:
            raise ValueError("Environment configuration (self.cfg.subtask_configs) is not properly set.")
        
        # 첫 번째 키 값을 EEF 이름으로 사용
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        # 액션에서 델타 위치 및 델타 회전 추출
        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        # 현재 EEF 포즈 가져오기
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # 타겟 위치 계산
        target_pos = curr_pos + delta_position

        # 델타 회전(축-각도)을 쿼터니언으로 변환
        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        
        # 0으로 나누는 것을 방지
        is_close_to_zero_angle = torch.isclose(delta_rotation_angle, torch.zeros_like(delta_rotation_angle)).squeeze(1)
        delta_rotation_axis = torch.zeros_like(delta_rotation)
        non_zero_mask = ~is_close_to_zero_angle
        delta_rotation_axis[non_zero_mask] = delta_rotation[non_zero_mask] / delta_rotation_angle[non_zero_mask]

        delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle.squeeze(1), delta_rotation_axis)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()

        return {eef_name: target_poses}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts the gripper actuation part from a sequence of env actions (compatible with env.step).
        For RB10 Paint, this is the spray on/off action.
        env 액션 시퀀스에서 그리퍼(스프레이) 액추에이션 부분을 추출합니다.
        RB10 페인트의 경우, 이것은 스프레이 켜기/끄기 액션입니다.

        Args:
            actions: environment actions. Shape is (num_envs, num steps in a demo, action_dim).

        Returns:
            A dictionary of torch.Tensor spray actions. Key to each dict is an eef_name.
        """
        if not hasattr(self, 'cfg') or not hasattr(self.cfg, 'subtask_configs') or not self.cfg.subtask_configs:
            raise ValueError("Environment configuration (self.cfg.subtask_configs) is not properly set.")
        
        # 첫 번째 키 값을 EEF 이름으로 사용
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        # 액션의 마지막 차원이 스프레이 액션 (gripper_action과 동일한 위치)
        spray_actions = actions[:, -1:]
        return {eef_name: spray_actions}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """
        Gets a dictionary of termination signal flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. The implementation of this method is
        required if intending to enable automatic subtask term signal annotation when running the
        dataset annotation tool.
        작업의 각 서브태스크에 대한 종료 신호 플래그 사전을 가져옵니다. 서브태스크가 완료되면 
        플래그는 1이고, 그렇지 않으면 0입니다. 이 메서드는 데이터셋 주석 도구를 실행할 때 
        자동 서브태스크 주석을 활성화하려는 경우 필요합니다.

        Args:
            env_ids: Environment indices to get the termination signals for. If None, all envs are considered.

        Returns:
            A dictionary termination signal flags (False or True) for each subtask.
        """
        if env_ids is None:
            env_ids = slice(None)

        signals = dict()

        # paint_env_cfg.py에서 정의된 서브태스크 관찰 그룹과 용어를 사용합니다
        # 관찰 그룹 이름은 ObservationsCfg.SubtaskCfg와 일치해야 합니다
        obs_group_name = "subtask_terms"
        
        # observations.py와 paint_env_cfg.py에서 두 개의 접근 서브태스크를 확인했습니다
        subtask_term_names = ["approach_1", "approach_2"]
        
        if obs_group_name in self.obs_buf:
            for term_name in subtask_term_names:
                if term_name in self.obs_buf[obs_group_name]:
                    signals[term_name] = self.obs_buf[obs_group_name][term_name][env_ids]
                else:
                    omni.log.warn(f"Subtask signal '{term_name}' not found in observation buffer's '{obs_group_name}' group.")
                    num_requested_envs = len(env_ids) if isinstance(env_ids, Sequence) else self.num_envs
                    signals[term_name] = torch.zeros(num_requested_envs, device=self.device, dtype=torch.bool)
        else:
            omni.log.warn(f"Observation group '{obs_group_name}' not found in observation buffer.")
            num_requested_envs = len(env_ids) if isinstance(env_ids, Sequence) else self.num_envs
            for term_name in subtask_term_names:
                signals[term_name] = torch.zeros(num_requested_envs, device=self.device, dtype=torch.bool)
        
        return signals