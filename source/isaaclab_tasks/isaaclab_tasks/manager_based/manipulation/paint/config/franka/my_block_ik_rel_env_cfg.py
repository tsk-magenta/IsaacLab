# my_block_ik_rel_env_cfg.py (상속 활용 방식)

from isaaclab.utils import configclass
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab_assets.robots.rb import RB10_CFG, RB10_HIGH_PD_CFG # <--- 이 라인을 추가하거나 수정
from . import paint_joint_pos_env_cfg # MyBlock 추가, 큐브 주석처리된 버전

@configclass
class MyBlockIKRelEnvCfg(paint_joint_pos_env_cfg.FrankaPaintEnvCfg):
    """Configuration for the RB10 MyBlock environment using inheritance."""

    def __post_init__(self):
        # 부모 클래스의 __post_init__ 호출
        super().__post_init__()

        # RB10 로봇 설정 변경
        self.scene.robot = RB10_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
                
        # RB10 로봇을 위한 IK 액션 설정
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            # RB10 로봇의 실제 관절 이름
            joint_names=["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"],
            body_name="link6", # RB10의 마지막 링크
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            # RB10의 마지막 링크 기준 오프셋
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, -0.28, 0.0]),
        )

        # # 팔 액션을 IK 방식으로 변경
        # self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        #     asset_name="robot",
        #     joint_names=["panda_joint.*"],
        #     body_name="panda_hand",
        #     controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        #     scale=0.5,
        #     body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        # )

        # --- 4. 팔 액션을 IK 방식으로 설정 (관절 이름 변경 필요) ---
# paint/config/franka/my_block_ik_rel_env_cfg.py
# ... (기존 import들) ...

# @configclass
# class MyBlockIKRelEnvCfg(paint_joint_pos_env_cfg.FrankaPaintEnvCfg):
#     # ...
#     def __post_init__(self):
#         super().__post_init__()
#         # ... (로봇, 그리퍼, 이벤트 설정) ...

#         self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
#             asset_name="robot",
#             # USD 파일에서 확인한 실제 관절 이름으로 수정
#             # 오류 메시지에 나온 이름들을 순서대로 나열하거나,
#             # 또는 각 이름에 맞는 정규 표현식 사용.
#             # 예시: 6개의 관절이 순서대로 base, shoulder, elbow, wrist1, wrist2, wrist3 라면
#             joint_names=["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"],
#             # 또는 만약 USD 내 이름이 "joint_base", "joint_shoulder" 등이라면
#             # joint_names_expr=["joint_base", "joint_shoulder", ...],
#             # 또는 정규표현식으로: joint_names_expr=["joint_.*"], (단, 순서 보장 및 다른 조인트와 혼동 주의)

#             # body_name도 USD에서 확인한 실제 엔드 이펙터 링크 이름으로 수정해야 할 수 있음
#             # (만약 link6가 아니라면)
#             body_name="wrist3", # 또는 실제 마지막 링크 이름 (예: 'link_ee', 'tool0')
#             controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
#             scale=0.5,
#             body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1]), # 이 값도 재확인 필요
#         )

#         # self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
#         #     asset_name="robot",
#         #     # RB10 로봇의 관절 이름으로 변경
#         #     joint_names=["root_joint", "link[0-6]"],  # Franka의 "panda_joint.*" 대신
#         #     body_name="link6",  # Franka의 "panda_hand" 대신
#         #     controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
#         #     scale=0.5,
#         #     # 오프셋도 적절히 조정 필요
#         #     body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1]),
#         # )


#         # ... (그리퍼 액션 설정) ...

#         # # --- 5. 그리퍼 액션 설정 ---
#         # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
#         #     asset_name="gripper",  # "robot" 대신 "gripper"로 변경
#         #     joint_names=["panda_finger_joint.*"],
#         #     open_command_expr={"panda_finger_joint.*": 0.04},
#         #     close_command_expr={"panda_finger_joint.*": 0.0},
#         # )

#         # 그리퍼 액션은 부모 클래스의 것을 그대로 사용 (변경 필요 시 여기서 덮어씀)
#         # self.actions.gripper_action = ...

#         # Observation, Termination, Event 등도 필요하다면 여기서 덮어쓰거나 수정 가능
#         # 예: self.observations.policy.my_new_obs = ObsTerm(...)
#         # 예: self.terminations.my_new_termination = DoneTerm(...)
