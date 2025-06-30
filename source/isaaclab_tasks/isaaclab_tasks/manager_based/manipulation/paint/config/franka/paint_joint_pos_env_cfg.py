# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# /home/hys/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/paint/config/franka/paint_joint_pos_env_cfg.py

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


from isaaclab_tasks.manager_based.manipulation.paint import mdp
# from isaaclab_tasks.manager_based.manipulation.paint.mdp import franka_paint_events
from isaaclab_tasks.manager_based.manipulation.paint.mdp import rb_paint_events
from isaaclab_tasks.manager_based.manipulation.paint.paint_env_cfg import PaintEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
# from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
from isaaclab_assets.robots.rb import RB10_CFG


## 내가 추가한 modules
import os
import isaaclab.sim as sim_utils
import math
from isaaclab.sensors import CameraCfg

# RB10 초기 자세 (도 단위)
rb10_initial_joint_angles_deg_for_event = [
    10.0,  # Base joint (예: 이 태스크에서는 약간 회전)
    -45.0, # Shoulder joint
    60.0,  # Elbow joint
    -10.0, # Wrist 1 joint
    -80.0, # Wrist 2 joint
    5.0    # Wrist 3 joint
]
# 도 단위를 라디안 단위로 변환
rb10_initial_joint_angles_rad_for_event = [math.radians(deg) for deg in rb10_initial_joint_angles_deg_for_event]

# 표준편차도 필요하다면 여기서 변환
randomize_std_rad_for_event = math.radians(1.0) # 예: 1도 표준편차

@configclass
class EventCfg:
    """Configuration for events."""
    # 이제 EventCfg 내에는 EventTermCfg 타입의 객체만 존재합니다.
    init_rb10_arm_pose = EventTerm(
        func=rb_paint_events.set_rb10_default_pose, # RB10 로봇용 함수
        mode="startup",
        params={
            "default_pose": rb10_initial_joint_angles_rad_for_event, # 미리 변환된 라디안 리스트 사용
        },
    )

    randomize_rb10_joint_state = EventTerm(
        func=rb_paint_events.randomize_rb10_joints_by_gaussian_offset, # RB10 로봇용 함수
        mode="reset",
        params={
            "mean": 0.0,
            "std": randomize_std_rad_for_event, # 미리 변환된 라디안 값 사용 (또는 직접 라디안 값 입력)
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )



@configclass
class FrankaPaintEnvCfg(PaintEnvCfg):
    def __post_init__(self):
        
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        self.scene.robot = RB10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        config_directory = os.path.dirname(__file__)
        my_block_usd_path = os.path.join(config_directory, "block_hys_no_materials.usd")

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]
        
        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"], 
            scale=0.5, 
            use_default_offset=True
        )
        # spray_action 대신 dummy_action 추가
        self.actions.dummy_action = mdp.dummy_action.DummyActionCfg(
            asset_name="robot" # 더미 액션은 어떤 에셋에든 연결될 수 있습니다.
        )

        # self.actions.spray_action = mdp.SprayOnOffActionCfg(
        #     asset_name="robot",
        #     ee_frame_asset_name="ee_frame",
        #     nozzle_tip_frame_name="end_effector",
        #     projectile_prim_name="NozzleProjectile",
        #     # projectile_type="Cube", # 기본값 사용 시 생략 가능
        #     # projectile_scale=[0.02, 0.02, 0.02], # 기본값
        #     # projectile_mass=0.01, # 기본값
        #     projectile_parent_link_rel_path="link6", 
        #     robot_prim_path_for_single_env=actual_robot_prim_path,
            
        #     # 새로운 설정값 (필요에 따라 여기서 오버라이드)
        #     spray_interval=0.25,  # 0.25초 간격
        #     max_projectiles=50,   # 최대 50개
        #     custom_projectile_initial_speed=100.0 # 발사 속도 100
        # )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/link6",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, -0.28, 0.0],
                    ),
                ),
            ],
        ) 

        self.scene.table_camera = CameraCfg(
            # 카메라 Prim이 생성될 경로 (환경 네임스페이스 아래에 생성)
            prim_path="{ENV_REGEX_NS}/table_camera",
            # 카메라 업데이트 주기 (0이면 매 렌더링 스텝)
            update_period=0.0, # 필요에 따라 조정 (예: 0.0333 -> ~30Hz)
            # 카메라 해상도
            height=480,
            width=640,
            # 수집할 데이터 종류 (rgb, depth 등 추가 가능)
            data_types=["rgb"],
            # 카메라 내부 속성 설정 (Pinhole 카메라 사용)
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5), # 물체가 보이는 최소/최대 거리
            ),
            # --- 카메라 위치 및 방향 설정 ---
            offset=CameraCfg.OffsetCfg(
                # prim_path 기준으로 상대적 위치 [X, Y, Z] (월드 좌표계 기준)
                # pos=(-3, -2.5, 4), # 예시: 테이블 중앙 위쪽
                pos=(-2, 1.56203, 1.32506), # 예시: 테이블 중앙 위쪽
                # prim_path 기준으로 상대적 회전 [w, x, y, z] (쿼터니언)
                # rot=(0.81215, 0.37003, -0.13992, -0.42885), # 예시: 아래를 약간 비스듬히 바라봄 
                rot=(-0.46447, -0.32107, 0.48129, 0.67047), # 예시: 아래를 약간 비스듬히 바라봄 

                convention="opengl", # 좌표계 관례 (보통 "ros" 또는 "opengl")
            ),

        )

        # Rigid body properties of each cube
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        self.scene.myblock = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/MyBlock",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.7, 0, 0.5], rot=[0.707, 0, 0, -0.707]), #rot=[0.707, 0, 0, 0.707]),
            spawn=UsdFileCfg(
                usd_path=my_block_usd_path,
                scale=(5.0, 5.0, 5.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "myblock")],
            ),
        )
