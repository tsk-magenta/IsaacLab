# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# /home/hys/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/paint/paint_env_cfg.py
from dataclasses import MISSING, field
from typing import TYPE_CHECKING, List, Dict

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp
scale = float(5.0)
TARGET_LOCATIONS = {
    "target1": [-0.07 * scale, 0.0 * scale, 0.09 * scale],
    "target2": [0.07 * scale, 0.0 * scale, 0.09 * scale],
    }
ths = 0.3
##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##

@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg = MISSING # 또는 IKActionCfg
    dummy_action: mdp.DummyActionCfg = MISSING

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        eef_to_current_target_dist = ObsTerm(
            func=mdp.eef_to_myblock_current_target_dist,
            params={
                "eef_frame_cfg": SceneEntityCfg("ee_frame"),
                "myblock_cfg": SceneEntityCfg("myblock")
            }
        )

        # current_spray_state = ObsTerm(
        #     func=mdp.spray_on_off_state,
        #     params={"spray_action_cfg_name": "spray_action"} # ActionsCfg의 필드 이름과 일치
        # )

        ####################################
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        approach_1 = ObsTerm(
        func=mdp.check_eef_near_myblock_target,
        params={
            "eef_frame_cfg": SceneEntityCfg("ee_frame"),
            "myblock_cfg": SceneEntityCfg("myblock"),
            "target_local_pos": TARGET_LOCATIONS["target1"],
            "threshold": ths,
            }
        )

        approach_2 = ObsTerm(
            func=mdp.check_eef_near_myblock_target,
            params={
                "eef_frame_cfg": SceneEntityCfg("ee_frame"),
                "myblock_cfg": SceneEntityCfg("myblock"),
                "target_local_pos": TARGET_LOCATIONS["target2"],
                "threshold": ths,
            }
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 새로운 종료 조건 - 모든 서브태스크 완료 확인
    # success = DoneTerm(
    #     func=mdp.check_all_subtasks_completed,
    #     params={
    #         "eef_frame_cfg": SceneEntityCfg("ee_frame"),
    #         "myblock_cfg": SceneEntityCfg("myblock"),
    #         "threshold": ths,
    #     }
    # )
    success = DoneTerm(
        func=mdp.check_eef_near_myblock_target,
        params={
            "eef_frame_cfg": SceneEntityCfg("ee_frame"),
            "myblock_cfg": SceneEntityCfg("myblock"),
            "target_local_pos": TARGET_LOCATIONS["target2"],  # 마지막 타겟(approach2) 위치
            "threshold": ths,
        }
    )
   


@configclass
class PaintEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the painting environment."""
    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands = None
    rewards = None
    events = None
    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(-0.1, -0.5, -1.05),
        anchor_rot=(0.866, 0, 0, -0.5),
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
