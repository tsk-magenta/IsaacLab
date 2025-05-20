# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import (
    agents,
    paint_ik_abs_env_cfg,
    paint_ik_rel_blueprint_env_cfg,
    paint_ik_rel_env_cfg,
    paint_ik_rel_instance_randomize_env_cfg,
    paint_ik_rel_visuomotor_env_cfg,
    paint_joint_pos_env_cfg,
    paint_joint_pos_instance_randomize_env_cfg,
    my_block_ik_rel_env_cfg,
    franka_paint_custom_env,
)

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Paint-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": paint_joint_pos_env_cfg.FrankaPaintEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Paint-Instance-Randomize-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": paint_joint_pos_instance_randomize_env_cfg.FrankaPaintInstanceRandomizeEnvCfg,
    },
    disable_env_checker=True,
)


##
# Inverse Kinematics - Relative Pose Control
##



################### 새로운 환경 등록 ###################
# gym.register(
#     id="Isaac-Paint-Cube-Franka-IK-Rel-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": stack_ik_rel_env_cfg.FrankaCubeStackEnvCfg,
#         "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
#     },
#     disable_env_checker=True,
# )

gym.register(
    id="Isaac-Paint-Franka-IK-Rel-v0",
    # --- entry_point 변경 ---
    # 형식: "패키지.경로.파일명:클래스명"
    # isaaclab_tasks.manager_based.manipulation.stack 이 기본 경로라고 가정
    entry_point="isaaclab_tasks.manager_based.manipulation.paint.config.franka.franka_paint_custom_env:FrankaPaintCustomEnv",
    kwargs={
        "env_cfg_entry_point": my_block_ik_rel_env_cfg.MyBlockIKRelEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

## rb 로봇 전용 환경 ##
gym.register(
    id="Isaac-Paint-RB10-IK-Rel-v0",
    entry_point="isaaclab_tasks.manager_based.manipulation.paint.config.franka.franka_paint_custom_env:FrankaPaintCustomEnv",
    kwargs={
        "env_cfg_entry_point": my_block_ik_rel_env_cfg.MyBlockIKRelEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

######################################################



gym.register(
    id="Isaac-Paint-Franka-IK-Rel-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": paint_ik_rel_visuomotor_env_cfg.FrankaPaintVisuomotorEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_84.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Paint-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": paint_ik_abs_env_cfg.FrankaPaintEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Paint-Instance-Randomize-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": paint_ik_rel_instance_randomize_env_cfg.FrankaPaintInstanceRandomizeEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Paint-Franka-IK-Rel-Blueprint-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": paint_ik_rel_blueprint_env_cfg.FrankaPaintBlueprintEnvCfg,
    },
    disable_env_checker=True,
)
