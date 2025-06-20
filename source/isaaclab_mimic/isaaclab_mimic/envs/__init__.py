# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sub-package with environment wrappers for Isaac Lab Mimic."""

import gymnasium as gym

from .franka_stack_ik_abs_mimic_env import FrankaCubeStackIKAbsMimicEnv
from .franka_stack_ik_abs_mimic_env_cfg import FrankaCubeStackIKAbsMimicEnvCfg
from .franka_stack_ik_rel_blueprint_mimic_env_cfg import FrankaCubeStackIKRelBlueprintMimicEnvCfg
from .franka_stack_ik_rel_mimic_env import FrankaCubeStackIKRelMimicEnv
from .franka_stack_ik_rel_mimic_env_cfg import FrankaCubeStackIKRelMimicEnvCfg
from .franka_stack_ik_rel_visuomotor_mimic_env_cfg import FrankaCubeStackIKRelVisuomotorMimicEnvCfg
from .my_rb_mimic_env_cfg import MyRbPaintMimicEnvCfg
from .my_rb_mimic_env import MyRBPaintMimicEnv
from .rainbow_paint_ik_rel_mimic_env import RainbowPaintIKRelMimicEnv
from .rainbow_paint_ik_rel_mimic_env_cfg import RainbowPaintIKRelMimicEnvCfg

##
# Inverse Kinematics - Relative Pose Control
##
gym.register(
    id="Isaac-Paint-Rainbow-IK-Rel-Mimic-v0",
    entry_point="isaaclab_mimic.envs:RainbowPaintIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": rainbow_paint_ik_rel_mimic_env_cfg.RainbowPaintIKRelMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Paint-RB10-IK-Rel-Mimic-v0",
    entry_point="isaaclab_mimic.envs:MyRBPaintMimicEnv",
    kwargs={
        "env_cfg_entry_point": my_rb_mimic_env_cfg.MyRbPaintMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_rel_blueprint_mimic_env_cfg.FrankaCubeStackIKRelBlueprintMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Abs-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKAbsMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_abs_mimic_env_cfg.FrankaCubeStackIKAbsMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_rel_visuomotor_mimic_env_cfg.FrankaCubeStackIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)
