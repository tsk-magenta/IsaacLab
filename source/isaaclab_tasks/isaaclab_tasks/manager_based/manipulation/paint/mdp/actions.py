# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium.spaces as spaces
import torch
from pxr import UsdGeom, Gf, Sdf, UsdPhysics, Usd, Kind # Usd import for Usd.TimeCode, Kind for ModelAPI
import omni.usd
import omni.kit.commands
import omni.log
import time 
from collections import deque

from isaaclab.envs.mdp.actions import ActionTermCfg, ActionTerm, JointPositionActionCfg, BinaryJointPositionActionCfg
from isaaclab.utils import configclass
from isaaclab.assets import Articulation
from isaaclab.sensors import FrameTransformer
from typing import TYPE_CHECKING, List, Sequence, Deque

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class SprayOnOffActionCfg(ActionTermCfg):
    """Configuration for the spray on/off action."""
    class_type: type[ActionTerm] = lambda cfg, env: SprayOnOffAction(cfg, env)
    asset_name: str = "robot"
    ee_frame_asset_name: str = "ee_frame"
    nozzle_tip_frame_name: str = "end_effector"
    projectile_prim_name: str = "NozzleProjectile"
    projectile_type: str = "Cube"
    projectile_scale: List[float] = [0.02, 0.02, 0.02]
    projectile_mass: float = 0.01
    projectile_parent_link_rel_path: str = "link6"
    robot_prim_path_for_single_env: str | None = None
    spray_interval: float = 0.25
    max_projectiles: int = 50
    custom_projectile_initial_speed: float = 100.0


class SprayOnOffAction(ActionTerm):
    cfg: SprayOnOffActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: SprayOnOffActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        self._raw_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._is_spray_currently_on = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._was_button_input_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._active_projectile_paths: List[Deque[str]] = [deque() for _ in range(self.num_envs)]
        self._last_spray_time: torch.Tensor = torch.full((self.num_envs,), -float('inf'), device=self.device, dtype=torch.float64)
        self._projectile_counter: List[int] = [0] * self.num_envs

        if not isinstance(self._asset, Articulation):
            omni.log.error(
                f"[{self.__class__.__name__}] Asset '{self.cfg.asset_name}' is not an Articulation instance. "
                f"Type is: {type(self._asset)}. Projectile spawning will likely fail."
            )

        self._ee_frame_asset: FrameTransformer | None = self._env.scene.sensors.get(self.cfg.ee_frame_asset_name)
        if not isinstance(self._ee_frame_asset, FrameTransformer):
            omni.log.warn(
                f"[{self.__class__.__name__}] FrameTransformer asset named '{self.cfg.ee_frame_asset_name}' "
                f"not found in env.scene.sensors or is not a FrameTransformer. Projectile positioning will fail."
            )
            if self._env.scene.sensors:
                omni.log.info(f"Available sensors in env.scene.sensors: {list(self._env.scene.sensors.keys())}")
            else:
                omni.log.info("env.scene.sensors is empty or not yet populated.")

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_action

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, action: torch.Tensor):
        self._raw_action[:] = action
        current_button_input_active = action < 0.0 
        self._processed_actions[:] = current_button_input_active.float()

    def get_action_space(self) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=torch.float32)

    def apply_actions(self):
        if self._ee_frame_asset is None or self._asset is None:
            omni.log.warn(f"[{self.__class__.__name__}] EE Frame asset or Robot asset is None, skipping apply_actions.")
            return

        current_time_sim = time.time() 
        current_button_input_flags = self._processed_actions.squeeze(-1) > 0.5

        target_idx = -1
        if hasattr(self._ee_frame_asset, 'cfg') and hasattr(self._ee_frame_asset.cfg, 'target_frames'):
            target_frame_cfg_list = self._ee_frame_asset.cfg.target_frames
            for i, frame_cfg_item in enumerate(target_frame_cfg_list):
                if frame_cfg_item.name == self.cfg.nozzle_tip_frame_name:
                    target_idx = i
                    break
        
        if target_idx == -1:
            configured_names_str = "None"
            if hasattr(self._ee_frame_asset, 'cfg') and hasattr(self._ee_frame_asset.cfg, 'target_frames'):
                configured_names = [cfg_item.name for cfg_item in self._ee_frame_asset.cfg.target_frames if hasattr(cfg_item, 'name')]
                configured_names_str = str(configured_names)
            omni.log.error(
                f"Target frame name '{self.cfg.nozzle_tip_frame_name}' not found in FrameTransformer Cfg "
                f"target_frames for SprayOnOffAction. Configured names: {configured_names_str}"
            )
            return

        ee_frame_data = self._ee_frame_asset.data
        if not (hasattr(ee_frame_data, 'target_pos_w') and hasattr(ee_frame_data, 'target_quat_w') and \
                ee_frame_data.target_pos_w.ndim == 3 and ee_frame_data.target_pos_w.shape[1] > target_idx and \
                ee_frame_data.target_quat_w.ndim == 3 and ee_frame_data.target_quat_w.shape[1] > target_idx):
            omni.log.error(
                f"FrameTransformerData for '{self.cfg.ee_frame_asset_name}' has missing or malformed "
                f"target_pos_w/target_quat_w or target_idx is out of bounds for SprayOnOffAction."
            )
            return
            
        nozzle_tip_orientations_w = ee_frame_data.target_quat_w[:, target_idx, :]
        nozzle_tip_positions_w = ee_frame_data.target_pos_w[:, target_idx, :]

        stage = omni.usd.get_context().get_stage()
        default_time_code = Usd.TimeCode.Default()

        robot_prim_paths_all_envs: List[str | None] = [None] * self.num_envs
        asset_name_in_scene = self.cfg.asset_name
        paths_resolved_for_all_envs = True

        for i in range(self.num_envs):
            current_env_path_resolved = False
            if self.num_envs == 1 and self.cfg.robot_prim_path_for_single_env:
                path_candidate = self.cfg.robot_prim_path_for_single_env
                if stage.GetPrimAtPath(path_candidate):
                    robot_prim_paths_all_envs[i] = path_candidate
                    omni.log.info(f"Env {i}: Using provided robot_prim_path_for_single_env: {path_candidate}")
                    current_env_path_resolved = True
                else:
                    omni.log.warn(f"Env {i}: Provided robot_prim_path_for_single_env '{path_candidate}' not found in stage.")
            
            if not current_env_path_resolved and hasattr(self._asset, 'cfg') and hasattr(self._asset.cfg, 'prim_path'):
                base_prim_path_expr = self._asset.cfg.prim_path
                env_namespace_prefix = ""
                if hasattr(self._env, 'env_prim_paths') and self._env.env_prim_paths and len(self._env.env_prim_paths) > i:
                    env_namespace_prefix = self._env.env_prim_paths[i]
                elif hasattr(self._env, 'env_ns') and self._env.env_ns and len(self._env.env_ns) > i:
                    env_namespace_prefix = self._env.env_ns[i]

                if "{ENV_REGEX_NS}" in base_prim_path_expr:
                    if env_namespace_prefix:
                        path_suffix = base_prim_path_expr.split("{ENV_REGEX_NS}")[-1]
                        constructed_path = env_namespace_prefix + path_suffix
                    else:
                        cleaned_expr = base_prim_path_expr.replace("{ENV_REGEX_NS}", "").lstrip("/")
                        constructed_path = f"/World/{cleaned_expr}"
                        omni.log.warn(f"Env {i}: Attempting default construction for '{base_prim_path_expr}' as '{constructed_path}' due to missing env_ns/env_prim_paths.")
                else:
                    constructed_path = base_prim_path_expr

                if stage.GetPrimAtPath(constructed_path):
                    robot_prim_paths_all_envs[i] = constructed_path
                    omni.log.info(f"Env {i}: Constructed robot prim path: {constructed_path}")
                    current_env_path_resolved = True
                else:
                    omni.log.warn(f"Env {i}: Constructed robot prim path '{constructed_path}' not found in stage.")

            if not current_env_path_resolved:
                paths_resolved_for_all_envs = False
                omni.log.error(
                    f"Env {i}: Failed to resolve a valid prim_path for robot asset '{asset_name_in_scene}'. "
                    f"Projectile parenting will fail for this environment."
                )

        if not paths_resolved_for_all_envs and self.num_envs > 0 :
             omni.log.error(f"One or more robot prim paths could not be resolved. Check warnings above.")

        for env_idx in range(self.num_envs):
            button_pressed_this_step = current_button_input_flags[env_idx].item()
            spray_state_toggled_this_step = False
            if button_pressed_this_step and not self._was_button_input_active[env_idx].item():
                self._is_spray_currently_on[env_idx] = not self._is_spray_currently_on[env_idx]
                spray_state_toggled_this_step = True
                omni.log.info(f"Env {env_idx}: Spray Toggled. New state: {'ON' if self._is_spray_currently_on[env_idx] else 'OFF'}")
                if not self._is_spray_currently_on[env_idx]:
                    self._last_spray_time[env_idx] = -float('inf') 

            if self._is_spray_currently_on[env_idx].item():
                if robot_prim_paths_all_envs[env_idx] is None:
                    if spray_state_toggled_this_step :
                        omni.log.warn(f"Env {env_idx}: Cannot spray because robot prim path is not resolved.")
                    continue

                can_spray_now = (current_time_sim - self._last_spray_time[env_idx].item()) >= self.cfg.spray_interval
                can_add_more_projectiles = len(self._active_projectile_paths[env_idx]) < self.cfg.max_projectiles

                if can_spray_now and can_add_more_projectiles:
                    self._last_spray_time[env_idx] = current_time_sim

                    robot_instance_root_path = robot_prim_paths_all_envs[env_idx]
                    parent_prim_for_projectile_str = robot_instance_root_path
                    if self.cfg.projectile_parent_link_rel_path and self.cfg.projectile_parent_link_rel_path.strip():
                        parent_prim_for_projectile_str = f"{robot_instance_root_path}/{self.cfg.projectile_parent_link_rel_path}"

                    if not stage.GetPrimAtPath(parent_prim_for_projectile_str):
                        omni.log.error(f"Env {env_idx}: Parent prim for projectile '{parent_prim_for_projectile_str}' NOT FOUND.")
                        continue

                    self._projectile_counter[env_idx] += 1
                    projectile_name_unique = f"{self.cfg.projectile_prim_name}_{env_idx}_{self._projectile_counter[env_idx]}_actor"
                    projectile_path_str = f"{parent_prim_for_projectile_str}/{projectile_name_unique}"
                    
                    if len(self._active_projectile_paths[env_idx]) >= self.cfg.max_projectiles:
                        oldest_projectile_path = self._active_projectile_paths[env_idx].popleft()
                        if stage.GetPrimAtPath(oldest_projectile_path):
                            omni.kit.commands.execute('DeletePrims', paths=[oldest_projectile_path])
                            omni.log.info(f"Env {env_idx}: Removed oldest projectile: {oldest_projectile_path}")
                    
                    if stage.GetPrimAtPath(projectile_path_str):
                         omni.kit.commands.execute('DeletePrims', paths=[projectile_path_str])

                    # 1. Define Prim and Set Kind
                    projectile_prim = stage.DefinePrim(projectile_path_str, self.cfg.projectile_type)
                    if not projectile_prim.IsValid():
                        omni.log.warn(f"Env {env_idx}: Failed to define projectile prim at {projectile_path_str}")
                        continue
                    model_api = Usd.ModelAPI(projectile_prim)
                    model_api.SetKind(Kind.Tokens.component) # Changed from .prop to .component

                    self._active_projectile_paths[env_idx].append(projectile_path_str)
                    
                    # 2. Set Scale (type-specific API or part of the transform matrix for generic prims)
                    if self.cfg.projectile_type == "Cube":
                        cube_api = UsdGeom.Cube.Get(stage, projectile_path_str)
                        if not cube_api: cube_api = UsdGeom.Cube.Define(stage, projectile_path_str)
                        cube_api.GetSizeAttr().Set(self.cfg.projectile_scale[0] * 2.0, time=default_time_code)
                    elif self.cfg.projectile_type == "Sphere":
                        sphere_api = UsdGeom.Sphere.Get(stage, projectile_path_str)
                        if not sphere_api: sphere_api = UsdGeom.Sphere.Define(stage, projectile_path_str)
                        sphere_api.GetRadiusAttr().Set(self.cfg.projectile_scale[0], time=default_time_code)
                    # For generic prims, scale will be part of the local_transform_matrix

                    # 3. Calculate Local Transform Matrix
                    nozzle_pos_w_tensor = nozzle_tip_positions_w[env_idx]
                    nozzle_quat_w_tensor = nozzle_tip_orientations_w[env_idx]
                    initial_offset_local_to_nozzle = Gf.Vec3f(self.cfg.projectile_scale[0] * 1.1, 0, 0)
                    nozzle_orientation_gf = Gf.Quatf(
                        nozzle_quat_w_tensor[0].item(), nozzle_quat_w_tensor[1].item(),
                        nozzle_quat_w_tensor[2].item(), nozzle_quat_w_tensor[3].item()
                    ).GetNormalized()
                    initial_offset_world = nozzle_orientation_gf.Transform(initial_offset_local_to_nozzle)
                    initial_projectile_pos_w_gf_base = Gf.Vec3f(
                        nozzle_pos_w_tensor[0].item(), nozzle_pos_w_tensor[1].item(), nozzle_pos_w_tensor[2].item()
                    )
                    initial_projectile_pos_w_gf = initial_projectile_pos_w_gf_base + initial_offset_world

                    parent_prim_obj = stage.GetPrimAtPath(parent_prim_for_projectile_str)
                    parent_world_transform_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(parent_prim_obj)
                    
                    world_rotation_matrix = Gf.Matrix4d().SetRotate(nozzle_orientation_gf)
                    world_translation_matrix = Gf.Matrix4d().SetTranslate(Gf.Vec3d(initial_projectile_pos_w_gf))
                    
                    world_transform_matrix = world_rotation_matrix * world_translation_matrix
                    if self.cfg.projectile_type != "Cube" and self.cfg.projectile_type != "Sphere":
                        world_scale_matrix = Gf.Matrix4d().SetScale(Gf.Vec3d(*self.cfg.projectile_scale))
                        world_transform_matrix = world_scale_matrix * world_transform_matrix 
                        
                    local_transform_matrix = world_transform_matrix * parent_world_transform_matrix.GetInverse()
                    
                    # 4. Set Local Transform using a single xformOp:transform attribute
                    xform_api = UsdGeom.Xformable(projectile_prim)
                    xform_api.ClearXformOpOrder() 
                    transform_op = xform_api.AddTransformOp(precision=UsdGeom.XformOp.PrecisionDouble) 
                    transform_op.Set(local_transform_matrix, time=default_time_code)
                    xform_api.SetXformOpOrder([transform_op])


                    # 5. Set Initial Velocities and Mass as USD attributes
                    shoot_direction_local_on_nozzle = Gf.Vec3f(1, 0, 0)
                    shoot_direction_world = nozzle_orientation_gf.Transform(shoot_direction_local_on_nozzle)
                    shoot_direction_world.Normalize()
                    initial_linear_velocity_world_val = shoot_direction_world * self.cfg.custom_projectile_initial_speed
                    
                    vel_attr = projectile_prim.CreateAttribute("physics:velocity", Sdf.ValueTypeNames.Vector3f)
                    vel_attr.Set(Gf.Vec3f(initial_linear_velocity_world_val), time=default_time_code)
                    ang_vel_attr = projectile_prim.CreateAttribute("physics:angularVelocity", Sdf.ValueTypeNames.Vector3f)
                    ang_vel_attr.Set(Gf.Vec3f(0, 0, 0), time=default_time_code)
                    
                    mass_attr_usd = projectile_prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float)
                    mass_attr_usd.Set(self.cfg.projectile_mass, time=default_time_code)
                    
                    # 6. Now Apply Physics APIs
                    rb_api = UsdPhysics.RigidBodyAPI.Apply(projectile_prim)
                    collision_api = UsdPhysics.CollisionAPI.Apply(projectile_prim)
                    
                    collision_enabled_attr = collision_api.GetCollisionEnabledAttr()
                    if not collision_enabled_attr.IsValid(): 
                        collision_enabled_attr = collision_api.CreateCollisionEnabledAttr()
                    collision_enabled_attr.Set(True, default_time_code)

                    omni.log.info(f"Env {env_idx}: Projectile SPAWNED ({len(self._active_projectile_paths[env_idx])}/{self.cfg.max_projectiles}) at {projectile_path_str}")

            elif spray_state_toggled_this_step and not self._is_spray_currently_on[env_idx].item():
                self._remove_all_projectiles(env_idx, stage)

            self._was_button_input_active[env_idx] = button_pressed_this_step

    def _remove_all_projectiles(self, env_idx: int, stage: Usd.Stage) -> None:
        if self._active_projectile_paths[env_idx]:
            paths_to_delete = list(self._active_projectile_paths[env_idx])
            omni.kit.commands.execute('DeletePrims', paths=paths_to_delete)
            omni.log.info(f"Env {env_idx}: Removed {len(paths_to_delete)} projectiles.")
            self._active_projectile_paths[env_idx].clear()
        self._projectile_counter[env_idx] = 0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        stage = omni.usd.get_context().get_stage()
        indices_to_reset = range(self.num_envs) if env_ids is None else env_ids

        for env_idx in indices_to_reset:
            self._remove_all_projectiles(env_idx, stage)
            self._is_spray_currently_on[env_idx] = False
            self._was_button_input_active[env_idx] = False
            self._last_spray_time[env_idx] = -float('inf')
        
        if env_ids is None:
            self._raw_action.zero_()
            self._processed_actions.zero_()
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
            else:
                env_ids_tensor = env_ids
            if env_ids_tensor.numel() > 0:
                 self._raw_action[env_ids_tensor] = 0.0
                 self._processed_actions[env_ids_tensor] = 0.0

__all__ = ["SprayOnOffActionCfg", "SprayOnOffAction", "JointPositionActionCfg", "BinaryJointPositionActionCfg"]
