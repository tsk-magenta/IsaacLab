# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to replay demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay demonstrations in Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
parser.add_argument("--task", type=str, default=None, help="Force to use the specified task.")
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="A list of episode indices to be replayed. Keep empty to replay all in the dataset file.",
)
parser.add_argument("--dataset_file", type=str, default="datasets/dataset.hdf5", help="Dataset file to be replayed.")
parser.add_argument(
    "--validate_states",
    action="store_true",
    default=False,
    help=(
        "Validate if the states, if available, match between loaded from datasets and replayed. Only valid if"
        " --num_envs is 1."
    ),
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--show_particles",
    action="store_true",
    default=False,
    help="Show particles when gripper is open during replay.",
)
parser.add_argument(
    "--force_particles",
    action="store_true",
    default=False,
    help="Force particles to show at regular intervals regardless of gripper state.",
)
parser.add_argument(
    "--debug_particles",
    action="store_true",
    default=False,
    help="Create debug particles in a spiral pattern to verify the particle system is working.",
)
parser.add_argument(
    "--show_episode_data",
    action="store_true",
    default=False,
    help="Show detailed gripper_open and eef_pos values for every step in replayed episodes.",
)
parser.add_argument(
    "--live_logging",
    action="store_true",
    default=False,
    help="Show gripper_open and eef_pos values for each step during replay.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# args_cli.headless = True

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import time
import random
import numpy as np
import h5py

from isaaclab.devices import Se3Keyboard
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

is_paused = False
particles_enabled = True


def get_current_gripper_eef_state(episode_data, step_index):
    """Get the current gripper state and end effector position from the episode data.
    
    Args:
        episode_data: The episode data loaded from HDF5
        step_index: Current step index in the episode
        
    Returns:
        gripper_is_open: Boolean indicating if gripper is open
        eef_position: Numpy array with end effector position
    """
    gripper_is_open = False
    eef_position = None
    
    try:
        # First attempt: Direct access through the nested dictionary structure
        # Based on HDF5 structure: data/demo_0/obs/gripper_open and data/demo_0/obs/eef_pos
        if 'obs' in episode_data.data:
            obs_data = episode_data.data['obs']
            
            # Get gripper state
            if 'gripper_open' in obs_data:
                if step_index < len(obs_data['gripper_open']):
                    gripper_value = obs_data['gripper_open'][step_index]
                    if isinstance(gripper_value, torch.Tensor):
                        gripper_is_open = bool(gripper_value.item())
                    else:
                        gripper_is_open = bool(gripper_value)
            
            # Get EEF position
            if 'eef_pos' in obs_data:
                if step_index < len(obs_data['eef_pos']):
                    eef_value = obs_data['eef_pos'][step_index]
                    if isinstance(eef_value, torch.Tensor):
                        eef_position = eef_value.cpu().numpy()
                    else:
                        eef_position = np.array(eef_value)
        
        # Second attempt: Access through dynamic attributes
        # Sometimes the data is loaded as attributes rather than dict keys
        if (gripper_is_open is False or eef_position is None) and hasattr(episode_data, 'obs'):
            obs = episode_data.obs
            
            # Check if 'obs' is a dict or has attributes
            if isinstance(obs, dict):
                # Dict-style access
                if 'gripper_open' in obs and step_index < len(obs['gripper_open']):
                    gripper_value = obs['gripper_open'][step_index]
                    if isinstance(gripper_value, torch.Tensor):
                        gripper_is_open = bool(gripper_value.item())
                    else:
                        gripper_is_open = bool(gripper_value)
                
                if 'eef_pos' in obs and step_index < len(obs['eef_pos']):
                    eef_value = obs['eef_pos'][step_index]
                    if isinstance(eef_value, torch.Tensor):
                        eef_position = eef_value.cpu().numpy()
                    else:
                        eef_position = np.array(eef_value)
            
            # Attribute-style access
            elif hasattr(obs, 'gripper_open') and hasattr(obs, 'eef_pos'):
                try:
                    if step_index < len(obs.gripper_open):
                        gripper_value = obs.gripper_open[step_index]
                        if isinstance(gripper_value, torch.Tensor):
                            gripper_is_open = bool(gripper_value.item())
                        else:
                            gripper_is_open = bool(gripper_value)
                except Exception as e:
                    print(f"Error accessing gripper_open as attribute: {e}")
                
                try:
                    if step_index < len(obs.eef_pos):
                        eef_value = obs.eef_pos[step_index]
                        if isinstance(eef_value, torch.Tensor):
                            eef_position = eef_value.cpu().numpy()
                        else:
                            eef_position = np.array(eef_value)
                except Exception as e:
                    print(f"Error accessing eef_pos as attribute: {e}")
        
        # Third attempt: Legacy method with observations attribute
        if (gripper_is_open is False or eef_position is None) and hasattr(episode_data, 'observations'):
            observations = episode_data.observations
            
            # Ensure index is valid
            if observations and step_index < len(observations):
                obs_data = observations[step_index]
                
                # Extract gripper state if available
                if isinstance(obs_data, dict) and "policy" in obs_data and "gripper_open" in obs_data["policy"]:
                    gripper_open_tensor = obs_data["policy"]["gripper_open"]
                    if isinstance(gripper_open_tensor, torch.Tensor):
                        try:
                            gripper_is_open = bool(gripper_open_tensor.item())
                        except Exception:
                            # If item() fails, try using the first element
                            try:
                                gripper_is_open = bool(gripper_open_tensor[0].item())
                            except Exception:
                                # If all else fails, convert to bool directly
                                gripper_is_open = bool(gripper_open_tensor)
                    else:
                        gripper_is_open = bool(gripper_open_tensor)
                
                # Extract end effector position if available
                if isinstance(obs_data, dict) and "policy" in obs_data and "eef_pos" in obs_data["policy"]:
                    eef_pos_tensor = obs_data["policy"]["eef_pos"]
                    if isinstance(eef_pos_tensor, torch.Tensor):
                        try:
                            eef_position = eef_pos_tensor.cpu().numpy()
                        except Exception:
                            # If converting to numpy fails, try converting to list
                            try:
                                eef_position = np.array([eef_pos_tensor[0].item(), 
                                                       eef_pos_tensor[1].item(), 
                                                       eef_pos_tensor[2].item()])
                            except Exception:
                                # Direct conversion
                                eef_position = np.array(eef_pos_tensor)
                    else:
                        eef_position = np.array(eef_pos_tensor)
                        
    except Exception as e:
        print(f"Error in get_current_gripper_eef_state: {e}")
        import traceback
        traceback.print_exc()
    
    return gripper_is_open, eef_position


# Custom Particle System that shows particles when gripper is open
class ParticleSystem:
    def __init__(self, env):
        self.particles = []
        self.stopped_count = 0
        self.env = env
        
        # Import omni modules after simulator is initialized
        import omni.usd
        from pxr import Usd, UsdGeom, Gf, Sdf, UsdShade
        self.omni_usd = omni.usd
        self.Usd = Usd
        self.UsdGeom = UsdGeom
        self.Gf = Gf
        self.Sdf = Sdf
        self.UsdShade = UsdShade
        
        # Create a test particle to verify the system is working
        print("Creating test particle to verify system...")
        self.create_test_particle()
        
    def create_test_particle(self):
        """Create a test particle at a fixed location for debugging"""
        try:
            stage = self.omni_usd.get_context().get_stage()
            if not stage:
                print("Error: No stage available")
                return
                
            test_path = "/World/TestParticle"
            
            # Create a larger red particle at a fixed position
            test_particle = self.UsdGeom.Cube.Define(stage, test_path)
            test_particle.GetSizeAttr().Set(0.02)  # Larger cube for visibility
            
            # Position it where it will be visible
            test_pos = self.Gf.Vec3d(0.5, 0.0, 0.3)
            self.UsdGeom.XformCommonAPI(test_particle).SetTranslate(test_pos)
            
            # Create red material 
            material_path = self.Sdf.Path("/World/Looks/Red")
            if not stage.GetPrimAtPath(str(material_path)).IsValid():
                material = self.UsdShade.Material.Define(stage, material_path)
                shader = self.UsdShade.Shader.Define(stage, material_path.AppendPath("Shader"))
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", self.Sdf.ValueTypeNames.Color3f).Set(self.Gf.Vec3f(1.0, 0.0, 0.0)) # Red
                shader.CreateInput("roughness", self.Sdf.ValueTypeNames.Float).Set(0.4)
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            else:
                material = self.UsdShade.Material(stage.GetPrimAtPath(material_path))
            
            prim = test_particle.GetPrim()
            self.UsdShade.MaterialBindingAPI(prim).Bind(material)
            
            print("✅ Test particle created successfully at position (0.5, 0.0, 0.3)")
        except Exception as e:
            print(f"Error creating test particle: {e}")
            import traceback
            traceback.print_exc()
        
    def create_particles(self, position, count=3):
        """Create particles at the given position"""
        if position is None or not isinstance(position, np.ndarray):
            return
            
        stage = self.omni_usd.get_context().get_stage()
        if not stage:
            return
            
        base_path = "/World/ReplayParticle"
        
        for _ in range(count):
            # Generate a unique name
            i = 0
            while stage.GetPrimAtPath(f"{base_path}{i}").IsValid():
                i += 1
                
            particle_path = f"{base_path}{i}"
            
            # Add random offset
            rand_offset = self.Gf.Vec3d(
                random.uniform(-0.03, 0.03),
                random.uniform(-0.03, 0.03),
                random.uniform(-0.03, 0.03),
            )
            
            # Create position
            spawn_pos = self.Gf.Vec3d(
                float(position[0]) + rand_offset[0],
                float(position[1]) + rand_offset[1], 
                float(position[2]) + rand_offset[2]
            )
            
            # Create visual element (cube)
            particle = self.UsdGeom.Cube.Define(stage, particle_path)
            particle.GetSizeAttr().Set(0.008)  # Slightly larger cube for better visibility
            self.UsdGeom.XformCommonAPI(particle).SetTranslate(spawn_pos)
            
            # Create magenta material for the standard particles
            material_path = self.Sdf.Path("/World/Looks/Magenta")
            if not stage.GetPrimAtPath(str(material_path)).IsValid():
                material = self.UsdShade.Material.Define(stage, material_path)
                shader = self.UsdShade.Shader.Define(stage, material_path.AppendPath("Shader"))
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", self.Sdf.ValueTypeNames.Color3f).Set(self.Gf.Vec3f(1.0, 0.0, 1.0)) # Magenta
                shader.CreateInput("roughness", self.Sdf.ValueTypeNames.Float).Set(0.4)
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            else:
                material = self.UsdShade.Material(stage.GetPrimAtPath(material_path))
            
            prim = particle.GetPrim()
            self.UsdShade.MaterialBindingAPI(prim).Bind(material)
            
            # Add custom data
            particle_data = {
                'prim': particle,
                'direction_x': random.uniform(1.0, 1.0),
                'direction_y': random.uniform(-0.25, 0.25),
                'direction_z': random.uniform(-0.25, 0.25),
                'stopped': False
            }
            
            self.particles.append(particle_data)
    
    def update_particles(self):
        """Move all particles according to their direction"""
        if not self.particles:
            return
            
        stage = self.omni_usd.get_context().get_stage()
        if not stage:
            return
        
        to_remove = []
        
        for idx, particle_data in enumerate(self.particles):
            particle = particle_data['prim']
            prim = particle.GetPrim()
            
            if not prim or not prim.IsValid():
                to_remove.append(idx)
                continue
            
            # Get current position
            translate_attr = prim.GetAttribute("xformOp:translate")
            if not translate_attr:
                to_remove.append(idx)
                continue
                
            current_pos = translate_attr.Get()
            if current_pos is None:
                to_remove.append(idx)
                continue
            
            # Remove if X position is too far
            if current_pos[0] > 1.0:
                stage.RemovePrim(prim.GetPath())
                to_remove.append(idx)
                continue
            
            # Check if particle should stop
            if current_pos[0] >= 0.6 and not particle_data['stopped']:
                if -0.38 <= current_pos[1] <= 0.38 and 0.0 <= current_pos[2] <= 0.93:
                    # Particle stops here
                    particle_data['stopped'] = True
                    self.stopped_count += 1
                    
                    # Log milestone
                    if self.stopped_count % 50 == 0:
                        print(f"✅ Magenta particles collected: {self.stopped_count}")
                    
                    # Fix position and scale
                    fixed_pos = self.Gf.Vec3d(0.65, current_pos[1], current_pos[2])
                    self.UsdGeom.XformCommonAPI(particle).SetTranslate(fixed_pos)
                    self.UsdGeom.XformCommonAPI(particle).SetScale(self.Gf.Vec3f(5.0, 5.0, 5.0))
                    continue
            
            # Skip movement for stopped particles
            if particle_data['stopped']:
                continue
            
            # Move particle
            move_factor = 0.05
            new_pos = self.Gf.Vec3d(
                current_pos[0] + particle_data['direction_x'] * move_factor,
                current_pos[1] + particle_data['direction_y'] * move_factor,
                current_pos[2] + particle_data['direction_z'] * move_factor,
            )
            
            self.UsdGeom.XformCommonAPI(particle).SetTranslate(new_pos)
        
        # Remove invalid particles from list (in reverse order)
        for idx in sorted(to_remove, reverse=True):
            if idx < len(self.particles):
                self.particles.pop(idx)
    
    def clear_all_particles(self):
        """Remove all particles"""
        stage = self.omni_usd.get_context().get_stage()
        if not stage:
            return
            
        # Remove all particles from USD stage
        for particle_data in self.particles:
            particle = particle_data['prim']
            prim = particle.GetPrim()
            if prim and prim.IsValid():
                stage.RemovePrim(prim.GetPath())
        
        # Clear particle list
        self.particles = []
        print(f"Cleared all magenta particles. Total stopped count: {self.stopped_count}")

    def create_red_particles(self, position, count=5):
        """Create larger red particles at the given position for better visibility"""
        if position is None or not isinstance(position, np.ndarray):
            return
            
        stage = self.omni_usd.get_context().get_stage()
        if not stage:
            return
            
        base_path = "/World/ReplayRedParticle"
        
        for _ in range(count):
            # Generate a unique name
            i = 0
            while stage.GetPrimAtPath(f"{base_path}{i}").IsValid():
                i += 1
                
            particle_path = f"{base_path}{i}"
            
            # Add random offset
            rand_offset = self.Gf.Vec3d(
                random.uniform(-0.02, 0.02),
                random.uniform(-0.02, 0.02),
                random.uniform(-0.02, 0.02),
            )
            
            # Create position
            spawn_pos = self.Gf.Vec3d(
                float(position[0]) + rand_offset[0],
                float(position[1]) + rand_offset[1], 
                float(position[2]) + rand_offset[2]
            )
            
            # Create visual element (cube)
            particle = self.UsdGeom.Cube.Define(stage, particle_path)
            particle.GetSizeAttr().Set(0.01)  # Larger cube for better visibility
            self.UsdGeom.XformCommonAPI(particle).SetTranslate(spawn_pos)
            
            # Create red material
            material_path = self.Sdf.Path("/World/Looks/Red")
            if not stage.GetPrimAtPath(str(material_path)).IsValid():
                material = self.UsdShade.Material.Define(stage, material_path)
                shader = self.UsdShade.Shader.Define(stage, material_path.AppendPath("Shader"))
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", self.Sdf.ValueTypeNames.Color3f).Set(self.Gf.Vec3f(1.0, 0.0, 0.0)) # Red
                shader.CreateInput("roughness", self.Sdf.ValueTypeNames.Float).Set(0.4)
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            else:
                material = self.UsdShade.Material(stage.GetPrimAtPath(material_path))
            
            prim = particle.GetPrim()
            self.UsdShade.MaterialBindingAPI(prim).Bind(material)
            
            # Add custom data
            particle_data = {
                'prim': particle,
                'direction_x': random.uniform(1.0, 1.0),
                'direction_y': random.uniform(-0.25, 0.25),
                'direction_z': random.uniform(-0.25, 0.25),
                'stopped': False
            }
            
            self.particles.append(particle_data)


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def toggle_particles_cb():
    global particles_enabled
    particles_enabled = not particles_enabled
    print(f"Particles {'enabled' if particles_enabled else 'disabled'}")


def force_create_particles_cb():
    """Create particles manually at the current end effector position."""
    try:
        # Get particle system reference from globals
        if 'particle_system' not in globals() or globals()['particle_system'] is None:
            print("Particle system not initialized")
            return
            
        particle_system = globals()['particle_system']
        env = globals().get('env')
        
        # Try to get the EEF position
        eef_pos = None
        
        # Method 1: Try from HDF5 file for first environment
        if 'env_episode_data_map' in globals() and globals()['env_episode_data_map'] and len(globals()['env_episode_data_map']) > 0:
            env_episode_data_map = globals()['env_episode_data_map']
            current_index = env_episode_data_map[0].next_action_index - 1
            _, eef_pos_from_hdf5 = get_current_gripper_eef_state(
                env_episode_data_map[0], 
                current_index
            )
            if eef_pos_from_hdf5 is not None:
                eef_pos = eef_pos_from_hdf5
                print(f"Got EEF position from HDF5 at step {current_index}: {eef_pos}")
        
        # Method 2: Try from environment's robot
        if eef_pos is None and hasattr(env.unwrapped, 'robot') and hasattr(env.unwrapped.robot, 'eef_pos'):
            eef_pos = env.unwrapped.robot.eef_pos[0].cpu().numpy()
            print(f"Got EEF position from robot: {eef_pos}")
        
        # Method 3: Try from USD stage
        if eef_pos is None and particle_system:
            try:
                stage = particle_system.omni_usd.get_context().get_stage()
                if stage:
                    # Try common end effector paths
                    ee_paths = [
                        "/World/envs/env_0/Franka/panda_rightfinger",
                        "/World/envs/env_0/Franka/panda_hand",
                        "/World/envs/env_0/Franka/panda_leftfinger",
                        "/World/envs/env_0/robot/panda_rightfinger",
                        "/World/envs/env_0/robot/panda_hand"
                    ]
                    
                    for path in ee_paths:
                        ee_prim = stage.GetPrimAtPath(path)
                        if ee_prim and ee_prim.IsValid():
                            # Get world transform
                            xform_cache = particle_system.UsdGeom.XformCache(particle_system.Usd.TimeCode.Default())
                            world_transform = xform_cache.GetLocalToWorldTransform(ee_prim)
                            ee_translation = world_transform.ExtractTranslation()
                            
                            eef_pos = np.array([
                                float(ee_translation[0]),
                                float(ee_translation[1]),
                                float(ee_translation[2])
                            ])
                            
                            print(f"Got EEF position from USD stage prim {path}: {eef_pos}")
                            break
            except Exception as e:
                print(f"Error accessing USD stage: {e}")
        
        # Fallback to a default position if we still don't have EEF position
        if eef_pos is None:
            print("Using fallback position for manual particles")
            eef_pos = np.array([0.5, 0.0, 0.5])  # Default position in front of the robot
        
        # Create a burst of particles (more than normal operation)
        # Use both magenta and red particles for better visibility
        particle_system.create_particles(eef_pos, count=20)  # Magenta particles
        particle_system.create_red_particles(eef_pos, count=10)  # Red particles
        
        # Create additional bursts slightly offset for volume
        offset1 = np.array([0.01, 0.01, 0.01])
        offset2 = np.array([-0.01, -0.01, 0.01])
        particle_system.create_particles(eef_pos + offset1, count=10)
        particle_system.create_particles(eef_pos + offset2, count=10)
        
        # Set last particle time to prevent immediate creation of more particles
        globals()['last_particle_time'] = time.time()
        print(f"Manually created particle burst at position {eef_pos}")
    except Exception as e:
        print(f"Error creating manual particles: {e}")
        import traceback
        traceback.print_exc()


def print_episode_summary_cb():
    """Print episode data summary for the current episode on demand."""
    try:
        if 'env_episode_data_map' in globals() and globals()['env_episode_data_map']:
            env_episode_data_map = globals()['env_episode_data_map']
            
            # Print summary for the first environment
            if 0 in env_episode_data_map and env_episode_data_map[0]:
                print("\n\nMANUALLY REQUESTED EPISODE DATA SUMMARY")
                episode_data = env_episode_data_map[0]
                
                # Log current episode and step info
                if hasattr(episode_data, 'next_action_index'):
                    current_step = episode_data.next_action_index - 1
                    print(f"Current step: {current_step}")
                
                # Check data structure to help with debugging
                print(f"Episode data attributes: {dir(episode_data)}")
                
                # Check for data in episode_data.data['obs']
                if hasattr(episode_data, 'data') and 'obs' in episode_data.data:
                    print(f"Found 'obs' in episode_data.data, keys: {list(episode_data.data['obs'].keys())}")
                    
                # Check for direct obs attribute
                if hasattr(episode_data, 'obs'):
                    print(f"Found 'obs' attribute, type: {type(episode_data.obs)}")
                    if hasattr(episode_data.obs, 'gripper_open'):
                        print(f"Found 'gripper_open' attribute in obs with length: {len(episode_data.obs.gripper_open)}")
                    if hasattr(episode_data.obs, 'eef_pos'):
                        print(f"Found 'eef_pos' attribute in obs with length: {len(episode_data.obs.eef_pos)}")
                
                # Check for observations attribute (legacy format)
                if hasattr(episode_data, 'observations'):
                    print(f"Found 'observations' attribute with length: {len(episode_data.observations) if episode_data.observations else 0}")
                    
                # Print summary of episode data
                print_episode_data_summary(episode_data)
            else:
                print("No episode data available for environment 0")
        else:
            print("No episode data map available")
    except Exception as e:
        print(f"Error printing episode summary: {e}")
        import traceback
        traceback.print_exc()


def compare_states(state_from_dataset, runtime_state, runtime_env_index) -> tuple[bool, str]:
    """Compare states from dataset and runtime.

    Args:
        state_from_dataset: State from dataset.
        runtime_state: State from runtime.
        runtime_env_index: Index of the environment in the runtime states to be compared.

    Returns:
        bool: True if states match, False otherwise.
        str: Log message if states don't match.
    """
    states_matched = True
    output_log = ""
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name][runtime_env_index]
                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name]
                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(f"State shape of {state_name} for asset {asset_name} don't match")
                for i in range(len(dataset_asset_state)):
                    if abs(dataset_asset_state[i] - runtime_asset_state[i]) > 0.01:
                        states_matched = False
                        output_log += f'\tState ["{asset_type}"]["{asset_name}"]["{state_name}"][{i}] don\'t match\r\n'
                        output_log += f"\t  Dataset:\t{dataset_asset_state[i]}\r\n"
                        output_log += f"\t  Runtime: \t{runtime_asset_state[i]}\r\n"
    return states_matched, output_log


def print_hdf5_structure(episode_data):
    """Print the structure of the HDF5 file for debugging purposes.
    
    Args:
        episode_data: EpisodeData object loaded from HDF5 file
    """
    print("\n\n===== HDF5 FILE STRUCTURE =====")
    
    # Print available attributes
    print(f"Episode data attributes: {dir(episode_data)}")
    
    # Check for data attribute (dictionary-based structure)
    if hasattr(episode_data, 'data'):
        print("\nFound 'data' attribute (dictionary structure)")
        print(f"Keys in data: {list(episode_data.data.keys())}")
        
        # Check for obs in data
        if 'obs' in episode_data.data:
            print(f"\nFound 'obs' in data dictionary")
            print(f"Keys in obs: {list(episode_data.data['obs'].keys())}")
            
            # Print key details for each observation
            for key in episode_data.data['obs'].keys():
                try:
                    data = episode_data.data['obs'][key]
                    shape = data.shape if hasattr(data, 'shape') else "unknown"
                    dtype = data.dtype if hasattr(data, 'dtype') else type(data)
                    print(f"  - {key}: shape={shape}, dtype={dtype}")
                except Exception as e:
                    print(f"  - {key}: Error accessing data: {e}")
        
        # Check for actions in data
        if 'actions' in episode_data.data:
            try:
                actions = episode_data.data['actions']
                print(f"\nFound 'actions' in data dictionary")
                print(f"Actions shape: {actions.shape}, dtype: {actions.dtype}")
            except Exception as e:
                print(f"Error accessing actions: {e}")
    
    # Check for direct observation attributes (attribute-based structure)
    elif hasattr(episode_data, 'obs'):
        print("\nFound direct 'obs' attribute")
        print(f"Obs attributes: {dir(episode_data.obs)}")
        
        # Try to print details for common observation components
        for attr in ['eef_pos', 'gripper_open', 'joint_pos', 'actions']:
            if hasattr(episode_data.obs, attr):
                try:
                    data = getattr(episode_data.obs, attr)
                    shape = data.shape if hasattr(data, 'shape') else "unknown"
                    dtype = data.dtype if hasattr(data, 'dtype') else type(data)
                    print(f"  - {attr}: shape={shape}, dtype={dtype}")
                except Exception as e:
                    print(f"  - {attr}: Error accessing data: {e}")
    
    # Check for observations attribute (legacy format)
    elif hasattr(episode_data, 'observations'):
        print("\nFound legacy 'observations' attribute")
        try:
            observations = episode_data.observations
            print(f"Observations type: {type(observations)}, length: {len(observations) if observations else 0}")
        except Exception as e:
            print(f"Error accessing observations: {e}")
    
    print("\n===== END OF HDF5 STRUCTURE =====\n")
    return True


def _print_episode_data_summary_legacy(episode_data):
    """Legacy method to print episode data summary for older HDF5 structures."""
    try:
        # Check for either 'observations' or 'obs' attribute
        observations = None
        if hasattr(episode_data, 'observations') and episode_data.observations:
            observations = episode_data.observations
            print("Using 'observations' attribute from episode data")
        elif hasattr(episode_data, 'obs') and isinstance(episode_data.obs, list):
            observations = episode_data.obs
            print("Using 'obs' attribute from episode data")
        
        if observations is None:
            print("ERROR: Episode data has neither usable 'observations' nor 'obs' attribute")
            print(f"Available attributes: {dir(episode_data)}")
            
            # Special handling for spray data if available
            if hasattr(episode_data, 'data') and 'obs' in episode_data.data:
                if 'spray' in episode_data.data['obs']:
                    spray_data = episode_data.data['obs']['spray']
                    print(f"\nFound 'spray' data in episode_data.data['obs']")
                    print(f"Spray data shape: {spray_data.shape}")
                    print(f"Spray data summary: min={spray_data.min().item()}, max={spray_data.max().item()}")
                    
                    # Print spray state changes
                    prev_state = None
                    for i in range(len(spray_data)):
                        curr_state = spray_data[i, 0].item()
                        if prev_state is None or curr_state != prev_state:
                            print(f"Step {i}: Spray {'ON' if curr_state else 'OFF'}")
                            prev_state = curr_state
            
            return False
            
        # Rest of the function remains the same
        num_steps = len(observations)
        print(f"\n===== Episode Data Summary (Legacy Method, {num_steps} steps) =====")
        print(f"{'Step':<6} {'Gripper Open':<15} {'EEF Position (X, Y, Z)':<40}")
        print("-" * 70)
        
        # Sample the first, middle, and last few observations to avoid excessive printing
        MAX_PRINT_STEPS = 50
        
        if num_steps <= MAX_PRINT_STEPS:
            steps_to_print = range(num_steps)
        else:
            # Print first 20, middle 10, and last 20 steps
            first_steps = list(range(20))
            middle_start = (num_steps // 2) - 5
            middle_steps = list(range(middle_start, middle_start + 10))
            last_steps = list(range(num_steps - 20, num_steps))
            steps_to_print = first_steps + middle_steps + last_steps
            
        for i in steps_to_print:
            gripper_is_open = False
            eef_pos = None
            
            # Get the observation for this step
            try:
                obs = observations[i]
                
                # Check if observation has the expected structure
                if isinstance(obs, dict) and "policy" in obs:
                    policy_obs = obs["policy"]
                    
                    # Extract gripper_open
                    if "gripper_open" in policy_obs:
                        gripper_open_val = policy_obs["gripper_open"]
                        if isinstance(gripper_open_val, torch.Tensor):
                            gripper_is_open = bool(gripper_open_val.item())
                        else:
                            gripper_is_open = bool(gripper_open_val)
                    else:
                        print(f"  [Step {i}: No 'gripper_open' in policy. Available keys: {list(policy_obs.keys())}]")
                        
                    # Extract eef_pos
                    if "eef_pos" in policy_obs:
                        eef_pos_val = policy_obs["eef_pos"]
                        if isinstance(eef_pos_val, torch.Tensor):
                            eef_pos = eef_pos_val.cpu().numpy()
                        else:
                            eef_pos = np.array(eef_pos_val)
                    else:
                        print(f"  [Step {i}: No 'eef_pos' in policy. Available keys: {list(policy_obs.keys())}]")
                else:
                    print(f"  [Step {i}: Observation missing 'policy'. Available keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}]")
            except Exception as e:
                print(f"Error processing step {i}: {e}")
                continue
            
            # Format the output
            gripper_str = f"{gripper_is_open}"
            if eef_pos is not None:
                try:
                    eef_pos_str = f"[{eef_pos[0]:.4f}, {eef_pos[1]:.4f}, {eef_pos[2]:.4f}]"
                except Exception as e:
                    eef_pos_str = f"Error formatting: {e}, Raw: {eef_pos}"
            else:
                eef_pos_str = "None"
            
            # Add visual indicator for transitions in gripper state
            if i > 0 and i-1 in steps_to_print:
                prev_obs = observations[i-1]
                prev_gripper_open = False
                if isinstance(prev_obs, dict) and "policy" in prev_obs and "gripper_open" in prev_obs["policy"]:
                    prev_gripper_open = bool(prev_obs["policy"]["gripper_open"].item())
                
                if prev_gripper_open != gripper_is_open:
                    indicator = ">>> GRIPPER STATE CHANGED <<<"
                else:
                    indicator = ""
            else:
                indicator = ""
            
            print(f"{i:<6} {gripper_str:<15} {eef_pos_str:<40} {indicator}")
            
            # If we're not printing consecutive steps, add a separator
            if i < num_steps - 1 and i + 1 in steps_to_print and i + 1 != steps_to_print[steps_to_print.index(i) + 1]:
                print("   ...   " + "-" * 62)
        
        print("=" * 70)
        return True
    except Exception as e:
        print(f"Error in _print_episode_data_summary_legacy: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_episode_data_summary(episode_data):
    """Print a summary of gripper_open and eef_pos values for an entire episode.
    
    Args:
        episode_data: EpisodeData object loaded from HDF5 file
    """
    print("\n\n*** STARTING EPISODE DATA SUMMARY ***")
    
    # First print the HDF5 structure to help debugging
    print_hdf5_structure(episode_data)
    
    # Try to access the observations using the dictionary-based approach first
    try:
        # Check for data in the nested dictionary format (preferred method)
        if hasattr(episode_data, 'data') and 'obs' in episode_data.data:
            print("Found 'obs' in episode_data.data")
            obs_data = episode_data.data['obs']
            
            # Check if we have both required fields
            has_gripper_open = 'gripper_open' in obs_data
            has_eef_pos = 'eef_pos' in obs_data
            has_spray = 'spray' in obs_data
            
            if has_gripper_open:
                print(f"Found 'gripper_open' with length: {len(obs_data['gripper_open'])}")
            if has_eef_pos:
                print(f"Found 'eef_pos' with length: {len(obs_data['eef_pos'])}")
            if has_spray:
                print(f"Found 'spray' with length: {len(obs_data['spray'])}")
                
            # If we don't have the required fields, fall back to the legacy method
            if not (has_gripper_open and has_eef_pos):
                print("Missing required observation fields, falling back to legacy method")
                return _print_episode_data_summary_legacy(episode_data)
                
            # Access and process the data
            num_steps = len(obs_data['eef_pos'])
            print(f"\n===== Episode Data Summary ({num_steps} steps) =====")
            print(f"{'Step':<6} {'Gripper Open':<15} {'Spray':<10} {'EEF Position (X, Y, Z)':<40}")
            print("-" * 70)
            
            # Sample the steps to print
            MAX_PRINT_STEPS = 50
            if num_steps <= MAX_PRINT_STEPS:
                steps_to_print = range(num_steps)
            else:
                # Print first 20, middle 10, and last 20 steps
                first_steps = list(range(20))
                middle_start = (num_steps // 2) - 5
                middle_steps = list(range(middle_start, middle_start + 10))
                last_steps = list(range(num_steps - 20, num_steps))
                steps_to_print = sorted(list(set(first_steps + middle_steps + last_steps)))
            
            # Print the selected steps
            for i in steps_to_print:
                try:
                    eef_pos = obs_data['eef_pos'][i][0]
                    gripper_open = "Open" if obs_data['gripper_open'][i][0] else "Closed"
                    spray_value = "ON" if (has_spray and obs_data['spray'][i][0]) else "OFF"
                    print(f"{i:<6} {gripper_open:<15} {spray_value:<10} {eef_pos}")
                except Exception as e:
                    print(f"Error processing step {i}: {e}")
            
            print("=" * 70)
            return True
        
        # If dictionary-based access didn't work, try attribute-based access
        if hasattr(episode_data, 'obs'):
            print("Found 'obs' attribute in episode_data")
            obs = episode_data.obs
            
            # Check if we can access gripper_open and eef_pos as attributes
            has_gripper_open = hasattr(obs, 'gripper_open')
            has_eef_pos = hasattr(obs, 'eef_pos')
            
            if has_gripper_open:
                print(f"Found 'gripper_open' attribute with length: {len(obs.gripper_open)}")
            if has_eef_pos:
                print(f"Found 'eef_pos' attribute with length: {len(obs.eef_pos)}")
            
            # Print summary if we have both attributes
            if has_gripper_open and has_eef_pos:
                num_steps = min(len(obs.gripper_open), len(obs.eef_pos))
                print(f"\n===== Episode Data Summary (Attribute Format, {num_steps} steps) =====")
                print(f"{'Step':<6} {'Gripper Open':<15} {'EEF Position (X, Y, Z)':<40}")
                print("-" * 70)
                
                # Sample steps to avoid excessive output
                MAX_PRINT_STEPS = 50
                
                if num_steps <= MAX_PRINT_STEPS:
                    steps_to_print = range(num_steps)
                else:
                    # Print first 20, middle 10, and last 20 steps
                    first_steps = list(range(20))
                    middle_start = (num_steps // 2) - 5
                    middle_steps = list(range(middle_start, middle_start + 10))
                    last_steps = list(range(num_steps - 20, num_steps))
                    steps_to_print = sorted(set(first_steps + middle_steps + last_steps))
                
                prev_gripper_open = None
                for i in steps_to_print:
                    try:
                        # Get values from attributes
                        gripper_val = obs.gripper_open[i]
                        eef_val = obs.eef_pos[i]
                        
                        # Convert to appropriate types
                        if isinstance(gripper_val, torch.Tensor):
                            gripper_is_open = bool(gripper_val.item())
                        else:
                            gripper_is_open = bool(gripper_val)
                            
                        if isinstance(eef_val, torch.Tensor):
                            eef_pos = eef_val.cpu().numpy()
                        else:
                            eef_pos = np.array(eef_val)
                        
                        # Format output strings
                        gripper_str = f"{gripper_is_open}"
                        eef_pos_str = f"[{eef_pos[0]:.4f}, {eef_pos[1]:.4f}, {eef_pos[2]:.4f}]" if eef_pos is not None else "None"
                        
                        # Add indicator for gripper state changes
                        indicator = ""
                        if prev_gripper_open is not None and prev_gripper_open != gripper_is_open:
                            indicator = ">>> GRIPPER STATE CHANGED <<<"
                        
                        print(f"{i:<6} {gripper_str:<15} {eef_pos_str:<40} {indicator}")
                        
                        # Save current gripper state for next iteration
                        prev_gripper_open = gripper_is_open
                        
                        # Add separator for non-consecutive steps
                        if i < num_steps - 1 and i + 1 in steps_to_print and i + 1 != steps_to_print[steps_to_print.index(i) + 1]:
                            print("   ...   " + "-" * 62)
                    except Exception as e:
                        print(f"Error processing step {i}: {e}")
                
                print("=" * 70)
                return True
    
        # If we get here, try the legacy method for backward compatibility
        print("Attempting legacy method for episode data summary...")
        return _print_episode_data_summary_legacy(episode_data)
        
    except Exception as e:
        print(f"Error in print_episode_data_summary: {e}")
        import traceback
        traceback.print_exc()
        
        # Try legacy method as a fallback
        print("Falling back to legacy method...")
        return _print_episode_data_summary_legacy(episode_data)


def clear_particles_cb():
    """Clear all particles manually."""
    try:
        if 'particle_system' not in globals() or globals()['particle_system'] is None:
            print("Particle system not initialized")
            return
        
        particle_system = globals()['particle_system']
        particle_system.clear_all_particles()
        print("Manually cleared all particles")
    except Exception as e:
        print(f"Error clearing particles: {e}")
        import traceback
        traceback.print_exc()


def get_spray_state_from_episode(episode_data, step_index):
    """Extract spray state from episode data at the given step index.
    
    This function tries multiple methods to access the spray state:
    1. Dictionary-based access through episode_data.data['obs']['spray']
    2. Attribute-based access through episode_data.obs.spray
    3. Legacy observation access through episode_data.observations
    
    Args:
        episode_data: EpisodeData object
        step_index: The step index to get spray state for
        
    Returns:
        bool: True if spray is enabled, False otherwise
    """
    try:
        # Method 1: Dictionary-based access (newest format)
        if hasattr(episode_data, 'data') and 'obs' in episode_data.data and 'spray' in episode_data.data['obs']:
            spray_data = episode_data.data['obs']['spray'][step_index]
            if isinstance(spray_data, torch.Tensor):
                return bool(spray_data[0].item())
            return bool(spray_data)
        
        # Method 2: Attribute-based access
        if hasattr(episode_data, 'obs') and hasattr(episode_data.obs, 'spray'):
            spray_data = episode_data.obs.spray[step_index]
            if isinstance(spray_data, torch.Tensor):
                return bool(spray_data[0].item())
            return bool(spray_data)
        
        # Method 3: Check observation list (legacy format)
        if hasattr(episode_data, 'observations') and episode_data.observations:
            obs = episode_data.observations[step_index]
            if isinstance(obs, dict) and 'policy' in obs and 'spray' in obs['policy']:
                spray_data = obs['policy']['spray']
                if isinstance(spray_data, torch.Tensor):
                    return bool(spray_data[0].item())
                return bool(spray_data)
        
        # Default: Return False if we can't find spray state
        return False
    except Exception as e:
        print(f"Error getting spray state: {e}")
        return False


def main():
    """Replay episodes loaded from a file.
    
    Use --show_particles to enable particle visualization during replay.
    Use --force_particles to show particles at regular intervals regardless of gripper state.
    Press 'P' to toggle particles on/off during replay.
    Press 'M' to manually create particles at the current position.
    Press 'S' to print detailed episode data summary.
    """
    global is_paused, particles_enabled, particle_system, last_particle_time, env, env_episode_data_map

    # Load dataset
    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()
    
    # Debug: Print HDF5 file structure
    print_hdf5_structure(dataset_file_handler)
    
    # Get episode names
    episode_names = list(dataset_file_handler.get_episode_names())
    
    # Load the first episode to inspect its structure
    if episode_count > 0:
        first_episode = dataset_file_handler.load_episode(episode_names[0], "cpu")
        
        print(f"\n===== Checking first episode for gripper state and EEF position =====")
        
        # Inspect gripper data access methods
        print("Attempting to find gripper_open and eef_pos...")
        success = False
        
        # Try using the get_current_gripper_eef_state function for step 0
        gripper_is_open, eef_pos = get_current_gripper_eef_state(first_episode, 0)
        if gripper_is_open is not False or eef_pos is not None:
            success = True
            print(f"✅ Successfully found data with get_current_gripper_eef_state() function")
            print(f"Step 0: gripper_open={gripper_is_open}, eef_pos={eef_pos}")
            
            # Check a few more steps
            for step in [1, 5, 10]:
                try:
                    gripper_is_open, eef_pos = get_current_gripper_eef_state(first_episode, step)
                    print(f"Step {step}: gripper_open={gripper_is_open}, eef_pos={eef_pos}")
                except Exception:
                    pass
        
        if not success:
            print("❌ Failed to find gripper_open and eef_pos data in the episode")
            print("This might affect particle creation during replay")
    
    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    episode_indices_to_replay = args_cli.select_episodes
    if len(episode_indices_to_replay) == 0:
        episode_indices_to_replay = list(range(episode_count))

    if args_cli.task is not None:
        env_name = args_cli.task
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    num_envs = args_cli.num_envs

    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=num_envs)

    # Disable all recorders and terminations
    env_cfg.recorders = {}
    env_cfg.terminations = {}

    # create environment from loaded config
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    # Initialize particle system if enabled
    particle_system = None
    if args_cli.show_particles:
        particle_system = ParticleSystem(env)
        print("Initialized particle system to show particles when gripper is open")
        
        # Add debug spiral particles if requested
        if args_cli.debug_particles:
            try:
                print("Creating debug spiral particles...")
                # Create a spiral of particles
                center = np.array([0.5, 0.0, 0.5])  # Center position
                radius = 0.2  # Initial radius
                height = 0.3  # Initial height
                num_points = 50  # Number of points in spiral
                
                for i in range(num_points):
                    # Calculate spiral position
                    angle = 0.5 * i
                    r = radius * (1 - i / num_points)  # Decreasing radius
                    h = height + (i / num_points) * 0.2  # Increasing height
                    
                    pos = np.array([
                        center[0] + r * np.cos(angle),
                        center[1] + r * np.sin(angle),
                        center[2] + h
                    ])
                    
                    # Create particle at this position
                    particle_system.create_particles(pos, count=1)
                    
                print(f"Created {num_points} particles in debug spiral pattern")
            except Exception as e:
                print(f"Error creating debug spiral: {e}")

    teleop_interface = Se3Keyboard(pos_sensitivity=0.1, rot_sensitivity=0.1, env=env)
    teleop_interface.add_callback("N", play_cb)
    teleop_interface.add_callback("B", pause_cb)
    teleop_interface.add_callback("P", toggle_particles_cb)
    teleop_interface.add_callback("M", force_create_particles_cb)
    teleop_interface.add_callback("S", print_episode_summary_cb)
    teleop_interface.add_callback("C", clear_particles_cb)
    
    print("\n===== REPLAY CONTROLS =====")
    print('Press "B" to pause and "N" to resume the replayed actions.')
    print('Press "P" to toggle particles on/off during replay.')
    print('Press "M" to manually create particles at the current position.')
    print('Press "S" to print detailed episode data summary.')
    print('Press "C" to clear all particles manually.')
    print("============================\n")
    
    # Variable to track previous gripper state for particle creation
    prev_gripper_open = False
    last_particle_time = time.time()

    # Determine if state validation should be conducted
    state_validation_enabled = False
    if args_cli.validate_states and num_envs == 1:
        state_validation_enabled = True
    elif args_cli.validate_states and num_envs > 1:
        print("Warning: State validation is only supported with a single environment. Skipping state validation.")

    # Get idle action (idle actions are applied to envs without next action)
    if hasattr(env_cfg, "idle_action"):
        idle_action = env_cfg.idle_action.repeat(num_envs, 1)
    else:
        idle_action = torch.zeros(env.action_space.shape)

    # reset before starting
    env.reset()
    teleop_interface.reset()

    # simulate environment -- run everything in inference mode
    replayed_episode_count = 0
    try:
        with torch.inference_mode():
            while simulation_app.is_running() and not simulation_app.is_exiting():
                env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}
                first_loop = True
                has_next_action = True
                while has_next_action:
                    # initialize actions with idle action so those without next action will not move
                    actions = idle_action
                    has_next_action = False
                    for env_id in range(num_envs):
                        env_next_action = env_episode_data_map[env_id].get_next_action()
                        if env_next_action is None:
                            next_episode_index = None
                            while episode_indices_to_replay:
                                next_episode_index = episode_indices_to_replay.pop(0)
                                if next_episode_index < episode_count:
                                    break
                                next_episode_index = None

                            if next_episode_index is not None:
                                replayed_episode_count += 1
                                print(f"{replayed_episode_count :4}: Loading #{next_episode_index} episode to env_{env_id}")
                                
                                # Clear all particles before loading a new episode
                                if args_cli.show_particles and particle_system:
                                    print("Clearing particles for new episode...")
                                    particle_system.clear_all_particles()
                                    # Reset the previous gripper state for the new episode
                                    prev_gripper_open = False
                                
                                episode_data = dataset_file_handler.load_episode(
                                    episode_names[next_episode_index], env.device
                                )
                                env_episode_data_map[env_id] = episode_data
                                
                                # Show episode data summary if requested
                                if args_cli.show_episode_data:
                                    print(f"\n\nPRINTING EPISODE DATA SUMMARY FOR EPISODE #{next_episode_index}")
                                    print(f"Episode name: {episode_names[next_episode_index]}")
                                    # Move to a separate thread to avoid blocking the main thread
                                    try:
                                        print_episode_data_summary(episode_data)
                                        print("SUCCESSFULLY PRINTED EPISODE DATA SUMMARY")
                                    except Exception as e:
                                        print(f"ERROR printing episode data summary: {e}")
                                        import traceback
                                        traceback.print_exc()
                                
                                # Set initial state for the new episode
                                initial_state = episode_data.get_initial_state()
                                env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)
                                # Get the first action for the new episode
                                env_next_action = env_episode_data_map[env_id].get_next_action()
                                has_next_action = True
                            else:
                                continue
                        else:
                            has_next_action = True
                        actions[env_id] = env_next_action
                    if first_loop:
                        first_loop = False
                    else:
                        while is_paused:
                            env.sim.render()
                            continue
                    env.step(actions)

                    # Store observations for checking gripper state
                    if isinstance(actions, tuple):
                        # Standard gym environment returns (obs, reward, done, info)
                        current_obs = actions[0]
                    else:
                        # Some environments might just return observations
                        current_obs = actions
                    
                    # Live logging of gripper and EEF state for each step
                    if args_cli.live_logging:
                        for env_id in range(num_envs):
                            if env_id in env_episode_data_map:
                                episode_data = env_episode_data_map[env_id]
                                if hasattr(episode_data, 'next_action_index'):
                                    current_index = episode_data.next_action_index - 1
                                    
                                    # Check both 'observations' and 'obs' attributes
                                    observations = None
                                    if hasattr(episode_data, 'observations') and len(episode_data.observations) > current_index >= 0:
                                        observations = episode_data.observations
                                    elif hasattr(episode_data, 'obs') and len(episode_data.obs) > current_index >= 0:
                                        observations = episode_data.obs
                                    
                                    if observations is not None:
                                        # Get the current step's data
                                        gripper_is_open, eef_pos = get_current_gripper_eef_state(episode_data, current_index)
                                        
                                        # Format and print
                                        eef_pos_str = f"[{eef_pos[0]:.4f}, {eef_pos[1]:.4f}, {eef_pos[2]:.4f}]" if eef_pos is not None else "None"
                                        print(f"[Env {env_id}, Step {current_index}] Gripper Open: {gripper_is_open}, EEF Position: {eef_pos_str}")
                    
                    # Handle particle creation based on gripper state
                    if args_cli.show_particles and particle_system:
                        try:
                            if env_episode_data_map and env_id in env_episode_data_map:
                                current_episode_data = env_episode_data_map[env_id]
                                
                                # Get current step index
                                step_index = current_episode_data.next_action_index - 1
                                if step_index >= 0:
                                    # Check both gripper and spray state
                                    current_gripper_open = gripper_is_open
                                    current_spray_enabled = get_spray_state_from_episode(current_episode_data, step_index)
                                    
                                    # Only create particles if BOTH gripper is open AND spray is enabled
                                    if current_gripper_open and current_spray_enabled:
                                        if not prev_particles_enabled:
                                            print(f"Starting particle creation (Gripper open and spray enabled)")
                                            prev_particles_enabled = True
                                        
                                        # Get current end effector position for particle creation
                                        eef_pos = eef_pos
                                        if eef_pos is not None:
                                            particle_system.create_particles(eef_pos, count=args_cli.particles_per_step)
                                    else:
                                        if prev_particles_enabled:
                                            if not current_gripper_open:
                                                print(f"Stopping particle creation (Gripper closed)")
                                            elif not current_spray_enabled:
                                                print(f"Stopping particle creation (Spray disabled)")
                                            prev_particles_enabled = False
                        except Exception as e:
                            print(f"Error handling particle creation: {e}")
                            import traceback
                            traceback.print_exc()
                break
    except KeyboardInterrupt:
        print("\nSimulation interrupted. Cleaning up...")
    finally:
        # Clean up particles before closing
        if args_cli.show_particles and particle_system:
            print("Cleaning up particles...")
            particle_system.clear_all_particles()
        
        # Close environment after replay is complete
        plural_trailing_s = "s" if replayed_episode_count > 1 else ""
        print(f"Finished replaying {replayed_episode_count} episode{plural_trailing_s}.")
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
