# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(3) control."""

import numpy as np
import weakref
import time
from collections.abc import Callable
from scipy.spatial.transform import Rotation

import carb
import omni

from ..device_base import DeviceBase

import omni.usd
import omni.kit.app  # ✅ 프레임 업데이트용
from pxr import UsdGeom, Gf, Usd, UsdShade, Sdf
import random
import math
import torch


class Se3Keyboard(DeviceBase):
    def __init__(self, pos_sensitivity: float = 0.4, rot_sensitivity: float = 0.8, env = None):
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self._env = env  # Store environment reference
        print(f"[Se3Keyboard] __init__: _env: {self._env}")
        
        # Print helpful key instructions
        print("═════════════════════════════════════════")
        print("Se3Keyboard initialized with special keys:")
        print(" • K: Toggle particle creation and spray")
        print(" • P: Clean all particles and reset count")
        print(" • R: Reset keyboard state")
        print("═════════════════════════════════════════")

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        self._create_key_bindings()
        self._close_gripper = False
        self._delta_pos = np.zeros(3)
        self._delta_rot = np.zeros(3)
        self._additional_callbacks = dict()

        # 파티클 관련
        self._is_creating_particles = False
        self._spheres = []  # 모든 구체를 저장하는 리스트
        self._update_sub = None  # 프레임 업데이트 구독 핸들

        self._stopped_count = 0
        self._last_print_time = 0  # Track last time EEF position was printed
        self._current_eef_pos = None  # Store current end effector position
        self._last_obv = None  # Store the last observation from env.step()
        
        # Make the stopped count globally accessible for termination functions
        if env is not None:
            env.keyboard_stopped_count = 0
            # Store reference for later lookup
            if not hasattr(env, 'keyboard'):
                env.keyboard = self

    def __del__(self):
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None
        if self._update_sub:
            self._update_sub.unsubscribe()

    def reset(self):
        try:
            print(f"[KB] Resetting particle state")
            
            self._close_gripper = False
            self._delta_pos = np.zeros(3)
            self._delta_rot = np.zeros(3)
            
            # stop creating particles
            self._is_creating_particles = False
            
            # Update the global state in the paint mdp module if available
            try:
                from isaaclab_tasks.manager_based.manipulation.paint.mdp.observations import set_keyboard_spray_state
                set_keyboard_spray_state(self._is_creating_particles)
            except ImportError:
                # Module not available, this is okay in other environments
                pass

            # Clean up all particles
            self._cleanup_all_particles()
            
            # Additional cleanup - delete any stray KeyboardSphere objects that might still exist
            try:
                stage = omni.usd.get_context().get_stage()
                base_path = "/World/KeyboardSphere"
                i = 0
                stray_count = 0
                while True:
                    sphere_path = f"{base_path}{i}"
                    sphere_prim = stage.GetPrimAtPath(sphere_path)
                    if sphere_prim.IsValid():
                        stage.RemovePrim(sphere_prim.GetPath())
                        stray_count += 1
                    else:
                        # No more particles with this index
                        if i > 1000:  # Safety limit
                            break
                        # Skip ahead if we've checked a lot of non-existent particles
                        if i > 10 and i % 10 == 0 and not stage.GetPrimAtPath(f"{base_path}{i+10}").IsValid():
                            i += 9  # Skip ahead but still increment by 1 in the loop
                    i += 1
                    
                    # Break after a reasonable number of checks
                    if i > 5000:
                        print("[KB] Safety limit reached when cleaning stray particles")
                        break
                
                if stray_count > 0:
                    print(f"[KB] Found and removed {stray_count} stray particles")
            except Exception as e:
                print(f"[KB] Error cleaning stray particles: {e}")
                
            # Also disable particle creation
            self._is_creating_particles = False
            
            # Update the global state again to ensure it's synchronized
            try:
                from isaaclab_tasks.manager_based.manipulation.paint.mdp.observations import set_keyboard_spray_state
                set_keyboard_spray_state(self._is_creating_particles)
            except ImportError:
                # Module not available, this is okay in other environments
                pass
            
            print("[KB] Reset complete")
        except Exception as e:
            print(f"[KB] Error during reset: {e}")
            # Make sure state is reset even if cleanup fails
            self._close_gripper = False
            self._delta_pos = np.zeros(3)
            self._delta_rot = np.zeros(3)
            self._stopped_count = 0
            self._is_creating_particles = False
            if self._env is not None:
                self._env.keyboard_stopped_count = 0

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    def advance(self) -> tuple[np.ndarray, bool]:
        rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()
        return np.concatenate([self._delta_pos, rot_vec]), self._close_gripper

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self.reset()
            if event.input.name == "K":
                print("K 누름")
                self._toggle_particle_creation()
            if event.input.name == "P":
                print("P 누름 - Cleaning all particles")
                self._cleanup_all_particles()
            elif event.input.name in self._INPUT_KEY_MAPPING:
                delta = self._INPUT_KEY_MAPPING[event.input.name]
                if isinstance(delta, np.ndarray):
                    if delta.shape[0] == 3:
                        if event.input.name in ["W", "S", "A", "D", "Q", "E"]:
                            self._delta_pos += delta
                        else:
                            self._delta_rot += delta

        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._INPUT_KEY_MAPPING:
                delta = self._INPUT_KEY_MAPPING[event.input.name]
                if isinstance(delta, np.ndarray):
                    if delta.shape[0] == 3:
                        if event.input.name in ["W", "S", "A", "D", "Q", "E"]:
                            self._delta_pos -= delta
                        else:
                            self._delta_rot -= delta

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()

        return True

    def _create_key_bindings(self):
        self._INPUT_KEY_MAPPING = {
            "K": True,
            "W": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "A": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,
            "Z": np.asarray([1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "X": np.asarray([-1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "T": np.asarray([0.0, 1.0, 0.0]) * self.rot_sensitivity,
            "G": np.asarray([0.0, -1.0, 0.0]) * self.rot_sensitivity,
            "C": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            "V": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,
        }

    def _toggle_particle_creation(self):
        self._is_creating_particles = not self._is_creating_particles
        
        # Update the global state in the paint mdp module if available
        try:
            from isaaclab_tasks.manager_based.manipulation.paint.mdp.observations import set_keyboard_spray_state
            set_keyboard_spray_state(self._is_creating_particles)
        except ImportError:
            # Module not available, this is okay in other environments
            pass
        
        if self._is_creating_particles:
            print("▶️ K key pressed: Starting particle generation...")
           
            # 프레임 업데이트 구독
            if self._update_sub is None:
                self._update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
                    self._on_update
                )
        else:
            print("⏹️ K key pressed again: Stopping particle generation...")

            # 구체 생성은 멈추고, 이동은 계속 진행
            self._is_creating_particles = False
            
            # Update the global state again to ensure it's synchronized
            try:
                from isaaclab_tasks.manager_based.manipulation.paint.mdp.observations import set_keyboard_spray_state
                set_keyboard_spray_state(self._is_creating_particles)
            except ImportError:
                # Module not available, this is okay in other environments
                pass

    def _on_update(self, _e):
        """매 프레임마다 구체 생성 + 이동"""
        try:
            self._move_all_spheres()

            # Check if particles are enabled and environment exists
            if not self._is_creating_particles:
                return
                
            # Get end effector position from environment if available
            if self._env is not None:
                # Debug timing
                current_time = time.time()
                
                # Method directly copying from record_demos.py line 413
                try:
                    # Try to get the most recent observation directly - this is what works in record_demos.py
                    eef_world_pos = None
                    
                    # First check if we have a recent observation stored from record_demos.py
                    if hasattr(self._env, 'last_obv') and self._env.last_obv is not None:
                        # Use the same exact approach from record_demos.py
                        obv = self._env.last_obv
                        if isinstance(obv, tuple) and len(obv) > 0:
                            obv = obv[0]  # First element might be the observation
                            
                        if isinstance(obv, dict) and "policy" in obv and "eef_pos" in obv["policy"]:
                            eef_world_pos = obv["policy"]["eef_pos"][0].cpu().numpy()
                            if current_time - self._last_print_time > 1.0:
                                print(f"[KB DEBUG] Got EEF from env.last_obv")
                    
                    # If we still don't have a position, try to get it from observation_manager
                    # This is our best fallback option
                    if eef_world_pos is None and hasattr(self._env, 'observation_manager'):
                        try:
                            obv = self._env.observation_manager.compute_observations()
                            if isinstance(obv, dict) and "policy" in obv and "eef_pos" in obv["policy"]:
                                eef_world_pos = obv["policy"]["eef_pos"][0].cpu().numpy()
                                if current_time - self._last_print_time > 1.0:
                                    print(f"[KB DEBUG] Got EEF from observation_manager")
                        except Exception as e:
                            if current_time - self._last_print_time > 1.0:
                                print(f"[KB DEBUG] Error with observation_manager: {e}")
                    
                    # Last resort: Use a fallback position
                    if eef_world_pos is None:
                        # Use the last known position if we have one
                        if self._current_eef_pos is not None:
                            eef_world_pos = self._current_eef_pos
                            if current_time - self._last_print_time > 1.0:
                                print("[KB DEBUG] Using last known EEF position")
                        else:
                            # Otherwise use a default position in front of the robot
                            eef_world_pos = np.array([0.5, 0.0, 0.5])
                            if current_time - self._last_print_time > 1.0:
                                print("[KB DEBUG] Using default position (0.5, 0.0, 0.5)")
                                
                                # Only print diagnostics data once per second to avoid spamming
                                print(f"[KB DEBUG] ENV TYPE: {type(self._env).__name__}")
                                print(f"[KB DEBUG] Has last_obv: {hasattr(self._env, 'last_obv')}")
                                if hasattr(self._env, 'last_obv'):
                                    print(f"[KB DEBUG] last_obv type: {type(self._env.last_obv)}")
                    
                    # Log the position periodically
                    if current_time - self._last_print_time > 1.0:
                        print(f"[KB] eef_world_pos: {eef_world_pos}")
                        self._last_print_time = current_time
                    
                    # Store the position for future reference 
                    if eef_world_pos is not None:
                        self._current_eef_pos = eef_world_pos
                        
                    
                    # Create new particles if we have a valid position
                    if eef_world_pos is not None:
                        # Create particles at end effector position
                        self._particleCCC()
                    
                except Exception as e:
                    # If there's an error getting EEF position, use a fallback
                    print(f"Error getting EEF position: {e}")
                    import traceback
                    traceback.print_exc()
                    # Move existing particles even if we can't create new ones
            else:
                # If no environment is available, just move existing particles
                print("Environment reference is None, cannot get EEF position")
                
        except Exception as e:
            print(f"Error in _on_update: {e}")
            import traceback
            traceback.print_exc()

    def _move_all_spheres(self):
        for sphere_prim in self._spheres[:]:  # 리스트 복사 순회
            if sphere_prim:
                translate_attr = sphere_prim.GetPrim().GetAttribute("xformOp:translate")
                if not translate_attr:
                    continue
                current_pos = translate_attr.Get()
                if current_pos is None:
                    continue

                # ✅ X가 1 이상이면 삭제
                if current_pos[0] > 1.0:
                    self._delete_sphere(sphere_prim)
                    self._spheres.remove(sphere_prim)
                    continue

                # ✅ X가 0.6 이상이면 필터 조건 체크
                if current_pos[0] >= 0.6:
                    if -0.38 <= current_pos[1] <= 0.38 and 0.0 <= current_pos[2] <= 0.93:
                        # 필터 조건 통과: 이동 멈춤
                        if not hasattr(sphere_prim, 'stopped') or not sphere_prim.stopped:
                            sphere_prim.stopped = True  # 'stopped' 마커 속성 설정
                            self._stopped_count += 1  # ✅ 멈춘 구체 수 증가
                            if self._stopped_count % 100 == 0:
                                print(f"✅ 누적 된 구체 수: {self._stopped_count}")  # ✅ 출력
                            
                            # Update the environment's keyboard_stopped_count
                            if self._env is not None:
                                self._env.keyboard_stopped_count = self._stopped_count

                            # ✅ X를 0.65로 고정하고 YZ는 유지
                            fixed_pos = Gf.Vec3d(0.65, current_pos[1], current_pos[2])
                            UsdGeom.XformCommonAPI(sphere_prim).SetTranslate(fixed_pos)
                            UsdGeom.XformCommonAPI(sphere_prim).SetScale(Gf.Vec3f(5.0, 5.0, 5.0))

                        continue  # 이동하지 않음

                # ✅ 최초 방향 설정
                if not hasattr(sphere_prim, 'direction_x'):
                    sphere_prim.direction_x = random.uniform(1.0, 1.0)
                    sphere_prim.direction_y = random.uniform(-0.25, 0.25)
                    sphere_prim.direction_z = random.uniform(-0.25, 0.25)
                    sphere_prim.stopped = False  # 초기엔 이동 가능

                # ✅ 이동 멈춘 경우는 건너뜀
                if sphere_prim.stopped:
                    continue

                # ✅ 이동 수행
                move_factor = 0.05
                new_pos = Gf.Vec3d(
                    current_pos[0] + sphere_prim.direction_x * move_factor,
                    current_pos[1] + sphere_prim.direction_y * move_factor,
                    current_pos[2] + sphere_prim.direction_z * move_factor,
                )

                UsdGeom.XformCommonAPI(sphere_prim).SetTranslate(new_pos)

    def _delete_sphere(self, sphere_prim):
        """구체를 삭제하는 함수"""
        prim = sphere_prim.GetPrim()
        if prim:
            stage = omni.usd.get_context().get_stage()
            stage.RemovePrim(prim.GetPath())  # prim의 경로를 전달

    def _particleCCC(self):
        stage = omni.usd.get_context().get_stage()
        hand_path = "/World/light"
        hand_prim = stage.GetPrimAtPath(hand_path)

        if not hand_prim.IsValid():
            return

        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        world_transform = xform_cache.GetLocalToWorldTransform(hand_prim)
        hand_translation = world_transform.ExtractTranslation() # (0, 0, 0)
        
        # Use EEF position if available, otherwise use default position
        if self._current_eef_pos is not None:
            # Unpack the NumPy array values individually to create the Vec3d
            hand_translation = Gf.Vec3d(
                float(self._current_eef_pos[0]), 
                float(self._current_eef_pos[1]), 
                float(self._current_eef_pos[2])
            )
        #     print(f"[KB] _particleCCC: Using EEF pos: {hand_translation}")
        # else:
        #     print(f"[KB] _particleCCC: Using default pos: {hand_translation}")

        offset = Gf.Vec3d(0.0, 0.0, 0.0)
        # print(f"[KB] _particleCCC: offset: {offset}")
        base_pos = hand_translation + offset
        # print(f"[KB] _particleCCC: base_pos: {base_pos}")

        base_path = "/World/KeyboardSphere"
       
        # ✅ 프레임당 3개 구체 생성
        for _ in range(10):
            # 유일한 이름 생성
            i = 0
            while stage.GetPrimAtPath(f"{base_path}{i}").IsValid():
                i += 1

            sphere_path = f"{base_path}{i}"
            rand_offset = Gf.Vec3d(
                random.uniform(-0.03, 0.03),
                random.uniform(-0.03, 0.03),
                random.uniform(-0.03, 0.03),
            )
            spawn_pos = base_pos + rand_offset

            sphere_prim = UsdGeom.Cube.Define(stage, sphere_path)

            sphere_prim.GetSizeAttr().Set(0.005)  # 큐브 한 변의 길이 (구체 지름과 유사하게 맞춤)
            #sphere_prim.GetRadiusAttr().Set(0.005)
            UsdGeom.XformCommonAPI(sphere_prim).SetTranslate(spawn_pos)

            prim = sphere_prim.GetPrim()
            material_path = Sdf.Path("/World/Looks/Purple")
            if not stage.GetPrimAtPath(str(material_path)).IsValid():
                material = UsdShade.Material.Define(stage, material_path)
                shader = UsdShade.Shader.Define(stage, material_path.AppendPath("Shader"))
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.99, 0.0, 0.99))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            else:
                material = UsdShade.Material(stage.GetPrimAtPath(material_path))

            UsdShade.MaterialBindingAPI(prim).Bind(material)

            self._spheres.append(sphere_prim)

    def get_stopped_count(self) -> int:
        """Get the current stopped particle count.
        
        Returns:
            The number of stopped particles.
        """
        return self._stopped_count

    def _cleanup_all_particles(self):
        """Clean up all particles without resetting other state"""
        try:
            # Get the stage
            stage = omni.usd.get_context().get_stage()
            
            # Track how many particles were deleted
            deleted_count = 0
            
            # Delete all spheres both from USD stage and our list
            if hasattr(self, '_spheres') and self._spheres:
                for sphere_prim in self._spheres[:]:
                    try:
                        # Get the prim path
                        if hasattr(sphere_prim, 'GetPrim'):
                            prim = sphere_prim.GetPrim()
                            if prim and prim.IsValid():
                                # Remove from stage
                                stage.RemovePrim(prim.GetPath())
                                deleted_count += 1
                    except Exception as e:
                        print(f"Error deleting particle: {e}")
                
                # Clear the list
                self._spheres.clear()
            
            # Reset the counts but keep other state
            previous_count = getattr(self, '_stopped_count', 0)
            self._stopped_count = 0
            
            # Reset the environment's keyboard_stopped_count if available
            if self._env is not None:
                self._env.keyboard_stopped_count = 0
            
            if deleted_count > 0 or previous_count > 0:
                print(f"[KB] Cleanup complete: Deleted {deleted_count} particles, reset count from {previous_count} to 0")
        except Exception as e:
            print(f"[KB] Error during cleanup: {e}")
            # Make sure counts are reset even if cleanup fails
            self._stopped_count = 0
            if self._env is not None:
                self._env.keyboard_stopped_count = 0
                