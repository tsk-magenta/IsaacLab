# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
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


class Se3Keyboard(DeviceBase):
    def __init__(self, pos_sensitivity: float = 0.4, rot_sensitivity: float = 0.8, env = None):
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self._env = env  # Store environment reference
        print(f"[Se3Keyboard] __init__: _env: {self._env}")

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

    def __del__(self):
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None
        if self._update_sub:
            self._update_sub.unsubscribe()

    def reset(self):
        self._close_gripper = False
        self._delta_pos = np.zeros(3)
        self._delta_rot = np.zeros(3)

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    def advance(self) -> tuple[np.ndarray, bool]:
        rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()
        return np.concatenate([self._delta_pos, rot_vec]), self._close_gripper

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "L":
                self.reset()
            if event.input.name == "K":
                print("K 누름")
                self._toggle_particle_creation()
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

    def _on_update(self, _e):
        """매 프레임마다 구체 생성 + 이동"""
        # Get end effector position from environment if available
        if self._env is not None:
            try:
                # Try different ways to access the observations based on environment type
                obs = None
                eef_world_pos = None
                
                # Method 1: Check if the environment has a get_observations method
                if hasattr(self._env.unwrapped, 'get_observations'):
                    obs = self._env.unwrapped.get_observations()
                # Method 2: Try to access the most recent observation if available
                elif hasattr(self._env, 'last_observations'):
                    obs = self._env.last_observations
                # Method 3: For FrankaPaintCustomEnv that might store observations differently
                elif hasattr(self._env.unwrapped, 'last_obs'):
                    obs = self._env.unwrapped.last_obs
                
                # Process observation if available
                if obs is not None and isinstance(obs, dict):
                    if "policy" in obs and "eef_pos" in obs["policy"]:
                        # Get end effector position
                        eef_world_pos = obs["policy"]["eef_pos"][0].cpu().numpy()
                        self._current_eef_pos = eef_world_pos
                    
                # Alternative direct access if observation dictionary access failed
                if eef_world_pos is None and hasattr(self._env.unwrapped, 'robot'):
                    if hasattr(self._env.unwrapped.robot, 'eef_pos'):
                        eef_world_pos = self._env.unwrapped.robot.eef_pos.cpu().numpy()
                        self._current_eef_pos = eef_world_pos
                
                # Print position every second if we have it
                if self._current_eef_pos is not None:
                    current_time = time.time()
                    if current_time - self._last_print_time >= 1.0:
                        print(f"[KB] EEF World Pos: [X={self._current_eef_pos[0]:.3f}, Y={self._current_eef_pos[1]:.3f}, Z={self._current_eef_pos[2]:.3f}]")
                        self._last_print_time = current_time
            except Exception as e:
                # Log the exception but continue execution
                print(f"Error _on_update: {e}")
                pass
                
        if self._is_creating_particles:
            self._particleCCC()  # 구체 생성
        self._move_all_spheres()  # 구체 이동은 계속 진행

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
                            print(f"✅ 누적 된 구체 수: {self._stopped_count}")  # ✅ 출력

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
