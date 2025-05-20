# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(3) control."""

import numpy as np
import weakref
from collections.abc import Callable
from scipy.spatial.transform import Rotation

import carb
import omni

from ..device_base import DeviceBase

# particle libs
import asyncio
import random
import itertools
from pxr import Usd, UsdGeom, Gf, Sdf, Vt
from pxr import PhysxSchema, UsdShade
from omni.physx.scripts import physicsUtils, particleUtils
from isaacsim.core.api.materials.omni_pbr import OmniPBR

# from . import Particle

class Se3Keyboard(DeviceBase):
    """A keyboard controller for sending SE(3) commands as delta poses and binary command (open/close).

    This class is designed to provide a keyboard controller for a robotic arm with a gripper.
    It uses the Omniverse keyboard interface to listen to keyboard events and map them to robot's
    task-space commands.

    The command comprises of two parts:

    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Toggle gripper (open/close)    K
        Move along x-axis              W                 S
        Move along y-axis              A                 D
        Move along z-axis              Q                 E
        Rotate along x-axis            Z                 X
        Rotate along y-axis            T                 G
        Rotate along z-axis            C                 V
        ============================== ================= =================

    .. seealso::

        The official documentation for the keyboard interface: `Carb Keyboard Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

    """

    def __init__(self, pos_sensitivity: float = 0.4, rot_sensitivity: float = 0.8):
        """Initialize the keyboard layer.

        Args:
            pos_sensitivity: Magnitude of input position command scaling. Defaults to 0.05.
            rot_sensitivity: Magnitude of scale input rotation commands scaling. Defaults to 0.5.
        """
        # store inputs
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # command buffers
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

        # particle variables
        self._is_creating_particles = False
        self._particle_task = None
        self._sphere_id_counter = itertools.count()

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for SE(3): {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tToggle gripper (open/close): K\n"
        msg += "\tMove arm along x-axis: W/S\n"
        msg += "\tMove arm along y-axis: A/D\n"
        msg += "\tMove arm along z-axis: Q/E\n"
        msg += "\tRotate arm along x-axis: Z/X\n"
        msg += "\tRotate arm along y-axis: T/G\n"
        msg += "\tRotate arm along z-axis: C/V"
        return msg

    """
    Operations
    """

    def reset(self):
        # default flags
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        A list of available keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> tuple[np.ndarray, bool]:
        """Provides the result from keyboard event state.

        Returns:
            A tuple containing the delta pose command and gripper commands.
        """
        # convert to rotation vector
        rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()
        # return the command and gripper state
        return np.concatenate([self._delta_pos, rot_vec]), self._close_gripper

    """
    Particle helpers.
    """
    def create_at_nozzle(self):
        stage = omni.usd.get_context().get_stage()
        # nozzle_path = "/World/franka_instanceable/panda_hand/Nozzle"
        nozzle_path = "/World/envs/env_0/Robot/panda_hand"
        print(nozzle_path)
        nozzle_prim = stage.GetPrimAtPath(nozzle_path)
        print(nozzle_prim)

        if not nozzle_prim or not nozzle_prim.IsA(UsdGeom.Xform):
            print(f"⚠️ Nozzle prim not found or invalid at {nozzle_path}")
            return Gf.Vec3f(0, 0, 0)

        try:
            xformable = UsdGeom.Xformable(nozzle_prim)
            world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            world_position = Gf.Vec3f(world_transform.ExtractTranslation())
            print(f"🌐 World position of Nozzle: {world_position}")
            return world_position
        except Exception as e:
            print(f"❌ Failed to compute world transform: {e}")
            return Gf.Vec3f(0, 0, 0)
        
    async def _create_particle_with_auto_stop(self):
        print("🟣 Creating particle system...")

        stage = omni.usd.get_context().get_stage()
        prim_id = next(self._sphere_id_counter)
        prim_path = f"/World/ParticleScene_{prim_id}"

        particle_system_path = prim_path + "/particleSystem"
        particle_system = PhysxSchema.PhysxParticleSystem.Define(stage, particle_system_path)
        print("PhysxSchema.PhysxParticleSystem.Define") # ok
        print(f"stage.GetDefaultPrim().GetPath(): {stage.GetDefaultPrim().GetPath()}")
        # particle_system.CreateSimulationOwnerRel().SetTargets([stage.GetDefaultPrim().GetPath()])
        particle_system.CreateSimulationOwnerRel().SetTargets(["/World/envs/env_0/Robot/panda_hand"])
        print("CreateSimulationOwnerRel().SetTargets") # not ok
        particle_system.CreateParticleContactOffsetAttr().Set(0.1)
        print("CreateParticleContactOffsetAttr") # not ok
        particle_system.CreateMaxVelocityAttr().Set(250.0)

        print("particle created") # not ok

        size = 0.2
        contact_offset = 0.1
        rest_offset = 0.99 * 0.6 * contact_offset
        spacing = 1.8 * rest_offset
        num_samples = round(size / spacing) + 1

        lower = Gf.Vec3f(-size * 0.2)
        positions, _ = particleUtils.create_particles_grid(
            lower, spacing, num_samples, num_samples, num_samples
        )

        print("particle: before velocities")

        velocities = [Gf.Vec3f(random.uniform(5.0, 10.0), 0.0, 0.0) for _ in positions]

        point_instancer_path = prim_path + "/particles"
        particleUtils.add_physx_particleset_pointinstancer(
            stage,
            Sdf.Path(point_instancer_path),
            Vt.Vec3fArray(positions),
            Vt.Vec3fArray(velocities),
            particle_system_path,
            self_collision=True,
            fluid=True,
            particle_group=0,
            particle_mass=1.1,
            density=1.0,
        )

        print("creating at nozzle")
        position = self.create_at_nozzle()
        point_instancer = UsdGeom.PointInstancer.Get(stage, point_instancer_path)
        physicsUtils.set_or_add_translate_op(point_instancer, translate=position)

        prototype_path = point_instancer_path + "/particlePrototype0"
        sphere = UsdGeom.Sphere.Get(stage, prototype_path)
        sphere.CreateRadiusAttr().Set(rest_offset * 0.2)

        material_path = prim_path + "/purpleMaterial"
        purple_material = OmniPBR(
            prim_path=material_path,
            name="omni_pbr_purple",
            color=np.array([0.5, 0.0, 1.0]),
        )
        material = purple_material._material

        UsdShade.MaterialBindingAPI(sphere).Bind(material)

        print(f"✅ Particle created with purple material at {position}")

    async def _create_particles_periodically(self):
        try:
            while self._is_creating_particles:
                await self._create_particle_with_auto_stop()
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            print("❌ Particle task was cancelled.")

    def _toggle_particle_creation(self):
        self._is_creating_particles = not self._is_creating_particles
        if self._is_creating_particles:
            print("▶️ K key pressed: Starting particle generation...")
            self._particle_task = asyncio.ensure_future(self._create_particles_periodically())
        else:
            print("⏹️ K key pressed again: Stopping particle generation...")
            # if self._particle_task:
            #     self._particle_task.cancel()

    """
    Internal helpers.
    """
    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "L":
                self.reset()
            if event.input.name == "K":
                self._close_gripper = not self._close_gripper
                print(f"K pressed")
                self._toggle_particle_creation()
            elif event.input.name in ["W", "S", "A", "D", "Q", "E"]:
                self._delta_pos += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["Z", "X", "T", "G", "C", "V"]:
                self._delta_rot += self._INPUT_KEY_MAPPING[event.input.name]
        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in ["W", "S", "A", "D", "Q", "E"]:
                self._delta_pos -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["Z", "X", "T", "G", "C", "V"]:
                self._delta_rot -= self._INPUT_KEY_MAPPING[event.input.name]
        # additional callbacks
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # toggle: gripper command
            "K": True,
            # x-axis (forward)
            "W": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            # y-axis (left-right)
            "A": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            # z-axis (up-down)
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,
            # roll (around x-axis)
            "Z": np.asarray([1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "X": np.asarray([-1.0, 0.0, 0.0]) * self.rot_sensitivity,
            # pitch (around y-axis)
            "T": np.asarray([0.0, 1.0, 0.0]) * self.rot_sensitivity,
            "G": np.asarray([0.0, -1.0, 0.0]) * self.rot_sensitivity,
            # yaw (around z-axis)
            "C": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            "V": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,
        }
