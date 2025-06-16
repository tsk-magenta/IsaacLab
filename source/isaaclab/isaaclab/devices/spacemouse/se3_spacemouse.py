# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spacemouse controller for SE(3) control."""

import hid
import numpy as np
import threading
import time
from collections.abc import Callable
from scipy.spatial.transform import Rotation

from ..device_base import DeviceBase
from .utils import convert_buffer

# particle libs
import asyncio
import random
import itertools
from pxr import Usd, UsdGeom, Gf, Sdf, Vt
from pxr import PhysxSchema, UsdShade
from omni.physx.scripts import physicsUtils, particleUtils
from isaacsim.core.api.materials.omni_pbr import OmniPBR

class Se3SpaceMouse(DeviceBase):
    """A space-mouse controller for sending SE(3) commands as delta poses.

    This class implements a space-mouse controller to provide commands to a robotic arm with a gripper.
    It uses the `HID-API`_ which interfaces with USD and Bluetooth HID-class devices across multiple platforms [1].

    The command comprises of two parts:

    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.

    Note:
        The interface finds and uses the first supported device connected to the computer.

    Currently tested for following devices:

    - SpaceMouse Compact: https://3dconnexion.com/de/product/spacemouse-compact/

    .. _HID-API: https://github.com/libusb/hidapi

    """

    def __init__(self, pos_sensitivity: float = 0.4, rot_sensitivity: float = 0.8):
        """Initialize the space-mouse layer.

        Args:
            pos_sensitivity: Magnitude of input position command scaling. Defaults to 0.4.
            rot_sensitivity: Magnitude of scale input rotation commands scaling. Defaults to 0.8.
        """
        # store inputs
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        # acquire device interface
        self._device = hid.device()
        self._find_device()
        # read rotations
        self._read_rotation = False

        # command buffers
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)
        # dictionary for additional callbacks
        self._additional_callbacks = dict()
        # run a thread for listening to device updates
        self._thread = threading.Thread(target=self._run_device)
        self._thread.daemon = True
        self._thread.start()

        # particle variables
        self._is_creating_particles = False
        self._particle_task = None
        self._sphere_id_counter = itertools.count()

    def __del__(self):
        """Destructor for the class."""
        self._thread.join()

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Spacemouse Controller for SE(3): {self.__class__.__name__}\n"
        msg += f"\tManufacturer: {self._device.get_manufacturer_string()}\n"
        msg += f"\tProduct: {self._device.get_product_string()}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tRight button: reset command\n"
        msg += "\tLeft button: toggle gripper command (open/close)\n"
        msg += "\tMove mouse laterally: move arm horizontally in x-y plane\n"
        msg += "\tMove mouse vertically: move arm vertically\n"
        msg += "\tTwist mouse about an axis: rotate arm about a corresponding axis"
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
        # check keys supported by callback
        if key not in ["L", "R"]:
            raise ValueError(f"Only left (L) and right (R) buttons supported. Provided: {key}.")
        # TODO: Improve this to allow multiple buttons on same key.
        self._additional_callbacks[key] = func

    def advance(self) -> tuple[np.ndarray, bool]:
        """Provides the result from spacemouse event state.

        Returns:
            A tuple containing the delta pose command and gripper commands.
        """
        rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()
        # if new command received, reset event flag to False until keyboard updated.
        return np.concatenate([self._delta_pos, rot_vec]), self._close_gripper

    """
    Internal helpers.
    """

    def _find_device(self):
        """Find the device connected to computer."""
        found = False
        # implement a timeout for device search
        for _ in range(5):
            for device in hid.enumerate():
                if (
                    device["product_string"] == "SpaceMouse Compact"
                    or device["product_string"] == "SpaceMouse Wireless"
                ):
                    # set found flag
                    found = True
                    vendor_id = device["vendor_id"]
                    product_id = device["product_id"]
                    # connect to the device
                    self._device.close()
                    self._device.open(vendor_id, product_id)
            # check if device found
            if not found:
                time.sleep(1.0)
            else:
                break
        # no device found: return false
        if not found:
            raise OSError("No device found by SpaceMouse. Is the device connected?")

    def _run_device(self):
        """Listener thread that keeps pulling new messages."""
        # keep running
        while True:
            # read the device data
            data = self._device.read(7)
            if data is not None:
                # readings from 6-DoF sensor
                if data[0] == 1:
                    self._delta_pos[1] = self.pos_sensitivity * convert_buffer(data[1], data[2])
                    self._delta_pos[0] = self.pos_sensitivity * convert_buffer(data[3], data[4])
                    self._delta_pos[2] = self.pos_sensitivity * convert_buffer(data[5], data[6]) * -1.0
                elif data[0] == 2 and not self._read_rotation:
                    self._delta_rot[1] = self.rot_sensitivity * convert_buffer(data[1], data[2])
                    self._delta_rot[0] = self.rot_sensitivity * convert_buffer(data[3], data[4])
                    self._delta_rot[2] = self.rot_sensitivity * convert_buffer(data[5], data[6]) * -1.0
                # readings from the side buttons
                elif data[0] == 3:
                    # press left button
                    if data[1] == 1:
                        # close gripper
                        self._close_gripper = not self._close_gripper
                        print(f"Left pressed")
                        # self._toggle_particle_creation() # thread runtime error
                        # additional callbacks
                        if "L" in self._additional_callbacks:
                            self._additional_callbacks["L"]()
                    # right button is for reset
                    if data[1] == 2:
                        # reset layer
                        self.reset()
                        # additional callbacks
                        if "R" in self._additional_callbacks:
                            self._additional_callbacks["R"]()
                    if data[1] == 3:
                        self._read_rotation = not self._read_rotation

    """
    Particle helpers.
    """
    def create_at_nozzle(self):
        stage = omni.usd.get_context().get_stage()
        # nozzle_path = "/World/franka_instanceable/panda_hand/Nozzle"
        # nozzle_path = "/World/envs/env_0/Robot/panda_hand"
        nozzle_path = "/World/envs/env_0/Robot2"
        print(nozzle_path)
        nozzle_prim = stage.GetPrimAtPath(nozzle_path)
        print(nozzle_prim)

        if not nozzle_prim or not nozzle_prim.IsA(UsdGeom.Xform):
            print(f"⚠️ Nozzle prim not found or invalid at {nozzle_path}")
            return Gf.Vec3f(0, 0, 0)

        try:
            xformable = UsdGeom.Xformable(nozzle_prim)
            world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            print(world_transform.ExtractTranslation())
            world_position = Gf.Vec3f(world_transform.ExtractTranslation())
            # world_position = (0.4, 0, 0.77)
            world_position = (0, 0, 0)
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
        # print(f"stage.GetDefaultPrim().GetPath(): {stage.GetDefaultPrim().GetPath()}") # nothing printed
        # particle_system.CreateSimulationOwnerRel().SetTargets([stage.GetDefaultPrim().GetPath()]) # correct: /World
        particle_system.CreateSimulationOwnerRel().SetTargets(["/World"])
        # print("CreateSimulationOwnerRel().SetTargets") # ok
        particle_system.CreateParticleContactOffsetAttr().Set(0.1)
        # print("CreateParticleContactOffsetAttr") # ok
        particle_system.CreateMaxVelocityAttr().Set(250.0)

        # print("particle created") # ok

        size = 0.2
        contact_offset = 0.1
        rest_offset = 0.99 * 0.6 * contact_offset
        spacing = 1.8 * rest_offset
        num_samples = round(size / spacing) + 1

        lower = Gf.Vec3f(-size * 0.2)
        positions, _ = particleUtils.create_particles_grid(
            lower, spacing, num_samples, num_samples, num_samples
        )

        # print("particle: before velocities")

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

        # print("creating at nozzle")
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
            if self._particle_task:
                self._particle_task.cancel()
