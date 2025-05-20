from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf
import os

# Paths
robot_usd_path = "/home/hys/Downloads/rb10_1300e/rb10_1300e/rb10_1300e/rb10_1300e.usd"
gripper_usd_path = "/home/hys/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/Franka/franka.usd"
combined_usd_path = "/home/hys/Downloads/rb10_with_panda_gripper_v2.usd"

# Remove existing file if any
if os.path.exists(combined_usd_path):
    print(f"Removing existing USD file: {combined_usd_path}")
    os.remove(combined_usd_path)

# Create new stage
stage = Usd.Stage.CreateNew(combined_usd_path)

# Create the root prim for the robot
robot_root = stage.DefinePrim("/rb10_gripper", "Xform")
stage.SetDefaultPrim(robot_root)

# Apply ArticulationRootAPI
UsdPhysics.ArticulationRootAPI.Apply(robot_root)

# Create links
link_names = ["link0", "link1", "link2", "link3", "link4", "link5", "link6"]

for link_name in link_names:
    # Create link prim
    link_prim = stage.DefinePrim(f"/rb10_gripper/{link_name}", "Xform")
    # Apply RigidBodyAPI
    UsdPhysics.RigidBodyAPI.Apply(link_prim)
    
    # Create visuals and collisions subfolders
    visuals = stage.DefinePrim(f"/rb10_gripper/{link_name}/visuals", "Xform")
    collisions = stage.DefinePrim(f"/rb10_gripper/{link_name}/collisions", "Xform")
    UsdPhysics.CollisionAPI.Apply(collisions)
    
    # Add reference to original robot USD for this link's geometry
    visuals.GetReferences().AddReference(
        assetPath=robot_usd_path,
        primPath=f"/rb10_1300e/{link_name}/visuals"
    )
    
    collisions.GetReferences().AddReference(
        assetPath=robot_usd_path,
        primPath=f"/rb10_1300e/{link_name}/collisions"
    )

# Create gripper components as separate link (not under link6)
# 1. Create panda_hand
panda_hand = stage.DefinePrim("/rb10_gripper/panda_hand", "Xform")
UsdPhysics.RigidBodyAPI.Apply(panda_hand)

# Add geometry for panda_hand
hand_geo = stage.DefinePrim("/rb10_gripper/panda_hand/geometry", "Xform")
hand_mesh = stage.DefinePrim("/rb10_gripper/panda_hand/geometry/panda_hand", "Mesh")
hand_mesh.GetReferences().AddReference(
    assetPath=gripper_usd_path,
    primPath="/panda/panda_hand/geometry/panda_hand"
)

# 2. Create tool_center reference point
tool_center = stage.DefinePrim("/rb10_gripper/panda_hand/tool_center", "Xform")

# 3. Create fingers
left_finger = stage.DefinePrim("/rb10_gripper/panda_leftfinger", "Xform")
UsdPhysics.RigidBodyAPI.Apply(left_finger)
left_geo = stage.DefinePrim("/rb10_gripper/panda_leftfinger/geometry", "Xform")
left_mesh = stage.DefinePrim("/rb10_gripper/panda_leftfinger/geometry/panda_leftfinger", "Mesh")
left_mesh.GetReferences().AddReference(
    assetPath=gripper_usd_path,
    primPath="/panda/panda_leftfinger/geometry/panda_leftfinger"
)

right_finger = stage.DefinePrim("/rb10_gripper/panda_rightfinger", "Xform")
UsdPhysics.RigidBodyAPI.Apply(right_finger)
right_geo = stage.DefinePrim("/rb10_gripper/panda_rightfinger/geometry", "Xform")
right_mesh = stage.DefinePrim("/rb10_gripper/panda_rightfinger/geometry/panda_rightfinger", "Mesh")
right_mesh.GetReferences().AddReference(
    assetPath=gripper_usd_path,
    primPath="/panda/panda_rightfinger/geometry/panda_rightfinger"
)

# 4. Create joints folder
joints = stage.DefinePrim("/rb10_gripper/joints", "Scope")

# 5. Add arm joints
joint_list = [
    ("base", "link0", "link1", "Z"),
    ("shoulder", "link1", "link2", "Y"),
    ("elbow", "link2", "link3", "Y"),
    ("wrist1", "link3", "link4", "Y"),
    ("wrist2", "link4", "link5", "Z"),
    ("wrist3", "link5", "link6", "Y"),
]

for name, body0, body1, axis in joint_list:
    joint = UsdPhysics.RevoluteJoint.Define(stage, f"/rb10_gripper/joints/{name}")
    joint.CreateBody0Rel().SetTargets([f"/rb10_gripper/{body0}"])
    joint.CreateBody1Rel().SetTargets([f"/rb10_gripper/{body1}"])
    joint.CreateAxisAttr().Set(axis)
    joint.CreateLowerLimitAttr().Set(-179.9087)
    joint.CreateUpperLimitAttr().Set(179.9087)

# 6. Add gripper joint between link6 and panda_hand
tool_joint = UsdPhysics.FixedJoint.Define(stage, "/rb10_gripper/joints/tool_joint")
tool_joint.CreateBody0Rel().SetTargets(["/rb10_gripper/link6"])
tool_joint.CreateBody1Rel().SetTargets(["/rb10_gripper/panda_hand"])

# 7. Add finger joints - using explicit joint names matching what your code expects
finger_joint1 = UsdPhysics.PrismaticJoint.Define(stage, "/rb10_gripper/joints/panda_finger_joint1")
finger_joint1.CreateBody0Rel().SetTargets(["/rb10_gripper/panda_hand"])
finger_joint1.CreateBody1Rel().SetTargets(["/rb10_gripper/panda_leftfinger"])
finger_joint1.CreateAxisAttr().Set("Y")
finger_joint1.CreateLowerLimitAttr().Set(0.0)
finger_joint1.CreateUpperLimitAttr().Set(0.04)

finger_joint2 = UsdPhysics.PrismaticJoint.Define(stage, "/rb10_gripper/joints/panda_finger_joint2")
finger_joint2.CreateBody0Rel().SetTargets(["/rb10_gripper/panda_hand"])
finger_joint2.CreateBody1Rel().SetTargets(["/rb10_gripper/panda_rightfinger"])
finger_joint2.CreateAxisAttr().Set("Y")
finger_joint2.CreateLowerLimitAttr().Set(0.0)
finger_joint2.CreateUpperLimitAttr().Set(0.04)

# Set correct transforms for each component
def set_transform(prim_path, position=(0, 0, 0), rotation=(1, 0, 0, 0), scale=(1, 1, 1)):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        print(f"Warning: Prim not found at {prim_path}")
        return
        
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    
    if position != (0, 0, 0):
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*position))
        
    if rotation != (1, 0, 0, 0):
        rotate_op = xformable.AddOrientOp()
        rotate_op.Set(Gf.Quatf(*rotation))
        
    if scale != (1, 1, 1):
        scale_op = xformable.AddScaleOp()
        scale_op.Set(Gf.Vec3f(*scale))

# Position the gripper relative to link6
set_transform("/rb10_gripper/panda_hand", position=(0, -0.1153, 0), rotation=(0.7071, 0.7071, 0, 0))
set_transform("/rb10_gripper/panda_leftfinger", position=(0, 0.04, 0.065))
set_transform("/rb10_gripper/panda_rightfinger", position=(0, -0.04, 0.065), rotation=(0, 0, 1, 0))

# Save the combined USD file
stage.Save()
print(f"Created combined USD file: {combined_usd_path}")