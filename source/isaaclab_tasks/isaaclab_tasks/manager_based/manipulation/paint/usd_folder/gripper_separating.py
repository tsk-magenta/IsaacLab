from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf # Tf는 이제 필요 없을 수 있습니다.
#######################################################################
# franka.usd의 gripper만 분리해서 저장하는 스크립트
# 2025.05.16 작성
#
#######################################################################

source_path = "/home/hys/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/Franka/franka.usd" 
target_path = "/home/hys/Downloads/panda_gripper_only.usd" # 필요시 다른 이름으로 변경

# 원본 USD 파일 로드
source_stage = Usd.Stage.Open(source_path)

# 새 USD 파일 생성
target_stage = Usd.Stage.CreateNew(target_path)

# 루트 프림 생성
root_prim = target_stage.DefinePrim("/panda_gripper", "Xform")
target_stage.SetDefaultPrim(root_prim)

# 그리퍼 부분만 복사
def copy_prim_hierarchy(source_stage, source_path, target_stage, target_path, include_properties=True):
    source_prim = source_stage.GetPrimAtPath(source_path)
    if not source_prim:
        print(f"Source prim not found: {source_path}")
        return None
    
    # 프림 타입 가져오기
    prim_type = source_prim.GetTypeName()
    
    # 대상 프림 생성
    target_prim = target_stage.DefinePrim(target_path, prim_type)
    
    # 속성 복사
    if include_properties:
        for prop in source_prim.GetProperties():
            if isinstance(prop, Usd.Attribute):
                attr = prop
                if attr.HasValue(): 
                    try:
                        value = attr.Get()
                        target_attr = target_prim.CreateAttribute(attr.GetName(), attr.GetTypeName())
                        target_attr.Set(value)
                    except Exception as e:
                        print(f"Could not copy attribute {attr.GetName()}: {e}")
    
    # 자식 프림 복사
    for child in source_prim.GetChildren():
        child_source_path = child.GetPath()
        child_target_path = Sdf.Path(str(target_path) + str(child_source_path).replace(str(source_path), ""))
        copy_prim_hierarchy(source_stage, child_source_path, target_stage, child_target_path)
    
    return target_prim

# 그리퍼 부분만 복사
print("Copying panda_hand...")
hand_prim = copy_prim_hierarchy(source_stage, "/panda/panda_hand", target_stage, "/panda_gripper/panda_hand")

print("Copying panda_leftfinger...")
leftfinger_prim = copy_prim_hierarchy(source_stage, "/panda/panda_leftfinger", target_stage, "/panda_gripper/panda_leftfinger")

print("Copying panda_rightfinger...")
rightfinger_prim = copy_prim_hierarchy(source_stage, "/panda/panda_rightfinger", target_stage, "/panda_gripper/panda_rightfinger")

# 손가락 관절 생성
print("Creating finger joints...")
# 첫 번째 손가락 관절
finger_joint1 = UsdPhysics.PrismaticJoint.Define(target_stage, "/panda_gripper/panda_hand/panda_finger_joint1")
finger_joint1.CreateBody0Rel().SetTargets([Sdf.Path("/panda_gripper/panda_hand")])
finger_joint1.CreateBody1Rel().SetTargets([Sdf.Path("/panda_gripper/panda_leftfinger")])
finger_joint1.CreateAxisAttr().Set("Y") # <--- 이 부분을 "Y" 문자열로 변경

# 관절 한계 설정
finger_joint1.CreateLowerLimitAttr().Set(0.0)
finger_joint1.CreateUpperLimitAttr().Set(0.04)

# 드라이브 속성 설정
drive_api1 = UsdPhysics.DriveAPI.Apply(finger_joint1.GetPrim(), "linear")
drive_api1.CreateTypeAttr().Set("force")
drive_api1.CreateMaxForceAttr().Set(200.0)
drive_api1.CreateTargetPositionAttr().Set(0.04) # 초기 상태 - 열림
drive_api1.CreateDampingAttr().Set(100.0)
drive_api1.CreateStiffnessAttr().Set(2000.0)

# 두 번째 손가락 관절
finger_joint2 = UsdPhysics.PrismaticJoint.Define(target_stage, "/panda_gripper/panda_hand/panda_finger_joint2")
finger_joint2.CreateBody0Rel().SetTargets([Sdf.Path("/panda_gripper/panda_hand")])
finger_joint2.CreateBody1Rel().SetTargets([Sdf.Path("/panda_gripper/panda_rightfinger")])
finger_joint2.CreateAxisAttr().Set("Y") # <--- 이 부분을 "Y" 문자열로 변경

# 관절 한계 설정
finger_joint2.CreateLowerLimitAttr().Set(0.0)
finger_joint2.CreateUpperLimitAttr().Set(0.04)

# 드라이브 속성 설정
drive_api2 = UsdPhysics.DriveAPI.Apply(finger_joint2.GetPrim(), "linear")
drive_api2.CreateTypeAttr().Set("force")
drive_api2.CreateMaxForceAttr().Set(200.0)
drive_api2.CreateTargetPositionAttr().Set(0.04) # 초기 상태 - 열림
drive_api2.CreateDampingAttr().Set(100.0)
drive_api2.CreateStiffnessAttr().Set(2000.0)

# 파일 저장
target_stage.Save()
print(f"Created new USD file: {target_path}")