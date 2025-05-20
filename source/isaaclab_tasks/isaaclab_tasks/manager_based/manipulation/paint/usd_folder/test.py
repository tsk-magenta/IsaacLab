from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf
import os

# 1. 원본 파일 경로 설정
robot_usd_path = "/home/hys/Downloads/rb10_1300e/rb10_1300e/rb10_1300e/rb10_1300e.usd"
gripper_usd_path = "/home/hys/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/Franka/franka.usd"  # 원본 Franka USD 사용

# 합쳐진 파일의 저장 경로
combined_usd_path = "/home/hys/Downloads/rb10_with_panda_gripper.usd"

# 기존 합쳐진 파일이 있으면 삭제
if os.path.exists(combined_usd_path):
    print(f"기존 합쳐진 USD 파일 삭제 중: {combined_usd_path}")
    os.remove(combined_usd_path)

# 2. 로봇 USD 스테이지 로드
robot_stage = Usd.Stage.Open(robot_usd_path)
if not robot_stage:
    print(f"오류: 로봇 USD 파일 '{robot_usd_path}'을(를) 열 수 없습니다.")
    exit()

# 3. 새로운 스테이지 생성 (합쳐진 로봇 시스템용)
combined_stage = Usd.Stage.CreateNew(combined_usd_path)

# 로봇 USD의 루트 Prim 경로 확인
robot_root_prim_path = Sdf.Path("/rb10_1300e")  # RB10 USD 파일의 실제 루트 Prim 경로 확인 필요!
robot_root_prim = robot_stage.GetPrimAtPath(robot_root_prim_path)

if not robot_root_prim:
    print(f"오류: 로봇 USD 파일 '{robot_usd_path}'에서 루트 Prim '{robot_root_prim_path}'을(를) 찾을 수 없습니다.")
    exit()

# Prim 타입이 xformable인지 확인하는 함수
def is_xformable(prim):
    """Prim이 Xformable 타입인지 확인합니다"""
    return prim.IsA(UsdGeom.Xformable)

# 물리 속성 적용 함수
def apply_physics_properties(prim, is_rigid_body=True):
    """Prim에 물리 속성을 적용합니다."""
    if is_xformable(prim) and (prim.GetTypeName() == "Xform" or prim.GetTypeName() == "Mesh"):
        if is_rigid_body and UsdPhysics.RigidBodyAPI.CanApply(prim):
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
            prim.CreateAttribute("physics:enabled", Sdf.ValueTypeNames.Bool).Set(True)
            
            if prim.GetTypeName() == "Mesh":
                collision_api = UsdPhysics.CollisionAPI.Apply(prim)
                prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set("convexHull")
                
            if not prim.HasAttribute("physics:mass"):
                prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Double).Set(0.1)

# 변환 설정 함수
def set_transform_relative_to_parent(prim, translation=(0,0,0), rotation=(0,0,0), scale=(1,1,1)):
    """지정된 Prim에 부모 기준 변환 설정"""
    if is_xformable(prim):
        xformable = UsdGeom.Xformable(prim)
        
        # 기존 변환 초기화
        xformable.ClearXformOpOrder()
        
        # 새로운 변환 적용 (부모 기준)
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*translation))
        
        rotate_op = xformable.AddRotateXYZOp()
        rotate_op.Set(Gf.Vec3f(*rotation))
        
        scale_op = xformable.AddScaleOp()
        scale_op.Set(Gf.Vec3f(*scale))

# 선택적 복사 함수 - root_joint를 제외하고 복사
def copy_robot_without_root_joint(source_prim, target_stage):
    """
    로봇 Prim을 root_joint를 제외하고 복사합니다.
    """
    # 로봇 루트 Prim 생성
    robot_root = target_stage.DefinePrim(source_prim.GetPath(), source_prim.GetTypeName())
    
    # ArticulationRootAPI 적용
    UsdPhysics.ArticulationRootAPI.Apply(robot_root)
    
    # Prim 복사 시 root_joint 제외
    for child in source_prim.GetChildren():
        if child.GetName() != "root_joint":  # root_joint 제외
            # 자식 Prim 생성 및 속성 복사
            child_path = robot_root.GetPath().AppendChild(child.GetName())
            child_target = target_stage.DefinePrim(child_path, child.GetTypeName())
            
            # 속성 및 관계 복사
            for prop in child.GetProperties():
                if isinstance(prop, Usd.Attribute) and prop.HasValue():
                    try:
                        value = prop.Get()
                        if value is not None:
                            attr = child_target.CreateAttribute(prop.GetName(), prop.GetTypeName())
                            attr.Set(value)
                    except Exception as e:
                        print(f"속성 복사 오류: {e}")
                elif isinstance(prop, Usd.Relationship):
                    try:
                        targets = prop.GetTargets()
                        if targets:
                            rel = child_target.CreateRelationship(prop.GetName())
                            rel.SetTargets(targets)
                    except Exception as e:
                        print(f"관계 복사 오류: {e}")
            
            # Xformable 속성 복사
            if is_xformable(child) and is_xformable(child_target):
                source_xform = UsdGeom.Xformable(child)
                target_xform = UsdGeom.Xformable(child_target)
                
                # 변환 연산 복사
                for op in source_xform.GetOrderedXformOps():
                    if op.GetName() == "xformOp:translate":
                        target_op = target_xform.AddTranslateOp()
                        if op.GetAttr().HasValue():
                            target_op.Set(op.Get())
                    elif op.GetName() == "xformOp:rotateXYZ":
                        target_op = target_xform.AddRotateXYZOp()
                        if op.GetAttr().HasValue():
                            target_op.Set(op.Get())
                    elif op.GetName() == "xformOp:scale":
                        target_op = target_xform.AddScaleOp()
                        if op.GetAttr().HasValue():
                            target_op.Set(op.Get())
            
            # 재귀적으로 자식들 복사
            for grandchild in child.GetChildren():
                # 재귀적으로 각 자식 복사
                copy_children_recursively(grandchild, child_target, target_stage)
    
    return robot_root

# 자식 Prim 재귀적 복사 함수
def copy_children_recursively(source_prim, target_parent, target_stage):
    """
    Prim의 자식들을 재귀적으로 복사합니다.
    """
    child_path = target_parent.GetPath().AppendChild(source_prim.GetName())
    child_target = target_stage.DefinePrim(child_path, source_prim.GetTypeName())
    
    # 속성 및 관계 복사
    for prop in source_prim.GetProperties():
        if isinstance(prop, Usd.Attribute) and prop.HasValue():
            try:
                value = prop.Get()
                if value is not None:
                    attr = child_target.CreateAttribute(prop.GetName(), prop.GetTypeName())
                    attr.Set(value)
            except Exception as e:
                print(f"속성 복사 오류: {e}")
        elif isinstance(prop, Usd.Relationship):
            try:
                targets = prop.GetTargets()
                if targets:
                    rel = child_target.CreateRelationship(prop.GetName())
                    rel.SetTargets(targets)
            except Exception as e:
                print(f"관계 복사 오류: {e}")
    
    # Xformable 속성 복사
    if is_xformable(source_prim) and is_xformable(child_target):
        source_xform = UsdGeom.Xformable(source_prim)
        target_xform = UsdGeom.Xformable(child_target)
        
        # 변환 연산 복사
        for op in source_xform.GetOrderedXformOps():
            if op.GetName() == "xformOp:translate":
                target_op = target_xform.AddTranslateOp()
                if op.GetAttr().HasValue():
                    target_op.Set(op.Get())
            elif op.GetName() == "xformOp:rotateXYZ":
                target_op = target_xform.AddRotateXYZOp()
                if op.GetAttr().HasValue():
                    target_op.Set(op.Get())
            elif op.GetName() == "xformOp:scale":
                target_op = target_xform.AddScaleOp()
                if op.GetAttr().HasValue():
                    target_op.Set(op.Get())
    
    # 자식들 복사
    for child in source_prim.GetChildren():
        copy_children_recursively(child, child_target, target_stage)

# 보다 효율적인 접근 방법 - 참조로 로봇을 복사하되, 루트 조인트를 제외
def generate_combined_usd():
    """
    RB10 로봇과 Franka 그리퍼를 결합한 USD 파일을 생성합니다.
    """
    print("로봇 Prim 복사 중...")
    # 로봇 루트 생성 (root_joint 제외)
    robot_root = copy_robot_without_root_joint(robot_root_prim, combined_stage)
    combined_stage.SetDefaultPrim(robot_root)
    
    # 4. 그리퍼 USD 스테이지 로드
    gripper_stage = Usd.Stage.Open(gripper_usd_path)
    if not gripper_stage:
        print(f"오류: 그리퍼 USD 파일 '{gripper_usd_path}'을(를) 열 수 없습니다.")
        return False
    
    # RB10의 마지막 링크 (엔드 이펙터) 경로
    rb10_ee_link_path = robot_root.GetPath().AppendChild("link6")
    rb10_ee_link_prim = combined_stage.GetPrimAtPath(rb10_ee_link_path)
    
    if not rb10_ee_link_prim:
        print(f"오류: 합쳐진 스테이지에서 RB10 엔드 이펙터 링크 '{rb10_ee_link_path}'를 찾을 수 없습니다.")
        return False
    
    print("그리퍼 Prim들을 로봇 엔드 이펙터 아래로 복사 중...")
    
    # 그리퍼 핸드 생성
    gripper_hand_path = rb10_ee_link_path.AppendChild("panda_hand")
    gripper_hand = combined_stage.DefinePrim(gripper_hand_path, "Xform")
    
    # 그리퍼 핸드 위치 설정 - RB10의 TCP 위치에 맞춤
    set_transform_relative_to_parent(
        gripper_hand, 
        translation=(0, -0.1153, 0),  # TCP 위치에 맞춤
        rotation=(90, 0, 0),         # X축 방향으로 90도 회전
        scale=(1, 1, 1)               # 크기 유지
    )
    
    # 물리 속성 적용
    apply_physics_properties(gripper_hand)
    
    # 그리퍼 핸드 지오메트리 복사
    hand_geo_path = gripper_hand_path.AppendChild("geometry")
    hand_geo = combined_stage.DefinePrim(hand_geo_path, "Xform")
    
    # 그리퍼 핸드 메시 참조 (geometry 폴더 내부)
    hand_mesh_path = hand_geo_path.AppendChild("panda_hand")
    hand_mesh = combined_stage.DefinePrim(hand_mesh_path, "Mesh")
    # 참조 추가 (메시 데이터와 재질 유지)
    hand_mesh.GetReferences().AddReference(
        assetPath=gripper_stage.GetRootLayer().identifier,
        primPath="/panda/panda_hand/geometry/panda_hand"
    )
    
    # 툴 센터 추가 (센서와 eef 참조용) - rigid body로 설정
    tool_center_path = gripper_hand_path.AppendChild("tool_center")
    tool_center = combined_stage.DefinePrim(tool_center_path, "Xform")
    # 툴 센터에 물리 속성 추가하여 rigid body로 만듦
    apply_physics_properties(tool_center)
    
    # 왼쪽 손가락 생성
    left_finger_path = gripper_hand_path.AppendChild("panda_leftfinger")
    left_finger = combined_stage.DefinePrim(left_finger_path, "Xform")
    
    # 왼쪽 손가락 위치 설정
    set_transform_relative_to_parent(
        left_finger, 
        translation=(0, 0.04, 0.065),   # 그리퍼 핸드에서 적절한 위치
        rotation=(0, 0, 0),
        scale=(1, 1, 1)
    )
    
    # 물리 속성 적용
    apply_physics_properties(left_finger)
    
    # 왼쪽 손가락 지오메트리 폴더
    left_geo_path = left_finger_path.AppendChild("geometry")
    left_geo = combined_stage.DefinePrim(left_geo_path, "Xform")
    
    # 왼쪽 손가락 메시 참조
    left_mesh_path = left_geo_path.AppendChild("panda_leftfinger")
    left_mesh = combined_stage.DefinePrim(left_mesh_path, "Mesh")
    left_mesh.GetReferences().AddReference(
        assetPath=gripper_stage.GetRootLayer().identifier,
        primPath="/panda/panda_leftfinger/geometry/panda_leftfinger"
    )
    
    # 오른쪽 손가락 생성
    right_finger_path = gripper_hand_path.AppendChild("panda_rightfinger")
    right_finger = combined_stage.DefinePrim(right_finger_path, "Xform")
    
    # 오른쪽 손가락 위치 설정 - Z축으로 180도 회전하여 대칭 배치
    set_transform_relative_to_parent(
        right_finger, 
        translation=(0, -0.04, 0.065),  # 그리퍼 핸드에서 적절한 위치
        rotation=(0, 0, 180),         # Z축 기준 180도 회전 - 왼쪽 손가락과 마주보도록
        scale=(1, 1, 1)
    )
    
    # 물리 속성 적용
    apply_physics_properties(right_finger)
    
    # 오른쪽 손가락 지오메트리 폴더
    right_geo_path = right_finger_path.AppendChild("geometry")
    right_geo = combined_stage.DefinePrim(right_geo_path, "Xform")
    
    # 오른쪽 손가락 메시 참조
    right_mesh_path = right_geo_path.AppendChild("panda_rightfinger")
    right_mesh = combined_stage.DefinePrim(right_mesh_path, "Mesh")
    right_mesh.GetReferences().AddReference(
        assetPath=gripper_stage.GetRootLayer().identifier,
        primPath="/panda/panda_rightfinger/geometry/panda_rightfinger"
    )
    
    # 그리퍼 조인트 생성
    print("그리퍼 내부 조인트 생성 중...")
    
    # 첫 번째 손가락 조인트
    finger_joint1 = UsdPhysics.PrismaticJoint.Define(combined_stage, gripper_hand_path.AppendChild("panda_finger_joint1"))
    finger_joint1.CreateBody0Rel().SetTargets([gripper_hand_path])
    finger_joint1.CreateBody1Rel().SetTargets([left_finger_path])
    finger_joint1.CreateAxisAttr().Set("Y")
    finger_joint1.CreateLowerLimitAttr().Set(0.0)
    finger_joint1.CreateUpperLimitAttr().Set(0.04)
    
    # 드라이브 속성 설정
    drive_api1 = UsdPhysics.DriveAPI.Apply(finger_joint1.GetPrim(), "linear")
    drive_api1.CreateTypeAttr().Set("force")
    drive_api1.CreateMaxForceAttr().Set(200.0)
    drive_api1.CreateTargetPositionAttr().Set(0.04)
    drive_api1.CreateDampingAttr().Set(100.0)
    drive_api1.CreateStiffnessAttr().Set(2000.0)
    
    # 두 번째 손가락 조인트
    finger_joint2 = UsdPhysics.PrismaticJoint.Define(combined_stage, gripper_hand_path.AppendChild("panda_finger_joint2"))
    finger_joint2.CreateBody0Rel().SetTargets([gripper_hand_path])
    finger_joint2.CreateBody1Rel().SetTargets([right_finger_path])
    finger_joint2.CreateAxisAttr().Set("Y")
    finger_joint2.CreateLowerLimitAttr().Set(0.0)
    finger_joint2.CreateUpperLimitAttr().Set(0.04)
    
    # 드라이브 속성 설정
    drive_api2 = UsdPhysics.DriveAPI.Apply(finger_joint2.GetPrim(), "linear")
    drive_api2.CreateTypeAttr().Set("force")
    drive_api2.CreateMaxForceAttr().Set(200.0)
    drive_api2.CreateTargetPositionAttr().Set(0.04)
    drive_api2.CreateDampingAttr().Set(100.0)
    drive_api2.CreateStiffnessAttr().Set(2000.0)
    
    # 합쳐진 USD 파일 저장
    combined_stage.Save()
    print(f"합쳐진 USD 파일 생성 완료: {combined_usd_path}")
    
    return True

# 메인 실행 코드
generate_combined_usd()