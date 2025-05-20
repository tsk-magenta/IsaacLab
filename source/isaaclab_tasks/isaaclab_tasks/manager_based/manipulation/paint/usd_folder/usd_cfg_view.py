from pxr import Usd, UsdPhysics, UsdShade, Sdf, Tf

def print_usd_structure_with_physics_and_materials(usd_path):
    """
    USD 파일의 Prim 계층 구조, Prim 타입, 물리 속성 및 재질 바인딩 정보를 출력합니다.

    Args:
        usd_path (str): 확인할 USD 파일의 경로.
    """
    stage = Usd.Stage.Open(usd_path)

    if not stage:
        print(f"오류: '{usd_path}' 파일을 열 수 없습니다.")
        return

    print(f"--- USD 파일 구조 분석: {usd_path} ---")

    # 모든 Prim을 순회하며 정보 출력
    for prim in stage.TraverseAll():
        # 경로 문자열에서 '/' 개수를 세어 들여쓰기 계산
        path = prim.GetPath()
        path_str = str(path)
        indent_level = path_str.count('/') 
        indent = "  " * indent_level

        print(f"{indent}Prim Path: {path_str}")
        print(f"{indent}  Type Name: {prim.GetTypeName()}")

        # 물리 속성 확인
        check_physics_properties(stage, prim, indent)
        
        # 재질 바인딩 확인
        check_material_bindings(prim, indent)
        
        # 조인트 속성 확인 (PhysicsJoint 타입이면)
        check_joint_properties(prim, indent)

    print(f"--- 분석 완료 ---")

def check_physics_properties(stage, prim, indent):
    """물리 관련 속성을 확인하고 출력합니다."""
    path = prim.GetPath()
    
    # RigidBodyAPI 확인
    rb = UsdPhysics.RigidBodyAPI(prim)
    if rb:
        print(f"{indent}  Physics: RigidBodyAPI 적용됨")
        
        # RigidBodyAPI 관련 속성 확인
        velocity_attr = rb.GetVelocityAttr()
        if velocity_attr and velocity_attr.IsValid():
            velocity = velocity_attr.Get()
            if velocity is not None:
                print(f"{indent}    Velocity: {velocity}")
        
        angular_vel_attr = rb.GetAngularVelocityAttr()
        if angular_vel_attr and angular_vel_attr.IsValid():
            angular_vel = angular_vel_attr.Get()
            if angular_vel is not None:
                print(f"{indent}    Angular Velocity: {angular_vel}")
        
        kinematic_attr = rb.GetKinematicEnabledAttr()
        if kinematic_attr and kinematic_attr.IsValid():
            kinematic = kinematic_attr.Get()
            print(f"{indent}    Kinematic: {kinematic}")
            
        enabled_attr = prim.GetAttribute("physics:rigidBodyEnabled")
        if enabled_attr and enabled_attr.IsValid():
            enabled = enabled_attr.Get()
            print(f"{indent}    Enabled: {enabled}")
    
    # CollisionAPI 확인
    collision = UsdPhysics.CollisionAPI(prim)
    if collision:
        print(f"{indent}  Physics: CollisionAPI 적용됨")
        
        # CollisionAPI 관련 속성 확인
        enabled_attr = collision.GetCollisionEnabledAttr()
        if enabled_attr and enabled_attr.IsValid():
            enabled = enabled_attr.Get()
            print(f"{indent}    Collision Enabled: {enabled}")
    
    # MassAPI 확인
    mass_api = UsdPhysics.MassAPI(prim)
    if mass_api:
        print(f"{indent}  Physics: MassAPI 적용됨")
        
        # 질량 관련 속성 확인
        mass_attr = mass_api.GetMassAttr()
        if mass_attr and mass_attr.IsValid():
            mass = mass_attr.Get()
            if mass is not None:
                print(f"{indent}    Mass: {mass}")
        
        com_attr = mass_api.GetCenterOfMassAttr()
        if com_attr and com_attr.IsValid():
            com = com_attr.Get()
            if com is not None:
                print(f"{indent}    Center of Mass: {com}")
    
    # ArticulationRootAPI 확인
    articulation = UsdPhysics.ArticulationRootAPI(prim)
    if articulation:
        print(f"{indent}  Physics: ArticulationRootAPI 적용됨")
        
    # XformStack reset 확인
    xform_reset_attr = prim.GetAttribute("physics:useXformStackReset")
    if xform_reset_attr and xform_reset_attr.IsValid():
        use_xformstack_reset = xform_reset_attr.Get()
        if use_xformstack_reset:
            print(f"{indent}  Physics: XformStack Reset 활성화됨")

def check_material_bindings(prim, indent):
    """재질 바인딩 정보를 확인하고 출력합니다."""
    # 재질 바인딩 확인
    material_binding_API = UsdShade.MaterialBindingAPI(prim)
    
    # 직접 바인딩 확인
    direct_binding = material_binding_API.GetDirectBinding()
    material = direct_binding.GetMaterial()
    if material:
        print(f"{indent}  Material Binding: {material.GetPath()}")
        
        # 타겟 경로 확인
        rel = prim.GetRelationship("material:binding")
        if rel:
            targets = rel.GetTargets()
            if targets:
                print(f"{indent}    Targets:")
                for target in targets:
                    print(f"{indent}      {target}")
    
    # 컬렉션 바인딩 확인
    collection_bindings = material_binding_API.GetCollectionBindings()
    if collection_bindings:
        for binding in collection_bindings:
            collection = binding.GetCollection()
            if collection:
                print(f"{indent}  Collection Material: {collection.GetName()}")
                material = binding.GetMaterial()
                if material:
                    print(f"{indent}    Material: {material.GetPath()}")

def check_joint_properties(prim, indent):
    """조인트 관련 속성을 확인하고 출력합니다."""
    # PhysicsJoint 타입 확인
    type_name = prim.GetTypeName()
    if "PhysicsJoint" in str(type_name) or "PhysicsRevoluteJoint" in str(type_name) or "PhysicsPrismaticJoint" in str(type_name):
        print(f"{indent}  Joint Type: {type_name}")
        
        # body0와 body1 관계 확인
        for rel_name in ["physics:body0", "physics:body1"]:
            rel = prim.GetRelationship(rel_name)
            if rel:
                targets = rel.GetTargets()
                if targets:
                    print(f"{indent}    {rel_name}:")
                    for target in targets:
                        print(f"{indent}      {target}")
                else:
                    print(f"{indent}    {rel_name}: 타겟 없음")
        
        # 조인트 축 속성 확인
        axis_attr = prim.GetAttribute("physics:axis")
        if axis_attr and axis_attr.IsValid():
            axis = axis_attr.Get()
            print(f"{indent}    Axis: {axis}")
        
        # 조인트 제한 확인
        for limit_attr in ["physics:lowerLimit", "physics:upperLimit"]:
            attr = prim.GetAttribute(limit_attr)
            if attr and attr.IsValid():
                limit = attr.Get()
                print(f"{indent}    {limit_attr.split(':')[1]}: {limit}")

if __name__ == "__main__":
    # usd_file_to_check = "C:\Users\yuns0\Downloads\rb10_1300e_nozzle\urdf\rb10_1300e_nozzle\rb10_1300e_nozzle.usd"  # 여기에 확인하고 싶은 usd 경로를 입력
    usd_file_to_check = r"/home/hys/Downloads/rb10_1300e_nozzle/urdf/rb10_1300e_nozzle/rb10_1300e_nozzle.usd"
    print_usd_structure_with_physics_and_materials(usd_file_to_check)