# isaaclab_assets/robots/rb.py

import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
# from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR # 현재 사용 안 함

RB10_USD_PATH = r"C:\Isaaclab\source\isaaclab_assets\isaaclab_assets\robots\rb10_1300e_nozzle\urdf\rb10_1300e_nozzle\rb10_1300e_nozzle.usd"

# USD 파일의 실제 관절 이름 (USD 파일 내 /rb10_1300e_with_nozzle/joints/ 하위의 조인트 프리미티브 이름)
joint_names_for_rb10_init = [
    "base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"
]

# --- RB10 초기 자세 설정 (도 단위 입력) ---
rb10_initial_joint_angles_deg = [
    90.0,   # Base joint
    45.0, # Shoulder joint (예시: 값을 약간 변경해 봄)
    -90.0,  # Elbow joint  (예시: 값을 약간 변경해 봄)
    0.0,   # Wrist 1 joint
    0.0, # Wrist 2 joint (예시: 값을 약간 변경해 봄)
    0.0    # Wrist 3 joint
]
rb10_initial_joint_angles_rad = [math.radians(deg) for deg in rb10_initial_joint_angles_deg]

initial_joint_positions_for_cfg = {
    name: rad_val for name, rad_val in zip(joint_names_for_rb10_init, rb10_initial_joint_angles_rad)
}
# 또는 리스트로 전달하려면 액추에이터의 joint_names_expr 순서와 일치해야 함
# initial_joint_positions_list_for_cfg = rb10_initial_joint_angles_rad

RB10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=RB10_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=initial_joint_positions_for_cfg, # 수정된 초기 자세 사용
        # joint_pos=initial_joint_positions_list_for_cfg, # 또는 리스트 사용
    ),
    actuators={
        "rb10_arm": ImplicitActuatorCfg(
            joint_names_expr=joint_names_for_rb10_init, # init_state와 일관성 있는 이름 사용
            effort_limit=200.0,
            velocity_limit=2.0,
            stiffness=500.0,
            damping=100.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of RB10-1300E robot."""

# IK 제어를 위한 고강성 버전
RB10_HIGH_PD_CFG = RB10_CFG.copy() # RB10_CFG를 복사하므로 init_state.joint_pos도 복사됨
RB10_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
RB10_HIGH_PD_CFG.actuators["rb10_arm"].stiffness = 1000.0
RB10_HIGH_PD_CFG.actuators["rb10_arm"].damping = 200.0
"""Configuration of RB10-1300E robot with stiffer PD control."""

# # rb.py

# import math # math 모듈 임포트
# import isaaclab.sim as sim_utils
# from isaaclab.actuators import ImplicitActuatorCfg
# from isaaclab.assets.articulation import ArticulationCfg
# from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# ##
# # RB10 Configuration (주석 수정: Panda Gripper 내용은 현재 설정에 없음)
# ##

# RB10_USD_PATH = "/home/hys/Downloads/rb10_1300e_nozzle/urdf/rb10_1300e_nozzle/rb10_1300e_nozzle.usd"

# # RB10의 관절 이름 (USD 파일의 실제 경로와 이름에 맞게 확인 필요)
# # init_state.joint_pos에 사용될 키값들입니다.
# # 아래 예시는 USD 파일 구조 분석 결과에서 추정된 이름과 유사하게 작성했습니다.
# # 실제 USD 파일의 /rb10_1300e_with_nozzle/joints/ 하위의 조인트 프리미티브 이름을 사용해야 합니다.
# joint_names_for_init = [
#     "base",     # 예: /rb10_1300e_with_nozzle/joints/base 의 마지막 부분
#     "shoulder",
#     "elbow",
#     "wrist1",
#     "wrist2",
#     "wrist3"
# ]

# # --- RB10 초기 자세 설정 (도 단위 입력 후 라디안 변환) ---
# rb10_initial_joint_angles_deg = [
#     180.0,   # Base joint
#     45.0, # Shoulder joint
#     -30.0,  # Elbow joint
#     90.0, # Wrist 1 joint
#     180.0,   # Wrist 2 joint
#     180.0    # Wrist 3 joint
# ]
# rb10_initial_joint_angles_rad = [math.radians(deg) for deg in rb10_initial_joint_angles_deg]

# # joint_pos 딕셔너리 생성
# # ArticulationCfg.InitialStateCfg의 joint_pos는 딕셔너리 (관절 이름: 값) 또는
# # 정렬된 값의 리스트/튜플 형태를 받을 수 있습니다.
# # 여기서는 딕셔너리 형태로 명확하게 지정합니다.
# initial_joint_positions = {
#     joint_names_for_init[0]: rb10_initial_joint_angles_rad[0],
#     joint_names_for_init[1]: rb10_initial_joint_angles_rad[1],
#     joint_names_for_init[2]: rb10_initial_joint_angles_rad[2],
#     joint_names_for_init[3]: rb10_initial_joint_angles_rad[3],
#     joint_names_for_init[4]: rb10_initial_joint_angles_rad[4],
#     joint_names_for_init[5]: rb10_initial_joint_angles_rad[5],
# }


# RB10_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=RB10_USD_PATH,
#         activate_contact_sensors=False,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             max_depenetration_velocity=5.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True,
#             solver_position_iteration_count=8,
#             solver_velocity_iteration_count=0,
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         # 초기 관절 자세를 위에서 정의한 딕셔너리로 설정
#         joint_pos=initial_joint_positions,
#     ),
#     actuators={
#         "rb10_arm": ImplicitActuatorCfg(
#             # USD 파일의 실제 관절 이름 또는 이를 포함하는 정규 표현식 사용
#             # 이 이름들은 init_state.joint_pos의 키와 일치해야 합니다 (딕셔너리 사용 시).
#             joint_names_expr=["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"],
#             effort_limit=200.0, # RB10 사양에 맞는 값으로 검토/수정 필요
#             velocity_limit=2.0, # RB10 사양에 맞는 값으로 검토/수정 필요
#             stiffness=500.0,    # RB10 사양에 맞는 값으로 검토/수정 필요
#             damping=100.0,      # RB10 사양에 맞는 값으로 검토/수정 필요
#         ),
#     },
#     soft_joint_pos_limit_factor=1.0,
# )
# """Configuration of RB10-1300E robot.""" # 주석 수정

# # IK 제어를 위한 고강성 버전
# RB10_HIGH_PD_CFG = RB10_CFG.copy()
# RB10_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
# RB10_HIGH_PD_CFG.actuators["rb10_arm"].stiffness = 1000.0 # 예시 값, 튜닝 필요
# RB10_HIGH_PD_CFG.actuators["rb10_arm"].damping = 200.0   # 예시 값, 튜닝 필요
# """Configuration of RB10-1300E robot with stiffer PD control."""

# # (이전 주석 처리된 RB10_1300E_CFG, PANDA_GRIPPER_CFG 등은 필요 없으므로 유지 또는 정리)