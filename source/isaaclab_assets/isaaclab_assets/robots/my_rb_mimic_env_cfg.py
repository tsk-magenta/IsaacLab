# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Configuration for the RB10 Paint Mimic environment.
RB10 페인트 Mimic 환경을 위한 설정 파일입니다.
franka_stack_ik_rel_mimic_env_cfg.py의 구조를 따릅니다.
"""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

# my_block_ik_rel_env_cfg.py에서 기본 환경 설정을 가져옵니다
from isaaclab_tasks.manager_based.manipulation.paint.config.franka.my_block_ik_rel_env_cfg import MyBlockIKRelEnvCfg


@configclass
class MyRbPaintMimicEnvCfg(MyBlockIKRelEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for RB10 Paint env.
    RB10 로봇을 이용한 페인팅 Mimic 환경을 위한 설정 클래스입니다.
    MyBlockIKRelEnvCfg와 MimicEnvCfg를 상속받아 구현합니다.
    """

    def __post_init__(self):
        """
        Initialize configuration after the object is created.
        객체 생성 후 설정을 초기화합니다.
        부모 클래스의 초기화 후 Mimic 관련 설정을 추가적으로 구성합니다.
        """
        # 부모 클래스들의 __post_init__ 호출
        # 이것은 MyBlockIKRelEnvCfg의 초기화를 수행합니다
        super().__post_init__()
        
        # 데이터 생성 관련 설정 오버라이드
        # 데이터셋 이름과 생성 관련 설정을 페인팅 작업에 맞게 조정합니다
        self.datagen_config.name = "demo_rb_paint_mimic_D0"
        self.datagen_config.generation_guarantee = True  # 모든 서브태스크에 대해 데이터 생성 보장
        self.datagen_config.generation_keep_failed = True  # 실패한 데모도 저장
        self.datagen_config.generation_num_trials = 10  # 각 서브태스크 당 시도 횟수
        self.datagen_config.generation_select_src_per_subtask = True  # 각 서브태스크마다 소스 데모 선택
        self.datagen_config.generation_transform_first_robot_pose = False  # 첫 로봇 포즈 변환 비활성화
        self.datagen_config.generation_interpolate_from_last_target_pose = True  # 마지막 타겟 포즈에서 보간 시작
        self.datagen_config.generation_relative = True  # 상대적 위치 사용
        self.datagen_config.max_num_failures = 25  # 최대 실패 허용 횟수
        self.datagen_config.seed = 1  # 랜덤 시드

        # 페인팅 작업을 위한 서브태스크 설정
        # paint_env_cfg.py의 ObservationsCfg.SubtaskCfg에 정의된 서브태스크를 활용합니다
        subtask_configs = []
        
        # 첫 번째 서브태스크: 첫 번째 타겟 지점에 접근
        subtask_configs.append(
            SubTaskConfig(
                # 각 서브태스크는 단일 객체 프레임에 대해 조작을 수행합니다
                object_ref="myblock",  # 페인팅 대상 객체 (paint_env_cfg.py에서 정의)
                
                # "datagen_info"에서 이 서브태스크가 완료되었음을 알리는 바이너리 인디케이터에 해당하는 키
                # paint_env_cfg.py의 ObservationsCfg.SubtaskCfg에 정의된 이름과 일치해야 합니다
                subtask_term_signal="approach_1",
                
                # 궤적을 서브태스크 세그먼트로 분할할 때 데이터 생성을 위한 시간 오프셋 지정
                # 종료 경계에 랜덤 오프셋이 추가됩니다
                subtask_term_offset_range=(10, 20),
                
                # 데이터 생성 중 소스 서브태스크 세그먼트를 선택하는 전략
                selection_strategy="nearest_neighbor_object",
                
                # 선택 전략 함수에 대한 선택적 매개변수
                selection_strategy_kwargs={"nn_k": 3},
                
                # 이 서브태스크 동안 적용할 액션 노이즈의 양
                action_noise=0.03,
                
                # 이 서브태스크 세그먼트로 연결하기 위한 보간 단계 수
                num_interpolation_steps=5,
                
                # 로봇이 필요한 포즈에 도달하기 위한 추가 고정 단계
                num_fixed_steps=0,
                
                # True면 보간 단계와 실행 중에 액션 노이즈 적용
                apply_noise_during_interpolation=False,
                
                # 서브태스크 설명 (선택 사항)
                description="Approach first paint target",
                
                # 다음 서브태스크에 대한 설명 (선택 사항)
                next_subtask_description="Approach second paint target",
            )
        )
        
        # 두 번째 서브태스크: 두 번째 타겟 지점에 접근
        subtask_configs.append(
            SubTaskConfig(
                object_ref="myblock",
                subtask_term_signal="approach_2",
                subtask_term_offset_range=(0,0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Approach second paint target",
                # 마지막 서브태스크이므로 next_subtask_description은 필요 없음
            )
        )
        
        # 로봇 엔드 이펙터에 대한 서브태스크 구성 설정
        # franka_stack_ik_rel_mimic_env_cfg.py에서는 "franka"를 키로 사용하지만,
        # RB10 로봇을 사용하더라도 일관성을 위해 기존 키를 유지하거나 
        # 환경에 맞게 수정할 수 있습니다.
        # 여기서는 일관성을 위해 "franka" 키를 사용합니다.
        self.subtask_configs["franka"] = subtask_configs