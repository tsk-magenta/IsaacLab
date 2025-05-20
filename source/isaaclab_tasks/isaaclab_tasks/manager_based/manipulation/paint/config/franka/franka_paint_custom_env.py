# franka_stack_custom_env.py 파일 내용


import torch
import omni.log # omni.log 임포트 추가 (사용하고 있다면)
from isaaclab.envs import ManagerBasedRLEnv
# ManagerBasedEnvCfg 대신, 더 구체적인 PaintEnvCfg를 임포트합니다.
# franka_paint_custom_env.py 파일은 paint_env_cfg.py 파일과 다른 디렉토리에 있으므로,
# 상대 경로 또는 절대 경로를 사용하여 임포트합니다.
# 현재 파일 위치: isaaclab_tasks/manager_based/manipulation/paint/config/franka/
# PaintEnvCfg 위치: isaaclab_tasks/manager_based/manipulation/paint/
from isaaclab_tasks.manager_based.manipulation.paint.paint_env_cfg import PaintEnvCfg

class FrankaPaintCustomEnv(ManagerBasedRLEnv):
    """Custom environment class for Franka Paint task to store state."""

    # 타입 힌트를 PaintEnvCfg로 명시
    cfg: PaintEnvCfg

    def __init__(self, cfg: PaintEnvCfg, **kwargs):
        # 속성을 초기화합니다
        self.all_target_local_positions = []
        
        # 부모 클래스 초기화
        super().__init__(cfg=cfg, **kwargs)
        
        # 이제 디바이스와 환경 개수에 접근할 수 있으므로 current_target_idx를 초기화합니다
        self.current_target_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # TARGET_LOCATIONS에서 타겟 위치 가져오기
        target_positions_loaded = False

        
        if hasattr(self.cfg, "TARGET_LOCATIONS"):
            target_locations = self.cfg.TARGET_LOCATIONS
            for key in ["target1", "target2"]:
                if key in target_locations:
                    self.all_target_local_positions.append(
                        torch.tensor(target_locations[key], device=self.device, dtype=torch.float32)
                    )
                    target_positions_loaded = True
                else:
                    omni.log.warn(f"Target key '{key}' not found in cfg.TARGET_LOCATIONS.")
        
        # 서브태스크 완료 상태 추적을 위한 변수 추가
        self.completed_subtasks = {}
        # 서브태스크 이름 목록 (순서 중요)
        self.subtask_names = ["approach_1", "approach_2"]  # 실제 서브태스크 이름으로 수정
        # 초기 상태: 모두 미완료
        for subtask in self.subtask_names:
            self.completed_subtasks[subtask] = False

            
        # 디버깅을 위한 로그 출력
        omni.log.info(f"Initialized {len(self.all_target_local_positions)} target positions: {self.all_target_local_positions}")

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if hasattr(self, "current_target_idx"):
            self.current_target_idx[env_ids] = 0
        for env_id in env_ids:
            for subtask in self.subtask_names:
                self.completed_subtasks[subtask] = False
        # ... (다른 상태 변수 리셋) ...
    def update_subtask_completion(self, subtask_states):
        """
        서브태스크 완료 상태를 업데이트합니다.
        
        Args:
            subtask_states: 각 서브태스크의 현재 상태를 담은 딕셔너리
        """
        for i, subtask in enumerate(self.subtask_names):
            if subtask in subtask_states:
                current_state = subtask_states[subtask]
                
                # 이미 완료된 서브태스크는 완료 상태 유지
                if self.completed_subtasks.get(subtask, False):
                    continue
                    
                # 이전 서브태스크가 모두 완료되었는지 확인
                all_previous_completed = True
                for prev_idx in range(i):
                    prev_subtask = self.subtask_names[prev_idx]
                    if not self.completed_subtasks.get(prev_subtask, False):
                        all_previous_completed = False
                        break
                
                # 현재 상태가 True이고 이전 서브태스크가 모두 완료되었다면 이 서브태스크 완료 처리
                if current_state and all_previous_completed:
                    self.completed_subtasks[subtask] = True
                    print(f"🎉 Subtask '{subtask}' completed!")
    def get_current_target_local_pos(self, env_ids_query: torch.Tensor | None = None) -> torch.Tensor:
        # 타겟 위치가 비어 있는 경우 기본값 설정
        if not hasattr(self, "all_target_local_positions") or not self.all_target_local_positions:
            # 기본값 설정
            default_scale = 5.0
            self.all_target_local_positions = [
                torch.tensor([-0.07 * default_scale, 0.0 * default_scale, 0.09 * default_scale], device=self.device, dtype=torch.float32),
                torch.tensor([0.07 * default_scale, 0.0 * default_scale, 0.09 * default_scale], device=self.device, dtype=torch.float32)
            ]
            omni.log.warn("Using default target positions in get_current_target_local_pos.")
        
        # 인덱스 확인
        if not hasattr(self, "current_target_idx"):
            self.current_target_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        if env_ids_query is None:
            indices_to_use = self.current_target_idx
        else:
            indices_to_use = self.current_target_idx[env_ids_query]
        
        # 빈 환경 처리
        if indices_to_use.numel() == 0:
            return torch.empty((0, 3), device=self.device, dtype=torch.float32)
        
        # 타겟 선택
        selected_targets_list = []
        for idx_val_tensor in indices_to_use:
            idx_val = idx_val_tensor.item()
            if 0 <= idx_val < len(self.all_target_local_positions):
                selected_targets_list.append(self.all_target_local_positions[idx_val])
            else:
                omni.log.error(f"Invalid target index: {idx_val} for all_target_local_positions of size {len(self.all_target_local_positions)}")
                selected_targets_list.append(self.all_target_local_positions[0])  # 첫 번째 타겟 사용
        
        # 결과 반환
        return torch.stack(selected_targets_list)

    def advance_to_next_subtask(self, env_ids_to_advance: torch.Tensor):
        if env_ids_to_advance.numel() == 0 or not self.all_target_local_positions:
            return

        current_indices = self.current_target_idx[env_ids_to_advance]
        next_indices = current_indices + 1
        
        num_total_targets = len(self.all_target_local_positions)
        # 다음 인덱스가 총 타겟 수를 넘지 않도록 클램핑
        effective_next_indices = torch.clamp(next_indices, max=num_total_targets - 1)
        
        self.current_target_idx[env_ids_to_advance] = effective_next_indices
        
        # 실제로 인덱스가 변경된 환경만 로그 출력
        advanced_mask = (effective_next_indices > current_indices) & (current_indices < num_total_targets - 1)
        if torch.any(advanced_mask):
            actual_advanced_env_ids = env_ids_to_advance[advanced_mask]
            new_indices_for_log = self.current_target_idx[actual_advanced_env_ids]
            omni.log.info(f"Envs {actual_advanced_env_ids.tolist()} advanced to next subtask index: {new_indices_for_log.tolist()}")

    # _compute_rewards, _check_termination 등의 메서드에서
    # self.get_current_target_local_pos() 와 self.advance_to_next_subtask()를 활용

# class FrankaPaintCustomEnv(ManagerBasedRLEnv):
#     """Custom environment class for Franka Paint task to store state."""

#     # 타입을 명확히 하기 위해 cfg 타입을 지정해줄 수 있습니다 (선택 사항)
#     cfg: ManagerBasedEnvCfg

#     def __init__(self, cfg: ManagerBasedEnvCfg, **kwargs):
#         """Initialize the environment and the counter."""
#         # 부모 클래스 초기화 (필수)
#         super().__init__(cfg=cfg, **kwargs)

#         # --- 여기에 카운터 변수 초기화 ---
#         # 디바이스와 환경 개수는 부모 클래스 초기화 후 접근 가능
#         self.eef_near_target_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
#         print(f"Initialized eef_near_target_counter on device {self.device} with shape {self.eef_near_target_counter.shape}")

#     def _reset_idx(self, env_ids: torch.Tensor):
#         """Reset the specified environment indices."""
#         # 부모 클래스의 리셋 로직 호출 (필수)
#         super()._reset_idx(env_ids)

#         # --- 해당 환경들의 카운터 리셋 ---
#         if hasattr(self, "eef_near_target_counter"):
#             self.eef_near_target_counter[env_ids] = 0
#         else:
#              # 혹시 모르니 초기화 안됐으면 경고 (디버깅용)
#              print("[Warning] Attempted to reset eef_near_target_counter before initialization.")

#     # 필요하다면 여기에 다른 커스텀 메소드 추가 가능
#     # 예: _step(actions) 메소드를 오버라이드하여 매 스텝 특정 로직 수행
#     # def _step(self, actions: torch.Tensor) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Dict[str, Any]]:
#     #     # ... pre-step logic ...
#     #     obs, rewards, terminated, truncated, info = super()._step(actions)
#     #     # ... post-step logic ...
#     #     return obs, rewards, terminated, truncated, info