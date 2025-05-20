# franka_stack_custom_env.py 파일 내용

import torch
import omni.log # omni.log 임포트 추가 (사용하고 있다면)
from isaaclab.envs import ManagerBasedRLEnv
# ManagerBasedEnvCfg 대신, 더 구체적인 PaintEnvCfg를 임포트합니다.
# franka_paint_custom_env.py 파일은 paint_env_cfg.py 파일과 다른 디렉토리에 있으므로,
# 상대 경로 또는 절대 경로를 사용하여 임포트합니다.
# 현재 파일 위치: isaaclab_tasks/manager_based/manipulation/paint/config/franka/
# PaintEnvCfg 위치: isaaclab_tasks/manager_based/manipulation/paint/
from isaaclab_tasks.manager_based.manipulation.paint.paint_env_cfg import PaintEnvCfg, TARGET_LOCATIONS

class FrankaPaintCustomEnv(ManagerBasedRLEnv):
    """Custom environment class for Franka Paint task to store state."""

    # 타입 힌트를 PaintEnvCfg로 명시
    cfg: PaintEnvCfg

    def __init__(self, cfg: PaintEnvCfg, **kwargs): # 생성자 파라미터 타입도 PaintEnvCfg로 변경
        """Initialize the environment and the custom state variables."""
        super().__init__(cfg=cfg, **kwargs)

        # --- 환경 클래스 내부 상태 변수 초기화 ---
        self.all_target_local_positions: list[torch.Tensor] = []

        try:
            for key, pos in TARGET_LOCATIONS.items():
                self.all_target_local_positions.append(
                    torch.tensor(pos, device=self.device, dtype=torch.float32)
                )
            omni.log.info(f"Loaded {len(self.all_target_local_positions)} target positions directly from TARGET_LOCATIONS")
        except Exception as e:
            omni.log.error(f"Error loading targets from TARGET_LOCATIONS: {e}")
            self.all_target_local_positions = []

        if not self.all_target_local_positions and hasattr(self.cfg, "task_internal_state") and \
           hasattr(self.cfg.task_internal_state, "subtask_target_keys") and \
           hasattr(self.cfg, "TARGET_LOCATIONS"): 

            target_keys = self.cfg.task_internal_state.subtask_target_keys
            target_locations_map = self.cfg.TARGET_LOCATIONS 

            for key in target_keys:
                if key in target_locations_map:
                    self.all_target_local_positions.append(
                        torch.tensor(target_locations_map[key], device=self.device, dtype=torch.float32)
                    )
                else:
                    omni.log.error(f"Target key '{key}' not found in cfg.TARGET_LOCATIONS.")

        if not self.all_target_local_positions:
            omni.log.warn(
                "No target positions were loaded. Adding default hardcoded target positions."
            )
            scale = getattr(self.cfg, "scale", 5.0)  # paint_env_cfg.py에 정의된 scale 값 사용
            
            self.all_target_local_positions = [
                torch.tensor([-0.07 * scale, 0.0 * scale, 0.09 * scale], device=self.device, dtype=torch.float32),
                torch.tensor([0.07 * scale, 0.0 * scale, 0.09 * scale], device=self.device, dtype=torch.float32)
            ]
            omni.log.info(f"Added {len(self.all_target_local_positions)} default target positions")

        self.current_target_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        self.eef_near_target_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if hasattr(self, "current_target_idx"):
            self.current_target_idx[env_ids] = 0
        
        # [추가] eef_near_target_counter 리셋
        if hasattr(self, "eef_near_target_counter"):
            self.eef_near_target_counter[env_ids] = 0
        # [추가 끝]

    def get_current_target_local_pos(self, env_ids_query: torch.Tensor | None = None) -> torch.Tensor | None:
        if not self.all_target_local_positions:
            omni.log.warn_once("get_current_target_local_pos: all_target_local_positions is empty.")
            return None

        if env_ids_query is None:
            indices_to_use = self.current_target_idx
        else:
            indices_to_use = self.current_target_idx[env_ids_query]
        
        if indices_to_use.numel() == 0: # 조회할 env_ids가 없는 경우
             return torch.empty((0, self.all_target_local_positions[0].shape[0] if self.all_target_local_positions else 3),
                                device=self.device, dtype=torch.float32)

        # [수정] 인덱스 범위 체크 및 예외 처리 개선
        selected_targets_list = []
        for env_idx, idx_val_tensor in enumerate(indices_to_use):
            idx_val = idx_val_tensor.item()
            if 0 <= idx_val < len(self.all_target_local_positions):
                selected_targets_list.append(self.all_target_local_positions[idx_val])
            else:
                omni.log.error(f"Invalid target index: {idx_val} for all_target_local_positions of size {len(self.all_target_local_positions)}")
                # 오류 발생 시 첫 번째 타겟 사용
                selected_targets_list.append(self.all_target_local_positions[0])
                # 인덱스 수정 (경계 내로 조정)
                self.current_target_idx[env_ids_query[env_idx] if env_ids_query is not None else env_idx] = 0
        # [수정 끝]
        
        if not selected_targets_list:
            return None
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
        advanced_mask = (effective_next_indices > current_indices) & (current_indices < num_total_targets -1) # 마지막 타겟으로 이미 간 경우는 제외
        if torch.any(advanced_mask):
             actual_advanced_env_ids = env_ids_to_advance[advanced_mask]
             new_indices_for_log = self.current_target_idx[actual_advanced_env_ids]
             omni.log.info(f"Envs {actual_advanced_env_ids.tolist()} advanced to next subtask index: {new_indices_for_log.tolist()}")
        
    # [추가] 매 스텝마다 현재 타겟과 상태 체크
    def _step(self, actions: torch.Tensor):
        # 부모 클래스의 step 호출
        obs, rewards, terminated, truncated, info = super()._step(actions)
        
        # 현재 타겟 인덱스와 위치 로깅 (디버깅용, 선택적)
        if self.sim_frame_count % 100 == 0:  # 매 100 프레임마다만 로그 출력 (줄이기 위해)
            sample_env_ids = torch.arange(min(3, self.num_envs), device=self.device)  # 처음 3개 환경만 샘플링
            current_targets = self.get_current_target_local_pos(sample_env_ids)
            if current_targets is not None:
                for i, env_id in enumerate(sample_env_ids):
                    omni.log.info(f"Env {env_id.item()}: Current target idx={self.current_target_idx[env_id].item()}, pos={current_targets[i].tolist()}")
        
        return obs, rewards, terminated, truncated, info
    # [추가 끝]
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