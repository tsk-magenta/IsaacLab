# paint/mdp/dummy_action.py
from __future__ import annotations

import gymnasium.spaces as spaces
import torch
from isaaclab.envs.mdp.actions import ActionTermCfg, ActionTerm
from isaaclab.utils import configclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class DummyActionCfg(ActionTermCfg):
    """Configuration for a dummy action that performs no operation."""
    class_type: type[ActionTerm] = lambda cfg, env: DummyAction(cfg, env)
    asset_name: str = "robot" # 어떤 에셋에든 연결될 수 있지만, 실제 사용되지 않음


class DummyAction(ActionTerm):
    cfg: DummyActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: DummyActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._raw_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1 # 1차원 입력을 받습니다.

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_action

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, action: torch.Tensor):
        # 액션을 받지만 아무것도 처리하지 않습니다.
        self._raw_action[:] = action
        self._processed_actions[:] = 0.0 # 항상 0으로 유지하거나, 그냥 무시합니다.

    def get_action_space(self) -> spaces.Box:
        # record_demos.py가 예상하는 [-1.0, 1.0] 범위의 1차원 공간을 반환합니다.
        return spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=torch.float32)

    def apply_actions(self):
        # 액션을 적용하지만 아무것도 하지 않습니다.
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # 리셋 시에도 아무것도 하지 않습니다.
        if env_ids is None:
            self._raw_action.zero_()
            self._processed_actions.zero_()
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
            else:
                env_ids_tensor = env_ids
            if env_ids_tensor.numel() > 0:
                 self._raw_action[env_ids_tensor] = 0.0
                 self._processed_actions[env_ids_tensor] = 0.0