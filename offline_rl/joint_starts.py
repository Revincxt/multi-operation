"""用全动作的共同模式起点缓解联合价值搜索中的协调局部最优。"""

import torch

from offline_rl.structured_value import StructuredValueActor


class JointStartActor(StructuredValueActor):
    """增加最佳全城统一动作起点，再执行原有自由逐网格改进。"""

    @torch.no_grad()
    def extra_starts(self, states):
        prepared, unused = self._prepare(states)
        prior = self.behavior_log_probs(states)[0]
        grids = torch.arange(self.grid_count, device=states.device)
        baseline = prior.argmax(-1)
        uniform = torch.arange(self.action_count, device=states.device)[:, None].expand(-1, self.grid_count)
        candidates = torch.cat((baseline[None], uniform), 0)
        values = self.cached_values(candidates, prepared, unused)
        differences = values[:, 1:]-values[:, :1]
        scores = differences.mean(0)-self.uncertainty_weight*differences.std(0, unbiased=False)
        scores += self.behavior_weight*(prior[grids, uniform].sum(-1)-prior[grids, baseline].sum())
        return [uniform[scores.argmax()].clone()]
