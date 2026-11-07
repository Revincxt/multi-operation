"""四期历史滑窗的无损内存表示；不改变数据、标签及采样分布。"""

import numpy as np


class CompactReplay:
    """历史相邻窗口共享三期，仅保留一次时间轴，逐块验证所有重叠值。"""

    def __init__(self, source, chunk_runs=128):
        self.metadata, self.periods, self.count = source.metadata, source.periods, source.count
        self.grids = self.metadata["grid_count"]
        self.steps = self.metadata["preprocessor"]["history_steps"]
        self.features = len(self.metadata["preprocessor"]["history_features"])
        self.history_size = self.grids*self.steps*self.features
        runs = self.count//self.periods
        context_size = self.metadata["state_size"]-self.history_size
        self.frames = np.empty((runs, self.periods+self.steps-1, self.grids, self.features), dtype="float32")
        self.context = np.empty((runs, self.periods, context_size), dtype="float32")
        for first in range(0, runs, chunk_runs):
            last = min(runs, first+chunk_runs)
            values = np.asarray(source.arrays["states"][first*self.periods:last*self.periods]).reshape(last-first, self.periods, -1)
            history = values[:, :, :self.history_size].reshape(last-first, self.periods, self.grids, self.steps, self.features)
            if not np.array_equal(history[:, :-1, :, 1:], history[:, 1:, :, :-1]):
                raise ValueError
            self.frames[first:last, :self.steps] = history[:, 0].transpose(0, 2, 1, 3)
            self.frames[first:last, self.steps:] = history[:, 1:, :, -1]
            self.context[first:last] = values[:, :, self.history_size:]
        self.actions = np.array(source.arrays["actions"])
        self.rewards = np.array(source.arrays["rewards"])
        self.bytes = sum(array.nbytes for array in (self.frames, self.context, self.actions, self.rewards))

    def states(self, indices):
        runs, periods = indices//self.periods, indices%self.periods
        history = self.frames[runs[:, None], periods[:, None]+np.arange(self.steps)]
        history = history.transpose(0, 2, 1, 3).reshape(len(indices), -1)
        return np.concatenate((history, self.context[runs, periods]), -1)

    def batch(self, indices):
        indices = np.asarray(indices)
        terminal = (indices+1)%self.periods == 0
        next_indices = np.where(terminal, indices, indices+1)
        next_states = self.states(next_indices)
        next_states[terminal] = 0
        return {"states": self.states(indices), "actions": self.actions[indices].astype("int64"),
                "rewards": self.rewards[indices], "next_states": next_states, "masks": (~terminal).astype("float32")}
