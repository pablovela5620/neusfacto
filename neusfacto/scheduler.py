from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Type

from torch.optim import Optimizer, lr_scheduler

from nerfstudio.engine.schedulers import Scheduler, SchedulerConfig

@dataclass
class MultiStepSchedulerConfig(SchedulerConfig):
    """Basic scheduler config with self-defined exponential decay schedule"""

    _target: Type = field(default_factory=lambda: MultiStepScheduler)
    """target class to instantiate"""
    max_steps: int = 1000000
    """The maximum number of steps."""


class MultiStepScheduler(Scheduler):
    """Starts with a flat lr schedule until it reaches N epochs then applies a given scheduler"""

    config: MultiStepSchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> lr_scheduler._LRScheduler:
        max_steps = self.config.max_steps
        scheduler = lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],
            gamma=0.33,
        )
        return scheduler