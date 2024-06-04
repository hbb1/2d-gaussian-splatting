from typing import List, Literal
from dataclasses import dataclass, field


@dataclass
class LightGaussian:
    prune_steps: List[int] = field(default_factory=lambda: [])
    prune_decay: float = 1.
    prune_percent: float = 0.66
    prune_type: Literal["v_important_score"] = "v_important_score"
    v_pow: float = 0.1
