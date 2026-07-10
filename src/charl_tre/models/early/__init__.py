"""Early fusion module for multi-antenna Dirichlet VAEs."""

from charl_tre.models.early.fusion import HierarchicalFusion
from charl_tre.models.early.trainer import Trainer, TrainerParams

__all__ = ["HierarchicalFusion", "Trainer", "TrainerParams"]
