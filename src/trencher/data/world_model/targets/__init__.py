"""Label-only future target engine (strictly post-T supervision).

PIT / observation / sequence input modules must NEVER import this package.
"""

from trencher.data.world_model.targets.build import build_target_dataset_v1

__all__ = ["build_target_dataset_v1"]
