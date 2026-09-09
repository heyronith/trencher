from __future__ import annotations

import inspect

from trencher.world_model.trainer import ThreePhaseTrainer


def test_health_gate_precedes_optimizer_and_periodic_checkpoint() -> None:
    source = inspect.getsource(ThreePhaseTrainer.train_phase)

    health_raise = source.index('raise RuntimeError(f"health gate failed: {health_msg}")')
    optimizer_step = source.index("self.optimizer.step()")
    periodic_save = source.index('self._save("periodic", phase=phase)')

    assert health_raise < optimizer_step < periodic_save
