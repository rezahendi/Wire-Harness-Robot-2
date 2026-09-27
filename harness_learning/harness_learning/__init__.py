"""Learning scaffolding for the wire-harness cell: Gymnasium env + expert demos."""

import gymnasium as gym

gym.register(
    id="HarnessRouting-v0",
    entry_point="harness_learning.env:HarnessRoutingEnv",
    kwargs={"randomize": True},
)

gym.register(
    id="HarnessRoutingNominal-v0",
    entry_point="harness_learning.env:HarnessRoutingEnv",
    kwargs={"randomize": False},
)
