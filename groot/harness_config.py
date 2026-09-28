# GR00T N1.7 modality config for the wire-harness cell (UR5e-class arm, wrist F/T sensor,
# parallel gripper, 500 Hz admittance controller underneath).
#
# Pass it to GR00T's fine-tuning script:
#     --embodiment-tag NEW_EMBODIMENT --modality-config-path <repo>/groot/harness_config.py
#
# Keys must match meta/modality.json written by `python -m harness_agent.groot_data record`
# (harness_agent/harness_agent/groot_features.py defines the layout).

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

ACTION_HORIZON = 16          # 0.8 s of 20 Hz actions per prediction

harness_config = {
    # a fixed camera over the formboard + the wrist camera, current frame only
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["scene", "wrist"],
    ),
    # tcp pose, commanded lead, gripper, force/torque, goal fixture + fixation, 8 wire keypoints
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["tcp", "command", "gripper", "wrench", "goal", "cable"],
    ),
    # the cell's 5-D action: TCP step (dx dy dz, x 1 cm), yaw step (x 0.15 rad), gripper (-1 open .. +1 closed).
    # The steps are already relative to the commanded pose, so the model predicts them as they are.
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["motion", "gripper"],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    # "route the wire into fork F2", ...
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(harness_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
