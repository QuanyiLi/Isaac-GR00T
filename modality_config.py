from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# Custom configuration for Panda with 8-dim action and 9-dim state
# Mapping: 7 joints (Relative) + 1 Gripper (Absolute)
modality_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "image_1",
            "wrist_camera",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "joint_pos",
            "gripper_pos",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 40)),
        modality_keys=[
            "joint_action",
            "gripper_action",
        ],
        action_configs=[
            # 7-dim joint action (RELATIVE aligns with N1.6 pretraining)
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # 1-dim gripper action
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="gripper_pos",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

# Register under NEW_EMBODIMENT as required for fine-tuning
register_modality_config(modality_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
