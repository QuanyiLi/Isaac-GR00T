from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig

# Custom configuration for Panda with 8-dim action and 9-dim state
# All actions use absolute representation (no relative conversion)
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
        delta_indices=list(range(0, 20)),
        modality_keys=[
            "joint_action",
            "gripper_action",
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

# Register under NEW_EMBODIMENT as required for fine-tuning
register_modality_config(modality_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
