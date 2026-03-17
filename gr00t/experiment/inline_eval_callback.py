"""
Inline evaluation callback for GR00T fine-tuning.

Runs simulation evaluation on config_0 (train + test splits) during training:
  - Before the first gradient step (sanity check)
  - Every time a checkpoint is saved

Only runs on rank 0. Uses the in-memory model directly (no checkpoint save/load).
"""

import logging
import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from transformers.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)


def _env_obs_to_groot(obs, modality_config):
    """
    Convert environment observations (flat TensorDict) to the nested dict
    format expected for GR00T model inference.

    Env provides:
        obs["observation.images.image_1"]    : (B, H, W, C) uint8 tensor
        obs["observation.images.wrist_image"] : (B, H, W, C) uint8 tensor
        obs["observation.state"]              : (B, 9) float32 tensor  [7 joint + 2 gripper]
        obs["task"]                           : encoded UTF-8 tensor

    Returns nested dict for processor.
    """
    from vla_align.utils.helpers import batch_tensor_to_string

    video_keys = modality_config["video"].modality_keys
    state_keys = modality_config["state"].modality_keys
    language_keys = modality_config["language"].modality_keys

    # Video key mapping: modality key -> env key
    video_key_map = {
        "image_1": "observation.images.image_1",
        "wrist_camera": "observation.images.wrist_image",
    }
    video_data = {}
    for mk in video_keys:
        env_key = video_key_map.get(mk)
        if env_key is None:
            raise ValueError(f"No env mapping for video modality key: {mk}")
        img = obs[env_key]  # (B, H, W, C) uint8
        img_np = img.cpu().numpy().astype(np.uint8)
        video_data[mk] = img_np[:, None, :, :, :]  # (B, 1, H, W, C)

    # State
    state_np = obs["observation.state"].cpu().numpy().astype(np.float32)  # (B, 9)
    state_data = {}
    for mk in state_keys:
        if mk == "joint_pos":
            state_data[mk] = state_np[:, :7][:, None, :]  # (B, 1, 7)
        elif mk == "gripper_pos":
            state_data[mk] = state_np[:, 7:][:, None, :]  # (B, 1, 2)
        else:
            raise ValueError(f"Unknown state modality key: {mk}")

    # Language
    task_strings = batch_tensor_to_string(obs["task"])  # list[str]
    language_data = {}
    for mk in language_keys:
        language_data[mk] = [[s] for s in task_strings]

    return {
        "video": video_data,
        "state": state_data,
        "language": language_data,
    }


def _make_inmemory_policy_fn(model, processor, modality_config, embodiment_tag, action_horizon, device):
    """
    Create a rollout-compatible policy function using the in-memory model.

    The model is a Gr00tN1d6 (PreTrainedModel) that may be wrapped in DDP/DeepSpeed.
    We unwrap it, set to eval mode, and run inference in bfloat16 (matching training dtype).
    """
    from gr00t.data.types import MessageType, VLAStepData

    action_buffer = []

    # Unwrap DDP / DeepSpeed wrapper if needed
    unwrapped_model = model
    if hasattr(model, "module"):
        unwrapped_model = model.module

    def forward(obs):
        nonlocal action_buffer

        if len(action_buffer) == 0:
            # Convert env obs to GR00T format
            groot_obs = _env_obs_to_groot(obs, modality_config)

            # --- Build processor inputs (same as Gr00tPolicy._get_action) ---
            # Unbatch observations
            batch_size = groot_obs["video"][list(groot_obs["video"].keys())[0]].shape[0]
            processed_inputs = []
            states_list = []

            for i in range(batch_size):
                single_obs = {
                    "video": {k: v[i] for k, v in groot_obs["video"].items()},
                    "state": {k: v[i] for k, v in groot_obs["state"].items()},
                    "language": {k: v[i] for k, v in groot_obs["language"].items()},
                }
                states_list.append(single_obs["state"])  # dict[str, (T, D)]

                vla_step_data = VLAStepData(
                    images=single_obs["video"],
                    states=single_obs["state"],
                    actions={},
                    text=single_obs["language"][modality_config["language"].modality_keys[0]][0],
                    embodiment=embodiment_tag,
                )
                messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
                processed_inputs.append(processor(messages))

            # Collate
            collated_inputs = processor.collator(processed_inputs)

            # The model's get_action handles dtype/device conversion internally
            # via prepare_input(), so we just pass the dict
            was_training = unwrapped_model.training
            unwrapped_model.eval()

            with torch.no_grad():
                model_pred = unwrapped_model.get_action(collated_inputs["inputs"])

            if was_training:
                unwrapped_model.train()

            # Decode actions: convert from normalized to physical space
            normalized_action = model_pred["action_pred"].float()

            # Build batched states for decode_action
            batched_states = {}
            for k in modality_config["state"].modality_keys:
                batched_states[k] = np.stack([s[k] for s in states_list], axis=0)  # (B, T, D)

            unnormalized_action = processor.decode_action(
                normalized_action.cpu().numpy(), embodiment_tag, batched_states
            )

            # Concatenate action components: (B, T_action, total_action_dim)
            action_keys = modality_config["action"].modality_keys
            all_actions = np.concatenate(
                [unnormalized_action[k].astype(np.float32) for k in action_keys], axis=-1
            )

            T_action = all_actions.shape[1]
            steps_to_use = min(T_action, action_horizon)
            for t in range(steps_to_use):
                action_buffer.append(all_actions[:, t, :])

        current_action = action_buffer.pop(0)
        action_tensor = torch.from_numpy(current_action).to(dtype=torch.float32, device=device)
        return action_tensor, action_tensor, {}

    return forward


def _run_eval_on_config0(model, processor, modality_config, embodiment_tag,
                         action_horizon, step_label, num_envs=12, eval_rounds=1):
    """Run evaluation on config_0 train+test splits and print results."""
    from vla_align.env.config import get_env_cfg, MAX_EPISODE_STEP_WORKSPACE_EVAL
    from vla_align.utils.env import build_endless_env
    from vla_align.utils.rollout import rollout

    results = {}

    for split in ["train", "test"]:
        scene_cfg = dict(
            robot_init_qpos_noise=0.0,
            cube_size_noise=0.0,
            cfg_name="config_0",
            mode=split,
        )
        env_cfg = get_env_cfg(
            num_env=num_envs,
            max_steps=MAX_EPISODE_STEP_WORKSPACE_EVAL,
            obs_mode="rgb+segmentation",
            scene_cfg_to_overwrite=scene_cfg,
        )
        envs = build_endless_env(env_cfg, record_video=False, data_record_dir="inline_eval_tmp")

        wrapped = _make_inmemory_policy_fn(
            model, processor, modality_config, embodiment_tag,
            action_horizon, device=envs.device,
        )

        logger.info(f"[InlineEval] Starting rollout for config_0 ({split}) at {step_label}")
        start_time = time.perf_counter()

        with torch.no_grad():
            performance = rollout(
                envs,
                wrapped,
                round_to_collect=eval_rounds,
                demo_saving_dir=None,
                debug_mode=True,
                indices_to_save=[],
            )

        elapsed = time.perf_counter() - start_time

        print("\n" + "=" * 60)
        print(f"[InlineEval] config_0_{split} @ {step_label} — {elapsed:.1f}s")
        print("=" * 60)
        for key, v in performance.items():
            print(f"  {key}: {v}")

        results[f"config_0_{split}"] = performance

        envs.unwrapped.close()

    return results


class InlineEvalCallback(TrainerCallback):
    """
    HuggingFace TrainerCallback that runs simulation evaluation on config_0
    (train + test splits) during GR00T fine-tuning.

    Evaluation triggers:
      1. Before the first gradient step (sanity check)
      2. Every time a checkpoint is saved

    Only runs on world process zero.
    """

    def __init__(
        self,
        processor,
        modality_config,
        embodiment_tag,
        num_envs: int = 12,
        eval_rounds: int = 1,
    ):
        """
        Args:
            processor: The Gr00tN1d6Processor instance (already initialized with stats).
            modality_config: Dict of modality configs for the embodiment.
            embodiment_tag: EmbodimentTag for the embodiment being fine-tuned.
            num_envs: Number of parallel simulation environments.
            eval_rounds: Number of evaluation rounds per split.
        """
        self.processor = processor
        self.modality_config = modality_config
        self.embodiment_tag = embodiment_tag
        self.num_envs = num_envs
        self.eval_rounds = eval_rounds
        self.action_horizon = len(modality_config["action"].delta_indices)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Run eval before the first gradient step as a sanity check."""
        if state.is_world_process_zero and model is not None:
            logger.info("[InlineEval] Running pre-training sanity check evaluation...")
            self.processor.eval()
            _run_eval_on_config0(
                model=model,
                processor=self.processor,
                modality_config=self.modality_config,
                embodiment_tag=self.embodiment_tag,
                action_horizon=self.action_horizon,
                step_label="step_0 (pre-training)",
                num_envs=self.num_envs,
                eval_rounds=self.eval_rounds,
            )
            self.processor.train()

    def on_save(self, args, state, control, model=None, **kwargs):
        """Run eval every time a checkpoint is saved."""
        if state.is_world_process_zero and model is not None:
            step = state.global_step
            logger.info(f"[InlineEval] Running evaluation at step {step}...")
            self.processor.eval()
            _run_eval_on_config0(
                model=model,
                processor=self.processor,
                modality_config=self.modality_config,
                embodiment_tag=self.embodiment_tag,
                action_horizon=self.action_horizon,
                step_label=f"step_{step}",
                num_envs=self.num_envs,
                eval_rounds=self.eval_rounds,
            )
            self.processor.train()
