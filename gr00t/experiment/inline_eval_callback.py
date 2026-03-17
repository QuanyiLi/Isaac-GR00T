"""
Inline evaluation callback for GR00T fine-tuning.

Runs simulation evaluation on config_0 (train + test splits) during training:
  - Before the first gradient step (sanity check)
  - Every time a checkpoint is saved

Only runs on rank 0. Evaluation is run in a **spawned subprocess** to fully
isolate the SAPIEN/Vulkan GPU context from the training process's CUDA state,
preventing segfaults in DataLoader workers.
"""

import logging
import os
import tempfile
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subprocess entry point — runs in a completely separate process
# ---------------------------------------------------------------------------

def _subprocess_eval(model_state_path, processor_dir, modality_config_path,
                     embodiment_tag_value, step_label, num_envs, eval_rounds,
                     result_file):
    """
    Entry point for the eval subprocess. Loads model from state_dict,
    builds a Gr00tPolicy-like inference pipeline, runs rollouts on config_0,
    and writes results to a file.

    This function runs in a fresh process with no SAPIEN/Vulkan state inherited.
    """
    import json
    import numpy as np
    import torch

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["SVT_LOG"] = "0"

    # Register modality config
    import importlib.util
    spec = importlib.util.spec_from_file_location("modality_config", modality_config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    # Load policy from the temp checkpoint dir (which has model weights + processor)
    device = torch.device("cuda")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag_value),
        model_path=model_state_path,
        device=device,
    )
    modality_config = policy.get_modality_config()
    action_horizon = len(modality_config["action"].delta_indices)

    # Import eval utilities
    from vla_align.env.config import get_env_cfg, MAX_EPISODE_STEP_WORKSPACE_EVAL
    from vla_align.utils.env import build_endless_env
    from vla_align.utils.helpers import batch_tensor_to_string
    from vla_align.utils.rollout import rollout

    # Observation conversion (same as gr00t_eval.py)
    def env_obs_to_groot(obs):
        video_keys = modality_config["video"].modality_keys
        state_keys = modality_config["state"].modality_keys
        language_keys = modality_config["language"].modality_keys

        video_key_map = {
            "image_1": "observation.images.image_1",
            "wrist_camera": "observation.images.wrist_image",
        }
        video_data = {}
        for mk in video_keys:
            env_key = video_key_map.get(mk)
            if env_key is None:
                raise ValueError(f"No env mapping for video modality key: {mk}")
            img = obs[env_key].cpu().numpy().astype(np.uint8)
            video_data[mk] = img[:, None, :, :, :]

        state_np = obs["observation.state"].cpu().numpy().astype(np.float32)
        state_data = {}
        for mk in state_keys:
            if mk == "joint_pos":
                state_data[mk] = state_np[:, :7][:, None, :]
            elif mk == "gripper_pos":
                state_data[mk] = state_np[:, 7:][:, None, :]
            else:
                raise ValueError(f"Unknown state modality key: {mk}")

        task_strings = batch_tensor_to_string(obs["task"])
        language_data = {}
        for mk in language_keys:
            language_data[mk] = [[s] for s in task_strings]

        return {"video": video_data, "state": state_data, "language": language_data}

    # Policy wrapper for rollout (same as gr00t_eval.py)
    action_buffer = []

    def policy_fn(obs):
        nonlocal action_buffer
        if len(action_buffer) == 0:
            groot_obs = env_obs_to_groot(obs)
            action_dict, _ = policy.get_action(groot_obs)
            action_keys = modality_config["action"].modality_keys
            all_actions = np.concatenate(
                [action_dict[k] for k in action_keys], axis=-1
            )
            T_action = all_actions.shape[1]
            steps_to_use = min(T_action, action_horizon)
            for t in range(steps_to_use):
                action_buffer.append(all_actions[:, t, :])

        current_action = action_buffer.pop(0)
        action_tensor = torch.from_numpy(current_action).to(dtype=torch.float32, device=device)
        return action_tensor, action_tensor, {}

    # Run eval on config_0 train + test
    all_results = {}
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

        # Reset action buffer for each split
        action_buffer.clear()

        print(f"\n[InlineEval-subprocess] Starting rollout for config_0 ({split}) at {step_label}")
        start_time = time.perf_counter()

        with torch.no_grad():
            performance = rollout(
                envs,
                policy_fn,
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

        # Convert numpy values for JSON serialization
        serializable = {}
        for k, v in performance.items():
            if hasattr(v, 'item'):
                serializable[k] = v.item()
            else:
                serializable[k] = float(v) if isinstance(v, (int, float, np.floating)) else str(v)
        all_results[f"config_0_{split}"] = serializable

        envs.unwrapped.close()

    # Write results to file so parent can read them
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class InlineEvalCallback(TrainerCallback):
    """
    HuggingFace TrainerCallback that runs simulation evaluation on config_0
    (train + test splits) during GR00T fine-tuning.

    Evaluation triggers:
      1. Before the first gradient step (sanity check)
      2. Every time a checkpoint is saved

    Only runs on world process zero. Eval runs in a **spawned subprocess**
    to isolate SAPIEN/Vulkan GPU contexts from the training CUDA state.
    """

    def __init__(
        self,
        processor,
        modality_config,
        embodiment_tag,
        modality_config_path: str,
        output_dir: str,
        num_envs: int = 12,
        eval_rounds: int = 1,
    ):
        """
        Args:
            processor: The Gr00tN1d6Processor instance (for saving to temp dir).
            modality_config: Dict of modality configs for the embodiment.
            embodiment_tag: EmbodimentTag for the embodiment being fine-tuned.
            modality_config_path: Path to modality_config.py file.
            output_dir: Training output directory (for saving temp checkpoints).
            num_envs: Number of parallel simulation environments.
            eval_rounds: Number of evaluation rounds per split.
        """
        self.processor = processor
        self.modality_config = modality_config
        self.embodiment_tag = embodiment_tag
        self.modality_config_path = modality_config_path
        self.output_dir = output_dir
        self.num_envs = num_envs
        self.eval_rounds = eval_rounds

    def _run_eval_subprocess(self, model, step_label):
        """Save model to temp dir, spawn eval subprocess, print results."""
        import json
        import shutil
        import subprocess

        # Unwrap DDP/DeepSpeed
        unwrapped_model = model
        if hasattr(model, "module"):
            unwrapped_model = model.module

        # Save model + processor to a temp checkpoint directory
        temp_ckpt_dir = os.path.join(self.output_dir, "_inline_eval_tmp_ckpt")
        os.makedirs(temp_ckpt_dir, exist_ok=True)

        try:
            # Save model weights
            unwrapped_model.save_pretrained(temp_ckpt_dir)
            # Save processor
            self.processor.save_pretrained(temp_ckpt_dir)

            # Create result file
            result_fd, result_file = tempfile.mkstemp(suffix=".json")
            os.close(result_fd)

            # Build the subprocess command
            # We call this module's _subprocess_eval function via a small wrapper
            eval_script = os.path.join(os.path.dirname(__file__), "_inline_eval_runner.py")

            # Write a small runner script if it doesn't exist
            if not os.path.exists(eval_script):
                with open(eval_script, "w") as f:
                    f.write('"""Auto-generated runner for inline eval subprocess."""\n')
                    f.write("import sys, json\n")
                    f.write("from gr00t.experiment.inline_eval_callback import _subprocess_eval\n")
                    f.write("if __name__ == '__main__':\n")
                    f.write("    args = json.loads(sys.argv[1])\n")
                    f.write("    _subprocess_eval(**args)\n")

            # Subprocess args
            eval_args = {
                "model_state_path": temp_ckpt_dir,
                "processor_dir": temp_ckpt_dir,
                "modality_config_path": self.modality_config_path,
                "embodiment_tag_value": self.embodiment_tag.value,
                "step_label": step_label,
                "num_envs": self.num_envs,
                "eval_rounds": self.eval_rounds,
                "result_file": result_file,
            }

            import json as json_mod
            args_json = json_mod.dumps(eval_args)

            logger.info(f"[InlineEval] Spawning eval subprocess for {step_label}...")
            proc = subprocess.run(
                [
                    "python", eval_script, args_json,
                ],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
                capture_output=False,  # Let output go to stdout/stderr
                timeout=600,  # 10 minute timeout
            )

            if proc.returncode != 0:
                logger.error(f"[InlineEval] Subprocess exited with code {proc.returncode}")
            elif os.path.exists(result_file) and os.path.getsize(result_file) > 0:
                with open(result_file) as f:
                    results = json.load(f)
                logger.info(f"[InlineEval] Results at {step_label}: {json.dumps(results, indent=2)}")

        except Exception as e:
            logger.error(f"[InlineEval] Error during evaluation: {e}")
        finally:
            # Clean up temp checkpoint
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            if 'result_file' in locals() and os.path.exists(result_file):
                os.remove(result_file)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Run eval before the first gradient step as a sanity check."""
        if state.is_world_process_zero and model is not None:
            logger.info("[InlineEval] Running pre-training sanity check evaluation...")
            self._run_eval_subprocess(model, "step_0 (pre-training)")

    def on_save(self, args, state, control, model=None, **kwargs):
        """Run eval every time a checkpoint is saved."""
        if state.is_world_process_zero and model is not None:
            step = state.global_step
            logger.info(f"[InlineEval] Running evaluation at step {step}...")
            self._run_eval_subprocess(model, f"step_{step}")
