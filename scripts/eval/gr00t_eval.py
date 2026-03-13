"""
Evaluate a fine-tuned GR00T N1.6 model in simulation across config subsets.

This script mirrors lerobot_eval.py but uses a GR00T policy (Gr00tPolicy)
instead of a LeRobot policy. It bridges the environment observation format
to the GR00T policy's expected input format.

Usage:
    python gr00t_eval.py \
        --model_path /path/to/checkpoint \
        --modality_config_path /path/to/modality_config.py \
        --start_subset 0 --end_subset 6 \
        --split both \
        --result_dir ./gr00t_eval_result \
        --eval_rounds 1
"""
import argparse
import json
import logging
import os
import shutil
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["SVT_LOG"] = "0"

import numpy as np
import torch

# vla_align imports
from vla_align.env.config import get_env_cfg, MAX_EPISODE_STEP_WORKSPACE_EVAL
from vla_align.utils.env import build_endless_env
from vla_align.utils.helpers import batch_tensor_to_string
from vla_align.utils.rollout import rollout, calculate_averages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Observation conversion: env TensorDict -> GR00T nested dict
# ---------------------------------------------------------------------------

def env_obs_to_groot(obs, modality_config):
    """
    Convert environment observations (flat TensorDict) to the nested dict
    format expected by Gr00tPolicy.get_action().

    Env provides:
        obs["observation.images.image_1"]  : (B, H, W, C) uint8 tensor
        obs["observation.images.wrist_image"] : (B, H, W, C) uint8 tensor
        obs["observation.state"]           : (B, 9) float32 tensor  [7 joint + 2 gripper]
        obs["task"]                        : encoded UTF-8 tensor

    GR00T expects:
        {
          "video": {"image_1": (B, T, H, W, C) uint8 np, "wrist_camera": ...},
          "state": {"joint_pos": (B, T, 7) f32 np, "gripper_pos": (B, T, 2) f32 np},
          "language": {"annotation.human.action.task_description": [[str], ...]}
        }
    """
    video_keys = modality_config["video"].modality_keys
    state_keys = modality_config["state"].modality_keys
    language_keys = modality_config["language"].modality_keys

    # --- Video ---
    # Map env key -> modality key
    video_key_map = {
        "image_1": "observation.images.image_1",
        "wrist_camera": "observation.images.wrist_image",
    }
    video_data = {}
    for mk in video_keys:
        env_key = video_key_map.get(mk)
        if env_key is None:
            raise ValueError(f"No env mapping for video modality key: {mk}")
        img = obs[env_key]  # (B, H, W, C) uint8 torch tensor
        img_np = img.cpu().numpy().astype(np.uint8)
        # Add temporal dimension T=1: (B, H, W, C) -> (B, 1, H, W, C)
        video_data[mk] = img_np[:, None, :, :, :]

    # --- State ---
    state_np = obs["observation.state"].cpu().numpy().astype(np.float32)  # (B, 9)
    state_data = {}
    for mk in state_keys:
        if mk == "joint_pos":
            state_data[mk] = state_np[:, :7][:, None, :]  # (B, 1, 7)
        elif mk == "gripper_pos":
            state_data[mk] = state_np[:, 7:][:, None, :]  # (B, 1, 2)
        else:
            raise ValueError(f"Unknown state modality key: {mk}")

    # --- Language ---
    task_strings = batch_tensor_to_string(obs["task"])  # list[str] of length B
    language_data = {}
    for mk in language_keys:
        # Each batch item needs to be [[str]] (batch of temporal sequences)
        language_data[mk] = [[s] for s in task_strings]

    return {
        "video": video_data,
        "state": state_data,
        "language": language_data,
    }


# ---------------------------------------------------------------------------
# GR00T policy wrapper for the rollout() function
# ---------------------------------------------------------------------------

def make_groot_policy_fn(policy, modality_config, action_horizon, device):
    """
    Create a callable that matches the signature expected by rollout():
        policy_fn(obs) -> (action_to_take, expert_action, info_dict)

    Uses chunked action execution: infer once, execute all `action_horizon`
    actions, then re-infer.
    """
    action_buffer = []  # list of (B, action_dim) numpy arrays

    def forward(obs):
        nonlocal action_buffer

        if len(action_buffer) == 0:
            # Convert env obs to GR00T format
            groot_obs = env_obs_to_groot(obs, modality_config)

            # Run inference
            action_dict, _ = policy.get_action(groot_obs)
            # action_dict: {"joint_action": (B, T_action, 7), "gripper_action": (B, T_action, 1)}

            action_keys = modality_config["action"].modality_keys
            # Concatenate action components along the last dimension
            # -> (B, T_action, 8)
            all_actions = np.concatenate(
                [action_dict[k] for k in action_keys], axis=-1
            )

            # Fill the action buffer with per-step actions
            T_action = all_actions.shape[1]
            steps_to_use = min(T_action, action_horizon)
            for t in range(steps_to_use):
                action_buffer.append(all_actions[:, t, :])  # (B, action_dim)

        # Pop the next action from the buffer
        current_action = action_buffer.pop(0)
        action_tensor = torch.from_numpy(current_action).to(dtype=torch.float32, device=device)

        return action_tensor, action_tensor, {}

    return forward


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_eval(args):
    result_dir = args.result_dir

    if args.aggregate_only:
        _aggregate_results(result_dir)
        return

    os.makedirs(result_dir, exist_ok=True)

    # Validate required args for rollout mode
    if not args.model_path:
        raise ValueError("--model_path is required when not using --aggregate_only")
    if not args.modality_config_path:
        raise ValueError("--modality_config_path is required when not using --aggregate_only")

    # Device setup
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # --- Load GR00T Policy ---
    logger.info("Loading GR00T policy...")

    # Register the custom modality config by importing the config file
    import importlib.util
    spec = importlib.util.spec_from_file_location("modality_config", args.modality_config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    logger.info(f"Registered modality config from {args.modality_config_path}")

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=args.model_path,
        device=device,
    )
    modality_config = policy.get_modality_config()
    logger.info(f"Modality config: {modality_config}")

    action_horizon = len(modality_config["action"].delta_indices)
    logger.info(f"Action horizon (chunk size): {action_horizon}")

    # Determine splits to evaluate
    splits = ["train", "test"] if args.split == "both" else [args.split]

    logger.info(f"Evaluating subsets {args.start_subset} to {args.end_subset - 1} "
                f"on splits: {splits}")

    # Evaluation loop
    for split in splits:
        for i in range(args.start_subset, args.end_subset):
            cfg_name = f"config_{i}"
            subset_result_dir = os.path.join(result_dir, f"config_{i}_{split}")

            # Skip if already evaluated
            metrics_file = os.path.join(subset_result_dir, "episode_metrics.json")
            if os.path.exists(metrics_file):
                logger.info(f"Skipping {cfg_name} ({split}) — already evaluated")
                continue

            if os.path.exists(subset_result_dir):
                shutil.rmtree(subset_result_dir)
            os.makedirs(subset_result_dir)

            # Build environment for this subset + split
            scene_cfg = dict(
                robot_init_qpos_noise=0.0,
                cube_size_noise=0.0,
                cfg_name=cfg_name,
                mode=split,
            )
            env_cfg = get_env_cfg(
                num_env=12,
                max_steps=MAX_EPISODE_STEP_WORKSPACE_EVAL,
                obs_mode="rgb+segmentation",
                scene_cfg_to_overwrite=scene_cfg,
            )
            envs = build_endless_env(env_cfg, record_video=False, data_record_dir="test")

            # Wrap GR00T policy for rollout
            wrapped = make_groot_policy_fn(
                policy, modality_config, action_horizon, device=envs.device
            )

            # Rollout
            print("\n" + "=" * 60)
            print(f"Starting Rollout for {cfg_name} ({split})")
            print("=" * 60)

            start_time = time.perf_counter()
            with torch.no_grad():
                performance = rollout(
                    envs,
                    wrapped,
                    round_to_collect=args.eval_rounds,
                    demo_saving_dir=subset_result_dir,
                    debug_mode=True,
                    indices_to_save=None if args.save_video else [],
                )
            elapsed = time.perf_counter() - start_time

            print("\n" + "=" * 60)
            print(f"Performance for {cfg_name} ({split}) — {elapsed:.1f}s")
            print("=" * 60)
            for key, v in performance.items():
                print(f"  {key}: {v}")

            envs.unwrapped.close()

    logger.info("All subsets evaluated. Done.")


def _aggregate_results(result_dir):
    """Compute final aggregated results for train and test splits."""
    for split in ["train", "test"]:
        pattern = os.path.join(result_dir, f"*{split}*")
        final_results = calculate_averages(pattern)
        if final_results:
            out_path = os.path.join(result_dir, f"final_results_{split}.json")
            with open(out_path, "w") as f:
                json.dump(final_results, f, indent=2)
            print(f"Final results saved to {out_path}")
        else:
            print(f"No results found for split '{split}'")


def parse_args():
    parser = argparse.ArgumentParser(description="GR00T Simulation Evaluation")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to GR00T fine-tuned checkpoint")
    parser.add_argument("--modality_config_path", type=str, default=None,
                        help="Path to modality_config.py for the embodiment")
    parser.add_argument("--start_subset", type=int, default=0,
                        help="Start config index (inclusive)")
    parser.add_argument("--end_subset", type=int, default=24,
                        help="End config index (exclusive)")
    parser.add_argument("--split", type=str, default="both",
                        choices=["train", "test", "both"],
                        help="Which split(s) to evaluate")
    parser.add_argument("--result_dir", type=str, default="./gr00t_eval_result",
                        help="Directory to save evaluation results")
    parser.add_argument("--eval_rounds", type=int, default=1,
                        help="Number of evaluation rounds per subset")
    parser.add_argument("--aggregate_only", action="store_true",
                        help="Skip rollouts, only compute final aggregation")
    parser.add_argument("--save_video", action="store_true",
                        help="Save episode videos to each subset result directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
