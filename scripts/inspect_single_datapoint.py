#!/usr/bin/env python
"""
Inspect a single datapoint from the GR00T training pipeline.

This script replicates the exact data loading path used during training:
  launch_finetune.py → experiment.py → BasicPipeline → DatasetFactory
    → ShardedSingleStepDataset → extract_step_data → processor

It saves BOTH the raw VLAStepData (before processor) and the processed
tensor dict (what the model actually sees) so you can verify correctness.

Usage:
  python scripts/inspect_single_datapoint.py \
    --dataset-path /path/to/dataset \
    --modality-config-path /path/to/modality_config.py \
    --output-dir /path/to/output \
    --episode-index 0 \
    --step-index 5
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_modality_config(modality_config_path: str):
    """Load and register the user's modality config (same as launch_finetune.py)."""
    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


def tensor_to_serialisable(obj):
    """Recursively convert tensors/arrays to lists for JSON."""
    if torch.is_tensor(obj):
        return {"__tensor__": True, "shape": list(obj.shape), "dtype": str(obj.dtype),
                "values": obj.tolist()}
    elif isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "shape": list(obj.shape), "dtype": str(obj.dtype),
                "values": obj.tolist()}
    elif isinstance(obj, dict):
        return {k: tensor_to_serialisable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [tensor_to_serialisable(v) for v in obj]
    else:
        return obj


def save_images_from_step_data(vla_step_data, output_dir: Path):
    """Save images from raw VLAStepData as PNG files."""
    img_dir = output_dir / "raw_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for cam_name, frames in vla_step_data.images.items():
        for t_idx, frame in enumerate(frames):
            if isinstance(frame, Image.Image):
                frame.save(str(img_dir / f"{cam_name}_t{t_idx}.png"))
            elif isinstance(frame, np.ndarray):
                Image.fromarray(frame).save(str(img_dir / f"{cam_name}_t{t_idx}.png"))
            elif torch.is_tensor(frame):
                arr = frame.permute(1, 2, 0).cpu().numpy() if frame.dim() == 3 else frame.cpu().numpy()
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                Image.fromarray(arr.astype(np.uint8)).save(str(img_dir / f"{cam_name}_t{t_idx}.png"))
            else:
                print(f"  Skipping {cam_name}_t{t_idx}: unknown type {type(frame)}")


def save_images_from_processed(processed, output_dir: Path):
    """Save image tensors from the processed dict as PNG files."""
    img_dir = output_dir / "processed_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for key, val in processed.items():
        if not torch.is_tensor(val):
            continue
        # Heuristic: image tensors have 3+ dims and one dim is 3 (channels)
        if val.dim() >= 3 and (val.shape[-3] == 3 or val.shape[-1] == 3):
            # Could be (C, H, W) or (T, C, H, W) or (H, W, C)
            if val.dim() == 3:
                imgs = [val]
            elif val.dim() == 4:
                imgs = [val[i] for i in range(val.shape[0])]
            else:
                continue
            for i, img_t in enumerate(imgs):
                if img_t.shape[0] == 3:  # (C, H, W) → (H, W, C)
                    img_t = img_t.permute(1, 2, 0)
                arr = img_t.cpu().numpy()
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
                Image.fromarray(arr).save(str(img_dir / f"{key}_t{i}.png"))


def main():
    parser = argparse.ArgumentParser(description="Inspect a single GR00T training datapoint")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the LeRobot dataset")
    parser.add_argument("--modality-config-path", type=str, required=True,
                        help="Path to modality_config.py")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save inspection outputs")
    parser.add_argument("--base-model-path", type=str, default="nvidia/GR00T-N1.6-3B",
                        help="Base model path for processor config")
    parser.add_argument("--video-backend", type=str, default="pyav",
                        help="Video decoding backend")
    parser.add_argument("--episode-index", type=int, default=0,
                        help="Episode index to inspect")
    parser.add_argument("--step-index", type=int, default=0,
                        help="Step index within episode to inspect")
    parser.add_argument("--embodiment-tag", type=str, default="NEW_EMBODIMENT",
                        help="Embodiment tag")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Register modality config (same as launch_finetune.py) ──
    load_modality_config(args.modality_config_path)

    from gr00t.configs.base_config import get_default_config
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType

    embodiment_tag = EmbodimentTag[args.embodiment_tag]

    # ── 2. Build config exactly like launch_finetune.py ──
    config = get_default_config().load_dict({
        "data": {
            "download_cache": False,
            "datasets": [{
                "dataset_paths": [args.dataset_path],
                "mix_ratio": 1.0,
                "embodiment_tag": args.embodiment_tag,
            }],
        }
    })
    config.data.video_backend = args.video_backend
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True
    # Apply the same augmentation params used in training
    config.model.state_dropout_prob = 0.0  # no dropout for inspection
    config.model.color_jitter_params = None  # no jitter for inspection

    modality_configs = config.data.modality_configs[args.embodiment_tag]

    # ── 3. Load a single episode via LeRobotEpisodeLoader ──
    print(f"\n{'='*60}")
    print(f"Loading episode {args.episode_index}, step {args.step_index}")
    print(f"Dataset: {args.dataset_path}")
    print(f"{'='*60}\n")

    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_configs,
        video_backend=args.video_backend,
    )

    print(f"Total episodes: {len(loader)}")
    print(f"Episode lengths: {loader.episode_lengths[:10]}...")
    ep_len = loader.get_episode_length(args.episode_index)
    print(f"Episode {args.episode_index} length: {ep_len}")

    episode_data = loader[args.episode_index]
    print(f"Episode DataFrame columns: {list(episode_data.columns)}")
    print(f"Episode DataFrame shape: {episode_data.shape}")

    # ── 4. Extract raw VLAStepData (before processor) ──
    vla_step_data = extract_step_data(
        episode_data,
        args.step_index,
        modality_configs,
        embodiment_tag,
        allow_padding=False,
    )

    print(f"\n--- Raw VLAStepData ---")
    print(f"  Text: {vla_step_data.text}")
    print(f"  Embodiment: {vla_step_data.embodiment}")
    for mod_name, mod_data in [("states", vla_step_data.states),
                                ("actions", vla_step_data.actions),
                                ("images", vla_step_data.images)]:
        print(f"  {mod_name}:")
        for k, v in mod_data.items():
            if isinstance(v, np.ndarray):
                print(f"    {k}: shape={v.shape}, dtype={v.dtype}, "
                      f"min={v.min():.6f}, max={v.max():.6f}")
            elif isinstance(v, list):
                print(f"    {k}: list len={len(v)}, type={type(v[0]) if v else 'empty'}")
            else:
                print(f"    {k}: {type(v)}")

    # Save raw images
    save_images_from_step_data(vla_step_data, output_dir)

    # Save raw numerical data
    raw_data = {
        "text": vla_step_data.text,
        "embodiment": str(vla_step_data.embodiment),
        "states": {k: v.tolist() if isinstance(v, np.ndarray) else v
                   for k, v in vla_step_data.states.items()},
        "actions": {k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in vla_step_data.actions.items()},
    }
    with open(output_dir / "raw_step_data.json", "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"\nSaved raw step data to {output_dir / 'raw_step_data.json'}")

    # ── 5. Run through processor (exactly what training sees) ──
    from gr00t.data.dataset.factory import DatasetFactory
    from gr00t.data.stats import generate_stats, generate_rel_stats

    # Generate stats (same as factory.py)
    generate_stats(args.dataset_path)
    generate_rel_stats(args.dataset_path, embodiment_tag)

    # Build dataset to get processor with statistics
    from gr00t.model import MODEL_REGISTRY
    save_cfg_dir = output_dir / "tmp_cfg"
    save_cfg_dir.mkdir(parents=True, exist_ok=True)
    pipeline = MODEL_REGISTRY.get(type(config.model))(config, save_cfg_dir)

    # Build processor without the full model
    processor = pipeline.processor_class(
        modality_configs=config.data.modality_configs,
        statistics=None,
        **config.model.processor_kwargs,
    )

    # Build a minimal ShardedSingleStepDataset to get proper stats merging
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
    shard_ds = ShardedSingleStepDataset(
        dataset_path=args.dataset_path,
        embodiment_tag=embodiment_tag,
        modality_configs=modality_configs,
        video_backend=args.video_backend,
        shard_size=2**10,
        episode_sampling_rate=0.1,
        seed=42,
        allow_padding=False,
    )

    # Get stats and set on processor
    stats = shard_ds.get_dataset_statistics()
    emb_key = embodiment_tag.value
    stats_by_emb = {emb_key: {}}
    for modality in ["state", "action", "relative_action"]:
        if modality in stats:
            stats_by_emb[emb_key][modality] = {modality: stats[modality]}
    # Use simpler approach: set processor on dataset and use get_datapoint
    shard_ds.set_processor(processor)

    # Now use the dataset's own stats merging path
    from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
    mixture = ShardedMixtureDataset(
        datasets=[shard_ds],
        weights=[1.0],
        processor=processor,
        seed=42,
        training=True,
        num_shards_per_epoch=1,
        override_pretraining_statistics=True,
    )

    # Now processor has correct statistics. Get the processed datapoint.
    messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
    processed = processor(messages)

    print(f"\n--- Processed datapoint (what the model sees) ---")
    for key, val in processed.items():
        if torch.is_tensor(val):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}, "
                  f"min={val.min().item():.6f}, max={val.max().item():.6f}")
        elif isinstance(val, np.ndarray):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key}: {type(val)} = {val}")

    # Save processed tensors
    torch.save(processed, output_dir / "processed_datapoint.pt")
    print(f"\nSaved processed datapoint to {output_dir / 'processed_datapoint.pt'}")

    # Save processed data as JSON for readability
    processed_json = tensor_to_serialisable(processed)
    with open(output_dir / "processed_datapoint.json", "w") as f:
        json.dump(processed_json, f, indent=2)
    print(f"Saved processed datapoint JSON to {output_dir / 'processed_datapoint.json'}")

    # Save processed images
    save_images_from_processed(processed, output_dir)

    # ── 6. Summary ──
    print(f"\n{'='*60}")
    print(f"Inspection complete! All outputs saved to: {output_dir}")
    print(f"{'='*60}")
    print(f"Files:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file() and "tmp_cfg" not in str(f):
            print(f"  {f.relative_to(output_dir)}")


if __name__ == "__main__":
    main()
