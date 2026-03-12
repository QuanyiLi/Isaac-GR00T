#!/usr/bin/env python
"""
Test script for verifying data loading in the GR00T fine-tuning pipeline.

This script replicates the data setup from launch_finetune.py but skips model
loading and training. It loads the first batch from the dataloader and saves
the decoded video frames as images for visual inspection.

Two levels of inspection:
  PART 1 – Raw video frames from the episode loader (before processor).
  PART 2 – First batch from the full DataLoader pipeline (after processor).

Usage (via SLURM):
    sbatch scripts/slurm/submit_test_dataloader.slurm
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Setup logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def save_tensor_as_image(tensor_or_array, save_path: Path, label: str = ""):
    """Save a tensor/ndarray image to disk. Handles float32 [0,1] -> uint8 conversion."""
    if isinstance(tensor_or_array, torch.Tensor):
        arr = tensor_or_array.cpu().numpy()
    else:
        arr = np.array(tensor_or_array)

    # If float and in [0, 1], convert to uint8
    if arr.dtype in (np.float32, np.float64):
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
            logger.info(f"  [{label}] Converted float32 [0,1] -> uint8")
        else:
            arr = arr.astype(np.uint8)

    # Handle CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    # Handle single-channel
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze(-1)

    img = Image.fromarray(arr)
    img.save(str(save_path))
    logger.info(f"  [{label}] Saved {save_path} | size={img.size} mode={img.mode}")


def main():
    import tyro
    from gr00t.configs.base_config import get_default_config
    from gr00t.configs.finetune_config import FinetuneConfig

    # ── Helper: Load modality config module ──────────────────────────────
    def load_modality_config(modality_config_path: str):
        import importlib
        path = Path(modality_config_path)
        if path.exists() and path.suffix == ".py":
            sys.path.append(str(path.parent))
            importlib.import_module(path.stem)
            logger.info(f"Loaded modality config: {path}")
        else:
            raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")

    # ── Parse CLI args (same as launch_finetune.py) ──────────────────────
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    embodiment_tag = ft_config.embodiment_tag.value

    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    # ── Build config (mirrors launch_finetune.py exactly) ────────────────
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [ft_config.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # Overwrite with finetune config
    config.data.video_backend = ft_config.video_backend
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = False  # never use wandb for test
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = "finetune-gr00t-n1d6"

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    config.validate()

    # ── Create output directory for saved images ─────────────────────────
    output_dir = Path(ft_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "test_dataloader_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Will save test images to: {images_dir}")

    # ======================================================================
    # PART 1: Raw video frames (before any processing)
    # This tests whether pyav correctly decodes the mp4 videos.
    # ======================================================================
    logger.info("=" * 80)
    logger.info("PART 1: Loading raw video frames from episode loader")
    logger.info("=" * 80)

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    dataset_path = ft_config.dataset_path
    modality_configs = config.data.modality_configs[embodiment_tag]

    episode_loader = LeRobotEpisodeLoader(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        video_backend=ft_config.video_backend,
    )

    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Video backend: {ft_config.video_backend}")
    logger.info(f"Number of episodes: {len(episode_loader)}")
    episode_lengths = episode_loader.get_episode_lengths()
    logger.info(f"First 5 episode lengths: {episode_lengths[:5]}")

    # Load the first episode
    episode_data = episode_loader[0]
    logger.info(f"Loaded episode 0 with {len(episode_data)} timesteps")
    logger.info(f"Columns: {list(episode_data.columns)}")

    # Save raw video frames from the first few timesteps
    video_keys = modality_configs["video"].modality_keys
    raw_dir = images_dir / "raw_frames"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for view_key in video_keys:
        col_name = f"video.{view_key}"
        if col_name in episode_data.columns:
            for t_idx in range(min(3, len(episode_data))):
                frame = episode_data[col_name].iloc[t_idx]
                if isinstance(frame, np.ndarray):
                    save_tensor_as_image(
                        frame,
                        raw_dir / f"raw_{view_key}_t{t_idx}.png",
                        label=f"raw/{view_key}/t{t_idx}",
                    )
                elif isinstance(frame, Image.Image):
                    save_path = raw_dir / f"raw_{view_key}_t{t_idx}.png"
                    frame.save(str(save_path))
                    logger.info(f"  [raw/{view_key}/t{t_idx}] Saved PIL: {save_path} | size={frame.size}")
                else:
                    logger.warning(f"  Unknown frame type: {type(frame)} for {view_key} t={t_idx}")
        else:
            logger.warning(f"  Column {col_name} not found in episode data!")

    # ======================================================================
    # PART 2: Processed batch from the full pipeline
    # This tests the complete data loading + processing pipeline.
    # ======================================================================
    logger.info("=" * 80)
    logger.info("PART 2: Loading first batch from the full data pipeline")
    logger.info("=" * 80)

    from gr00t.data.dataset.factory import DatasetFactory
    from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
    from transformers import AutoProcessor

    # Build processor exactly as Gr00tN1d6Pipeline._create_dataset does
    transformers_loading_kwargs = {"trust_remote_code": True}

    processor = AutoProcessor.from_pretrained(
        ft_config.base_model_path,
        # Overrides
        modality_configs=config.data.modality_configs,
        image_crop_size=config.model.image_crop_size,
        image_target_size=config.model.image_target_size,
        random_rotation_angle=config.model.random_rotation_angle,
        color_jitter_params=config.model.color_jitter_params,
        model_name=config.model.model_name,
        model_type=config.model.backbone_model_type,
        formalize_language=config.model.formalize_language,
        apply_sincos_state_encoding=config.model.apply_sincos_state_encoding,
        max_action_horizon=config.model.action_horizon,
        use_albumentations=config.model.use_albumentations_transforms,
        shortest_image_edge=config.model.shortest_image_edge,
        crop_fraction=config.model.crop_fraction,
        transformers_loading_kwargs=transformers_loading_kwargs,
        use_relative_action=config.model.use_relative_action,
        **transformers_loading_kwargs,
    )
    logger.info("Processor loaded successfully")

    # Build dataset via factory (same as the training pipeline)
    dataset_factory = DatasetFactory(config)
    train_dataset, _ = dataset_factory.build(processor)
    logger.info("Dataset built successfully")

    # Use a DataLoader to get the first batch
    from torch.utils.data import DataLoader

    collator = processor.collator
    dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        num_workers=0,  # single worker for deterministic test
        collate_fn=collator,
    )

    logger.info("Getting first batch from dataloader...")
    batch = next(iter(dataloader))

    # ── Inspect and log the batch structure ──────────────────────────────
    logger.info(f"Batch type: {type(batch)}")
    if isinstance(batch, dict):
        batch_to_inspect = batch
    elif hasattr(batch, "data"):
        # BatchFeature stores data in .data attribute
        batch_to_inspect = batch.data if isinstance(batch.data, dict) else batch
    else:
        batch_to_inspect = batch

    # The Gr00tN1d6DataCollator wraps everything in {"inputs": {...}}
    if "inputs" in batch_to_inspect:
        inner = batch_to_inspect["inputs"]
        logger.info(f"Batch['inputs'] keys: {list(inner.keys()) if isinstance(inner, dict) else type(inner)}")
        batch_to_inspect = inner
    else:
        logger.info(f"Batch keys: {list(batch_to_inspect.keys()) if isinstance(batch_to_inspect, dict) else type(batch_to_inspect)}")

    for key, val in (batch_to_inspect.items() if isinstance(batch_to_inspect, dict) else []):
        if isinstance(val, torch.Tensor):
            logger.info(
                f"  {key}: shape={val.shape}, dtype={val.dtype}, "
                f"min={val.min().item():.4f}, max={val.max().item():.4f}"
            )
        elif isinstance(val, np.ndarray):
            logger.info(f"  {key}: np.array shape={val.shape}, dtype={val.dtype}")
        elif isinstance(val, list):
            logger.info(f"  {key}: list len={len(val)}, first_type={type(val[0]) if val else 'empty'}")
        else:
            logger.info(f"  {key}: type={type(val)}")

    # ── Save pixel_values images (post-VLM-processing) ───────────────────
    processed_dir = images_dir / "processed_batch"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(batch_to_inspect, dict) and "pixel_values" in batch_to_inspect:
        pv = batch_to_inspect["pixel_values"]
        if isinstance(pv, torch.Tensor):
            pv = pv.cpu()
        logger.info(f"pixel_values: shape={pv.shape}, dtype={pv.dtype}")

        if pv.ndim == 5:
            # (batch, num_patches, C, H, W) – typical for any-resolution VLMs
            for b_idx in range(min(2, pv.shape[0])):
                for p_idx in range(min(6, pv.shape[1])):
                    save_tensor_as_image(
                        pv[b_idx, p_idx],
                        processed_dir / f"pixel_b{b_idx}_patch{p_idx}.png",
                        label=f"processed/b{b_idx}/p{p_idx}",
                    )
        elif pv.ndim == 4:
            # (batch, C, H, W)
            for b_idx in range(min(4, pv.shape[0])):
                save_tensor_as_image(
                    pv[b_idx],
                    processed_dir / f"pixel_b{b_idx}.png",
                    label=f"processed/b{b_idx}",
                )
    else:
        logger.info("No 'pixel_values' found in batch. Skipping processed image saving.")

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 80)
    logger.info(f"✅ Test completed successfully!")
    logger.info(f"   Raw frames:      {raw_dir}")
    logger.info(f"   Processed batch: {processed_dir}")
    logger.info(f"   All images:      {images_dir}")
    logger.info("=" * 80)

    # Cleanup
    if hasattr(train_dataset, "close"):
        train_dataset.close()


if __name__ == "__main__":
    main()
