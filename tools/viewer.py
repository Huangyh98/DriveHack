"""
Standalone viser viewer for browsing 3D Gaussians without running evaluation.
Usage:
    export PYTHONPATH=$(pwd)
    python tools/viewer.py --resume_from outputs/waymo_omnire/scene23/checkpoint_final.pth
"""
from omegaconf import OmegaConf
import os
import time
import logging
import argparse

import torch
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def main(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build dataset
    logger.info("Loading dataset...")
    dataset = DrivingDataset(data_cfg=cfg.data)

    # setup trainer
    logger.info("Setting up trainer...")
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )

    # Resume from checkpoint
    trainer.resume_from_checkpoint(
        ckpt_path=args.resume_from,
        load_only_model=True,
    )
    trainer.set_eval()
    logger.info(f"Loaded checkpoint from {args.resume_from}")

    # Start viewer
    trainer.init_viewer(port=args.port)
    logger.info(f"Viewer running at http://localhost:{args.port}")
    logger.info("Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down viewer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Standalone 3DGS Viewer")
    parser.add_argument(
        "--resume_from", type=str, required=True,
        help="Path to checkpoint, e.g. outputs/waymo_omnire/scene23/checkpoint_final.pth",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viewer port")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, default=None,
        help="Override config options",
    )
    args = parser.parse_args()
    main(args)
