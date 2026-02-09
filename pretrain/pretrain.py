import os
import random
import sys
from typing import Dict, Any
import numpy as np
from tqdm import tqdm
import argparse
import yaml
import logging
from datetime import datetime, timedelta, timezone

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
print(os.path.abspath(os.path.dirname(__file__)))

import torch

from sklearn.cluster import KMeans
from torch_geometric.data import Data

from models.pretrain_model import *
from utils.utils import *
from datasets.pretrain_dataset import *
from utils.losses import *


def collect_all_regions(model, city_data: Dict[str, Data], cfg):
    
    model.eval()
    regions = []
    with torch.no_grad():
        for c, data in city_data.items():
            data = data.to(cfg.device)
            out = model(data, city_name=c)
            region_features = out["region_features"]
            regions.append(region_features.detach().cpu())
    if len(regions) == 0:
        return None
    all_R = torch.cat(regions, dim=0)
    return all_R.numpy()

def train_model(model: CrossCityReprModel, city_data: Dict[str, Data], cfg, logger: logging.Logger):
    model.to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay
    )
    os.makedirs(cfg.save_dir, exist_ok=True)

    best_avg_loss = float("inf")
    best_ckpt_path = os.path.join(cfg.save_dir, "repr_best.pt")

    for epoch in range(1, cfg.train.epochs + 1):
        all_R_np = collect_all_regions(model, city_data, cfg)
        if all_R_np is None:
            raise RuntimeError("No regions collected. Check city_data.")

        kmeans = KMeans(
            n_clusters=cfg.model.prototype_k,
            random_state=cfg.seed,
            n_init=10
        )
        kmeans.fit(all_R_np)
        prototypes = torch.tensor(
            kmeans.cluster_centers_,
            dtype=torch.float32,
            device=cfg.device
        )

        model.train()
        total_loss = 0.0
        num_cities = 0

        for c, data in city_data.items():
            num_cities += 1
            data = data.to(cfg.device)
            out = model(data, city_name=c)

            init_segment_features = out["init_segment_features"]
            reconstructed_segment_features = out["reconstructed_segment_features"]
            low_segment_features = out["low_segment_features"]
            high_segment_features = out["high_segment_features"]
            region_features = out["region_features"]
            segment2region_assignment = out["segment2region_assignment"]

            L_rec = reconstruction_loss(
                init_segment_features,
                reconstructed_segment_features
            )
            L_ortho = orthogonality_loss(
                low_segment_features,
                high_segment_features
            )
            L_smooth, L_topo, L_traj = combined_smoothness_loss(
                low_segment_features,
                data.edge_index,
                getattr(data, "traj_edge_index", None),
                getattr(data, "traj_edge_weight", None),
                alpha_topo=1.0,
                beta_traj=1.0,
            )

            L_proto, proto_stats = prototype_loss_v2(
                region_features,
                prototypes,
                tau=0.1,
                w_contrast=1.0,
                w_soft=0.1,
                w_balance=0.1,
            )


            loss = (
                cfg.loss_weights.lambda_rec * L_rec
                + cfg.loss_weights.lambda_ortho * L_ortho
                + cfg.loss_weights.lambda_smooth * L_smooth
                + cfg.loss_weights.lambda_cons * L_proto
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, num_cities)
        logger.info(
            f"[Epoch {epoch:03d}/{cfg.train.epochs}] "
            f"avg_loss={avg_loss:.6f}  "
            f"(L_rec={L_rec.item():.4f} "
            f"L_ortho={L_ortho.item():.4f} "
            f"L_smooth={L_smooth.item():.4f} "
            f"L_proto={L_proto.item():.4f})"
        )

        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            ckpt = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_avg_loss": best_avg_loss,
            }
            torch.save(ckpt, best_ckpt_path)
            logger.info(
                f"  -> New best avg_loss={best_avg_loss:.6f}, "
                f"checkpoint saved to {best_ckpt_path}"
            )

    export_dir = os.path.join(cfg.run_dir, "output")
    os.makedirs(export_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for c, data in city_data.items():
            data = data.to(cfg.device)
            out = model(data, city_name=c)

            s2r = segment2region_assignment.detach().cpu().numpy()
            s2r_soft = s2r.transpose(1, 0)
            s2r_hard = s2r_soft.argmax(axis=1).astype(np.int64)

            region_feats = out["region_features"].detach().cpu().numpy()
            seg_feats = out["reconstructed_segment_features"].detach().cpu().numpy()

            np.save(os.path.join(export_dir, f"{c}_s2r_soft.npy"), s2r_soft)
            np.save(os.path.join(export_dir, f"{c}_s2r_hard.npy"), s2r_hard)
            np.save(os.path.join(export_dir, f"{c}_region_feats.npy"), region_feats)
            np.save(os.path.join(export_dir, f"{c}_segment_feats.npy"), seg_feats)

            logger.info(
                f"[Final Export] {c}: s2r_soft {s2r_soft.shape}, "
                f"s2r_hard {s2r_hard.shape}, region_feats {region_feats.shape}, "
                f"segment_feats {seg_feats.shape}"
            )


def main(args):

    setup_seed(args.seed)

    with open(args.config, "r") as f:
        cfg_raw = yaml.safe_load(f)
    cfg = dict_to_object(cfg_raw)
    cfg.seed = args.seed
    cfg.device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")

    beijing_tz = timezone(timedelta(hours=8))
    timestamp = datetime.now(beijing_tz).strftime("%Y%m%d_%H%M%S")
    run_name = f"repr_{timestamp}"

    exp_root = getattr(cfg, "exp_root", "./runs_pretrain")
    run_dir = os.path.join(exp_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    cfg.run_dir = run_dir

    log_path = os.path.join(run_dir, "experiment.log")
    ckpt_root_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_root_dir, exist_ok=True)

    cfg.save_dir = ckpt_root_dir

    with open(os.path.join(run_dir, "pretrain_cfg.yml"), "w") as f:
        yaml.safe_dump(cfg_raw, f, sort_keys=False, allow_unicode=True)
    args_dict = vars(args)
    with open(os.path.join(run_dir, "args.yml"), "w") as f:
        yaml.safe_dump(args_dict, f, sort_keys=False, allow_unicode=True)

    logger = logging.getLogger("repr_pretrain")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"=== New pretrain run: {run_name} ===")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Checkpoint dir: {ckpt_root_dir}")
    logger.info(f"Device: {cfg.device}")
    logger.info(f"Seed: {cfg.seed}")
    logger.info(f"Configuration (dict): {cfg_raw}")

    cities = ["bj", "cd", "xa"]
    city_data = {}
    for city_name in cities:
        data = load_city_dataset(cfg, city_name)
        city_data[city_name] = data
    logger.info(f"Datasets loaded for cities: {list(city_data.keys())}")

    model = CrossCityReprModel(cfg, city_list=cities)
    logger.info("Model created. Training start ...")

    train_model(model, city_data, cfg, logger)

    logger.info("Training finished.")
    logger.info(f"Best checkpoint saved under: {ckpt_root_dir}")
    logger.info(f"Final outputs exported to: {os.path.join(run_dir, 'output')}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--config",
        type=str,
        default="./pretrain/configs/pretrain.yml",
        help="Path to config file"
    )

    args = parser.parse_args()
    main(args)
 