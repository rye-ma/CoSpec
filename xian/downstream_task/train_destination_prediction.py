import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse

import yaml
from tqdm import tqdm
import logging
from datetime import datetime, timedelta, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.utils import dict_to_object, setup_seed, load_destination_prediction_data
from pretrain.datasets.pretrain_dataset import load_city_dataset
from datasets.destination_prediction_dataset import DestinationPredictionDataset
from models.destination_prediction_model import DestinationPredictionModel


@torch.no_grad()
def evaluate(model, city_data, city_name, data_loader, device):
    model.eval()
    right_1 = 0
    right_5 = 0
    total_num = 0

    for src, label in data_loader:
        src = src.to(device)
        label = label.to(device)

        logits = model(city_data, city_name, src)
        probs = F.softmax(logits, dim=-1)

        top1 = probs.argmax(dim=-1)
        right_1 += (top1 == label).sum().item()

        _, top5 = probs.topk(5, dim=-1)
        match_top5 = (top5 == label.unsqueeze(-1)).any(dim=-1)
        right_5 += match_top5.sum().item()

        total_num += label.size(0)

    acc1 = right_1 / max(1, total_num)
    acc5 = right_5 / max(1, total_num)
    return acc1, acc5


def train_for_city(city_name, city_cfg, dest_cfg, backbone_cfg, city_list, pretrain_ckpt_path, device, logger):
    logger.info(f"\n========== Training destination prediction for city: {city_name} ==========")

    city_data = load_city_dataset(dest_cfg, city_name).to(device)

    train_des_set, test_des_set = load_destination_prediction_data(city_cfg)

    train_dataset = DestinationPredictionDataset(
        raw_batches=train_des_set,
        input_len=city_cfg.input_len,
        min_len=city_cfg.train_min_len,
        max_len=city_cfg.train_max_len,
    )
    test_dataset = DestinationPredictionDataset(
        raw_batches=test_des_set,
        input_len=city_cfg.input_len,
        min_len=city_cfg.test_min_len,
        max_len=city_cfg.test_max_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=dest_cfg.dest_train.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=dest_cfg.dest_train.batch_size,
        shuffle=False,
        num_workers=0,
    )

    num_segments = city_cfg.num_segments

    freeze_backbone = dest_cfg.dest_model.freeze_backbone
    logger.info(f"[{city_name}] freeze_backbone(backbone) = {freeze_backbone}")

    model = DestinationPredictionModel(
        backbone_cfg=backbone_cfg,
        city_list=city_list,
        num_segments=num_segments,
        hidden_dim_head=dest_cfg.dest_model.hidden_dim,
        num_layers_head=dest_cfg.dest_model.num_layers,
        dropout_head=dest_cfg.dest_model.dropout,
        freeze_backbone=freeze_backbone,
    ).to(device)

    ckpt = torch.load(pretrain_ckpt_path, map_location="cpu")
    model.backbone.load_state_dict(ckpt["model_state"])
    logger.info(f"[{city_name}] Loaded backbone from {pretrain_ckpt_path}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=dest_cfg.dest_train.lr,
        weight_decay=dest_cfg.dest_train.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_acc1 = 0.0
    best_acc5 = 0.0

    root_save_dir = getattr(dest_cfg, "save_dir", "./destination_prediction_ckpt")
    city_save_dir = os.path.join(root_save_dir, city_name)
    os.makedirs(city_save_dir, exist_ok=True)

    for epoch in tqdm(range(1, dest_cfg.dest_train.epochs + 1), desc=f"[{city_name}] Destination Prediction Training"):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for src, label in train_loader:
            src = src.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            logits = model(city_data, city_name, src)
            
            loss = criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), dest_cfg.dest_train.clip)
            optimizer.step()

            total_loss += loss.item() * src.size(0)
            total_samples += src.size(0)

        train_loss = total_loss / max(1, total_samples)

        acc1, acc5 = evaluate(model, city_data, city_name, test_loader, device)

        if acc1 > best_acc1 or acc5 > best_acc5:
            best_acc1 = max(acc1, best_acc1)
            best_acc5 = max(acc5, best_acc5)
            ckpt_path = os.path.join(city_save_dir, "best_dest_model.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_acc1": best_acc1,
                    "best_acc5": best_acc5,
                    "city": city_name,
                    "city_config": city_cfg.__dict__,
                    "dest_config": dest_cfg.__dict__,
                },
                ckpt_path,
            )
            logger.info(
                f"[{city_name}]  -> New best ACC@1={best_acc1:.4f}, "
                f"ACC@5={best_acc5:.4f}, saved to {ckpt_path}"
            )

        logger.info(
            f"[{city_name}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | Top-1={acc1:.4f} Top-5={acc5:.4f} | Best Top-1={best_acc1:.4f} | Best Top-5={best_acc5:.4f}"
        )

    logger.info(f"[{city_name}] Training completed. Best ACC@1={best_acc1:.4f}, ACC@5={best_acc5:.4f}")


def main(args):
    with open(args.config, "r") as f:
        dest_cfg_raw = yaml.safe_load(f)
    dest_cfg = dict_to_object(dest_cfg_raw)

    with open(args.pretrain_config, "r") as f:
        backbone_cfg_raw = yaml.safe_load(f)
    backbone_cfg = dict_to_object(backbone_cfg_raw)

    seed = getattr(args, "seed", getattr(dest_cfg, "seed", 42))
    setup_seed(seed)
    dest_cfg.seed = seed

    if isinstance(dest_cfg.device, str):
        if dest_cfg.device.lower() == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device(dest_cfg.device)
    else:
        if dest_cfg.device >= 0 and torch.cuda.is_available():
            device = torch.device(f"cuda:{dest_cfg.device}")
        else:
            device = torch.device("cpu")

    print("Using device:", device)

    beijing_tz = timezone(timedelta(hours=8))
    timestamp = datetime.now(beijing_tz).strftime("%Y%m%d_%H%M%S")

    run_name = f"destpred_{timestamp}"

    exp_root = getattr(dest_cfg, "exp_root", "./runs_destination")
    run_dir = os.path.join(exp_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "experiment.log")

    ckpt_root_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_root_dir, exist_ok=True)
    dest_cfg.save_dir = ckpt_root_dir

    with open(os.path.join(run_dir, "dest_cfg.yml"), "w") as f:
        yaml.safe_dump(dest_cfg_raw, f, sort_keys=False, allow_unicode=True)
    with open(os.path.join(run_dir, "backbone_cfg.yml"), "w") as f:
        yaml.safe_dump(backbone_cfg_raw, f, sort_keys=False, allow_unicode=True)

    args_dict = vars(args)
    with open(os.path.join(run_dir, "args.yml"), "w") as f:
        yaml.safe_dump(args_dict, f, sort_keys=False, allow_unicode=True)

    logger = logging.getLogger("dest_pred")
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

    logger.info(f"=== New run: {run_name} ===")
    logger.info(f"Using device: {device}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Destination config: {args.config}")
    logger.info(f"Pretrain config: {args.pretrain_config}")
    logger.info(f"Pretrain checkpoint: {args.pretrain_ckpt}")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Checkpoint root dir: {ckpt_root_dir}")

    cities_topo_raw = backbone_cfg_raw["cities"]
    city_list = list(cities_topo_raw.keys())
    logger.info(f"Backbone city_list (pretrain): {city_list}")

    all_city_cfg_raw = dest_cfg_raw["cities"]

    if args.cities is None or args.cities.lower() == "all":
        target_cities = list(all_city_cfg_raw.keys())
    else:
        target_cities = [c.strip() for c in args.cities.split(",") if c.strip()]

    logger.info(f"Target cities to train (from args): {target_cities}")

    valid_target_cities = []
    for city_name in target_cities:
        if city_name not in all_city_cfg_raw:
            logger.warning(f"City '{city_name}' not found in destination config, skip.")
        else:
            valid_target_cities.append(city_name)

    if not valid_target_cities:
        logger.error("No valid cities to train. Please check --cities and destination config.")
        return

    logger.info(f"Final valid target cities: {valid_target_cities}")

    for city_name in valid_target_cities:
        city_cfg_dict = all_city_cfg_raw[city_name]
        city_cfg = dict_to_object(city_cfg_dict)
        train_for_city(
            city_name=city_name,
            city_cfg=city_cfg,
            dest_cfg=dest_cfg,
            backbone_cfg=backbone_cfg,
            city_list=city_list,
            pretrain_ckpt_path=args.pretrain_ckpt,
            device=device,
            logger=logger
        )

    logger.info(f"=== Run finished: {run_name} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./xian/configs/destination_prediction.yml",
    )
    parser.add_argument(
        "--pretrain_config",
        type=str,
        default="./pretrain/configs/pretrain.yml",
    )
    parser.add_argument(
        "--pretrain_ckpt",
        type=str,
        default=".",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--cities",
        type=str,
        default="xa",
    )
    args = parser.parse_args()
    main(args)