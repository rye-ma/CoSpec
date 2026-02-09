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

from utils.utils import dict_to_object, setup_seed, load_next_location_prediction_data
from pretrain.datasets.pretrain_dataset import load_city_dataset
from datasets.next_location_prediction_dataset import NextLocationPredictionDataset
from models.next_location_prediction_model import NextLocationPredictionModel


def collate_fn_train(batch):
    src_list, label_list, length_list = zip(*batch)
    lengths = torch.tensor(length_list, dtype=torch.long)

    max_len = lengths.max().item()

    padded_src = []
    padded_label = []
    for src, label in zip(src_list, label_list):
        L = src.size(0)
        pad_len = max_len - L
        if pad_len > 0:
            src_pad = F.pad(src, (0, pad_len), value=0)
            label_pad = F.pad(label, (0, pad_len), value=0)
        else:
            src_pad = src
            label_pad = label
        padded_src.append(src_pad)
        padded_label.append(label_pad)

    src_padded = torch.stack(padded_src, dim=0)
    label_padded = torch.stack(padded_label, dim=0)

    return src_padded, label_padded, lengths


def collate_fn_test(batch):
    src_list, label_list, length_list = zip(*batch)
    lengths = torch.tensor(length_list, dtype=torch.long)
    labels = torch.stack(label_list, dim=0)

    max_len = lengths.max().item()
    padded_src = []
    for src in src_list:
        L = src.size(0)
        pad_len = max_len - L
        if pad_len > 0:
            src_pad = F.pad(src, (0, pad_len), value=0)
        else:
            src_pad = src
        padded_src.append(src_pad)

    src_padded = torch.stack(padded_src, dim=0)

    return src_padded, labels, lengths


@torch.no_grad()
def evaluate(model, city_data, city_name, data_loader, device):
    model.eval()
    right_1 = 0
    right_5 = 0
    total = 0

    for src, labels, lengths in data_loader:
        src = src.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        logits = model(city_data, city_name, src, lengths, labels=None)
        probs = F.softmax(logits, dim=-1)

        top1 = probs.argmax(dim=-1)
        right_1 += (top1 == labels).sum().item()

        _, top5 = probs.topk(5, dim=-1)
        match_top5 = (top5 == labels.unsqueeze(-1)).any(dim=-1)
        right_5 += match_top5.sum().item()

        total += labels.size(0)

    acc1 = right_1 / max(1, total)
    acc5 = right_5 / max(1, total)
    return acc1, acc5


def train_for_city(city_name, city_cfg, next_cfg, backbone_cfg, city_list,
                   pretrain_ckpt_path, device, logger):
    logger.info(f"\n========== Training NEXT-LOCATION for city: {city_name} ==========")

    city_data = load_city_dataset(next_cfg, city_name).to(device)

    train_next_set, test_next_set = load_next_location_prediction_data(city_cfg)

    train_dataset = NextLocationPredictionDataset(
        raw_batches=train_next_set,
        mode="train",
        min_len=city_cfg.train_min_len,
        max_len=city_cfg.train_max_len,
    )
    test_dataset = NextLocationPredictionDataset(
        raw_batches=test_next_set,
        mode="test",
        min_len=city_cfg.test_min_len,
        max_len=city_cfg.test_max_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=next_cfg.next_train.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_train,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=next_cfg.next_train.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_test,
    )

    num_segments = city_cfg.num_segments
    freeze_backbone = next_cfg.next_model.freeze_backbone
    logger.info(f"[{city_name}] freeze_backbone(backbone) = {freeze_backbone}")

    model = NextLocationPredictionModel(
        backbone_cfg=backbone_cfg,
        city_list=city_list,
        num_segments=num_segments,
        hidden_dim_head=next_cfg.next_model.hidden_dim,
        num_layers_head=next_cfg.next_model.num_layers,
        dropout_head=next_cfg.next_model.dropout,
        freeze_backbone=freeze_backbone,
    ).to(device)

    ckpt = torch.load(pretrain_ckpt_path, map_location="cpu")
    model.backbone.load_state_dict(ckpt["model_state"])
    logger.info(f"[{city_name}] Loaded backbone from {pretrain_ckpt_path}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=next_cfg.next_train.lr,
        weight_decay=next_cfg.next_train.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_acc1 = 0.0
    best_acc5 = 0.0

    root_save_dir = getattr(next_cfg, "save_dir", "./next_location_ckpt")
    city_save_dir = os.path.join(root_save_dir, city_name)
    os.makedirs(city_save_dir, exist_ok=True)

    for epoch in tqdm(range(1, next_cfg.next_train.epochs + 1), desc=f"[{city_name}] Next Location Prediction Training"):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for src, label_seq, lengths in train_loader:

            src = src.to(device)
            label_seq = label_seq.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            logits_seq = model(city_data, city_name, src, lengths, labels=label_seq)

            B, T_max, C = logits_seq.shape
            time_ids = torch.arange(T_max, device=device).unsqueeze(0).expand(B, -1)
            mask = time_ids < lengths.unsqueeze(1)

            logits_flat = logits_seq[mask]
            labels_flat = label_seq[mask]

            loss = criterion(logits_flat, labels_flat)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * logits_flat.size(0)
            total_tokens += logits_flat.size(0)

        train_loss = total_loss / max(1, total_tokens)

        acc1, acc5 = evaluate(model, city_data, city_name, test_loader, device)

        if acc1 > best_acc1:
            best_acc1 = acc1
            best_acc5 = acc5
            ckpt_path = os.path.join(city_save_dir, "best_next_model.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_acc1": best_acc1,
                    "best_acc5": best_acc5,
                    "city": city_name,
                    "city_config": city_cfg.__dict__,
                    "next_config": next_cfg.__dict__,
                },
                ckpt_path,
            )
            logger.info(
                f"[{city_name}]  -> New BEST ACC@1={best_acc1:.4f}, "
                f"ACC@5={best_acc5:.4f}, saved to {ckpt_path}"
            )

        logger.info(
            f"[{city_name}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"Top-1={acc1:.4f} Top-5={acc5:.4f} | "
            f"Best Top-1={best_acc1:.4f} Best Top-5={best_acc5:.4f}"
        )

    logger.info(f"[{city_name}] Training completed. Best ACC@1={best_acc1:.4f}, ACC@5={best_acc5:.4f}")


def main(args):
    with open(args.config, "r") as f:
        next_cfg_raw = yaml.safe_load(f)
    next_cfg = dict_to_object(next_cfg_raw)

    with open(args.pretrain_config, "r") as f:
        backbone_cfg_raw = yaml.safe_load(f)
    backbone_cfg = dict_to_object(backbone_cfg_raw)

    seed = getattr(args, "seed", getattr(next_cfg, "seed", 42))
    setup_seed(seed)
    next_cfg.seed = seed

    if isinstance(next_cfg.device, str):
        if next_cfg.device.lower() == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device(next_cfg.device)
    else:
        if next_cfg.device >= 0 and torch.cuda.is_available():
            device = torch.device(f"cuda:{next_cfg.device}")
        else:
            device = torch.device("cpu")

    print("Using device:", device)

    beijing_tz = timezone(timedelta(hours=8))
    timestamp = datetime.now(beijing_tz).strftime("%Y%m%d_%H%M%S")
    run_name = f"nextpred_{timestamp}"

    exp_root = getattr(next_cfg, "exp_root", "./runs_next_location")
    run_dir = os.path.join(exp_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "experiment.log")
    ckpt_root_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_root_dir, exist_ok=True)
    next_cfg.save_dir = ckpt_root_dir

    with open(os.path.join(run_dir, "next_cfg.yml"), "w") as f:
        yaml.safe_dump(next_cfg_raw, f, sort_keys=False, allow_unicode=True)
    with open(os.path.join(run_dir, "backbone_cfg.yml"), "w") as f:
        yaml.safe_dump(backbone_cfg_raw, f, sort_keys=False, allow_unicode=True)
    args_dict = vars(args)
    with open(os.path.join(run_dir, "args.yml"), "w") as f:
        yaml.safe_dump(args_dict, f, sort_keys=False, allow_unicode=True)

    logger = logging.getLogger("next_pred")
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
    logger.info(f"Next-location config: {args.config}")
    logger.info(f"Pretrain config: {args.pretrain_config}")
    logger.info(f"Pretrain checkpoint: {args.pretrain_ckpt}")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Checkpoint root dir: {ckpt_root_dir}")

    cities_topo_raw = backbone_cfg_raw["cities"]
    city_list = list(cities_topo_raw.keys())
    logger.info(f"Backbone city_list (pretrain): {city_list}")

    all_city_cfg_raw = next_cfg_raw["cities"]

    if args.cities is None or args.cities.lower() == "all":
        target_cities = list(all_city_cfg_raw.keys())
    else:
        target_cities = [c.strip() for c in args.cities.split(",") if c.strip()]

    logger.info(f"Target cities to train (from args): {target_cities}")

    valid_target_cities = []
    for city_name in target_cities:
        if city_name not in all_city_cfg_raw:
            logger.warning(f"City '{city_name}' not found in next config, skip.")
        else:
            valid_target_cities.append(city_name)

    if not valid_target_cities:
        logger.error("No valid cities to train. Please check --cities and config.")
        return

    logger.info(f"Final valid target cities: {valid_target_cities}")

    for city_name in valid_target_cities:
        city_cfg_dict = all_city_cfg_raw[city_name]
        city_cfg = dict_to_object(city_cfg_dict)
        train_for_city(
            city_name=city_name,
            city_cfg=city_cfg,
            next_cfg=next_cfg,
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
        default="./chengdu/configs/next_location_prediction.yml",
    )
    parser.add_argument(
        "--pretrain_config",
        type=str,
        default="./pretrain/configs/pretrain.yml",
    )
    parser.add_argument(
        "--pretrain_ckpt",
        type=str,
        default="path/to/pretrain_ckpt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--cities",
        type=str,
        default="cd",
    )
    args = parser.parse_args()
    main(args)