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

from utils.utils import (
    dict_to_object,
    setup_seed,
    load_label_prediction_data,
)
from pretrain.datasets.pretrain_dataset import load_city_dataset
from datasets.label_prediction_dataset import LabelPredictionDataset
from models.label_prediction_model import LabelPredictionModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

@torch.no_grad()
def evaluate(model, city_data, city_name, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for seg_ids, labels in data_loader:
        seg_ids = seg_ids.to(device)
        labels = labels.to(device)

        logits = model(city_data, city_name, seg_ids)
        probs = F.softmax(logits, dim=-1)

        preds = probs.argmax(dim=-1)
        prob_pos = probs[:, 1]

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_probs.append(prob_pos.cpu())

    if len(all_labels) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    return accuracy, precision, recall, f1, auc



def train_for_city(city_name, city_cfg, label_cfg, backbone_cfg, city_list, pretrain_ckpt_path, device, logger):
    logger.info(f"\n========== Training label prediction for city: {city_name} ==========")

    city_data = load_city_dataset(label_cfg, city_name).to(device)

    train_seg_ids, train_labels, test_seg_ids, test_labels = load_label_prediction_data(city_cfg)

    train_dataset = LabelPredictionDataset(train_seg_ids, train_labels)
    test_dataset  = LabelPredictionDataset(test_seg_ids,  test_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=label_cfg.label_train.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=label_cfg.label_train.batch_size,
        shuffle=False,
        num_workers=0,
    )

    num_segments = city_cfg.num_segments

    num_classes = getattr(label_cfg.label_model, "num_classes", 2)

    freeze_backbone = label_cfg.label_model.freeze_backbone
    logger.info(f"[{city_name}] freeze_backbone(backbone) = {freeze_backbone}")

    model = LabelPredictionModel(
        backbone_cfg=backbone_cfg,
        city_list=city_list,
        num_segments=num_segments,
        num_classes=num_classes,
        hidden_dim_head=label_cfg.label_model.hidden_dim,
        num_layers_head=label_cfg.label_model.num_layers,
        dropout_head=label_cfg.label_model.dropout,
        freeze_backbone=freeze_backbone,
    ).to(device)

    ckpt = torch.load(pretrain_ckpt_path, map_location="cpu")
    model.backbone.load_state_dict(ckpt["model_state"])
    logger.info(f"[{city_name}] Loaded backbone from {pretrain_ckpt_path}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=label_cfg.label_train.lr,
        weight_decay=label_cfg.label_train.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_auc = 0.0

    root_save_dir = getattr(label_cfg, "save_dir", "./label_prediction_ckpt")
    city_save_dir = os.path.join(root_save_dir, city_name)
    os.makedirs(city_save_dir, exist_ok=True)

    for epoch in tqdm(range(1, label_cfg.label_train.epochs + 1), desc=f"[{city_name}] Label Prediction Training"):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for seg_ids, labels in train_loader:
            seg_ids = seg_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(city_data, city_name, seg_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * seg_ids.size(0)
            total_samples += seg_ids.size(0)

        train_loss = total_loss / max(1, total_samples)

        accuracy, precision, recall, f1, auc = evaluate(model, city_data, city_name, test_loader, device)

        if f1 > best_f1 or auc > best_auc:
            if f1 > best_f1:
                best_f1 = f1
            if auc > best_auc:
                best_auc = auc
            ckpt_path = os.path.join(city_save_dir, "best_label_model.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "best_auc": best_auc,
                    "city": city_name,
                    "city_config": city_cfg.__dict__,
                    "label_config": label_cfg.__dict__,
                },
                ckpt_path,
            )
            logger.info(
                f"[{city_name}]  -> New best F1={best_f1:.4f}, best AUC={best_auc:.4f}, saved to {ckpt_path}"
            )

        logger.info(
            f"[{city_name}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | Accuracy={accuracy:.4f} | Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f} | AUC={auc:.4f} | Best F1={best_f1:.4f} | Best AUC={best_auc:.4f}"
        )

    logger.info(f"[{city_name}] Training completed. Best F1={best_f1:.4f}, Best AUC={best_auc:.4f}")


def main(args):
    with open(args.config, "r") as f:
        label_cfg_raw = yaml.safe_load(f)
    label_cfg = dict_to_object(label_cfg_raw)

    with open(args.pretrain_config, "r") as f:
        backbone_cfg_raw = yaml.safe_load(f)
    backbone_cfg = dict_to_object(backbone_cfg_raw)

    seed = getattr(args, "seed", getattr(label_cfg, "seed", 42))
    setup_seed(seed)
    label_cfg.seed = seed

    if isinstance(label_cfg.device, str):
        if label_cfg.device.lower() == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device(label_cfg.device)
    else:
        if label_cfg.device >= 0 and torch.cuda.is_available():
            device = torch.device(f"cuda:{label_cfg.device}")
        else:
            device = torch.device("cpu")

    print("Using device:", device)

    beijing_tz = timezone(timedelta(hours=8))
    timestamp = datetime.now(beijing_tz).strftime("%Y%m%d_%H%M%S")

    run_name = f"labelpred_{timestamp}"

    exp_root = getattr(label_cfg, "exp_root", "./runs_label")
    run_dir = os.path.join(exp_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "experiment.log")

    ckpt_root_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_root_dir, exist_ok=True)
    label_cfg.save_dir = ckpt_root_dir

    with open(os.path.join(run_dir, "label_cfg.yml"), "w") as f:
        yaml.safe_dump(label_cfg_raw, f, sort_keys=False, allow_unicode=True)
    with open(os.path.join(run_dir, "backbone_cfg.yml"), "w") as f:
        yaml.safe_dump(backbone_cfg_raw, f, sort_keys=False, allow_unicode=True)

    args_dict = vars(args)
    with open(os.path.join(run_dir, "args.yml"), "w") as f:
        yaml.safe_dump(args_dict, f, sort_keys=False, allow_unicode=True)

    logging.Formatter.converter = lambda *ct: (datetime.utcnow() + timedelta(hours=8)).timetuple()

    logger = logging.getLogger("label_pred")
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
    logger.info(f"Label config: {args.config}")
    logger.info(f"Pretrain config: {args.pretrain_config}")
    logger.info(f"Pretrain checkpoint: {args.pretrain_ckpt}")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Checkpoint root dir: {ckpt_root_dir}")

    cities_topo_raw = backbone_cfg_raw["cities"]
    city_list = list(cities_topo_raw.keys())
    logger.info(f"Backbone city_list (pretrain): {city_list}")

    all_city_cfg_raw = label_cfg_raw["cities"]

    if args.cities is None or args.cities.lower() == "all":
        target_cities = list(all_city_cfg_raw.keys())
    else:
        target_cities = [c.strip() for c in args.cities.split(",") if c.strip()]

    logger.info(f"Target cities to train (from args): {target_cities}")

    valid_target_cities = []
    for city_name in target_cities:
        if city_name not in all_city_cfg_raw:
            logger.warning(f"City '{city_name}' not found in label config, skip.")
        else:
            valid_target_cities.append(city_name)

    if not valid_target_cities:
        logger.error("No valid cities to train. Please check --cities and label config.")
        return

    logger.info(f"Final valid target cities: {valid_target_cities}")

    for city_name in valid_target_cities:
        city_cfg_dict = all_city_cfg_raw[city_name]
        city_cfg = dict_to_object(city_cfg_dict)
        train_for_city(
            city_name=city_name,
            city_cfg=city_cfg,
            label_cfg=label_cfg,
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
        default="./beijing/configs/label_prediction.yml",
        help="Path to label prediction config file",
    )
    parser.add_argument(
        "--pretrain_config",
        type=str,
        default="./pretrain/configs/pretrain.yml",
        help="Path to backbone pretrain config file",
    )
    parser.add_argument(
        "--pretrain_ckpt",
        type=str,
        default="./pretrain/runs/repr_20251203_145530/checkpoints/repr_best.pt",
        help="Path to backbone pretrain checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--cities",
        type=str,
        default="bj",
        help="Comma-separated list of cities to train, e.g. 'bj,cd'. "
             "If not set or 'all', train all cities in config.",
    )
    args = parser.parse_args()
    main(args)
