import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import logging
from datetime import datetime, timedelta, timezone

import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils.utils import dict_to_object, setup_seed, load_route_plan_data, edit_distance
from pretrain.datasets.pretrain_dataset import load_city_dataset
from datasets.route_plan_dataset import RoutePlanDataset
from models.route_plan_model import RoutePlanModel

PAD_ID = -1


class RoutePlanEvalDataset(Dataset):
    def __init__(
        self,
        raw_batches,
        min_len: int = 2,
        max_len: int = 1000,
    ):
        super().__init__()

        self.trajs = []
        self.lengths = []

        for batch in raw_batches:
            if len(batch) == 0:
                continue
            for traj in batch:
                traj = np.asarray(traj, dtype=np.int64)
                L = len(traj)
                if L < min_len or L > max_len:
                    continue
                self.trajs.append(traj)
                self.lengths.append(L)

        print(
            f"[RoutePlanEvalDataset] collected {len(self.trajs)} samples "
            f"(min_len={min_len}, max_len={max_len})"
        )

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        traj_np = self.trajs[idx]
        length = self.lengths[idx]
        traj = torch.tensor(traj_np, dtype=torch.long)
        return traj, length


def collate_fn_train(batch):
    src_list, label_list, length_list = zip(*batch)
    lengths = torch.tensor(length_list, dtype=torch.long)
    B = len(src_list)
    max_len = lengths.max().item()

    src_pad = torch.full((B, max_len), PAD_ID, dtype=torch.long)
    label_pad = torch.full((B, max_len), PAD_ID, dtype=torch.long)

    for i, (src, label) in enumerate(zip(src_list, label_list)):
        L = src.size(0)
        src_pad[i, :L] = src
        label_pad[i, :L] = label

    dest_positions = torch.minimum(
        torch.full_like(lengths, 10),
        lengths - 1
    )

    dest_list = []
    for i in range(B):
        pos = dest_positions[i].item()
        dest_list.append(src_list[i][pos].item())

    dest = torch.tensor(dest_list, dtype=torch.long)

    return src_pad, label_pad, lengths, dest


def collate_fn_test(batch):
    traj_list, length_list = zip(*batch)
    lengths = torch.tensor(length_list, dtype=torch.long)
    B = len(traj_list)
    max_len = lengths.max().item()

    traj_pad = torch.full((B, max_len), PAD_ID, dtype=torch.long)
    for i, traj in enumerate(traj_list):
        L = traj.size(0)
        traj_pad[i, :L] = traj

    return traj_pad, lengths


@torch.no_grad()
def evaluate(model, city_data, city_name, test_loader, cfg, logger=None):
    model.eval()
    pred_right = 0
    recall_right = 0
    pred_sum = 0
    recall_sum = 0
    edt_sum = 0
    edt_count = 0
    pred_list = []

    device = cfg.device

    for traj_pad, lengths in test_loader:
        keep_mask = lengths >= 15
        if keep_mask.sum().item() == 0:
            continue

        traj_pad = traj_pad[keep_mask].to(device)
        lengths = lengths[keep_mask]

        L_eval = 15
        batch = traj_pad[:, :L_eval].detach().cpu().numpy()

        input_arr = batch[:, :6]
        label_arr = batch[:, 6:-1]
        destination_arr = batch[:, -1]

        destination = torch.tensor(destination_arr, dtype=torch.long, device=device)

        input_cur = input_arr.copy()

        for _ in range(batch.shape[1] - 7):
            input_tensor = torch.tensor(input_cur, dtype=torch.long, device=device)
            logits = model(city_data, city_name, input_tensor, destination)
            step_logits = logits[:, -1, :]

            pred_loc = step_logits.argmax(dim=-1).detach().cpu().numpy()
            input_cur = np.concatenate([input_cur, pred_loc[:, None]], axis=1)

        pred_batch = input_cur[:, 6:]
        pred_list.extend(pred_batch.reshape(-1).tolist())

        for traj_pred, traj_label in zip(pred_batch.tolist(), label_arr.tolist()):
            edt_sum += edit_distance(traj_pred, traj_label)
            edt_count += (len(traj_pred) + len(traj_label)) / 2.0

            for item in traj_pred:
                if item in traj_label:
                    pred_right += 1
            for item in traj_label:
                if item in traj_pred:
                    recall_right += 1

            pred_sum += len(traj_pred)
            recall_sum += len(traj_label)

    if pred_sum == 0 or recall_sum == 0 or edt_count == 0:
        if logger:
            logger.warning(f"[{city_name}] evaluate: pred_sum/recall_sum/edt_count has zero, skip metric.")
        return 0.0, 0.0, 0.0, 0.0

    precision = float(pred_right) / pred_sum
    recall = float(recall_right) / recall_sum
    f1 = (2 * precision * recall) / (precision + recall + 1e-9)
    edt = edt_sum / edt_count * 10.0

    if logger:
        logger.info(f"[{city_name}] eval pred_right={pred_right}, recall_right={recall_right}, "
                    f"pred_sum={pred_sum}, recall_sum={recall_sum}")
        logger.info(f"[{city_name}] eval unique_pred_nodes={len(set(pred_list))}")

    return precision, recall, f1, edt


def train_for_city(city_name, city_cfg, route_cfg, backbone_cfg, city_list, pretrain_ckpt_path, device, logger):
    logger.info(f"\n========== Training route plan for city: {city_name} ==========")

    city_data = load_city_dataset(route_cfg, city_name).to(device)

    train_route_set, test_route_set = load_route_plan_data(city_cfg)
    logger.info(f"[{city_name}] train_route_set size={len(train_route_set)}, "
                f"test_route_set size={len(test_route_set)}")

    train_dataset = RoutePlanDataset(
        raw_batches=train_route_set,
        mode="train",
        min_len=city_cfg.train_min_len,
        max_len=city_cfg.train_max_len,
    )

    test_dataset = RoutePlanEvalDataset(
        raw_batches=test_route_set,
        min_len=city_cfg.test_min_len,
        max_len=city_cfg.test_max_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=route_cfg.route_train.batch_size,
        shuffle=True,
        num_workers=getattr(route_cfg.route_train, "num_workers", 0),
        collate_fn=collate_fn_train,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=route_cfg.route_train.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_test,
        drop_last=False,
    )

    num_segments = city_cfg.num_segments
    freeze_backbone = route_cfg.route_model.freeze_backbone
    logger.info(f"[{city_name}] freeze_backbone(backbone) = {freeze_backbone}")

    model = RoutePlanModel(
        backbone_cfg=backbone_cfg,
        city_list=city_list,
        num_segments=num_segments,
        hidden_dim_head=route_cfg.route_model.hidden_dim,
        num_layers_head=route_cfg.route_model.num_layers,
        dropout_head=route_cfg.route_model.dropout,
        freeze_backbone=freeze_backbone,
    ).to(device)

    ckpt = torch.load(pretrain_ckpt_path, map_location="cpu")
    model.backbone.load_state_dict(ckpt["model_state"])
    logger.info(f"[{city_name}] Loaded backbone from {pretrain_ckpt_path}")

    ce_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=route_cfg.route_train.lr,
        weight_decay=route_cfg.route_train.weight_decay,
    )

    best_f1 = -1.0
    best_edt = float("inf")

    root_save_dir = getattr(route_cfg, "save_dir", "./route_plan_ckpt")
    city_save_dir = os.path.join(root_save_dir, city_name)
    os.makedirs(city_save_dir, exist_ok=True)

    for epoch in tqdm(range(1, route_cfg.route_train.epochs + 1), desc=f"[{city_name}] Route Plan Training"):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for src_pad, label_pad, lengths, destination in train_loader:
            src_pad = src_pad.to(device)
            label_pad = label_pad.to(device)
            lengths = lengths.to(device)
            destination = destination.to(device)

            B, T = src_pad.shape

            time_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            valid_mask = time_ids < lengths.unsqueeze(1)

            optimizer.zero_grad()
            logits = model(city_data, city_name, src_pad, destination)
            B2, T2, V = logits.shape
            assert T2 == T, "logits time dim must match input"

            logits_flat = logits.reshape(B2 * T2, V)
            labels_flat = label_pad.reshape(B2 * T2)
            mask_flat = valid_mask.reshape(B2 * T2)

            logits_flat = logits_flat[mask_flat]
            labels_flat = labels_flat[mask_flat]

            if logits_flat.numel() == 0:
                continue

            loss = ce_criterion(logits_flat, labels_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), route_cfg.route_train.clip)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        if total_batches == 0:
            logger.warning(f"[{city_name}] No valid batches in training epoch {epoch}.")
            continue

        train_loss = total_loss / total_batches

        precision, recall, f1, edt = evaluate(
            model, city_data, city_name, test_loader, route_cfg, logger
        )

        improved = False
        if f1 > best_f1 or edt < best_edt:
            improved = True
            best_f1 = max(best_f1, f1)
            best_edt = min(best_edt, edt)

            ckpt_path = os.path.join(city_save_dir, "route_plan_best.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "best_edt": best_edt,
                    "city": city_name,
                    "city_config": city_cfg.__dict__,
                    "route_config": route_cfg.__dict__,
                },
                ckpt_path,
            )
            logger.info(
                f"[{city_name}] -> New best F1={best_f1:.4f}, EDT={best_edt:.4f}, "
                f"saved to {ckpt_path}"
            )

        logger.info(
            f"[{city_name}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f} EDT={edt:.4f} | "
            f"Best_F1={best_f1:.4f} Best_EDT={best_edt:.4f} | "
            f"Improved={improved}"
        )

    logger.info(f"[{city_name}] Training completed. Best F1={best_f1:.4f}, Best EDT={best_edt:.4f}")


def main(args):
    with open(args.config, "r") as f:
        route_cfg_raw = yaml.safe_load(f)
    route_cfg = dict_to_object(route_cfg_raw)

    with open(args.pretrain_config, "r") as f:
        backbone_cfg_raw = yaml.safe_load(f)
    backbone_cfg = dict_to_object(backbone_cfg_raw)

    seed = getattr(args, "seed", getattr(route_cfg, "seed", 42))
    setup_seed(seed)
    route_cfg.seed = seed

    if isinstance(route_cfg.device, str):
        if route_cfg.device.lower() == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device(route_cfg.device)
    else:
        if route_cfg.device >= 0 and torch.cuda.is_available():
            device = torch.device(f"cuda:{route_cfg.device}")
        else:
            device = torch.device("cpu")

    print("Using device:", device)

    beijing_tz = timezone(timedelta(hours=8))
    timestamp = datetime.now(beijing_tz).strftime("%Y%m%d_%H%M%S")
    run_name = f"routeplan_{timestamp}"

    exp_root = getattr(route_cfg, "exp_root", "./runs_route_plan")
    run_dir = os.path.join(exp_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "experiment.log")
    ckpt_root_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_root_dir, exist_ok=True)

    route_cfg.save_dir = ckpt_root_dir

    with open(os.path.join(run_dir, "route_cfg.yml"), "w") as f:
        yaml.safe_dump(route_cfg_raw, f, sort_keys=False, allow_unicode=True)
    with open(os.path.join(run_dir, "backbone_cfg.yml"), "w") as f:
        yaml.safe_dump(backbone_cfg_raw, f, sort_keys=False, allow_unicode=True)
    args_dict = vars(args)
    with open(os.path.join(run_dir, "args.yml"), "w") as f:
        yaml.safe_dump(args_dict, f, sort_keys=False, allow_unicode=True)

    logger = logging.getLogger("route_plan")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"=== New run: {run_name} ===")
    logger.info(f"Using device: {device}")
    logger.info(f"Seed: {seed}")
    logger.info(f"RoutePlan config: {args.config}")
    logger.info(f"Pretrain config: {args.pretrain_config}")
    logger.info(f"Pretrain checkpoint: {args.pretrain_ckpt}")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Checkpoint root dir: {ckpt_root_dir}")

    cities_topo_raw = backbone_cfg_raw["cities"]
    city_list = list(cities_topo_raw.keys())
    logger.info(f"Backbone city_list (pretrain): {city_list}")

    all_city_cfg_raw = route_cfg_raw["cities"]

    if args.cities is None or args.cities.lower() == "all":
        target_cities = list(all_city_cfg_raw.keys())
    else:
        target_cities = [c.strip() for c in args.cities.split(",") if c.strip()]

    logger.info(f"Target cities from args: {target_cities}")

    valid_target_cities = []
    for city_name in target_cities:
        if city_name not in all_city_cfg_raw:
            logger.warning(f"City '{city_name}' not found in route_plan config, skip.")
        else:
            valid_target_cities.append(city_name)

    if not valid_target_cities:
        logger.error("No valid cities to train. Please check --cities and route_plan config.")
        return

    logger.info(f"Final valid target cities: {valid_target_cities}")

    route_cfg.device = device
    for city_name in valid_target_cities:
        city_cfg_dict = all_city_cfg_raw[city_name]
        city_cfg = dict_to_object(city_cfg_dict)
        if not hasattr(route_cfg, "cities"):
            route_cfg.cities = dict_to_object(all_city_cfg_raw)
        train_for_city(
            city_name=city_name,
            city_cfg=city_cfg,
            route_cfg=route_cfg,
            backbone_cfg=backbone_cfg,
            city_list=city_list,
            pretrain_ckpt_path=args.pretrain_ckpt,
            device=device,
            logger=logger,
        )

    logger.info(f"=== Run finished: {run_name} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./chengdu/configs/route_plan.yml",
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
        default="cd",
    )
    args = parser.parse_args()
    main(args)