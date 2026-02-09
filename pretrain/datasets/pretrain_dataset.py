import os
import torch
import numpy as np
import pickle
import scipy.sparse as sp
from torch_geometric.data import Data

def load_city_dataset(cfg, city_name: str) -> Data:
    base = os.path.join("data", city_name)
    feature_path = os.path.join(base, "road_features")
    adj_path = os.path.join(base, "road_adj_matrix")
    traj_adj_path = os.path.join(base, "road_trajectory_adj_matrix")

    if not (os.path.exists(feature_path) and
            os.path.exists(adj_path) and
            os.path.exists(traj_adj_path)):
        raise FileNotFoundError(
            f"expecting {feature_path}, {adj_path}, {traj_adj_path}"
        )

    with open(feature_path, "rb") as f:
        segment_features = list(pickle.load(f))

    city_cfg = getattr(cfg.cities, city_name)
    while len(segment_features) < city_cfg.num_segments:
        segment_features.append([0] * segment_features[0].shape[0])
    segment_features = [[int(v) for v in arr] for arr in segment_features]
    segment_features = np.array(segment_features)

    with open(adj_path, "rb") as f:
        segment_adj_matrix = pickle.load(f)

    if isinstance(segment_adj_matrix, sp.spmatrix):
        seg_row, seg_col = segment_adj_matrix.nonzero()
    else:
        segment_adj_matrix = np.array(segment_adj_matrix)
        seg_row, seg_col = segment_adj_matrix.nonzero()

    segment_edge_index = np.vstack((seg_row, seg_col))

    with open(traj_adj_path, "rb") as f:
        traj_adj = pickle.load(f)

    if isinstance(traj_adj, sp.spmatrix):
        traj_row, traj_col = traj_adj.nonzero()
        traj_vals = np.asarray(traj_adj[traj_row, traj_col]).ravel()
    else:
        traj_adj = np.array(traj_adj)
        traj_row, traj_col = traj_adj.nonzero()
        traj_vals = traj_adj[traj_row, traj_col]

    traj_edge_index = np.vstack((traj_row, traj_col))

    segment_features = torch.tensor(segment_features, dtype=torch.long)
    segment_edge_index = torch.tensor(segment_edge_index, dtype=torch.long)

    traj_edge_index = torch.tensor(traj_edge_index, dtype=torch.long)
    traj_edge_weight = torch.tensor(traj_vals, dtype=torch.float32)

    data = Data(
        x=segment_features,
        edge_index=segment_edge_index,
    )
    data.city_name = city_name

    data.traj_edge_index = traj_edge_index
    data.traj_edge_weight = traj_edge_weight

    return data
