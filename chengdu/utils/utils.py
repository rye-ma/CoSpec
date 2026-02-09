import torch
import numpy as np
import random
import pickle
from types import SimpleNamespace
from sklearn.utils import shuffle

def dict_to_object(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_object(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_object(x) for x in d]
    else:
        return d

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_destination_prediction_data(config):
    with open(config.train_des_set, "rb") as f:
        train_des_set = pickle.load(f)
    with open(config.test_des_set, "rb") as f:
        test_des_set = pickle.load(f)
    return train_des_set, test_des_set

def load_next_location_prediction_data(config):
    with open(config.train_loc_set, "rb") as f:
        train_loc_set = pickle.load(f)
    with open(config.test_loc_set, "rb") as f:
        test_loc_set = pickle.load(f)
    return train_loc_set, test_loc_set

def load_route_plan_data(config):
    with open(config.train_route_set, "rb") as f:
        train_route_set = pickle.load(f)
    with open(config.test_route_set, "rb") as f:
        test_route_set = pickle.load(f)
    return train_route_set, test_route_set

def load_label_prediction_data(config):
    label_pred_train = pickle.load(open(config.train_pred_train_set, "rb"))
    
    label_train_true = label_pred_train[:-100]

    label_train_false = []
    while len(label_train_false) < len(label_train_true):
        node = random.randint(0, config.num_segments - 1)
        if (node not in label_train_false) and (node not in label_train_true):
            label_train_false.append(node)

    train_seg_ids = label_train_false + label_train_true
    train_labels = [0 for _ in range(len(label_train_false))] + \
                   [1 for _ in range(len(label_train_true))]

    label_test_false = pickle.load(open(config.train_pred_train_set_false, "rb"))
    label_test_true = label_pred_train[-100:]

    test_seg_ids = label_test_false + label_test_true
    test_labels = [0 for _ in range(len(label_test_false))] + \
                  [1 for _ in range(len(label_test_true))]

    train_seg_ids = np.array(train_seg_ids, dtype=np.int64)
    train_labels = np.array(train_labels, dtype=np.int64)
    test_seg_ids = np.array(test_seg_ids, dtype=np.int64)
    test_labels = np.array(test_labels, dtype=np.int64)

    return train_seg_ids, train_labels, test_seg_ids, test_labels

def edit_distance(a, b):
    la, lb = len(a), len(b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[la][lb]