import numpy as np
import torch
from torch.utils.data import Dataset


class DestinationPredictionDataset(Dataset):

    def __init__(
        self,
        raw_batches,
        input_len: int = 5,
        min_len: int = 6,
        max_len: int = 12,
    ):
        super().__init__()
        self.inputs = []
        self.labels = []
        self.input_len = input_len

        for batch in raw_batches:
            if len(batch) == 0:
                continue
            batch_arr = np.array(batch, dtype=object)
            for traj in batch_arr:
                traj = np.array(traj, dtype=np.int64)
                L = len(traj)
                if L < min_len or L > max_len:
                    continue

                src = traj[:input_len]
                label = traj[-1]

                self.inputs.append(src)
                self.labels.append(label)

        print(
            f"[DestinationPredictionDataset] "
            f"collected {len(self.inputs)} samples "
            f"(input_len={input_len}, min_len={min_len}, max_len={max_len})"
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        src = torch.tensor(self.inputs[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return src, label