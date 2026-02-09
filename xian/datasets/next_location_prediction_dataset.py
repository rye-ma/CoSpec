import numpy as np
import torch
from torch.utils.data import Dataset

class NextLocationPredictionDataset(Dataset):
    def __init__(
        self,
        raw_batches,
        mode: str = "train",
        min_len: int = 2,
        max_len: int = 1000,
    ):
        super().__init__()
        assert mode in ["train", "test"], "mode should be 'train' or 'test'"
        self.mode = mode

        self.inputs = []
        self.labels = []
        self.lengths = []

        for batch in raw_batches:
            if len(batch) == 0:
                continue

            for traj in batch:
                traj = np.asarray(traj, dtype=np.int64)
                L = len(traj)

                if L < min_len or L > max_len:
                    continue
                if L < 2:
                    continue

                if mode == "train":
                    src = traj[:-1]
                    label = traj[1:]
                    src = np.asarray(src, dtype=np.int64).reshape(-1)
                    label = np.asarray(label, dtype=np.int64).reshape(-1)
                else:
                    src = traj[:-1]
                    label = traj[-1]
                    src = np.asarray(src, dtype=np.int64).reshape(-1)
                    label = int(label)

                self.inputs.append(src)
                self.labels.append(label)
                self.lengths.append(len(src))

        print(
            f"[NextLocationPredictionDataset][{mode}] "
            f"collected {len(self.inputs)} samples "
            f"(min_len={min_len}, max_len={max_len})"
        )

        if len(self.inputs) > 0:
            lens = {len(x) for x in self.inputs}
            print(f"[NextLocationPredictionDataset][{mode}] src length set: {lens}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        src_np = self.inputs[idx]
        label = self.labels[idx]
        length = self.lengths[idx]

        src = torch.tensor(src_np, dtype=torch.long)
        if self.mode == "train":
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = torch.tensor(label, dtype=torch.long)

        return src, label, length