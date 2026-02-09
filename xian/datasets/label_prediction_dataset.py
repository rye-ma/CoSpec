import numpy as np
from torch.utils.data import Dataset

class LabelPredictionDataset(Dataset):
    def __init__(self, segment_ids, labels):
        super().__init__()
        assert len(segment_ids) == len(labels), "segment_ids 和 labels 长度不一致"
        self.segment_ids = np.asarray(segment_ids, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.segment_ids)

    def __getitem__(self, idx):
        seg_id = int(self.segment_ids[idx])
        label = int(self.labels[idx])
        return seg_id, label