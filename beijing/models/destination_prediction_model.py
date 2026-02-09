import torch
import torch.nn as nn
from torch_geometric.data import Data

from pretrain.models.pretrain_model import CrossCityReprModel


class DestinationPredictionModel(nn.Module):

    def __init__(
        self,
        backbone_cfg,
        city_list,
        num_segments: int,
        hidden_dim_head: int = 256,
        num_layers_head: int = 1,
        dropout_head: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.backbone = CrossCityReprModel(backbone_cfg, city_list=city_list)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.num_segments = num_segments
        self.backbone_hidden_dim = backbone_cfg.model.hidden_dims

        self.rnn = nn.GRU(
            input_size=self.backbone_hidden_dim,
            hidden_size=hidden_dim_head,
            num_layers=num_layers_head,
            batch_first=True,
            dropout=dropout_head if num_layers_head > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim_head, num_segments)

    def forward(self, city_data: Data, city_name: str, seq: torch.LongTensor):
        out = self.backbone(city_data, city_name)
        init_feats = out["init_segment_features"]
        seg_feats = out["reconstructed_segment_features"]

        x = seg_feats[seq]

        _, h_n = self.rnn(x)
        h_last = h_n[-1, :, :]
        logits = self.fc(h_last)

        return logits