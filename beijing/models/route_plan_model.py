import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from pretrain.models.pretrain_model import CrossCityReprModel

class RoutePlanModel(nn.Module):
    def __init__(
        self,
        backbone_cfg,
        city_list,
        num_segments,
        hidden_dim_head=256,
        num_layers_head=1,
        dropout_head=0.1,
        freeze_backbone=True,
    ):
        super().__init__()

        self.num_segments = num_segments
        self.hidden_dim_head = hidden_dim_head

        self.backbone = CrossCityReprModel(backbone_cfg, city_list=city_list)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        backbone_dim = backbone_cfg.model.hidden_dims

        self.seg_proj = nn.Linear(backbone_dim, hidden_dim_head)
        self.dest_proj = nn.Linear(hidden_dim_head, hidden_dim_head)

        self.rnn = nn.GRU(
            input_size=backbone_dim * 2,
            hidden_size=hidden_dim_head,
            num_layers=num_layers_head,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout_head)
        self.fc = nn.Linear(hidden_dim_head, num_segments)

    def forward(self, city_data: Data, city_name: str,
                input_seq: torch.Tensor, destination: torch.Tensor):

        device = input_seq.device
        backbone_out = self.backbone(city_data, city_name)
        seg_feats = backbone_out["reconstructed_segment_features"]
        init_seg_feats = backbone_out["init_segment_features"]

        B, T = input_seq.shape

        rnn_input = torch.cat([seg_feats[input_seq], init_seg_feats[input_seq]], dim=-1)

        rnn_out, _ = self.rnn(rnn_input)
        rnn_out = self.dropout(rnn_out)

        logits = self.fc(rnn_out)

        return logits