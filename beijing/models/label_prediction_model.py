import torch
import torch.nn as nn
from pretrain.models.pretrain_model import CrossCityReprModel

class LabelPredictionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=1, dropout=0.1):
        super().__init__()
        layers = []
        if num_layers <= 1:
            layers.append(nn.Linear(in_dim, num_classes))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class LabelPredictionModel(nn.Module):
    def __init__(
        self,
        backbone_cfg,
        city_list,
        num_segments,
        num_classes,
        hidden_dim_head=256,
        num_layers_head=1,
        dropout_head=0.1,
        freeze_backbone=True,
    ):
        super().__init__()
        self.backbone_cfg = backbone_cfg
        self.city_list = list(city_list)
        self.num_segments = num_segments
        self.num_classes = num_classes
        self.backbone = CrossCityReprModel(backbone_cfg, city_list=self.city_list)
        feat_dim = backbone_cfg.model.hidden_dims
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = LabelPredictionHead(
            in_dim=feat_dim*2,
            hidden_dim=hidden_dim_head,
            num_classes=num_classes,
            num_layers=num_layers_head,
            dropout=dropout_head,
        )

    def forward(self, city_data, city_name, seg_ids):
        out = self.backbone(city_data, city_name=city_name)
        init_feats = out["init_segment_features"]
        seg_feats = out["reconstructed_segment_features"]
        x = torch.cat((seg_feats[seg_ids], init_feats[seg_ids]), dim=-1)
        logits = self.head(x)
        return logits