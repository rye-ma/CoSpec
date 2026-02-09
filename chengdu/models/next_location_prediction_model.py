import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from pretrain.models.pretrain_model import CrossCityReprModel


class NextLocationPredictionModel(nn.Module):

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

        self.num_segments = num_segments
        self.backbone_hidden_dim = backbone_cfg.model.hidden_dims

        self.backbone = CrossCityReprModel(backbone_cfg, city_list=city_list)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.freeze_backbone = freeze_backbone

        self.rnn = nn.GRU(
            input_size=self.backbone_hidden_dim*2,
            hidden_size=hidden_dim_head,
            num_layers=num_layers_head,
            batch_first=True,
            dropout=dropout_head if num_layers_head > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_dim_head, num_segments)

    def _get_segment_embeddings(self, city_data, city_name):
        out = self.backbone(city_data, city_name)
        init_seg_feats = out["init_segment_features"]
        seg_feats = out["reconstructed_segment_features"]
        return seg_feats, init_seg_feats

    def forward(self, city_data, city_name, src, lengths, labels=None):
        device = src.device

        seg_table, init_seg_feats = self._get_segment_embeddings(city_data, city_name)
        if self.freeze_backbone:
            seg_table = seg_table.detach()

        x = torch.cat([seg_table[src], init_seg_feats[src]], dim=-1)

        lengths_cpu = lengths.cpu()
        packed = pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        out_seq, _ = pad_packed_sequence(packed_out, batch_first=True)

        if labels is None:
            B = out_seq.size(0)
            idx = (lengths - 1).clamp(min=0)
            h_last = out_seq[torch.arange(B, device=device), idx]
            logits_last = self.fc(h_last)
            return logits_last
        else:
            logits_seq = self.fc(out_seq)
            return logits_seq