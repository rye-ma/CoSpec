import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, ChebConv, TransformerConv
from torch_geometric.utils import remove_self_loops, coalesce

class FeatureEmbeddingModule(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.segment_lane_emb_layer = nn.Embedding(
            self.cfg.model.segment_lane_num, self.cfg.model.segment_lane_dims
        )

        self.segment_type_emb_layer = nn.Embedding(
            self.cfg.model.segment_type_num, self.cfg.model.segment_type_dims
        )

        self.segment_length_emb_layer = nn.Embedding(
            self.cfg.model.segment_length_num, self.cfg.model.segment_length_dims
        )

        self.hidden_dimss = (
            self.cfg.model.segment_type_dims
            + self.cfg.model.segment_length_dims
            + self.cfg.model.segment_lane_dims
        )

        self.proj_layer = nn.Linear(
            self.hidden_dimss, self.cfg.model.hidden_dims
        )

    def forward(self, segment_features):

        segment_lane_emb = self.segment_lane_emb_layer(segment_features[:, 0])
        segment_type_emb = self.segment_type_emb_layer(segment_features[:, 1])
        segment_length_emb = self.segment_length_emb_layer(segment_features[:, 2])

        segment_emb = torch.cat([
            segment_lane_emb,
            segment_type_emb,
            segment_length_emb,
        ], dim=1)

        init_segment_features = self.proj_layer(segment_emb)
        return init_segment_features

class SegmentEncoder(nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.conv1 = GCNConv(self.hidden_dims, self.hidden_dims)
        self.conv2 = GCNConv(self.hidden_dims, self.hidden_dims)
        self.norm1 = nn.LayerNorm(self.hidden_dims)
        self.norm2 = nn.LayerNorm(self.hidden_dims)
        self.act = nn.GELU()

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.act(self.norm1(h))
        h = self.conv2(h, edge_index)
        h = self.act(self.norm2(h))
        return h

class CrossAttentionPoolingModule(nn.Module):
    def __init__(self, hidden_dims, region_num, city_list, rank=8, use_adapter=True):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.region_num = region_num
        self.use_adapter = use_adapter

        self.global_region_centers = nn.Parameter(
            torch.randn(region_num, hidden_dims)
        )

        self.city_list = list(city_list)
        self.city2idx = {c: i for i, c in enumerate(self.city_list)}

        if use_adapter:
            self.city_emb = nn.Embedding(len(self.city_list), rank)
            self.adapter_up = nn.Linear(rank, region_num * hidden_dims, bias=False)
        else:
            self.city_region_centers = nn.Parameter(
                torch.randn(len(self.city_list), region_num, hidden_dims)
            )

        self.linear_q = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.linear_k = nn.Linear(self.hidden_dims, self.hidden_dims)

    def get_city_emb(self, city_name: str):
        idx = self.city2idx[city_name]
        return self.city_emb.weight[idx]

    def get_region_centers(self, city_name: str):
        idx = self.city2idx[city_name]
        if self.use_adapter:
            city_emb = self.city_emb.weight[idx]
            delta = self.adapter_up(city_emb)
            delta = delta.view(self.region_num, self.hidden_dims)
            centers = self.global_region_centers + delta
        else:
            centers = self.city_region_centers[idx]
        return centers

    def forward(self, segment_features, city_name: str):
        region_centers = self.get_region_centers(city_name)

        queries = self.linear_q(region_centers)
        keys = self.linear_k(segment_features)
        values = segment_features

        attention_scores = torch.matmul(queries, keys.T) / (
            region_centers.size(1) ** 0.5
        )

        segment2region_assignments = F.softmax(attention_scores, dim=1)

        region_features = torch.matmul(segment2region_assignments, values) + region_centers

        return region_features, segment2region_assignments

class RegionEncoder(nn.Module):
    def __init__(self, hidden_dims, heads=4, dropout=0.1):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.heads = heads

        self.conv1 = TransformerConv(
            in_channels=hidden_dims,
            out_channels=hidden_dims // heads,
            heads=heads,
            edge_dim=1,
            dropout=dropout,
            beta=False
        )

        self.conv2 = TransformerConv(
            in_channels=hidden_dims,
            out_channels=hidden_dims // heads,
            heads=heads,
            edge_dim=1,
            dropout=dropout,
            beta=False
        )

        self.norm1 = nn.LayerNorm(hidden_dims)
        self.norm2 = nn.LayerNorm(hidden_dims)
        self.act = nn.GELU()

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None:
            edge_attr = edge_weight.unsqueeze(-1)
        else:
            edge_attr = None

        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        h = self.act(self.norm1(h))

        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        h = self.act(self.norm2(h))

        return h

class RegionProjector(nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, hidden_dims)
        )
        self.norm = nn.LayerNorm(hidden_dims)

    def forward(self, R):
        h = self.mlp(R)
        return self.norm(h)

class LowFrequencyPath(nn.Module):

    def __init__(self, hidden_dims, cheb_K=5):
        super().__init__()
        self.hidden_dims = hidden_dims

        self.seg_cheb = ChebConv(hidden_dims, hidden_dims, K=cheb_K)
        self.norm = nn.LayerNorm(hidden_dims)

    def forward(self, projected_region_features, segment2region_assignment, edge_index):

        seg_low_raw = torch.matmul(segment2region_assignment, projected_region_features)

        seg_low = self.seg_cheb(seg_low_raw, edge_index)
        seg_low = self.norm(F.gelu(seg_low))

        return seg_low, seg_low_raw


class CityFiLM(nn.Module):

    def __init__(self, hidden_dims, rank=8):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.rank = rank
        self.affine = nn.Linear(rank, 2 * hidden_dims, bias=False)

    def forward(self, x, city_emb):

        gamma_beta = self.affine(city_emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(0)
        beta = beta.unsqueeze(0)
        return x * (1.0 + gamma) + beta


class CityHighFrequencyPath(nn.Module):
    def __init__(self, hidden_dims, cheb_K=3, rank=8):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.rank = rank
        self.film = CityFiLM(hidden_dims, rank=rank)
        self.cheb = ChebConv(hidden_dims, hidden_dims, K=cheb_K)
        self.norm = nn.LayerNorm(hidden_dims)

    def forward(self, init_segment_features, low_segment_raw, edge_index, city_emb):

        residual = init_segment_features - low_segment_raw

        residual = self.film(residual, city_emb)
        high = self.cheb(residual, edge_index)
        high = self.norm(F.gelu(high))
        return high

class FrequencyFusion(nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dims * 2, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, 1)
        )
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, hidden_dims)
        )

    def forward(self, low, high):
        h = torch.cat([low, high], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(h))
        fused = gate * low + (1.0 - gate) * high
        fused = self.out_mlp(fused)
        return fused


class CrossCityReprModel(nn.Module):
    def __init__(self, cfg, city_list):
        super().__init__()

        self.cfg = cfg
        self.city_list = list(city_list)

        hidden_dims = self.cfg.model.hidden_dims
        region_num = self.cfg.model.region_num
        cap_rank = getattr(self.cfg.model, "cap_rank", 8)
        use_adapter = getattr(self.cfg.model, "cap_use_adapter", True)

        self.feature_embedding_module = FeatureEmbeddingModule(self.cfg)
        self.segment_encoder = SegmentEncoder(hidden_dims)
        self.cap_module = CrossAttentionPoolingModule(
                            hidden_dims=hidden_dims,
                            region_num=region_num,
                            city_list=self.city_list,
                            rank=cap_rank,
                            use_adapter=use_adapter,
                        )
        self.region_encoder = RegionEncoder(hidden_dims)
        self.region_projector = RegionProjector(hidden_dims)
        self.low_path = LowFrequencyPath(hidden_dims, cheb_K=5)
        high_rank = getattr(self.cfg.model, "high_rank", 8)
        self.high_path = CityHighFrequencyPath(hidden_dims, cheb_K=3, rank=high_rank)
        self.fusion = FrequencyFusion(hidden_dims)


    def forward(self, data: Data, city_name: str):
        init_segment_features = self.feature_embedding_module(data.x)
        segment_features = self.segment_encoder(init_segment_features, data.edge_index)
        region_features, segment2region_assignment = self.cap_module(segment_features, city_name)
        if segment2region_assignment.dim() == 2 and segment2region_assignment.size(0) == region_features.size(0):
            segment2region_assignment = segment2region_assignment.t()

        seg2reg = segment2region_assignment.argmax(dim=-1)
        src_seg, dst_seg = data.edge_index
        src_reg = seg2reg[src_seg]
        dst_reg = seg2reg[dst_seg]
        region_edge_index = torch.stack([src_reg, dst_reg], dim=0)
        region_edge_weight = torch.ones(
            region_edge_index.size(1),
            device=region_edge_index.device,
            dtype=region_features.dtype,
        )

        region_edge_index, region_edge_weight = coalesce(
            region_edge_index,
            region_edge_weight,
            region_features.size(0)
        )
        row = region_edge_index[0]
        deg = torch.zeros(region_features.size(0), device=region_edge_weight.device).scatter_add_(
            0, row, region_edge_weight
        )
        norm = deg[row].clamp(min=1e-6)
        region_edge_weight = region_edge_weight / norm

        updated_region_features = self.region_encoder(region_features, region_edge_index, edge_weight=region_edge_weight)
        projected_region_features = self.region_projector(updated_region_features)

        low_segment_features, low_segment_raw = self.low_path(
            projected_region_features=projected_region_features,
            segment2region_assignment=segment2region_assignment,
            edge_index=data.edge_index,
        )

        city_emb = self.cap_module.get_city_emb(city_name)

        high_segment_features = self.high_path(
            init_segment_features=init_segment_features,
            low_segment_raw=low_segment_raw,
            edge_index=data.edge_index,
            city_emb=city_emb,
        )

        reconstructed_segment_features = self.fusion(
            low_segment_features,
            high_segment_features,
        )

        return {
            "init_segment_features": init_segment_features, 
            "segment2region_assignment": segment2region_assignment, 
            "region_features": region_features,
            "projected_region_features": projected_region_features, 
            "low_segment_features": low_segment_features, 
            "high_segment_features": high_segment_features, 
            "reconstructed_segment_features": reconstructed_segment_features,
        }