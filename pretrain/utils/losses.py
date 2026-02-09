import torch
import torch.nn.functional as F


def reconstruction_loss(init_segment_features, reconstructed_segment_features):
    
    return F.mse_loss(reconstructed_segment_features, init_segment_features)

def orthogonality_loss(low_segment_features, high_segment_features):
    
    N = low_segment_features.size(0)
    l = low_segment_features - low_segment_features.mean(dim=0, keepdim=True)
    h = high_segment_features - high_segment_features.mean(dim=0, keepdim=True)
    cov = (l.t() @ h) / max(1, N)
    return (cov**2).sum()

def smoothness_loss(low_segment_features, edge_index):
    
    row, col = edge_index
    return ((low_segment_features[row] - low_segment_features[col]).pow(2).sum(dim=1)).mean()

def topo_smoothness_loss(x, edge_index):

    row, col = edge_index
    diff2 = (x[row] - x[col]).pow(2).sum(dim=1)
    return diff2.mean()


def traj_smoothness_loss(x, traj_edge_index, traj_edge_weight=None, eps=1e-9):

    row, col = traj_edge_index
    diff2 = (x[row] - x[col]).pow(2).sum(dim=1)

    if traj_edge_weight is None:
        return diff2.mean()

    w = traj_edge_weight.to(x.device)
    w = w / (w.sum() + eps)
    loss = (w * diff2).sum()
    return loss


def combined_smoothness_loss(
    x,
    edge_index,
    traj_edge_index=None,
    traj_edge_weight=None,
    alpha_topo=1.0,
    beta_traj=1.0,
):

    L_topo = topo_smoothness_loss(x, edge_index)

    if traj_edge_index is not None:
        L_traj = traj_smoothness_loss(x, traj_edge_index, traj_edge_weight)
        L_smooth = alpha_topo * L_topo + beta_traj * L_traj
    else:
        L_traj = torch.zeros((), device=x.device)
        L_smooth = alpha_topo * L_topo

    return L_smooth, L_topo, L_traj


def prototype_consistency_loss(all_region_embeddings, prototype_matrix):

    dists = torch.cdist(all_region_embeddings, prototype_matrix)
    min_dists, _ = torch.min(dists, dim=1)
    return min_dists.mean()

def prototype_loss_v2(
    all_region_embeddings,
    prototype_matrix,
    tau=0.1,
    w_contrast=1.0,
    w_soft=0.1,
    w_balance=0.1,
):
    sim = F.cosine_similarity(
        all_region_embeddings.unsqueeze(1),
        prototype_matrix.unsqueeze(0),
        dim=-1
    )

    with torch.no_grad():
        pos_idx = sim.argmax(dim=1)
    logits = sim / tau
    loss_contrast = F.cross_entropy(logits, pos_idx)

    assign = F.softmax(logits, dim=1)
    dist2 = torch.cdist(all_region_embeddings, prototype_matrix) ** 2
    loss_soft = (assign * dist2).sum(dim=1).mean()

    p_k = assign.mean(dim=0)
    Kp = assign.size(1)
    loss_balance = ((p_k - 1.0 / Kp) ** 2).sum()

    loss = (
        w_contrast * loss_contrast
        + w_soft * loss_soft
        + w_balance * loss_balance
    )

    return loss, {
        "loss_contrast": loss_contrast.detach(),
        "loss_soft": loss_soft.detach(),
        "loss_balance": loss_balance.detach(),
    }
