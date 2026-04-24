"""
Evaluation utilities for static (E-GraphSAGE, MoE) and temporal (TGN) models.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import Optional, List


@torch.no_grad()
def eval_egraphsage(
    model,
    data: Data,
    edge_indices: Optional[List[int]] = None,
    device: str = "cpu",
    use_quantile: bool = True,
    batch_size: int = 50000,
) -> dict:
    """
    Evaluate E-GraphSAGE on a graph (or subset of edges).
    Returns dict with y_true, y_pred, y_score.
    """
    model.eval()
    model.to(device)
    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr  = (data.edge_attr_q if use_quantile else data.edge_attr).to(device)
    edge_label = data.edge_label

    if edge_indices is not None:
        idx = torch.as_tensor(edge_indices, dtype=torch.long)
        edge_attr_eval = edge_attr[idx]
        ei_eval        = data.edge_index[:, idx].to(device)
        y_true         = edge_label[idx].numpy()
    else:
        edge_attr_eval = edge_attr
        ei_eval        = edge_index
        y_true         = edge_label.numpy()

    all_preds, all_scores = [], []
    for start in range(0, ei_eval.shape[1], batch_size):
        end = start + batch_size
        logits = model(x, ei_eval[:, start:end], edge_attr_eval[start:end])
        probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_scores.append(probs)

    return {
        "y_true":  y_true,
        "y_pred":  np.concatenate(all_preds),
        "y_score": np.concatenate(all_scores),
    }


@torch.no_grad()
def eval_tgn(
    model,
    data: Data,
    neighbor_loader,
    assoc: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 200,
    use_quantile: bool = True,
) -> dict:
    """
    Evaluate TGN in causal order (memory is updated after each batch).
    Resets memory state before evaluation.
    """
    model.eval()
    model.memory.reset_state()
    neighbor_loader.reset_state()

    edge_attr = (data.edge_attr_q if use_quantile else data.edge_attr).to(device)
    src_all   = data.edge_index[0]
    dst_all   = data.edge_index[1]
    t_all     = data.edge_time
    y_true    = data.edge_label.numpy()

    all_preds, all_scores = [], []
    E = data.edge_index.shape[1]
    for start in range(0, E, batch_size):
        end   = min(start + batch_size, E)
        src   = src_all[start:end].to(device)
        dst   = dst_all[start:end].to(device)
        t     = t_all[start:end].to(device)
        msg   = edge_attr[start:end]

        logits = model(src, dst, t, msg, neighbor_loader, assoc)
        probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_scores.append(probs)

        with torch.no_grad():
            model.memory.update_state(src, dst, t, msg)
            neighbor_loader.insert(src, dst)

    return {
        "y_true":  y_true,
        "y_pred":  np.concatenate(all_preds),
        "y_score": np.concatenate(all_scores),
    }
