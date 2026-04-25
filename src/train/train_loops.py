"""
Training loops for all model types.
Each loop returns the best model state dict (by val MCC).
"""

import logging
import time
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader

from src.utils.metrics import compute_mcc
from src.train.eval import eval_egraphsage, eval_tgn

log = logging.getLogger(__name__)


def _class_weights(labels: torch.Tensor, num_classes: int = 2, device: str = "cpu"):
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = counts.clamp(min=1)
    freq = counts / counts.sum()
    weights = 1.0 / freq          # 1/frequency, not 1/sqrt(count)
    return (weights / weights.sum() * num_classes).to(device)


def train_egraphsage(
    model: nn.Module,
    train_data: Data,
    val_split: Optional[dict],
    device: str = "cpu",
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 50,
    patience: int = 10,
    min_epochs: int = 10,
    batch_size: int = 2048,
    use_quantile: bool = True,
    neighbor_sizes: list = None,
) -> dict:
    """Train E-GraphSAGE with mini-batch edge sampling and early stopping."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    cw = _class_weights(train_data.edge_label, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    log.info(f"Class weights: benign={cw[0]:.3f} attack={cw[1]:.3f}")

    edge_attr  = (train_data.edge_attr_q if use_quantile else train_data.edge_attr).to(device)
    x          = train_data.x.to(device)
    edge_index = train_data.edge_index.to(device)
    all_labels = train_data.edge_label

    best_mcc, best_state, patience_cnt = -2.0, None, 0

    n_edges = edge_index.shape[1]
    if val_split is None:
        split_pt  = int(n_edges * 0.8)
        train_idx = list(range(split_pt))
        val_idx   = list(range(split_pt, n_edges))
    else:
        train_idx = list(val_split["train"])
        val_idx   = list(val_split["val"])

    train_idx_arr = np.array(train_idx, dtype=np.int64)

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        np.random.shuffle(train_idx_arr)

        for start in range(0, len(train_idx_arr), batch_size):
            batch_ids = train_idx_arr[start: start + batch_size]
            ei_b = edge_index[:, batch_ids]
            ea_b = edge_attr[batch_ids]
            y_b  = all_labels[batch_ids].to(device)

            optimizer.zero_grad()
            logits = model(x, ei_b, ea_b)
            loss   = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        result   = eval_egraphsage(model, train_data, val_idx, device, use_quantile)
        val_mcc  = compute_mcc(result["y_true"], result["y_pred"])
        attack_pred_rate = result["y_pred"].mean()
        log.info(f"epoch {epoch+1:02d}  loss={avg_loss:.4f}  val_mcc={val_mcc:.4f}  "
                 f"attack_pred%={attack_pred_rate*100:.1f}")

        if val_mcc > best_mcc:
            best_mcc    = val_mcc
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            if epoch >= min_epochs:
                patience_cnt += 1
                if patience_cnt >= patience:
                    log.info(f"Early stopping at epoch {epoch+1}")
                    break

    log.info(f"Best val MCC: {best_mcc:.4f}")
    return best_state


def train_tgn(
    model,
    train_data: Data,
    val_data: Optional[Data],
    device: str = "cpu",
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 50,
    patience: int = 10,
    min_epochs: int = 10,
    batch_size: int = 200,
    use_quantile: bool = True,
    reset_memory_between_datasets: bool = False,
    dataset_boundaries: Optional[List[int]] = None,
) -> dict:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    cw = _class_weights(train_data.edge_label, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    neighbor_loader = LastNeighborLoader(model.num_nodes, size=10, device=device)
    assoc = torch.empty(model.num_nodes, dtype=torch.long, device=device)

    edge_attr = (train_data.edge_attr_q if use_quantile else train_data.edge_attr).to(device)
    src_all   = train_data.edge_index[0].to(device)
    dst_all   = train_data.edge_index[1].to(device)
    t_all     = train_data.edge_time.to(device)
    y_all     = train_data.edge_label.to(device)

    best_mcc, best_state, patience_cnt = -2.0, None, 0
    E = train_data.edge_index.shape[1]

    for epoch in range(epochs):
        model.train()
        model.memory.reset_state()
        neighbor_loader.reset_state()
        epoch_loss = 0.0

        for start in range(0, E, batch_size):
            end = min(start + batch_size, E)
            src = src_all[start:end]
            dst = dst_all[start:end]
            t   = t_all[start:end]
            msg = edge_attr[start:end]
            y   = y_all[start:end]

            if reset_memory_between_datasets and dataset_boundaries:
                for b in dataset_boundaries:
                    if start <= b < end:
                        model.memory.reset_state()
                        neighbor_loader.reset_state()

            optimizer.zero_grad()
            logits = model(src, dst, t, msg, neighbor_loader, assoc,
                           t_full=t_all, msg_full=edge_attr)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            with torch.no_grad():
                model.memory.update_state(src, dst, t, msg)
                neighbor_loader.insert(src, dst)

        if val_data is not None:
            val_nl = LastNeighborLoader(model.num_nodes, size=10, device=device)
            val_assoc = torch.empty(model.num_nodes, dtype=torch.long, device=device)
            result = eval_tgn(model, val_data, val_nl, val_assoc, device, batch_size, use_quantile)
            val_mcc = compute_mcc(result["y_true"], result["y_pred"])
        else:
            val_mcc = 0.0

        log.info(f"TGN epoch {epoch+1:02d}  loss={epoch_loss:.4f}  val_mcc={val_mcc:.4f}")

        if val_mcc > best_mcc:
            best_mcc    = val_mcc
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            if epoch >= min_epochs:
                patience_cnt += 1
                if patience_cnt >= patience:
                    log.info(f"Early stopping at epoch {epoch+1}")
                    break

    log.info(f"Best val MCC: {best_mcc:.4f}")
    return best_state


def train_moe(
    model,
    train_data: Data,
    val_split: Optional[dict] = None,
    device: str = "cpu",
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 50,
    patience: int = 10,
    min_epochs: int = 10,
    batch_size: int = 2048,
    use_quantile: bool = True,
    lam: float = 0.1,
    uniform_gate: bool = False,
) -> dict:
    from src.models.moe_ids import augment_feature_noise, augment_density, augment_label_balance

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    cw = _class_weights(train_data.edge_label, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    edge_attr = (train_data.edge_attr_q if use_quantile else train_data.edge_attr).to(device)
    edge_label = train_data.edge_label.to(device)
    edge_index = train_data.edge_index.to(device)
    x = train_data.x.to(device)

    n_edges = edge_index.shape[1]
    if val_split is None:
        split_pt  = int(n_edges * 0.8)
        train_idx = list(range(split_pt))
        val_idx   = list(range(split_pt, n_edges))
    else:
        train_idx = list(val_split["train"])
        val_idx   = list(val_split["val"])

    if uniform_gate:
        for p in model.gate.parameters():
            p.requires_grad_(False)

    best_mcc, best_state, patience_cnt = -2.0, None, 0

    for epoch in range(epochs):
        model.train()
        np.random.shuffle(train_idx)
        epoch_loss = 0.0

        for start in range(0, len(train_idx), batch_size):
            batch_ids = train_idx[start: start + batch_size]
            ei_b  = edge_index[:, batch_ids]
            ea_b  = edge_attr[batch_ids]
            y_b   = edge_label[batch_ids]

            # Build per-expert augmented features
            aug1 = augment_feature_noise(ea_b)
            aug2 = augment_feature_noise(ea_b, sigma=0.05)
            _, y_aug3 = augment_label_balance(ea_b, y_b)
            aug3 = ea_b  # label balance doesn't change edge_attr values

            augmented = [aug1, aug2, aug3]
            optimizer.zero_grad()
            logits, L_align = model(
                x, ei_b, ea_b,
                augmented_attrs=augmented,
                return_loss_components=True,
            )
            L_task = criterion(logits, y_b)
            loss   = L_task + lam * L_align
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(x, edge_index[:, val_idx], edge_attr[val_idx])
        y_pred = val_logits.argmax(dim=-1).cpu().numpy()
        y_true = edge_label[val_idx].cpu().numpy()
        val_mcc = compute_mcc(y_true, y_pred)
        log.info(f"MoE epoch {epoch+1:02d}  loss={epoch_loss:.4f}  val_mcc={val_mcc:.4f}")

        if val_mcc > best_mcc:
            best_mcc    = val_mcc
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            if epoch >= min_epochs:
                patience_cnt += 1
                if patience_cnt >= patience:
                    log.info(f"Early stopping at epoch {epoch+1}")
                    break

    log.info(f"Best val MCC: {best_mcc:.4f}")
    return best_state
