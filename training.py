from typing import Tuple

import torch
from torch import nn, optim
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)
from torchmetrics.metric import Metric

METRICS = {
    "Accuracy": BinaryAccuracy,
    "Precision": BinaryPrecision,
    "Recall": BinaryRecall,
    "F1": BinaryF1Score,
    "AUPRC": BinaryAveragePrecision,  # AUPRC
}


def forward(
    data: HeteroData,
    model: nn.Module,
    criterion: nn.Module,
    full_edge_type: EdgeType = ("virus", "infects", "host"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run a forward pass through the model and compute the loss.

    Args:
        data (HeteroData): _description_
        model (nn.Module): _description_
        criterion (nn.Module): _description_,
        full_edge_type: EdgeType

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: model outputs and the loss
    """
    out = model(
        data.x_dict, data.edge_index_dict, data[full_edge_type].edge_label_index,
    )
    targets = data[full_edge_type].edge_label
    loss = criterion(out, targets)
    return out, loss


def train(
    data: HeteroData,
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    full_edge_type: EdgeType = ("virus", "infects", "host"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A single training iteration.

    Args:
        data (HeteroData): training data partition
        model (nn.Module): _description_
        optimizer (optim.Optimizer): _description_
        criterion (nn.Module): _description_

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: _description_
    """
    model.train()
    optimizer.zero_grad()
    out, loss = forward(data, model, criterion, full_edge_type)
    loss.backward()
    optimizer.step()
    return out, loss


@torch.no_grad()
def predict(y_out: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert model outputs to predictions on edges. The outputs are first
    pushed through a sigmoid function to produce probabilities for each 
    edge. Then any probabilities above `threshold` are considered
    positive predictions for that tested edge.
    """
    return (torch.sigmoid(y_out) >= threshold).long()


def evaluate(y_out: torch.Tensor, y_true: torch.Tensor, metric: Metric) -> float:
    """Evaluate model performance with a provided score type"""
    y_pred = predict(y_out)
    return float(metric()(y_true, y_pred))
