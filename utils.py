import re
from copy import copy
from functools import partial, update_wrapper
from typing import Iterator, List, Tuple, TypeVar

import torch
from numpy import ndarray
from ray.tune import CLIReporter
from ray.tune.experiment.trial import Trial
from sklearn.model_selection import KFold
from torch_geometric import transforms
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType
from torch_geometric.utils import negative_sampling

SEED = 123
CONV_NAME_PATT = re.compile(r"\<class 'torch_geometric\.nn\.conv\.\w+\.(.*)'\>")
_DT = TypeVar("_DT", Data, HeteroData)


class MinMaxFeatureNormalize(transforms.BaseTransform):
    """Row-normalize matrix using min-max normalization"""

    def __init__(self, attrs: List[str] = ["x"]) -> None:
        self.attrs = attrs

    def __call__(self, data: _DT) -> _DT:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                rowmin = torch.min(value, dim=1)[0].reshape(-1, 1)
                rowmax = torch.max(value, dim=1)[0].reshape(-1, 1)
                store[key] = (value - rowmin) / (rowmax - rowmin)
        return data


def wrapped_partial(func, *args, **kwargs):
    """Wrap a partial func obj to keep __name__ attr of original func"""
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def num_params(model: torch.nn.Module):
    """
    A helper function to compute the number of trainable parameters in a neural network model
    :param model: neural network model
    """
    # From https://stackoverflow.com/a/49201237
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_data(file: str) -> HeteroData:
    """Load a heterogeneous graph. Then, convert it to undirected, add self 
    loops, and perform min-max feature normalization."""
    data: HeteroData = torch.load(file)
    data = transforms.Compose(
        [
            transforms.ToUndirected(),
            transforms.AddSelfLoops(),
            MinMaxFeatureNormalize(),
        ]
    )(data)
    return data


def reset_weights(model: torch.nn.Module) -> None:
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def add_negatively_sampled_edges(
    data: HeteroData,
    full_edge_type: EdgeType = ("virus", "infects", "host"),
    shuffle_edges: bool = True,
) -> None:
    """Negatively sample edges from the indicated `edge_type`, and add
    those to the input `data` object.

    Args:
        data (HeteroData): _description_
        full_edge_type (EdgeType, optional): _description_. Defaults to ("virus", "infects", "host").
        shuffle_edges (bool, optional): _description_. Defaults to True.
    """
    src_node_type, edge_type, dst_node_type = full_edge_type
    n_src_nodes = data[src_node_type].x.size(0)
    n_dst_nodes = data[dst_node_type].x.size(0)

    negative_edges = negative_sampling(
        data[edge_type].edge_index, num_nodes=(n_src_nodes, n_dst_nodes)
    )

    # add negative edges to data object
    data[edge_type].edge_label_index = torch.cat(
        (data[edge_type].edge_index, negative_edges), dim=-1
    )

    data[edge_type].edge_label = torch.cat(
        (
            torch.ones(data[edge_type].edge_index.size(1)),
            torch.zeros(negative_edges.size(1)),
        )
    )

    if shuffle_edges:
        randidx = torch.randperm(data[edge_type].edge_label_index.size(1))

        data[edge_type].edge_label_index = data[edge_type].edge_label_index[:, randidx]
        data[edge_type].edge_label = data[edge_type].edge_label[randidx]


def kfolds_splitter(
    data: HeteroData,
    kfolds: int,
    full_edge_type: EdgeType = ("virus", "infects", "host"),
    shuffle_edges: bool = True,
    random_state: int = SEED,
) -> Iterator[Tuple[ndarray, ndarray]]:
    """Yield data indices from k partitions of the edges from the `data` for
    a link prediction task. Since this is a binary classification, negative 
    edges are sampled first before partitioning. 

    Args:
        data (HeteroData): `PyG` Heterogeneous data
        kfolds (int): number of partitions
        edge_type (Union[str, EdgeType], optional): edge type to partition. Defaults to "infects".
        random_state (int, optional): random state seed var. Defaults to SEED.

    Yields:
        Iterator[Tuple[ndarray, ndarray]]: `KFold` iterator that will yield
        `k` partitions of the dataset edges
    """
    add_negatively_sampled_edges(data, full_edge_type, shuffle_edges)

    kf = KFold(n_splits=kfolds, shuffle=True, random_state=random_state)

    yield from kf.split(range(data[full_edge_type].edge_label_index.size(1)))


def train_validation_split(
    data: HeteroData,
    kfolds: int,
    full_edge_type: EdgeType = ("virus", "infects", "host"),
    shuffle_edges: bool = True,
    random_state: int = SEED,
) -> Iterator[Tuple[HeteroData, HeteroData]]:
    """Yield train/validation data partitions after edge negative sampling
    and then splitting data into `kfolds` partitions with
    `sklearn.model_selection.KFold`

    Args:
        data (HeteroData): _description_
        kfolds (int): _description_
        full_edge_type (EdgeType, optional): _description_. Defaults to ("virus", "infects", "host").
        shuffle_edges (bool, optional): _description_. Defaults to True.
        random_state (int, optional): _description_. Defaults to SEED.

    Yields:
        Iterator[Tuple[HeteroData, HeteroData]]: _description_
    """
    kf = kfolds_splitter(data, kfolds, full_edge_type, shuffle_edges, random_state)
    for train_idx, val_idx in kf:
        train_data = copy(data)
        val_data = copy(data)

        train_data[full_edge_type].edge_label_index = data[
            full_edge_type
        ].edge_label_index[:, train_idx]
        train_data[full_edge_type].edge_label = data[full_edge_type].edge_label[
            train_idx
        ]

        val_data[full_edge_type].edge_label_index = data[
            full_edge_type
        ].edge_label_index[:, val_idx]
        val_data[full_edge_type].edge_label = data[full_edge_type].edge_label[val_idx]

        yield train_data, val_data


class ExperimentTerminationReporter(CLIReporter):
    """Only report on experiment termination"""

    def should_report(self, trials: List[Trial], done: bool = False):
        return done
