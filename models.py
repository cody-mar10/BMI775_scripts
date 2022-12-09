from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing, to_hetero
from torch_geometric.typing import EdgeType

from utils import SEED

torch.manual_seed(SEED)


class GNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[int, Tuple[int, ...]],
        conv_layer: MessagePassing,
        dropout: float = 0.5,
        **conv_kwargs,
    ) -> None:
        torch.manual_seed(SEED)
        super(GNNEncoder, self).__init__()

        self.dropout = dropout

        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self._add_conv_layers(input_dim, hidden_dims, conv_layer, **conv_kwargs)

    def _add_conv_layers(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...],
        conv_layer: MessagePassing,
        **conv_kwargs,
    ) -> None:
        # NOTE: it is critical to add conv layers this way so that they are not all references
        # to the same object
        prev_dim = input_dim
        for dim in hidden_dims:
            self.layers.append(
                conv_layer(
                    in_channels=(prev_dim, prev_dim), out_channels=dim, **conv_kwargs
                )
            )
            self.activations.append(nn.ReLU())
            prev_dim = dim

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        for layer, activation in zip(self.layers, self.activations):
            x_dict = activation(
                F.dropout(
                    layer(x_dict, edge_index_dict),
                    p=self.dropout,
                    training=self.training,
                )
            )

        return x_dict


class AttentionConv(GNNEncoder):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[int, Tuple[int, ...]],
        conv_layer: MessagePassing,
        dropout: float = 0.5,
        **conv_kwargs,
    ) -> None:
        if "heads" not in conv_kwargs:
            raise ValueError(
                "Missing 'heads' parameter for multi-headed attention layers"
            )

        super(AttentionConv, self).__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            conv_layer=conv_layer,
            dropout=dropout,
            **conv_kwargs,
        )

    def _add_conv_layers(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...],
        conv_layer: MessagePassing,
        **conv_kwargs,
    ) -> None:
        heads = conv_kwargs["heads"]
        prev_dim = input_dim
        for dim in hidden_dims:
            conv = conv_layer(
                in_channels=(prev_dim, prev_dim),
                out_channels=dim,
                **conv_kwargs,  # contains heads arg
            )
            self.layers.append(conv)
            self.activations.append(nn.ReLU())
            prev_dim = dim * heads


class LinkDecoder(nn.Module):
    def __init__(
        self, starting_dim: int, hidden_dims: Tuple[int, ...], dropout: float = 0.5
    ) -> None:
        torch.manual_seed(SEED)
        super(LinkDecoder, self).__init__()

        self.dropout = dropout

        self.layers = nn.ModuleList()
        prev_dim = starting_dim
        for hdim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hdim))
            self.layers.append(nn.ReLU())
            prev_dim = hdim
        self.layers.append(nn.Linear(prev_dim, 1))

    def forward(
        self, x_dict: Dict[str, torch.Tensor], edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        row, col = edge_label_index

        vir_feats = x_dict["virus"][row]
        host_feats = x_dict["host"][col]
        z = vir_feats - host_feats
        for layer in self.layers:
            z = F.dropout(layer(z), p=self.dropout, training=self.training)
        return z


class CherryModel(nn.Module):
    def __init__(
        self,
        data: HeteroData,
        gnn_hidden_dims: Union[int, Tuple[int, ...]],
        linear_hidden_dims: Tuple[int, ...],
        conv_layer: MessagePassing,
        dropout: float = 0.5,
        **conv_kwargs,
    ) -> None:
        torch.manual_seed(SEED)
        super(CherryModel, self).__init__()
        self.node_dim = data["virus"].x.size(1)

        gnn_output_dim = (
            gnn_hidden_dims if isinstance(gnn_hidden_dims, int) else gnn_hidden_dims[-1]
        )

        if "heads" in conv_kwargs:
            conv_cls = AttentionConv
            gnn_output_dim *= conv_kwargs["heads"]
        else:
            conv_cls = GNNEncoder

        self.encoder = conv_cls(
            input_dim=self.node_dim,
            hidden_dims=gnn_hidden_dims,
            conv_layer=conv_layer,
            dropout=dropout,
            **conv_kwargs,
        )
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr="sum")

        self.decoder = LinkDecoder(gnn_output_dim, linear_hidden_dims, dropout=dropout)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index).squeeze()
