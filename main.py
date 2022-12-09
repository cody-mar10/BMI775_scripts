#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from functools import partial
from typing import Any, Dict, List, Tuple, Union, Iterator, TextIO

import numpy as np
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn, optim
from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    FiLMConv,
    GATv2Conv,
    GENConv,
    GraphConv,
    LEConv,
    MessagePassing,
    ResGatedGraphConv,
    SAGEConv,
)
from torch_geometric.typing import EdgeType

from models import CherryModel
from training import evaluate, forward, precision_recall_auc, train
from utils import (
    CONV_NAME_PATT,
    SEED,
    ExperimentTerminationReporter,
    load_data,
    num_params,
    reset_weights,
    train_validation_split,
    wrapped_partial,
)

np.random.seed(SEED)
torch.manual_seed(SEED)

PHYSICAL_THREADS = os.cpu_count() // 2  # assuming dual-threaded cores

CONV_LAYERS = {
    "GCN": GraphConv,
    "SAGE": SAGEConv,
    "GAT": GATv2Conv,
    "ResGated": ResGatedGraphConv,
    "LE": LEConv,
    "GEN": GENConv,
    "FiLM": FiLMConv,
}

USES_ATTENTION = {GATv2Conv}
DONT_ADD_SELF_LOOPS = {GATv2Conv}

SCORERS = [
    accuracy_score,
    wrapped_partial(precision_score, zero_division=0.0),
    wrapped_partial(recall_score, zero_division=0.0),
    wrapped_partial(f1_score, zero_division=0.0),
    precision_recall_auc,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    io_args = parser.add_argument_group("IO")
    io_args.add_argument(
        "-i", "--input", help="required input pytorch geometric data file",
    )
    io_args.add_argument(
        "-o",
        "--output",
        default="model_performance_summary.tsv",
        help="output summary file (default: %(default)s)",
    )
    io_args.add_argument(
        "-od",
        "--outdir",
        default="./model_eval",
        help="output directory for ray-tune testing (default: %(default)s)",
    )

    model_hyperparam_args = parser.add_argument_group("MODEL HYPERPARAMETERS")
    model_hyperparam_args.add_argument(
        "-k",
        "--kfolds",
        type=int,
        choices={5, 10},
        default=10,
        help="number of k-fold cross validation partitions (default: %(default)s)",
    )
    model_hyperparam_args.add_argument(
        "-n",
        "--n-conv-layers",
        type=int,
        default=2,
        help="number of convolution layers all with same hidden dim (default: %(default)s)",
    )
    model_hyperparam_args.add_argument(
        "-gh",
        "--gnn-hidden-dim",
        type=int,
        default=256,
        help="hidden dimension for graph convolution (default: %(default)s)",
    )
    model_hyperparam_args.add_argument(
        "-lh",
        "--linear-hidden-dims",
        nargs="+",
        type=int,
        default=[128, 32],
        help="hidden dimensions for fully connected layers (default: %(default)s)",
    )
    model_hyperparam_args.add_argument(
        "-c",
        "--conv-layer",
        type=str,
        choices=CONV_LAYERS.keys(),
        default="GCN",
        help="type of convolution layers (default: %(default)s)",
    )
    model_hyperparam_args.add_argument(
        "-a",
        "--attention-heads",
        type=int,
        default=0,
        help="number of attention heads for convolution layers with multi-headed attention (default: %(default)s -- leave at 0 if layer doesn't use attention)",
    )

    training_hyperparam_args = parser.add_argument_group("TRAINING HYPERPARAMETERS")
    training_hyperparam_args.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        help="number of training epochs (default: %(default)s)",
    )
    training_hyperparam_args.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        default=0.001,
        help="learning rate (default: %(default)s)",
    )
    training_hyperparam_args.add_argument(
        "-d",
        "--dropout",
        type=float,
        default=0.5,
        help="dropout proportion during training (default: %(default)s)",
    )
    training_hyperparam_args.add_argument(
        "-w",
        "--weight-decay",
        type=float,
        default=0.01,
        help="optimizer weight decay (default: %(default)s)",
    )
    training_hyperparam_args.add_argument(
        "-r",
        "--logging-rate",
        type=int,
        default=5,
        help="how often to log performance in units of epochs (default: %(default)s)",
    )

    parallelization_args = parser.add_argument_group("PARALLELIZATION ARGS")
    parallelization_args.add_argument(
        "-t",
        "--threads",
        type=int,
        default=PHYSICAL_THREADS,
        help="number of pytorch CPU worker threads (default: %(default)s)",
    )
    parallelization_args.add_argument(
        "-g",
        "--gpus",
        type=int,
        default=1,
        help="number of GPUs to use if any are available (default: %(default)s)",
    )

    tuning_args = parser.add_argument_group("TUNING ARGS")
    tuning_args.add_argument(
        "--tune-parameters",
        default=False,
        action="store_true",
        help="use if you want to sample from the space of all possible models rather than run a single setup",
    )
    tuning_args.add_argument(
        "--n-trials",
        type=int,
        default=1e3,
        help="number of tuning trials to run (default: %(default)s)",
    )
    return parser.parse_args()


def print_metrics(metrics: List[Union[str, int, float]]) -> List[str]:
    metrics_line: List[str] = list()
    for metric in metrics:
        if isinstance(metric, float):
            metric = str(round(metric, 5))
        elif isinstance(metric, int):
            metric = str(metric)

        metrics_line.append(metric)

    return metrics_line


def training_loop(
    n_epochs: int,
    log_epochs: int,
    kfold: int,
    train_data: HeteroData,
    val_data: HeteroData,
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    full_edge_type: EdgeType,
    conv_layer: MessagePassing,
    gnn_hidden_dims: Tuple[int, ...],
    lr: float,
    weight_decay: float,
    dropout: float,
    filehandle: TextIO,
    **conv_kwargs,
) -> Tuple[List[float], List[float]]:
    val_loss_arr = list()
    val_acc_arr = list()
    for epoch in range(n_epochs):
        out, loss = train(train_data, model, optimizer, criterion, full_edge_type)

        if epoch % log_epochs == 0:
            conv_name = CONV_NAME_PATT.findall(str(conv_layer))[0]
            n_conv_layers = (
                len(gnn_hidden_dims) if isinstance(gnn_hidden_dims, tuple) else 1
            )
            heads = conv_kwargs.get("heads", 0)
            n_params = num_params(model)

            metrics = dict()
            with torch.no_grad():
                model.eval()
                val_out, val_loss = forward(val_data, model, criterion, full_edge_type)
                val_loss_arr.append(val_loss)

                for dcat, l, y_out, y_true in zip(
                    ("train", "val"),
                    (loss, val_loss),
                    (out, val_out),
                    (
                        train_data[full_edge_type].edge_label,
                        val_data[full_edge_type].edge_label,
                    ),
                ):
                    metrics[dcat] = [
                        conv_name,
                        kfold,
                        epoch,
                        n_conv_layers,
                        heads,
                        n_params,
                        lr,
                        weight_decay,
                        dropout,
                        dcat,
                        float(l),
                    ]
                    for scorer in SCORERS:
                        score = evaluate(y_out, y_true, scorer)
                        metrics[dcat].append(score)

                        if dcat == "val" and scorer == accuracy_score:
                            val_acc_arr.append(score)

                    metrics_line = print_metrics(metrics[dcat])
                    print("\t".join(metrics_line), flush=True, file=filehandle)
    return val_loss_arr, val_acc_arr


def cross_validation_loop(
    kfolds_splitter: Iterator[Tuple[HeteroData, HeteroData]],
    gnn_hidden_dims: Tuple[int, ...],
    linear_hidden_dims: Tuple[int, int],
    conv_layer: MessagePassing,
    criterion: nn.Module,
    full_edge_type: EdgeType,
    dropout: float,
    lr: float,
    weight_decay: float,
    n_epochs: int,
    log_epochs: int,
    filehandle: TextIO,
    DEVICE: torch.device,
    **conv_kwargs,
) -> Tuple[float, float]:
    val_loss_mat = list()
    val_acc_mat = list()
    for kfold, (train_data, val_data) in enumerate(kfolds_splitter):
        # TODO: technically don't need to transfer all attr
        train_data = train_data.to(device=DEVICE)
        val_data = val_data.to(device=DEVICE)
        model = CherryModel(
            data=train_data,
            gnn_hidden_dims=gnn_hidden_dims,
            linear_hidden_dims=linear_hidden_dims,
            conv_layer=conv_layer,
            dropout=dropout,
            **conv_kwargs,
        ).to(device=DEVICE)
        model.apply(reset_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        val_loss, val_acc = training_loop(
            n_epochs=n_epochs,
            log_epochs=log_epochs,
            kfold=kfold,
            train_data=train_data,
            val_data=val_data,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            full_edge_type=full_edge_type,
            conv_layer=conv_layer,
            gnn_hidden_dims=gnn_hidden_dims,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            filehandle=filehandle,
            **conv_kwargs,
        )
        val_loss_mat.append(val_loss)
        val_acc_mat.append(val_acc)

    val_loss_mat = np.array(val_loss_mat)
    val_acc_mat = np.array(val_acc_mat)

    avg_val_loss = val_loss_mat.mean(axis=0)
    avg_val_acc = val_acc_mat.mean(axis=0)
    return avg_val_loss.max(), avg_val_acc.max()


def main(
    datafile: str,
    kfolds: int,
    n_conv_layers: int,
    gnn_hidden_dim: int,
    linear_hidden_dims: Tuple[int, int],
    conv_layer: MessagePassing,
    dropout: float,
    lr: float,
    weight_decay: float,
    n_epochs: int,
    logfile: str,
    log_epochs: int,
    using_tune: bool = True,
    **conv_kwargs,
) -> None:
    gnn_hidden_dims = tuple([gnn_hidden_dim] * n_conv_layers)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(datafile).to(device=DEVICE)
    full_edge_type = ("virus", "infects", "host")
    criterion = nn.BCEWithLogitsLoss()
    with open(logfile, "w") as fp:
        header = "model\tfold\tepoch\tlayers\theads\tn_params\tlr\tweight_decay\tdropout\tdata\tloss\taccuracy\tprecision\trecall\tf1\tauprc"
        print(header, flush=True, file=fp)
        if any(issubclass(conv_layer, c) for c in DONT_ADD_SELF_LOOPS):
            conv_kwargs["add_self_loops"] = False
        splitter = train_validation_split(
            data, kfolds, full_edge_type, random_state=SEED
        )
        loss, acc = cross_validation_loop(
            kfolds_splitter=splitter,
            gnn_hidden_dims=gnn_hidden_dims,
            linear_hidden_dims=linear_hidden_dims,
            conv_layer=conv_layer,
            criterion=criterion,
            full_edge_type=full_edge_type,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            log_epochs=log_epochs,
            filehandle=fp,
            DEVICE=DEVICE,
            **conv_kwargs,
        )
        if using_tune:
            tune.report(loss=loss, accuracy=acc)


def main_helper(config: Dict[str, Any], checkpoint_dir=None) -> None:
    main(**config)


def main_tune(
    datafile: str,
    max_epochs: int,
    kfolds: int,
    logfile: str,
    log_epochs: int,
    outdir: str,
    threads: int,
    gpus: int,
    n_samples: int,
):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    config = {
        "n_conv_layers": tune.choice([1, 2, 3, 4, 5]),
        "gnn_hidden_dim": tune.sample_from(lambda _: 2 ** np.random.randint(6, 9)),
        "linear_hidden_dims": tune.sample_from(
            lambda _: tuple(sorted(2 ** np.random.randint(2, 8, size=2), reverse=True))
        ),
        "conv_layer": tune.choice(list(CONV_LAYERS.values())),
        "dropout": tune.uniform(0.25, 0.5),
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
        "kfolds": kfolds,
        "datafile": datafile,
        "n_epochs": max_epochs,
        "log_epochs": log_epochs,
        "logfile": logfile,
    }

    if config["conv_layer"] in USES_ATTENTION:
        config["heads"] = tune.choice(range(1, 9))

    scheduler = ASHAScheduler(
        metric="loss", mode="min", max_t=max_epochs, grace_period=1, reduction_factor=2
    )
    reporter = ExperimentTerminationReporter(metric_columns=["loss", "accuracy"])
    results = tune.run(
        main_helper,
        resources_per_trial={"cpu": threads, "gpu": gpus},
        num_samples=n_samples,
        config=config,
        scheduler=scheduler,
        local_dir=outdir,
        progress_reporter=reporter,
    )


if __name__ == "__main__":
    args = parse_args()

    threads = min(PHYSICAL_THREADS, args.threads)
    datafile = Path(args.input).resolve().as_posix()

    if args.tune_parameters:
        main_tune(
            datafile=datafile,
            max_epochs=args.epochs,
            kfolds=args.kfolds,
            logfile=args.output,
            log_epochs=args.logging_rate,
            threads=threads,
            gpus=args.gpus,
            outdir=args.outdir,
            n_samples=args.n_trials,
        )
    else:
        conv_kwargs = dict()
        if args.attention_heads > 0:
            conv_kwargs["heads"] = args.attention_heads

        torch.set_num_threads(threads)

        main(
            datafile=datafile,
            kfolds=args.kfolds,
            n_conv_layers=args.n_conv_layers,
            gnn_hidden_dim=args.gnn_hidden_dim,
            linear_hidden_dims=tuple(args.linear_hidden_dims),
            conv_layer=CONV_LAYERS[args.conv_layer],
            dropout=args.dropout,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            n_epochs=args.epochs,
            logfile=args.output,
            log_epochs=args.logging_rate,
            using_tune=False,
            **conv_kwargs,
        )

