#!/usr/bin/env python3
import argparse
import os
from itertools import product
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, TextIO, Tuple, Union

import numpy as np
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
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
from training import METRICS, evaluate, forward, train
from utils import (
    CONV_NAME_PATT,
    SEED,
    ExperimentTerminationReporter,
    load_data,
    num_params,
    reset_weights,
    train_validation_split,
)

np.random.seed(SEED)
torch.manual_seed(SEED)

PHYSICAL_THREADS = os.cpu_count() // 2  # assuming dual-threaded cores
# TODO: remove a bunch of stuff into submodules
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    io_args = parser.add_argument_group("IO")
    io_args.add_argument(
        "-i", "--input", required=True, help="required input pytorch geometric data file",
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
    io_args.add_argument(
        "-s",
        "--save",
        help="name to save model state dict",
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
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
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
        "--conv-layers",
        nargs="+",
        type=str,
        choices=CONV_LAYERS.keys(),
        default="all",
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
    DEVICE: torch.device,
    using_tune: bool = True,
    **conv_kwargs,
) -> None:
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
                val_acc = 0.0

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
                    for metric_name, metric in METRICS.items():
                        score = evaluate(y_out, y_true, metric, DEVICE)
                        metrics[dcat].append(score)

                        if dcat == "val" and metric_name == "Accuracy":
                            val_acc = score

                    metrics_line = print_metrics(metrics[dcat])
                    print("\t".join(metrics_line), flush=True, file=filehandle)

            if using_tune:
                # TODO: report more frequently by placing this in the epochs training loop
                tune.report(loss=val_loss, accuracy=val_acc)


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
    using_tune: bool = True,
    save=None,
    **conv_kwargs,
):
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
        training_loop(
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
            DEVICE=DEVICE,
            using_tune=using_tune,
            **conv_kwargs,
        )
    if save is not None:
        torch.save(model.state_dict(), save)


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
    logfile_handle: Union[str, TextIO],
    log_epochs: int,
    using_tune: bool = True,
    save = None,
    **conv_kwargs,
) -> None:
    gnn_hidden_dims = tuple([gnn_hidden_dim] * n_conv_layers)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(datafile).to(device=DEVICE)
    full_edge_type = ("virus", "infects", "host")
    criterion = nn.BCEWithLogitsLoss()
    if any(issubclass(conv_layer, c) for c in DONT_ADD_SELF_LOOPS):
        conv_kwargs["add_self_loops"] = False
    splitter = train_validation_split(data, kfolds, full_edge_type, random_state=SEED)

    cx_val_loop = partial(
        cross_validation_loop,
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
        # filehandle=logfile_handle,
        DEVICE=DEVICE,
        using_tune=using_tune,
        save=save,
        **conv_kwargs,
    )

    if isinstance(logfile_handle, str):
        with open(logfile_handle, "w") as fp:
            cx_val_loop(filehandle=fp)
    else:
        cx_val_loop(filehandle=logfile_handle)


def main_helper(config: Dict[str, Any], checkpoint_dir=None) -> None:
    main(**config)


def main_tune(
    datafile: str,
    n_epochs: int,
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
    # TODO: can write a function that will choose n_conv_layers first to affect length of gnn_hidden_dim
    config = {
        "n_conv_layers": tune.choice([1, 2, 3, 4, 5]),
        "gnn_hidden_dim": tune.sample_from(lambda _: 2 ** np.random.randint(6, 9)),
        "linear_hidden_dims": tune.sample_from(
            lambda _: tuple(sorted(2 ** np.random.randint(2, 8, size=2), reverse=True))
        ),
        "conv_layer": tune.choice(list(CONV_LAYERS.values())),
        "dropout": tune.uniform(0.25, 0.5),
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-5, 1e-1),
        "kfolds": kfolds,
        "datafile": datafile,
        "n_epochs": n_epochs,
        "log_epochs": log_epochs,
        "logfile_handle": logfile,
    }

    if config["conv_layer"] in USES_ATTENTION:
        config["heads"] = tune.choice(range(1, 9))

    # reporting every log_epochs out of n_epochs
    # within kfolds cross-validation
    max_epochs = kfolds * n_epochs // log_epochs
    scheduler = ASHAScheduler(
        metric="loss", mode="min", max_t=max_epochs, grace_period=1, reduction_factor=2,
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
    logbase = os.path.basename(args.output).rsplit(".")[0]
    gh = args.gnn_hidden_dim
    lh = args.linear_hidden_dims
    heads = args.attention_heads
    logfile = f"{logbase}_heads-{heads}_gh-{gh}_lh-{','.join(map(str,lh))}.tsv"

    if args.tune_parameters:
        main_tune(
            datafile=datafile,
            n_epochs=args.epochs,
            kfolds=args.kfolds,
            logfile=logfile,
            log_epochs=args.logging_rate,
            threads=threads,
            gpus=args.gpus,
            outdir=args.outdir,
            n_samples=args.n_trials,
        )
    else:
        # conv_kwargs = dict()
        # if args.attention_heads > 0:
        #     conv_kwargs["heads"] = args.attention_heads

        if args.conv_layers == "all":
            conv_layers = list(CONV_LAYERS.values())
        else:
            conv_layers = [CONV_LAYERS[c] for c in args.conv_layers]

        torch.set_num_threads(threads)
        with open(logfile, "w") as fp:
            header = "model\tfold\tepoch\tlayers\theads\tn_params\tlr\tweight_decay\tdropout\tdata\tloss\taccuracy\tprecision\trecall\tf1\tauprc"
            print(header, flush=True, file=fp)
            for n_conv_layers, conv_layer in product(args.n_conv_layers, conv_layers):
                conv_kwargs = dict()
                if conv_layer in USES_ATTENTION:
                    conv_kwargs["heads"] = args.attention_heads

                # if conv_layer in DONT_ADD_SELF_LOOPS:
                #     conv_kwargs["add_self_loops"] = False
                main(
                    datafile=datafile,
                    kfolds=args.kfolds,
                    n_conv_layers=n_conv_layers,
                    gnn_hidden_dim=args.gnn_hidden_dim,
                    linear_hidden_dims=tuple(args.linear_hidden_dims),
                    conv_layer=conv_layer,
                    dropout=args.dropout,
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                    n_epochs=args.epochs,
                    logfile_handle=fp,
                    log_epochs=args.logging_rate,
                    using_tune=False,
                    save=args.save,
                    **conv_kwargs,
                )

