import argparse
import os
import csv
import time
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.models as models

from model.wide_res_net import WideResNet
from model.resnet_18 import ResNet18Cifar
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from data.svhn import Svhn
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

import sys

sys.path.append("..")
from sam import SAM


def make_run_name(args):
    parts = [
        args.dataset,
        args.model,
        args.optimizer,
    ]
    if args.optimizer.lower().startswith("sam"):
        parts.append(f"rho{args.rho}")
        parts.append(f"adaptive{int(bool(args.adaptive))}")
    parts.extend(
        [
            f"lr{args.learning_rate}",
            f"wd{args.weight_decay}",
            f"bs{args.batch_size}",
            f"seed{args.seed}",
        ]
    )
    return "_".join(str(p) for p in parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model choice
    parser.add_argument(
        "--model", default="wideresnet", type=str, help="wideresnet | resnet18"
    )

    # Optimizer choice
    parser.add_argument(
        "--optimizer",
        default="sam_sgd",
        type=str,
        help="sgd | adamw | sam_sgd | sam_adamw",
    )

    # Dataset choice
    parser.add_argument("--dataset", default="cifar10", type=str, help="cifar10 | svhn")

    # SAM
    parser.add_argument(
        "--adaptive",
        default=False,
        type=bool,
        help="True for ASAM; False = SAM original.",
    )
    parser.add_argument(
        "--rho", default=0.05, type=float, help="Rho for SAM (paper uses 0.05)."
    )

    # Training hyperparameters (as original)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--depth", default=16, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--learning_rate", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--threads", default=2, type=int)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--width_factor", default=8, type=int)
    parser.add_argument("--output_dir", default="./results", type=str)

    # NEW: seed untuk multi-run (3 seed dsb.)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    # Init random seed
    initialize(args, seed=args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # PATH & FILE LOGGING SETUP
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = make_run_name(args)

    # results_root = "results"
    # epochs_dir = os.path.join(results_root, "epochs")
    # ckpt_dir = os.path.join(results_root, "checkpoints")
    # summary_path = os.path.join(results_root, "summary.csv")

    # os.makedirs(results_root, exist_ok=True)
    # os.makedirs(epochs_dir, exist_ok=True)
    # os.makedirs(ckpt_dir, exist_ok=True)

    # untuk run kaggle
    results_root = args.output_dir
    epochs_dir = os.path.join(results_root, "epochs")
    ckpt_dir = os.path.join(results_root, "checkpoints")
    summary_path = os.path.join(results_root, "summary.csv")

    os.makedirs(epochs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    epoch_csv_path = os.path.join(epochs_dir, f"{run_name}_{timestamp}.csv")
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}_best.pt")

    # tulis header epoch CSV
    with open(epoch_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
        )

    # DATASET
    if args.dataset.lower() == "cifar10":
        dataset = Cifar(args.batch_size, args.threads)
    elif args.dataset.lower() == "svhn":
        dataset = Svhn(args.batch_size, args.threads)
    else:
        raise ValueError("Unknown dataset. Choose 'cifar10' or 'svhn'.")

    log = Log(log_each=10)

    # ==== MODEL ====
    if args.model.lower() == "wideresnet":
        model = WideResNet(
            args.depth, args.width_factor, args.dropout, in_channels=3, labels=10
        ).to(device)
    elif args.model.lower() == "resnet18":
        model = ResNet18Cifar(num_classes=10).to(device)
    else:
        raise ValueError("Unknown model: choose wideresnet or resnet18")

    # OPTIMIZER
    opt_name = args.optimizer.lower()
    use_sam = opt_name.startswith("sam")

    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

    elif opt_name == "sam_sgd":
        optimizer = SAM(
            model.parameters(),
            torch.optim.SGD,
            rho=args.rho,
            adaptive=args.adaptive,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    elif opt_name == "sam_adamw":
        optimizer = SAM(
            model.parameters(),
            torch.optim.AdamW,
            rho=args.rho,
            adaptive=args.adaptive,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError("Unknown optimizer")

    # Scheduler → must use base optimizer when SAM
    scheduler = (
        StepLR(optimizer.base_optimizer, args.learning_rate, args.epochs)
        if use_sam
        else StepLR(optimizer, args.learning_rate, args.epochs)
    )

    # TRAIN LOOP + LOGGING
    best_val_acc = 0.0
    best_val_epoch = 0
    best_train_acc_at_best = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        # aggregator train
        train_loss_sum = 0.0
        train_correct_sum = 0
        train_total = 0

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            if use_sam:
                # first step
                enable_running_stats(model)
                predictions = model(inputs)
                loss = smooth_crossentropy(
                    predictions, targets, smoothing=args.label_smoothing
                )
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)

                # second step
                disable_running_stats(model)
                smooth_crossentropy(
                    model(inputs), targets, smoothing=args.label_smoothing
                ).mean().backward()
                optimizer.second_step(zero_grad=True)

            else:
                predictions = model(inputs)
                loss = smooth_crossentropy(
                    predictions, targets, smoothing=args.label_smoothing
                )
                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()

            with torch.no_grad():
                correct = torch.argmax(predictions, 1) == targets
                batch_size = targets.size(0)
                train_total += batch_size
                train_correct_sum += correct.sum().item()
                train_loss_sum += loss.mean().item() * batch_size

                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        # hitung train metrics epoch
        train_loss_epoch = train_loss_sum / train_total
        train_acc_epoch = train_correct_sum / train_total

        # EVAL
        model.eval()
        log.eval(len_dataset=len(dataset.test))

        val_loss_sum = 0.0
        val_correct_sum = 0
        val_total = 0

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets

                batch_size = targets.size(0)
                val_total += batch_size
                val_correct_sum += correct.sum().item()
                val_loss_sum += loss.mean().item() * batch_size

                log(model, loss.cpu(), correct.cpu())

        val_loss_epoch = val_loss_sum / val_total
        val_acc_epoch = val_correct_sum / val_total
        current_lr = scheduler.lr()

        # SAVE BEST CHECKPOINT
        if val_acc_epoch > best_val_acc:
            best_val_acc = val_acc_epoch
            best_val_epoch = epoch
            best_train_acc_at_best = train_acc_epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": val_acc_epoch,
                    "train_acc": train_acc_epoch,
                    "args": vars(args),
                },
                ckpt_path,
            )

        # APPEND EPOCH CSV
        with open(epoch_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    train_loss_epoch,
                    train_acc_epoch,
                    val_loss_epoch,
                    val_acc_epoch,
                    current_lr,
                ]
            )

    # SUMMARY LOG
    runtime_sec = time.time() - start_time
    final_val_acc = val_acc_epoch
    final_train_acc = train_acc_epoch

    summary_header = [
        "timestamp",
        "run_name",
        "dataset",
        "model",
        "optimizer",
        "rho",
        "adaptive",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "epochs",
        "seed",
        "best_val_acc",
        "best_val_epoch",
        "train_acc_at_best",
        "final_val_acc",
        "final_train_acc",
        "runtime_sec",
    ]

    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(summary_header)
        writer.writerow(
            [
                timestamp,
                run_name,
                args.dataset,
                args.model,
                args.optimizer,
                args.rho if hasattr(args, "rho") else None,
                bool(args.adaptive),
                args.learning_rate,
                args.weight_decay,
                args.batch_size,
                args.epochs,
                args.seed,
                best_val_acc,
                best_val_epoch,
                best_train_acc_at_best,
                final_val_acc,
                final_train_acc,
                runtime_sec,
            ]
        )

    log.flush()
