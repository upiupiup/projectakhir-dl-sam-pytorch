import argparse
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

    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.dataset.lower() == "cifar10":
        dataset = Cifar(args.batch_size, args.threads)
    elif args.dataset.lower() == "svhn":
        dataset = Svhn(args.batch_size, args.threads)
    else:
        raise ValueError("Unknown dataset. Choose 'cifar10' or 'svhn'.")

    log = Log(log_each=10)

    if args.model.lower() == "wideresnet":
        model = WideResNet(
            args.depth, args.width_factor, args.dropout, in_channels=3, labels=10
        ).to(device)

    elif args.model.lower() == "resnet18":
        model = ResNet18Cifar(num_classes=10).to(device)

    else:
        raise ValueError("Unknown model: choose wideresnet or resnet18")

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

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            if use_sam:
                enable_running_stats(model)
                predictions = model(inputs)
                loss = smooth_crossentropy(
                    predictions, targets, smoothing=args.label_smoothing
                )
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)

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
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())

    log.flush()
