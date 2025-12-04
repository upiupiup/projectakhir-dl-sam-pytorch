import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset


class Svhn:
    def __init__(self, batch_size, threads):
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(size=(32, 32), padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        train_set = torchvision.datasets.SVHN(
            root="./data_svhn",
            split="train",
            download=True,
            transform=train_transform,
        )
        extra_set = torchvision.datasets.SVHN(
            root="./data_svhn",
            split="extra",
            download=True,
            transform=train_transform,
        )
        test_set = torchvision.datasets.SVHN(
            root="./data_svhn",
            split="test",
            download=True,
            transform=test_transform,
        )

        full_train = ConcatDataset([train_set, extra_set])

        self.train = DataLoader(
            full_train,  # ✅ pakai gabungan train + extra
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Windows-friendly
            pin_memory=False,
        )

        self.test = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        self.classes = tuple(str(i) for i in range(10))

    def _get_statistics(self):
        pass
