import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset


class Svhn:
    def __init__(self, batch_size, threads):
        mean, std = self._get_statistics()

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

        # gunakan train + extra sebagai training data (sesuai paper)
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
            full_train, batch_size=batch_size, shuffle=True, num_workers=threads
        )
        self.test = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=threads
        )

        self.classes = tuple(str(i) for i in range(10))

    def _get_statistics(self):
        # hitung mean/std dari train + extra (tanpa augment)
        raw_train = torchvision.datasets.SVHN(
            root="./data_svhn",
            split="train",
            download=True,
            transform=transforms.ToTensor(),
        )
        raw_extra = torchvision.datasets.SVHN(
            root="./data_svhn",
            split="extra",
            download=True,
            transform=transforms.ToTensor(),
        )

        merged = ConcatDataset([raw_train, raw_extra])
        loader = DataLoader(merged, batch_size=512)

        data = torch.cat([d[0] for d in loader])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
