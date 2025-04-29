import torch
import random

from torch.utils.data import Dataset
from torchvision import transforms, datasets

from .logger import logger

class MnistDataset(Dataset):
    def __init__(
        self,
        transform=None,
        train=True,
        path='./data',
        indices = None
    ):
        logger.info(f"Loading MNIST dataset (train={train})")

        self.mnist = datasets.MNIST(
            root=path,
            train=train,
            download=True,
            transform=transform
        )

        self.indices = indices if indices is not None else list(range(len(self.mnist)))

        logger.info(f"Using {len(self.indices)} samples from MNIST dataset")

        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image, label = self.mnist[actual_idx]
        return image, label

    def get_n_classes(self):
        return 10

def get_transforms(train=True, image_size=(28, 28)):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(3),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(3),
            transforms.ToTensor(),
        ])


def discrimination_collate_fn(batch, n_distractors=4, seed=42):
    # random.seed(seed)
    # poss = []

    images, labels = zip(*batch)
    images = list(images)

    images_vectors_sender = []
    images_vectors_receiver = []
    sender_positions = []

    for i in range(len(images)):
        sender_image = images[i]
        sender_label = labels[i]
        images_vectors_sender.append(sender_image)

        distractor_candidates = [
            (img, lbl) for img, lbl in zip(images, labels) if lbl != sender_label
        ]

        sampled_distractors = random.sample(distractor_candidates, n_distractors)
        distractor_images = [img for img, _ in sampled_distractors]

        insert_pos = random.randint(0, n_distractors)
        receiver_images = distractor_images.copy()
        receiver_images.insert(insert_pos, sender_image)

        sender_positions.append(insert_pos)
        images_vectors_receiver.append(receiver_images)
    # poss.append(insert_pos)
    # poss

    images_vectors_sender = torch.stack(images_vectors_sender)
    images_vectors_receiver = torch.stack(
        [torch.stack(receivers) for receivers in images_vectors_receiver]
    )
    labels = torch.tensor(sender_positions, dtype=torch.float)

    return images_vectors_sender, labels, images_vectors_receiver, {}


def get_datasets(opts):
    logger.info(f"Loading dataset: {opts.dataset}")
    opts.dataset = "mnist"
    if opts.dataset == "mnist":
        train_dataset = MnistDataset(
            transform=get_transforms(
                train=True,
                image_size=(
                    opts.image_width,
                    opts.image_height
                )
            ),
            train=True,
        )

        test_dataset = MnistDataset(
            transform=get_transforms(
                train=False,
                image_size=(
                    opts.image_width,
                    opts.image_height
                )
            ),
            train=False,
        )

    else:
        raise ValueError(f"Dataset {opts.dataset} not recognized")
    if opts.pretrain_data_split > 0:
        logger.info("Splitting training data for pretraining")

        total_train_indices = list(range(len(train_dataset)))
        total_test_indices = list(range(len(test_dataset)))

        vision_train_indices, train_indices = torch.utils.data.random_split(
            total_train_indices,
            [opts.pretrain_data_split, 1 - opts.pretrain_data_split],
            generator=torch.Generator().manual_seed(opts.seed)
        )

        vision_test_indices, test_indices = torch.utils.data.random_split(
            total_test_indices,
            [opts.pretrain_data_split, 1 - opts.pretrain_data_split],
            generator=torch.Generator().manual_seed(opts.seed)
        )

        vision_train_dataset = MnistDataset(
            transform=train_dataset.transform,
            indices=vision_train_indices.indices,
            train=True,
        )
        vision_test_dataset = MnistDataset(
            transform=test_dataset.transform,
            indices=vision_test_indices.indices,
            train=False,
        )

        train_dataset = MnistDataset(
            transform=train_dataset.transform,
            indices=train_indices.indices,
            train=True,
        )

        test_dataset = MnistDataset(
            transform=test_dataset.transform,
            indices=test_indices.indices,
            train=False,
        )

    else:
        vision_train_dataset = train_dataset
        vision_test_dataset = test_dataset

    return train_dataset, test_dataset, vision_train_dataset, vision_test_dataset


def get_loaders(opts, train_dataset, test_dataset, vision_train_dataset, vision_test_dataset):
    kwargs = {"num_workers": opts.n_workers, "pin_memory": True} if opts.cuda else {}

    vision_train_loader = torch.utils.data.DataLoader(
        dataset=vision_train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        drop_last=True,
        **kwargs
    )

    vision_test_loader = torch.utils.data.DataLoader(
        dataset=vision_test_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        drop_last=True,
        **kwargs
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        collate_fn=lambda batch: discrimination_collate_fn(
            batch, n_distractors=opts.n_distractors, seed=opts.random_seed
        ),
        drop_last=True,
        **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        collate_fn=lambda batch: discrimination_collate_fn(
            batch, n_distractors=opts.n_distractors, seed=opts.random_seed
        ),
        drop_last=True,
        **kwargs
    )
    return train_loader, test_loader, vision_train_loader, vision_test_loader

