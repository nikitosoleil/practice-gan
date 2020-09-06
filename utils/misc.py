from typing import Tuple

from torch.utils.data import Dataset, random_split, Subset

from configs import Config


def train_test_split(dataset: Dataset) -> Tuple[Subset, ...]:
    train_size = int(len(dataset) * Config.train_portion)
    test_size = len(dataset) - train_size
    return tuple(random_split(dataset, [train_size, test_size]))
