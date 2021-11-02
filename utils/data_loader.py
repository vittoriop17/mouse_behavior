import pandas as pd
from torch.utils.data import DataLoader, random_split
from utils.dataset import MarkersDataset


def train_test_dataloader(args):
    setattr(args, "file_path", "train_dataset.csv")
    train_dataset = MarkersDataset(args, train=True)
    mean, std = train_dataset.get_stats()
    setattr(args, "file_path", "test_dataset.csv")
    setattr(args, "train", False)
    test_dataset = MarkersDataset(args, mean=mean, std=std, train=False)
    train_dl = DataLoader(train_dataset, batch_size=getattr(args, 'batch_size'), shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=getattr(args, 'batch_size'), shuffle=True)
    return train_dl, test_dl