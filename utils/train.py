import torch
from torch.utils.data import DataLoader, random_split
from utils.dataset import MarkersDataset
from utils.utils import load_existing_model
from model.simple_lstm import *
import time
import numpy as np


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


def train_wrapper(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setattr(args, "device", device)
    train_ds = MarkersDataset(args, train=True)
    mean, std = train_ds.get_stats()
    test_ds = MarkersDataset(args, mean=mean, std=std, train=False)
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)
    model = None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if getattr(args, "load_model", False) and getattr(args, "checkpoint_path", False):
        raise NotImplementedError("Implement network and optimizer")
        model = load_existing_model(model, optimizer, checkpoint_path=args.checkpoint_path)
    model = model.to(args.device)
    model, history = train_model(model, optimizer, train_dataloader, test_dataloader, args)


def train_model(model, optimizer, train_dataloader, test_dataloader, args):
    n_epochs = getattr(args, 'n_epochs')
    model.train()

    criterion = nn.MSELoss().to(args.device)

    history = dict(train=[], val=[])

    for epoch in range(1, n_epochs + 1):
        model = model.train()
        ts = time.time()
        train_losses = []

        for in_seq, true_seq in train_dataloader:
            optimizer.zero_grad()

            in_seq = in_seq.to(args.device)
            true_seq = true_seq.to(args.device)
            pred_seq = model(in_seq)

            loss = criterion(pred_seq, true_seq)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for in_seq, true_seq in test_dataloader:
                in_seq = in_seq.to(args.device)
                true_seq = true_seq.to(args.device)
                pred_seq = model(in_seq)

                loss = criterion(pred_seq, true_seq)

                val_losses.append(loss.item())
        te = time.time()
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        print(f"Epoch: {epoch}  train loss: {train_loss}  val loss: {val_loss}  time: {te - ts} ")

    return model.eval(), history