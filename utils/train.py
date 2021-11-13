import torch
from torch.utils.data import DataLoader, random_split
from utils.dataset import MarkersDataset
from utils.utils import load_existing_model
from model.simple_lstm import *
import time
import numpy as np
from model import new_lstm


def train_test_dataloader(args):
    train_dataset = MarkersDataset(args, train=True)
    mean, std = train_dataset.get_stats()
    test_dataset = MarkersDataset(args, mean=mean, std=std, train=False)
    train_dl = DataLoader(train_dataset, batch_size=getattr(args, 'batch_size'), shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=getattr(args, 'batch_size'), shuffle=True)
    return train_dl, test_dl


def train_wrapper(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setattr(args, "device", device)
    train_dataloader, test_dataloader = train_test_dataloader(args)
    model = new_lstm.Net(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if getattr(args, "load_model", False) and getattr(args, "checkpoint_path", False):
        try:
            model = load_existing_model(model, optimizer, checkpoint_path=args.checkpoint_path)
        except Exception as e:
            print(f"During loading the model, the following exception occured: {e}")
            print("The execution will continue anyway")
    model = model.to(args.device)
    model, history = train_model(model, optimizer, train_dataloader, test_dataloader, args)


def train_model(model, optimizer, train_dataloader, test_dataloader, args, alpha=0.5):
    n_epochs = getattr(args, 'n_epochs')
    model.train()
    # TODO - define loss criterions
    classification_criterion = NotImplementedError  # nn.MSELoss().to(args.device)
    denoising_criterion = NotImplementedError  # nn.MSELoss().to(args.device)

    history = dict(train_classification_losses=[],
                   train_denoising_losses=[],
                   test_classification_losses=[])

    for epoch in range(1, n_epochs + 1):
        model = model.train()
        ts = time.time()
        train_batch_classification_losses, train_batch_denoising_losses = [], []
        for _, _, _ in train_dataloader:
            optimizer.zero_grad()
            # TODO - prepare input from train_dataloader
            # in_seq = in_seq.to(args.device)
            # true_seq = true_seq.to(args.device)
            # pred_seq = model(in_seq)

            # Remember, this is a MULTI-TASK network.
            # The two losses are evaluated only during training
            classification_loss = classification_criterion(_, _)
            denoising_loss = denoising_criterion(_, _)
            multi_task_loss = alpha * classification_loss + (1 - alpha) * denoising_loss
            multi_task_loss.backward()
            optimizer.step()
            train_batch_classification_losses.append(classification_loss.item())
            train_batch_denoising_losses.append(denoising_loss.item())

        test_batch_classification_losses = []
        model = model.eval()
        with torch.no_grad():
            for _, _, _ in test_dataloader:
                # TODO - prepare input data from dataloader
                # in_seq = in_seq.to(args.device)
                # true_seq = true_seq.to(args.device)
                # pred_seq = model(in_seq)

                test_classification_loss = classification_loss(_, _)
                test_batch_classification_losses.append(test_classification_loss.item())

        te = time.time()
        train_classification_loss = np.mean(train_batch_classification_losses)
        test_classification_loss = np.mean(test_batch_classification_losses)
        train_denoising_loss = np.mean(train_batch_denoising_losses)
        history['train_classification_losses'].append(train_classification_loss)
        history['test_classification_losses'].append(test_classification_loss)
        history['train_denoising_losses'].append(train_denoising_loss)

        print(f"Epoch: {epoch}  \t(time: {te - ts} )"
              f"\tClassification: train loss: {train_classification_loss}  "
              f"test loss: {test_classification_loss} \n"
              f"\tDenoising train loss: {train_denoising_loss}")

    return model.eval(), history
