import torch
from torch import nn
from model.simple_lstm import *
import time
import numpy as np


def train_model(train_dataset, val_dataset, params):
    model = LSTMAutoEncoder(params)
    model = model.to(params.device)
    n_epochs = getattr(params,'n_epochs')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    criterion = nn.MSELoss().to(params.device)

    history = dict(train=[], val=[])

    for epoch in range(1, n_epochs + 1):
        model = model.train()
        ts = time.time()
        train_losses = []

        for in_seq, true_seq in train_dataset:
            optimizer.zero_grad()

            in_seq = in_seq.to(params.device)
            true_seq = true_seq.to(params.device)
            pred_seq = model(in_seq)

            loss = criterion(pred_seq, true_seq)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for in_seq, true_seq in val_dataset:
                in_seq = in_seq.to(params.device)
                true_seq = true_seq.to(params.device)
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