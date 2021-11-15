import torch
from torch.utils.data import DataLoader, random_split
from utils.dataset import MarkersDataset
from utils.utils import load_existing_model
from model.simple_lstm import *
import time
import numpy as np
from model import new_lstm
from sklearn.metrics import f1_score


def train_test_dataloader(args):
    train_dataset = MarkersDataset(args, train=True)
    mean, std = train_dataset.get_stats()
    test_dataset = MarkersDataset(args, mean=mean, std=std, train=False)
    train_dl = DataLoader(train_dataset, batch_size=getattr(args, 'batch_size'), shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=getattr(args, 'batch_size'), shuffle=True)
    if np.any(train_dataset.cols_coords!=test_dataset.cols_coords):
        raise RuntimeError("Coordinate columns should be equal for train and test dataset")
    return train_dl, test_dl, train_dataset.cols_coords[1:-1]


def train_wrapper(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setattr(args, "device", device)
    train_dataloader, test_dataloader, coord_cols = train_test_dataloader(args)
    model = new_lstm.Net(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if getattr(args, "load_model", False) and getattr(args, "checkpoint_path", False):
        try:
            model = load_existing_model(model, optimizer, checkpoint_path=args.checkpoint_path)
        except Exception as e:
            print(f"During loading the model, the following exception occured: {e}")
            print("The execution will continue anyway")
    model = model.to(args.device)
    model, history = train_model(model, optimizer, train_dataloader, test_dataloader, args, coord_cols=coord_cols)


def collapse_predictions(batch_pred_behaviors, batch_frame_ids, accumulator):
    """
    :param batch_pred_behaviors: tensor. Shape: (N, SEQ_LENGTH, 3)
    :param batch_frame_ids: tensor. Shape: (N, SEQ_LENGTH)
    :param accumulator: numpy array. Shape: (N_TOTAL_FRAMES, 3)
    :return:
    """
    for batch_idx, seq_frame_ids in enumerate(batch_frame_ids):
        seq_frame_ids = seq_frame_ids.type(torch.int)
        accumulator[seq_frame_ids, :] += batch_pred_behaviors[batch_idx].detach().cpu().numpy()
    return accumulator

def train_model(model, optimizer, train_dataloader, test_dataloader, args, coord_cols, alpha=0.5):
    n_epochs = getattr(args, 'n_epochs')
    model.train()
    # TODO - define WEIGHT for NLLLoss
    classification_criterion = nn.NLLLoss().to(args.device)  # nn.MSELoss().to(args.device)
    denoising_criterion = nn.MSELoss().to(args.device)  # nn.MSELoss().to(args.device)

    history = dict(train_classification_losses=[],
                   train_denoising_losses=[],
                   test_classification_losses=[],
                   train_f1_score=[],
                   test_f1_score=[])
    # *_dataloader.dataset.dataset.shape[0] represents the whole number of frames inside the respective
    # dataloader (train or test). This value is necessary in order to collapse different predictions for the same frame
    n_train_frames, n_test_frames = \
        train_dataloader.dataset.dataset.shape[0], test_dataloader.dataset.dataset.shape[0]
    # train_true_behaviors shape: [n_train_frames, 1]
    train_true_behaviors, test_true_behaviors = \
        train_dataloader.dataset.get_all_classes(), test_dataloader.dataset.get_all_classes()
    for epoch in range(1, n_epochs + 1):
        train_frame_pred_accumulator, test_frame_pred_accumulator = \
            np.zeros((n_train_frames, args.n_behaviors)), np.zeros((n_test_frames, args.n_behaviors))
        model = model.train()
        ts = time.time()
        train_batch_classification_losses, train_batch_denoising_losses = [], []
        for (batch_frame_ids, batch_sequences, batch_likelihoods, batch_behaviors) in train_dataloader:
            # shapes: (where N=BATCH_SIZE
            # batch_frame_ids: [N, SEQ_LENGTH, 1]
            # batch_sequences: [N, SEQ_LENGTH, INPUT_SIZE]
            # batch_likelihoods: [N, SEQ_LENGTH, n_markers] ---> n_markers=input_size/3 (input: x,y,L for each marker)
            # batch_behaviors: [N, SEQ_LENGTH, 1]
            optimizer.zero_grad()
            # prepare input
            (batch_frame_ids, batch_sequences, batch_likelihoods, batch_behaviors) = \
                (batch_frame_ids.to(args.device), batch_sequences.to(args.device),
                 batch_likelihoods.to(args.device), batch_behaviors.to(args.device))
            # Predict behaviors and denoised trajectories
            pred_behaviors, pred_trajectories = model(batch_sequences)

            # Remember, this is a MULTI-TASK network.
            # The two losses are evaluated only during training
            # each element in target has to have 0 <= value < C (target is the ground truth)
            classification_loss = \
                classification_criterion(pred_behaviors.view(-1, args.n_behaviors), batch_behaviors.view(-1, ))
            denoising_loss = \
                denoising_criterion(pred_trajectories, batch_sequences[:, :, coord_cols])
            multi_task_loss = alpha * classification_loss + (1 - alpha) * denoising_loss
            multi_task_loss.backward()
            optimizer.step()
            train_batch_classification_losses.append(classification_loss.item())
            train_batch_denoising_losses.append(denoising_loss.item())
            # Collapse predictions by frame id (remember: same frame may be in several sequences --> then, collapse)
            train_frame_pred_accumulator = collapse_predictions(pred_behaviors, batch_frame_ids, train_frame_pred_accumulator)
        test_batch_classification_losses = []
        model = model.eval()
        with torch.no_grad():
            for (batch_frame_ids, batch_sequences, batch_behaviors) in test_dataloader:
                # prepare input
                (batch_frame_ids, batch_sequences, batch_behaviors) = \
                    (batch_frame_ids.to(args.device), batch_sequences.to(args.device), batch_behaviors.to(args.device))
                pred_behaviors, _ = model(batch_sequences)
                # TODO - check pred_behaviors after 'view'
                test_classification_loss = \
                    classification_criterion(pred_behaviors.view(-1, args.n_behaviors), batch_behaviors.view(-1,))
                test_batch_classification_losses.append(test_classification_loss.item())
                # Collapse predictions by frame id (remember: same frame may be in several sequences --> then, collapse)
                test_frame_pred_accumulator = collapse_predictions(pred_behaviors, batch_frame_ids, test_frame_pred_accumulator)
        te = time.time()
        train_classification_loss = np.mean(train_batch_classification_losses)
        test_classification_loss = np.mean(test_batch_classification_losses)
        train_denoising_loss = np.mean(train_batch_denoising_losses)
        history['train_classification_losses'].append(train_classification_loss)
        history['test_classification_losses'].append(test_classification_loss)
        history['train_denoising_losses'].append(train_denoising_loss)
        history['train_f1_score'].append(f1_score(train_true_behaviors, train_frame_pred_accumulator.argmax(axis=-1), average='micro'))
        history['test_f1_score'].append(f1_score(test_true_behaviors, test_frame_pred_accumulator.argmax(axis=-1), average='micro'))
        print(f"Epoch: {epoch}  \t(time: {te - ts} )\n"
              f"\tCLASSIFICATION:\t train loss: {train_classification_loss}  "
              f"test loss: {test_classification_loss} \n"
              f"\tCLASSIFICATION MICRO F1 score: \t train: {history['train_f1_score'][-1]}"
              f"test: {history['test_f1_score'][-1]}\n"
              f"\tDENOISING:\t train loss: {train_denoising_loss}")

    return model.eval(), history
