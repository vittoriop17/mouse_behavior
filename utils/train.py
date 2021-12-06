import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from utils.dataset import MarkersDataset
from utils.utils import load_existing_model, save_confusion_matrix
from model.simple_lstm import *
import time
import numpy as np
from model import new_lstm
from sklearn.metrics import f1_score, confusion_matrix
from utils.loss import weighted_mse
import wandb
import pandas as pd
import matplotlib.pyplot as plt


# TODO - plot center movement (in function of time)

def behavior_line(checkpoint_path, args):
    """
    Plot a histogram representing the behavior for each frame
    """
    setattr(args, "stride", 1)
    setattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    model = new_lstm.Net_w_conv(args) if args.with_conv else new_lstm.Net(args)
    checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    test_dataset = MarkersDataset(args, train=False)
    test_dl = DataLoader(test_dataset, batch_size=64)
    all_predictions = np.zeros((test_dataset.dataset.shape[0], args.n_behaviors))
    test_true_behaviors = test_dataset.get_all_classes()
    for (batch_frame_ids, batch_sequences, batch_classes) in test_dl:
        (batch_frame_ids, batch_sequences, batch_behaviors) = \
            (batch_frame_ids.to(args.device), batch_sequences.to(args.device), batch_classes.to(args.device))
        pred_behaviors, _ = model(batch_sequences)
        # Collapse predictions by frame id (remember: same frame may be in several sequences --> then, collapse)
        all_predictions = collapse_predictions(pred_behaviors, batch_frame_ids, all_predictions)
    all_predictions = all_predictions.argmax(axis=-1)

    y = np.array([1] * all_predictions.shape[0])
    values = np.arange(0, all_predictions.shape[0])
    bins_groom = all_predictions == 0
    bins_non_groom = all_predictions != 0
    plt.scatter(y=y[bins_groom], x=values[bins_groom], color='blue', alpha=.7, label="Grooming", s=5)
    plt.scatter(y=y[bins_non_groom], x=values[bins_non_groom], color="red", alpha=.7, label='non grooming', s=5)

    plt.show()
    plt.savefig("prova.png")
    bins_groom = values[bins_groom] / 50
    print(bins_groom)



def test_model(checkpoint_path, args):
    setattr(args, "stride", 1)
    setattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    model = new_lstm.Net_w_conv(args) if args.with_conv else new_lstm.Net(args)
    checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    test_dataset = MarkersDataset(args, train=False)
    test_dl = DataLoader(test_dataset, batch_size=64)
    all_predictions = np.zeros((test_dataset.dataset.shape[0], args.n_behaviors))
    test_true_behaviors = test_dataset.get_all_classes()
    for (batch_frame_ids, batch_sequences, batch_classes) in test_dl:
        (batch_frame_ids, batch_sequences, batch_behaviors) = \
            (batch_frame_ids.to(args.device), batch_sequences.to(args.device), batch_classes.to(args.device))
        pred_behaviors, _ = model(batch_sequences)
        # Collapse predictions by frame id (remember: same frame may be in several sequences --> then, collapse)
        all_predictions = collapse_predictions(pred_behaviors, batch_frame_ids, all_predictions)
    all_predictions = all_predictions.argmax(axis=-1)
    test_f1_score_by_class = f1_score(test_true_behaviors, all_predictions, average=None)
    test_f1_score_lab0 = f1_score(test_true_behaviors, all_predictions, pos_label=0, average='binary')
    test_f1_score_lab1 = f1_score(test_true_behaviors, all_predictions, pos_label=1, average='binary')
    print(f"TEST RESULTS: \n"
          f"\tf1 score (label 0): {test_f1_score_lab0}\n"
          f"\tf1 score (label 1): {test_f1_score_lab1}\n"
          f"\tGrooming/non-grooming f1 scores: {test_f1_score_by_class}")
    save_confusion_matrix(y_true=test_true_behaviors, y_pred=all_predictions,
                          classes=['grooming', 'non-grooming'], name_method="LSTM-based architecture")


def train_test_dataloader(args):
    train_dataset = MarkersDataset(args, train=True)
    # mean, std = train_dataset.get_stats()
    test_dataset = MarkersDataset(args, train=False)
    train_dl = DataLoader(train_dataset, batch_size=getattr(args, 'batch_size'), shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=getattr(args, 'batch_size'), shuffle=True)
    if np.any(train_dataset.cols_coords!=test_dataset.cols_coords):
        raise RuntimeError("Coordinate columns should be equal for train and test dataset")
    return train_dl, test_dl, train_dataset.cols_coords[1:-1]


def train_wrapper(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alpha = getattr(args, "alpha", 0.5)
    setattr(args, "device", device)
    train_dataloader, test_dataloader, coord_cols = train_test_dataloader(args)
    model = new_lstm.Net_w_conv(args) if args.with_conv else new_lstm.Net(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if getattr(args, "load_model", False) and getattr(args, "checkpoint_path", False):
        try:
            load_existing_model(model, optimizer, checkpoint_path=args.checkpoint_path)
        except Exception as e:
            print(f"During loading the model, the following exception occured: {e}")
            print("The execution will continue anyway")
    model = model.to(args.device)
    if getattr(args, "train_only", False):
        model, history = only_train_model(model, optimizer, train_dataloader, args,
                                          coord_cols=coord_cols, alpha=alpha)
        test_model(args.checkpoint_path, args)

    else:
        model, history = train_evaluation_model(model, optimizer, train_dataloader, test_dataloader, args,
                                                coord_cols=coord_cols, alpha=alpha)


def collapse_predictions(batch_pred_behaviors: torch.tensor, batch_frame_ids, accumulator):
    """
    :param batch_pred_behaviors: tensor. Shape: (N, SEQ_LENGTH, 3)
    :param batch_frame_ids: tensor. Shape: (N, SEQ_LENGTH)
    :param accumulator: numpy array. Shape: (N_TOTAL_FRAMES, 3)
    :return:
    """
    batch_pred_behaviors = batch_pred_behaviors.clone().cpu().detach().numpy()
    for batch_idx, seq_frame_ids in enumerate(batch_frame_ids):
        seq_frame_ids = seq_frame_ids.type(torch.int)
        accumulator[seq_frame_ids.cpu(), :] += batch_pred_behaviors[batch_idx]
    return accumulator


def train_evaluation_model(model, optimizer, train_dataloader, test_dataloader, args, coord_cols, alpha=0.5):
    checkpoint_path = args.checkpoint_path if getattr(args, "checkpoint_path", None) is not None else "checkpoint_path"
    flag_checkpoint = False
    print(f"TRAIN + EVALUATION  IS STARTING...\n"
          f"Experiment: {args.name}")
    if getattr(args, "device", "cpu") is not "cpu":
        wandb.init(project="mouse_project", entity="vittoriop", name=args.name, config=args.__dict__)
        wandb.watch(model)
    n_epochs = getattr(args, 'n_epochs')
    model.train()
    classification_criterion = nn.NLLLoss(weight=torch.tensor([0.9, 0.1])).to(args.device)  # nn.MSELoss().to(args.device)
    denoising_criterion = weighted_mse  # nn.MSELoss().to(args.device)  # nn.MSELoss().to(args.device)

    history = dict(train_classification_losses=[],
                   train_denoising_losses=[],
                   test_classification_losses=[],
                   train_f1_score=[],
                   test_f1_score=[],
                   micro_train_f1_score=[],
                   micro_test_f1_score=[],
                   best_grooming_f1_score=0)
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
            if args.multitask:
                denoising_loss = \
                    denoising_criterion(pred_trajectories, batch_sequences[:, :, coord_cols], batch_likelihoods)
                multi_task_loss = alpha * classification_loss + (1 - alpha) * denoising_loss
                train_batch_denoising_losses.append(denoising_loss.item())
            else:
                multi_task_loss = classification_loss
            multi_task_loss.backward()
            optimizer.step()
            train_batch_classification_losses.append(classification_loss.item())
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
                test_classification_loss = \
                    classification_criterion(pred_behaviors.view(-1, args.n_behaviors), batch_behaviors.view(-1,))
                test_batch_classification_losses.append(test_classification_loss.item())
                # Collapse predictions by frame id (remember: same frame may be in several sequences --> then, collapse)
                test_frame_pred_accumulator = collapse_predictions(pred_behaviors, batch_frame_ids, test_frame_pred_accumulator)
        te = time.time()
        train_classification_loss = np.mean(train_batch_classification_losses)
        test_classification_loss = np.mean(test_batch_classification_losses)
        if args.multitask:
            train_denoising_loss = np.mean(train_batch_denoising_losses)
            history['train_denoising_losses'].append(train_denoising_loss)
            print(f"\tDENOISING:\t\t train loss: {train_denoising_loss}")
        history['train_classification_losses'].append(train_classification_loss)
        history['test_classification_losses'].append(test_classification_loss)
        history['train_f1_score'].append(f1_score(train_true_behaviors, train_frame_pred_accumulator.argmax(axis=-1), average=None))
        history['test_f1_score'].append(f1_score(test_true_behaviors, test_frame_pred_accumulator.argmax(axis=-1), average=None))
        history['micro_train_f1_score'].append(f1_score(train_true_behaviors, train_frame_pred_accumulator.argmax(axis=-1), average="micro"))
        history['micro_test_f1_score'].append(f1_score(test_true_behaviors, test_frame_pred_accumulator.argmax(axis=-1), average="micro"))
        print(f"Epoch: {epoch}  \t(time: {te - ts} )\n"
              f"\tCLASSIFICATION:\t\t train loss: {train_classification_loss}  "
              f"test loss: {test_classification_loss} \n"
              f"\tCLASSIFICATION MICRO F1 score: \t train: {history['train_f1_score'][-1]}  "
              f"test: {history['test_f1_score'][-1]}\n")
        if args.device != 'cpu':
            wandb.log({'test_grooming_f1_score': history['test_f1_score'][-1][0]})
        if history["best_grooming_f1_score"] < history['test_f1_score'][-1][0] and getattr(args, "save_model", False):
            previous_best_score = history["best_grooming_f1_score"]
            current_best_score = history['test_f1_score'][-1][0]
            print(f"SAVING CURRENT MODEL ...")
            print(f"Previous best grooming F1-score: {previous_best_score},"
                  f"\tCurrent best F1-score: {current_best_score}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_grooming_f1_score': current_best_score,
                'args': args,
                'all_train_f1_scores_by_class': history['train_f1_score'],
                'all_test_f1_scores_by_class': history['test_f1_score'],
                'all_train_f1_scores': history['micro_train_f1_score'],
                'all_test_f1_scores': history['micro_test_f1_score']
            }, checkpoint_path)
            history["best_grooming_f1_score"] = history['test_f1_score'][-1][0]
            flag_checkpoint = True
        if getattr(args, "device", "cpu") != "cpu":
            log_all_losses(history)
    if flag_checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        best_model = checkpoint['model_state_dict']
        best_score = checkpoint['best_grooming_f1_score']
        epoch = checkpoint['epoch']
        print(f"SAVING FINAL MODEL ...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': best_model,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_grooming_f1_score': best_score,
            'args': args,
            'all_train_f1_scores_by_class': history['train_f1_score'],
            'all_test_f1_scores_by_class': history['test_f1_score'],
            'all_train_f1_scores': history['micro_train_f1_score'],
            'all_test_f1_scores': history['micro_test_f1_score']
        }, checkpoint_path)
    elif getattr(args, "save_model", False):
        current_score = history['test_f1_score'][-1][0]
        print(f"SAVING FINAL MODEL ...")
        torch.save({
            'epoch': n_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_grooming_f1_score': current_score,
            'args': args,
            'all_train_f1_scores_by_class': history['train_f1_score'],
            'all_test_f1_scores_by_class': history['test_f1_score'],
            'all_train_f1_scores': history['micro_train_f1_score'],
            'all_test_f1_scores': history['micro_test_f1_score']
        }, checkpoint_path)
    print(f"BEST GROOMING F1 SCORE: {history['best_grooming_f1_score']}")
    return model.eval(), history


def only_train_model(model, optimizer, train_dataloader, args, coord_cols, alpha=0.5):
    checkpoint_path = args.checkpoint_path if getattr(args, "checkpoint_path", None) is not None else "checkpoint_path"
    flag_checkpoint = False
    print(f"TRAIN ONLY IS STARTING...\n"
          f"Experiment: {args.name} ")
    if getattr(args, "device", "cpu") is not "cpu":
        wandb.init(project="mouse_project", entity="vittoriop", name=args.name, config=args.__dict__)
        wandb.watch(model)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    n_epochs = getattr(args, 'n_epochs')
    model.train()
    classification_criterion = nn.NLLLoss(weight=torch.tensor([0.9, 0.1])).to(args.device)  # nn.MSELoss().to(args.device)
    denoising_criterion = weighted_mse  # nn.MSELoss().to(args.device)  # nn.MSELoss().to(args.device)

    history = dict(train_classification_losses=[],
                   train_denoising_losses=[],
                   train_f1_score=[],
                   micro_train_f1_score=[],
                   train_grooming_f1_score=[],
                   best_train_grooming_f1_score=0)
    # *_dataloader.dataset.dataset.shape[0] represents the whole number of frames inside the respective
    # dataloader (train or test). This value is necessary in order to collapse different predictions for the same frame
    n_train_frames = train_dataloader.dataset.dataset.shape[0]
    # train_true_behaviors shape: [n_train_frames, 1]
    train_true_behaviors = train_dataloader.dataset.get_all_classes()
    for epoch in range(1, n_epochs + 1):
        train_frame_pred_accumulator = np.zeros((n_train_frames, args.n_behaviors))
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
            if args.multitask:
                denoising_loss = \
                    denoising_criterion(pred_trajectories, batch_sequences[:, :, coord_cols], batch_likelihoods)
                multi_task_loss = alpha * classification_loss + (1 - alpha) * denoising_loss
                train_batch_denoising_losses.append(denoising_loss.item())
            else:
                multi_task_loss = classification_loss
            multi_task_loss.backward()
            optimizer.step()
            train_batch_classification_losses.append(classification_loss.item())
            # Collapse predictions by frame id (remember: same frame may be in several sequences --> then, collapse)
            train_frame_pred_accumulator = collapse_predictions(pred_behaviors, batch_frame_ids, train_frame_pred_accumulator)
        # scheduler.step()
        te = time.time()
        train_classification_loss = np.mean(train_batch_classification_losses)
        if args.multitask:
            train_denoising_loss = np.mean(train_batch_denoising_losses)
            history['train_denoising_losses'].append(train_denoising_loss)
            print(f"\tDENOISING:\t\t train loss: {train_denoising_loss}")
        history['train_classification_losses'].append(train_classification_loss)
        history['train_f1_score'].append(f1_score(train_true_behaviors, train_frame_pred_accumulator.argmax(axis=-1), average=None))
        history['micro_train_f1_score'].append(f1_score(train_true_behaviors, train_frame_pred_accumulator.argmax(axis=-1), average="micro"))
        print(f"Epoch: {epoch}  \t(time: {te - ts} )\n"
              f"\tCLASSIFICATION:\t\t train loss: {train_classification_loss}  "
              f"\tCLASSIFICATION MICRO F1 score: \t train: {history['train_f1_score'][-1]}  ")
        if args.device != 'cpu':
            wandb.log({'train_grooming_f1_score': history['train_f1_score'][-1][0]})
        if history["best_train_grooming_f1_score"] < history['train_f1_score'][-1][0] and getattr(args, "save_model", False):
            previous_best_score = history["best_train_grooming_f1_score"]
            current_best_score = history['train_f1_score'][-1][0]
            print(f"SAVING CURRENT MODEL ...")
            print(f"Previous best TRAIN grooming F1-score: {previous_best_score},"
                  f"\tCurrent best TRAIN grooming F1-score: {current_best_score}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_train_grooming_f1_score': current_best_score,
                'args': args,
                'all_train_f1_scores_by_class': history['train_f1_score'],
                'all_train_f1_scores': history['micro_train_f1_score']
            }, checkpoint_path)
            history["best_train_grooming_f1_score"] = history['train_f1_score'][-1][0]
            flag_checkpoint = True
        if getattr(args, "device", "cpu") != "cpu":
            log_all_losses(history)
    if flag_checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        best_model = checkpoint['model_state_dict']
        best_score = checkpoint['best_train_grooming_f1_score']
        epoch = checkpoint['epoch']
        print(f"SAVING FINAL MODEL ...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': best_model,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_train_grooming_f1_score': best_score,
            'args': args,
            'all_train_f1_scores_by_class': history['train_f1_score'],
            'all_train_f1_scores': history['micro_train_f1_score']
        }, checkpoint_path)
    elif getattr(args, "save_model", False):
        current_score = history['train_f1_score'][-1][0]
        print(f"SAVING FINAL MODEL ...")
        torch.save({
            'epoch': n_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_train_grooming_f1_score': current_score,
            'args': args,
            'all_train_f1_scores_by_class': history['train_f1_score'],
            'all_train_f1_scores': history['micro_train_f1_score']
        }, checkpoint_path)
    print(f"BEST GROOMING F1 SCORE: {history['best_train_grooming_f1_score']}")
    return model.eval(), history


def log_all_losses(history):
    for k, v in history.items():
        if isinstance(v, list) and len(v)>0:
            wandb.log({k: v[-1]})


