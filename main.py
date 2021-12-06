import torch
from utils import utils
from model import new_lstm
from utils import dataset, train
import numpy as np
import pandas as pd


def save_trajectories(train_dataset: dataset.MarkersDataset, test_dataset: dataset.MarkersDataset, train_pred, test_pred, params, threshold=0.5):
    cols_lh = train_dataset.cols_likelihood
    cols_coords = train_dataset.cols_coords
    train_mask = train_dataset.original_dataset[:, cols_lh] < threshold
    train_mask = np.concatenate([np.concatenate((row.reshape(-1,1), row.reshape(-1,1)), axis=1) for row in train_mask.T], axis=1)
    train_mask = train_mask.reshape(train_dataset.original_dataset.shape[0], -1)
    final_train = np.zeros((train_dataset.original_dataset[:, cols_coords].shape))
    test_mask = test_dataset.original_dataset[:, cols_lh] < threshold
    test_mask = np.concatenate([np.concatenate((row.reshape(-1,1), row.reshape(-1,1)), axis=1) for row in test_mask.T], axis=1)
    final_test = np.zeros((test_dataset.original_dataset[:, cols_coords].shape))

    for frame_id in range(final_train.shape[0]):
        if train_mask[frame_id].sum() == 0:
            continue
        idx = np.array(generate_sequence_idx(frame_id, params.final_input_size, train_dataset.n_sequences))
        frame_preds = np.array([train_pred[seq_id, in_seq_id] for seq_id, in_seq_id in idx])
        frame_preds = np.average(frame_preds, axis=0)
        np.argwhere(train_mask[frame_id] == True)
        current_mask = train_mask[frame_id]
        final_train[frame_id, current_mask] = frame_preds[current_mask]
    final_train[~train_mask] = train_dataset.original_dataset[:, cols_coords][~train_mask]

    for frame_id in range(final_test.shape[0]):
        if test_mask[frame_id].sum() == 0:
            continue
        idx = np.array(generate_sequence_idx(frame_id, params.final_input_size, test_dataset.n_sequences))
        frame_preds = np.array([test_pred[seq_id, in_seq_id] for seq_id, in_seq_id in idx])
        frame_preds = np.average(frame_preds, axis=0)
        np.argwhere(test_mask[frame_id] == True)
        current_mask = test_mask[frame_id]
        final_test[frame_id, current_mask] = frame_preds[current_mask]

    final_test[~test_mask] = test_dataset.original_dataset[:, cols_coords][~test_mask]
    final_csv_pd = pd.DataFrame(np.concatenate((final_train, final_test)))
    final_csv_pd.to_csv("final_pred.csv", header=False)


def generate_sequence_idx(frame_id, sequence_length, tot_sequences):
    # e.g.: frame_id = 0: then it will be in [0, 0]
    # frame_id = 6: then this frame will be in [6, 0], [5, 1], [4, 2], [3, 3], [2,4], [1,5]
    # the above results are true if we suppose that the sequence length is 6
    idx = [[i, frame_id - i] for i in range(frame_id, frame_id - sequence_length, -1) if (0 <= i < tot_sequences)]
    return idx[:sequence_length]


def main(params):
    train.train_wrapper(params)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # setattr(params, "device", device)
    # print("Using {} device".format(device))
    # train_dl, test_dl = data_loader.train_test_dataloader(params)
    # model, history = train_test.train_model(train_dl, test_dl, args=params)
    # torch.save(model.state_dict(), "checkpoint.pt")
    # # setattr(params, "file_path", "train_dataset.csv")
    # # train_dataset = dataset.MarkersDataset(params, train=True)
    # # setattr(params, "file_path", "test_dataset.csv")
    # # mean, std = train_dataset.get_stats()
    # # test_dataset = dataset.MarkersDataset(params, train=False, mean=mean, std=std)
    # # cols_coords = train_dataset.cols_coords
    # # train_pred = model(train_dataset.input_dataset.to(device))
    # # train_pred = (train_pred.detach().cpu().numpy() * std[cols_coords]) + mean[cols_coords]
    # # test_pred = model(test_dataset.input_dataset.to(device))
    # # test_pred = (test_pred.detach().cpu().numpy() * std[cols_coords]) + mean[cols_coords]
    # # save_trajectories(train_dataset, test_dataset, train_pred, test_pred, params)
    #
    # # print video
    # frames = utils.get_frames_from_video("groom1_edited.avi")
    # coords = np.loadtxt("final_pred.csv", delimiter=',', skiprows=0)[:, 1:]
    # utils.print_dots_on_frames(frames, coords)
    # utils.build_video(frames)

if __name__ == '__main__':
    args = utils.upload_args()
    # main(args)
    # checkpoint_path = "data\\checkpoint.pt"
    # train.denoise_trajectories_from_checkpoint(checkpoint_path, args)
    # setattr(args, "device", "cpu")
    # setattr(args, "name", "prova")
    # checkpoint_path = "data\\CHECKPOINTS\\seq_length\\checkpoint_Evaluation_seq_length_200.pt"
    train.test_model(args.checkpoint_path, args)