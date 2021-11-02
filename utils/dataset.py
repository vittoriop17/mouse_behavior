import utils
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sklearn


def split_dataset(file_path, train_size=0.8):
    cont = -1
    cols = list()
    with open(file_path) as fp:
        for line in fp.readlines():
            cont += 1
            if cont == 3:
                break
            if cont == 0:
                cols = ["" for _ in line.split(",")]
                continue
            line = line.replace("\n", "")
            cols = [pre + "_" + post for post, pre in zip(cols, line.split(","))]
    dataset = pd.read_csv(file_path, skiprows=2, header=0, names=cols)
    # remove first column
    dataset.drop(columns=dataset.columns[0], inplace=True)
    n_train_sample = int(train_size * dataset.shape[0])
    columns = dataset.columns
    dataset[:n_train_sample].to_csv(path_or_buf="..\\train_dataset.csv", header=columns, index=False)
    dataset[n_train_sample:].to_csv(path_or_buf="..\\test_dataset.csv", header=columns, index=False)


class MarkersDataset(Dataset):
    def __init__(self, args, train: bool = False, mean: np.array = None, std: np.array = None):
        self.args = None
        self.original_dataset = None
        self.dataset = None
        self.cols_likelihood = None
        self.cols_coords = None
        self.columns = None
        self.input_dataset = None
        self.target_dataset = None
        self.n_sequences = None
        self.train = train
        self.mean = mean
        self.std = std
        self.transform = StandardScaler(with_std=True, with_mean=True) if train else None
        self.check_args(args)
        if not os.path.isfile(getattr(self.args, "file_path")):
            raise FileNotFoundError(f"{getattr(self.args, 'file_path')} does not exist!")
        self.initialize_dataset()
        self.create_sequences()


    def __len__(self):
        return self.input_dataset.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.input_dataset[idx]), torch.tensor(self.target_dataset[idx])

    def get_stats(self):
        if self.train:
            return self.transform.mean_, np.sqrt(self.transform.var_)
        return self.mean, self.std

    def initialize_dataset(self):
        self.dataset = pd.read_csv(getattr(self.args, 'file_path'))
        self.columns = self.dataset.columns
        self.cols_likelihood = [col.startswith("likelihood") for col in self.dataset.columns]
        self.cols_coords = [not m for m in self.cols_likelihood]
        self.dataset = np.array(self.dataset)
        self.original_dataset = self.dataset
        partial_mask = self.dataset[:, self.cols_likelihood] < getattr(self.args, "threshold")
        full_mask = np.array(
            [np.concatenate((column.reshape(-1, 1), column.reshape(-1, 1), False * np.ones((len(column), 1))), axis=1)
             for column in partial_mask.T], dtype=np.bool8)
        full_mask = np.concatenate(full_mask, axis=-1)
        self.dataset[full_mask] = 0
        self.fill_fake_dataset()
        if self.train:
            self.transform.fit(self.fake_dataset)
            self.dataset = self.transform.transform(self.dataset)
            self.fake_dataset = self.transform.transform(self.fake_dataset)
        else:
            self.dataset = (self.dataset - self.mean) / self.std
            self.fake_dataset = (self.fake_dataset - self.mean) / self.std

    def fill_fake_dataset(self):
        self.fake_dataset = self.dataset
        for index, row in enumerate(self.fake_dataset):
            app_index = index
            mask = row == 0
            if index == 0:
                while mask.sum() != 0:
                    self.fake_dataset[index][mask] = self.fake_dataset[app_index + 1][mask]
                    mask = self.fake_dataset[index] == 0
                    app_index += 1
                continue
            if index == (self.fake_dataset.shape[0] - 1):
                while mask.sum() != 0:
                    self.fake_dataset[index][mask] = self.fake_dataset[app_index - 1][mask]
                    mask = self.fake_dataset[index] == 0
                    app_index -= 1
                continue
            inc_row = np.zeros((mask.shape), dtype=np.float32)
            while mask.sum() != 0 and (app_index + 1) < self.fake_dataset.shape[0]:
                inc_row[mask] = (self.fake_dataset[app_index + 1][mask] - self.fake_dataset[index - 1][mask]) / (
                            app_index - index + 2)
                app_index += 1
                mask = np.bitwise_and(mask, self.fake_dataset[app_index] == 0)
                inc_row[mask] = 0
            mask = row == 0
            self.fake_dataset[index][mask] = self.fake_dataset[index - 1][mask] + inc_row[mask]

    def create_sequences(self):
        self.n_sequences = len(self.dataset) - self.args.sequence_length + 1
        # input_dataset: the dataset used as input for our lstm architecture (the one filled with ZEROS where the
        # likelihood is lower than 'threshold'.
        # target_dataset: the dataset used as target for our prediction, the one filled with fake values
        self.input_dataset = torch.empty((self.n_sequences, self.args.sequence_length, self.args.input_size),
                                         dtype=torch.float32)
        self.target_dataset = torch.empty((self.n_sequences, self.args.sequence_length, self.args.input_size),
                                          dtype=torch.float32)
        for sequence_idx in range(self.n_sequences):
            for time_idx in range(self.args.sequence_length):
                row_index = sequence_idx + time_idx
                self.input_dataset[sequence_idx][time_idx] = torch.from_numpy(self.dataset[row_index][self.cols_coords])
                self.target_dataset[sequence_idx][time_idx] = torch.from_numpy(
                    self.fake_dataset[row_index][self.cols_coords])
        # print(f"Input dataset shape: {self.input_dataset.shape}")
        # print(f"Target dataset shape: {self.target_dataset.shape}")

    def check_args(self, args):
        if not hasattr(args, "file_path"):
            raise Exception("Argument not found: file_path")
        if not hasattr(args, "threshold"):
            raise Exception("Argument not found: threshold")
        if not hasattr(args, "sequence_length"):
            raise Exception("Argument not found: sequence_length")
        self.args = args


if __name__ == '__main__':
    args = utils.upload_args_from_json()
    setattr(args, "file_path", "..\\data\\video_4DLC_resnet101_For_Video_October14Oct14shuffle1_111600.csv")
    setattr(args, "threshold", 0.3)
    split_dataset(args.file_path)
    setattr(args, "file_path", "..\\train_dataset.csv")
    ds = MarkersDataset(args)
