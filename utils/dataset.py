import utils
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sklearn
import cv2
import utils


def check_start_end(start_end, n_frames):
    flags = None
    for idx, (start, end) in enumerate(start_end):
        start, end = int(start), int(end)
        if start >= end and not (end == -1 and idx == 0):
            raise ValueError("Inconsistent relation. start_frame must be < than end_frame")
        if idx == 0:
            end = n_frames if end==-1 else end
            flags = np.zeros((end - start, 1), dtype=int)
            continue
        flags[start:end] += 1
        if np.max(flags) > 1:
            raise ValueError(
                f"Wrong interval! The current interval {start}-{end} is overlapped with an existing interval")


def merge_behavior_and_trajectories(file_behavior,
                                    file_trajectories,
                                    save_custom_frames=False,
                                    frames_path="..\\data\\evaluation_frames_1h6min",
                                    video_path="..\\..\\ALL_VIDEOSSSS\\S1740002.MP4",
                                    merge=False, adjust_frame_id=True):
    """
    N.B: file_behavior, file_trajectories and video_path are somehow related to each other. In particular,
    they must be referred to the same video (which should be the one represented by video_path)
    File_trajectories must contain the column 'FRAME_ID': this can start from whatever number
    (e.g.: for test dataset, this should start from a number different from 0).
    Obv, the data inside file_behavior must be consistent with the frame_id inside file_trajectories
    :param video_path: str. Path to the video. This is used for extracting custom frames (if save_custom_frames=True)
    :param merge: boolean, default: False. Merge existing labels in existing TRAJECTORY DATAFRAME
    :param frames_path: str. Path to the folder that will contain the frames extracted for evaluation.
    :param save_custom_frames: boolean, default: False. Save the frames with a MARKER on them. Save only the
        frames associated to start_frame and end_frame reported in the file_behavior
    :param file_behavior: str to the .csv file representing the behavior formatted in the following way:
        format:
            class_id, start_frame, end_frame
        where
            -class_id can be: 0 or 1 (0 for grooming, 1 for climbing)
            -start_frame: absolute value of the frame where the associated behavior starts
            -end_frame: absolute value of the frame where the associated behavior ends
        Note that there is no need to specify 'general' behavior. We suppose that's the common condition, when the mouse
        is not grooming, neither climbing.
    :param file_trajectories: str to the .csv file representing the trajectories.
        format: (see train_dataset.csv and test_dataset.csv for instance)

        PLEASE NOTE:
            the numeration between trajectory frames and behavior frames must be meaningful!!!!
            Be sure that the behaviors and the trajectories are correctly linked together
    :return:
    """
    df_trajectories = pd.read_csv(file_trajectories)
    num_frames = df_trajectories.shape[0]
    behavior_start_end = np.loadtxt(file_behavior, delimiter=",", skiprows=1)
    check_start_end(behavior_start_end[:, 1:], df_trajectories.frame_id.max())
    behavior_classes = {0: "grooming", 1: "climbing", 2: "general"}
    # by default, all behaviors are set to 0
    behavior_by_frame = 1 + np.zeros((num_frames, 1)) if not merge else df_trajectories.label.to_numpy()
    for idx, row in enumerate(behavior_start_end):
        behavior, start_frame, end_frame = row
        start_frame, end_frame = int(start_frame), int(end_frame)
        if adjust_frame_id:
            start_frame, end_frame = (start_frame * 2, end_frame * 2) if end_frame != -1 else (
            start_frame * 2, end_frame)
        if behavior == -1 and idx == 0:
            end_frame = df_trajectories.frame_id.max()+1 if end_frame == -1 else end_frame
            if df_trajectories.frame_id.min() > start_frame or df_trajectories.frame_id.max() < end_frame-1:
                raise ValueError("Start or end frame not valid. Trajectory dataframe has different boundaries "
                                 f"for frame_id. \nTrajectory frame ids "
                                 f"({df_trajectories.frame_id.min()}, {df_trajectories.frame_id.max()})\n"
                                 f"Behavior frame ids ({start_frame}, {end_frame})")
            # pandas indexing works differently. It takes also the last element with index end_frame-1
            pre_mask = df_trajectories.frame_id >= start_frame
            post_mask = df_trajectories.frame_id < end_frame
            df_trajectories = df_trajectories[pre_mask & post_mask]
            behavior_by_frame = behavior_by_frame[start_frame:end_frame]
            continue
        if behavior not in behavior_classes.keys():
            raise ValueError(
                f'Invalid behavior found! Expected one of {behavior_classes.keys()}, found {behavior} instead')
        if start_frame >= num_frames:
            print("There is no trajectory information associated to the specified frame number. "
                  f"Thus, the behavior label must be discarded! {start_frame} > {num_frames}, where"
                  f"{num_frames} is the total number of frame trajectories inside the trajectory file")
            continue
        behavior_by_frame[start_frame:end_frame] = behavior
        if save_custom_frames and os.path.exists(frames_path):
            # save only two frames (the first and the last one associated to the behavior)
            for frame_number in [start_frame, end_frame]:
                utils.save_ith_frames_from_video(video_name=video_path, root_path=frames_path, frame_seq=frame_number)
                frame_path = os.path.join(frames_path, f"{frame_number}.png")
                img = cv2.imread(frame_path)
                # write dot on top of img
                RGB = (255, 0, 0) if behavior == 0 else ((0, 255, 0) if behavior == 1 else (0, 0, 0))
                img = utils.print_dots_on_frames(img.reshape(-1, *img.shape), np.array([[0, 0]]), RGB=RGB, radius=30,
                                                 thickness=30)
                cv2.imwrite(frame_path, img.squeeze(), [cv2.IMWRITE_PNG_COMPRESSION, 9])

    df_traj_n_label = pd.concat([df_trajectories,
                                 pd.DataFrame(behavior_by_frame, columns=["label"], index=df_trajectories.index)],
                                axis=1)
    df_traj_n_label.to_csv(str.replace(file_trajectories, ".csv", "_w_behaviors.csv"), index=False)


def check_params_for_sequence(in_size, kernel, stride):
    conv1d_out_size = (in_size - kernel) / stride + 1
    assert conv1d_out_size % 1 == 0, "Something went wront. The output of conv1d should have an integer dimension. Not float"
    return int(conv1d_out_size)


def get_n_frames(in_size, kernel, stride):
    out_size = None
    for new_in_size in range(in_size, in_size // 2, -1):
        try:
            out_size = check_params_for_sequence(new_in_size, kernel, stride)
            break
        except:
            pass
    if out_size is None:
        raise ValueError("No feasible input value. Try to change kernel (sequence length) and stride values")
    return new_in_size, out_size


def save_trajectories_from_dlc(dlc_file_path):
    """
    :param dlc_file_path: csv file: it must be the output generated by DLC with analyze_videos (and save_as_csv=True)
    :param train_size: float: 0<train_size<1. Percentage of frames to include in the train dataset
    Prepare the train and test dataset. The function extracts 100*train_size% frames from the input csv file
    and stores them inside ..\\train_dataset.csv. The remaining frames are saved inside ..\\test_dataset.csv

    PLEASE NOTE: this is the final structure of the csv file:
        HEADER: frame_id, x_nose_, y_nose_, likelihood_nose_, x_leftear_, y_leftear, likelihood_leftear_, ....

    """
    cont = -1
    cols = list()
    with open(dlc_file_path) as fp:
        for line in fp.readlines():
            cont += 1
            if cont == 3:
                break
            if cont == 0:
                cols = ["" for _ in line.split(",")]
                continue
            line = line.replace("\n", "")
            cols = [pre + "_" + post for post, pre in zip(cols, line.split(","))]
    dataset = pd.read_csv(dlc_file_path, skiprows=2, header=0, names=cols)
    # remove first column
    dataset.drop(columns=dataset.columns[0], inplace=True)
    dataset.index.name = "frame_id"
    columns = dataset.columns
    dataset.to_csv(path_or_buf="..\\data\\trajectories_without_labels.csv", header=columns, index=True)


class MarkersDataset(Dataset):
    def __init__(self, args, train: bool = True):
        self.args = None
        self.original_dataset = None
        self.dataset = None
        self.cols_likelihood = None
        self.cols_coords = None
        self.columns = None
        self.input_dataset = None
        self.input_size = None
        self.target_dataset = None
        self.n_sequences = None
        self.with_likelihood = True
        self.preprocess = getattr(args, "preprocess", "recenter")
        self.train = train
        self.check_args(args)
        self.device = args.device
        self.seq_length = args.sequence_length  # represents the number of frames in a single sequence
        self.stride = int(args.stride * self.seq_length)  # represents the stride between two consecutive sequences
        self.dataset_path = args.train_dataset_path if self.train else args.test_dataset_path
        self.transform = StandardScaler(with_std=True, with_mean=True)
        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(f"{self.dataset_path} does not exist!")
        self.initialize_dataset()
        self.create_sequences()

    def __len__(self):
        return self.input_dataset.shape[0]

    def __getitem__(self, idx):
        if self.with_likelihood:
            # remember, the shape of input_dataset is [N_SEQUENCES, SEQ_LENGTH, INPUT_SIZE+1]
            # where the +1 represents the frame_id metadata
            seq_frame_ids = self.input_dataset[idx, :, 0].clone().detach()
            sequence_w_likeli = self.input_dataset[idx, :, self.cols_coords + self.cols_likelihood].clone().detach()
            likelihoods = self.input_dataset[idx, :, self.cols_likelihood].clone().detach()
            classes = self.input_dataset[idx, :, self.col_class].clone().detach()
            # shapes:
            # seq_frame_ids: [SEQ_LENGTH, 1]
            # sequence_w_likeli: [SEQ_LENGTH, INPUT_SIZE]
            # likelihoods: [SEQ_LENGTH, n_markers]   --->   n_markers=input_size/3 (input: x,y,L for each marker)
            # classes: [SEQ_LENGTH, 1]
            classes = classes.type(torch.long).to(self.device)
            if self.train:
                return seq_frame_ids, sequence_w_likeli, likelihoods, classes
            else:
                return seq_frame_ids, sequence_w_likeli, classes
        else:
            raise NotImplementedError

    def get_all_classes(self):
        return np.array(self.dataset[:, self.col_class], dtype=np.int32)

    def _initialize_and_check_dataset_metadata(self):
        """
        the function reads the dataset from the csv file (WITH HEADER!)
        it verifies the presence of likelihoods and frame_id
        Then sets all the parameters needed for next steps
        Finally, it returns the dataset, which should have the following structure:
            frame_id,x_n,y_n,[L_n],x_le,y_le,[L_le],...
        Likelihoods are optional.
        :return: dataset as pandas dataframe, with FRAME_ID as first column and the other ones contain data
        (coords and likelihoods. The latter, only if present)
        """
        self.dataset = pd.read_csv(self.dataset_path, index_col=False)
        self.columns = self.dataset.columns
        self.cols_likelihood = [col.startswith("likelihood") for col in self.dataset.columns]
        self.with_likelihood = True if sum(self.cols_likelihood) > 0 else False
        self.cols_coords = [col.startswith("x_") or col.startswith("y_") for col in self.dataset.columns]
        if len(list(filter(lambda x: x == "class" or x == 'label', self.columns))) == 0:
            raise ValueError("Label not found inside dataset")
        self.col_class = [col == "class" or col == "label" for col in self.columns]
        self.input_size = sum(self.cols_coords) + sum(self.cols_likelihood)
        if self.columns[0] != "frame_id":
            self.dataset = pd.concat([pd.DataFrame(np.arange(0, self.dataset.shape[0]).reshape(-1, 1)),
                                      self.dataset],
                                     axis=1)
            app, app_l, app_c = ["frame_id"], [False], [False]
            app.extend(self.columns), app_l.extend(self.cols_likelihood), app_c.extend(self.cols_coords)
            self.columns = app
            self.cols_likelihood = app_l
            self.cols_coords = app_c
        # NB: if the likelihood is present, then n_likelihoods==n_markers,
        # sum(self.cols_likelihood) is necessary, since cols_likelihood is a BOOLEAN MASK
        # instead the *2 is necessary since self.cols_coords contains the total number of xs and ys which is twice
        # the number of markers
        assert sum(self.cols_likelihood) * 2 == sum(self.cols_coords), \
            f"Number of likelihoods and number of coordinates wrong! Check the header of the dataset file ({self.dataset_path})"
        self.cols_likelihood, self.cols_coords, self.col_class = \
            np.array(self.cols_likelihood), np.array(self.cols_coords), np.array(self.col_class)

    def get_center(self, points):
        """
        :param points: 2-D numpy array: shape: (n_points, 2): n_points represents the number of total markers in a
        single frame
        :return: (x_c, y_c): center coordinates
        """
        return np.sum(points, axis=0) / points.shape[0]

    def get_sequence_center(self, seq_points):
        """
        :param points: 3-D numpy array: shape: (seq_length, n_points, 2): n_points represents the number of total markers in a
        single frame. Seq_length represents the number of frames inside the sequence
        :return: (x_c, y_c): center coordinates
        """
        return self.get_center(np.reshape(seq_points, (-1, 2)))

    def recenter(self, points):
        points_ = points.reshape(-1, 2)
        points_ = points_ - self.get_center(points_)
        return points_.reshape(-1, )

    def normalize_wrt_sequence_center(self, sequence):
        sequence_center = self.get_sequence_center(sequence)
        N_SEQ, SEQ_LEN = sequence.shape
        app_sequence = sequence.reshape(-1, 2)
        app_sequence = app_sequence - sequence_center
        return app_sequence.reshape(N_SEQ, SEQ_LEN)

    def normalize_wrt_frame_center(self, dataset: np.array):
        """
        :param dataset: 2-D numpy array: shape: (N_frames, N_points*2): N.B.: the dataset does not contain the likelihood.
        N_points represents the number of markers
        :return:
        """
        return np.apply_along_axis(self.recenter, arr=dataset, axis=-1)

    def initialize_dataset(self):
        self._initialize_and_check_dataset_metadata()
        self.dataset = np.array(self.dataset)
        # copy the dataset. In this way, we can still be able to recover the original likelihoods and original coords
        self.original_dataset = self.dataset.copy()
        if self.preprocess == 'recenter':  # Apply recentering (by frame) and normalization (w.r.t the whole dataset)
            # center all the points by frame
            self.dataset[:, self.cols_coords] = self.normalize_wrt_frame_center(self.dataset[:, self.cols_coords])
            self.transform.fit(self.dataset[:, self.cols_coords])
            self.dataset[:, self.cols_coords] = self.transform.transform(self.dataset[:, self.cols_coords])
        elif self.preprocess == 'normalize':  # apply only normalization (without recentering)
            self.transform.fit(self.dataset[:, self.cols_coords])
            self.dataset[:, self.cols_coords] = self.transform.transform(self.dataset[:, self.cols_coords])
        elif self.preprocess == "recenter_by_sequence":
            # recenter all the trajectories considering the mean-center of each input sequence
            # this can't be done here. This will be done during the creation of the sequences!
            pass
        # else, do nothing (use raw trajectories)

        # remove offset from frame_id (necessary for the test dataset)
        self.dataset[:, 0] -= self.dataset[:, 0].min()

    def create_sequences(self):
        new_in_frames, self.n_sequences = get_n_frames(self.dataset.shape[0], self.seq_length, self.stride)
        # the following check is necessary in order to avoid that the few frames can be discarded
        if self.dataset.shape[0] - new_in_frames > 0:
            self.n_sequences += 1
        # each input sequence has length self.input_size + 2, because we also include the frame_id and behavior_class.
        # Obv, this value
        # must be removed inside _getitem_. It is just necessary as metadata, in order to keep track of the relation
        # between the frame_id and the input inside the sequence
        self.input_dataset = torch.empty((self.n_sequences, self.args.sequence_length, 2 + self.input_size),
                                         dtype=torch.float32)
        for sequence_idx in range(self.n_sequences - 1):
            for time_idx in range(self.seq_length):
                row_index = sequence_idx * self.stride + time_idx
                self.input_dataset[sequence_idx][time_idx] = torch.from_numpy(self.dataset[row_index])
            if self.preprocess == 'recenter_by_sequence':
                self.input_dataset[sequence_idx, :, self.cols_coords] = \
                    torch.tensor(self.normalize_wrt_sequence_center(self.input_dataset[sequence_idx, :, self.cols_coords].detach().numpy()))
        # construct the last sequence
        row_index = self.dataset.shape[0] - self.seq_length
        for time_idx in range(self.seq_length):
            self.input_dataset[self.n_sequences-1][time_idx] = torch.from_numpy(self.dataset[row_index])
            row_index += 1
        if self.preprocess == 'recenter_by_sequence':
            self.input_dataset[sequence_idx, :, self.cols_coords] = \
                torch.tensor(self.normalize_wrt_sequence_center(
                    self.input_dataset[-1, :, self.cols_coords].detach().numpy()))

    def get_behavior_proportions(self):
        all_behaviors = self.dataset[:, self.col_class].astype(np.int32)
        return np.bincount(all_behaviors.reshape(-1, )) / len(all_behaviors)

    def check_args(self, args):
        if not hasattr(args, "device"):
            raise Exception("Argument not found: device")
        if not hasattr(args, "threshold"):
            raise Exception("Argument not found: threshold")
        if not hasattr(args, "sequence_length"):
            raise Exception("Argument not found: sequence_length")
        if not hasattr(args, "train_dataset_path"):
            raise Exception("Argument not found: train_dataset_path")
        if not hasattr(args, "test_dataset_path"):
            raise Exception("Argument not found: test_dataset_path")
        if not hasattr(args, "stride"):
            raise Exception("Argument not found: stride")
        self.args = args


if __name__ == '__main__':
    args = utils.upload_args("..\\config.json")
    setattr(args, "device", "cpu")
    # setattr(args, "train_dataset_path", "..\\data\\trajectories1h6min_w_behaviors.csv")
    # split_dataset("..\\data\\video_4DLC_resnet101_For_Video_October14Oct14shuffle1_111600.csv")
    # ds = MarkersDataset(args)
    # ds.get_behavior_proportions()
    # breakpoint()
    file_behavior = "..\\data\\only_behavior_labels\\behavior_labels_22min_right.csv"
    file_trajectories = "..\\data\\trajectories_without_behavior\\trajectories_22min_right.csv"
    # save_trajectories_from_dlc(file_trajectories)
    merge_behavior_and_trajectories(file_behavior, file_trajectories, save_custom_frames=False)
