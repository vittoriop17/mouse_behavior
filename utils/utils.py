import argparse
import json
import os
import cv2
import numpy as np
import h5py
import torch
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import itertools


def upload_args(file_path="config.json"):
    parser = argparse.ArgumentParser(description=f'Arguments from json')
    parser.add_argument("--name", required=False, type=str, help="Name of the experiment "
                                                                "(e.g.: 'evaluate preprocessing: recenter_wrt_frame' or"
                                                                " 'test sequence length: 300')")
    parser.add_argument("--n_epochs", required=False, type=int, help="Number of epochs")
    parser.add_argument("--test_only", required=False, type=bool, help="Specify if only test is required")
    parser.add_argument("--save_model", required=False, type=bool, default=False, help="Boolean flag: set it if you want to save the model")
    parser.add_argument("--input_size", required=False, type=int, help="Input size of a singular time sample")
    parser.add_argument("--hidden_size", required=False, type=int)
    parser.add_argument("--train_only", required=False, type=bool, help="If True, apply only train. The Test score is evaluated at the end of the training. Otherwise, apply train and evaluation")
    parser.add_argument("--num_layers", required=False, type=int)
    parser.add_argument("--sequence_length", required=False, type=int)
    parser.add_argument("--lr", required=False, type=float)
    parser.add_argument("--batch_size", required=False, type=int)
    parser.add_argument("--train", required=False, type=bool)
    parser.add_argument("--video", required=False, type=str, help="Video path. Video used for evaluation of results")
    parser.add_argument("--multitask", required=False, type=bool, help="Training the multitask network or the classificatio network")
    parser.add_argument("--train_dataset_path", required=False, type=str, help="Train dataset path.")
    parser.add_argument("--checkpoint_path", required=False, type=str, help="path to checkpoint")
    parser.add_argument("--load_model", required=False, type=bool, help="Specify if load an existing model or not. If 'True', checkpoint_path must be specified as well")
    parser.add_argument("--test_dataset_path", required=False, type=str, help="Test dataset path.")
    parser.add_argument("--preprocess", required=False, type=str, help="Possible options: "
                                                                       "recenter: apply centering by frame and normalization by coordinate "
                                                                       "normalize: apply only normalization by coordinate "
                                                                       "recenter_by_sequence: apply centering by frame considering the mean-center of the current sequence "
                                                                       "... otherwise, do nothing (raw trajectories)")
    parser.add_argument("--dropout", required=False, type=float, help="Network dropout.")
    parser.add_argument("--alpha", required=False, type=float, help="Parameter for weighting the 2 losses (needed for training)")
    parser.add_argument("--stride", required=False, type=float, help="Window stride (for sequence definition)."
                                                                     "To be intended in relative terms (perc %).")
    parser.add_argument("--with_conv", required=False, type=bool, help="Specify if use 1-D Convolution, in order"
                                                                      " to preprocess the input sequences")
    args = parser.parse_args()
    args = upload_args_from_json(args, file_path)
    print(args)
    return args


def upload_args_from_json(args, file_path="config.json"):
    if args is None:
        parser = argparse.ArgumentParser(description=f'Arguments from json')
        args = parser.parse_args()
    json_params = json.loads(open(file_path).read())
    for option, option_value in json_params.items():
        # do not override pre-existing arguments, if present.
        # In other terms, the arguments passed through CLI have the priority
        if hasattr(args, option) and getattr(args, option) is not None:
            continue
        if option_value == 'None':
            option_value = None
        if option_value == "True":
            option_value = True
        if option_value == "False":
            option_value = False
        setattr(args, option, option_value)
    return args


def get_frames_from_video(video_path: str):
    """
    :return: np.array containing the frames extracted from the video
    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frames = list()
    frames.append(image)
    while success:
        success, image = vidcap.read()
        frames.append(image) if success else None
    return np.array(frames)


def save_frames_from_video(video_path: str, root_path, skip_n_frames=0, idx=None):
    try:
        os.makedirs(root_path)
    except:
        pass
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    success, image = vidcap.read()
    while success and (idx is not None or len(idx) > 0):
        if skip_n_frames > count and idx is None:
            success, image = vidcap.read()
            count += 1
            continue
        elif idx is not None:
            if count in idx:
                idx.remove(count)
                frame_path = os.path.join(root_path, f"{count}.png")
                cv2.imwrite(frame_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                success, image = vidcap.read()
                count += 1
            else:
                success, image = vidcap.read()
                count += 1
        else:
            frame_path = os.path.join(root_path, f"{count}.png")
            cv2.imwrite(frame_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            success, image = vidcap.read()
            count += 1


def save_ith_frame_from_video(video_name: str, root_path, frame_seq):
    # Open the video file
    frame = get_ith_frame_from_video(video_name, frame_seq)
    frame_path = os.path.join(root_path, f"{frame_seq}.png")
    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def get_ith_frame_from_video(video_path, frame_id):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_id)
    # Read the next frame from the video
    ret, frame = cap.read()
    return frame

def save_frame(frame, path):
    cv2.imwrite(path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])

def print_dots_on_frames(frames, coords: np.array, RGB=(0, 0, 255), radius=5, thickness=5):
    """
    :param frames: np.array containing the frames extracted from the video (result returned by get_frames_from_video)
    :param coords: 2-D numpy array. For each row, it contains a set of coordinates
        e.g.:
            FIRST ROW:  x1,y1,x2,y2,x3,y3,.....
        check file 'final_pred.csv' for a clearer example
    :return: np.array with frames marked with dots
    """
    radius = radius
    thickness = thickness
    R, G, B = RGB
    n_points = int(coords.shape[1] / 2)
    if frames.shape[0] != coords.shape[0]:
        Warning("Number of frames differs from size of 'coords'")
    for frame_id, row in enumerate(coords):
        for id_point in range(n_points):
            id_x = id_point * 2
            id_y = (id_point * 2) + 1
            x = int(np.round(coords[frame_id][id_x]))
            y = int(np.round(coords[frame_id][id_y]))
            frames[frame_id] = cv2.circle(frames[frame_id], (x, y), radius, (B, G, R), thickness)
    return frames


def build_video(frames):
    """
    :param frames: np.array with marked frames
    :return:
    """
    video_name = "lstm_video.avi"
    frame = frames[0]
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, fourcc=fourcc, frameSize=(width, height), fps=30)
    for image in frames:
        video.write(image)
    cv2.destroyAllWindows()
    video.release()


def mark_video_with_dots(video_path, coords_path):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"{video_path} not found!")
    if not os.path.isfile(coords_path):
        raise FileNotFoundError(f"{coords_path} not found!")
    frames = get_frames_from_video(video_path)
    coords = np.loadtxt(coords_path, delimiter=',', skiprows=0)
    frames = print_dots_on_frames(frames, coords)
    build_video(frames)


def load_existing_model(model, optimizer, checkpoint_path):
    try:
        print(f"Trying to load existing model from checkpoint @ {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(checkpoint['model_state_dict']) if hasattr(checkpoint, "model_state_dict") else model.load_state_dict(checkpoint)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("...existing model loaded")
        max_test_f1_score = getattr(checkpoint, "max_test_f1_score", 0)
        epoch = getattr(checkpoint, "epoch", 0)
    except Exception as e:
        print("...loading failed")
        print(f"During loading the existing model, the following exception occured: \n{e}")
        print("The execution will continue anyway")
        max_test_f1_score = 0
        epoch = 0
    return max_test_f1_score, epoch


def save_confusion_matrix(y_true: np.array, y_pred: np.array, classes: list, name_method: str):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average=None)
    plot_confusion_matrix(cm, classes, title=str.upper(name_method)+f", F1-score (G, NG): {micro_f1[0]:.3f}, {micro_f1[1]:.3f}", normalize=False)
    plt.savefig(name_method+"confusion_mat.png")


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_checkpoint_data(checkpoint_path):
    checkpoint = None
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"Something happened during loading the checkpoint: {e}")
    return checkpoint


def plot_losses_by_epoch(values: np.array, name, loss_type, **kwargs):
    epoch = np.arange(1, values.shape[0])
    plt.gca().plot(epoch, values, kwargs)
    plt.xlabel("Epoch")
    plt.ylabel(loss_type)
    plt.title(name)
    plt.show()
    plt.grid(True)


def multiple_model_losses(checkpoint_paths: list, title, test_only=True, start=0):
    colors = ['r', 'b', 'y', 'c', 'g', 'm', 'orange', "lightblue"]
    epochs = 100
    plt.figure()
    for idx, checkpoint_path in enumerate(checkpoint_paths):
        checkpoint = get_checkpoint_data(checkpoint_path)
        test_micro_f1_score = checkpoint['all_test_f1_scores']
        name = checkpoint['args'].name
        name = ' '.join([token for idx, token in enumerate(name.split("_")) if idx > 0])
        color = colors[idx]
        if not test_only:
            train_micro_f1_score = checkpoint['all_train_f1_scores']
            plt.gca().plot(np.arange(1, epochs+1)[start:], train_micro_f1_score[start:], alpha=.5, color=color, label="TRAIN_SET: "+name, linestyle="--")
        plt.gca().plot(np.arange(1, epochs+1)[start:], test_micro_f1_score[start:], alpha=.5, color=color, label="EVALUATION_SET: "+name)
    plt.xlabel("Epoch")
    plt.ylabel("Micro F1 score")
    # title = "Pre-process techniques comparison. Metric: micro F1 score for train and evaluation sets"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def from_seconds_to_hh_mm_ss(seconds):
    mm_ss = divmod(seconds, 60)
    mm, ss = mm_ss[0], mm_ss[1]
    hh_mm = divmod(mm, 60)
    hh = hh_mm[0]
    mm = hh_mm[1]
    return f"{str(hh).zfill(2)}:{str(mm).zfill(2)}:{str(ss).zfill(2)}"


def get_grooming_time_from_predictions(file_path, fps=50):
    preds = np.loadtxt(file_path, delimiter='\n', dtype=np.int32)
    grooming_frames = np.argwhere(preds == 0)
    grooming_set_seconds = set()
    grooming_set_seconds.update((set([int(np.round(frame/fps)) for frame in grooming_frames])))
    grooming_list_seconds = sorted(list(grooming_set_seconds))
    grooming_intervals, flag_interval_closed, start_interval = dict(), False, grooming_list_seconds[0]
    for idx in range(1, len(grooming_list_seconds) - 1):
        previous = grooming_list_seconds[idx - 1]
        current = grooming_list_seconds[idx]
        next_g = grooming_list_seconds[idx + 1]
        if flag_interval_closed:
            start_interval = current
            flag_interval_closed = False
        elif (previous + 1) == current == (next_g - 1):
            continue
        elif current != (next_g - 1):
            grooming_intervals[start_interval] = current  # key: value -> start_interval: close_interval
            flag_interval_closed = True
        elif (previous + 1) != current:
            grooming_intervals[start_interval] = start_interval
            flag_interval_closed = True

    grooming_set_hh_mm_ss = list()
    grooming_set_hh_mm_ss.extend([[from_seconds_to_hh_mm_ss(start), from_seconds_to_hh_mm_ss(end)]
                                  for start, end in grooming_intervals.items()])
    list(map(lambda x: print(f"Grooming detected from {x[0]} to {x[1]} (hh:mm:ss format)"), grooming_set_hh_mm_ss))

    return grooming_set_seconds, grooming_intervals


def create_bash_files_for_predictions(root_path, video_path):
    predictions_path = "..\\data\\frame_predictions.txt"
    _, pred_grooming_intervals = get_grooming_time_from_predictions(predictions_path)
    # ground_truth_path = ""
    # _, true_grooming_intervals = get_grooming_time_from_predictions(ground_truth_path)
    for start, end in pred_grooming_intervals.items():
        file_name = os.path.join(root_path, "predictions", f"time_interval__{start}_{end}.sh")
        with open(file_name, "w") as fp:
            lines = "#!/bin/bash\n"\
                    f"vlc --start-time={start} --stop-time={end} '{video_path}'"
            fp.writelines(lines)


if __name__ == '__main__':
    file_path = "..\\data\\DLC_resnet101_groomOct29shuffle1_117500-snapshot-117500.h5"
    video_path = "..\\..\\ALL_VIDEOSSSS\\final\\front_57min.MP4"
    root_path = "..\\data\\frames_video_22min"
    # frames = get_frames_from_video(video_path=video_path)
    # save_frames(root_path=, frames=frames)
    # save_frames_from_video(video_path, root_path)
    # save_ith_frames_from_video(video_name=video_path, root_path=root_path, frame_seq=10721 * 2)
    # check_paths = ["..\\data\\CHECKPOINTS\\checkpoint_Evaluation_preprocess_recenter_by_sequence.pt",
    #                "..\\data\\CHECKPOINTS\\checkpoint_Evaluation_preprocess_recenter_by_frame.pt",
    #                "..\\data\\CHECKPOINTS\\checkpoint_Evaluation_preprocess_normalization.pt"]
    # multiple_model_losses(check_paths, title="Pre-process techniques comparison. Metric: micro F1 score for evaluation set")
    # plt.tight_layout()
    # plt.savefig("Preprocess_losses.png")
    # seq_lengths = [50, 100, 150, 200, 250, 300, 350, 400]
    # check_paths = ["..\\data\\CHECKPOINTS\\seq_length\\checkpoint_Evaluation_seq_length_"+str(seq_length)+".pt"
    #                for seq_length in seq_lengths]
    #
    # multiple_model_losses(check_paths, title="Sequence length comparison. Metric: micro F1 score for evaluation set. \n From 50th to 100th epoch", start=49)
    # plt.tight_layout()
    # plt.savefig("seq_length_evaluation_losses.png")
    create_bash_files_for_predictions(root_path="..\\visual_analysis", video_path="..\\S1170001.MP4")
