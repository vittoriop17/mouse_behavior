import argparse
import json
import os
import cv2
import numpy as np
import h5py
import torch


def upload_args(file_path="config.json"):
    parser = argparse.ArgumentParser(description=f'Arguments from json')
    parser.add_argument("--n_epochs", required=False, type=int, help="Number of epochs")
    parser.add_argument("--input_size", required=False, type=int, help="Input size of a singular time sample")
    parser.add_argument("--hidden_size", required=False, type=int)
    parser.add_argument("--num_layers", required=False, type=int)
    parser.add_argument("--sequence_length", required=False, type=int)
    parser.add_argument("--lr", required=False, type=float)
    parser.add_argument("--batch_size", required=False, type=int)
    parser.add_argument("--train", required=False, type=bool)
    parser.add_argument("--video", required=False, type=str, help="Video path. Video used for evaluation of results")
    parser.add_argument("--train_dataset_path", required=False, type=str, help="Train dataset path.")
    parser.add_argument("--test_dataset_path", required=False, type=str, help="Test dataset path.")
    parser.add_argument("--dropout", required=False, type=float, help="Network dropout.")
    parser.add_argument("--stride", required=False, type=int, help="Window stride (for sequence definition)."
                                                                   "To be intended in absolute terms.")
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


def save_ith_frames_from_video(video_name: str, root_path, frame_seq):
    # Open the video file
    cap = cv2.VideoCapture(video_name)
    cap.set(1, frame_seq)
    # Read the next frame from the video
    ret, frame = cap.read()
    frame_path = os.path.join(root_path, f"{frame_seq}.png")
    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])


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
        model.load_state_dict(checkpoint['model_state_dict'])
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


if __name__ == '__main__':
    file_path = "..\\data\\DLC_resnet101_groomOct29shuffle1_117500-snapshot-117500.h5"
    video_path = "..\\..\\ALL_VIDEOSSSS\\final\\front_57min.MP4"
    root_path = "..\\data\\frames_video_22min"
    # frames = get_frames_from_video(video_path=video_path)
    # save_frames(root_path=, frames=frames)
    # save_frames_from_video(video_path, root_path)
    save_ith_frames_from_video(video_name=video_path, root_path=root_path, frame_seq=10721 * 2)
