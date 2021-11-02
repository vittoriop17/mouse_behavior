import argparse
import json
import os
import cv2
import numpy as np


def upload_args():
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
    args = parser.parse_args()
    args = upload_args_from_json(args)
    print(args)
    return args


def upload_args_from_json(args, file_path=os.path.join("config.json")):
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

def print_dots_on_frames(frames, coords: np.array):
    """
    :param frames: np.array containing the frames extracted from the video (result returned by get_frames_from_video)
    :param coords: 2-D numpy array. For each row, it contains a set of coordinates
        e.g.:
            FIRST ROW:  x1,y1,x2,y2,x3,y3,.....
        check file 'final_pred.csv' for a clearer example
    :return: np.array with frames marked with dots
    """
    radius = 5
    thickness = 5
    B, G, R = 0, 0, 255
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