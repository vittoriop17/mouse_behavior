import os
import matplotlib.pyplot as plt
import numpy as np
from pandas.tests.io.parser import test_skiprows


def plot_points_and_time(point_coords, marker_names):
    """
    :param point_coords: 2-D numpy array: shape: (N_frames, N_points*2)
    The plots are very dense, thus consider to plot a subset of trajectories related to a portion of the original video
    e.g.: reduce the size of N_frames to 500 (500 frames represents a portion of 10 seconds of the original video)
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    points = point_coords.reshape(point_coords.shape[0], point_coords.shape[1]//2, 2)
    n_markers = point_coords.shape[1]//2
    markers = ['.', 'x', 'o', '+', '*']
    assert len(markers) == n_markers and len(marker_names) == n_markers
    xs = range(point_coords.shape[0])
    for idx, m in enumerate(markers):
        ys = points[:, idx, 0]
        zs = points[:, idx, 1]
        ax.scatter(xs, ys, zs, marker=m, label=marker_names[idx], alpha=.3)

    ax.set_xlabel('time')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.legend()
    plt.show()

def plot_trajectories(point_coords, coord_names=None):
    """
    :param point_coords: 2-D numpy array: shape: (N_frames, N_points*2)
    The plots are very dense, thus consider to plot a subset of trajectories related to a portion of the original video
    e.g.: reduce the size of N_frames to 500 (500 frames represents a portion of 10 seconds of the original video)
    """
    fig, axs = plt.subplots(point_coords.shape[1]//2, 2)
    time = range(point_coords.shape[0])
    for idx, ax in enumerate(axs):
        x = point_coords[:, idx*2]
        y = point_coords[:, idx*2+1]
        ax[0].plot(time, x)
        ax[1].plot(time, y)
        if coord_names is not None and len(coord_names)==point_coords.shape[1]:
            ax[0].set_title(coord_names[idx*2])
            ax[1].set_title(coord_names[idx*2+1])
    plt.tight_layout()

def get_center(points):
    """
    :param points: 2-D numpy array: shape: (n_points, 2): n_points represents the number of total markers in a
    single frame
    :return: (x_c, y_c): center coordinates
    """
    return np.sum(points, axis=0) / points.shape[0]

def recenter(points):
    points_ = points.reshape(-1, 2)
    points_ = points_ - get_center(points_)
    return points_.reshape(-1, )

def normalize_wrt_frame_center(dataset: np.array):
    """
    :param dataset: 2-D numpy array: shape: (N_frames, N_points*2): N.B.: the dataset does not contain the likelihood.
    N_points represents the number of markers
    :return:
    """
    return np.apply_along_axis(recenter, arr=dataset, axis=-1)

if __name__=='__main__':
    dataset = np.loadtxt("..\\train_dataset.csv", skiprows=1, delimiter=',')
    likelihood_cols = [2,5,8,11,14]
    coord_cols = list(set(range(0, 15)) - set(likelihood_cols))
    coord_names = ['x_Nose','y_Nose',
                   'x_Left_Ear','y_Left_Ear',
                   'x_Right_Ear','y_Right_Ear',
                   'x_Left_Paw','y_Left_Paw',
                   'x_Right_Paw','y_Right_Paw']
    marker_names = ['nose', 'left_ear', 'right_ear', 'left_paw', 'right_paw']
    plot_trajectories(normalize_wrt_frame_center(dataset[:, coord_cols]), coord_names=coord_names)
    # plot_points_and_time(dataset[500:1000, coord_cols], )
    plot_points_and_time(normalize_wrt_frame_center(dataset[500:1000, coord_cols]), marker_names)