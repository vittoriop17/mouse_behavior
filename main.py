import torch
from utils import utils
from model import new_lstm
from utils import dataset, train


def main(params):
    if getattr(params, "test_only", False):
        assert hasattr(params, "checkpoint_path"), "Checkpoint_path argument not found! " \
                                                   "Checkpoint_path must be provided for testing the model"
        train.test_model(params.checkpoint_path, params)
    else:
        train.train_wrapper(params)


if __name__ == '__main__':
    args = utils.upload_args()
    main(args)
    # checkpoint_path = "data\\checkpoint.pt"
    # train.denoise_trajectories_from_checkpoint(checkpoint_path, args)
    # setattr(args, "device", "cpu")
    # setattr(args, "name", "prova")
    # checkpoint_path = "data\\CHECKPOINTS\\seq_length\\checkpoint_Evaluation_seq_length_200.pt"
