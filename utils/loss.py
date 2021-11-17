import torch
from torch import nn


def weighted_mse(predicted: torch.tensor, target: torch.tensor, weights: torch.tensor):
    """
    :param predicted: tensor. shape (N, SEQ_LENGTH, N_MARKERS * 2)
    :param target: tensor. shape (N, SEQ_LENGTH, N_MARKERS * 2)
    :param weights: tensor. shape (N,SEQ_LENGTH, N_MARKERS)
    :return: tensor
    """
    N, SEQ_LENGTH, INPUT_SIZE = predicted.shape
    predicted = predicted.view(N*SEQ_LENGTH, -1)
    target = target.view(N*SEQ_LENGTH, -1)
    square_diffs = torch.pow(target - predicted, 2)
    square_diffs = square_diffs.view(N*SEQ_LENGTH*(INPUT_SIZE//2), 2)
    square_diffs = torch.sum(square_diffs, dim=1)
    weights = weights.view(N*SEQ_LENGTH*(INPUT_SIZE//2), )
    return torch.sum(weights * square_diffs) / ( N * SEQ_LENGTH * INPUT_SIZE )


if __name__=='__main__':
    pred = torch.rand((2, 5, 6))
    true = pred + 10
    weights = torch.zeros((2, 5, 3))
    weights[0,:,:] =1
    weights[1,:,:] = 2
    weights[:,:,0] += 10
    weights[:,:,1] += 100
    weights[:,:,2] += 1000
    weighted_mse(pred, true, weights)