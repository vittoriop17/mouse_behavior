from torch.nn import modules
from torch import nn
import torch
import os


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.input_size = params.input_size  # Number of expected features in the input x
        self.hidden_size = params.hidden_size  # number of features in the hidden state h
        self.num_layers = params.num_layers if hasattr(params, "num_layers") else 2  # set the number of recurrent layers
        self.proj_size = params.proj_size if hasattr(params, "proj_size") else self.hidden_size / 2
        self.device = params.device
        self.lstm_encoder = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, device=self.device,
                                    batch_first=True, num_layers=self.num_layers)

    def forward(self, x: torch.Tensor):
        # x.shape should be (Batch_size, sequence length, input_size)
        # N.B.: sequence length represents the length of the sequence in terms of temporal samples
        # e.g.: sequence length=3 if we feed the network with 3 samples taken in three consecutive time instants
        # print(f"Input shape: {x.shape}")
        x_enc, (_, _) = self.lstm_encoder(x)
        # x_enc.shape should be: (N, L, H_out): (batch_size, final_input_size, proj_size)
        # print(f"Input encoded shape: {x_enc.shape}")
        return x_enc

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.output_size = params.input_size  # Number of expected features in the input x --> it corresponds to the output size
        self.num_layers = params.num_layers if hasattr(params, "num_layers") else 2  # set the number of recurrent layers
        self.input_size = params.hidden_size
        self.device = params.device
        self.lstm_decoder = nn.LSTM(input_size=self.input_size, hidden_size=self.output_size, num_layers=self.num_layers,
                                    device=self.device, batch_first=True)

    def forward(self, x):
        x_dec, (_, _) = self.lstm_decoder(x)
        # print(f"Decoder: x decoded shape: {x_dec.shape}")
        return x_dec


class LSTMAutoEncoder(nn.Module):
    def __init__(self, params):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        latent_code = self.encoder(x)
        x_dec = self.decoder(latent_code)
        # print(f"Decoded input shape {x_dec.shape}. It should be the same of x.shape")
        return x_dec