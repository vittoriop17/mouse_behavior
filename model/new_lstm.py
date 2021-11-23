from torch.nn import modules
from torch import nn
import torch
import os
from utils.utils import upload_args
from model.conv_lstm_net import PreProcessNet


HIDDEN_SIZE = 100
SEQ_LENGTH = 100  # 50fps --> 100 frames mean a sequence of length 2 seconds
DROPOUT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_LAYERS = 1  # num_layers for LSTM encoder and decoder


class Attention(nn.Module):
    """ Applies attention_behavior mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention_behavior score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         # >>> attention_behavior = Attention(256)
         # >>> query = torch.randn(5, 1, 256)
         # >>> context = torch.randn(5, 5, 256)
         # >>> output, weights = attention_behavior(query, context)
         # >>> output.size()
         # torch.Size([5, 1, 256])
         # >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention_behavior type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention_behavior mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention_behavior weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights


class EncoderLSTM(nn.Module):
    def __init__(self, args, merge_out=False, bidirectional=True, override_input_size=None):
        super(EncoderLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.D = 2 if self.bidirectional else 1
        self.merge_out = merge_out
        self.dropout = getattr(args, "dropout", DROPOUT)
        self.device = getattr(args, "device", DEVICE)
        self.n_markers = getattr(args, "n_markers", KeyError("n_markers argument not found!"))
        self.wl = 3 if getattr(args, "with_likelihood", True) else 2
        self.input_size = self.wl * self.n_markers if override_input_size is None else override_input_size
        self.hidden_size = getattr(args, "hidden_size", HIDDEN_SIZE)
        self.num_layers = getattr(args, "num_layers", NUM_LAYERS)
        self.encoder = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               bidirectional=self.bidirectional, batch_first=True,
                               device=self.device, dropout=self.dropout)
        self.output_size = 2 * self.hidden_size

    def forward(self, x):
        """
        :param x: tensor. Shape: [batch_size, seq_length, input_size]
        :return:
        """
        # D=1 if bidirectional=False, D=2 otherwise
        # N=batch_size
        # [N, seq_length, D*hidden_size], ([D*num_layers, N, hidden_size], ...)
        x_enc, (h_n, _) = self.encoder(x)
        if self.merge_out and self.bidirectional:
            # assuming the LSTM net is bidirectional
            x_enc = (x_enc[:, :, :self.hidden_size] + x_enc[:, :, self.hidden_size:]) / 2
        return x_enc, h_n


class DecoderLSTM(nn.Module):
    # N.B.: this module can be used fot both tasks: behavior classification and trajectory de-noising
    #
    def __init__(self, args):
        super(DecoderLSTM, self).__init__()
        self.dropout = getattr(args, "dropout", DROPOUT)
        self.device = getattr(args, "device", DEVICE)
        self.hidden_size = getattr(args, "hidden_size", HIDDEN_SIZE)
        self.num_layers = getattr(args, "num_layers", NUM_LAYERS)
        raise NotImplementedError("Implement LSTM Decoder")


class Net(nn.Module):
    def __init__(self, args, bidirectional=True, override_input_size=None):
        super(Net, self).__init__()
        self.bidirectional = bidirectional
        self.D = 2 if self.bidirectional else 1
        self.sequence_length = getattr(args, "sequence_length", SEQ_LENGTH)
        self.hidden_size = getattr(args, "hidden_size", HIDDEN_SIZE)
        self.device = getattr(args, "device", DEVICE)
        self.n_behaviors = getattr(args, "n_behaviors", KeyError("n_behaviors argument not found!"))
        self.n_markers = getattr(args, "n_markers", KeyError("n_markers argument not found!"))
        self.wl = 3 if getattr(args, "with_likelihood", True) else 2
        self.encoder = EncoderLSTM(args, override_input_size=override_input_size)
        self.dec_input_size = self.encoder.output_size
        self.attention_behavior = Attention(self.D*self.hidden_size)
        self.attention_denoising = Attention(self.D*self.hidden_size)
        self.decoder_behavior = nn.LSTMCell(input_size=self.dec_input_size, hidden_size=self.D*self.hidden_size,
                                            device=self.device)
        self.fc_behavior = nn.Sequential(nn.Linear(in_features=self.D*self.hidden_size, out_features=50,
                                                   device=self.device),
                                         nn.ReLU(),
                                         nn.Linear(in_features=50, out_features=self.n_behaviors,
                                                   device=self.device),
                                         nn.LogSoftmax(dim=-1))
        self.decoder_denoising = nn.LSTMCell(input_size=self.dec_input_size, hidden_size=self.D*self.hidden_size,
                                             device=self.device)
        self.fc_denoising = nn.Sequential(nn.Linear(in_features=self.D*self.hidden_size, out_features=100,
                                                    device=self.device),
                                          nn.ReLU(),
                                          nn.Linear(in_features=100, out_features=2*self.n_markers,
                                                    device=self.device))

    def forward(self, x):
        batch_size = x.shape[0]
        out_enc, h_n = self.encoder(x)
        assert out_enc.shape == (batch_size, self.sequence_length, self.dec_input_size)
        query = torch.zeros((batch_size, self.D*self.hidden_size), device=self.device, dtype=torch.float32)
        c_i = torch.zeros((batch_size, self.D*self.hidden_size), device=self.device, dtype=torch.float32)
        decoded_behavior_sequence = []
        decoded_trajectory_sequence = None
        trajectory_sequence = None
        if self.encoder.training:
            query_denoising = torch.zeros((batch_size, self.D*self.hidden_size), device=self.device, dtype=torch.float32)
            c_i_denoising = torch.zeros((batch_size, self.D*self.hidden_size), device=self.device, dtype=torch.float32)
            decoded_trajectory_sequence = []

        for i in range(self.sequence_length):
            # out_attn shape: [N, 1, self.dec_input_size]
            query = torch.unsqueeze(query, dim=1)
            out_attn, attn_w = self.attention_behavior(context=out_enc, query=query)
            out_attn = torch.squeeze(out_attn, dim=1)
            query = torch.squeeze(query, dim=1)
            query, c_i = self.decoder_behavior(out_attn, (query, c_i))
            decoded_behavior_sequence.append(query)
            if self.encoder.training:  # Predict also the de-noised trajectory
                query_denoising = torch.unsqueeze(query_denoising, dim=1)
                out_attn_denoising, attn_w_denoising = self.attention_denoising(context=out_enc, query=query_denoising)
                out_attn_denoising = torch.squeeze(out_attn_denoising, dim=1)
                query_denoising = torch.squeeze(query_denoising, dim=1)
                query_denoising, c_i_denoising = self.decoder_denoising(out_attn_denoising,
                                                                        (query_denoising, c_i_denoising))
                decoded_trajectory_sequence.append(query_denoising)

        decoded_behavior_sequence = torch.cat([torch.unsqueeze(h, dim=1) for h in decoded_behavior_sequence], dim=1)
        behavior_sequence = self.fc_behavior(decoded_behavior_sequence)
        if self.encoder.training:
            decoded_trajectory_sequence = torch.cat([torch.unsqueeze(h, dim=1) for h in decoded_trajectory_sequence], dim=1)
            trajectory_sequence = self.fc_denoising(decoded_trajectory_sequence)
        # [N, seq_length, n_behaviors], [N, seq_length, n_markers*self.wl] or None, if not training
        return behavior_sequence, trajectory_sequence


class Net_w_conv(nn.Module):
    def __init__(self, args):
        super(Net_w_conv, self).__init__()
        self.conv_net = PreProcessNet(args)
        # the temporal dimension had length 'sequence_length' (usually set to 120).
        # now, after the conv_net preprocessing, the size became self.conv_net.final_size
        # this new value should be lower than the previous one, thanks to the convolutions
        self.num_sequences = self.conv_net.final_size
        # the original input size (for a single frame) was n_markers * 3 (x,y,likelihood)
        # now, after the conv_net preprocessing, the size became self.conv_net.num_channels
        self.input_size = self.conv_net.num_channels
        setattr(args, "sequence_length", self.num_sequences)
        self.lstm_net = Net(args, override_input_size=self.input_size)

    def forward(self, x):
        # x must have the following shape:
        # [Batch_size, INPUT_CHANNELS, SEQ_LENGTH]
        # where INPUT_CHANNELS represents the size of a temporal input (number of coordinates + number of likelihoods)
        x = x.transpose(1, 2)
        x = self.conv_net(x)
        x = x.transpose(1, 2)
        return self.lstm_net(x)


if __name__=='__main__':
    args = upload_args("..\\config.json")
    setattr(args, "device", "cpu")
    net = Net_w_conv(args)
    # x_seq shape: [Batch_size, Seq_length, input_size]...
    # Input size refers to the length of the input in a specified instant time.
    # input_size = 24, since we have 8 triplets (x,y,L) for each marker
    # seq_length=120: this is just a design choice. We should keep this in a way that it should be equal to the
    # MAXIMUM period of each distinct marker trajectory (evaluate them through auto-correlation)
    x_seq = torch.rand((1, 120, 24))
    decoded_behavior_seq, decoded_trajectory_seq = net(x_seq)

