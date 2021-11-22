from torch.nn import Module
from torch.nn.modules import LSTM, Linear, Softmax, Conv1d, MaxPool1d, Sequential, ReLU, BatchNorm1d, Dropout, \
    AvgPool1d, LeakyReLU
import torch


# modify the Dataset code in order to handle input of different length (normal case: 3 seconds audio)
def check_conv1d_out_dim(in_size, kernel, padding, stride, dilation):
    conv1d_out_size = (in_size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    assert conv1d_out_size % 1 == 0, "Something went wront. The output of conv1d should have an integer dimension. Not float"
    return int(conv1d_out_size)


class PreProcessNet(Module):
    def __init__(self, args):
        super(PreProcessNet, self).__init__()
        self.input_size = args.sequence_length
        self.device = args.device
        self.args = args
        self.channels = [args.encoder_input_size,
                         args.encoder_input_size//3,
                         args.encoder_input_size//3,
                         args.encoder_input_size,
                         args.encoder_input_size,
                         args.encoder_input_size * 3]
        self.strides = [1, 1, 1, 1, 1]
        self.kernel_sizes = [3, 5, 5, 7, 11]
        self.output_sizes = []
        self.pool_kernel = 5
        self.pool_stride = 1
        self.modules = [Conv1d, ResBlock,
                        Conv1d, ResBlock,
                        Conv1d]
        self.down_sampling_modules = []
        in_size = self.input_size
        for (s, k) in zip(self.strides, self.kernel_sizes):
            in_size = check_conv1d_out_dim(in_size=in_size, kernel=k, stride=s, padding=0, dilation=1)
            # pooling operation
            in_size = check_conv1d_out_dim(in_size=in_size, kernel=self.pool_kernel, stride=self.pool_stride, padding=0, dilation=1)
            self.output_sizes.append(in_size)
        for (idx, (module, k, s)) in enumerate(zip(self.modules, self.kernel_sizes, self.strides)):
            self.down_sampling_modules.append(module(in_channels=self.channels[idx],
                                                 out_channels=self.channels[idx+1],
                                                 kernel_size=k,
                                                 dilation=1,
                                                 stride=s,
                                                 device=args.device))
            self.down_sampling_modules.append(MaxPool1d(self.pool_kernel, stride=self.pool_stride))
            self.down_sampling_modules.append(LeakyReLU(0.1))

        self.down_sampling_net = Sequential(*self.down_sampling_modules)

        self.num_channels = self.channels[-1]
        self.final_size = self.output_sizes[-1]


    def forward(self, x):
        return self.down_sampling_net(x)


class ResBlock(Module):
    def __init__(self, in_channels, out_channels, dilation, stride, kernel_size, device):
        super(ResBlock, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.device = device
        self.in_c = in_channels
        self.out_c = out_channels
        self.left = Sequential(MaxPool1d(kernel_size=self.kernel_size, dilation=self.dilation, stride=stride),
                               ReLU())
        self.right = Sequential(Conv1d(in_channels=self.in_c, out_channels=self.in_c, kernel_size=1, device=self.device),
                                Conv1d(in_channels=self.in_c, out_channels=self.out_c, kernel_size=self.kernel_size, dilation=self.dilation, stride=stride, device=self.device))

    def forward(self, x):
        x_left = self.left(x)
        x_right = self.right(x)
        return x_left + x_right
