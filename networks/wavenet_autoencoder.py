import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

class wavenet_autoencoder(nn.Module):
    def __init__(self,
                 filter_width,
                 quantization_channel,
                 dilations,

                 en_residual_channel,
                 en_dilation_channel,

                 en_bottleneck_width,
                 en_pool_kernel_size,

                 use_bias):

        super(wavenet_autoencoder, self).__init__()

        self.filter_width = filter_width
        self.quantization_channel = quantization_channel
        self.dilations = dilations

        self.en_residual_channel = en_residual_channel
        self.en_dilation_channel = en_dilation_channel

        self.en_bottleneck_width = en_bottleneck_width
        self.en_pool_kernel_size = en_pool_kernel_size

        self.use_bias = use_bias

        self.receptive_field = self._calc_receptive_field()

        self._init_encoding()
        self._init_causal_layer()

    def _init_causal_layer(self):

        self.en_causal_layer = nn.Conv1d(self.quantization_channel, self.en_residual_channel, self.filter_width,
                                         bias=self.use_bias)

        self.bottleneck_layer = nn.Conv1d(self.en_residual_channel, self.en_bottleneck_width, 1, bias=self.use_bias)

    def _calc_receptive_field(self):

        return (self.filter_width - 1) * (sum(self.dilations) + 1) + 1

    def _init_encoding(self):

        self.en_dilation_layer_stack = nn.ModuleList()
        self.en_dense_layer_stack = nn.ModuleList()

        for dilation in self.dilations:
            self.en_dilation_layer_stack.append(nn.Conv1d(

                self.en_residual_channel,
                self.en_dilation_channel,
                self.filter_width,
                dilation=dilation,
                bias=self.use_bias
            ))

            self.en_dense_layer_stack.append(nn.Conv1d(

                self.en_dilation_channel,
                self.en_residual_channel,
                1,
                bias=self.use_bias
            ))

    def _encode(self, sample):
        sample = self.en_causal_layer(sample)

        for i, (dilation_layer, dense_layer) in enumerate(zip(self.en_dilation_layer_stack, self.en_dense_layer_stack)):
            current = sample

            sample = F.relu(sample)
            sample = dilation_layer(sample)
            sample = F.relu(sample)
            sample = dense_layer(sample)
            _, _, current_length = sample.size()
            current_in_sliced = current[:, :, -current_length:]
            sample = sample + current_in_sliced

        sample = self.bottleneck_layer(sample)
        sample = F.relu(sample)

        pool1d = nn.AdaptiveAvgPool1d(output_size=self.en_pool_kernel_size)
        sample = pool1d(sample)
        return sample


    def _encoding_conv(self, encoding, channel_in, channel_out, kernel_size):

        conv1d = nn.Conv1d(channel_in, channel_out, kernel_size)
        en = conv1d(encoding)

        return en

    def forward(self, wave_sample):
        # batch_size, original_channels, seq_len = wave_sample.size()
        # output_width = seq_len - self.receptive_field + 1
        # print(self.receptive_field)
        encoding = self._encode(wave_sample)

        return encoding