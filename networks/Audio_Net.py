import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.wavenet_autoencoder import wavenet_autoencoder
from torch.autograd import Variable
from utils.utils import weights_init_normal

# AUDIO only network
class DeepVAD_audio(nn.Module):
    def __init__(self, args):
        self.lstm_layers = args.lstm_layers
        self.lstm_hidden_size = args.lstm_hidden_size
        self.test_batch_size = args.test_batch_size
        self.batch_size = args.batch_size

        super(DeepVAD_audio, self).__init__()

        import json
        with open('./params/model_params.json', 'r') as f:
            params = json.load(f)

        self.wavenet_en = wavenet_autoencoder(
            **params)  # filter_width, dilations, dilation_channels, residual_channels, skip_channels, quantization_channels, use_bias
        self.lstm_audio = nn.LSTM(input_size=params["en_bottleneck_width"],
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_layers,
                            bidirectional=False)

        self.vad_audio = nn.Linear(self.lstm_hidden_size, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.bn = torch.nn.BatchNorm1d(params["en_bottleneck_width"], eps=1e-05, momentum=0.1, affine=True)

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.named_parameters():
            weights_init_normal(m, mean=mean, std=std)

    def forward(self, x,h):
        x = self.wavenet_en(x) # output shape - Batch X Features X seq len
        x = self.bn(x)
        # Reshape to (seq_len, batch, input_size)
        x = x.permute(2, 0, 1)
        x = self.dropout(x)
        out, h = self.lstm_audio(x, h) # output shape - seq len X Batch X lstm size
        out = out[-1] # select last time step. many -> one
        out = self.dropout(out)
        out = F.sigmoid(self.vad_audio(out))
        return out

    def init_hidden(self,is_train):
        if is_train:
            return (Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_hidden_size)).cuda(),
                      Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(self.lstm_layers, self.test_batch_size, self.lstm_hidden_size)).cuda(),
                    Variable(torch.zeros(self.lstm_layers, self.test_batch_size, self.lstm_hidden_size)).cuda())