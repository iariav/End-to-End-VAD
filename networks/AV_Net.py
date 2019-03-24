import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from networks.wavenet_autoencoder import wavenet_autoencoder
from networks.compact_bilinear_pooling import CountSketch, CompactBilinearPooling
from torch.autograd import Variable
from utils.utils import weights_init_normal

# Audio_Visual network
class DeepVAD_AV(nn.Module):

    def __init__(self, args):
        super(DeepVAD_AV, self).__init__()

        self.lstm_layers = args.lstm_layers
        self.lstm_hidden_size = args.lstm_hidden_size
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.dropout = nn.Dropout(p=0.05)
        self.use_mcb = args.use_mcb

        # video related init

        resnet = models.resnet18(pretrained=True)  # set self.num_video_ftrs = 512

        self.num_video_ftrs = 512
        self.features = nn.Sequential(
            *list(resnet.children())[:-1]# drop the last FC layer
        )

        #audio related init

        import json
        with open('./params/model_params.json', 'r') as f:
            params = json.load(f)

        self.wavenet_en = wavenet_autoencoder(
            **params)  # filter_width, dilations, dilation_channels, residual_channels, skip_channels, quantization_channels, use_bias

        self.num_audio_ftrs = params["en_bottleneck_width"]
        self.bn = torch.nn.BatchNorm1d(params["en_bottleneck_width"], eps=1e-05, momentum=0.1, affine=True)

        # general init

        if self.use_mcb:
            self.lstm_input_size = self.mcb_output_size = args.mcb_output_size
            self.mcb = CompactBilinearPooling(self.num_audio_ftrs, self.num_video_ftrs, self.mcb_output_size).cuda()
            self.mcb_bn = torch.nn.BatchNorm1d(self.mcb_output_size, eps=1e-05, momentum=0.1, affine=True)
        else:
            self.lstm_input_size = self.num_audio_ftrs + self.num_video_ftrs

        self.lstm_merged = nn.LSTM(input_size=self.lstm_input_size ,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_layers,
                            bidirectional=False)

        self.vad_merged = nn.Linear(self.lstm_hidden_size, 2)


    def weight_init(self, mean=0.0, std=0.02):
        for m in self.named_parameters():
            weights_init_normal(m, mean=mean, std=std)

    def init_hidden(self,is_train):
        if is_train:
            return (Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_hidden_size)).cuda(),
                    Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(self.lstm_layers, self.test_batch_size, self.lstm_hidden_size)).cuda(),
                    Variable(torch.zeros(self.lstm_layers, self.test_batch_size, self.lstm_hidden_size)).cuda())

    def forward(self, audio, video, h):

        # Video branch
        batch,frames,channels,height,width = video.squeeze().size()
        # Reshape to (batch * seq_len, channels, height, width)
        video = video.view(batch*frames,channels,height,width)
        video = self.features(video).squeeze() # output shape - Batch X Features X seq len
        video = self.dropout(video)
        # Reshape to (batch , seq_len, Features)
        video = video.view(batch , frames, -1)
        # Reshape to (seq_len, batch, Features)
        video = video.permute(1, 0, 2)

        # Audio branch
        audio = self.wavenet_en(audio) # output shape - Batch X Features X seq len
        audio = self.bn(audio)
        audio = self.dropout(audio)  # output shape - Batch X Features X seq len
        # Reshape to (seq_len, batch, input_size)
        audio = audio.permute(2, 0, 1)

        # Merging branches
        if self.use_mcb:
            y = self.mcb(audio, video)
            # signed square root
            y =  torch.mul(torch.sign(x), torch.sqrt(torch.abs(x) + 1e-12)) # or y = torch.sqrt(F.relu(x)) - torch.sqrt(F.relu(-x))
            # L2 normalization
            y = y / torch.norm(y, p=2).detach()

            y = y.permute(1, 2, 0).contiguous()
            y = self.mcb_bn(y)
            y = y.permute(2, 0, 1).contiguous()

        else:
            y = torch.cat([audio,video],dim=2)

        # Merged branch
        y = self.dropout(y)
        out, h = self.lstm_merged(y, h)  # output shape - seq len X Batch X lstm size
        out = self.dropout(out[-1]) # select last time step. many -> one
        out = F.sigmoid(self.vad_merged(out))
        return out

