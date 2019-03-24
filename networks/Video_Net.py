import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import weights_init_normal

# VIDEO only network
class DeepVAD_video(nn.Module):

    def __init__(self, args):
        super(DeepVAD_video, self).__init__()

        resnet = models.resnet18(pretrained=True) # set num_ftrs = 512
        #resnet = models.resnet34(pretrained=True) # set num_ftrs = 512

        num_ftrs = 512

        self.lstm_input_size = num_ftrs
        self.lstm_layers = args.lstm_layers
        self.lstm_hidden_size = args.lstm_hidden_size
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size

        self.features = nn.Sequential(
            *list(resnet.children())[:-1]# drop the last FC layer
        )

        self.lstm_video = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_layers,
                            bidirectional=False)

        self.vad_video = nn.Linear(self.lstm_hidden_size, 2)
        self.dropout = nn.Dropout(p=0.5)

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.named_parameters():
            weights_init_normal(m, mean=mean, std=std)

    def forward(self, x, h):
        batch,frames,channels,height,width = x.squeeze().size()
        # Reshape to (batch * seq_len, channels, height, width)
        x = x.view(batch*frames,channels,height,width)
        x = self.features(x).squeeze() # output shape - Batch X Features X seq len
        x = self.dropout(x)
        # Reshape to (batch , seq_len, Features)
        x = x.view(batch , frames, -1)
        # Reshape to (seq_len, batch, Features)
        x = x.permute(1, 0, 2)
        out, _ = self.lstm_video(x, h)  # output shape - seq len X Batch X lstm size
        out = self.dropout(out[-1])  # select last time step. many -> one
        out = F.sigmoid(self.vad_video(out))
        return out

    def init_hidden(self,is_train):
        if is_train:
            return (Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_hidden_size)).cuda(),
                      Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(self.lstm_layers, self.test_batch_size, self.lstm_hidden_size)).cuda(),
                    Variable(torch.zeros(self.lstm_layers, self.test_batch_size, self.lstm_hidden_size)).cuda())