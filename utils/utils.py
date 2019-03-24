import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def to_var(x):
    if (torch.cuda.is_available()):
            x = x.cuda()
    return Variable(x)


def print_network(net):
    
    num_params = 0
    
    for param in net.parameters():
        num_params += param.numel()
    
    print(net)
    print('Total number of parameters: %d' % num_params)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, best_accuracy,filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        checkpoint_name = "'results\model_best_%.3f.pth.tar'" % (best_accuracy)
        shutil.copyfile(filename, 'results\model_best.pth.tar')

def weights_init_normal(m, mean=0.0, std=0.005):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('ConvTranspose2d') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('lstm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()

def import_dataset(args):
    if args.arch =='Video':
        from datasets import VideoDataset
        dataset = VideoDataset
    elif args.arch =='Audio':
        from datasets import AudioDataset
        dataset = AudioDataset
    else: # Audio-Visual
        from datasets import AVDataset
        dataset = AVDataset

    return dataset

def import_network(args):
    if args.arch =='Video':
        print('loaded Video model')
        from networks.Video_Net import DeepVAD_video
        net = DeepVAD_video(args)
    elif args.arch =='Audio':
        print('loaded Audio model')
        from networks.Audio_Net import DeepVAD_audio
        net = DeepVAD_audio(args)
    else:
        print('loaded AV model')
        from networks.AV_Net import DeepVAD_AV
        net = DeepVAD_AV(args)

    net.weight_init()
    if (torch.cuda.is_available()):
            net = net.cuda()

    if args.debug:
        print_network(net)

    return net
