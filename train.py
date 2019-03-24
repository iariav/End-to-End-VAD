from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
from utils import utils as utils
from torch.utils.data import DataLoader
import time
import torch.nn.utils as torchutils
from torch.autograd import Variable
from utils.logger import Logger
import os

if __name__ == '__main__':

    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=16, help='test batch size')
    parser.add_argument('--time_depth', type=int, default=15, help='number of time frames in each video\audio sample')
    parser.add_argument('--workers', type=int, default=0, help='num workers for data loading')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay factor')
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum factor')
    parser.add_argument('--save_freq', type=int, default=1, help='freq of saving the model')
    parser.add_argument('--print_freq', type=int, default=50, help='freq of printing stats')
    parser.add_argument('--seed', type=int, default=44974274, help='random seed')
    parser.add_argument('--lstm_layers', type=int, default=2, help='number of lstm layers in the model')
    parser.add_argument('--lstm_hidden_size', type=int, default=1024, help='number of neurons in each lstm layer in the model')
    parser.add_argument('--use_mcb', action='store_true', help='wether to use MCB or concat')
    parser.add_argument('--mcb_output_size', type=int, default=1024, help='the size of the MCB outputl')
    parser.add_argument('--debug', action='store_true', help='print debug outputs')
    parser.add_argument('--freeze_layers', action='store_true', help='wether to freeze the first layers of the model')
    parser.add_argument('--arch', type=str, default='AV', choices=['Audio', 'Video', 'AV'], help='which modality to train - Video\Audio\Multimodal')
    parser.add_argument('--pre_train', type=str, default='', help='path to a pre-trained network')

    args = parser.parse_args()
    print(args, end='\n\n')

    torch.manual_seed(args.seed)

    # set the logger
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logger = Logger('logs')

    # create a saved models folder
    save_dir = os.path.join('saved_models', args.arch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # create train + val datasets
    dataset = utils.import_dataset(args)

    train_dataset = dataset(DataDir='data/train/', timeDepth = args.time_depth, is_train=True)
    val_dataset = dataset(DataDir='data/test/', timeDepth = args.time_depth, is_train=False)

    print('{} samples found, {} train samples and {} test samples.'.format(len(val_dataset)+len(train_dataset),
                                                                           len(train_dataset),
                                                                           len(val_dataset)))

    # create the data loaders

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False,
        drop_last=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False,
        drop_last=True)

    # import network
    net = utils.import_network(args)

    # create optimizer and loss (optionaly assign each class with different weight
    weight = torch.FloatTensor(2)
    weight[0] = 1  # class 0 - non-speech
    weight[1] = 1  # class 1 - speech
    criterion = nn.CrossEntropyLoss(weight=weight).cuda()

    optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 5, 8], gamma=0.1)

    # init from a saved checkpoint

    if args.pre_train is not '':
        model_name = os.path.join('pre_trained', args.arch, args.pre_train)

        if os.path.isfile(model_name):
            print("=> loading checkpoint '{}'".format(model_name))
            checkpoint = torch.load(model_name)
            pretrained = checkpoint['state_dict']
            net.load_state_dict(pretrained,strict=False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_name, checkpoint['epoch']))
        else:
            print('Couldn\'t load model from {}'.format(model_name))
    else:
        print('Training the model from scratch.')

    # freeze layeres

    def freeze_layer(layer):
        for param in layer.parameters():
            param.requires_grad = False

    if args.arch == 'Video' and args.freeze_layers == True:
        freeze_layer(net.features)

    if args.arch == 'Audio' and args.freeze_layers == True:
        freeze_layer(net.wavenet_en)
        freeze_layer(net.bn)

    if args.arch == 'AV' and args.freeze_layers == True:
        freeze_layer(net.features)
        freeze_layer(net.wavenet_en)
        freeze_layer(net.bn)

    # def test method
    def test():

        test_acc  = utils.AverageMeter()
        test_loss = utils.AverageMeter()

        net.eval()
        print('Test started.')

        for i, data in enumerate(val_loader):

            states_test = net.init_hidden(is_train=False)

            if args.arch == 'Video' or args.arch == 'Audio':  # single modality

                input, target = data  # input is of shape torch.Size([batch, channels, frames, width, height])
                input_var = Variable(input.unsqueeze(1)).cuda()
                target_var = Variable(target.squeeze()).cuda()

                output = net(input_var, states_test)

            else:  # multiple modalities

                audio, video, target = data
                audio_var = Variable(audio.unsqueeze(1)).cuda()
                video_var = Variable(video.unsqueeze(1)).cuda()
                target_var = Variable(target.squeeze()).cuda()

                output = net(audio_var, video_var, states_test)

            loss = criterion(output.squeeze(), target_var)

            # measure accuracy and record loss
            _, predicted = torch.max(output.data, 1)
            accuracy = (predicted == target.squeeze().cuda()).sum().type(torch.FloatTensor)
            accuracy.mul_((100.0 / args.test_batch_size))
            test_loss.update(loss.item(), args.test_batch_size)
            test_acc.update(accuracy.item(), args.test_batch_size)

            if i>0 and i % args.print_freq == 0:
                print('Test: [{0}][{1}/{2}] - loss = {3} , acc = {4}'.format(epoch, i, len(val_loader), test_loss.avg, test_acc.avg))

        net.train()
        print('Test finished.')
        return test_acc.avg, test_loss.avg


    ### main training loop ###

    best_accuracy = 0
    best_epoch = 0
    step = 0

    for epoch in range(0,args.num_epochs):

        train_loss = utils.AverageMeter()
        train_acc = utils.AverageMeter()
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()

        # learning rate decay
        scheduler.step()

        end = time.time()

        # train for one epoch
        for i, data in enumerate(train_loader):

            states = net.init_hidden(is_train=True)

            if args.arch == 'Video' or args.arch == 'Audio': # single modality

                input, target = data
                input_var = Variable(input.unsqueeze(1)).cuda()
                target_var = Variable(target.squeeze()).cuda()

                # measure data loading time
                data_time.update(time.time() - end)

                output = net(input_var, states)

            else: # multiple modalities

                audio, video, target = data
                audio_var = Variable(audio.unsqueeze(1)).cuda()
                video_var = Variable(video.unsqueeze(1)).cuda()
                target_var = Variable(target.squeeze()).cuda()

                # measure data loading time
                data_time.update(time.time() - end)

                output = net(audio_var,video_var, states)

            loss = criterion(output.squeeze(), target_var)

            # compute gradient and do SGD step
            net.zero_grad()
            loss.backward()
            torchutils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # measure accuracy and record loss
            _, predicted = torch.max(output.data, 1)
            accuracy = (predicted == target.squeeze().cuda()).sum().type(torch.FloatTensor)
            accuracy.mul_((100.0 / args.batch_size))
            train_loss.update(loss.item(), args.batch_size)
            train_acc.update(accuracy.item(), args.batch_size)

            # tensorboard logging
            logger.scalar_summary('train loss', loss.item(), step + 1)
            logger.scalar_summary('train accuracy', accuracy.item(), step + 1)
            step+=1

            if i > 0 and i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}] , \t'
                      'LR {3} , \t'
                      'Time {batch_time.avg:.3f} , \t'
                      'Data {data_time.avg:.3f} , \t'
                      'Loss {loss.avg:.4f} , \t'
                      'Acc {top1.avg:.3f}'.format(
                    epoch, i, len(train_loader), optimizer.param_groups[0]['lr'], batch_time=batch_time,
                    data_time=data_time, loss=train_loss, top1=train_acc))

        # evaluate on validation set
        accuracy, loss = test()

        # logger
        logger.scalar_summary('Test Accuracy', accuracy, epoch)
        logger.scalar_summary('Test Loss ', loss, epoch)
        logger.scalar_summary('LR ', optimizer.param_groups[0]['lr'], epoch)

        # remember best prec@1 and save checkpoint
        is_best = False
        if accuracy > best_accuracy:
            is_best = True
            best_epoch = epoch

        best_accuracy = max(accuracy, best_accuracy)

        print('Average accuracy on validation set is: {}%'.format(accuracy))
        print('Best accuracy so far is: {}% , at epoch #{}'.format(best_accuracy,best_epoch))

        if epoch % args.save_freq == 0:
            checkpoint_name = "%s\\acc_%.3f_epoch_%03d_arch_%s.pkl" % (save_dir, accuracy, epoch, args.arch)
            utils.save_checkpoint(state={
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'accuracy': accuracy,
                'best_accuracy': best_accuracy,
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best,best_accuracy=best_accuracy,filename=checkpoint_name)
            model_name = "%s\\acc_%.3f_epoch_%03d_arch_%s_model.pkl" % (save_dir, accuracy, epoch, args.arch)

            torch.save(net, model_name)