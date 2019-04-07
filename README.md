# End-to-End-VAD
an Audio-Visual Voice Activity Detection using Deep Learning

### 1. Introduction

This is my pytorch implementation of the Audio-Visual voice activity detector presented in "An End-to-End Multimodal Voice Activity
Detection Using WaveNet Encoder and Residual Networks" (https://ieeexplore.ieee.org/document/8649655).

### 2. Requirements

the requirements are listed in the requirements.txt file and can be installed via
```shell
$pip install -r requirements.txt
```
you will also need to install FFmpeg for video loading.

Note: the code was tested on a windows machine with python 3.6, some minor changes may be required for other OS.

### 3. Training

#### 3.1 Data preparation 

The dataset used in this repo can be downloaded from https://www.dropbox.com/s/70jyy3d3hdumbpx/End-to-End-VAD_data.rar?dl=0.
the 11 speakers should be divided into train\val\test splits and placed under `./data/split` directory. At the initial run of train.py, these files will be processed and divided into smaller files of length "time_depth" [frames].

#### 3.2 Training

Using `train.py`. The parameters are as following:

```shell
$ python train.py -h
Usage: train.py [-h] 
                [--num_epochs]
                [--batch_size]
                [--test_batch_size]
                [--time_depth]
                [--workers]
                [--lr]
                [--weight_decay]
                [--momentum]
                [--save_freq]
                [--print_freq]
                [--seed]
                [--lstm_layers]
                [--lstm_hidden_size]
                [--use_mcb]
                [--mcb_output_size]
                [--debug]
                [--freeze_layers]
                [--arch]
                [--pre_train]               
                
```

Check the `train.py` for more details. Majority of these parameters already come with reasonable default values. We usually used a large batch_size for the audio network (e.g., 64 or 128) and a smaller batch_size for the video network and the multimodal network (e.g., 16).
other hyperparameters (e.g., batch size, weight decay, learning rate decay, etc.) should also be adjusted according to the network you wish to train (Audio\Video\AV)

Note: it is recommended to perform a two-stage training as described in the paper - 

(1) Train each single-modal network separately

(2) initialize the multimodal network with the pre-trained weights from (1) and then train.

### 4. Evaluation

Using `eval.py` to evaluate the validation or test dataset. The parameters are as following:

```shell
$ python eval.py -h
Usage: eval.py [-h] 
               [--batch_size]    
               [--time_depth]
               [--workers]
               [--print_freq]               
               [--lstm_layers]
               [--lstm_hidden_size]
               [--use_mcb]
               [--mcb_output_size]
               [--debug]       
               [--arch]
               [--pre_train]  
```

Check the `eval.py` for more details. This script will print out the average loss and accuracy on the evaluation set.

-------

### Credits:

this implementation borrows heavily from:
- the wavenet implementation of https://github.com/yangorwell/wavenet_autoencoder 
- the compact bilinear pooling implementation of https://github.com/gdlg/pytorch_compact_bilinear_pooling 





 
