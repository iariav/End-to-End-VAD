# End-to-End-VAD
an Audio-Visual Voice Activity Detection using Deep Learning

### 1. Introduction

This is my pytorch implementation of the Audio-Visual voice activity detector presented in "An End-to-End Multimodal Voice Activity
Detection Using WaveNet Encoder and Residual Networks" (https://ieeexplore.ieee.org/document/8649655).

### 2. Requirements

- pytorch 0.4 (newer versions may work too)
- python 3
- tensorboardX (for logging)
- librosa (for audio loading\saving)
- scipy
- pickle

### 3. Training

#### 3.1 Data preparation 

the dataset used in this repo can be downloaded from ###.
the 11 speakers should be devided to train\val\test splits and placed under `./data/split` directory. 
at the initial run of train.py, these files will be processed and devided into smaller files of length "time_depth" [frames].

#### 3.2 Training

Using `train.py`. The parameters are as following:

```shell
$ python train.py -h
usage: train.py [-h] 
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

Check the `train.py` for more details. Majority of these parameters already come with resonable default values. 

Note: it is recomended to perform a two-stage training as described in the paper - 

(1) Train each single-modal network seperatly

(2) initialize the multimodal network with the pre-trained weights from (1) and then train.

### 4. Evaluation

Using `eval.py` to evaluate the validation or test dataset. The parameters are as following:

```shell
$ python eval.py -h
usage: eval.py [-h] 
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

Check the `eval.py` for more details. 
this script will print out the average loss and accuracy on the evaluation set.

-------

### Credits:

this implementations borrows heavily from the wavenet implementation of ### and the compact bilinear pooling of ### 





 
