import os
from torch.utils.data import Dataset
import librosa
import scipy.io.wavfile as scp
import torch
import math
import pickle
from utils.NoiseInjection import *
import skvideo
skvideo.setFFmpegPath('C:/Deep/Tools/ffmpeg/bin/')
import skvideo.io
import torchvision.transforms
import re

#Audio ground truth
AudioGt = {}
AudioGt["Speaker1"] =  np.array(
           [[1.7,2.7], [5,6.9],[10.3,14.1],[18.3,20.0],[23.5,25.2],[30.4,32.6],[38.3,41],[46.2,50.1],[55.2,57.3],[63.7,65.2],[68.3,69.2],[72.6,74.7],[78.2,81.4],[86,87.5],[91.1,93],[96.8,99],[102.6,103.6],[106.5,108.3],[112.4,115.9],[118.9,119.9]],
           dtype=np.float32)
AudioGt["Speaker2"] =  np.array(
           [[1.4,2.4],[2.9,5.5],[9.5,11.4],[11.7,13.4],[18.2,24.6],[28,28.8],[32,35.7],[39.7,44.9],[49.3,50.3],[55.3,57.1],[60.6,65.2],[69.3,72.8],[77.6,84],[87.5,88.8],[94,99.4],[104.1,107.7],[112.8,116]],
           dtype=np.float32)
AudioGt["Speaker3"] =  np.array(
           [[1.7,7.2],[9.9,19.7],[25.1,29],[32.3,39.5],[44.1,48.2],[51.1,56.5],[61.5,70.3],[73.2,75.1],[75.7,77.1],[82,83],[83.3,85],[88.5,97.8],[103.4,111.4],[115.1,119.9]],
           dtype=np.float32)
AudioGt["Speaker4"] =  np.array(
           [[2.2,5.3],[8.3,10.8],[21.6,22.5],[23.2,26.5],[30,33.4],[36.8,37.4],[37.7,40.6],[43.5,45],[45.4,48.7],[51.5,52.8],[53.2,56.8],[60.7,63],[66.7,68.7],[71.3,74.7],[78.3,80.2],[83.6,86.6],[89.8,93.3],[96,98.3],[101.8,103.9],[108,110.2],[111,115.8],[119.7,119.9]],
           dtype=np.float32)
AudioGt["Speaker5"] =  np.array(
           [[2.7,5.3],[9.2,11.6],[17.2,22.1],[27.3,28.2],[29.1,32],[32.2,32.6],[32.8,33.3],[38,40.7],[40.9,41.7],[46.9,47.4],[47.9,49.6],[49.9,51.2],[56.6,57.9],[58.3,59.9],[60,61.7],[66.4,67.6],[71.5,72.1],[72.5,73.7],[74.1,76.2],[81.5,83.9],[84.9,85.6],[86.7,87.4],[93,96.7],[101.7,103.4],[108.4,111.5],[116.5,118.6],[119.1,119.9]],
           dtype=np.float32)
AudioGt["Speaker6"] =  np.array(
           [[1.3,4.4],[6.1,8.2],[10,14.7],[16.8,19],[21.2,23.2],[24.8,28.5],[30.7,33.2],[35.3,38.4],[40.5,42.3],[44.7,47.9],[50,53.4],[55.4,57.5],[59.9,62.3],[64.7,67.2],[70,71.8],[74.5,76.9],[79.8,83.4],[85.4,90.7],[93.7,95.3],[98.1,101.5],[104.1,107.2],[109.6,112.5],[115.3,118.2]],
           dtype=np.float32)
AudioGt["Speaker7"] =  np.array(
           [[2,4.4],[7.8,10.3],[13.6,15],[18.2,19.7],[23.1,26],[29.4,30.5],[34.4,37.6],[40.9,42.2],[46,48.2],[51.6,54],[57.6,59.1],[62.4,65.7],[69.4,72.3],[76,77.4],[80.9,84.2],[88,91.8],[95.7,97.6],[101.7,103.6],[107.4,109.8],[113.5,114.8],[119,119.9]],
           dtype=np.float32)
AudioGt["Speaker8"] =  np.array(
           [[2,5.3],[7.7,10],[12.1,14.4],[16.6,18.9],[21.1,23.4],[26.4,29.2],[29.5,30.9],[33.5,34.8],[37.4,39.1],[41.9,43.7],[46.3,49.4],[51.7,53.6],[56.4,59.1],[62.2,64.5],[67.5,70.5],[72.9,75.3],[78.5,80.7],[81.7,82],[82.5,85.6],[88.2,90.2],[92.7,94.7],[97.4,100.6],[104.7,107.6],[110.3,112.9],[116.3,117.8]],
           dtype=np.float32)
AudioGt["Speaker9"] =  np.array(
           [[1.1,4.6],[6.4,10.2],[12.5,15],[17.2,20.3],[22.9,26.4],[28.6,31.9],[34.5,37],[39.6,43.5],[45.9,48.1],[50.7,53.1],[55.1,57.4],[59.5,62.1],[64.3,67.8],[70.1,73.4],[75.6,77.9],[80.6,83.2],[86,90],[92.8,96.4],[99.5,102.7],[103.5,105.5],[108.1,110.8],[113.3,117],[117.6,118.7]],
           dtype=np.float32)
AudioGt["Speaker10"] =  np.array(
           [[1.3,2.6],[3.2,4.1],[7.9,9.8],[10.1,10.5],[10.9,11.4],[11.9,12.3],[17,19.6],[23.5,25.6],[29.5,32],[37.7,38.3],[38.7,40.9],[45.3,47.1],[51,53.2],[57.6,60.3],[64.1,68.6],[73.5,76.9],[80.6,82.8],[87.6,89.5],[90.4,91.1],[91.7,92.6],[98.9,102.4],[103,104.6],[109.1,112.7],[117.1,119.9]],
           dtype=np.float32)
AudioGt["Speaker11"] =  np.array(
           [[1,2.6],[4.6,9.4],[11.7,14.3],[17,20.6],[23.3,26],[28,30],[33.2,35.4],[37.5,40],[42.8,46.5],[48.8,50.2],[53.1,57.3],[59.3,61.4],[64.1,67.5],[70.2,72.7],[75.1,79.3],[81.5,84.4],[85.8,89.9],[92.1,94.7],[96.7,99.6],[103.2,105.2],[107.7,109.1],[111,114.7],[117.8,119.8]],
           dtype=np.float32)

#Video ground truth
VideoGt = {}

VideoGt["Speaker1"] = np.array(
   [[43,64],[128,176],[260,353],[456,499],[592,631],[762,824],[961,1034],[1160,1263],[1388,1441],[1605,1645],
    [1717,1741],[1825,1879],[1967,2051],[2163,2200],[2294,2345],[2437,2494],[2580,2603],[2682,2722],[2826,2912],
    [2991,3027]],
           dtype=np.int32)
VideoGt["Speaker2"] = np.array(
    [[26, 137], [238, 335], [445, 614], [694, 715], [796, 894], [996, 1127], [1235, 1260], [1371, 1436], [1513, 1514],
    [1515, 1634], [1731, 1827], [1941, 2111], [2190, 2230], [2353, 2500], [2608, 2707], [2824, 2913]],
           dtype=np.int32)
VideoGt["Speaker3"] = np.array(
    [[47, 171], [239, 480], [626, 722], [797, 986], [1099, 1203], [1283, 1410], [1541, 1765], [1843, 1933],
     [2060, 2131], [2226, 2455], [2603, 2798], [2890, 3019]],
           dtype=np.int32)
VideoGt["Speaker4"] = np.array(
    [[53, 126], [208, 259], [356, 441], [541, 556], [586, 658], [759, 832], [925, 1013], [1103, 1217], [1299, 1419],
     [1511, 1514], [1526, 1577], [1680, 1718], [1799, 1867], [1969, 2008], [2105, 2162], [2244, 2338], [2413, 2462],
     [2562, 2604], [2717, 2757], [2791, 2905], [3010, 3028]],
           dtype=np.int32)
VideoGt["Speaker5"] = np.array(
    [[49, 133], [224, 296], [412, 554], [682, 838], [945, 1054], [1174, 1289], [1414, 1514], [1515, 1553], [1655, 1702],
     [1792, 1916], [2044, 2122], [2138, 2155], [2185, 2203], [2341, 2435], [2537, 2605], [2717, 2806], [2921, 3028]],
           dtype=np.int32)
VideoGt["Speaker6"] = np.array(
    [[35, 105], [151, 196], [253, 356], [416, 468], [532, 585], [621, 717], [745, 830], [888, 956], [1013, 1057],
     [1124, 1191], [1258, 1345], [1393, 1447], [1503, 1560], [1630, 1679], [1759, 1800], [1886, 1926], [2010, 2093],
     [2151, 2280], [2357, 2392], [2466, 2538], [2613, 2694], [2759, 2826], [2902, 2964]],
           dtype=np.int32)
VideoGt["Speaker7"] = np.array(
    [[42, 112], [181, 261], [334, 378], [454, 493], [572, 655], [739, 767], [861, 942], [1027, 1065], [1154, 1212],
     [1291, 1361], [1442, 1487], [1564, 1651], [1742, 1819], [1907, 1951], [2031, 2122], [2209, 2310], [2403, 2457],
     [2549, 2606], [2696, 2762], [2857, 2888], [2987, 3028]],
           dtype=np.int32)
VideoGt["Speaker8"] = np.array(
    [[35, 131], [185, 253], [295, 362], [409, 482], [530, 591], [661, 784], [846, 872], [949, 984], [1054, 1112],
     [1175, 1254], [1320, 1354], [1420, 1486], [1563, 1621], [1702, 1778], [1845, 1907], [1987, 2160], [2218, 2274],
     [2339, 2388], [2451, 2544], [2637, 2710], [2774, 2847], [2934, 2975]],
           dtype=np.int32)
VideoGt["Speaker9"] = np.array(
    [[21, 114], [158, 254], [312, 379], [441, 515], [571, 665], [712, 811], [873, 936], [999, 1101], [1166, 1219],
     [1277, 1344], [1385, 1451], [1499, 1572], [1626, 1719], [1773, 1858], [1906, 1970], [2042, 2106], [2178, 2281],
     [2351, 2441], [2526, 2600], [2625, 2673], [2739, 2808], [2864, 3012]],
           dtype=np.int32)
VideoGt["Speaker10"] = np.array(
    [[33, 43], [74, 104], [204, 282], [300, 312], [432, 494], [596, 647], [744, 809], [954, 1035], [1161, 1189],
     [1294, 1343], [1458, 1527], [1622, 1740], [1862, 1944], [2039, 2094], [2220, 2265], [2296, 2304], [2328, 2348],
     [2507, 2652], [2751, 2863], [2953, 3028]],
           dtype=np.int32)
VideoGt["Speaker11"] = np.array(
    [[26, 58], [109, 235], [281, 257], [420, 522], [581, 653], [698, 750], [829, 900], [943, 1004], [1078, 1166],
     [1230, 1271], [1332, 1455], [1497, 1549], [1618, 1705], [1776, 1836], [1897, 2010], [2062, 2138], [2171, 2273],
     [2338, 2399], [2455, 2527], [2603, 2655], [2727, 2764], [2806, 2907], [2978, 3028]],
           dtype=np.int32)

def get_audio_ground_truth(speaker,frame_rate,sample_length):

    gt = np.zeros(int(frame_rate*sample_length),dtype=np.uint8)
    GT_temp = AudioGt[speaker]
    GT_temp = np.round(GT_temp*frame_rate).astype(np.int32)
    for i in range (GT_temp.shape[0]):
        gt[GT_temp[i][0]:GT_temp[i][1]] = 1

    return gt

def get_video_ground_truth(speaker,video_length):

    gt = np.zeros(int(video_length),dtype=np.uint8)
    GT_temp = VideoGt[speaker]
    for i in range (GT_temp.shape[0]):
        gt[GT_temp[i][0]-1:GT_temp[i][1]-1] = 1  # added -1 because of difference in indexing between numpy and matlab

    return gt

class AudioDataset(Dataset):

    """Dataset Class for Loading audio files"""

    def __init__(self, DataDir, timeDepth, is_train):
        """
        Args:
        DataDir (string): Directory with all the data.
        timeDepth: Number of frames to be loaded in a sample
        is_train(bool): Is train or test dataset
        """
        import json
        with open('./params/audio_dataset_params.json', 'r') as f:
            params = json.load(f)

        self.DataDir = DataDir
        self.is_train = is_train
        self.timeDepth = timeDepth

        self.Audio_frame_Length = params["Audio_frame_Length"]
        self.sampling_rate = params["sampling_rate"]
        self.GlobalFrameRate = params["GlobalFrameRate"]
        self.audio_duration = params["audio_duration"]
        self.noise_prob = params["noise_prob"]
        self.trans_prob = params["trans_prob"]

        self.audio_len_in_frames = math.floor(self.audio_duration * self.sampling_rate / self.Audio_frame_Length)
        self.normalize = True

        # init noise injector
        self.noiseInjector = NoiseInjection(Noise_path='data/noises/',
                                             transient_path='data/transients/',
                                             sample_length=self.timeDepth * self.Audio_frame_Length,
                                             sample_rate=self.sampling_rate,
                                             SNR=(0, 20))

        raw_audio = [x for x in os.listdir(self.DataDir) if ".wav" in x]

        labels_stack = []
        labels_pkl_file = os.path.join(self.DataDir, "audio_labels.pkl")

        if not os.path.exists(labels_pkl_file):

            if not os.path.exists(self.DataDir + 'samples'):
                os.makedirs(self.DataDir + 'samples')

            piece_num = 0
            for i ,f in enumerate(raw_audio):
                print("processing audio file %s" % f)
                full_path = os.path.join(self.DataDir,f)
                audio, Fs = librosa.load(full_path,sr=self.sampling_rate,duration=self.audio_duration)
                if self.normalize:
                    audio = librosa.util.normalize(audio)
                SpeakerNum = re.findall('\d+', f)
                labels = get_audio_ground_truth("Speaker%s" % SpeakerNum[0], self.GlobalFrameRate, self.audio_duration)

                for end_frame in range(self.timeDepth,self.audio_len_in_frames):
                    audio_seq = audio[(end_frame - self.timeDepth) * self.Audio_frame_Length:end_frame * self.Audio_frame_Length]

                    if not self.is_train: # at test add noises and trans only once for consistency
                        add_noise = np.random.binomial(1, self.noise_prob)
                        add_trans = np.random.binomial(1, self.trans_prob)

                        if add_noise:
                            audio_seq = self.noiseInjector.inject_noise_sample(audio_seq)

                        if add_trans:
                            audio_seq = self.noiseInjector.inject_trans_sample(audio_seq)

                    audio_path = self.DataDir + 'samples/sample_' + str(piece_num) + '.wav'
                    librosa.output.write_wav(audio_path, audio_seq, sr=self.sampling_rate)
                    piece_num += 1

                    label = torch.LongTensor(1)
                    label[0] = int(labels[end_frame])
                    labels_stack.append(label)

            labels_dump = open(labels_pkl_file, 'wb')
            pickle.dump(labels_stack, labels_dump)

        labels = open(labels_pkl_file, 'rb')
        labels = pickle.load(labels)

        self.audio_labels = np.ravel(np.array(labels))

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):

        audio_path = self.DataDir + 'samples/sample_' + str(idx) + '.wav'
        Fs, audio = scp.read(audio_path)

        # at train randomly add noises and trans
        if self.noiseInjector and self.is_train:
            add_noise = np.random.binomial(1, self.noise_prob)
            add_trans = np.random.binomial(1, self.trans_prob)

            if add_noise:
                audio = self.noiseInjector.inject_noise_sample(audio)

            if add_trans:
                audio = self.noiseInjector.inject_trans_sample(audio)

        sample = torch.from_numpy(audio).type(torch.FloatTensor)
        return sample, self.audio_labels[idx]


class VideoDataset(Dataset):

    """Dataset Class for Loading video files"""

    def __init__(self, DataDir, timeDepth, is_train):
        """
        Args:
        DataDir (string): Directory with all the data.
        timeDepth: Number of frames to be loaded in a sample
        is_train(bool): Is train or test dataset
        """

        self.DataDir = DataDir
        self.is_train = is_train
        self.timeDepth = timeDepth
        self.video_duration_in_frames = 3059

        raw_video = [x for x in os.listdir(self.DataDir) if ".avi" in x]

        labels_stack = []
        videos_stack = []
        labels_pkl_file = os.path.join(self.DataDir, "video_labels.pkl")

        if not os.path.exists(labels_pkl_file):

            normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            t = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                normalize,
            ])

            for i ,f in enumerate(raw_video):
                print("processing video file %s" % f)
                full_path = os.path.join(self.DataDir,f)
                video = skvideo.io.vread(full_path)
                T, M, N, C = video.shape
                video = video[:self.video_duration_in_frames, :, :, :]
                video_tensor = torch.FloatTensor(self.video_duration_in_frames,C,224,224)
                SpeakerNum = re.findall('\d+', f)
                labels = get_video_ground_truth("Speaker%s" % SpeakerNum[0], self.video_duration_in_frames)

                for end_frame in range(self.timeDepth,self.video_duration_in_frames):

                    label = torch.LongTensor(1)
                    label[0] = int(labels[end_frame])
                    labels_stack.append(label)

                for frame in range(self.video_duration_in_frames):
                    #possibly augment data here
                    video_tensor[frame,:,:,:] = t(video[frame,:,:,:])

                video_path = self.DataDir + 'VideoTensor_' + str(i) + '.pt'
                torch.save(video_tensor, video_path)

            labels_dump = open(labels_pkl_file, 'wb')
            pickle.dump(labels_stack, labels_dump)

        labels = open(labels_pkl_file, 'rb')
        labels = pickle.load(labels)

        for i, f in enumerate(raw_video):
            video_path = self.DataDir + 'VideoTensor_' + str(i) + '.pt'
            videos_stack.append(torch.load(video_path))


        self.video_labels = np.ravel(np.array(labels))
        self.videos = videos_stack

    def __len__(self):
        return len(self.video_labels)

    def __getitem__(self, idx):

        vid_idx = int(idx / (self.video_duration_in_frames - self.timeDepth))
        frame_idx = idx % (self.video_duration_in_frames - self.timeDepth)

        sample = self.videos[vid_idx][frame_idx:frame_idx+self.timeDepth,:,:,:]
        return sample, self.video_labels[idx]


class AVDataset(Dataset):

    """Dataset Class for Loading audio & video files"""

    def __init__(self, DataDir, timeDepth,is_train):
        """
        Args:
        DataDir (string): Directory with all the data.
        timeDepth: Number of frames to be loaded in a sample
        is_train(bool): Is train or test dataset
        """

        self.audio_dataset = AudioDataset(DataDir=DataDir, timeDepth=timeDepth, is_train=is_train)
        self.video_dataset = VideoDataset(DataDir=DataDir, timeDepth=timeDepth, is_train=is_train)

    def __len__(self):
        return len(self.audio_dataset)

    def __getitem__(self, idx):

        audio_sample, audio_label = self.audio_dataset.__getitem__(idx)
        video_sample, video_label = self.video_dataset.__getitem__(idx)
        av_label = audio_label | video_label
        return audio_sample, video_sample, av_label