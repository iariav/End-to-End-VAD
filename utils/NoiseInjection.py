import os
import librosa
import numpy as np

class NoiseInjection(object):
    def __init__(self,
                 Noise_path=None,
                 transient_path=None,
                 sample_length=1, # seconds
                 sample_rate=8000,
                 SNR=(0, 20)):
        """
        Adds noise to an input signal with specific SNR. Higher the noise level, the more noise added.
        Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py
        """

        self.sample_rate = sample_rate
        self.SNR = SNR
        self.sample_length = sample_length

        # load noises audio files

        if not os.path.exists(Noise_path):
            print("Directory doesn't exist: {}".format(Noise_path))
            raise IOError

        raw_audio = [x for x in os.listdir(Noise_path) if ".wav" in x]

        noises = []
        noises_lengths = []

        for i, f in enumerate(raw_audio):
            print("processing noise file %s" % f)
            full_path = os.path.join(Noise_path, f)
            audio, Fs = librosa.load(full_path, sr=self.sample_rate)
            noises.append(audio)
            noises_lengths.append(librosa.get_duration(filename=full_path, sr=self.sample_rate) * self.sample_rate)

        # load transients audio files

        if not os.path.exists(transient_path):
            print("Directory doesn't exist: {}".format(transient_path))
            raise IOError

        raw_audio = [x for x in os.listdir(transient_path) if ".wav" in x]

        trans = []
        trans_lengths = []

        for i, f in enumerate(raw_audio):
            print("processing trans file %s" % f)
            full_path = os.path.join(transient_path, f)
            audio, Fs = librosa.load(full_path, sr=self.sample_rate)
            trans.append(audio)
            trans_lengths.append(librosa.get_duration(filename=full_path, sr=self.sample_rate) * self.sample_rate)

        self.noises = noises
        self.trans = trans
        self.noise_lengths = noises_lengths
        self.trans_lengths = trans_lengths


    def inject_noise_sample(self, sample,noise_idx=None,SNR=None):

        if noise_idx is None:
            noise_idx = np.random.randint(0,len(self.noises))
        if SNR is None:
            SNR = np.random.randint(*self.SNR)

        noise_len = self.noise_lengths[noise_idx]

        noise_start = int(np.floor(np.random.rand() * (noise_len - self.sample_length)))
        noise_end = noise_start + self.sample_length
        noise = self.noises[noise_idx][noise_start:noise_end]

        assert len(sample) == len(noise)

        # adjust noise energy according to SNR

        sample_std = np.std(sample)
        noise_std = np.std(noise)
        new_noise_std = sample_std/(10 ** (SNR / 20))
        noise/=noise_std
        noise*=new_noise_std

        return sample+noise

    def inject_trans_sample(self, sample,trans_idx=None):

        if trans_idx is None:
            trans_idx = np.random.randint(0, len(self.trans))

        trans_len = self.trans_lengths[trans_idx]

        trans_start = int(np.floor(np.random.rand() * (trans_len - self.sample_length)))
        trans_end = trans_start + self.sample_length
        trans = self.trans[trans_idx][trans_start:trans_end]

        assert len(sample) == len(trans)

        return sample+(trans*2)
