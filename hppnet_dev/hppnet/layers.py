import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import numpy as np
from torchaudio.transforms import Spectrogram
import nnAudio.Spectrogram

class WaveformToLogSpecgram(nn.Module):
    def __init__(self, sample_rate, n_fft, fmin, bins_per_octave, freq_bins, hop_length, logspecgram_type, device):
        super().__init__()

        e = freq_bins/bins_per_octave
        fmax = fmin * (2 ** e)

        self.logspecgram_type = logspecgram_type
        self.n_fft = n_fft
        self.hamming_window = torch.hann_window(self.n_fft).to(device)
        self.hamming_window = torch.unsqueeze(self.hamming_window, 0)

        # torch.hann_window()

        fre_resolution = sample_rate/n_fft

        idxs = torch.arange(0, freq_bins, device=device)

        log_idxs = fmin * (2**(idxs/bins_per_octave)) / fre_resolution

        # 线性插值： y_k = y_i * (k-i) + y_{i+1} * ((i+1)-k)
        self.log_idxs_floor = torch.floor(log_idxs).long()
        self.log_idxs_floor_w = (log_idxs - self.log_idxs_floor).reshape([1, freq_bins, 1])
        self.log_idxs_ceiling = torch.ceil(log_idxs).long()
        self.log_idxs_ceiling_w = (self.log_idxs_ceiling - log_idxs).reshape([1, freq_bins, 1])

        self.waveform_to_specgram = torchaudio.transforms.Spectrogram(n_fft, hop_length=hop_length).to(device)

        assert(bins_per_octave % 12 == 0)
        bins_per_semitone = bins_per_octave // 12
        
        if(logspecgram_type == 'logspecgram'):
            self.spec_layer = nnAudio.Spectrogram.STFT(
                n_fft=n_fft, 
                freq_bins=freq_bins, 
                hop_length=hop_length, 
                sr=sample_rate,
                freq_scale='log',
                fmin=fmin,
                fmax=fmax,
                output_format='Magnitude'
            ).to(device)
        elif(logspecgram_type == 'cqt'):
            self.spec_layer = nnAudio.Spectrogram.CQT(
                sr=sample_rate, hop_length=hop_length,
                fmin=fmin,
                n_bins=freq_bins,
                bins_per_octave=bins_per_octave,
                output_format='Magnitude',
            ).to(device)

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def forward(self, waveforms):
        # inputs: [b x wave_len]
        # outputs: [b x T x n_bins]


        if(self.logspecgram_type == 'logharmgram'):
            # return self.waveform_to_harmgram(waveforms)[:, :, :, 0]

            # [b x (n_fft/2 + 1) x T]
            # specgram = self.waveform_to_specgram(waveforms)

            waveforms = waveforms * self.hamming_window
            specgram =  torch.fft.fft(waveforms)
            specgram = torch.abs(specgram[:, :self.n_fft//2 + 1])
            specgram = specgram * specgram
            specgram = torch.unsqueeze(specgram, dim=2)

            # => [b x freq_bins x T]
            specgram = specgram[:, self.log_idxs_floor] * self.log_idxs_floor_w + specgram[:, self.log_idxs_ceiling] * self.log_idxs_ceiling_w
        elif(self.logspecgram_type == 'cqt' or self.logspecgram_type == 'logspecgram'):
            specgram = self.spec_layer(waveforms)

        specgram_db = self.amplitude_to_db(specgram)
        # specgram_db = (specgram_db + 80)/80
        # specgram_db = specgram_db[:, :, :-1] # remove the last frame.
        specgram_db = specgram_db.permute([0, 2, 1])
        return specgram_db




