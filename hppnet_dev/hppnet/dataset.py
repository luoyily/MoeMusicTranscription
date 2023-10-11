import os

import numpy as np
from torch.utils.data import Dataset
import torch
import torch.utils.data
import torch.nn.functional as F
import h5py

from hppnet.constants import *



os.environ['HDF5_USE_FILE_LOCKING'] = "FALSE"

def max_pooling(x, pooling_size):
    # => [1 x W x H]
    x = torch.unsqueeze(x, dim=0)
    x = torch.max_pool2d(x, pooling_size)
    # => [W x H]
    x = torch.squeeze(x, dim=0)
    return x


class PianoRollAudioDatasetH5(Dataset):
    
    def __init__(self, folders=[],sequence_length=None, seed=42) -> None:
        self.sequence_length = sequence_length
        self.random = np.random.RandomState(seed)
        self.data = []
        for folder in folders :
            self.data += [os.path.join(folder, fn) for fn in os.listdir(folder) if fn.endswith('h5')]

    def __getitem__(self, index):
        '''
        # reutrn :
        # [n_step, midi_bins]
        '''

        h5_path = self.data[index]
        with h5py.File(h5_path, 'r') as data:
            result = dict(path=data['path'][()])
            audio_length = data['audio'].shape[0]
            
            if self.sequence_length is not None and audio_length > self.sequence_length:
                # audio_length = data['audio'].shape[0] # len(data['audio'])
                step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
                n_steps = self.sequence_length // HOP_LENGTH
                step_end = step_begin + n_steps

                begin = step_begin * HOP_LENGTH
                end = begin + self.sequence_length

                result['audio'] = data['audio'][begin:end]
                result['label'] = data['label'][step_begin:step_end, :]
                result['velocity'] = data['velocity'][step_begin:step_end, :]
            else:

                result['audio'] = data['audio'][:]
                result['label'] = data['label'][:]
                result['velocity'] = data['velocity'][:]

            result['audio'] = torch.tensor(result['audio'])
            result['label'] = torch.tensor(result['label'])
            result['velocity'] = torch.tensor(result['velocity'])

            if audio_length < self.sequence_length:
                audio_pad = self.sequence_length - audio_length
                label_pad = (self.sequence_length // HOP_LENGTH) - result['label'].size()[0]
                result['audio'] = F.pad(result['audio'],(0,audio_pad), mode="constant", value=0)
                result['label'] = F.pad(result['label'],(0,0,0,label_pad), mode="constant", value=0)
                result['velocity'] = F.pad(result['velocity'],(0,0,0,label_pad), mode="constant", value=0)
            
            result['audio'] = result['audio'].float().div_(32768.0)
            result['onset'] = (result['label'] == 3).float()
            result['offset'] = (result['label'] == 1).float()
            result['frame'] = (result['label'] > 1).float()
            result['velocity'] = result['velocity'].float().div_(128.0)

        return result
    def __len__(self):
        return len(self.data)
