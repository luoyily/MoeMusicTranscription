

import torch
import torch.nn.functional as F
from torch import nn
import torchaudio

from .nets import CNNTrunk, FreqGroupLSTM

from .constants import *

from nnAudio import features

amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
e = 2**(1/24)
to_cqt = features.CQT(sr=SAMPLE_RATE, hop_length=HOP_LENGTH, fmin=27.5/e, n_bins=88*4,
                      bins_per_octave=BINS_PER_SEMITONE*12, output_format='Magnitude')

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),)+self.shape)


class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.squeeze(x, self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.unsqueeze(x, self.dim)


class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


class Head(nn.Module):
    def __init__(self, model_size) -> None:
        super().__init__()
        self.head = FreqGroupLSTM(model_size, 1, model_size)

    def forward(self, x):
        # input: [B x model_size x T x 88]
        # output: [B x 1 x T x 88]
        y = self.head(x)
        return y


class SubNet(nn.Module):
    def __init__(self, model_size=128,  head_names=['head'], concat=False, time_pooling=False) -> None:
        super().__init__()
        # Trunk
        self.trunk = CNNTrunk(c_in=1, c_har=16, embedding=model_size)

        # Heads
        head_size = model_size
        self.concat = concat
        if (concat):
            head_size *= 2
        self.head_names = head_names
        self.heads = nn.ModuleDict()
        for name in head_names:
            self.heads[name] = Head(head_size)

        self.time_pooling = time_pooling

    def forward(self, x):
        # input: [B x 2 x T x 352], [B x 1 x T x 88]
        # output:
        #   {"head_1": [B x T x 88],
        #    "head_2": [B x T x 88],...
        # }

        if (self.time_pooling):
            src_size = list(x.size())
            src_size[-1] = 88
            x = F.max_pool2d(x, [2, 1])

        # => [B x model_size x T x 88]
        y = self.trunk(x)
        # print(y.shape)
        output = {}
        for head in self.head_names:
            # => [B x 1 x T x 88]
            output[head] = self.heads[head](y)
            if (self.time_pooling):
                output[head] = F.interpolate(
                    output[head], size=src_size[-2:], mode='bilinear')
                # output[head] = F.upsample(output[head], size=src_size[-2:], mode='bilinear')

            output[head] = torch.clip(output[head], 1e-7, 1-1e-7)

        return output


class HPPNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_size = config['model_size']
        self.subnets = {}
        # [self.subnet_onset, self.subnet_frame]
        self.subnets['all'] = nn.ModuleList()
        if 'onset_subnet' in self.config['SUBNETS_TO_TRAIN']:
            self.subnet_onset = SubNet(model_size, config['onset_subnet_heads'])
            self.subnets['onset_subnet'] = self.subnet_onset
            self.subnets['all'].append(self.subnet_onset)
        if 'frame_subnet' in self.config['SUBNETS_TO_TRAIN']:
            self.subnet_frame = SubNet(model_size, config['frame_subnet_heads'], time_pooling=True)
            self.subnets['frame_subnet'] = self.subnet_frame
            self.subnets['all'].append(self.subnet_frame)

        self.inference_mode = False
        

    def forward(self, waveform):
        '''
        inputs:
            waveform [b x num_samples] (sample rate: 16000)
        '''
        device = self.config['device']
        global to_cqt
        to_cqt = to_cqt.to(device)
        waveform = waveform.to(device)
        # => [b x T x 352]
        cqt = to_cqt(waveform).permute([0, 2, 1]).float()
        cqt_db = amplitude_to_db(cqt)
        return self.forward_logspecgram(cqt_db)

    def forward_logspecgram(self, cqt_db):
        # inputs: [b x n], [b x T x 88]
        # inputs: cqt_db [b x T x 352]
        # outputs:
        '''
        {
            "onset":[b x T x 88],
            "frame":
            "offset":
            "velocity":
        }
        '''

        specgram_db = torch.unsqueeze(cqt_db, dim=1).to(self.config['device'])

        if self.inference_mode == False:
            specgram_db = specgram_db[:, :, :self.frame_num, :]
            pad_len = self.frame_num - specgram_db.size()[2]
            if (pad_len > 0):
                print(f'frame len < {self.frame_num}, zero_pad_len:{pad_len}')
                # => [B x 2 x T x 352]
                specgram_db = F.pad(
                    specgram_db, [0, 0, 0, pad_len], mode='replicate')
                assert specgram_db.size()[2] == self.frame_num

        # specgram_db = cqt_db
        # print(specgram_db.shape)
        # eg 1, 1, 2603, 352
        results = {}
        if 'onset_subnet' in self.config['SUBNETS_TO_TRAIN']:
            results_1 = self.subnet_onset(specgram_db)
            results.update(results_1)
        if 'frame_subnet' in self.config['SUBNETS_TO_TRAIN']:
            results_2 = self.subnet_frame(specgram_db)
            results.update(results_2)

        if self.inference_mode == True:
            del results['offset']

        return results

    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        velocity_label = batch['velocity']

        self.frame_num = frame_label.size()[-2]
        self.piano_roll_size = frame_label.size()[-2:]

        # [:, :-1]
        audio_label_reshape = audio_label.reshape(-1, audio_label.shape[-1])
        # => [n_mel x T] => [T x n_mel]
        # mel = melspectrogram(audio_label_reshape).transpose(-1, -2)

        # => [T x 88]
        # onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)

        results = self.forward(audio_label_reshape)
        predictions = {
            'onset': torch.clip(onset_label, 0, 0),
            'offset': torch.clip(offset_label, 0, 0),
            'frame': torch.clip(frame_label, 0, 0),
            'velocity': torch.clip(velocity_label, 0, 0)
        }
        losses = {}
        def bce(x, y): return -y * torch.log(x) - (1-y)*torch.log(1-x)
        if 'onset' in results.keys():
            predictions['onset'] = results['onset'].reshape(*onset_label.shape)
            # [b x T x 88]
            losses['loss/onset'] = - 2 * onset_label * torch.log(predictions['onset']) - (
                1 - onset_label) * torch.log(1-predictions['onset'])
            losses['loss/onset'] = losses['loss/onset'].mean()
            # losses['loss/onset'] = F.binary_cross_entropy(predictions['onset'], onset_label)
        if 'offset' in results.keys():
            predictions['offset'] = results['offset'].reshape(
                *offset_label.shape)
            losses['loss/offset'] = F.binary_cross_entropy(
                predictions['offset'], offset_label)
        if 'frame' in results.keys():
            predictions['frame'] = results['frame'].reshape(*frame_label.shape)
            losses['loss/frame'] = bce(predictions['frame'],
                                       frame_label).mean()
        if 'velocity' in results.keys():
            predictions['velocity'] = results['velocity'].reshape(
                *velocity_label.shape)
            losses['loss/velocity'] = self.velocity_loss(
                predictions['velocity'], velocity_label, onset_label)

        losses['loss/all'] = sum(losses.values())

        if 'onset_subnet' in self.config['SUBNETS_TO_TRAIN']:
            losses['loss/onset_subnet'] = torch.tensor(
                0.0).to(self.config['device'])
            for head in self.config['onset_subnet_heads']:
                losses['loss/onset_subnet'] += losses['loss/' + head]

        if 'frame_subnet' in self.config['SUBNETS_TO_TRAIN']:
            losses['loss/frame_subnet'] = torch.tensor(
                0.0).to(self.config['device'])
            for head in self.config['frame_subnet_heads']:
                losses['loss/frame_subnet'] += losses['loss/' + head]

        # onset_pred,  _, frame_pred, velocity_pred = self(mel)

        # predictions = {
        #     'onset': onset_pred.reshape(*onset_label.shape),
        #     # 'offset': offset_pred.reshape(*offset_label.shape),
        #     'frame': frame_pred.reshape(*frame_label.shape),
        #     'velocity': velocity_pred.reshape(*velocity_label.shape)
        # }

        # losses = {
        #     'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
        #     # 'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
        #     'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
        #     'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        # }

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator
