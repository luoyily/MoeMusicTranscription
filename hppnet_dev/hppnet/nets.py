import torch
import torch.nn as nn


from .constants import *
from .lstm import BiLSTM


class FreqGroupLSTM(nn.Module):
    def __init__(self, channel_in, channel_out, lstm_size) -> None:
        super().__init__()

        self.channel_out = channel_out

        self.lstm = BiLSTM(channel_in, lstm_size//2)
        self.linear = nn.Linear(lstm_size, channel_out)

    def forward(self, x):
        # inputs: [b x c_in x T x freq]
        # outputs: [b x c_out x T x freq]

        b, c_in, t, n_freq = x.size() 

        # => [b x freq x T x c_in] 
        x = torch.permute(x, [0, 3, 2, 1])

        # => [(b*freq) x T x c_in]
        x = x.reshape([b*n_freq, t, c_in])
        # => [(b*freq) x T x lstm_size]
        x = self.lstm(x)
        # => [(b*freq) x T x c_out]
        x = self.linear(x)
        # => [b x freq x T x c_out]
        x = x.reshape([b, n_freq, t, self.channel_out])
        # => [b x c_out x T x freq]
        x = torch.permute(x, [0, 3, 2, 1])
        x = torch.sigmoid(x)
        return x


# Modified the padding parameter 'same' to an equivalent value for onnx exports
class HarmonicDilatedConv(nn.Module):
    def __init__(self, c_in, c_out) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(c_in, c_out, [1, 3], padding=(0,48), dilation=[1, 48])
        self.conv_2 = nn.Conv2d(c_in, c_out, [1, 3], padding=(0,76), dilation=[1, 76])
        self.conv_3 = nn.Conv2d(c_in, c_out, [1, 3], padding=(0,96), dilation=[1, 96])
        self.conv_4 = nn.Conv2d(c_in, c_out, [1, 3], padding=(0,111), dilation=[1, 111])
        self.conv_5 = nn.Conv2d(c_in, c_out, [1, 3], padding=(0,124), dilation=[1, 124])
        self.conv_6 = nn.Conv2d(c_in, c_out, [1, 3], padding=(0,135), dilation=[1, 135])
        self.conv_7 = nn.Conv2d(c_in, c_out, [1, 3], padding=(0,144), dilation=[1, 144])
        self.conv_8 = nn.Conv2d(c_in, c_out, [1, 3], padding=(0,152), dilation=[1, 152])
    def forward(self, x):
        x = self.conv_1(x) + self.conv_2(x) + self.conv_3(x) + self.conv_4(x) +\
            self.conv_5(x) + self.conv_6(x) + self.conv_7(x) + self.conv_8(x)
        x = torch.relu(x)
        return x


class CNNTrunk(nn.Module):
    def get_conv2d_block(self, channel_in,channel_out, kernel_size = [1, 3], pool_size = None, dilation = [1, 1],padding=None):
        if(pool_size == None):
            return nn.Sequential( 
                nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding=padding, dilation=dilation),
                nn.ReLU(),
                nn.InstanceNorm2d(channel_out),
                
            )
        else:
            return nn.Sequential( 
                nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding=padding, dilation=dilation),
                nn.ReLU(),
                nn.MaxPool2d(pool_size),
                nn.InstanceNorm2d(channel_out)
            )

    def __init__(self, c_in = 1, c_har = 16,  embedding = 128) -> None:
        super().__init__()

        self.block_1 = self.get_conv2d_block(c_in, c_har, kernel_size=7,padding=3)
        self.block_2 = self.get_conv2d_block(c_har, c_har, kernel_size=7,padding=3)
        self.block_2_5 = self.get_conv2d_block(c_har, c_har, kernel_size=7,padding=3)

        c3_out = embedding
        
        self.conv_3 = HarmonicDilatedConv(c_har, c3_out)

        self.block_4 = self.get_conv2d_block(c3_out, c3_out, pool_size=[1, 4], dilation=[1, 48],padding=(0,48))
        self.block_5 = self.get_conv2d_block(c3_out, c3_out, dilation=[1, 12],padding=(0,12))
        self.block_6 = self.get_conv2d_block(c3_out, c3_out, [5,1],padding=(2,0))
        self.block_7 = self.get_conv2d_block(c3_out, c3_out, [5,1],padding=(2,0))
        self.block_8 = self.get_conv2d_block(c3_out, c3_out, [5,1],padding=(2,0))


    def forward(self, log_gram_db):
        # inputs: [b x 2 x T x n_freq] , [b x 1 x T x 88]
        # outputs: [b x T x 88]

        x = self.block_1(log_gram_db)
        x = self.block_2(x)
        x = self.block_2_5(x)
        x = self.conv_3(x)
        x = self.block_4(x)
        # => [b x 1 x T x 88]

        x = self.block_5(x)
        # => [b x ch x T x 88]
        x = self.block_6(x) # + x
        x = self.block_7(x) # + x
        x = self.block_8(x) # + x


        return x