from sklearn import model_selection
from torch import nn
import torch
from models.quartznet.base.layers import MainBlock
from utils.model_utils import init_weights


_quartznet5x5_config = [
    {'filters': 256, 'repeat': 1, 'kernel': 33, 'stride': 2, 'dilation': 1, 'dropout': 0.2, 'residual': False, 'separable': True},

    {'filters': 256, 'repeat': 5, 'kernel': 33, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': True, 'separable': True},

    {'filters': 256, 'repeat': 5, 'kernel': 39, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': True, 'separable': True},

    {'filters': 512, 'repeat': 5, 'kernel': 51, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': True, 'separable': True},

    {'filters': 512, 'repeat': 5, 'kernel': 63, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': True, 'separable': True},

    {'filters': 512, 'repeat': 5, 'kernel': 75, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': True, 'separable': True},

    {'filters': 512, 'repeat': 1, 'kernel': 87, 'stride': 1, 'dilation': 2, 'dropout': 0.2, 'residual': False, 'separable': True},

    {'filters': 1024, 'repeat': 1, 'kernel': 1, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': False, 'separable': False}
]


class QuartzNet(nn.Module):
    def __init__(
            self,
            feat_in,
            vocab_size,
            activation='relu',
            normalization_mode="batch",
            norm_groups=-1,
            frame_splicing=1,
            init_mode='xavier_uniform',
            **kwargs
    ):
        super(QuartzNet, self).__init__()

        feat_in = feat_in * frame_splicing
        self.stride = 1

        residual_panes = []
        layers = []
        model_config = _quartznet5x5_config
        for lcfg in model_config:
            self.stride *= lcfg['stride']

            groups = lcfg.get('groups', 1)
            separable = lcfg.get('separable', False)
            residual = lcfg.get('residual', True)
            layers.append(
                MainBlock(feat_in,
                    lcfg['filters'],
                    repeat=lcfg['repeat'],
                    kernel_size=lcfg['kernel'],
                    stride=lcfg['stride'],
                    dilation=lcfg['dilation'],
                    dropout=lcfg['dropout'] if 'dropout' in lcfg else 0.0,
                    residual=residual,
                    groups=groups,
                    separable=separable,
                    normalization=normalization_mode,
                    norm_groups=norm_groups,
                    activation=activation))
            feat_in = lcfg['filters']

        self.encoder = nn.Sequential(*layers)
        self.classify = nn.Conv1d(1024, vocab_size,
                      kernel_size=1, bias=True)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, audio_signal):
        feat = self.encoder(audio_signal)
        # BxCxT
        return self.classify(feat)

    def load_weights(self, path, map_location='cpu'):
        weights = torch.load(path, map_location=map_location)
        print(self.load_state_dict(weights, strict=False))