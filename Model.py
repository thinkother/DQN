import numpy as np
import cPickle
from time import time
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer import optimizers
from chainer import Chain, ChainList
import chainer.serializers as S
import chainer.computational_graph as c


class CNN(Chain):

    def __init__(self, ch_1, ksize_1, st_1, ch_2, ksize_2, st_2):
        super(CNN, self).__init__(
            conv_1=L.Convolution2D(
                in_channels=1,
                out_channels=ch_1,
                ksize=ksize_1,
                stride=st_1
            ),
            conv_2=L.Convolution2D(
                in_channels=1,
                out_channels=ch_2,
                ksize=ksize_2,
                stride=st_2
            ),
            bn_1=L.BatchNormalization(ch_1),
            bn_2=L.BatchNormalization(ch_2)
        )

    def __call__(self, x, is_train):
        x = F.expand_dims(x, 1)
        y = self.bn_1(self.conv_1(x), test=not is_train)
        y = F.swapaxes(y, 1, 3)
        y = F.relu(y)
        y = F.max_pooling_2d(y, ksize=(3, 1))
        y = self.bn_2(self.conv_2(y), test=not is_train)
        y = F.relu(y)
        return y


class Shared(Chain):

    def __init__(self, _params):
        super(Shared, self).__init__(
            conv_25_ticks=CNN(
                ch_1=_params['conv_num'][0],
                ksize_1=(_params['conv_size'][0], _params['input_size']),
                st_1=_params['conv_stride'][0],
                ch_2=_params['conv_num'][1],
                ksize_2=(_params['conv_size'][1], _params['conv_num'][0]),
                st_2=_params['conv_stride'][1]
            ),
            conv_5_ticks=CNN(
                ch_1=_params['conv_num'][2],
                ksize_1=(_params['conv_size'][2], _params['input_size']),
                st_1=_params['conv_stride'][2],
                ch_2=_params['conv_num'][3],
                ksize_2=(_params['conv_size'][3], _params['conv_num'][2]),
                st_2=_params['conv_stride'][3]
            ),
            conv_1_ticks=CNN(
                ch_1=_params['conv_num'][4],
                ksize_1=(_params['conv_size'][4], _params['input_size']),
                st_1=_params['conv_stride'][4],
                ch_2=_params['conv_num'][5],
                ksize_2=(_params['conv_size'][5], _params['conv_num'][4]),
                st_2=_params['conv_stride'][5]
            ),
        )

    def __call__(self, x, posit_x, is_train):
        x_25_ticks = x
        y_25_ticks = self.conv_25_ticks(x_25_ticks, is_train)
        x_5_ticks = F.split_axis(x_25_ticks, 5, 1)[-1]
        y_5_ticks = self.conv_5_ticks(x_5_ticks, is_train)
        x_1_ticks = F.split_axis(x_5_ticks, 5, 1)[-1]
        y_1_ticks = self.conv_1_ticks(x_1_ticks, is_train)

        y_25_ticks = F.reshape(
            y_25_ticks,
            (y_25_ticks.data.shape[0], y_25_ticks.data.shape[1] *
             y_25_ticks.data.shape[2] * y_25_ticks.data.shape[3])
        )
        y_5_ticks = F.reshape(
            y_5_ticks,
            (y_5_ticks.data.shape[0], y_5_ticks.data.shape[1] *
             y_5_ticks.data.shape[2] * y_5_ticks.data.shape[3])
        )
        y_1_ticks = F.reshape(
            y_1_ticks,
            (y_1_ticks.data.shape[0], y_1_ticks.data.shape[1] *
             y_1_ticks.data.shape[2] * y_1_ticks.data.shape[3])
        )
        y = F.concat([y_25_ticks, y_5_ticks, y_1_ticks, posit_x], 1)

        return y


class HEAD(ChainList):

    def __init__(self, _params):
        self.train = True
        self.config = _params
        self.shared = Shared(self.config)
        l_list = [self.shared]
        for i in range(self.config['K']):
            l_list.append(
                L.Linear(self.config['class_input'],
                         self.config['class_hidden_1'])
            )
            l_list.append(
                L.Linear(self.config['class_hidden_1'],
                         self.config['class_hidden_2'])
            )
            l_list.append(
                L.Linear(self.config['class_hidden_2'],
                         self.config['class_output'])
            )
        super(HEAD, self).__init__(*l_list)

    def __call__(self, x, posit_x):
        y_shared = self.shared(x, posit_x, self.train)
        y_shared = F.dropout(y_shared, ratio=0.1, train=self.train)

        y = []
        for i in range(self.config['K']):
            hidden_1 = F.relu(self[3 * i + 1](y_shared))
            hidden_2 = F.relu(self[3 * i + 2](hidden_1))
            y.append(self[3 * i + 3](hidden_2))

        # g = c.build_computational_graph(y)
        # with open('graph', 'w') as o:
        #     o.write(g.dump())
        # raw_input()
        return y

    def training(self):
        self.train = True

    def evaluating(self):
        self.train = False


def buildModel(_params, _pre_model=None):
    q_func = HEAD(_params)
    if _pre_model:
        _params['epsilon'] = 0.
        _params['beta'] = 1.
        S.load_npz(_pre_model, q_func)
    target_q_func = q_func.copy()
    return q_func, target_q_func
