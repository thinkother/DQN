import numpy as np
import chainer.serializers as S
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from chainer import Variable
import cPickle
from bokeh.plotting import figure, show, output_file, vplot

from Agent import *

params = {}
params['input_size'] = 12
params['forward_tick'] = 750
params['conv_num'] = [30, 50, 30, 50, 30, 50]
params['conv_size'] = [25, 3, 5, 3, 3, 3]
params['conv_stride'] = [12, 1, 3, 1, 1, 1]
params['class_input'] = 2107
params['class_hidden_1'] = 500
params['class_hidden_2'] = 250
params['class_output'] = 3
params['K'] = 10

params['gpu'] = True
params['begin_tick'] = 1000
params['end_tick'] = 5000
params['position_rate'] = 1
params['stop_loss'] = 0
params['stop_win'] = 0.05
print params

X_test = np.load('./utils/dataset/X_test.npy')
Y_test = np.load('./utils/dataset/Y_test.npy')

if params['gpu']:
    X_test = [cuda.to_gpu(d) for d in X_test]
    Y_test = [cuda.to_gpu(d) for d in Y_test]

agent = Agent(params, X_test, Y_test, _is_train=False)

agent.test('./models/during_10000', 0)
