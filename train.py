import numpy as np
import cupy
import cPickle
from time import time
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers, cuda
import chainer.serializers as S
import chainer.computational_graph as c
import random
import cupy
import math

from Agent import *
from Replay import *

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
params['epsilon'] = 0.5
params['epsilon_decay'] = 0.0001
params['gpu'] = True
params['begin_tick'] = 1000
params['end_tick'] = -100
params['end_game_tick'] = -50
params['p'] = 0.5  # for binomial
params['alpha'] = 0.7
params['beta'] = 0.5
params['beta_add'] = 0.0001
params['gamma'] = 0.99
params['stop_game'] = 2000
# params['lambda'] = 0.5  # mean - lambda * std
params['replay_p'] = 0.5
params['replay_size'] = 100000
params['batch_size'] = 64
params['train_rate'] = 10
params['train_during'] = 1
params['update_target_q_func_during'] = 10000
params['save_during'] = params['update_target_q_func_during']
params['grad_clip'] = 5
params['reward_clip'] = 10

print params

X_train = np.load('./utils/dataset/X_train.npy')
Y_train = np.load('./utils/dataset/Y_train.npy')

if params['gpu']:
    X_train = [cuda.to_gpu(d) for d in X_train]
    Y_train = [cuda.to_gpu(d) for d in Y_train]

replay_memory = Replay(params['replay_size'])
agent = Agent(params, X_train, Y_train, replay_memory)

during = 0
while 1:
    agent.step()

    during += 1
    if during % params['train_during'] == 0:
        start_time = time()
        err = agent.train()
        print '\tduring:', during, 'time:', time() - start_time, 'err:', err
        params['train_during'] = int(params['train_rate'] / err)
        params['train_during'] = max(params['train_during'], 1)
        params['train_during'] = min(params['train_during'], params['batch_size'])
    if during % params['update_target_q_func_during'] == 0:
        agent.update_target_q_func()
    if during % params['save_during'] == 0:
        agent.save(during)
