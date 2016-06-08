import random
from chainer import serializers
from bokeh.plotting import figure, show, output_file, vplot
import cupy
from chainer import cuda

from Model import *
from Replay import ReplayTuple


class Agent():

    def __init__(self, _config, _env, _replay=None, _pre_model=None):
        self.config = _config
        self.env = _env
        self.replay = _replay

        # model for train, model for target
        self.q_func, self.target_q_func = buildModel(_config, _pre_model)

    def step(self):
        if not self.env.in_game:
            self.env.startNewGame()

        # get current state
        cur_state = self.env.getState()
        # choose action in step
        action = self.chooseAction(self.q_func, cur_state)
        # do action and get reward
        reward = self.env.doAction(action)
        # get new state
        next_state = self.env.getState()

        # randomly decide to store tuple into pool
        if random.random() < self.config.replay_p:
            # store replay_tuple into memory pool
            replay_tuple = ReplayTuple(
                cur_state, action, reward, next_state,
                # get mask for bootstrap
                np.random.binomial(1, self.config.p, (self.config.K))
            )
            self.replay.push(replay_tuple)

    def train(self):
        # clear grads
        self.q_func.zerograds()

        # pull tuples from memory pool
        batch_tuples = self.replay.pull(self.config.batch_size)
        if not len(batch_tuples):
            return

        # stack inputs
        cur_x = [self.env.getX(t.state) for t in batch_tuples]
        next_x = [self.env.getX(t.next_state) for t in batch_tuples]
        # merge inputs into one array
        if self.config.gpu:
            cur_x = [cupy.expand_dims(t, 0) for t in cur_x]
            cur_x = cupy.concatenate(cur_x, 0)
            next_x = [cupy.expand_dims(t, 0) for t in next_x]
            next_x = cupy.concatenate(next_x, 0)
        else:
            cur_x = np.stack(cur_x)
            next_x = np.stack(next_x)

        # get cur outputs
        cur_output = self.QFunc(self.q_func, cur_x)
        # get next outputs, NOT target
        next_output = self.QFunc(self.q_func, next_x)
        # choose next action for each output
        next_action = [
            self.env.getBestAction(
                o.data,
                [t.next_state for t in batch_tuples]
            ) for o in next_output  # for each head in Model
        ]
        # get next outputs, target
        next_output = self.QFunc(self.target_q_func, next_x)

        # clear err of tuples
        for t in batch_tuples:
            t.err = 0.
        # store err count
        err_count_list = [0.] * len(batch_tuples)
        # compute grad's weights
        weights = np.array([t.P for t in batch_tuples], np.float32)
        if self.config.gpu:
            weights = cuda.to_gpu(weights)
        if self.replay.getPoolSize():
            weights *= self.replay.getPoolSize()
        weights = weights ** -self.config.beta
        weights /= weights.max()
        if self.config.gpu:
            weights = cupy.expand_dims(weights, 1)
        else:
            weights = np.expand_dims(weights, 1)

        # update beta
        self.config.beta = min(1, self.config.beta + self.config.beta_add)

        # compute grad for each head
        for k in range(self.config.K):
            if self.config.gpu:
                cur_output[k].grad = cupy.zeros_like(cur_output[k].data)
            else:
                cur_output[k].grad = np.zeros_like(cur_output[k].data)
            # compute grad from each tuples
            for i in range(len(batch_tuples)):
                if batch_tuples[i].mask[k].tolist():
                    cur_action_value = \
                        cur_output[k].data[i][batch_tuples[i].action].tolist()
                    reward = batch_tuples[i].reward
                    next_action_value = \
                        next_output[k].data[i][next_action[k][i]].tolist()
                    target_value = reward
                    # if not empty position, not terminal state
                    if batch_tuples[i].next_state.in_game:
                        target_value += self.params['gamma'] * next_action_value
                    loss = cur_action_value - target_value
                    cur_output[k].grad[i][batch_tuples[i].action] = 2 * loss
                    # count err
                    if cur_action_value:
                        batch_tuples[i].err += abs(loss / cur_action_value)
                        err_count_list[i] += 1

            # multiply weights with grad
            if self.params['gpu']:
                cur_output[k].grad = cupy.multiply(
                    cur_output[k].grad, weights)
                cur_output[k].grad = cupy.clip(cur_output[k].grad, -1, 1)
            else:
                cur_output[k].grad = np.multiply(
                    cur_output[k].grad, weights)
                cur_output[k].grad = np.clip(cur_output[k].grad, -1, 1)
            # backward
            cur_output[k].backward()

        # adjust grads
        for param in self.q_func.shared.params():
            param.grad /= self.config.K

        # update params
        self.optimizer.update()

        # avg err
        for i in range(len(batch_tuples)):
            if err_count_list[i] > 0:
                batch_tuples[i].err /= err_count_list[i]

        self.replay.merge(self.config.alpha)

        return np.mean([t.err for t in batch_tuples])

    def chooseAction(self, _model, _state):
        # update epsilon
        self.config.epsilon = max(
            self.config.epsilon_underline,
            self.config.epsilon * self.config.epsilon_decay
        )
        random_value = random.random()
        if random_value < self.config.epsilon:
            # randomly choose
            return self.env.getRandomAction(_state)
        else:
            # use model to choose
            x_data = self.env.getX(_state)
            output = self.QFunc(_model, x_data)
            return self.env.getBestAction(output, [_state])[0]

    def QFunc(self, _model, _x_data):
        def toVariable(_data):
            if type(_data) is list:
                return [toVariable(d) for d in _data]
            else:
                return Variable(_data)
        return _model(toVariable(_x_data))

    def update_target_q_func(self):
        self.target_q_func.copyparams(self.q_func)

    def save(self, _during):
        serializers.save_npz('./models/during_' + str(_during), self.q_func)
