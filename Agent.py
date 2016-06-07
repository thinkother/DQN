import random
from chainer import serializers
from bokeh.plotting import figure, show, output_file, vplot
import cupy
from chainer import cuda

from Model import *


class Agent():

    def __init__(self, _config, _env, _replay=None, _pre_model=None):
        self.config = _config
        self.env = _env
        self.replay = _replay

        self.q_func, self.target_q_func = buildModel(_config, _pre_model)

    def step(self):
        if not self.env.in_game:
            self.env.startNewGame()

        # get cur_state
        cur_state = self.env.getState()

        # choose action in step
        action = self.chooseAction(self.q_func)


    def train(self):
        # clear grads
        self.q_func.zerograds()

        # pull tuples from memory pool
        batch_tuples = self.memory_pool.pull(self.params['batch_size'])
        if not len(batch_tuples):
            return 1.

        # stack inputs
        cur_x = []
        cur_posit_x = []
        next_x = []
        next_posit_x = []
        for t in batch_tuples:
            cur_x.append(
                self.getX(self.X_list[t.state.list_idx], t.state.tick))
            cur_lastprice, cur_bidprice, cur_askprice = self.getY(
                self.Y_list[t.state.list_idx], t.state.tick)
            cur_posit_x.append(
                self.getPositX(cur_lastprice, cur_bidprice, cur_askprice,
                               t.state.tick, t.state.position_list))
            next_x.append(
                self.getX(self.X_list[t.next_state.list_idx], t.next_state.tick))
            next_lastprice, next_bidprice, next_askprice = self.getY(
                self.Y_list[t.next_state.list_idx], t.state.tick)
            next_posit_x.append(
                self.getPositX(next_lastprice, next_bidprice, next_askprice,
                               t.next_state.tick, t.next_state.position_list))
        if self.params['gpu']:
            cur_x = [cupy.expand_dims(t, 0) for t in cur_x]
            cur_x = cupy.concatenate(cur_x, 0)
            cur_posit_x = [cupy.expand_dims(t, 0) for t in cur_posit_x]
            cur_posit_x = cupy.concatenate(cur_posit_x, 0)
            next_x = [cupy.expand_dims(t, 0) for t in next_x]
            next_x = cupy.concatenate(next_x, 0)
            next_posit_x = [cupy.expand_dims(t, 0) for t in next_posit_x]
            next_posit_x = cupy.concatenate(next_posit_x, 0)
        else:
            cur_x = np.stack(cur_x)
            cur_posit_x = np.stack(cur_posit_x)
            next_x = np.stack(next_x)
            next_posit_x = np.stack(next_posit_x)

        cur_output = self.QFunc(
            self.q_func, cur_x, cur_posit_x)  # get cur outputs
        next_output = self.QFunc(
            self.q_func, next_x, next_posit_x)  # get next outputs, NOT target
        next_action = [self.getBestAction(
            o.data, [t.next_state.position_list for t in batch_tuples]
        ) for o in next_output]  # choose next action for each output
        next_output = self.QFunc(
            self.target_q_func, next_x, next_posit_x)  # get next outputs, target

        # clear err of tuples
        for i in range(len(batch_tuples)):
            batch_tuples[i].err = 0.
        # store err count
        err_count_list = [0.] * len(batch_tuples)
        # compute grad's weights
        weights = np.array([t.P for t in batch_tuples], np.float32)
        if self.params['gpu']:
            weights = cuda.to_gpu(weights)
        if len(self.memory_pool.memory_pool):
            weights *= len(self.memory_pool.memory_pool)
        weights = weights ** -self.params['beta']
        self.params['beta'] = min(
            1, self.params['beta'] + self.params['beta_add'])
        weights /= weights.max()
        if self.params['gpu']:
            weights = cupy.expand_dims(weights, 1)
        else:
            weights = np.expand_dims(weights, 1)
        # compute grad for each head
        for k in range(self.params['K']):
            if self.params['gpu']:
                cur_output[k].grad = cupy.zeros_like(cur_output[k].data)
            else:
                cur_output[k].grad = np.zeros_like(cur_output[k].data)
            # compute grad from each tuples
            for i in range(len(batch_tuples)):
                if batch_tuples[i].mask[k].tolist():
                    cur_action_value = cur_output[k].data[
                        i][batch_tuples[i].action].tolist()
                    reward = batch_tuples[i].reward
                    next_action_value = next_output[k].data[
                        i][next_action[k][i]].tolist()
                    target_value = reward
                    # if not empty position, not terminal state
                    # if batch_tuples[i].next_state.in_game:
                    target_value += self.params['gamma'] * next_action_value
                    loss = cur_action_value - target_value
                    if cur_action_value:
                        batch_tuples[i].err += abs(loss / cur_action_value)
                        err_count_list[i] += 1
                    cur_output[k].grad[i][batch_tuples[i].action] = 2 * loss

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
            param.grad /= self.params['K']
        if self.params['grad_clip']:
            for param in self.q_func.params():
                param.grad = cupy.clip(
                    param.grad,
                    -self.params['grad_clip'], self.params['grad_clip']
                )

        # update params
        self.optimizer.update()

        # avg err
        for i in range(len(batch_tuples)):
            if err_count_list[i] > 0:
                batch_tuples[i].err /= err_count_list[i]

        self.memory_pool.merge(self.params['alpha'])

        return np.mean([t.err for t in batch_tuples])

    def chooseAction(self, _model, _state):
        self.config.epsilon = max(
            self.config.epsilon_underline,
            self.config.epsilon * self.config.epsilon_decay
        )
        random_value = random.random()
        if random_value < self.config.epsilon:
            return self.env.randomChooseAction(_state)
        else:
            x_data = self.env.getX(_state)
            output = self.QFunc(_model, x_data)
            return self.env.chooseBestAction(output, [_state])[0]

    def getState(self):
        return State(self.list_idx, self.tick, self.position_list, self.in_game)

    def getX(self, _X, _tick):
        return _X[_tick + 1 - self.params['forward_tick']: _tick + 1]

    def getPositX(self, _lastprice, _bidprice, _askprice, _tick, _position_list):
        if len(_position_list):
            if len(_position_list[0].price_list):
                price_list = np.array(_position_list[0].price_list)
                price_list -= _position_list[0].open_price
                price_list *= _position_list[0].direction
                min_value = price_list.min().tolist()
                max_value = price_list.max().tolist()
            else:
                min_value = 0.
                max_value = 0.
            if self.params['gpu']:
                return cupy.array(
                    [_lastprice - _position_list[0].open_price,
                     _bidprice - _position_list[0].open_price,
                     _askprice - _position_list[0].open_price,
                     _position_list[0].direction,
                     len(_position_list),
                     min_value, max_value],
                    cupy.float32
                )
            return np.array(
                [_lastprice - _position_list[0].open_price,
                 _bidprice - _position_list[0].open_price,
                 _askprice - _position_list[0].open_price,
                 _position_list[0].direction,
                 len(_position_list),
                 min_value, max_value],
                np.float32
            )
        else:
            if self.params['gpu']:
                return cupy.array([0, 0, 0, 0, 0, 0, 0], cupy.float32)
            return np.array([0, 0, 0, 0, 0, 0, 0], np.float32)

    def getY(self, _Y, _tick):
        return _Y[_tick]

    def QFunc(self, _model, _x_data):
        def toVariable(_data):
            if type(_data) is list:
                return [toVariable(d) for d in _data]
            else:
                return Variable(_data)
        return _model(toVariable(_x_data))

    def getBestAction(self, _output, _position_list_list):
        action_list = []
        for i in range(len(_position_list_list)):
            position_list = _position_list_list[i]
            o = _output[i]
            if not len(position_list):
                action_list.append(np.argmax(o).tolist())
            elif position_list[0].direction == 1:
                if o[0] >= o[2]:
                    action_list.append(0)
                else:
                    action_list.append(2)
            elif position_list[0].direction == -1:
                if o[0] >= o[1]:
                    action_list.append(0)
                else:
                    action_list.append(1)
            else:
                raise Exception()
        return action_list

    def getRandomAction(self, _output, _position_list_list):
        action_list = []
        for i in range(len(_position_list_list)):
            position_list = _position_list_list[i]
            o = _output[i]
            if not len(position_list):
                action_list.append(random.randint(0, 2))
            elif position_list[0].direction == 1:
                action = random.randint(0, 1)
                if action == 1:
                    action = 2
                action_list.append(action)
            elif position_list[0].direction == -1:
                action_list.append(random.randint(0, 1))
            else:
                raise Exception()
        return action_list

    def getActionValue(self, _output, _action):
        return _output[_action.T]

    def getReward(self):
        if len(self.position_list):
            if len(self.position_list[0].price_list):
                price_list = np.array(self.position_list[0].price_list)
                direction = self.position_list[0].direction
                tmp = (price_list -
                       self.position_list[0].open_price) * direction
            else:
                tmp = np.array([0.])

            final = tmp[-1].tolist()
            maxback = min(tmp).tolist()
            if final > 0 and final >= -maxback:
                reward = 1
                print '\t\t### WIN ###'
            else:
                reward = -1
                print '\t\t### LOSS ###'

            self.position_list[0].show()

            print '\t\tfinal:', final, 'maxback:', maxback, 'reward:', reward
            print '\t\t############'
            return reward
        else:
            return 0.

    def update_target_q_func(self):
        self.target_q_func.copyparams(self.q_func)

    def save(self, _during):
        serializers.save_npz('./models/during_' + str(_during), self.q_func)

    def test(self, _filename, _list_idx):
        self.q_func = HEAD(self.params)
        if self.params['gpu']:
            self.q_func.to_gpu()
        serializers.load_npz(_filename, self.q_func)

        self.X = self.X_list[_list_idx]
        self.Y = self.Y_list[_list_idx]

        buy_open_list = []
        sell_open_list = []
        buy_close_list = []
        sell_close_list = []

        for tick in range(self.params['begin_tick'], self.params['end_tick']):
            x_data = self.getX(self.X, tick)
            output = self.QFunc(self.q_func, x_data)
            action_list = [0] * 3
            for o in output:
                print o.data
                action = self.getBestAction(o.data, [self.position_list])[0]
                action_list[action] += 1
            print action_list
            action = np.argmax(action_list).tolist()
            print 'action:', action

            if action == 0:
                # hold on
                pass
            elif action == 1:
                # buy
                if len(self.position_list) and self.position_list[0].direction == -1:
                    # close position
                    self.position_list = self.position_list[1:]
                    buy_close_list.append(tick)
                else:
                    # open position
                    self.position_list.append(
                        PositionTuple(askprice.tolist(), 1))
                    buy_open_list.append(tick)
            elif action == 2:
                # sell
                if len(self.position_list) and self.position_list[0].direction == 1:
                    # close position
                    self.position_list = self.position_list[1:]
                    sell_close_list.append(tick)
                else:
                    # open position
                    if not len(self.position_list):
                        self.position_list.append(
                            PositionTuple(bidprice.tolist(), -1))
                        sell_open_list.append(tick)
            else:
                print '!!! err action'

            # stop loss
            if self.params['stop_loss']:
                if len(self.position_list) and self.position_list[0].direction == 1:
                    if bidprice - self.position_list[0].open_price < -self.params['stop_loss']:
                        sell_close_list.append(tick)
                        self.position_list = []
                if len(self.position_list) and self.position_list[0].direction == -1:
                    if askprice - self.position_list[0].open_price > self.params['stop_loss']:
                        buy_close_list.append(tick)
                        self.position_list = []

        win = 0
        loss = 0
        tie = 0
        for open_price, close_price in zip(
                cuda.to_cpu(self.Y[:, 2])[buy_open_list],
                cuda.to_cpu(self.Y[:, 1])[sell_close_list]):
            if close_price > open_price:
                win += close_price - open_price
            elif close_price < open_price:
                loss += open_price - close_price
            else:
                tie += 1
        for open_price, close_price in zip(
                cuda.to_cpu(self.Y[:, 1])[sell_open_list],
                cuda.to_cpu(self.Y[:, 2])[buy_close_list]):
            if close_price < open_price:
                win += open_price - close_price
            elif close_price > open_price:
                loss += close_price - open_price
            else:
                tie += 1
        print 'win:', win, 'loss:', loss, 'tie:', tie

        output_file("fund.html", title="fund")
        p1 = figure(plot_width=1000)
        price_array = self.Y[self.params['begin_tick']:self.params['end_tick']]
        if self.params['gpu']:
            price_array = cuda.to_cpu(price_array)
        # last price
        p1.line(range(self.params['begin_tick'], self.params['end_tick']),
                price_array[:, 0],
                color='orange', line_width=3)
        # bid price
        p1.line(range(self.params['begin_tick'], self.params['end_tick']),
                price_array[:, 1],
                color='green', alpha=0.5, line_width=1)
        # ask price
        p1.line(range(self.params['begin_tick'], self.params['end_tick']),
                price_array[:, 2],
                color='red', alpha=0.5, line_width=1)

        def get_price_array(_list, _col):
            return cuda.to_cpu(self.Y[:, _col])[_list]

        p1.square(buy_open_list,
                  get_price_array(buy_open_list, 2),
                  color='red', alpha=0.5, size=8)
        p1.square(sell_open_list,
                  get_price_array(sell_open_list, 1),
                  color='green', alpha=0.5, size=8)

        p1.circle(buy_close_list,
                  get_price_array(buy_close_list, 2),
                  color='red', alpha=0.5, size=8)
        p1.circle(sell_close_list,
                  get_price_array(sell_close_list, 1),
                  color='green', alpha=0.5, size=8)

        show(p1)


if __name__ == '__main__':
    agent = Agent(params)
    tmp_list = [PositionTuple(1, 1), PositionTuple(2, -1)]
    state = State(1, tmp_list)
    tmp_list.append(PositionTuple(3, 1))
    state.show()
