import logging

##########################################
## simple env in bootstrap DQN for demo ##
##########################################

class State:

    def __init__(self, _in_game, _index):
        self.in_game = _in_game

    def show(self):
        pass


class Env:

    def __init__(self):
        self.in_game = False

    def startNewGame(self):
        self.in_game = True
        logging.info('Start new game')

    def getState(self):
        return State(self.in_game)

    def doAction(self, _action):
        pass

    def getX(self):
        return

    def getY(self):
        return

    def getRandomAction(self, _state):
        return

    def getBestAction(self, _data, _state_list):
        return
