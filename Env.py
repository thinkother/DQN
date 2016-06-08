class State:

    def __init__(self):
        pass

    def show(self):
        pass


class Env:

    def __init__(self):
        self.in_game = False

    def startNewGame(self):
        self.in_game = True

    def getState(self):
        return

    def doAction(self):
        pass

    def getX(self):
        return

    def getY(self):
        return

    def getRandomAction(self, _state):
        return

    def getBestAction(self, _data, _state_list):
        return
