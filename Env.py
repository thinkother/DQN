class State:

    def __init__(self, _list_idx, _tick, _position_list, _in_game):
        self.list_idx = _list_idx
        self.tick = _tick
        self.position_list = _position_list[:]
        self.in_game = _in_game

    def show(self):
        print 'list_idx:', self.list_idx, 'tick:', self.tick, 'position:'
        for position in self.position_list:
            position.show()
        print 'in_game:', self.in_game

class Env:
    def __init__(self):
        self.in_game = False

    def startNewGame(self):
        self.in_game = True

    def getState(self):
        return
