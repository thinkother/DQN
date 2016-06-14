from Agent import *
from Env import *
import Config
from Replay import *
from Train import Train
import logging

FORMAT = '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

agent = Agent(Env(), Replay())
train = Train(agent)
train.run()
