import tensorflow as tf

from .tfagent import TFAgent


class Actor(TFAgent):
    def __init__(self, lr, discount):
