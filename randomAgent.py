import numpy as np

class Agent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
    def act(self, state):

        return np.random.choice(self.num_actions)