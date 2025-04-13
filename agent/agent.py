import json
import os
import random

from .state import State


class Q_State(State):
    '''Augments the game state with Q-learning information'''

    def __init__(self, string):
        super().__init__(string)

        # key stores the state's key string (see notes in _compute_key())
        self.key = self._compute_key()

    def _compute_key(self):
        '''
        Returns a key used to index this state.

        The key should reduce the entire game state to something much smaller
        that can be used for learning. When implementing a Q table as a
        dictionary, this key is used for accessing the Q values for this
        state within the dictionary.
        '''

        # this simple key uses the 3 object characters above the frog
        # and combines them into a key string
        return ''.join([
            self.get(self.frog_x - 1, self.frog_y - 1) or '_',
            self.get(self.frog_x, self.frog_y - 1) or '_',
            self.get(self.frog_x + 1, self.frog_y - 1) or '_',
            self.get(self.frog_x - 1, self.frog_y) or '_',
            self.get(self.frog_x + 1, self.frog_y) or '_',
            self.get(self.frog_x - 1, self.frog_y + 1) or '_',
            self.get(self.frog_x, self.frog_y + 1) or '_',
            self.get(self.frog_x + 1, self.frog_y + 1) or '_'
        ])

    def reward(self):
        '''Returns a reward value for the state.'''

        if self.at_goal:
            return self.score
        elif self.is_done:
            return -36
        else:
            return (self.frog_y - self.max_y) * 4


class Agent:

    def __init__(self, train=None):

        # train is either a string denoting the name of the saved
        # Q-table file, or None if running without training
        self.train = train

        # q is the dictionary representing the Q-table
        self.q = {}

        # name is the Q-table filename
        # (you likely don't need to use or change this)
        self.name = train or 'q'

        # path is the path to the Q-table file
        # (you likely don't need to use or change this)
        self.path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), 'train', self.name + '.json')
        self.load()

        # q learning parameters
        self.alpha = 0.1 # Learning rate
        self.gamma = 0.9 # Discount factor
        self.epsilon = 0.2 # Exploration rate

        # previous state and action
        self.previous_state = None
        self.previous_action = None

    def load(self):
        '''Loads the Q-table from the JSON file'''
        try:
            with open(self.path, 'r') as f:
                self.q = json.load(f)
            if self.train:
                print('Training {}'.format(self.path))
            else:
                print('Loaded {}'.format(self.path))
        except IOError:
            if self.train:
                print('Training {}'.format(self.path))
            else:
                raise Exception('File does not exist: {}'.format(self.path))
        return self

    def save(self):
        '''Saves the Q-table to the JSON file'''
        with open(self.path, 'w') as f:
            json.dump(self.q, f)
        return self

    def choose_action(self, state_string):
        '''
        Returns the action to perform.

        This is the main method that interacts with the game interface:
        given a state string, it should return the action to be taken
        by the agent.

        The initial implementation of this method is simply a random
        choice among the possible actions. You will need to augment
        the code to implement Q-learning within the agent.
        '''

        current_state = Q_State(state_string)

        if current_state.key not in self.q.keys():
            self.q[current_state.key] = {action : 0.0 for action in State.ACTIONS}

        if not self.train:
            return max(self.q[current_state.key], key = lambda action : self.q[current_state.key][action])

        if random.uniform(0, 1) <= self.epsilon:
            weighted_actions = ["u"] * 8 + ["d", "l", "r", "_"]
            action = random.choice(weighted_actions)
        else:
            action = max(self.q[current_state.key], key = lambda action : self.q[current_state.key][action])

        if self.previous_state is not None:
            self.update_q_value(self.previous_state, self.previous_action, current_state)

        self.previous_state = current_state.key
        self.previous_action = action
        
        return action

    def update_q_value(self, previous_state, previous_action, current_state):
        reward = current_state.reward()
        if(not current_state.at_goal):
            self.q[previous_state][previous_action] = ((1 - self.alpha) * (self.q[previous_state][previous_action])) + (self.alpha * (reward + (self.gamma * max(self.q[current_state.key].values()))))
        else:
            self.q[previous_state][previous_action] = ((1 - self.alpha) * (self.q[previous_state][previous_action])) + (self.alpha * (reward + (self.gamma * 650)))
        self.save()
        self.load()