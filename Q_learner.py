import numpy as np
import tensorflow as tf
from collections import deque
import multiprocessing

import logging

logging.basicConfig(filename='example.log', level=logging.DEBUG)

def dense(x, weights, bias, activation=tf.identity, **activation_kwargs):
    """Dense layer."""
    z = tf.matmul(x, weights) + bias
    return activation(z, **activation_kwargs)


def init_weights(shape, initializer):
    """Initialize weights for tensorflow layer."""
    weights = tf.Variable(
        initializer(shape,dtype=tf.float64),
        trainable=True,
        dtype=tf.float64
    )

    return weights


class Network(object):
    """Q-function approximator."""

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size=[50, 50],
                 weights_initializer=tf.initializers.glorot_uniform(),
                 bias_initializer=tf.initializers.zeros(),
                 optimizer=tf.optimizers.Adam,
                 **optimizer_kwargs):
        """Initialize weights and hyperparameters."""
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        np.random.seed(41)

        self.initialize_weights(weights_initializer, bias_initializer)
        self.optimizer = optimizer(**optimizer_kwargs)

    def initialize_weights(self, weights_initializer, bias_initializer):
        """Initialize and store weights."""
        wshapes = [
            [self.input_size, self.hidden_size[0]],
            [self.hidden_size[0], self.hidden_size[1]],
            [self.hidden_size[1], self.output_size]
        ]

        bshapes = [
            [1, self.hidden_size[0]],
            [1, self.hidden_size[1]],
            [1, self.output_size]
        ]

        self.weights = [init_weights(s, weights_initializer) for s in wshapes]
        self.biases = [init_weights(s, bias_initializer) for s in bshapes]

        self.trainable_variables = self.weights + self.biases

    def model(self, inputs):
        """Given a state vector, return the Q values of actions."""
        h1 = dense(inputs, self.weights[0], self.biases[0], tf.nn.relu)
        h2 = dense(h1, self.weights[1], self.biases[1], tf.nn.relu)

        out = dense(h2, self.weights[2], self.biases[2])

        return out

    def train_step(self, inputs, targets, actions_one_hot):
        """Update weights."""
        with tf.GradientTape() as tape:
            qvalues = tf.squeeze(self.model(inputs))
            preds = tf.reduce_sum(qvalues * actions_one_hot, axis=1)
            loss = tf.losses.mean_squared_error(targets, preds)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))


class Memory(object):
    """Memory buffer for Experience Replay."""

    def __init__(self, max_size):
        """Initialize a buffer containing max_size experiences."""
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        """Add an experience to the buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        buffer_size = len(self.buffer)
        index = np.random.choice(
            np.arange(buffer_size),
            size=batch_size,
            replace=False
        )

        return [self.buffer[i] for i in index]

    def size(self):
        """get the size of the buffer."""
        return len(self.buffer)

    def __len__(self):
        """Interface to access buffer length."""
        return len(self.buffer)


class Agent(object):
    """Deep Q-learning agent."""

    def __init__(self,
                 state_space_size,
                 action_space_size,
                 discount=0.99,
                 batch_size=32,
                 max_explore=1,
                 min_explore=0.05,
                 anneal_rate=(1 / 100),
                 replay_memory_size=100000,
                 replay_start_size=50):
        """Set parameters, initialize network."""
        self.action_space_size = action_space_size

        self.learning_model = Network(state_space_size, action_space_size)

        # training parameters
        self.discount = discount
        self.batch_size = batch_size

        # policy during learning
        self.max_explore = max_explore + (anneal_rate * replay_start_size)
        self.min_explore = min_explore
        self.anneal_rate = anneal_rate
        self.steps = 0

        # replay memory
        self.memory = Memory(replay_memory_size)
        self.replay_start_size = replay_start_size


    def handle_episode_start(self):
        self.last_states, self.last_actions = dict(), dict()

    def step(self, mote, observation, training=True):
        """Observe state and rewards, select action.
        It is assumed that `observation` will be an object with
        a `state` vector and a `reward` float or integer. The reward
        corresponds to the action taken in the previous step.
        """
        last_state, last_action = self.last_states.get(mote), self.last_actions.get(mote)
        last_reward = observation.reward
        state = observation.state

        action = self.policy(state, training)

        if training:
            self.steps += 1

            if last_state is not None:
                experience = {
                    "state": last_state,
                    "action": last_action,
                    "reward": last_reward,
                    "next_state": state
                }

                self.memory.add(experience)
            else:
                self.steps -= 1

            if self.steps > self.replay_start_size:
                self.train_network()

        self.last_states[mote] = state
        self.last_actions[mote] = action

        return action

    def policy(self,state, training):
        """Epsilon-greedy policy for training, greedy policy otherwise."""
        explore_prob = self.max_explore - (self.steps * self.anneal_rate)
        explore = max(explore_prob, self.min_explore) > np.random.rand()

        if training and explore:
            action = np.random.randint(self.action_space_size)
        else:
            print("made a choice")
            inputs = np.array(state, dtype=np.float64)
            inputs = np.expand_dims(inputs, 0)

            qvalues = self.learning_model.model(inputs)
            action = np.squeeze(np.argmax(qvalues, axis=-1))

        return int(action)


    def train_network(self):
        """Update online network weights."""
        batch = self.memory.sample(self.batch_size)
        inputs = np.array([b["state"] for b in batch],dtype=np.float64)
        actions = np.array([b["action"] for b in batch],dtype=int)
        rewards = np.array([b["reward"] for b in batch],dtype=np.float64)
        next_inputs = np.array([b["next_state"] for b in batch],dtype=np.float64)

        actions_one_hot = np.eye(self.action_space_size)[actions]

        next_qvalues = np.squeeze(self.learning_model.model(next_inputs))
        targets = rewards + self.discount * np.amax(next_qvalues, axis=-1)

        self.learning_model.train_step(inputs, targets, actions_one_hot)