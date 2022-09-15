import json
import os
from random import random

import numpy as np
import tensorflow as tf
from collections import deque

import logging

logging.basicConfig(filename='loss_function.log', level=logging.DEBUG)

def dense(x, weights, bias, activation=tf.identity, **activation_kwargs):
    """Dense layer."""
    z = tf.matmul(x, weights) + bias
    return activation(z, **activation_kwargs)


def init_weights(shape, initializer):
    """Initialize weights for tensorflow layer."""
    weights = tf.Variable(
        initializer(shape, dtype=tf.float64),
        trainable=True,
        dtype=tf.float64
    )

    return weights


class Network(object):
    """Q-function approximator."""

    def __init__(self,time_step, feature_size, output_size, learning_rate = 0.01):
        init = tf.keras.initializers.HeUniform()
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=(time_step,feature_size,1)))
        self.model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation=None, kernel_initializer=init))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
        #self.model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation=None, kernel_initializer=init))
        #self.model.add(tf.keras.layers.BatchNormalization())
        #self.model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
        self.model.add(tf.keras.layers.Dropout(0.25))
        self.model.add(tf.keras.layers.Dense(120, activation='relu', kernel_initializer=init))
        self.model.add(tf.keras.layers.Dense(120, activation='relu', kernel_initializer=init))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(output_size, activation='tanh'))


        self.model.compile(
            loss=tf.keras.losses.MeanSquaredLogarithmicError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['accuracy'])

        checkpoint_path = "training_1_full/cp.ckpt"

        self.model.load_weights(checkpoint_path)


    def model(self, state):
        return self.model.predict(state)[0]

    def train_step(self, train_batch,validation_batch):
        learning_rate = 0.7  # Learning rate
        discount_factor = 0.618

        current_states = np.array([experience["state"] for experience in train_batch])
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([experience["next_state"] for experience in train_batch])
        future_qs_list = self.model.predict(new_current_states)

        X = []
        Y = []
        for index, experience in enumerate(train_batch):
            max_future_q = experience["reward"] + discount_factor * np.max(future_qs_list[index])

            current_qs = current_qs_list[index]
            current_qs[experience["action"]] = (1 - learning_rate) * current_qs[experience["action"]] + learning_rate * max_future_q

            X.append(experience["state"])
            Y.append(current_qs)

        current_states = np.array([experience["state"] for experience in validation_batch])
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([experience["next_state"] for experience in validation_batch])
        future_qs_list = self.model.predict(new_current_states)

        X_val = []
        Y_val = []
        for index, experience in enumerate(validation_batch):
            max_future_q = experience["reward"] + discount_factor * np.max(future_qs_list[index])

            current_qs = current_qs_list[index]
            current_qs[experience["action"]] = (1 - learning_rate) * current_qs[
                experience["action"]] + learning_rate * max_future_q

            X_val.append(experience["state"])
            Y_val.append(current_qs)

        checkpoint_path = "training_1/cp.ckpt"

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)


        hist = self.model.fit(np.array(X), np.array(Y), batch_size=128, shuffle=True, epochs=10, validation_data=(np.array(X_val), np.array(Y_val)),callbacks=[cp_callback])




class Memory(object):
    """Memory buffer for Experience Replay."""

    def __init__(self, max_size,output_experiences):
        """Initialize a buffer containing max_size experiences."""
        self.buffer = deque(maxlen=max_size)
        self.buffer_file = open("experiences.txt", "a")
        self.output_experiences = output_experiences

    def add(self, experience):
        """Add an experience to the buffer."""
        self.buffer.append(experience)
        state_list = list()
        if self.output_experiences:
            for transmission in experience["state"]:
                state_list.append(transmission.tolist())
            next_state_list = list()
            for transmission in experience["next_state"]:
                next_state_list.append(transmission.tolist())
            self.buffer_file.write(json.dumps({"state":state_list,"reward":experience["reward"],"action":experience["action"],"next_state":next_state_list}))
            self.buffer_file.write("\n")

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
                 time_Steps,
                 feature_size,
                 action_space_size,
                 discount=0.99,
                 batch_size=128,
                 max_explore=1,
                 min_explore=0.05,
                 anneal_rate=(1 / 400),
                 replay_memory_size=100000,
                 replay_start_size=50,output_experiences = False):
        """Set parameters, initialize network."""
        self.action_space_size = action_space_size

        self.learning_model = Network(time_Steps, feature_size, action_space_size)


        # training parameters
        self.discount = discount
        self.batch_size = batch_size

        # policy during learning
        self.max_explore = max_explore + (anneal_rate * replay_start_size)
        self.min_explore = min_explore
        self.anneal_rate = anneal_rate
        self.steps = 0

        # replay memory
        self.memory = Memory(replay_memory_size,output_experiences)
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
            if self.steps > 0 and ((self.steps % (self.batch_size + int(self.batch_size/2))) == 0):
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
            inputs = np.array(state, dtype=np.float64)
            inputs = np.expand_dims(inputs, 0)

            qvalues = self.learning_model.model(inputs)
            action = np.squeeze(np.argmax(qvalues, axis=-1))

        return int(action)


    def train_network(self):
        """Update online network weights."""
        batch = self.memory.sample(self.batch_size)
        validation_batch = self.memory.sample(int(self.batch_size/2))
        self.learning_model.train_step(batch, validation_batch)

