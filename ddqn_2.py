# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
import cv2 as cv
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPool2D, Flatten, Activation
from keras.layers import DepthwiseConv2D
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf

EPISODES = 5000

class DQNAgent:
    def __init__(self, state_size, action_size, input_size):
        self.state_size = state_size
        self.action_size = action_size
        self.input_size = input_size
        self.memory = deque(maxlen=1000000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.learning_rate = 0.00025
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    """Huber loss for Q Learning

    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    '''
    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))
    '''

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Convolution2D(
            batch_input_shape=(None, self.input_size[0], self.input_size[1], self.input_size[2]),
            filters=32,
            kernel_size=8,
            strides=4,
            padding='same',
            data_format='channels_last',
            activation='relu'
        ))
        model.add(MaxPool2D(
            pool_size=2,
            strides=2,
            padding='same',
            data_format='channels_last'
        ))
        model.add(Convolution2D(64, 4, strides=2, padding='same', data_format='channels_last', activation='relu'))
        model.add(MaxPool2D(2, 2, 'same', data_format='channels_last'))
        model.add(Convolution2D(64, 3, strides=1, padding='same', data_format='channels_last', activation='relu'))
        model.add(MaxPool2D(5, 5, 'same', data_format='channels_first'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = cv.cvtColor(state, cv.COLOR_BGR2GRAY)
        state = cv.resize(state, (84, 84))
        state = state[np.newaxis, :, :, np.newaxis]
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = cv.cvtColor(state, cv.COLOR_BGR2GRAY)
            state = cv.resize(state, (84, 84))
            state = state[np.newaxis, :, :, np.newaxis]
            next_state = cv.cvtColor(next_state, cv.COLOR_BGR2GRAY)
            next_state = cv.resize(next_state, (84, 84))
            next_state = next_state[np.newaxis, :, :, np.newaxis]
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                # target[0][action] = reward + self.gamma * np.amax(t)
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
                # print(t[np.argmax(a)])
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('SpaceInvaders-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    input_size = [84, 84, 1]
    agent = DQNAgent(state_size, action_size, input_size)
    # agent.load("./save/SpaceInvaders-ddqn.h5")
    done = False
    batch_size = 32
    a_ = 3
    start = 1

    for e in range(EPISODES):
        state = env.reset()
        for time in range(1000000000000):
            env.render()
            action = agent.act(state)
            next_state, reward, done, a = env.step(action)
            if reward>0:
                reward = 1
            if a['ale.lives'] < a_:
                reward -= 1
            a_ = a['ale.lives']
            print(reward)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done and start == 0:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                    .format(e-e_start, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > 50000:
                agent.replay(batch_size)
                if start == 1:
                    e_start = e
                    start = 0
        # if e % 10 == 0:
        #     agent.save("./save/DemonAttack-ddqn.h5")
