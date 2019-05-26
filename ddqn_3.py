# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
import cv2 as cv
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPool2D, Flatten
from keras.optimizers import RMSprop


class DQNAgent:
    def __init__(self, state_size, action_size, input_size):
        self.state_size = state_size
        self.action_size = action_size
        self.input_size = input_size
        self.memory = deque(maxlen=1000000) # 记忆库大小
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1 
        self.epsilon_decay = 0.9999977 # epsilon缩小速率，按照论文中所著的经过100万帧后缩小到0.1所得
        self.learning_rate = 0.00025
        # 此处的model与target_model就是论文中所述两种消除数据间相关性的方法之一
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    '''
    按照论文上面的模型构建进行构建，由三对卷积层与池化层，以及两个全连接层组成，优化器选择了RMSprop，
    损失函数为常用的差平方函数。
    '''
    def _build_model(self):
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
        rmsprop = RMSprop(lr=self.learning_rate, rho=0.95, decay=0.01)
        model.compile(loss='mse',
                      optimizer=rmsprop)
        return model

    '''
    在一定的帧数之后，将model中的weights拷贝到target_model中
    '''
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    '''
    将该步信息存入记忆库
    '''
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    '''
    依据epsilon进行行动：有epsilon的概率随机行动，此时epsilon不断减小；
    有1-epsilon的概率依据model预测值行动

    Return: 
        行动索引
        平均action_values的值
    '''
    def act(self, state, done = False):
        global action_aver
        if np.random.rand() <= self.epsilon:
            return [random.randrange(self.action_size), 1]
        # 对于图像进行下采样以及灰度处理
        state = cv.cvtColor(state, cv.COLOR_BGR2GRAY)
        state = cv.resize(state, (84, 84))
        state = state[np.newaxis, :, :, np.newaxis]
        act_values = self.model.predict(state)
        if done:
            action_aver = 0
        action_aver += act_values[0][np.argmax(act_values[0])]
        return [np.argmax(act_values[0]), action_aver]  # return action

    '''
    从记忆库中抽取batch_size大小的数据，投进model中进行学习，并且更新epsilon
    '''
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
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
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
    EPISODES = 50000 # 迭代轮数
    batch_size = 32
    a_ = 3 # 游戏生命初始值
    start = 1 # 设置此参数的目的是为了对应论文中所说的，当记忆库的大小达到50000时才开始学习，所以我们设定当此参数为0时才为第一轮训练
    action_aver = 0 # 每一局游戏的action_values的平均值
    target_time = 0 # 当target_time达到10000时更新target_model

    for e in range(EPISODES):
        state = env.reset()
        for time in range(1000000000000):
            env.render()
            action = agent.act(state)[0]
            next_state, reward, done, a = env.step(action)
            if a['ale.lives'] < a_:
                reward -= 10
            a_ = a['ale.lives']
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done and start == 0:
                # 当帧数达到10000时更新target_model
                if target_time == 10000:
                    agent.update_target_model()
                    target_time = 0
                print("episode: {}/{}, score: {}, e: {:.2}, action_aver: {}"
                    .format(e-e_start, EPISODES, time, agent.epsilon, agent.act(state, done)[1]/time))
                break
            if len(agent.memory) > 50000:
                agent.replay(batch_size)
                if start == 1:
                    e_start = e
                    start = 0
            target_time += 1
        # if e % 10 == 0:
        #     agent.save("./save/DemonAttack-ddqn.h5")
