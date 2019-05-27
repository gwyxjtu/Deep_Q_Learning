import random
import cv2 as cv
import numpy as np
from game_env import GameEnv


class ReplayMemory:
    """
    在Replaymemory中实现论文中的预处理
    """
    def __init__(self,
                 size=1000000,
                 frame_height=84,
                 frame_width=84,
                 hist_len=4,
                 batch_size=32):
        """
        :param size:  记忆库大小
        :param frame_height:  图像高度
        :param frame_width:  图像宽度
        :param hist_len:  向前组合his_len个图像作为网络的输入
        :param batch_size: emmm....batch_size
        """
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.hist_len = hist_len
        self.batch_size = batch_size

        self.populate = 0  # 当前的记忆个数(实际值)
        self.current = 0  # 当前的记忆个数（索引值）

        # 预先分配记忆(size)
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.int32)
        self.frames = np.empty(
            (self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

        # 预先分配记忆(batch_size)
        self.states = np.empty((self.batch_size,
                               self.hist_len,
                               self.frame_height,
                               self.frame_width), dtype=np.uint8)

        self.new_states = np.empty((self.batch_size,
                                    self.hist_len,
                                    self.frame_height,
                                    self.frame_width), dtype=np.uint8)

        # 用于训练时的随机抽取
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experience(self, action, frame, reward, terminal):

        assert frame.shape == (
            self.frame_height, self.frame_width), "Incorrect Dimension"

        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        if self.populate == 1 or self.populate == 0:
            self.frames[self.current, ...] = frame
        else:
            self.frames[self.current, ...] = np.maximum(frame, self.frames[self.current - 1, ...])
        self.populate = max(self.populate, self.current+1)
        self.current = (self.current + 1) % self.size

    #  从当前库中抽取记忆
    def _get_state(self, index):
        # 这里的index是索引值，index + 1 为图片实际排序：取出思路是：
        # ex：
        # index = 3,则取出：0,1,2,3；即：[0 : 4]

        assert self.populate > 0, "Replay memory cannot be empty"
        index %= self.populate
        # 按照论文中的思路，将四帧整合在一起,作为输入
        if index >= self.hist_len - 1:
            return self.frames[index - self.hist_len + 1:index + 1, ...]
        else:
            # 小于3时，从末尾索引
            """
            可能会有的问题：列表未填充！！！
            """
            return self.frames[index - self.hist_len + 1:-1] + self.frame[0:index + 1, ...]

    # 随机抽取
    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                # 先随即一个值
                index = random.randint(
                    self.hist_len, self.populate - 1)
                # index 要比his_len长，避免一些奇奇怪怪的问题（从末尾索引），编程方便
                if index < self.hist_len:
                    continue
                # 索引的时候不能有时间断隔，因为记忆库是覆盖填充，而取出的4帧（his_len）必须是连续的
                if self.current <= index <= self.current + self.hist_len:
                    continue
                # 索引的图片应该是同一局游戏中的，不能是两个回合，否则图片变化会很大，不连续....预处理烦死了...我还有考试呢0&0
                if self.terminal_flags[(index - self.hist_len):index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self):
        if self.populate < self.hist_len:
            raise ValueError("Not enough memory to retrieve a mini-batch")
        self._get_valid_indices()

        for i, idx in enumerate(self.indices):
            self.states[i, ...] = self._get_state(idx - 1)
            self.new_states[i, ...] = self._get_state(idx)

        # 原frame结构为[size, channel, height, weight],要变成[size height, weight, channel]来满足TensorFlow的输入条件
        return (np.transpose(self.states, axes=[0, 2, 3, 1]),
                self.terminal_flags[self.indices],
                self.actions[self.indices],
                self.rewards[self.indices],
                np.transpose(self.new_states, axes=[0, 2, 3, 1]))


if __name__ == '__main__':
    env = GameEnv("PongDeterministic-v4", 4)
    frame = env.reset()
    replay_exp = ReplayMemory(1000)
    for i in range(10000):
        env.render()
        action = env.get_action_random()
        terminal_score_lost, observation, reward, is_done, info = env.step(
            action)
        # print(observation[:, :, 0].shape)
        observation_gray = observation[:, :, 0]
        replay_exp.add_experience(
            action, observation_gray, reward, terminal_score_lost)
        if i >= 120:
            mini_batch = replay_exp.get_minibatch()
            # emmm 高维列表，处理时要注意
            # mini_batch[0], mini_batch[1], mini_batch[2], mini_batch[3] 分别为 s, score_lost, a, r, s_
            print(mini_batch[3].shape)
