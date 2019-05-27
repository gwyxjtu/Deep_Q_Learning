import cv2 as cv
import gym
import numpy as np
from frame_process import FrameProcess


class GameEnv:

    def __init__(self, EnvName, hist_len):
        self.env = gym.make(EnvName)
        self.frame_process = FrameProcess()
        self.state = None  # 应该是[84 84 n]
        self.last_lives = 1
        self.hist_len = hist_len

        print("Environment Name: {}".format(EnvName))
        print("This environment has the {} actions : {}"
              .format(self.env.action_space.n,
                      self.get_action_meanings()))

    def get_num_actions(self):
        return self.env.action_space.n

    def get_action_meanings(self):
        return self.env.unwrapped.get_action_meanings()

    def get_action_random(self):
        return self.env.action_space.sample()

    def step(self, a):
        """
        对于pong这个游戏，reward只是每一帧的reward，并不会叠加，得分即为1，失分即为-1，
        游戏结束：当某一方达到20分，is_done的True or False指的是一个游戏是否结束
        函数中参数score_lost指的是是否失分
        :param a: 动作
        :return:
                score_lost：是否失分
                observation：（84， 84， 1）
                reward：0 +-1
                is_done：一个episode的结束
                info：emmm，没啥用（-_-）
        """
        observation, reward, is_done, info = self.env.step(a)
        if reward == 0:
            reward = 0
        elif reward > 0:
            reward = 1
        else:
            reward = -1
        observation = self.frame_process.process(observation)
        score_lost = True if reward == -1 else False
        if is_done:
            self.reset()

        # 去掉state[:,:,0],并在state上加上最新的状态，以此类推，反复跟新
        # 但始终保持[84 84 4]的结构
        new_state = np.append(self.state[:, :, 1:], observation, axis=2)
        self.state = new_state
        return score_lost, observation, reward, is_done, info

    def render(self):
        return self.env.render()

    def reset(self):
        observation = self.env.reset()
        observation = self.frame_process.process(observation)
        # [84 84 1]
        self.state = np.repeat(observation, self.hist_len, axis=2)
        # [84 84 4]
        # 为了作为choose_action的输入，
        # 由于每次输入是四帧的总和，若想预测下一个动作，那么第一二三次会维度不能缺失，因此第一次重复了四次
        # 每次reset()，state都应该更新自己

    def close(self):
        self.env.close()

    #  评估模型时要中断训练，应该为后续的训练储存当前的状态参数
    def clone_state(self):
        return self.state, self.env.unwrapped.clone_full_state()

    def restore_state(self, state, saved_state):
        self.state = state
        self.env.unwrapped.restore_full_state(saved_state)


if __name__ == "__main__":
    env = GameEnv('PongDeterministic-v4', 4)
    env.reset()
    for _ in range(1, 10000):
        env.render()
        action = env.env.action_space.sample()
        # is_done indicates the end of an episode
        terminal_score_lost, observation, reward, is_done, info = env.step(
            action)
        cv.waitKey(5)
        if not terminal_score_lost:
            print(reward)