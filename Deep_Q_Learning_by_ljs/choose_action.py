import numpy as np


class ChooseAction:
    """
    用eps-greedy(模拟退火法)的方式选择动作
    """
    def __init__(self,
                 n_actions,
                 ini_eps=1,
                 final_eps=0.1,
                 start_frame=50000,
                 annealing_frame=1000000):

        self.n_actions = n_actions
        self.ini_eps = ini_eps
        self.final_eps = final_eps
        self.start_frame = start_frame
        self.annealing_frame = annealing_frame

        # eps-greedy 的 eps 以线性增长：
        self.slope = (self.final_eps - self.ini_eps) / self.annealing_frame
        self.ori = self.ini_eps - self.slope * self.start_frame

    def choose_action(self,
                      sess,
                      frame_number,
                      state,
                      eval_dqn):
        """
        参数解释
        :param sess:
        :param frame_number: 第几张图像
        :param state: 状态,是灰度图，二维数组(84, 84，4)
        :param eval_dqn: DQN网络，生成预测值
        :return: action（integer）
        """
        # 更新 eps
        if frame_number < self.start_frame:
            # 初始时始终随机选择
            eps = self.ini_eps
        if self.start_frame <= frame_number < self.start_frame + self.annealing_frame:
            # 从零开始线性递增
            eps = self.slope * frame_number + self.ori
        if frame_number >= self.start_frame + self.annealing_frame:
            # 最大不能超过
            eps = self.final_eps

        # 选择动作
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)
        else:
            # TODO: 有点问题.....和另一个类联动问题**（solved，but still may occur pro)
            return sess.run(eval_dqn.best_action,
                            feed_dict={eval_dqn.input: state[np.newaxis, ...]})[0]
