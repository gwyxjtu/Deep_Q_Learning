from game_env import  GameEnv
from experience_replay import ReplayMemory
import cv2
import tensorflow as tf
import numpy as np

VALID_NAMES = ("eval_dqn", "target_dqn")
# TODO: 1、向量维度问题。 2、loss设置问题


class DeepQNetwork:

    def __init__(self,
                 n_actions,
                 batch_size=32,
                 lr=0.0001,
                 hist_len=4,
                 frame_height=84,
                 frame_width=84,
                 name=None):
        """
        :param n_actions: 动作的数量
        :param batch_size: emmmmm....batch_size
        :param lr:  learning_rate
        :param hist_len:  将his_len个帧作为一个输入[84 * 84 * his_len]
        :param frame_height:  图像长
        :param frame_width:  图像宽
        :param name:  网络的名称(eval_dqn, target_dqn)
        """
        # 防止乱命名
        assert name in VALID_NAMES, "Name is not one of {}".format(VALID_NAMES)
        # 属性
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.lr = lr
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.hist_len = hist_len
        self.name = name
        # 方法
        self._build_model(self.name)
        if self.name == 'eval_dqn':
            self._build_optimizer()

    def variables(self):
        # 赋值时使用
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 scope=self.name)

    def _build_model(self, name):
        initializer = tf.variance_scaling_initializer(scale=2)
        self.input = tf.placeholder(shape=[None,
                                           self.frame_height,
                                           self.frame_width,
                                           self.hist_len],
                                    dtype=tf.float32)

        with tf.variable_scope(name):
            """
            根据论文，网络结构应该如下：
            　　第一个卷积层：卷积核大小为 8*8 ， 然后有 32个filter，步长为4；后面跟一个非线性激活函数；
　　            第二个卷积层：卷积核大小为 4*4， 然后有 64个filter，步长为2；后面跟一个非线性激活函数；
            　　第三个卷积层，64个filter，卷积核大小为 3*3，步长为1，后面也跟着一个 rectifier；
　　            最后的隐层是：全连接层，由512个rectifier units 构成；
            　　输出层是 一个全连接，每一个有效地动作对应一个输出。
            """
            # input [None 84 84 4]
            # output [None 20 20 32]
            self.conv1 = tf.layers.conv2d(inputs=self.input,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=4,
                                          kernel_initializer=initializer,
                                          padding='VALID',
                                          activation=tf.nn.relu,
                                          use_bias=True,
                                          bias_initializer=initializer,
                                          name='conv1')
            # input [None 20 20 32]
            # output [None 9 9 64]
            self.conv2 = tf.layers.conv2d(inputs=self.conv1,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=2,
                                          kernel_initializer=initializer,
                                          padding='VALID',
                                          activation=tf.nn.relu,
                                          use_bias=True,
                                          bias_initializer=initializer,
                                          name='conv2')
            # input[None 9 9 64]
            # output[None 7 7 64]
            self.conv3 = tf.layers.conv2d(inputs=self.conv2,
                                          filters=64,
                                          kernel_size=[3, 3],
                                          strides=1,
                                          kernel_initializer=initializer,
                                          padding='VALID',
                                          activation=tf.nn.relu,
                                          use_bias=True,
                                          bias_initializer=initializer,
                                          name='conv3')

            self.hidden = tf.reshape(self.conv3, [-1, 7 * 7 * 64])

            self.fc1 = tf.layers.dense(inputs=self.hidden,
                                       units=512,
                                       activation=tf.nn.relu,
                                       kernel_initializer=initializer,
                                       name='fc1')

            # output [None, 64, 6]
            self.q_value = tf.layers.dense(self.fc1,
                                           self.n_actions,
                                           kernel_initializer=initializer,
                                           name='q_value')

            self.best_action = tf.argmax(self.q_value, axis=1)  # 返回索引值

    def _build_optimizer(self):
        self.target_q = tf.placeholder(shape=[None, 6], dtype=tf.float32)
        self.loss = tf.reduce_mean(tf.square(self.q_value - self.target_q))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.99, momentum=0.95)
        self.update = self.optimizer.minimize(self.loss)


if __name__ == "__main__":
    env = GameEnv('PongDeterministic-v4', 4)
    dqn = DeepQNetwork(n_actions=6, hist_len=4, name="eval_dqn")
    env.reset()
    replay = ReplayMemory()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100000):
            dummy_q = np.zeros((32, 6))
            # dummy_q = [dummy_q]
            action = env.env.action_space.sample()
            terminal_life_lost, observation, reward, is_done, info = env.step(
                action)
            observation = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_NEAREST)
            replay.add_experience(action, observation,
                                  reward, terminal_life_lost)

            if i > 10000:
                states, actions, rewards, new_states, terminal_flags = replay.get_minibatch()
                loss, _, best_action = sess.run([dqn.loss, dqn.update, dqn.best_action],
                                                feed_dict={dqn.input: states,
                                                           dqn.target_q: dummy_q
                                                           })
                print(best_action)

