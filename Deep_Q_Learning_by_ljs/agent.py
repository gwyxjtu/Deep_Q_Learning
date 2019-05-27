from game_env import GameEnv
from DQN_model import DeepQNetwork
from experience_replay import ReplayMemory
from choose_action import ChooseAction

import tensorflow as tf
import numpy as np


class Agent:

    def __init__(self,
                 env,
                 lr=0.00025,
                 batch_size=32,
                 gamma=0.99,
                 n_frames=3000000,
                 start_frame=50000,
                 anneal_frame=10**6,
                 update_freq=5000,
                 hist_len=4,
                 num_reward=200,
                 experience_size=10**6,
                 check_point_path=r'../checkpoints',
                 save_freq=1000,
                 no_ops=10,
                 eval_times=10,
                 restore=False
                 ):

        # 环境搭建
        self.env = GameEnv(env, hist_len)
        # 训练用到的参数
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.hist_len = hist_len
        self.experience_size = experience_size
        # 容易乱的参数.....（- & -）
        self.n_frames = n_frames  # 总共训练的帧数
        self.start_frame = start_frame  # 开始eps递增
        self.anneal_frame = anneal_frame  # eps递增到最大值
        self.update_freq = update_freq  # target_q_net_work的参数更新，每过update_freq个图，更新一次
        self.num_reward = num_reward  # 储存的reward的个数
        self.no_ops = no_ops  # 什么都不做的步数
        self.eval_times = eval_times  # 评估局数(默认十局)

        self.sess = tf.Session()
        self.save_freq = save_freq
        self.check_point_path = check_point_path

        n_actions = self.env.get_num_actions()
        self.action_chooser = ChooseAction(n_actions=n_actions,
                                           start_frame=self.start_frame,
                                           annealing_frame=self.anneal_frame)

        self.eval_dqn = DeepQNetwork(n_actions,
                                     batch_size=self.batch_size,
                                     lr=self.lr,
                                     name='eval_dqn')

        self.target_dqn = DeepQNetwork(n_actions,
                                       batch_size=self.batch_size,
                                       name='target_dqn')

        self.replay_memory = ReplayMemory(size=self.experience_size,
                                          batch_size=self.batch_size)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(name="model")
        self.restore = restore
        self.step = 0

    # 读取以往的检查点
    def _restore(self):
        if self.restore:
            print("Checkpoint Path: ", self.check_point_path)
            print("Checkpoint to be Restored:",
                  tf.train.latest_checkpoint(self.check_point_path))
            self.saver.restore(self.sess,
                               tf.train.latest_checkpoint(self.check_point_path))

    def _save(self):
        self.saver.save(self.sess, self.check_point_path + '/model.ckpt',
                        global_step=self.step)

    # 评估训练结果（论文要求）
    def _eval(self):
        print("Evaluating...")
        # 保存当前状态
        internal_state, system_state = self.env.clone_state()
        for eval_episodes in range(self.eval_times):
            self.env.reset()
            start_ep = True
            eval_reward = 0
            eval_rewards = []
            no_op = 0
            no_ops = np.random.randint(0, self.no_ops)
            while True:
                self.env.render()
                if start_ep:
                    no_op += 1
                    action = 0
                else:
                    action = self.action_chooser.choose_action(self.sess,
                                                               3000000,
                                                               self.env.state,
                                                               self.eval_dqn)
                _, _, reward, is_done, _ = self.env.step(action)
                eval_reward += reward
                if no_op == no_ops:
                    start_ep = False
                if is_done:
                    eval_rewards.append(eval_reward)
                    break
        avg_eval_rewards = np.mean(eval_rewards)
        print("Evaluation average reward: {}".format(avg_eval_rewards))
        # 恢复状态，继续训练
        self.env.restore_state(internal_state, system_state)

    def _learn(self):
        # 从记忆库中取出记忆
        states, score_lost, actions, rewards, new_states = self.replay_memory.get_minibatch()
        # [None 84 84 4] boolen [64] [64] [None 84 84 4]
        # 令下个状态回报最大 的 最优动作
        best_action = self.sess.run(self.eval_dqn.best_action,
                                    feed_dict={self.eval_dqn.input: new_states})

        # mini_batch每个状态时，下个状态的 reward
        target_q_val = self.sess.run(self.target_dqn.q_value,
                                     feed_dict={self.target_dqn.input: new_states})

        # [batch_size, best_action]---每个状态下的最优动作值
        target_q_val = target_q_val[range(self.batch_size), best_action]

        # 因为每次失分会导致整个游戏reset（），所以每失分一次，算作一个结束
        # 根据论文算法，每次结束都采用reward作为target_q
        target_q = rewards + self.gamma * target_q_val * (1 - score_lost)

        # 为了反向传播....妈妈咪...-_-
        pred_q = self.sess.run(self.eval_dqn.q_value,
                               feed_dict={self.eval_dqn.input: states})
        target_q_transition = pred_q.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = actions
        target_q_transition[batch_index, eval_act_index] = target_q
        target_q = target_q_transition
        # print(target_q - pred_q)
        loss, _ = self.sess.run([self.eval_dqn.loss,
                                 self.eval_dqn.update],
                                feed_dict={self.eval_dqn.input: states,
                                           self.eval_dqn.target_q: target_q})
        return pred_q, target_q, loss

    def _update_target_q_network(self):
        eval_vars = self.eval_dqn.variables()
        target_vars = self.target_dqn.variables()
        update_operation = []
        for eval_var, target_var in zip(eval_vars, target_vars):
            update_operation.append(tf.assign(target_var, tf.identity(eval_var)))
        copy_operation = tf.group(*update_operation)  # 返回一个操作而非数组
        self.sess.run(copy_operation)
        # 防止参数没有更新
        check_before = self.sess.run(eval_vars[0])
        check_after = self.sess.run(target_vars[0])
        assert (check_before == check_after).all(), "Parameters not updated"

    def train(self):
        self._restore()
        self.env.reset()
        start_ep = True
        no_op = 0
        no_ops = np.random.randint(0, self.no_ops)
        train_rewards = []
        train_reward = 0
        num_dones = 0
        # current_reward = 0
        # last_reward = 0

        print("Training for {} frames...".format(self.n_frames))

        # Training loop
        for elapsed_frame in range(0, self.n_frames):
            self.env.render()
            # 每过4帧换一次动作
            if elapsed_frame % 4 == 0:
                # TODO 矩阵维度问题...-_-(done)
                action = self.action_chooser.choose_action(self.sess,
                                                           elapsed_frame,
                                                           self.env.state,
                                                           self.eval_dqn)
            # print(action)
            score_lost, observation, reward, is_done, _ = self.env.step(action)
            train_reward += reward

            # 是否结束了一局（某一方得20分）
            if is_done:
                num_dones += 1
                if len(train_rewards) < self.num_reward:
                    train_rewards.append(train_reward)
                else:
                    train_rewards[num_dones % self.num_reward] = train_reward
                last_reward = sum(train_rewards) / len(train_rewards)
                current_reward = train_reward
                train_reward = 0
                print("Training Reward(average):", last_reward)
                print("Training Reward(current):", current_reward)

            self.replay_memory.add_experience(action,
                                              observation[:, :, 0],
                                              reward,
                                              score_lost)

            # start_frame之前是随机走动...只是为了丰富记忆库（林子大了什么鸟都有）
            if elapsed_frame > self.start_frame:
                _, _, loss = self._learn()
                print('loss:', loss)
                self.step += 1

            if (elapsed_frame % self.update_freq == 0
                    and elapsed_frame > self.start_frame):
                print("Updating target network params", end=', ')
                print("Current number of frames elapsed: {}".format(elapsed_frame))
                self._update_target_q_network()

            if (elapsed_frame % self.save_freq == 0
                    and elapsed_frame > self.start_frame):
                # 保存参数,检测训练结果
                self._save()
                self._eval()

        print("Training finished")
        self.env.close()


if __name__ == '__main__':
    agent = Agent('PongDeterministic-v4')
    agent.train()
