---
### 序
DQN有很多优势，相比于Q-Learning：DQN使用了卷积神经网络来逼近行为值函数，还使用了target Q network来更新target，还有就是使用了经验回放Experience replay。
由于在强化学习中，得到的观测数据是有序的，步步相关的，用这样的数据去更新神经网络的参数会有问题。在有监督学习中，数据之间都是独立的。因此DQN中使用经验回放，即用一个Memory来存储经历过的数据，每次更新参数的时候从Memory中抽取一部分的数据来用于更新，以此来打破数据间的关联。
当然DQN的数学思想以及相较于以往的RL算法的优势不止这些，这里都不一一展开，我在学习DQN的过程中参考了以下几篇博客，受益良多：
[参考链接-1](https://wanjun0511.github.io/2017/11/05/DQN/)
[参考链接-2](https://www.jianshu.com/p/b92dac7a4225)
[参考链接-3](https://www.cnblogs.com/wangxiaocvpr/p/5580963.html)
这里主要讲一下我的**编程思路，代码内容，复现论文的程度以及运行流程**
----

### 编程思路：
#### 整体思路
根据参阅论文与其他相关学习资料，了解到，一个DQN，其实就是一个agent在玩游戏，在这个过程中，他需要执行众多操作，其中包括：根据当前状态选择动作，执行动作并返回值（obs，rewards..），添加至记忆库等。事实上，神经网络是没有显式的参与到决策中去的，只是，每帧时都自我训练一次，并且在动作选择时提供reward

在玩游戏的过程中，一个agent应该按以下逻辑顺序来进行学习：
1. 选一个动作。在每局最开始时，有随机的几帧或十几帧保持不动，而后每过4帧，更新一次动作（no_operation仅在evaluation部分才使用，在平时训练时不使用）
2. 得到动作返回值（obs，reward，states...）
3. 储存记忆 
4. 检查是否完成一局（若完成一局，则总结分数，更新以及重置参数）
5. 学习(通过minibatch，更新网络参数)
6. **(if，以一定频率地)** 更新target_Q参数
7. **(if，以一定频率地)** 保存参数与评估当前模型

#### 细化各部分
那么相应的将模型细化为几个类，一来，分工明确，思路比较清楚；二来，方便debug与局部更改。
每个类功能与方法如下：
1. **图像处理类**：用来读取游戏界面，并将游戏图像转化为灰度图，并resize成[84, 84, 1]的结构
```
class FrameProcess:

    def __init__(self):
        pass

    def process(self, frame):
    '''
    实现功能
    基本的图像读取，
    三通道转灰度，
    更改图像大小维度
    '''
        pass
        return frame[:, :, np.newaxis]
```
2. **游戏环境类**:创建环境，重写step函数，使得其输出满足编程需要，为gym的多个接口提供相应的类方法。同时，在训练过程中，加入了evaluation（论文中的evaluation应该是独立的，这里为了方便观察训练效果，我加进了训练过程），因此需要中断训练保存当前的所有状态，以便评估结束后继续
```
class GameEnv:

    def __init__(self):
        pass

    def get_num_actions(self):  # 返回动作个数
        return self.env.action_space.n

    def get_action_meanings(self):  # 返回动作意义
        return self.env.unwrapped.get_action_meanings()

    def get_action_random(self):  # 随机选择动作
        return self.env.action_space.sample()

    def step(self, a):
    '''
     :return:
     score_lost：是否失分
     observation：（84， 84， 1）
     reward：0 or +-1
     is_done：一个episode的结束
    '''
        pass
        return score_lost, observation, reward, is_done, info

    def render(self):
        return self.env.render()

    def reset(self): # 重置
        pass

    def close(self): # 关闭环境
        self.env.close()

    #  评估模型时要中断训练，应该为后续的训练储存当前的状态参数
    def clone_state(self):
        return self.state, self.env.unwrapped.clone_full_state()

    def restore_state(self, state, saved_state):
        self.state = state
        self.env.unwrapped.restore_full_state(saved_state)
```
3. **experience_replay类**：实现论文的预处理部分，储存记忆，以及为神经网络提供数据（get_mini_batch）。另外由于论文中将4帧打包输入，这里在抽取时要注意打包的四帧是否连续，有意义，是否跨局或者跨回合。
```
class ReplayMemory:
    """
    在Replaymemory中实现论文中的预处理
    储存frames，是否失分（丢命），rewards，actions
    """
    def __init__(self):
        pass
        
    def add_experience(self, action, frame, reward, terminal):
        # 将新的状态添加到记忆库中
        pass

    #  从当前库中抽取记忆
    def _get_state(self, index):
        pass
        return state  # [4 84 84]

    # 随机抽取
    def _get_valid_indices(self):
       # 保证随机抽取的过程中，抽取的四帧是连续的，有意义的
       # 返回或生成一个有效的序列，使得序列中的每帧以及该帧的前3帧满足连续条件
       pass

    # 抽取记忆
    def get_minibatch(self):
      pass
      # 返回：当前状态，是否失分，reward， action， next_state
      return *  # [32 84 84 4], [32], [32], [32], [32 84 84 4]
```
4. **choose_action类**:通过eps-greedy选择动作.根据论文中的参数，eps是线性递减的，有1经过一百万帧递减到0.1
```
class ChooseAction:
    """
    用eps-greedy(模拟退火法)的方式选择动作
    其中eps以线性递减，1000000帧时由1递减到0.1
    """
    def __init__(self):
        pass

    def choose_action(self): # 选择动作
        pass
        return action
```
5. **神经网络类**：eval_dqn与target_dqn公用的网络模型与优化器
   1. 根据论文，网络结构应该如下：
      * 第一个卷积层：卷积核大小为 8*8 ， 然后有 32个filter，步长为4；后面跟一个非线性激活函数；
      * 第二个卷积层：卷积核大小为 4*4， 然后有 64个filter，步长为2；后面跟一个非线性激活函数；
      * 第三个卷积层，64个filter，卷积核大小为 3*3，步长为1，后面也跟着一个 rectifier；
      * 最后的隐层是：全连接层，由512个rectifier units 构成；
      * 输出层是 一个全连接，每一个有效地动作对应一个输出。
    2. 优化器采用 RMSProp， loss为 tf.reduce_mean(tf.square(self.q_value - self.target_q))
```
VALID_NAMES = ("eval_dqn", "target_dqn") # 防止乱命名

class DeepQNetwork:

    def __init__(self)
        # 初始化参数，创建网络，建立优化器
        pass

    def variables(self):
        # 赋值时使用
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 scope=self.name)

    def _build_model(self, name):
        # 搭建网络，返回预测值
        pass
        return q_value, best_action

    def _build_optimizer(self):
       # 搭建优化器，定义loss并更新网络
       pass
```
6. **agent类**：
   * 学习：调用记忆库更新网络参数
   * 更新target_net参数、保存以及读取参数、
   * 评估模型：初始时，随机几帧保持不动，eps为0.1，进行模拟
   * 自我训练：选择动作、得到反馈（obs，action,reward...），储存新状态，调用类方法更新保存参数
```
class Agent:

    def __init__(self):
        # 环境搭建
        # 定义用到的参数
        # 创建动作选择对象，经验回放对象，两个网络对象
        pass
        
    def _restore(self):  # 读取以往的检查点
        pass

    def _save(self):  # 保存训练参数
        pass

    def _eval(self):  # 评估训练结果（论文要求）
        pass

    def _learn(self):  # 训练神经网络
        pass
        return pred_q, target_q, loss

    def _update_target_q_network(self):  # 更新target_q_net网络参数
        pass

    def train(self):
        '''
        agent玩游戏的过程
        选择动作
        得到动作返回值（obs，reward，states...）
        储存记忆 
        检查是否完成一局（若完成一局，则总结分数，更新以及重置参数）
        学习(通过minibatch，更新网络参数)
        (if，以一定频率地) 更新target_Q参数
        (if，以一定频率地) 保存参数与评估当前模型
        '''
        pass
```

---

### 编程过程中遇到的问题：
#### 一、 预处理部分
预处理主要有两步：
1. 比较当前帧与前一帧的对应像素值的大小，并取二者中的最大值，以此来消除图像中存在的闪烁问题
2. 取出图像的Y通道，并resize成84 * 84大小。将当前帧与前三帧打包作为输入，输入维度为[None 84 84 4]

**实现思路：**
在实际处理时，我将所有的输入图像都转化为了灰度图，在experience_replay类中的add_experience方法中，储存图像（状态）时通过np.maximumm（）来实现预处理一
在实现第二步时，抽取过程就不能是完全随机的。首先，由于记忆库的填充过程是新的覆盖旧的，就有可能导致取出的四帧画面属于不同的两局游戏，由于每局游戏结束时，画面会重置，那么这四帧之间就没有任何关系，上下文没有联系，就失去了打包的意义；其次由于每次失分时，游戏画面都会重置，那么提取的四帧中应该隶属同一回合（或者同一条命），否则原因同上。
#### 二、loss的反向传播问题
在编写代码的过程中，最终计算得到的target_q的维度实际上是[batch_size， 1]，而反向传播所要求的维度是[batch_size, n_actions]，那么为了能够反向传播，就要更改target_q的维度，并且使得二者在正确的位置上作差（[size, best_action]），而其余位置是0

**实现方法：**
```
        target_q = rewards + self.gamma * target_q_val * (1 - score_lost)
        pred_q = self.sess.run(self.eval_dqn.q_value,
                               feed_dict={self.eval_dqn.input: states})
        target_q_transition = pred_q.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = actions
        target_q_transition[batch_index, eval_act_index] = target_q
        target_q = target_q_transition
        loss, _ = self.sess.run([self.eval_dqn.loss,
                                 self.eval_dqn.update],
                                feed_dict={self.eval_dqn.input: states,
                                           self.eval_dqn.target_q: target_q})
```
#### 三、类之间、函数之间向量维度问题
很多时候，程序报错都是因为维度错误，但大多不是严重的维度不匹配而是那种[84, 84]和[84, 84, 1]的区别，导致程序难以运行

**解决办法：**
emmm-_-,一行一行的改，先磨合类内的维度不匹配问题，再更改类之间的维度不匹配问题

---

### 论文复现的完整程度
基本已经完全复现，包括**预处理，跳帧技术，评估，每局初始时的no_operation，记忆库，eps-greedy，double net，reward的规范化设置，兼容多个游戏**等。所选的各个参数也都和论文保持相一致。
但由于..电脑性能问题，无法得出和论文相同的结果（准确的说是跑不完...，基本上20万帧我的电脑要跑12个小时左右，跑完300万帧可能要花上一周。所以对于过拟合、收敛于局部最优的问题，我没能很好的处理）

---

### 代码运行流程
1. 训练可以在agent类中的main函数下进行，如果想更换游戏，只需更改Agent('Game_name')即可
2. 各个类的具体参数可以参考__ini__函数，命名都尽量易于理解，和论文保持一致，比较琐碎的参数我都加上了注释
3. 程序会自动保存训练参数，默认每隔一千帧保存一次，会在程序的上级目录下生成一个新的文件夹（checkpoints），参数记录都在其中
4. 如果想继续或者读取先前的检查点，只需把Agent的restore参数设为True（默认为False）
---
### 代码不足以及之后改进的地方
虽然网络，变量都有命名但是由于一些问题（考试有点多..自身能力不足等等），最后我没有做出网络可视化，summary也没有做，如果日后有时间，我希望能够补上做完