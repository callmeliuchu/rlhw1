"""
最小实现的 Actor-Critic (AC) - 完全独立版本
集成了 BC 框架的数据加载、评估等功能
AC 的核心方法（update_critic, update_actor, bc）留空，供用户自己实现
"""

import pickle
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Independent

# 尝试导入 gymnasium，如果失败则使用 gym
# 注意：linter 可能显示警告，但这是正常的（运行时动态导入）
try:
    import gymnasium as gym  # type: ignore
except ImportError:
    try:
        import gym  # type: ignore
    except ImportError:
        raise ImportError("Please install either 'gymnasium' or 'gym' package")


# ============================================================================
# 1. 超参数配置（写死）
# ============================================================================
class Config:
    # 环境配置
    ENV_NAME = "Ant-v4"
    
    # 网络结构
    HIDDEN_DIM = 256         # Actor 和 Critic 的隐藏层维度
    NUM_CRITICS = 2          # Critic 网络数量（用于 Double Q-learning）
    STD = 0.1                # Actor 输出的标准差
    
    # 训练配置
    LEARNING_RATE = 1e-3     # 学习率
    BATCH_SIZE = 256         # 训练 batch size
    N_EPOCHS = 100           # 训练轮数（用于 BC）
    
    # AC 特定配置
    CRITIC_TARGET_TAU = 0.01  # Critic target 网络软更新系数
    STDDEV_CLIP = 0.3         # 标准差裁剪范围
    
    # 数据路径
    EXPERT_DATA_PATH = "cs224r/expert_data/expert_data_Ant-v4.pkl"
    
    # 设备
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    
    # 其他
    USE_TB = False  # 是否使用 tensorboard（暂时不用）


# ============================================================================
# 2. 辅助工具函数（utils 模块）
# ============================================================================
def weight_init(m):
    """初始化网络权重"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def to_torch(batch, device):
    """
    将 batch 数据转换为 torch tensor
    
    Args:
        batch: tuple of (observation, action, reward, discount, next_observation)
        device: torch device
    
    Returns:
        tuple of torch tensors
    """
    obs, action, reward, discount, next_obs = batch
    obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
    action = torch.as_tensor(action, device=device, dtype=torch.float32)
    reward = torch.as_tensor(reward, device=device, dtype=torch.float32)
    discount = torch.as_tensor(discount, device=device, dtype=torch.float32)
    next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
    return obs, action, reward, discount, next_obs


class TruncatedNormal:
    """
    截断正态分布（用于 Actor 输出）
    将动作限制在 [-1, 1] 范围内（通过 tanh）
    """
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std
        self.dist = Normal(mu, std)
    
    def sample(self, clip=None):
        """
        采样动作
        
        Args:
            clip: 裁剪范围，如果为 None 则不裁剪
        
        Returns:
            action: [batch_size, action_dim]
        """
        action = self.dist.sample()
        # 通过 tanh 将动作限制在 [-1, 1]
        # action = torch.tanh(action)
        # if clip is not None:
        #     action = torch.clamp(action, -clip, clip)
        return action
    
    @property
    def mean(self):
        """返回分布的均值（经过 tanh）"""
        return torch.tanh(self.mu)
    
    def log_prob(self, action):
        """
        计算 log 概率
        
        Args:
            action: [batch_size, action_dim]，应该在 [-1, 1] 范围内
        
        Returns:
            log_prob: [batch_size]
        """
        # 将 action 从 [-1, 1] 映射回未截断空间
        # 使用 atanh 的逆变换
        eps = 1e-6
        action = torch.clamp(action, -1 + eps, 1 - eps)
        u = 0.5 * torch.log((1 + action) / (1 - action))
        
        # 计算 log_prob，需要加上 tanh 变换的雅可比行列式
        log_prob = self.dist.log_prob(u)
        # tanh 的雅可比行列式：log(1 - tanh^2(u)) = log(1 - action^2)
        log_prob -= torch.log(1 - action.pow(2) + eps)
        
        # 对动作维度求和
        return log_prob.sum(dim=-1)


# ============================================================================
# 3. Actor 网络
# ============================================================================
class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim, std=0.1):
        super().__init__()

        self.std = std
        self.policy = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0])
        )

        self.apply(weight_init)

    def forward(self, obs):
        mu = self.policy(obs)
        # mu = torch.tanh(mu)
        std = torch.ones_like(mu) * self.std

        dist = TruncatedNormal(mu, std)
        return dist


# ============================================================================
# 4. Critic 网络
# ============================================================================
class Critic(nn.Module):
    def __init__(self, obs_shape, action_shape, num_critics, hidden_dim):
        super().__init__()

        self.critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_shape[0] + action_shape[0], hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(num_critics)
        ])

        self.apply(weight_init)

    def forward(self, obs, action):
        h_action = torch.cat([obs, action], dim=-1)
        return [critic(h_action) for critic in self.critics]


# ============================================================================
# 5. AC Agent
# ============================================================================
class ACAgent:
    def __init__(self, obs_shape, action_shape, device, lr,
                 hidden_dim, num_critics, critic_target_tau, stddev_clip, use_tb):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.stddev_clip = stddev_clip

        # models
        self.actor = Actor(obs_shape, action_shape, hidden_dim).to(device)

        self.critic = Critic(obs_shape, action_shape, num_critics, hidden_dim).to(device)
        self.critic_target = Critic(obs_shape, action_shape, num_critics, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, eval_mode):
        # 确保使用 float32（MPS 不支持 float64）
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        # 推理时不需要梯度
        with torch.no_grad():
            dist = self.actor(obs.unsqueeze(0))
            if eval_mode:
                action = dist.mean
            else:
                action = dist.sample(clip=None)
        # 分离计算图并转换为 numpy
        return action.cpu().detach().numpy()[0]

    def update_critic(self, replay_iter):
        '''
        This function updates the critic and target critic parameters.

        Args:

        replay_iter:
            An iterable that produces batches of tuples
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, D] of states
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, D] of states

        Returns:

        metrics: dictionary of relevant metrics to be logged. Add any metrics
                 that you find helpful to log for debugging, such as the critic
                 loss, or the mean Bellman targets.
        '''

        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = to_torch(batch, self.device)

        ### YOUR CODE HERE ###


        #####################
        return metrics

    def update_actor(self, replay_iter):
        '''
        This function updates the policy parameters.

        Args:

        replay_iter:
            An iterable that produces batches of tuples
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, D] of states
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, D] of states

        Returns:

        metrics: dictionary of relevant metrics to be logged. Add any metrics
                 that you find helpful to log for debugging, such as the actor
                 loss.
        '''
        metrics = dict()

        batch = next(replay_iter)
        obs, _, _, _, _ = to_torch(batch, self.device)

        ### YOUR CODE HERE ###


        return metrics

    def bc(self, replay_iter):
        '''
        This function updates the policy with end-to-end
        behavior cloning

        Args:

        replay_iter:
            An iterable that produces batches of tuples
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, D] of states
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, D] of states

        Returns:

        metrics: dictionary of relevant metrics to be logged. Add any metrics
                 that you find helpful to log for debugging, such as the loss.
        '''

        metrics = dict()

        batch = next(replay_iter)
        obs, action, _, _, _ = to_torch(batch, self.device)

        ### YOUR CODE HERE ###
        dist = self.actor(obs)
        log_prob = dist.log_prob(action)
        loss = -log_prob.mean()
        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        metrics['loss'] = loss.item()
        return metrics


# ============================================================================
# 6. 数据加载：从专家数据文件加载
# ============================================================================
def load_expert_data(filepath):
    """
    加载专家数据
    
    Args:
        filepath: 专家数据 pickle 文件路径
    
    Returns:
        observations: [N, obs_dim] numpy array
        actions: [N, ac_dim] numpy array
    """
    print(f"Loading expert data from {filepath}...")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # 处理不同的数据格式
    if isinstance(data, dict):
        # 格式 1: dict with 'observations' and 'actions'
        obs = data.get('observations', data.get('obs'))
        acs = data.get('actions', data.get('acs'))
        
        if obs is None or acs is None:
            raise ValueError("Expert data must contain 'observations' and 'actions'")
        
        # 如果是列表，转换为数组
        if isinstance(obs, list):
            obs = np.array(obs)
        if isinstance(acs, list):
            acs = np.array(acs)
        
        # 如果是多条轨迹，展平
        if len(obs.shape) == 3:  # [n_trajs, T, obs_dim]
            obs = obs.reshape(-1, obs.shape[-1])
        if len(acs.shape) == 3:  # [n_trajs, T, ac_dim]
            acs = acs.reshape(-1, acs.shape[-1])
        
        # 确保是 2D
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        if len(acs.shape) == 1:
            acs = acs.reshape(1, -1)
            
    elif isinstance(data, list):
        # 格式 2: list of trajectories
        obs_list = []
        acs_list = []
        for traj in data:
            if isinstance(traj, dict):
                obs_list.append(traj['observation'])
                acs_list.append(traj['action'])
            else:
                obs_list.append(traj[0])
                acs_list.append(traj[1])
        
        obs = np.concatenate(obs_list, axis=0)
        acs = np.concatenate(acs_list, axis=0)
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")
    
    print(f"Loaded {len(obs)} expert (state, action) pairs")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action shape: {acs.shape}")
    
    return obs, acs


# ============================================================================
# 7. 简单的 Replay Buffer（用于 AC 训练）
# ============================================================================
class ReplayBuffer:
    """
    简单的经验回放缓冲区
    用于存储和采样 (obs, action, reward, discount, next_obs) 元组
    """
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, obs, action, reward, discount, next_obs):
        """添加一个经验到缓冲区"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (obs, action, reward, discount, next_obs)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """随机采样一个 batch"""
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, discount, next_obs = zip(*batch)
        return (
            np.array(obs),
            np.array(action),
            np.array(reward),
            np.array(discount),
            np.array(next_obs)
        )
    
    def __len__(self):
        return len(self.buffer)


def create_replay_iter(replay_buffer, batch_size):
    """
    创建一个无限迭代器，用于从 replay buffer 中采样
    
    Args:
        replay_buffer: ReplayBuffer 实例
        batch_size: batch 大小
    
    Yields:
        batch: tuple of (obs, action, reward, discount, next_obs)
    """
    while True:
        yield replay_buffer.sample(batch_size)


# ============================================================================
# 8. 评估/演示函数
# ============================================================================
def evaluate_policy(agent, env, n_episodes=5, max_steps=1000, render=False):
    """
    评估训练好的策略
    
    Args:
        agent: ACAgent 实例
        env: Gym 环境
        n_episodes: 评估的 episode 数量
        max_steps: 每个 episode 最大步数
        render: 是否渲染
    
    Returns:
        returns: 每个 episode 的回报列表
        episode_lengths: 每个 episode 的长度列表
    """
    print(f"\n{'='*60}")
    print("Evaluating Policy")
    print(f"{'='*60}")
    
    agent.train(False)  # 设置为评估模式
    returns = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        # 重置环境
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        
        total_reward = 0
        steps = 0
        
        while steps < max_steps:
            # 获取动作（eval_mode=True 使用均值）
            action = agent.act(obs, eval_mode=True)
            
            # 执行动作（兼容 gym 和 gymnasium）
            step_result = env.step(action)
            if len(step_result) == 5:  # gymnasium 返回 5 个值
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:  # gym 返回 4 个值
                next_obs, reward, done, info = step_result
            
            # 渲染
            if render:
                env.render()
            
            total_reward += reward
            steps += 1
            obs = next_obs
            
            if done:
                break
        
        returns.append(total_reward)
        episode_lengths.append(steps)
        
        print(f"Episode {episode+1}: Return = {total_reward:.2f}, Length = {steps}")
    
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    avg_length = np.mean(episode_lengths)
    
    print(f"\n{'='*60}")
    print(f"Average Return: {avg_return:.2f} ± {std_return:.2f}")
    print(f"Average Episode Length: {avg_length:.1f}")
    print(f"{'='*60}\n")
    
    return returns, episode_lengths


# ============================================================================
# 9. 主函数：完整的 AC 流程框架
# ============================================================================
def main():
    """
    完整的 Actor-Critic 流程框架：
    1. 创建环境
    2. 加载专家数据（可选，用于 BC 预训练）
    3. 创建 AC Agent
    4. 训练策略（需要用户实现 update_critic, update_actor, bc）
    5. 评估策略
    """
    print("="*60)
    print("Actor-Critic - Minimal Standalone Implementation")
    print("="*60)
    
    # ========================================================================
    # 步骤 1: 创建环境
    # ========================================================================
    print("\n[Step 1] Creating environment...")
    env_kwargs = {"render_mode": "rgb_array"}
    if Config.ENV_NAME == "Ant-v4":
        env_kwargs["use_contact_forces"] = True
    
    try:
        env = gym.make(Config.ENV_NAME, **env_kwargs)
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR: Failed to create environment")
        print(f"{'='*60}")
        print(f"Error message: {str(e)}")
        print(f"\nPossible solutions:")
        print("1. Make sure MuJoCo is installed correctly")
        print("2. Check if the environment name is correct")
        print("3. Try installing gymnasium: pip install gymnasium")
        print("4. For MuJoCo environments, you may need to install mujoco-py")
        print(f"{'='*60}\n")
        raise
    
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    
    print(f"Environment: {Config.ENV_NAME}")
    print(f"Observation shape: {obs_shape}")
    print(f"Action shape: {action_shape}")
    
    # ========================================================================
    # 步骤 2: 创建 AC Agent
    # ========================================================================
    print("\n[Step 2] Creating AC Agent...")
    agent = ACAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=Config.DEVICE,
        lr=Config.LEARNING_RATE,
        hidden_dim=Config.HIDDEN_DIM,
        num_critics=Config.NUM_CRITICS,
        critic_target_tau=Config.CRITIC_TARGET_TAU,
        stddev_clip=Config.STDDEV_CLIP,
        use_tb=Config.USE_TB
    )
    
    # 打印网络结构
    actor_params = sum(p.numel() for p in agent.actor.parameters())
    critic_params = sum(p.numel() for p in agent.critic.parameters())
    print(f"AC Agent created:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Actor parameters: {actor_params:,}")
    print(f"  Critic parameters: {critic_params:,} (x{Config.NUM_CRITICS})")
    
    # ========================================================================
    # 步骤 3: 加载专家数据（可选，用于 BC 预训练）
    # ========================================================================
    print("\n[Step 3] Loading expert data (for BC pre-training)...")
    try:
        observations, actions = load_expert_data(Config.EXPERT_DATA_PATH)
        
        # 创建 replay buffer 并填充专家数据
        replay_buffer = ReplayBuffer()
        # 假设 discount = 0.99（可以根据实际情况调整）
        discount = 0.99
        for i in range(len(observations) - 1):
            replay_buffer.push(
                obs=observations[i],
                action=actions[i],
                reward=0.0,  # 专家数据中可能没有 reward，设为 0
                discount=discount,
                next_obs=observations[i + 1]
            )
        print(f"Replay buffer filled with {len(replay_buffer)} expert transitions")
        
    except FileNotFoundError:
        print(f"WARNING: Expert data file not found: {Config.EXPERT_DATA_PATH}")
        print("Creating empty replay buffer. You'll need to collect data first.")
        replay_buffer = ReplayBuffer()
    
    # ========================================================================
    # 步骤 4: 训练策略（示例：BC 预训练）
    # ========================================================================
    print("\n[Step 4] Training policy...")
    print("NOTE: You need to implement update_critic, update_actor, and bc methods!")
    print("For now, this is just a framework. Implement your training loop here.")
    
    # 示例：BC 预训练（如果实现了 bc 方法）
    if len(replay_buffer) > 0:
        print("\nExample: Running BC pre-training...")
        replay_iter = create_replay_iter(replay_buffer, Config.BATCH_SIZE)
        
        # 这里需要用户实现 bc 方法
        for epoch in range(Config.N_EPOCHS):
            epoch_loss = 0.0
            n_batches  = len(replay_buffer) // Config.BATCH_SIZE
            for _ in range(n_batches):
                metrics = agent.bc(replay_iter)
                loss = metrics.get('loss', 0)
                # 如果是 tensor，调用 .item()，否则直接使用
                if isinstance(loss, torch.Tensor):
                    epoch_loss += loss.item()
                else:
                    epoch_loss += float(loss)
            epoch_loss /= n_batches
            if (epoch + 1) % 10 == 0:
                print(f"BC Epoch {epoch+1}/{Config.N_EPOCHS} | Loss: {epoch_loss}")
    
    # ========================================================================
    # 步骤 5: 评估策略
    # ========================================================================
    print("\n[Step 5] Evaluating trained policy...")
    returns, lengths = evaluate_policy(
        agent=agent,
        env=env,
        n_episodes=5,
        max_steps=1000,
        render=True
    )
    
    # ========================================================================
    # 步骤 6: 保存策略参数
    # ========================================================================
    print("\n[Step 6] Saving trained agent...")
    agent_path = "ac_agent_standalone.pt"
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'critic_target': agent.critic_target.state_dict(),
    }, agent_path)
    print(f"Agent saved to: {agent_path}")
    
    # 关闭环境
    env.close()
    
    print("\n" + "="*60)
    print("AC Framework Setup Complete!")
    print(f"Agent saved to: {agent_path}")
    print("="*60)
    print("\nNext steps:")
    print("1. Implement update_critic() method")
    print("2. Implement update_actor() method")
    print("3. Implement bc() method (for BC pre-training)")
    print("4. Implement your training loop")
    print("="*60)


# ============================================================================
# 运行主函数
# ============================================================================
if __name__ == "__main__":
    main()
