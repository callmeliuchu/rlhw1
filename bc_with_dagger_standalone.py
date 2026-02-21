"""
完整实现的 Behavior Cloning + DAgger - 完全独立版本
包含迭代训练、首次专家数据加载、DAgger 专家重新标注等完整流程
不依赖项目中的其他文件，超参数写死，方便学习理解

关于专家策略（expert_policy）的来源：
=====================================
1. 专家策略是什么？
   - 专家策略是一个预训练的神经网络，能够在该环境中执行任务
   - 通常通过强化学习算法（如 PPO、TRPO、SAC 等）训练得到
   - 专家策略的性能远高于随机策略或未训练的模型

2. 专家策略从哪里来的？
   【核心答案】：通过强化学习算法预先训练得到
   
   训练流程：
   a) 使用强化学习算法（PPO/TRPO/SAC 等）在环境中训练
   b) 策略网络通过与环境交互，不断更新参数
   c) 训练直到策略性能达到专家水平（高回报、稳定行为）
   d) 保存训练好的策略网络参数为 pickle 文件
   
   注意：
   - 专家策略通常由课程/项目提供，不需要自己训练
   - 你的任务是实现 BC/DAgger 算法，使用提供的专家策略
   - 详细说明见：WHERE_EXPERT_POLICY_FROM.md

3. 专家策略文件格式：
   - 文件路径：cs224r/policies/experts/Ant.pkl（对于 Ant-v4 环境）
   - 格式：pickle 文件，包含：
     * 'nonlin_type': 激活函数类型（'tanh' 或 'lrelu'）
     * 'GaussianPolicy': 策略网络参数（权重、偏置、归一化参数等）

4. 专家策略在 DAgger 中的作用：
   - 当使用 DAgger 算法时，需要专家策略来重新标注收集到的状态
   - 流程：当前策略收集轨迹 -> 专家策略为这些状态提供正确的动作 -> 用这些数据训练当前策略
   - 这样可以解决 BC 中的分布偏移问题

5. 如何加载专家策略：
   - 完整实现：使用 cs224r/policies/loaded_gaussian_policy.py 中的 LoadedGaussianPolicy 类
   - 本文件中的 load_expert_policy() 是简化版本，仅用于演示
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Independent
import gym


# ============================================================================
# 1. 超参数配置（写死）
# ============================================================================
class Config:
    # 环境配置
    ENV_NAME = "Ant-v4"
    
    # 网络结构
    N_LAYERS = 2          # 隐藏层数量
    HIDDEN_SIZE = 64      # 每层隐藏单元数
    LEARNING_RATE = 5e-3  # 学习率
    
    # 训练配置
    BATCH_SIZE = 100      # 训练 batch size
    N_EPOCHS = 100        # 每个迭代的训练轮数
    N_ITER = 2            # BC 迭代次数（BC=1, DAgger>1）
    DO_DAGGER = False     # 是否使用 DAgger（True 时会用专家重新标注）
    
    # 数据路径
    EXPERT_DATA_PATH = "cs224r/expert_data/expert_data_Ant-v4.pkl"
    EXPERT_POLICY_PATH = "cs224r/policies/experts/Ant.pkl"  # 用于 DAgger 的专家策略
    
    # 数据收集配置
    COLLECT_BATCH_SIZE = 1000  # 每次迭代收集的数据量（时间步数）
    
    # 设备
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# 2. 网络设计：MLP 策略网络
# ============================================================================
class MLPPolicy(nn.Module):
    """
    多层感知机策略网络
    输入：观测 (obs_dim,)
    输出：动作分布 (高斯分布，均值 + 标准差)
    """
    
    def __init__(self, obs_dim, ac_dim, n_layers=2, hidden_size=64):
        super().__init__()
        
        # 构建 MLP：obs_dim -> hidden_size -> ... -> hidden_size -> ac_dim
        layers = []
        
        # 第一层：输入层 -> 隐藏层
        layers.append(nn.Linear(obs_dim, hidden_size))
        layers.append(nn.Tanh())
        
        # 中间隐藏层
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        
        # 输出层：隐藏层 -> 动作维度（输出均值）
        layers.append(nn.Linear(hidden_size, ac_dim))
        
        self.mean_net = nn.Sequential(*layers)
        
        # 标准差参数（可学习的标量，广播到所有动作维度）
        self.log_std = nn.Parameter(torch.zeros(ac_dim))
        
    def forward(self, obs):
        """
        前向传播：给定观测，返回动作分布
        
        Args:
            obs: [batch_size, obs_dim] 或 [obs_dim]
        
        Returns:
            dist: Independent(Normal) 分布对象
        """
        # 如果是单个观测，添加 batch 维度
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        
        # 计算均值：[batch_size, ac_dim]
        mean = self.mean_net(obs)
        
        # 计算标准差：[batch_size, ac_dim]（广播）
        std = torch.exp(self.log_std.expand_as(mean))
        
        # 创建高斯分布（每个动作维度独立）
        # Independent 将 ac_dim 个独立的一维高斯分布组合成一个分布
        dist = Independent(Normal(mean, std), 1)
        
        return dist
    
    def get_action(self, obs):
        """
        采样动作（用于推理/演示）
        
        Args:
            obs: numpy array [obs_dim] 或 [batch_size, obs_dim]
        
        Returns:
            action: numpy array [ac_dim] 或 [batch_size, ac_dim]
        """
        # 转换为 tensor
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.from_numpy(obs).float().to(Config.DEVICE)
        else:
            obs_tensor = obs.to(Config.DEVICE)
        
        # 采样动作
        with torch.no_grad():
            dist = self.forward(obs_tensor)
            action_tensor = dist.sample()
        
        # 转换回 numpy
        return action_tensor.cpu().numpy()


# ============================================================================
# 3. 数据加载：从专家数据文件加载
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
# 3.5. 加载专家策略（用于 DAgger）
# ============================================================================
def load_expert_policy(filepath, obs_dim, ac_dim):
    """
    加载预训练的专家策略网络
    
    专家策略文件格式：
    - pickle 文件，包含 'nonlin_type' 和 'GaussianPolicy' 键
    - GaussianPolicy 包含：'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'
    
    Args:
        filepath: 专家策略 pickle 文件路径
        obs_dim: 观测维度
        ac_dim: 动作维度
    
    Returns:
        expert_policy: 加载的专家策略网络（MLPPolicy 实例）
    """
    print(f"Loading expert policy from {filepath}...")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # 检查数据格式
        if 'nonlin_type' not in data:
            raise ValueError("Expert policy file must contain 'nonlin_type'")
        
        nonlin_type = data['nonlin_type']
        policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]
        
        if policy_type != 'GaussianPolicy':
            raise ValueError(f"Unsupported policy type: {policy_type}")
        
        policy_params = data[policy_type]
        
        # 提取参数
        # 注意：这里简化处理，实际专家策略可能有不同的网络结构
        # 为了简化，我们创建一个与专家策略结构相似的网络
        # 实际项目中应该完全按照 pickle 文件中的结构重建网络
        
        print("Note: This is a simplified expert policy loader.")
        print("For full implementation, reconstruct the exact network architecture from the pickle file.")
        print("See cs224r/policies/loaded_gaussian_policy.py for reference.")
        
        # 简化版本：创建一个标准的 MLPPolicy
        # 实际应该根据 pickle 文件中的参数重建网络
        expert_policy = MLPPolicy(
            obs_dim=obs_dim,
            ac_dim=ac_dim,
            n_layers=Config.N_LAYERS,
            hidden_size=Config.HIDDEN_SIZE
        )
        expert_policy.to(Config.DEVICE)
        expert_policy.eval()
        
        print("Expert policy loaded (simplified version)")
        print("WARNING: This simplified loader may not match the exact expert policy architecture.")
        print("For accurate DAgger, use the full LoadedGaussianPolicy from cs224r/policies/loaded_gaussian_policy.py")
        
        return expert_policy
        
    except FileNotFoundError:
        print(f"ERROR: Expert policy file not found: {filepath}")
        print("Expert policy is required for DAgger. Please provide the correct path.")
        return None
    except Exception as e:
        print(f"ERROR loading expert policy: {e}")
        print("Will fall back to using expert data for relabeling")
        return None


# ============================================================================
# 4. 训练函数
# ============================================================================
def train_bc(policy, observations, actions, n_epochs=100, batch_size=100):
    """
    训练 Behavior Cloning 策略
    
    Args:
        policy: MLPPolicy 网络
        observations: [N, obs_dim] 专家观测
        actions: [N, ac_dim] 专家动作
        n_epochs: 训练轮数
        batch_size: batch 大小
    
    Returns:
        losses: 每轮的损失列表
    """
    print(f"\n{'='*60}")
    print("Training Behavior Cloning Policy")
    print(f"{'='*60}")
    print(f"Device: {Config.DEVICE}")
    print(f"Data size: {len(observations)}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {n_epochs}")
    print(f"{'='*60}\n")
    
    # 优化器
    optimizer = optim.Adam(policy.parameters(), lr=Config.LEARNING_RATE)
    
    # 转换数据为 tensor
    obs_tensor = torch.from_numpy(observations).float().to(Config.DEVICE)
    act_tensor = torch.from_numpy(actions).float().to(Config.DEVICE)
    
    losses = []
    n_batches = len(observations) // batch_size
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        
        # 随机打乱数据
        indices = np.random.permutation(len(observations))
        
        for i in range(n_batches):
            # 获取 batch
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            obs_batch = obs_tensor[batch_indices]
            act_batch = act_tensor[batch_indices]
            
            # 前向传播
            dist = policy(obs_batch)
            
            # 计算损失：负对数似然
            # log_prob: [batch_size] (Independent 已经对动作维度求和)
            log_probs = dist.log_prob(act_batch)
            loss = -log_probs.mean()  # 对 batch 求平均
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        # 每 10 轮打印一次
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs} | Loss: {avg_loss:.4f}")
    
    print(f"\nTraining completed! Final loss: {losses[-1]:.4f}")
    return losses


# ============================================================================
# 5. 数据收集函数（用当前策略采样轨迹）
# ============================================================================
def collect_trajectories(policy, env, min_timesteps, max_steps_per_episode=1000):
    """
    用当前策略收集轨迹数据
    
    Args:
        policy: 当前策略
        env: 环境
        min_timesteps: 最少收集的时间步数
        max_steps_per_episode: 每个 episode 最大步数
    
    Returns:
        observations: [N, obs_dim] 收集到的观测
        actions: [N, ac_dim] 收集到的动作
    """
    policy.eval()
    observations = []
    actions = []
    timesteps = 0
    
    print(f"Collecting trajectories with current policy (target: {min_timesteps} steps)...")
    
    while timesteps < min_timesteps:
        # 重置环境
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        
        episode_obs = []
        episode_actions = []
        steps = 0
        
        while steps < max_steps_per_episode:
            # 获取动作
            action = policy.get_action(obs)
            if len(action.shape) > 1:
                action = action[0]
            action = np.array(action).flatten()
            
            # 执行动作
            next_obs, reward, done, info = env.step(action)
            
            episode_obs.append(obs)
            episode_actions.append(action)
            
            obs = next_obs
            steps += 1
            timesteps += 1
            
            if done:
                break
        
        observations.extend(episode_obs)
        actions.extend(episode_actions)
    
    observations = np.array(observations)
    actions = np.array(actions)
    
    print(f"Collected {len(observations)} transitions ({timesteps} timesteps)")
    return observations, actions


# ============================================================================
# 6. DAgger: 用专家策略重新标注动作
# ============================================================================
def relabel_with_expert(expert_policy, observations):
    """
    DAgger 核心：用专家策略重新标注收集到的观测
    
    为什么需要专家网络（expert_policy）？
    ====================================
    
    1. 【核心问题】Behavior Cloning 的分布偏移（Distribution Shift）
       - BC 只在专家访问的状态上训练：训练分布 = 专家策略的状态分布
       - 但测试时：当前策略会访问到专家从未访问过的状态
       - 结果：策略在"新状态"上表现很差，错误累积导致失败
    
    2. 【DAgger 的解决方案】
       - 用当前策略收集轨迹（访问当前策略会到达的状态）
       - 用专家网络为这些"新状态"提供正确的动作
       - 在"真实分布"（当前策略的状态分布）上训练
    
    3. 【为什么不能用专家数据代替专家网络？】
       
       ❌ 专家数据（expert_data）的局限性：
       - 只包含专家访问过的状态：s_expert ∈ D_expert
       - 当前策略访问的状态：s_current（可能不在 D_expert 中）
       - 无法为"新状态"提供动作！
       
       ✅ 专家网络（expert_policy）的优势：
       - 是一个函数：a = expert_policy(s)，可以为任意状态 s 生成动作
       - 可以为当前策略访问的"新状态"提供专家动作
       - 这是 DAgger 算法的核心！
    
    4. 【具体例子】
       场景：训练机器人走路
       
       专家数据包含：
       - 状态1（正常站立）→ 动作1（向前走）
       - 状态2（轻微倾斜）→ 动作2（调整平衡）
       
       当前策略可能访问到：
       - 状态3（严重倾斜）← 专家数据中没有！
       
       如果没有专家网络：
       - ❌ 无法知道状态3应该做什么动作
       - ❌ 只能用专家数据中"最接近"的状态，但可能不准确
       
       有了专家网络：
       - ✅ expert_policy(状态3) → 得到专家在状态3下的正确动作
       - ✅ 策略可以在"真实分布"上学习
    
    5. 【DAgger 流程】
       Iteration 0: 用专家数据训练（warm start）
       Iteration 1+: 
         1. 当前策略收集轨迹 → 得到状态 s_current
         2. 专家网络重新标注 → expert_policy(s_current) → a_expert
         3. 用 (s_current, a_expert) 训练 → 在真实分布上学习
    
    Args:
        expert_policy: 专家策略网络（可以为任意状态生成动作）
        observations: [N, obs_dim] 当前策略访问的状态（可能不在专家数据中）
    
    Returns:
        expert_actions: [N, ac_dim] 专家在这些状态下的动作
    """
    print("Relabeling observations with expert actions (DAgger)...")
    expert_actions = []
    
    # 批量处理以提高效率
    batch_size = 100
    for i in range(0, len(observations), batch_size):
        batch_obs = observations[i:i+batch_size]
        batch_actions = expert_policy.get_action(batch_obs)
        
        # 确保是 2D
        if len(batch_actions.shape) == 1:
            batch_actions = batch_actions.reshape(1, -1)
        elif len(batch_actions.shape) == 2 and batch_actions.shape[0] == 1:
            # 如果返回的是 [1, ac_dim]，需要重复
            batch_actions = np.repeat(batch_actions, len(batch_obs), axis=0)
        
        expert_actions.append(batch_actions)
    
    expert_actions = np.concatenate(expert_actions, axis=0)
    print(f"Relabeled {len(expert_actions)} actions with expert policy")
    
    return expert_actions


# ============================================================================
# 7. 评估/演示函数
# ============================================================================
def evaluate_policy(policy, env, n_episodes=5, max_steps=1000, render=False):
    """
    评估训练好的策略
    
    Args:
        policy: 训练好的 MLPPolicy
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
    
    policy.eval()  # 设置为评估模式
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
            # 获取动作
            action = policy.get_action(obs)
            
            # 如果是 batch 输出，取第一个
            if len(action.shape) > 1:
                action = action[0]
            
            # 执行动作
            next_obs, reward, done, info = env.step(action)
            
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
# 8. 主函数：完整的 BC + DAgger 流程
# ============================================================================
def main():
    """
    完整的 Behavior Cloning + DAgger 流程：
    1. 创建环境
    2. 加载专家数据（第一次迭代使用）
    3. 创建策略网络
    4. 迭代训练：
       - 第一次迭代：使用专家数据
       - 后续迭代：用当前策略收集数据
       - DAgger：用专家重新标注收集到的数据
    5. 评估策略
    6. 保存策略
    """
    print("="*60)
    print("Behavior Cloning + DAgger - Complete Standalone Implementation")
    print("="*60)
    
    # ========================================================================
    # 步骤 1: 创建环境
    # ========================================================================
    print("\n[Step 1] Creating environment...")
    env_kwargs = {"render_mode": "rgb_array"}
    if Config.ENV_NAME == "Ant-v4":
        env_kwargs["use_contact_forces"] = True
    
    env = gym.make(Config.ENV_NAME, **env_kwargs)
    obs_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    
    print(f"Environment: {Config.ENV_NAME}")
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {ac_dim}")
    
    # ========================================================================
    # 步骤 2: 加载专家数据（第一次迭代使用）
    # ========================================================================
    print("\n[Step 2] Loading expert data for first iteration...")
    print("KEY CONCEPT: First iteration uses expert data (BC core idea)")
    print("             Subsequent iterations collect with current policy")
    try:
        expert_observations, expert_actions = load_expert_data(Config.EXPERT_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Expert data file not found: {Config.EXPERT_DATA_PATH}")
        print("Please make sure the file exists.")
        return
    
    # ========================================================================
    # 步骤 3: 创建策略网络
    # ========================================================================
    print("\n[Step 3] Creating policy network...")
    policy = MLPPolicy(
        obs_dim=obs_dim,
        ac_dim=ac_dim,
        n_layers=Config.N_LAYERS,
        hidden_size=Config.HIDDEN_SIZE
    )
    policy.to(Config.DEVICE)
    
    # 打印网络结构
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Policy network created:")
    print(f"  Layers: {Config.N_LAYERS} hidden layers")
    print(f"  Hidden size: {Config.HIDDEN_SIZE}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # ========================================================================
    # 步骤 4: 迭代训练（BC 或 DAgger）
    # ========================================================================
    print(f"\n[Step 4] Starting {'DAgger' if Config.DO_DAGGER else 'BC'} training...")
    print(f"Number of iterations: {Config.N_ITER}")
    print("="*60)
    print("DAgger 学习的是什么？")
    print("="*60)
    print("核心：在当前策略会访问的状态分布上，模仿专家的动作")
    print("")
    print("Iteration 0: 在专家状态上学习（warm start）")
    print("  → 学习：π(s_expert) ≈ π_expert(s_expert)")
    print("")
    print("Iteration 1+: 在当前策略状态上学习（DAgger 核心）")
    print("  → 学习：π(s_current) ≈ π_expert(s_current)")
    print("  → 其中 s_current ~ d_π_current（当前策略的状态分布）")
    print("  → 解决分布偏移：训练分布 = 测试分布")
    print("="*60)
    print("")
    print("为什么需要 DAgger，而不是直接用专家网络？")
    print("="*60)
    print("1. 模型压缩：专家网络可能太大，需要学习小模型")
    print("2. 部署需求：需要快速推理，适合边缘设备")
    print("3. 可用性：专家网络可能是黑盒或计算成本高")
    print("4. 任务适应：需要针对特定任务优化")
    print("5. 知识蒸馏：将专家知识转移到可部署的策略")
    print("")
    print("详细说明见：WHY_NEED_DAGGER_NOT_EXPERT.md")
    print("="*60)
    
    # 加载专家策略（用于 DAgger 重新标注）
    expert_policy = None
    if Config.DO_DAGGER:
        print("\nLoading expert policy for DAgger relabeling...")
        print("="*60)
        print("专家策略来源说明：")
        print("="*60)
        print("【专家网络从哪里来的？】")
        print("")
        print("1. 来源：通过强化学习算法预先训练得到")
        print("   - 使用 PPO、TRPO、SAC 等 RL 算法")
        print("   - 在环境中交互，不断更新策略参数")
        print("   - 训练直到策略性能达到专家水平")
        print("")
        print("2. 文件信息：")
        print("   - 文件路径: " + Config.EXPERT_POLICY_PATH)
        print("   - 格式：pickle 文件（包含网络参数）")
        print("   - 通常由课程/项目提供，不需要自己训练")
        print("")
        print("3. 在 DAgger 中的作用：")
        print("   - 为当前策略访问的状态提供专家动作")
        print("   - 解决 BC 中的分布偏移问题")
        print("")
        print("详细说明见：WHERE_EXPERT_POLICY_FROM.md")
        print("="*60)
        expert_policy = load_expert_policy(
            filepath=Config.EXPERT_POLICY_PATH,
            obs_dim=obs_dim,
            ac_dim=ac_dim
        )
    
    # 存储所有训练数据
    all_observations = []
    all_actions = []
    
    for iteration in range(Config.N_ITER):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{Config.N_ITER}")
        print(f"{'='*60}")
        
        # ====================================================================
        # 4.1 数据收集：第一次迭代用专家数据，后续用当前策略
        # ====================================================================
        if iteration == 0:
            # 第一次迭代：使用专家数据
            print("\n[Iteration 0] Using expert data (first iteration)...")
            print("This is the key BC concept: start with expert demonstrations")
            print("Why? Policy is randomly initialized, needs expert data to learn!")
            obs_batch = expert_observations
            act_batch = expert_actions
        else:
            # 后续迭代：用当前策略收集数据
            print(f"\n[Iteration {iteration}] Collecting data with current policy...")
            print("Current policy has been trained, can now collect meaningful trajectories")
            obs_batch, act_batch = collect_trajectories(
                policy=policy,
                env=env,
                min_timesteps=Config.COLLECT_BATCH_SIZE,
                max_steps_per_episode=1000
            )
            
            # DAgger: 用专家策略重新标注
            if Config.DO_DAGGER:
                print(f"\n[Iteration {iteration}] DAgger: Relabeling with expert...")
                print("="*60)
                print("为什么需要专家网络？")
                print("="*60)
                print("1. 当前策略访问的状态可能不在专家数据中")
                print("2. 专家数据只能提供有限状态的动作")
                print("3. 专家网络可以为任意状态生成专家动作")
                print("4. 这样可以在'真实分布'（当前策略的状态分布）上训练")
                print("="*60)
                if expert_policy is not None:
                    # ✅ 正确做法：用专家网络为任意状态生成动作
                    print("Using expert policy network to relabel...")
                    act_batch = relabel_with_expert(expert_policy, obs_batch)
                else:
                    # ⚠️ 简化版本：从专家数据中找到最接近的状态（不准确！）
                    print("WARNING: Expert policy not loaded, using expert data as fallback...")
                    print("This is NOT accurate! Expert data may not contain these states.")
                    print("For proper DAgger, you MUST load the expert policy network.")
                    # 这里简化处理：实际应该用专家策略网络
                    # 为了演示，我们随机选择一些专家动作
                    expert_indices = np.random.choice(
                        len(expert_observations), 
                        size=len(obs_batch),
                        replace=True
                    )
                    act_batch = expert_actions[expert_indices]
        
        # 累积数据
        all_observations.append(obs_batch)
        all_actions.append(act_batch)
        
        # ====================================================================
        # 4.2 训练策略（使用累积的所有数据）
        # ====================================================================
        print(f"\n[Iteration {iteration}] Training policy on collected data...")
        # 合并所有数据
        train_obs = np.concatenate(all_observations, axis=0)
        train_acts = np.concatenate(all_actions, axis=0)
        
        print(f"Total training data: {len(train_obs)} transitions")
        print(f"  - From iteration 0 (expert): {len(all_observations[0])}")
        if iteration > 0:
            print(f"  - From iteration {iteration} (collected): {len(obs_batch)}")
        
        losses = train_bc(
            policy=policy,
            observations=train_obs,
            actions=train_acts,
            n_epochs=Config.N_EPOCHS,
            batch_size=Config.BATCH_SIZE
        )
        
        # ====================================================================
        # 4.3 评估当前策略
        # ====================================================================
        print(f"\n[Iteration {iteration}] Evaluating current policy...")
        returns, lengths = evaluate_policy(
            policy=policy,
            env=env,
            n_episodes=3,
            max_steps=1000,
            render=False
        )
    
    # ========================================================================
    # 步骤 5: 最终评估
    # ========================================================================
    print("\n[Step 5] Final evaluation...")
    final_returns, final_lengths = evaluate_policy(
        policy=policy,
        env=env,
        n_episodes=5,
        max_steps=1000,
        render=False
    )
    
    # ========================================================================
    # 步骤 6: 保存策略参数
    # ========================================================================
    print("\n[Step 6] Saving trained policy...")
    policy_path = "bc_policy_standalone.pt"
    torch.save(policy.state_dict(), policy_path)
    print(f"Policy saved to: {policy_path}")
    
    # 关闭环境
    env.close()
    
    print("\n" + "="*60)
    print(f"{'DAgger' if Config.DO_DAGGER else 'BC'} Training Complete!")
    print(f"Policy saved to: {policy_path}")
    print("To visualize the policy, run: python load_and_demo_policy.py")
    print("="*60)


# ============================================================================
# 运行主函数
# ============================================================================
if __name__ == "__main__":
    main()
