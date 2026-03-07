"""
独立脚本：加载保存的 AC Agent 并演示
不依赖训练过程，可以直接加载 .pt 文件并显示动画
"""

import numpy as np
import torch
import torch.nn as nn

# 尝试导入 gymnasium，如果失败则使用 gym
try:
    import gymnasium as gym  # type: ignore
except ImportError:
    try:
        import gym  # type: ignore
    except ImportError:
        raise ImportError("Please install either 'gymnasium' or 'gym' package")


# ============================================================================
# 辅助工具函数（必须和训练时完全一致）
# ============================================================================
def weight_init(m):
    """初始化网络权重"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class TruncatedNormal:
    """
    截断正态分布（用于 Actor 输出）
    将动作限制在 [-1, 1] 范围内（通过 tanh）
    """
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std
        self.dist = torch.distributions.Normal(mu, std)
    
    def sample(self, clip=None):
        """采样动作"""
        action = self.dist.sample()
        # action = torch.tanh(action)
        # if clip is not None:
        #     action = torch.clamp(action, -clip, clip)
        return action
    
    @property
    def mean(self):
        """返回分布的均值（经过 tanh）"""
        return torch.tanh(self.mu)
    
    def log_prob(self, action):
        """计算 log 概率"""
        eps = 1e-6
        action = torch.clamp(action, -1 + eps, 1 - eps)
        u = 0.5 * torch.log((1 + action) / (1 - action))
        log_prob = self.dist.log_prob(u)
        log_prob -= torch.log(1 - action.pow(2) + eps)
        return log_prob.sum(dim=-1)


# ============================================================================
# Actor 网络（必须和训练时完全一致）
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
# Critic 网络（必须和训练时完全一致）
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
# AC Agent（简化版，只用于推理）
# ============================================================================
class ACAgent:
    def __init__(self, obs_shape, action_shape, device, hidden_dim, std=0.1):
        self.device = device
        self.actor = Actor(obs_shape, action_shape, hidden_dim, std).to(device)
        self.train(False)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)

    def act(self, obs, eval_mode=True):
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


# ============================================================================
# 主函数：加载并演示
# ============================================================================
def main():
    # 配置（必须和训练时完全一致）
    ENV_NAME = "Ant-v4"
    AGENT_PATH = "ac_agent_standalone.pt"  # 训练时保存的文件
    HIDDEN_DIM = 256
    STD = 0.1
    N_EPISODES = 3
    MAX_STEPS = 1000
    
    # 设备
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("Load and Demonstrate Trained AC Agent")
    print("="*60)
    
    # 1. 创建环境
    print("\n[Step 1] Creating environment...")
    env_kwargs = {}
    if ENV_NAME == "Ant-v4":
        env_kwargs["use_contact_forces"] = True
    env_kwargs["render_mode"] = "human"  # 可视化模式
    
    try:
        env = gym.make(ENV_NAME, **env_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to create environment: {e}")
        return
    
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    
    print(f"Environment: {ENV_NAME}")
    print(f"Observation shape: {obs_shape}")
    print(f"Action shape: {action_shape}")
    
    # 2. 创建 AC Agent（只创建 Actor，用于推理）
    print("\n[Step 2] Creating AC Agent...")
    agent = ACAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        hidden_dim=HIDDEN_DIM,
        std=STD
    )
    
    # 3. 加载保存的参数
    print(f"\n[Step 3] Loading agent from {AGENT_PATH}...")
    try:
        checkpoint = torch.load(AGENT_PATH, map_location=device)
        
        # 检查保存的格式
        if isinstance(checkpoint, dict):
            if 'actor' in checkpoint:
                agent.actor.load_state_dict(checkpoint['actor'])
                print("Loaded actor parameters")
            else:
                # 如果直接是 state_dict
                agent.actor.load_state_dict(checkpoint)
                print("Loaded actor parameters (direct state_dict)")
        else:
            agent.actor.load_state_dict(checkpoint)
            print("Loaded actor parameters")
        
        print("Agent loaded successfully!")
    except FileNotFoundError:
        print(f"ERROR: Agent file not found: {AGENT_PATH}")
        print("Please run ac_minimal_standalone.py first to train and save the agent.")
        env.close()
        return
    except Exception as e:
        print(f"ERROR loading agent: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return
    
    agent.train(False)  # 设置为评估模式
    
    # 4. 演示
    print("\n[Step 4] Visual demonstration...")
    print("Close the MuJoCo window or press Ctrl+C to stop.\n")
    
    try:
        for episode in range(N_EPISODES):
            print(f"Episode {episode + 1}/{N_EPISODES}")
            
            # 重置环境
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result
            else:
                obs = reset_result
            
            total_reward = 0
            steps = 0
            
            while True:
                # 获取动作（使用 eval_mode=True 使用均值）
                action = agent.act(obs, eval_mode=True)
                
                # 确保 action 是 1D
                if len(action.shape) > 1:
                    action = action[0]
                action = np.array(action).flatten()
                
                # 裁剪动作到有效范围
                action = np.clip(action, env.action_space.low, env.action_space.high)
                
                # 执行动作（兼容 gym 和 gymnasium）
                step_result = env.step(action)
                if len(step_result) == 5:  # gymnasium 返回 5 个值
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:  # gym 返回 4 个值
                    next_obs, reward, done, info = step_result
                
                # 渲染
                env.render()
                
                total_reward += reward
                steps += 1
                obs = next_obs
                
                if done or steps >= MAX_STEPS:
                    break
            
            print(f"  Steps: {steps}, Total Reward: {total_reward:.2f}")
    
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()
        print("\nDemonstration complete!")


if __name__ == "__main__":
    main()
