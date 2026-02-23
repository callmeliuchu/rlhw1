"""
独立脚本：加载保存的策略参数并演示
不依赖训练过程，可以直接加载 .pt 文件并显示动画
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent

# 尝试导入 gymnasium，如果失败则使用 gym
try:
    import gymnasium as gym  # type: ignore
except ImportError:
    try:
        import gym  # type: ignore
    except ImportError:
        raise ImportError("Please install either 'gymnasium' or 'gym' package")


# ============================================================================
# 网络定义（必须和训练时完全一致）
# ============================================================================
class MLPPolicy(nn.Module):
    """多层感知机策略网络（和 bc_minimal_standalone.py 中完全一致）"""
    
    def __init__(self, obs_dim, ac_dim, n_layers=2, hidden_size=64):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(obs_dim, hidden_size))
        layers.append(nn.Tanh())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_size, ac_dim))
        self.mean_net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(ac_dim))
        
    def forward(self, obs):
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std.expand_as(mean))
        dist = Independent(Normal(mean, std), 1)
        return dist
    
    def get_action(self, obs):
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.from_numpy(obs).float().to(self.log_std.device)
        else:
            obs_tensor = obs.to(self.log_std.device)
        with torch.no_grad():
            dist = self.forward(obs_tensor)
            action_tensor = dist.sample()
        return action_tensor.cpu().numpy()


# ============================================================================
# 主函数：加载并演示
# ============================================================================
def main():
    # 配置（必须和训练时一致）
    ENV_NAME = "Ant-v4"
    POLICY_PATH = "bc_policy_standalone.pt"  # 训练时保存的文件
    N_LAYERS = 2
    HIDDEN_SIZE = 64
    N_EPISODES = 3
    MAX_STEPS = 1000
    
    # 设备
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("Load and Demonstrate Trained Policy")
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
        return
    obs_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    
    print(f"Environment: {ENV_NAME}")
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {ac_dim}")
    
    # 2. 创建策略网络
    print("\n[Step 2] Creating policy network...")
    policy = MLPPolicy(
        obs_dim=obs_dim,
        ac_dim=ac_dim,
        n_layers=N_LAYERS,
        hidden_size=HIDDEN_SIZE
    )
    policy.to(device)
    
    # 3. 加载保存的参数
    print(f"\n[Step 3] Loading policy from {POLICY_PATH}...")
    try:
        state_dict = torch.load(POLICY_PATH, map_location=device)
        policy.load_state_dict(state_dict)
        print("Policy loaded successfully!")
    except FileNotFoundError:
        print(f"ERROR: Policy file not found: {POLICY_PATH}")
        print("Please run bc_minimal_standalone.py first to train and save the policy.")
        env.close()
        return
    except Exception as e:
        print(f"ERROR loading policy: {e}")
        env.close()
        return
    
    policy.eval()
    
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
                # 获取动作
                action = policy.get_action(obs)
                
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
