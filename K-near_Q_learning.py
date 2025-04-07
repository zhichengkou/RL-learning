import gym
import numpy as np
import random
from collections import deque
from sklearn.neighbors import NearestNeighbors

# 环境设置
env = gym.make("FrozenLake-v1", is_slippery=False)
n_actions = env.action_space.n

# 超参数
gamma = 0.99
epsilon = 0.1
episodes = 200
K = 3  # 邻居数量

# 经验池
memory = []

# 简单的 Q 表估计函数（用 KNN）
def estimate_Q(state, action, memory, k=K):
    if not memory:
        return 0.0
    X = np.array([[s, a] for (s, a, r, ns, done) in memory])
    y = np.array([r + (0 if done else gamma * max_Q(ns, memory)) for (s, a, r, ns, done) in memory])

    # 查询最近邻
    query = np.array([[state, action]])
    nn = NearestNeighbors(n_neighbors=min(k, len(memory))).fit(X)
    dists, indices = nn.kneighbors(query)
    return np.mean(y[indices[0]])

def max_Q(state, memory):
    return max([estimate_Q(state, a, memory) for a in range(n_actions)])

# 主训练循环
for ep in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        # ε-贪婪策略
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_vals = [estimate_Q(state, a, memory) for a in range(n_actions)]
            action = int(np.argmax(q_vals))

        next_state, reward, done, _, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

    if (ep + 1) % 20 == 0:
        print(f"Episode {ep + 1}, Total Reward: {total_reward}")

env.close()
