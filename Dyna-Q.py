import gym
import numpy as np
import random
from collections import defaultdict

# 环境设置
env = gym.make("FrozenLake-v1", is_slippery=False)  # 设置为 deterministic
n_states = env.observation_space.n
n_actions = env.action_space.n

# 超参数
alpha = 0.1
gamma = 0.95
epsilon = 0.1
n_planning_steps = 5
n_episodes = 200

# 初始化 Q 表和模型
Q = np.zeros((n_states, n_actions))
model = dict()  # 用于保存模型 {(s, a): (r, s')}

# ε-贪婪策略
def choose_action(state):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

for episode in range(n_episodes):
    state = env.reset()[0]
    done = False

    while not done:
        action = choose_action(state)
        next_state, reward, done, _, _ = env.step(action)

        # 1. 实际 Q-learning 更新
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        # 2. 存储模型
        model[(state, action)] = (reward, next_state)

        # 3. 模拟更新（虚拟经验）
        for _ in range(n_planning_steps):
            s_sim, a_sim = random.choice(list(model.keys()))
            r_sim, s_next_sim = model[(s_sim, a_sim)]
            Q[s_sim, a_sim] += alpha * (
                r_sim + gamma * np.max(Q[s_next_sim]) - Q[s_sim, a_sim]
            )

        state = next_state

# 查看学习后的策略
print("Learned Policy:")
policy = np.argmax(Q, axis=1)
print(policy.reshape(4, 4))  # 针对 4x4 FrozenLake
