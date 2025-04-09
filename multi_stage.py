import numpy as np
import random

# 环境定义：机器人只能在一维线性轨道上移动
class Simple1DRobotEnv:
    def __init__(self, goal=10):
        self.goal = goal
        self.reset()

    def reset(self):
        self.pos = 0
        return self.pos

    def step(self, action):
        # 动作代表速度方向和大小：-1、0、+1
        self.pos += action
        done = abs(self.pos - self.goal) < 1e-1
        reward = -abs(self.goal - self.pos)  # 越接近目标奖励越高
        return self.pos, reward, done

# 初始化
env = Simple1DRobotEnv()
actions = [-1, 0, 1]  # 方向（幕1）或速度（幕2）
q_table = {}  # Q表：state -> action-value 映射
alpha = 0.1
gamma = 0.9
epsilon = 0.2

# 半梯度Q-learning更新函数
def update_q(state, action, reward, next_state):
    state, next_state = round(state, 1), round(next_state, 1)
    q_table.setdefault(state, np.zeros(len(actions)))
    q_table.setdefault(next_state, np.zeros(len(actions)))

    best_next = np.max(q_table[next_state])
    td_target = reward + gamma * best_next
    td_error = td_target - q_table[state][action]
    q_table[state][action] += alpha * td_error

# 分幕式训练过程
for stage in [1, 2]:
    print(f"\n====== Stage {stage} ======")
    for episode in range(100):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_r = round(state, 1)
            q_table.setdefault(state_r, np.zeros(len(actions)))

            if random.random() < epsilon:
                act_idx = random.randint(0, len(actions) - 1)
            else:
                act_idx = np.argmax(q_table[state_r])

            action = actions[act_idx]

            # 分幕控制逻辑：
            if stage == 1:
                action = np.sign(action)  # 只考虑方向
            elif stage == 2:
                action = action  # 保留原始速度

            next_state, reward, done = env.step(action)
            update_q(state, act_idx, reward, next_state)
            state = next_state
            total_reward += reward

        if episode % 20 == 0:
            print(f"Ep {episode}, Total Reward: {total_reward:.2f}")

# 展示最终策略
print("\nLearned Q-table (partial):")
for k in sorted(q_table)[:10]:
    print(f"State {k}: {q_table[k]}")
