import numpy as np
from collections import defaultdict

# 初始化 Q 表
Q = defaultdict(lambda: np.zeros(2))  # 2 个动作：Hit (0), Stand (1)

# 记录行为策略和目标策略
behavior_policy = {0: 0.5, 1: 0.5}  # 随机策略
target_policy = {0: 1.0, 1: 0.0}  # 贪婪策略 (Hit: 1, Stand: 0)

# 重要性采样权重
W = 1.0

# 采样轨迹
trajectory = [(13, 0), (18, 1)]  # (总点数, 选择的动作)
reward = 1  # 这场游戏赢了

# 反向遍历轨迹并更新 Q 值
for state, action in reversed(trajectory):
    W *= target_policy[action] / behavior_policy[action]  # 计算重要性采样比率
    Q[state][action] += W * (reward - Q[state][action])  # 更新 Q 值
    if W == 0:
        break  # 重要性采样权重太小，则提前结束更新

print(Q)  # 打印最终 Q 值
