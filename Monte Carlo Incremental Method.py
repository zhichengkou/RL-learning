import numpy as np

class MonteCarlo:
    def __init__(self, gamma=1.0):
        self.gamma = gamma  # 折扣因子
        self.V = {}  # 状态的价值函数
        self.returns = {}  # 每个状态的回报记录
    
    def update(self, episode):
        """
        增量式更新状态价值函数
        episode: list of (state, reward) tuples
        """
        G = 0  # 累计回报
        for state, reward in reversed(episode):
            # 使用折扣因子更新累计回报
            G = reward + self.gamma * G

            # 如果是第一次访问该状态，初始化
            if state not in self.returns:
                self.returns[state] = []
            
            # 将当前回报G记录下来
            self.returns[state].append(G)
            
            # 增量式更新状态的价值
            self.V[state] = np.mean(self.returns[state])

    def get_value(self, state):
        """
        获取某个状态的价值
        """
        return self.V.get(state, 0.0)

    def print_value_function(self):
        """
        打印所有状态的价值函数
        """
        for state in self.V:
            print(f"State: {state}, Value: {self.V[state]}")

# 示例使用
if __name__ == "__main__":
    mc = MonteCarlo(gamma=0.9)

    # 假设一个简单的回合，状态和奖励
    # 这里使用 (state, reward) 元组来表示
    episode1 = [('A', 1), ('B', -1), ('C', 2), ('D', 0)]
    episode2 = [('A', 1), ('C', 1), ('B', -2), ('D', 0)]
    
    # 增量式更新
    mc.update(episode1)
    mc.update(episode2)

    # 打印每个状态的价值
    mc.print_value_function()
