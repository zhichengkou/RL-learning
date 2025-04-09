# Python重写：强化学习用于A股择时策略的最小Demo（随机策略，结构演示）

import numpy as np
import pandas as pd
import random

# 模拟股票价格
prices = [0.5 + 0.5 * np.sin(i * 0.1) for i in range(200)]

# --------- 模拟交易环境 ---------
class StockTradingEnv:
    def __init__(self, prices):
        self.prices = prices
        self.n_steps = len(prices)
        self.current_step = 0
        self.cash = 10000
        self.stock = 0

    def reset(self):
        self.current_step = 0
        self.cash = 10000
        self.stock = 0
        return self._get_obs()

    def _get_obs(self):
        return self.prices[self.current_step:self.current_step+5]

    def step(self, action):
        price = self.prices[self.current_step + 4] * 100

        if action == 1 and self.cash >= price:
            self.stock += 1
            self.cash -= price
        elif action == 2 and self.stock > 0:
            self.stock -= 1
            self.cash += price

        total_asset = self.cash + self.stock * price
        reward = total_asset - 10000

        self.current_step += 1
        done = self.current_step + 5 >= self.n_steps
        return self._get_obs(), reward, done

# --------- 主程序 ---------
if __name__ == '__main__':
    env = StockTradingEnv(prices)

    for ep in range(10):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = random.randint(0, 2)
            next_state, reward, done = env.step(action)
            total_reward += reward

        print(f"Episode {ep + 1}: Reward = {total_reward:.2f}")