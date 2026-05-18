from copy import deepcopy

import numpy as np
import torch.nn as nn

def dictsum(dic,t):
    return sum([dic[key][t] for key in dic if t in dic[key]])

def nestdictsum(dict):
    return sum([sum([dict[i][t] for t in dict[i]]) for i in dict])


class DailyStatsAccumulator:
    """Accumulates per-step env info into a daily summary vector for the meta-policy.

    Usage (day loop, issue #5):
        accumulator.reset(env)                          # start of each day
        for each step:
            accumulator.update(info, system_info, prices)
        state = accumulator.daily_state(agent_id, day, num_days, reward_scalar)
        env.update_brand_momentum(accumulator.served, accumulator.total_demand)
    """

    def reset(self, env):
        # Snapshot M_o(d-1) at day start — before update_brand_momentum fires
        self.momentum_snapshot = dict(env.brand_momentum)
        self.profit = {0: 0.0, 1: 0.0}
        self.reb_cost = {0: 0.0, 1: 0.0}
        self.served = {0: 0, 1: 0}
        self.total_demand = 0
        self._price_sum = {0: 0.0, 1: 0.0}
        self._price_steps = 0

    def update(self, info, system_info, prices):
        """Call once per env step.

        prices: {agent_id: np.ndarray} of per-region price scalars from the low-level action,
                or None for steps where no pricing action was taken (e.g. pure reb steps).
        """
        for a in [0, 1]:
            self.profit[a] += info[a].get('true_profit', 0.0)
            self.reb_cost[a] += info[a].get('rebalancing_cost', 0.0)
            self.served[a] += info[a].get('served_demand', 0)
        self.total_demand += system_info.get('total_demand', 0)
        if prices is not None:
            for a in [0, 1]:
                if a in prices:
                    self._price_sum[a] += float(np.mean(prices[a]))
            self._price_steps += 1

    def daily_state(self, agent_id, day, num_days, reward_scalar):
        """Return the 7-element daily summary vector for the meta-policy.

        [M_o, profit_o, avg_price_o, avg_price_opp, reb_cost_o, capture_rate_o, d/N]
        """
        opp_id = 1 - agent_id
        avg_price_o = self._price_sum[agent_id] / self._price_steps if self._price_steps > 0 else 0.0
        avg_price_opp = self._price_sum[opp_id] / self._price_steps if self._price_steps > 0 else 0.0
        capture_rate = self.served[agent_id] / self.total_demand if self.total_demand > 0 else 0.0

        return np.array([
            self.momentum_snapshot[agent_id],
            self.profit[agent_id] / reward_scalar,
            avg_price_o,
            avg_price_opp,
            self.reb_cost[agent_id] / reward_scalar,
            capture_rate,
            day / num_days,
        ], dtype=np.float32)