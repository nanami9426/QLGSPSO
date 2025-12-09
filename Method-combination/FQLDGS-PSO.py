import numpy as np
import math
from scipy.stats import norm
from typing import Dict, Tuple
from Function import Fun

class FeatureSelectorConfig:
    """配置参数容器类"""

    def __init__(self, dim: int, opts: Dict):
        self.dim = dim
        self.N = opts['N']
        self.T = opts['T']
        self.max_fes = self.N * self.T
        self.num_intervals = int(np.round(4 * math.log10(dim + 50) + 4))
        self.min_range = max(3, int(dim / self.num_intervals * 0.5))
        self.valued_num = 20
        self.q_params = [0.5, 0.3, 0.2]


class PSOSwarm:
    """粒子群优化器封装"""

    def __init__(self, config: FeatureSelectorConfig):
        self.config = config
        self.positions = np.random.rand(config.num_intervals, config.dim)
        self.velocities = np.random.rand(config.num_intervals, config.dim)
        self.pbest = self.positions.copy()
        self.gbest = np.random.binomial(1, 0.5, size=config.dim)
        self.pbest_fitness = np.full(config.num_intervals, np.inf)
        self.gbest_fitness = np.inf
        self.current_best = np.zeros(config.dim, dtype=int)

    def _update_velocity(self, i: int):
        """更新粒子速度"""
        self.velocities[i] = (
                0.7 * self.velocities[i] +
                1.5 * np.random.rand() * (self.pbest[i] - self.positions[i]) +
                0.25 * np.random.rand() * (self.gbest - self.positions[i])
        )

    def _prune_solution(self, pos: np.ndarray, k: int) -> np.ndarray:
        """修剪解到指定维度"""
        selected = pos >= 0.5
        count = selected.sum()

        if count < k:
            candidates = np.where(~selected)[0]
            add_indices = np.random.choice(candidates, k - count, replace=False)
            pos[add_indices] = 0.75
        elif count > k:
            candidates = np.where(selected)[0]
            remove_indices = np.random.choice(candidates, count - k, replace=False)
            pos[remove_indices] = 0.25
        return pos

    def evolve(self, intervals: np.ndarray, evaluate_fn):
        """执行一代进化"""
        for i in range(self.config.num_intervals):
            min_d, max_d = intervals[i], intervals[i + 1]

            for _ in range(self.config.valued_num):
                self._update_velocity(i)
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], 0, 1)
                target_dim = np.random.randint(min_d + 1, max_d + 1)
                self.positions[i] = self._prune_solution(self.positions[i], target_dim)

                current_fit = evaluate_fn((self.positions[i] >= 0.5).astype(int))

                if current_fit < self.pbest_fitness[i]:
                    self.pbest[i] = self.positions[i].copy()
                    self.pbest_fitness[i] = current_fit

                if current_fit < self.gbest_fitness:
                    self.gbest = (self.positions[i] >= 0.5).astype(int).copy()
                    self.gbest_fitness = current_fit
                    self.current_best = self.gbest.copy()


class QLearningController:
    """Q学习控制器"""

    def __init__(self):
        self.q_table = np.ones((4, 4))  # 状态×动作
        self.state_map = {
            0: "右半区最优",
            1: "左半区最优",
            2: "跨区分布",
            3: "终止状态"
        }

    def get_action(self, state: int) -> int:
        """基于策略选择动作"""
        probs = self.q_table[state] / self.q_table[state].sum()
        return np.random.choice(len(probs), p=probs)

    def update_table(self, prev_state: int, action: int, reward: float):
        """更新Q表"""
        self.q_table[prev_state][action] = 0.8 * self.q_table[prev_state][action] + 0.2 * reward


def partition_intervals(start: int, end: int, num: int, method: int, min_range: int) -> np.ndarray:
    """区间划分核心函数"""
    if method == 0:
        points = np.linspace(start, end, num + 1)
    elif method == 1:
        points = norm.ppf(np.linspace(0.01, 0.99, num + 1))
        points = (points - points.min()) / (points.ptp()) * (end - start) + start
    elif method == 2:
        t = np.linspace(0, 1, num + 1)
        points = np.exp(t) - 1
        points = points / points.max() * (end - start) + start
    elif method == 3:
        points = np.linspace(0, 1, num + 1) ** 2 * (end - start) + start
    else:
        raise ValueError("Invalid partition method")

    # 区间长度约束处理
    adjusted = [points[0]]
    for p in points[1:]:
        adjusted.append(p if (p - adjusted[-1]) >= min_range else adjusted[-1] + min_range)
    adjusted[-1] = end
    return np.unique(np.round(adjusted)).astype(int)


# 辅助函数保持不变
def update_search_state(pbest_fitness: np.ndarray,
                        intervals: np.ndarray,
                        current_start: int,
                        current_end: int,
                        min_range: int) -> Tuple[int, int, int]:
    """更新搜索空间状态
    Args:
        pbest_fitness: 各区间最优适应度值数组
        intervals: 当前区间划分点数组
        current_start: 当前搜索空间起点
        current_end: 当前搜索空间终点
        min_range: 最小区间长度
    Returns:
        (新状态, 新区间起点, 新区间终点)
    """
    num_intervals = len(intervals) - 1
    state = 3  # 默认终止状态

    # 计算区间宽度是否达到终止条件
    if (current_end - current_start) / num_intervals <= min_range:
        return state, current_start, current_end

    # 获取各区间25%分位数适应度
    k = len(pbest_fitness) // 4
    quartile_values = np.partition(pbest_fitness, k, axis=0)[k]

    # 找到最优区间
    min_idx = np.argmin(quartile_values)
    max_idx1 = np.argsort(quartile_values)[-2]
    max_idx2 = np.argsort(quartile_values)[-1]

    # 状态转移逻辑
    if quartile_values[min_idx] < quartile_values[max_idx1] and quartile_values[min_idx] < quartile_values[max_idx2]:
        # 最优区间在中间区域
        new_start = intervals[min_idx]
        new_end = intervals[min_idx + 1]
        state = 2
    elif max_idx1 < min_idx and max_idx2 < min_idx:
        # 最优区间在左半区
        new_start = current_start
        new_end = intervals[min_idx + 1]
        state = 1
    else:
        # 最优区间在右半区
        new_start = intervals[min_idx]
        new_end = current_end
        state = 0

    return state, int(new_start), int(new_end)


def calculate_reward(params: list,
                     prev_fitness: float,
                     current_fitness: float,
                     prev_start: int,
                     prev_end: int,
                     current_start: int,
                     current_end: int,
                     prev_best_len: int,
                     current_best_len: int,
                     dim: int) -> float:
    """计算强化学习奖励值
    Args:
        params: 奖励权重参数 [w1, w2, w3]
        prev_fitness: 前次迭代最优适应度
        current_fitness: 当前最优适应度
        prev_start: 前次区间起点
        prev_end: 前次区间终点
        current_start: 当前区间起点
        current_end: 当前区间终点
        prev_best_len: 前次最优特征数
        current_best_len: 当前最优特征数
        dim: 总特征维度
    Returns:
        综合奖励值
    """
    # 中心对齐奖励
    r_center = (dim - abs((current_start + current_end) / 2 - current_best_len)) / dim

    # 区间稳定性奖励
    prev_center = (prev_start + prev_end) / 2
    current_center = (current_start + current_end) / 2
    r_stability = (dim - abs(prev_center - current_center)) / dim

    # 最优保持奖励
    r_consistency = 0
    if current_fitness < prev_fitness:
        r_consistency = (dim - abs(prev_best_len - current_best_len)) / dim

    # 综合奖励
    return (params[0] * r_center +
            params[1] * r_stability +
            params[2] * r_consistency)

def fs(xtrain, xvalid, ytrain, yvalid, opts) :
    """主优化流程"""
    # 初始化配置
    dim = xtrain.shape[1]
    config = FeatureSelectorConfig(dim, opts)
    pso = PSOSwarm(config)
    q_controller = QLearningController()

    # 状态跟踪
    current_state = 0
    start_dim, end_dim = 0, dim
    curve = np.full(config.max_fes + 500, np.inf)

    def evaluate_fn(x):
        return Fun(xtrain, xvalid, ytrain, yvalid, x, opts)

    # 主循环
    t = 0
    while t < config.max_fes and current_state != 3:
        # 动态划分维度空间
        action = q_controller.get_action(current_state)
        intervals = partition_intervals(start_dim, end_dim,
                                        config.num_intervals, action, config.min_range)

        # 执行PSO优化
        pso.evolve(intervals, evaluate_fn)

        # 更新状态和Q表
        prev_state = current_state
        current_state, start_dim, end_dim = update_search_state(
            pso.pbest_fitness, intervals,
            start_dim, end_dim, config.min_range
        )

        if (end_dim - start_dim) / config.num_intervals <= config.min_range:
            break

        reward = calculate_reward(
            config.q_params, pso.gbest_fitness,
            start_dim, end_dim, dim
        )
        q_controller.update_table(prev_state, action, reward)

        # 记录收敛曲线
        curve[t:t + config.valued_num * config.num_intervals] = pso.gbest_fitness
        t += config.valued_num * config.num_intervals

    return {
        'sf': pso.current_best,
        'c': curve[:config.max_fes],
        'nf': pso.current_best.sum()
    }





