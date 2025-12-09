import numpy as np
from Function import Fun
import random
import math
from scipy.stats import norm, t, cauchy, weibull_min, weibull_max

# 这个代码最完备,集齐了区间分层，聚类分层，区间最优引导，缺乏下层强化学习，下层搜索特征只有互信息
# 增加pbest初始化的时候使用概率
def dim_divide(dim, dim_divide):
    interval_width = dim // dim_divide
    intervals = np.array([i * interval_width for i in range(dim_divide + 1)])
    if intervals[-1] > dim:
        intervals[-1] = dim
    return intervals






def init_Xc(dim, divide_min, divide_max):
    num_dim = random.randint(divide_min + 1, divide_max)
    X = np.zeros(dim)
    indices = np.random.choice(dim, num_dim, replace=False)
    X[indices] = 1
    return X, num_dim


# 判断优先的函数，
# fitness_qutor 为四等分点
# cut_percentage 为删减的百分比
# dim_divide_num 划分的区间数
def judge_area(fitness_qutor, cut_percentage, dim_divide_num, best_num):
    dim_divide_flag = np.ones(dim_divide_num)
    fitness_sort = np.argsort(fitness_qutor)
    middle_num = dim_divide_num // 2
    cut_count = int(np.ceil(dim_divide_num * cut_percentage))
    # 如果末尾的在中位数以下可以直接删减
    best_fitness = fitness_qutor[fitness_sort[best_num]]
    count = 0
    cut_flag = False
    start_num = 0
    end_num = dim_divide_num
    ##  优先删除高维的，如果高维不足可以从低维起删除
    for i in range(dim_divide_num - 1, 0, -1):
        if fitness_qutor[i] > best_fitness:
            dim_divide_flag[i] = 0
            end_num = i
            count += 1
        # 如果高维度删除但还没删除满就遇到了最优值先退出从低维删除起
        if fitness_qutor[i] <= best_fitness:
            cut_flag = False
            break
        if count >= cut_count:
            cut_flag = True
            break
    if cut_flag == False:
        for i in range(dim_divide_num):
            if fitness_qutor[i] > best_fitness:
                dim_divide_flag[i] = 0
                start_num = i
                count += 1
            if fitness_qutor[i] <= best_fitness:
                cut_flag = False
                break
            if count >= cut_count:
                cut_flag = True
                break
    return start_num, end_num, cut_flag


def partition_intervals(start, end, num_intervals, method, min_range):
    if method == 0:  # 均匀区间划分
        intervals = np.linspace(start, end, num_intervals + 1)
    elif method == 1:  # 中间的区间间隔较少，两边的间隔较大
        intervals = norm.ppf(np.linspace(0.01, 0.99, num_intervals + 1))
        intervals = (intervals - intervals.min()) / (intervals.max() - intervals.min())
        intervals = start + intervals * (end - start)
    elif method == 2:  # 前面区间的间隔较小，后面区间间隔较大
        intervals = np.linspace(0, 1, num_intervals + 1)
        intervals = np.exp(intervals) - 1
        intervals = intervals / intervals.max()
        intervals = start + intervals * (end - start)
    elif method == 3:  # 间隔较少的区间划分线在左侧
        intervals = np.linspace(0, 1, num_intervals + 1) ** 2
        intervals = start + intervals * (end - start)
    else:
        raise ValueError("Method should be one of 1, 2, 3, or 4")

    # 检查并调整区间长度
    adjusted_intervals = [intervals[0]]
    for i in range(1, len(intervals)):
        if intervals[i] - adjusted_intervals[-1] < min_range:
            adjusted_intervals.append(adjusted_intervals[-1] + min_range)
        else:
            adjusted_intervals.append(intervals[i])

    # 确保最后一个点是 end
    adjusted_intervals[-1] = end
    return np.array(adjusted_intervals).astype(int)


def compute_len_state(fit_num_quarter, intervals, divide_num, start, end, min_range):
    state = 3
    # start = 0
    # end = 0
    # 如果区间达到最小间隔，返回暂停状态三
    if (end - start) / divide_num <= min_range:
        return 3, start, end
    min_index_list = np.argsort(fit_num_quarter)
    min_fit_index = min_index_list[0]
    max_fit_index1 = min_index_list[-2]
    max_fit_index2 = min_index_list[-1]

    if max_fit_index1 >= min_fit_index and max_fit_index2 >= min_fit_index:
        state = 0
        end = intervals[(max_fit_index1 + min_fit_index) // 2]
    elif max_fit_index1 < min_fit_index and max_fit_index2 < min_fit_index:
        state = 1
        start = intervals[0] # max_fit_index2 + 1
    else:
        state = 2
        if max_fit_index1 < min_fit_index:
            start = intervals[0] # max_fit_index1 + 1
            end = intervals[(max_fit_index2 + min_fit_index) // 2]
        else:
            start = intervals[0] # max_fit_index2 + 1
            end = intervals[(max_fit_index1 + min_fit_index) // 2]

    return state, start, end


def weighted_random_choice(values):
    total_sum = np.sum(values)
    probabilities = values / total_sum
    selected_index = np.random.choice(len(values), p=probabilities)
    return selected_index


def compute_len_action(state, qtable):
    qchoose = qtable[state, :]
    action = weighted_random_choice(qchoose)
    return action


def select_len_action(state, qtable, action_num, use_q_learning, random_actions, default_action):
    if random_actions:
        return np.random.randint(action_num)
    if use_q_learning:
        return compute_len_action(state, qtable)
    default_action = max(0, min(action_num - 1, int(default_action)))
    return default_action


def compute_len_reward(param, fit_best_pre, fit_best_cur, fit_best_pre_len, fit_best_cur_len, fit_area_pre,
                       fit_area_pre_start, fit_area_pre_end, fit_area_cur, fit_area_cur_start, fit_area_cur_end, dim):
    r1 = (dim - np.abs((fit_area_cur_start + fit_area_cur_end) / 2 - fit_best_cur_len)) / dim
    r2 = (dim - np.abs((fit_area_pre_start + fit_area_pre_end) - (fit_area_cur_start + fit_area_cur_end)) / 2) / dim
    if fit_best_cur != fit_best_pre:
        r3 = (dim - np.abs(fit_best_pre_len - fit_best_cur_len)) / dim
    else:
        r3 = 0
    r = param[0] * r1 + param[1] * r2 + param[2] * r3
    return r


def renew_len_qtable(state_pre, state_cur, action, r, qtable, lr=0.2, gamma=0.0):
    lr = np.clip(lr, 0.0, 1.0)
    gamma = max(0.0, gamma)
    best_next = 0.0
    if 0 <= state_cur < qtable.shape[0]:
        best_next = np.max(qtable[state_cur, :])
    td_target = r + gamma * best_next
    qtable[state_pre, action] = (1 - lr) * qtable[state_pre, action] + lr * td_target


# 寻找区间适应度的最小区间所在的位置
def find_min_area(value_fitness_quarter, intervals):
    index = np.argmin(value_fitness_quarter)
    return value_fitness_quarter[index], intervals[index], intervals[index + 1]


def prune_solution(x, k):
    # Ensure the solution has exactly k 1's
    x1 = (x >= 0.5).astype(int)
    selected_features = np.where(x1 == 1)[0]  # Get indices of selected features
    n_selected = len(selected_features)
    if n_selected < k:
        # Add 1's to the solution
        available_features = np.where(x1 == 0)[0]  # Features not selected
        additional = np.random.choice(available_features, k - n_selected, replace=False)
        x[additional] = 0.75
    elif n_selected > k:
        # Remove 1's from the solution
        to_remove = np.random.choice(selected_features, n_selected - k, replace=False)
        x[to_remove] = 0.25

    return x

def prune_solution_fast(x, k):
    x1 = (x >= 0.5)
    n_selected = np.count_nonzero(x1)
    if n_selected < k:
        # 添加1
        available = np.flatnonzero(~x1)
        x[np.random.choice(available, k - n_selected, replace=False)] = 0.75
    elif n_selected > k:
        # 移除1
        selected = np.flatnonzero(x1)
        x[np.random.choice(selected, n_selected - k, replace=False)] = 0.25

    return x
# 这个版本是使用区间最优个体迭代更新,使用Qlearning进行特征挑选

def make_increasing(a, step=5):
    for i in range(1, len(a)):
        if a[i] <= a[i - 1]:  # 检查是否下降
            # 如果下降，将当前元素增加至前一个元素加上固定间隔
            a[i] = a[i - 1] + step
    return a


def baseline_pso_run(xtrain, xvalid, ytrain, yvalid, opts):
    dim = xtrain.shape[1]
    N = opts['N']
    T = opts['T']
    w = opts.get('baseline_w', 0.9)
    c1 = opts.get('baseline_c1', 2.0)
    c2 = opts.get('baseline_c2', 2.0)
    thres = opts.get('baseline_thres', 0.5)
    X = np.random.rand(N, dim)
    V = np.random.uniform(-1, 1, size=(N, dim))
    pbest = X.copy()
    gbest = X[0, :].copy()
    pbest_fit = np.full(N, np.inf)
    gbest_fit = np.inf
    curve = np.zeros((1, T))
    for t in range(T):
        Xbin = (X > thres).astype(int)
        for i in range(N):
            fit = Fun(xtrain, xvalid, ytrain, yvalid, Xbin[i, :], opts)
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest[i, :] = X[i, :]
            if fit < gbest_fit:
                gbest_fit = fit
                gbest = X[i, :]
        curve[0, t] = gbest_fit
        r1 = np.random.rand(N, dim)
        r2 = np.random.rand(N, dim)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = np.clip(X + V, 0, 1)
    gsel = (gbest > thres).astype(int)
    nf = np.count_nonzero(gsel)
    return {'sf': gsel, 'c': curve, 'nf': nf}


def fs(xtrain, xvalid, ytrain, yvalid, opts):
    # ********************固定的参数**********************
    dim = xtrain.shape[1]  # 特征的维度
    N = opts['N']
    T = opts['T']
    ql_lr = float(opts.get('ql_lr', 0.2))
    ql_gamma = float(opts.get('ql_gamma', 0.0))
    ql_mode = str(opts.get('ql_mode', 'full')).lower()
    if ql_mode not in ['full', 'dim_only', 'random_policy', 'baseline']:
        ql_mode = 'full'
    if ql_mode == 'baseline':
        return baseline_pso_run(xtrain, xvalid, ytrain, yvalid, opts)
    use_q_learning = ql_mode == 'full'
    random_len_actions = ql_mode == 'random_policy'
    default_len_action = int(opts.get('len_action_default', 0))
    Max_FEs = T * N
    curve = np.ones([1, Max_FEs + 500], dtype='float')
    curves = np.ones([1, Max_FEs], dtype='float')
    # Nsfc = 0
    # fitGB = float('inf')
    Nsfb = dim
    # fitc = 0
    Xgb = np.zeros(dim, dtype='int')

    # *******************初始化层*********************
    default_intervals = int(np.round(4 * math.log10(dim + 50) + 4))
    num_intervals = int(opts.get('interval_num', default_intervals))
    num_intervals = max(2, min(num_intervals, max(2, dim)))
    min_range = int(max(1, min(5, np.round(dim / num_intervals * 0.5))))  # 区间最小间隔 初始时设置为5
    len_state_num = 4
    len_action_num = 4
    len_qtable = np.ones((len_state_num, len_action_num))
    len_reward_param = [0.5, 0.3, 0.2]  # 长度相关的三种奖励的权重
    gbest_fit_cur = 1  # 当前全局最优的适应度
    gbest_len_cur = dim  # 当前全局最优的特征长度
    # ************************* 循环-搜索 **********************************************************************

    start_dim = 0  # 当前分区开始的维度
    end_dim = dim  # 当前分区结束的维度
    # dim_divide_flag = np.ones(dim_divide_num)  # 判断空间是否可用
    t = 0
    state_cur = 0  # 初始化设置状态为0
    step_total = 0  # 总消耗的迭代次数
    valued_num = max(1, int(opts.get('interval_iterations', 20)))
    # 随机生成一个划分区间，用于最优生成
    # action_cur =
    action_cur = select_len_action(state_cur, len_qtable, len_action_num, use_q_learning, random_len_actions,
                                   default_len_action)
    if not (use_q_learning or random_len_actions):
        state_cur = 3
    intervals = partition_intervals(start_dim, end_dim, num_intervals, action_cur, min_range)
    # print(f"dim = {dim}, num = {num_intervals}, intervals = {intervals} ")
    temp_fitness = np.zeros((num_intervals, valued_num))  # 储存每个区间适应度的变量
    pso_x = np.random.rand(num_intervals, dim)   # pso的位置，用于寻找特征
    pso_v = np.random.rand(num_intervals, dim)  # pso的速度，用于寻找特征
    pso_xbest = np.zeros((num_intervals, dim))  # pso的区间最优位置
    pso_gbest = np.random.binomial(1, 0.5, size=dim)  # pso的区间最优位置
    g_best = (pso_gbest >= 0.5).astype(int)
    fitGB = Fun(xtrain, xvalid, ytrain, yvalid, g_best, opts)
    pso_fitbest = np.ones(num_intervals)    # pso的最优适应度
    intervals = make_increasing(intervals, min_range)
    for i in range(num_intervals - 1, -1, -1):
        # 初始化粒子和个体最优解
        x = np.random.rand(dim)
        pbest = np.random.rand(dim)
        X_best = (pbest >= 0.5).astype(int)
        fit_best = Fun(xtrain, xvalid, ytrain, yvalid, X_best, opts)
        # 初始化速度
        v = pso_v[i, :]
        # 随机数预生成
        rand1 = np.random.random(valued_num)
        rand2 = np.random.random(valued_num)
        Nsfn_list = np.random.randint(intervals[i] + 1, intervals[i + 1] + 1, size=valued_num)

        for j in range(valued_num):
            Nsfc = Nsfn_list[j]
            # 更新速度和位置
            v = 0.7 * v + 1.5 * rand1[j] * (pbest - x) + 0.25 * rand2[j] * (pso_gbest - x)
            x = np.clip(x + v, 0, 1)
            # 剪枝 + 二值化
            x = prune_solution_fast(x, Nsfc)
            X_cur = (x >= 0.5).astype(int)
            fitc = Fun(xtrain, xvalid, ytrain, yvalid, X_cur, opts)
            # 保存当前解和适应度
            temp_fitness[i, j] = fitc
            # 更新个体最优
            if fitc <= fit_best:
                pbest = x.copy()
                fit_best = fitc
            # 更新全局最优
            if fitc < fitGB or (fitc == fitGB and Nsfc < Nsfb):
                pso_gbest = x.copy()
                Xgb = X_cur.copy()
                fitGB = fitc
                Nsfb = Nsfc
                gbest_fit_cur = fitc
                gbest_len_cur = Nsfc

        # 写回当前粒子的最新状态
        pso_x[i, :] = x
        pso_xbest[i, :] = pbest
        pso_fitbest[i] = fit_best
        pso_v[i, :] = v
    k = 0  # 四分点
    # k = valued_num // 4
    value_fitness_quarter = np.partition(temp_fitness, k, axis=1)[:, k]  # 寻找适应度最小的四等分点
    quarter_fit_cur, quarter_len_start_cur, quarter_len_end_cur = find_min_area(value_fitness_quarter, intervals)
    while t < Max_FEs and state_cur != 3:
        # 更新一些状态值为pre
        gbest_fit_pre = gbest_fit_cur
        gbest_len_pre = gbest_len_cur
        quarter_fit_pre = quarter_fit_cur
        quarter_len_start_pre = quarter_len_start_cur
        quarter_len_end_pre = quarter_len_end_cur
        state_pre = state_cur
        # 根据当前长度状态计算获得长度行动
        action_cur = select_len_action(state_cur, len_qtable, len_action_num, use_q_learning, random_len_actions,
                                       default_len_action)
        # 根据q值得到的行动使用对应的方式划分区间
        intervals = partition_intervals(start_dim, end_dim, num_intervals, action_cur, min_range)
        # temp_x = np.zeros((num_intervals, valued_num, dim))
        # temp_fitness = np.zeros((num_intervals, valued_num))  # 储存每个区间适应度的变量
        ##########################################################################
        # 下面两行赋值上一次评价前的最优历史值
        # print(f"结束的t为{t}, intervals 为{intervals}")
        intervals = make_increasing(intervals, min_range)
        # print(f'{intervals}')
        for i in range(num_intervals - 1, -1, -1):
            # 预生成随机数
            rand1 = np.random.random(valued_num)
            rand2 = np.random.random(valued_num)

            Nsfn_list = np.random.randint(intervals[i] + 1, intervals[i + 1] + 1, size=valued_num)
            # 遍历粒子
            for j in range(valued_num):
                Nsfn = Nsfn_list[j]
                # 向量化更新速度
                pso_v[i, :] = 0.7 * pso_v[i, :] + 1.5 * rand1[j] * (pso_xbest[i, :] - pso_x[i, :]) + \
                              0.25 * rand2[j] * (pso_gbest - pso_x[i, :])

                # 更新位置并限制在 0 到 1 之间
                pso_x[i, :] = np.clip(pso_x[i, :] + pso_v[i, :], 0, 1)

                # 剪枝操作
                pso_x[i, :] = prune_solution_fast(pso_x[i, :], Nsfn)

                # 计算当前解
                X_cur = (pso_x[i, :] >= 0.5).astype(int)
                fitc = Fun(xtrain, xvalid, ytrain, yvalid, X_cur, opts)
                temp_fitness[i, j] = fitc
                # 更新个体最优解
                if fitc <= pso_fitbest[i]:
                    pso_xbest[i, :] = pso_x[i, :].copy()  # 使用 .copy() 来避免引用
                    pso_fitbest[i] = fitc
                    # 更新全局最优解
                    if fitc < fitGB or (fitc == fitGB and Nsfn < Nsfb):
                        pso_gbest = pso_x[i, :].copy()  # 直接赋值而不使用额外的 temp_x 复制
                        Xgb = X_cur.copy()
                        fitGB = fitc
                        Nsfb = Nsfn
                        gbest_fit_cur = fitc
                        gbest_len_cur = Nsfn
                # 记录全局最优的适应度
                curve[0, t] = fitGB
                t += 1
            #################################################################
        k = 0  # 四分点
        value_fitness_quarter = pso_fitbest.copy()
        # value_fitness_quarter = np.partition(temp_fitness, k, axis=1)[:, k]  # 寻找适应度最小的四等分点
        quarter_fit_cur, quarter_len_start_cur, quarter_len_end_cur = find_min_area(value_fitness_quarter,
                                                                                    intervals)
        state_cur, start_dim, end_dim = compute_len_state(value_fitness_quarter, intervals, num_intervals, start_dim,
                                                          end_dim, min_range)
        if ((end_dim - start_dim) // num_intervals <= min_range):
            intervals = partition_intervals(start_dim, end_dim, num_intervals, action_cur, 1)
            break
        len_r = compute_len_reward(len_reward_param, gbest_fit_pre, gbest_fit_cur, gbest_len_pre, gbest_len_cur,
                                   quarter_fit_pre, quarter_len_start_pre, quarter_len_end_pre,
                                   quarter_fit_cur, quarter_len_start_cur, quarter_len_end_cur, dim)

        renew_len_qtable(state_pre, state_cur, action_cur, len_r, len_qtable, lr=ql_lr, gamma=ql_gamma)
    # print(f"结束的t为{t}, intervals 为{intervals}, 最优长度{Nsfb}, 最优适应度{fitGB}")
    intervals = make_increasing(intervals, min_range)
    while t < Max_FEs:

        # print(f'{intervals}')
        for i in range(num_intervals - 1, -1, -1):
            # 预生成随机数
            rand1 = np.random.random(valued_num)
            rand2 = np.random.random(valued_num)
            Nsfn_list = np.random.randint(intervals[i] + 1, intervals[i + 1] + 1, size=valued_num)
            # 遍历粒子
            for j in range(valued_num):
                Nsfn = Nsfn_list[j]
                # 向量化更新速度
                pso_v[i, :] = 0.7 * pso_v[i, :] + 1.5 * rand1[j] * (pso_xbest[i, :] - pso_x[i, :]) + \
                              0.25 * rand2[j] * (pso_gbest - pso_x[i, :])
                # 更新位置并限制在 0 到 1 之间
                pso_x[i, :] = np.clip(pso_x[i, :] + pso_v[i, :], 0, 1)
                # 剪枝操作
                pso_x[i, :] = prune_solution_fast(pso_x[i, :], Nsfn)
                # 计算当前解
                X_cur = (pso_x[i, :] >= 0.5).astype(int)
                fitc = Fun(xtrain, xvalid, ytrain, yvalid, X_cur, opts)
                temp_fitness[i, j] = fitc
                # 更新个体最优解
                if fitc <= pso_fitbest[i]:
                    pso_xbest[i, :] = pso_x[i, :].copy()  # 使用 .copy() 来避免引用
                    pso_fitbest[i] = fitc
                    # 更新全局最优解
                    if fitc < fitGB or (fitc == fitGB and Nsfn < Nsfb):
                        pso_gbest = pso_x[i, :].copy()  # 直接赋值而不使用额外的 temp_x 复制
                        Xgb = X_cur.copy()
                        fitGB = fitc
                        Nsfb = Nsfn
                    # 记录全局最优的适应度
                curve[0, t] = fitGB
                t += 1

    # curve[0, -(Max_FEs - step_total):] = curve[0, -(Max_FEs - step_total + 1)]
    curves[0, :] = curve[0, :Max_FEs]

    num_points = opts['T']
    # 等差选取数据点的索引
    indices = np.round(np.linspace(0, Max_FEs - 1, num_points)).astype(int)
    # 从Cost中选取数据填入curve
    curves = curves[:, indices]
    # 将curve变形为所需的大小
    curves = curves.reshape(1, -1)
    index = np.where(Xgb == 1)[0]
    # print(f"QLDGS-PSO{index}")
    ceeh_data = {'sf': Xgb, 'c': curves, 'nf': Nsfb}
    return ceeh_data
