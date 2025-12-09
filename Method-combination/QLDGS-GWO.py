import random

import numpy as np
from numpy.random import rand
from sklearn.feature_selection import mutual_info_classif

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


def dim_divide2(start_dim, end_dim, dim_divide):
    interval_width = (end_dim - start_dim) / dim_divide
    intervals = np.array([start_dim + int(np.ceil(i * interval_width)) for i in range(dim_divide + 1)])
    if intervals[-1] > end_dim:
        intervals[-1] = end_dim
    return intervals


def compute_state(Nsf, state_list):
    state = 0
    for i in range(state_list.shape[0] - 1):
        if Nsf <= state_list[i + 1]:
            state = i - 1
            break
    return state


def elite_divide(dim, divide_num):
    result = dim // divide_num
    # 如果有余数，则结果加1
    if dim % divide_num != 0:
        result += 1
    return result


def compute_action(state, qtable):
    qchoose = qtable[state, :]
    action_dim = np.size(qchoose)
    prob = random.random()
    if prob < 0.2:
        action = random.randint(0, action_dim - 1)
    else:
        action = np.argmax(qchoose)
    return action


def select_features_by_score(scores, num_features_to_select):
    """
    根据特征得分选择特征，得分越高被选择的概率越高

    参数:
    scores (array-like): 特征得分向量
    num_features_to_select (int): 选择的特征数量

    返回:
    selected_features (list): 被选择特征的索引列表
    """
    scores = np.array(scores)
    non_zero_indices = scores.nonzero()[0]  # 获取所有非零得分的索引
    zero_indices = np.where(scores == 0)[0]  # 获取所有零得分的索引

    if len(non_zero_indices) >= num_features_to_select:
        # 如果非零得分的数量足够，按概率选择
        probabilities = scores[non_zero_indices] / scores[non_zero_indices].sum()  # 归一化得分以作为概率
        selected_indices = np.random.choice(non_zero_indices, size=num_features_to_select, replace=False,
                                            p=probabilities)
    else:
        # 如果非零得分的数量不足，从零得分项中补足
        selected_non_zero_indices = np.random.choice(non_zero_indices, size=len(non_zero_indices), replace=False)
        additional_indices_needed = num_features_to_select - len(non_zero_indices)
        selected_zero_indices = np.random.choice(zero_indices, size=additional_indices_needed, replace=False)
        selected_indices = np.concatenate((selected_non_zero_indices, selected_zero_indices))

    return selected_indices


def init_position(lb, ub, dim, num, choose_pb):
    X = np.full(dim, lb, dtype='int')  # 使用整数类型
    num_ub = num
    indices = select_features_by_score(choose_pb, num_ub)
    X[indices] = ub
    return X


def init_position2(lb, ub, dim, num):
    X = np.full(dim, lb, dtype='int')  # 使用整数类型
    num_ub = num
    indices = np.random.choice(dim, num_ub, replace=False)
    X[indices] = ub
    return X


# 按照区间最优个体生成
def init_position3(lb, ub, dim, num, best_x, best_num, choose_pb):
    X = best_x.copy()
    if num > best_num:
        unselected_x = np.where(X == 0)[0]  # 没有被挑选的特征
        choose_num = num - best_num
        pb = choose_pb[unselected_x]
        pb_dim = pb.shape[0]
        # indices = select_features_by_score(pb, choose_num)
        indices = np.random.choice(pb_dim, choose_num, replace=False)
        X[unselected_x[indices]] = ub
    elif num < best_num:
        selected_x = np.where(X == 1)[0]
        choose_num = best_num - num
        pb = 1 - choose_pb[selected_x] + 0.2
        pb_dim = pb.shape[0]
        # indices = select_features_by_score(pb, choose_num)
        indices = np.random.choice(pb_dim, choose_num, replace=False)
        X[selected_x[indices]] = lb
    elif num == best_num:
        X = np.full(dim, lb, dtype='int')  # 使用整数类型
        num_ub = num
        indices = select_features_by_score(choose_pb, num_ub)
        X[indices] = ub
    return X


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


def renew_len_qtable(state_pre, state_cur, action, r, qtable):
    qtable[state_pre, action] = 0.8 * qtable[state_pre, action] + 0.2 * r


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
# 这个版本是使用区间最优个体迭代更新,使用Qlearning进行特征挑选
def fs(xtrain, xvalid, ytrain, yvalid, opts):
    # ********************固定的参数**********************
    dim = xtrain.shape[1]  # 特征的维度
    N = opts['N']
    T = opts['T']
    Max_FEs = T * N
    curve = np.ones([1, Max_FEs + 500], dtype='float')
    curves = np.ones([1, Max_FEs], dtype='float')
    Nsfc = 0
    fitGB = float('inf')
    Nsfb = dim
    fitc = 0
    alpha_rate = 0.9
    Xgb = np.zeros(dim, dtype='int')

    # *******************初始化层*********************
    # num_intervals = 20  # 区间个数
    num_intervals = int(np.round(4 * math.log10(dim+50) + 4))
    min_range = min(5, np.round(dim / num_intervals * 0.5) )  # 区间最小间隔 初始时设置为5
    len_state_num = 4
    len_action_num = 4
    len_qtable = np.ones((len_state_num, len_action_num))
    len_reward_param = [0.5, 0.3, 0.2]  # 长度相关的三种奖励的权重
    gbest_fit_cur = 1  # 当前全局最优的适应度
    gbest_len_cur = dim  # 当前全局最优的特征长度
    gbest_fit_pre = 1  # 上一次全局最优的特征长度
    gbest_len_pre = dim  #

    # ************************* 循环-搜索 **********************************************************************

    # corr_sort = np.argsort(selector)[::-1]
    start_dim = 0  # 当前分区开始的维度
    end_dim = dim  # 当前分区结束的维度
    # dim_divide_flag = np.ones(dim_divide_num)  # 判断空间是否可用
    t = 0
    # agt = FsqAgent(dim)  # 挑选特征的强化学习agent初始化
    # agt.score_init(xtrain, xvalid, ytrain, yvalid)
    state_cur = 0  # 初始化设置状态为0
    step_total = 0  # 总消耗的迭代次数
    valued_num = 20
    # 随机生成一个划分区间，用于最优生成
    # action_cur =
    action_cur = compute_len_action(state_cur, len_qtable)
    intervals = partition_intervals(start_dim, end_dim, num_intervals, action_cur, min_range)
    # print(f"dim = {dim}, num = {num_intervals}, intervals = {intervals} ")
    temp_x = np.zeros((num_intervals, valued_num, dim))
    temp_fitness = np.zeros((num_intervals, valued_num))  # 储存每个区间适应度的变量
    gwo_x = np.random.rand(num_intervals, dim)   # gwo的位置，用于寻找特征
    a_x = np.random.rand(dim)  # alpha狼的位置，用于寻找特征
    aXbin = (a_x >= 0.5).astype(int)
    afit = Fun(xtrain, xvalid, ytrain, yvalid, aXbin, opts)
    b_x = np.random.rand(dim)  # alpha狼的位置，用于寻找特征
    bXbin = (b_x >= 0.5).astype(int)
    bfit = Fun(xtrain, xvalid, ytrain, yvalid, bXbin, opts)
    d_x = np.random.rand(dim)  # alpha狼的位置，用于寻找特征
    dXbin = (d_x >= 0.5).astype(int)
    dfit = Fun(xtrain, xvalid, ytrain, yvalid, dXbin, opts)
    gwo_fitness = np.zeros(num_intervals)    # gwo的适应度
    gwo_xbest = np.zeros((num_intervals, dim))  # gwo的区间最优位置
    gwo_gbest = np.random.binomial(1, 0.5, size=dim)  # gwo的区间最优位置
    g_best = (gwo_gbest >= 0.5).astype(int)
    fitGB = Fun(xtrain, xvalid, ytrain, yvalid, g_best, opts)
    gwo_fitbest = np.zeros(num_intervals)    # gwo的最优适应度
    for i in range(num_intervals - 1, -1, -1):
        # X_cur, Nsfc = init_Xc(dim, intervals[i], intervals[i + 1])
        # fitc = Fun(xtrain, xvalid, ytrain, yvalid, X_cur, opts)
        # Nsfc = int(np.sum(X_cur))
        _,Nsfc = init_Xc(dim, intervals[i], intervals[i + 1])
        gwo_x[i, :] = np.random.rand(dim)
        gwo_xbest[i, :] = np.random.rand(dim)
        X_best = (gwo_xbest[i, :] >= 0.5).astype(int)
        gwo_fitbest[i] = Fun(xtrain, xvalid, ytrain, yvalid, X_best, opts)

        if gwo_fitbest[i] < afit:
            afit = gwo_fitbest[i]
            a_x = gwo_x[i, :].copy()
        elif gwo_fitbest[i] < bfit:
            bfit = gwo_fitbest[i]
            b_x = gwo_x[i, :].copy()
        elif gwo_fitbest[i] < dfit:
            dfit = gwo_fitbest[i]
            d_x = gwo_x[i, :].copy()
        for j in range(valued_num):
            Nsfn = random.randint(intervals[i] + 1, intervals[i + 1])


            newX = gwo_x[i, :].copy()
            for o in range(dim):
                r1, r2 = np.random.random(), np.random.random()
                A = 2 * alpha_rate * r1 - alpha_rate  # alpha调节因子
                C = 2 * r2  # alpha调节因子
                r3 = np.random.random()
                D = 2 * r3

                # 更新特征选择
                newX[o] = gwo_x[i, o] + A * (a_x[o] -gwo_x[i, o]) + C * (b_x[j] - gwo_x[i, o]) + D * (d_x[0] - gwo_x[i, o])

                # 确保新的选择位置在[0, 1]范围内
                if newX[j] > 1:
                    newX[j] = 1
                if newX[j] < 0:
                    newX[j] = 0
            gwo_x[i, :] = newX.copy()

            gwo_x[i, :] = prune_solution(gwo_x[i, :], Nsfn)
            X_cur = (gwo_x[i, :] >= 0.5).astype(int)
            fitc = Fun(xtrain, xvalid, ytrain, yvalid, X_cur, opts)
            temp_x[i, j, :] = X_cur.copy()
            temp_fitness[i, j] = fitc
            if fitc <= gwo_fitbest[i]:
                gwo_xbest[i, :] = gwo_x[i, :].copy()
                gwo_fitbest[i] = fitc
            if fitc < fitGB or (fitc == fitGB and Nsfc < Nsfb):
                gwo_gbest = gwo_x[i, :].copy()
                Xgb = temp_x[i, j, :].copy()  # 不必要的复制，直接赋值即可
                fitGB = fitc
                Nsfb = Nsfc
                gbest_fit_cur = fitc
                gbest_len_cur = Nsfc
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
        action_cur = compute_len_action(state_cur, len_qtable)
        # 根据q值得到的行动使用对应的方式划分区间
        intervals = partition_intervals(start_dim, end_dim, num_intervals, action_cur, min_range)
        temp_x = np.zeros((num_intervals, valued_num, dim))
        temp_fitness = np.zeros((num_intervals, valued_num))  # 储存每个区间适应度的变量
        ##########################################################################
        # 下面两行赋值上一次评价前的最优历史值
        # print(f"结束的t为{t}, intervals 为{intervals}")
        for i in range(num_intervals - 1, -1, -1):
            Nsfc = int(np.sum(X_cur))
            gwo_x[i, :] = np.random.rand(dim)
            gwo_xbest[i, :] = np.random.rand(dim)
            X_best = (gwo_xbest[i, :] >= 0.5).astype(int)
            gwo_fitbest[i] = Fun(xtrain, xvalid, ytrain, yvalid, X_best, opts)
            if gwo_fitbest[i] < afit:
                afit = gwo_fitbest[i]
                a_x = gwo_x[i, :].copy()
            elif gwo_fitbest[i] < bfit:
                bfit = gwo_fitbest[i]
                b_x = gwo_x[i, :].copy()
            elif gwo_fitbest[i] < dfit:
                dfit = gwo_fitbest[i]
                d_x = gwo_x[i, :].copy()
            for j in range(valued_num):
                Nsfn = random.randint(intervals[i] + 1, intervals[i + 1])
                newX = gwo_x[i, :].copy()
                for o in range(dim):
                    r1, r2 = np.random.random(), np.random.random()
                    A = 2 * alpha_rate * r1 - alpha_rate  # alpha调节因子
                    C = 2 * r2  # alpha调节因子
                    r3 = np.random.random()
                    D = 2 * r3

                    # 更新特征选择
                    newX[o] = gwo_x[i, o] + A * (a_x[o] - gwo_x[i, o]) + C * (b_x[o] - gwo_x[i, o]) + D * (
                                d_x[o] - gwo_x[i, o])

                    # 确保新的选择位置在[0, 1]范围内
                    if newX[j] > 1:
                        newX[j] = 1
                    if newX[j] < 0:
                        newX[j] = 0
                gwo_x[i, :] = newX.copy()

                gwo_x[i, :] = prune_solution(gwo_x[i, :], Nsfn)
                X_cur = (gwo_x[i, :] >= 0.5).astype(int)
                fitc = Fun(xtrain, xvalid, ytrain, yvalid, X_cur, opts)
                temp_x[i, j, :] = X_cur.copy()
                temp_fitness[i, j] = fitc
                if fitc <= gwo_fitbest[i]:
                    gwo_xbest[i, :] = gwo_x[i, :].copy()
                    gwo_fitbest[i] = fitc
                if fitc < fitGB or (fitc == fitGB and Nsfc < Nsfb):
                    gwo_gbest = gwo_x[i, :].copy()
                    Xgb = temp_x[i, j, :].copy()  # 不必要的复制，直接赋值即可
                    fitGB = fitc
                    Nsfb = Nsfc
                    gbest_fit_cur = fitc
                    gbest_len_cur = Nsfc
                curve[0, t] = fitGB
                t = t + 1

            #################################################################
        k = valued_num // 4  # 四分点
        value_fitness_quarter = np.partition(temp_fitness, k, axis=1)[:, k]  # 寻找适应度最小的四等分点
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

        renew_len_qtable(state_pre, state_cur, action_cur, len_r, len_qtable)
    # print(f"结束的t为{t}, intervals 为{intervals}, 最优长度{Nsfb}, 最优适应度{fitGB}")




    while (t < Max_FEs):
        for i in range(num_intervals - 1, -1, -1):
            if intervals[i] >= intervals[i + 1]:
                intervals[i + 1] = intervals[i] + min_range
            gwo_x[i, :] = np.random.rand(dim)
            gwo_xbest[i, :] = np.random.rand(dim)
            X_best = (gwo_xbest[i, :] >= 0.5).astype(int)
            gwo_fitbest[i] = Fun(xtrain, xvalid, ytrain, yvalid, X_best, opts)
            if gwo_fitbest[i] < afit:
                afit = gwo_fitbest[i]
                a_x = gwo_x[i, :].copy()
            elif gwo_fitbest[i] < bfit:
                bfit = gwo_fitbest[i]
                b_x = gwo_x[i, :].copy()
            elif gwo_fitbest[i] < dfit:
                dfit = gwo_fitbest[i]
                d_x = gwo_x[i, :].copy()
            for j in range(valued_num):
                Nsfn = random.randint(intervals[i] + 1, intervals[i + 1])
                newX = gwo_x[i, :].copy()
                for o in range(dim):
                    r1, r2 = np.random.random(), np.random.random()
                    A = 2 * alpha_rate * r1 - alpha_rate  # alpha调节因子
                    C = 2 * r2  # alpha调节因子
                    r3 = np.random.random()
                    D = 2 * r3

                    # 更新特征选择
                    newX[o] = gwo_x[i, o] + A * (a_x[o] - gwo_x[i, o]) + C * (b_x[o] - gwo_x[i, o]) + D * (
                            d_x[o] - gwo_x[i, o])

                    # 确保新的选择位置在[0, 1]范围内
                    if newX[j] > 1:
                        newX[j] = 1
                    if newX[j] < 0:
                        newX[j] = 0
                gwo_x[i, :] = newX.copy()

                gwo_x[i, :] = prune_solution(gwo_x[i, :], Nsfn)
                X_cur = (gwo_x[i, :] >= 0.5).astype(int)
                fitc = Fun(xtrain, xvalid, ytrain, yvalid, X_cur, opts)
                temp_x[i, j, :] = X_cur.copy()
                temp_fitness[i, j] = fitc
                if fitc <= gwo_fitbest[i]:
                    gwo_xbest[i, :] = gwo_x[i, :].copy()
                    gwo_fitbest[i] = fitc
                if fitc < fitGB or (fitc == fitGB and Nsfc < Nsfb):
                    gwo_gbest = gwo_x[i, :].copy()
                    Xgb = temp_x[i, j, :].copy()  # 不必要的复制，直接赋值即可
                    fitGB = fitc
                    Nsfb = Nsfc
                    gbest_fit_cur = fitc
                    gbest_len_cur = Nsfc
                curve[0, t] = fitGB
                t = t + 1
    # print(f"best = {fitGB},{agt.param},最优长度{Nsfb}, 最优适应度{fitGB}")
    # print(t)

    # for i in range()

    # curve[0, -(Max_FEs - step_total):] = curve[0, -(Max_FEs - step_total + 1)]
    curves[0, :] = curve[0, :Max_FEs]

    num_points = opts['T']
    # 等差选取数据点的索引
    indices = np.round(np.linspace(0, Max_FEs - 1, num_points)).astype(int)
    # 从Cost中选取数据填入curve
    curves = curves[:, indices]
    # 将curve变形为所需的大小
    curves = curves.reshape(1, -1)
    ceeh_data = {'sf': Xgb, 'c': curves, 'nf': Nsfb}
    return ceeh_data
