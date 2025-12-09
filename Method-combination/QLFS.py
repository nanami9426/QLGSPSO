import numpy as np
from Function import Fun
from numpy.random import rand
import math
import warnings


def qlfs(train_data, labels):
    """
    Q-learning based Feature Selection (QLFS) algorithm.

    :param train_data: mxn numpy array, where m is the number of samples and n is the number of features.
    :param labels: 1-dimensional numpy array with length m, containing the labels for each sample.
    :return: ranked: 1-dimensional numpy array with the ranking of features.
    """
    # 转置特征矩阵，使其成为(n, m)的形状
    train_data = train_data.T

    # 特征数量和样本数量
    feature = train_data.shape[0]
    number = train_data.shape[1]

    # 初始化状态矩阵，添加一列用于时间步长
    state = np.zeros((feature + 1, number))
    state[0, :] = 1  # 第一列为全1，用于偏置项

    # 填充状态矩阵，将特征数据填充到状态矩阵中
    state[1:, :] = train_data

    # 初始化权重矩阵和排名数组
    c = np.unique(labels).size  # 计算类别数
    w = np.zeros((c, feature + 1))  # 权重矩阵，添加一列用于偏置项
    ranked = np.zeros(feature)  # 特征排名数组

    # 学习率
    alpha = 30

    # 主循环
    for t in range(10):  # 根据需要设置迭代次数
        w = np.zeros((c, feature + 1))  # 重置权重矩阵
        for k in range(1000):  # 外部循环，可以设置为需要的迭代次数
            w0 = w.copy()  # 保存当前权重矩阵状态

            for i1 in range(number):  # 内部循环
                # 随机选择动作，探索概率为0.8
                if np.random.rand() > 0.8:
                    v = np.dot(w, state[:, i1])  # 计算所有类别的判别分数
                    d = np.argmax(v)  # 选择最大判别分数的类别
                else:
                    d = np.random.randint(0, c)  # 随机选择类别

                # 计算奖励
                R = 1 if d == labels[i1] else -1
                if i1 < number - 1:
                    R += 0.1 * np.max(np.dot(w, state[:, i1 + 1]))

                # 更新权重
                for z in range(c):
                    if z == d:
                        temp = (R - np.dot(w[z, :], state[:, i1])) * state[:, i1]
                        w[z, :] += (1 / (alpha * ((i1 + 1) ** 0.2))) * temp
                        break

            # 如果权重矩阵变化很小，则提前结束循环
            if np.linalg.norm(w - w0) < 0.000001:
                break

        # 特征选择和排名逻辑
        # 这里需要根据QLFS算法的逻辑来实现特征的删除和排名
        # 以下是一个示例逻辑，可能需要根据实际算法调整
        w_abs = np.abs(w)
        w_abs_sum = np.sum(w_abs, axis=0)[:-1]
        ranked[w_abs_sum.argsort()[::-1]] = np.arange(feature)  # 根据权重的绝对值和进行排名

    return ranked


def QLFS(train_new_data):
    feature = train_new_data.shape[1]  # 特征数量
    number = train_new_data.shape[0]  # 样本数量

    state = np.zeros((feature+1, number))
    state[0, :] = 1  # 第一行是全1，偏置项

    train_new_data = train_new_data.T  # 转置数据
    state[1:, :] = train_new_data  # 填充特征数据

    c = int(np.max(state[-1, :]))+1  # 计算类别数量
    w = np.zeros((c, feature))  # 初始化权重矩阵
    ranked = np.zeros(feature - 1)  # 初始化特征排名数组

    t = 0
    maxiter = 1
    c1 = np.random.permutation(number)  # 随机排列样本索引
    state1 = state.copy()
    for paixu in range(number):
        state[:, paixu] = state1[:, c1[paixu]]  # 打乱样本顺序

    alpha = 100  # 学习率

    while t != maxiter:
        w = np.zeros((c, feature))  # 重置权重矩阵
        for k in range(10):  # 外部循环
            w0 = w.copy()
            for i1 in range(number):  # 内部循环
                if np.random.rand() < 0.8:  # 选择动作
                    v = np.dot(w, state[:-1, i1])  # 计算所有类别的判别分数
                    d = np.argmax(v)  # 选择最大判别分数的类别
                else:
                    d = np.random.randint(0, c)  # 随机选择类别

                # 计算奖励
                if d == state[-1, i1]:
                    R = 1
                else:
                    R = -1

                # 计算下一步的奖励
                if i1 < number - 1:
                    v_next = np.dot(w, state[:-1, i1 + 1])
                    R += 0.1 * np.max(v_next)

                # 更新权重
                tidu = (R - np.dot(w[d, :], state[:-1, i1]))  # 更新值
                w[d, :] += (1 / (alpha * (i1 + 1) ** 0.2)) * tidu * state[:-1, i1]

            # 如果权重变化非常小，则停止
            if np.linalg.norm(w - w0) < 0.00001:
                break

        # 删除特征逻辑
        WW = np.abs(w[:, 1:])  # 去掉偏置项列
        WW = WW ** 2  # 权重的平方
        WW = np.sum(WW, axis=0)  # 对每个特征求和
        sum_ww = np.sum(WW)  # 所有特征权重之和
        del_idx = np.where(WW <= (sum_ww / (feature - 1)))[0]  # 删除特征
        zero = np.where(WW == 0)[0]  # 找到权重为0的特征
        t = len(zero)

        if t != len(del_idx):
            for idx in del_idx:
                state[idx + 1, :] = 0  # 删除特征
            del_idx_sorted = np.argsort(WW[del_idx])  # 排序删除的特征
            for i in range(len(del_idx) - t):
                ranked[i + t] = del_idx[del_idx_sorted[i + t]]
        else:
            ranked[:t] = np.argsort(WW)[:t]  # 排序所有特征

        maxiter = len(del_idx)

    return ranked


def fs(xtrain, xvalid, ytrain, yvalid, opts):
    # x = np.concatenate((xtrain, xvalid), axis=0)
    # y = np.concatenate((ytrain, yvalid), axis=0)
    x = xtrain.copy()
    y = ytrain.copy()
    # train_data = np.concatenate((x, y), axis=1)
    X_with_labels = np.hstack((x, y.reshape(-1, 1)))
    # rank = qlfs(x, y)
    rank = QLFS(X_with_labels)
    dim = np.size(xtrain, 1)
    N = opts['N']
    max_iter = opts['T']
    value_num = int(dim * 0.4)
    curve = np.zeros([1, value_num], dtype='float')
    Xpb = np.zeros([value_num, dim], dtype='int')
    # fitP = float('inf') * np.ones([max_iter, 1], dtype='float')
    Xgb = np.zeros([1, dim], dtype='int')
    fitG = float('inf')
    for i in range(value_num):
        sf = rank[:value_num - i - 1]
        Xpb[i, sf.astype(int)] = 1
        fitP = Fun(xtrain, xvalid, ytrain, yvalid, Xpb[i, :], opts)
        if fitP <= fitG:
            Xgb[0, :] = Xpb[i, :]
            fitG = fitP
        curve[0, i] = fitG.copy()
    num_feat = np.sum(Xgb[0, :])
    expanded_vector = np.zeros((1,N * max_iter))

    # 计算每个元素需要重复的次数，如果n是max_n的因数，则每个元素重复n次；否则，重复次数为n-1
    repeat_times = (N * max_iter // value_num) + 1

    # 填充扩展后的向量
    for i in range(value_num):
        # 将当前元素值重复指定次数并添加到扩展向量中
        expanded_vector[0, i * repeat_times:(i + 1) * repeat_times] = curve[0, i]

    num_points = opts['T']
    # 等差选取数据点的索引
    indices = np.round(np.linspace(0, N * max_iter - 1, num_points)).astype(int)
    # 从Cost中选取数据填入curve
    expanded_vector = expanded_vector[:, indices]
    # 将curve变形为所需的大小
    expanded_vector = expanded_vector.reshape(1, -1)
    inf_data = {'sf': Xgb, 'c': expanded_vector, 'nf': num_feat}

    return inf_data

