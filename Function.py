import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from CCM import calculate_metrics

def Fun(X_train, X_valid, Y_train, Y_valid, x, opts):
    # Parameters
    alpha = float(opts.get('phi', 0.99))
    if alpha < 0:
        alpha = 0.0
    if alpha > 1:
        alpha = 1.0
    beta = 1 - alpha
    # Original feature size
    max_feat = len(x)
    if 'dim' in opts:
        max_feat = opts['dim']
    # Number of selected features
    num_feat = np.sum(x == 1)
    # Solve if no feature selected
    if num_feat == 0:
        cost = 1
    else:
        # Get error rate using cross validation
        error = cross_validation_value(X_train, X_valid, Y_train, Y_valid, x, opts)
        # Objective function
        cost = alpha * error + beta * (num_feat / max_feat)
    return cost

# error rate using cross validation
def cross_validation_value(X_train, X_valid, Y_train, Y_valid, x, opts):
    split = opts['split']
    # 当split=0时，使用分开固定的一个训练集及验证集划分；当split为其他时，将训练及验证混合起来使用k折交叉验证
    if split == 0:
        error_sum = error_rate(X_train, X_valid, Y_train, Y_valid, x, opts)
        num_splits = 1
    else:
        num_splits = split
        error_sum = 0
        X = np.vstack((X_train, X_valid))  # Combine X_train and X_valid
        Y = np.hstack((Y_train, Y_valid))  # Combine Y_train and Y_valid
        skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
        for train_index, valid_index in skf.split(X, Y):
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = Y[train_index], Y[valid_index]
            error_sum += error_rate(X_train, X_valid, y_train, y_valid, x, opts)
    return error_sum / num_splits

# error rate
def error_rate(X_train, X_valid, Y_train, Y_valid, x, opts):
    x = x.flatten()
    # parameters
    # Number of instances
    num_train = np.size(X_train, 0)
    num_valid = np.size(X_valid, 0)
    # Define selected features
    X_train = X_train[:, x == 1]
    # Y_train = Y_train.reshape(num_train)  # Solve bug
    Y_train = pd.DataFrame(Y_train).to_numpy().reshape(-1, )
    X_valid = X_valid[:, x == 1]
    # Y_valid = Y_valid.reshape(num_valid)  # Solve bug
    Y_valid = pd.DataFrame(Y_valid).to_numpy().reshape(-1, )
    # Training Prediction
    ypred = classifier_method(X_train, X_valid, Y_train, Y_valid, opts)
    if opts['func'] == 0:
        acc = np.sum(Y_valid == ypred) / num_valid
        error = 1 - acc
    elif opts['func'] == 1:
        acc = np.sum(Y_valid == ypred) / num_valid
        error1 = 1 - acc
        auc = calculate_metrics(Y_valid, ypred, 'roc_auc')
        error2 = 1 - auc['ROC AUC (Macro)']
        error = 0.6 * error1 + 0.4 * error2
    elif opts['func'] == 2:
        acc = np.sum(Y_valid == ypred) / num_valid
        error1 = 1 - acc
        f1 = calculate_metrics(Y_valid, ypred, 'f1')
        error2 = 1 - f1['F1Score (Macro)']
        error = 0.7 * error1 + 0.3 * error2
    elif opts['func'] == 4:
        x_all = np.vstack([X_train, X_valid])
        y_all = np.concatenate((Y_train, Y_valid))
        silhouette = silhouette_score(x_all, ypred)
        calinski_harabasz = calinski_harabasz_score(x_all, ypred)
        davies_bouldin = davies_bouldin_score(x_all, ypred)
        weights = [1, 0.3, 0.6]  # 权重可以根据实际需求调整
        error = 1 - 0.005*calinski_harabasz+silhouette*0.5
    return error

def classifier_method(X_train, X_valid, Y_train, Y_valid,opts):
    # 根据 opts['classify'] 的取值选择相应的分类器
    if opts['classify'] == 'knn':  # K最近邻（KNN）算法
        # 在 KNN 中，para 表示选取最近的 para 个邻居来进行投票
        para = opts.get('knn_para', 3)
        classifier = KNeighborsClassifier(n_neighbors=para)
    elif opts['classify'] == 'svm':  # 支持向量机（SVM）算法
        # 在 SVM 中，para 不是算法的一部分，而是用于 RBF 核函数的参数，代表高斯核的宽度
        para = opts.get('svm_para', 1.0)
        classifier = SVC(kernel='rbf', C=para)
    elif opts['classify'] == 'rf':  # 随机森林（RF）算法
        # 在随机森林中，para 代表森林中树木的数量
        para = opts.get('rf_para', 100)
        classifier = RandomForestClassifier(n_estimators=para)
    elif opts['classify'] == 'dt':  # 决策树（Decision Tree）算法
        # 在决策树中，para 可以代表树的深度、节点最少样本数等参数
        para = opts.get('dt_para', None)
        classifier = DecisionTreeClassifier(max_depth=para)
    elif opts['classify'] == 'lr':  # 逻辑回归（Logistic Regression）算法
        # 在逻辑回归中，通常不涉及 para 参数
        classifier = LogisticRegression()
    # else:
    #     raise ValueError("Invalid classifier specified in opts['classify']")
    if opts['cluster'] == 'kmeans':
        x_all = np.vstack([X_train, X_valid])
        y_all = np.concatenate((Y_train, Y_valid))
        # kmeans = KMeans(n_clusters=len(np.unique(y_all)))  # 假设我们希望聚类成3个簇
        # 对 X_train 数据进行聚类
        dbscan = DBSCAN(eps=0.5, min_samples=len(np.unique(y_all)))
        # 3. 对数据进行聚类
        labels = dbscan.fit_predict(x_all)
        # kmeans.fit(x_all)
        # # 输出预测标签
        Y_pred = labels
    else:
        mdl = classifier
        mdl.fit(X_train, Y_train)
        Y_pred = mdl.predict(X_valid)
    return Y_pred

def calculate_model_complexity(model, opts, X):
    n, m = X.shape    # n 为样本数，m 为特征数
    # 对于每个特征，计算取值范围内的离散点数量
    features_discretization = []
    for j in range(m):
        unique_values = np.unique(X[:, j])
        if len(unique_values) > 1:
            feature_range = max(unique_values) - min(unique_values)
            feature_discretization = int(np.ceil(feature_range / (0.1 * feature_range)))
        else:
            feature_discretization = 1
        features_discretization.append(feature_discretization)
    # 计算网格的总像素数
    N = 1
    for d in features_discretization:
        N *= d
    # 计算分类边界复杂度
    h = .02  # 步长
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    boundary_area = np.sum(Z != Z[1:])
    complexity = boundary_area / N
    # if opts['classify'] == 'knn' or opts['classify'] == 'svm' or opts['classify'] == 'lr':
    #     boundary_length = np.sum(np.abs(np.diff(model.predict(X).reshape(-1, 1), axis=0)))
    #     complexity = boundary_length / N
    # elif opts['classify'] == 'dt':
    #     boundary_area = np.sum(model.predict(X) != model.predict(X[1:]))
    #     complexity = boundary_area / N
    # else:
    #     h = .02  # 步长
    #     x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    #     y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    #     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #     Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    #     boundary_area = np.sum(Z != Z[1:])
    #     complexity = boundary_area / N
    return complexity
