import sys
import time
import importlib
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import MinMaxScaler
from CCM import calculate_metrics
from Save import *
from Draw import *
from Function import *
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore", category=UserWarning)
# 忽略所有警告
warnings.filterwarnings("ignore")


def map_labels_hungarian(true_labels, cluster_labels):
    unique_true = np.unique(true_labels)
    unique_cluster = np.unique(cluster_labels)
    cost_matrix = np.zeros((len(unique_true), len(unique_cluster)))

    for i, true in enumerate(unique_true):
        for j, cluster in enumerate(unique_cluster):
            mask = (cluster_labels == cluster)
            true_labels_cluster = true_labels[mask]
            cost_matrix[i, j] = -np.sum(true_labels_cluster == true)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    label_map = {cluster: true for cluster, true in zip(unique_cluster[col_ind], unique_true[row_ind])}

    return label_map

def map_labels(true_labels, cluster_labels):
    unique_true = np.unique(true_labels)
    unique_cluster = np.unique(cluster_labels)
    label_map = {}

    for cluster in unique_cluster:
        mask = (cluster_labels == cluster)
        true_labels_cluster = true_labels[mask]
        counts = np.bincount(true_labels_cluster)
        max_count = np.argmax(counts)
        label_map[cluster] = max_count

    return label_map
# 多线程处理函数
def process_name(data_name, method_list, opts, runs, current_date, save_index = False):
    result_list = []  # 创建一个空列表存放算法结果
    result_curve = []  # 创建一个空列表存放curve画图数据
    results = {}  # 创建一个空字典用于保存结果
    result_run = []  # 初始化列表保存每次循环的值
    if save_index:
        result_index = [] # 初始化列表保存每次结果存储的特征index
    method_num = len(method_list)
    result_run_best = np.zeros((runs, method_num))  # 创建迭代最优保存
    result_run_best_name = np.array([['string{}'.format(i + 1)] for i in range(runs)])  # 创建迭代最优保存的行名
    results[data_name] = {}  # 在字典中创建一个新的键值对，值是一个空字典
    path = 'Dataset'
    Maxiter = opts['T']
    # 读取数据
    data = pd.read_csv("%s/%s.csv" % (path, data_name))
    # 有时候上面读取不到要换成下面这个
    # data = pd.read_csv("%s\\%s.csv" % (path, data_name))
    X = data.iloc[:, :-1].values.copy()
    Y = data.iloc[:, -1].values.copy()
    # 将一维数据转化为二维数据
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    X = np.nan_to_num(X)
    scaler = MinMaxScaler(feature_range=(0, 1))  # 归一化到 [0,1] 范围
    X = scaler.fit_transform(X)

    # # 对每一列进行归一化
    # for i in range(X.shape[1]):
    #     col = X[:, i]
    #     min_val = np.min(col)
    #     max_val = np.max(col)
    #     denominator = max_val - min_val
    #     denominator = denominator if denominator != 0 else 1e-8  # 避免除零错误
    #     X[:, i] = np.divide((col - min_val), denominator)

    X = np.nan_to_num(X)
    sample = np.size(X, 0)
    dim = np.size(X, 1)
    result_run_best_name[runs * 0:runs * (0 + 1), 0] = data_name
    print("数据集【%s】，样本数量：%d；特征数量：%d" % (data_name, sample, dim))
    # print("【%s：%s】" % (data_name, methodset))
    for m, method in enumerate(method_list):
        # 进行特征选择
        sf = np.zeros((runs, dim))
        nf = np.zeros((runs, 1))
        curve = np.zeros((runs, Maxiter))
        timecal = np.zeros((runs, 1))
        acc_valid = np.zeros((runs, 1))
        acc_test = np.zeros((runs, 1))
        f1_valid = np.zeros((runs, 1))
        f1_test = np.zeros((runs, 1))
        auc_valid = np.zeros((runs, 1))
        auc_test = np.zeros((runs, 1))
        recall_valid = np.zeros((runs, 1))
        recall_test = np.zeros((runs, 1))
        precision_valid = np.zeros((runs, 1))
        precision_test = np.zeros((runs, 1))

        for run in range(runs):
            # 划分数据（训练：验证： 测试 = 6：2：2）
            random_seed = run
            # # # 先将数据分成训练集和测试集
            # X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=random_seed)
            # # 再将训练分成训练集和验证集
            # X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_val, Y_train_val, test_size=0.2, stratify=Y_train_val, random_state=random_seed)
            # 只划分训练集和测试集（8：2）

            X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.3, stratify=Y,
                                                                  random_state=random_seed)
            X_test = X_valid.copy()
            Y_test = Y_valid.copy()
            opts['random_seed'] = random_seed
            module = importlib.import_module(method)  # 动态加载模块
            fs_method = getattr(module, 'fs', None)  # 获取函数对象
            if fs_method is not None and callable(fs_method):
                # 记录开始时间
                start_time = time.time()
                # 条件判断语句，用于检查获取到的方法对象是否不为空且可调用。
                FS = fs_method(X_train.copy(), X_valid.copy(), Y_train.copy(), Y_valid.copy(), opts)
                # 记录结束时间
                end_time = time.time()
                # 记录部分结果
                sf[run, :] = FS['sf']
                nsf = sf[run, :]
                nf[run, 0] = FS['nf']
                curve[run, :] = FS['c']
                timecal[run, 0] = end_time - start_time
                if opts['classify'] == 'kmeans':
                    pred_valid = classifier_method(X_train[:, nsf == 1], X_valid[:, nsf == 1], Y_train, Y_valid, opts)  # 训练集和验证集
                    pred_test  = classifier_method(X_train[:, nsf == 1], X_test[:, nsf == 1], Y_train, Y_valid, opts)  # 测试集
                    y_all = np.concatenate((Y_train, Y_valid))
                    pred_test_map = map_labels_hungarian(y_all, pred_test)
                    pred_test = np.array([pred_test_map[label] for label in pred_test])
                    pred_valid_map = map_labels_hungarian(y_all, pred_valid)
                    pred_valid = np.array([pred_valid_map[label] for label in pred_valid ])
                    acc_valid[run, 0] = calculate_metrics(y_all, pred_valid, metric='accuracy')
                    acc_test[run, 0] = calculate_metrics(y_all, pred_test, metric='accuracy')
                    f1_valid_metrics = calculate_metrics(y_all, pred_valid, metric='f1')
                    f1_valid[run, 0] = f1_valid_metrics['F1Score (Macro)']
                    f1_test_metrics = calculate_metrics(y_all, pred_valid, metric='f1')
                    f1_test[run, 0] = f1_test_metrics['F1Score (Macro)']
                    auc_valid_metrics = calculate_metrics(y_all, pred_valid, metric='roc_auc')
                    auc_valid[run, 0] = auc_valid_metrics['ROC AUC (Macro)']
                    auc_test_metrics = calculate_metrics(y_all, pred_test, metric='roc_auc')
                    auc_test[run, 0] = auc_test_metrics['ROC AUC (Macro)']
                    recall_valid_metrics = calculate_metrics(y_all, pred_valid, metric='recall')
                    recall_valid[run, 0] = recall_valid_metrics['Recall (Macro)']
                    recall_test_metrics = calculate_metrics(y_all, pred_test, metric='recall')
                    recall_test[run, 0] = recall_test_metrics['Recall (Macro)']
                    precision_valid_metrics = calculate_metrics(y_all, pred_valid, metric='precision')
                    precision_valid[run, 0] = precision_valid_metrics['Precision (Macro)']
                    precision_test_metrics = calculate_metrics(y_all, pred_test, metric='precision')
                    precision_test[run, 0] = precision_test_metrics['Precision (Macro)']
                    run_result = {
                        'Data Name': data_name,
                        'Method': method,
                        'Run': run + 1,  # 通常我们从1开始计数
                        'Feature Number': nf[run, 0],
                        'Time Calculation': timecal[run, 0],
                        'Accuracy Valid': acc_valid[run, 0],
                        'Accuracy Test': acc_test[run, 0],
                        'F1 Valid': f1_valid[run, 0],
                        'F1 Test': f1_test[run, 0],
                        'AUC Valid': auc_valid[run, 0],
                        'AUC Test': auc_test[run, 0],
                        'Recall Valid': recall_valid[run, 0],
                        'Recall Test': recall_test[run, 0],
                        'Precision Valid': precision_valid[run, 0],
                        'Precision Test': precision_test[run, 0]
                    }
                    result_run.append(run_result)
                    if save_index:
                        run_index = {'Data Name': data_name,
                                     'Method': method,
                                     'index': np.where(nsf == 1)[0]}
                        result_index.append(run_index)

                    continue
                # 得到模型预测
                pred_valid = classifier_method(X_train[:, nsf == 1], X_valid[:, nsf == 1], Y_train, Y_valid, opts)  # 训练集和验证集
                pred_test = classifier_method(X_train[:, nsf == 1], X_test[:, nsf == 1], Y_train, Y_valid, opts)  # 测试集

                acc_valid[run, 0] = calculate_metrics(Y_valid, pred_valid, metric='accuracy')
                acc_test[run, 0] = calculate_metrics(Y_test, pred_test, metric='accuracy')
                f1_valid_metrics = calculate_metrics(Y_valid, pred_valid, metric='f1')
                f1_valid[run, 0] = f1_valid_metrics['F1Score (Macro)']
                f1_test_metrics = calculate_metrics(Y_valid, pred_valid, metric='f1')
                f1_test[run, 0] = f1_test_metrics['F1Score (Macro)']
                auc_valid_metrics = calculate_metrics(Y_valid, pred_valid, metric='roc_auc')
                auc_valid[run, 0] = auc_valid_metrics['ROC AUC (Macro)']
                auc_test_metrics = calculate_metrics(Y_test, pred_test, metric='roc_auc')
                auc_test[run, 0] = auc_test_metrics['ROC AUC (Macro)']
                recall_valid_metrics = calculate_metrics(Y_valid, pred_valid, metric='recall')
                recall_valid[run, 0] = recall_valid_metrics['Recall (Macro)']
                recall_test_metrics = calculate_metrics(Y_test, pred_test, metric='recall')
                recall_test[run, 0] = recall_test_metrics['Recall (Macro)']
                precision_valid_metrics = calculate_metrics(Y_valid, pred_valid, metric='precision')
                precision_valid[run, 0] = precision_valid_metrics['Precision (Macro)']
                precision_test_metrics = calculate_metrics(Y_test, pred_test, metric='precision')
                precision_test[run, 0] = precision_test_metrics['Precision (Macro)']
            else:
                print("没有调取到相关算法，请检查算法名称或算法包")
            # 收集每次运行的详细结果，并创建一个字典
            run_result = {
                'Data Name': data_name,
                'Method': method,
                'Run': run + 1,  # 通常我们从1开始计数
                'Feature Number': nf[run, 0],
                'Time Calculation': timecal[run, 0],
                'Accuracy Valid': acc_valid[run, 0],
                'Accuracy Test': acc_test[run, 0],
                'F1 Valid': f1_valid[run, 0],
                'F1 Test': f1_test[run, 0],
                'AUC Valid': auc_valid[run, 0],
                'AUC Test': auc_test[run, 0],
                'Recall Valid': recall_valid[run, 0],
                'Recall Test': recall_test[run, 0],
                'Precision Valid': precision_valid[run, 0],
                'Precision Test': precision_test[run, 0]
            }
            result_run.append(run_result)
            if save_index:
                run_index = {'Data Name': data_name,
                             'Method': method,
                             'index':np.where(nsf == 1)[0]}
                result_index.append(run_index)

        best_indices = np.argmin(curve[:, -1])  # 默认适应度函数为最小最优
        sf_best = sf[best_indices, :]
        curve_best = curve[best_indices, :]
        curve_mean = np.mean(curve, axis=0, keepdims=True)
        curve_std = np.std(curve, axis=0, keepdims=True)
        fitness_best = curve_best[-1]
        fitness_mean = curve_mean[0, -1]
        acc_valid_best = np.max(acc_valid)
        acc_test_best = np.max(acc_test)
        nf_mean = np.mean(nf)
        timecal_mean = np.mean(timecal)
        acc_valid_mean = np.mean(acc_valid)
        acc_test_mean = np.mean(acc_test)
        acc_valid_min = np.min(acc_valid)
        acc_test_min = np.min(acc_test)
        nf_std = np.std(nf)
        timecal_std = np.std(timecal)
        acc_valid_std = np.std(acc_valid)
        acc_test_std = np.std(acc_test)
        acc_gen = acc_test_mean - acc_valid_mean
        f1_valid_mean = np.mean(f1_valid)
        f1_test_mean = np.mean(f1_test)
        auc_valid_mean = np.mean(auc_valid)
        auc_test_mean = np.mean(auc_test)
        auc_gen = auc_test_mean - auc_valid_mean
        recall_valid_mean = np.mean(recall_valid)
        recall_test_mean = np.mean(recall_test)
        precision_valid_mean = np.mean(precision_valid)
        precision_test_mean = np.mean(precision_test)
        result_run_best[runs * 0:runs * (0 + 1), m] = np.copy(acc_test[:, 0])
        # 将所有结果添加到列表中
        # result_list.append(
        #     [data_name, method, fitness_best, fitness_mean, acc_valid_best, acc_test_best, nf_mean, nf_std,
        #      timecal_mean, timecal_std, acc_valid_mean, acc_valid_std, acc_test_mean, acc_test_std, acc_gen,
        #      auc_valid_mean, auc_test_mean, auc_gen])
        result_list.append(
            [data_name, method, nf_mean, nf_std, acc_test_best, acc_test_mean, acc_test_std, acc_test_min, f1_test_mean,
             auc_test_mean, recall_test_mean, precision_test_mean, timecal_mean])
        result_curve_mean = [data_name, method] + curve_mean.tolist()[0]  # 生成要添加到 result_curve 中的数据
        result_curve.append(result_curve_mean)  # 将数据添加到 result_curve 中
        # 最后进行分类预测得到其他指标
        valid_pred = classifier_method(X_train[:, sf_best == 1], X_valid[:, sf_best == 1], Y_train,Y_valid, opts)
        test_pred = classifier_method(X_train[:, sf_best == 1], X_test[:, sf_best == 1], Y_train,Y_valid, opts)

        # 进行算法评估指标计算
        if opts['classify'] == 'kmeans':
            y_all = np.concatenate((Y_train, Y_valid))
            # x_all = np.vstack([X_train, X_valid])

            pred_test_map = map_labels_hungarian(y_all, pred_test)
            pred_test = np.array([pred_test_map[label] for label in pred_test])
            pred_valid_map = map_labels_hungarian(y_all, pred_test)
            pred_valid = np.array([pred_valid_map[label] for label in pred_valid])
            map_labels(y_all, valid_pred)
            valid_result = calculate_metrics(y_all, valid_pred, 'all')
            test_result = calculate_metrics(y_all, test_pred, 'all')
        else:
            valid_result = calculate_metrics(Y_valid, valid_pred, 'all')
            test_result = calculate_metrics(Y_test, test_pred, 'all')
        result = {'sf': sf, 'nf': nf, 'timecal': timecal, 'acc_valid': acc_valid, 'acc_test': acc_test,
                  'f1_valid': f1_valid, 'f1_test': f1_test, 'auc_valid': auc_valid, 'auc_test': auc_test,
                  'valid_result': valid_result, 'test_result': test_result, 'curve_mean': curve_mean}
        results[data_name][method] = result  # 将结果保存到字典中
        print(
            "【%s】【%-8s】NF: %.2f ;   1-Fitness: %.2f ;   Acc_test: %.2f ± %.2f ;   F1_test: %.2f%%;  Auc_test:%.2f%%;  Time: %.2f"
            % (data_name, method, nf_mean, 1 - fitness_mean, acc_test_mean, acc_test_std, f1_test_mean * 100,
               auc_test_mean * 100, timecal_mean))
        # print("NF: %d ± %.2f ;   Acc_valid: %.2f ± %.2f ;   Acc_test: %.2f ± %.2f ;   Time: %.2f" % (nf_mean, nf_std, acc_valid_mean, acc_valid_std, acc_test_mean, acc_test_std, timecal_mean))

    save_result_many(data_name, results, result_list, result_curve, result_run, result_run_best, result_run_best_name, method_list, current_date,result_index)

def main():
    # 忽略特定类型的警告
    warnings.filterwarnings("ignore", category=UserWarning)
    # 忽略所有警告
    warnings.filterwarnings("ignore")

    # 添加文件夹路径
    sys.path.append('Method-tradition')
    sys.path.append('Method-contrast')
    sys.path.append('Method-advanced')
    sys.path.append('Method-transfer')
    sys.path.append('Generalization')
    sys.path.append('Method-combination')
    sys.path.append('Method-add')
    path = 'Dataset'  # 数据集文件夹路径
    file_names = os.listdir(path)  # 获取文件夹中所有文件名
    sorted_file_names = sorted(file_names)  # 对文件名进行排序
    # 数据集合

    # 运行的数据集
    dataset = sorted_file_names
    dataset = ['Colon', 'BT1']
    dataset = [filename.rstrip('.csv') for filename in dataset]  # 去除.csv后缀
    data_num = len(dataset)
    # methodset = ['rlpsoasm', "tmgwo", "essa", "SFE", "PSO", "BBPSO", "VLPSO", "SFEPSO", "QLDGS-PSO", "QLDGS-PSO-Elite", 'lapsodr', "igpso"]
    methodset = ["QLGSPSO"]
    method_num = len(methodset)
    # 共同参数
    runs = 1  # 方法运行次数
    N = 128  # 种群规模
    Maxiter = 300  # 种群最大迭代次数
    classify = 'knn'  # 分类方法
    cluster = ','  # 如果标记为kmeans就可以使用cluster作为无监督的结果，如果不是这个名字就是使用分类器
    split = 0  # 适应度函数中训练集与验证集划分方式
    func = 0  # 选择不同的适应度函数
    opts = {'maxLt': runs, 'N': N, 'T': Maxiter, 'cluster':cluster,'classify': classify, 'split': split, 'func': func, 'knn_para':3}
    # 获取当前日期并格式化为字符串，例如 '06-27'
    # current_date = datetime.datetime.now().strftime("%m-%d")
    current_date = "y"
    # max_workers为上限进程核数
    with ProcessPoolExecutor(max_workers=8) as executor:
        # 根据数据个数创建进程
        futures = []
        for data in dataset:
            for method in methodset:
                futures.append(executor.submit(process_name, data, [method], opts, runs, current_date, True))


       # 等待任务完成并获取结果
       # 等待所有任务完成并收集结果
        for future in futures:
            result = future.result()
                # results.append(result)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    main()

