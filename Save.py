import json
import os
import numpy as np
import datetime

import pandas as pd


# 定义递归函数，将字典转换为原生数据类型
def convert_dict_to_native(results_dict):
    results_native = {}
    for key, value in results_dict.items():
        if isinstance(value, dict):
            results_native[key] = convert_dict_to_native(value)
        elif isinstance(value, np.ndarray):
            results_native[key] = value.tolist()
        else:
            results_native[key] = value
    return results_native


# 定义递归函数，将原生数据类型转换为字典
def convert_native_to_dict(data):
    converted_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            converted_data[key] = convert_native_to_dict(value)
        else:
            converted_data[key] = value
    return converted_data

def read_result():
    # 从文件中读取结果
    folder_path = "Result"
    file_path = os.path.join(folder_path, "results.json")
    with open(file_path, "r") as f:
        results_native = json.load(f)
    # 将原生数据类型转换为字典
    results_dict = convert_native_to_dict(results_native)
    # result_ind = ['sf', 'nf', 'c_mid', 'c_mean', 'v_res', 't_res']
    # res = ['Accuracy', 'Recall (Micro)', 'Recall (Macro)', 'Precision (Micro)', 'Precision (Macro)', 'F1-Score (Micro)', 'F1-Score (Macro)']
    results = results_dict
    print("结果已成功读取")
    return results

'''
# 重新画图
# 跟Main函数中值一样，主要是跟results一样
from Save import *
dataset = ['BCW.csv',  'HV.csv', 'SRBCT.csv', 'BT1.csv']
dataset = [filename.rstrip('.csv') for filename in dataset]  # 去除.csv后缀
methodset = ['all', 'psoV1', 'pso']
results = read_result()
draw_curve(dataset, methodset, results)
draw_nf(dataset, methodset, results)
'''
# 定义一个函数来追加数据到CSV文件
# def append_to_csv(file_path, data, delimiter, header=None):
#     mode = 'a' if os.path.exists(file_path) else 'w'
#     with open(file_path, mode, newline='') as f:
#         # 如果文件是新创建的，则写入表头
#         if mode == 'w' and header is not None:
#             np.savetxt(f, [header], delimiter=delimiter, fmt='%s')
#         np.savetxt(f, data, delimiter=delimiter, fmt='%s')


def append_to_csv(file_path, data, delimiter=',', header=None):
    """
    将数据追加到 CSV 文件中，支持 pandas.DataFrame 和其他类型的数据（如列表、numpy 数组）

    参数:
        file_path (str): 文件路径
        data: 要保存的数据，可以是 pandas.DataFrame、列表、numpy 数组等
        delimiter (str): 分隔符，默认为 ','
        header: 表头，默认为 None
    """
    mode = 'a' if os.path.exists(file_path) else 'w'

    # 如果是 pandas.DataFrame，直接使用 to_csv 方法
    if isinstance(data, pd.DataFrame):
        data.to_csv(file_path, mode=mode, index=False, header=header is not None, sep=delimiter)
    else:
        # 如果是其他类型的数据（如列表或 numpy 数组），使用 numpy.savetxt
        with open(file_path, mode, newline='') as f:
            # 如果文件是新创建的，则写入表头
            if mode == 'w' and header is not None:
                np.savetxt(f, [header], delimiter=delimiter, fmt='%s')
            np.savetxt(f, data, delimiter=delimiter, fmt='%s')
def save_result_many(data_name, results, result_list, result_curve, result_run, result_run_best, result_run_best_name, methodset, current_date, run_index = []):
    # 保存结果
    result_data = np.array(result_list)  # 将列表转换为NumPy数组
    result_curve = np.array(result_curve)
    result_best = np.concatenate((result_run_best_name, result_run_best), axis=1)
    # if run_index != []:
    #     # run_index = np.array()
    # 将字典列表转换为 NumPy 数组，需要先确定列的顺序和数据类型
    result_run_data = np.array([[item['Data Name'], item['Method'], item['Run'], item['Feature Number'],
                                 item['Time Calculation'], item['Accuracy Valid'], item['Accuracy Test'],
                                 item['F1 Valid'], item['F1 Test'], item['AUC Valid'], item['AUC Test']]
                                for item in result_run])
    if run_index != []:
        run_index_data = [[item['Data Name'], item['Method']] + item['index'].tolist() for item in run_index]
        run_index_data = pd.DataFrame(run_index_data)
    # 将列名作为第一行
    run_result_var_names = ['Data Name', 'Method', 'Run', 'Feature Number', 'Time Calculation', 'Accuracy Valid', 'Accuracy Test', 'F1 Valid', 'F1 Test', 'AUC Valid', 'AUC Test']
    # 添加变量名
    result_data_var_names = ['Data Name', 'Method', 'NF Mean', 'NF Std', 'Accuracy Test Best', 'Accuracy Test Mean', 'Accuracy Test Std', 'Accuracy Test Min', 'F1 Test Mean', 'AUC Test Mean','Recall Test Mean', 'Precision Test Mean', 'Timecal Mean']
    result_best_var_names = ['Data Name'] + methodset
    folder_path = "Result"
    os.makedirs(folder_path, exist_ok=True)  # 确保文件夹存在

    # 准备数据
    result_data = np.array(result_list)
    result_curve = np.array(result_curve)
    result_best = np.concatenate((result_run_best_name, result_run_best), axis=1)
    # 创建 NumPy 数组，跳过前面重复添加的 'Data Name' 和 'Method'
    result_run_data = np.array([[item[key] for key in run_result_var_names if key in item] for item in result_run])
    # 列名
    result_data_header = result_data_var_names
    # result_curve 不需要表头，所以 header 参数为 None
    result_best_header = result_best_var_names
    result_run_data_header = run_result_var_names

    # 保存路径
    data_save_path = os.path.join(folder_path, f"result_{current_date}.csv")
    curve_save_path = os.path.join(folder_path, f"result_curve_{current_date}.csv")
    best_save_path = os.path.join(folder_path, f"result_best_{current_date}.csv")
    run_save_path = os.path.join(folder_path, f"result_run_{current_date}.csv")
    if run_index != []:
        index_save_path = os.path.join(folder_path, f"result_index_{current_date}.csv")

    # 调用 append_to_csv 函数保存数据
    # 只有当文件不存在时，才写入表头
    append_to_csv(data_save_path, result_data, delimiter=',', header=result_data_header)
    append_to_csv(curve_save_path, result_curve, delimiter=',', header=None)  # result_curve 不需要表头
    append_to_csv(best_save_path, result_best, delimiter=',', header=result_best_header)
    append_to_csv(run_save_path, result_run_data, delimiter=',', header=result_run_data_header)
    if run_index != []:
        append_to_csv(index_save_path, run_index_data, delimiter=',', header=None)

    # results_native = convert_dict_to_native(results)  # 将结果字典转换为原生数据类型
    # os.makedirs(folder_path, exist_ok=True)  # 如果文件夹不存在则创建文件夹
    # file_path = os.path.join(folder_path, "results.json")  # 文件路径
    # with open(file_path, "w") as f:
    #     json.dump(results_native, f)  # 将数据字典存储到文件中
    print(f"{data_name} - 结果已成功保存")