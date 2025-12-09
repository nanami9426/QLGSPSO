import math
import matplotlib.pyplot as plt
import numpy as np
from ordered_set import OrderedSet


def draw_curve(dataset, methodset, results):
    # 画图-迭代次数寻优
    # 准备数据
    data_num = len(dataset)
    fig_col = 4  # 图列数
    fig_row = math.ceil(data_num / fig_col)  # 图行数
    if fig_row == 0 or fig_col == 0:
        print("没有数据集或者图表列数为0")
        return
    fig, axs = plt.subplots(fig_row, fig_col, figsize=(4*fig_col, 4*fig_row))
    markers = ['.', 'o', '^', '*', 'x', 'p', 'v', '+', 's']  # 可用的marker样式
    # 'o': 圆圈    # '.': 点    # ',': 像素点    # 'v': 倒三角形    # '^': 正三角形    # 's': 方块    # 'p': 五边形    # '*': 星型    # '+': 加号    # 'x': 叉号
    legend_labels = []  # 存储所有子图的图例标签
    for i, data_name in enumerate(dataset):
        ax_row = i // fig_col
        ax_col = i % fig_col
        if fig_row > 1:  # 当行数大于1时，需要用二维数组来索引
            ax = axs[ax_row, ax_col]
        else:  # 当只有一行时，axs变成了一个一维数组，直接用一维索引即可
            ax = axs[ax_col]
        lines_found = False
        for j, method in enumerate(methodset):
            if method == 'all':  # 跳过初始算法
                continue
            if method in results[data_name]:
                lines_found = True
                y = results[data_name][method]['curve_mean']
                y = y[0]
                Maxiter = len(y)
                x = list(range(1, Maxiter + 1))
                marker_index = j % len(markers)  # 按顺序选择marker样式
                ax.plot(x, y, label="%s" % (method), marker=markers[marker_index], markevery=200)
                legend_labels.append(method)  # 添加图例标签
        if not lines_found:
            continue
        ax.set_title("%s" % data_name)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Fitness Value")
        # 注释掉子图的图例
        # ax.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=3)
        # 设置 x 轴坐标间距为 50
        ax.set_xticks(np.arange(0, Maxiter + 1, 200))
    # 隐藏多余的子图
    for i in range(data_num, fig_row*fig_col):
        if fig_row > 1:  # 当行数大于1时，需要用二维数组来索引
            ax = axs[i // fig_col, i % fig_col]
        else:
            ax = axs[i % fig_col]
        ax.axis('off')
    # 去除重复的图例标签
    legend_labels = list(OrderedSet(legend_labels))
    # 添加大图的图例并放置在底部中央
    fig.legend(legend_labels, loc='lower center', ncol=len(legend_labels), bbox_to_anchor=(0.5, 0))
    plt.tight_layout()
    plt.show()


def draw_nf(dataset, methodset, results):
    ## 画图-特征数量
    # 准备数据
    data_num = len(dataset)
    # 创建新的图表和子图
    fig_col = 4  # 图列数
    fig_row = math.ceil(data_num / fig_col)  # 图行数
    fig, axs = plt.subplots(fig_row, fig_col, figsize=(5*fig_col, 5*fig_row))
    # 遍历所有数据集，每个数据集在不同的子图上绘制
    for i, data_name in enumerate(dataset):
        ax_row = i // fig_col  # 计算行索引
        ax_col = i % fig_col   # 计算列索引
        if fig_row > 1:  # 当行数大于1时，需要用二维数组来索引
            ax = axs[ax_row, ax_col]
        else:  # 当只有一行时，axs变成了一个一维数组，直接用一维索引即可
            ax = axs[ax_col]
        # 检查数据集中是否存在要绘制的柱状图
        column_found = False
        x_labels = []  # 用于保存横坐标标签
        x_positions = []  # 用于保存横坐标位置
        for j, method in enumerate(methodset):
            if method == 'all':  # 如果是'all'则跳过当前循环
                continue
            if method in results[data_name]:  # 检查方法是否存在于结果中
                column_found = True
                y = results[data_name][method]['nf']
                x_positions.append(j)  # 保存 x 轴位置
                x_labels.append(method)  # 保存 x 轴标签
                ax.bar(j, y, label="%s" % (method))
                ax.text(j, y, "%.2f" % y, ha='center', va='bottom', color='black')  # 添加标签并格式化显示
        if not column_found:  # 如果没有找到要绘制的柱状图，则跳过当前数据集
            continue
        # 添加标题和标签
        ax.set_title("%s" % data_name)
        ax.set_ylabel("Number of Selected Features")
        ax.set_xticks(x_positions)  # 设置 x 轴刻度为保存的位置
        ax.set_xticklabels(x_labels)  # 设置 x 轴标签为保存的标签
    # 隐藏多余的子图
    for i in range(data_num, fig_row*fig_col):
        if fig_row > 1:  # 当行数大于1时，需要用二维数组来索引
            ax = axs[i // fig_col, i % fig_col]
        else:
            ax = axs[i % fig_col]
        ax.axis('off')
    # 自动调整子图的间距
    plt.tight_layout()
    # 显示所有图表
    plt.show()


def draw_time( dataset, methodset, results):
    ## 画图-算法运行时间
    # 准备数据
    data_num = len(dataset)
    # 创建新的图表和子图
    fig_col = 3  # 图列数
    fig_row = math.ceil(data_num / fig_col)  # 图行数
    fig, axs = plt.subplots(fig_row, fig_col, figsize=(5*fig_col, 5*fig_row))
    # 遍历所有数据集，每个数据集在不同的子图上绘制
    for i, data_name in enumerate(dataset):
        ax_row = i // fig_col  # 计算行索引
        ax_col = i % fig_col   # 计算列索引
        if fig_row > 1:  # 当行数大于1时，需要用二维数组来索引
            ax = axs[ax_row, ax_col]
        else:  # 当只有一行时，axs变成了一个一维数组，直接用一维索引即可
            ax = axs[ax_col]
        # 检查数据集中是否存在要绘制的柱状图
        column_found = False
        for j, method in enumerate(methodset):
            if method in results[data_name]:  # 检查方法是否存在于结果中
                column_found = True
                y = results[data_name][method]['time']
                x_pos = [j]  # 设置 x 轴位置为方法的索引位置
                ax.bar(x_pos, y, label="%s" % (method))
        if not column_found:  # 如果没有找到要绘制的柱状图，则跳过当前数据集
            continue
        # 添加标题和标签
        ax.set_title("%s" % data_name)
        ax.set_ylabel("Time of Algorithm Operation")
        ax.set_xticks(range(len(methodset)))  # 设置 x 轴刻度为 methodset
        ax.set_xticklabels(methodset)  # 设置 x 轴标签为 methodset
    # 隐藏多余的子图
    for i in range(data_num, fig_row*fig_col):
        if fig_row > 1:  # 当行数大于1时，需要用二维数组来索引
            ax = axs[i // fig_col, i % fig_col]
        else:
            ax = axs[i % fig_col]
        ax.axis('off')
    # 自动调整子图的间距
    plt.tight_layout()
    # 显示所有图表
    plt.show()

def draw_eva(dataset, methodset, results):
    ## 画图-其他算法评估值
    # 准备数据
    data_num = len(dataset)
    # 创建新的图表和子图
    fig_col = 3  # 图列数
    fig_row = math.ceil(data_num / fig_col)  # 图行数
    fig, axs = plt.subplots(fig_row, fig_col, figsize=(5*fig_col, 5*fig_row))
    # 遍历所有数据集，每个数据集在不同的子图上绘制
    for i, data_name in enumerate(dataset):
        ax_row = i // fig_col  # 计算行索引
        ax_col = i % fig_col   # 计算列索引
        if fig_row > 1:  # 当行数大于1时，需要用二维数组来索引
            ax = axs[ax_row, ax_col]
        else:  # 当只有一行时，axs变成了一个一维数组，直接用一维索引即可
            ax = axs[ax_col]
        # 检查数据集中是否存在要绘制的柱状图
        column_found = False
        for j, method in enumerate(methodset):
            if method in results[data_name]:  # 检查方法是否存在于结果中
                column_found = True
                evaluation = 'Accuracy'
                y = results[data_name][method]['v_res'][evaluation]
                x_pos = [j]  # 设置 x 轴位置为方法的索引位置
                ax.bar(x_pos, y, label="%s" % (method))
        if not column_found:  # 如果没有找到要绘制的柱状图，则跳过当前数据集
            continue
        # 添加标题和标签
        ax.set_title("%s" % data_name)
        ax.set_ylabel(evaluation)
        ax.set_xticks(range(len(methodset)))  # 设置 x 轴刻度为 methodset
        ax.set_xticklabels(methodset)  # 设置 x 轴标签为 methodset
    # 隐藏多余的子图
    for i in range(data_num, fig_row*fig_col):
        if fig_row > 1:  # 当行数大于1时，需要用二维数组来索引
            ax = axs[i // fig_col, i % fig_col]
        else:
            ax = axs[i % fig_col]
        ax.axis('off')
    # 自动调整子图的间距
    plt.tight_layout()
    # 显示所有图表
    plt.show()

def draw_data(dataset, methodset, result_data, evaluation):
    ## 画图-其他评估指标
    # 准备数据
    data_num = len(dataset)
    # 获取第一行作为列名
    var_names = result_data[0, :]
    # 找到目标列名所在的索引
    target_col_index = np.where(var_names == evaluation)[0]
    # 提取目标列数据
    target_col_data = result_data[1:, target_col_index].astype(float)  # 将数据转换为浮点型
    col_name_data = result_data[1:, 0:2]
    new_data = np.concatenate((col_name_data, target_col_data), axis=1)
    # 创建新的图表和子图
    fig_col = 3  # 图列数
    fig_row = math.ceil(data_num / fig_col)  # 图行数
    fig, axs = plt.subplots(fig_row, fig_col, figsize=(5*fig_col, 5*fig_row))
    # 遍历所有数据集，每个数据集在不同的子图上绘制
    for i, data_name in enumerate(dataset):
        ax_row = i // fig_col  # 计算行索引
        ax_col = i % fig_col   # 计算列索引
        if fig_row > 1:  # 当行数大于1时，需要用二维数组来索引
            ax = axs[ax_row, ax_col]
        else:  # 当只有一行时，axs变成了一个一维数组，直接用一维索引即可
            ax = axs[ax_col]
        # 检查数据集中是否存在要绘制的柱状图
        column_found = False
        x_labels = []  # 用于保存横坐标标签
        x_positions = []  # 用于保存横坐标位置
        for j, method in enumerate(methodset):
            column_found = True
            # 使用条件筛选获取符合条件的结果
            mask = (new_data[:, 0] == data_name) & (new_data[:, 1] == method)
            y = new_data[mask][:, 2].astype(float)  # 将数据转换为浮点型
            y = y * 100  # 将 y 值乘以 100
            x_positions.append(j)  # 保存 x 轴位置
            x_labels.append(method)  # 保存 x 轴标签
            ax.bar(j, y, label="%s" % (method))
        if not column_found:  # 如果没有找到要绘制的柱状图，则跳过当前数据集
            continue
        # 添加标题和标签
        ax.set_title("%s" % data_name)
        ax.set_ylabel(evaluation + " (%)")  # 在标签上添加百分号
        ax.set_xticks(x_positions)  # 设置 x 轴刻度为保存的位置
        ax.set_xticklabels(x_labels)  # 设置 x 轴标签为保存的标签
    # 隐藏多余的子图
    for i in range(data_num, fig_row*fig_col):
        if fig_row > 1:  # 当行数大于1时，需要用二维数组来索引
            ax = axs[i // fig_col, i % fig_col]
        else:
            ax = axs[i % fig_col]
        ax.axis('off')
    # 自动调整子图的间距
    plt.tight_layout()
    # 显示所有图表
    plt.show()