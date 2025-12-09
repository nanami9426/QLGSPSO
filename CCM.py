import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer

# calculate confusion metrics 计算混淆矩阵
# average 设置为 'macro'，意味着会使用宏观平均值来计算，Micro意味着微观计算。
# Micro通过考虑所有类别的总体统计量来计算指标，对每个单独的预测进行求和并计算指标，适用于样本数不平衡的情况。
# Macro通过对每个类别分别计算指标，然后取平均值来得到最终结果，对每个类别都给予了相同的权重。
def calculate_metrics(label, pred, metric):
    # 混淆矩阵
    confusion_matrix = metrics.confusion_matrix(label, pred)

    # 计算指标
    if metric == 'all':
        accuracy = metrics.accuracy_score(label, pred)
        recall_micro = metrics.recall_score(label, pred, average='micro')
        recall_macro = metrics.recall_score(label, pred, average='macro')
        precision_micro = metrics.precision_score(label, pred, average='micro', zero_division=1)
        precision_macro = metrics.precision_score(label, pred, average='macro', zero_division=1)
        f1_micro = metrics.f1_score(label, pred, average='micro')
        f1_macro = metrics.f1_score(label, pred, average='macro')
        roc_auc_micro, roc_auc_macro = calculate_roc_auc(label, pred)
        return {
            'Accuracy': accuracy,
            'Recall (Micro)': recall_micro,
            'Recall (Macro)': recall_macro,
            'Precision (Micro)': precision_micro,
            'Precision (Macro)': precision_macro,
            'F1-Score (Micro)': f1_micro,
            'F1-Score (Macro)': f1_macro,
            'ROC AUC (Micro)': roc_auc_micro,
            'ROC AUC (Macro)': roc_auc_macro
        }

    if metric == 'accuracy':
        # 对于准确率这个指标来说，因为其没有对类别进行加权或考虑样本不平衡的情况，所以不存在Micro与Macro之分。
        return metrics.accuracy_score(label, pred)

    if metric == 'recall':
        return {
            'Recall (Micro)': metrics.recall_score(label, pred, average='micro'),
            'Recall (Macro)': metrics.recall_score(label, pred, average='macro')
        }

    if metric == 'precision':
        return {
            'Precision (Micro)': metrics.precision_score(label, pred, average='micro', zero_division=1),
            'Precision (Macro)': metrics.precision_score(label, pred, average='macro', zero_division=1)
        }

    if metric == 'f1':
        return {
            'F1Score (Micro)': metrics.f1_score(label, pred, average='micro'),
            'F1Score (Macro)': metrics.f1_score(label, pred, average='macro')
        }

    if metric == 'roc_auc':
        roc_auc_micro, roc_auc_macro = calculate_roc_auc(label, pred)
        return {
            'ROC AUC (Micro)': roc_auc_micro,
            'ROC AUC (Macro)': roc_auc_macro
        }

def calculate_roc_auc(label, pred):
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(label)
    label_binary = label_binarizer.transform(label)
    pred_binary = label_binarizer.transform(pred)
    num_classes = label_binary.shape[1]
    roc_auc_micro = metrics.roc_auc_score(label_binary.ravel(), pred_binary.ravel())
    roc_auc_macro = 0.0
    for i in range(num_classes):
        roc_auc_macro += metrics.roc_auc_score(label_binary[:, i], pred_binary[:, i])
    roc_auc_macro /= num_classes
    return roc_auc_micro, roc_auc_macro