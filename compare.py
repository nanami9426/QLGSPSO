import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 读取两个CSV文件
df1 = pd.read_csv('Dataset/Colon.csv', header=None)
df2 = pd.read_csv('Dataset/SRBCT.csv',header=None)

# 步骤1：检查基础结构
print("数据集1形状:", df1.shape)
print("数据集2形状:", df2.shape)
print("列名是否一致:", all(df1.columns == df2.columns))

# 步骤2：检查数值完全相同的行
comparison = df1.equals(df2)
print("数据集是否完全相同:", comparison)

# 步骤3：检查逐列差异（处理浮点数精度）
tolerance = 1e-6
numeric_cols = df1.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    col_diff = np.abs(df1[col] - df2[col])
    max_diff = col_diff.max()
    if max_diff > tolerance:
        print(f"列 {col} 存在差异，最大差异值: {max_diff}")

# 步骤4：统计信息对比
print("\n数据集1统计信息:")
print(df1.describe())
print("\n数据集2统计信息:")
print(df2.describe())

# 步骤5：检查缺失值和重复行
print("\n数据集1缺失值:", df1.isnull().sum().sum())
print("数据集2缺失值:", df2.isnull().sum().sum())
print("数据集1重复行:", df1.duplicated().sum())
print("数据集2重复行:", df2.duplicated().sum())

# 步骤6：检查分类标签分布（假设目标列为'target'）
if 'target' in df1.columns:
    print("\n数据集1类别分布:")
    print(df1['target'].value_counts())
    print("数据集2类别分布:")
    print(df2['target'].value_counts())


# 步骤7：比较KNN分类效果（示例）
def evaluate_knn(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)


if 'target' in df1.columns:
    acc1 = evaluate_knn(df1)
    acc2 = evaluate_knn(df2)
    print(f"\nKNN准确率对比 - 数据集1: {acc1:.4f}, 数据集2: {acc2:.4f}")

# 步骤8：可视化特征分布（示例）
import matplotlib.pyplot as plt

if 'feature1' in df1.columns:
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.hist(df1['feature1'], bins=20, alpha=0.5, label='Dataset1')
    plt.hist(df2['feature1'], bins=20, alpha=0.5, label='Dataset2')
    plt.title('Feature1 Distribution')
    plt.legend()
    plt.show()