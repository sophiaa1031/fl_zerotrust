import os
import matplotlib.pyplot as plt
import numpy as np

# 指定包含文件的目录路径
acc_path = "plot_data/mnist_noniid_acc.txt"

def plot_acc(path):
    training_acc = []
    testing_acc = []

    with open(path, 'r') as file:
        for line in file:
            # 将每行数据按逗号分割，然后取第一个元素（第一列的数据）
            columns = line.strip().split(',')
            if len(columns) >= 1:
                training_acc.append(float(columns[0]))  # 假设数据是浮点数
                testing_acc.append(float(columns[1]))  # 假设数据是浮点数

    x_labels = ["benign","attack_20","attack_10","remove_20", "remove_10", "remove_0", "zerotrust_20", "zerotrust_10", "zerotrust_0"]
    y_data = training_acc

    # 创建一个直方图
    plt.figure(figsize=(8, 8))  # 10是宽度，5是高度
    plt.bar(x_labels, y_data)

    # 添加标题和标签
    plt.xlabel("Methods", fontsize=12)
    plt.ylabel("Training Accuracy", fontsize=12)

    # 自定义X轴标签的角度，以防止重叠
    plt.xticks(rotation=45)
    # plt.savefig('save/mnist_noniid_acc.png')  # 保存为PNG格式图片

    # 显示图形
    plt.show()

plot_acc(acc_path)