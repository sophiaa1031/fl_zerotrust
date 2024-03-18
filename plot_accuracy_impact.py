import matplotlib.pyplot as plt

def read_first_column(file_path):
    # 读取文件的第一列数据
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = [float(line.strip().split()[0]) for line in lines[:50]]
    return data

data1 = []
data2 = []

for i in range(3):
    # 从第一个文件中读取第一列数据
    folder_path = 'plot_data/accuracy_impact/try' + str(i + 1) + '/'
    file1_path = folder_path + 'mnist_noniid_acc.txt'
    data1.append(read_first_column(file1_path))

    # 从第二个文件中读取第一列数据
    file2_path = folder_path + '_acc.txt'
    data2.append(read_first_column(file2_path))

# 计算每个文件数据的平均值
data_sum1 = [(x + y + z) / 3 for x, y, z in zip(data1[0], data1[1], data1[2])]
data_sum2 = [(x + y + z) / 3 for x, y, z in zip(data2[0], data2[1], data2[2])]


# 计算两列数据之差
difference = [x - y for x, y in zip(data_sum1, data_sum2)]

# 创建横轴数据，长度为数据的长度
x_axis = range(1, len(difference) + 1)

# 绘制图表
plt.plot(x_axis, difference)
plt.xlabel('FL Epochs')
plt.ylabel('Accuracy Impact')
plt.savefig('save/accuracy_impact.png')  # 保存为PNG格式图片
plt.show()
