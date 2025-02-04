import numpy as np


def load_data(data_file):
    # 首行（列标题）
    feature_names = np.genfromtxt(data_file, delimiter=",", max_rows=1, dtype=str)
    # 二维数据
    data = np.genfromtxt(data_file, delimiter=",", skip_header=1)

    # 归一化
    max_values = data.max(axis=0)
    min_values = data.min(axis=0)
    data = (data - min_values) / (max_values - min_values)

    # 划分训练集
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    test_data = data[offset:]

    return training_data, test_data, max_values, min_values
