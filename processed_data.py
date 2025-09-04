import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter

# 加载数据
data = pd.read_excel("chenji7.xlsx", engine='openpyxl')


# 添加高斯噪声并进行滤波处理
def add_gaussian_noise_and_filter(data, noise_level=0.01, sigma=1.0):
    # 选择第三到第七列 (注意列索引从0开始)
    data_subset = data.iloc[:, 2:7]

    # 添加高斯噪声
    noisy_data = data_subset + np.random.normal(0, noise_level, data_subset.shape)

    # 应用高斯滤波
    filtered_data = gaussian_filter(noisy_data, sigma=sigma)

    # 将处理后的数据替换回原数据
    data.iloc[:, 2:7] = filtered_data

    return data


# 设置噪声水平和高斯滤波的sigma值
noise_level = 0.05  # 调整噪声水平
sigma = 0.4  # 调整滤波强度

# 添加噪声并进行滤波处理
processed_data = add_gaussian_noise_and_filter(data, noise_level, sigma)

# 保存处理后的数据
processed_data.to_excel('processed_data.xlsx', index=False)
