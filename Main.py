import argparse
import ast
import os
import numpy as np
import pandas as pd
from Processor import Pred_Processor
from Example_model import Pred_model
from Evaluator import Pred_Evaluator
from viz import visualization
import matplotlib.pyplot as plt
import csv
#import openpyxl
import pandas as pd
import matplotlib.pyplot as plt

def get_parser():
    parser = argparse.ArgumentParser(
        description='PLM')
    parser.add_argument('--DataSet', type=str, default='E_trajectory_MultiSubject')   #数据名称 E_trajectory  E_trajectory_left   E_trajectory_SIND   E_trajectory_MultiSubject  Apollo_MultiSubject
    parser.add_argument('--sample_dir', type=str, default="My_Data_20240820_2")  # 存储数据地址 data_set_NoneLDA_0608
    parser.add_argument('--data_dir', type=str, default='E:/开源代码/Preference_Learning_Model/data')  # 数据地址

    parser.add_argument('--if_const_data', type=str, default=False)  # False  True
    parser.add_argument('--if_train', type=str, default=False)  # True
    parser.add_argument('--if_test', type=str, default=False)
    parser.add_argument('--if_vis', type=str, default=True)
    parser.add_argument('--if_enhance_data', type=str, default=True)  # True
    parser.add_argument('--if_iter_pre', type=str, default=False)
    parser.add_argument('--Pred_length', type=int, default=25)  #25  预测时长 45=9*5  6
    parser.add_argument('--Obs_length', type=int, default=20)  #20  预测时长 27=9*3   4
    parser.add_argument('--Decision_interval', type=int, default=5)  #我的数据是5
    parser.add_argument('--max_bicycle_num', type=int, default=3)
    parser.add_argument('--occupancy_value', type=int, default=0)
    parser.add_argument('--learning_rate', type=int, default=0.001)  #0.001
    parser.add_argument('--threshold_acc', type=int, default=0.5)
    parser.add_argument('--threshold_jerk', type=int, default=0.3)
    parser.add_argument('--threshold_curvature', type=int, default=0.01)
    parser.add_argument('--threshold_angle', type=int, default=10)  #5
    parser.add_argument('--threshold_angle_var', type=int, default=5)  #5
    parser.add_argument('--penalty_coefficient', type=int, default=0.001)  #0.001
    parser.add_argument('--feature_num', type=int, default=3)

    return parser


def sliding_window_smooth(data, window_size):
    num_rows, num_cols = data.shape  # 获取数据的行数和列数
    smoothed_data = np.zeros_like(data)  # 创建与原始数据相同形状的数组来存储平滑后的数据

    for col in range(num_cols):
        for row in range(num_rows):
            start = max(0, row - window_size // 2)  # 窗口的起始位置
            end = min(num_rows, row + window_size // 2 + 1)  # 窗口的结束位置
            smoothed_data[row, col] = np.mean(data[start:end, col])

    return smoothed_data


def data_cleaning(A, angle_threshold):
    if args.DataSet == 'E_trajectory_MultiSubject':
        E_trajectory[:, 3] = np.where(np.abs(E_trajectory[:, 3]) > 10, np.median(E_trajectory[:, 3]),
                                      E_trajectory[:, 3])
        E_trajectory[:, 4] = np.where(np.abs(E_trajectory[:, 4]) > 4,
                                      4 * np.sign(E_trajectory[:, 4]), E_trajectory[:, 4])
        E_trajectory[:, 5] = np.where(np.abs(E_trajectory[:, 5]) > 10,
                                      10 * np.sign(E_trajectory[:, 5]), E_trajectory[:, 5])
        E_trajectory[:, 6] = np.where(np.abs(E_trajectory[:, 6]) > 10,
                                      10 * np.sign(E_trajectory[:, 6]), E_trajectory[:, 6])

    A_filtered = E_trajectory
    """
    过滤掉速度角度超过一定限度的行人轨迹。

    Parameters
    ----------
    A : numpy.ndarray
        带有行人编号的矩阵，列分别为[帧ID、特征数、行人轨迹点编号、行人编号，类型]。
    angle_threshold : float
        速度向量的角度阈值范围，如 [-angle_threshold, angle_threshold]。

    Returns
    -------
    A_filtered : numpy.ndarray
        过滤后的新矩阵，不包含速度角度超过阈值的行人轨迹。
    #"""
    """
    angle_threshold = 90
    # 提取唯一的行人编号
    pedestrian_ids = np.unique(A[:, 12])

    # 用于存放保留的行人轨迹
    valid_trajectories = []

    # 遍历每个行人
    for pid in pedestrian_ids:
        # 获取该行人的所有轨迹点
        trajectory = A[A[:, 12] == pid]

        # 确保轨迹点数大于1
        if len(trajectory) < 2:
            continue

        #trajectory[:, 1:3] = sliding_window_smooth(trajectory[:, 1:3], 5)


        # 获取起点和终点的坐标 (假设特征数包含 x, y 坐标)
        start_point = trajectory[0, 2:4]  # 假设特征列中 [1:3] 是 [x, y] 坐标
        end_point = trajectory[-1, 2:4]  # 终点的 [x, y] 坐标

        # 计算起点到终点的速度向量
        velocity_vector = end_point - start_point

        # 计算该向量与 x 轴的夹角（以度数表示）
        angle = np.degrees(np.arctan2(velocity_vector[1], velocity_vector[0]))

        # 判断角度是否在 [-angle_threshold, angle_threshold] 范围内
        if -angle_threshold <= angle <= angle_threshold:
            # 若在范围内，则保留该行人的轨迹
            valid_trajectories.append(trajectory)

    # 将所有保留的轨迹组合成新的矩阵
    if valid_trajectories:
        A_filtered = np.vstack(valid_trajectories)
    else:
        A_filtered = np.array([])  # 如果没有符合条件的轨迹，返回空数组
    """
    return A_filtered


def split_data_by_id_Nem(A, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
    """
    将带有行人编号的矩阵随机划分成训练集、验证集和测试集。

    Parameters
    ----------
    A : numpy.ndarray
        带有行人编号的矩阵，列分别为[帧ID、特征数、行人轨迹点编号、行人编号，类型]。
    train_ratio : float, optional
        训练集比例，默认为 0.7。
    valid_ratio : float, optional
        验证集比例，默认为 0.2。
    test_ratio : float, optional
        测试集比例，默认为 0.1。

    Returns
    -------
    train_set : numpy.ndarray
        训练集。
    valid_set : numpy.ndarray
        验证集。
    test_set : numpy.ndarray
        测试集。
    """
    # 提取唯一的自行车编号
    pedestrian_ids = np.unique(A[:, 12])

    # 随机打乱行人编号
    np.random.shuffle(pedestrian_ids)

    # 计算各个数据集的长度
    a_len = int(len(pedestrian_ids) * train_ratio)
    b_len = int(len(pedestrian_ids) * (train_ratio + valid_ratio))

    # 按照比例划分行人编号
    train_set_id, valid_set_id, test_set_id = np.split(pedestrian_ids, [a_len, b_len])

    # 根据行人编号划分训练集、验证集和测试集
    train_set = A[np.isin(A[:, 12], train_set_id)]
    valid_set = A[np.isin(A[:, 12], valid_set_id)]
    test_set = A[np.isin(A[:, 12], test_set_id)]

    return train_set, valid_set, test_set


def split_data_by_timestamp(A, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
    """
    将带有 timestamp 的矩阵划分成训练集、验证集和测试集。

    Parameters
    ----------
    A : numpy.ndarray
        带有 ID 的矩阵。
    train_ratio : float, optional
        训练集比例，默认为 0.7。
    valid_ratio : float, optional
        验证集比例，默认为 0.2。
    test_ratio : float, optional
        测试集比例，默认为 0.1。

    Returns
    -------
    train_set : pandas.DataFrame
        训练集。
    valid_set : pandas.DataFrame
        验证集。
    test_set : pandas.DataFrame
        测试集。
    """
    ids = np.unique(A[:, 0])
    ids_new = ids.copy()  # 复制 ids 数组，以免修改原始数组  时间戳
    #np.random.shuffle(ids_new)  # 打乱 ids_new 数组的顺序
    a_len = int(len(ids_new) * 0.7)
    b_len = int(len(ids_new) * 0.9)
    train_set_id, valid_set_id, test_set_id = np.split(ids_new, [a_len, b_len])

    test_set = A[np.isin(A[:, 0], test_set_id)]
    valid_set = A[np.isin(A[:, 0], valid_set_id)]
    train_set = A[np.isin(A[:, 0], train_set_id)]

    return train_set, valid_set, test_set


def normalizing(input_data):
    # 特征归一化
    output = np.zeros((input_data.shape[0], input_data.shape[1]))
    mmin = np.zeros((1, input_data.shape[1]))
    mmax = np.zeros((1, input_data.shape[1]))

    for i in range(input_data.shape[1]):
        # 输出每列最小值
        mmin[0, i] = np.min(input_data[:, i])
        # 输出每列最大值，留作以后反归一化
        mmax[0, i] = np.max(input_data[:, i])

        # 对每列（指标）进行归一化
        for j in range(input_data.shape[0]):
            output[j, i] = (input_data[j, i] - mmin[0, i]) / (mmax[0, i] - mmin[0, i])

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if np.isnan(output[i, j]):
                # 查找如果等于nan的话，就写成0；
                output[i, j] = 0

        return output, mmin, mmax


def normalize_and_split_matrices(args, x_train_input, y_train_input, x_val_input, y_val_input, x_test_input,
                                 y_test_input):

    # 归一化方法1
    # 第一步变换形状
    step_x = args.Obs_length / args.Decision_interval
    fea_num = int(x_train_input.shape[1] / step_x / args.max_bicycle_num)  # 求取特征值数目  一个人的
    #fea_num = int(x_train_input.shape[1] / step_x)  # 求取特征值数目  一个人
    sample_num_x = x_train_input.shape[1]
    sample_num_y = y_train_input.shape[1]

    sample_num_x_test = x_test_input.shape[1]
    sample_num_y_test = y_test_input.shape[1]

    x_train_short = x_train_input.reshape(-1, fea_num)
    y_train_short = y_train_input.reshape(-1, fea_num)
    x_val_short = x_val_input.reshape(-1, fea_num)
    y_val_short = y_val_input.reshape(-1, fea_num)
    x_test_short = x_test_input.reshape(-1, fea_num)
    y_test_short = y_test_input.reshape(-1, fea_num)

    #"""
    # 将无数据的行换成nan
    x_train_short[np.all(x_train_short == 0, axis=1), :] = np.nan
    y_train_short[np.all(y_train_short == 0, axis=1), :] = np.nan
    x_val_short[np.all(x_val_short == 0, axis=1), :] = np.nan
    y_val_short[np.all(y_val_short == 0, axis=1), :] = np.nan
    x_test_short[np.all(x_test_short == 0, axis=1), :] = np.nan
    y_test_short[np.all(y_test_short == 0, axis=1), :] = np.nan
    #"""

    m_all = np.vstack([x_train_short, y_train_short, x_val_short, y_val_short, x_test_short, y_test_short])

    if args.DataSet == 'E_trajectory_MultiSubject':
        #"""
        # 修正超范围之数值
        m_all[:, 2][m_all[:, 2] > 5] = 5
        m_all[:, 3][m_all[:, 3] < - 2] = - 2

        x_train_short[:, 2][x_train_short[:, 2] > 5] = 5
        x_train_short[:, 3][x_train_short[:, 3] < - 2] = - 2

        y_train_short[:, 2][y_train_short[:, 2] > 5] = 5
        y_train_short[:, 3][y_train_short[:, 3] < - 2] = - 2

        x_val_short[:, 2][x_val_short[:, 2] > 5] = 5
        x_val_short[:, 3][x_val_short[:, 3] < - 2] = - 2

        y_val_short[:, 2][y_val_short[:, 2] > 5] = 5
        y_val_short[:, 3][y_val_short[:, 3] < - 2] = - 2

        x_test_short[:, 2][x_test_short[:, 2] > 5] = 5
        x_test_short[:, 3][x_test_short[:, 3] < - 2] = - 2

        y_test_short[:, 2][y_test_short[:, 2] > 5] = 5
        y_test_short[:, 3][y_test_short[:, 3] < - 2] = - 2
        #"""
    new_min = np.nanmin(m_all, axis=0)
    new_max = np.nanmax(m_all, axis=0)
    #plt.hist(m_all[:, 7])
    #plt.show()

    x_train_nor = normalize_column(x_train_short, new_min, new_max)
    y_train_nor = normalize_column(y_train_short, new_min, new_max)
    x_val_nor = normalize_column(x_val_short, new_min, new_max)
    y_val_nor = normalize_column(y_val_short, new_min, new_max)
    x_test_nor = normalize_column(x_test_short, new_min, new_max)
    y_test_nor = normalize_column(y_test_short, new_min, new_max)

    # 检查是否存在nan，若存在赋予0
    x_train_nor[np.isnan(x_train_nor)] = args.occupancy_value
    y_train_nor[np.isnan(y_train_nor)] = args.occupancy_value
    x_val_nor[np.isnan(x_val_nor)] = args.occupancy_value
    y_val_nor[np.isnan(y_val_nor)] = args.occupancy_value
    x_test_nor[np.isnan(x_test_nor)] = args.occupancy_value
    y_test_nor[np.isnan(y_test_nor)] = args.occupancy_value


    # 形状变换
    x_train_output = x_train_nor.reshape(-1, sample_num_x)
    y_train_output = y_train_nor.reshape(-1, sample_num_y)
    x_val_output = x_val_nor.reshape(-1, sample_num_x)
    y_val_output = y_val_nor.reshape(-1, sample_num_y)
    x_test_output = x_test_nor.reshape(-1, sample_num_x_test)
    y_test_output = y_test_nor.reshape(-1, sample_num_y_test)
    #""""
    '''

    # 归一化方法2
    #"""
    step_x = args.Obs_length / args.Decision_interval
    fea_num = int(x_train_input.shape[1] / step_x / args.max_bicycle_num)  #求取特征值数目
    sample_num_x = x_train_input.shape[1]
    sample_num_y = y_train_input.shape[1]

    # 第一步变换形状
    x_train_short = x_train_input.reshape(-1, fea_num)
    y_train_short = y_train_input.reshape(-1, fea_num)
    x_val_short = x_val_input.reshape(-1, fea_num)
    y_val_short = y_val_input.reshape(-1, fea_num)
    x_test_short = x_test_input.reshape(-1, fea_num)
    y_test_short = y_test_input.reshape(-1, fea_num)


    # 第二步：将所有矩阵拼接到一起
    big_data = np.concatenate((x_train_short, y_train_short, x_val_short, y_val_short, x_test_short, y_test_short), axis=0)

    # 第三步：进行归一化操作
    big_data_nor, min_boundary_0, max_boundary_0 = normalizing(big_data[:, ])

    new_min = min_boundary_0
    new_max = max_boundary_0
    # 检查是否存在nan，若存在赋予0
    big_data_nor[np.isnan(big_data_nor)] = 0

    # 第四步：读取窄矩阵上每个原参数的行ID
    k_x_train = x_train_short.shape[0]
    k_y_train = x_train_short.shape[0] + y_train_short.shape[0]
    k_x_val = x_train_short.shape[0] + y_train_short.shape[0] + x_val_short.shape[0]
    k_y_val = x_train_short.shape[0] + y_train_short.shape[0] + x_val_short.shape[0] + y_val_short.shape[0]
    k_x_test = x_train_short.shape[0] + y_train_short.shape[0] + x_val_short.shape[0] + y_val_short.shape[0]\
               + x_test_short.shape[0]
    #k_y_test = x_train_short.shape[0] + y_train_short.shape[0] + x_val_short.shape[0] + y_val_short.shape[0] \
               #+ x_test_short.shape[0] + y_test_short.shape[0]

    #print(np.shape(big_data_nor[k_x_train: k_y_train, :]))
    # 第五步 分开矩阵并返回
    x_train_output = big_data_nor[:k_x_train, :].reshape(-1, sample_num_x)
    y_train_output = big_data_nor[k_x_train: k_y_train, :].reshape(-1, sample_num_y)
    x_val_output = big_data_nor[k_y_train: k_x_val, :].reshape(-1, sample_num_x)
    y_val_output = big_data_nor[k_x_val: k_y_val, :].reshape(-1, sample_num_y)
    x_test_output = big_data_nor[k_y_val: k_x_test, :].reshape(-1, sample_num_x)
    y_test_output = big_data_nor[k_x_test:, :].reshape(-1, sample_num_y)
    #"""
    '''
    return x_train_output, y_train_output, x_val_output, y_val_output, x_test_output,\
           y_test_output, new_min, new_max


def normalize_all(x_train_eff, y_train_eff, x_val_eff, y_val_eff, x_test_eff, y_test_eff):
    # """
    # 赋予无数据样本nan
    x_train_eff[(x_train_eff == 0)] = np.nan
    y_train_eff[(y_train_eff == 0)] = np.nan
    x_val_eff[(x_val_eff == 0)] = np.nan
    y_val_eff[(y_val_eff == 0)] = np.nan
    x_test_eff[(x_test_eff == 0)] = np.nan
    y_test_eff[(y_test_eff == 0)] = np.nan
    #"""

    new_min = np.nanmin([np.nanmin(x_train_eff), np.nanmin(y_train_eff), np.nanmin(x_val_eff),
                     np.nanmin(y_val_eff), np.nanmin(x_test_eff), np.nanmin(y_test_eff)])
    new_max = np.nanmax([np.nanmax(x_train_eff), np.nanmax(y_train_eff), np.nanmax(x_val_eff),
                     np.nanmax(y_val_eff), np.nanmax(x_test_eff), np.nanmax(y_test_eff)])
    #plt.hist(x_train_eff.flatten())
    #plt.show()




    # 归一化
    x_train_eff_nor = normalize_matrix(x_train_eff, new_min, new_max)
    y_train_eff_nor = normalize_matrix(y_train_eff, new_min, new_max)
    x_val_eff_nor = normalize_matrix(x_val_eff, new_min, new_max)
    y_val_eff_nor = normalize_matrix(y_val_eff, new_min, new_max)
    x_test_eff_nor = normalize_matrix(x_test_eff, new_min, new_max)
    y_test_eff_nor = normalize_matrix(y_test_eff, new_min, new_max)


    # 检查是否存在nan，若存在赋予0
    x_train_eff_nor[np.isnan(x_train_eff_nor)] = args.occupancy_value
    y_train_eff_nor[np.isnan(y_train_eff_nor)] = args.occupancy_value
    x_val_eff_nor[np.isnan(x_val_eff_nor)] = args.occupancy_value
    y_val_eff_nor[np.isnan(y_val_eff_nor)] = args.occupancy_value
    x_test_eff_nor[np.isnan(x_test_eff_nor)] = args.occupancy_value
    y_test_eff_nor[np.isnan(y_test_eff_nor)] = args.occupancy_value

    return x_train_eff_nor, y_train_eff_nor, x_val_eff_nor, y_val_eff_nor, x_test_eff_nor, y_test_eff_nor


def normalize_matrix(matrix, new_min, new_max):

    # 对矩阵进行线性归一化
    normalized_matrix = (matrix - new_min) / (new_max - new_min)+0.0001

    return normalized_matrix


def normalize_column(matrix, new_min, new_max):
    normalized_matrix = np.zeros((matrix.shape[0], matrix.shape[1]))
    # 对矩阵进行线性归一化
    for i in range(matrix.shape[1]):
        normalized_matrix[:, i] = (matrix[:, i] - new_min[i]) / (new_max[i] - new_min[i]) + 0.001

    return normalized_matrix


def Data_saving(model_path,  min_boundary, max_boundary, x_train, y_train, x_ver, y_ver, x_test, y_test, test_id):
    # 如果路径不存在，则创建路径
    train_data_path = os.path.join(model_path, "train_data")
    test_data_path = os.path.join(model_path, "test_data")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)
    np.savetxt(os.path.join(train_data_path, "x_train.csv"), x_train, delimiter=",")
    np.savetxt(os.path.join(train_data_path, "y_train.csv"), y_train, delimiter=",")
    np.savetxt(os.path.join(train_data_path, "x_ver.csv"), x_ver, delimiter=",")
    np.savetxt(os.path.join(train_data_path, "y_ver.csv"), y_ver, delimiter=",")
    np.savetxt(os.path.join(train_data_path, "x_test.csv"), x_test, delimiter=",")
    np.savetxt(os.path.join(train_data_path, "y_test.csv"), y_test, delimiter=",")
    np.savetxt(os.path.join(model_path, "min_boundary.csv"), min_boundary, delimiter=",")  # 归一化的最小值
    np.savetxt(os.path.join(model_path, "max_boundary.csv"), max_boundary, delimiter=",")  # 最大值mmax
    np.savetxt(os.path.join(model_path, "test_id.csv"), test_id, delimiter=",")  # 测试集id

    for i in range(x_test.shape[0]):
        file_path = os.path.join(test_data_path, 'x_test_' + str(i + 1) + '.txt')
        np.savetxt(file_path, x_test[i, :], delimiter=',')


def data_set_saving(args, data_set_dir1, x_train, y_train, x_val, y_val, x_test, y_test, max_boundary, min_boundary):

    if not os.path.exists(data_set_dir1):
        os.makedirs(data_set_dir1)
    max_boundary_path = os.path.join(data_set_dir1, "max_boundary.txt")
    min_boundary_path = os.path.join(data_set_dir1, "min_boundary.txt")

    x_train_path = os.path.join(data_set_dir1, "x_train.txt")
    y_train_path = os.path.join(data_set_dir1, "y_train.txt")

    x_val_path = os.path.join(data_set_dir1, "x_val.txt")
    y_val_path = os.path.join(data_set_dir1, "y_val.txt")

    x_test_path = os.path.join(data_set_dir1, "x_test.txt")
    y_test_path = os.path.join(data_set_dir1, "y_test.txt")
    # 保存数据集到本地
    np.savetxt(max_boundary_path, max_boundary, delimiter=",")
    np.savetxt(min_boundary_path, min_boundary, delimiter=",")

    np.savetxt(x_train_path, x_train, delimiter=",")
    np.savetxt(y_train_path, y_train, delimiter=",")
    #print(np.shape(x_train))
    np.savetxt(x_val_path, x_val, delimiter=",")
    np.savetxt(y_val_path, y_val, delimiter=",")

    np.savetxt(x_test_path, x_test, delimiter=",")
    np.savetxt(y_test_path, y_test, delimiter=",")


if __name__ == '__main__':

    ######################### Step 1:数据预处理############################
    parser = get_parser()
    args = parser.parse_args()

    # 文件路径和文件名
    dir_path = 'data'
    filename = args.DataSet + '.csv'  #E_trajectory.csv  E_trajectory_left   E_trajectory_SIND   E_trajectory_MultiSubject

    # 拼接完整的文件路径
    file_path = os.path.join(dir_path, filename)

    # 检查文件是否存在
    if os.path.isfile(file_path):
        # 如果文件存在，则读取 csv 数据
        df = pd.read_csv(file_path, delimiter='\t', header=None)
        # 将 DataFrame 转换为 NumPy 数组
        E_trajectory = df.to_numpy()
        print('数据读取中。。。')
    else:
        print("错误：数据文件不存在")
        E_trajectory = []

    #test demo

    data_set_dir1 = os.path.join(args.data_dir, args.sample_dir, 'data_set')  #修改此处即可修改保存数据的文件夹
    data_set_dir2 = os.path.join(args.data_dir, args.sample_dir, 'data_efficiency')
    data_set_dir3 = os.path.join(args.data_dir, args.sample_dir, 'data_safety')

    # 数据清洗
    E_trajectory = data_cleaning(E_trajectory, 90)
    #"""
    print('The data has been cleaned')


    ################# Step 2： 构建训练数据集###########################
    #"""
    if args.if_const_data:
        # ① 数据集划分
        train_data, val_data, test_data = split_data_by_timestamp(E_trajectory, train_ratio=0.7, valid_ratio=0.2,
                                                               test_ratio=0.1)

        dp = Pred_Processor.DataSetProcessing(args)
        if args.if_iter_pre:   #若迭代预测，则预测步长赋予一个间隔
            pred_length = args.Decision_interval
        else:
            pred_length = args.Pred_length

        # ② 构建数据集
        x_train, y_train, x_train_efficiency, y_train_efficiency, x_train_safety, y_train_safety = dp.data_set_construction(train_data, 'train_set', pred_length)
        x_val, y_val, x_val_efficiency, y_val_efficiency, x_val_safety, y_val_safety = dp.data_set_construction(val_data, 'val_set', pred_length)
        x_test, y_test, x_test_efficiency, y_test_efficiency, x_test_safety, y_test_safety = dp.data_set_construction(test_data, 'test_set', args.Pred_length)

        # ③ 归一化
        x_train, y_train, x_val, y_val, x_test, y_test, min_boundary, \
            max_boundary = normalize_and_split_matrices(args, x_train, y_train, x_val, y_val, x_test, y_test)

        x_train_efficiency, y_train_efficiency, x_val_efficiency, y_val_efficiency, x_test_efficiency, \
            y_test_efficiency = normalize_all(x_train_efficiency, y_train_efficiency, x_val_efficiency,
                                                y_val_efficiency, x_test_efficiency, y_test_efficiency)

        x_train_safety, y_train_safety, x_val_safety, y_val_safety, x_test_safety, y_test_safety = \
            normalize_all(x_train_safety, y_train_safety, x_val_safety, y_val_safety, x_test_safety, y_test_safety)

        # ④ 保存数据
        data_set_saving(args, data_set_dir1, x_train, y_train, x_val, y_val, x_test, y_test, max_boundary, min_boundary)
        data_set_saving(args, data_set_dir2, x_train_efficiency, y_train_efficiency, x_val_efficiency, y_val_efficiency, x_test_efficiency, y_test_efficiency, max_boundary, min_boundary)
        data_set_saving(args, data_set_dir3, x_train_safety, y_train_safety, x_val_safety, y_val_safety, x_test_safety, y_test_safety, max_boundary, min_boundary)

        print('The dataset has been saved')

    ########################### Step 3： 训练模型###########################
    if args.if_train:
        # 训练预测模型
        Pred_model.motion_prediction_train(args, data_set_dir1, data_set_dir2, data_set_dir3)
        print('The PLM model has been trained and saved')

    ############################ Step 4： 测试模型###########################
    # 测试预测结果 y_pred_all, ade, fde
    if args.if_test:
        _, _, _, ade, fde = Pred_Evaluator.trajectory_prediction(args, data_set_dir1, data_set_dir2, data_set_dir3)
        print("ADE:", ade)
        print("FDE:", fde)

    ########################### Step 5： 结果可视化###########################
    if args.if_vis:
        print('Visualization....')
        visualization.Visualization(args, [])

    print('ALL work has been done')



