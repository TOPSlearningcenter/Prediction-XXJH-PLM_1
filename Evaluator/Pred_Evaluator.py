#该脚本的作用是利用已知的起终点进行轨迹生成，完成轨迹预测
# 以下开始
import numpy as np
import os
from scipy import interpolate
import pandas as pd
import tensorflow as tf
from MotionPredictionTrain import MotionConstraintLoss
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.signal import savgol_filter

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def plot_histogram(data1, data2, bins):
    plt.hist(data1, alpha=0.5, label='Simulation', bins=bins)
    plt.hist(data2, alpha=0.5, label='Empirical data', bins=bins)  #, color='orange'
    plt.legend(prop={'size': 12})  #字体大小
    plt.xlabel('Steering angle (°/3s)', fontsize=18)  #Velocity  Lateral motion rate (m/3s)
    plt.ylabel('Frequency', fontsize=18)
    #plt.title('Histogram Comparison', fontsize=14)
    plt.tick_params(axis='both', which='both', direction='in', labelsize=12)
    plt.show()

def calculate_trajectory_indicators(A):
    # 计算侧向位移
    lateral_displacement = np.abs(np.max(A[:, 1]) - np.min(A[:, 1])) / 3

    # 计算平均速度
    velocity = np.sqrt(np.sum(np.diff(A[:, 0])**2 + np.diff(A[:, 1])**2)) / 3 * 3.6

    # 计算平均加速度
    acceleration = np.sum(np.diff(np.diff(A, axis=0), axis=0)) / (3**2)

    # 计算转向速率
    angle_change = np.diff(np.arctan2(np.diff(A[:, 1]), np.diff(A[:, 0])))

    # 转换弧度为角度
    angle_change = np.degrees(angle_change)

    # 计算平均转向速率
    turning_rate = np.mean(np.abs(angle_change))

    return lateral_displacement, velocity, acceleration, angle_change


def calculate_position(data, T):
    # 提取位置和速度信息
    x = data[:, 0]
    y = data[:, 1]
    vx = data[:, 2]
    vy = data[:, 3]

    # 计算步长位置
    positions = np.zeros_like(data)
    positions[:, 0] = x[0]
    positions[:, 1] = y[0]
    for i in range(1, len(data)):
        positions[i, 0] = positions[i-1, 0] + vx[i-1] * T
        positions[i, 1] = positions[i-1, 1] + vy[i-1] * T

    return positions


def exponential_smoothing(data, alpha):
    smoothed_data = [data[0]]  # 初始值与原始数据相同
    for i in range(1, len(data)):
        smoothed_value = alpha * data[i] + (1 - alpha) * smoothed_data[i-1]
        smoothed_data.append(smoothed_value)
    return smoothed_data


def savitzky_golay(data, window_size, poly_order):
    smoothed_data = savgol_filter(data, window_size, poly_order)
    return smoothed_data


def sliding_window_smooth(data, window_size):
    num_rows, num_cols = data.shape  # 获取数据的行数和列数
    smoothed_data = np.zeros_like(data)  # 创建与原始数据相同形状的数组来存储平滑后的数据

    for col in range(num_cols):
        for row in range(num_rows):
            start = max(0, row - window_size // 2)  # 窗口的起始位置
            end = min(num_rows, row + window_size // 2 + 1)  # 窗口的结束位置
            smoothed_data[row, col] = np.mean(data[start:end, col])

    return smoothed_data[1:, :]


##三次样条曲线插值函数
def cubic_hermite_interpolation(valid_frame_labe,valid_x_center,invalid_frame_labe):
    f = interpolate.interp1d(valid_frame_labe, valid_x_center, kind=3)
    return f(invalid_frame_labe)


def bench_based_last_point(pred, histor, T):
    # 读取历史点最后轨迹点的信息
    last_p = histor[-1, :2]
    last_v = histor[-1, 2:4]

    cv_p = last_p + last_v * T #恒速度推导
    pred_out = pred - (pred[0, :2] - cv_p)  #归化

    return pred_out


def custom_loss(occupancy_value):
    def loss_function(y_true, y_pred):
        mask = tf.math.not_equal(y_true, occupancy_value)
        masked_squared_error = tf.square(tf.boolean_mask(y_true - y_pred, mask))
        return tf.reduce_mean(masked_squared_error)

    return loss_function


def restore_original_position(n_input, position_difference, A, T):
    # 获取起点位置和速度
    x0, y0, vx0, vy0 = A[0, 0], A[0, 1], A[0, 2], A[0, 3]

    # 还原原始轨迹
    num_positions = len(position_difference)
    restored_trajectory = np.zeros((num_positions, 2))
    restored_trajectory[0, :] = np.array([x0, y0])  # 初始位置
    if num_positions>1:
        for i in range(n_input, num_positions):
            dt = T * i  # 当前时间戳
            dx = vx0 * dt  # x方向的位移
            dy = vy0 * dt  # y方向的位移
            restored_trajectory[i, :] = position_difference[i, :2] + (np.array([dx, dy]) + A[0, :2])

    return restored_trajectory[n_input:, :]


def is_sorted(trj):
    flag = 0
    if(trj == sorted(trj, reverse=True)):
        flag=1
    return flag


def anti_norm(nor_data, max_boundary, min_boundary):
    # 反归一化
    # 此处显示详细说明
    max_boundary = max_boundary.flatten()
    min_boundary = min_boundary.flatten()
    orin_data = nor_data.copy()
    if not (nor_data == 0).all():
        for i in range(nor_data.shape[1]):
            for j in range(nor_data.shape[0]):
                orin_data[j, i] = ((max_boundary[i] - min_boundary[i]) * (nor_data[j, i] - 0.001)) + min_boundary[i]

    return orin_data


def ade_calculation(input_arg1, input_arg2):
    mde = []
    diff = input_arg1 - input_arg2
    diff = np.reshape(diff, [-1, 2])
    for i in range(diff.shape[0]):
            mde.append(np.linalg.norm(diff[i][:]))
    output = np.mean(mde)

    return output


def save_prediction_results(args, ADE, FDE, All_histor_motion_output, All_pred_motion_output, All_obse_motion_output):
    data_set_dir1 = os.path.join(args.data_dir, args.sample_dir, 'data_set')
    # 创建预测结果文件夹
    prediction_data_path = os.path.join(data_set_dir1, "预测结果")

    if not os.path.exists(prediction_data_path):
        os.makedirs(prediction_data_path)

    # 保存
    np.save(os.path.join(prediction_data_path, "All_histor_motion_output.npy"), All_histor_motion_output)
    np.save(os.path.join(prediction_data_path, "All_pred_motion_output.npy"), All_pred_motion_output)
    np.save(os.path.join(prediction_data_path, "All_obse_motion_output.npy"), All_obse_motion_output)
    np.save(os.path.join(prediction_data_path, "ADE.npy"), ADE)
    np.save(os.path.join(prediction_data_path, "FDE.npy"), FDE)

    """
    # 保存历史轨迹、预测轨迹和观测轨迹到.csv文件
    np.savetxt(os.path.join(prediction_data_path, "All_histor_motion_output.csv"),
               All_histor_motion_output.reshape(-1, 1), delimiter=",")
    np.savetxt(os.path.join(prediction_data_path, "All_pred_motion_output.csv"),
               All_pred_motion_output.reshape(-1, 1), delimiter=",")
    np.savetxt(os.path.join(prediction_data_path, "All_obse_motion_output.csv"),
               All_obse_motion_output.reshape(-1, 1), delimiter=",")
    np.savetxt(os.path.join(prediction_data_path, "ADE.csv"),
               ADE.reshape(-1, ADE.shape[-1]), delimiter=",")
    np.savetxt(os.path.join(prediction_data_path, "FDE.csv"),
               FDE.reshape(-1, FDE.shape[-1]), delimiter=",")
    """

    print(f"Prediction results saved successfully in {prediction_data_path}")

"""
def plot_spiderweb(categories, values, title='Spider Web Chart', fill_color='#FFDEAD', line_color='#CD853F',
                   bg_color='#FFFFFF'):
    '''
    绘制蛛网图

    参数：
    categories -- str 数组，表示每个数据点对应的类别标签
    values -- float 数组，表示每个数据点的值
    title -- str，表示蛛网图的标题，默认值为 "Spider Web Chart"
    fill_color -- str，表示填充颜色的十六进制字符串，默认为 "#FFDEAD"（桃色）
    line_color -- str，表示线条颜色的十六进制字符串，默认为 "#CD853F"（秘鲁棕色）
    bg_color -- str，表示背景颜色的十六进制字符串，默认为 "#FFFFFF"（白色）

    返回值：
    无返回值，直接显示绘制出的蛛网图。
    '''
    # 计算角度
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 闭合

    # 绘制蛛网图
    fig = plt.figure(facecolor=bg_color)
    ax = plt.subplot(111, polar=True)  # 极坐标系
    ax.plot(angles, values, 'o-', linewidth=1, color=line_color)  # 连线
    ax.fill(angles, values, color=fill_color, alpha=0.25)  # 填充

    # 设置刻度标签
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, categories)

    # 设置极径范围
    ymax = max(values) + 1
    ax.set_rlim(0, ymax)

    # 设置标题
    plt.title(title)

    # 显示图形
    plt.show()
"""


# 预测
def trajectory_prediction(args, data_dir1, data_set_dir2, data_set_dir3):

    T = args.Decision_interval * 0.12
    #n_input = 4 #输入轨迹点的个数
    #n_output = 6
    n_input = int(args.Obs_length / args.Decision_interval)  #4 #输入轨迹点的个数   n_input*n_feature
    n_output = int(args.Pred_length / args.Decision_interval)  #6

    # 读取数据

    x_test = pd.read_csv(os.path.join(data_dir1, 'x_test.txt'), header=None)
    y_test = pd.read_csv(os.path.join(data_dir1, 'y_test.txt'), header=None)
    max_boundary = pd.read_csv(os.path.join(data_dir1, 'max_boundary.txt'), header=None)
    min_boundary = pd.read_csv(os.path.join(data_dir1, 'min_boundary.txt'), header=None)

    # 读取数据2
    path_x_test2 = os.path.join(data_set_dir2, "x_test.txt")
    path_y_test2 = os.path.join(data_set_dir2, "y_test.txt")

    x_test_efficiency = np.array(pd.read_csv(path_x_test2, header=None))
    y_test_efficiency = np.array(pd.read_csv(path_y_test2, header=None))

    # 读取数据3
    path_x_test3 = os.path.join(data_set_dir3, "x_test.txt")
    path_y_test3 = os.path.join(data_set_dir3, "y_test.txt")

    x_test_safety = np.array(pd.read_csv(path_x_test3, header=None))
    y_test_safety = np.array(pd.read_csv(path_y_test3, header=None))


    # 读取特征数
    find_dim = np.array(x_test)
    find_dim = find_dim[0, :].reshape(n_input, -1)
    n_feature = int(find_dim.shape[1] / args.max_bicycle_num)  # 读取输入特征数
    n_input = int(args.Obs_length / args.Decision_interval)  #4 #输入轨迹点的个数   n_input*n_feature
    n_output = int(args.Pred_length / args.Decision_interval)  #6

    # 读取模型
    # 自定义loss 1
    custom_loss_obj = MotionConstraintLoss(args, n_output, n_feature)
    loss = custom_loss_obj.my_loss
    custom_objects = {'my_loss': loss}



    # 自定义loss 2
    #custom_loss_obj = custom_loss(args.occupancy_value)
    #custom_objects = {'loss_function': custom_loss_obj}

    # 读取
    #trained_model = tf.keras.models.load_model(os.path.join(data_dir1, "MotionPrediction_model.h5"), custom_objects=
                                #{'TransformerLayer': TransformerLayer, 'my_loss': loss}) #训练好的模型
    #trained_model = tf.keras.models.load_model(os.path.join(data_dir1, "MotionPrediction_model.h5"), custom_objects=
                                #{'my_loss': loss})  # 训练好的模型
    trained_model = load_model(os.path.join(data_dir1, "MotionPrediction_model.h5"))  # 训练好的模型

    trained_model.summary()
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  #设置环境变量，设置结果显示的级别，解释在word里
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    max_boundary = np.array(max_boundary)
    min_boundary = np.array(min_boundary)

    y_pred_all = []
    ade_all =[]
    fde_all = []
    Pred_Traj_features = []
    Obsed_Traj_features = []
    total = x_test.shape[0]

    #permutation = np.random.permutation(x_test.shape[0]) # 打乱样本
    #x_test = x_test[permutation]
    #y_test = y_test[permutation]

    All_histor_motion_output = np.zeros((total, args.max_bicycle_num, n_input, 2))
    All_pred_motion_output = np.zeros((total, args.max_bicycle_num, n_output, 2))
    All_obse_motion_output = np.zeros((total, args.max_bicycle_num, n_output, 2))
    ADE = np.zeros((total, args.max_bicycle_num))
    FDE = np.zeros((total, args.max_bicycle_num))

    for i in range(total):  #x_test.shape[0]   #820,880,2
        # 基于训练好的模型进行预测
        x_input = x_test[i, :]
        x_eff = x_test_efficiency[i, :]
        x_safety = x_test_safety[i, :]
        if not args.if_iter_pre:
            y_pred = trained_model.predict([x_input.reshape(1, -1), x_eff.reshape(1, -1), x_safety.reshape(1, -1)])
        else:
            x_input_iter = x_input.reshape(1, -1)
            x_eff_iter = x_eff.reshape(1, -1)
            x_safety_iter = x_safety.reshape(1, -1)
            # 由于迭代预测需要安全和效率持续输入，因为放弃这个了
            # for frame_index in range(n_output):
            y_pred = trained_model.predict([x_input_iter, x_eff_iter, x_safety_iter])


        if len(y_pred_all) == 0:
            y_pred_all = y_pred
        else:
            y_pred_all = np.vstack((y_pred_all, y_pred))
        # 换算回来
        pred_motion = y_pred.reshape(-1, args.max_bicycle_num, 2)  #n_feature  这里不缩减y时，就会是n_feature
        obse_motion = y_test[i, :].reshape(-1, args.max_bicycle_num, n_feature)
        histor_motion = x_input.reshape(-1, args.max_bicycle_num, n_feature)

        obse_motion_output_0 = []
        pred_motion_output_0 = []
        # 拼接回原来形状
        pred_motion = np.concatenate((pred_motion, np.zeros((n_output, args.max_bicycle_num, (n_feature - 2)))), axis=2)
        for j in range(args.max_bicycle_num):#对于每个自行车来说
            # 若预测时长不满足要求，直接跳过

            #print(j)
            # 逆归一化
            pred_motion_output = anti_norm(pred_motion[:, j, :], max_boundary, min_boundary)
            obse_motion_output = anti_norm(obse_motion[:, j, :], max_boundary, min_boundary)
            histor_motion_output = anti_norm(histor_motion[:, j, :], max_boundary, min_boundary)

            # 基于历史趋势优化轨迹
            pred_motion_output[:, :2] = bench_based_last_point(pred_motion_output[:, :2], histor_motion_output[:, :4],
                                                               0.12*args.Decision_interval)


            # 判断是否是错误数据，若是则跳过
            #if obse_motion_output[0,8] != 1:
                #continue
            if (histor_motion_output[:, 0] == args.occupancy_value).any() or (obse_motion_output[:, 0] == args.occupancy_value).any():
                continue
            if not is_sorted((list(histor_motion_output[:, 0])+list(obse_motion_output[:, 0]))):
                continue

            # 保存数据
            All_histor_motion_output[i, j, :, :] = histor_motion_output[:, :2]
            All_pred_motion_output[i, j, :, :] = pred_motion_output[:, :2]
            All_obse_motion_output[i, j, :, :] = obse_motion_output[:, :2]

            if len(obse_motion_output_0) > 0:
                obse_motion_output_0 = np.hstack((obse_motion_output_0, obse_motion_output[:, :2]))
                pred_motion_output_0 = np.hstack((pred_motion_output_0, pred_motion_output[:, :2]))
            else:
                obse_motion_output_0 = obse_motion_output[:, :2]
                pred_motion_output_0 = pred_motion_output[:, :2]

            #plt.show()

            #计算误差


            ade = ade_calculation(pred_motion_output[:, :2], obse_motion_output[:, :2])
            fde = ade_calculation(pred_motion_output[-1, :2], obse_motion_output[-1, :2])

            ADE[i, j] = ade
            FDE[i, j] = fde

            ade_all.append(ade)
            fde_all.append(fde)

            # 计算轨迹特征
            pred_lateral_displacement, pred_velocity, pred_acceleration, pred_turning_rate = calculate_trajectory_indicators(pred_motion_output[:, :2])
            obse_lateral_displacement, obse_velocity, obse_acceleration, obse_turning_rate = calculate_trajectory_indicators(obse_motion_output[:, :2])

            Pred_Traj_features.append(np.hstack((pred_lateral_displacement, pred_velocity, pred_acceleration, pred_turning_rate)))
            Obsed_Traj_features.append(np.hstack((obse_lateral_displacement, obse_velocity, obse_acceleration, obse_turning_rate)))


        progress = (i + 1) / total  # 计算完成百分比
        if i % 10 == 0:
            print(f"Progress: {progress:.2%}")  # 打印当前完成百分比
        #ade_all = np.array(ade_all)
        #fde_all = np.array(fde_all)

    ade_all = np.array(ade_all)
    fde_all = np.array(fde_all)


    #plt.hist(ade_all)
    #plt.show()

    # 写出数据
    save_prediction_results(args, ADE, FDE, All_histor_motion_output, All_pred_motion_output, All_obse_motion_output)

    return All_histor_motion_output, All_pred_motion_output, All_obse_motion_output, np.mean(ade_all), np.mean(fde_all)

