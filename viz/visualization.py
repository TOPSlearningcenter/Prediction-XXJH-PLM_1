# 该脚本的作用是利用已知的起终点进行轨迹生成，完成轨迹预测
# 以下开始
import numpy as np
import os
import matplotlib.pyplot as plt


def load_prediction_results(prediction_data_path):
    """
    # 读取CSV文件
    All_histor_motion_output = np.loadtxt(os.path.join(prediction_data_path, "All_histor_motion_output.csv"),
                                          delimiter=",")
    All_pred_motion_output = np.loadtxt(os.path.join(prediction_data_path, "All_pred_motion_output.csv"), delimiter=",")
    All_obse_motion_output = np.loadtxt(os.path.join(prediction_data_path, "All_obse_motion_output.csv"), delimiter=",")

    ADE = np.loadtxt(os.path.join(prediction_data_path, "ADE.csv"), delimiter=",")
    FDE = np.loadtxt(os.path.join(prediction_data_path, "FDE.csv"), delimiter=",")
    """
    # 读取npy文件
    All_histor_motion_output = np.load(os.path.join(prediction_data_path, "All_histor_motion_output.npy"))
    All_pred_motion_output = np.load(os.path.join(prediction_data_path, "All_pred_motion_output.npy"))
    All_obse_motion_output = np.load(os.path.join(prediction_data_path, "All_obse_motion_output.npy"))
    ADE = np.load(os.path.join(prediction_data_path, "ADE.npy"))
    FDE = np.load(os.path.join(prediction_data_path, "FDE.npy"))



    # 返回加载的数据
    return ADE, FDE, All_histor_motion_output, All_pred_motion_output, All_obse_motion_output


def sliding_window_smooth(data, window_size):
    num_rows, num_cols = data.shape  # 获取数据的行数和列数
    smoothed_data = np.zeros_like(data)  # 创建与原始数据相同形状的数组来存储平滑后的数据

    for col in range(num_cols):
        for row in range(num_rows):
            start = max(0, row - window_size // 2)  # 窗口的起始位置
            end = min(num_rows, row + window_size // 2 + 1)  # 窗口的结束位置
            smoothed_data[row, col] = np.mean(data[start:end, col])

    return smoothed_data[1:, :]


def visualize_trajectories(args, ID, Sample_number, model_path, all_ade, All_histor_motion_output, All_pred_motion_output,
                           All_obse_motion_output):
    if len(ID) != 0:
        Sample_number = len(ID)
    # 创建存储图片的文件夹
    visualizations_path = os.path.join(model_path, "可视化")
    if not os.path.exists(visualizations_path):
        os.makedirs(visualizations_path)

    # 将误差为0的地方替换为 NaN
    all_ade_nan = np.where(all_ade == 0, np.nan, all_ade)
    average_all_ade = np.nanmean(all_ade_nan, axis=1)

    # 将误差为0的地方替换为 NaN
    all_ade_with_nan = np.where(average_all_ade == 0, np.nan, average_all_ade)

    # 按误差从小到大排序
    sorted_indices = np.argsort(all_ade_with_nan)

    # 计数器，用于跟踪已输出的图片数
    count = 0

    for idx, i in enumerate(sorted_indices):
        ade = all_ade_nan[i, :]
        N_bicycle = len(ade[~np.isnan(ade)])
        #if N_bicycle is not (args.max_bicycle_num - 0):
        if N_bicycle is 1:
            #print('个体数不足')
            continue
        if len(ID) >= (count+1):
            i = ID[count]
        else:
            if len(ID) != 0:
                continue

        plt.figure()
        for j in range(All_histor_motion_output.shape[1]):
            # 停止条件：如果已经输出的图片数达到了 Sample_number，则停止循环
            if count >= Sample_number:
                return

            histor_motion_output = All_histor_motion_output[i, j, :, :]
            pred_motion_output = All_pred_motion_output[i, j, :, :]
            obse_motion_output = All_obse_motion_output[i, j, :, :]

            # 忽略轨迹中包含0值的点
            histor_motion_output[histor_motion_output == 0] = np.nan
            obse_motion_output[obse_motion_output == 0] = np.nan
            pred_motion_output[pred_motion_output == 0] = np.nan

            # 绘制轨迹
            pred_motion_output = sliding_window_smooth(np.concatenate(([histor_motion_output[-1, :]], pred_motion_output)), 3)

            plt.plot(histor_motion_output[:, 0], histor_motion_output[:, 1], 'b*-')
            plt.plot(np.concatenate(([histor_motion_output[-1, 0]], obse_motion_output[:, 0])),
                     np.concatenate(([histor_motion_output[-1, 1]], obse_motion_output[:, 1])), 'go-')
            plt.plot(np.concatenate(([histor_motion_output[-1, 0]], pred_motion_output[:, 0])),
                     np.concatenate(([histor_motion_output[-1, 1]], pred_motion_output[:, 1])), 'r^-')
            #plt.show()

        plt.plot([], [], 'b*-', label='Histories')  # 空数据用于图例
        plt.plot([], [], 'go-', label='Observations')  # 空数据用于图例
        plt.plot([], [], 'r^-', label='Prediction')  # 空数据用于图例
        plt.title(f'Trajectory Visualization for Sample {i}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()

        # 保存图片
        img_file = os.path.join(visualizations_path, f"bicycle_trajectory_sample_{i}.png")
        plt.savefig(img_file)
        plt.close()

        # 计数
        count += 1

    print(f"Trajectory visualizations saved in {visualizations_path}")


############## 开始做图#########################
def Visualization(args, ID):

    Sample_number = 1  # 样本数目

    # 指定文件路径
    data_set_dir1 = os.path.join(args.data_dir, args.sample_dir, 'data_set')
    # 创建预测结果文件夹
    prediction_data_path = os.path.join(data_set_dir1, "预测结果")

    # 读取预测数据
    ADE, FDE, All_histor_motion_output, All_pred_motion_output, All_obse_motion_output = \
        load_prediction_results(prediction_data_path)


    n_input = int(args.Obs_length / args.Decision_interval)  # 4 #输入轨迹点的个数   n_input*n_feature
    n_output = int(args.Pred_length / args.Decision_interval)  # 6

    """
    # 恢复到原来的形状
    All_histor_motion_output = All_histor_motion_output.reshape(-1, args.max_bicycle_num, n_input, 2)
    All_pred_motion_output = All_pred_motion_output.reshape(-1, args.max_bicycle_num, n_output, 2)
    All_obse_motion_output = All_obse_motion_output.reshape(-1, args.max_bicycle_num, n_output, 2)

    ADE = ADE.reshape(-1, args.max_bicycle_num)
    FDE = FDE.reshape(-1, args.max_bicycle_num)
    """


    # 做图
    visualize_trajectories(args,ID, Sample_number, prediction_data_path, ADE, All_histor_motion_output, All_pred_motion_output,
                           All_obse_motion_output)





############ start #############################
