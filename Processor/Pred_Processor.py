import numpy as np
import os
import random
import dill
import matplotlib.pyplot as plt


class DataSetProcessing:
    def __init__(self, args):
        self.args = args

    def data_set_construction(self, handle_data, data_type, pred_length):
        ## 构造深度学习的数据集
        handle_data[:, 0] = (handle_data[:, 0] - 1).astype(int)  #因为不是从0开始的

        x_handle_data = []
        y_handle_data = []
        x_efficiency_data = []
        y_efficiency_data = []
        x_safety_data = []
        y_safety_data = []

        sample_num = 0
        num = np.unique(handle_data[:, 0])  #全部帧ID
        total_num = int(len(num))

        for frame in range(total_num):  # 对于每一个样本
            frame_seq = np.arange(num[frame], num[frame] + self.args.Obs_length + pred_length)
            extract_sample_data = handle_data[np.isin(handle_data[:, 0], frame_seq), :]
            bicycle_id_seq = np.unique(extract_sample_data[:, 12])
            feat_num_orin = 11  # 9 = 7个特征值+ 帧ID和车子ID
            sample = np.zeros((len(frame_seq), self.args.max_bicycle_num, feat_num_orin))  # 用来存放样本
            bike_id_dict = {}
            for i, bike_id in enumerate(bicycle_id_seq):
                bike_id_dict[bike_id] = i

            for j in range(
                    self.args.max_bicycle_num):  # 每个id  range(np.min((len(bicycle_id_seq), self.args.max_bicycle_num))):
                for index_i in range(len(frame_seq)):  # 每个帧
                    try:
                        feat_all_sample = extract_sample_data[(extract_sample_data[:, 0] == frame_seq[index_i]) & (
                                extract_sample_data[:, 12] == bicycle_id_seq[j]), :]

                        if bicycle_id_seq[j] not in bike_id_dict:
                            continue

                        idx = bike_id_dict[bicycle_id_seq[j]]
                        sample[index_i, idx, 0] = feat_all_sample[0, 0]  # 帧
                        sample[index_i, idx, 1] = feat_all_sample[0, 12]  # 自行车id
                        sample[index_i, idx, 2:8] = feat_all_sample[0, 1:7]
                        #sample[index_i, idx, 2:8] = feat_all_sample[0, 1:7]
                        sample[index_i, idx, -1] = feat_all_sample[0, -1]
                    except Exception as e:
                         #sample[index_i, j, :] = np.nan
                         #print('Error: ', e)  # 输出异常信息
                        #print('error')
                        continue

            # 按照起点CV基准化
            for index_b in range(self.args.max_bicycle_num):
                sample[:, index_b, 8:10] = self.calculate_position_difference(sample[:, index_b, 2:6], 0.12)

            """
            for index_i in range(len(frame_seq)):# 每个帧
                #index_i = frame_seq[i]
                for j in range(np.min((len(bicycle_id_seq), self.args.max_bicycle_num))): #每个id
                    try:
                        feat_all_sample = extract_sample_data[(extract_sample_data[:, 0] == frame_seq[index_i]) & (
                                    extract_sample_data[:, 12] == bicycle_id_seq[j]), :]
                        sample[index_i, j, 0] = feat_all_sample[0, 0]  # 帧
                        sample[index_i, j, 1] = feat_all_sample[0, 12]  # 自行车id
                        sample[index_i, j, 2:8] = feat_all_sample[0, 1:7]
                        sample[index_i, j, -1] = feat_all_sample[0, -1]
                    except Exception as e:
                        #sample[index_i, j, :] = np.nan
                        #print('Error: ', e)  # 输出异常信息
                        continue
            """
            # 按照数据饱满程度重新排列第二个维度
            lengths = np.sum(np.sign(sample[:, :, 3]), axis=0)
            sorted_indices = np.argsort(lengths)[::-1]
            sample = sample[:, sorted_indices]
            if np.max(lengths) < sample.shape[0]:
                continue

            # input2: 在这里实行计算效率序列
            efficiency_sample = self.calculate_bicycle_movement(sample)  #_every
            # input3: 在这里实行计算安全序列
            safety_sample = self.calculate_bicycle_safety(sample)

            # 操作 去掉时间帧和ID信息

            sample = sample[:, :, 2:] #特征值维度仅保留后面的
            ##
            #sample = self.rotate_system_by_average_angle(sample)  #旋转一个角度
            #sample[:, :, 2:-1] = 0  #旋转后没再计算运动参数，这里应该计算
            #if not self.is_kinematically_valid(sample):
                #print('gg')
                #continue
            #print('gg')
            # 下面仅在测试时使用
            """
            import matplotlib.pyplot as plt
            plt.plot(sample[:,0, 0], sample[:,0, 1], 'b*-', label='Histories')
            plt.plot(sample[:, 1, 0], sample[:, 1, 1], 'b*-', label='Histories')
            plt.plot(sample[:, 2, 0], sample[:, 2, 1], 'b*-', label='Histories')
            plt.show()
            """

            if self.args.if_enhance_data:
                expanding_num = 1  #10
                random_thr = 0    #1
                #"""
                if data_type is "test_set": # 测试集不扩充
                    expanding_num = 1
                    random_thr = 0
                #"""
                for inter_random in range(0, expanding_num):
                    #print(inter_random)
                    if inter_random is 0:
                        random_thr = 0

                    sample_random = sample
                    sample_random = np.where(sample_random == 0, np.nan, sample_random)
                    sample_random[:, :, 0] = sample_random[:, :, 0] + np.random.uniform(-(random_thr / 2),
                                               (random_thr / 2), size=sample_random[:, :, 0].shape)
                    sample_random[:, :, 1] = sample_random[:, :, 1] + np.random.uniform(-(random_thr / 4),
                                                (random_thr / 4), size=sample_random[:, :, 1].shape)

                    sample_random = np.nan_to_num(sample_random)

                    #操作2 分开训练和预测集
                    x_train_sample = sample_random[:self.args.Obs_length, :, :] #在时间帧上操作
                    y_train_sample = sample_random[self.args.Obs_length:, :, :]

                    x_efficiency_sample = efficiency_sample[:self.args.Obs_length, :, :] #在时间帧上操作
                    y_efficiency_sample = efficiency_sample[self.args.Obs_length:, :, :]

                    x_safety_sample = safety_sample[:self.args.Obs_length, :, :, :] #在时间帧上操作
                    y_safety_sample = safety_sample[self.args.Obs_length:, :, :, :]

                    #"""
                    #操作3 切片操作，涉及所取位置是头、还是尾
                    x_train_sample_sli = x_train_sample[0: x_train_sample.shape[0]: self.args.Decision_interval, :, :] #切片
                    y_train_sample_sli = y_train_sample[0: y_train_sample.shape[0]: self.args.Decision_interval, :, :]

                    x_efficiency_sample_sli = x_efficiency_sample[0: x_efficiency_sample.shape[0]: self.args.Decision_interval, :, :] #切片
                    y_efficiency_sample_sli = y_efficiency_sample[0: y_efficiency_sample.shape[0]: self.args.Decision_interval, :, :]

                    x_safety_sample_sli = x_safety_sample[0: x_safety_sample.shape[0]: self.args.Decision_interval, :, :, :] #切片
                    y_safety_sample_sli = y_safety_sample[0: y_safety_sample.shape[0]: self.args.Decision_interval, :, :, :]
                    #"""
                    # 样本筛选1：判断是否有中间为0的矩阵，有的话就全赋予0
                    id_zeros = np.where((x_train_sample_sli[-1, :, 0] == 0)|(y_train_sample_sli[0, :, 0] == 0))
                    for idx_id in id_zeros:
                        x_train_sample_sli[:, idx_id, :] = 0
                        y_train_sample_sli[:, idx_id, :] = 0

                    # 样本筛选2：去掉无效样本,若观测预测集无一个全时间数据，则跳过
                    if np.max(np.sign(abs(x_train_sample_sli[-1, :, 0])) + np.sign(abs(y_train_sample_sli[0, :, 0]))) < 2:
                        continue


                    #if np.all(x_train_sample_sli[0, :, 0] == 0) or np.all(y_train_sample_sli[-1, :, 0] == 0):
                        #continue  #刚刚未运行
                    #
                    if np.all(np.sum(np.sign(np.abs(sample[:, :, 0])), axis=0) != sample.shape[0]):  #np.all(np.sum(np.sign(np.abs(sample[:, :, 0])), axis=0) != sample.shape[0]):
                        continue


                    """
                    plt.plot(x_train_sample_sli[:, 0, 0], x_train_sample_sli[:, 0, 1], 'g*-')
                    plt.plot(y_train_sample_sli[:, 0, 0], y_train_sample_sli[:, 0, 1], 'go-')
                    plt.plot(sample[:, 0, 2], sample[:, 0, 3], 'go-')
                    
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.title('预测轨迹图')
                    plt.show()
                    """
                    #若不是完全满自行车则排除   #我的数据不要需要运行  aop数据需要
                    #if np.sum(np.sign(np.abs(sample[:, :, 0]))) != sample.shape[0] * self.args.max_bicycle_num:
                        #continue
                    #data_type == 'test_set' and
                    dausuu = sample[:, 0, :]

                    # 追加
                    x_handle_data.append(x_train_sample_sli.flatten().tolist())  #追加
                    y_handle_data.append(y_train_sample_sli.flatten().tolist())  #追加

                    x_efficiency_data.append(x_efficiency_sample_sli.flatten().tolist())  #追加
                    y_efficiency_data.append(y_efficiency_sample_sli.flatten().tolist())  #追加

                    x_safety_data.append(x_safety_sample_sli.flatten().tolist())  #追加
                    y_safety_data.append(y_safety_sample_sli.flatten().tolist())  #追加

                    progress = (frame + 1) / total_num * 100
                    sample_num += 1
                    if frame % 10 == 0:
                        print(f"{data_type.capitalize()} Processing {frame + 1}/{total_num} ({progress:.2f}%)，样本量为{sample_num}个")
                    #print(f"str(data_type) Processing {id + 1}/{total_num} ({progress:.2f}%)")

        x_handle_data = np.array(x_handle_data)
        y_handle_data = np.array(y_handle_data)
        x_efficiency_data = np.array(x_efficiency_data)
        y_efficiency_data = np.array(y_efficiency_data)
        x_safety_data = np.array(x_safety_data)
        y_safety_data = np.array(y_safety_data)
        #np.max(x_efficiency_data)
        return x_handle_data, y_handle_data, x_efficiency_data, y_efficiency_data, x_safety_data, y_safety_data

    @staticmethod
    def sliding_window_smooth(data, window_size):
        num_rows, num_cols = data.shape  # 获取数据的行数和列数
        smoothed_data = np.zeros_like(data)  # 创建与原始数据相同形状的数组来存储平滑后的数据

        for col in range(num_cols):
            for row in range(num_rows):
                start = max(0, row - window_size // 2)  # 窗口的起始位置
                end = min(num_rows, row + window_size // 2 + 1)  # 窗口的结束位置
                smoothed_data[row, col] = np.mean(data[start:end, col])

        return smoothed_data

    @staticmethod
    def is_kinematically_valid(sample, max_speed_threshold=20, max_acceleration_threshold=5,
                               max_angle_change_threshold=45):
        time_inter = 0.5
        """
        检查每个行人的运动是否符合运动学规律。

        Parameters
        ----------
        sample : numpy.ndarray
            维度为 [时间步长, 对象id, 特征] 的矩阵。
            其中特征维度的前两列为 [x, y] 坐标。
        max_speed_threshold : float
            最大速度阈值，如果速度超过此值则认为不合理。
        max_acceleration_threshold : float
            最大加速度阈值，如果加速度超过此值则认为不合理。
        max_angle_change_threshold : float
            最大角度变化阈值，单位为度，如果角度变化超过此值则认为不合理。

        Returns
        -------
        valid : bool
            返回每个对象是否符合运动学规律的布尔值。
        """
        num_time_steps, num_pedestrians, _ = sample.shape

        for pid in range(num_pedestrians):
            # 提取该对象的轨迹
            trajectory = sample[:, pid, :2]

            """
            # 判断是否有相同的轨迹点
            trajectory_points = set()  # 使用集合来存储每个行人的轨迹点

            for t in range(num_time_steps):
                position = tuple(sample[t, pid, :2])  # 将 (x, y) 坐标转换为元组以便于比较

                if position in trajectory_points:
                    return False  # 如果发现重复的轨迹点，立即返回 "否"
                trajectory_points.add(position)
            """

            #"""
            # 如果轨迹只有一个点，无法判断运动学规律，跳过
            if len(trajectory) < 2 or np.all(trajectory == 0):
                continue

            for t in range(1, num_time_steps):
                # 计算速度（距离）
                displacement = trajectory[t] - trajectory[t - 1]
                distance = np.linalg.norm(displacement)
                speed = distance / time_inter # 假设时间步长为1

                if speed > max_speed_threshold:
                    return False  # 速度过大，不符合运动学规律

                # 计算加速度
                if t > 1:
                    previous_displacement = trajectory[t - 1] - trajectory[t - 2]
                    previous_speed = np.linalg.norm(previous_displacement)
                    acceleration = (speed - previous_speed)/time_inter  # 假设时间步长为1
                    if abs(acceleration) > max_acceleration_threshold:
                        return False  # 加速度过大，不符合运动学规律

                # 计算方向变化（角度变化）
                if np.all(displacement != 0):  # 确保不是静止状态
                    current_angle = np.arctan2(displacement[1], displacement[0])
                    if t > 1 and np.all(previous_displacement != 0):
                        previous_angle = np.arctan2(previous_displacement[1], previous_displacement[0])
                        angle_change = np.degrees(current_angle - previous_angle)

                        # 确保角度变化合理
                        if abs(angle_change) > max_angle_change_threshold:
                            return False  # 角度变化过大，不符合运动学规律
            #"""

        return True  # 所有对象都符合运动学规律

    @staticmethod
    def calculate_average_angle(angles_dict):
        """
        计算角度字典中的平均角度。

        Parameters
        ----------
        angles_dict : dict
            每个行人 ID 对应的角度，单位为度。

        Returns
        -------
        average_angle : float
            所有行人角度的平均值。
        """
        valid_angles = [angle for angle in angles_dict.values() if angle is not None]
        if valid_angles:
            return np.mean(valid_angles)
        else:
            return 0  # 如果没有有效角度，返回 0 度

    @staticmethod
    def calculate_all_change_angles(sample):
        """
        计算所有行人的起点和终点之间的变化角度。

        Parameters
        ----------
        sample : numpy.ndarray
            维度为 [时间步长, 行人id, 特征] 的矩阵。
            其中特征维度的前两列为 [x, y] 坐标。

        Returns
        -------
        angles : dict
            每个行人 ID 对应的角度，单位为度。
        """
        angles = {}
        num_pedestrians = sample.shape[1]

        for pid in range(num_pedestrians):
            trajectory = sample[:, pid, :2]
            if len(trajectory) < 2 or np.all(trajectory == 0):
                angles[pid] = None  # 如果轨迹全为0或只有一个点，角度无效
                continue

            start_point = trajectory[0]
            end_point = trajectory[-1]
            if np.all(start_point == 0) or np.all(end_point == 0):
                angles[pid] = None  # 忽略起点或终点全为0的行人
                continue

            velocity_vector = end_point - start_point
            if np.all(velocity_vector == 0):
                angles[pid] = None  # 如果起点和终点相同，角度无效
                continue

            angle = np.arctan2(velocity_vector[1], velocity_vector[0])
            angle_in_degrees = np.degrees(angle)
            angles[pid] = angle_in_degrees

        return angles

    def rotate_system_by_average_angle(self, sample):
        """
        根据所有对象的起终点变化角度的平均值旋转整个坐标系，并将其对齐到最靠近右下角对象的起点处为 (1, 1)。
        """
        angles_dict = self.calculate_all_change_angles(sample)
        average_angle = self.calculate_average_angle(angles_dict)

        # 将角度转换为负轴方向，计算旋转的角度
        rotate_angle = -average_angle
        radians = np.radians(rotate_angle)

        rotation_matrix = np.array([[np.cos(radians), -np.sin(radians)],
                                    [np.sin(radians), np.cos(radians)]])

        sample_rotated = sample.copy()

        # Step 1: 旋转所有对象坐标
        for t in range(sample.shape[0]):
            for pid in range(sample.shape[1]):
                # 旋转坐标
                position = sample[t, pid, :2]
                if not np.all(position == 0):  # 忽略坐标全为0的点
                    sample_rotated[t, pid, :2] = np.dot(rotation_matrix, position)
                """
                # 旋转速度 (vx, vy)
                velocity = sample[t, pid, 2:4]
                if not np.all(velocity == 0):  # 忽略速度全为0的点
                    sample_rotated[t, pid, 2:4] = np.dot(rotation_matrix, velocity)

                # 旋转加速度 (ax, ay)
                acceleration = sample[t, pid, 4:6]
                if not np.all(acceleration == 0):  # 忽略加速度全为0的点
                    sample_rotated[t, pid, 4:6] = np.dot(rotation_matrix, acceleration)
                """

        # Step 2: 找出最靠近右下角的行人起点
        start_positions = sample_rotated[0, :, :2]  # 获取所有行人的起点坐标

        # 标记起点为 (0, 0) 的行人
        non_zero_mask = ~np.all(start_positions == 0, axis=1)

        # 过滤掉起点为 (0, 0) 的行人
        non_zero_start_positions = start_positions[non_zero_mask]

        # 如果没有非零起点的行人，需要处理这种情况
        if len(non_zero_start_positions) == 0:
            raise ValueError("所有行人的起点均为 (0, 0)，无法找到最靠近右下角的行人。")

        # 找到 x 最大且 y 最小的非零行人索引
        max_x_index = np.argmax(non_zero_start_positions[:, 0])  # 找到 x 最大的非零行人索引
        min_y_index = np.argmin(non_zero_start_positions[:, 1])  # 找到 y 最小的非零行人索引

        # 获取在原数组中的实际索引
        selected_index = np.arange(start_positions.shape[0])[non_zero_mask][min_y_index]

        reference_start_point = start_positions[selected_index]  # 找到参照行人的起点

        # Step 3: 将参照行人的起点对齐到 (1, 1)
        translation_vector = np.array([1, 1]) - reference_start_point

        # 平移整个 sample 使得参照行人的起点为 (1, 1)
        for t in range(sample_rotated.shape[0]):
            for pid in range(sample_rotated.shape[1]):
                sample_rotated[t, pid, :2] += translation_vector

        # Step 4: 将原本为 (0, 0) 的点重新设置为 (0, 0)
        for t in range(sample.shape[0]):
            for pid in range(sample.shape[1]):
                if np.all(sample[t, pid, :2] == 0):  # 如果原始数据中该点是 (0, 0)
                    sample_rotated[t, pid, :2] = [0, 0]

        # Step 4 平滑轨迹
        for pid in range(sample_rotated.shape[1]):
            # 获取该行人的轨迹
            trajectory = sample_rotated[:, pid, :2]

            # 标记轨迹中为 (0, 0) 的点
            zero_mask = np.all(trajectory == 0, axis=1)

            # 只对非 (0, 0) 的点进行平滑
            non_zero_trajectory = trajectory[~zero_mask]

            if len(non_zero_trajectory) > 0:  # 确保有非零点存在
                smoothed_trajectory = self.sliding_window_smooth(non_zero_trajectory, 3)

                # 将平滑后的轨迹回填到非 (0, 0) 的位置
                sample_rotated[~zero_mask, pid, :2] = smoothed_trajectory

            # 保留原本为 (0, 0) 的点不变
            sample_rotated[zero_mask, pid, :2] = [0, 0]

        return sample_rotated

    @staticmethod
    def calculate_bicycle_movement(matrix):
        # 获取矩阵的形状
        frame_seq, max_bicycle_num, feat_num = matrix.shape

        # 创建一个三维矩阵，用于存储每个帧每个自行车的位移量
        displacement_matrix = np.zeros((frame_seq, max_bicycle_num, 7))

        # 创建一个二维矩阵，用于存储每个帧每个自行车的动作量
        feature_num = 3
        movement_matrix = np.zeros((frame_seq, max_bicycle_num, feature_num))

        # 遍历所有帧
        for i in range(frame_seq):
            # 获取当前帧的数据
            current_frame = matrix[i, :, :]

            # 遍历所有自行车
            for j in range(max_bicycle_num):
                # 如果当前自行车的数据不完整，则跳过
                if np.all(current_frame[j][:2] == 0):
                    continue

                # 获取当前自行车的 ID、位置和速度
                bike_id = int(current_frame[j][1])
                frame_id = int(current_frame[j][0])
                bike_pos = current_frame[j][2:4]
                bike_vel = current_frame[j][4:6]

                # 赋值
                displacement_matrix[i, j, :] = np.concatenate((bike_pos, bike_vel, [frame_id, bike_id, 0]))

                # 找到起点

                samebike_start_frame = np.where(displacement_matrix[:, :, -2] == bike_id)[0]
                if len(samebike_start_frame) == 0:
                    start_frame = j
                else:
                    start_frame = np.min(samebike_start_frame)


                    # 方法①：计算基于起点的位置和速度

                    start_pos = displacement_matrix[start_frame, j, :2]
                    start_vel = displacement_matrix[start_frame, j, 2:4]
                    prev_timestamp = int(displacement_matrix[start_frame, j, -3])
                    time_diff = displacement_matrix[i, j, -3] - prev_timestamp
                    base_pos = start_pos + start_vel * time_diff * 0.12  # 0.12是时间间隔  base_pos是恒速度运行至此


                    # 继续运行
                    # 计算位移量并添加到矩阵中
                    displacement = bike_pos - base_pos  #矢量
                    total_displacement = np.sqrt(displacement[0] ** 2 + displacement[1] ** 2)  #标量
                    displacement_matrix[i, j, -1] = total_displacement

                    movement_matrix[i, j, 0] = displacement[0]   #矢量动作
                    movement_matrix[i, j, 1] = displacement[1]
                    #movement_matrix[i, j, 2] = displacement_matrix[i, j, -1]  #总量动作
        #if np.any(movement_matrix == 0):
            #print('hhh')
        return movement_matrix

    @staticmethod
    def calculate_bicycle_movement_every(matrix):
        # 获取矩阵的形状
        frame_seq, max_bicycle_num, feat_num = matrix.shape

        # 创建一个三维矩阵，用于存储每个帧每个自行车的位移量
        displacement_matrix = np.zeros((frame_seq, max_bicycle_num, 7))

        # 创建一个二维矩阵，用于存储每个帧每个自行车的动作量
        feature_num = 3
        movement_matrix = np.zeros((frame_seq, max_bicycle_num, feature_num))

        # 遍历所有帧
        for i in range(frame_seq):
            # 获取当前帧的数据
            current_frame = matrix[i, :, :]

            # 遍历所有自行车
            for j in range(max_bicycle_num):
                # 如果当前自行车的数据不完整，则跳过
                if np.all(current_frame[j][:2] == 0):
                    continue

                # 获取当前自行车的 ID、位置和速度
                bike_id = int(current_frame[j][1])
                frame_id = int(current_frame[j][0])
                bike_pos = current_frame[j][2:4]
                bike_vel = current_frame[j][4:6]

                # 赋值
                displacement_matrix[i, j, :] = np.concatenate((bike_pos, bike_vel, [frame_id, bike_id, 0]))

                # 找到起点
                samebike_start_frame = np.where(displacement_matrix[:, :, -2] == bike_id)[0]
                if len(samebike_start_frame) == 0:
                    start_frame = j
                else:
                    start_frame = np.min(samebike_start_frame)

                # 计算上一个步长的位置和速度
                prev_pos = displacement_matrix[start_frame, j, :2]
                prev_vel = displacement_matrix[start_frame, j, 2:4]

                # 计算基于上一个步长的位置和速度
                prev_timestamp = int(displacement_matrix[start_frame, j, -3])
                time_diff = frame_id - prev_timestamp
                base_pos = prev_pos + prev_vel * time_diff * 0.12

                # 计算位移量并添加到矩阵中
                displacement = bike_pos - base_pos
                total_displacement = np.sqrt(displacement[0] ** 2 + displacement[1] ** 2)
                displacement_matrix[i, j, -1] = total_displacement

                movement_matrix[i, j, 0] = displacement[0]
                movement_matrix[i, j, 1] = displacement[1]
                movement_matrix[i, j, 2] = displacement_matrix[i, j, -1]

        return movement_matrix

    @staticmethod
    def calculate_bicycle_safety(matrix):
        # 获取矩阵的形状
        frame_seq, max_bicycle_num, feat_num = matrix.shape

        # 创建一个字典，用于判断自行车是否为首次出现
        bike_dict = {}

        # 创建一个三维矩阵，用于存储每个帧下各个自行车之间的安全距离变化情况
        feature_num = 3
        safety_matrix = np.zeros((frame_seq, max_bicycle_num, max_bicycle_num, feature_num))

        # 遍历所有帧
        for i in range(frame_seq):
            # 获取当前帧的各个自行车状态信息
            current_frame = matrix[i]

            # 遍历所有自行车
            for j in range(max_bicycle_num):
                # 如果当前自行车的数据不完整，则跳过
                if np.isnan(current_frame[j]).any():
                    continue

                # 获取当前自行车的 ID、位置和速度
                bike_id = int(current_frame[j][1])
                bike_pos = current_frame[j][2:4]
                bike_vel = current_frame[j][4:6]

                # 如果之前没有保存该自行车的状态，则将当前状态作为起点
                if bike_id not in bike_dict:
                    bike_dict[bike_id] = {'pos': bike_pos, 'vel': bike_vel, 'frame': i}
                else:
                    # 计算基于起点的位置和速度
                    start_pos = bike_dict[bike_id]['pos']
                    start_vel = bike_dict[bike_id]['vel']
                    prev_timestamp = int(bike_dict[bike_id]['frame'])
                    time_diff = i - prev_timestamp
                    base_pos = start_pos + start_vel * time_diff * 0.12  # 0.12是时间间隔

                    # 计算同一帧下两个自行车之间欧式距离的减少量
                    # 标量
                    distance_before = np.linalg.norm(start_pos - current_frame[:, 2:4], axis=1)
                    distance_after = np.linalg.norm(bike_pos - current_frame[:, 2:4], axis=1)
                    safety_decrease = distance_before - distance_after

                    # 矢量
                    distance_change = (start_pos - current_frame[:, 2:4]) - (bike_pos - current_frame[:, 2:4])

                    # 将当前自行车与其他自行车的安全减少量保存到安全矩阵中
                    for k in range(max_bicycle_num):
                        if k == j or np.isnan(current_frame[k]).any():
                            continue
                        #safety_matrix[i, j, k, 0:2] = distance_change[k]  #这里是赋值3个特征值
                        safety_matrix[i, j, k, 2] = safety_decrease[k]
                        safety_matrix[i, k, j] = safety_decrease[k]  # 对称赋值

                    # 更新字典状态
                    bike_dict[bike_id]['pos'] = bike_pos
                    bike_dict[bike_id]['vel'] = bike_vel
                    bike_dict[bike_id]['frame'] = i

        return safety_matrix

    @staticmethod
    def calculate_bicycle_safety_every(matrix):
        frame_seq, max_bicycle_num, feat_num = matrix.shape

        bike_dict = {}

        feature_num = 3
        safety_matrix = np.zeros((frame_seq, max_bicycle_num, max_bicycle_num, feature_num))

        for i in range(frame_seq):
            current_frame = matrix[i]

            for j in range(max_bicycle_num):
                if np.isnan(current_frame[j]).any():
                    continue

                bike_id = int(current_frame[j][1])
                bike_pos = current_frame[j][2:4]
                bike_vel = current_frame[j][4:6]

                if bike_id not in bike_dict:
                    bike_dict[bike_id] = {'pos': bike_pos, 'vel': bike_vel, 'frame': i}
                else:
                    start_pos = bike_dict[bike_id]['pos']
                    start_vel = bike_dict[bike_id]['vel']
                    prev_timestamp = int(bike_dict[bike_id]['frame'])
                    time_diff = i - prev_timestamp
                    base_pos = start_pos + start_vel * time_diff * 0.12

                    distance_before = np.linalg.norm(start_pos - current_frame[:, 2:4], axis=1)
                    distance_after = np.linalg.norm(bike_pos - current_frame[:, 2:4], axis=1)
                    safety_decrease = distance_before - distance_after

                    distance_change = (start_pos - current_frame[:, 2:4]) - (bike_pos - current_frame[:, 2:4])

                    for k in range(max_bicycle_num):
                        if k == j or np.isnan(current_frame[k]).any():
                            continue
                        safety_matrix[i, j, k, 0:2] = distance_change[k]
                        safety_matrix[i, j, k, 2] = safety_decrease[k]

                    bike_dict[bike_id]['pos'] = bike_pos
                    bike_dict[bike_id]['vel'] = bike_vel
                    bike_dict[bike_id]['frame'] = i

        return safety_matrix

    @staticmethod
    def sort_by_distance(data):
        """
        按照两两之间距离的远近对交通参与者进行排序
        :param data: 二维数组，第一列为帧ID，第二列为位置x轴，第三列为位置y轴，第12列为个体ID
        :return: 排序后的数据，二维数组格式
        """
        # 计算每个参与者与其他参与者之间的欧氏距离
        num_participant = len(np.unique(data[:, 11]))  # 参与者的数量
        dist_matrix = np.zeros((num_participant, num_participant))  # 初始化距离矩阵为0
        for i in range(num_participant):
            for j in range(i + 1, num_participant):
                dist = np.sqrt((data[i, 1]-data[j, 1]) ** 2
                               + (data[i, 2]-data[j, 2]) ** 2)
                dist_matrix[i][j] = dist  # 距离矩阵为对称矩阵

        # 将距离矩阵进行unique以后进行升序排列
        unique_dist_matrix = np.unique(dist_matrix[dist_matrix > 0])
        sorted_dist_matrix = np.sort(unique_dist_matrix)

        # 按照距离最小的原则（除了0）将不同的id进行排序
        order_index = []
        used_id = set()
        for dist in sorted_dist_matrix:
            rows, cols = np.where(dist_matrix == dist)
            for r, c in zip(rows, cols):
                if r not in used_id and c not in used_id:
                    order_index.append(r)
                    used_id.add(r)
                    order_index.append(c)
                    used_id.add(c)

        # 将未选过的id保存下来
        remaining_index = np.array(list(set(range(num_participant)) - used_id))
        order_index = np.concatenate((order_index, remaining_index)).astype(int)
        sorted_ids = data[:, 12][order_index]

        return sorted_ids

    @staticmethod
    def features_select(sample):
        # 这里是取用一些特征值，调整模型请从这里修改
        sample_1 = sample[:, 0:10]  # 取前10列特征
        sample_1 = np.hstack((sample_1, sample[:, 13].reshape(-1, 1)))  # 删除一部分特征
        #print(np.shape(sample[:, 13]))
        sample_1 = np.hstack((sample_1, sample[:, 14:21]))
        sample_1 = np.hstack((sample_1, sample[:, 24].reshape(-1, 1)))

        return sample_1

    @staticmethod
    def calculate_position_difference(A, T):
        # 获取起点位置和速度
        x0, y0, vx0, vy0 = A[0, 0], A[0, 1], A[0, 2], A[0, 3]

        # 计算每个时间戳上的位置差异
        position_difference = np.zeros_like(A[:, :2])
        position_difference[0] = np.array([0, 0])  # 起点位置差异为0

        for i in range(1, len(A)):
            dt = T * i  # 当前时间戳
            dx = vx0 * dt  # x方向的位移
            dy = vy0 * dt  # y方向的位移
            position_difference[i, :] = A[i, :2] - (np.array([dx, dy])+ A[0, :2])

        return position_difference
