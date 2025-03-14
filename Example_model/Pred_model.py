import numpy as np  #基础包，可对N维数组进行操作
import pandas as pd   ###强大的分析结构化数据的工具集
import os  #获取当前目录，python 的工作目
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
import tensorflow as tf
from scipy.special import comb
import math
from tensorflow.keras.layers import Input, Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense

# 自定义moss函数
class MotionConstraintLoss:
    def __init__(self, args, n_output, n_feature):  #这里n_feature是默认
        self.args = args
        self.n_output = n_output
        self.n_feature = 2
        #self.max_boundary = np.array(max_boundary)
        #self.min_boundary = np.array(min_boundary)

    @staticmethod
    def anti_norm(nor_data, max_boundary, min_boundary):
        # 反归一化
        # 此处显示详细说明
        max_boundary = tf.convert_to_tensor(max_boundary.flatten(), dtype=tf.float32)
        min_boundary = tf.convert_to_tensor(min_boundary.flatten(), dtype=tf.float32)

        orin_data = tf.cast(nor_data, dtype=tf.float32)  # 创建 nor_data 的副本并更改数据类型

        if not tf.reduce_all(tf.equal(nor_data, 0)):
            orin_data = ((max_boundary - min_boundary) * nor_data - 0.001) + min_boundary

        return orin_data

    @staticmethod
    def compute_curvature(points):
        x = points[:, 0]
        y = points[:, 1]

        curvature = []
        for i in range(1, len(points) - 1):
            dx = x[i + 1] - x[i - 1]
            dy = y[i + 1] - y[i - 1]
            d2x = x[i + 1] - 2 * x[i] + x[i - 1]
            d2y = y[i + 1] - 2 * y[i] + y[i - 1]

            # 添加一个小的数，避免分母过小导致计算结果为 NaN
            denominator = (dx ** 2 + dy ** 2) ** 1.5 + 0.00001
            curv = abs(dx * d2y - dy * d2x) / denominator
            curvature.append(curv)

        max_cur = tf.reduce_max(tf.abs(curvature))

        return max_cur

    @staticmethod
    def compute_trajectory_angle(points):
        angles = []

        for i in range(len(points) - 2):
            dx1 = -(points[i + 1][0] - points[i][0])
            dy1 = points[i + 1][1] - points[i][1]
            dx2 = -(points[i + 2][0] - points[i + 1][0])
            dy2 = points[i + 2][1] - points[i + 1][1]

            if dx1 == 0:
                angle1 = 0.0
            else:
                angle1 = tf.math.atan2(dy1, dx1)

            if dx2 == 0:
                angle2 = 0.0
            else:
                angle2 = tf.math.atan2(dy2, dx2)

            angle_diff = angle2 - angle1
            angles.append(angle_diff)

        max_angle_rad = tf.reduce_max(tf.abs(angles))
        var = tf.math.reduce_variance(angles)

        max_angle_deg = max_angle_rad * 180 / tf.constant(math.pi)
        var = tf.convert_to_tensor(var)

        return max_angle_deg, var

    @staticmethod
    def compute_acceleration_and_jerk(points, T):
        x = points[:, 0]
        y = points[:, 1]

        dx = (x[2:] - x[:-2]) / (2 * T)
        dy = (y[2:] - y[:-2]) / (2 * T)

        acceleration = tf.sqrt(dx ** 2 + dy ** 2)

        d2x = (x[2:] - 2 * x[1:-1] + x[:-2]) / (T ** 2)
        d2y = (y[2:] - 2 * y[1:-1] + y[:-2]) / (T ** 2)

        jerk = tf.sqrt(d2x ** 2 + d2y ** 2)

        return acceleration, jerk

    def my_loss(self, gt, pre):
        n_samples = tf.shape(pre)[0]
        gt = tf.reshape(gt, (n_samples, self.n_output, self.args.max_bicycle_num, self.n_feature))
        pre = tf.reshape(pre, (n_samples, self.n_output, self.args.max_bicycle_num, self.n_feature))
        loss_all = 0.0
        for i in range(n_samples):
            gt_frame = gt[i, :, :, :]
            pre_frame = pre[i, :, :, :]
            for j in range(self.args.max_bicycle_num):
                #gt_current = self.anti_norm(gt_frame[:, j, :], self.max_boundary, self.min_boundary)
                #pre_current = self.anti_norm(pre_frame[:, j, :], self.max_boundary, self.min_boundary)

                # 不逆归一化
                gt_current = gt_frame[:, j, :]
                pre_current = pre_frame[:, j, :]

                #acc_pre, jerk_pre = self.compute_acceleration_and_jerk(pre_current[:, :2], 0.12 * self.args.Decision_interval)
                angle, var = self.compute_trajectory_angle(pre_current[:, :2])

                #acc_penalty = tf.maximum((tf.reduce_mean(acc_pre) / self.args.threshold_acc), 1.0)
                #jerk_penalty = tf.maximum((tf.reduce_mean(jerk_pre) / self.args.threshold_jerk), 1.0)
                angle_penalty = tf.maximum((angle / self.args.threshold_angle), 1.0)
                var_penalty = tf.maximum((var / self.args.threshold_angle_var), 1.0)

                penalty = tf.reduce_mean([angle_penalty, var_penalty]) * self.args.penalty_coefficient

                distance = tf.abs(gt_frame[:, j, :] - pre_frame[:, j, :])
                mde = tf.reduce_mean(distance)

                loss = mde + (mde * penalty)

                loss_all += tf.reduce_sum(loss)
        mul = tf.cast(tf.multiply(n_samples, tf.convert_to_tensor(self.args.max_bicycle_num)), dtype=tf.float32)

        loss_out = tf.divide(loss_all, mul)
        return loss_out

    def mean_absolute_error(self, y_true, y_pred):
        n_samples = tf.shape(y_true)[0]
        y_true = tf.reshape(y_true, (n_samples, self.n_output, self.args.max_bicycle_num, self.n_feature))
        y_pred = tf.reshape(y_pred, (n_samples, self.n_output, self.args.max_bicycle_num, self.n_feature))

        # 计算每个样本的绝对误差
        absolute_error = tf.abs(y_true - y_pred)

        # 计算每个样本的平均绝对误差
        mae = tf.reduce_mean(absolute_error, axis=-1)

        # 返回批次中所有样本的平均绝对误差
        return tf.reduce_mean(mae)


def bezier_curve(points, num_samples):
    num_points = len(points)
    t = np.linspace(0, 1, num_samples)

    curve = np.zeros((num_samples, 2))

    for i in range(num_samples):
        for j in range(num_points):
            curve[i] += comb(num_points-1, j) * ((1-t[i])**(num_points-1-j)) * (t[i]**j) * points[j]

    return curve


def sparsemax(logits):
    """稀疏化注意力"""
    z = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
    softmax = tf.exp(z)
    normalization = tf.reduce_sum(softmax, axis=-1, keepdims=True)
    return tf.divide(softmax, normalization)


def weighted_average(logits):
    """加权平均"""
    weights = tf.nn.softmax(logits, axis=-1)
    weight_sum = tf.reduce_sum(weights, axis=-1, keepdims=True)
    return tf.divide(weights, weight_sum)


# 时间维度和个体维度
def dynamic_balance_cdp(x, efficiency, safety, args, n_input, n_feature, n_feature_all, n_output, n_feat_output):
    # 轨迹编码数据通道
    #x = tf.keras.layers.Dense(n_feature)(x)
    """
    efficiency = Reshape((n_input, args.max_bicycle_num, args.feature_num))(efficiency_input)
    safety = Reshape((n_input, args.max_bicycle_num, args.max_bicycle_num, args.feature_num))(safety_input)
    """
    # 效率通道处理
    #efficiency = tf.reshape(efficiency, (tf.shape(efficiency)[0], n_input*args.feature_num))
    efficiency = tf.keras.layers.Flatten()(efficiency)
    efficiency = tf.keras.layers.Dense(n_feature*n_output)(efficiency)  # 映射到与轨迹编码数据相同的维度
    #efficiency = tf.reshape(efficiency, (tf.shape(safety)[0], n_output, (n_feature*n_input)))
    #efficiency = tf.keras.layers.Dense(n_feature)(efficiency)
    efficiency_att = tf.nn.softmax(efficiency, axis=-1)   # 使用稀疏化注意力  #tf.nn.softmax(efficiency, axis=-1)  # 使用 softmax 激活函数得到注意力权重

    # 安全通道处理
    safety = tf.reshape(safety, (tf.shape(safety)[0], n_input*args.feature_num*args.max_bicycle_num))  #
    safety = tf.keras.layers.Flatten()(safety)
    safety = tf.keras.layers.Dense(n_feature*n_output)(safety)  # 映射到与轨迹编码数据相同的维度
    #safety = tf.reshape(safety, (tf.shape(safety)[0], n_output, (n_feature*n_input)))
    #safety = tf.keras.layers.Dense(n_feature)(safety)
    safety_att = tf.nn.softmax(safety, axis=-1)  # sparsemax 使用加权平均  #tf.nn.softmax(safety, axis=-1)  # 使用 softmax 激活函数得到注意力权重

    # 将注意力权重应用到轨迹编码数据上
    x_efficiency_time = tf.multiply(x, efficiency_att)
    x_safety_time = tf.multiply(x, safety_att)

    # 获取每个骑行者对应的注意力权重
    attention_weights_time_efficiency = tf.reduce_sum(x_efficiency_time, axis=-1)
    attention_weights_time_safety = tf.reduce_sum(x_safety_time, axis=-1)

    #"""
    # 构建骑行者之间的安全和效率平衡关系
    x_efficiency_time = tf.multiply(x_efficiency_time, attention_weights_time_efficiency[:, tf.newaxis])
    x_safety_time = tf.multiply(x_safety_time,  attention_weights_time_safety[:, tf.newaxis])
    
    #标红的这个原来是1-效率
    # 在新的轴上扩展维度
    x_efficiency_time_expanded = tf.expand_dims(x_efficiency_time, axis=-1)
    x_safety_time_expanded = tf.expand_dims(x_safety_time, axis=-1)
    #"""
    # 在新的轴上进行拼接
    output_cdp = tf.concat(
        [x_efficiency_time, x_safety_time], axis=-1)

    # 将两部分合并，得到最终的输出

    #output_cdp = tf.reshape(output_cdp, (tf.shape(output_cdp)[0], n_output*n_feature*2))

    return output_cdp

train_loss_history = []
val_loss_history = []

# 在每个epoch结束后记录损失值
def append_loss(history):
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)


def custom_loss(occupancy_value):
    def loss_function(y_true, y_pred):
        mask = tf.math.not_equal(y_true, occupancy_value)
        masked_squared_error = tf.square(tf.boolean_mask(y_true - y_pred, mask))
        return tf.reduce_mean(masked_squared_error)

    return loss_function


def compute_curvature(points):
    dx = tf.gradients(points[:, 0], [points])[0]
    dy = tf.gradients(points[:, 1], [points])[0]

    dx_dt = tf.gradients(points[:, 0], [points])[0]
    dy_dt = tf.gradients(points[:, 1], [points])[0]

    curvature = tf.abs(dx_dt * dy - dx * dy_dt) / (dx ** 2 + dy ** 2) ** 1.5

    return curvature


def motion_constraint_loss(gt, pre, threshold, penalty_coefficient, args, n_output, n_feature):
    gt = tf.reshape(gt, (tf.shape(gt)[0], n_output, args.max_bicycle_num, n_feature))
    pre = tf.reshape(pre, (tf.shape(pre)[0], n_output, args.max_bicycle_num, n_feature))
    loss_all = []
    for i in range(tf.shape(pre)[0]):
        # 计算曲率
        curvature_pre = compute_curvature(pre[:, :2])
        # 计算超过阈值的平均曲率与阈值的比例，并乘以惩罚因子
        mean_curvature = tf.reduce_mean(curvature_pre)
        curvature_penalty = tf.maximum((mean_curvature - threshold), 1.0) * penalty_coefficient

        # 计算平均绝对误差
        distance = tf.abs(gt - pre)
        penalty_distance = tf.reduce_mean(distance)

        loss = curvature_penalty * penalty_distance
        loss_all.append(loss)

    return tf.reduce_sum(loss_all)


"""

def custom_loss(occupancy_value):
    def loss_function(y_true, y_pred):
        mask = tf.math.not_equal(y_true, occupancy_value)
        masked_absolute_error = tf.abs(tf.boolean_mask(y_true - y_pred, mask))
        return tf.reduce_mean(masked_absolute_error)

    return loss_function
"""


def masked_dense(units, activation='relu'):
    # 创建全连接层
    dense_layer = Dense(units, activation=activation)

    def apply_mask(inputs):
        # 拆分输入为数据和掩码
        data, mask = inputs

        # 将掩码与数据进行元素乘法
        masked_data = tf.multiply(data, mask)

        # 应用全连接层到掩码后的数据
        output = dense_layer(masked_data)

        return output

    return apply_mask


# 在每个epoch结束后调用append_loss()函数并绘制损失曲线
def plot_loss(train_loss_history, val_loss_history):
    plt.figure()
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def individual_model(args, n_input, n_feature, n_feat_output,n_feature_all, main_input, efficiency_input, safety_input,
                     n_output):
    # 该函数用来为一个个体建模，返回数据流
    #x = tf.reshape(main_input, (tf.shape(main_input)[0], n_input, n_feature, 1))
    x = tf.keras.layers.Reshape((n_input, n_feature,1))(main_input)
    x = Conv2D(128, (2, 1), padding='VALID')(x)  #128
    x = Activation('relu')(x)  #relu
    x = MaxPooling2D(pool_size=(2, 1), strides=None, padding='VALID', data_format=None)(x)
    x = TimeDistributed(Flatten())(x)

    masking_layer = Masking(mask_value=args.occupancy_value)  # 指定掩码值为0.0
    masked = masking_layer(x)
    x = LSTM(128, dropout=0.01, recurrent_dropout=0.01, activation='relu', return_sequences=True)(masked) #128
    x = tf.keras.layers.Flatten()(x)
    x = Dense(n_output * n_feature, activation='relu')(x)

    # 平衡机制
    output_cdp = dynamic_balance_cdp(x, efficiency_input, safety_input, args, n_input, n_feature, n_feature_all,
                                     n_output, n_feat_output)

    return x, output_cdp


def motion_prediction_train(args, data_set_dir1, data_set_dir2,data_set_dir3):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 读取数据1
    path_x_train1 = os.path.join(data_set_dir1, "x_train.txt")
    path_y_train1 = os.path.join(data_set_dir1, "y_train.txt")
    path_x_ver1 = os.path.join(data_set_dir1, "x_val.txt")
    path_y_ver1 = os.path.join(data_set_dir1, "y_val.txt")

    x_train = np.array(pd.read_csv(path_x_train1, header=None))
    y_train = np.array(pd.read_csv(path_y_train1, header=None))
    x_val = np.array(pd.read_csv(path_x_ver1, header=None))
    y_val = np.array(pd.read_csv(path_y_ver1, header=None))

    # 读取数据2
    path_x_train2 = os.path.join(data_set_dir2, "x_train.txt")
    path_y_train2 = os.path.join(data_set_dir2, "y_train.txt")
    path_x_ver2 = os.path.join(data_set_dir2, "x_val.txt")
    path_y_ver2 = os.path.join(data_set_dir2, "y_val.txt")

    x_train_efficiency = np.array(pd.read_csv(path_x_train2, header=None))
    y_train_efficiency = np.array(pd.read_csv(path_y_train2, header=None))
    x_val_efficiency = np.array(pd.read_csv(path_x_ver2, header=None))
    y_val_efficiency = np.array(pd.read_csv(path_y_ver2, header=None))

    # 读取数据3
    path_x_train3 = os.path.join(data_set_dir3, "x_train.txt")
    path_y_train3 = os.path.join(data_set_dir3, "y_train.txt")
    path_x_ver3 = os.path.join(data_set_dir3, "x_val.txt")
    path_y_ver3 = os.path.join(data_set_dir3, "y_val.txt")

    x_train_safety = np.array(pd.read_csv(path_x_train3, header=None))
    y_train_safety = np.array(pd.read_csv(path_y_train3, header=None))
    x_val_safety = np.array(pd.read_csv(path_x_ver3, header=None))
    y_val_safety = np.array(pd.read_csv(path_y_ver3, header=None))

    max_boundary = pd.read_csv(os.path.join(data_set_dir1, 'max_boundary.txt'), header=None)
    min_boundary = pd.read_csv(os.path.join(data_set_dir1, 'min_boundary.txt'), header=None)

    # 设置模型保存地址
    save_path = os.path.join(data_set_dir1, "MotionPrediction_model.h5")
    if not os.path.exists(os.path.dirname(data_set_dir1)):
        os.makedirs(os.path.dirname(data_set_dir1))

    # 根据数据计算参数
    n_input = int(args.Obs_length / args.Decision_interval)  #4 #输入轨迹点的个数   n_input*n_feature
    n_output = int(args.Pred_length / args.Decision_interval)  #6
    find_dim = np.array(x_train)
    find_dim = find_dim[0, :].reshape(n_input, -1)
    n_feature = int(find_dim.shape[1] / args.max_bicycle_num) # 读取输入特征数
    n_feature_all = int(n_feature * n_input * args.max_bicycle_num)  #起终点全部的特征数
    n_feart_efficiency = int(n_input * args.max_bicycle_num * args.feature_num)
    n_feart_safety = int(n_input * args.max_bicycle_num * args.max_bicycle_num * args.feature_num)
    n_feat_output = int(n_output * n_feature * args.max_bicycle_num)

    #optimizer = Adam(learning_rate=args.learning_rate)  # 使用Adam优化器，指定学习率

    # 打乱样本
    # 对训练集
    num_rows_train = x_train.shape[0]
    random_order_train = np.random.permutation(num_rows_train)

    x_train = x_train[random_order_train]
    y_train = y_train[random_order_train]
    x_train_efficiency = x_train_efficiency[random_order_train]
    x_train_safety = x_train_safety[random_order_train]

    # 对验证集
    num_rows_val = x_val.shape[0]
    random_order_val = np.random.permutation(num_rows_val)

    x_val = x_val[random_order_val]
    y_val = y_val[random_order_val]
    x_val_efficiency = x_val_efficiency[random_order_val]
    x_val_safety = x_val_safety[random_order_val]


    # 删减y,只映射至x和y
    y_train = y_train.reshape(-1, n_output, args.max_bicycle_num, n_feature)
    y_train = y_train[:, :, :, :2]
    y_train = y_train.reshape(x_train.shape[0], -1)

    y_val = y_val.reshape(-1, n_output, args.max_bicycle_num, n_feature)
    y_val = y_val[:, :, :, :2]
    y_val = y_val.reshape(x_val.shape[0], -1)


    #### 模型    定义###########################
    #mask_train = (x_train != args.occupancy_value).astype(float)
    #mask_efficiency = (x_train_efficiency != args.occupancy_value).astype(float)
    #mask_safety = (x_train_safety != args.occupancy_value).astype(float)

    main_input = Input(shape=(n_feature_all,))
    efficiency_input = Input(shape=(n_feart_efficiency,))
    safety_input = Input(shape=(n_feart_safety,))

    # 输入三个通道
    x = Reshape((n_input, args.max_bicycle_num, n_feature))(main_input)
    efficiency = Reshape((n_input, args.max_bicycle_num, args.feature_num))(efficiency_input)
    safety = Reshape((n_input, args.max_bicycle_num, args.max_bicycle_num, args.feature_num))(safety_input)

    # 各自建模
    all_output = []
    all_cdp = []
    for i in range(args.max_bicycle_num):   #为每个自行车建立轨迹预测模型
        individual_output, output_cdp = individual_model(args, n_input, n_feature, n_feat_output, n_feature_all,
                                             x[:, :, i, :], efficiency[:, :, i, :],
                                             safety[:, :, i, :], n_output)

        output_cdp = Dense(n_feature * n_output)(output_cdp)
        all_output.append(individual_output)
        all_cdp.append(output_cdp)

    # 方法0
    #
    traj_out = []
    output_cdp_combined = tf.keras.layers.concatenate(all_cdp)
    output_cdp_combined = Dense(args.max_bicycle_num * n_feature * n_output)(output_cdp_combined)  #我的数据在运行
    #output_cdp_combined = Dense(n_feature * n_output)(output_cdp_combined)
    output_cdp_combined = tf.keras.layers.Reshape((args.max_bicycle_num, n_output*n_feature))(output_cdp_combined)

    for i in range(args.max_bicycle_num):
        #output_combined = Dense(n_output * 2)(tf.multiply(all_output[i], output_cdp_combined))
        output_combined = Dense(n_output * 2)(tf.multiply(all_output[i], output_cdp_combined[:, i, :]))
        #output_combined = Dense(n_output*2)(output_cdp_combined[:,i,:])

        #n_feature * n_output
        traj_out.append(output_combined)


    # 输出个体预测
    output = tf.keras.layers.concatenate(traj_out)

    model = Model(inputs=[main_input, efficiency_input, safety_input], outputs=output)
    Loss = MotionConstraintLoss(args, n_output, n_feature)
    loss = Loss.my_loss  #my_loss
    #model.compile(loss=custom_loss(occupancy_value=args.occupancy_value), optimizer='Adam')
    #model.compile(loss=loss, optimizer='Adam')
    model.compile(loss='mean_absolute_error', optimizer='Adam')   #optimizer    model.compile(loss=custom_loss(occupancy_value=args.occupancy_value), optimizer=optimizer)
    model.summary()

    # 训练模型
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    history2 = model.fit([x_train, x_train_efficiency, x_train_safety], y_train, batch_size=20, epochs=200,
                         verbose=1, validation_data=([x_val, x_val_efficiency, x_val_safety], y_val),
                         callbacks=[early_stopping])

    print('history', history2)

    # 在每个epoch结束后记录损失值
    #append_loss(history2)
    #plot_loss(train_loss_history, val_loss_history)
    model.save(save_path)  #保存模型 ，保存格式是h5
    with open(os.path.join(data_set_dir1, "MotionPrediction_model.csv"), 'w') as ff:  #以写入的方式打开file文件，并将文件存储到变量中
        ff.write(str(history2.history))  ##write表示将字符串str写入到文件，str()将对象转化为适合人阅读的格式
    print('save model success')
    print('done!')

