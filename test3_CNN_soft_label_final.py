from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_curve, auc
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib.patches as patches
import time
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, Reshape
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, concatenate, Multiply
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dense, Input
from tensorflow.keras.models import Model
from tqdm import tqdm
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Define experiment variables
sample_num = 119
channel_num = 300
way = 'CWT'  # CWT ISD
algorithm = 'SAMME.R'  # 'SAMME.R' or 'SAMME'
input_path = r"C:\Users\final\Desktop\岩性识别\代码\code\code"
output_path = rf"C:\Users\final\Desktop\岩性识别\代码\code\code\{way}"
plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})

# Set random seed to ensure result reproducibility
seed = 50
np.random.seed(seed)
tensorflow.random.set_seed(seed)

# Load image data
def load_images(image_folder):
    images = []
    bmp_files = [f for f in os.listdir(image_folder) if f.endswith('.bmp')]
    sorted_files = sorted(bmp_files)

    for filename in sorted_files:
        print(f"Reading file: {filename}")
        image_path = os.path.join(image_folder, filename)
        img = load_img(image_path)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
    return np.array(images)

# Modified load_images function with batch processing
def load_images_in_batches(image_folder, batch_size):
    bmp_files = [f for f in os.listdir(image_folder) if f.endswith('.bmp')]
    sorted_files = sorted(bmp_files)
    for i in range(0, len(sorted_files), batch_size):
        batch_files = sorted_files[i:i + batch_size]
        images = []
        for filename in batch_files:
            print(f"Reading file: {filename}")
            image_path = os.path.join(image_folder, filename)
            img = load_img(image_path)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
        yield np.array(images), batch_files

# Read label data
def load_labels(labels_file):
    labels_df = pd.read_excel(labels_file, header=0, usecols=[0, 1, 2])  # 读取前三列为类别概率
    labels = labels_df.values  # 返回为概率分布
    return labels


class PrintLearningRate(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = K.get_value(self.model.optimizer.lr)
        print(f"\nEpoch {epoch + 1}: Current learning rate = {lr}")


# 逐通道计算余弦相似度并求平均
def calculate_rgb_cosine_similarity(img1, img2):
    # 对每个通道分别计算余弦相似度
    red_similarity = cosine_similarity(img1[:, :, 0].flatten().reshape(1, -1),
                                       img2[:, :, 0].flatten().reshape(1, -1))[0, 0]
    green_similarity = cosine_similarity(img1[:, :, 1].flatten().reshape(1, -1),
                                         img2[:, :, 1].flatten().reshape(1, -1))[0, 0]
    blue_similarity = cosine_similarity(img1[:, :, 2].flatten().reshape(1, -1),
                                        img2[:, :, 2].flatten().reshape(1, -1))[0, 0]
    # 通过加权平均得到总相似度
    total_similarity = (red_similarity + green_similarity + blue_similarity) / 3
    return total_similarity

# 修改的过滤函数，逐通道计算余弦相似度
def filter_similar_images_rgb(X_source, X_target, y, threshold=0.8):
    retained_indices = []
    for i in range(len(X_source)):
        # 计算两张彩色图片的相似性
        similarity = calculate_rgb_cosine_similarity(X_source[i], X_target[i])
        if similarity >= threshold:
            retained_indices.append(i)

    # 根据保留的索引过滤 X_source 和 y
    X_filtered = X_source[retained_indices]
    y_filtered = y[retained_indices]
    return X_filtered, y_filtered, retained_indices

def baseline_model(img_height, img_width, img_channels, n_classes):
    model = Sequential()
    model.add(Conv2D(128, (3, 3), padding="same", input_shape=(img_height, img_width, img_channels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    adam_optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['categorical_crossentropy'])

    # reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, min_lr=0.0001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='min')

    return model, early_stopping
#

def Ada_baseline_model(img_height, img_width, img_channels, n_classes):
    model = Sequential()
    model.add(Conv2D(128, (3, 3), padding="same", input_shape=(img_height, img_width, img_channels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    adam_optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['categorical_crossentropy'])

    # reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, min_lr=0.0001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='min')

    return model, [early_stopping]
#

def Ada_baseline_model_False(img_height, img_width, img_channels, n_classes):
    model = Sequential()
    model.add(Conv2D(128, (3, 3), padding="same", input_shape=(img_height, img_width, img_channels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    adam_optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['categorical_crossentropy'])

    # reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, min_lr=0.0001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='min')

    return model, [early_stopping]
#, reduce_lr
def plot_and_save_confusion_matrix(y_true, y_pred, classes, output_path, file_name="confusion_matrix.tif"):
    cm = confusion_matrix(y_true, y_pred)
    class_labels = [classes[i] for i in sorted(classes)]

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.savefig(os.path.join(output_path, file_name), bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()


def save_operating_data(metrics):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_path = os.path.join(output_path, f"{metrics['algorithm']}_Operating_data.txt")

    # 通过将内容格式化为字符串来保存数据
    content = f"""
    Algorithm: {metrics['algorithm']}
    """

    # 根据是否是回归任务来保存不同的指标
    if 'mse_train' in metrics:  # 说明是回归任务
        content += f"""
        {metrics['algorithm']}_train_MSE: {metrics['mse_train']}
        {metrics['algorithm']}_test_MSE: {metrics['mse_test']}
        {metrics['algorithm']}_train_MAE: {metrics['mae_train']}
        {metrics['algorithm']}_test_MAE: {metrics['mae_test']}
        {metrics['algorithm']}_train_RMSE: {metrics['rmse_train']}
        {metrics['algorithm']}_test_RMSE: {metrics['rmse_test']}
        {metrics['algorithm']}_train_R2: {metrics['r2_train']}
        {metrics['algorithm']}_test_R2: {metrics['r2_test']}
        {metrics['algorithm']}_train_cosine_similarity: {metrics['cosine_similarity_train']}
        {metrics['algorithm']}_test_cosine_similarity: {metrics['cosine_similarity_test']}
        """
    else:  # 否则是分类任务
        content += f"""
        {metrics['algorithm']}_train_accuracy_: {metrics['accuracy_train']}
        {metrics['algorithm']}_test_accuracy_: {metrics['accuracy_test']}
        {metrics['algorithm']}_train_Precision: {metrics['precision_train']}
        {metrics['algorithm']}_test_Precision: {metrics['precision_test']}
        {metrics['algorithm']}_train_Recall: {metrics['recall_train']}
        {metrics['algorithm']}_test_Recall: {metrics['recall_test']}
        {metrics['algorithm']}_train_F1-score: {metrics['f1_train']}
        {metrics['algorithm']}_test_F1-score: {metrics['f1_test']}
        {metrics['algorithm']}_train_cosine_similarity: {metrics['cosine_similarity_train']}
        {metrics['algorithm']}_test_cosine_similarity: {metrics['cosine_similarity_test']}
        """

    # 写入文件
    with open(file_path, "w") as file:
        file.write(content)

    print(f"Operating data saved to: {file_path}")


# 计算标准指标
def calculate_standard_metrics(y_true, y_pred, y_true_classes, y_pred_classes):
    """
    计算准确率、精确率、召回率和 F1 分数
    """
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes, average='macro')
    recall = recall_score(y_true_classes, y_pred_classes, average='macro')
    f1 = f1_score(y_true_classes, y_pred_classes, average='macro')

    return accuracy, precision, recall, f1

# 计算余弦相似度
def calculate_cosine_metrics(y_true, y_pred):
    """
    计算余弦相似度
    """
    cosine_sim = cosine_similarity(y_true, y_pred)
    # 提取对角线元素
    diagonal_similarities = np.diagonal(cosine_sim)
    # 计算对角线元素的平均值
    average_similarity = np.mean(diagonal_similarities)

    return average_similarity


# 计算回归指标：MSE、MAE、RMSE、MAPE 和 R²
def calculate_regression_metrics(y_true, y_pred, epsilon=1e-10):
    """
    计算回归模型的评估指标：MSE、MAE、RMSE、MAPE、R²
    epsilon: 平滑值，用于避免 MAPE 计算时的除以零错误
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return mse, mae, rmse, r2

def plot_and_save_combined_scatter_probabilities_with_stats(
        y_true,
        y_pred,
        output_path,
        file_name="combined_scatter_probabilities",
        classes=None,
        colors=None,
        alpha=0.8,
        beta=0.8,
        file_format="png",
):
    """
    绘制真实值和预测值的合并散点图，单独保存图例（仅保存一次），并统计样本分布情况。

    参数:
        y_true (numpy.ndarray): 真实值数组，形状为 (样本数, 类别数)。
        y_pred (numpy.ndarray): 预测值数组，形状为 (样本数, 类别数)。
        output_path (str): 保存图表和统计结果的输出路径。
        file_name (str): 保存图表的文件名，默认为 "combined_scatter_probabilities"。
        classes (dict): 类别索引到类别名称的映射字典，可选。
        colors (list): 每个类别的颜色列表，可选。
        alpha (float): 散点透明度，默认值为 0.8。
        beta (float): 控制虚线与实线的距离 (0-1)，数值越大越接近黑色实线。
        file_format (str): 保存图像的文件格式，默认为 "png"。
    """
    try:
        # 确保输入是 NumPy 数组
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # 检查输入形状是否一致
        assert y_true.shape == y_pred.shape, "y_true 和 y_pred 的形状必须一致！"

        # 确保 Beta 在有效范围内
        assert 0 <= beta <= 1, "Beta 参数必须在 0 和 1 之间！"

        # 最大偏移范围等于概率图的最大范围
        max_offset = 1.0
        # 计算虚线的偏移距离
        offset = (1 - beta) * max_offset

        num_classes = y_true.shape[1]
        figsize = (7, 7)  # 合并后的图像大小
        plt.figure(figsize=figsize)

        # 默认颜色列表
        if colors is None:
            default_colors = ["red", "yellow", "blue", "green", "gray"]
            colors = [default_colors[i % len(default_colors)] for i in range(num_classes)]

        # 创建统计结果
        stats = []
        total_samples = y_true.shape[0]
        stats.append(f"Total samples: {total_samples}\n")

        # 绘制散点图，区分类别
        for i in range(num_classes):
            color = colors[i]
            class_name = classes[i] if classes and i in classes else f"Class {i + 1}"
            plt.scatter(
                y_true[:, i],
                y_pred[:, i],
                alpha=alpha,
                edgecolor=color,
                facecolor="none",
                label=class_name,
                s=50,
                linewidths=1.5,
            )

            # 统计类别内的样本数量
            inside = np.sum((y_pred[:, i] <= y_true[:, i] + offset) & (y_pred[:, i] >= y_true[:, i] - offset))
            outside = total_samples - inside
            stats.append(f"{class_name}:\n  Inside tolerance: {inside}\n  Outside tolerance: {outside}\n")

        # 保存所有类别统计到一个文件中
        stats_file_path = os.path.join(output_path, f"{file_name}_stats.txt")
        os.makedirs(output_path, exist_ok=True)
        with open(stats_file_path, "w") as f:
            f.writelines("\n".join(stats))

        # 检查并保存图例（仅保存一次）
        legend_save_path = os.path.join(output_path, f"scatter_probabilities_legend.{file_format}")
        if not os.path.exists(legend_save_path):
            handles, labels = plt.gca().get_legend_handles_labels()
            legend_fig = plt.figure(figsize=(4, 2))
            legend_fig.legend(handles, labels, loc='center', fontsize=10, frameon=False)
            plt.axis('off')
            legend_fig.savefig(legend_save_path, bbox_inches='tight', pad_inches=0, dpi=600)
            plt.close(legend_fig)
            print(f"图例已成功保存至: {legend_save_path}")
        else:
            print(f"图例已存在，未重复保存: {legend_save_path}")

        # 添加虚线和实线到图
        plt.plot([0, 1], [0, 1], 'k-')  # 添加黑色实线 (y=x)
        plt.plot([0, 1], [offset, 1 + offset], 'k--')  # 添加虚线 (y=x+offset)
        plt.plot([0, 1], [-offset, 1 - offset], 'k--')  # 添加虚线 (y=x-offset)

        # 设置坐标轴范围和标签
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("True Probability")
        plt.ylabel("Predicted Probability")
        # plt.title("Combined Scatter Plot of True vs. Predicted Probabilities")

        # 保存合并的散点图
        scatter_save_path = os.path.join(output_path, f"{file_name}.{file_format}")
        plt.savefig(scatter_save_path, bbox_inches='tight', pad_inches=0, dpi=600)
        plt.close()
        print(f"图像已成功保存至: {scatter_save_path}")
        print(f"统计结果已成功保存至: {stats_file_path}")

    except Exception as e:
        print(f"绘制散点图时出错: {e}")
        print(f"输入参数: y_true.shape={y_true.shape}, y_pred.shape={y_pred.shape}, beta={beta}")

# 模型评估函数
# 修改后的 model_test 函数
def model_test(model, X_train, y_train, X_test, y_test, classes, algorithm, is_regression=False, colors=None):
    """
    执行模型测试，计算标准指标、回归指标及余弦相似度，保存混淆矩阵和操作数据
    """
    metrics = {}

    # 计算模型预测的值或概率分布
    if algorithm == 'CNN':
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    else:
        y_train_pred = model.predict_proba(X_train)
        y_test_pred = model.predict_proba(X_test)

    # 如果是回归任务
    if is_regression:
        # 计算回归指标
        mse_train, mae_train, rmse_train, r2_train = calculate_regression_metrics(y_train, y_train_pred)
        mse_test, mae_test, rmse_test, r2_test = calculate_regression_metrics(y_test, y_test_pred)

        # 保存真实值与预测值的交叉图，传入 classes 参数
        plot_and_save_combined_scatter_probabilities_with_stats(y_test, y_test_pred, output_path,
                                            file_name=f"{algorithm}_scatter_probabilities_test", classes=classes, colors=colors)
        plot_and_save_combined_scatter_probabilities_with_stats(y_train, y_train_pred, output_path,
                                            file_name=f"{algorithm}_scatter_probabilities_train", classes=classes, colors=colors)

        # 计算回归任务的余弦相似度
        cosine_train = calculate_cosine_metrics(y_train, y_train_pred)
        cosine_test = calculate_cosine_metrics(y_test, y_test_pred)

        # 将回归指标和余弦相似度添加到 metrics 中
        metrics['algorithm'] = algorithm
        metrics['mse_train'] = mse_train
        metrics['mse_test'] = mse_test
        metrics['mae_train'] = mae_train
        metrics['mae_test'] = mae_test
        metrics['rmse_train'] = rmse_train
        metrics['rmse_test'] = rmse_test
        metrics['r2_train'] = r2_train
        metrics['r2_test'] = r2_test
        metrics['cosine_similarity_train'] = cosine_train
        metrics['cosine_similarity_test'] = cosine_test

    else:
        # 如果是分类任务，继续计算标准分类指标
        y_train_pred_classes = np.argmax(y_train_pred, axis=1)
        y_test_pred_classes = np.argmax(y_test_pred, axis=1)

        y_train_classes = np.argmax(y_train, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        # 计算标准指标
        accuracy_train, precision_train, recall_train, f1_score_train = calculate_standard_metrics(y_train, y_train_pred, y_train_classes, y_train_pred_classes)
        accuracy_test, precision_test, recall_test, f1_score_test = calculate_standard_metrics(y_test, y_test_pred, y_test_classes, y_test_pred_classes)

        # 将标准指标添加到 metrics 中
        metrics['algorithm'] = algorithm
        metrics['accuracy_train'] = accuracy_train
        metrics['accuracy_test'] = accuracy_test
        metrics['precision_train'] = precision_train
        metrics['precision_test'] = precision_test
        metrics['recall_train'] = recall_train
        metrics['recall_test'] = recall_test
        metrics['f1_train'] = f1_score_train
        metrics['f1_test'] = f1_score_test

    # 绘制和保存混淆矩阵（仅分类任务）
    if not is_regression:
        plot_and_save_confusion_matrix(y_train_classes, y_train_pred_classes, classes, output_path,
                                       rf"{algorithm}_confusion_matrix_train.png")
        plot_and_save_confusion_matrix(y_test_classes, y_test_pred_classes, classes, output_path,
                                       rf"{algorithm}_confusion_matrix_test.png")

    # 保存操作数据
    save_operating_data(metrics)

    print("Metrics calculated and saved successfully.")

def plot_and_save_lithology(labels, output_path, colors, file_name="lithology.png"):
    lithology_image = np.zeros((len(labels), 1, 3))

    for i, label in enumerate(labels):
        if isinstance(label, np.ndarray):
            label = label[0]
        lithology_image[i, 0, :] = mcolors.to_rgb(colors[int(label)])

    fig, ax = plt.subplots(figsize=(2, 10))
    ax.imshow(lithology_image, aspect='auto')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, file_name), bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()

def plot_and_save_probability_lithology(probabilities, output_path, colors, file_name="probability_distribution.png"):
    fig, ax = plt.subplots(figsize=(2, 10))

    for index, prob in enumerate(probabilities):
        start = 0
        for class_index, class_prob in enumerate(prob):
            ax.add_patch(
                patches.Rectangle(
                    (start, index),
                    class_prob,
                    1,
                    facecolor=colors[class_index]
                )
            )
            start += class_prob

    ax.set_ylim(0, len(probabilities))
    ax.set_aspect('auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, file_name), bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()


def plot_and_save_probability_lithology_curve(probabilities, output_path, colors,
                                              file_name="probability_distribution_curve.png"):
    """
    绘制并保存概率岩性曲线图。
    横轴表示概率（0-1），纵轴表示样本深度（索引），每种类别的概率曲线用不同颜色表示。
    """
    fig, ax = plt.subplots(figsize=(2, 10))

    depth = np.arange(len(probabilities))
    class_labels = ["Gas sandstone", "Non-gas sandstone", "Mudstone"]  # 定义类别标签

    # 遍历每个类别的概率并绘制曲线
    for i, class_prob in enumerate(np.array(probabilities).T):
        ax.plot(class_prob, depth, label=class_labels[i], color=colors[i], linewidth=2.5)  # 使用指定颜色绘制每个类别的曲线

    # 设置横轴为概率范围 [0, 1]，纵轴为深度方向
    # ax.set_xlim(0, 1)
    # ax.set_ylim(len(probabilities), 0)
    # ax.set_xlabel("Probability")
    # ax.set_ylabel("Depth (Index)")
    # ax.legend(loc="upper right")
    ax.invert_yaxis()
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, file_name), bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()

def calculate_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    percentages = (counts / len(y)) * 100
    return unique, counts, percentages

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{p:.1f}%  ({v:d})'.format(p=pct, v=val)
    return my_format

def plot_and_save_distribution(unique, counts, percentages, classes, colors, title, output_path):
    # 将浮点数标签四舍五入为整数标签，并获取颜色
    pie_colors = [colors.get(int(round(i)), 'gray') for i in unique]  # 找不到时默认为'gray'
    labels = [classes.get(int(round(i)), 'Unknown') for i in unique]  # 找不到时标为'Unknown'

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=140, colors=pie_colors)
    plt.title(f'{title} Class Distribution')

    plt.subplot(1, 2, 2)
    plt.bar(labels, counts, color=pie_colors)
    plt.title(f'{title} Class Distribution')
    plt.ylabel('Counts')
    for i, count in enumerate(counts):
        plt.text(i, count + 0.2, str(count), ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{title}_Distribution.png'), bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()


sample_length = '25'
num = '2'
m = 'Full' # 'Interval'

image_folder_63 = rf'{input_path}\training and testing set{num}\{way}\{way}_063_{sample_length}'
image_folder_64 = rf'{input_path}\training and testing set{num}\{way}\{way}_064_{sample_length}'
image_folder_65 = rf'{input_path}\training and testing set{num}\{way}\{way}_065_{sample_length}'
image_folder_240 = rf'{input_path}\training and testing set{num}\{way}\{way}_240_{sample_length}'
image_folder_241 = rf'{input_path}\training and testing set{num}\{way}\{way}_241_{sample_length}'
image_folder_242 = rf'{input_path}\training and testing set{num}\{way}\{way}_242_{sample_length}'
labels_file1 = rf'{input_path}\WELL1_labels_{sample_num}_time_new5_3class_prob.xlsx' # 概率型标签,假设有一个样本3种类别的概率分别为[0.2,0.6,0.2]
labels_file2 = rf'{input_path}\WELl2_labels_{sample_num}_time_new5_3class_prob.xlsx' # 概率型标签
pred_file = rf'{input_path}\predicting set\{m}\{way}\{way}_block_{sample_length}'

X63 = load_images(image_folder_63)
X64 = load_images(image_folder_64)
X65 = load_images(image_folder_65)
X240 = load_images(image_folder_240)
X241 = load_images(image_folder_241)
X242 = load_images(image_folder_242)
y1 = load_labels(labels_file1)
y2 = load_labels(labels_file2)

# X = X64
# y = y1
# X = X241
# y = y2
X = np.concatenate((X64, X241), axis=0)
y = np.concatenate((y1, y2), axis=0)

X_train = X
y_train = y
X_val = np.concatenate((X65, X242), axis=0)
y_val = np.concatenate((y1, y2), axis=0)
# X_val = X65
# y_val = y1
# X_val = X242
# y_val = y2
# X_test = X64
# y_test = y1
X_test = np.concatenate((X63, X240), axis=0)
y_test = np.concatenate((y1, y2), axis=0)
X_well1 = X63
X_well2 = X240

n_classes = y.shape[1]
classes = {
    0: '''Gas sandstone''',
    1: '''Non-gas sandstone''',
    2: '''Mudstone'''
}
colors = {0: 'red', 1: 'yellow', 2: 'gray'}

y1_class = np.argmax(y1, axis=1)
y2_class = np.argmax(y2, axis=1)
y_class = np.argmax(y, axis=1)

unique1, counts1, percentages1 = calculate_distribution(y1_class)
unique2, counts2, percentages2 = calculate_distribution(y2_class)
unique3, counts3, percentages3 = calculate_distribution(y_class)

plot_and_save_distribution(unique1, counts1, percentages1, classes, colors, 'Well 1', output_path)
plot_and_save_distribution(unique2, counts2, percentages2, classes, colors, 'Well 2', output_path)
plot_and_save_distribution(unique3, counts3, percentages3, classes, colors, 'ALL', output_path)

plot_and_save_lithology(y1_class, output_path, colors, file_name="WELL1_real_lithology.png")
plot_and_save_lithology(y2_class, output_path, colors, file_name="WELL2_real_lithology.png")
plot_and_save_probability_lithology(y1, output_path, colors, file_name="WELL1_real_lithology_prob.png")
plot_and_save_probability_lithology(y2, output_path, colors, file_name="WELL2_real_lithology_prob.png")
plot_and_save_probability_lithology_curve(y1, output_path, colors, file_name="WELL1_real_lithology_prob_curve.png")
plot_and_save_probability_lithology_curve(y2, output_path, colors, file_name="WELL2_real_lithology_prob_curve.png")

y_train_encoded = y_train
y_test_encoded = y_test
y_val_encoded = y_val

batch_size = 32
n_estimators = 30
epochs = 30
img_height, img_width = X64.shape[1], X64.shape[2]
img_channels = 3
learning_rate = 0.01

from multi_AdaBoost_CNN_regression import AdaBoostClassifier as Ada_CNN
from realtime_plot_callback_classification import RealTimeTrainingPlotCallback

# 设置不同模型的前缀
training_plot_callback_model = RealTimeTrainingPlotCallback(output_path, prefix="model")
training_plot_callback_Ada_CNN_model = RealTimeTrainingPlotCallback(output_path, prefix="Ada_CNN_model")
# training_plot_callback_Ada_CNN_model_False = RealTimeTrainingPlotCallback(output_path, prefix="Ada_CNN_model_False")

# 创建不同模型的回调列表，确保它们各自独立
# 定义并获取基准模型和回调
model, callbacks = baseline_model(img_height, img_width, img_channels, n_classes)
learning_rate_printer = PrintLearningRate()
# callbacks.append(learning_rate_printer)  # 如果有自定义的学习率打印回调，可以加到列表中
callbacks_model = [training_plot_callback_model, callbacks]  # 基准模型使用独立回调
model.fit(X_train, y_train_encoded, epochs=epochs, batch_size=batch_size, callbacks=callbacks_model, validation_data=(X_val, y_val_encoded)) #

# 创建 AdaBoost 模型及其回调
Ada_model, Ada_callbacks = Ada_baseline_model(img_height, img_width, img_channels, n_classes)
callbacks_Ada_CNN_model = [training_plot_callback_Ada_CNN_model, Ada_callbacks]  # AdaBoost-CNN模型使用独立回调
Ada_CNN_model = Ada_CNN(
    base_estimator=Ada_model,
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    epochs=epochs,
    batch_size=batch_size,
    algorithm=algorithm,
    callbacks=callbacks_Ada_CNN_model  # 使用Ada_baseline_model产生的回调
)

# # 创建不使用之前估计器的 AdaBoost 模型及其回调
# Ada_model_False, Ada_callbacks_False = Ada_baseline_model_False(img_height, img_width, img_channels, n_classes)
# callbacks_Ada_CNN_model_False = [training_plot_callback_Ada_CNN_model_False, Ada_callbacks_False]  # AdaBoost-CNN模型(禁用之前估计器)使用独立回调
# Ada_CNN_model_False = Ada_CNN(
#     base_estimator=Ada_model_False,
#     n_estimators=n_estimators,
#     learning_rate=learning_rate,
#     epochs=epochs,
#     batch_size=batch_size,
#     algorithm=algorithm,
#     callbacks=callbacks_Ada_CNN_model_False,  # 使用Ada_baseline_model_False产生的回调
#     copy_previous_estimator=False
# )

print(f"Training with copy_previous_estimator=True")
start_time_True = time.time()
Ada_CNN_model.fit(X_train, y_train, X_val, y_val) #
end_time_True = time.time()
# print(f"Training with copy_previous_estimator=False")
# start_time_False = time.time()
# Ada_CNN_model_False.fit(X_train, y_train, X_val, y_val) #
# end_time_False = time.time()

# # print(f"Training with copy_previous_estimator=False took {end_time_False - start_time_False:.2f} seconds")
# print(f"Training with copy_previous_estimator=True took {end_time_True - start_time_True:.2f} seconds")
#
# model_test(model, X_train, y_train_encoded, X_test, y_test_encoded, classes, algorithm='CNN', is_regression=True, colors=colors)
# # model_test(Ada_CNN_model_False, X_train, y_train_encoded, X_test, y_test_encoded, classes, algorithm='Ada_CNN_False', is_regression=True)
# model_test(Ada_CNN_model, X_train, y_train_encoded, X_test, y_test_encoded, classes, algorithm='Ada_CNN', is_regression=True, colors=colors)
#
# CNN_y_proba1 = model.predict(X_well1)
# CNN_y_proba2 = model.predict(X_well2)
# Ada_CNN_y_pred1 = Ada_CNN_model.predict(X_well1)
# Ada_CNN_y_pred2 = Ada_CNN_model.predict(X_well2)
#
# plot_and_save_lithology(np.argmax(CNN_y_proba1, axis=1), output_path, colors, file_name="CNN_WELL1_pred_lithology.png")
# plot_and_save_lithology(np.argmax(CNN_y_proba2, axis=1), output_path, colors, file_name="CNN_WELL2_pred_lithology.png")
# plot_and_save_lithology(Ada_CNN_y_pred1, output_path, colors, file_name="Ada_CNN_WELL1_pred_lithology.png")
# plot_and_save_lithology(Ada_CNN_y_pred2, output_path, colors, file_name="Ada_CNN_WELL2_pred_lithology.png")
#
# if algorithm == 'SAMME.R':
#     Ada_CNN_y_proba1 = Ada_CNN_model.predict_proba(X_well1)
#     Ada_CNN_y_proba2 = Ada_CNN_model.predict_proba(X_well2)
#
#     plot_and_save_probability_lithology(CNN_y_proba1, output_path, colors, file_name="CNN_WELL1_pred_proba_lithology.png")
#     plot_and_save_probability_lithology(CNN_y_proba2, output_path, colors, file_name="CNN_WELL2_pred_proba_lithology.png")
#     plot_and_save_probability_lithology(Ada_CNN_y_proba1, output_path, colors, file_name="Ada_CNN_WELL1_pred_proba_lithology.png")
#     plot_and_save_probability_lithology(Ada_CNN_y_proba2, output_path, colors, file_name="Ada_CNN_WELL2_pred_proba_lithology.png")
#
#     plot_and_save_probability_lithology_curve(CNN_y_proba1, output_path, colors, file_name="CNN_WELL1_pred_proba_lithology_curve.png")
#     plot_and_save_probability_lithology_curve(CNN_y_proba2, output_path, colors, file_name="CNN_WELL2_pred_proba_lithology_curve.png")
#     plot_and_save_probability_lithology_curve(Ada_CNN_y_proba1, output_path, colors, file_name="Ada_CNN_WELL1_pred_proba_lithology_curve.png")
#     plot_and_save_probability_lithology_curve(Ada_CNN_y_proba2, output_path, colors, file_name="Ada_CNN_WELL2_pred_proba_lithology_curve.png")

batch_size = sample_num
print(f"Predicting profile")



all_CNN_predictions = []

for i, (X_batch, batch_files) in enumerate(load_images_in_batches(pred_file, batch_size)):
    print(f"Processing batch {i + 1}/{len(range(0, len(os.listdir(pred_file)), batch_size))}...")
    y_pred_prob_batch = model.predict(X_batch)
    y_pred_batch = np.argmax(y_pred_prob_batch, axis=1)
    all_CNN_predictions.append(y_pred_batch)

all_CNN_predictions_array = np.column_stack(all_CNN_predictions)

predictions_CNN_df = pd.DataFrame(all_CNN_predictions_array)
predictions_CNN_df.to_excel(rf"{output_path}\{way}_CNN_predictions_{n_classes}.xlsx", index=False)

colors = {0: 'red', 1: 'yellow', 2: 'gray'}
rgb_image = np.zeros((*predictions_CNN_df.shape, 3))

for value, color in colors.items():
    mask = predictions_CNN_df == value
    rgb_image[mask.values] = mcolors.to_rgb(color)

labels_df1 = pd.read_excel(labels_file1, header=None)
labels_df2 = pd.read_excel(labels_file2, header=None)

white_color = [1, 1, 1]

well1_rows = min(sample_num, len(labels_df1))
rgb_image[:, 64, :] = white_color

well2_rows = min(sample_num, len(labels_df2))
rgb_image[14:14 + well2_rows, 241, :] = white_color

fig, ax = plt.subplots()
ax.imshow(rgb_image, interpolation='nearest')
ax.axis('off')

fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
ax.set_position([0, 0, 1, 1])

plt.tight_layout()
plt.savefig(rf'{output_path}\{way}_CNN_predictions_image_{n_classes}.png', format='png', bbox_inches='tight', pad_inches=0,
            dpi=600)

all_predictions = []

for i, (X_batch, batch_files) in enumerate(load_images_in_batches(pred_file, batch_size)):
    print(f"Processing batch {i + 1}/{len(range(0, len(os.listdir(pred_file)), batch_size))}...")
    y_pred_batch = Ada_CNN_model.predict(X_batch)
    all_predictions.append(y_pred_batch)

all_predictions_array = np.column_stack(all_predictions)

predictions_df = pd.DataFrame(all_predictions_array)
predictions_df.to_excel(rf"{output_path}\{way}_{algorithm}_predictions_{n_classes}.xlsx", index=False)

colors = {0: 'red', 1: 'yellow', 2: 'gray'}
rgb_image = np.zeros((*predictions_df.shape, 3))

for value, color in colors.items():
    mask = predictions_df == value
    rgb_image[mask.values] = mcolors.to_rgb(color)

labels_df1 = pd.read_excel(labels_file1, header=None)
labels_df2 = pd.read_excel(labels_file2, header=None)

white_color = [1, 1, 1]

well1_rows = min(sample_num, len(labels_df1))
rgb_image[:, 64, :] = white_color

well2_rows = min(sample_num, len(labels_df2))
rgb_image[14:14 + well2_rows, 241, :] = white_color

fig, ax = plt.subplots()
ax.imshow(rgb_image, interpolation='nearest')
ax.axis('off')

fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
ax.set_position([0, 0, 1, 1])

plt.tight_layout()
plt.savefig(rf'{output_path}\{way}_{algorithm}_predictions_image_{n_classes}.png', format='png', bbox_inches='tight', pad_inches=0,
            dpi=600)




all_CNN_proba_predictions = []

for i, (X_batch, batch_files) in enumerate(load_images_in_batches(pred_file, batch_size)):
    print(f"Processing batch {i + 1}/{len(range(0, len(os.listdir(pred_file)), batch_size))}...")
    y_proba_batch = model.predict(X_batch)
    all_CNN_proba_predictions.extend(y_proba_batch)

all_CNN_proba_predictions_array = np.array(all_CNN_proba_predictions)

for class_index in range(n_classes):
    class_proba = all_CNN_proba_predictions_array[:, class_index]
    class_image_data = class_proba.reshape(-1, batch_size).T

    class_proba_df = pd.DataFrame(class_image_data)
    class_proba_df.to_excel(
        rf"{output_path}\{way}_CNN_sorted_proba_class_{n_classes}_{class_index}.xlsx",
        index=False)

colors = {0: 'red', 1: 'yellow', 2: 'gray'}
gray_Green_yellow_cmap = LinearSegmentedColormap.from_list(
    "GrayGreenYellow",
    ['gray', 'green', 'yellow', 'red']
)
cmap = gray_Green_yellow_cmap

labels_df1 = pd.read_excel(labels_file1, header=None)
labels_df2 = pd.read_excel(labels_file2, header=None)

for class_index in range(n_classes):
    sorted_proba_df = pd.read_excel(
        rf"{output_path}\{way}_CNN_sorted_proba_class_{n_classes}_{class_index}.xlsx")
    sorted_class_proba = sorted_proba_df.values

    class_image = cmap(sorted_class_proba)

    white_color = [1, 1, 1, 1]

    well1_rows = min(sample_num, len(labels_df1))
    class_image[:, 64, :] = white_color

    well2_rows = min(sample_num, len(labels_df2))
    class_image[14:14 + well2_rows, 241, :] = white_color

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(class_image, aspect='auto')
    ax.axis('off')
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
    ax.set_position([0, 0, 1, 1])

    plt.tight_layout()
    plt.savefig(
        rf'{output_path}\{way}_CNN_sorted_proba_predictions_image_class_{n_classes}_{class_index}.png',
        format='png',
        bbox_inches='tight', pad_inches=0, dpi=600)

all_proba_predictions = []

for i, (X_batch, batch_files) in enumerate(load_images_in_batches(pred_file, batch_size)):
    print(f"Processing batch {i + 1}/{len(range(0, len(os.listdir(pred_file)), batch_size))}...")
    y_proba_batch = Ada_CNN_model.predict_proba(X_batch)
    all_proba_predictions.extend(y_proba_batch)

all_proba_predictions_array = np.array(all_proba_predictions)

for class_index in range(n_classes):
    class_proba = all_proba_predictions_array[:, class_index]
    class_image_data = class_proba.reshape(-1, batch_size).T

    class_proba_df = pd.DataFrame(class_image_data)
    class_proba_df.to_excel(rf"{output_path}\{way}_{algorithm}_sorted_proba_class_{n_classes}_{class_index}.xlsx", index=False)

colors = {0: 'red', 1: 'yellow', 2: 'gray'}
gray_Green_yellow_cmap = LinearSegmentedColormap.from_list(
    "GrayGreenYellow",
    ['gray', 'green', 'yellow', 'red']
)
cmap = gray_Green_yellow_cmap

labels_df1 = pd.read_excel(labels_file1, header=None)
labels_df2 = pd.read_excel(labels_file2, header=None)

for class_index in range(n_classes):
    sorted_proba_df = pd.read_excel(rf"{output_path}\{way}_{algorithm}_sorted_proba_class_{n_classes}_{class_index}.xlsx")
    sorted_class_proba = sorted_proba_df.values

    class_image = cmap(sorted_class_proba)

    white_color = [1, 1, 1, 1]

    well1_rows = min(sample_num, len(labels_df1))
    class_image[:, 64, :] = white_color

    well2_rows = min(sample_num, len(labels_df2))
    class_image[14:14 + well2_rows, 241, :] = white_color

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(class_image, aspect='auto')
    ax.axis('off')
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
    ax.set_position([0, 0, 1, 1])

    plt.tight_layout()
    plt.savefig(rf'{output_path}\{way}_{algorithm}_sorted_proba_predictions_image_class_{n_classes}_{class_index}.png', format='png',
                bbox_inches='tight', pad_inches=0, dpi=600)
