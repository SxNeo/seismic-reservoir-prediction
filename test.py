import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score  # 导入 accuracy_score 评估模型
from sklearn.metrics import f1_score, confusion_matrix,precision_score, recall_score, roc_curve, auc
import tensorflow
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import sys
import seaborn as sns
import matplotlib.patches as patches
import time
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback


# 定义实验编号变量
sample_num = 119
channel_num = 300
way = 'ISD'
algorithm = 'SAMME.R'  # 'SAMME.R' 或 'SAMME'
input_path = r"C:\01 CodeofPython\AdaBoost_CNN\AdaBoost_CNN-master\INPUT"
output_path = rf"C:\01 CodeofPython\AdaBoost_CNN\AdaBoost_CNN-master\OUTPUT\{way}"
plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})
#####################################################在test_base的基础上做了修改，优化了函数的调用，其余没变###############################
# 设置随机种子以保证结果的可重复性
seed = 50
np.random.seed(seed)
tensorflow.random.set_seed(seed)

# 加载图像数据
def load_images(image_folder):
    images = []
    # 获取所有以 .bmp 结尾的文件名
    bmp_files = [f for f in os.listdir(image_folder) if f.endswith('.bmp')]
    # 按字母顺序排序文件名
    sorted_files = sorted(bmp_files)

    for filename in sorted_files:
        print(f"正在读取文件: {filename}")  # 打印文件名
        image_path = os.path.join(image_folder, filename)
        img = load_img(image_path)  # 加载彩色图像，默认为彩色模式
        img_array = img_to_array(img) / 255.0  # 归一化
        images.append(img_array)
    return np.array(images)

# 修改后的 load_images 函数，增加了批处理功能
def load_images_in_batches(image_folder, batch_size):
    bmp_files = [f for f in os.listdir(image_folder) if f.endswith('.bmp')]
    sorted_files = sorted(bmp_files)
    for i in range(0, len(sorted_files), batch_size):
        batch_files = sorted_files[i:i + batch_size]
        images = []
        for filename in batch_files:
            print(f"正在读取文件: {filename}")  # 打印文件名
            image_path = os.path.join(image_folder, filename)
            img = load_img(image_path)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
        yield np.array(images), batch_files

# 读取标签数据
def load_labels(labels_file):
    # 使用 header=None 读取数据，避免跳过第一行
    labels_df = pd.read_excel(labels_file, header=None)
    # 假设标签位于第一列，这里用 iloc[:, 0] 获取第一列的所有行
    labels = labels_df.iloc[:, 0].values
    return labels

class PrintLearningRate(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # 获取当前的学习率
        lr = K.get_value(self.model.optimizer.lr)
        # 打印学习率
        print(f"\nEpoch {epoch + 1}: Current learning rate = {lr}")

# 定义构建基础 CNN 模型的函数
def baseline_model(img_height, img_width, img_channels, n_classes):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding="same", input_shape=(img_height, img_width, img_channels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    adam_optimizer = Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.25, patience=4, min_lr=0.0001)
    early_stopping = EarlyStopping(monitor='accuracy', patience=4, verbose=1)

    return model, [reduce_lr, early_stopping]

def baseline_model_False(img_height, img_width, img_channels, n_classes):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding="same", input_shape=(img_height, img_width, img_channels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    adam_optimizer = Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.25, patience=4, min_lr=0.0001)

    return model, reduce_lr

def plot_and_save_confusion_matrix(y_true, y_pred, classes, output_path, file_name="confusion_matrix.tif"):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 构造类别标签列表
    class_labels = [classes[i] for i in sorted(classes)]

    # 使用seaborn绘制混淆矩阵
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('Real label')
    plt.xlabel('Predicted label')

    # 检查输出路径是否存在，如果不存在则创建
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 保存图像
    plt.savefig(os.path.join(output_path, file_name), bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()

def save_operating_data(way):
    # 确保输出路径存在
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 定义文件全路径
    file_path = os.path.join(output_path, f"{way['algorithm']}_Operating_data.txt")

    # 构造要保存的内容
    content = f"""
    {way['algorithm']}_train_accuracy_: {way['acc_train']}
    {way['algorithm']}_test_accuracy_: {way['acc_test']}
    {way['algorithm']}_training_Precision: {way['pre_train']}
    {way['algorithm']}_test_Precision: {way['pre_test']}
    {way['algorithm']}_training_Recall: {way['rec_train']}
    {way['algorithm']}_test_Recall: {way['rec_test']}
    {way['algorithm']}_training_F1-score: {way['f1_train']}
    {way['algorithm']}_test_F1-score: {way['f1_test']}
    """

    # 将内容写入文件
    with open(file_path, "w") as file:
        file.write(content)

    print(f"操作数据已保存到：{file_path}")

def model_test(model, X_train, y_train, X_test, y_test, classes, algorithm):
    # 判断模型类型，如果是CNN，确保使用独热编码的标签
    if algorithm == 'CNN':
        # 确保这里的y_train和y_test已经是独热编码形式
        # 假设y_train_encoded和y_test_encoded是独热编码后的标签
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # 将预测结果从概率转换为标签索引
        y_train_pred = np.argmax(y_train_pred, axis=1)
        y_test_pred = np.argmax(y_test_pred, axis=1)

        # 因为y_train和y_test是独热编码，所以要转换回原始标签形式进行评估
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)
    else:
        # 对于Ada_CNN，直接使用原始标签
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_train = precision_score(y_train, y_train_pred, average='macro')
    precision_test = precision_score(y_test, y_test_pred, average='macro')
    recall_train = recall_score(y_train, y_train_pred, average='macro')
    recall_test = recall_score(y_test, y_test_pred, average='macro')
    f1_score_train = f1_score(y_train, y_train_pred, average='macro')
    f1_score_test = f1_score(y_test, y_test_pred, average='macro')

    # 示例：保存测试集的混淆矩阵图片
    plot_and_save_confusion_matrix(y_train, y_train_pred, classes, output_path,
                                   rf"{algorithm}_confusion_matrix_train.png")
    plot_and_save_confusion_matrix(y_test, y_test_pred, classes, output_path,
                                   rf"{algorithm}_confusion_matrix_test.png")

    way = {
        'algorithm': algorithm,
        'acc_train': accuracy_train,
        'acc_test': accuracy_test,
        'pre_train': precision_train,
        'pre_test': precision_test,
        'rec_train': recall_train,
        'rec_test': recall_test,
        'f1_train': f1_score_train,
        'f1_test': f1_score_test,
    }

    # 用实际的准确率和F1分数调用save_operating_data函数
    save_operating_data(way)

def plot_and_save_lithology(labels, output_path, colors, file_name="lithology.png"):
    # 创建一个空的RGB图像，高度为标签数量，宽度为1，3是颜色通道数量
    lithology_image = np.zeros((len(labels), 1, 3))

    for i, label in enumerate(labels):
        # 确保label是一个整数而不是一个数组
        if isinstance(label, np.ndarray):
            label = label[0]  # 如果label是一个NumPy数组，获取第一个元素
        lithology_image[i, 0, :] = mcolors.to_rgb(colors[int(label)])  # 使用int确保键是整数类型

    # 创建图像并调整布局
    fig, ax = plt.subplots(figsize=(2, 10))  # 调整图像大小
    ax.imshow(lithology_image, aspect='auto')
    ax.axis('off')

    # 显示并保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, file_name), bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()

def plot_and_save_probability_lithology(probabilities, output_path, colors, file_name="probability_distribution.png"):
    # 创建图形和轴对象
    fig, ax = plt.subplots(figsize=(2, 10))  # 宽度固定为6，高度根据样本数量适当调整

    # 遍历每个样本
    for index, prob in enumerate(probabilities):
        start = 0  # 开始位置
        for class_index, class_prob in enumerate(prob):
            # 为每个类别添加一个彩色部分
            ax.add_patch(
                patches.Rectangle(
                    (start, index),  # (x, y)矩形左下角
                    class_prob,      # width
                    1,               # height
                    facecolor=colors[class_index]
                )
            )
            start += class_prob  # 更新下一个矩形的开始位置

    # 设置图形属性
    # ax.set_xlim(0, 1)
    ax.set_ylim(0, len(probabilities))
    ax.set_aspect('auto')
    ax.set_xticks([])  # 移除x轴标记，因为每行代表一个样本
    ax.set_yticks([])  # 移除y轴标记，因为每行代表一个样本
    ax.invert_yaxis()  # 反转y轴，使第一个样本出现在顶部

    # 显示并保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, file_name), bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()

# 计算每个类别的占比和数量
def calculate_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    percentages = (counts / len(y)) * 100
    return unique, counts, percentages

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.1f}%  ({v:d})'.format(p=pct, v=val)
    return my_format

# 绘制并保存饼图和条形图
def plot_and_save_distribution(unique, counts, percentages, classes, colors, title, output_path):
    # 根据实际存在的类别索引从颜色字典中选择颜色
    pie_colors = [colors[i] for i in unique]

    labels = [classes[i] for i in unique]

    # 饼图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=140, colors=pie_colors)
    plt.title(f'{title} Class Distribution')

    # 条形图
    plt.subplot(1, 2, 2)
    plt.bar(labels, counts, color=pie_colors)  # 使用正确的颜色
    plt.title(f'{title} Class Distribution')
    plt.ylabel('Counts')
    for i, count in enumerate(counts):
        plt.text(i, count + 0.2, str(count), ha='center')

    plt.tight_layout()

    # 保存图表
    plt.savefig(os.path.join(output_path, f'{title}_Distribution.png'), bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()  # 关闭当前的绘图窗口，防止在Jupyter中重复显示

name1 = '3'
name2 = '4'
name3 = '20'

# 主程序
image_folder_63 = rf'{input_path}\测试窗口数据集{name1}\{way}\{way}_063_{name3}'
image_folder_64 = rf'{input_path}\测试窗口数据集{name1}\{way}\{way}_064_{name3}'
image_folder_65 = rf'{input_path}\测试窗口数据集{name1}\{way}\{way}_065_{name3}'
image_folder_240 = rf'{input_path}\测试窗口数据集{name1}\{way}\{way}_240_{name3}'
image_folder_241 = rf'{input_path}\测试窗口数据集{name1}\{way}\{way}_241_{name3}'
image_folder_242 = rf'{input_path}\测试窗口数据集{name1}\{way}\{way}_242_{name3}'
labels_file1 = rf'{input_path}\WELL1_labels_{sample_num}_time_new3_3class.xlsx'
labels_file2 = rf'{input_path}\WELl2_labels_{sample_num}_time_new3_3class.xlsx'
pred_file = rf'{input_path}\测试窗口数据集{name2}\{way}_block_{name3}'

# 加载图像和标签数据
X1 = load_images(image_folder_63)
X2 = load_images(image_folder_64)
X3 = load_images(image_folder_65)
X4 = load_images(image_folder_240)
X5 = load_images(image_folder_241)
X6 = load_images(image_folder_242)
y1 = load_labels(labels_file1)
y2 = load_labels(labels_file2)

# 合并图像数据集
# X = np.concatenate((X1, X2, X3, X4, X5, X6), axis=0)
# y = np.concatenate((y1, y1, y1, y2, y2, y2), axis=0)

X = np.concatenate((X2,X3, X5,X6), axis=0)
y = np.concatenate((y1,y1, y2, y2), axis=0)
X_test1 = X1
X_test2 = X4
X_train = X
y_train = y
X_test = np.concatenate((X1, X4), axis=0)
y_test = np.concatenate((y1, y2), axis=0)

n_classes = len(np.unique(y1))
classes = {
    0: '''Gas
sandstone''',
    1: '''Non-gas
sandstone''',
    2: '''Mudstone'''
}
colors = {0: 'red', 1: 'yellow', 2: 'gray'}
# classes = {0: 'Sandstone', 1: 'Mudstone'}
# colors = {0: 'yellow', 1: 'gray'}

# 计算分布
unique1, counts1, percentages1 = calculate_distribution(y1)
unique2, counts2, percentages2 = calculate_distribution(y2)
unique3, counts3, percentages3 = calculate_distribution(y)

# 绘制并保存WELL1的分布图表
plot_and_save_distribution(unique1, counts1, percentages1, classes, colors, 'Well 1', output_path)

# 绘制并保存WELL2的分布图表
plot_and_save_distribution(unique2, counts2, percentages2, classes, colors, 'Well 2', output_path)

# 绘制并保存所有的分布图表
plot_and_save_distribution(unique3, counts3, percentages3, classes, colors, 'ALL', output_path)

# 绘制并保存岩性图
plot_and_save_lithology(y1, output_path, colors, file_name="WELL1_real_lithology.png")
plot_and_save_lithology(y2, output_path, colors, file_name="WELL2_real_lithology.png")

# 首先，将数据集分为训练集+验证集和测试集，比例为80%:20%
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 对合并后的标签进行独热编码
encoder = OneHotEncoder(sparse=False)
encoder.fit(y.reshape(-1, 1))  # 使用合并后的标签进行fit
y_train_encoded = encoder.transform(y_train.reshape(-1, 1))
# y_val_encoded = encoder.transform(y_val.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

batch_size = 32  # 设置批量大小
n_estimators = 10  # 设置估计器数量
epochs = 30  # 设置训练周期
# img_height, img_width = X_train.shape[1], X_train.shape[2]
img_height, img_width = X.shape[1], X.shape[2]
img_channels = 3  # 使用 3 个颜色通道，因为图像是彩色的
learning_rate = 0.01

# AdaBoost+CNN 部分
from multi_AdaBoost_CNN import AdaBoostClassifier as Ada_CNN  # 导入自定义的 AdaBoost 与 CNN 结合的分类器

model, callbacks = baseline_model(img_height, img_width, img_channels, n_classes)
learning_rate_printer = PrintLearningRate()
callbacks.append(learning_rate_printer) # 确保把自定义的回调函数加到回调函数列表中

model.fit(X_train, y_train_encoded, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

Ada_CNN_model = Ada_CNN(base_estimator=model, n_estimators=n_estimators, learning_rate=learning_rate,
                            epochs=epochs, batch_size=batch_size, algorithm=algorithm, callbacks=callbacks)

model_False, callbacks_False = baseline_model_False(img_height, img_width, img_channels, n_classes)
learning_rate_printer = PrintLearningRate()

Ada_CNN_model_False = Ada_CNN(base_estimator=model_False, n_estimators=n_estimators, learning_rate=learning_rate,
                            epochs=epochs, batch_size=batch_size, algorithm=algorithm, callbacks=callbacks_False, copy_previous_estimator=False)  # 创建 AdaBoost+CNN 模型(不含迁移技术)

print(f"Training with copy_previous_estimator=False")
start_time_False = time.time()  # 记录开始时间
Ada_CNN_model_False.fit(X_train, y_train)
end_time_False = time.time()  # 记录结束时间
print(f"Training with copy_previous_estimator=True")
start_time_True = time.time()  # 记录开始时间
Ada_CNN_model.fit(X_train, y_train)
end_time_True = time.time()  # 记录结束时间
print(f"Training with copy_previous_estimator=False took {end_time_False - start_time_False:.2f} seconds")
print(f"Training with copy_previous_estimator=True took {end_time_True - start_time_True:.2f} seconds")

model_test(model_False, X_train, y_train_encoded, X_test, y_test_encoded, classes, algorithm= 'CNN')
model_test(Ada_CNN_model_False, X_train, y_train, X_test, y_test, classes, algorithm= 'Ada_CNN_False')
model_test(Ada_CNN_model, X_train, y_train, X_test, y_test, classes, algorithm= 'Ada_CNN')

# 对于Ada_CNN，直接使用原始标签s
CNN_y_proba1 = model_False.predict(X_test1)
CNN_y_proba2 = model_False.predict(X_test2)
Ada_CNN_y_pred1 = Ada_CNN_model.predict(X_test1)
Ada_CNN_y_pred2 = Ada_CNN_model.predict(X_test2)

# 绘制并保存岩性图
plot_and_save_lithology(np.argmax(CNN_y_proba1, axis=1), output_path, colors, file_name="CNN_WELL1_pred_lithology.png")
plot_and_save_lithology(np.argmax(CNN_y_proba2, axis=1), output_path, colors, file_name="CNN_WELL2_pred_lithology.png")
plot_and_save_lithology(Ada_CNN_y_pred1, output_path, colors, file_name="Ada_CNN_WELL1_pred_lithology.png")
plot_and_save_lithology(Ada_CNN_y_pred2, output_path, colors, file_name="Ada_CNN_WELL2_pred_lithology.png")

if algorithm == 'SAMME.R':
    # 对于CNN已经是概率预测
    Ada_CNN_y_proba1 = Ada_CNN_model.predict_proba(X_test1)
    Ada_CNN_y_proba2 = Ada_CNN_model.predict_proba(X_test2)

    # 调用绘制概率岩性图的函数
    plot_and_save_probability_lithology(CNN_y_proba1, output_path, colors, file_name="CNN_WELL1_pred_proba_lithology.png")
    plot_and_save_probability_lithology(CNN_y_proba2, output_path, colors, file_name="CNN_WELL2_pred_proba_lithology.png")
    plot_and_save_probability_lithology(Ada_CNN_y_proba1, output_path, colors, file_name="Ada_CNN_WELL1_pred_proba_lithology.png")
    plot_and_save_probability_lithology(Ada_CNN_y_proba2, output_path, colors, file_name="Ada_CNN_WELL2_pred_proba_lithology.png")

########################################################
batch_size = sample_num
print(f"正在预测剖面")
Ada_CNN_model_all = Ada_CNN(base_estimator=model, n_estimators=n_estimators, learning_rate=learning_rate,
                            epochs=epochs, algorithm=algorithm, callbacks=callbacks)
Ada_CNN_model_all.fit(X_train, y_train)# 用全部的数据集作为训练集训练模型
# Ada_CNN_model_all = Ada_CNN_model

if algorithm == 'SAMME':
    all_CNN_predictions = []  # 用于存储所有批次的预测结果

    for i, (X_batch, batch_files) in enumerate(load_images_in_batches(pred_file, batch_size)):
        print(f"正在处理批次 {i + 1}/{len(range(0, len(os.listdir(pred_file)), batch_size))}...")
        y_pred_prob_batch = model.predict(X_batch)
        y_pred_batch = np.argmax(y_pred_prob_batch, axis=1)
        all_CNN_predictions.append(y_pred_batch)  # 将每批次的预测结果作为一列添加到列表中

    # 将预测结果转换为 NumPy 数组
    all_CNN_predictions_array = np.column_stack(all_CNN_predictions)

    # 保存预测结果矩阵到Excel文件
    predictions_CNN_df = pd.DataFrame(all_CNN_predictions_array)
    predictions_CNN_df.to_excel(rf"{output_path}\{algorithm}_CNN_{sample_num}_predictions_{sample_num}_{n_classes}.xlsx", index=False)

    # 定义颜色映射
    # colors = {0: 'yellow',1: 'gray'}
    colors = {0: 'red', 1: 'yellow', 2: 'gray'}

    # 创建一个空的RGB图像，其尺寸与 predictions_df 相同，3是颜色通道数量
    rgb_image = np.zeros((*predictions_CNN_df.shape, 3))

    # 填充图像
    for value, color in colors.items():
        mask = predictions_CNN_df == value
        rgb_image[mask.values] = mcolors.to_rgb(color)

    # 使用 Pandas 读取 Excel 文件到 DataFrame
    labels_df1 = pd.read_excel(labels_file1, header=None)
    labels_df2 = pd.read_excel(labels_file2, header=None)

    white_color = [1, 1, 1]

    # 对于WELL1的标签原来替换第64列的位置，现在使用白色填充
    well1_rows = min(sample_num, len(labels_df1))
    rgb_image[:, 64, :] = white_color

    # 使用WELL2的标签从第16行开始替换第241列
    well2_rows = min(sample_num, len(labels_df2))  # WELL2标签的数量
    rgb_image[14:14 + well2_rows, 241, :] = white_color

    # 创建图像并调整布局
    fig, ax = plt.subplots()
    ax.imshow(rgb_image, interpolation='nearest')
    ax.axis('off')

    # 调整图像边界以去除白边
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
    ax.set_position([0, 0, 1, 1])

    # 显示并保存图像
    plt.tight_layout()
    plt.savefig(rf'{output_path}\{algorithm}_CNN_{way}_predictions_image_{sample_num}_{n_classes}.png', format='png', bbox_inches='tight', pad_inches=0,
                dpi=600)
##################################################################################
    all_predictions = []  # 用于存储所有批次的预测结果

    for i, (X_batch, batch_files) in enumerate(load_images_in_batches(pred_file, batch_size)):
        print(f"正在处理批次 {i + 1}/{len(range(0, len(os.listdir(pred_file)), batch_size))}...")
        y_pred_batch = Ada_CNN_model_all.predict(X_batch)
        all_predictions.append(y_pred_batch)  # 将每批次的预测结果作为一列添加到列表中

    # 将预测结果转换为 NumPy 数组
    all_predictions_array = np.column_stack(all_predictions)

    # 保存预测结果矩阵到Excel文件
    predictions_df = pd.DataFrame(all_predictions_array)
    predictions_df.to_excel(rf"{output_path}\{algorithm}_{sample_num}_predictions_{sample_num}_{n_classes}.xlsx", index=False)

    # 定义颜色映射
    # colors = {0: 'yellow',1: 'gray'}
    colors = {0: 'red', 1: 'yellow', 2: 'gray'}

    # 创建一个空的RGB图像，其尺寸与 predictions_df 相同，3是颜色通道数量
    rgb_image = np.zeros((*predictions_df.shape, 3))

    # 填充图像
    for value, color in colors.items():
        mask = predictions_df == value
        rgb_image[mask.values] = mcolors.to_rgb(color)

    # 使用 Pandas 读取 Excel 文件到 DataFrame
    labels_df1 = pd.read_excel(labels_file1, header=None)
    labels_df2 = pd.read_excel(labels_file2, header=None)

    white_color = [1, 1, 1]

    # 对于WELL1的标签原来替换第64列的位置，现在使用白色填充
    well1_rows = min(sample_num, len(labels_df1))
    rgb_image[:, 64, :] = white_color

    # 使用WELL2的标签从第16行开始替换第241列
    well2_rows = min(sample_num, len(labels_df2))  # WELL2标签的数量
    rgb_image[14:14 + well2_rows, 241, :] = white_color

    # 创建图像并调整布局
    fig, ax = plt.subplots()
    ax.imshow(rgb_image, interpolation='nearest')
    ax.axis('off')

    # 调整图像边界以去除白边
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
    ax.set_position([0, 0, 1, 1])

    # 显示并保存图像
    plt.tight_layout()
    plt.savefig(rf'{output_path}\{algorithm}_{way}_predictions_image_{sample_num}_{n_classes}.png', format='png', bbox_inches='tight', pad_inches=0,
                dpi=600)

else:
    all_CNN_proba_predictions = []  # 用于存储所有批次的概率预测结果

    for i, (X_batch, batch_files) in enumerate(load_images_in_batches(pred_file, batch_size)):
        print(f"正在处理批次 {i + 1}/{len(range(0, len(os.listdir(pred_file)), batch_size))}...")
        y_proba_batch = model.predict(X_batch)  # 进行概率预测
        all_CNN_proba_predictions.extend(y_proba_batch)  # 每一列对应一个类别,将每批次的预测结果添加到列表中

    # 将概率预测结果转换为 NumPy 数组
    all_CNN_proba_predictions_array = np.array(all_CNN_proba_predictions)

    # 保存每个类别的预测结果矩阵到Excel文件
    for class_index in range(n_classes):
        # 获取并重塑当前类别的概率数据
        class_proba = all_CNN_proba_predictions_array[:, class_index]
        class_image_data = class_proba.reshape(-1, batch_size).T

        # 将重塑后的数据保存到Excel文件中
        class_proba_df = pd.DataFrame(class_image_data)
        class_proba_df.to_excel(
            rf"{output_path}\{algorithm}_CNN_{way}_sorted_proba_class_{class_index}_{sample_num}_{n_classes}.xlsx",
            index=False)

    # 定义颜色映射
    # colors = {0: 'yellow',1: 'gray'}
    colors = {0: 'red', 1: 'yellow', 2: 'gray'}

    # 创建颜色渐变映射
    gray_Green_yellow_cmap = LinearSegmentedColormap.from_list(
        "GrayGreenYellow",  # 名称
        ['gray', 'green', 'yellow', 'red']  # 颜色范围：从灰色(0.5, 0.5, 0.5)到绿色(0, 1, 0)再到黄色(1, 1, 0)
    )
    cmap = gray_Green_yellow_cmap

    # 使用 Pandas 读取 Excel 文件到 DataFrame
    labels_df1 = pd.read_excel(labels_file1, header=None)
    labels_df2 = pd.read_excel(labels_file2, header=None)

    # 对每个类别绘制图像
    for class_index in range(n_classes):
        # 读取每个类别的排序后概率数据
        sorted_proba_df = pd.read_excel(
            rf"{output_path}\{algorithm}_CNN_{way}_sorted_proba_class_{class_index}_{sample_num}_{n_classes}.xlsx")
        sorted_class_proba = sorted_proba_df.values

        # 根据概率值获取颜色
        class_image = cmap(sorted_class_proba)

        white_color = [1, 1, 1, 1]  # 假设第四个元素代表透明度

        # 对于WELL1的标签原来替换第64列的位置，现在使用白色填充
        well1_rows = min(sample_num, len(labels_df1))
        class_image[:, 64, :] = white_color

        # 使用WELL2的标签从第16行开始替换第241列
        well2_rows = min(sample_num, len(labels_df2))  # WELL2标签的数量
        class_image[14:14 + well2_rows, 241, :] = white_color

        # 创建图像并调整布局
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(class_image, aspect='auto')
        ax.axis('off')
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
        ax.set_position([0, 0, 1, 1])

        # 显示并保存图像
        plt.tight_layout()
        plt.savefig(
            rf'{output_path}\{algorithm}_CNN_{way}_sorted_proba_predictions_image_class_{class_index}_{sample_num}_{n_classes}.png',
            format='png',
            bbox_inches='tight', pad_inches=0, dpi=600)
    #########################################################################
    all_proba_predictions = []  # 用于存储所有批次的概率预测结果

    for i, (X_batch, batch_files) in enumerate(load_images_in_batches(pred_file, batch_size)):
        print(f"正在处理批次 {i + 1}/{len(range(0, len(os.listdir(pred_file)), batch_size))}...")
        y_proba_batch = Ada_CNN_model_all.predict_proba(X_batch)  # 进行概率预测
        all_proba_predictions.extend(y_proba_batch)  # 每一列对应一个类别,将每批次的预测结果添加到列表中

    # 将概率预测结果转换为 NumPy 数组
    all_proba_predictions_array = np.array(all_proba_predictions)

    # 保存每个类别的预测结果矩阵到Excel文件
    for class_index in range(n_classes):
        # 获取并重塑当前类别的概率数据
        class_proba = all_proba_predictions_array[:, class_index]
        class_image_data = class_proba.reshape(-1, batch_size).T

        # 将重塑后的数据保存到Excel文件中
        class_proba_df = pd.DataFrame(class_image_data)
        class_proba_df.to_excel(rf"{output_path}\{algorithm}_{way}_sorted_proba_class_{class_index}_{sample_num}_{n_classes}.xlsx", index=False)

    # 定义颜色映射
    # colors = {0: 'yellow',1: 'gray'}
    colors = {0: 'red', 1: 'yellow', 2: 'gray'}

    # 创建颜色渐变映射
    gray_Green_yellow_cmap = LinearSegmentedColormap.from_list(
        "GrayGreenYellow",  # 名称
        ['gray', 'green', 'yellow', 'red']  # 颜色范围：从灰色(0.5, 0.5, 0.5)到绿色(0, 1, 0)再到黄色(1, 1, 0)
    )
    cmap = gray_Green_yellow_cmap

    # 使用 Pandas 读取 Excel 文件到 DataFrame
    labels_df1 = pd.read_excel(labels_file1, header=None)
    labels_df2 = pd.read_excel(labels_file2, header=None)

    # 对每个类别绘制图像
    for class_index in range(n_classes):
        # 读取每个类别的排序后概率数据
        sorted_proba_df = pd.read_excel(rf"{output_path}\{algorithm}_{way}_sorted_proba_class_{class_index}_{sample_num}_{n_classes}.xlsx")
        sorted_class_proba = sorted_proba_df.values

        # 根据概率值获取颜色
        class_image = cmap(sorted_class_proba)

        white_color = [1, 1, 1, 1]  # 假设第四个元素代表透明度

        # 对于WELL1的标签原来替换第64列的位置，现在使用白色填充
        well1_rows = min(sample_num, len(labels_df1))
        class_image[:, 64, :] = white_color

        # 使用WELL2的标签从第16行开始替换第241列
        well2_rows = min(sample_num, len(labels_df2))  # WELL2标签的数量
        class_image[14:14+well2_rows, 241, :] = white_color

        # 创建图像并调整布局
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(class_image, aspect='auto')
        ax.axis('off')
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
        ax.set_position([0, 0, 1, 1])

        # 显示并保存图像
        plt.tight_layout()
        plt.savefig(rf'{output_path}\{algorithm}_{way}_sorted_proba_predictions_image_class_{class_index}_{sample_num}_{n_classes}.png', format='png',
                    bbox_inches='tight', pad_inches=0, dpi=600)