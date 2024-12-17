from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_curve, auc
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from multi_AdaBoost_CNN_regression import AdaBoostClassifier as Ada_CNN
from realtime_plot_callback_classification import RealTimeTrainingPlotCallback
import time  # 导入time模块

# 定义实验编号变量
sample_num = 119
channel_num = 300
algorithm = 'SAMME.R'  # 'SAMME.R' 或 'SAMME'
input_path = rf"C:\Users\final\Desktop\岩性识别\代码\code\code"
output_path = rf"C:\Users\final\Desktop\岩性识别\代码\code\code\sample length"
# 设置全局字体大小和字体
plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})

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
    labels_df = pd.read_excel(labels_file, header=0, usecols=[0, 1, 2])  # 读取前三列为类别概率
    labels = labels_df.values  # 返回为概率分布
    return labels

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
    early_stopping = EarlyStopping(monitor='loss', patience=4, verbose=1, mode='min')

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
    early_stopping = EarlyStopping(monitor='loss', patience=4, verbose=1, mode='min')

    return model, [early_stopping]

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
    early_stopping = EarlyStopping(monitor='loss', patience=4, verbose=1, mode='min')

    return model, [early_stopping]

# 模型评估函数
# 修改后的 model_test 函数
def model_test(model, X_train, y_train, X_test, y_test, n_classes, algorithm, is_regression=False, training_time=None):
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
        # 计算训练集和测试集的回归指标
        metrics_train = {
            'mse': mean_squared_error(y_train, y_train_pred),
            'mae': mean_absolute_error(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'r2': r2_score(y_train, y_train_pred),
            'training_time': training_time  # 加入训练时间
        }
        metrics_test = {
            'mse': mean_squared_error(y_test, y_test_pred),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'r2': r2_score(y_test, y_test_pred),
            'training_time': training_time  # 加入训练时间
        }

        return metrics_train, metrics_test  # 分别返回训练集和测试集的回归指标

    else:
        # 如果是分类任务，继续计算标准分类指标
        y_train_pred_classes = np.argmax(y_train_pred, axis=1)
        y_test_pred_classes = np.argmax(y_test_pred, axis=1)

        y_train_classes = np.argmax(y_train, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        # 对于分类问题：评估准确率和其他指标
        metrics_train = {
            'accuracy': accuracy_score(y_train_classes, np.argmax(y_train_pred_classes, axis=1)),
            'f1': f1_score(y_train_classes, np.argmax(y_train_pred_classes, axis=1), average='weighted'),
            'precision': precision_score(y_train_classes, np.argmax(y_train_pred_classes, axis=1), average='weighted'),
            'recall': recall_score(y_train_classes, np.argmax(y_train_pred_classes, axis=1), average='weighted')
        }
        metrics_test = {
            'accuracy': accuracy_score(y_test_classes, np.argmax(y_test_pred_classes, axis=1)),
            'f1': f1_score(y_test_classes, np.argmax(y_test_pred_classes, axis=1), average='weighted'),
            'precision': precision_score(y_test_classes, np.argmax(y_test_pred_classes, axis=1), average='weighted'),
            'recall': recall_score(y_test_classes, np.argmax(y_test_pred_classes, axis=1), average='weighted')
        }

        return metrics_train, metrics_test  # 分别返回训练集和测试集的分类指标


# 定义 AdaBoost + CNN 模型的训练和评估函数
def train_adaboost_cnn_with_cross_validation(X, y, factor, is_regression=False):# , X_val, y_val
    # y_val_encoded = y_val
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 用于存储每个样本长度的训练和测试指标
    # CNN_all_avg_metrics_train = []
    # CNN_all_avg_metrics_test = []

    # Ada_CNN_False_all_avg_metrics_train = []
    # Ada_CNN_False_all_avg_metrics_test = []

    Ada_CNN_all_avg_metrics_train = []
    Ada_CNN_all_avg_metrics_test = []

    # 将概率标签转换为类别标签
    y_classes = np.argmax(y, axis=1)  # 每个样本的类别标签

    # 遍历每一折
    for fold, (train_val_index, test_index) in enumerate(skf.split(X, y_classes)):
        print(f"Starting fold {fold + 1}")
        X_train, X_test = X[train_val_index], X[test_index]  # 每折的测试集和训练集
        y_train, y_test = y[train_val_index], y[test_index]

        img_height, img_width = X_train.shape[1], X_train.shape[2]
        img_channels = 3  # 图像是彩色的
        n_classes = y_train.shape[1] if y_train.ndim > 1 else len(np.unique(y_train))

        y_train_encoded = y_train
        y_test_encoded = y_test

        # 创建和训练模型
        # model, callbacks = baseline_model(img_height, img_width, img_channels, n_classes)
        # start_time = time.time()  # 开始计时
        # model.fit(X_train, y_train_encoded, epochs=30, batch_size=32, callbacks=callbacks) #, validation_data=(X_val, y_val_encoded)
        # cnn_training_time = time.time() - start_time  # 记录训练时间

        # Ada_model_False, Ada_callbacks_False = Ada_baseline_model_False(img_height, img_width, img_channels, n_classes)
        # Ada_CNN_model_False = Ada_CNN(base_estimator=Ada_model_False,n_estimators=factor,learning_rate=0.01,
        #                             epochs=30,batch_size=32,algorithm=algorithm,callbacks=Ada_callbacks_False, copy_previous_estimator=False)
        # start_time = time.time()  # 开始计时
        # Ada_CNN_model_False.fit(X_train, y_train) # , X_val, y_val
        # ada_cnn_false_training_time = time.time() - start_time  # 记录训练时间

        # 创建 AdaBoost + CNN 模型
        Ada_model, Ada_callbacks = Ada_baseline_model(img_height, img_width, img_channels, n_classes)
        Ada_CNN_model = Ada_CNN(base_estimator=Ada_model, n_estimators=factor, learning_rate=0.01,
                                epochs=30, batch_size=32, algorithm=algorithm, callbacks=Ada_callbacks)
        start_time = time.time()  # 开始计时
        Ada_CNN_model.fit(X_train, y_train) #, X_val, y_val
        ada_cnn_training_time = time.time() - start_time  # 记录训练时间

        # 调用model_test并获取训练集和测试集指标
        # CNN_metrics_train, CNN_metrics_test = model_test(model, X_train, y_train_encoded, X_test, y_test_encoded,
        #                                                  n_classes, algorithm='CNN', is_regression=is_regression, training_time=cnn_training_time)
        # Ada_CNN_False_metrics_train, Ada_CNN_False_metrics_test = model_test(Ada_CNN_model_False, X_train, y_train_encoded, X_test,
        #                                                          y_test_encoded, n_classes, algorithm='Ada_CNN_False',
        #                                                          is_regression=is_regression, training_time=ada_cnn_false_training_time)
        Ada_CNN_metrics_train, Ada_CNN_metrics_test = model_test(Ada_CNN_model, X_train, y_train_encoded, X_test,
                                                                 y_test_encoded, n_classes, algorithm='Ada_CNN',
                                                                 is_regression=is_regression, training_time=ada_cnn_training_time)

        # 添加训练集和测试集指标
        # CNN_all_avg_metrics_train.append(CNN_metrics_train)
        # CNN_all_avg_metrics_test.append(CNN_metrics_test)
        # Ada_CNN_False_all_avg_metrics_train.append(Ada_CNN_False_metrics_train)
        # Ada_CNN_False_all_avg_metrics_test.append(Ada_CNN_False_metrics_test)
        Ada_CNN_all_avg_metrics_train.append(Ada_CNN_metrics_train)
        Ada_CNN_all_avg_metrics_test.append(Ada_CNN_metrics_test)

    # 计算每个样本长度下的指标均值
    # CNN_avg_metrics_train = {key: np.mean([metrics[key] for metrics in CNN_all_avg_metrics_train]) for key in CNN_all_avg_metrics_train[0]}
    # CNN_avg_metrics_test = {key: np.mean([metrics[key] for metrics in CNN_all_avg_metrics_test]) for key in CNN_all_avg_metrics_test[0]}
    # Ada_CNN_False_avg_metrics_train = {key: np.mean([metrics[key] for metrics in Ada_CNN_False_all_avg_metrics_train]) for key in Ada_CNN_False_all_avg_metrics_train[0]}
    # Ada_CNN_False_avg_metrics_test = {key: np.mean([metrics[key] for metrics in Ada_CNN_False_all_avg_metrics_test]) for key in Ada_CNN_False_all_avg_metrics_test[0]}
    Ada_CNN_avg_metrics_train = {key: np.mean([metrics[key] for metrics in Ada_CNN_all_avg_metrics_train]) for key in Ada_CNN_all_avg_metrics_train[0]}
    Ada_CNN_avg_metrics_test = {key: np.mean([metrics[key] for metrics in Ada_CNN_all_avg_metrics_test]) for key in Ada_CNN_all_avg_metrics_test[0]}

    METRICS_KEYS = ['mse', 'mae', 'rmse', 'r2', 'training_time']

    # 将每个样本长度的4x4矩阵指标结果存储在对应的列表中
    # CNN_avg_metrics_train_list = [CNN_avg_metrics_train[key] for key in METRICS_KEYS]
    # CNN_avg_metrics_test_list = [CNN_avg_metrics_test[key] for key in METRICS_KEYS]

    # Ada_CNN_False_avg_metrics_train_list = [Ada_CNN_False_avg_metrics_train[key] for key in METRICS_KEYS]
    # Ada_CNN_False_avg_metrics_test_list = [Ada_CNN_False_avg_metrics_test[key] for key in METRICS_KEYS]

    Ada_CNN_avg_metrics_train_list = [Ada_CNN_avg_metrics_train[key] for key in METRICS_KEYS]
    Ada_CNN_avg_metrics_test_list = [Ada_CNN_avg_metrics_test[key] for key in METRICS_KEYS]

    return  Ada_CNN_avg_metrics_train_list, Ada_CNN_avg_metrics_test_list
#, CNN_avg_metrics_train_list, CNN_avg_metrics_test_list,Ada_CNN_False_avg_metrics_train_list, Ada_CNN_False_avg_metrics_test_list

# 使用示例
num = '2'
all_results = {}
ways = ['CWT', 'ISD']
factors = [2,3,4,5,6,7,8,9,10] # ,20,30,40
sample_lengths = [25]# [5,10,15,20,25,30] [3,7,11,15,19,23,27]

# 主程序
for way in ways:
    # 定义标签文件路径
    labels_file1 = rf'{input_path}\WELL1_labels_{sample_num}_time_new5_3class_prob.xlsx'  # 概率型标签
    labels_file2 = rf'{input_path}\WELl2_labels_{sample_num}_time_new5_3class_prob.xlsx'  # 概率型标签

    # 创建文件夹，确保每个方式（'CWT'/'ISD'）都有自己的文件夹
    way_folder = os.path.join(output_path, way)
    os.makedirs(way_folder, exist_ok=True)  # 创建 'CWT' 或 'ISD' 文件夹

    # 初始化数据容器
    CNN_avg_metrics_all = []
    Ada_CNN_False_avg_metrics_all = []
    Ada_CNN_avg_metrics_all = []

    for sample_length in sample_lengths:
        # 图像数据文件夹路径
        # image_folder_63 = rf'{input_path}\training and testing set{num}\{way}\{way}_063_{sample_length}'
        image_folder_64 = rf'{input_path}\training and testing set{num}\{way}\{way}_064_{sample_length}'
        # image_folder_65 = rf'{input_path}\training and testing set{num}\{way}\{way}_065_{sample_length}'
        # image_folder_240 = rf'{input_path}\training and testing set{num}\{way}\{way}_240_{sample_length}'
        image_folder_241 = rf'{input_path}\training and testing set{num}\{way}\{way}_241_{sample_length}'
        # image_folder_242 = rf'{input_path}\training and testing set{num}\{way}\{way}_242_{sample_length}'

        # 加载图像和标签数据
        # X_63 = load_images(image_folder_63)
        X_64 = load_images(image_folder_64)
        # X_65 = load_images(image_folder_65)
        # X_240 = load_images(image_folder_240)
        X_241 = load_images(image_folder_241)
        # X_242 = load_images(image_folder_242)
        y1 = load_labels(labels_file1)
        y2 = load_labels(labels_file2)

        # 合并图像数据集和标签数据集
        # X = np.concatenate((X_63,X_64,X_65,X_240, X_241, X_242), axis=0)
        # y = np.concatenate((y1,y1,y1, y2, y2, y2), axis=0)
        X = np.concatenate((X_64, X_241), axis=0)
        y = np.concatenate((y1, y2), axis=0)
        # X_val = np.concatenate((X_65, X_242), axis=0)
        # y_val = np.concatenate((y1, y2), axis=0)

        for factor in factors:
            # 训练并获取每个参数的四个指标均值 CNN_avg_metrics_train, CNN_avg_metrics_test, Ada_CNN_False_metrics_train, Ada_CNN_False_avg_metrics_test
            Ada_CNN_avg_metrics_train, Ada_CNN_avg_metrics_test = train_adaboost_cnn_with_cross_validation(
                X, y, factor, is_regression=True) # , X_val, y_val
            # 将每个参数的四个指标均值保存为二维数据
            # CNN_avg_metrics_all.append([CNN_avg_metrics_train, CNN_avg_metrics_test])
            # Ada_CNN_False_avg_metrics_all.append([Ada_CNN_False_metrics_train, Ada_CNN_False_avg_metrics_test])
            Ada_CNN_avg_metrics_all.append([Ada_CNN_avg_metrics_train, Ada_CNN_avg_metrics_test])

    # 将所有的二维数据转换为三维数据
    # CNN_avg_metrics_all = np.array(CNN_avg_metrics_all)  # shape: (factors, 2, 5)，其中2表示训练或测试，5表示5种评价指标
    # Ada_CNN_False_avg_metrics_all = np.array(Ada_CNN_False_avg_metrics_all)
    Ada_CNN_avg_metrics_all = np.array(Ada_CNN_avg_metrics_all)

    # 打印形状检查
    # print(f"CNN_avg_metrics_all shape: {CNN_avg_metrics_all.shape}")
    # print(f"Ada_CNN_False_avg_metrics_all shape: {Ada_CNN_False_avg_metrics_all.shape}")
    print(f"Ada_CNN_avg_metrics_all shape: {Ada_CNN_avg_metrics_all.shape}")

    # 保存三维数据到对应的文件夹中
    for i, factor in enumerate(factors):
        # 构建文件名
        # filename_CNN = f'CNN_{factor}.npy'
        # filename_Ada_CNN_False = f'Ada_CNN_False_{factor}.npy'
        filename_Ada_CNN = f'Ada_CNN_{factor}.npy'

        # 保存为.npy格式
        # np.save(os.path.join(way_folder, filename_CNN), CNN_avg_metrics_all[i])  # 使用索引 i 访问
        # np.save(os.path.join(way_folder, filename_Ada_CNN_False), Ada_CNN_False_avg_metrics_all[i])  # 使用索引 i 访问
        np.save(os.path.join(way_folder, filename_Ada_CNN), Ada_CNN_avg_metrics_all[i])  # 使用索引 i 访问


