import numpy as np
import os
import matplotlib.pyplot as plt

# 设置文件夹路径和评价指标
output_path = rf"C:\Users\final\Desktop\岩性识别\代码\code\code\sample length"
ways = ['CWT', 'ISD']
sample_lengths = [5, 10, 15, 20, 25, 30]
metrics = ['CVM-MSE', 'CVM-MAE', 'CVM-RMSE', r'CVM-$R^{2}$']  # 使用 LaTeX 格式化 R^2
# 设置全局字体大小和字体
plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})

# 加载 .npy 数据
def load_metrics(way, sample_length):
    cnn_file = os.path.join(output_path, way, f'CNN_{sample_length}.npy')
    ada_cnn_file = os.path.join(output_path, way, f'Ada_CNN_{sample_length}.npy')

    cnn_data = np.load(cnn_file)
    ada_cnn_data = np.load(ada_cnn_file)

    return cnn_data, ada_cnn_data


# 绘制每个评价指标的图形
def plot_metrics(metrics, ways, factors):
    handles = []  # 用于存储图例句柄
    labels = []   # 用于存储图例标签
    legend_saved = False  # 确保图例只保存一次

    for metric_index, metric in enumerate(metrics):
        fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(14, 6))

        for way in ways:
            train_values = []
            test_values = []
            ada_train_values = []
            ada_test_values = []

            for sample_length in factors:
                cnn_data, ada_cnn_data = load_metrics(way, sample_length)
                train_values.append(cnn_data[0, metric_index])  # 训练值
                test_values.append(cnn_data[1, metric_index])  # 测试值
                ada_train_values.append(ada_cnn_data[0, metric_index])  # AdaBoost-CNN 训练值
                ada_test_values.append(ada_cnn_data[1, metric_index])  # AdaBoost-CNN 测试值

            # 设置符号大小
            marker_size = 10

            if way == 'CWT':
                line1, = ax_train.plot(factors, train_values, 'bo-', label='CWT-CNN', markersize=marker_size)
                line2, = ax_test.plot(factors, test_values, 'bo-', label='CWT-CNN', markersize=marker_size)
                line5, = ax_train.plot(factors, ada_train_values, 'b^-', label='CWT-AdaBoost-CNN', markersize=marker_size)
                line6, = ax_test.plot(factors, ada_test_values, 'b^-', label='CWT-AdaBoost-CNN', markersize=marker_size)
            elif way == 'ISD':
                line1, = ax_train.plot(factors, train_values, 'ro-', label='ISD-CNN', markersize=marker_size)
                line2, = ax_test.plot(factors, test_values, 'ro-', label='ISD-CNN', markersize=marker_size)
                line5, = ax_train.plot(factors, ada_train_values, 'r^-', label='ISD-AdaBoost-CNN', markersize=marker_size)
                line6, = ax_test.plot(factors, ada_test_values, 'r^-', label='ISD-AdaBoost-CNN', markersize=marker_size)

            if not legend_saved:  # 收集图例句柄和标签
                handles.extend([line1, line5])
                labels.extend([f'{way}-CNN', f'{way}-AdaBoost-CNN'])

        ax_train.set_xlabel('Sample Length')
        ax_train.set_ylabel(f'{metric} on cross-training sets') #  on the training set
        ax_test.set_xlabel('Sample Length')
        ax_test.set_ylabel(f'{metric} on cross-validation sets') #  on the testing set

        os.makedirs(output_path, exist_ok=True)
        plt.tight_layout()
        fig.savefig(os.path.join(output_path, f'{metric}_plot.png'), bbox_inches='tight', pad_inches=0, dpi=600)
        plt.close(fig)

        legend_saved = True  # 确保图例只保存一次

    # 保存图例
    fig_leg = plt.figure(figsize=(3, 2))
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.legend(handles, labels, loc='center')
    ax_leg.axis('off')
    fig_leg.savefig(os.path.join(output_path, 'legend.png'), bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close(fig_leg)



# 主程序
plot_metrics(metrics, ways, sample_lengths)
