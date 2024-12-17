import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
import os


class RealTimeTrainingPlotCallback(Callback):
    def __init__(self, output_path, prefix="model"):
        super().__init__()
        self.output_path = output_path
        self.prefix = prefix

        # 为回归问题定义指标列表
        self.train_losses = []
        self.val_losses = []
        self.train_mses = []
        self.val_mses = []
        self.train_maes = []
        self.val_maes = []

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        plt.ion()
        self.fig, (self.ax_loss, self.ax_mse, self.ax_mae) = plt.subplots(1, 3, figsize=(18, 5))

    def on_epoch_end(self, epoch, logs=None):
        # 从日志中获取损失、MSE 和 MAE 的值
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.train_mses.append(logs.get('mse'))
        self.val_mses.append(logs.get('val_mse'))
        self.train_maes.append(logs.get('mae'))
        self.val_maes.append(logs.get('val_mae'))

        # 清空并更新图像
        self.ax_loss.cla()
        self.ax_mse.cla()
        self.ax_mae.cla()

        # 绘制损失曲线
        self.ax_loss.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Train Loss')
        self.ax_loss.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Validation Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.set_title('Loss Curve')
        self.ax_loss.legend()

        # 绘制 MSE 曲线
        self.ax_mse.plot(range(1, len(self.train_mses) + 1), self.train_mses, label='Train MSE')
        self.ax_mse.plot(range(1, len(self.val_mses) + 1), self.val_mses, label='Validation MSE')
        self.ax_mse.set_xlabel('Epoch')
        self.ax_mse.set_ylabel('MSE')
        self.ax_mse.set_title('MSE Curve')
        self.ax_mse.legend()

        # 绘制 MAE 曲线
        self.ax_mae.plot(range(1, len(self.train_maes) + 1), self.train_maes, label='Train MAE')
        self.ax_mae.plot(range(1, len(self.val_maes) + 1), self.val_maes, label='Validation MAE')
        self.ax_mae.set_xlabel('Epoch')
        self.ax_mae.set_ylabel('MAE')
        self.ax_mae.set_title('MAE Curve')
        self.ax_mae.legend()

        plt.draw()
        plt.pause(0.1)

    def on_train_end(self, logs=None):
        plot_filename = os.path.join(self.output_path, f"{self.prefix}_final_training_plot.png")
        self.fig.savefig(plot_filename, bbox_inches='tight', pad_inches=0, dpi=600)
        plt.close(self.fig)
        print(f"Final training plot saved to {plot_filename}")

