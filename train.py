import os
import numpy as np
import torch
import torch.nn as nn
import time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import matplotlib.pyplot as plt
from M3T_Attention import *


# from test1 import *


class EEGDataset(Dataset):
    def __init__(self, eeg_data, kinematic_data, window_size=500,
                 noise_level=0.05, warp_prob=0.3, drop_prob=0.1,
                 enable_augment=True):
        """
        新增参数说明：
        :param noise_level: 高斯噪声标准差系数 (建议0.03-0.1)
        :param warp_prob: 时间扭曲应用概率 (建议0.2-0.5)
        :param drop_prob: 通道丢弃概率 (建议0.05-0.2)
        :param enable_augment: 是否启用增强 (验证集应关闭)
        """
        # 数据校验
        assert len(eeg_data) == len(kinematic_data), \
            f"数据维度不匹配: EEG样本数 {len(eeg_data)} vs 运动数据样本数 {len(kinematic_data)}"
        assert eeg_data.shape[1] == window_size, \
            f"窗口尺寸不匹配: EEG数据窗口 {eeg_data.shape[1]} vs 设定窗口 {window_size}"
        assert kinematic_data.shape[1] == window_size, \
            f"窗口尺寸不匹配: 运动数据窗口 {kinematic_data.shape[1]} vs 设定窗口 {window_size}"

        self.eeg_data = eeg_data
        self.kinematic_data = kinematic_data
        self.window_size = window_size
        self.enable_augment = enable_augment

        # 增强参数配置
        self.noise_level = noise_level
        self.warp_prob = warp_prob
        self.drop_prob = drop_prob

        # 时间扭曲参数
        self.warp_scales = [0.9, 1.1]  # 时间缩放范围
        self.num_warps = 3  # 最大扭曲点数

    def __len__(self):
        """返回数据集总样本数"""
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg_window = self.eeg_data[idx]
        kinematic_target = self.kinematic_data[idx]

        if self.enable_augment:
            # 按顺序应用增强方法
            eeg_window = self._gaussian_noise(eeg_window)
            eeg_window = self._time_warp(eeg_window)
            eeg_window = self._channel_dropout(eeg_window)
            eeg_window = self._scale_augmentation(eeg_window)

        return torch.tensor(eeg_window, dtype=torch.float32), \
            torch.tensor(kinematic_target, dtype=torch.float32)

    def _gaussian_noise(self, data):
        """高斯噪声增强"""
        if np.random.rand() > 0.5:  # 50%概率应用
            noise = np.random.normal(
                loc=0.0,
                scale=self.noise_level * np.std(data),
                size=data.shape
            )
            return data + noise
        return data

    def _time_warp(self, data):
        """时间扭曲增强（保持时序连续性）"""
        if np.random.rand() > self.warp_prob:
            return data

        orig_steps = np.arange(data.shape[0])
        warp_steps = orig_steps.copy()

        # 随机选择扭曲点
        for _ in range(self.num_warps):
            warp_point = np.random.choice(orig_steps[1:-1])
            scale = np.random.uniform(*self.warp_scales)

            # 前半部分不变，后半部分缩放
            warp_steps[warp_point:] = warp_point + \
                                      (warp_steps[warp_point:] - warp_point) * scale

        # 线性插值保持数据长度
        from scipy.interpolate import interp1d
        warped_data = np.zeros_like(data)
        for ch in range(data.shape[1]):
            f = interp1d(orig_steps, data[:, ch], kind='linear')
            warped_steps = np.clip(warp_steps, 0, orig_steps[-1])
            warped_data[:, ch] = f(warped_steps)

        return warped_data

    def _channel_dropout(self, data):
        """模拟电极接触不良（通道丢弃）"""
        if np.random.rand() > 0.3:  # 30%概率应用
            return data

        num_channels = data.shape[1]
        drop_mask = np.random.rand(num_channels) < self.drop_prob
        data[:, drop_mask] = 0  # 直接置零
        return data

    def _scale_augmentation(self, data):
        """幅度缩放增强（通道独立）"""
        scales = np.random.uniform(0.8, 1.2, size=data.shape[1])
        return data * scales


# 在线统计量计算类（皮尔逊相关系数）
class OnlinePearsonCalculator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_xy = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.n = 0

    def update(self, preds, tars):
        preds_flat = preds.flatten()
        tars_flat = tars.flatten()

        self.sum_x += np.sum(tars_flat)
        self.sum_y += np.sum(preds_flat)
        self.sum_xy += np.sum(tars_flat * preds_flat)
        self.sum_x2 += np.sum(tars_flat ** 2)
        self.sum_y2 += np.sum(preds_flat ** 2)
        self.n += len(tars_flat)

    def compute(self):
        if self.n < 2:
            return 0.0

        cov_xy = (self.sum_xy - (self.sum_x * self.sum_y) / self.n)
        var_x = self.sum_x2 - (self.sum_x ** 2) / self.n
        var_y = self.sum_y2 - (self.sum_y ** 2) / self.n

        if var_x <= 1e-10 or var_y <= 1e-10:
            return 0.0
        return cov_xy / np.sqrt(var_x * var_y)


# 在线RMSE计算类
class OnlineRMSECalculator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_squared_error = 0.0
        self.n = 0

    def update(self, preds, tars):
        preds_flat = preds.flatten()
        tars_flat = tars.flatten()

        squared_error = np.sum((tars_flat - preds_flat) ** 2)
        self.sum_squared_error += squared_error
        self.n += len(tars_flat)

    def compute(self):
        if self.n == 0:
            return 0.0
        mse = self.sum_squared_error / self.n
        return np.sqrt(mse)


# 模型训练相关代码
def train_model(model, train_loader, val_loader, num_epochs, l1_lambda=0.000002):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 初始化TensorBoard Writer
    writer = SummaryWriter(f'runs/EEG_Transformer_P{number}')  # 添加被试者编号到日志路径

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # 初始化统计器 - 增加RMSE计算器
    train_stats = {'x': {'pearson': OnlinePearsonCalculator(), 'rmse': OnlineRMSECalculator()},
                   'y': {'pearson': OnlinePearsonCalculator(), 'rmse': OnlineRMSECalculator()},
                   'z': {'pearson': OnlinePearsonCalculator(), 'rmse': OnlineRMSECalculator()}}

    val_stats = {'x': {'pearson': OnlinePearsonCalculator(), 'rmse': OnlineRMSECalculator()},
                 'y': {'pearson': OnlinePearsonCalculator(), 'rmse': OnlineRMSECalculator()},
                 'z': {'pearson': OnlinePearsonCalculator(), 'rmse': OnlineRMSECalculator()}}

    # 存储训练损失和验证损失
    train_losses = []
    val_losses = []
    # 记录总开始时间
    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # 记录epoch开始时间
        # 训练阶段
        model.train()
        running_loss = 0.0
        # 重置所有统计器
        for axis_stats in train_stats.values():
            axis_stats['pearson'].reset()
            axis_stats['rmse'].reset()

        for eeg_batch, kinematic_batch in train_loader:
            eeg_batch = eeg_batch.to(device)
            kinematic_batch = kinematic_batch.to(device)

            optimizer.zero_grad()
            outputs = model(eeg_batch)

            # 计算各轴损失
            loss_x = criterion(outputs[:, :, 0], kinematic_batch[:, :, 0])
            loss_y = criterion(outputs[:, :, 1], kinematic_batch[:, :, 1])
            loss_z = criterion(outputs[:, :, 2], kinematic_batch[:, :, 2])
            loss = loss_x + loss_y + loss_z

            # L1正则化
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += l1_lambda * l1_norm

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.9)  # 0.9
            optimizer.step()

            running_loss += loss.item()

            # 更新统计量
            with torch.no_grad():
                for i, axis in enumerate(['x', 'y', 'z']):
                    train_stats[axis]['pearson'].update(
                        outputs[:, :, i].cpu().numpy(),
                        kinematic_batch[:, :, i].cpu().numpy()
                    )
                    train_stats[axis]['rmse'].update(
                        outputs[:, :, i].cpu().numpy(),
                        kinematic_batch[:, :, i].cpu().numpy()
                    )

        # 计算训练损失
        train_losses.append(running_loss / len(train_loader))

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        # 重置验证统计器
        for axis_stats in val_stats.values():
            axis_stats['pearson'].reset()
            axis_stats['rmse'].reset()

        with torch.no_grad():
            for eeg_batch, kinematic_batch in val_loader:
                eeg_batch = eeg_batch.to(device)
                kinematic_batch = kinematic_batch.to(device)

                outputs = model(eeg_batch)

                # 计算损失
                loss_x = criterion(outputs[:, :, 0], kinematic_batch[:, :, 0])
                loss_y = criterion(outputs[:, :, 1], kinematic_batch[:, :, 1])
                loss_z = criterion(outputs[:, :, 2], kinematic_batch[:, :, 2])
                val_running_loss += (loss_x + loss_y + loss_z).item()

                # 更新统计量
                for i, axis in enumerate(['x', 'y', 'z']):
                    val_stats[axis]['pearson'].update(
                        outputs[:, :, i].cpu().numpy(),
                        kinematic_batch[:, :, i].cpu().numpy()
                    )
                    val_stats[axis]['rmse'].update(
                        outputs[:, :, i].cpu().numpy(),
                        kinematic_batch[:, :, i].cpu().numpy()
                    )

        # 计算验证损失
        val_losses.append(val_running_loss / len(val_loader))

        # 计算epoch耗时
        epoch_duration = time.time() - epoch_start_time
        writer.add_scalar('Time/Epoch', epoch_duration, epoch)

        # TensorBoard记录指标
        writer.add_scalar('Loss/Train', running_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/Validation', val_running_loss / len(val_loader), epoch)

        # 记录各轴的Pearson和RMSE
        for axis in ['x', 'y', 'z']:
            writer.add_scalar(f'Pearson/Train_{axis.upper()}', train_stats[axis]['pearson'].compute(), epoch)
            writer.add_scalar(f'Pearson/Val_{axis.upper()}', val_stats[axis]['pearson'].compute(), epoch)
            writer.add_scalar(f'RMSE/Train_{axis.upper()}', train_stats[axis]['rmse'].compute(), epoch)
            writer.add_scalar(f'RMSE/Val_{axis.upper()}', val_stats[axis]['rmse'].compute(), epoch)

        # 计算平均指标
        train_avg_pearson = np.mean([train_stats[axis]['pearson'].compute() for axis in ['x', 'y', 'z']])
        val_avg_pearson = np.mean([val_stats[axis]['pearson'].compute() for axis in ['x', 'y', 'z']])
        train_avg_rmse = np.mean([train_stats[axis]['rmse'].compute() for axis in ['x', 'y', 'z']])
        val_avg_rmse = np.mean([val_stats[axis]['rmse'].compute() for axis in ['x', 'y', 'z']])

        writer.add_scalar('Pearson/Train_Avg', train_avg_pearson, epoch)
        writer.add_scalar('Pearson/Val_Avg', val_avg_pearson, epoch)
        writer.add_scalar('RMSE/Train_Avg', train_avg_rmse, epoch)
        writer.add_scalar('RMSE/Val_Avg', val_avg_rmse, epoch)

        # 打印结果
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {running_loss / len(train_loader):.4f} | '
              f'训练集Pearson: X={train_stats["x"]["pearson"].compute():.4f}, '
              f'Y={train_stats["y"]["pearson"].compute():.4f}, '
              f'Z={train_stats["z"]["pearson"].compute():.4f} | '
              f'训练集RMSE: X={train_stats["x"]["rmse"].compute():.4f}, '
              f'Y={train_stats["y"]["rmse"].compute():.4f}, '
              f'Z={train_stats["z"]["rmse"].compute():.4f}')
        print(f'Val Loss: {val_running_loss / len(val_loader):.4f} | '
              f'测试集Pearson: X={val_stats["x"]["pearson"].compute():.4f}, '
              f'Y={val_stats["y"]["pearson"].compute():.4f}, '
              f'Z={val_stats["z"]["pearson"].compute():.4f} | '
              f'测试集RMSE: X={val_stats["x"]["rmse"].compute():.4f}, '
              f'Y={val_stats["y"]["rmse"].compute():.4f}, '
              f'Z={val_stats["z"]["rmse"].compute():.4f}\n')

    # 计算总训练时间
    total_time = time.time() - total_start_time
    print(F'P{number}训练结束----------------------------------------------------------------------')
    print(f'总训练时间: {total_time:.2f}秒 ({total_time / 60:.2f}分钟)')

    # 训练结束后关闭writer
    writer.close()

    # 训练以及测试损失可视化
    plot_loss_curve(train_losses, val_losses)

    # 返回最终的性能指标
    final_metrics = {
        'train': {
            'loss': train_losses[-1],
            'pearson': {axis: train_stats[axis]['pearson'].compute() for axis in ['x', 'y', 'z']},
            'rmse': {axis: train_stats[axis]['rmse'].compute() for axis in ['x', 'y', 'z']}
        },
        'val': {
            'loss': val_losses[-1],
            'pearson': {axis: val_stats[axis]['pearson'].compute() for axis in ['x', 'y', 'z']},
            'rmse': {axis: val_stats[axis]['rmse'].compute() for axis in ['x', 'y', 'z']}
        }
    }

    return final_metrics


# 损失函数可视化
def plot_loss_curve(train_losses, val_losses):
    """绘制训练损失和验证损失随 epoch 变化的曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker=None, linestyle='-')
    plt.plot(val_losses, label='Validation Loss', marker=None, linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid()
    plt.show()


def plot_axis_comparison(model, loader, device, title='', max_points=10000, save_path=None):
    """
    绘制三个轴向（X, Y, Z）的真实值与预测值对比图
    参数:
        model: 训练好的模型
        loader: 数据加载器
        device: 计算设备 (如 'cuda' 或 'cpu')
        title: 图表标题
        max_points: 最大显示数据点数
        save_path: 图像保存路径，如果为None则不保存
    """
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for eeg_batch, kinematic_batch in loader:
            eeg_batch = eeg_batch.to(device)
            kinematic_batch = kinematic_batch.to(device)

            prediction = model(eeg_batch)
            true = kinematic_batch[:, 1:, :].cpu().numpy()  # 跳过第一个时间步
            pred = prediction.cpu().numpy()

            # 展平批次和时间步维度
            all_true.append(true.reshape(-1, 3))
            all_pred.append(pred.reshape(-1, 3))

            # 提前终止如果已收集足够数据点
            total_points = sum(arr.shape[0] for arr in all_true)
            if total_points >= max_points:
                break

    # 合并所有数据
    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    # 限制数据点为前max_points个
    if len(all_true) > max_points:
        all_true = all_true[:max_points]
        all_pred = all_pred[:max_points]

    # 设置全局字体大小
    plt.rcParams.update({
        'font.size': 14,  # 全局字体大小
        'axes.titlesize': 16,  # 标题大小
        'axes.labelsize': 14,  # 坐标轴标签大小
        'xtick.labelsize': 12,  # X轴刻度标签大小
        'ytick.labelsize': 12,  # Y轴刻度标签大小
        'legend.fontsize': 12  # 图例字体大小
    })

    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    axes = axes.flatten()

    # 绘制各轴数据
    for i, axis in enumerate(['X', 'Y', 'Z']):
        # 绘制真实值和预测值
        axes[i].plot(all_true[:, i], label='True', alpha=0.8, linewidth=1.2, color='blue')
        axes[i].plot(all_pred[:, i], label='Predicted', alpha=0.8, linewidth=1.2, color='red')

        # 设置标题和标签
        axes[i].set_title(f'{axis} Axis Comparison ({title})')
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Value')
        axes[i].legend()

        # 添加网格线提高可读性
        axes[i].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # 增加子图间距

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")

    plt.show()




# 获取三维轨迹数据（仅验证集）
def get_3d_trajectories(model, loader, device, max_points=10000):
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for eeg_batch, kinematic_batch in loader:
            eeg_batch = eeg_batch.to(device)
            kinematic_batch = kinematic_batch.to(device)

            prediction = model(eeg_batch)
            true_traj = kinematic_batch[:, 1:, :].cpu().numpy()
            pred_traj = prediction.cpu().numpy()

            all_true.append(true_traj.reshape(-1, 3))
            all_pred.append(pred_traj.reshape(-1, 3))

    true_trajectory = np.concatenate(all_true, axis=0)
    pred_trajectory = np.concatenate(all_pred, axis=0)

    # 限制数据点为前10,000个
    if len(true_trajectory) > max_points:
        true_trajectory = true_trajectory[:max_points]
        pred_trajectory = pred_trajectory[:max_points]

    return true_trajectory, pred_trajectory


# 三维空间轨迹可视化
def plot_3d_comparison(true_traj, pred_traj, title='3D Trajectory Comparison', save_path=None):
    """三维轨迹对比可视化"""
    plt.figure(figsize=(15, 10))
    ax = plt.axes(projection='3d')

    # 计算坐标范围
    combined = np.vstack([true_traj, pred_traj])
    max_range = (combined.max(axis=0) - combined.min(axis=0)).max() / 2.0

    # 绘制轨迹
    ax.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2],
            label='True Trajectory', color='blue', alpha=0.6, linewidth=0.8)
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2],
            label='Predicted Trajectory', color='red', alpha=0.6, linewidth=0.8)

    # 设置坐标轴
    mid_x = (true_traj[:, 0].max() + true_traj[:, 0].min()) * 0.5
    mid_y = (true_traj[:, 1].max() + true_traj[:, 1].min()) * 0.5
    mid_z = (true_traj[:, 2].max() + true_traj[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 设置标签和视角
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.view_init(elev=20, azim=45)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        print(f"三维轨迹图已保存至: {save_path}")

    plt.show()
# 修改后的主程序部分
if __name__ == "__main__":
    number=1  #被试者编号
    p1_folder = f'C:\\Users\\jsp\\Desktop\\way-eeg-gal\\P{number}'

    # 创建保存图像的目录
    save_dir = f'P{number}_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建保存目录: {save_dir}")
    all_sessions = []

    # 1. 按session独立加载数据

    for session_num in range(1, 10):
        eeg_path = os.path.join(p1_folder, f'eeg_HS_P{number}_S{session_num}.mat')  #EEG
        kinematic_path = os.path.join(p1_folder, f'Pro_HS_P{number}_S{session_num}.mat') #运动学数据

        if os.path.exists(eeg_path) and os.path.exists(kinematic_path):
            try:
                eeg_data = loadmat(eeg_path)['data']  # shape: (channels, time)
                kinematic_data = loadmat(kinematic_path)['downsampled_data']  # shape: (time, features)

                # 验证时间对齐
                assert eeg_data.shape[1] == kinematic_data.shape[0], \
                    f"Session {session_num}时间步不匹配: EEG({eeg_data.shape[1]}) vs Kinematic({kinematic_data.shape[0]})"

                all_sessions.append({
                    'eeg': eeg_data,
                    'kinematic': kinematic_data,
                    'session_id': session_num
                })
            except Exception as e:
                print(f"Session {session_num}加载失败: {str(e)}")
    print(
        f'P{number}开始训练---------------------------------------------------------------------------------------------')
    # 2. 按会话划分训练集和验证集（前8个会话训练，第9个验证）
    train_sessions = [s for s in all_sessions if s['session_id'] != 9]
    val_sessions = [s for s in all_sessions if s['session_id'] == 9]

    # 3. 会话独立的窗口处理
    def process_session(session_dict, window_size=500, stride=100):
        """为单个会话生成滑动窗口数据"""
        eeg = session_dict['eeg']
        kinematic = session_dict['kinematic']

        def sliding_window(data, window_size, stride, is_eeg=True):
            """
            通用滑动窗口函数
            参数：
                data: EEG数据形状为(channels, time)
                      或运动数据形状为(time, features)
                window_size: 窗口大小（单位：时间步）
                stride: 滑动步长（单位：时间步）
                is_eeg: 是否为EEG数据
            返回：
                windows: 滑动窗口数组
                        EEG形状为(num_windows, window_size, channels)
                        运动数据形状为(num_windows, window_size, features)
                        可以显著增加训练样本量（特别是小步长时），同时保持时间序列的连续性，适合需要时序建模的Transformer/LSTM等模型。
            """
            if is_eeg:
                # EEG数据形状 (channels, total_time)
                num_channels, total_time = data.shape
                num_windows = (total_time - window_size) // stride + 1
                windows = np.zeros((num_windows, window_size, num_channels))

                for i in range(num_windows):
                    start = i * stride
                    end = start + window_size
                    windows[i] = data[:, start:end].T  # 转置为(time, channels)

            else:
                # 运动数据形状 (total_time, features)
                total_time, num_features = data.shape
                num_windows = (total_time - window_size) // stride + 1
                windows = np.zeros((num_windows, window_size, num_features))

                for i in range(num_windows):
                    start = i * stride
                    end = start + window_size
                    windows[i] = data[start:end]

            return windows

        # EEG数据滑动窗口处理
        reshaped_eeg = sliding_window(eeg, window_size, stride, is_eeg=True)

        # 运动数据滑动窗口处理（保持与EEG相同的窗口参数）
        reshaped_kinematic = sliding_window(kinematic, window_size, stride, is_eeg=False)

        # 时间对齐验证
        assert reshaped_eeg.shape[0] == reshaped_kinematic.shape[0], \
            f"Session {session_dict['session_id']}窗口数量不匹配，EEG: {reshaped_eeg.shape[0]} vs Motion: {reshaped_kinematic.shape[0]}"

        # 验证窗口对齐
        sample_window_idx = 0  # 可以改为其他索引测试
        assert reshaped_eeg[sample_window_idx].shape[0] == window_size, "EEG窗口尺寸错误"
        assert reshaped_kinematic[sample_window_idx].shape[0] == window_size, "运动数据窗口尺寸错误"

        return reshaped_eeg, reshaped_kinematic


    # 处理所有训练会话
    train_eeg = []
    train_kinematic = []
    for session in train_sessions:
        eeg, kinematic = process_session(session)
        train_eeg.append(eeg)
        train_kinematic.append(kinematic)


    # 处理验证会话
    val_eeg, val_kinematic = process_session(val_sessions[0])

    # 4. 合并会话数据（保持会话独立性）
    train_eeg = np.concatenate(train_eeg, axis=0)
    train_kinematic = np.concatenate(train_kinematic, axis=0)
    print('train_eeg的形状为：', train_eeg.shape)
    print('val_eeg的形状为：', val_eeg.shape)

    # 训练集启用增强
    train_dataset = EEGDataset(
        train_eeg,
        train_kinematic,
        enable_augment=True,  # 开启增强
        noise_level=0.01,  # 噪声强度1%
        warp_prob=0.00,  # 00%概率时间扭曲
        drop_prob=0.00  # 00%通道丢弃概率
    )

    # 验证集关闭增强
    val_dataset = EEGDataset(
        val_eeg,
        val_kinematic,
        enable_augment=False  # 验证集不应用增强
    )

    # 数据加载器保持不变
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    d_model = 36
    model = EEGEncoder(
        d_model=d_model,
        n_head=6,
        max_len=500,
        ffn_hidden=64,
        n_layers=8,
        drop_prob=0.25,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # 训练模型并获取最终指标
    final_metrics = train_model(model, train_loader, val_loader, num_epochs=300)

    # 打印最终性能指标
    print("\n=== 最终性能指标 ===")
    print("训练集:")
    print(f"  损失: {final_metrics['train']['loss']:.4f}")
    print(
        f"  Pearson - X: {final_metrics['train']['pearson']['x']:.4f}, Y: {final_metrics['train']['pearson']['y']:.4f}, Z: {final_metrics['train']['pearson']['z']:.4f}")
    print(
        f"  RMSE - X: {final_metrics['train']['rmse']['x']:.4f}, Y: {final_metrics['train']['rmse']['y']:.4f}, Z: {final_metrics['train']['rmse']['z']:.4f}")

    print("验证集:")
    print(f"  损失: {final_metrics['val']['loss']:.4f}")
    print(
        f"  Pearson - X: {final_metrics['val']['pearson']['x']:.4f}, Y: {final_metrics['val']['pearson']['y']:.4f}, Z: {final_metrics['val']['pearson']['z']:.4f}")
    print(
        f"  RMSE - X: {final_metrics['val']['rmse']['x']:.4f}, Y: {final_metrics['val']['rmse']['y']:.4f}, Z: {final_metrics['val']['rmse']['z']:.4f}")

    # 保存模型
    # model_save_path = f'EEG_Transformer_P{number}.pth'
    # torch.save(model.state_dict(), model_save_path)
    # print(f'模型已保存至 {model_save_path}')

    # 可视化训练集和验证集结果
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 保存验证集的轴向对比图
    axis_plot_path = os.path.join(save_dir, f'P{number}_validation_axis_comparison.png')
    plot_axis_comparison(model, val_loader, device, title='Validation Set', max_points=10000, save_path=axis_plot_path)

    # 获取三维轨迹数据
    train_true, train_pred = get_3d_trajectories(model, train_loader, device)
    val_true, val_pred = get_3d_trajectories(model, val_loader, device)

    # 保存验证集三维轨迹图
    val_3d_plot_path = os.path.join(save_dir, f'P{number}_validation_3d_trajectory.png')
    plot_3d_comparison(val_true, val_pred, title='Validation Set 3D Trajectory', save_path=val_3d_plot_path)

    # 可选：保存前1000个点的对比图
    short_3d_plot_path = os.path.join(save_dir, f'P{number}_validation_first_1000_points.png')
    plot_3d_comparison(val_true[:1000], val_pred[:1000], title='First 1000 Points Comparison',
                       save_path=short_3d_plot_path)
