"""
M3T-Attention Minimal Demo
运行此脚本可快速验证模型的核心功能
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# 简化版的模型定义（只包含核心组件）
class MinimalM3TAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 简化的时空卷积块
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv1d(18, 36, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(36),
            torch.nn.GELU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(36, 36, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(36),
            torch.nn.GELU(),
        )

        # Transformer编码器
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=36,
            nhead=6,
            dim_feedforward=64,
            dropout=0.25,
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)

        # 回归头
        self.regression_head = torch.nn.Sequential(
            torch.nn.Linear(36, 18),
            torch.nn.GELU(),
            torch.nn.Linear(18, 3)
        )

        # 上采样层
        self.upsample = torch.nn.ConvTranspose1d(3, 3, kernel_size=4, stride=4)

    def forward(self, x):
        # 输入形状: [batch, 500, 18]
        x = x.permute(0, 2, 1)  # [batch, 18, 500]
        x = self.conv_block(x)  # [batch, 36, 250]
        x = x.permute(0, 2, 1)  # [batch, 250, 36]

        x = self.transformer_encoder(x)  # [batch, 250, 36]
        x = self.regression_head(x)  # [batch, 250, 3]

        x = x.permute(0, 2, 1)  # [batch, 3, 250]
        x = self.upsample(x)  # [batch, 3, 1000]
        x = x.permute(0, 2, 1)  # [batch, 1000, 3]

        return x[:, :500, :]  # 截取前500个时间点


def main():
    print("=" * 60)
    print("M3T-Attention 最小化演示")
    print("=" * 60)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载样本数据
    print("\n1. 加载样本数据...")
    try:
        with open('demo_assets/sample_data.pkl', 'rb') as f:
            sample_data = pickle.load(f)

        sample_eeg = sample_data['eeg']  # 形状: [1, 500, 18]
        sample_trajectory = sample_data['trajectory']  # 形状: [1, 500, 3]

        print(f"EEG数据形状: {sample_eeg.shape}")
        print(f"轨迹数据形状: {sample_trajectory.shape}")

    except Exception as e:
        print(f"错误: 无法加载样本数据 - {e}")
        print("请确保 'demo_assets/sample_data.pkl' 文件存在")
        return

    # 加载模型
    print("\n2. 加载预训练模型...")
    try:
        model = MinimalM3TAttention().to(device)
        model.load_state_dict(torch.load('demo_assets/sample_model.pth', map_location=device, weights_only=True))
        model.eval()
        print("模型加载成功!")

    except Exception as e:
        print(f"错误: 无法加载模型 - {e}")
        print("请确保 'demo_assets/sample_model.pth' 文件存在")
        return

    # 模型推理
    print("\n3. 运行模型推理...")
    with torch.no_grad():
        input_tensor = torch.tensor(sample_eeg, dtype=torch.float32).to(device)
        predicted_trajectory = model(input_tensor)

        # 转换为numpy
        pred_np = predicted_trajectory.cpu().numpy().squeeze()  # [500, 3]
        true_np = sample_trajectory.squeeze()  # [500, 3]

    # 计算性能指标
    print("\n4. 计算性能指标...")
    pcc_results = []
    rmse_results = []

    for i, axis in enumerate(['X', 'Y', 'Z']):
        pcc = pearsonr(pred_np[:, i], true_np[:, i])[0]
        rmse = np.sqrt(np.mean((pred_np[:, i] - true_np[:, i]) ** 2))

        pcc_results.append(pcc)
        rmse_results.append(rmse)

        print(f"   {axis}轴 - PCC: {pcc:.4f}, RMSE: {rmse:.4f}")

    avg_pcc = np.mean(pcc_results)
    avg_rmse = np.mean(rmse_results)
    print(f"\n   平均PCC: {avg_pcc:.4f}, 平均RMSE: {avg_rmse:.4f}")

    # 可视化结果
    print("\n5. 生成可视化结果...")
    plt.figure(figsize=(15, 10))

    # 绘制三个轴向的对比
    time_steps = np.arange(500)
    axes = ['X', 'Y', 'Z']
    colors = ['red', 'green', 'blue']

    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(time_steps, true_np[:, i], 'b-', linewidth=1.5, label='真实轨迹', alpha=0.8)
        plt.plot(time_steps, pred_np[:, i], 'r--', linewidth=1.5, label='预测轨迹', alpha=0.8)
        plt.title(f'{axes[i]}轴轨迹对比 (PCC: {pcc_results[i]:.4f})')
        plt.xlabel('时间步')
        plt.ylabel('位置')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_output.png', dpi=300, bbox_inches='tight')
    print("可视化结果已保存至 'demo_output.png'")

    # 3D轨迹可视化（可选）
    try:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 只绘制前100个点以便清晰查看
        points_to_plot = 100
        ax.plot(true_np[:points_to_plot, 0], true_np[:points_to_plot, 1], true_np[:points_to_plot, 2],
                'b-', label='真实轨迹', linewidth=2, alpha=0.7)
        ax.plot(pred_np[:points_to_plot, 0], pred_np[:points_to_plot, 1], pred_np[:points_to_plot, 2],
                'r--', label='预测轨迹', linewidth=2, alpha=0.7)

        ax.set_xlabel('X轴')
        ax.set_ylabel('Y轴')
        ax.set_zlabel('Z轴')
        ax.set_title('3D轨迹对比 (前100个时间点)')
        ax.legend()

        plt.savefig('demo_3d_output.png', dpi=300, bbox_inches='tight')
        print("3D可视化结果已保存至 'demo_3d_output.png'")

    except ImportError:
        print("注意: 3D可视化需要matplotlib的3D支持")

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()