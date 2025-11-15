"""
创建演示所需的样本数据和模型权重
"""

import torch
import numpy as np
import pickle
import os


def create_sample_data():
    """创建模拟的样本数据"""
    # 设置随机种子以确保可复现性
    np.random.seed(42)
    torch.manual_seed(42)

    # 创建模拟EEG数据 (1个样本, 500时间点, 18通道)
    # 模拟真实的EEG信号特征：低频振荡+噪声
    t = np.linspace(0, 5, 500)  # 5秒数据

    eeg_data = np.zeros((1, 500, 18))
    for ch in range(18):
        # 每个通道有不同的振荡模式
        freq = 2 + ch * 0.5  # 2-11Hz
        phase = ch * 0.3
        amplitude = 0.5 + 0.2 * np.random.randn()

        signal = amplitude * np.sin(2 * np.pi * freq * t + phase)
        noise = 0.1 * np.random.randn(500)

        eeg_data[0, :, ch] = signal + noise

    # 创建模拟轨迹数据 (1个样本, 500时间点, 3维度)
    # 模拟平滑的手部运动轨迹
    trajectory = np.zeros((1, 500, 3))

    # X轴: 缓慢的线性运动 + 小振荡
    trajectory[0, :, 0] = 0.5 * t + 0.1 * np.sin(2 * np.pi * 0.5 * t)

    # Y轴: 正弦波运动
    trajectory[0, :, 1] = 0.3 * np.sin(2 * np.pi * 0.8 * t)

    # Z轴: 更复杂的运动模式
    trajectory[0, :, 2] = 0.2 * np.sin(2 * np.pi * 0.3 * t) + 0.1 * np.sin(2 * np.pi * 1.2 * t)

    # 添加少量噪声使轨迹更真实
    trajectory += 0.02 * np.random.randn(1, 500, 3)

    return {
        'eeg': eeg_data.astype(np.float32),
        'trajectory': trajectory.astype(np.float32)
    }


def create_sample_model():
    """创建并保存一个示例模型"""
    from demo import MinimalM3TAttention  # 从主演示脚本导入

    model = MinimalM3TAttention()

    # 使用预定义的权重（在实际应用中，这里应该加载真实训练好的权重）
    # 这里我们只是保存随机初始化的权重作为示例
    return model


def main():
    print("创建演示资源...")

    # 创建目录
    os.makedirs('demo_assets', exist_ok=True)

    # 创建样本数据
    print("生成样本数据...")
    sample_data = create_sample_data()

    with open('demo_assets/sample_data.pkl', 'wb') as f:
        pickle.dump(sample_data, f)

    # 创建示例模型
    print("创建示例模型...")
    model = create_sample_model()
    torch.save(model.state_dict(), 'demo_assets/sample_model.pth')

    print("演示资源创建完成!")
    print("文件已保存至 'demo_assets/' 目录")

    # 验证文件
    print("\n验证生成的文件:")
    print(f"样本数据: demo_assets/sample_data.pkl ({os.path.getsize('demo_assets/sample_data.pkl')} bytes)")
    print(f"模型权重: demo_assets/sample_model.pth ({os.path.getsize('demo_assets/sample_model.pth')} bytes)")


if __name__ == "__main__":
    main()