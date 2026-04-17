"""
EEG 数据增强效果可视化脚本
使用 selfEEG 库演示时间平移增强效果

注意：selfEEG 库中没有名为 RandomTimeShift 的类，
      时间平移增强是通过 `shift_horizontal` 函数实现的。
      可以配合 `DynamicSingleAug` 类实现随机参数的时间平移。

运行要求:
- selfEEG 库已安装
- matplotlib 已安装 (用于可视化)
- numpy < 2.0 (如果遇到 numpy 版本兼容性问题)

如果遇到 numpy 版本错误，请运行:
    pip install "numpy<2"
"""

import torch
import numpy as np

from selfeeg import augmentation as aug

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib 未安装，无法生成可视化图像")
    print("请运行: pip install matplotlib")
except Exception as e:
    MATPLOTLIB_AVAILABLE = False
    print(f"警告: matplotlib 导入失败: {e}")
    print("如果是 numpy 版本兼容性问题，请尝试: pip install \"numpy<2\"")


def generate_simulated_eeg(n_channels: int = 2, duration: float = 10.0, fs: float = 128.0) -> torch.Tensor:
    """
    生成模拟 EEG 信号
    
    该函数生成包含多个正弦波分量和高斯噪声的模拟 EEG 信号。
    信号包含典型的脑电频率分量：delta (2Hz), theta (5Hz), alpha (10Hz), beta (15Hz), gamma (20Hz)
    
    参数:
        n_channels: 通道数 (默认: 2)
        duration: 持续时间（秒） (默认: 10.0)
        fs: 采样率（Hz） (默认: 128.0)
    
    返回:
        torch.Tensor: 形状为 (n_channels, n_samples) 的 EEG 信号
    """
    n_samples = int(duration * fs)
    t = torch.linspace(0, duration, n_samples)
    
    eeg_data = torch.zeros((n_channels, n_samples))
    
    freq_components = [2, 5, 10, 15, 20]
    amplitudes = [0.8, 0.5, 0.3, 0.2, 0.1]
    
    for chan in range(n_channels):
        for freq, amp in zip(freq_components, amplitudes):
            phase_offset = torch.rand(1) * 2 * np.pi
            eeg_data[chan, :] += amp * torch.sin(2 * np.pi * freq * t + phase_offset)
        
        eeg_data[chan, :] += 0.1 * torch.randn(n_samples)
    
    return eeg_data


def create_visualization(
    original_eeg: torch.Tensor,
    augmented_eeg_forward: torch.Tensor,
    augmented_eeg_backward: torch.Tensor,
    augmented_eeg_random: torch.Tensor,
    n_channels: int = 2,
    duration: float = 10.0,
    fs: float = 128.0,
    shift_time: float = 2.0
) -> None:
    """
    创建可视化对比图
    
    参数:
        original_eeg: 原始 EEG 信号
        augmented_eeg_forward: 向前平移后的 EEG 信号
        augmented_eeg_backward: 向后平移后的 EEG 信号
        augmented_eeg_random: 随机参数平移后的 EEG 信号
        n_channels: 通道数
        duration: 信号时长（秒）
        fs: 采样率（Hz）
        shift_time: 平移时间（秒）
    """
    if not MATPLOTLIB_AVAILABLE:
        print("跳过可视化步骤，因为 matplotlib 不可用")
        return
    
    print("\n[3/4] 生成可视化对比图...")
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    
    t = np.linspace(0, duration, int(duration * fs))
    channel_names = ['Channel 1', 'Channel 2']
    colors = ['#1f77b4', '#ff7f0e']
    
    ax = axes[0]
    for chan in range(n_channels):
        ax.plot(t, original_eeg[chan, :].numpy(), 
                label=f'{channel_names[chan]}', 
                color=colors[chan], 
                linewidth=1.5,
                alpha=0.8)
    ax.set_title('原始 EEG 信号', fontsize=14, fontweight='bold')
    ax.set_xlabel('时间 (秒)', fontsize=12)
    ax.set_ylabel('幅值', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration)
    
    ax = axes[1]
    for chan in range(n_channels):
        ax.plot(t, augmented_eeg_forward[chan, :].numpy(), 
                label=f'{channel_names[chan]} (向前平移 {shift_time}s)', 
                color=colors[chan], 
                linewidth=1.5,
                alpha=0.8)
    ax.set_title(f'时间平移增强效果 (向前平移 {shift_time} 秒)', fontsize=14, fontweight='bold')
    ax.set_xlabel('时间 (秒)', fontsize=12)
    ax.set_ylabel('幅值', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration)
    ax.axvline(x=shift_time, color='red', linestyle='--', alpha=0.5, label=f'平移开始点 ({shift_time}s)')
    ax.legend(loc='upper right')
    
    ax = axes[2]
    for chan in range(n_channels):
        ax.plot(t, augmented_eeg_backward[chan, :].numpy(), 
                label=f'{channel_names[chan]} (向后平移 {shift_time}s)', 
                color=colors[chan], 
                linewidth=1.5,
                alpha=0.8)
    ax.set_title(f'时间平移增强效果 (向后平移 {shift_time} 秒)', fontsize=14, fontweight='bold')
    ax.set_xlabel('时间 (秒)', fontsize=12)
    ax.set_ylabel('幅值', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration)
    ax.axvline(x=duration - shift_time, color='red', linestyle='--', alpha=0.5, label=f'平移结束点 ({duration-shift_time}s)')
    ax.legend(loc='upper right')
    
    ax = axes[3]
    for chan in range(n_channels):
        ax.plot(t, augmented_eeg_random[chan, :].numpy(), 
                label=f'{channel_names[chan]} (随机参数平移)', 
                color=colors[chan], 
                linewidth=1.5,
                alpha=0.8)
    ax.set_title('时间平移增强效果 (随机参数)', fontsize=14, fontweight='bold')
    ax.set_xlabel('时间 (秒)', fontsize=12)
    ax.set_ylabel('幅值', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration)
    
    plt.tight_layout()
    
    plt.savefig('eeg_augmentation_comparison.png', dpi=150, bbox_inches='tight')
    print("  图像已保存为: eeg_augmentation_comparison.png")
    
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 10))
    
    ax = axes2[0]
    for chan in range(n_channels):
        ax.plot(t, original_eeg[chan, :].numpy(), 
                label=f'原始 - {channel_names[chan]}', 
                color=colors[chan], 
                linewidth=1.5,
                alpha=0.6)
        ax.plot(t, augmented_eeg_forward[chan, :].numpy(), 
                label=f'增强后 - {channel_names[chan]}', 
                color=colors[chan], 
                linewidth=1.5,
                linestyle='--',
                alpha=0.9)
    ax.set_title('原始信号 vs 向前平移增强信号 (叠图对比)', fontsize=14, fontweight='bold')
    ax.set_xlabel('时间 (秒)', fontsize=12)
    ax.set_ylabel('幅值', fontsize=12)
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration)
    ax.axvline(x=shift_time, color='red', linestyle='--', alpha=0.5)
    ax.annotate(f'信号向右平移 {shift_time}s\n前 {shift_time}s 补零', 
                xy=(shift_time, 0), xytext=(shift_time + 0.5, -1.5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    ax = axes2[1]
    for chan in range(n_channels):
        ax.plot(t, original_eeg[chan, :].numpy(), 
                label=f'原始 - {channel_names[chan]}', 
                color=colors[chan], 
                linewidth=1.5,
                alpha=0.6)
        ax.plot(t, augmented_eeg_backward[chan, :].numpy(), 
                label=f'增强后 - {channel_names[chan]}', 
                color=colors[chan], 
                linewidth=1.5,
                linestyle='--',
                alpha=0.9)
    ax.set_title('原始信号 vs 向后平移增强信号 (叠图对比)', fontsize=14, fontweight='bold')
    ax.set_xlabel('时间 (秒)', fontsize=12)
    ax.set_ylabel('幅值', fontsize=12)
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration)
    ax.axvline(x=duration - shift_time, color='red', linestyle='--', alpha=0.5)
    ax.annotate(f'信号向左平移 {shift_time}s\n后 {shift_time}s 补零', 
                xy=(duration - shift_time, 0), xytext=(duration - shift_time - 3, -1.5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    plt.tight_layout()
    
    plt.savefig('eeg_augmentation_overlay.png', dpi=150, bbox_inches='tight')
    print("  图像已保存为: eeg_augmentation_overlay.png")
    
    print("\n[4/4] 显示图像...")
    print("注意: 如果在非图形环境中运行，plt.show() 可能不会显示图像")
    print("      但图像已保存为 PNG 文件，可以直接查看。")
    plt.show()


def main():
    """
    主函数：生成 EEG 信号，应用时间平移增强，并可视化对比
    """
    
    print("=" * 70)
    print("EEG 数据增强效果可视化 - selfEEG 演示脚本")
    print("=" * 70)
    
    print("""
重要说明:
    selfEEG 库中没有名为 RandomTimeShift 的类。
    时间平移增强是通过 `shift_horizontal` 函数实现的。
    
    可以通过以下方式使用:
    1. 直接调用 `aug.shift_horizontal()` 函数（固定参数）
    2. 使用 `DynamicSingleAug` 类包装（随机参数）
    
    本脚本将演示这两种用法。
    """)
    
    # 参数设置
    N_CHANNELS = 2
    DURATION = 10.0
    FS = 128.0
    SHIFT_TIME = 2.0
    
    print(f"参数设置:")
    print(f"  - 通道数: {N_CHANNELS}")
    print(f"  - 信号时长: {DURATION} 秒")
    print(f"  - 采样率: {FS} Hz")
    print(f"  - 时间平移量: {SHIFT_TIME} 秒")
    print(f"  - 总样本数: {int(DURATION * FS)}")
    
    # 1. 生成模拟 EEG 信号
    print("\n[1/4] 生成模拟 EEG 信号...")
    original_eeg = generate_simulated_eeg(N_CHANNELS, DURATION, FS)
    print(f"  原始信号形状: {original_eeg.shape} (通道数 x 样本数)")
    print(f"  信号范围: [{original_eeg.min():.3f}, {original_eeg.max():.3f}]")
    
    # 2. 应用时间平移增强
    print("\n[2/4] 应用时间平移增强...")
    
    print(f"\n方法 1: 直接调用 shift_horizontal 函数（固定参数）")
    print(f"  向前平移 (forward=True): 信号向右移动，前面补零")
    augmented_eeg_forward = aug.shift_horizontal(
        original_eeg, 
        shift_time=SHIFT_TIME, 
        Fs=FS, 
        forward=True
    )
    print(f"  向前平移后的信号形状: {augmented_eeg_forward.shape}")
    
    print(f"\n  向后平移 (forward=False): 信号向左移动，后面补零")
    augmented_eeg_backward = aug.shift_horizontal(
        original_eeg, 
        shift_time=SHIFT_TIME, 
        Fs=FS, 
        forward=False
    )
    print(f"  向后平移后的信号形状: {augmented_eeg_backward.shape}")
    
    print(f"\n方法 2: 使用 DynamicSingleAug 实现随机时间平移")
    print(f"  这相当于 RandomTimeShift 的效果 - 每次调用参数随机")
    print(f"  配置: 随机方向 (向前/向后), 随机平移量 (0.5-3.0秒)")
    
    random_time_shift = aug.DynamicSingleAug(
        aug.shift_horizontal,
        discrete_arg={
            'Fs': FS,
            'forward': [True, False]
        },
        range_arg={
            'shift_time': [0.5, 3.0]
        },
        range_type={'shift_time': False}
    )
    
    augmented_eeg_random = random_time_shift(original_eeg)
    print(f"  随机平移后的信号形状: {augmented_eeg_random.shape}")
    
    # 3. 验证增强效果
    print("\n验证增强效果:")
    shift_samples = int(SHIFT_TIME * FS)
    print(f"  平移样本数: {shift_samples}")
    
    print(f"\n  向前平移验证:")
    print(f"    前 {shift_samples} 个样本是否全为零: {torch.all(augmented_eeg_forward[:, :shift_samples] == 0)}")
    print(f"    第 {shift_samples} 个样本是否与原始第 0 个样本相等: {torch.allclose(augmented_eeg_forward[:, shift_samples], original_eeg[:, 0])}")
    
    print(f"\n  向后平移验证:")
    print(f"    后 {shift_samples} 个样本是否全为零: {torch.all(augmented_eeg_backward[:, -shift_samples:] == 0)}")
    print(f"    第 0 个样本是否与原始第 {shift_samples} 个样本相等: {torch.allclose(augmented_eeg_backward[:, 0], original_eeg[:, shift_samples])}")
    
    # 4. 可视化
    create_visualization(
        original_eeg,
        augmented_eeg_forward,
        augmented_eeg_backward,
        augmented_eeg_random,
        N_CHANNELS,
        DURATION,
        FS,
        SHIFT_TIME
    )
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)
    
    print("""
总结:
    本脚本演示了如何使用 selfEEG 库进行时间平移增强:
    
    1. 核心函数: `aug.shift_horizontal()`
       - 参数:
         * x: 输入信号 (形状: 通道数 x 样本数 或 批次 x 通道数 x 样本数)
         * shift_time: 平移时间（秒）
         * Fs: 采样率（Hz）
         * forward: True=向前平移（右移，前面补零）, False=向后平移（左移，后面补零）
         * random_shift: 是否使用随机平移量（shift_time 作为上限）
         * batch_equal: 是否对批次中所有信号应用相同的平移
    
    2. 随机参数增强: 使用 `DynamicSingleAug` 类
       - 可以配置离散参数选项 (discrete_arg)
       - 可以配置连续参数范围 (range_arg)
       - 每次调用时随机选择参数值
    
    3. 可视化效果:
       - 向前平移: 信号整体向右移动，前 N 秒为零
       - 向后平移: 信号整体向左移动，后 N 秒为零
    
生成的文件:
    - eeg_augmentation_comparison.png: 四张子图对比
    - eeg_augmentation_overlay.png: 叠图对比（更直观看到平移效果）

如果遇到问题:
    - numpy 版本错误: 运行 `pip install "numpy<2"`
    - matplotlib 未安装: 运行 `pip install matplotlib`
    - selfEEG 未安装: 运行 `pip install selfeeg` 或在仓库根目录运行 `pip install -e .`
    """)


if __name__ == "__main__":
    main()
