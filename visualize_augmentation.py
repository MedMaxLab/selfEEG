"""
EEG 数据增强效果可视化脚本
使用 selfEEG 库演示多种数据增强效果

增强方法包括：
1. 原始信号
2. 时间平移 (Time Shift) - 信号在时间轴上平移
3. 振幅缩放 (Amplitude Scaling) - 信号幅值缩放
4. 高斯噪声 (Gaussian Noise) - 添加随机噪声

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
    from matplotlib import font_manager
    
    MATPLOTLIB_AVAILABLE = True
    
    # 设置中文字体 - 尝试多种常见中文字体
    font_names = [
        'SimHei',           # 黑体
        'Microsoft YaHei',  # 微软雅黑
        'KaiTi',            # 楷体
        'FangSong',         # 仿宋
        'Arial Unicode MS',
        'PingFang SC',      # Mac
        'Heiti TC',         # Mac 繁体
        'Noto Sans CJK SC', # Noto 字体
    ]
    
    font_found = False
    for font_name in font_names:
        try:
            font_manager.fontManager.addfont(font_name)
            plt.rcParams['font.family'] = font_name
            font_found = True
            print(f"已设置字体: {font_name}")
            break
        except:
            continue
    
    if not font_found:
        # 尝试查找系统中的中文字体
        system_fonts = [f.name for f in font_manager.fontManager.ttflist]
        chinese_fonts = [f for f in system_fonts if any(keyword in f for keyword in ['Hei', 'Kai', 'Song', 'Fang', 'YaHei', 'Noto', 'Ping', 'Sim'])]
        if chinese_fonts:
            plt.rcParams['font.family'] = chinese_fonts[0]
            print(f"已设置字体: {chinese_fonts[0]}")
        else:
            print("警告: 未找到中文字体，中文可能显示为方块")
            print("建议安装中文字体后重试，或使用英文标签")
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    
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
    augmented_eeg_timeshift: torch.Tensor,
    augmented_eeg_scaling: torch.Tensor,
    augmented_eeg_noise: torch.Tensor,
    n_channels: int = 2,
    duration: float = 10.0,
    fs: float = 128.0,
    shift_time: float = 2.0,
    scaling_factor: float = 1.5,
    noise_std: float = 0.3
) -> None:
    """
    创建可视化对比图
    
    参数:
        original_eeg: 原始 EEG 信号
        augmented_eeg_timeshift: 时间平移后的 EEG 信号
        augmented_eeg_scaling: 振幅缩放后的 EEG 信号
        augmented_eeg_noise: 添加高斯噪声后的 EEG 信号
        n_channels: 通道数
        duration: 信号时长（秒）
        fs: 采样率（Hz）
        shift_time: 平移时间（秒）
        scaling_factor: 缩放因子
        noise_std: 噪声标准差
    """
    if not MATPLOTLIB_AVAILABLE:
        print("跳过可视化步骤，因为 matplotlib 不可用")
        return
    
    print("\n[3/4] 生成可视化对比图...")
    
    t = np.linspace(0, duration, int(duration * fs))
    channel_names = ['通道 1', '通道 2']
    colors = ['#1f77b4', '#ff7f0e']
    
    # ==================== 图1: 4张子图对比 ====================
    fig1 = plt.figure(figsize=(14, 18))
    
    gs = fig1.add_gridspec(4, 1, hspace=0.5)
    
    # 子图1: 原始信号
    ax1 = fig1.add_subplot(gs[0])
    for chan in range(n_channels):
        ax1.plot(t, original_eeg[chan, :].numpy(), 
                label=f'{channel_names[chan]}', 
                color=colors[chan], 
                linewidth=1.5,
                alpha=0.8)
    ax1.set_title(
        f'【图1】原始 EEG 信号\n'
        f'({n_channels} 通道, {duration} 秒, {fs} Hz 采样率)', 
        fontsize=14, 
        fontweight='bold',
        pad=15
    )
    ax1.set_xlabel('时间 (秒)', fontsize=12, labelpad=10)
    ax1.set_ylabel('幅值', fontsize=12, labelpad=10)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, duration)
    
    # 子图2: 时间平移增强
    ax2 = fig1.add_subplot(gs[1])
    for chan in range(n_channels):
        ax2.plot(t, augmented_eeg_timeshift[chan, :].numpy(), 
                label=f'{channel_names[chan]} (向前平移 {shift_time}秒)', 
                color=colors[chan], 
                linewidth=1.5,
                alpha=0.8)
    ax2.set_title(
        f'【图2】时间平移增强 (Time Shift)\n'
        f'信号向右平移 {shift_time} 秒，前 {shift_time} 秒补零', 
        fontsize=14, 
        fontweight='bold',
        pad=15
    )
    ax2.set_xlabel('时间 (秒)', fontsize=12, labelpad=10)
    ax2.set_ylabel('幅值', fontsize=12, labelpad=10)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, duration)
    ax2.axvline(x=shift_time, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax2.annotate(
        f'平移开始点\n({shift_time}秒)', 
        xy=(shift_time, 0), 
        xytext=(shift_time + 0.3, -1.2),
        arrowprops=dict(arrowstyle='->', color='red', linewidth=1.5),
        fontsize=11, 
        color='red',
        fontweight='bold'
    )
    
    # 子图3: 振幅缩放增强
    ax3 = fig1.add_subplot(gs[2])
    for chan in range(n_channels):
        ax3.plot(t, original_eeg[chan, :].numpy(), 
                label=f'{channel_names[chan]} (原始)', 
                color=colors[chan], 
                linewidth=1.2,
                alpha=0.4)
        ax3.plot(t, augmented_eeg_scaling[chan, :].numpy(), 
                label=f'{channel_names[chan]} (缩放 {scaling_factor}x)', 
                color=colors[chan], 
                linewidth=1.8,
                alpha=0.9)
    ax3.set_title(
        f'【图3】振幅缩放增强 (Amplitude Scaling)\n'
        f'信号幅值放大 {scaling_factor} 倍 (原始信号以淡色显示)', 
        fontsize=14, 
        fontweight='bold',
        pad=15
    )
    ax3.set_xlabel('时间 (秒)', fontsize=12, labelpad=10)
    ax3.set_ylabel('幅值', fontsize=12, labelpad=10)
    ax3.legend(loc='upper right', fontsize=10, ncol=2)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(0, duration)
    
    # 子图4: 高斯噪声增强
    ax4 = fig1.add_subplot(gs[3])
    for chan in range(n_channels):
        ax4.plot(t, original_eeg[chan, :].numpy(), 
                label=f'{channel_names[chan]} (原始)', 
                color=colors[chan], 
                linewidth=1.2,
                alpha=0.4)
        ax4.plot(t, augmented_eeg_noise[chan, :].numpy(), 
                label=f'{channel_names[chan]} (加噪声)', 
                color=colors[chan], 
                linewidth=1.5,
                alpha=0.9)
    ax4.set_title(
        f'【图4】高斯噪声增强 (Gaussian Noise)\n'
        f'添加标准差为 {noise_std} 的高斯噪声 (原始信号以淡色显示)', 
        fontsize=14, 
        fontweight='bold',
        pad=15
    )
    ax4.set_xlabel('时间 (秒)', fontsize=12, labelpad=10)
    ax4.set_ylabel('幅值', fontsize=12, labelpad=10)
    ax4.legend(loc='upper right', fontsize=10, ncol=2)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xlim(0, duration)
    
    plt.subplots_adjust(top=0.92, bottom=0.06, left=0.08, right=0.95)
    
    plt.savefig('eeg_augmentation_comparison.png', dpi=150, bbox_inches='tight')
    print("  图像已保存为: eeg_augmentation_comparison.png")
    
    # ==================== 图2: 叠图对比 ====================
    fig2 = plt.figure(figsize=(14, 12))
    
    gs2 = fig2.add_gridspec(2, 1, hspace=0.4)
    
    # 子图1: 时间平移叠图对比
    ax2_1 = fig2.add_subplot(gs2[0])
    for chan in range(n_channels):
        ax2_1.plot(t, original_eeg[chan, :].numpy(), 
                label=f'{channel_names[chan]} - 原始信号', 
                color=colors[chan], 
                linewidth=2.0,
                alpha=0.6,
                linestyle='-')
        ax2_1.plot(t, augmented_eeg_timeshift[chan, :].numpy(), 
                label=f'{channel_names[chan]} - 平移后信号', 
                color=colors[chan], 
                linewidth=2.0,
                alpha=0.9,
                linestyle='--')
    ax2_1.set_title(
        f'原始信号 vs 时间平移增强 (叠图对比)\n'
        f'虚线 = 增强后信号 (向右平移 {shift_time} 秒)', 
        fontsize=14, 
        fontweight='bold',
        pad=15
    )
    ax2_1.set_xlabel('时间 (秒)', fontsize=12, labelpad=10)
    ax2_1.set_ylabel('幅值', fontsize=12, labelpad=10)
    ax2_1.legend(loc='upper right', fontsize=10, ncol=2)
    ax2_1.grid(True, alpha=0.3, linestyle='--')
    ax2_1.set_xlim(0, duration)
    ax2_1.axvline(x=shift_time, color='green', linestyle='-.', alpha=0.8, linewidth=2)
    ax2_1.annotate(
        f'信号向右平移 {shift_time}秒\n'
        f'前 {shift_time}秒补零，信号整体右移', 
        xy=(shift_time, 0), 
        xytext=(shift_time + 0.5, -1.5),
        arrowprops=dict(arrowstyle='->', color='green', linewidth=1.5),
        fontsize=11, 
        color='darkgreen',
        fontweight='bold'
    )
    
    # 子图2: 振幅缩放 vs 高斯噪声 对比
    ax2_2 = fig2.add_subplot(gs2[1])
    for chan in range(n_channels):
        ax2_2.plot(t, original_eeg[chan, :].numpy(), 
                label=f'{channel_names[chan]} - 原始', 
                color=colors[chan], 
                linewidth=2.0,
                alpha=0.4)
        ax2_2.plot(t, augmented_eeg_scaling[chan, :].numpy(), 
                label=f'{channel_names[chan]} - 缩放 {scaling_factor}x', 
                color=colors[chan], 
                linewidth=1.5,
                alpha=0.8,
                linestyle='--')
        ax2_2.plot(t, augmented_eeg_noise[chan, :].numpy(), 
                label=f'{channel_names[chan]} - 加噪声', 
                color=colors[chan], 
                linewidth=1.0,
                alpha=0.6,
                linestyle=':')
    ax2_2.set_title(
        f'原始信号 vs 振幅缩放 vs 高斯噪声 (三图对比)\n'
        f'虚线=缩放 {scaling_factor}x, 点线=加噪声 (std={noise_std})', 
        fontsize=14, 
        fontweight='bold',
        pad=15
    )
    ax2_2.set_xlabel('时间 (秒)', fontsize=12, labelpad=10)
    ax2_2.set_ylabel('幅值', fontsize=12, labelpad=10)
    ax2_2.legend(loc='upper right', fontsize=10, ncol=3)
    ax2_2.grid(True, alpha=0.3, linestyle='--')
    ax2_2.set_xlim(0, duration)
    
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.95)
    
    plt.savefig('eeg_augmentation_overlay.png', dpi=150, bbox_inches='tight')
    print("  图像已保存为: eeg_augmentation_overlay.png")
    
    print("\n[4/4] 显示图像...")
    print("注意: 如果在非图形环境中运行，plt.show() 可能不会显示图像")
    print("      但图像已保存为 PNG 文件，可以直接查看。")
    plt.show()


def main():
    """
    主函数：生成 EEG 信号，应用多种数据增强，并可视化对比
    """
    
    print("=" * 80)
    print("EEG 数据增强效果可视化 - selfEEG 演示脚本")
    print("=" * 80)
    
    print("""
本脚本演示 selfEEG 库中常用的 3 种数据增强方法：

1. 时间平移 (Time Shift)
   - 函数: aug.shift_horizontal()
   - 效果: 信号在时间轴上向左或向右平移，超出部分补零

2. 振幅缩放 (Amplitude Scaling)
   - 函数: aug.scaling()
   - 效果: 信号幅值按比例放大或缩小

3. 高斯噪声 (Gaussian Noise)
   - 函数: aug.add_gaussian_noise()
   - 效果: 添加随机高斯噪声到信号中
""")
    
    # 参数设置
    N_CHANNELS = 2
    DURATION = 10.0
    FS = 128.0
    SHIFT_TIME = 2.0
    SCALING_FACTOR = 1.5
    NOISE_STD = 0.3
    
    print(f"参数设置:")
    print(f"  - 通道数: {N_CHANNELS}")
    print(f"  - 信号时长: {DURATION} 秒")
    print(f"  - 采样率: {FS} Hz")
    print(f"  - 总样本数: {int(DURATION * FS)}")
    print(f"  - 时间平移量: {SHIFT_TIME} 秒")
    print(f"  - 振幅缩放因子: {SCALING_FACTOR}x")
    print(f"  - 噪声标准差: {NOISE_STD}")
    
    # 1. 生成模拟 EEG 信号
    print("\n[1/4] 生成模拟 EEG 信号...")
    original_eeg = generate_simulated_eeg(N_CHANNELS, DURATION, FS)
    print(f"  原始信号形状: {original_eeg.shape} (通道数 x 样本数)")
    print(f"  信号范围: [{original_eeg.min():.3f}, {original_eeg.max():.3f}]")
    print(f"  信号均值: {original_eeg.mean():.3f}")
    print(f"  信号标准差: {original_eeg.std():.3f}")
    
    # 2. 应用数据增强
    print("\n[2/4] 应用数据增强...")
    
    # 增强1: 时间平移
    print(f"\n【增强 1】时间平移 (Time Shift)")
    print(f"  函数: aug.shift_horizontal()")
    print(f"  参数: shift_time={SHIFT_TIME}秒, forward=True (向右平移)")
    augmented_eeg_timeshift = aug.shift_horizontal(
        original_eeg, 
        shift_time=SHIFT_TIME, 
        Fs=FS, 
        forward=True
    )
    print(f"  增强后信号形状: {augmented_eeg_timeshift.shape}")
    print(f"  前 {int(SHIFT_TIME * FS)} 个样本是否全为零: {torch.all(augmented_eeg_timeshift[:, :int(SHIFT_TIME * FS)] == 0)}")
    
    # 增强2: 振幅缩放
    print(f"\n【增强 2】振幅缩放 (Amplitude Scaling)")
    print(f"  函数: aug.scaling()")
    print(f"  参数: value={SCALING_FACTOR} (放大 {SCALING_FACTOR} 倍)")
    augmented_eeg_scaling = aug.scaling(
        original_eeg, 
        value=SCALING_FACTOR
    )
    print(f"  增强后信号形状: {augmented_eeg_scaling.shape}")
    print(f"  验证: 增强后最大值 / 原始最大值 = {augmented_eeg_scaling.max() / original_eeg.max():.3f} (预期: {SCALING_FACTOR})")
    
    # 增强3: 高斯噪声
    print(f"\n【增强 3】高斯噪声 (Gaussian Noise)")
    print(f"  函数: aug.add_gaussian_noise()")
    print(f"  参数: std={NOISE_STD} (噪声标准差)")
    augmented_eeg_noise, noise = aug.add_gaussian_noise(
        original_eeg, 
        std=NOISE_STD,
        get_noise=True
    )
    print(f"  增强后信号形状: {augmented_eeg_noise.shape}")
    print(f"  添加的噪声标准差: {noise.std():.3f} (预期约: {NOISE_STD})")
    
    # 3. 验证增强效果
    print("\n" + "-" * 60)
    print("验证增强效果:")
    print("-" * 60)
    
    shift_samples = int(SHIFT_TIME * FS)
    print(f"\n1. 时间平移验证:")
    print(f"   平移样本数: {shift_samples}")
    print(f"   前 {shift_samples} 个样本是否全为零: {torch.all(augmented_eeg_timeshift[:, :shift_samples] == 0)}")
    print(f"   第 {shift_samples} 个样本是否与原始第 0 个样本相等: {torch.allclose(augmented_eeg_timeshift[:, shift_samples], original_eeg[:, 0])}")
    
    print(f"\n2. 振幅缩放验证:")
    scaling_ratio = augmented_eeg_scaling / original_eeg
    print(f"   增强后信号 / 原始信号 的平均值: {scaling_ratio.mean():.3f} (预期: {SCALING_FACTOR})")
    print(f"   增强后信号 / 原始信号 的标准差: {scaling_ratio.std():.6f} (预期接近 0)")
    
    print(f"\n3. 高斯噪声验证:")
    added_noise = augmented_eeg_noise - original_eeg
    print(f"   实际添加的噪声标准差: {added_noise.std():.3f} (预期约: {NOISE_STD})")
    print(f"   实际添加的噪声均值: {added_noise.mean():.4f} (预期接近 0)")
    
    # 4. 可视化
    create_visualization(
        original_eeg,
        augmented_eeg_timeshift,
        augmented_eeg_scaling,
        augmented_eeg_noise,
        N_CHANNELS,
        DURATION,
        FS,
        SHIFT_TIME,
        SCALING_FACTOR,
        NOISE_STD
    )
    
    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)
    
    print("""
总结:
    本脚本演示了 selfEEG 库中 3 种常用的数据增强方法:

    ============================================================================
    1. 时间平移 (Time Shift) - aug.shift_horizontal()
    ============================================================================
       函数签名:
           shift_horizontal(x, shift_time, Fs, forward=None, random_shift=False, batch_equal=True)
       
       参数说明:
           x: 输入信号 (形状: 通道数 x 样本数 或 批次 x 通道数 x 样本数)
           shift_time: 平移时间（秒）
           Fs: 采样率（Hz）
           forward: True=向前平移（右移，前面补零）, False=向后平移（左移，后面补零）
           random_shift: 是否使用随机平移量（shift_time 作为上限）
           batch_equal: 是否对批次中所有信号应用相同的平移
       
       效果:
           信号在时间轴上平移，常用于模拟信号出现时间的变化。
           向前平移: 信号整体右移，前 N 秒为零
           向后平移: 信号整体左移，后 N 秒为零

    ============================================================================
    2. 振幅缩放 (Amplitude Scaling) - aug.scaling()
    ============================================================================
       函数签名:
           scaling(x, value=None, batch_equal=True)
       
       参数说明:
           x: 输入信号
           value: 缩放因子。如果为 None，则随机选择 [0.5, 2] 范围内的值
           batch_equal: 是否对批次中所有信号应用相同的缩放
       
       效果:
           信号幅值按比例放大或缩小，常用于模拟信号强度的变化。
           value > 1: 放大信号
           0 < value < 1: 缩小信号

    ============================================================================
    3. 高斯噪声 (Gaussian Noise) - aug.add_gaussian_noise()
    ============================================================================
       函数签名:
           add_gaussian_noise(x, std=1.0, mean=0.0, get_noise=False)
       
       参数说明:
           x: 输入信号
           std: 噪声的标准差（越大噪声越强）
           mean: 噪声的均值（通常为 0）
           get_noise: 是否返回生成的噪声本身
       
       效果:
           添加随机高斯噪声到信号中，常用于模拟环境噪声或仪器噪声。
           噪声越大，信号越"模糊"，越难以识别原始波形。

    ============================================================================
    4. 随机参数增强 - DynamicSingleAug 类
    ============================================================================
       如果想要每次调用时参数随机变化（如 RandomTimeShift 的效果），
       可以使用 DynamicSingleAug 类包装增强函数:
       
           random_aug = aug.DynamicSingleAug(
               aug.shift_horizontal,
               discrete_arg={'Fs': 128, 'forward': [True, False]},
               range_arg={'shift_time': [0.5, 3.0]},
               range_type={'shift_time': False}
           )
           
           augmented = random_aug(original)
       
       这样每次调用 random_aug() 时，参数都会随机选择。

生成的文件:
    - eeg_augmentation_comparison.png: 4 张子图对比
      * 图1: 原始 EEG 信号
      * 图2: 时间平移增强
      * 图3: 振幅缩放增强
      * 图4: 高斯噪声增强
    
    - eeg_augmentation_overlay.png: 叠图对比
      * 上图: 原始信号 vs 时间平移 (虚线 = 增强后)
      * 下图: 原始信号 vs 振幅缩放 vs 高斯噪声

如果遇到问题:
    - numpy 版本错误: 运行 `pip install "numpy<2"`
    - matplotlib 未安装: 运行 `pip install matplotlib`
    - selfEEG 未安装: 运行 `pip install selfeeg` 或在仓库根目录运行 `pip install -e .`
    - 中文显示为方块: 确保系统安装了中文字体（SimHei, Microsoft YaHei 等）
""")


if __name__ == "__main__":
    main()
