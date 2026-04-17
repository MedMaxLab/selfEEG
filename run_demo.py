"""
EEG 数据增强效果演示脚本
运行此脚本可以生成并查看多种数据增强效果

功能:
1. 单张图演示：2x2 网格布局，展示原始信号和 3 种增强效果
2. 批量处理：生成 5 组不同的随机 EEG 数据，每组都进行时间平移增强
3. 保存高清对比图到 results/ 目录

增强方法:
- 时间平移 (Time Shift): aug.shift_horizontal()
- 振幅缩放 (Amplitude Scaling): aug.scaling()
- 高斯噪声 (Gaussian Noise): aug.add_gaussian_noise()

运行:
    python run_demo.py
"""

import torch
import numpy as np
import os

from selfeeg import augmentation as aug

from eeg_viz import (
    generate_simulated_eeg,
    plot_comparison_grid,
    plot_batch_comparison,
    plot_overlay_comparison
)


def single_sample_demo():
    """
    单样本演示：2x2 网格布局展示原始信号和 3 种增强效果
    """
    print("=" * 80)
    print("【演示 1】单样本 2x2 网格对比")
    print("=" * 80)
    
    N_CHANNELS = 2
    DURATION = 10.0
    FS = 128.0
    SHIFT_TIME = 2.0
    SCALING_FACTOR = 1.5
    NOISE_STD = 0.3
    
    print(f"\n参数设置:")
    print(f"  - 通道数: {N_CHANNELS}")
    print(f"  - 信号时长: {DURATION} 秒")
    print(f"  - 采样率: {FS} Hz")
    print(f"  - 总样本数: {int(DURATION * FS)}")
    print(f"  - 时间平移量: {SHIFT_TIME} 秒")
    print(f"  - 振幅缩放因子: {SCALING_FACTOR}x")
    print(f"  - 噪声标准差: {NOISE_STD}")
    
    print("\n[1/5] 生成模拟 EEG 信号...")
    original_eeg = generate_simulated_eeg(N_CHANNELS, DURATION, FS, seed=42)
    print(f"  信号形状: {original_eeg.shape}")
    print(f"  信号范围: [{original_eeg.min():.3f}, {original_eeg.max():.3f}]")
    
    print("\n[2/5] 应用时间平移增强...")
    augmented_timeshift = aug.shift_horizontal(
        original_eeg, 
        shift_time=SHIFT_TIME, 
        Fs=FS, 
        forward=True
    )
    shift_samples = int(SHIFT_TIME * FS)
    print(f"  前 {shift_samples} 个样本是否全为零: {torch.all(augmented_timeshift[:, :shift_samples] == 0)}")
    
    print("\n[3/5] 应用振幅缩放增强...")
    augmented_scaling = aug.scaling(
        original_eeg, 
        value=SCALING_FACTOR
    )
    scaling_ratio = augmented_scaling.max() / original_eeg.max()
    print(f"  缩放后最大值 / 原始最大值 = {scaling_ratio:.3f} (预期: {SCALING_FACTOR})")
    
    print("\n[4/5] 应用高斯噪声增强...")
    augmented_noise, noise = aug.add_gaussian_noise(
        original_eeg, 
        std=NOISE_STD,
        get_noise=True
    )
    print(f"  添加的噪声标准差: {noise.std():.3f} (预期约: {NOISE_STD})")
    
    print("\n[5/5] 生成 2x2 网格对比图...")
    
    aug_list = [
        {
            'data': augmented_timeshift,
            'title': '时间平移增强',
            'show_original': False
        },
        {
            'data': augmented_scaling,
            'title': '振幅缩放增强',
            'show_original': True
        },
        {
            'data': augmented_noise,
            'title': '高斯噪声增强',
            'show_original': True
        }
    ]
    
    os.makedirs('results', exist_ok=True)
    
    plot_comparison_grid(
        original_eeg=original_eeg,
        aug_list=aug_list,
        fs=FS,
        figsize=(18, 16),
        suptitle=f'EEG 信号增强效果对比 (2通道, 10秒, 128Hz)',
        save_path='results/single_sample_comparison.png',
        show_plot=False,
        channel_names=['通道 1 (蓝色)', '通道 2 (橙色)']
    )
    
    print("\n生成叠图对比...")
    plot_overlay_comparison(
        original_eeg=original_eeg,
        augmented_eeg=augmented_timeshift,
        fs=FS,
        figsize=(16, 9),
        title='原始信号 vs 时间平移增强 (叠图对比)',
        save_path='results/overlay_timeshift.png',
        show_plot=False,
        annotate_shift=True,
        shift_time=SHIFT_TIME
    )
    
    print("\n✓ 单样本演示完成!")
    print(f"  生成的文件:")
    print(f"    - results/single_sample_comparison.png")
    print(f"    - results/overlay_timeshift.png")
    
    return original_eeg, augmented_timeshift, augmented_scaling, augmented_noise


def batch_processing_demo():
    """
    批量处理演示：生成 5 组不同的随机 EEG 数据，
    对每组都进行时间平移增强，然后把 5 个结果拼成一张大图
    """
    print("\n" + "=" * 80)
    print("【演示 2】批量处理对比 (5组样本)")
    print("=" * 80)
    
    N_SAMPLES = 5
    N_CHANNELS = 2
    DURATION = 10.0
    FS = 128.0
    SHIFT_TIME = 2.0
    
    print(f"\n参数设置:")
    print(f"  - 样本组数: {N_SAMPLES}")
    print(f"  - 每组通道数: {N_CHANNELS}")
    print(f"  - 信号时长: {DURATION} 秒")
    print(f"  - 采样率: {FS} Hz")
    print(f"  - 时间平移量: {SHIFT_TIME} 秒")
    
    batch_data = []
    
    print(f"\n生成并处理 {N_SAMPLES} 组样本...")
    
    for i in range(N_SAMPLES):
        seed = 100 + i
        
        print(f"\n  [样本 {i+1}/{N_SAMPLES}]")
        print(f"    种子: {seed}")
        
        original = generate_simulated_eeg(N_CHANNELS, DURATION, FS, seed=seed)
        print(f"    原始信号形状: {original.shape}")
        
        augmented = aug.shift_horizontal(
            original, 
            shift_time=SHIFT_TIME, 
            Fs=FS, 
            forward=True
        )
        shift_samples = int(SHIFT_TIME * FS)
        print(f"    前 {shift_samples} 个样本是否全为零: {torch.all(augmented[:, :shift_samples] == 0)}")
        
        batch_data.append({
            'original': original,
            'augmented': augmented,
            'label': f'样本 {i+1}',
            'aug_name': '时间平移'
        })
    
    print(f"\n生成批量对比图 ({N_SAMPLES} 行 x 2 列)...")
    
    os.makedirs('results', exist_ok=True)
    
    plot_batch_comparison(
        batch_data=batch_data,
        fs=FS,
        figsize=(18, 6 + N_SAMPLES * 3),
        suptitle=f'批量 EEG 信号增强效果对比 ({N_SAMPLES} 组样本, 时间平移 {SHIFT_TIME} 秒)',
        save_path='results/batch_comparison.png',
        show_plot=False,
        channel_names=['通道 1', '通道 2']
    )
    
    print("\n✓ 批量处理演示完成!")
    print(f"  生成的文件:")
    print(f"    - results/batch_comparison.png")
    
    return batch_data


def main():
    """
    主函数：运行所有演示
    """
    print("=" * 80)
    print("EEG 数据增强效果可视化 - selfEEG 演示脚本")
    print("=" * 80)
    
    print("""
本脚本演示 selfEEG 库中常用的数据增强方法：

1. 时间平移 (Time Shift)
   - 函数: aug.shift_horizontal()
   - 效果: 信号在时间轴上向左或向右平移，超出部分补零

2. 振幅缩放 (Amplitude Scaling)
   - 函数: aug.scaling()
   - 效果: 信号幅值按比例放大或缩小

3. 高斯噪声 (Gaussian Noise)
   - 函数: aug.add_gaussian_noise()
   - 效果: 添加随机高斯噪声到信号中

本脚本包含两个演示：
- 演示 1: 单样本 2x2 网格对比
- 演示 2: 批量处理对比 (5组样本)
""")
    
    print("\n" + "-" * 80)
    print("开始运行演示...")
    print("-" * 80)
    
    single_sample_demo()
    
    batch_processing_demo()
    
    print("\n" + "=" * 80)
    print("所有演示完成!")
    print("=" * 80)
    
    print("""
生成的文件汇总:
├── results/
│   ├── single_sample_comparison.png  (2x2 网格对比图)
│   ├── overlay_timeshift.png          (时间平移叠图对比)
│   └── batch_comparison.png           (5组样本批量对比)

图像说明:
1. single_sample_comparison.png
   - 2行2列网格布局
   - 左上角: 原始 EEG 信号
   - 右上角: 时间平移增强
   - 左下角: 振幅缩放增强 (原始信号淡色对比)
   - 右下角: 高斯噪声增强 (原始信号淡色对比)

2. overlay_timeshift.png
   - 叠图对比：原始信号 (实线) vs 时间平移 (虚线)
   - 绿色虚线标注平移位置
   - 箭头+文字说明平移效果

3. batch_comparison.png
   - 5行2列布局
   - 左列: 各组样本的原始信号
   - 右列: 各组样本的增强后信号 (原始信号淡色对比)
   - 每组样本使用不同的随机种子生成

注意事项:
- 所有图像使用 dpi=200 保存，高清分辨率
- 保存时使用 bbox_inches='tight'，避免标题和标签被截断
- 自动创建 results/ 目录保存所有输出文件

如果中文显示为方块:
- 确保系统安装了中文字体 (SimHei, Microsoft YaHei 等)
- Windows 系统通常自带这些字体
- eeg_viz.py 会自动检测并设置可用的中文字体
""")


if __name__ == "__main__":
    main()
