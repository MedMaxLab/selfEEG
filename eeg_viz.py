"""
EEG 数据增强可视化模块
包含多种可视化函数，用于展示 EEG 信号及其增强效果

核心函数:
- plot_comparison_grid: 2x2 网格布局，展示原始信号和多种增强效果
- plot_batch_comparison: 批量处理结果的可视化
- generate_simulated_eeg: 生成模拟 EEG 信号
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional

try:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.gridspec import GridSpec
    
    MATPLOTLIB_AVAILABLE = True
    
    def setup_chinese_font():
        """
        设置 matplotlib 使用中文字体
        自动检测并选择可用的中文字体
        """
        font_names = [
            'SimHei',
            'Microsoft YaHei',
            'KaiTi',
            'FangSong',
            'Arial Unicode MS',
            'PingFang SC',
            'Heiti TC',
            'Noto Sans CJK SC',
            'STSong',
            'STHeiti',
            'STKaiti',
            'STFangSong',
        ]
        
        font_found = False
        selected_font = None
        
        for font_name in font_names:
            try:
                font_manager.fontManager.addfont(font_name)
                plt.rcParams['font.family'] = font_name
                font_found = True
                selected_font = font_name
                break
            except:
                continue
        
        if not font_found:
            system_fonts = [f.name for f in font_manager.fontManager.ttflist]
            chinese_fonts = [f for f in system_fonts if any(
                keyword.lower() in f.lower() 
                for keyword in ['hei', 'kai', 'song', 'fang', 'yahei', 'noto', 'ping', 'sim', 'st']
            )]
            if chinese_fonts:
                plt.rcParams['font.family'] = chinese_fonts[0]
                selected_font = chinese_fonts[0]
                font_found = True
        
        plt.rcParams['axes.unicode_minus'] = False
        
        return font_found, selected_font
    
    setup_chinese_font()
    
except ImportError:
    MATPLOTLIB_AVAILABLE = False
except Exception as e:
    MATPLOTLIB_AVAILABLE = False
    print(f"matplotlib 初始化警告: {e}")


def generate_simulated_eeg(
    n_channels: int = 2, 
    duration: float = 10.0, 
    fs: float = 128.0,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    生成模拟 EEG 信号
    
    该函数生成包含多个正弦波分量和高斯噪声的模拟 EEG 信号。
    信号包含典型的脑电频率分量：delta (2Hz), theta (5Hz), alpha (10Hz), 
    beta (15Hz), gamma (20Hz)
    
    参数:
        n_channels: 通道数 (默认: 2)
        duration: 持续时间（秒） (默认: 10.0)
        fs: 采样率（Hz） (默认: 128.0)
        seed: 随机种子 (默认: None)
    
    返回:
        torch.Tensor: 形状为 (n_channels, n_samples) 的 EEG 信号
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
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


def plot_comparison_grid(
    original_eeg: torch.Tensor,
    aug_list: List[Dict],
    fs: float = 128.0,
    figsize: Tuple[int, int] = (16, 14),
    suptitle: str = "EEG 信号增强效果对比",
    save_path: Optional[str] = None,
    show_plot: bool = False,
    channel_colors: Optional[List[str]] = None,
    channel_names: Optional[List[str]] = None
) -> plt.Figure:
    """
    使用 2x2 网格布局展示 EEG 信号及其增强效果
    
    参数:
        original_eeg: 原始 EEG 信号，形状为 (n_channels, n_samples)
        aug_list: 增强结果列表，每个元素是字典，包含:
            - 'data': 增强后的信号 tensor
            - 'title': 子图标题
            - 'show_original': 是否在同一张图中显示原始信号作为对比 (默认: False)
        fs: 采样率（Hz）(默认: 128.0)
        figsize: 图像尺寸 (宽度, 高度)，单位英寸 (默认: (16, 14))
        suptitle: 总标题 (默认: "EEG 信号增强效果对比")
        save_path: 保存路径，如果为 None 则不保存 (默认: None)
        show_plot: 是否显示图像 (默认: False)
        channel_colors: 各通道的颜色列表 (默认: ['#1f77b4', '#ff7f0e'])
        channel_names: 各通道的名称列表 (默认: ['通道 1', '通道 2'])
    
    返回:
        plt.Figure: matplotlib Figure 对象
    
    示例:
        >>> original = generate_simulated_eeg()
        >>> aug1 = {'data': augmented1, 'title': '时间平移增强'}
        >>> aug2 = {'data': augmented2, 'title': '振幅缩放增强'}
        >>> aug3 = {'data': augmented3, 'title': '高斯噪声增强'}
        >>> plot_comparison_grid(original, [aug1, aug2, aug3], 
        ...                      save_path='comparison.png')
    """
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib 不可用，无法绘制图像")
    
    n_channels = original_eeg.shape[0]
    n_samples = original_eeg.shape[1]
    duration = n_samples / fs
    
    t = np.linspace(0, duration, n_samples)
    
    if channel_colors is None:
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        channel_colors = default_colors[:n_channels]
    
    if channel_names is None:
        channel_names = [f'通道 {i+1}' for i in range(n_channels)]
    
    fig = plt.figure(figsize=figsize)
    
    gs = GridSpec(2, 2, figure=fig, 
                  hspace=0.35,
                  wspace=0.25,
                  height_ratios=[1, 1],
                  width_ratios=[1, 1])
    
    all_axes = []
    
    ax_original = fig.add_subplot(gs[0, 0])
    all_axes.append(ax_original)
    
    for chan in range(n_channels):
        ax_original.plot(t, original_eeg[chan, :].numpy(), 
                        label=f'{channel_names[chan]}', 
                        color=channel_colors[chan], 
                        linewidth=1.8,
                        alpha=0.9)
    
    ax_original.set_title('【原始信号】\n未经过任何增强的 EEG 信号', 
                          fontsize=14, 
                          fontweight='bold',
                          pad=12)
    ax_original.set_xlabel('时间 (秒)', fontsize=11, labelpad=8)
    ax_original.set_ylabel('幅值', fontsize=11, labelpad=8)
    ax_original.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax_original.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax_original.set_xlim(0, duration)
    ax_original.tick_params(axis='both', labelsize=9)
    
    grid_positions = [(0, 1), (1, 0), (1, 1)]
    
    for idx, aug_info in enumerate(aug_list[:3]):
        row, col = grid_positions[idx]
        ax = fig.add_subplot(gs[row, col])
        all_axes.append(ax)
        
        aug_data = aug_info['data']
        aug_title = aug_info.get('title', f'增强 {idx+1}')
        show_original = aug_info.get('show_original', False)
        
        if show_original:
            for chan in range(n_channels):
                ax.plot(t, original_eeg[chan, :].numpy(), 
                        label=f'{channel_names[chan]} (原始)', 
                        color=channel_colors[chan], 
                        linewidth=1.2,
                        alpha=0.3,
                        zorder=1)
        
        for chan in range(n_channels):
            linestyle = '--' if show_original else '-'
            alpha = 0.9 if show_original else 0.9
            linewidth = 1.8 if show_original else 1.8
            
            label = f'{channel_names[chan]} (增强后)' if show_original else channel_names[chan]
            
            ax.plot(t, aug_data[chan, :].numpy(), 
                    label=label, 
                    color=channel_colors[chan], 
                    linewidth=linewidth,
                    alpha=alpha,
                    linestyle=linestyle,
                    zorder=2)
        
        ax.set_title(f'【{aug_title}】', 
                      fontsize=14, 
                      fontweight='bold',
                      pad=12)
        ax.set_xlabel('时间 (秒)', fontsize=11, labelpad=8)
        ax.set_ylabel('幅值', fontsize=11, labelpad=8)
        
        ncol = 2 if show_original else 1
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9, ncol=ncol)
        ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
        ax.set_xlim(0, duration)
        ax.tick_params(axis='both', labelsize=9)
    
    fig.suptitle(suptitle, fontsize=18, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95)
    
    if save_path:
        plt.savefig(save_path, 
                    dpi=200, 
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none')
        print(f"图像已保存: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_batch_comparison(
    batch_data: List[Dict],
    fs: float = 128.0,
    figsize: Tuple[int, int] = (16, 20),
    suptitle: str = "批量 EEG 信号增强效果对比",
    save_path: Optional[str] = None,
    show_plot: bool = False,
    channel_colors: Optional[List[str]] = None,
    channel_names: Optional[List[str]] = None
) -> plt.Figure:
    """
    批量对比多个 EEG 样本及其增强效果
    使用 N 行 2 列布局，每行显示一个样本的原始信号和增强信号
    
    参数:
        batch_data: 批量数据列表，每个元素是字典，包含:
            - 'original': 原始信号 tensor
            - 'augmented': 增强后信号 tensor
            - 'label': 样本标签/名称 (可选)
            - 'aug_name': 增强方法名称 (可选)
        fs: 采样率（Hz）(默认: 128.0)
        figsize: 图像尺寸 (默认: (16, 20))
        suptitle: 总标题 (默认: "批量 EEG 信号增强效果对比")
        save_path: 保存路径 (默认: None)
        show_plot: 是否显示图像 (默认: False)
        channel_colors: 各通道的颜色列表
        channel_names: 各通道的名称列表
    
    返回:
        plt.Figure: matplotlib Figure 对象
    """
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib 不可用，无法绘制图像")
    
    n_samples = len(batch_data)
    
    if n_samples == 0:
        raise ValueError("batch_data 不能为空")
    
    n_channels = batch_data[0]['original'].shape[0]
    n_samples_per_signal = batch_data[0]['original'].shape[1]
    duration = n_samples_per_signal / fs
    
    t = np.linspace(0, duration, n_samples_per_signal)
    
    if channel_colors is None:
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        channel_colors = default_colors[:n_channels]
    
    if channel_names is None:
        channel_names = [f'通道 {i+1}' for i in range(n_channels)]
    
    fig = plt.figure(figsize=figsize)
    
    n_rows = n_samples
    n_cols = 2
    
    gs = GridSpec(n_rows, n_cols, figure=fig,
                  hspace=0.45,
                  wspace=0.25)
    
    for sample_idx, sample_info in enumerate(batch_data):
        original = sample_info['original']
        augmented = sample_info['augmented']
        label = sample_info.get('label', f'样本 {sample_idx+1}')
        aug_name = sample_info.get('aug_name', '增强')
        
        ax_original = fig.add_subplot(gs[sample_idx, 0])
        
        for chan in range(n_channels):
            ax_original.plot(t, original[chan, :].numpy(), 
                            label=f'{channel_names[chan]}', 
                            color=channel_colors[chan], 
                            linewidth=1.6,
                            alpha=0.9)
        
        ax_original.set_title(f'【{label}】原始信号', 
                              fontsize=13, 
                              fontweight='bold',
                              pad=10)
        ax_original.set_xlabel('时间 (秒)', fontsize=10, labelpad=6)
        ax_original.set_ylabel('幅值', fontsize=10, labelpad=6)
        ax_original.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax_original.grid(True, alpha=0.3, linestyle='--')
        ax_original.set_xlim(0, duration)
        ax_original.tick_params(axis='both', labelsize=8)
        
        ax_augmented = fig.add_subplot(gs[sample_idx, 1])
        
        for chan in range(n_channels):
            ax_augmented.plot(t, original[chan, :].numpy(), 
                            label=f'{channel_names[chan]} (原始)', 
                            color=channel_colors[chan], 
                            linewidth=1.0,
                            alpha=0.25)
        
        for chan in range(n_channels):
            ax_augmented.plot(t, augmented[chan, :].numpy(), 
                            label=f'{channel_names[chan]} ({aug_name})', 
                            color=channel_colors[chan], 
                            linewidth=1.8,
                            alpha=0.9,
                            linestyle='--')
        
        ax_augmented.set_title(f'【{label}】{aug_name}后信号\n(原始信号以淡色显示)', 
                              fontsize=13, 
                              fontweight='bold',
                              pad=10)
        ax_augmented.set_xlabel('时间 (秒)', fontsize=10, labelpad=6)
        ax_augmented.set_ylabel('幅值', fontsize=10, labelpad=6)
        ax_augmented.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=1)
        ax_augmented.grid(True, alpha=0.3, linestyle='--')
        ax_augmented.set_xlim(0, duration)
        ax_augmented.tick_params(axis='both', labelsize=8)
    
    fig.suptitle(suptitle, fontsize=17, fontweight='bold', y=0.99)
    
    plt.subplots_adjust(top=0.94, bottom=0.05, left=0.07, right=0.96)
    
    if save_path:
        import os
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(save_path, 
                    dpi=200, 
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none')
        print(f"批量对比图已保存: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_overlay_comparison(
    original_eeg: torch.Tensor,
    augmented_eeg: torch.Tensor,
    fs: float = 128.0,
    figsize: Tuple[int, int] = (14, 8),
    title: str = "原始信号 vs 增强信号 (叠图对比)",
    save_path: Optional[str] = None,
    show_plot: bool = False,
    channel_colors: Optional[List[str]] = None,
    channel_names: Optional[List[str]] = None,
    annotate_shift: bool = False,
    shift_time: Optional[float] = None
) -> plt.Figure:
    """
    叠图对比：在同一张图中显示原始信号和增强信号
    
    参数:
        original_eeg: 原始 EEG 信号
        augmented_eeg: 增强后的 EEG 信号
        fs: 采样率（Hz）
        figsize: 图像尺寸
        title: 图像标题
        save_path: 保存路径
        show_plot: 是否显示图像
        channel_colors: 各通道颜色
        channel_names: 各通道名称
        annotate_shift: 是否标注时间平移效果
        shift_time: 平移时间（秒），用于标注
    
    返回:
        plt.Figure: matplotlib Figure 对象
    """
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib 不可用，无法绘制图像")
    
    n_channels = original_eeg.shape[0]
    n_samples = original_eeg.shape[1]
    duration = n_samples / fs
    
    t = np.linspace(0, duration, n_samples)
    
    if channel_colors is None:
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        channel_colors = default_colors[:n_channels]
    
    if channel_names is None:
        channel_names = [f'通道 {i+1}' for i in range(n_channels)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for chan in range(n_channels):
        ax.plot(t, original_eeg[chan, :].numpy(), 
                label=f'{channel_names[chan]} (原始)', 
                color=channel_colors[chan], 
                linewidth=2.0,
                alpha=0.6,
                linestyle='-')
    
    for chan in range(n_channels):
        ax.plot(t, augmented_eeg[chan, :].numpy(), 
                label=f'{channel_names[chan]} (增强后)', 
                color=channel_colors[chan], 
                linewidth=2.0,
                alpha=0.9,
                linestyle='--')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('时间 (秒)', fontsize=12, labelpad=10)
    ax.set_ylabel('幅值', fontsize=12, labelpad=10)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, duration)
    ax.tick_params(axis='both', labelsize=10)
    
    if annotate_shift and shift_time is not None:
        ax.axvline(x=shift_time, color='green', linestyle='-.', alpha=0.8, linewidth=2.5)
        ax.annotate(
            f'信号向右平移 {shift_time} 秒\n'
            f'前 {shift_time} 秒补零，信号整体右移', 
            xy=(shift_time, 0), 
            xytext=(shift_time + 0.5, -1.5),
            arrowprops=dict(arrowstyle='->', color='darkgreen', linewidth=2),
            fontsize=12, 
            color='darkgreen',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, 
                    dpi=200, 
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none')
        print(f"叠图对比已保存: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig
