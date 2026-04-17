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
import os
import platform
from typing import List, Dict, Tuple, Optional

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties

MATPLOTLIB_AVAILABLE = True

CHINESE_FONT_PATH = None
CHINESE_FONT_PROP = None
CHINESE_FONT_NAME = None
FONT_SETUP_SUCCESS = False


def find_chinese_font_files():
    """
    查找系统中的中文字体文件
    返回字体文件路径列表
    """
    system = platform.system()
    
    font_candidates = []
    
    if system == 'Windows':
        font_dirs = [
            'C:\\Windows\\Fonts',
            os.path.expanduser('~\\AppData\\Local\\Microsoft\\Windows\\Fonts'),
        ]
        
        font_patterns = [
            ('simhei.ttf', 'SimHei'),
            ('msyh.ttc', 'Microsoft YaHei'),
            ('msyhbd.ttc', 'Microsoft YaHei Bold'),
            ('simkai.ttf', 'KaiTi'),
            ('simsun.ttc', 'SimSun'),
            ('simfang.ttf', 'FangSong'),
            ('STSONG.TTF', 'STSong'),
            ('STHEITI.TTF', 'STHeiti'),
            ('STKAI.TTF', 'STKaiti'),
            ('STFANGSO.TTF', 'STFangSong'),
        ]
        
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for font_file, font_name in font_patterns:
                    font_path = os.path.join(font_dir, font_file)
                    if os.path.exists(font_path):
                        font_candidates.append((font_path, font_name))
    
    elif system == 'Darwin':
        font_dirs = [
            '/System/Library/Fonts',
            '/Library/Fonts',
            os.path.expanduser('~/Library/Fonts'),
        ]
        
        font_patterns = [
            ('PingFang SC.ttc', 'PingFang SC'),
            ('STHeiti Light.ttc', 'STHeiti'),
            ('STKaiti.ttf', 'STKaiti'),
            ('Songti.ttc', 'Songti'),
            ('Arial Unicode.ttf', 'Arial Unicode'),
        ]
        
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for font_file, font_name in font_patterns:
                    font_path = os.path.join(font_dir, font_file)
                    if os.path.exists(font_path):
                        font_candidates.append((font_path, font_name))
    
    else:
        font_dirs = [
            '/usr/share/fonts',
            '/usr/local/share/fonts',
            os.path.expanduser('~/.fonts'),
            '/usr/share/fonts/opentype',
            '/usr/share/fonts/truetype',
        ]
        
        font_patterns = [
            ('NotoSansCJK-Regular.ttc', 'Noto Sans CJK'),
            ('NotoSansSC-Regular.otf', 'Noto Sans SC'),
            ('wqy-zenhei.ttc', 'WenQuanYi Zen Hei'),
            ('uming.ttc', 'AR PL UKai'),
        ]
        
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for root, dirs, files in os.walk(font_dir):
                    for font_file, font_name in font_patterns:
                        if font_file in files:
                            font_path = os.path.join(root, font_file)
                            font_candidates.append((font_path, font_name))
    
    return font_candidates


def get_font_properties_from_ttflist():
    """
    从 matplotlib 的 ttflist 中查找中文字体
    """
    chinese_fonts = []
    
    all_fonts = [(f.name, f.fname) for f in fm.fontManager.ttflist]
    
    chinese_keywords = [
        'SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'SimSun',
        'PingFang', 'Heiti', 'STSong', 'STHeiti', 'STKaiti', 'STFang',
        'Noto Sans CJK', 'Noto Sans SC', 'WenQuanYi', 'AR PL'
    ]
    
    for font_name, font_path in all_fonts:
        for keyword in chinese_keywords:
            if keyword.lower() in font_name.lower():
                chinese_fonts.append((font_path, font_name))
                break
    
    return chinese_fonts


def setup_chinese_font():
    """
    设置 matplotlib 使用中文字体
    自动检测并选择可用的中文字体
    """
    global CHINESE_FONT_PATH, CHINESE_FONT_PROP, CHINESE_FONT_NAME, FONT_SETUP_SUCCESS
    
    print("=" * 60)
    print("正在设置中文字体...")
    print("=" * 60)
    
    font_candidates = []
    
    print("\n[1/3] 查找系统字体文件...")
    file_fonts = find_chinese_font_files()
    font_candidates.extend(file_fonts)
    if file_fonts:
        print(f"  找到 {len(file_fonts)} 个字体文件:")
        for path, name in file_fonts:
            print(f"    - {name}: {path}")
    else:
        print("  未找到常见字体文件")
    
    print("\n[2/3] 查找 matplotlib 已注册字体...")
    ttflist_fonts = get_font_properties_from_ttflist()
    for font_path, font_name in ttflist_fonts:
        if font_path not in [f[0] for f in font_candidates]:
            font_candidates.append((font_path, font_name))
    
    if ttflist_fonts:
        print(f"  找到 {len(ttflist_fonts)} 个已注册字体:")
        for path, name in ttflist_fonts:
            print(f"    - {name}")
    else:
        print("  未找到已注册的中文字体")
    
    print("\n[3/3] 尝试设置字体...")
    
    if font_candidates:
        for font_path, font_name in font_candidates:
            try:
                print(f"\n  尝试字体: {font_name}")
                print(f"  路径: {font_path}")
                
                if os.path.exists(font_path):
                    fm.fontManager.addfont(font_path)
                    print(f"  ✓ 字体已添加到 fontManager")
                else:
                    print(f"  警告: 字体文件不存在: {font_path}")
                
                font_prop = FontProperties(fname=font_path) if os.path.exists(font_path) else None
                
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                
                plt.rcParams['axes.unicode_minus'] = False
                
                CHINESE_FONT_PATH = font_path
                CHINESE_FONT_PROP = font_prop
                CHINESE_FONT_NAME = font_name
                FONT_SETUP_SUCCESS = True
                
                print(f"\n✓ 字体设置成功!")
                print(f"  - 字体名称: {font_name}")
                print(f"  - 字体文件: {font_path}")
                
                return True, font_name, font_path
                
            except Exception as e:
                print(f"  ✗ 设置失败: {e}")
                continue
    
    print("\n" + "!" * 60)
    print("警告: 未能找到可用的中文字体!")
    print("中文可能会显示为方块字。")
    print("!" * 60)
    
    print("""
建议解决方案:

Windows 系统:
  1. 确保系统安装了以下字体之一:
     - SimHei (黑体) - 通常位于 C:\Windows\Fonts\simhei.ttf
     - Microsoft YaHei (微软雅黑) - 通常位于 C:\Windows\Fonts\msyh.ttc
     - KaiTi (楷体) - 通常位于 C:\Windows\Fonts\simkai.ttf
  
  2. 如果缺少字体，可以:
     - 从其他 Windows 系统复制字体文件
     - 或下载并安装中文字体

Mac 系统:
  1. 确保系统安装了以下字体之一:
     - PingFang SC
     - STHeiti
     - Songti SC

Linux 系统:
  1. 安装中文字体包:
     - Ubuntu/Debian: sudo apt-get install fonts-noto-cjk
     - CentOS/RHEL: sudo yum install google-noto-sans-cjk-sc-fonts
     - 或安装 WenQuanYi 字体: sudo apt-get install fonts-wqy-zenhei
""")
    
    plt.rcParams['axes.unicode_minus'] = False
    
    return False, None, None


FONT_SETUP_SUCCESS, SELECTED_FONT, SELECTED_FONT_PATH = setup_chinese_font()

print(f"\n" + "=" * 60)
print("字体设置结果:")
print("=" * 60)
print(f"  成功: {'是' if FONT_SETUP_SUCCESS else '否'}")
if SELECTED_FONT:
    print(f"  字体名称: {SELECTED_FONT}")
if SELECTED_FONT_PATH:
    print(f"  字体文件: {SELECTED_FONT_PATH}")

plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def get_chinese_font_prop(size=None):
    """
    获取中文字体的 FontProperties 对象
    用于在绘图时显式指定字体
    
    参数:
        size: 字体大小，如果为 None 则使用默认大小
    
    返回:
        FontProperties 对象或 None
    """
    global CHINESE_FONT_PROP, CHINESE_FONT_PATH
    
    if CHINESE_FONT_PATH and os.path.exists(CHINESE_FONT_PATH):
        if size:
            return FontProperties(fname=CHINESE_FONT_PATH, size=size)
        else:
            return FontProperties(fname=CHINESE_FONT_PATH)
    elif CHINESE_FONT_PROP:
        return CHINESE_FONT_PROP
    return None


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
    figsize: Tuple[int, int] = (18, 16),
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
        figsize: 图像尺寸 (宽度, 高度)，单位英寸 (默认: (18, 16))
        suptitle: 总标题 (默认: "EEG 信号增强效果对比")
        save_path: 保存路径，如果为 None 则不保存 (默认: None)
        show_plot: 是否显示图像 (默认: False)
        channel_colors: 各通道的颜色列表 (默认: ['#1f77b4', '#ff7f0e'])
        channel_names: 各通道的名称列表 (默认: ['通道 1', '通道 2'])
    
    返回:
        plt.Figure: matplotlib Figure 对象
    """
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib 不可用，无法绘制图像")
    
    font_prop_title = get_chinese_font_prop(size=14)
    font_prop_label = get_chinese_font_prop(size=12)
    font_prop_legend = get_chinese_font_prop(size=10)
    font_prop_suptitle = get_chinese_font_prop(size=18)
    
    n_channels = original_eeg.shape[0]
    n_samples = original_eeg.shape[1]
    duration = n_samples / fs
    
    t = np.linspace(0, duration, n_samples)
    
    if channel_colors is None:
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        channel_colors = default_colors[:n_channels]
    
    if channel_names is None:
        channel_names = [f'通道 {i+1}' for i in range(n_channels)]
    
    fig = plt.figure(figsize=figsize, dpi=120)
    
    gs = GridSpec(2, 2, figure=fig, 
                  hspace=0.4,
                  wspace=0.25,
                  height_ratios=[1, 1],
                  width_ratios=[1, 1])
    
    ax_original = fig.add_subplot(gs[0, 0])
    
    for chan in range(n_channels):
        ax_original.plot(t, original_eeg[chan, :].numpy(), 
                        label=f'{channel_names[chan]}', 
                        color=channel_colors[chan], 
                        linewidth=1.8,
                        alpha=0.9)
    
    title_text = '【原始信号】\n未经过任何增强的 EEG 信号'
    if font_prop_title:
        ax_original.set_title(title_text, fontproperties=font_prop_title, fontweight='bold', pad=12)
    else:
        ax_original.set_title(title_text, fontsize=14, fontweight='bold', pad=12)
    
    if font_prop_label:
        ax_original.set_xlabel('时间 (秒)', fontproperties=font_prop_label, labelpad=8)
        ax_original.set_ylabel('幅值', fontproperties=font_prop_label, labelpad=8)
    else:
        ax_original.set_xlabel('时间 (秒)', fontsize=12, labelpad=8)
        ax_original.set_ylabel('幅值', fontsize=12, labelpad=8)
    
    if font_prop_legend:
        ax_original.legend(loc='upper right', prop=font_prop_legend, framealpha=0.9)
    else:
        ax_original.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    ax_original.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax_original.set_xlim(0, duration)
    ax_original.tick_params(axis='both', labelsize=10)
    
    grid_positions = [(0, 1), (1, 0), (1, 1)]
    
    for idx, aug_info in enumerate(aug_list[:3]):
        row, col = grid_positions[idx]
        ax = fig.add_subplot(gs[row, col])
        
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
        
        title_text = f'【{aug_title}】'
        if font_prop_title:
            ax.set_title(title_text, fontproperties=font_prop_title, fontweight='bold', pad=12)
        else:
            ax.set_title(title_text, fontsize=14, fontweight='bold', pad=12)
        
        if font_prop_label:
            ax.set_xlabel('时间 (秒)', fontproperties=font_prop_label, labelpad=8)
            ax.set_ylabel('幅值', fontproperties=font_prop_label, labelpad=8)
        else:
            ax.set_xlabel('时间 (秒)', fontsize=12, labelpad=8)
            ax.set_ylabel('幅值', fontsize=12, labelpad=8)
        
        ncol = 2 if show_original else 1
        if font_prop_legend:
            ax.legend(loc='upper right', prop=font_prop_legend, framealpha=0.9, ncol=ncol)
        else:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9, ncol=ncol)
        
        ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
        ax.set_xlim(0, duration)
        ax.tick_params(axis='both', labelsize=10)
    
    if font_prop_suptitle:
        fig.suptitle(suptitle, fontproperties=font_prop_suptitle, fontweight='bold', y=0.98)
    else:
        fig.suptitle(suptitle, fontsize=18, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95)
    
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
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
    figsize: Tuple[int, int] = (18, 22),
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
        figsize: 图像尺寸 (默认: (18, 22))
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
    
    font_prop_title = get_chinese_font_prop(size=14)
    font_prop_label = get_chinese_font_prop(size=12)
    font_prop_legend = get_chinese_font_prop(size=10)
    font_prop_suptitle = get_chinese_font_prop(size=17)
    
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
    
    fig = plt.figure(figsize=figsize, dpi=120)
    
    n_rows = n_samples
    n_cols = 2
    
    gs = GridSpec(n_rows, n_cols, figure=fig,
                  hspace=0.5,
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
        
        title_text = f'【{label}】原始信号'
        if font_prop_title:
            ax_original.set_title(title_text, fontproperties=font_prop_title, fontweight='bold', pad=10)
        else:
            ax_original.set_title(title_text, fontsize=14, fontweight='bold', pad=10)
        
        if font_prop_label:
            ax_original.set_xlabel('时间 (秒)', fontproperties=font_prop_label, labelpad=6)
            ax_original.set_ylabel('幅值', fontproperties=font_prop_label, labelpad=6)
        else:
            ax_original.set_xlabel('时间 (秒)', fontsize=12, labelpad=6)
            ax_original.set_ylabel('幅值', fontsize=12, labelpad=6)
        
        if font_prop_legend:
            ax_original.legend(loc='upper right', prop=font_prop_legend, framealpha=0.9)
        else:
            ax_original.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        ax_original.grid(True, alpha=0.3, linestyle='--')
        ax_original.set_xlim(0, duration)
        ax_original.tick_params(axis='both', labelsize=10)
        
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
        
        title_text = f'【{label}】{aug_name}后信号\n(原始信号以淡色显示)'
        if font_prop_title:
            ax_augmented.set_title(title_text, fontproperties=font_prop_title, fontweight='bold', pad=10)
        else:
            ax_augmented.set_title(title_text, fontsize=14, fontweight='bold', pad=10)
        
        if font_prop_label:
            ax_augmented.set_xlabel('时间 (秒)', fontproperties=font_prop_label, labelpad=6)
            ax_augmented.set_ylabel('幅值', fontproperties=font_prop_label, labelpad=6)
        else:
            ax_augmented.set_xlabel('时间 (秒)', fontsize=12, labelpad=6)
            ax_augmented.set_ylabel('幅值', fontsize=12, labelpad=6)
        
        if font_prop_legend:
            ax_augmented.legend(loc='upper right', prop=font_prop_legend, framealpha=0.9, ncol=1)
        else:
            ax_augmented.legend(loc='upper right', fontsize=10, framealpha=0.9, ncol=1)
        
        ax_augmented.grid(True, alpha=0.3, linestyle='--')
        ax_augmented.set_xlim(0, duration)
        ax_augmented.tick_params(axis='both', labelsize=10)
    
    if font_prop_suptitle:
        fig.suptitle(suptitle, fontproperties=font_prop_suptitle, fontweight='bold', y=0.99)
    else:
        fig.suptitle(suptitle, fontsize=17, fontweight='bold', y=0.99)
    
    plt.subplots_adjust(top=0.94, bottom=0.05, left=0.07, right=0.96)
    
    if save_path:
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
    figsize: Tuple[int, int] = (16, 10),
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
    
    font_prop_title = get_chinese_font_prop(size=16)
    font_prop_label = get_chinese_font_prop(size=12)
    font_prop_legend = get_chinese_font_prop(size=11)
    font_prop_annot = get_chinese_font_prop(size=12)
    
    n_channels = original_eeg.shape[0]
    n_samples = original_eeg.shape[1]
    duration = n_samples / fs
    
    t = np.linspace(0, duration, n_samples)
    
    if channel_colors is None:
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        channel_colors = default_colors[:n_channels]
    
    if channel_names is None:
        channel_names = [f'通道 {i+1}' for i in range(n_channels)]
    
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    
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
    
    if font_prop_title:
        ax.set_title(title, fontproperties=font_prop_title, fontweight='bold', pad=15)
    else:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    if font_prop_label:
        ax.set_xlabel('时间 (秒)', fontproperties=font_prop_label, labelpad=10)
        ax.set_ylabel('幅值', fontproperties=font_prop_label, labelpad=10)
    else:
        ax.set_xlabel('时间 (秒)', fontsize=12, labelpad=10)
        ax.set_ylabel('幅值', fontsize=12, labelpad=10)
    
    if font_prop_legend:
        ax.legend(loc='upper right', prop=font_prop_legend, framealpha=0.9, ncol=2)
    else:
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9, ncol=2)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, duration)
    ax.tick_params(axis='both', labelsize=10)
    
    if annotate_shift and shift_time is not None:
        ax.axvline(x=shift_time, color='green', linestyle='-.', alpha=0.8, linewidth=2.5)
        
        annot_text = f'信号向右平移 {shift_time} 秒\n前 {shift_time} 秒补零，信号整体右移'
        
        if font_prop_annot:
            ax.annotate(
                annot_text, 
                xy=(shift_time, 0), 
                xytext=(shift_time + 0.5, -1.5),
                arrowprops=dict(arrowstyle='->', color='darkgreen', linewidth=2),
                fontproperties=font_prop_annot,
                color='darkgreen',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )
        else:
            ax.annotate(
                annot_text, 
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
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
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
