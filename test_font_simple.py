"""
快速测试中文字体设置
运行此脚本验证中文字体是否正常工作
"""

import os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

print("=" * 60)
print("Kuai Su Ce Shi Zhong Wen Ti Ti Xian Shi")
print("=" * 60)

font_dir = 'C:\\Windows\\Fonts'
font_files_to_try = [
    ('simhei.ttf', 'SimHei'),
    ('msyh.ttc', 'Microsoft YaHei'),
    ('simkai.ttf', 'KaiTi'),
    ('simsun.ttc', 'SimSun'),
]

selected_font_path = None
selected_font_name = None

for font_file, font_name in font_files_to_try:
    font_path = os.path.join(font_dir, font_file)
    if os.path.exists(font_path):
        print(f"\nFound font: {font_name}")
        print(f"  Path: {font_path}")
        selected_font_path = font_path
        selected_font_name = font_name
        break

if selected_font_path is None:
    print("\nNot found common Chinese font files")
    print("\nTrying to find from matplotlib registered fonts...")
    
    from matplotlib import font_manager as fm
    all_fonts = [(f.name, f.fname) for f in fm.fontManager.ttflist]
    
    chinese_keywords = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'SimSun', 'PingFang', 'Heiti', 'STSong', 'STHeiti', 'STKaiti', 'Noto']
    
    for font_name, font_path in all_fonts:
        for keyword in chinese_keywords:
            if keyword.lower() in font_name.lower():
                print(f"Found font: {font_name}")
                print(f"  Path: {font_path}")
                selected_font_path = font_path
                selected_font_name = font_name
                break
        if selected_font_path:
            break

if selected_font_path:
    print(f"\nUsing font: {selected_font_name}")
    
    font_prop_title = FontProperties(fname=selected_font_path, size=16)
    font_prop_label = FontProperties(fname=selected_font_path, size=12)
    font_prop_legend = FontProperties(fname=selected_font_path, size=11)
    font_prop_text = FontProperties(fname=selected_font_path, size=14)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    ax.plot(x, y1, label='通道 1 (正弦波)', color='#1f77b4', linewidth=2)
    ax.plot(x, y2, label='通道 2 (余弦波)', color='#ff7f0e', linewidth=2)
    
    ax.set_title('这是一个中文标题 - 测试字体显示', fontproperties=font_prop_title, fontweight='bold', pad=20)
    ax.set_xlabel('时间 (秒)', fontproperties=font_prop_label, labelpad=10)
    ax.set_ylabel('幅值', fontproperties=font_prop_label, labelpad=10)
    
    ax.legend(loc='upper right', prop=font_prop_legend, framealpha=0.9)
    
    ax.text(5, 0, '这是一段中文测试文字\n用来验证字体是否正常显示', 
            fontproperties=font_prop_text,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    save_path = 'results/font_test_final.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"\nTest image saved: {save_path}")
    print("\nPlease open this image to check if Chinese text displays correctly.")
    
else:
    print("\n" + "!" * 60)
    print("WARNING: Could not find any Chinese font!")
    print("!" * 60)

print("\n" + "=" * 60)
print("Test completed")
print("=" * 60)
