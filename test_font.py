"""
测试字体设置
"""
import os
import platform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

print("=" * 60)
print("测试 matplotlib 中文字体设置")
print("=" * 60)

print(f"\n操作系统: {platform.system()}")

font_dir = 'C:\\Windows\\Fonts'
font_files = [
    ('simhei.ttf', 'SimHei'),
    ('msyh.ttc', 'Microsoft YaHei'),
    ('simkai.ttf', 'KaiTi'),
    ('simsun.ttc', 'SimSun'),
    ('simfang.ttf', 'FangSong'),
]

print(f"\n检查 Windows 字体目录: {font_dir}")
found_fonts = []
for font_file, font_name in font_files:
    font_path = os.path.join(font_dir, font_file)
    exists = os.path.exists(font_path)
    print(f"  {font_file} ({font_name}): {'存在' if exists else '不存在'}")
    if exists:
        found_fonts.append((font_path, font_name))

print(f"\n找到的字体文件: {found_fonts}")

if found_fonts:
    font_path, font_name = found_fonts[0]
    print(f"\n尝试添加字体: {font_path}")
    try:
        font_manager.fontManager.addfont(font_path)
        print(f"  字体添加成功")
    except Exception as e:
        print(f"  字体添加失败: {e}")
    
    print(f"\n设置 font.family = {font_name}")
    plt.rcParams['font.family'] = font_name
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False
else:
    print("\n警告: 未找到任何字体文件")

print(f"\n当前 rcParams 设置:")
print(f"  font.family: {plt.rcParams.get('font.family', '未设置')}")
print(f"  font.sans-serif: {plt.rcParams.get('font.sans-serif', '未设置')}")
print(f"  axes.unicode_minus: {plt.rcParams.get('axes.unicode_minus', '未设置')}")

print("\n" + "=" * 60)
print("创建测试图像")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

import numpy as np
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

ax.plot(x, y1, label='通道 1 (正弦波)', color='#1f77b4', linewidth=2)
ax.plot(x, y2, label='通道 2 (余弦波)', color='#ff7f0e', linewidth=2)

ax.set_title('测试中文字体显示 - 这是一个标题', fontsize=16, fontweight='bold')
ax.set_xlabel('时间 (秒)', fontsize=12)
ax.set_ylabel('幅值', fontsize=12)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

ax.text(5, 0, '这是一段中文文字测试\n用来验证字体是否正常工作', 
        fontsize=14, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

os.makedirs('results', exist_ok=True)
save_path = 'results/font_test.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n测试图像已保存: {save_path}")

plt.close(fig)

print("\n" + "=" * 60)
print("完成!")
print("=" * 60)
print(f"\n请查看 {save_path} 确认中文是否正常显示")
