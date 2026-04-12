import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
# 注意：文件使用分号作为分隔符
df = pd.read_csv(r'D:/py_test/毕设/data/Erosion fault state-Vw=5.3.csv', sep=';')

# 查看数据基本信息  
print("数据形状:", df.shape)
print("\n前几行数据:")
print(df.head())
print("\n数据统计信息:")
print(df.describe())

# 创建图形
fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

# 1. 时域波形图
axes[0].plot(df['Time - Voltage_1'], df['Amplitude - Voltage_1'], 
             color='steelblue', linewidth=0.8, alpha=0.9)
axes[0].set_xlabel('Time (s)', fontsize=12)
axes[0].set_ylabel('Amplitude (V)', fontsize=12)
axes[0].set_title('Time Domain Signal - Crack Fault (Vw=4)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

# 添加统计信息文本框
stats_text = f'Max: {df["Amplitude - Voltage_1"].max():.4f} V\nMin: {df["Amplitude - Voltage_1"].min():.4f} V\nMean: {df["Amplitude - Voltage_1"].mean():.4f} V\nStd: {df["Amplitude - Voltage_1"].std():.4f} V'
axes[0].text(0.02, 0.95, stats_text, transform=axes[0].transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 2. 振幅分布直方图
axes[1].hist(df['Amplitude - Voltage_1'], bins=50, color='coral', 
             edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Amplitude (V)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Amplitude Distribution', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('crack_fault_time_domain.png', dpi=300, bbox_inches='tight')
plt.show()

# 额外：绘制局部放大图（前0.05秒）
fig2, ax = plt.subplots(figsize=(12, 5))

# 选取前0.05秒的数据
mask = df['Time - Voltage_1'] <= 0.05
ax.plot(df.loc[mask, 'Time - Voltage_1'], 
        df.loc[mask, 'Amplitude - Voltage_1'], 
        color='steelblue', linewidth=1.2, marker='o', markersize=3, alpha=0.8)

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Amplitude (V)', fontsize=12)
ax.set_title('Time Domain Signal (Zoomed: First 0.05s)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig('crack_fault_zoomed.png', dpi=300, bbox_inches='tight')
plt.show()

# 如果需要频域分析（FFT），可以添加以下代码
from scipy.fft import fft, fftfreq

# 采样频率计算
dt = df['Time - Voltage_1'].iloc[1] - df['Time - Voltage_1'].iloc[0]  # 时间步长
fs = 1 / dt  # 采样频率

print(f"\n采样信息:")
print(f"时间步长: {dt:.6f} s")
print(f"采样频率: {fs:.2f} Hz")
print(f"总采样点数: {len(df)}")
print(f"信号时长: {df['Time - Voltage_1'].iloc[-1]:.3f} s")

# 进行FFT变换
n = len(df)
signal = df['Amplitude - Voltage_1'].values
yf = fft(signal)
xf = fftfreq(n, dt)[:n//2]

# 只取正频率部分
amplitude_spectrum = 2.0/n * np.abs(yf[:n//2])

# 绘制频谱图
fig3, ax = plt.subplots(figsize=(12, 5))
ax.plot(xf, amplitude_spectrum, color='forestgreen', linewidth=0.8)
ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.set_title('Frequency Spectrum (FFT)', fontsize=14, fontweight='bold')
ax.set_xlim([0, fs/2])  # 显示到奈奎斯特频率
ax.grid(True, alpha=0.3)

# 标注主要频率峰值
if len(xf) > 0:
    # 找到前5个主要峰值
    peak_indices = np.argsort(amplitude_spectrum)[-5:][::-1]
    for idx in peak_indices:
        if amplitude_spectrum[idx] > amplitude_spectrum.max() * 0.1:  # 只标注大于最大峰值10%的
            ax.annotate(f'{xf[idx]:.1f} Hz', 
                       xy=(xf[idx], amplitude_spectrum[idx]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, alpha=0.7,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('crack_fault_spectrum.png', dpi=300, bbox_inches='tight')
plt.show()