import numpy as np
import matplotlib.pyplot as plt

# 参数设置
num_phi = 2048  # 方位角采样点数
phi = np.linspace(0, 2 * np.pi, num_phi, endpoint=False)

# 目标OAM模及功率分布
oam_modes = np.array([-9,7])
target_power = np.array([1,1])
target_power = target_power / np.sum(target_power)  # 归一化
target_amp = np.sqrt(target_power)

# 迭代参数
max_iter = 100
alpha = 0.5  # 初始步长
min_alpha = 0.01
max_alpha = 1.0
tol = 1e-6
last_err = None

# 只在-20到20范围内迭代
m_min, m_max = -20, 20
m_range = np.arange(m_min, m_max + 1)
C = np.zeros(len(m_range), dtype=complex)
for i, l in enumerate(oam_modes):
    idx = np.where(m_range == l)[0][0]
    C[idx] = target_amp[i]  # 初始幅度，初始相位为0

# 初始化
g_phi = np.exp(1j * np.random.uniform(0, 2*np.pi, num_phi))  # 随机初相位

err_list = []
for it in range(max_iter):
    # 1. 用C合成S_phi
    S_phi = np.zeros_like(phi, dtype=complex)
    for idx, m in enumerate(m_range):
        S_phi += C[idx] * np.exp(1j * m * phi)
    # 2. 取相位，生成g_phi
    phase_phi = np.angle(S_phi)
    g_phi = np.exp(1j * phase_phi)
    # 3. OAM域分析
    B = np.fft.fft(g_phi) / num_phi
    B = np.fft.fftshift(B)
    m_fft = np.fft.fftshift(np.fft.fftfreq(num_phi, 1/num_phi)).astype(int)
    # 4. 对目标OAM模做比例幅度修正
    for i, l in enumerate(oam_modes):
        idx = np.where(m_range == l)[0][0]
        B_idx = np.where(m_fft == l)[0][0]
        # 论文式：|C_k| <- |C_k| + alpha * (|A_k| - |B_k|)
        C_abs_new = np.abs(C[idx]) + alpha * (target_amp[i] - np.abs(B[B_idx]))
        C[idx] = C_abs_new * np.exp(1j * np.angle(C[idx]))
    # 5. 误差判断等...
    # 计算当前目标OAM模的幅度
    B_amp = []
    for l in oam_modes:
        idx = np.where(m_fft == l)[0][0]
        B_amp.append(np.abs(B[idx]))
    B_amp = np.array(B_amp)
    delta = target_amp - B_amp
    err = np.max(np.abs(delta))
    err_list.append(err)
    if it % 100 == 0 or err < tol:
        print(f'Iter {it+1}: max error={err:.4e}')
    if err < tol:
        print(f'Converged at iteration {it+1}, max error={err:.2e}')
        break
else:
    print(f'Not fully converged, final max error={err:.2e}')

# 计算最终功率分布
S_phi = np.zeros_like(phi, dtype=complex)
for idx, m in enumerate(m_range):
    S_phi += C[idx] * np.exp(1j * m * phi)
phase_phi = np.angle(S_phi)
g_phi = np.exp(1j * phase_phi)
B = np.fft.fft(g_phi) / num_phi
B = np.fft.fftshift(B)
m_fft = np.fft.fftshift(np.fft.fftfreq(num_phi, 1/num_phi)).astype(int)
B_power = np.abs(B)**2

# 输出目标OAM模的实际功率分布
actual_power = []
for l in oam_modes:
    idx = np.where(m_fft == l)[0][0]
    actual_power.append(B_power[idx])
actual_power = np.array(actual_power)
actual_power = actual_power / np.sum(B_power)  # 归一化

print("目标功率分布:", target_power)
print("实际功率分布:", actual_power)
print("功率比:", actual_power / actual_power[0])

# 可视化（横坐标范围优化）
mask = (m_fft >= -20) & (m_fft <= 20)

# 生成颜色列表：目标OAM为蓝色，其余为绿色
bar_colors = []
for m in m_fft[mask]:
    if m in oam_modes:
        bar_colors.append('blue')
    else:
        bar_colors.append('green')

plt.figure(figsize=(8,4))
plt.bar(m_fft[mask], (B_power / np.sum(B_power))[mask], color=bar_colors, width=0.8)
plt.xlabel('OAM mode (l)')
plt.ylabel('Normalized Power')
plt.title('OAM Power Spectrum after Optimization')
plt.xlim(-20, 20)
plt.tight_layout()
plt.show()

# 生成二维OAM相位图
N = 1080  # 方形图像分辨率
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)
PHI = np.arctan2(Y, X)  # [-pi, pi]
PHI[PHI < 0] += 2 * np.pi  # 转为 [0, 2pi)

# 将PHI映射到你的phase_phi
phi_1d = np.linspace(0, 2 * np.pi, num_phi, endpoint=False)
# 用插值获得每个像素的相位
phase_map = np.interp(PHI.flatten(), phi_1d, phase_phi)
phase_map = phase_map.reshape(N, N)

# 假设 phase_map.shape == (1080, 1080)
target_h, target_w = 1080, 1920
src_h, src_w = phase_map.shape

# 计算左右需要补的像素数
pad_left = (target_w - src_w) // 2
pad_right = target_w - src_w - pad_left

# 上下不用补
phase_map_padded = np.pad(
    phase_map,
    ((0, 0), (pad_left, pad_right)),
    mode='constant',
    constant_values=0
)

print("padded shape:", phase_map_padded.shape)  # 应为 (1080, 1920)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.imshow(phase_map_padded, cmap='gray', extent=[0, target_w, 0, target_h])
plt.title('Padded OAM Phase Plate (Centered, 1920x1080)')
plt.axis('off')
plt.tight_layout()
plt.show()

# 保存图片到指定路径
import os
save_dir = r"C:\Users\胡涛\Desktop\qipan"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "316phase_map.bmp")
plt.imsave(save_path, phase_map_padded, cmap='gray')
print(f"Image saved to: {save_path}")
