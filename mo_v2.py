import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# === 1. 加载音频 ===
audio_path = "/Users/zack/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_209ay89m1i9422_6c6e/msg/file/2025-08/red12.wav"  # 改成你的音频路径
y, sr = librosa.load(audio_path, sr=None, mono=False)

# === 2. 参数设置 ===
frame_length = sr // 5
hop_length = frame_length // 2

threshold_ratio = 1.5        # 主通道与其他通道能量比
min_amplitude = 0.05         # 最小 RMS 门限
min_vector_jump = 0.2        # RMS 向量突变阈值
consecutive_required = 2     # 连续帧满足才判定为触摸
debounce_time = 0            # 去抖秒数（s）
window_size = 1.2            # 窗口大小（秒）：2秒内的事件视为一组

# === 3. 计算 RMS 特征 ===
rms_matrix = []
frame_times = []

for i in range(0, y.shape[1] - frame_length, hop_length):
    frame = y[:, i:i+frame_length]
    rms = np.sqrt(np.mean(frame **2, axis=1))  # 计算每个通道的RMS能量
    rms_matrix.append(rms)
    frame_times.append(i / sr)  # 记录当前帧的时间

rms_matrix = np.array(rms_matrix).T  # shape: (通道数, 帧数)
frame_times = np.array(frame_times)
num_channels = rms_matrix.shape[0]  # 获取麦克风通道数量

# === 4. 计算 RMS 向量变化 ===
rms_norms = []
distances = [0]  # 存储相邻帧RMS向量的欧氏距离

for j in range(rms_matrix.shape[1]):
    vec = rms_matrix[:, j]  # 当前帧的RMS向量（各通道能量组成）
    norm_vec = vec / (np.linalg.norm(vec) + 1e-8)  # 归一化，避免能量绝对值影响
    rms_norms.append(norm_vec)

# 计算相邻帧RMS向量的变化（欧氏距离）
for j in range(1, len(rms_norms)):
    dist = euclidean(rms_norms[j], rms_norms[j-1])
    distances.append(dist)

# === 5. 触摸事件检测（核心：记录每次触发的通道顺序） ===
touch_events = []  # 存储所有触摸事件 (时间, 通道)
event_channel_sequence = []  # 记录每次触摸事件的通道顺序（按触发时间排列）
last_touch_time = -10  # 上次触摸事件的时间（用于去抖）
consecutive_count = 0  # 连续满足条件的帧数
high_rms_channels = []  # 记录RMS异常高的通道

for j in range(len(frame_times)):
    rms = rms_matrix[:, j]  # 当前帧各通道RMS值
    max_ch = np.argmax(rms)  # 能量最高的通道（当前主导麦克风）
    max_val = rms[max_ch]  # 最高能量值
    other_avg = np.mean(np.delete(rms, max_ch))  # 其他通道的平均能量
    vec_jump = distances[j]  # 当前帧与上一帧的RMS向量变化
    t = frame_times[j]  # 当前帧时间


    # 触摸事件判定条件：
    # 1. 主通道能量远高于其他通道  2. 能量高于最小门限  3. 能量分布发生显著突变
    if (max_val > threshold_ratio * other_avg and
        max_val > min_amplitude and
        vec_jump > min_vector_jump):
        consecutive_count += 1  # 连续满足条件的帧数+1
    else:
        consecutive_count = 0  # 不满足则重置

    # 当连续满足条件且超过去抖时间，判定为有效触摸事件
    if consecutive_count >= consecutive_required and (t - last_touch_time) > debounce_time:
        touch_events.append((t, max_ch))  # 记录事件时间和对应通道
        event_channel_sequence.append(max_ch)  # 按触发顺序记录通道
        last_touch_time = t  # 更新上次触摸时间
        print(f"✅ 触摸事件 @ {t:.2f}s, 触发通道: {max_ch}")  # 实时打印触发信息
        consecutive_count = 0  # 重置连续计数

# === 6. 窗口内通道顺序分组（核心新增功能） ===
window_sequences = []  # 存储每个窗口内的通道顺序
if touch_events:
    current_window = [touch_events[0][1]]  # 第一个窗口初始化
    last_event_time = touch_events[0][0]   # 第一个事件时间

    # 遍历剩余事件，按窗口分组
    for t, ch in touch_events[1:]:
        if t - last_event_time <= window_size:  # 在2秒窗口内，加入当前组
            current_window.append(ch)
        else:  # 超过窗口，保存当前组并新建组
            window_sequences.append(current_window)
            current_window = [ch]
        last_event_time = t  # 更新组内最后事件时间

    # 加入最后一个窗口
    window_sequences.append(current_window)

# === 7. 输出结果摘要（重点展示窗口内顺序） ===
print("\n===== 触摸事件通道顺序总结 =====")
print(f"总触摸事件数: {len(touch_events)}")
print(f"通道切换轨迹: {' → '.join(map(str, event_channel_sequence))}")  # 完整轨迹

# 打印窗口内顺序（如 3-2；3-2；0-1）
if window_sequences:
    print(f"\n{window_size}秒窗口内的通道顺序:")
    window_strings = ["-".join(map(str, seq)) for seq in window_sequences]
    print("； ".join(window_strings))  # 用分号分隔不同窗口

# 显示RMS异常高的通道
if high_rms_channels:
    print("\nRMS异常高的通道:")
    unique_high_ch = set(ch for _, ch, _ in high_rms_channels)
    for ch in unique_high_ch:
        ch_events = [(t, val) for t, c, val in high_rms_channels if c == ch]
        max_val = max(val for _, val in ch_events)
        print(f"通道 {ch}: 出现 {len(ch_events)} 次，最高值 {max_val:.4f}")

# === 8. 可视化（突出窗口分组） ===
plt.figure(figsize=(14, 12))

# 子图1：音频波形与触摸事件标记
for ch in range(num_channels):
    plt.subplot(4, 1, 1)
    plt.plot(np.arange(len(y[ch])) / sr, y[ch], alpha=0.6, label=f"Mic {ch}")
plt.title("wave and event", fontsize=12)
plt.ylabel("amplitude")
plt.grid(True, alpha=0.3)
# 标注每个触摸事件及其通道
for i, (t, ch) in enumerate(touch_events):
    plt.axvline(t, color='green', linestyle='--', alpha=0.6)
    plt.text(t, 0.5, f"Event {i+1}: Mic {ch}", 
             color='green', rotation=90, va='bottom', fontsize=9)

# 子图2：各通道RMS能量曲线
plt.subplot(4, 1, 2)
for ch in range(num_channels):
    plt.plot(frame_times, rms_matrix[ch], label=f"Mic {ch} RMS", alpha=0.8)

plt.title("rms change", fontsize=12)
plt.ylabel("RMS")
plt.grid(True, alpha=0.3)
# 标注触摸事件时间点
for t, ch in touch_events:
    plt.axvline(t, color='green', linestyle='--', alpha=0.6)

# 子图3：通道触发顺序标注
plt.subplot(4, 1, 3)
plt.plot(frame_times, [0]*len(frame_times), alpha=0)  # 空图用于标注
plt.ylim(-1, num_channels)
plt.yticks(range(num_channels), [f"Mic {ch}" for ch in range(num_channels)])
plt.title("sequence", fontsize=12)
plt.xlabel("time (s)")
plt.grid(True, alpha=0.3)
# 按时间顺序标注每个事件的通道
for i, (t, ch) in enumerate(touch_events):
    plt.scatter(t, ch, color='blue', s=50, zorder=3)
    plt.text(t, ch+0.2, f"Event {i+1}", color='blue', fontsize=9)

# 子图4：窗口分组标记
plt.subplot(4, 1, 4)
plt.plot(frame_times, [0]*len(frame_times), alpha=0)  # 空图用于标注
plt.ylim(-0.5, 0.5)
plt.yticks([])  # 隐藏y轴刻度
plt.title(f"{window_size}s window groups", fontsize=12)
plt.xlabel("time (s)")
plt.grid(True, alpha=0.3, axis='x')

# 标注窗口分组（用背景色区分）
if window_sequences and touch_events:
    window_start_idx = 0  # 每个窗口在touch_events中的起始索引
    for i, seq in enumerate(window_sequences):
        window_end_idx = window_start_idx + len(seq) - 1  # 窗口结束索引
        window_start_time = touch_events[window_start_idx][0]
        window_end_time = touch_events[window_end_idx][0]
        # 绘制窗口背景
        plt.axvspan(window_start_time, window_end_time, 
                    color=f"C{i%10}", alpha=0.2, label=f"Window {i+1}: {'-'.join(map(str, seq))}")
        window_start_idx = window_end_idx + 1  # 更新下一个窗口起始索引
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放右侧

plt.tight_layout()
plt.show()