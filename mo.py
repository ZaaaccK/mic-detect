import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# === 1. 加载音频 ===
audio_path = "/Users/zack/Desktop/mic—detct/red5.wav"  # 改成你的音频路径
y, sr = librosa.load(audio_path, sr=None, mono=False)


def compute_decay_rate(rms_series, start_idx, steps=5):
    decay = []
    for k in range(steps):
        if start_idx + k < len(rms_series):
            decay.append(rms_series[start_idx + k])
    if len(decay) < 2:
        return 0
    return (decay[0] - decay[-1]) / (steps / (sr / hop_length))  # 每秒 RMS 降幅


# === 2. 参数设置 ===
frame_length = sr // 5
hop_length = frame_length // 2

threshold_ratio = 3.0        # 主通道与其他通道能量比
min_amplitude = 0.05         # 最小 RMS 门限
min_vector_jump = 0.3        # RMS 向量突变阈值
max_rms_limit = 0.7          # 如果超过此 RMS，认为是敲击（太剧烈）
consecutive_required = 2     # 连续帧满足才判定为触摸
debounce_time = 0          # 去抖秒数（s）

# === 3. 计算 RMS 特征 ===
rms_matrix = []
frame_times = []

for i in range(0, y.shape[1] - frame_length, hop_length):
    frame = y[:, i:i+frame_length]
    rms = np.sqrt(np.mean(frame ** 2, axis=1))
    rms_matrix.append(rms)
    frame_times.append(i / sr)

rms_matrix = np.array(rms_matrix).T  # shape: (4, num_frames)
frame_times = np.array(frame_times)

# === 4. 计算 RMS 向量变化 ===
rms_norms = []
distances = [0]

for j in range(rms_matrix.shape[1]):
    vec = rms_matrix[:, j]
    norm_vec = vec / (np.linalg.norm(vec) + 1e-8)
    rms_norms.append(norm_vec)

for j in range(1, len(rms_norms)):
    dist = euclidean(rms_norms[j], rms_norms[j-1])
    distances.append(dist)

# === 5. 触摸事件检测 + 敲击排除 ===
touch_events = []
knock_events = []
last_touch_time = -10
consecutive_count = 0

for j in range(len(frame_times)):
    rms = rms_matrix[:, j]
    norm_rms = rms / (np.linalg.norm(rms) + 1e-8)
    max_ch = np.argmax(rms)
    max_val = rms[max_ch]
    other_avg = np.mean(np.delete(rms, max_ch))
    vec_jump = distances[j]
    t = frame_times[j]

    # 判定敲击 vs 触摸
    if max_val > max_rms_limit:
        knock_events.append((t, max_ch))
        continue

    if (max_val > threshold_ratio * other_avg and
        max_val > min_amplitude and
        vec_jump > min_vector_jump):
        consecutive_count += 1
    else:
        consecutive_count = 0

    if consecutive_count >= consecutive_required and (t - last_touch_time) > debounce_time:
        touch_events.append((t, max_ch))
        last_touch_time = t
        print(f"✅ 触摸事件 @ {t:.2f}s, 通道 {max_ch}")
        consecutive_count = 0

# === 6. 可视化 ===
plt.figure(figsize=(12, 6))
for ch in range(4):
    plt.plot(np.arange(len(y[ch])) / sr, y[ch], alpha=0.6, label=f"Mic {ch}")

# 标注触摸（绿）和敲击（红）
for t, ch in touch_events:
    plt.axvline(t, color='green', linestyle='--', alpha=0.6)
    plt.text(t, 0.5, f"Touch @Mic {ch}", color='green', rotation=90, va='bottom')

for t, ch in knock_events:
    plt.axvline(t, color='red', linestyle='--', alpha=0.6)
    plt.text(t, 0.5, f"Knock @Mic {ch}", color='red', rotation=90, va='bottom')

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("触摸事件与敲击事件检测")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
