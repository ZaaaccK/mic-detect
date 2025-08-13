import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
import matplotlib.gridspec as gridspec
import librosa
from scipy.spatial.distance import euclidean

# === 敲击/拍打检测相关函数 ===
def load_audio_tap_knock(path):
    audio, sr_loaded = sf.read(path)
    if audio.ndim > 1:
        audio = audio[:, 0]  # 转为单声道
    return audio, sr_loaded

def compute_rms(audio, hop_length):
    rms = []
    for i in range(0, len(audio), hop_length):
        frame = audio[i:i + hop_length]
        if len(frame) == hop_length:
            rms.append(np.sqrt(np.mean(frame **2)))
    return np.array(rms)

def estimate_background_rms(rms, window=50):
    baseline = median_filter(rms, size=window)
    return baseline

def detect_events_tap_knock(rms, rms_threshold, event_spacing):
    peaks, _ = find_peaks(rms, height=rms_threshold, distance=event_spacing)
    return peaks

def compute_relative_decay(rms_series, start_idx, steps=15):
    peak_val = rms_series[start_idx]
    if peak_val < 1e-6:
        return 0
    end_idx = min(start_idx + 1 + steps, len(rms_series))
    tail_vals = rms_series[start_idx+1 : end_idx]
    if len(tail_vals) == 0:
        return 0
    mean_tail = np.mean(tail_vals)
    decay_ratio = (peak_val - mean_tail) / peak_val
    return decay_ratio

def compute_peak_width(rms_series, start_idx, threshold_ratio=0.5):
    peak_val = rms_series[start_idx]
    threshold = peak_val * threshold_ratio
    
    # 向左找边界
    left = start_idx
    while left > 0 and rms_series[left] > threshold:
        left -= 1
    
    # 向右找边界
    right = start_idx
    while right < len(rms_series)-1 and rms_series[right] > threshold:
        right += 1
    
    return right - left  # 峰值宽度（帧数）

def classify_tap_or_knock(decay_ratio, variance, peak_rms, peak_width,
                          decay_threshold=0.75,
                          var_threshold=0.02,
                          rms_min=0.2,
                          width_threshold=8):
    if (decay_ratio > decay_threshold and 
        variance < var_threshold and 
        peak_rms > rms_min and 
        peak_width < width_threshold):
        return "knock"
    else:
        return "tap"

def classify_events_tap_knock(rms, event_idxs, baseline):
    results = []
    for idx in event_idxs:
        if rms[idx] - baseline[idx] < 0.05:
            continue
        decay = compute_relative_decay(rms, idx, steps=15)
        end_var_idx = min(idx + 15, len(rms))
        var = np.var(rms[idx:end_var_idx])
        peak = rms[idx]
        peak_width = compute_peak_width(rms, idx)
        label = classify_tap_or_knock(decay, var, peak, peak_width)
        results.append((idx, peak, label, decay, var, peak_width))
    return results

# === 触摸检测相关函数 ===
def compute_decay_rate(rms_series, start_idx, steps=5, sr=None, hop_length=None):
    decay = []
    for k in range(steps):
        if start_idx + k < len(rms_series):
            decay.append(rms_series[start_idx + k])
    if len(decay) < 2 or sr is None or hop_length is None:
        return 0
    return (decay[0] - decay[-1]) / (steps / (sr / hop_length))  # 每秒 RMS 降幅

def detect_touch_events(audio_path):
    # 加载音频
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    
    # 参数设置
    frame_length = sr // 5
    hop_length = frame_length // 2

    threshold_ratio = 3.0        # 主通道与其他通道能量比
    min_amplitude = 0.05         # 最小 RMS 门限
    min_vector_jump = 0.3        # RMS 向量突变阈值
    max_rms_limit = 0.7          # 如果超过此 RMS，认为是敲击（太剧烈）
    consecutive_required = 2     # 连续帧满足才判定为触摸
    debounce_time = 0.5          # 去抖秒数（s）

    # 计算 RMS 特征
    rms_matrix = []
    frame_times = []

    for i in range(0, y.shape[1] - frame_length, hop_length):
        frame = y[:, i:i+frame_length]
        rms = np.sqrt(np.mean(frame** 2, axis=1))
        rms_matrix.append(rms)
        frame_times.append(i / sr)

    rms_matrix = np.array(rms_matrix).T  # shape: (num_channels, num_frames)
    frame_times = np.array(frame_times)

    # 计算 RMS 向量变化
    rms_norms = []
    distances = [0]

    for j in range(rms_matrix.shape[1]):
        vec = rms_matrix[:, j]
        norm_vec = vec / (np.linalg.norm(vec) + 1e-8)
        rms_norms.append(norm_vec)

    for j in range(1, len(rms_norms)):
        dist = euclidean(rms_norms[j], rms_norms[j-1])
        distances.append(dist)

    # 触摸事件检测 + 敲击排除
    touch_events = []
    knock_events_librosa = []  # 这里的敲击事件与第一个检测器的敲击不同
    last_touch_time = -10
    consecutive_count = 0

    for j in range(len(frame_times)):
        rms = rms_matrix[:, j]
        max_ch = np.argmax(rms)
        max_val = rms[max_ch]
        other_avg = np.mean(np.delete(rms, max_ch)) if len(rms) > 1 else 0
        vec_jump = distances[j]
        t = frame_times[j]

        # 判定敲击 vs 触摸
        if max_val > max_rms_limit:
            knock_events_librosa.append((t, max_ch))
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
            consecutive_count = 0

    return touch_events, knock_events_librosa, y, sr

# === 事件协调与去重函数 ===
def coordinate_events(tap_knock_events, touch_events, hop_length_tap_knock, sr_tap_knock, conflict_window=1.0):
    """
    冲突处理逻辑：
    1. 若冲突窗口内有"摸"事件，优先保留"摸"，移除所有冲突的"敲/拍"
    2. 若冲突窗口内只有"敲/拍"事件，只保留第一个事件
    3. 不同窗口的事件正常保留
    """
    # 转换为统一格式：(时间, 类型, 其他信息)
    all_events = []
    
    # 添加敲击/拍打事件
    for event in tap_knock_events:
        idx, peak, label, decay, var, width = event
        time = idx * hop_length_tap_knock / sr_tap_knock
        all_events.append((time, label, event))
    
    # 添加触摸事件
    for event in touch_events:
        time, channel = event
        all_events.append((time, "touch", event))
    
    # 按时间排序
    all_events.sort(key=lambda x: x[0])
    
    # 应用规则过滤事件
    filtered_events = []
    processed_events = []  # 记录已处理的事件：(时间, 类型)
    
    for event in all_events:
        current_time, current_type, current_details = event
        conflict_found = False
        
        # 检查当前事件与已处理事件是否存在冲突
        for processed_time, processed_type in processed_events:
            time_diff = abs(current_time - processed_time)
            if time_diff < conflict_window:
                # 冲突情况1：当前是"摸"，已处理的是"敲/拍" → 移除已处理的，保留当前"摸"
                if current_type == "touch" and processed_type in ["touch", "tap"]:
                    # 移除已处理的冲突事件
                    filtered_events = [e for e in filtered_events 
                                      if not (abs(e[0] - current_time) < conflict_window 
                                              and e[1] in ["knock", "tao"])]
                    processed_events = [e for e in processed_events 
                                       if not (abs(e[0] - current_time) < conflict_window 
                                               and e[1] in ["knock", "tap"])]
                    conflict_found = False  # 清除冲突标记，因为要添加当前"摸"
                    break
                # 冲突情况2：当前是"敲/拍"，已处理的是任何类型 → 跳过当前事件
                elif current_type in ["knock", "tap"]:
                    conflict_found = True
                    break
                # 冲突情况3：当前是"摸"，已处理的也是"摸" → 保留第一个"摸"
                elif current_type == "touch" and processed_type == "touch":
                    conflict_found = True
                    break
        
        if not conflict_found:
            filtered_events.append(event)
            processed_events.append((current_time, current_type))
    
    # 按时间重新排序
    filtered_events.sort(key=lambda x: x[0])
    return filtered_events

# === 可视化函数 ===
def plot_combined_results(audio, tap_knock_audio, filtered_events, tap_knock_events,
                          rms, baseline, sr, hop_length_tap_knock, sr_tap_knock):
    # 创建主图
    plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 2, 1])
    
    # 波形图
    ax1 = plt.subplot(gs[0])
    time_full = np.arange(len(audio[0])) / sr if audio.ndim > 1 else np.arange(len(audio)) / sr
    if audio.ndim > 1:
        for ch in range(min(4, audio.shape[0])):
            ax1.plot(time_full, audio[ch], alpha=0.5, label=f"channel {ch}")
    else:
        ax1.plot(time_full, audio, alpha=0.7, label='wave')
    
    # 标记所有过滤后的事件
    for time, event_type, details in filtered_events:
        color = 'blue' if event_type == 'touch' else 'red' if event_type == 'knock' else 'green'
        ax1.axvline(time, color=color, linestyle='--', alpha=0.8)
        ax1.text(time, ax1.get_ylim()[1]*0.9, event_type, color=color, fontsize=10)
    
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("amplitude")
    ax1.set_title("detect")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # RMS能量图
    ax2 = plt.subplot(gs[1])
    time_rms = np.arange(len(rms)) * hop_length_tap_knock / sr_tap_knock
    ax2.plot(time_rms, rms, label="RMS", alpha=0.8)
    ax2.plot(time_rms, baseline, label="baseline", color='orange', linestyle='--')
    ax2.axhline(0.2, label="threshold", color='r', alpha=0.5, linestyle=':')
    
    for time, event_type, details in filtered_events:
        if event_type in ['knock', 'tap']:
            # 找到对应的峰值
            for idx, peak, label, _, _, _ in tap_knock_events:
                event_time = idx * hop_length_tap_knock / sr_tap_knock
                if abs(event_time - time) < 0.01:  # 时间接近
                    color = 'red' if event_type == 'tap' else 'green'
                    ax2.scatter(time, peak, color=color, s=50, zorder=3)
                    ax2.text(time, peak, event_type, color=color, fontsize=9, ha='right')
    
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("RMS")
    ax2.set_title("RMS&knock/tap")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 事件统计
    ax3 = plt.subplot(gs[2])
    event_counts = {"knock": 0, "tap": 0, "touch": 0}
    for _, event_type, _ in filtered_events:
        event_counts[event_type] += 1
    
    ax3.bar(event_counts.keys(), event_counts.values(), color=['red', 'green', 'blue'])
    ax3.set_title("overall")
    ax3.set_ylabel("count")
    
    plt.tight_layout()
    plt.show()

# === 主函数 ===
def main(audio_path, conflict_window=1.0):
    # 1. 配置敲击/拍打检测参数
    sr_tap_knock = 16000
    hop_length_tap_knock = 512
    rms_threshold = 0.2
    event_spacing = int(0.2 * sr_tap_knock / hop_length_tap_knock)
    
    # 2. 运行敲击/拍打检测
    tap_knock_audio, sr_tap_knock = load_audio_tap_knock(audio_path)
    rms = compute_rms(tap_knock_audio, hop_length_tap_knock)
    baseline = estimate_background_rms(rms)
    events_tap_knock_idxs = detect_events_tap_knock(rms, rms_threshold, event_spacing)
    tap_knock_events = classify_events_tap_knock(rms, events_tap_knock_idxs, baseline)
    
    # 3. 运行触摸检测
    touch_events, knock_events_librosa, full_audio, sr_full = detect_touch_events(audio_path)
    
    # 4. 协调所有事件 - 处理所有类型的冲突
    filtered_events = coordinate_events(tap_knock_events, touch_events, 
                                       hop_length_tap_knock, sr_tap_knock, conflict_window)
    
    # 5. 打印结果
    print("检测到的事件（按时间排序）：")
    print("时间 (秒) | 类型 | 详情")
    print("-" * 50)
    for time, event_type, details in filtered_events:
        if event_type in ['knock', 'tap']:
            idx, peak, _, decay, var, width = details
            print(f"{time:.2f}     | {event_type:2s} | 峰值: {peak:.3f}, 衰减率: {decay:.3f}, 宽度: {width}帧")
        else:  # 摸
            _, channel = details
            print(f"{time:.2f}     | {event_type:2s} | 通道: {channel}")
    
    # 6. 可视化
    plot_combined_results(full_audio, tap_knock_audio, filtered_events, tap_knock_events,
                          rms, baseline, sr_full, hop_length_tap_knock, sr_tap_knock)

# 运行主程序
audio_path = "/Users/zack/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_209ay89m1i9422_6c6e/msg/file/2025-08/c.wav"
if __name__ == "__main__":
    main(audio_path, conflict_window=2)  # 冲突窗口设为1秒
    