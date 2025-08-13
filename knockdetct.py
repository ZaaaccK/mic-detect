import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
import matplotlib.gridspec as gridspec



sr = 16000
hop_length = 512
rms_threshold = 0.2
event_spacing = int(0.2 * sr / hop_length)

def load_audio(path):
    audio, sr_loaded = sf.read(path)
    if audio.ndim > 1:
        audio = audio[:, 0]  # 转为单声道
    return audio, sr_loaded

def compute_rms(audio):
    rms = []
    for i in range(0, len(audio), hop_length):
        frame = audio[i:i + hop_length]
        if len(frame) == hop_length:
            rms.append(np.sqrt(np.mean(frame **2)))
    return np.array(rms)

def estimate_background_rms(rms, window=50):
    baseline = median_filter(rms, size=window)
    return baseline

def detect_events(rms):
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
    """新增特征：计算峰值宽度（超过峰值一半的持续帧数），敲的宽度通常更窄"""
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
                          # 调整阈值：降低rms_min，提高衰减阈值，放宽方差
                          decay_threshold=0.75,  # 敲的衰减更快（要求更高）
                          var_threshold=0.02,    # 拍的方差可能更大
                          rms_min=0.2,           # 降低峰值要求，匹配实际数据
                          width_threshold=8):    # 敲的峰值更窄（帧数更少）
    # 敲的特征：衰减快、方差小、峰值足够、宽度窄
    if (decay_ratio > decay_threshold and 
        variance < var_threshold and 
        peak_rms > rms_min and 
        peak_width < width_threshold):
        return "敲"
    else:
        return "拍"

def classify_events(rms, event_idxs, baseline):
    results = []
    for idx in event_idxs:
        if rms[idx] - baseline[idx] < 0.05:
            continue
        decay = compute_relative_decay(rms, idx, steps=15)
        end_var_idx = min(idx + 15, len(rms))
        var = np.var(rms[idx:end_var_idx])
        peak = rms[idx]
        peak_width = compute_peak_width(rms, idx)  # 计算新特征
        label = classify_tap_or_knock(decay, var, peak, peak_width)
        results.append((idx, peak, label, decay, var, peak_width))  # 保存宽度特征
    return results

def plot_waveform(audio, results, sr, hop_length):
    time = np.arange(len(audio)) / sr
    plt.figure(figsize=(12, 6))
    plt.plot(time, audio, alpha=0.7, label='音频波形')
    
    for idx, peak, label, decay, var, width in results:
        t = idx * hop_length / sr
        color = 'r' if label == '敲' else 'g'
        plt.axvline(t, color=color, linestyle='--', alpha=0.8)
        plt.text(t, max(audio)*0.9, label, color=color, fontsize=10)
    
    plt.xlabel("时间 (秒)")
    plt.ylabel("振幅")
    plt.title("原始音频波形与检测事件")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_event_details(audio, rms, results, sr, hop_length, window_size=0.2):
    num_events = len(results)
    if num_events == 0:
        print("未检测到任何事件")
        return
    
    cols = 2
    rows = (num_events + cols - 1) // cols
    
    plt.figure(figsize=(12, 5 * rows))
    gs = gridspec.GridSpec(rows, cols)
    
    for i, (idx, peak, label, decay, var, width) in enumerate(results):
        event_time = idx * hop_length / sr
        samples_per_window = int(window_size * sr)
        start_sample = max(0, int(event_time * sr) - samples_per_window // 2)
        end_sample = min(len(audio), start_sample + samples_per_window)
        
        audio_segment = audio[start_sample:end_sample]
        time_segment = np.arange(start_sample, end_sample) / sr
        
        start_rms_idx = max(0, idx - int((window_size/2) * sr / hop_length))
        end_rms_idx = min(len(rms), idx + int((window_size/2) * sr / hop_length) + 1)
        rms_segment = rms[start_rms_idx:end_rms_idx]
        rms_time = np.arange(start_rms_idx, end_rms_idx) * hop_length / sr
        
        ax = plt.subplot(gs[i])
        ax.plot(time_segment, audio_segment, alpha=0.7, label='波形')
        ax.set_ylabel('振幅')
        ax.legend(loc='upper left')
        
        ax2 = ax.twinx()
        ax2.plot(rms_time, rms_segment, 'r-', alpha=0.8, label='RMS能量')
        ax2.set_ylabel('RMS值', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.legend(loc='upper right')
        
        ax.axvline(event_time, color='purple', linestyle='--', alpha=0.8)
        ax.set_title(f"事件 {i+1}: {label} (衰减率: {decay:.3f}, 宽度: {width})")
        ax.set_xlabel("时间 (秒)")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_rms_with_baseline(rms, baseline, results, sr, hop_length):
    time = np.arange(len(rms)) * hop_length / sr
    plt.figure(figsize=(12, 6))
    plt.plot(time, rms, label="RMS能量", alpha=0.8)
    plt.plot(time, baseline, label="背景基线", color='orange', linestyle='--')
    plt.axhline(rms_threshold, label="检测阈值", color='r', alpha=0.5, linestyle=':')
    
    for idx, peak, label, decay, var, width in results:
        t = idx * hop_length / sr
        color = 'r' if label == '敲' else 'g'
        plt.scatter(t, peak, color=color, s=50, zorder=3)
        plt.text(t, peak, label, color=color, fontsize=9, ha='right')
    
    plt.xlabel("时间 (秒)")
    plt.ylabel("RMS值")
    plt.title("RMS能量与背景基线对比")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main(audio_path, detail_window_size=0.2):
    audio, sr_loaded = load_audio(audio_path)
    global sr
    sr = sr_loaded
    rms = compute_rms(audio)
    baseline = estimate_background_rms(rms)
    events = detect_events(rms)
    results = classify_events(rms, events, baseline)
    
    # 打印结果时增加峰值宽度特征
    print("事件帧号 | RMS值 | 类型 | 衰减比率 | 方差 | 峰值宽度(帧)")
    print("-" * 60)
    for idx, peak, label, decay, var, width in results:
        print(f"{idx:7d} | {peak:.3f} | {label:2s} | {decay:.5f} | {var:.5f} | {width:4d}")
    
    plot_waveform(audio, results, sr, hop_length)
    plot_rms_with_baseline(rms, baseline, results, sr, hop_length)
    plot_event_details(audio, rms, results, sr, hop_length, window_size=detail_window_size)

audio_path = "/Users/zack/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_209ay89m1i9422_6c6e/msg/file/2025-08/f.wav"
if __name__ == "__main__":
    main(audio_path, detail_window_size=0.2)
