import numpy as np
import librosa
import matplotlib.pyplot as plt
import yaml
import argparse  # 新增：用于解析命令行参数
from scipy.spatial.distance import euclidean
from pathlib import Path

# === 辅助函数 ===
def remove_consecutive_duplicates(seq):
    """合并连续重复元素"""
    if not seq:
        return []
    result = [seq[0]]
    for item in seq[1:]:
        if item != result[-1]:
            result.append(item)
    return result

def load_config(config_path):
    """加载YAML配置文件"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print(f"✅ 配置文件加载成功: {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ 配置文件未找到: {config_path}")
        exit(1)
    except Exception as e:
        print(f"❌ 配置文件解析错误: {str(e)}")
        exit(1)

def detect_clap_events(rms_matrix, frame_times, config):
    """检测拍动作"""
    clap_cfg = config["clap"]
    num_channels, num_frames = rms_matrix.shape
    consecutive_clap_frames = 0
    clap_start_frame = None
    clap_times = []

    for j in range(num_frames):
        all_high = np.all(rms_matrix[:, j] >= clap_cfg["rms_threshold"])
        if all_high:
            consecutive_clap_frames += 1
            if consecutive_clap_frames == 1:
                clap_start_frame = j
        else:
            if consecutive_clap_frames >= clap_cfg["min_frames"]:
                clap_end_frame = j - 1
                clap_time = (frame_times[clap_start_frame] + frame_times[clap_end_frame]) / 2
                if not clap_times or (clap_time - clap_times[-1] >= clap_cfg["min_interval"]):
                    clap_times.append(clap_time)
                    print(f"Clap detected @ {clap_time:.2f}s")
            consecutive_clap_frames = 0

    if consecutive_clap_frames >= clap_cfg["min_frames"]:
        clap_time = frame_times[-1]
        if not clap_times or (clap_time - clap_times[-1] >= clap_cfg["min_interval"]):
            clap_times.append(clap_time)
            print(f"Clap detected @ {clap_time:.2f}s")
    return clap_times

def check_condition(condition, first_touch):
    """检查单个条件是否满足"""
    op = condition["operator"]
    a = condition["channel_a"]
    b = condition["channel_b"]
    
    if a not in first_touch or b not in first_touch:
        return False
    
    if op == "before":
        return first_touch[a] < first_touch[b]
    else:
        print(f"⚠️ 不支持的操作符: {op}")
        return False

def determine_mode(touch_events, config, channel_mapping={0:1, 1:2, 2:3, 3:4}):
    """根据YAML配置动态判断模式"""
    modes_cfg = config["modes"]
    if len(touch_events) < 2:
        return None

    first_touch = {}
    for t, ch in touch_events:
        mapped_ch = channel_mapping[ch]
        if mapped_ch not in first_touch:
            first_touch[mapped_ch] = t

    for mode_name, mode_cfg in modes_cfg.items():
        if not mode_cfg["enabled"]:
            continue
        
        conditions = mode_cfg["conditions"]
        logic = mode_cfg["logic"].lower()
        condition_results = [check_condition(cond, first_touch) for cond in conditions]
        
        if logic == "and" and all(condition_results):
            return mode_cfg["name"]
        if logic == "or" and any(condition_results):
            return mode_cfg["name"]

    return None

# === 主程序 ===
if __name__ == "__main__":
    # === 新增：解析命令行参数 ===
    parser = argparse.ArgumentParser(description="音频事件检测程序")
    parser.add_argument("-f", "--file", required=True, help="YAML配置文件路径（例如：config.yaml）")
    args = parser.parse_args()  # 解析参数

    # 加载配置文件（使用命令行传入的路径）
    config = load_config(args.file)
    
    # 加载音频（从配置文件读取路径）
    audio_path = config["audio"]["path"]
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        print(f"✅ 音频文件加载成功: {audio_path}")
    except Exception as e:
        print(f"❌ 音频文件加载失败: {str(e)}")
        exit(1)

    # 确保4个通道
    if y.shape[0] > 4:
        y = y[:4, :]
    num_channels = y.shape[0]
    assert num_channels == 4, "需要4个通道的音频"

    # 解析参数
    frame_cfg = config["frame"]
    touch_cfg = config["touch"]
    clap_cfg = config["clap"]
    post_clap_cfg = config["post_clap"]

    frame_duration = frame_cfg["duration"]
    frame_length = int(sr * frame_duration)
    hop_length = int(frame_length * frame_cfg["hop_length_ratio"])

    # 计算RMS特征
    rms_matrix = []
    frame_times = []
    for i in range(0, y.shape[1] - frame_length, hop_length):
        frame = y[:, i:i+frame_length]
        rms = np.sqrt(np.mean(frame **2, axis=1))
        rms_matrix.append(rms)
        frame_times.append(i / sr)
    rms_matrix = np.array(rms_matrix).T
    frame_times = np.array(frame_times)

    # 检测拍动作
    clap_times = detect_clap_events(rms_matrix, frame_times, config)
    if not clap_times:
        print("❌ 未检测到拍动作，模式检测禁用")
        exit()
    print(f"检测到拍动作数量: {len(clap_times)}")

    # 计算RMS向量变化
    rms_norms = []
    distances = [0]
    for j in range(rms_matrix.shape[1]):
        vec = rms_matrix[:, j]
        norm_vec = vec / (np.linalg.norm(vec) + 1e-8)
        rms_norms.append(norm_vec)
    for j in range(1, len(rms_norms)):
        distances.append(euclidean(rms_norms[j], rms_norms[j-1]))

    # 检测触摸事件并判断模式
    all_touch_events = []
    all_modes = []
    for clap_idx, clap_time in enumerate(clap_times):
        touch_events = []
        last_trigger_times = [-10] * num_channels
        consecutive_counts = [0] * num_channels
        window_start = clap_time + clap_cfg["cooldown"]
        window_end = window_start + post_clap_cfg["active_window"]

        for j in range(len(frame_times)):
            t = frame_times[j]
            if t < window_start:
                continue
            if clap_idx < len(clap_times) - 1 and t >= clap_times[clap_idx + 1]:
                break
            if t > window_end:
                break

            rms = rms_matrix[:, j]
            max_ch = np.argmax(rms)
            max_val = rms[max_ch]
            other_avg = np.mean(np.delete(rms, max_ch))

            reason = []
            if max_val <= touch_cfg["threshold_ratio"] * other_avg:
                reason.append("能量比不足")
            if max_val <= touch_cfg["min_amplitude"]:
                reason.append("振幅过低")

            if not reason:
                consecutive_counts[max_ch] += 1
            else:
                consecutive_counts[max_ch] = 0

            if (consecutive_counts[max_ch] >= touch_cfg["consecutive_required"] and 
                (t - last_trigger_times[max_ch]) > touch_cfg["debounce_time"]):
                touch_events.append((t, max_ch))
                last_trigger_times[max_ch] = t
                consecutive_counts[max_ch] = 0

        all_touch_events.append((clap_time, touch_events))
        mode = determine_mode(touch_events, config) if touch_events else None
        all_modes.append(mode)
        print(f"第 {clap_idx+1} 次拍动作检测到 {len(touch_events)} 个触摸事件，模式: {mode}")

    # 结果汇总
    print("\n===== 结果汇总 =====")
    for i, (clap_time, touch_events) in enumerate(all_touch_events):
        mode = all_modes[i]
        print(f"\n第 {i+1} 次拍动作 @ {clap_time:.2f}s")
        print(f"触摸检测开始时间: {clap_time + clap_cfg['cooldown']:.2f}s")
        print(f"触摸事件数量: {len(touch_events)}")
        if touch_events:
            window_sequences = []
            current_window = [touch_events[0][1]]
            last_event_time = touch_events[0][0]
            for t, ch in touch_events[1:]:
                if t - last_event_time <= post_clap_cfg["sequence_window"]:
                    current_window.append(ch)
                else:
                    window_sequences.append(current_window)
                    current_window = [ch]
                last_event_time = t
            window_sequences.append(current_window)
            simplified = [remove_consecutive_duplicates(seq) for seq in window_sequences]
            print(f"简化序列: {'; '.join(['-'.join(map(str, s)) for s in simplified])}")
        print(f"检测到的模式: {mode if mode else '无有效模式'}")

    # 可视化
    fig, axs = plt.subplots(2 * num_channels, 1, figsize=(14, 4 * num_channels))
    fig.suptitle("Waveforms, RMS & Event Markers", fontsize=16)

    for ch in range(num_channels):
        ax_wave = axs[2 * ch]
        time_axis = np.arange(len(y[ch])) / sr
        ax_wave.plot(time_axis, y[ch], alpha=0.7, color=f"C{ch}")
        ax_wave.set_title(f"Channel {ch} Waveform")
        ax_wave.set_ylabel("Amplitude")
        ax_wave.grid(True, alpha=0.3)

        for i, clap_time in enumerate(clap_times):
            cooldown_end = clap_time + clap_cfg["cooldown"]
            window_end = cooldown_end + post_clap_cfg["active_window"]
            if i < len(clap_times) - 1:
                window_end = min(window_end, clap_times[i+1])
            ax_wave.axvline(clap_time, color='purple', linestyle='-', alpha=0.8, label='Clap' if i == 0 else "")
            ax_wave.axvspan(clap_time, cooldown_end, color='red', alpha=0.1, label='Cooldown' if i == 0 else "")
            ax_wave.axvspan(cooldown_end, window_end, color='green', alpha=0.1, label='Detection Window' if i == 0 else "")

        for clap_idx, (clap_time, touch_events) in enumerate(all_touch_events):
            ch_events = [t for t, c in touch_events if c == ch]
            for t in ch_events:
                ax_wave.axvline(t, color='blue', linestyle='--', alpha=0.6, label='Touch' if clap_idx == 0 else "")
        ax_wave.legend()

        ax_rms = axs[2 * ch + 1]
        ax_rms.plot(frame_times, rms_matrix[ch], alpha=0.8, color=f"C{ch}")
        ax_rms.axhline(y=touch_cfg["min_amplitude"], color='green', linestyle='--', label='Min Amplitude')
        ax_rms.axhline(y=clap_cfg["rms_threshold"], color='purple', linestyle='--', label='Clap Threshold')
        ax_rms.set_title(f"Channel {ch} RMS Energy")
        ax_rms.set_ylabel("RMS")
        ax_rms.grid(True, alpha=0.3)
        ax_rms.legend()

    axs[-1].set_xlabel("Time (s)")

    plt.figure(figsize=(16, 6))
    plt.ylim(-1, num_channels)
    plt.yticks(range(num_channels), [f"Channel {ch}" for ch in range(num_channels)])
    plt.title("Event Sequence & Modes", fontsize=14)
    plt.xlabel("Time (s)")
    plt.grid(True, alpha=0.3)

    for i, clap_time in enumerate(clap_times):
        cooldown_end = clap_time + clap_cfg["cooldown"]
        window_end = cooldown_end + post_clap_cfg["active_window"]
        if i < len(clap_times) - 1:
            window_end = min(window_end, clap_times[i+1])
        plt.axvline(clap_time, color='purple', linestyle='-', alpha=0.8, label='Clap' if i == 0 else "")
        plt.axvspan(clap_time, cooldown_end, color='red', alpha=0.1, label='Cooldown' if i == 0 else "")
        plt.axvspan(cooldown_end, window_end, color='green', alpha=0.1, label='Detection Window' if i == 0 else "")
        mode = all_modes[i]
        if mode:
            plt.text(cooldown_end, num_channels - 0.5, f"Mode: {mode}", color='green', fontweight='bold')

    for clap_idx, (clap_time, touch_events) in enumerate(all_touch_events):
        for i, (t, ch) in enumerate(touch_events):
            plt.scatter(t, ch, color='blue', s=80, zorder=3, edgecolors='black')
            plt.text(t, ch+0.2, f"E{i+1}", fontsize=8)

    plt.legend()
    plt.tight_layout()
    plt.show()