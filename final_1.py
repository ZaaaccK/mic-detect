import numpy as np
import librosa
import matplotlib.pyplot as plt
import yaml
import argparse
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

def determine_mode(touch_events, config, simplified_sequence, channel_mapping={0:1, 1:2, 2:3, 3:4}):
    """基于模式简化序列判断模式（核心修改）"""
    modes_cfg = config["modes"]
    if len(touch_events) < 2 or not simplified_sequence:
        return None

    # 将简化序列映射为目标通道编号（原始→映射后）
    mapped_simplified = [channel_mapping[ch] for ch in simplified_sequence]
    print(f"用于模式判断的映射后简化序列: {mapped_simplified}")

    for mode_name, mode_cfg in modes_cfg.items():
        if not mode_cfg["enabled"]:
            continue
        
        conditions = mode_cfg["conditions"]
        logic = mode_cfg["logic"].lower()
        condition_results = []
        
        for cond in conditions:
            a = cond["channel_a"]  # 模式中定义的通道a（映射后）
            b = cond["channel_b"]  # 模式中定义的通道b（映射后）
            
            # 检查a是否在b之前出现在简化序列中
            try:
                idx_a = mapped_simplified.index(a)  # a在序列中的位置
                idx_b = mapped_simplified.index(b)  # b在序列中的位置
                condition_results.append(idx_a < idx_b)  # a位置在b前则满足条件
            except ValueError:
                # 若a或b未出现在序列中，条件不满足
                condition_results.append(False)
        
        # 根据逻辑判断是否满足模式
        if (logic == "and" and all(condition_results)) or (logic == "or" and any(condition_results)):
            print(f"模式 {mode_name} 条件满足: {condition_results}")
            return mode_cfg["name"]

    return None

def print_debug_info(debug_logs, clap_idx, clap_time, window_start, window_end):
    """打印检测窗口中的调试信息"""
    print(f"\n===== 第 {clap_idx+1} 次拍动作的检测窗口调试信息 =====")
    print(f"拍动作时间: {clap_time:.2f}s")
    print(f"检测窗口: {window_start:.2f}s 至 {window_end:.2f}s")
    print(f"总帧数: {len(debug_logs)}")
    
    # 按通道分组统计
    channel_stats = {}
    for log in debug_logs:
        for ch in log["channels"]:
            if ch["channel"] not in channel_stats:
                channel_stats[ch["channel"]] = {
                    "total_frames": 0,
                    "valid_frames": 0,
                    "reasons": {},
                    "near_trigger": 0
                }
            
            stats = channel_stats[ch["channel"]]
            stats["total_frames"] += 1
            
            if ch["valid"]:
                stats["valid_frames"] += 1
                if ch["consecutive_count"] >= ch["required_consecutive"] - 1:
                    stats["near_trigger"] += 1
            else:
                for reason in ch["reasons"]:
                    if reason not in stats["reasons"]:
                        stats["reasons"][reason] = 0
                    stats["reasons"][reason] += 1
    
    # 打印各通道统计信息
    for ch in sorted(channel_stats.keys()):
        stats = channel_stats[ch]
        print(f"\n通道 {ch} 统计:")
        print(f"  总帧数: {stats['total_frames']}")
        print(f"  有效帧数: {stats['valid_frames']} ({stats['valid_frames']/stats['total_frames']*100:.1f}%)")
        print(f"  接近触发的帧数: {stats['near_trigger']}")
        print(f"  未满足条件的原因:")
        for reason, count in stats["reasons"].items():
            print(f"    - {reason}: {count} 次 ({count/stats['total_frames']*100:.1f}%)")
    
    # 打印触发事件的详细信息
    trigger_events = [log for log in debug_logs if log["triggered"]]
    if trigger_events:
        print(f"\n触发事件详情 ({len(trigger_events)} 个):")
        for event in trigger_events:
            print(f"  时间: {event['time']:.4f}s, 通道: {event['triggered_channel']}, 连续帧数: {event['consecutive_count']}")

# === 主程序 ===
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="基于模式简化序列的音频事件检测程序")
    parser.add_argument("-f", "--file", required=True, help="YAML配置文件路径（例如：config.yaml）")
    parser.add_argument("-v", "--verbose", action="store_true", help="启用详细调试日志")
    args = parser.parse_args()

    # 加载配置文件
    config = load_config(args.file)
    
    # 加载音频
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
    compare_mode = touch_cfg.get("compare_mode", "average")
    min_channels_to_exceed = touch_cfg.get("min_channels_to_exceed", 1)
    
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

    # 检测触摸事件并判断模式（基于模式简化序列）
    all_touch_events = []
    all_modes = []
    for clap_idx, clap_time in enumerate(clap_times):
        touch_events = []
        last_trigger_times = [-10] * num_channels
        consecutive_counts = [0] * num_channels
        window_start = clap_time + clap_cfg["cooldown"]
        window_end = window_start + post_clap_cfg["active_window"]
        
        # 用于记录调试信息
        debug_logs = []

        for j in range(len(frame_times)):
            t = frame_times[j]
            if t < window_start:
                continue
            if clap_idx < len(clap_times) - 1 and t >= clap_times[clap_idx + 1]:
                break
            if t > window_end:
                break

            rms = rms_matrix[:, j]
            frame_debug = {
                "time": t,
                "channels": [],
                "triggered": False,
                "triggered_channel": None,
                "consecutive_count": 0
            }

            # 多通道独立检测
            for ch in range(num_channels):
                ch_val = rms[ch]
                other_channels = np.delete(rms, ch)
                reason = []
                valid = False

                # 两种能量比较模式
                if compare_mode == "average":
                    other_avg = np.mean(other_channels) if len(other_channels) > 0 else 0
                    if ch_val <= touch_cfg["threshold_ratio"] * other_avg:
                        reason.append(f"能量比不足 (当前: {ch_val:.4f}, 其他平均: {other_avg:.4f}, 阈值: {touch_cfg['threshold_ratio']})")
                    if ch_val > touch_cfg["threshold_ratio"] * other_avg:
                        valid = True

                elif compare_mode == "single_channel":
                    exceeded_count = sum(ch_val > (touch_cfg["threshold_ratio"] * other_val) 
                                        for other_val in other_channels)
                    if exceeded_count < min_channels_to_exceed:
                        reason.append(f"超过的通道数不足 (当前: {exceeded_count}, 要求: {min_channels_to_exceed}, 阈值: {touch_cfg['threshold_ratio']})")
                    else:
                        valid = True

                # 检查振幅最小值
                if ch_val <= touch_cfg["min_amplitude"]:
                    reason.append(f"振幅过低 (当前: {ch_val:.4f}, 阈值: {touch_cfg['min_amplitude']})")
                    valid = False

                # 更新连续计数
                if valid:
                    consecutive_counts[ch] += 1
                else:
                    consecutive_counts[ch] = 0

                # 记录通道调试信息
                frame_debug["channels"].append({
                    "channel": ch,
                    "rms": ch_val,
                    "other_channels": other_channels.tolist(),
                    "valid": valid,
                    "reasons": reason,
                    "consecutive_count": consecutive_counts[ch],
                    "required_consecutive": touch_cfg["consecutive_required"],
                    "time_since_last_trigger": t - last_trigger_times[ch]
                })

                # 检查是否触发事件
                if (valid and 
                    consecutive_counts[ch] >= touch_cfg["consecutive_required"] and 
                    (t - last_trigger_times[ch]) > touch_cfg["debounce_time"]):
                    touch_events.append((t, ch))
                    last_trigger_times[ch] = t
                    frame_debug["triggered"] = True
                    frame_debug["triggered_channel"] = ch
                    frame_debug["consecutive_count"] = consecutive_counts[ch]
                    consecutive_counts[ch] = 0

            # 添加帧调试信息
            debug_logs.append(frame_debug)

        # 启用详细日志
        if args.verbose:
            print_debug_info(debug_logs, clap_idx, clap_time, window_start, window_end)

        # 生成模式简化序列（用于判断模式）
        raw_sequence = [ch for t, ch in touch_events]
        consecutive_simplified = remove_consecutive_duplicates(raw_sequence)
        pattern_window = post_clap_cfg.get("pattern_window", 3)
        pattern_simplified = []
        i = 0
        while i < len(consecutive_simplified):
            if i + pattern_window <= len(consecutive_simplified):
                current_pattern = consecutive_simplified[i:i+pattern_window]
                if i + 2*pattern_window <= len(consecutive_simplified) and \
                consecutive_simplified[i+pattern_window:i+2*pattern_window] == current_pattern:
                    pattern_simplified.append(f"[{'-'.join(map(str, current_pattern))}]*2")
                    i += 2*pattern_window
                    continue
            pattern_simplified.append(str(consecutive_simplified[i]))
            i += 1

        # 基于模式简化序列判断模式（核心修改：传入simplified_sequence）
        mode = determine_mode(
            touch_events, 
            config, 
            simplified_sequence=consecutive_simplified  # 使用连续重复合并后的序列作为判断依据
        ) if touch_events else None
        
        all_touch_events.append((clap_time, touch_events))
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
            raw_sequence = [ch for t, ch in touch_events]
            print(f"原始序列: {'-'.join(map(str, raw_sequence))}")
            
            consecutive_simplified = remove_consecutive_duplicates(raw_sequence)
            print(f"连续重复合并: {'-'.join(map(str, consecutive_simplified))}")
            
            pattern_window = post_clap_cfg.get("pattern_window", 3)
            pattern_simplified = []
            i_seq = 0
            while i_seq < len(consecutive_simplified):
                if i_seq + pattern_window <= len(consecutive_simplified):
                    current_pattern = consecutive_simplified[i_seq:i_seq+pattern_window]
                    if i_seq + 2*pattern_window <= len(consecutive_simplified) and \
                    consecutive_simplified[i_seq+pattern_window:i_seq+2*pattern_window] == current_pattern:
                        pattern_simplified.append(f"[{'-'.join(map(str, current_pattern))}]*2")
                        i_seq += 2*pattern_window
                        continue
                pattern_simplified.append(str(consecutive_simplified[i_seq]))
                i_seq += 1
            print(f"模式简化序列: {'-'.join(pattern_simplified)}")

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
