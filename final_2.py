import numpy as np
import librosa
import matplotlib.pyplot as plt
import yaml
import argparse
from scipy.spatial.distance import euclidean
from pathlib import Path
import time

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

def check_single_channel_condition(touch_events, channel, ratio_threshold, rms_matrix, frame_times):
    """检查触摸事件是否满足单通道RMS高于其他通道的比例条件"""
    valid_triggers = 0
    # 遍历该通道的所有触摸事件
    for t, ch, phase in touch_events:
        if ch != (channel - 1):  # 映射回原始通道（配置中channel是映射后的值）
            continue
        
        # 找到该事件对应的帧索引
        frame_idx = np.argmin(np.abs(frame_times - t))
        if frame_idx >= len(rms_matrix[0]):
            continue
        
        # 当前通道RMS值
        current_rms = rms_matrix[ch, frame_idx]
        # 其他三个通道的RMS平均值
        other_rms = np.mean([rms_matrix[other_ch, frame_idx] for other_ch in range(4) if other_ch != ch])
        
        # 检查是否满足比例条件（当前通道RMS > 其他通道平均值 * 比例阈值）
        if other_rms == 0:
            # 避免除零错误，若其他通道为0则直接判定有效（当前通道有信号）
            valid_triggers += 1
        elif current_rms > other_rms * ratio_threshold:
            valid_triggers += 1
    
    return valid_triggers

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

def determine_mode(touch_events, config, simplified_sequence, rms_matrix, frame_times, channel_mapping={0:1, 1:2, 2:3, 3:4}):
    """扩展支持单通道连拍模式判断"""
    modes_cfg = config["modes"]
    if len(touch_events) < 2 or not simplified_sequence:
        return None

    # 映射简化序列（原有逻辑保持）
    mapped_simplified = [channel_mapping[ch] for ch in simplified_sequence]
    print(f"用于模式判断的映射后简化序列: {mapped_simplified}")

    for mode_name, mode_cfg in modes_cfg.items():
        if not mode_cfg["enabled"]:
            continue
        
        conditions = mode_cfg["conditions"]
        logic = mode_cfg["logic"].lower()
        condition_results = []
        
        for cond in conditions:
            # 处理原有"通道顺序"条件
            if "type" not in cond or cond["type"] == "channel_order":
                a = cond["channel_a"]
                b = cond["channel_b"]
                try:
                    idx_a = mapped_simplified.index(a)
                    idx_b = mapped_simplified.index(b)
                    condition_results.append(idx_a < idx_b)
                except ValueError:
                    condition_results.append(False)
            
            # 处理新增"单通道连拍"条件
            elif cond["type"] == "single_channel":
                target_channel = cond["channel"]  # 映射后的目标通道
                ratio_threshold = cond["ratio_threshold"]
                min_triggers = cond["min_triggers"]
                
                # 检查该通道的有效触发次数（满足RMS比例条件）
                valid_count = check_single_channel_condition(
                    touch_events, 
                    target_channel, 
                    ratio_threshold, 
                    rms_matrix, 
                    frame_times
                )
                # 判断是否达到最小触发次数
                condition_results.append(valid_count >= min_triggers)
        
        # 根据逻辑判断是否满足模式
        if (logic == "and" and all(condition_results)) or (logic == "or" and any(condition_results)):
            print(f"模式 {mode_name} 条件满足: {condition_results}")
            return mode_cfg["name"]

    return None

def print_debug_info(debug_logs, clap_time, window_start, window_end):
    """打印检测窗口中的调试信息"""
    print(f"\n===== 拍动作触发的检测窗口调试信息 =====")
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
    parser = argparse.ArgumentParser(description="支持多次检测的clap触发+触摸识别程序")
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

    # 计算RMS特征（一次计算，多次使用）
    rms_matrix = []
    frame_times = []
    for i in range(0, y.shape[1] - frame_length, hop_length):
        frame = y[:, i:i+frame_length]
        rms = np.sqrt(np.mean(frame **2, axis=1))
        rms_matrix.append(rms)
        frame_times.append(i / sr)
    rms_matrix = np.array(rms_matrix).T
    frame_times = np.array(frame_times)
    print(f"✅ 特征提取完成，共 {len(frame_times)} 帧")

    # 主循环：支持多次检测
    cycle_count = 1
    try:
        while True:
            print(f"\n\n===== 检测周期 {cycle_count} 开始 =====")
            print("等待clap触发信号...")

            # 检测clap事件（每次循环重新检测）
            clap_times = detect_clap_events(rms_matrix, frame_times, config)
            if not clap_times:
                print("❌ 未检测到clap触发信号，3秒后重试...")
                time.sleep(3)
                cycle_count += 1
                continue

            # 取最新的clap作为触发点
            current_clap_time = clap_times[-1]
            print(f"✅ 检测到clap触发信号 @ {current_clap_time:.2f}s")

            # 初始化检测参数（每次循环重置）
            touch_events = []
            last_trigger_times = [-10] * num_channels
            consecutive_counts = [0] * num_channels
            window_start = current_clap_time + clap_cfg["cooldown"]
            window_end = window_start + 5  # 初始5秒窗口
            j = 0  # 帧索引
            debug_logs = []

            # 查找当前clap对应的起始帧索引
            while j < len(frame_times) and frame_times[j] < current_clap_time:
                j += 1

            while j < len(frame_times):
                t = frame_times[j]
                
                # 跳过窗口开始前的帧
                if t < window_start:
                    j += 1
                    continue
                
                # 计算当前窗口阶段
                time_since_start = t - window_start
                middle_window_start = post_clap_cfg["middle_window_start"]
                middle_window_end = middle_window_start + post_clap_cfg["middle_window_duration"]
                
                # 确定当前窗口阶段和对应参数
                if time_since_start < middle_window_start:
                    window_phase = "main"
                    threshold_ratio = touch_cfg["main_window"]["threshold_ratio"]
                    consecutive_required = touch_cfg["main_window"]["consecutive_required"]
                elif time_since_start < middle_window_end:
                    window_phase = "middle"
                    threshold_ratio = touch_cfg["middle_window"]["threshold_ratio"]
                    consecutive_required = touch_cfg["middle_window"]["consecutive_required"]
                else:
                    window_phase = "final"
                    threshold_ratio = touch_cfg["main_window"]["threshold_ratio"]  # 复用主窗口参数
                    consecutive_required = touch_cfg["main_window"]["consecutive_required"]
                
                # 计算当前窗口结束时间
                if window_phase == "main":
                    current_window_end = window_start + post_clap_cfg["main_window_initial"]
                elif window_phase == "middle":
                    current_window_end = window_start + middle_window_end
                else:  # final阶段
                    current_window_end = window_end  # 动态延长的窗口
                
                # 5秒无操作则结束当前检测
                if t > current_window_end:
                    print(f"⏹️ {window_phase}窗口超时，当前检测周期结束 @ {t:.2f}s")
                    break
                
                # 处理当前帧
                rms = rms_matrix[:, j]
                frame_debug = {
                    "time": t,
                    "window_phase": window_phase,
                    "channels": [],
                    "triggered": False,
                    "triggered_channel": None,
                    "consecutive_count": 0
                }

                # 多通道触摸检测（使用当前窗口阶段的参数）
                for ch in range(num_channels):
                    ch_val = rms[ch]
                    other_channels = np.delete(rms, ch)
                    reason = []
                    valid = False

                    # 能量比较模式（使用当前窗口的阈值）
                    if compare_mode == "average":
                        other_avg = np.mean(other_channels) if len(other_channels) > 0 else 0
                        if ch_val > threshold_ratio * other_avg:
                            valid = True
                        else:
                            reason.append(f"能量比不足 (当前: {ch_val:.4f}, 其他平均: {other_avg:.4f}, 阈值: {threshold_ratio})")

                    elif compare_mode == "single_channel":
                        exceeded_count = sum(ch_val > (threshold_ratio * other_val) 
                                            for other_val in other_channels)
                        if exceeded_count >= min_channels_to_exceed:
                            valid = True
                        else:
                            reason.append(f"超过的通道数不足 (当前: {exceeded_count}, 要求: {min_channels_to_exceed}, 阈值: {threshold_ratio})")

                    # 振幅最小值检查
                    if ch_val <= touch_cfg["min_amplitude"]:
                        reason.append(f"振幅过低 (当前: {ch_val:.4f}, 阈值: {touch_cfg['min_amplitude']})")
                        valid = False

                    # 更新连续计数（使用当前窗口的连续帧要求）
                    consecutive_counts[ch] = consecutive_counts[ch] + 1 if valid else 0

                    # 记录调试信息
                    frame_debug["channels"].append({
                        "channel": ch,
                        "rms": ch_val,
                        "valid": valid,
                        "reasons": reason,
                        "consecutive_count": consecutive_counts[ch],
                        "required_consecutive": consecutive_required,
                        "time_since_last_trigger": t - last_trigger_times[ch]
                    })

                    # 触发触摸事件（使用当前窗口的连续帧要求）
                    if (valid and 
                        consecutive_counts[ch] >= consecutive_required and 
                        (t - last_trigger_times[ch]) > touch_cfg["debounce_time"]):
                        touch_events.append((t, ch, window_phase))  # 记录事件所在窗口阶段
                        last_trigger_times[ch] = t
                        consecutive_counts[ch] = 0
                        frame_debug["triggered"] = True
                        frame_debug["triggered_channel"] = ch
                        frame_debug["consecutive_count"] = consecutive_counts[ch]
                        
                        # 动态延长最终窗口
                        if window_phase == "final":
                            window_end = t + post_clap_cfg["final_window_extend"]
                            print(f"触摸事件 @ {t:.2f}s (通道{ch}，{window_phase}窗口)，窗口延长至 {window_end:.2f}s")
                        else:
                            print(f"触摸事件 @ {t:.2f}s (通道{ch}，{window_phase}窗口)")

                debug_logs.append(frame_debug)
                j += 1


            # 输出详细调试信息
            if args.verbose:
                print_debug_info(debug_logs, current_clap_time, window_start, window_end)

            # 生成触摸序列并判断模式
            raw_sequence = [ch for t, ch in touch_events]
            consecutive_simplified = remove_consecutive_duplicates(raw_sequence)
            print(f"\n原始触摸序列: {'-'.join(map(str, raw_sequence)) if raw_sequence else '无'}")
            print(f"连续重复合并后序列: {'-'.join(map(str, consecutive_simplified)) if consecutive_simplified else '无'}")

            # 判断模式
            # 判断模式（修改调用参数）
            mode = determine_mode(
                touch_events, 
                config, 
                simplified_sequence=consecutive_simplified,
                rms_matrix=rms_matrix,  # 新增参数
                frame_times=frame_times  # 新增参数
            ) if touch_events else None

            # 结果汇总
            print("\n===== 检测周期结果 =====")
            print(f"clap触发时间: {current_clap_time:.2f}s")
            print(f"检测窗口: {window_start:.2f}s 至 {window_end:.2f}s")
            print(f"触摸事件数量: {len(touch_events)}")
            print(f"识别到的模式: {mode if mode else '无有效模式'}")

            # 可视化当前检测周期结果
            fig, axs = plt.subplots(2 * num_channels, 1, figsize=(14, 4 * num_channels))
            fig.suptitle(f"检测周期 {cycle_count} - 波形与RMS能量", fontsize=16)

            for ch in range(num_channels):
                # 波形图
                ax_wave = axs[2 * ch]
                time_axis = np.arange(len(y[ch])) / sr
                ax_wave.plot(time_axis, y[ch], alpha=0.7, color=f"C{ch}")
                ax_wave.set_title(f"通道 {ch} 波形")
                ax_wave.set_ylabel("振幅")
                ax_wave.grid(True, alpha=0.3)
                
                # 标记关键时间点
                ax_wave.axvline(current_clap_time, color='purple', linestyle='-', alpha=0.8, label='Clap触发')
                ax_wave.axvspan(current_clap_time, window_start, color='red', alpha=0.1, label='冷却期')
                ax_wave.axvspan(window_start, window_end, color='green', alpha=0.1, label='检测窗口')
                
                # 标记触摸事件
                ch_events = [t for t, c in touch_events if c == ch]
                for t in ch_events:
                    ax_wave.axvline(t, color='blue', linestyle='--', alpha=0.6, label='触摸事件')
                ax_wave.legend()

                # RMS能量图
                ax_rms = axs[2 * ch + 1]
                ax_rms.plot(frame_times, rms_matrix[ch], alpha=0.8, color=f"C{ch}")
                ax_rms.axhline(y=touch_cfg["min_amplitude"], color='green', linestyle='--', label='最小振幅阈值')
                ax_rms.axhline(y=clap_cfg["rms_threshold"], color='purple', linestyle='--', label='Clap阈值')
                ax_rms.set_title(f"通道 {ch} RMS能量")
                ax_rms.set_ylabel("RMS")
                ax_rms.grid(True, alpha=0.3)
                ax_rms.legend()

            axs[-1].set_xlabel("时间 (s)")

            # 事件序列图
            plt.figure(figsize=(16, 6))
            plt.ylim(-1, num_channels)
            plt.yticks(range(num_channels), [f"通道 {ch}" for ch in range(num_channels)])
            plt.title(f"检测周期 {cycle_count} - 触摸事件序列", fontsize=14)
            plt.xlabel("时间 (s)")
            plt.grid(True, alpha=0.3)

            # 标记时间区间
            plt.axvline(current_clap_time, color='purple', linestyle='-', alpha=0.8, label='Clap触发')
            plt.axvspan(current_clap_time, window_start, color='red', alpha=0.1, label='冷却期')
            plt.axvspan(window_start, window_end, color='green', alpha=0.1, label='检测窗口')

            # 标记触摸事件
            for i, (t, ch) in enumerate(touch_events):
                plt.scatter(t, ch, color='blue', s=80, zorder=3, edgecolors='black')
                plt.text(t, ch+0.2, f"E{i+1}", fontsize=8)

            # 标记识别到的模式
            if mode:
                plt.text(window_start, num_channels - 0.5, f"识别模式: {mode}", color='green', fontweight='bold')
            else:
                plt.text(window_start, num_channels - 0.5, "未识别到有效模式", color='red', fontweight='bold')

            plt.legend()
            plt.tight_layout()
            plt.show()

            # 准备下一个检测周期
            cycle_count += 1
            if not mode:
                print("\n未检测到有效模式，准备重新监听clap触发信号...")
            else:
                print("\n检测完成，3秒后重新监听clap触发信号...")
                time.sleep(3)

    except KeyboardInterrupt:
        print("\n\n用户中断程序，退出检测")
        exit(0)
