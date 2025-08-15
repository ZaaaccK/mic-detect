import numpy as np
import argparse
import yaml
import librosa
import matplotlib.pyplot as plt
import os
import sys
from detect_touch_pattern import (
    init_detector_state,
    detect_touch_pattern
)


def read_yaml(config_path):
    """读取YAML配置文件"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ 配置文件错误: {str(e)}")
        sys.exit(1)

def visualize_results(audio, sr, results, config, state):
    """可视化检测结果"""
    num_channels = config["audio"].get("num_channels", 4)
    frame_len = int(config["frame"]["duration"] * sr)
    
    # 创建画布
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("触摸模式检测结果", fontsize=16)
    
    # 1. 原始音频波形
    time_axis = np.arange(len(audio[0])) / sr if audio.ndim > 1 else np.arange(len(audio)) / sr
    for ch in range(num_channels):
        axes[0].plot(time_axis, audio[ch] if audio.ndim > 1 else audio, 
                    alpha=0.7, label=f"通道 {ch}")
    axes[0].set_title("原始音频波形")
    axes[0].set_xlabel("时间 (秒)")
    axes[0].set_ylabel("幅度")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 2. RMS能量
    rms_data = np.array([r["rms"] for r in results])
    segment_time = np.array([r["time"] for r in results])
    for ch in range(num_channels):
        axes[1].plot(segment_time, rms_data[:, ch], label=f"通道 {ch} RMS")
    axes[1].axhline(config["touch"]["min_amplitude"], color='r', linestyle='--', 
                   label="最小振幅阈值")
    axes[1].set_title("各通道RMS能量")
    axes[1].set_xlabel("时间 (秒)")
    axes[1].set_ylabel("RMS值")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # 3. 检测结果
    events = [r for r in results if r["trigger_event"]]
    if events:
        event_times = [e["time"] for e in events]
        event_channels = [e["trigger_event"][1] for e in events]
        axes[2].scatter(event_times, event_channels, color='g', s=80, 
                       edgecolors='black', label="触摸事件")
        
        # 标记模式
        mode_changes = []
        prev_mode = None
        for r in results:
            if r["current_mode"] and r["current_mode"] != prev_mode:
                mode_changes.append((r["time"], r["current_mode"]))
                prev_mode = r["current_mode"]
        
        for t, mode in mode_changes:
            axes[2].text(t, num_channels - 0.5, f"模式: {mode}", 
                        color='blue', fontweight='bold')
    
    axes[2].set_ylim(-1, num_channels)
    axes[2].set_yticks(range(num_channels))
    axes[2].set_yticklabels([f"通道 {ch}" for ch in range(num_channels)])
    axes[2].set_title("触摸事件与模式检测")
    axes[2].set_xlabel("时间 (秒)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("touch_detection_results.png", dpi=150)
    print("检测结果图像已保存为 touch_detection_results.png")
    plt.close()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="触摸模式检测程序")
    parser.add_argument('-f', '--file', type=str, metavar="", required=True,
                        help='配置文件路径（YAML格式）')
    args = parser.parse_args()

    # 读取配置文件
    config = read_yaml(args.file)

    # 参数检查
    required_keys = ["wav_path", "frame", "touch", "modes", "audio"]
    for key in required_keys:
        if key not in config:
            sys.exit(f"配置文件缺少必要键: '{key}'")
    
    # 检查音频文件
    if not os.path.isfile(config["wav_path"]):
        sys.exit(f"音频文件不存在: {config['wav_path']}")

    # 加载音频
    try:
        audio, sr = librosa.load(
            config["wav_path"], 
            sr=None, 
            mono=False  # 保持多通道
        )
        print(f"✅ 成功加载音频: {config['wav_path']} (采样率: {sr})")
    except Exception as e:
        sys.exit(f"❌ 音频加载失败: {str(e)}")

    # 确保音频通道数正确
    num_channels = config["audio"].get("num_channels", 4)
    if audio.ndim == 1:
        audio = np.tile(audio, (num_channels, 1))  # 单通道转多通道
    if audio.shape[0] > num_channels:
        audio = audio[:num_channels, :]  # 截断多余通道

    # 初始化检测器状态和结果列表
    detector_state = init_detector_state(config)
    results = []
    frame_len = int(config["frame"]["duration"] * sr)
    hop_len = int(frame_len * config["frame"]["hop_length_ratio"])

    # # 逐帧处理音频
    # print("开始检测触摸模式...")
    # for i in range(0, audio.shape[1] - frame_len, hop_len):
    #     # 提取当前帧
    #     frame = audio[:, i:i + frame_len]
        
    #     # 调用核心检测函数
    #     result = detect_touch_pattern(frame, config, detector_state)
    #     results.append(result)
        
    #     # 打印检测到的事件
    #     if result["trigger_event"]:
    #         t, ch = result["trigger_event"]
    #         print(f"[{t:.2f}s] 检测到触摸事件 - 通道 {ch} | 序列: {result['simplified_sequence']}")
    # 逐帧处理音频（添加6秒限制）
    print("开始检测触摸模式（仅处理前6秒）...")
    max_process_time = 6.0  # 最大处理时间：6秒
    frame_len = int(config["frame"]["duration"] * sr)
    hop_len = int(frame_len * config["frame"]["hop_length_ratio"])

    for i in range(0, audio.shape[1] - frame_len, hop_len):
        # 计算当前帧的开始时间（用于判断是否超过6秒）
        current_frame_time = i / sr  # 当前帧的起始时间（秒）
        
        # 如果当前帧开始时间已超过6秒，停止处理
        if current_frame_time >= max_process_time:
            print(f"已处理完前{max_process_time}秒音频，停止处理")
            break
        
        # 提取当前帧
        frame = audio[:, i:i + frame_len]
        
        # 调用核心检测函数
        result = detect_touch_pattern(frame, config, detector_state)
        results.append(result)
        
        # 打印检测到的事件
        if result["trigger_event"]:
            t, ch = result["trigger_event"]
            print(f"[{t:.2f}s] 检测到触摸事件 - 通道 {ch} | 序列: {result['simplified_sequence']}")

    # 输出最终结果
    print("\n===== 检测完成 =====")
    print(f"总触摸事件数: {len(detector_state['touch_events'])}")
    print(f"最终序列: {detector_state['simplified_sequence']}")
    print(f"最终识别模式: {detector_state['current_mode'] or '无'}")

    # 可视化结果
    visualize_results(audio, sr, results, config, detector_state)

if __name__ == '__main__':
    main()
