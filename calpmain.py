import numpy as np
import argparse
from module.utils.yaml_read import read_yaml
from clap import isClap
import librosa
import matplotlib.pyplot as plt
import os
import sys

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['SimHei']       # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False         # 正常显示负号

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Novelty detection training")
    parser.add_argument('-f', '--file', type=str, metavar="", required=True,
                        help='path of config.yaml file')
    args = parser.parse_args()

    config_file = read_yaml(args.file)

    # 参数检查
    for key in ["wav_path", "frame_len", "threshold"]:
        if key not in config_file:
            sys.exit(f"Config file missing '{key}' key.")
    if not os.path.isfile(config_file["wav_path"]):
        sys.exit(f"Wav file not found: {config_file['wav_path']}")

    # 读音频
    audio, sr = librosa.load(config_file["wav_path"], sr=None)

    # 分段检测
    results = []
    for i in range(0, len(audio), config_file["frame_len"]):
        frame = audio[i:i + config_file["frame_len"]]
        result = isClap(frame, config_file["threshold"])
        results.append(result)

    # 画布
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # 图1：原始时域
    time_axis = np.arange(len(audio)) / sr
    axes[0].plot(time_axis, audio, color="blue")
    axes[0].set_title("原始音频时域图")
    axes[0].set_xlabel("时间 (秒)")
    axes[0].set_ylabel("幅度")

    # 图2：RMS检测结果
    segment_time = np.arange(len(results)) * (config_file["frame_len"] / sr)
    axes[1].stem(segment_time, results, basefmt=" ", linefmt="r-", markerfmt="ro", use_line_collection=True)
    axes[1].set_title("RMS 拍手检测结果")
    axes[1].set_xlabel("时间 (秒)")
    axes[1].set_ylabel("检测结果 (1=拍手, 0=无)")

    plt.tight_layout()
    plt.savefig("clap_detection.png", dpi=150)
    plt.close()

    print("图像已保存为 clap_detection.png")
    print('end')
