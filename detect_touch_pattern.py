import numpy as np

def remove_consecutive_duplicates(seq):
    """合并连续重复元素"""
    if not seq:
        return []
    result = [seq[0]]
    for item in seq[1:]:
        if item != result[-1]:
            result.append(item)
    return result

def determine_mode(touch_events, config, simplified_sequence, channel_mapping={0:1, 1:2, 2:3, 3:4}):
    """基于模式简化序列判断模式，完善主导通道模式检测"""
    modes_cfg = config["modes"]
    if len(touch_events) < 1 or not simplified_sequence:  # 主导模式至少需要1个事件
        return None

    mapped_simplified = [channel_mapping[ch] for ch in simplified_sequence]
    
    for mode_name, mode_cfg in modes_cfg.items():
        if not mode_cfg["enabled"]:
            continue
        
        # 处理主导通道模式
        if mode_name == "dominant_channel":
            # 检查是否配置了必要参数
            if "dominant_ratio" not in mode_cfg:
                continue
                
            # 获取最后一个触发事件的完整数据
            last_event = touch_events[-1]
            last_time, last_ch, last_rms, other_channels_rms = last_event  # 从事件中提取RMS数据
            
            # 检查该通道是否高于所有其他通道指定比例
            all_higher = all(last_rms > mode_cfg["dominant_ratio"] * rms for rms in other_channels_rms)
            if all_higher:
                return f"{mode_cfg['name']} (通道{last_ch})"
            continue
        
        # 原有模式判断逻辑
        conditions = mode_cfg["conditions"]
        logic = mode_cfg["logic"].lower()
        condition_results = []
        
        for cond in conditions:
            a, b = cond["channel_a"], cond["channel_b"]
            try:
                idx_a = mapped_simplified.index(a)
                idx_b = mapped_simplified.index(b)
                condition_results.append(idx_a < idx_b)
            except ValueError:
                condition_results.append(False)
        
        if (logic == "and" and all(condition_results)) or (logic == "or" and any(condition_results)):
            return mode_cfg["name"]
    return None

def init_detector_state(config):
    """初始化检测器状态"""
    num_channels = config["audio"].get("num_channels", 4)
    return {
        "consecutive_counts": [0] * num_channels,
        "last_trigger_times": [-100] * num_channels,
        "touch_events": [],  # 存储格式: [(时间, 通道, 本通道RMS, 其他通道RMS列表), ...]
        "simplified_sequence": [],
        "current_mode": None,
        "processed_frames": 0
    }

def detect_touch_pattern(frame, config, detector_state):
    """
    单通道模式下的触摸检测核心函数
    完善主导通道模式的支持
    """
    # 从配置获取参数
    touch_cfg = config["touch"]
    num_channels = config["audio"].get("num_channels", 4)
    frame_duration = config["frame"]["duration"]
    
    # 计算当前时间和RMS能量
    current_time = detector_state["processed_frames"] * frame_duration
    detector_state["processed_frames"] += 1
    rms = np.sqrt(np.mean(frame **2, axis=1))  # 计算各通道RMS
    
    trigger_event = None
    compare_mode = touch_cfg.get("compare_mode", "average")
    min_exceed = touch_cfg.get("min_channels_to_exceed", 2)

    for ch in range(num_channels):
        ch_val = rms[ch]
        other_channels = np.delete(rms, ch)  # 获取其他所有通道
        valid = False

        # 单通道模式判断逻辑
        if compare_mode == "single_channel":
            exceeded_count = sum(ch_val > touch_cfg["threshold_ratio"] * other_val 
                               for other_val in other_channels)
            valid = exceeded_count >= min_exceed and ch_val > touch_cfg["min_amplitude"]

        # 更新连续计数
        if valid:
            detector_state["consecutive_counts"][ch] += 1
        else:
            detector_state["consecutive_counts"][ch] = 0

        # 检查是否触发事件
        if (valid and 
            detector_state["consecutive_counts"][ch] >= touch_cfg["consecutive_required"] and 
            (current_time - detector_state["last_trigger_times"][ch]) > touch_cfg["debounce_time"]):
            
            # 记录事件，保存完整的RMS数据（关键修改）
            detector_state["touch_events"].append((
                current_time, 
                ch, 
                ch_val,  # 本通道RMS值
                other_channels.tolist()  # 其他通道RMS值（转为列表方便存储）
            ))
            detector_state["last_trigger_times"][ch] = current_time
            detector_state["consecutive_counts"][ch] = 0
            trigger_event = (current_time, ch)

            # 更新简化序列
            raw_sequence = [c for t, c, _, _ in detector_state["touch_events"]]
            detector_state["simplified_sequence"] = remove_consecutive_duplicates(raw_sequence)

    # 检测模式
    if trigger_event:
        detector_state["current_mode"] = determine_mode(
            detector_state["touch_events"],
            config,
            detector_state["simplified_sequence"]
        )

    return {
        "time": current_time,
        "rms": rms,
        "trigger_event": trigger_event,
        "simplified_sequence": detector_state["simplified_sequence"],
        "current_mode": detector_state["current_mode"]
    }
