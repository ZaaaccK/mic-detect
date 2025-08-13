import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# === Helper Functions ===
def remove_consecutive_duplicates(seq):
    """Merge consecutive duplicate elements"""
    if not seq:
        return []
    result = [seq[0]]
    for item in seq[1:]:
        if item != result[-1]:
            result.append(item)
    return result

def detect_clap_events(rms_matrix, frame_times, clap_rms_threshold, clap_min_frames=2, min_clap_interval=1.0):
    """Detect multiple clap events"""
    num_channels, num_frames = rms_matrix.shape
    consecutive_clap_frames = 0
    clap_start_frame = None
    clap_times = []

    for j in range(num_frames):
        all_high = np.all(rms_matrix[:, j] >= clap_rms_threshold)
        
        if all_high:
            consecutive_clap_frames += 1
            if consecutive_clap_frames == 1:
                clap_start_frame = j
        else:
            if consecutive_clap_frames >= clap_min_frames:
                clap_end_frame = j - 1
                clap_time = (frame_times[clap_start_frame] + frame_times[clap_end_frame]) / 2
                if not clap_times or (clap_time - clap_times[-1] >= min_clap_interval):
                    clap_times.append(clap_time)
                    print(f"Clap detected @ {clap_time:.2f}s")
            consecutive_clap_frames = 0

    if consecutive_clap_frames >= clap_min_frames:
        clap_time = frame_times[-1]
        if not clap_times or (clap_time - clap_times[-1] >= min_clap_interval):
            clap_times.append(clap_time)
            print(f"Clap detected @ {clap_time:.2f}s")
    
    return clap_times

def determine_mode(touch_events, channel_mapping={0:1, 1:2, 2:3, 3:4}):
    """Determine mode based on touch event order"""
    if len(touch_events) < 2:
        return None

    first_touch = {}
    for t, ch in touch_events:
        mapped_ch = channel_mapping[ch]
        if mapped_ch not in first_touch:
            first_touch[mapped_ch] = t

    # Mode 1: 2 before 1 and 3 before 4
    if 2 in first_touch and 1 in first_touch and 3 in first_touch and 4 in first_touch:
        if (first_touch[2] < first_touch[1]) and (first_touch[3] < first_touch[4]):
            return 1
    # Mode 2: 1 before 2 and 4 before 3
    if 1 in first_touch and 2 in first_touch and 4 in first_touch and 3 in first_touch:
        if (first_touch[1] < first_touch[2]) and (first_touch[4] < first_touch[3]):
            return 2
    # Mode 3: 1 before 4 and 2 before 3
    if 1 in first_touch and 4 in first_touch and 2 in first_touch and 3 in first_touch:
        if (first_touch[1] < first_touch[4]) and (first_touch[2] < first_touch[3]):
            return 3
    # Mode 4: 4 before 1 and 3 before 2
    if 4 in first_touch and 1 in first_touch and 3 in first_touch and 2 in first_touch:
        if (first_touch[4] < first_touch[1]) and (first_touch[3] < first_touch[2]):
            return 4
    return None

# === 1. Load Audio ===
audio_path = "/Users/zack/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_209ay89m1i9422_6c6e/msg/file/2025-08/red14.wav"  # Replace with your audio path
y, sr = librosa.load(audio_path, sr=None, mono=False)

# Ensure 4 channels
if y.shape[0] > 4:
    y = y[:4, :]
num_channels = y.shape[0]
assert num_channels == 4, "Code requires 4 audio channels"

# === 2. Parameter Settings ===
frame_duration = 0.1  # Frame length in seconds
frame_length = int(sr * frame_duration) 
hop_length = frame_length // 2

# Touch event parameters
threshold_ratio = 1.8
min_amplitude = 0.05
consecutive_required = 1
debounce_time = 0.1

# Clap detection parameters
clap_rms_threshold = 0.3  # Adjust based on your audio
clap_min_frames = 2       # Minimum consecutive frames for clap
min_clap_interval = 1.0   # Minimum interval between claps (seconds)
clap_cooldown = 0.5       # 新增：拍后静默期（秒），此期间不检测触摸动作

# Post-clap detection parameters
active_window_after_clap = 3.0  # Detection window after clap (seconds)
window_size = 0.35              # Touch event grouping window

# === 3. Calculate RMS Features ===
rms_matrix = []
frame_times = []

for i in range(0, y.shape[1] - frame_length, hop_length):
    frame = y[:, i:i+frame_length]
    rms = np.sqrt(np.mean(frame **2, axis=1))  # Calculate RMS for each channel
    rms_matrix.append(rms)
    frame_times.append(i / sr)

rms_matrix = np.array(rms_matrix).T  # Shape: (num_channels, num_frames)
frame_times = np.array(frame_times)

# === 4. Detect All Clap Events ===
clap_times = detect_clap_events(
    rms_matrix, 
    frame_times, 
    clap_rms_threshold=clap_rms_threshold,
    clap_min_frames=clap_min_frames,
    min_clap_interval=min_clap_interval
)

if not clap_times:
    print("❌ No clap events detected. Mode detection disabled.")
    exit()
print(f"Total claps detected: {len(clap_times)}")

# === 5. Calculate RMS Vector Changes ===
rms_norms = []
distances = [0]
for j in range(rms_matrix.shape[1]):
    vec = rms_matrix[:, j]
    norm_vec = vec / (np.linalg.norm(vec) + 1e-8)
    rms_norms.append(norm_vec)

for j in range(1, len(rms_norms)):
    distances.append(euclidean(rms_norms[j], rms_norms[j-1]))

# === 6. Detect Touch Events for Each Clap ===
all_touch_events = []  # Stores (clap_time, [touch_events...]) for each clap
all_modes = []         # Stores detected mode for each clap

for clap_idx, clap_time in enumerate(clap_times):
    touch_events = []
    last_trigger_times = [-10] * num_channels
    consecutive_counts = [0] * num_channels
    window_start = clap_time + clap_cooldown  # 触摸检测窗口起始时间（拍后静默期结束）
    window_end = window_start + active_window_after_clap  # 触摸检测窗口结束时间

    for j in range(len(frame_times)):
        t = frame_times[j]
        # 只处理静默期之后、窗口结束之前的事件，且不与下一个拍动作重叠
        if t < window_start:  # 新增：跳过拍后静默期内的事件
            continue
        if clap_idx < len(clap_times) - 1 and t >= clap_times[clap_idx + 1]:
            break
        if t > window_end:
            break

        rms = rms_matrix[:, j]
        max_ch = np.argmax(rms)
        max_val = rms[max_ch]
        other_avg = np.mean(np.delete(rms, max_ch))

        # Check touch conditions
        reason = []
        if max_val <= threshold_ratio * other_avg:
            reason.append("insufficient ratio")
        if max_val <= min_amplitude:
            reason.append("low amplitude")

        if not reason:
            consecutive_counts[max_ch] += 1
        else:
            consecutive_counts[max_ch] = 0

        # Trigger valid touch event
        if (consecutive_counts[max_ch] >= consecutive_required and 
            (t - last_trigger_times[max_ch]) > debounce_time):
            touch_events.append((t, max_ch))
            last_trigger_times[max_ch] = t
            consecutive_counts[max_ch] = 0

    all_touch_events.append((clap_time, touch_events))
    # Determine mode for current clap
    mode = determine_mode(touch_events) if touch_events else None
    all_modes.append(mode)
    print(f"Clap {clap_idx+1} detected {len(touch_events)} touch events, Mode: {mode}")

# === 7. Result Summary ===
print("\n===== Final Results Summary =====")
for i, (clap_time, touch_events) in enumerate(all_touch_events):
    mode = all_modes[i]
    print(f"\nClap {i+1} @ {clap_time:.2f}s")
    print(f"Touch detection starts at: {clap_time + clap_cooldown:.2f}s")  # 显示触摸检测开始时间
    print(f"Number of touch events: {len(touch_events)}")
    if touch_events:
        # Generate simplified sequences
        window_sequences = []
        current_window = [touch_events[0][1]]
        last_event_time = touch_events[0][0]
        for t, ch in touch_events[1:]:
            if t - last_event_time <= window_size:
                current_window.append(ch)
            else:
                window_sequences.append(current_window)
                current_window = [ch]
            last_event_time = t
        window_sequences.append(current_window)
        simplified = [remove_consecutive_duplicates(seq) for seq in window_sequences]
        print(f"Simplified sequences: {'; '.join(['-'.join(map(str, s)) for s in simplified])}")
    print(f"Detected mode: {mode if mode else 'No valid mode'}")

# === 8. Visualization ===
# Figure 1: Channel waveforms and event markers
fig, axs = plt.subplots(2 * num_channels, 1, figsize=(14, 4 * num_channels))
fig.suptitle("Waveforms, RMS & Event Markers (With Clap Cooldown)", fontsize=16)

for ch in range(num_channels):
    # Waveform subplot
    ax_wave = axs[2 * ch]
    time_axis = np.arange(len(y[ch])) / sr
    ax_wave.plot(time_axis, y[ch], alpha=0.7, color=f"C{ch}")
    ax_wave.set_title(f"Channel {ch} Waveform")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.grid(True, alpha=0.3)
    
    # Mark all claps, cooldown periods and detection windows
    for i, clap_time in enumerate(clap_times):
        cooldown_end = clap_time + clap_cooldown
        window_end = cooldown_end + active_window_after_clap
        if i < len(clap_times) - 1:
            window_end = min(window_end, clap_times[i+1])
        
        # 标记拍动作
        ax_wave.axvline(clap_time, color='purple', linestyle='-', alpha=0.8, 
                       label='Clap event' if i == 0 else "")
        # 标记拍后静默期（红色阴影）
        ax_wave.axvspan(clap_time, cooldown_end, 
                       color='red', alpha=0.1, 
                       label='Clap cooldown' if i == 0 else "")
        # 标记有效触摸检测窗口（绿色阴影）
        ax_wave.axvspan(cooldown_end, window_end, 
                       color='green', alpha=0.1, 
                       label='Touch detection window' if i == 0 else "")
    
    # Mark touch events for this channel
    for clap_idx, (clap_time, touch_events) in enumerate(all_touch_events):
        ch_events = [t for t, c in touch_events if c == ch]
        for t in ch_events:
            ax_wave.axvline(t, color='blue', linestyle='--', alpha=0.6, label='Touch event' if clap_idx == 0 else "")

    ax_wave.legend()

    # RMS subplot
    ax_rms = axs[2 * ch + 1]
    ax_rms.plot(frame_times, rms_matrix[ch], alpha=0.8, color=f"C{ch}")
    ax_rms.axhline(y=min_amplitude, color='green', linestyle='--', label='Min amplitude threshold')
    ax_rms.axhline(y=clap_rms_threshold, color='purple', linestyle='--', label='Clap threshold')
    ax_rms.set_title(f"Channel {ch} RMS Energy")
    ax_rms.set_ylabel("RMS")
    ax_rms.grid(True, alpha=0.3)
    ax_rms.legend()

axs[-1].set_xlabel("Time (s)")

# Figure 2: Event sequence and modes
plt.figure(figsize=(16, 6))
plt.ylim(-1, num_channels)
plt.yticks(range(num_channels), [f"Channel {ch}" for ch in range(num_channels)])
plt.title("Event Sequence with Clap Cooldown", fontsize=14)
plt.xlabel("Time (s)")
plt.grid(True, alpha=0.3)

# Mark all claps, cooldown periods and detection windows
for i, clap_time in enumerate(clap_times):
    cooldown_end = clap_time + clap_cooldown
    window_end = cooldown_end + active_window_after_clap
    if i < len(clap_times) - 1:
        window_end = min(window_end, clap_times[i+1])
    
    plt.axvline(clap_time, color='purple', linestyle='-', alpha=0.8, 
               label='Clap event' if i == 0 else "")
    plt.axvspan(clap_time, cooldown_end, 
               color='red', alpha=0.1, 
               label='Clap cooldown' if i == 0 else "")
    plt.axvspan(cooldown_end, window_end, 
               color='green', alpha=0.1, 
               label='Touch window' if i == 0 else "")
    
    # Mark detected mode
    mode = all_modes[i]
    if mode:
        plt.text(cooldown_end, num_channels - 0.5, f"Mode {mode}", 
                color='green', fontweight='bold')

# Mark all touch events
for clap_idx, (clap_time, touch_events) in enumerate(all_touch_events):
    for i, (t, ch) in enumerate(touch_events):
        plt.scatter(t, ch, color='blue', s=80, zorder=3, edgecolors='black')
        plt.text(t, ch+0.2, f"E{i+1}", fontsize=8)

plt.legend()
plt.tight_layout()
plt.show()
