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

def detect_clap_event(rms_matrix, frame_times, clap_rms_threshold, clap_min_frames=2):
    """
    Detect clap event (all channels have high RMS simultaneously)
    Args:
        rms_matrix: RMS features (num_channels x num_frames)
        frame_times: Time of each frame
        clap_rms_threshold: RMS threshold for clap (all channels must exceed)
        clap_min_frames: Minimum consecutive frames to confirm clap
    Returns:
        clap_time: Timestamp of clap event (None if not detected)
    """
    num_channels, num_frames = rms_matrix.shape
    consecutive_clap_frames = 0
    clap_start_frame = None

    for j in range(num_frames):
        # Check if all channels exceed clap threshold in current frame
        all_high = np.all(rms_matrix[:, j] >= clap_rms_threshold)
        
        if all_high:
            consecutive_clap_frames += 1
            if consecutive_clap_frames == 1:
                clap_start_frame = j  # Record first frame of potential clap
        else:
            # Check if we had enough consecutive frames to confirm clap
            if consecutive_clap_frames >= clap_min_frames:
                # Return mid-time of clap event
                clap_end_frame = j - 1
                clap_time = (frame_times[clap_start_frame] + frame_times[clap_end_frame]) / 2
                return clap_time
            consecutive_clap_frames = 0

    # Final check for clap at end of audio
    if consecutive_clap_frames >= clap_min_frames:
        clap_time = frame_times[-1]
        return clap_time
    return None

def determine_mode(touch_events, clap_time, active_window=3.0):
    """
    Determine mode based on touch event order after clap
    Args:
        touch_events: List of (time, channel) events after clap
        clap_time: Time of clap event
        active_window: Time window after clap to consider events (seconds)
    Returns:
        mode: Detected mode (1-4) or None if no valid mode
    """
    # Filter events within active window after clap
    valid_events = [(t, ch) for t, ch in touch_events 
                   if (t >= clap_time) and (t <= clap_time + active_window)]
    if len(valid_events) < 2:  # Need at least 2 events to determine mode
        return None

    # Get first occurrence time of each channel (1-4, assuming channels 0=1, 1=2, 2=3, 3=4)
    # Note: Adjust channel mapping if your channels are numbered differently
    channel_mapping = {0:1, 1:2, 2:3, 3:4}  # Map 0→1, 1→2, 2→3, 3→4
    first_touch = {}
    for t, ch in valid_events:
        mapped_ch = channel_mapping[ch]
        if mapped_ch not in first_touch:
            first_touch[mapped_ch] = t

    # Check if all required channels have events (simplified: need at least 2 pairs)
    required_pairs = [
        (2,1), (3,4),  # Mode 1
        (1,2), (4,3),  # Mode 2
        (1,4), (2,3),  # Mode 3
        (4,1), (3,2)   # Mode 4
    ]
    
    # Check each mode's conditions
    # Mode 1: 2 triggers before 1; 3 triggers before 4
    if 2 in first_touch and 1 in first_touch and 3 in first_touch and 4 in first_touch:
        if (first_touch[2] < first_touch[1]) and (first_touch[3] < first_touch[4]):
            return 1
    # Mode 2: 1 triggers before 2; 4 triggers before 3
    if 1 in first_touch and 2 in first_touch and 4 in first_touch and 3 in first_touch:
        if (first_touch[1] < first_touch[2]) and (first_touch[4] < first_touch[3]):
            return 2
    # Mode 3: 1 triggers before 4; 2 triggers before 3
    if 1 in first_touch and 4 in first_touch and 2 in first_touch and 3 in first_touch:
        if (first_touch[1] < first_touch[4]) and (first_touch[2] < first_touch[3]):
            return 3
    # Mode 4: 4 triggers before 1; 3 triggers before 2
    if 4 in first_touch and 1 in first_touch and 3 in first_touch and 2 in first_touch:
        if (first_touch[4] < first_touch[1]) and (first_touch[3] < first_touch[2]):
            return 4
    return None

# === 1. Load Audio ===
audio_path = "/Users/zack/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_209ay89m1i9422_6c6e/msg/file/2025-08/red11.wav"  # Update your audio path
y, sr = librosa.load(audio_path, sr=None, mono=False)

# Ensure only 0-3 channels are used
if y.shape[0] > 4:
    y = y[:4, :]
num_channels = y.shape[0]
assert num_channels == 4, "This code requires exactly 4 channels"  # Mode detection needs 4 channels

# === 2. Parameter Settings ===
# Basic parameters
frame_duration = 0.1
frame_length = int(sr * frame_duration) 
hop_length = frame_length // 2

# Touch event parameters
threshold_ratio = 1.8
min_amplitude = 0.05
consecutive_required = 1
debounce_time = 0.1

# Clap detection parameters
clap_rms_threshold = 0.3  # Adjust based on your audio (all channels must exceed this)
clap_min_frames = 2  # Minimum 2 consecutive frames for clap
active_window_after_clap = 3.0  # Only detect touches within 3s after clap

# Window grouping parameters
window_size = 0.35

# === 3. Calculate RMS Features ===
rms_matrix = []
frame_times = []

for i in range(0, y.shape[1] - frame_length, hop_length):
    frame = y[:, i:i+frame_length]
    rms = np.sqrt(np.mean(frame **2, axis=1))  # RMS for each channel
    rms_matrix.append(rms)
    frame_times.append(i / sr)

rms_matrix = np.array(rms_matrix).T  # Shape: (num_channels, num_frames)
frame_times = np.array(frame_times)

# === 4. Detect Clap Event First ===
clap_time = detect_clap_event(
    rms_matrix, 
    frame_times, 
    clap_rms_threshold=clap_rms_threshold, 
    clap_min_frames=clap_min_frames
)

if not clap_time:
    print("❌ No clap event detected. Mode detection disabled.")
    # Exit if no clap (or add further handling)
    exit()
print(f"✅ Clap event detected at {clap_time:.2f}s. Starting touch detection...")

# === 5. Calculate RMS Vector Changes (for touch event refinement) ===
rms_norms = []
distances = [0]  # Euclidean distances between consecutive RMS vectors

for j in range(rms_matrix.shape[1]):
    vec = rms_matrix[:, j]
    norm_vec = vec / (np.linalg.norm(vec) + 1e-8)
    rms_norms.append(norm_vec)

for j in range(1, len(rms_norms)):
    distances.append(euclidean(rms_norms[j], rms_norms[j-1]))

# === 6. Touch Event Detection (Only after clap) ===
touch_events = []  # (time, channel)
last_trigger_times = [-10] * num_channels
consecutive_counts = [0] * num_channels

for j in range(len(frame_times)):
    t = frame_times[j]
    # Only detect touches after clap and within active window
    if t < clap_time or t > clap_time + active_window_after_clap:
        continue

    rms = rms_matrix[:, j]
    max_ch = np.argmax(rms)
    max_val = rms[max_ch]
    other_avg = np.mean(np.delete(rms, max_ch))

    # Check touch conditions
    reason = []
    if max_val <= threshold_ratio * other_avg:
        reason.append(f"ratio low")
    if max_val <= min_amplitude:
        reason.append(f"amplitude low")

    if not reason:
        consecutive_counts[max_ch] += 1
    else:
        consecutive_counts[max_ch] = 0

    # Trigger valid touch event
    if (consecutive_counts[max_ch] >= consecutive_required and 
        (t - last_trigger_times[max_ch]) > debounce_time):
        touch_events.append((t, max_ch))
        last_trigger_times[max_ch] = t
        print(f"👉 Touch event @ {t:.2f}s, Channel {max_ch}")
        consecutive_counts[max_ch] = 0

if not touch_events:
    print(f"❌ No touch events detected within {active_window_after_clap}s after clap.")
    exit()

# === 7. Simplify Sequences and Determine Mode ===
# Group events into windows
window_sequences = []
if touch_events:
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

simplified_sequences = [remove_consecutive_duplicates(seq) for seq in window_sequences]

# Determine mode
detected_mode = determine_mode(
    touch_events, 
    clap_time, 
    active_window=active_window_after_clap
)

# === 8. Result Summary ===
print("\n===== Final Results =====")
print(f"Clap event time: {clap_time:.2f}s")
print(f"Touch events detected: {len(touch_events)}")
print(f"Simplified sequences: {'; '.join(['-'.join(map(str, s)) for s in simplified_sequences])}")
print(f"Detected mode: {detected_mode if detected_mode else 'No valid mode'}")

# === 9. Visualization ===
# Plot 1: Channel waveforms and clap/touch markers
fig, axs = plt.subplots(2 * num_channels, 1, figsize=(14, 4 * num_channels))
fig.suptitle("Waveforms, RMS & Events (Clap + Touches)", fontsize=16)

for ch in range(num_channels):
    # Waveform subplot
    ax_wave = axs[2 * ch]
    time_axis = np.arange(len(y[ch])) / sr
    ax_wave.plot(time_axis, y[ch], alpha=0.7, color=f"C{ch}")
    ax_wave.set_title(f"Channel {ch} Waveform")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.grid(True, alpha=0.3)
    
    # Mark clap event
    ax_wave.axvline(clap_time, color='purple', linestyle='-', alpha=0.8, label='Clap Event')
    # Mark active window after clap
    ax_wave.axvspan(clap_time, clap_time + active_window_after_clap, 
                   color='yellow', alpha=0.1, label='Active Detection Window')
    
    # Mark touch events for this channel
    ch_events = [t for t, c in touch_events if c == ch]
    for t in ch_events:
        ax_wave.axvline(t, color='red', linestyle='--', alpha=0.6)

    ax_wave.legend()

    # RMS subplot
    ax_rms = axs[2 * ch + 1]
    ax_rms.plot(frame_times, rms_matrix[ch], alpha=0.8, color=f"C{ch}")
    ax_rms.axhline(y=min_amplitude, color='green', linestyle='--', label='Min Amplitude')
    ax_rms.axhline(y=clap_rms_threshold, color='purple', linestyle='--', label='Clap Threshold')
    ax_rms.set_title(f"Channel {ch} RMS Energy")
    ax_rms.set_ylabel("RMS")
    ax_rms.grid(True, alpha=0.3)
    ax_rms.legend()

axs[-1].set_xlabel("Time (s)")

# Plot 2: Event sequence and mode
plt.figure(figsize=(16, 6))
plt.ylim(-1, num_channels)
plt.yticks(range(num_channels), [f"Channel {ch}" for ch in range(num_channels)])
plt.title(f"Event Sequence (Clap at {clap_time:.2f}s | Mode: {detected_mode})", fontsize=14)
plt.xlabel("Time (s)")
plt.grid(True, alpha=0.3)

# Mark clap and active window
plt.axvline(clap_time, color='purple', linestyle='-', alpha=0.8, label='Clap Event')
plt.axvspan(clap_time, clap_time + active_window_after_clap, 
           color='yellow', alpha=0.1, label='Active Window')

# Mark touch events
for i, (t, ch) in enumerate(touch_events):
    plt.scatter(t, ch, color=f"C{ch}", s=80, zorder=3, edgecolors='black')
    plt.text(t, ch+0.2, f"E{i+1}", fontsize=8)

plt.legend()
plt.tight_layout()
plt.show()