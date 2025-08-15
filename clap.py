import numpy as np
import librosa
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class ClapDetectorState:
    consecutive_clap_frames: int = 0
    last_clap_time: float = -1000.0
    frame_count: int = 0

class ClapDetector:
    def __init__(self, config):
        self.sr = config["sample_rate"]
        self.frame_duration = config["frame_duration"]
        self.frame_size = int(self.sr * self.frame_duration)
        self.rms_threshold = config["rms_threshold"]
        self.min_consecutive_frames = config["min_consecutive_frames"]
        self.min_interval = config["min_interval"]
        self.state = ClapDetectorState()

    def process_frame(self, frame):
        current_time = self.state.frame_count * self.frame_duration
        self.state.frame_count += 1

        rms = np.sqrt(np.mean(frame **2, axis=1))
        all_above = np.all(rms >= self.rms_threshold)

        if all_above:
            self.state.consecutive_clap_frames += 1
        else:
            self.state.consecutive_clap_frames = 0

        detected = False
        if (self.state.consecutive_clap_frames >= self.min_consecutive_frames and 
            (current_time - self.state.last_clap_time) >= self.min_interval):
            detected = True
            self.state.last_clap_time = current_time
            self.state.consecutive_clap_frames = 0

        return detected, rms
    
def load_audio(file_path):
    try:
        # Load audio with original sample rate and preserve channels
        y, sr = librosa.load(file_path, sr=None, mono=False)
        
        # Ensure we have at least 1 channel, maximum 4 channels
        if y.ndim == 1:
            # Convert mono to single channel array
            y = y[np.newaxis, :]
        
        # Extract only the first 4 channels if there are more
        if y.shape[0] > 4:
            y = y[:4, :]
            print(f"Audio has more than 4 channels. Using first 4 channels.")
        
        print(f"Successfully loaded audio: {file_path}")
        print(f"Channels: {y.shape[0]}, Samples: {y.shape[1]}, Sample rate: {sr} Hz")
        return y, sr
    except Exception as e:
        print(f"Error loading audio: {str(e)}")
        raise

def simulate_streaming_detection(audio_data, sr, detector_config):
    # Initialize detector with configuration
    detector = ClapDetector({
        "sample_rate": sr,
        "frame_duration": detector_config["frame_duration"],
        "rms_threshold": detector_config["rms_threshold"],
        "min_consecutive_frames": detector_config["min_consecutive_frames"],
        "min_interval": detector_config["min_interval"]
    })

    frame_size = detector.frame_size
    num_channels = audio_data.shape[0]  # This will be <=4 due to load_audio processing
    total_samples = audio_data.shape[1]
    num_frames = total_samples // frame_size

    # Initialize storage arrays
    detection_events = []
    frame_rms = np.zeros((num_frames, num_channels))
    frame_times = np.zeros(num_frames)

    print(f"Starting processing: {num_frames} frames, {num_channels} channels")

    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        if end > total_samples:
            break
            
        frame = audio_data[:, start:end]
        detected, rms = detector.process_frame(frame)
        
        # Store results
        frame_times[i] = i * detector_config["frame_duration"]
        frame_rms[i] = rms
        
        if detected:
            detection_events.append(frame_times[i])
            print(f"Clap detected @ {frame_times[i]:.2f}s")

    # Trim arrays to actual number of processed frames
    frame_rms = frame_rms[:i+1]
    frame_times = frame_times[:i+1]

    print(f"Processing complete: {i+1} frames processed")
    return {
        "detection_events": detection_events,
        "frame_rms": frame_rms,
        "frame_times": frame_times,
        "sr": sr,
        "audio_data": audio_data
    }

def visualize_results(results, detector_config):
    # Extract data from results
    num_channels = results["audio_data"].shape[0]
    sr = results["sr"]
    audio_data = results["audio_data"]
    frame_rms = results["frame_rms"]
    frame_times = results["frame_times"]
    rms_threshold = detector_config["rms_threshold"]

    print(f"Visualizing results: {frame_rms.shape[0]} RMS frames, {num_channels} channels")

    # Create figure with subplots for each channel + 1 for RMS
    fig, axs = plt.subplots(num_channels + 1, 1, figsize=(12, 3 * (num_channels + 1)))
    fig.suptitle("Audio Analysis (First 4 Channels)", fontsize=16)

    # Plot waveforms for each channel
    for ch in range(num_channels):
        ax = axs[ch]
        time_axis = np.arange(audio_data.shape[1]) / sr
        ax.plot(time_axis, audio_data[ch], alpha=0.6, color=f"C{ch}")
        ax.set_title(f"Channel {ch} Waveform")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)

    # Plot RMS with threshold
    rms_ax = axs[-1]
    for ch in range(num_channels):
        rms_ax.plot(frame_times, frame_rms[:, ch], label=f"Channel {ch}", alpha=0.8)
    
    # Add threshold line
    rms_ax.axhline(y=rms_threshold, color='purple', linestyle='-', linewidth=2,
                  label=f"Threshold: {rms_threshold:.3f}")
    
    rms_ax.set_title("RMS Energy per Channel")
    rms_ax.set_xlabel("Time (s)")
    rms_ax.set_ylabel("RMS Energy")
    rms_ax.legend()
    rms_ax.grid(alpha=0.3)

    # Adjust layout and display
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

if __name__ == "__main__":
    # Configuration parameters
    DETECTOR_CONFIG = {
        "frame_duration": 0.06,  # 60ms frames
        "rms_threshold": 0.2,   # Adjust based on your audio level
        "min_consecutive_frames": 1,
        "min_interval": 0.2
    }

    # Audio path - replace with your actual file path
    audio_path = "/Users/zack/Desktop/mic-detect/redf1.wav"

    try:
        # Load audio (will automatically extract first 4 channels)
        audio_data, sr = load_audio(audio_path)
        
        # Process audio
        results = simulate_streaming_detection(audio_data, sr, DETECTOR_CONFIG)
        
        # Verify results
        if results["frame_rms"].size == 0:
            print("Warning: No RMS data generated!")
        else:
            print(f"Generated RMS data: {results['frame_rms'].shape}")
        
        # Visualize results
        visualize_results(results, DETECTOR_CONFIG)
        
    except Exception as e:
        print(f"Program error: {str(e)}")
