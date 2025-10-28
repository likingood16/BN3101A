import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from collections import deque
import numpy as np

class RealTimeDataBuffer:
    def __init__(self, window_size=180):
        self.window_size = window_size
        # Create circular buffers for each signal
        self.time_buffer = deque(maxlen=window_size)
        self.hr_buffer = deque(maxlen=window_size)
        self.spo2_buffer = deque(maxlen=window_size)
        self.eeg_buffer = deque(maxlen=window_size)
        
    def update(self, time_val, hr_val, spo2_val, eeg_val=None):
        self.time_buffer.append(time_val)
        self.hr_buffer.append(hr_val)
        self.spo2_buffer.append(spo2_val)
        if eeg_val is not None:
            self.eeg_buffer.append(eeg_val)
            
    def get_data(self):
        return {
            'Time': list(self.time_buffer),
            'HeartRate': list(self.hr_buffer),
            'SPO2': list(self.spo2_buffer),
            'EEG': list(self.eeg_buffer) if self.eeg_buffer else None
        }

def detect_abnormalities_p(hr_data, spo2_data):
    """Detect abnormal HR (<50 or >120 bpm) and SpO₂ (<90%)."""
    abnormal_hr = np.logical_or(np.array(hr_data) < 50, np.array(hr_data) > 120)
    abnormal_spo2 = np.array(spo2_data) < 90
    return abnormal_hr, abnormal_spo2

def detect_abnormalities_bis(eeg_data, low_threshold=20, high_threshold=80):
    """Detect abnormal EEG (too low or too high)."""
    eeg_array = np.array(eeg_data)
    abnormal_eeg = np.logical_or(eeg_array < low_threshold, eeg_array > high_threshold)
    return abnormal_eeg

def draw_alert_boxes(ax, time_series, abnormal_series, color='red', alpha=0.2):
    """Highlight time segments with abnormal values."""
    if len(time_series) == 0 or not any(abnormal_series):
        return
        
    time_array = np.array(time_series)
    abnormal_indices = np.where(abnormal_series)[0]
    if len(abnormal_indices) == 0:
        return
        
    start_idx = abnormal_indices[0]
    prev_idx = start_idx
    
    for idx in abnormal_indices[1:]:
        if idx != prev_idx + 1:
            ax.add_patch(patches.Rectangle(
                (time_array[start_idx], ax.get_ylim()[0]),
                time_array[prev_idx] - time_array[start_idx],
                ax.get_ylim()[1] - ax.get_ylim()[0],
                linewidth=0, facecolor=color, alpha=alpha))
            start_idx = idx
        prev_idx = idx
        
    ax.add_patch(patches.Rectangle(
        (time_array[start_idx], ax.get_ylim()[0]),
        time_array[prev_idx] - time_array[start_idx],
        ax.get_ylim()[1] - ax.get_ylim()[0],
        linewidth=0, facecolor=color, alpha=alpha))

def update_plot(fig, axes, data_buffer, low_eeg=20, high_eeg=80):
    """Update the plot with new data."""
    data = data_buffer.get_data()
    if not data['Time']:  # No data yet
        return

    # Clear all axes
    for ax in axes:
        ax.clear()

    current_time = data['Time'][-1]
    window_start = max(0, current_time - 180)  # Show last 180 seconds
    
    # Update title
    fig.suptitle(f"Real-time Signals Monitoring (Window: {window_start}s - {current_time}s)", fontsize=14)

    # HR panel
    ax = axes[0]
    ax.plot(data['Time'], data['HeartRate'], color='blue', label='Heart Rate (bpm)')
    abnormal_hr, abnormal_spo2 = detect_abnormalities_p(data['HeartRate'], data['SPO2'])
    draw_alert_boxes(ax, data['Time'], abnormal_hr)
    ax.set_ylabel("Heart Rate (bpm)")
    ax.set_ylim(40, 130)
    ax.legend(loc="upper right")
    ax.grid(True)

    # SpO2 panel
    ax = axes[1]
    ax.plot(data['Time'], data['SPO2'], color='green', label='SpO₂ (%)')
    draw_alert_boxes(ax, data['Time'], abnormal_spo2)
    ax.set_ylabel("SpO₂ (%)")
    ax.set_ylim(85, 100)
    ax.legend(loc="upper right")
    ax.grid(True)

    # EEG panel
    ax = axes[2]
    if data['EEG'] and len(data['EEG']) > 0:
        ax.plot(data['Time'], data['EEG'], color='purple', label='EEG (BIS)')
        abnormal_eeg = detect_abnormalities_bis(data['EEG'], low_eeg, high_eeg)
        draw_alert_boxes(ax, data['Time'], abnormal_eeg)
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "No BIS EEG data available", ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, color='gray')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("EEG Level")
    ax.set_ylim(0, 100)
    ax.grid(True)

    # Set x-axis limits for all plots
    for ax in axes:
        ax.set_xlim(window_start, current_time)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    fig.canvas.draw()
    plt.pause(0.01)

def simulate_realtime_monitoring(data_folder, update_interval=1.0, low_eeg=20, high_eeg=80):
    """Simulate real-time monitoring of vital signs."""
    # Set up the plot
    plt.ion()  # Turn on interactive mode
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    
    # Initialize data buffer
    data_buffer = RealTimeDataBuffer(window_size=180)  # Store 180 seconds of data
    
    # Load data files
    p_files = [f for f in os.listdir(data_folder) if f.startswith("P_") and f.endswith(".csv")]
    if not p_files:
        print("No data files found!")
        return
        
    # Load first P file
    filename = p_files[0]
    num = filename.split('_')[1].split('.')[0]
    p_data = pd.read_csv(os.path.join(data_folder, filename))
    p_data.columns = ['Time', 'HeartRate', 'SPO2']
    
    # Try to load corresponding BIS file
    bis_data = None
    bis_filename = f"BIS_{num}.csv"
    if os.path.exists(os.path.join(data_folder, bis_filename)):
        bis_data = pd.read_csv(os.path.join(data_folder, bis_filename))
        bis_data.columns = ['Time', 'EEG']
    
    print(f"Starting real-time monitoring simulation...")
    print("Press Ctrl+C to stop the monitoring")
    
    try:
        # Simulate real-time data stream at 5-second intervals
        for i in range(0, len(p_data), 5):  # Step by 5 to simulate 5-second intervals
            # Get current data point
            time_val = p_data.iloc[i]['Time']
            hr_val = p_data.iloc[i]['HeartRate']
            spo2_val = p_data.iloc[i]['SPO2']
            
            # Get corresponding BIS data if available
            eeg_val = None
            if bis_data is not None:
                matching_bis = bis_data[bis_data['Time'] == time_val]
                if not matching_bis.empty:
                    eeg_val = matching_bis.iloc[0]['EEG']
            
            # Update data buffer
            data_buffer.update(time_val, hr_val, spo2_val, eeg_val)
            
            # Update plot
            update_plot(fig, axes, data_buffer, low_eeg, high_eeg)
            
            # Wait for next update
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\nStopping monitoring simulation...")
    finally:
        plt.ioff()
        plt.close('all')

if __name__ == "__main__":
    data_folder = input("Enter the full directory path containing P_ and BIS_ files: ").strip()
    update_interval = 5.0  # Update every 5 seconds
    simulate_realtime_monitoring(data_folder, update_interval)
    print("Monitoring simulation stopped.")