import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ========== ABNORMAL DETECTION FUNCTIONS ==========

def detect_abnormalities_p(df):
    """Detect abnormal HR (<50 or >120 bpm) and SpO2 (<90%)."""
    abnormal_hr = (df['HeartRate'] < 50) | (df['HeartRate'] > 120)
    abnormal_spo2 = (df['SPO2'] < 90)
    return abnormal_hr, abnormal_spo2

def detect_abnormalities_bis(df, low_threshold=20, high_threshold=80):
    """Detect abnormal EEG (too low or too high)."""
    abnormal_eeg = (df['EEG'] < low_threshold) | (df['EEG'] > high_threshold)
    return abnormal_eeg

def draw_alert_boxes(ax, time_series, abnormal_series, color='red', alpha=0.2):
    """Highlight time segments with abnormal values using translucent boxes."""
    abnormal_indices = abnormal_series[abnormal_series].index
    if abnormal_indices.empty:
        return
    start = abnormal_indices[0]
    end = start
    for i in range(1, len(abnormal_indices)):
        if abnormal_indices[i] != abnormal_indices[i-1] + 1:
            ax.add_patch(patches.Rectangle(
                (time_series[start], ax.get_ylim()[0]),
                time_series[end] - time_series[start],
                ax.get_ylim()[1] - ax.get_ylim()[0],
                linewidth=0,
                facecolor=color,
                alpha=alpha))
            start = abnormal_indices[i]
        end = abnormal_indices[i]
    ax.add_patch(patches.Rectangle(
        (time_series[start], ax.get_ylim()[0]),
        time_series[end] - time_series[start],
        ax.get_ylim()[1] - ax.get_ylim()[0],
        linewidth=0,
        facecolor=color,
        alpha=alpha))

# ========== SYNCHRONIZED MULTI-CHANNEL VISUALIZATION ==========

def visualize_combined(data_folder, low_eeg=20, high_eeg=80):
    """Display synchronized interactive graphs for HR, SpO2, and EEG."""
    # Try to find a BIS_ file (EEG data)
    bis_file = next((f for f in os.listdir(data_folder)
                     if f.startswith("BIS_") and f.endswith(".csv")), None)
    bis_df = None
    abnormal_eeg = None
    if bis_file:
        bis_df = pd.read_csv(os.path.join(data_folder, bis_file))
        bis_df.columns = ['Time', 'EEG']
        bis_df['Time'] = pd.to_datetime(bis_df['Time'], errors='coerce')
        abnormal_eeg = detect_abnormalities_bis(bis_df, low_eeg, high_eeg)

    # Process P_number files
    for filename in os.listdir(data_folder):
        if filename.startswith("P_") and filename.endswith(".csv"):
            print(f"Visualizing synchronized data for {filename}...")
            df = pd.read_csv(os.path.join(data_folder, filename))
            df.columns = ['Time', 'HeartRate', 'SPO2']  # Corrected column order
            df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

            abnormal_hr, abnormal_spo2 = detect_abnormalities_p(df)

            # Build a shared time axis
            if bis_df is not None:
                all_times = pd.concat([df['Time'], bis_df['Time']])
            else:
                all_times = df['Time']

            # Create vertically aligned, time-synchronized subplots
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 8))
            fig.suptitle(f"Synchronized Vital Sign Visualization: {filename}", fontsize=14)

            # Heart Rate subplot
            ax = axes[0]
            ax.plot(df['Time'], df['HeartRate'], color='blue', label='Heart Rate (bpm)')
            draw_alert_boxes(ax, df['Time'], abnormal_hr)
            ax.set_ylabel("Heart Rate (bpm)")
            ax.legend(loc="upper right")
            ax.grid(True)

            # SpO2 subplot
            ax = axes[1]
            ax.plot(df['Time'], df['SPO2'], color='green', label='SpO₂ (%)')
            draw_alert_boxes(ax, df['Time'], abnormal_spo2)
            ax.set_ylabel("SpO₂ (%)")
            ax.legend(loc="upper right")
            ax.grid(True)

            # EEG subplot
            ax = axes[2]
            if bis_df is not None:
                ax.plot(bis_df['Time'], bis_df['EEG'], color='purple', label='EEG (BIS)')
                draw_alert_boxes(ax, bis_df['Time'], abnormal_eeg)
                ax.legend(loc="upper right")
            else:
                ax.text(0.5, 0.5, "No BIS EEG data found", ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_xlabel("Time")
            ax.set_ylabel("EEG Level")
            ax.grid(True)

            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            plt.show()

# ========== EXECUTION ==========

if __name__ == "__main__":
    data_folder = input("Enter the full directory path containing P_ and BIS_ files: ").strip()
    visualize_combined(data_folder, low_eeg=20, high_eeg=80)
    print("Interactive synchronized visualization complete.")