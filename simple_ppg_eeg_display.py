"""
Simple PPG and EEG Real-Time Display
====================================
This is a simplified version ready to run.
Just upload the Arduino code and run this Python script.

Requirements:
pip install numpy matplotlib scipy pyserial

Author: For NUS Student
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
import serial
import time
from datetime import datetime
from collections import deque

# ----------------------------------------------------------------------------
# CONFIGURATION - CHANGE THESE FOR YOUR SETUP
# ----------------------------------------------------------------------------
SERIAL_PORT = 'COM3'   # Change this (Windows: COM3, Mac/Linux: /dev/ttyUSB0)
BAUD_RATE = 115200
PPG_SAMPLING_RATE = 100  # Hz
EEG_SAMPLING_RATE = 250  # Hz
DISPLAY_WINDOW = 10       # seconds
UPDATE_INTERVAL = 100     # ms

# ----------------------------------------------------------------------------
# STEP 1: DESIGN FILTERS
# ----------------------------------------------------------------------------
def design_filters():
    nyq_ppg = 0.5 * PPG_SAMPLING_RATE
    ppg_sos = signal.butter(4, [0.5/nyq_ppg, 15/nyq_ppg], btype='band', output='sos')

    nyq_eeg = 0.5 * EEG_SAMPLING_RATE
    eeg_sos = signal.butter(4, [0.5/nyq_eeg, 50/nyq_eeg], btype='band', output='sos')

    notch_freq = 50.0 / nyq_eeg  # 50 Hz (Singapore)
    notch_b, notch_a = signal.iirnotch(notch_freq, 30)

    return ppg_sos, eeg_sos, notch_b, notch_a

# ----------------------------------------------------------------------------
# STEP 2: CONNECT ARDUINO
# ----------------------------------------------------------------------------
def connect_arduino(port, baud):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        print(f"✓ Connected to Arduino on {port}")
        return ser
    except Exception as e:
        print(f"✗ Error: Could not connect to {port}")
        print(e)
        return None

# ----------------------------------------------------------------------------
# STEP 3: FILTER FUNCTIONS
# ----------------------------------------------------------------------------
def filter_ppg(data, sos):
    if len(data) < 20:
        return data
    return signal.sosfilt(sos, data)

def filter_eeg(data, sos, notch_b, notch_a):
    if len(data) < 20:
        return data
    filtered = signal.sosfilt(sos, data)
    return signal.filtfilt(notch_b, notch_a, filtered)

# ----------------------------------------------------------------------------
# STEP 4: PEAK DETECTION & HEART RATE
# ----------------------------------------------------------------------------
def detect_peaks(ppg_data, rate):
    if len(ppg_data) < 10:
        return np.array([])
    threshold = np.mean(ppg_data) + 0.5*np.std(ppg_data)
    dist = int(rate * 0.5)
    peaks, _ = signal.find_peaks(ppg_data, height=threshold, distance=dist)
    return peaks

def calculate_heart_rate(peaks, rate):
    if len(peaks) < 2:
        return 0
    intervals = np.diff(peaks) / rate
    hr = 60.0 / np.mean(intervals)
    return hr if 40 < hr < 200 else 0

# ----------------------------------------------------------------------------
# STEP 5: REAL-TIME DISPLAY CLASS
# ----------------------------------------------------------------------------
class RealTimeDisplay:
    def __init__(self, ser):
        self.ser = ser
        self.ppg_sos, self.eeg_sos, self.notch_b, self.notch_a = design_filters()

        self.time_data = deque(maxlen=3000)
        self.ppg_raw, self.eeg_raw = deque(maxlen=3000), deque(maxlen=3000)
        self.ppg_filt, self.eeg_filt = deque(maxlen=3000), deque(maxlen=3000)

        self.start_time = time.time()
        self.heart_rate = 0
        self.peaks = np.array([])

        self.log = open(f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 'w')
        self.log.write("Time,PPG_Raw,PPG_Filtered,EEG_Raw,EEG_Filtered,HeartRate\n")

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.setup_plot()

    def setup_plot(self):
        self.ax1.set_title("PPG Signal")
        self.ax1.set_ylabel("Amplitude")
        self.line_ppg, = self.ax1.plot([], [], lw=1.5)
        self.peaks_plot, = self.ax1.plot([], [], 'r*', ms=10)
        self.hr_text = self.ax1.text(0.02, 0.95, '', transform=self.ax1.transAxes)

        self.ax2.set_title("EEG Signal")
        self.ax2.set_ylabel("µV")
        self.ax2.set_xlabel("Time (s)")
        self.line_eeg, = self.ax2.plot([], [], lw=1.5)

    def read_data(self):
        try:
            if self.ser.in_waiting:
                line = self.ser.readline().decode().strip().split(',')
                if len(line) == 2:
                    return float(line[0]), float(line[1])
        except:
            return None, None
        return None, None

    def update(self, frame):
        ppg_val, eeg_val = self.read_data()
        if ppg_val is not None:
            t = time.time() - self.start_time
            self.time_data.append(t)
            self.ppg_raw.append(ppg_val)
            self.eeg_raw.append(eeg_val)

            if len(self.ppg_raw) > 50:
                ppg_filtered = filter_ppg(self.ppg_raw, self.ppg_sos)
                eeg_filtered = filter_eeg(self.eeg_raw, self.eeg_sos, self.notch_b, self.notch_a)
                self.ppg_filt = deque(ppg_filtered, maxlen=3000)
                self.eeg_filt = deque(eeg_filtered, maxlen=3000)

                self.peaks = detect_peaks(ppg_filtered, PPG_SAMPLING_RATE)
                if len(self.peaks) > 1:
                    self.heart_rate = calculate_heart_rate(self.peaks, PPG_SAMPLING_RATE)

            self.log.write(f"{t:.3f},{ppg_val},{self.ppg_filt[-1] if self.ppg_filt else 0},"
                           f"{eeg_val},{self.eeg_filt[-1] if self.eeg_filt else 0},"
                           f"{self.heart_rate:.1f}\n")

        # Update plots
        if self.time_data:
            t = np.array(self.time_data)
            self.line_ppg.set_data(t, list(self.ppg_filt))
            self.line_eeg.set_data(t, list(self.eeg_filt))
            self.hr_text.set_text(f"Heart Rate: {self.heart_rate:.1f} BPM")

            self.ax1.set_xlim(max(0, t[-1]-DISPLAY_WINDOW), t[-1])
            self.ax2.set_xlim(max(0, t[-1]-DISPLAY_WINDOW), t[-1])

        return self.line_ppg, self.line_eeg

    def run(self):
        print("Starting real-time plot...")
        FuncAnimation(self.fig, self.update, interval=UPDATE_INTERVAL)
        plt.show()
        self.log.close()
        if self.ser:
            self.ser.close()

# ----------------------------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------------------------
def main():
    print("Connecting to Arduino...")
    ser = connect_arduino(SERIAL_PORT, BAUD_RATE)
    if not ser:
        print("Connection failed. Check cable/port.")
        return
    display = RealTimeDisplay(ser)
    display.run()

if __name__ == "__main__":
    main()
