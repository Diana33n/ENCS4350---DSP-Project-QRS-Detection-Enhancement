import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# ----- Step 1: Load ECG from MIT-BIH Record 100 -----
record = wfdb.rdrecord('100', pn_dir='mitdb', sampto=6500)
ecg = record.p_signal[:, 0]  # Use MLII lead
fs = record.fs

# ----- Step 2: Pan-Tompkins pipeline until MWI (Point 3) -----
def bandpass_filter(signal):
    low_b = [1, 0, -2, 0, 1]
    low_a = [1, -2, 1]
    low_passed = lfilter(low_b, low_a, signal)

    high_b = [-1/32] + [0]*15 + [1, -1] + [0]*14 + [1/32]
    high_passed = lfilter(high_b, [1], low_passed)
    return high_passed

def derivative_filter(signal):
    b = [1, 2, 0, -2, -1]
    return lfilter(b, [1], signal) * (1 / 8)

def squaring(signal):
    return signal ** 2

def moving_window_integration(signal, window_size=30):
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='same')

filtered = bandpass_filter(ecg)
derivative = derivative_filter(filtered)
squared = squaring(derivative)
mwi = moving_window_integration(squared)

# ----- Step 3: LMS Adaptive Thresholding (Point 4) -----
mu = 1e-4
alpha = 0.9
T = 0.5 * np.max(mwi[:fs])
peaks_lms = []
last_qrs = -np.inf
T_history = []

for n in range(len(mwi)):
    e = mwi[n] - T
    T = T + mu * e
    T_history.append(T)
    if mwi[n] > T and (n - last_qrs) > 0.2 * fs:
        peaks_lms.append(n)
        T *= alpha
        last_qrs = n

# ----- Step 4: Plot results -----
plt.figure(figsize=(14, 6))
plt.plot(mwi, label='MWI Signal')
plt.plot(T_history, label='Adaptive Threshold', linestyle='--')
plt.scatter(peaks_lms, [mwi[p] for p in peaks_lms], color='red', label='Detected QRS', zorder=5)
plt.title('QRS Detection using LMS Adaptive Threshold (MIT-BIH Record 100)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----- Step 5: Print summary -----
print("Number of detected QRS complexes:", len(peaks_lms))
print("First 10 detected peak indices:", peaks_lms[:10])
