import wfdb
import numpy as np
from scipy.signal import lfilter

# Signal Processing Functions
def bandpass_filter(signal):
    low_b = [1, 0, -2, 0, 1]
    low_a = [1, -2, 1]
    low_passed = lfilter(low_b, low_a, signal)
    high_b = [-1/32] + [0]*15 + [1, -1] + [0]*14 + [1/32]
    return lfilter(high_b, [1], low_passed)

def derivative_filter(signal):
    return lfilter([1, 2, 0, -2, -1], [1], signal) * (1 / 8)

def squaring(signal):
    return signal ** 2

def moving_window_integration(signal, window_size=30):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

# Static Threshold Detection
def detect_static(mwi, fs):
    T = 0.6 * np.max(mwi[:fs])
    peaks = []
    last_qrs = -np.inf
    for n in range(1, len(mwi)-1):
        if mwi[n] > T and mwi[n] > mwi[n-1] and mwi[n] > mwi[n+1]:
            if (n - last_qrs) > 0.25 * fs:
                peaks.append(n)
                last_qrs = n
    return peaks

# Robust LMS Adaptive Detection
def detect_lms_robust(mwi, fs, base_mu=0.005, alpha=0.94, floor_gain=0.4):
    baseline = np.median(mwi[:fs])
    T = baseline
    peaks = []
    last_qrs = -np.inf
    refractory = int(0.25 * fs)
    adaptive_floor = baseline + floor_gain * np.std(mwi)

    for n in range(1, len(mwi)-1):
        val = mwi[n]
        if val < adaptive_floor:
            continue
        error = val - T
        mu = base_mu * (1 + abs(error))
        T += mu * error

        if val > T and val > mwi[n-1] and val > mwi[n+1]:
            if (n - last_qrs) > refractory:
                peaks.append(n)
                last_qrs = n
                T *= alpha
    return peaks

# Evaluation Metrics
def evaluate(detected, truth, fs, tol=0.1):
    tolerance = int(tol * fs)
    TP = 0
    matched = np.zeros(len(truth), dtype=bool)
    for d in detected:
        for i, t in enumerate(truth):
            if not matched[i] and abs(d - t) <= tolerance:
                matched[i] = True
                TP += 1
                break
    FP = len(detected) - TP
    FN = len(truth) - TP
    Se = TP / (TP + FN) if TP + FN else 0
    PPV = TP / (TP + FP) if TP + FP else 0
    F1 = 2 * Se * PPV / (Se + PPV) if Se + PPV else 0
    return TP, FP, FN, Se, PPV, F1

# Comparison Function
def compare(record_id):
    print(f"\n📊 Evaluation for Record: {record_id}")
    record = wfdb.rdrecord(record_id, pn_dir='mitdb')
    ann = wfdb.rdann(record_id, 'atr', pn_dir='mitdb')
    ecg = record.p_signal[:, 0]
    fs = record.fs
    true_peaks = ann.sample

    # Pan-Tompkins Pipeline
    filtered = bandpass_filter(ecg)
    deriv = derivative_filter(filtered)
    squared = squaring(deriv)
    mwi_signal = moving_window_integration(squared)

    # Detection
    peaks_static = detect_static(mwi_signal, fs)
    peaks_lms = detect_lms_robust(mwi_signal, fs)

    # Evaluation
    r_static = evaluate(peaks_static, true_peaks, fs)
    r_lms = evaluate(peaks_lms, true_peaks, fs)

    print(f"{'Method':<30}{'Se':>10} {'PPV':>10} {'F1 Score':>10}")
    print(f"{'-'*60}")
    print(f"{'Pan-Tompkins (Static)':<30}{r_static[3]:>10.4f} {r_static[4]:>10.4f} {r_static[5]:>10.4f}")
    print(f"{'LMS Adaptive Threshold':<30}{r_lms[3]:>10.4f} {r_lms[4]:>10.4f} {r_lms[5]:>10.4f}")

# Run on multiple records
compare('100')
compare('108')
