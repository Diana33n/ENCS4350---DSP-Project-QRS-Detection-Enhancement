
# ENCS4350 - DSP Project: QRS Detection Enhancement

This project reproduces and enhances the Pan-Tompkins QRS detection algorithm for ECG signals using Python and DSP techniques. It also implements an adaptive LMS-based thresholding method to improve detection in noisy conditions.

## Overview

The project uses ECG data from the MIT-BIH Arrhythmia Database and applies a full Pan-Tompkins pipeline:

1. Bandpass filtering (5–15 Hz)
2. Derivative filter to highlight slopes
3. Squaring function for non-linear amplification
4. Moving Window Integration (MWI)
5. Thresholding (static and adaptive LMS-based)

Performance is evaluated using Sensitivity, PPV, and F1-Score on clean and noisy ECG records.

---

## Files

| File                            | Description                                                                |
| ------------------------------- | -------------------------------------------------------------------------- |
| `lms_qrs_detection.py`          | Implements LMS-based adaptive threshold QRS detection with plots.          |
| `evaluate_performance.py`       | Compares Pan-Tompkins static threshold and LMS adaptive threshold methods. |

---

## How to Run

### Requirements

* Python 3.x
* Libraries: `wfdb`, `numpy`, `scipy`, `matplotlib`
* MIT-BIH ECG data (via PhysioNet)


---

##

| Name           | Student ID |
| -------------- | ---------- |
| Diana Nasser   | 1210363    |
| Jouwana Daibes | 1210123    |


Course: ENCS4350 – Digital Signal Processing

---
