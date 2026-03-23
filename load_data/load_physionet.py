import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from Utils import compute_eps_per_channel
import argparse

def bandpass_filter(signal, low_freq, high_freq, fs, order=5):
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def remove_baseline_drift(signal, wavelet="db6", level=9):
    coeff = pywt.wavedec(signal, wavelet, level=level)
    coeff[0] = np.zeros_like(coeff[0])  # Remove baseline drift
    reconstructed_signal = pywt.waverec(coeff, wavelet)
    return reconstructed_signal


def load_ecg_data(record_path):
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    ecg_signal = record.p_signal[:, 0]
    r_peaks = annotation.sample
    
    # Apply baseline drift removal and bandpass filter
    ecg_signal = remove_baseline_drift(ecg_signal)
    ecg_signal = bandpass_filter(ecg_signal, low_freq=0.5, high_freq=40, fs=500)
    
    return ecg_signal, r_peaks

def normalize_and_segment(signal, r_peaks, window_size=180):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    signal_normalized = scaler.fit_transform(signal.reshape(-1, 1)).flatten()

    segments = []
    half_window = window_size // 2
    for r_peak in r_peaks:
        start = max(r_peak - half_window, 0)
        end = min(r_peak + half_window, len(signal_normalized))
        if end - start == window_size:
            segments.append(signal_normalized[start:end])
    return np.array(segments)

def visualize_ecg_signals(ecg_signal, segments, r_peaks, record_label):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(ecg_signal, label='Filtered ECG Signal')
    plt.scatter(r_peaks, [ecg_signal[j] for j in r_peaks], color='red', label='R-peaks')
    plt.title(f'ECG Signal: {record_label}')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(segments[0], label='First Processed Segment')
    plt.title('First Segment After Processing')
    plt.legend()

    plt.tight_layout()
    plt.show()

def load_and_process_all_records(base_directory, visualize=False, lis = ['Person_01', 'Person_02','Person_09', 'Person_10']):
    all_segments = []
    all_labels = []
    person_dirs = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]    
    print(person_dirs)
    for person_id in lis:
        person_path = os.path.join(base_directory, person_id)
        for record_file in os.listdir(person_path):
            if record_file.endswith('.dat'):
                record_base = record_file[:-4]
                record_path = os.path.join(person_path, record_base)
                ecg_signal, r_peaks = load_ecg_data(record_path)
                segments = normalize_and_segment(ecg_signal, r_peaks)
                all_segments.extend(segments)
                all_labels.extend([person_id] * len(segments))
                if visualize and person_id == 'Person_01':  # Example visualization for one person
                    visualize_ecg_signals(ecg_signal, segments, r_peaks, record_base)

    return np.array(all_segments), np.array(all_labels)

def prep_physionet_dataset(raw_data_dir, savedir, lis, verbose=False):
    all_segments, all_labels = load_and_process_all_records(raw_data_dir, visualize=False ,lis=lis)

    # Encoding labels
    label_encoder = LabelEncoder()
    all_labels_encoded = label_encoder.fit_transform(all_labels)

    # Prepare data for the LSTM model
    X = np.array(all_segments).reshape((len(all_segments), -1, 1))
    y = np.array(all_labels_encoded)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
    }
    eps_per_channel = compute_eps_per_channel(result['X_train'])
    result['eps_per_channel'] = eps_per_channel

    if verbose:
        print(np.unique(y_train), np.unique(y_test))

    for key, val in result.items():
        np.save(f'{savedir}/{key}.npy', val)
        if verbose:
            print(val.shape)
            print(f'{key} saved as {savedir}/{key}.npy')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', default='data/PhysioNet')
    parser.add_argument('--processed_data_dir', default='data/PhysioNetProcessed')
    parser.add_argument('--verbose', action='store_true')
    args=parser.parse_args()
    os.makedirs(args.processed_data_dir, exist_ok=True)
    prep_physionet_dataset(raw_data_dir=args.raw_data_dir, savedir=args.processed_data_dir, lis=os.listdir(args.raw_data_dir), verbose=args.verbose)