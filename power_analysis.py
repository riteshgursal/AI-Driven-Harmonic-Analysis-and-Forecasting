import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import os
import time

# --- PHASE 2: STRUCTURING DEEP LEARNING MODELS ---

class HarmonicPredictorLSTM(nn.Module):
    """
    LSTM Model for time-series forecasting (Harmonic Prediction - P4).
    Input: Sequential FFT-derived features (time-steps, feature_dim).
    Output: Predicted harmonic magnitude/phase for the next time step.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(HarmonicPredictorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        print("Model 1: LSTM Harmonic Predictor initialized (Placeholder)")

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Use the output from the last time step for prediction
        return self.fc(lstm_out[:, -1, :])

class HarmonicClassifierCNN(nn.Module):
    """
    CNN Model for Harmonic Classification/Diagnosis (P4).
    Input: A window (sequence) of raw signal or FFT features.
    Output: Classification of source/severity (e.g., [PV, Wind, Baseline]).
    """
    def __init__(self, sequence_length=50, num_classes=3):
        super(HarmonicClassifierCNN, self).__init__()
        # Simplified 1D CNN structure for time-series/spectral data
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(self._get_conv_output(sequence_length), num_classes)
        print("Model 2: 1D CNN Harmonic Classifier initialized (Placeholder)")

    def _get_conv_output(self, seq_len):
        # Helper to calculate the size after convolution and pooling
        output_len = (seq_len - 3) // 2 + 1
        return output_len * 16

    def forward(self, x):
        # Assuming input shape: (batch_size, 1, sequence_length)
        x = self.pool(self.relu(self.conv1(x.unsqueeze(1))))
        x = x.view(x.size(0), -1) # Flatten
        return self.fc(x)

# --- PHASE 1: DATA SIMULATION, FFT, AND FEATURE EXTRACTION ---

def generate_synthetic_data(duration_sec=2, sample_rate=1000, fundamental_freq=60, harmonic_mag=0.1):
    """Simulates voltage/current data with fundamental and harmonic frequencies."""
    t = np.linspace(0, duration_sec, sample_rate * duration_sec, endpoint=False)
    # Fundamental frequency (60 Hz is common in power systems)
    signal = 1.0 * np.sin(2 * np.pi * fundamental_freq * t)
    # Add 5th and 7th order harmonics (common issues in PV/Wind inverters)
    signal += harmonic_mag * np.sin(2 * np.pi * (5 * fundamental_freq) * t)
    signal += 0.05 * np.sin(2 * np.pi * (7 * fundamental_freq) * t)
    # Add noise
    signal += 0.02 * np.random.randn(len(t))
    
    df = pd.DataFrame({'Time_s': t, 'Voltage_V': signal})
    df.to_csv('synthetic_grid_data.csv', index=False)
    return df

def perform_fft_analysis(data_series, sample_rate):
    """Performs FFT to extract frequency domain features."""
    N = len(data_series)
    T = 1.0 / sample_rate
    # Calculate FFT
    yf = fft(data_series.values)
    xf = fftfreq(N, T)[:N//2]
    # Calculate magnitude spectrum (ignoring negative frequencies)
    magnitude = 2.0/N * np.abs(yf[0:N//2])
    
    # Store relevant harmonic magnitudes (e.g., 1st, 5th, 7th order)
    harmonic_data = {}
    fundamental_index = np.argmax(magnitude) 
    
    # Simple feature extraction: Magnitude of 5th and 7th harmonics
    freqs = xf[1:]
    mags = magnitude[1:]

    # Find index closest to 5th harmonic (approx 300Hz)
    idx_5th = np.argmin(np.abs(freqs - 5 * 60))
    # Find index closest to 7th harmonic (approx 420Hz)
    idx_7th = np.argmin(np.abs(freqs - 7 * 60))

    harmonic_data['5th_Harmonic_Mag'] = mags[idx_5th]
    harmonic_data['7th_Harmonic_Mag'] = mags[idx_7th]

    return xf, magnitude, harmonic_data

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    SAMPLE_RATE = 1000 # 1000 samples per second
    
    print("--- Starting AI Harmonic Analysis POC (P4 Alignment) ---")
    
    # 1. Generate/Load Time-Series Data
    if not os.path.exists('synthetic_grid_data.csv'):
        print("Generating synthetic power grid data...")
        data_df = generate_synthetic_data(sample_rate=SAMPLE_RATE)
    else:
        print("Loading existing synthetic power grid data...")
        data_df = pd.read_csv('synthetic_grid_data.csv')

    voltage_series = data_df['Voltage_V']
    
    # 2. Signal Feature Extraction (FFT)
    xf, magnitude, features = perform_fft_analysis(voltage_series, SAMPLE_RATE)
    print(f"Extracted Harmonic Features: {features}")
    
    # 3. Instantiate PyTorch Models (Demonstrating Phase 2 and 3 Structure)
    # Input size is 2 (e.g., 5th and 7th harmonic magnitudes)
    lstm_model = HarmonicPredictorLSTM(input_size=len(features))
    cnn_model = HarmonicClassifierCNN(sequence_length=50) 

    # --- 4. Visualization of Results (Key POC Demonstration) ---
    
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Time-Series Waveform
    ax1.plot(data_df['Time_s'][:200], voltage_series[:200], label='Voltage Waveform', color='blue')
    ax1.set_title('Time-Series Voltage Waveform (First 0.2s)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage (V)')
    ax1.grid(True)

    # Plot 2: Frequency Spectrum (FFT)
    ax2.plot(xf[:200], magnitude[:200], label='Magnitude Spectrum', color='red')
    ax2.set_title('Frequency Spectrum Analysis (FFT)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.axvline(x=60, color='green', linestyle='--', label='Fundamental (60Hz)')
    ax2.axvline(x=300, color='orange', linestyle='--', label='5th Harmonic')
    ax2.axvline(x=420, color='purple', linestyle='--', label='7th Harmonic')
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout(pad=3.0)
    plt.show()

    print("\n--- POC Complete. Check the plots for FFT visualization ---")