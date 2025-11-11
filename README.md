**AI-Driven Harmonic Analysis and Forecasting**

This repository details a Proof-of-Concept (POC) addressing the critical challenge of Power Quality Assessment in systems integrated with renewable energy sources (Solar PV, Wind). The project establishes a robust data-centric pipeline for Harmonic Prediction and fault Classification using advanced sequential Deep Learning techniques.

The **Phase 1** module demonstrates core proficiency in signal processing (FFT), data visualization, and structuring the necessary Deep Learning architectures for future research expansion.

**🚀 Execution Instructions**

**A. Setup and Installation**

*Clone the Repository:*

git clone https://github.com/riteshgursal/AI-Driven-Harmonic-Analysis-and-Forecasting.git
cd harmonic-analysis-poc


*Install Dependencies:* (Requires a Python environment with scientific libraries)

**pip install -r requirements_p4.txt**


**B. Running the Analysis**

Run the main Python script. It will generate a synthetic power signal, perform the Fast Fourier Transform (FFT) analysis, and display two critical plots (Time-Series and Frequency Spectrum).

**python power_analysis.py**

**💡 Technical Highlights & Research Roadmap**

**Technical Highlights:**

**Mandatory Signal Feature Extraction:** The script utilizes SciPy's FFT function and NumPy to perform spectral analysis, isolating crucial non-linear harmonic components from the raw time-series data. This technique is fundamental for converting complex signal patterns into actionable data features.

**Data Processing & Visualization:** The code employs Pandas for structured data handling and Matplotlib/Seaborn to visually validate the Time-Series waveform and the spectral frequency components.

**Predictive Model Architecture:** Defines a conceptual PyTorch LSTM (HarmonicPredictorLSTM) optimized for processing sequential, time-dependent FFT features.

**Diagnostic Classifier:** Defines a conceptual PyTorch 1D CNN (HarmonicClassifierCNN) structured to categorize harmonic patterns for automated fault identification.

**Research & Innovation:**

**Harmonic Detection:** The POC executes a core function for Harmonic Detection required in advanced power system analysis, demonstrating readiness to tackle critical infrastructure problems.

**Advanced Sequence Modeling:** Establishing the LSTM foundation sets the stage for real-time Harmonic Prediction, transitioning from simple signal analysis to complex time-series forecasting.

**Fault Diagnostics:** The conceptual 1D CNN module supports automated fault Classification & Diagnostics, which is a high-value application for improving grid stability and reliability.
