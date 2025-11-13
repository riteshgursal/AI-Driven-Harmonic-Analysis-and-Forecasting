**AI-Driven Harmonic Analysis and Forecasting**

This repository details a Proof-of-Concept (POC) addressing the critical challenge of Power Quality Assessment in systems integrated with renewable energy sources (Solar PV, Wind). The project establishes a robust data-centric pipeline for Harmonic Prediction and fault Classification using advanced sequential Deep Learning techniques.

---

The **Phase 1** module demonstrates core proficiency in signal processing (FFT), data visualization, and structuring the necessary Deep Learning architectures for future research expansion.

**🚀 Execution Instructions**

**A. Setup and Installation**

*Clone the Repository:*

git clone https://github.com/riteshgursal/AI-Driven-Harmonic-Analysis-and-Forecasting.git
```
cd harmonic-analysis-poc
```

*Install Dependencies:* (Requires a Python environment with scientific libraries provided in requirements1.txt file)
```
pip install -r requirements_p4.txt
```

**B. Running the Analysis**

Run the main Python script. It will generate a synthetic power signal, perform the Fast Fourier Transform (FFT) analysis, and display two critical plots (Time-Series and Frequency Spectrum).
```
python power_analysis.py
```

---
**💡 Technical Highlights & Research Roadmap**

**Technical Highlights:**

**Mandatory Signal Feature Extraction:** The script utilizes SciPy's FFT function and NumPy to perform spectral analysis, isolating crucial non-linear harmonic components from the raw time-series data. This technique is fundamental for converting complex signal patterns into actionable data features.

**Data Processing & Visualization:** The code employs Pandas for structured data handling and Matplotlib/Seaborn to visually validate the Time-Series waveform and the spectral frequency components.

**Predictive Model Architecture:** Defines a conceptual PyTorch LSTM (HarmonicPredictorLSTM) optimized for processing sequential, time-dependent FFT features.

**Diagnostic Classifier:** Defines a conceptual PyTorch 1D CNN (HarmonicClassifierCNN) structured to categorize harmonic patterns for automated fault identification.

---

**Research & Innovation:**

**Harmonic Detection:** The POC executes a core function for Harmonic Detection required in advanced power system analysis, demonstrating readiness to tackle critical infrastructure problems.

**Advanced Sequence Modeling:** Establishing the LSTM foundation sets the stage for real-time Harmonic Prediction, transitioning from simple signal analysis to complex time-series forecasting.

**Fault Diagnostics:** The conceptual 1D CNN module supports automated fault Classification & Diagnostics, which is a high-value application for improving grid stability and reliability.

---

**🎯 Objective**

In predictive maintenance and energy analytics, it’s essential not only to make accurate forecasts but also to understand why the model behaves a certain way.
This section explains how explainable AI (XAI) and evaluation metrics are used to interpret and validate model performance.

---

**⚙️ Model Evaluation Process**

The system uses time-series data (e.g., power signals, vibrations, or current harmonics) and forecasts anomalies or degradation patterns using LSTM networks.


| Metric                             | Purpose                                                                         | Formula / Description                                      |                     |   |
| ---------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------- | - |
| **MAE (Mean Absolute Error)**      | Measures average prediction error magnitude.                                    | ( MAE = \frac{1}{n} \sum                                   | y_{pred} - y_{true} |  |
| **RMSE (Root Mean Squared Error)** | Highlights larger deviations, suitable for continuous time-series data.         | ( RMSE = \sqrt{\frac{1}{n} \sum (y_{pred} - y_{true})^2}  |                     |   |
| **R² Score**                       | Indicates how well predictions follow actual trends (closer to 1 → better fit). | ( R^2 = 1 - \frac{SS_{res}}{SS_{tot}} )                    |                     |   |



**📈 Performance Visualization**

To interpret results, the following visualizations are included:

1. Predicted vs. Actual Curve

Displays how closely forecasted signals match real-time values.

Used to assess time-step accuracy and temporal lag.	
```
import matplotlib.pyplot as plt
plt.plot(y_true, label="Actual", linewidth=2)
plt.plot(y_pred, label="Predicted", linestyle='--')
plt.title("Predicted vs Actual Harmonic Levels")
plt.xlabel("Time")
plt.ylabel("Signal Magnitude")
plt.legend()
plt.show()
```

2. Residual Plot

Shows error distribution over time.

Helps detect systematic bias or underfitting.

3. Anomaly Detection Threshold

Marks areas where deviation exceeds defined thresholds → potential equipment anomaly.
---

​🧠 Explainable AI Insights

To enhance interpretability, optional XAI tools such as SHAP or LIME can be integrated to:

Visualize which time-series features most influence predictions.

Identify conditions (e.g., temperature rise, current imbalance) contributing to anomalies.

Support engineers in decision-making rather than treating the model as a black box.

---

**Predictive Maintenance Lifecycle Diagram**

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/47984bbf-5837-4876-b57b-377e1c231348" />

---
