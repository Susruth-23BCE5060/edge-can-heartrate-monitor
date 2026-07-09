# 🏥 Hospital Edge Device Monitor & Real-Time Activity Recognition

An end-to-end IoT and Deep Learning pipeline that performs real-time Human Activity Recognition (HAR) and patient health anomaly detection using raw smartwatch sensor data. The project processes the clinical **PAMAP2 Dataset**, builds a custom **1D-Convolutional Neural Network (1D-CNN)** for inference, simulates multi-patient streams via **MQTT**, and visualizes real-time status telemetry on an IoT platform dashboard (**ThingsBoard**).

---

## 📌 Project Architecture & Pipeline

The framework is structured as a **4-stage decoupled pipeline** optimized for hospital deployment:

```
  ┌────────────────────────┐      Preprocessed       ┌──────────────────────────┐
  │ 1. CNN Model Trainer   │  ───────────────────>   │ 2. Edge Device Simulator │
  │   (Artifact Generation)│                         │  (Inference Engine)      │
  └────────────────────────┘                         └──────────────────────────┘
                                                               │
                                                 Inference &   │ MQTT Broker
                                                 Telemetry     │ (Port 1883)
                                                               ▼
  ┌────────────────────────┐                         ┌──────────────────────────┐
  │ 4. Local Dashboard     │ <────────────────────── │ 3. ThingsBoard IoT Cloud │
  │ (Matplotlib Live Plot) │    Optional Offline     │ (Production Dashboard)   │
  └────────────────────────┘         Testing         └──────────────────────────┘

```

1. **`1_train_cnn_model.py` (Trainer):** Cleans, linearly interpolates, and extracts overlapping sliding windows ($2.56\text{ s}$ windows @ $100\text{ Hz}$) from raw PAMAP2 telemetry. Fits a 3-block 1D-CNN with Global Average Pooling and extracts data-derived baseline physiological rules[cite: 3].
2. **`2_simulate_edge_devices.py` (Edge Core):** A multi-threaded simulation engine mapping individual patients to background threads. Each thread feeds data windows into the scaled CNN, evaluates a **Two-Layer Clinical Anomaly Logic**, and streams telemetry packets via MQTT[cite: 2].
3. **`dashboard_config.json` (IoT Config):** A ready-to-import configuration manifest enabling rapid widget setup and alarm routing rule chains inside ThingsBoard[cite: 4].
4. **`4_local_realtime_dashboard.py` (Local UI):** A standalone UI displaying continuous streaming plots, top class probability distributions, and a system health state machine using a dark-mode Matplotlib loop[cite: 1, 3].

---

## 🔬 Model Design & Anomaly Detection Logic

### 1D-CNN Deep Learning Model

The model architecture uses a 1D-CNN to process multi-channel IMU time series without manual feature engineering[cite: 3]:

* **Block 1 & 2:** Conv1D layers (filters: 64, 128; kernels: 11, 7) coupled with Batch Normalization, Max Pooling ($2\times$), and Spatial Dropout to capture structural context[cite: 3].
* **Block 3:** High-capacity Conv1D (256 filters, kernel 5) followed by a **Global Average Pooling 1D** layer to reduce spatial dimensions while minimizing overfitting[cite: 3].
* **Head:** Dense layers with Softmax activation outputting confidence values for 18 distinct physiological activities[cite: 3].

### Two-Layer Clinical Anomaly State Machine

Unlike naive threshold engines, this project uses an intelligent hybrid safety runtime:

* **Layer 1 (Data-Driven Baselines):** Quantifies standard deviation bands ($\mu \pm 2\sigma$ for Warning, $\mu \pm 3\sigma$ for Critical Alarm) dynamically evaluated *per activity* during training (e.g., catching high pulse spikes relative to a resting state)[cite: 2, 3].
* **Layer 2 (Hard Clinical Overrides):** Protects against outliers in training data by capping sedentary thresholds. For example, if a patient is classified as `lying` or `sitting`, any heart rate exceeding **$100\text{ bpm}$** triggers `TACHYCARDIA_WARN`, and **$120\text{ bpm}$** triggers `TACHYCARDIA_ALARM`, overriding any learned rules[cite: 2].
* **Gravity-Aware Vector Magnitude Thresholding:** To prevent false positives from constant gravitational forces ($\approx 9.8\text{ m/s}^2$) hitting sensor plates, anomalous motion is calculated via full 3-axis vector magnitudes[cite: 2]:

$$\text{AccMag} = \sqrt{a_x^2 + a_y^2 + a_z^2}$$



Thresholds are adaptively scaled based on the classified activity context (Sedentary: $19.5\text{ m/s}^2$, Light: $22.0\text{ m/s}^2$, Intense Sports: $30.0\text{ m/s}^2$)[cite: 2].

---

## 🛠️ Installation & Setup

### 1. Prerequisites

Ensure you have Python 3.8+ installed. Install the explicit tracking dependencies[cite: 1, 3]:

```bash
pip install tensorflow paho-mqtt scikit-learn pandas numpy matplotlib

```

### 2. Dataset Provisioning

1. Download the clinical **PAMAP2 Physical Activity Monitoring Dataset** (available via Kaggle or the UCI Machine Learning Repository)[cite: 3].
2. Extract the archive content into the following path structured in the project directory[cite: 3]:

```
data/PAMAP2_Dataset/Protocol/subject101.dat

```

---

## 🚀 Execution Guide

### Step 1: Train the Deep Learning Artifacts

Run the compilation pipeline to filter raw records, construct features, and dump trained model serialization files into `./model/`[cite: 3]:

```bash
python 1_train_cnn_model.py

```

*Outputs generated:* `cnn_model.h5`, `scaler.pkl`, `label_encoder.pkl`, `hr_thresholds.json`, and `training_history.png`[cite: 3].

### Step 2: Configure the IoT Cloud Portal (ThingsBoard)

1. Log in to your account at [ThingsBoard Live Demo Portal](https://demo.thingsboard.io) or load your local environment[cite: 1, 2].
2. Navigate to **Entities** ➡️ **Devices** ➡️ Click **`+ Add Device`**[cite: 1, 4].
3. Provision your clinical client instances, naming them exactly `Patient_1`, `Patient_2`, and `Patient_3`[cite: 1, 4].
4. Open each device record, select **Manage Credentials**, and copy the uniquely generated **Access Token** string[cite: 1].
5. Update the configuration map inside `2_simulate_edge_devices.py`[cite: 1, 2]:
```python
PATIENT_TOKENS = {
    "Patient_3": "YOUR_PATIENT_3_TOKEN_HERE",
    "Patient_2": "YOUR_PATIENT_2_TOKEN_HERE",
    "Patient_1": "YOUR_PATIENT_1_TOKEN_HERE",
}

```



### Step 3: Run the Multi-Threaded Edge Simulator

Launch the real-time simulation streaming engine to start parsing multi-patient directories and publishing MQTT payloads[cite: 1, 2]:

```bash
python 2_simulate_edge_devices.py

```

*(Note: If a device token field remains unpopulated or uses placeholders, the code automatically falls back to safe local-only monitoring logs to ensure continuity)[cite: 2].*

### Step 4: Launch the Dashboards

To view your production widgets, import `dashboard_config.json` via the **Dashboards** menu in your ThingsBoard UI[cite: 4].

If you prefer offline validation, launch the local dashboard to open an advanced monitoring window right on your workstation[cite: 3]:

```bash
python 4_local_realtime_dashboard.py

```

---

## 📊 Telemetry Schema

The simulation engine constructs and publishes structured JSON telemetry objects to the ThingsBoard target topic `v1/devices/me/telemetry` every $1.5\text{ seconds}$[cite: 1, 2]:

```json
{
  "heart_rate": 78.4,
  "acc_magnitude": 16.421,
  "gyro_magnitude": 0.114,
  "temperature": 32.45,
  "predicted_activity": "sitting",
  "confidence_pct": 98.7,
  "hr_status": "NORMAL",
  "hr_low_alarm": 45,
  "hr_high_alarm": 100,
  "sensor_fault": 0,
  "motion_flag": 0,
  "overall_status": "NORMAL",
  "alert_msg": "All Clear"
}

```

---

## 📂 Repository Layout

```
├── 1_train_cnn_model.py          # Deep Learning Model compiling script
├── 2_simulate_edge_devices.py    # Threaded MQTT core publisher simulation
├── 4_local_realtime_dashboard.py # Live Matplotlib UI renderer
├── dashboard_config.json         # ThingsBoard importable layout definition
├── data/                         # Patient data directory
│   └── PAMAP2_Dataset/Protocol/  # Extracted .dat target stream files
└── model/                        # Serialized deep learning artifacts
    ├── cnn_model.h5              # Trained Conv1D parameters
    ├── hr_thresholds.json        # Activity-correlated safety arrays
    └── scaler.pkl                # Standard Normalizer object

```
