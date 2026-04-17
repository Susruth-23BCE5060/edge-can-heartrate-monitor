'''"""
=============================================================
  PAMAP2 Edge Device Simulator → ThingsBoard
  File 2 of 4: Simulates N smartwatch patients in real-time
=============================================================
WHAT THIS DOES:
  - Loads the trained CNN model + artifacts from ./model/
  - Reads real PAMAP2 sensor windows as "live" patient data
  - Runs inference every ~1.5 seconds (simulated streaming)
  - Publishes telemetry to ThingsBoard via MQTT (one device per patient):
      heart_rate, acc_magnitude, gyro_magnitude, temperature,
      predicted_activity, confidence, hr_status, overall_status, alert_msg
  - Flags anomalies: BRADYCARDIA / TACHYCARDIA / SENSOR_FAULT / HIGH_MOTION

SETUP (ThingsBoard):
  1. Log in to https://demo.thingsboard.io  (or your local TB instance)
  2. Go to Entities → Devices → click "+ Add Device"
  3. Create one Device per patient  (e.g. "Patient_1", "Patient_2", "Patient_3")
  4. Click each device → Manage Credentials → copy the ACCESS TOKEN
  5. Paste each token into PATIENT_TOKENS below (replace the placeholders)

INSTALL:
  pip install paho-mqtt tensorflow scikit-learn pandas numpy

USAGE:
  python 2_simulate_edge_devices.py

TROUBLESHOOTING:
  - Device shows "Inactive" in ThingsBoard → token is wrong or port 1883 blocked
  - All ❌ in console → MQTT publish failing; check on_connect log printed at startup
  - UNEXPECTED_MOTION for lying → fixed in this version (gravity-aware thresholds)
"""

import os, json, pickle, time, random, threading, warnings
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
import tensorflow as tf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# THINGSBOARD CONFIG  ← EDIT THESE
# ─────────────────────────────────────────────
TB_HOST  = "thingsboard.cloud"   # or "localhost" for local install
TB_PORT  = 1883
TB_TOPIC = "v1/devices/me/telemetry"

# !! IMPORTANT !!
# Replace these placeholder tokens with the ACTUAL Access Tokens
# from your ThingsBoard devices (Entities → Devices → Manage Credentials)
PATIENT_TOKENS = {
    "Patient_3": "",
    "Patient_2": "",
    "Patient_1": "",
}

PUBLISH_INTERVAL = 1.5   # seconds between telemetry pushes
DATA_DIR = "data/PAMAP2_Dataset/Protocol"

# ─────────────────────────────────────────────
# PAMAP2 COLUMN NAMES (same as trainer)
# ─────────────────────────────────────────────
COL_NAMES = ["timestamp", "activityID", "heart_rate"]
for imu in ["hand", "chest", "ankle"]:
    for feat in ["temp","acc16_x","acc16_y","acc16_z","acc6_x","acc6_y","acc6_z",
                 "gyro_x","gyro_y","gyro_z","mag_x","mag_y","mag_z",
                 "ori_0","ori_1","ori_2","ori_3"]:
        COL_NAMES.append(f"{imu}_{feat}")

SENSOR_COLS  = [c for c in COL_NAMES if c not in ["timestamp","activityID"] and "ori_" not in c]
FEATURE_COLS = [c for c in SENSOR_COLS if c != "heart_rate"]
WINDOW_SIZE  = 256
STEP_SIZE    = 64   # faster for simulation

ACTIVITY_NAMES = {
    1:"lying", 2:"sitting", 3:"standing", 4:"walking", 5:"running",
    6:"cycling", 7:"nordic_walking", 9:"watching_TV", 10:"pc_work",
    11:"car_driving", 12:"ascending_stairs", 13:"descending_stairs",
    16:"vacuum_cleaning", 17:"ironing", 18:"folding_laundry",
    19:"house_cleaning", 20:"playing_soccer", 24:"rope_jumping"
}

# ─────────────────────────────────────────────
# MQTT RESULT CODES — for diagnostics
# ─────────────────────────────────────────────
MQTT_RC_CODES = {
    0: "Connection accepted ✅",
    1: "Connection refused — bad protocol version",
    2: "Connection refused — client ID rejected",
    3: "Connection refused — server unavailable (check host/port)",
    4: "Connection refused — bad username/password (WRONG TOKEN?)",
    5: "Connection refused — not authorized",
}

# ─────────────────────────────────────────────
# LOAD MODEL ARTIFACTS
# ─────────────────────────────────────────────
print("Loading model artifacts …")
model = tf.keras.models.load_model("model/cnn_model.h5")

with open("model/scaler.pkl","rb") as f:
    scaler = pickle.load(f)
with open("model/label_encoder.pkl","rb") as f:
    le = pickle.load(f)
with open("model/hr_thresholds.json") as f:
    hr_thresholds = json.load(f)

print(f"  Model input shape : {model.input_shape}")
print(f"  Activities        : {list(le.classes_)}")
print(f"  HR threshold rules: {len(hr_thresholds)} activities loaded")

# ─────────────────────────────────────────────
# ANOMALY DETECTION
# ─────────────────────────────────────────────
def get_hr_status(heart_rate: float, activity: str) -> tuple:
    """
    Returns (status_code, message, severity)
    severity: 0=NORMAL, 1=WARNING, 2=ALARM
    """
    rules = hr_thresholds.get(activity, hr_thresholds.get("walking", {}))
    if not rules:
        return "UNKNOWN", "No threshold data", 0

    hr = heart_rate
    if hr < rules["low_alarm"]:
        return "BRADYCARDIA_ALARM", f"HR {hr:.0f} bpm — dangerously low!", 2
    if hr < rules["low_warn"]:
        return "BRADYCARDIA_WARN",  f"HR {hr:.0f} bpm — below normal for {activity}", 1
    if hr > rules["high_alarm"]:
        return "TACHYCARDIA_ALARM", f"HR {hr:.0f} bpm — dangerously high!", 2
    if hr > rules["high_warn"]:
        return "TACHYCARDIA_WARN",  f"HR {hr:.0f} bpm — elevated for {activity}", 1
    return "NORMAL", f"HR {hr:.0f} bpm — normal", 0


def detect_sensor_fault(segment_df: pd.DataFrame) -> bool:
    """Flag if >20 % of any sensor column is NaN in this window."""
    nan_frac = segment_df[FEATURE_COLS].isna().mean().max()
    return nan_frac > 0.20


def motion_alert(acc_mag: float, activity: str) -> bool:
    """
    Flag unexpected high-magnitude acceleration.

    FIX: PAMAP2 acc16 columns include gravity (~9.8 m/s²) so a stationary
    sensor at rest still shows ~9.8 in the dominant axis.  The original
    threshold of 12.0 for sedentary activities was too close to the gravity
    baseline, causing false UNEXPECTED_MOTION alerts for 'lying', 'sitting',
    etc.  Thresholds below are chosen empirically:
      - lying/sitting/sedentary: 15.0  (gravity + small postural movement)
      - walking/standing/light:  18.0
      - high-intensity sports:   25.0
    """
    HIGH_MOTION_ACTIVITIES = {
        "running", "rope_jumping", "playing_soccer", "nordic_walking", "cycling"
    }
    SEDENTARY_ACTIVITIES = {
        "lying", "sitting", "watching_TV", "pc_work", "car_driving", "ironing",
        "folding_laundry", "house_cleaning", "vacuum_cleaning"
    }

    if activity in HIGH_MOTION_ACTIVITIES:
        threshold = 25.0
    elif activity in SEDENTARY_ACTIVITIES:
        threshold = 15.0   # comfortably above gravity baseline + postural noise
    else:
        threshold = 18.0   # walking, standing, ascending/descending stairs, etc.

    return acc_mag > threshold


# ─────────────────────────────────────────────
# CNN INFERENCE ON A WINDOW
# ─────────────────────────────────────────────
def run_inference(window: np.ndarray) -> tuple:
    """
    window: (WINDOW_SIZE, num_features) — raw (unscaled)
    Returns (activity_name, confidence_pct)
    """
    w, f = window.shape
    scaled = scaler.transform(window).reshape(1, w, f)
    probs  = model.predict(scaled, verbose=0)[0]
    idx    = np.argmax(probs)
    label  = le.inverse_transform([idx])[0]
    name   = ACTIVITY_NAMES.get(int(label), str(label))
    return name, round(float(probs[idx]) * 100, 1)


# ─────────────────────────────────────────────
# LOAD SUBJECT DATA
# ─────────────────────────────────────────────
def load_subject_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=" ", header=None, names=COL_NAMES)
    df.replace("NaN", np.nan, inplace=True)
    df[COL_NAMES[2:]] = df[COL_NAMES[2:]].astype(float)
    df = df[df["activityID"] != 0]
    df[SENSOR_COLS] = df[SENSOR_COLS].interpolate().ffill().bfill()
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# PATIENT SIMULATOR  (one thread per patient)
# ─────────────────────────────────────────────
class PatientDevice:
    def __init__(self, patient_name: str, token: str, subject_file: str):
        self.name         = patient_name
        self.token        = token
        self.data         = load_subject_data(subject_file)
        self.pointer      = 0
        self.connected    = False
        self.connect_rc   = -1

        self.client = mqtt.Client(client_id=f"tb_{patient_name}_{random.randint(1000,9999)}")
        self.client.username_pw_set(token)

        # ── Callbacks for connection diagnostics ──
        self.client.on_connect    = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish    = self._on_publish

        self._connect()

    # ── MQTT callbacks ──────────────────────────
    def _on_connect(self, client, userdata, flags, rc):
        self.connect_rc = rc
        msg = MQTT_RC_CODES.get(rc, f"Unknown rc={rc}")
        if rc == 0:
            self.connected = True
            print(f"  [{self.name}] MQTT connected → {msg}")
        else:
            self.connected = False
            print(f"  [{self.name}] MQTT FAILED    → {msg}")
            if rc == 4:
                print(f"  [{self.name}] ⚠️  Double-check the ACCESS TOKEN for this device in ThingsBoard.")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        if rc != 0:
            print(f"  [{self.name}] Unexpected disconnect (rc={rc}). Will retry …")

    def _on_publish(self, client, userdata, mid):
        pass   # successful publish confirmed by broker

    def _connect(self):
        try:
            self.client.connect(TB_HOST, TB_PORT, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            print(f"  [{self.name}] TCP connection FAILED: {e}")
            print(f"  [{self.name}] ⚠️  Check that port 1883 is not blocked by your firewall/network.")

    # ── Data windowing ───────────────────────────
    def _next_window(self) -> pd.DataFrame:
        start = self.pointer
        end   = start + WINDOW_SIZE
        if end > len(self.data):          # wrap around when we reach the end
            self.pointer = 0
            start, end = 0, WINDOW_SIZE
        segment      = self.data.iloc[start:end].copy()
        self.pointer += STEP_SIZE
        return segment

    # ── Publish one telemetry message ────────────
    def publish_telemetry(self):
        seg = self._next_window()
        win = seg[FEATURE_COLS].values     # (256, num_features)
        hr  = float(seg["heart_rate"].mean())

        # CNN inference
        activity, confidence = run_inference(win)

        # Derived metrics
        acc_cols = [c for c in FEATURE_COLS if "acc16" in c]
        gyr_cols = [c for c in FEATURE_COLS if "gyro"  in c]
        tmp_cols = [c for c in FEATURE_COLS if "temp"  in c]

        acc_mag  = float(np.sqrt((seg[acc_cols].values ** 2).sum(axis=1)).mean())
        gyr_mag  = float(np.sqrt((seg[gyr_cols].values ** 2).sum(axis=1)).mean())
        temp_avg = float(seg[tmp_cols].mean().mean())

        # Anomaly checks
        hr_status, hr_msg, hr_sev = get_hr_status(hr, activity)
        sensor_fault = detect_sensor_fault(seg)
        motion_flag  = motion_alert(acc_mag, activity)

        alerts = []
        if sensor_fault: alerts.append("SENSOR_FAULT")
        if motion_flag:  alerts.append("UNEXPECTED_MOTION")
        if hr_sev >= 1:  alerts.append(hr_status)

        overall_status = "NORMAL"
        if hr_sev == 1 or motion_flag:  overall_status = "WARNING"
        if hr_sev == 2 or sensor_fault: overall_status = "ALARM"

        payload = {
            "heart_rate":        round(hr, 1),
            "acc_magnitude":     round(acc_mag, 3),
            "gyro_magnitude":    round(gyr_mag, 3),
            "temperature":       round(temp_avg, 2),
            "predicted_activity": activity,
            "confidence_pct":    confidence,
            "hr_status":         hr_status,
            "hr_low_alarm":      hr_thresholds.get(activity, {}).get("low_alarm", 40),
            "hr_high_alarm":     hr_thresholds.get(activity, {}).get("high_alarm", 200),
            "sensor_fault":      int(sensor_fault),
            "motion_flag":       int(motion_flag),
            "overall_status":    overall_status,
            "alert_msg":         "; ".join(alerts) if alerts else "All Clear",
        }

        result = self.client.publish(TB_TOPIC, json.dumps(payload), qos=1)

        # rc=0 means the message was queued successfully by paho;
        # actual delivery confirmation comes via on_publish callback
        ok     = result.rc == mqtt.MQTT_ERR_SUCCESS
        icon   = "✅" if ok else "❌"
        status_note = "" if ok else f" (MQTT rc={result.rc})"

        print(
            f"[{self.name}] {icon}  "
            f"Activity={activity:20s} HR={hr:5.1f}  "
            f"AccMag={acc_mag:6.2f}  Status={overall_status:7s}  "
            f"Alert={payload['alert_msg']}{status_note}"
        )
        return payload

    # ── Main loop ────────────────────────────────
    def run_loop(self):
        # Give the MQTT loop a moment to complete the handshake
        time.sleep(1.5)
        if not self.connected:
            print(f"  [{self.name}] ⚠️  Not connected — publishing anyway (messages will queue/fail).")

        print(f"  [{self.name}] Starting simulation loop …")
        while True:
            try:
                self.publish_telemetry()
            except Exception as e:
                print(f"  [{self.name}] Error in publish loop: {e}")
            time.sleep(PUBLISH_INTERVAL + random.uniform(-0.2, 0.2))


# ─────────────────────────────────────────────
# LOCAL-ONLY MODE  (no ThingsBoard token)
# ─────────────────────────────────────────────
def local_loop(patient_name: str, subject_file: str):
    """
    Run inference and print results locally without any MQTT connection.
    Useful for testing the model before setting up ThingsBoard.
    """
    data    = load_subject_data(subject_file)
    pointer = 0

    print(f"  [{patient_name}] Running in LOCAL mode (no ThingsBoard token set).")
    while True:
        start = pointer
        end   = start + WINDOW_SIZE
        if end > len(data):
            pointer = 0
            start, end = 0, WINDOW_SIZE
        seg     = data.iloc[start:end]
        pointer += STEP_SIZE

        win      = seg[FEATURE_COLS].values
        hr       = float(seg["heart_rate"].mean())
        activity, confidence = run_inference(win)
        hr_status, _, hr_sev = get_hr_status(hr, activity)

        icon = "🟡" if hr_sev == 1 else ("🔴" if hr_sev == 2 else "🔵")
        print(
            f"[{patient_name}] {icon} (local)  "
            f"Activity={activity:20s} HR={hr:5.1f}  "
            f"Confidence={confidence:5.1f}%  HRStatus={hr_status}"
        )
        time.sleep(PUBLISH_INTERVAL)


# ─────────────────────────────────────────────
# MAIN — load subjects, spawn threads
# ─────────────────────────────────────────────
def main():
    subject_files = sorted([
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR) if f.endswith(".dat")
    ])

    if not subject_files:
        raise FileNotFoundError(f"No .dat files found in {DATA_DIR}")

    patients = list(PATIENT_TOKENS.items())
    threads  = []

    for i, (patient_name, token) in enumerate(patients):
        subject_file = subject_files[i % len(subject_files)]
        print(f"\nSetting up {patient_name} → {os.path.basename(subject_file)}")

        if "YOUR_TOKEN" in token or not token.strip():
            # ── Local mode: no valid token provided ──
            t = threading.Thread(
                target=local_loop,
                args=(patient_name, subject_file),
                daemon=True
            )
        else:
            # ── ThingsBoard MQTT mode ──
            dev = PatientDevice(patient_name, token, subject_file)
            t   = threading.Thread(target=dev.run_loop, daemon=True)

        threads.append(t)
        t.start()
        time.sleep(0.5)   # stagger thread starts

    print(
        f"\n🚀  Simulating {len(threads)} patient device(s) …  "
        f"Press Ctrl+C to stop.\n"
        f"{'─'*70}"
    )

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()'''

"""
=============================================================
  PAMAP2 Edge Device Simulator → ThingsBoard
  File 2 of 4: Simulates N smartwatch patients in real-time
=============================================================
WHAT THIS DOES:
  - Loads the trained CNN model + artifacts from ./model/
  - Reads real PAMAP2 sensor windows as "live" patient data
  - Runs inference every ~1 second (simulated streaming)
  - Publishes telemetry to ThingsBoard via MQTT (one device per patient):
      heart_rate, acc_magnitude, gyro_magnitude, temperature,
      predicted_activity, confidence, hr_status, overall_status, alert_msg
  - Flags anomalies:  BRADYCARDIA / TACHYCARDIA / SENSOR_FAULT / HIGH_MOTION

SETUP (ThingsBoard):
  1. Log in to https://demo.thingsboard.io  (or your local TB instance)
  2. Create one Device per patient  (e.g. "Patient_1", "Patient_2", …)
  3. Copy each device's ACCESS TOKEN into PATIENT_TOKENS below
  4. pip install paho-mqtt tensorflow scikit-learn pandas numpy

USAGE:
  python 2_simulate_edge_devices.py
"""

import os, json, pickle, time, random, threading, warnings
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
import tensorflow as tf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# THINGSBOARD CONFIG  ← EDIT THESE
# ─────────────────────────────────────────────
TB_HOST = "thingsboard.cloud"   # or "localhost" for local install
TB_PORT = 1883
TB_TOPIC = "v1/devices/me/telemetry"

# Map patient name → ThingsBoard device ACCESS TOKEN
# Create devices in ThingsBoard first, then paste tokens here
PATIENT_TOKENS = {
    "Patient_3": "oTUaBl7OObNa3TRYdKDh",
    "Patient_2": "CdCboTGBpu9qoFwkMdZA",
    "Patient_1": "NEl6YEQQEg4sWV65l2il",
}

PUBLISH_INTERVAL = 1.5   # seconds between telemetry pushes
DATA_DIR = "data/PAMAP2_Dataset/Protocol"

# ─────────────────────────────────────────────
# PAMAP2 COLUMN NAMES (same as trainer)
# ─────────────────────────────────────────────
COL_NAMES = ["timestamp", "activityID", "heart_rate"]
for imu in ["hand", "chest", "ankle"]:
    for feat in ["temp","acc16_x","acc16_y","acc16_z","acc6_x","acc6_y","acc6_z",
                 "gyro_x","gyro_y","gyro_z","mag_x","mag_y","mag_z",
                 "ori_0","ori_1","ori_2","ori_3"]:
        COL_NAMES.append(f"{imu}_{feat}")

SENSOR_COLS  = [c for c in COL_NAMES if c not in ["timestamp","activityID"] and "ori_" not in c]
FEATURE_COLS = [c for c in SENSOR_COLS if c != "heart_rate"]
WINDOW_SIZE  = 256
STEP_SIZE    = 64   # faster for simulation

ACTIVITY_NAMES = {
    1:"lying",2:"sitting",3:"standing",4:"walking",5:"running",
    6:"cycling",7:"nordic_walking",9:"watching_TV",10:"pc_work",
    11:"car_driving",12:"ascending_stairs",13:"descending_stairs",
    16:"vacuum_cleaning",17:"ironing",18:"folding_laundry",
    19:"house_cleaning",20:"playing_soccer",24:"rope_jumping"
}

# Activities where a person is largely stationary.
# Used by BOTH motion_alert (gravity-aware thresholds) and
# get_hr_status (hard clinical HR caps).
SEDENTARY_ACTIVITIES = {
    "lying", "sitting", "watching_TV", "pc_work", "car_driving",
    "ironing", "folding_laundry", "house_cleaning", "vacuum_cleaning"
}

# Hard clinical HR caps for sedentary activities.
# These override data-derived thresholds when the learned value is too
# permissive (e.g. training outliers pushed the learned high_warn above 100).
# A patient lying/sitting with HR > 100 should ALWAYS be flagged.
SEDENTARY_HR_WARN_CAP  = 100   # bpm → WARNING
SEDENTARY_HR_ALARM_CAP = 120   # bpm → ALARM

# ─────────────────────────────────────────────
# LOAD MODEL ARTIFACTS
# ─────────────────────────────────────────────
print("Loading model artifacts …")
model = tf.keras.models.load_model("model/cnn_model.h5")

with open("model/scaler.pkl","rb") as f:
    scaler = pickle.load(f)
with open("model/label_encoder.pkl","rb") as f:
    le = pickle.load(f)
with open("model/hr_thresholds.json") as f:
    hr_thresholds = json.load(f)

print(f"  Model input shape: {model.input_shape}")
print(f"  Activities: {list(le.classes_)}")
print(f"  HR threshold rules for {len(hr_thresholds)} activities")

# ─────────────────────────────────────────────
# ANOMALY DETECTION
# ─────────────────────────────────────────────
def get_hr_status(heart_rate: float, activity: str) -> tuple:
    """
    Returns (status_code, message, severity)
    severity: 0=NORMAL, 1=WARNING, 2=ALARM

    TWO-LAYER LOGIC
    ───────────────
    Layer 1 — Data-derived thresholds from hr_thresholds.json
              (mean ± 2σ warn, mean ± 3σ alarm, computed per activity at
               training time).  Catches activity-specific anomalies.

    Layer 2 — Hard clinical caps for sedentary activities only.
              Runs AFTER layer 1 to catch cases where the learned threshold
              was too permissive (e.g. noisy training data pushed high_warn
              well above 100 bpm for "lying").
              Rule: HR > 100 while lying/sitting → TACHYCARDIA_WARN
                    HR > 120 while lying/sitting → TACHYCARDIA_ALARM
    """
    rules = hr_thresholds.get(activity, hr_thresholds.get("walking", {}))
    hr    = heart_rate

    # ── Layer 1: data-derived ──────────────────────────────────────────
    if rules:
        if hr < rules["low_alarm"]:
            return "BRADYCARDIA_ALARM", f"HR {hr:.0f} bpm — dangerously low!", 2
        if hr < rules["low_warn"]:
            return "BRADYCARDIA_WARN",  f"HR {hr:.0f} bpm — below normal for {activity}", 1
        if hr > rules["high_alarm"]:
            return "TACHYCARDIA_ALARM", f"HR {hr:.0f} bpm — dangerously high!", 2
        if hr > rules["high_warn"]:
            return "TACHYCARDIA_WARN",  f"HR {hr:.0f} bpm — elevated for {activity}", 1

    # ── Layer 2: hard clinical cap for sedentary activities ────────────
    # Only reached when data-derived rule returned NORMAL.  Catches cases
    # where the learned high_warn was too permissive.
    if activity in SEDENTARY_ACTIVITIES:
        if hr > SEDENTARY_HR_ALARM_CAP:
            return "TACHYCARDIA_ALARM", f"HR {hr:.0f} bpm — dangerously high for {activity}!", 2
        if hr > SEDENTARY_HR_WARN_CAP:
            return "TACHYCARDIA_WARN",  f"HR {hr:.0f} bpm — elevated for {activity}", 1

    return "NORMAL", f"HR {hr:.0f} bpm — normal", 0


def detect_sensor_fault(segment_df: pd.DataFrame) -> bool:
    """Flag if >20% of any sensor column is NaN in this window."""
    nan_frac = segment_df[FEATURE_COLS].isna().mean().max()
    return nan_frac > 0.20


def motion_alert(acc_mag: float, activity: str) -> bool:
    """
    Flag unexpected high-magnitude acceleration.

    WHY THE ORIGINAL THRESHOLD (12.0 / 15.0) WAS WRONG
    ─────────────────────────────────────────────────────
    PAMAP2 acc16 columns are RAW accelerometer data that include gravity
    (≈ 9.8 m/s²). The 3-axis vector magnitude is:

        AccMag = sqrt(ax² + ay² + az²)

    For a person lying perfectly still, gravity sits on one axis:
        AccMag_ideal = sqrt(0² + 9.8² + 0²) = 9.8 m/s²

    In practice, sensor tilt + measurement noise push this to 15–18 m/s²
    consistently — which is exactly what you saw in the console output.
    Any sedentary threshold at or below 18 will fire on EVERY lying window.

    CORRECT THRESHOLDS
    ──────────────────
      Sedentary (lying, sitting, …)    19.5 m/s²  — clears the ~17–18 floor
      Light activity (walking, stairs) 22.0 m/s²
      High intensity (running, soccer) 30.0 m/s²
    """
    HIGH_MOTION_ACTIVITIES = {
        "running", "rope_jumping", "playing_soccer", "nordic_walking", "cycling"
    }

    if activity in HIGH_MOTION_ACTIVITIES:
        threshold = 30.0
    elif activity in SEDENTARY_ACTIVITIES:
        threshold = 19.5   # safely above the gravity+noise floor of ~17–18 m/s²
    else:
        threshold = 22.0   # walking, standing, ascending/descending stairs

    return acc_mag > threshold

# ─────────────────────────────────────────────
# CNN INFERENCE ON A WINDOW
# ─────────────────────────────────────────────
def run_inference(window: np.ndarray) -> tuple:
    """
    window: (WINDOW_SIZE, num_features) — raw (unscaled)
    Returns (activity_name, confidence_pct)
    """
    w, f = window.shape
    scaled = scaler.transform(window).reshape(1, w, f)
    probs  = model.predict(scaled, verbose=0)[0]
    idx    = np.argmax(probs)
    label  = le.inverse_transform([idx])[0]
    name   = ACTIVITY_NAMES.get(int(label), str(label))
    return name, round(float(probs[idx]) * 100, 1)

# ─────────────────────────────────────────────
# LOAD SUBJECT DATA
# ─────────────────────────────────────────────
def load_subject_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=" ", header=None, names=COL_NAMES)
    df.replace("NaN", np.nan, inplace=True)
    df[COL_NAMES[2:]] = df[COL_NAMES[2:]].astype(float)
    df = df[df["activityID"] != 0]
    df[SENSOR_COLS] = df[SENSOR_COLS].interpolate().ffill().bfill()
    return df.reset_index(drop=True)

# ─────────────────────────────────────────────
# PATIENT SIMULATOR  (one thread per patient)
# ─────────────────────────────────────────────
class PatientDevice:
    def __init__(self, patient_name: str, token: str, subject_file: str):
        self.name    = patient_name
        self.token   = token
        self.data    = load_subject_data(subject_file)
        self.pointer = 0
        self.client  = mqtt.Client(client_id=f"tb_{patient_name}")
        self.client.username_pw_set(token)
        self._connect()

    def _connect(self):
        try:
            self.client.connect(TB_HOST, TB_PORT, keepalive=60)
            self.client.loop_start()
            print(f"  [{self.name}] Connected to ThingsBoard ({TB_HOST})")
        except Exception as e:
            print(f"  [{self.name}] Connection FAILED: {e}")

    def _next_window(self) -> pd.DataFrame:
        start = self.pointer
        end   = start + WINDOW_SIZE
        if end > len(self.data):
            self.pointer = 0
            start, end = 0, WINDOW_SIZE
        segment = self.data.iloc[start:end]
        self.pointer += STEP_SIZE
        return segment

    def publish_telemetry(self):
        seg = self._next_window()
        win = seg[FEATURE_COLS].values    # (256, num_features)
        hr  = float(seg["heart_rate"].mean())

        # CNN inference
        activity, confidence = run_inference(win)

        # Compute derived metrics
        acc_cols = [c for c in FEATURE_COLS if "acc16" in c]
        gyr_cols = [c for c in FEATURE_COLS if "gyro"  in c]
        acc_mag  = float(np.sqrt((seg[acc_cols].values**2).sum(axis=1)).mean())
        gyr_mag  = float(np.sqrt((seg[gyr_cols].values**2).sum(axis=1)).mean())
        temp_avg = float(seg[[c for c in FEATURE_COLS if "temp" in c]].mean().mean())

        # Anomaly checks
        hr_status, hr_msg, hr_sev = get_hr_status(hr, activity)
        sensor_fault  = detect_sensor_fault(seg)
        motion_flag   = motion_alert(acc_mag, activity)

        alerts = []
        if sensor_fault:  alerts.append("SENSOR_FAULT")
        if motion_flag:   alerts.append("UNEXPECTED_MOTION")
        if hr_sev == 2:   alerts.append(hr_status)
        elif hr_sev == 1: alerts.append(hr_status)

        overall_status = "NORMAL"
        if hr_sev == 1 or motion_flag:  overall_status = "WARNING"
        if hr_sev == 2 or sensor_fault: overall_status = "ALARM"

        payload = {
            "heart_rate":          round(hr, 1),
            "acc_magnitude":       round(acc_mag, 3),
            "gyro_magnitude":      round(gyr_mag, 3),
            "temperature":         round(temp_avg, 2),
            "predicted_activity":  activity,
            "confidence_pct":      confidence,
            "hr_status":           hr_status,
            "hr_low_alarm":        hr_thresholds.get(activity, {}).get("low_alarm", 40),
            "hr_high_alarm":       hr_thresholds.get(activity, {}).get("high_alarm", 200),
            "sensor_fault":        int(sensor_fault),
            "motion_flag":         int(motion_flag),
            "overall_status":      overall_status,
            "alert_msg":           "; ".join(alerts) if alerts else "All Clear",
        }

        result = self.client.publish(TB_TOPIC, json.dumps(payload), qos=1)
        status = "✅" if result.rc == mqtt.MQTT_ERR_SUCCESS else "❌"
        print(
            f"[{self.name}] {status}  Activity={activity:18s} "
            f"HR={hr:5.1f}  Status={overall_status:7s}  "
            f"Alert={payload['alert_msg']}"
        )
        return payload

    def run_loop(self):
        print(f"  [{self.name}] Starting simulation loop …")
        while True:
            try:
                self.publish_telemetry()
            except Exception as e:
                print(f"  [{self.name}] Error: {e}")
            time.sleep(PUBLISH_INTERVAL + random.uniform(-0.2, 0.2))

# ─────────────────────────────────────────────
# MAIN — load subjects, spawn threads
# ─────────────────────────────────────────────
def main():
    subject_files = sorted([
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR) if f.endswith(".dat")
    ])

    if not subject_files:
        raise FileNotFoundError(f"No .dat files found in {DATA_DIR}")

    patients = list(PATIENT_TOKENS.items())
    threads  = []

    for i, (patient_name, token) in enumerate(patients):
        subject_file = subject_files[i % len(subject_files)]
        print(f"\nSetting up {patient_name} → {os.path.basename(subject_file)}")

        if "YOUR_ACCESS_TOKEN" in token:
            print(f"  ⚠️  Token not set for {patient_name} — skipping MQTT, showing local output only.")
            # Still run inference locally so you can test without TB
            dev = PatientDevice.__new__(PatientDevice)
            dev.name    = patient_name
            dev.token   = token
            dev.data    = load_subject_data(subject_file)
            dev.pointer = 0
            dev.client  = None

            def local_loop(d=dev):
                while True:
                    seg = d._next_window()
                    win = seg[FEATURE_COLS].values
                    hr  = float(seg["heart_rate"].mean())
                    activity, confidence = run_inference(win)
                    hr_status, hr_msg, hr_sev = get_hr_status(hr, activity)
                    print(f"[{d.name}] 🔵 (local) Activity={activity:18s} "
                          f"HR={hr:5.1f}  HRStatus={hr_status}")
                    time.sleep(PUBLISH_INTERVAL)

            t = threading.Thread(target=local_loop, daemon=True)
        else:
            dev = PatientDevice(patient_name, token, subject_file)
            t   = threading.Thread(target=dev.run_loop, daemon=True)

        threads.append(t)
        t.start()
        time.sleep(0.5)

    print(f"\n🚀  Simulating {len(threads)} patient device(s) …  Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()