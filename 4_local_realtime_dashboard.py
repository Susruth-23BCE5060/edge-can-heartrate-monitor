"""
=============================================================
  Local Real-Time Dashboard  (no ThingsBoard needed)
  File 4 of 4: Live Matplotlib plots for testing & demo
=============================================================
WHAT THIS DOES:
  - Streams inferences from the CNN on real PAMAP2 windows
  - Draws live graphs: Heart Rate, Acc/Gyro, Confidence
  - Shows current Activity label + Status badge (NORMAL/WARN/ALARM)
  - Useful for demos, screenshots, or when offline

USAGE:
  python 4_local_realtime_dashboard.py

INSTALL:
  pip install tensorflow scikit-learn pandas numpy matplotlib
"""

import os, json, pickle, time, warnings
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import tensorflow as tf

warnings.filterwarnings("ignore")
plt.style.use("dark_background")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR       = "data/PAMAP2_Dataset/Protocol"
SUBJECT_FILE   = None   # None = first .dat found
WINDOW_SIZE    = 256
STEP_SIZE      = 50
HISTORY_POINTS = 120    # x-axis length of live charts
UPDATE_MS      = 800    # plot refresh interval (ms)
PATIENT_ID     = "Patient_1"

ACTIVITY_NAMES = {
    1:"lying",2:"sitting",3:"standing",4:"walking",5:"running",
    6:"cycling",7:"nordic_walking",9:"watching_TV",10:"pc_work",
    11:"car_driving",12:"ascending_stairs",13:"descending_stairs",
    16:"vacuum_cleaning",17:"ironing",18:"folding_laundry",
    19:"house_cleaning",20:"playing_soccer",24:"rope_jumping"
}

STATUS_COLORS = {"NORMAL":"#00e676","WARNING":"#ffab40","ALARM":"#ff1744"}

COL_NAMES = ["timestamp","activityID","heart_rate"]
for imu in ["hand","chest","ankle"]:
    for feat in ["temp","acc16_x","acc16_y","acc16_z","acc6_x","acc6_y","acc6_z",
                 "gyro_x","gyro_y","gyro_z","mag_x","mag_y","mag_z",
                 "ori_0","ori_1","ori_2","ori_3"]:
        COL_NAMES.append(f"{imu}_{feat}")

SENSOR_COLS  = [c for c in COL_NAMES if c not in ["timestamp","activityID"] and "ori_" not in c]
FEATURE_COLS = [c for c in SENSOR_COLS if c != "heart_rate"]

# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
print("Loading model …")
model  = tf.keras.models.load_model("model/cnn_model.h5")
with open("model/scaler.pkl","rb") as f:  scaler = pickle.load(f)
with open("model/label_encoder.pkl","rb") as f:  le = pickle.load(f)
with open("model/hr_thresholds.json") as f:  hr_thresholds = json.load(f)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
if SUBJECT_FILE is None:
    files = sorted([os.path.join(DATA_DIR,f)
                    for f in os.listdir(DATA_DIR) if f.endswith(".dat")])
    SUBJECT_FILE = files[0]

print(f"Streaming from: {os.path.basename(SUBJECT_FILE)}")
raw_df = pd.read_csv(SUBJECT_FILE, sep=" ", header=None, names=COL_NAMES)
raw_df.replace("NaN", np.nan, inplace=True)
raw_df[COL_NAMES[2:]] = raw_df[COL_NAMES[2:]].astype(float)
raw_df = raw_df[raw_df["activityID"] != 0].reset_index(drop=True)
raw_df[SENSOR_COLS] = raw_df[SENSOR_COLS].interpolate().ffill().bfill()

# ─────────────────────────────────────────────
# ANOMALY HELPERS
# ─────────────────────────────────────────────
def get_hr_status(hr, activity):
    rules = hr_thresholds.get(activity, hr_thresholds.get("walking",{}))
    if not rules:
        return "UNKNOWN", 0
    if hr < rules.get("low_alarm",40):  return "BRADYCARDIA_ALARM", 2
    if hr < rules.get("low_warn",50):   return "BRADYCARDIA_WARN",  1
    if hr > rules.get("high_alarm",200):return "TACHYCARDIA_ALARM", 2
    if hr > rules.get("high_warn",170): return "TACHYCARDIA_WARN",  1
    return "NORMAL", 0

def run_inference(window):
    w, f    = window.shape
    scaled  = scaler.transform(window).reshape(1,w,f)
    probs   = model.predict(scaled, verbose=0)[0]
    idx     = np.argmax(probs)
    label   = le.inverse_transform([idx])[0]
    name    = ACTIVITY_NAMES.get(int(label), str(label))
    return name, float(probs[idx])*100, probs

# ─────────────────────────────────────────────
# STREAMING STATE
# ─────────────────────────────────────────────
pointer = [0]

hr_hist    = deque([0.0]*HISTORY_POINTS, maxlen=HISTORY_POINTS)
acc_hist   = deque([0.0]*HISTORY_POINTS, maxlen=HISTORY_POINTS)
gyr_hist   = deque([0.0]*HISTORY_POINTS, maxlen=HISTORY_POINTS)
conf_hist  = deque([0.0]*HISTORY_POINTS, maxlen=HISTORY_POINTS)
t_hist     = deque(range(HISTORY_POINTS), maxlen=HISTORY_POINTS)
counter    = [0]

current = {
    "activity":"---","confidence":0.0,"hr":0.0,
    "hr_status":"NORMAL","overall_status":"NORMAL",
    "alert_msg":"","probs":np.zeros(len(le.classes_))
}

def advance():
    p = pointer[0]
    if p + WINDOW_SIZE > len(raw_df):
        pointer[0] = 0
        p = 0
    seg = raw_df.iloc[p:p+WINDOW_SIZE]
    pointer[0] += STEP_SIZE
    counter[0]  += 1

    win  = seg[FEATURE_COLS].values
    hr   = float(seg["heart_rate"].mean())
    acc_cols = [c for c in FEATURE_COLS if "acc16" in c]
    gyr_cols = [c for c in FEATURE_COLS if "gyro"  in c]
    acc_mag  = float(np.sqrt((seg[acc_cols].values**2).sum(axis=1)).mean())
    gyr_mag  = float(np.sqrt((seg[gyr_cols].values**2).sum(axis=1)).mean())

    activity, conf, probs = run_inference(win)
    hr_status, hr_sev = get_hr_status(hr, activity)

    overall = "NORMAL"
    if hr_sev == 1:  overall = "WARNING"
    if hr_sev == 2:  overall = "ALARM"

    alerts = []
    if hr_sev > 0:  alerts.append(hr_status)

    current.update({
        "activity":activity,"confidence":conf,"hr":hr,
        "hr_status":hr_status,"overall_status":overall,
        "alert_msg":"; ".join(alerts) if alerts else "All Clear",
        "probs":probs
    })

    hr_hist.append(hr)
    acc_hist.append(acc_mag)
    gyr_hist.append(gyr_mag)
    conf_hist.append(conf)
    t_hist.append(counter[0])

# ─────────────────────────────────────────────
# BUILD FIGURE
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11), facecolor="#0d1117")
fig.suptitle(f"🏥  Hospital Edge Device Monitor — {PATIENT_ID}",
             fontsize=16, color="white", weight="bold", y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Row 0: status cards
ax_status  = fig.add_subplot(gs[0, 0])
ax_hr_val  = fig.add_subplot(gs[0, 1])
ax_act_val = fig.add_subplot(gs[0, 2])

# Row 1: time series charts
ax_hr_ts   = fig.add_subplot(gs[1, :2])
ax_acc_ts  = fig.add_subplot(gs[1, 2])

# Row 2: gyro + bar chart
ax_gyr_ts  = fig.add_subplot(gs[2, :2])
ax_bar     = fig.add_subplot(gs[2, 2])

for ax in [ax_status, ax_hr_val, ax_act_val]:
    ax.set_facecolor("#161b22");  ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# ─────────────────────────────────────────────
# ANIMATE
# ─────────────────────────────────────────────
def update(_):
    advance()

    xs = list(t_hist)

    # ---- status card ----
    ax_status.cla(); ax_status.set_facecolor("#161b22")
    ax_status.set_xticks([]); ax_status.set_yticks([])
    s = current["overall_status"]
    col = STATUS_COLORS.get(s, "white")
    ax_status.set_facecolor(col + "22")   # translucent tint
    ax_status.text(0.5, 0.55, s, transform=ax_status.transAxes,
                   ha="center", va="center", fontsize=22, weight="bold", color=col)
    ax_status.text(0.5, 0.2, current["alert_msg"], transform=ax_status.transAxes,
                   ha="center", va="center", fontsize=8, color="#aaaaaa",
                   wrap=True)
    ax_status.set_title("Overall Status", color="white", fontsize=10, pad=4)

    # ---- HR value card ----
    ax_hr_val.cla(); ax_hr_val.set_facecolor("#161b22")
    ax_hr_val.set_xticks([]); ax_hr_val.set_yticks([])
    hr_col = STATUS_COLORS.get(
        "NORMAL" if "WARN" not in current["hr_status"] and "ALARM" not in current["hr_status"]
        else "WARNING" if "WARN" in current["hr_status"] else "ALARM", "white")
    ax_hr_val.text(0.5, 0.55, f"{current['hr']:.0f}", transform=ax_hr_val.transAxes,
                   ha="center", va="center", fontsize=34, weight="bold", color=hr_col)
    ax_hr_val.text(0.5, 0.15, "bpm", transform=ax_hr_val.transAxes,
                   ha="center", fontsize=12, color="#888888")
    ax_hr_val.set_title("Heart Rate", color="white", fontsize=10, pad=4)

    # ---- Activity card ----
    ax_act_val.cla(); ax_act_val.set_facecolor("#161b22")
    ax_act_val.set_xticks([]); ax_act_val.set_yticks([])
    ax_act_val.text(0.5, 0.6, current["activity"].replace("_"," ").title(),
                    transform=ax_act_val.transAxes,
                    ha="center", va="center", fontsize=14, weight="bold", color="#58a6ff",
                    wrap=True)
    ax_act_val.text(0.5, 0.2, f"Confidence: {current['confidence']:.1f}%",
                    transform=ax_act_val.transAxes,
                    ha="center", fontsize=9, color="#aaaaaa")
    ax_act_val.set_title("Predicted Activity", color="white", fontsize=10, pad=4)

    # ---- HR time series ----
    ax_hr_ts.cla(); ax_hr_ts.set_facecolor("#0d1117")
    rules = hr_thresholds.get(current["activity"], {})
    ax_hr_ts.fill_between(xs,
        rules.get("low_warn",50), rules.get("high_warn",170),
        color="#00e676", alpha=0.08, label="Normal band")
    ax_hr_ts.axhline(rules.get("low_alarm",40),  color="#ff1744", ls="--", lw=0.8, alpha=0.7)
    ax_hr_ts.axhline(rules.get("high_alarm",200), color="#ff1744", ls="--", lw=0.8, alpha=0.7)
    ax_hr_ts.plot(xs, list(hr_hist), color="#e53935", lw=1.8)
    ax_hr_ts.set_ylabel("bpm", color="white"); ax_hr_ts.tick_params(colors="white")
    ax_hr_ts.set_facecolor("#0d1117"); ax_hr_ts.set_title("Heart Rate", color="white", pad=4)
    for sp in ax_hr_ts.spines.values(): sp.set_edgecolor("#30363d")

    # ---- Acc time series ----
    ax_acc_ts.cla(); ax_acc_ts.set_facecolor("#0d1117")
    ax_acc_ts.plot(xs, list(acc_hist), color="#1976D2", lw=1.5)
    ax_acc_ts.set_ylabel("m/s²", color="white"); ax_acc_ts.tick_params(colors="white")
    ax_acc_ts.set_title("Acc Magnitude", color="white", pad=4)
    for sp in ax_acc_ts.spines.values(): sp.set_edgecolor("#30363d")

    # ---- Gyro time series ----
    ax_gyr_ts.cla(); ax_gyr_ts.set_facecolor("#0d1117")
    ax_gyr_ts.plot(xs, list(gyr_hist), color="#7B1FA2", lw=1.5)
    ax_gyr_ts.set_ylabel("rad/s", color="white"); ax_gyr_ts.tick_params(colors="white")
    ax_gyr_ts.set_title("Gyro Magnitude", color="white", pad=4)
    for sp in ax_gyr_ts.spines.values(): sp.set_edgecolor("#30363d")

    # ---- Confidence bar chart ----
    ax_bar.cla(); ax_bar.set_facecolor("#0d1117")
    class_names = [ACTIVITY_NAMES.get(int(c), str(c))[:12] for c in le.classes_]
    probs = current["probs"]
    top_n = 6
    top_idx = np.argsort(probs)[-top_n:][::-1]
    colors_bar = ["#58a6ff" if i != np.argmax(probs) else "#00e676" for i in top_idx]
    ax_bar.barh([class_names[i] for i in top_idx], probs[top_idx], color=colors_bar)
    ax_bar.set_xlim(0, 1); ax_bar.tick_params(colors="white")
    ax_bar.set_title("Top Activity Probs", color="white", pad=4)
    for sp in ax_bar.spines.values(): sp.set_edgecolor("#30363d")
    ax_bar.set_facecolor("#0d1117")

from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, update, interval=UPDATE_MS, cache_frame_data=False)

print("🖥️   Local dashboard running — close the window to stop.")
plt.show()