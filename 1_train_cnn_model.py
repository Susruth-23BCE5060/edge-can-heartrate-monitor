"""
=============================================================
  PAMAP2 CNN Model Trainer — Activity Recognition + Anomaly
  File 1 of 4: Train the model, save artifacts
=============================================================
WHAT THIS DOES:
  - Loads PAMAP2 .dat files from data/PAMAP2_Dataset/Protocol/
  - Cleans & interpolates missing values
  - Builds sliding-window segments (2.56 s @ 100 Hz = 256 samples)
  - Trains a 1-D CNN for activity recognition
  - Also fits per-class heart-rate thresholds used as anomaly rules
  - Saves:
      model/cnn_model.h5          (trained Keras model)
      model/label_encoder.pkl     (activity label encoder)
      model/scaler.pkl            (StandardScaler for features)
      model/hr_thresholds.json    (normal HR ranges per activity)

INSTALL:
  pip install tensorflow scikit-learn pandas numpy matplotlib
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf  # type: ignore
from tensorflow.keras import layers, models, callbacks

warnings.filterwarnings("ignore")
os.makedirs("model", exist_ok=True)

# ─────────────────────────────────────────────
# 1. DATASET CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = "data/PAMAP2_Dataset/Protocol"   # <-- unzip Kaggle data here
WINDOW_SIZE = 256    # 2.56 s at 100 Hz
STEP_SIZE   = 128    # 50 % overlap
FS          = 100    # sampling frequency

# 18 activity labels (PAMAP2 spec)
ACTIVITY_NAMES = {
    1: "lying",       2: "sitting",      3: "standing",    4: "walking",
    5: "running",     6: "cycling",      7: "nordic_walking", 9: "watching_TV",
    10: "pc_work",   11: "car_driving", 12: "ascending_stairs",
    13: "descending_stairs", 16: "vacuum_cleaning", 17: "ironing",
    18: "folding_laundry",   19: "house_cleaning",  20: "playing_soccer",
    24: "rope_jumping"
}
REMOVE_ACTIVITY = 0   # transient / unlabelled

# 54 raw columns per PAMAP2 spec
COL_NAMES = ["timestamp", "activityID", "heart_rate"]
for imu in ["hand", "chest", "ankle"]:
    for feat in ["temp",
                 "acc16_x","acc16_y","acc16_z",
                 "acc6_x", "acc6_y", "acc6_z",
                 "gyro_x", "gyro_y", "gyro_z",
                 "mag_x",  "mag_y",  "mag_z",
                 "ori_0",  "ori_1",  "ori_2",  "ori_3"]:
        COL_NAMES.append(f"{imu}_{feat}")

# Useful sensor features (drop orientation as recommended by authors)
SENSOR_COLS = [c for c in COL_NAMES
               if c not in ["timestamp", "activityID"]
               and "ori_" not in c]

# ─────────────────────────────────────────────
# 2. LOAD & CLEAN DATA
# ─────────────────────────────────────────────
def load_subject(path):
    df = pd.read_csv(path, sep=" ", header=None, names=COL_NAMES)
    df.replace("NaN", np.nan, inplace=True)
    df[COL_NAMES[2:]] = df[COL_NAMES[2:]].astype(float)
    df = df[df["activityID"] != REMOVE_ACTIVITY]
    # forward-fill then back-fill missing sensor values
    df[SENSOR_COLS] = df[SENSOR_COLS].interpolate(method="linear").ffill().bfill()
    return df

print("Loading PAMAP2 subjects …")
dfs = []
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(
        f"\n  [!] DATA_DIR not found: {DATA_DIR}\n"
        "  Please download PAMAP2_Dataset from Kaggle and extract it so that\n"
        "  'data/PAMAP2_Dataset/Protocol/subject101.dat' exists.\n"
    )

for f in sorted(os.listdir(DATA_DIR)):
    if f.endswith(".dat"):
        fp = os.path.join(DATA_DIR, f)
        print(f"  Loading {f} …", end=" ")
        sub = load_subject(fp)
        print(f"{len(sub):,} rows, activities: {sorted(sub['activityID'].unique())}")
        dfs.append(sub)

full_df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal rows: {len(full_df):,}")
print("Activity distribution:\n", full_df["activityID"].value_counts())

# ─────────────────────────────────────────────
# 3. HEART-RATE THRESHOLDS (per activity)
#    Used as medical anomaly rules at runtime
# ─────────────────────────────────────────────
hr_stats = full_df.groupby("activityID")["heart_rate"].agg(["mean","std"]).to_dict("index")
hr_thresholds = {}
for act_id, stats in hr_stats.items():
    mean, std = stats["mean"], stats["std"]
    name = ACTIVITY_NAMES.get(int(act_id), f"activity_{int(act_id)}")
    hr_thresholds[name] = {
        "low_alarm":  round(max(40,  mean - 3*std), 1),
        "low_warn":   round(max(50,  mean - 2*std), 1),
        "mean":       round(mean, 1),
        "high_warn":  round(min(200, mean + 2*std), 1),
        "high_alarm": round(min(220, mean + 3*std), 1),
    }

with open("model/hr_thresholds.json", "w") as f:
    json.dump(hr_thresholds, f, indent=2)
print("\nHeart-rate thresholds saved → model/hr_thresholds.json")

# ─────────────────────────────────────────────
# 4. SLIDING WINDOW SEGMENTATION
# ─────────────────────────────────────────────
print("\nBuilding sliding-window segments …")
feature_cols = [c for c in SENSOR_COLS if c != "heart_rate"]

X_list, y_list = [], []
for act_id in full_df["activityID"].unique():
    sub = full_df[full_df["activityID"] == act_id]
    arr = sub[feature_cols].values
    for start in range(0, len(arr) - WINDOW_SIZE, STEP_SIZE):
        seg = arr[start:start + WINDOW_SIZE]
        if seg.shape[0] == WINDOW_SIZE:
            X_list.append(seg)
            y_list.append(int(act_id))

X = np.array(X_list)   # (N, 256, num_features)
y_raw = np.array(y_list)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(le.classes_)

print(f"  Segments: {X.shape}  Classes: {num_classes}")

# Scale features (fit on flattened, reshape back)
scaler = StandardScaler()
n, w, f = X.shape
X_flat = X.reshape(-1, f)
X_scaled = scaler.fit_transform(X_flat).reshape(n, w, f)

with open("model/scaler.pkl", "wb") as fh:
    pickle.dump(scaler, fh)
with open("model/label_encoder.pkl", "wb") as fh:
    pickle.dump(le, fh)
print("  Scaler & label encoder saved.")

# ─────────────────────────────────────────────
# 5. TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

print(f"\nTrain: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# ─────────────────────────────────────────────
# 6. CNN MODEL
# ─────────────────────────────────────────────
def build_cnn(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv1D(64, 11, activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.Conv1D(128, 7, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Block 3
    x = layers.Conv1D(256, 5, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)

    # Dense head
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_cnn((WINDOW_SIZE, X_train.shape[2]), num_classes)
model.summary()

# ─────────────────────────────────────────────
# 7. TRAIN
# ─────────────────────────────────────────────
cbs = [
    callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
    callbacks.ModelCheckpoint("model/cnn_model.h5", save_best_only=True, verbose=1),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=cbs,
    verbose=1
)

# ─────────────────────────────────────────────
# 8. EVALUATE
# ─────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅  Test accuracy: {acc*100:.2f}%   Loss: {loss:.4f}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
label_names = [ACTIVITY_NAMES.get(int(c), str(c)) for c in le.classes_]
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_names))

# ─────────────────────────────────────────────
# 9. SAVE TRAINING PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history["accuracy"],    label="Train")
axes[0].plot(history.history["val_accuracy"],label="Val")
axes[0].set_title("Accuracy");  axes[0].legend(); axes[0].grid(True)

axes[1].plot(history.history["loss"],    label="Train")
axes[1].plot(history.history["val_loss"],label="Val")
axes[1].set_title("Loss");  axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig("model/training_history.png", dpi=120)
print("\nTraining plot saved → model/training_history.png")
print("\n🎉  Training complete!  All artifacts in ./model/")
print("    Next → run  2_simulate_edge_devices.py")