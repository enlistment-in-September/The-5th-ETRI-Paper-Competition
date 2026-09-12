import os
import gc
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

# CONFIG
from config import DATA_DIR as _DATA_DIR, OUTPUT_DIR as _OUTPUT_DIR

DATA_DIR = str(_DATA_DIR)   # 입력: 원본 parquet/csv
BASE_DIR = str(_OUTPUT_DIR)  # 출력: OOF / 제출 파일
TRAIN_CSV = os.path.join(DATA_DIR, "ch2026_metrics_train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "ch2026_submission_sample.csv")

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

SEEDS = [42, 1, 2024, 8765, 9999]
N_FOLDS = 5

LOOKBACK = 14
BATCH_SIZE = 64
EPOCHS = 50
LR = 7e-4
WD = 1e-4
HIDDEN = 96
NUM_LAYERS = 2
DROPOUT = 0.25

MAX_SEQ_FEATURES = 256
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SEED
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEEDS[0])

# UTILS
def pfile(name):
    return os.path.join(DATA_DIR, name)

def safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce")

def ensure_datetime_cols(df, ts_col="timestamp"):
    df = df.copy()
    df[ts_col] = safe_to_datetime(df[ts_col])
    df = df[df[ts_col].notna()].copy()
    df["lifelog_date"] = df[ts_col].dt.date.astype(str)
    df["hour"] = df[ts_col].dt.hour
    return df

def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else np.nan

def safe_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x)) if len(x) else np.nan

def safe_sum(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sum(x)) if len(x) else 0.0

def safe_max(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.max(x)) if len(x) else np.nan

def safe_entropy(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    s = arr.sum()
    if s <= 0:
        return 0.0
    p = arr / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0

def merge_feature_list(base_df, feat_list):
    out = base_df.copy()
    for i, feat in enumerate(feat_list):
        if feat is None or len(feat) == 0:
            continue
        out = out.merge(feat, on=["subject_id", "lifelog_date"], how="left")
        print(f"[MERGE] {i+1}/{len(feat_list)} -> {out.shape}")
    return out

def add_calendar_feats(df):
    df = df.copy()
    dt = pd.to_datetime(df["lifelog_date"])
    df["dayofweek"] = dt.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    return df

def add_subject_stats(df, exclude_cols):
    df = df.copy()
    num_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    for col in num_cols:
        g = df.groupby("subject_id")[col]
        subj_mean = g.transform("mean")
        subj_std = g.transform("std")
        df[f"{col}_subj_mean"] = subj_mean
        df[f"{col}_subj_std"] = subj_std
        df[f"{col}_subj_z"] = (df[col] - subj_mean) / (subj_std + 1e-6)
    return df

def add_recent_stats(df, exclude_cols, max_cols=60):
    df = df.copy()
    df = df.sort_values(["subject_id", "lifelog_date"]).reset_index(drop=True)

    num_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in num_cols if df[c].nunique(dropna=True) > 1][:max_cols]

    for col in num_cols:
        g = df.groupby("subject_id")[col]
        shifted = g.shift(1)

        df[f"{col}_lag1"] = shifted
        df[f"{col}_lag2"] = g.shift(2)
        df[f"{col}_diff1"] = df[col] - shifted

        df[f"{col}_roll3_mean"] = (
            shifted.groupby(df["subject_id"])
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"{col}_roll7_mean"] = (
            shifted.groupby(df["subject_id"])
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return df

def agg_num_features(df, value_col, prefix):
    if value_col not in df.columns:
        return None
    return (
        df.groupby(["subject_id", "lifelog_date"], as_index=False)
          .agg(**{
              f"{prefix}_mean": (value_col, "mean"),
              f"{prefix}_std": (value_col, "std"),
              f"{prefix}_min": (value_col, "min"),
              f"{prefix}_max": (value_col, "max"),
              f"{prefix}_median": (value_col, "median"),
              f"{prefix}_count": (value_col, "count"),
          })
    )

def agg_hour_bins(df, value_col, prefix):
    if value_col not in df.columns:
        return None

    bins = {
        "night": [0,1,2,3,4,5],
        "morning": [6,7,8,9,10,11],
        "afternoon": [12,13,14,15,16,17],
        "evening": [18,19,20,21,22,23],
    }

    out = (
        df.groupby(["subject_id", "lifelog_date"], as_index=False)
          .agg(**{f"{prefix}_daily_mean": (value_col, "mean")})
    )

    for name, hours in bins.items():
        tmp = (
            df[df["hour"].isin(hours)]
            .groupby(["subject_id", "lifelog_date"], as_index=False)
            .agg(**{f"{prefix}_{name}_mean": (value_col, "mean")})
        )
        out = out.merge(tmp, on=["subject_id", "lifelog_date"], how="left")

    for name in bins:
        out[f"{prefix}_{name}_ratio"] = out[f"{prefix}_{name}_mean"] / (out[f"{prefix}_daily_mean"] + 1e-6)

    return out

def sanitize_numeric_df(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    return df

def clip_probs(x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.5, posinf=1.0-eps, neginf=eps)
    x = np.clip(x, eps, 1.0 - eps)
    return x

# LOAD
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

train["lifelog_date"] = pd.to_datetime(train["lifelog_date"]).dt.date.astype(str)
train["sleep_date"]   = pd.to_datetime(train["sleep_date"]).dt.date.astype(str)

test["lifelog_date"] = pd.to_datetime(test["lifelog_date"]).dt.date.astype(str)
test["sleep_date"]   = pd.to_datetime(test["sleep_date"]).dt.date.astype(str)

base_all = pd.concat(
    [
        train[["subject_id", "lifelog_date"]],
        test[["subject_id", "lifelog_date"]],
    ],
    axis=0
).drop_duplicates().reset_index(drop=True)

print("train:", train.shape, "test:", test.shape, "base_all:", base_all.shape)

# FEATURE ENGINEERING
feature_tables = []

# 1) mLight
if os.path.exists(pfile("ch2025_mLight.parquet")):
    df = pd.read_parquet(pfile("ch2025_mLight.parquet"))
    df = ensure_datetime_cols(df)
    feature_tables += [agg_num_features(df, "m_light", "m_light"), agg_hour_bins(df, "m_light", "m_light")]
    del df; gc.collect()

# 2) wLight
if os.path.exists(pfile("ch2025_wLight.parquet")):
    df = pd.read_parquet(pfile("ch2025_wLight.parquet"))
    df = ensure_datetime_cols(df)
    feature_tables += [agg_num_features(df, "w_light", "w_light"), agg_hour_bins(df, "w_light", "w_light")]
    del df; gc.collect()

# 3) mACStatus
if os.path.exists(pfile("ch2025_mACStatus.parquet")):
    df = pd.read_parquet(pfile("ch2025_mACStatus.parquet"))
    df = ensure_datetime_cols(df)
    if "m_charging" in df.columns:
        feat1 = (
            df.groupby(["subject_id", "lifelog_date"], as_index=False)
              .agg(
                  m_charging_mean=("m_charging", "mean"),
                  m_charging_sum=("m_charging", "sum"),
                  m_charging_std=("m_charging", "std"),
                  m_charging_count=("m_charging", "count"),
              )
        )
        feat2 = agg_hour_bins(df, "m_charging", "m_charging")
        feature_tables += [feat1, feat2]
    del df; gc.collect()

# 4) mScreenStatus
if os.path.exists(pfile("ch2025_mScreenStatus.parquet")):
    df = pd.read_parquet(pfile("ch2025_mScreenStatus.parquet"))
    df = ensure_datetime_cols(df)
    if "m_screen_use" in df.columns:
        feat1 = (
            df.groupby(["subject_id", "lifelog_date"], as_index=False)
              .agg(
                  m_screen_use_mean=("m_screen_use", "mean"),
                  m_screen_use_sum=("m_screen_use", "sum"),
                  m_screen_use_std=("m_screen_use", "std"),
                  m_screen_use_count=("m_screen_use", "count"),
              )
        )
        feat2 = agg_hour_bins(df, "m_screen_use", "m_screen_use")
        feature_tables += [feat1, feat2]
    del df; gc.collect()

# 5) mActivity
if os.path.exists(pfile("ch2025_mActivity.parquet")):
    df = pd.read_parquet(pfile("ch2025_mActivity.parquet"))
    df = ensure_datetime_cols(df)
    if "m_activity" in df.columns:
        feat_num = (
            df.groupby(["subject_id", "lifelog_date"], as_index=False)
              .agg(
                  m_activity_mean=("m_activity", "mean"),
                  m_activity_std=("m_activity", "std"),
                  m_activity_min=("m_activity", "min"),
                  m_activity_max=("m_activity", "max"),
                  m_activity_median=("m_activity", "median"),
                  m_activity_nunique=("m_activity", "nunique"),
                  m_activity_count=("m_activity", "count"),
              )
        )
        act_counts = (
            df.groupby(["subject_id", "lifelog_date", "m_activity"])
              .size()
              .reset_index(name="cnt")
        )
        pivot = act_counts.pivot_table(
            index=["subject_id", "lifelog_date"],
            columns="m_activity",
            values="cnt",
            fill_value=0
        )
        pivot.columns = [f"m_activity_code_{c}_cnt" for c in pivot.columns]
        pivot = pivot.reset_index()
        feature_tables += [feat_num, pivot]
    del df; gc.collect()

# 6) wPedo
if os.path.exists(pfile("ch2025_wPedo.parquet")):
    df = pd.read_parquet(pfile("ch2025_wPedo.parquet"))
    df = ensure_datetime_cols(df)
    pedo_cols = ["step", "step_frequency", "running_step", "walking_step", "distance", "speed", "burned_calories"]
    use_cols = [c for c in pedo_cols if c in df.columns]
    if len(use_cols):
        agg_dict = {}
        for c in use_cols:
            agg_dict[f"{c}_mean"] = (c, "mean")
            agg_dict[f"{c}_sum"] = (c, "sum")
            agg_dict[f"{c}_std"] = (c, "std")
            agg_dict[f"{c}_max"] = (c, "max")
            agg_dict[f"{c}_median"] = (c, "median")
        feat = df.groupby(["subject_id", "lifelog_date"], as_index=False).agg(**agg_dict)
        feature_tables.append(feat)
    del df; gc.collect()

# 7) wHr
if os.path.exists(pfile("ch2025_wHr.parquet")):
    df = pd.read_parquet(pfile("ch2025_wHr.parquet"))
    df = ensure_datetime_cols(df)
    if "heart_rate" in df.columns:
        rows = []
        for r in df.itertuples(index=False):
            hr = getattr(r, "heart_rate", None)
            try:
                arr = np.asarray(hr, dtype=float) if isinstance(hr, (list, tuple, np.ndarray)) else np.asarray([hr], dtype=float)
                arr = arr[np.isfinite(arr)]
            except:
                arr = np.array([])
            if len(arr) == 0:
                rows.append([r.subject_id, r.lifelog_date, np.nan, np.nan, np.nan, np.nan, 0])
            else:
                rows.append([r.subject_id, r.lifelog_date, arr.mean(), arr.std(), arr.min(), arr.max(), len(arr)])

        hr_df = pd.DataFrame(rows, columns=["subject_id","lifelog_date","hr_mean","hr_std","hr_min","hr_max","hr_len"])
        feat = (
            hr_df.groupby(["subject_id", "lifelog_date"], as_index=False)
                 .agg(
                     hr_mean_mean=("hr_mean", "mean"),
                     hr_mean_std=("hr_mean", "std"),
                     hr_std_mean=("hr_std", "mean"),
                     hr_std_max=("hr_std", "max"),
                     hr_min_mean=("hr_min", "mean"),
                     hr_max_mean=("hr_max", "mean"),
                     hr_len_sum=("hr_len", "sum"),
                     hr_len_mean=("hr_len", "mean"),
                 )
        )
        feature_tables.append(feat)
    del df; gc.collect()

# 8) mUsageStats
if os.path.exists(pfile("ch2025_mUsageStats.parquet")):
    df = pd.read_parquet(pfile("ch2025_mUsageStats.parquet"))
    df = ensure_datetime_cols(df)
    if "m_usage_stats" in df.columns:
        rows = []
        for r in df.itertuples(index=False):
            items = getattr(r, "m_usage_stats", None)
            if items is None or not isinstance(items, (list, tuple)) or len(items) == 0:
                rows.append([r.subject_id, r.lifelog_date, 0, 0, 0.0, 0.0])
                continue

            total_times, app_names = [], []
            for item in items:
                if isinstance(item, dict):
                    app_names.append(str(item.get("app_name", "UNK")))
                    try:
                        total_times.append(float(item.get("total_time", 0)))
                    except:
                        total_times.append(0.0)

            rows.append([
                r.subject_id,
                r.lifelog_date,
                len(items),
                len(set(app_names)),
                safe_sum(total_times),
                safe_entropy(total_times),
            ])

        feat = pd.DataFrame(rows, columns=[
            "subject_id", "lifelog_date",
            "usage_event_app_cnt", "usage_event_app_unique",
            "usage_event_total_time", "usage_event_time_entropy"
        ])
        feat = (
            feat.groupby(["subject_id", "lifelog_date"], as_index=False)
                .agg(
                    usage_event_app_cnt_mean=("usage_event_app_cnt", "mean"),
                    usage_event_app_cnt_sum=("usage_event_app_cnt", "sum"),
                    usage_event_app_unique_mean=("usage_event_app_unique", "mean"),
                    usage_event_app_unique_sum=("usage_event_app_unique", "sum"),
                    usage_event_total_time_mean=("usage_event_total_time", "mean"),
                    usage_event_total_time_sum=("usage_event_total_time", "sum"),
                    usage_event_time_entropy_mean=("usage_event_time_entropy", "mean"),
                    usage_event_time_entropy_max=("usage_event_time_entropy", "max"),
                )
        )
        feature_tables.append(feat)
    del df; gc.collect()

# 9) mWifi
if os.path.exists(pfile("ch2025_mWifi.parquet")):
    df = pd.read_parquet(pfile("ch2025_mWifi.parquet"))
    df = ensure_datetime_cols(df)
    if "m_wifi" in df.columns:
        rows = []
        for r in df.itertuples(index=False):
            items = getattr(r, "m_wifi", None)
            if items is None or not isinstance(items, (list, tuple)) or len(items) == 0:
                rows.append([r.subject_id, r.lifelog_date, 0, 0, np.nan, np.nan])
                continue

            rssis, bssids = [], []
            for item in items:
                if isinstance(item, dict):
                    bssids.append(str(item.get("bssid", "UNK")))
                    try:
                        rssis.append(float(item.get("rssi", np.nan)))
                    except:
                        pass

            rows.append([r.subject_id, r.lifelog_date, len(items), len(set(bssids)), safe_mean(rssis), safe_max(rssis)])

        feat = pd.DataFrame(rows, columns=[
            "subject_id","lifelog_date","wifi_scan_cnt","wifi_unique_bssid","wifi_rssi_mean","wifi_rssi_max"
        ])
        feat = (
            feat.groupby(["subject_id", "lifelog_date"], as_index=False)
                .agg(
                    wifi_scan_cnt_mean=("wifi_scan_cnt", "mean"),
                    wifi_scan_cnt_sum=("wifi_scan_cnt", "sum"),
                    wifi_unique_bssid_mean=("wifi_unique_bssid", "mean"),
                    wifi_unique_bssid_sum=("wifi_unique_bssid", "sum"),
                    wifi_rssi_mean_mean=("wifi_rssi_mean", "mean"),
                    wifi_rssi_mean_std=("wifi_rssi_mean", "std"),
                    wifi_rssi_max_mean=("wifi_rssi_max", "mean"),
                    wifi_rssi_max_max=("wifi_rssi_max", "max"),
                )
        )
        feature_tables.append(feat)
    del df; gc.collect()

# 10) mBle
if os.path.exists(pfile("ch2025_mBle.parquet")):
    df = pd.read_parquet(pfile("ch2025_mBle.parquet"))
    df = ensure_datetime_cols(df)
    if "m_ble" in df.columns:
        rows = []
        for r in df.itertuples(index=False):
            items = getattr(r, "m_ble", None)
            if items is None or not isinstance(items, (list, tuple)) or len(items) == 0:
                rows.append([r.subject_id, r.lifelog_date, 0, 0, np.nan, 0])
                continue

            rssis, addrs, classes = [], [], []
            for item in items:
                if isinstance(item, dict):
                    addrs.append(str(item.get("address", "UNK")))
                    classes.append(str(item.get("device_class", "UNK")))
                    try:
                        rssis.append(float(item.get("rssi", np.nan)))
                    except:
                        pass

            rows.append([r.subject_id, r.lifelog_date, len(items), len(set(addrs)), safe_mean(rssis), len(set(classes))])

        feat = pd.DataFrame(rows, columns=[
            "subject_id","lifelog_date","ble_scan_cnt","ble_unique_addr","ble_rssi_mean","ble_device_class_unique"
        ])
        feat = (
            feat.groupby(["subject_id", "lifelog_date"], as_index=False)
                .agg(
                    ble_scan_cnt_mean=("ble_scan_cnt", "mean"),
                    ble_scan_cnt_sum=("ble_scan_cnt", "sum"),
                    ble_unique_addr_mean=("ble_unique_addr", "mean"),
                    ble_unique_addr_sum=("ble_unique_addr", "sum"),
                    ble_rssi_mean_mean=("ble_rssi_mean", "mean"),
                    ble_rssi_mean_std=("ble_rssi_mean", "std"),
                    ble_device_class_unique_mean=("ble_device_class_unique", "mean"),
                    ble_device_class_unique_sum=("ble_device_class_unique", "sum"),
                )
        )
        feature_tables.append(feat)
    del df; gc.collect()

# 11) mGps
if os.path.exists(pfile("ch2025_mGps.parquet")):
    df = pd.read_parquet(pfile("ch2025_mGps.parquet"))
    df = ensure_datetime_cols(df)
    if "m_gps" in df.columns:
        rows = []
        for r in df.itertuples(index=False):
            items = getattr(r, "m_gps", None)
            if items is None or not isinstance(items, (list, tuple)) or len(items) == 0:
                rows.append([r.subject_id, r.lifelog_date, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                continue

            lats, lons, alts, speeds = [], [], [], []
            for item in items:
                if isinstance(item, dict):
                    try: lats.append(float(item.get("latitude", np.nan)))
                    except: pass
                    try: lons.append(float(item.get("longitude", np.nan)))
                    except: pass
                    try: alts.append(float(item.get("altitude", np.nan)))
                    except: pass
                    try: speeds.append(float(item.get("speed", np.nan)))
                    except: pass

            lat_std = safe_std(lats)
            lon_std = safe_std(lons)
            spread = np.sqrt(lat_std**2 + lon_std**2) if np.isfinite(lat_std) and np.isfinite(lon_std) else np.nan

            rows.append([
                r.subject_id, r.lifelog_date, len(items),
                safe_mean(lats), safe_mean(lons),
                lat_std, lon_std,
                safe_mean(speeds), spread
            ])

        feat = pd.DataFrame(rows, columns=[
            "subject_id","lifelog_date","gps_point_cnt","gps_lat_mean","gps_lon_mean",
            "gps_lat_std","gps_lon_std","gps_speed_mean","gps_spatial_spread"
        ])
        feat = (
            feat.groupby(["subject_id", "lifelog_date"], as_index=False)
                .agg(
                    gps_point_cnt_mean=("gps_point_cnt", "mean"),
                    gps_point_cnt_sum=("gps_point_cnt", "sum"),
                    gps_lat_mean_mean=("gps_lat_mean", "mean"),
                    gps_lon_mean_mean=("gps_lon_mean", "mean"),
                    gps_lat_std_mean=("gps_lat_std", "mean"),
                    gps_lon_std_mean=("gps_lon_std", "mean"),
                    gps_speed_mean_mean=("gps_speed_mean", "mean"),
                    gps_speed_mean_max=("gps_speed_mean", "max"),
                    gps_spatial_spread_mean=("gps_spatial_spread", "mean"),
                    gps_spatial_spread_max=("gps_spatial_spread", "max"),
                )
        )
        feature_tables.append(feat)
    del df; gc.collect()

# 12) mAmbience
if os.path.exists(pfile("ch2025_mAmbience.parquet")):
    df = pd.read_parquet(pfile("ch2025_mAmbience.parquet"))
    df = ensure_datetime_cols(df)
    if "m_ambience" in df.columns:
        rows = []
        for r in df.itertuples(index=False):
            items = getattr(r, "m_ambience", None)
            if items is None or not isinstance(items, (list, tuple)) or len(items) == 0:
                rows.append([r.subject_id, r.lifelog_date, 0, 0, 0.0, 0.0, 0.0, 0.0])
                continue

            labels = []
            music_score, speech_score, vehicle_score, outside_score = 0.0, 0.0, 0.0, 0.0

            for item in items:
                label, score = None, 0.0
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    label = str(item[0])
                    try: score = float(item[1])
                    except: score = 0.0
                elif isinstance(item, dict):
                    label = str(item.get("label", "UNK"))
                    try: score = float(item.get("score", 0.0))
                    except: score = 0.0

                if label is None:
                    continue

                labels.append(label)
                if "Music" in label: music_score += score
                if "Speech" in label: speech_score += score
                if ("Vehicle" in label) or ("Car" in label) or ("Truck" in label): vehicle_score += score
                if "Outside" in label: outside_score += score

            rows.append([
                r.subject_id, r.lifelog_date,
                len(labels), len(set(labels)),
                music_score, speech_score, vehicle_score, outside_score
            ])

        feat = pd.DataFrame(rows, columns=[
            "subject_id","lifelog_date","amb_label_cnt","amb_label_unique",
            "amb_music_score","amb_speech_score","amb_vehicle_score","amb_outside_score"
        ])
        feat = (
            feat.groupby(["subject_id", "lifelog_date"], as_index=False)
                .agg(
                    amb_label_cnt_mean=("amb_label_cnt", "mean"),
                    amb_label_cnt_sum=("amb_label_cnt", "sum"),
                    amb_label_unique_mean=("amb_label_unique", "mean"),
                    amb_label_unique_sum=("amb_label_unique", "sum"),
                    amb_music_score_mean=("amb_music_score", "mean"),
                    amb_music_score_sum=("amb_music_score", "sum"),
                    amb_speech_score_mean=("amb_speech_score", "mean"),
                    amb_speech_score_sum=("amb_speech_score", "sum"),
                    amb_vehicle_score_mean=("amb_vehicle_score", "mean"),
                    amb_vehicle_score_sum=("amb_vehicle_score", "sum"),
                    amb_outside_score_mean=("amb_outside_score", "mean"),
                    amb_outside_score_sum=("amb_outside_score", "sum"),
                )
        )
        feature_tables.append(feat)
    del df; gc.collect()

feature_tables = [x for x in feature_tables if x is not None and len(x) > 0]
print("feature_tables:", len(feature_tables))

# DAILY TABLE
daily_feat = merge_feature_list(base_all, feature_tables)
daily_feat = add_calendar_feats(daily_feat)
daily_feat["num_missing"] = daily_feat.isna().sum(axis=1)

exclude_cols = ["subject_id", "lifelog_date"]
daily_feat = add_subject_stats(daily_feat, exclude_cols=exclude_cols)
daily_feat = add_recent_stats(daily_feat, exclude_cols=exclude_cols, max_cols=60)
daily_feat = sanitize_numeric_df(daily_feat)

print("daily_feat:", daily_feat.shape)

train_df = train.merge(daily_feat, on=["subject_id", "lifelog_date"], how="left")
test_df  = test.merge(daily_feat, on=["subject_id", "lifelog_date"], how="left")

print("train_df:", train_df.shape)
print("test_df :", test_df.shape)

# TIME-AWARE FOLD
def assign_time_folds(df, n_folds=5):
    df = df.copy()
    df["fold"] = -1
    df["date_dt"] = pd.to_datetime(df["lifelog_date"])

    for sid in sorted(df["subject_id"].unique()):
        idx = df[df["subject_id"] == sid].sort_values("date_dt").index.tolist()
        chunks = np.array_split(idx, n_folds)
        for f, chunk in enumerate(chunks):
            df.loc[chunk, "fold"] = f

    return df.drop(columns=["date_dt"])

train_df = assign_time_folds(train_df, N_FOLDS)

# FEATURE SELECTION FOR LSTM
DROP_COLS = ["sleep_date", "fold"] + TARGETS
FEATURE_COLS = [c for c in train_df.columns if c not in DROP_COLS]
BASE_SEQ_FEATURE_COLS = [c for c in FEATURE_COLS if c not in ["subject_id", "lifelog_date"]]
BASE_SEQ_FEATURE_COLS = [c for c in BASE_SEQ_FEATURE_COLS if pd.api.types.is_numeric_dtype(train_df[c])]

print("raw num seq features:", len(BASE_SEQ_FEATURE_COLS))

def select_stable_features(train_part, cols, max_features=256, missing_th=0.95):
    tmp = train_part[cols].copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan)

    miss = tmp.isna().mean()
    valid_cols = miss[miss < missing_th].index.tolist()

    if len(valid_cols) == 0:
        return cols[:min(len(cols), max_features)]

    var_series = tmp[valid_cols].fillna(tmp[valid_cols].median()).var()
    var_series = var_series.sort_values(ascending=False)

    selected = var_series.index.tolist()[:max_features]
    return selected

# SCALER
def fit_feature_stats_from_daily(daily_part, cols):
    x = daily_part[cols].copy()
    x = x.replace([np.inf, -np.inf], np.nan)

    med = x.median()
    x = x.fillna(med)
    x = x.replace([np.inf, -np.inf], 0.0)
    x = x.fillna(0.0)

    scaler = StandardScaler()
    scaler.fit(x)
    return med, scaler

def transform_features(df, cols, med, scaler):
    x = df[cols].copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(med)
    x = x.replace([np.inf, -np.inf], 0.0)
    x = x.fillna(0.0)

    arr = scaler.transform(x)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    x = pd.DataFrame(arr, columns=cols, index=df.index)
    return x

# SUBJECT ENCODING
all_subjects = sorted(pd.concat([train_df["subject_id"], test_df["subject_id"]]).unique())
subject2idx = {s: i for i, s in enumerate(all_subjects)}

# BUILD DAILY ARRAYS
def build_subject_daily_arrays(daily_df, feature_cols, med, scaler):
    out = {}
    tmp = daily_df[["subject_id", "lifelog_date"] + feature_cols].copy()
    tmp["date_dt"] = pd.to_datetime(tmp["lifelog_date"])

    tmp_scaled = transform_features(tmp, feature_cols, med, scaler)
    tmp2 = pd.concat(
        [
            tmp[["subject_id", "lifelog_date", "date_dt"]].reset_index(drop=True),
            tmp_scaled.reset_index(drop=True),
        ],
        axis=1
    )

    for sid, sdf in tmp2.groupby("subject_id"):
        sdf = sdf.sort_values("date_dt").reset_index(drop=True)
        feat = sdf[feature_cols].values.astype(np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        out[sid] = {
            "dates": sdf["lifelog_date"].tolist(),
            "date_dt": sdf["date_dt"].tolist(),
            "feat": feat,
        }
    return out

# DATASET
class SleepSequenceDataset(Dataset):
    def __init__(self, sample_df, subject_daily_dict, feature_cols, lookback=14, targets=None, is_test=False):
        self.df = sample_df.reset_index(drop=True).copy()
        self.subject_daily_dict = subject_daily_dict
        self.feature_cols = feature_cols
        self.lookback = lookback
        self.targets = targets
        self.is_test = is_test
        self.df["date_dt"] = pd.to_datetime(self.df["lifelog_date"])

    def __len__(self):
        return len(self.df)

    def _make_seq(self, subject_id, current_date):
        info = self.subject_daily_dict[subject_id]
        dates = info["date_dt"]
        feat = info["feat"]

        idxs = [i for i, d in enumerate(dates) if d <= current_date]
        idxs = idxs[-self.lookback:]

        seq = np.zeros((self.lookback, feat.shape[1]), dtype=np.float32)
        mask = np.zeros((self.lookback,), dtype=np.float32)

        if len(idxs) > 0:
            use_feat = feat[idxs]
            use_feat = np.nan_to_num(use_feat, nan=0.0, posinf=0.0, neginf=0.0)
            seq[-len(idxs):] = use_feat
            mask[-len(idxs):] = 1.0

        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        return seq, mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row["subject_id"]
        sidx = subject2idx[sid]
        current_date = row["date_dt"]

        seq, mask = self._make_seq(sid, current_date)
        static = seq[-1].copy()

        item = {
            "seq": torch.tensor(seq, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "static": torch.tensor(static, dtype=torch.float32),
            "subject_id": torch.tensor(sidx, dtype=torch.long),
        }

        if not self.is_test:
            y = row[self.targets].values.astype(np.float32)
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
            item["target"] = torch.tensor(y, dtype=torch.float32)

        return item

# MODEL
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask):
        score = self.attn(x).squeeze(-1)
        score = score.masked_fill(mask == 0, -1e9)
        weight = torch.softmax(score, dim=1)
        pooled = torch.bmm(weight.unsqueeze(1), x).squeeze(1)
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        return pooled, weight

class MultiTaskBiLSTM(nn.Module):
    def __init__(self, input_dim, n_subjects, hidden_dim=96, num_layers=2, dropout=0.25, out_dim=7):
        super().__init__()

        self.subject_emb = nn.Embedding(n_subjects, 12)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.attn_pool = AttentionPooling(hidden_dim * 2)

        self.static_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        fusion_dim = hidden_dim * 2 + hidden_dim * 2 + hidden_dim + 12

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, seq, mask, static, subject_id):
        seq = torch.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        static = torch.nan_to_num(static, nan=0.0, posinf=0.0, neginf=0.0)

        x = self.input_proj(seq)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        lstm_out, _ = self.lstm(x)
        lstm_out = torch.nan_to_num(lstm_out, nan=0.0, posinf=0.0, neginf=0.0)

        attn_vec, _ = self.attn_pool(lstm_out, mask)

        lengths = mask.sum(dim=1).long().clamp(min=1)
        last_idx = lengths - 1
        batch_idx = torch.arange(seq.size(0), device=seq.device)
        last_vec = lstm_out[batch_idx, last_idx]

        static_vec = self.static_mlp(static)
        subj_vec = self.subject_emb(subject_id)

        fused = torch.cat([attn_vec, last_vec, static_vec, subj_vec], dim=1)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)

        logits = self.head(fused)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        return logits

# TRAIN / VALID
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []

    for batch in loader:
        seq = batch["seq"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        static = batch["static"].to(DEVICE)
        subject_id = batch["subject_id"].to(DEVICE)
        target = batch["target"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(seq, mask, static, subject_id)
        loss = criterion(logits, target)

        if not torch.isfinite(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses)) if len(losses) else 999.0

@torch.no_grad()
def valid_one_epoch(model, loader, criterion):
    model.eval()
    losses = []
    preds = []

    for batch in loader:
        seq = batch["seq"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        static = batch["static"].to(DEVICE)
        subject_id = batch["subject_id"].to(DEVICE)
        target = batch["target"].to(DEVICE)

        logits = model(seq, mask, static, subject_id)
        loss = criterion(logits, target)

        if torch.isfinite(loss):
            losses.append(loss.item())

        p = torch.sigmoid(logits).cpu().numpy()
        p = clip_probs(p)
        preds.append(p)

    preds = np.concatenate(preds, axis=0)
    return (float(np.mean(losses)) if len(losses) else 999.0), preds

@torch.no_grad()
def predict_loader(model, loader):
    model.eval()
    preds = []

    for batch in loader:
        seq = batch["seq"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        static = batch["static"].to(DEVICE)
        subject_id = batch["subject_id"].to(DEVICE)

        logits = model(seq, mask, static, subject_id)
        p = torch.sigmoid(logits).cpu().numpy()
        p = clip_probs(p)
        preds.append(p)

    return np.concatenate(preds, axis=0)

def multi_target_logloss(y_true, y_pred, targets):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = clip_probs(y_pred)

    scores = {}
    vals = []
    for i, t in enumerate(targets):
        s = log_loss(y_true[:, i], y_pred[:, i], labels=[0, 1])
        scores[t] = s
        vals.append(s)
    scores["avg"] = float(np.mean(vals))
    return scores


# BASE FEATURE LIST
DROP_COLS = ["sleep_date", "fold"] + TARGETS
FEATURE_COLS = [c for c in train_df.columns if c not in DROP_COLS]
BASE_SEQ_FEATURE_COLS = [c for c in FEATURE_COLS if c not in ["subject_id", "lifelog_date"]]
BASE_SEQ_FEATURE_COLS = [c for c in BASE_SEQ_FEATURE_COLS if pd.api.types.is_numeric_dtype(train_df[c])]
print("raw num seq features:", len(BASE_SEQ_FEATURE_COLS))

# OOF TRAINING
oof_pred = np.zeros((len(train_df), len(TARGETS)), dtype=np.float32)
test_pred = np.zeros((len(test_df), len(TARGETS)), dtype=np.float32)

for seed in SEEDS:
    seed_everything(seed)
    print(f"\n================ SEED {seed} ================")

    oof_seed = np.zeros((len(train_df), len(TARGETS)), dtype=np.float32)
    test_seed = np.zeros((len(test_df), len(TARGETS)), dtype=np.float32)

    for fold in range(N_FOLDS):
        print(f"\n---- FOLD {fold} ----")

        tr_idx = train_df[train_df["fold"] != fold].index
        va_idx = train_df[train_df["fold"] == fold].index

        tr_part = train_df.loc[tr_idx].copy()
        va_part = train_df.loc[va_idx].copy()

        selected_cols = select_stable_features(
            tr_part, BASE_SEQ_FEATURE_COLS,
            max_features=MAX_SEQ_FEATURES, missing_th=0.95
        )

        train_subjects = tr_part["subject_id"].unique().tolist()
        daily_tr = daily_feat[daily_feat["subject_id"].isin(train_subjects)].copy()

        med, scaler = fit_feature_stats_from_daily(daily_tr, selected_cols)
        subject_daily_dict = build_subject_daily_arrays(daily_feat, selected_cols, med, scaler)

        tr_ds = SleepSequenceDataset(tr_part, subject_daily_dict, selected_cols, lookback=LOOKBACK, targets=TARGETS,
                                     is_test=False)
        va_ds = SleepSequenceDataset(va_part, subject_daily_dict, selected_cols, lookback=LOOKBACK, targets=TARGETS,
                                     is_test=False)
        te_ds = SleepSequenceDataset(test_df, subject_daily_dict, selected_cols, lookback=LOOKBACK, targets=TARGETS,
                                     is_test=True)

        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        model = MultiTaskBiLSTM(
            input_dim=len(selected_cols),
            n_subjects=len(subject2idx),
            hidden_dim=HIDDEN,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            out_dim=len(TARGETS),
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        criterion = nn.BCEWithLogitsLoss()

        best_score = 1e9
        best_state = None
        patience = 8
        wait = 0

        y_val = va_part[TARGETS].values.astype(np.float32)

        for epoch in range(1, EPOCHS + 1):
            tr_loss = train_one_epoch(model, tr_loader, optimizer, criterion)
            va_loss, va_pred = valid_one_epoch(model, va_loader, criterion)
            va_score = multi_target_logloss(y_val, va_pred, TARGETS)["avg"]

            print(
                f"Seed {seed} Fold {fold} | Epoch {epoch:02d} | train {tr_loss:.5f} | valid {va_loss:.5f} | logloss {va_score:.5f}")

            if va_score < best_score:
                best_score = va_score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        model.load_state_dict(best_state)

        fold_oof = predict_loader(model, va_loader)
        oof_seed[va_idx] = fold_oof

        fold_test = predict_loader(model, te_loader)
        test_seed += fold_test / N_FOLDS

        fold_scores = multi_target_logloss(y_val, fold_oof, TARGETS)
        print(f"[Seed {seed} FOLD {fold}] scores: {fold_scores}")

        del model, tr_loader, va_loader, te_loader, tr_ds, va_ds, te_ds, subject_daily_dict
        gc.collect()
        torch.cuda.empty_cache()

    oof_pred += oof_seed / len(SEEDS)
    test_pred += test_seed / len(SEEDS)
# FINAL OOF SCORE
final_scores = multi_target_logloss(train_df[TARGETS].values.astype(np.float32), oof_pred, TARGETS)

print("\n========== LSTM OOF ==========")
for k, v in final_scores.items():
    print(f"{k}: {v:.6f}")

# SAVE
oof_df = train_df[["subject_id", "sleep_date", "lifelog_date"] + TARGETS].copy()
for i, t in enumerate(TARGETS):
    oof_df[f"{t}_pred"] = clip_probs(oof_pred[:, i])

oof_path = os.path.join(BASE_DIR, "oof_5fold_bilstm_multitask_safe.csv")
oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
print("saved:", oof_path)

sub = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
for i, t in enumerate(TARGETS):
    sub[t] = clip_probs(test_pred[:, i])

sub_path = os.path.join(BASE_DIR, "submission_5fold_bilstm_multitask_safe.csv")
sub.to_csv(sub_path, index=False, encoding="utf-8-sig")
print("saved:", sub_path)
