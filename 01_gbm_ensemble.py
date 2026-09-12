import warnings


warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from scipy.optimize import minimize
from sklearn.metrics import f1_score, classification_report

# 경로 설정

from config import DATA_DIR as ITEMS_DIR, OUTPUT_DIR

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']

# 라벨 데이터 로드

print('Loading raw data...')

train_df = pd.read_csv(ITEMS_DIR / 'ch2026_metrics_train.csv')
test_df = pd.read_csv(ITEMS_DIR / 'ch2026_submission_sample.csv')

train_df['lifelog_date'] = pd.to_datetime(train_df['lifelog_date'])
test_df['lifelog_date'] = pd.to_datetime(test_df['lifelog_date'])
train_df['sleep_date'] = pd.to_datetime(train_df['sleep_date'])
test_df['sleep_date'] = pd.to_datetime(test_df['sleep_date'])

all_keys = pd.concat([
    train_df[['subject_id', 'lifelog_date']],
    test_df[['subject_id', 'lifelog_date']]
]).drop_duplicates().reset_index(drop=True)

sleep_keys = pd.concat([
    train_df[['subject_id', 'sleep_date']],
    test_df[['subject_id', 'sleep_date']]
]).drop_duplicates().reset_index(drop=True)


# 공통 유틸 함수

def agg_stats(vals, prefix):
    vals = np.asarray(vals, dtype=float)

    if len(vals) == 0:
        return {
            f'{prefix}_mean': np.nan,
            f'{prefix}_std': np.nan,
            f'{prefix}_min': np.nan,
            f'{prefix}_max': np.nan,
            f'{prefix}_median': np.nan,
            f'{prefix}_q25': np.nan,
            f'{prefix}_q75': np.nan
        }

    return {
        f'{prefix}_mean': np.nanmean(vals),
        f'{prefix}_std': np.nanstd(vals),
        f'{prefix}_min': np.nanmin(vals),
        f'{prefix}_max': np.nanmax(vals),
        f'{prefix}_median': np.nanmedian(vals),
        f'{prefix}_q25': np.nanpercentile(vals, 25),
        f'{prefix}_q75': np.nanpercentile(vals, 75),
    }


def safe_mean(vals):
    arr = np.asarray(vals, dtype=float)
    return np.nanmean(arr) if len(arr) > 0 else np.nan


def parse_hr(v, lo=40, hi=200):
    """심박수 파싱 + 생리적 범위[40, 200] 필터링. 범위 외 이상치는 제외."""
    try:
        if isinstance(v, (list, np.ndarray)):
            arr = np.asarray(v, dtype=float).ravel()
        else:
            arr = np.array([float(v)])

        arr = arr[(arr >= lo) & (arr <= hi)]
    except Exception:
        arr = np.array([])

    return arr


# 데이터 로드
def load_parquet(name):
    df = pd.read_parquet(ITEMS_DIR / f'ch2025_{name}.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


# 라이프로그 10분 단위 레코드를 하루 단위로 집계

def extract_activity(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        acts = grp['m_activity'].values
        h = grp['timestamp'].dt.hour.values
        for a in [0, 3, 4, 7, 8]:
            row[f'act_{a}_ratio'] = (acts == a).mean()
        row['act_active_ratio'] = ((acts == 7) | (acts == 8) | (acts == 3)).mean()
        row['act_still_ratio'] = (acts == 0).mean()
        row['act_n_records'] = len(acts)
        for seg, mask in [('morn', (h >= 6) & (h < 12)), ('aftn', (h >= 12) & (h < 18)),
                          ('eve', (h >= 18) & (h < 22)), ('night', (h >= 22) | (h < 6))]:
            s_acts = acts[mask]
            row[f'act_{seg}_active'] = ((s_acts == 7) | (s_acts == 8)).mean() if len(s_acts) > 0 else np.nan
            row[f'act_{seg}_still'] = (s_acts == 0).mean() if len(s_acts) > 0 else np.nan
        pre = acts[(h >= 22) & (h < 24)]
        row['act_presleep_active'] = ((pre == 7) | (pre == 8)).mean() if len(pre) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_pedo(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        row['pedo_total_steps'] = grp['step'].sum()
        row['pedo_total_distance'] = grp['distance'].sum()
        row['pedo_total_calories'] = grp['burned_calories'].sum()
        row['pedo_max_speed'] = grp['speed'].max()
        row['pedo_mean_speed'] = grp['speed'].mean()
        row['pedo_running_steps'] = grp['running_step'].sum()
        row['pedo_walking_steps'] = grp['walking_step'].sum()
        row['pedo_run_ratio'] = grp['running_step'].sum() / (grp['step'].sum() + 1)
        eve = grp[grp['timestamp'].dt.hour.between(18, 21)]
        row['pedo_evening_steps'] = eve['step'].sum()
        row['pedo_step_freq_mean'] = grp['step_frequency'].mean()
        row['pedo_step_freq_max'] = grp['step_frequency'].max()
        hourly = grp.groupby(grp['timestamp'].dt.hour)['step'].sum()
        row['pedo_active_hours'] = (hourly > 50).sum()
        feats.append(row)
    return pd.DataFrame(feats)


def extract_hr(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        all_v, seg_v = [], {'morn': [], 'aftn': [], 'eve': [], 'night': []}
        for ts, v in zip(grp['timestamp'], grp['heart_rate']):
            arr = parse_hr(v)
            all_v.extend(arr.tolist())
            h = ts.hour
            if 6 <= h < 12:  # 아침
                seg_v['morn'].extend(arr.tolist())
            elif 12 <= h < 18:  # 오후
                seg_v['aftn'].extend(arr.tolist())
            elif 18 <= h < 22:  # 저녁
                seg_v['eve'].extend(arr.tolist())
            else:  # 밤(새벽)
                seg_v['night'].extend(arr.tolist())
        hr = np.array(all_v)
        for k, v in agg_stats(hr, 'hr').items():
            row[k] = v
        row['hr_resting'] = float(np.nanpercentile(hr, 10)) if len(hr) > 0 else np.nan  # 안정시 심박수 추정
        row['hr_active'] = float(np.nanpercentile(hr, 90)) if len(hr) > 0 else np.nan  # 활동 중 심박수 추정
        row['hr_rmssd'] = float(np.sqrt(np.nanmean(np.diff(hr) ** 2))) if len(hr) > 1 else np.nan
        row['hr_n_records'] = len(hr)
        for seg, vals in seg_v.items():
            a = np.array(vals)
            row[f'hr_{seg}_mean'] = float(np.nanmean(a)) if len(a) > 0 else np.nan
            row[f'hr_{seg}_std'] = float(np.nanstd(a)) if len(a) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_screen(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        sc = grp['m_screen_use'].values
        h = grp['timestamp'].dt.hour.values
        row['screen_on_total'] = (sc > 0).sum()
        row['screen_on_ratio'] = (sc > 0).mean()
        row['screen_unlock_cnt'] = ((sc[1:] > sc[:-1])).sum()
        for seg, mask in [('night', (h >= 22) | (h < 2)), ('eve', (h >= 20) & (h <= 23)),
                          ('presleep', (h >= 22) & (h < 24))]:
            s_sc = sc[mask]
            row[f'screen_{seg}_on'] = (s_sc > 0).sum()
            row[f'screen_{seg}_ratio'] = (s_sc > 0).mean() if len(s_sc) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_light(df_raw, col, prefix, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        vals = grp[col].dropna().values
        for k, v in agg_stats(vals, f'{prefix}_all').items():
            row[k] = v
        h = grp['timestamp'].dt.hour
        for seg, (lo, hi) in [('eve', (18, 22)), ('morn', (6, 10)), ('night', (22, 24))]:
            sv = grp.loc[h.between(lo, hi - 1), col].dropna().values
            row[f'{prefix}_{seg}_mean'] = safe_mean(sv)
        row[f'{prefix}_dark_ratio'] = (vals < 10).mean() if len(vals) > 0 else np.nan
        row[f'{prefix}_bright_ratio'] = (vals > 1000).mean() if len(vals) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_ac(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        ch = grp['m_charging'].values
        h = grp['timestamp'].dt.hour.values
        row['ac_charging_ratio'] = ch.mean()
        for seg, mask in [('eve', (h >= 21) & (h <= 23)), ('night', (h >= 22) | (h < 4)),
                          ('presleep', (h >= 22) & (h < 24))]:
            sc = ch[mask]
            row[f'ac_{seg}_charging'] = sc.mean() if len(sc) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_gps(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        speeds, lats, lons = [], [], []
        for v in grp['m_gps']:
            if isinstance(v, list):
                for pt in v:
                    if isinstance(pt, dict):
                        speeds.append(pt.get('speed', 0))
                        lats.append(pt.get('latitude', 0))
                        lons.append(pt.get('longitude', 0))
        speeds = np.array(speeds)
        row['gps_mean_speed'] = np.nanmean(speeds) if len(speeds) > 0 else np.nan
        row['gps_max_speed'] = np.nanmax(speeds) if len(speeds) > 0 else np.nan
        row['gps_moving_ratio'] = (speeds > 0.5).mean() if len(speeds) > 0 else np.nan
        row['gps_lat_std'] = np.nanstd(lats) if len(lats) > 0 else np.nan
        row['gps_lon_std'] = np.nanstd(lons) if len(lons) > 0 else np.nan
        if len(lats) > 1:
            dlat = np.diff(lats);
            dlon = np.diff(lons)
            row['gps_total_disp'] = float(np.sum(np.sqrt(dlat ** 2 + dlon ** 2)))
        else:
            row['gps_total_disp'] = 0.0
        feats.append(row)
    return pd.DataFrame(feats)


def extract_usage(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        total_time, late_time, eve_time, n_apps = 0, 0, 0, 0
        for ts, v in zip(grp['timestamp'], grp['m_usage_stats']):
            if isinstance(v, list):
                for app in v:
                    if isinstance(app, dict):
                        t = app.get('total_time', 0) or 0
                        total_time += t;
                        n_apps += 1
                        if ts.hour >= 22 or ts.hour < 2:
                            late_time += t
                        if ts.hour >= 18:
                            eve_time += t
        row['usage_total_time'] = total_time
        row['usage_n_apps'] = n_apps
        row['usage_late_time'] = late_time
        row['usage_late_ratio'] = late_time / (total_time + 1)
        row['usage_eve_time'] = eve_time
        row['usage_eve_ratio'] = eve_time / (total_time + 1)
        feats.append(row)
    return pd.DataFrame(feats)


def extract_wifi(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        all_bssids, rssi_vals = set(), []
        for v in grp['m_wifi']:
            if isinstance(v, list):
                for net in v:
                    if isinstance(net, dict):
                        all_bssids.add(net.get('bssid', ''))
                        rssi_vals.append(net.get('rssi', -100))
        row['wifi_n_unique'] = len(all_bssids)
        row['wifi_mean_rssi'] = np.mean(rssi_vals) if rssi_vals else np.nan
        row['wifi_max_rssi'] = np.max(rssi_vals) if rssi_vals else np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_ble(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        addrs = set()
        for v in grp['m_ble']:
            if isinstance(v, list):
                for dev in v:
                    if isinstance(dev, dict):
                        addrs.add(dev.get('address', ''))
        row['ble_n_unique'] = len(addrs)
        row['ble_n_scans'] = len(grp)
        feats.append(row)
    return pd.DataFrame(feats)


def extract_wlight(df_raw, keys):
    return extract_light(df_raw, 'w_light', 'wlight', keys)


def extract_ambience(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        music_s, speech_s, silence_s = [], [], []
        for v in grp['m_ambience']:
            if isinstance(v, list):
                d_map = {item[0]: item[1] for item in v if isinstance(item, list) and len(item) == 2}
                music_s.append(d_map.get('Music', 0))
                speech_s.append(d_map.get('Speech', 0))
                silence_s.append(d_map.get('Silence', 0))
        row['amb_music_mean'] = np.mean(music_s) if music_s else np.nan
        row['amb_speech_mean'] = np.mean(speech_s) if speech_s else np.nan
        row['amb_silence_mean'] = np.mean(silence_s) if silence_s else np.nan
        row['amb_n_records'] = len(grp)
        feats.append(row)
    return pd.DataFrame(feats)


# 수면 구간(전날 밤~오전) 피처 추출

def extract_sleep_hr(df_raw, sleep_keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    df_m = df_raw[df_raw['timestamp'].dt.hour < 9].copy()
    feats = []
    for (sid, d), grp in df_m.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        hour_vals = {h: [] for h in range(9)}
        all_v = []
        for ts, v in zip(grp['timestamp'], grp['heart_rate']):
            arr = parse_hr(v)
            all_v.extend(arr.tolist())
            hour_vals[ts.hour].extend(arr.tolist())
        sleep_hrs = np.array(all_v)
        for k, v in agg_stats(sleep_hrs, 'slp_hr').items():
            row[k] = v
        row['slp_hr_deep_ratio'] = (sleep_hrs < 55).mean() if len(sleep_hrs) > 0 else np.nan
        row['slp_hr_awake_ratio'] = (sleep_hrs > 75).mean() if len(sleep_hrs) > 0 else np.nan
        row['slp_hr_light_ratio'] = ((sleep_hrs >= 55) & (sleep_hrs <= 75)).mean() if len(sleep_hrs) > 0 else np.nan
        if len(sleep_hrs) > 1:
            diffs = np.diff(sleep_hrs)
            row['slp_hr_rmssd'] = float(np.sqrt(np.nanmean(diffs ** 2)))
        else:
            row['slp_hr_rmssd'] = np.nan
        row['slp_hr_n_records'] = len(grp)
        row['slp_hr_early_mean'] = safe_mean(sum([hour_vals[h] for h in range(3)], []))
        row['slp_hr_late_mean'] = safe_mean(sum([hour_vals[h] for h in range(6, 9)], []))
        row['slp_hr_mid_mean'] = safe_mean(sum([hour_vals[h] for h in range(3, 6)], []))
        row['slp_hr_range'] = float(np.ptp(sleep_hrs)) if len(sleep_hrs) > 0 else np.nan
        row['slp_hr_median'] = float(np.median(sleep_hrs)) if len(sleep_hrs) > 0 else np.nan
        if len(sleep_hrs) > 5:
            rolling = pd.Series(sleep_hrs).rolling(5, min_periods=1).mean().values
            row['slp_hr_spike_count'] = int((np.abs(sleep_hrs - rolling) > 15).sum())
        else:
            row['slp_hr_spike_count'] = np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_sleep_pedo(df_raw, sleep_keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        morn = grp[grp['timestamp'].dt.hour < 9]
        row['slp_pedo_steps'] = morn['step'].sum()
        row['slp_pedo_active'] = (morn['step'] > 5).sum()
        row['slp_pedo_calories'] = morn['burned_calories'].sum()
        row['slp_pedo_n_records'] = len(morn)
        mid = grp[grp['timestamp'].dt.hour.between(2, 4)]
        row['slp_pedo_mid_steps'] = mid['step'].sum()
        feats.append(row)
    return pd.DataFrame(feats)


def extract_sleep_activity(df_raw, sleep_keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        morn = grp[grp['timestamp'].dt.hour < 9]
        if len(morn) == 0:
            row.update({'slp_act_still_ratio': np.nan, 'slp_act_active_ratio': np.nan, 'slp_act_n_records': 0})
        else:
            acts = morn['m_activity'].values
            row['slp_act_still_ratio'] = (acts == 0).mean()
            row['slp_act_active_ratio'] = ((acts == 7) | (acts == 8)).mean()
            row['slp_act_n_records'] = len(acts)
        feats.append(row)
    return pd.DataFrame(feats)


def extract_sleep_screen(df_raw, sleep_keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        morn = grp[grp['timestamp'].dt.hour < 9]
        if len(morn) > 0:
            sc = morn['m_screen_use'].values
            row['slp_screen_on'] = (sc > 0).sum()
            row['slp_screen_ratio'] = (sc > 0).mean()
        else:
            row['slp_screen_on'] = row['slp_screen_ratio'] = np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_sleep_light(df_raw, sleep_keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        morn = grp[grp['timestamp'].dt.hour < 9]
        if len(morn) > 0:
            vals = morn['w_light'].dropna().values
            row['slp_wlight_mean'] = safe_mean(vals)
            row['slp_wlight_dark'] = (vals < 5).mean() if len(vals) > 0 else np.nan
            row['slp_wlight_light'] = (vals > 100).mean() if len(vals) > 0 else np.nan
        else:
            row['slp_wlight_mean'] = row['slp_wlight_dark'] = row['slp_wlight_light'] = np.nan
        feats.append(row)
    return pd.DataFrame(feats)


# 피처 추출 및 데이터셋 준비

print('Extracting features...')

# 낮 시간대
feat_dfs = []
for name, fn, col, prefix in [
    ('mActivity', extract_activity, None, None),
    ('wPedo', extract_pedo, None, None),
    ('wHr', extract_hr, None, None),
    ('mScreenStatus', extract_screen, None, None),
    ('mLight', extract_light, 'm_light', 'mlight'),
    ('wLight', extract_wlight, None, None),
    ('mACStatus', extract_ac, None, None),
    ('mGps', extract_gps, None, None),
    ('mUsageStats', extract_usage, None, None),
    ('mWifi', extract_wifi, None, None),
    ('mBle', extract_ble, None, None),
    ('mAmbience', extract_ambience, None, None),
]:
    print(f'  {name}...')
    df = load_parquet(name)
    feat_dfs.append(fn(df, col, prefix, all_keys) if col else fn(df, all_keys))

# 수면 시간대
sleep_feat_dfs = []
for name, fn in [
    ('wHr', extract_sleep_hr),
    ('wPedo', extract_sleep_pedo),
    ('mActivity', extract_sleep_activity),
    ('mScreenStatus', extract_sleep_screen),
    ('wLight', extract_sleep_light),
]:
    print(f'  sleep_morning: {name}...')
    df = load_parquet(name)
    sleep_feat_dfs.append(fn(df, sleep_keys))

sleep_feats = sleep_feat_dfs[0]
for df in sleep_feat_dfs[1:]:
    sleep_feats = sleep_feats.merge(df, on=['subject_id', 'sleep_date'], how='outer')

feat_all = feat_dfs[0]
for df in feat_dfs[1:]:
    feat_all = feat_all.merge(df, on=['subject_id', 'lifelog_date'], how='outer')

# Time features
feat_all['dow'] = feat_all['lifelog_date'].dt.dayofweek
feat_all['month'] = feat_all['lifelog_date'].dt.month
feat_all['week'] = feat_all['lifelog_date'].dt.isocalendar().week.astype(int)
feat_all['is_weekend'] = (feat_all['dow'] >= 5).astype(int)
feat_all['subject_num'] = feat_all['subject_id'].str.extract(r'(\d+)').astype(int)
feat_all = feat_all.sort_values(['subject_id', 'lifelog_date']).reset_index(drop=True)
import holidays
kr_holidays = holidays.KR(years=2024)

feat_all['is_holiday'] = feat_all['lifelog_date'].apply(
    lambda x: 1 if x in kr_holidays else 0
)
feat_all['is_holiday_or_weekend'] = (
    (feat_all['dow'] >= 5) | (feat_all['is_holiday'] == 1)
).astype(int)

# 다음날 주말/공휴일 여부
feat_all['next_day'] = feat_all['lifelog_date'] + pd.Timedelta(days=1)
feat_all['is_next_holiday'] = feat_all['next_day'].apply(
    lambda x: 1 if x in kr_holidays else 0
)
feat_all['is_next_weekend'] = (feat_all['next_day'].dt.dayofweek >= 5).astype(int)
feat_all['is_before_freeday'] = (
    (feat_all['is_next_weekend'] == 1) | (feat_all['is_next_holiday'] == 1)
).astype(int)

# 임시 컬럼 제거
feat_all.drop(columns=['next_day'], inplace=True)
# Lag / rolling
roll_cols = [
    'pedo_total_steps', 'pedo_total_calories', 'pedo_total_distance',
    'screen_on_ratio', 'screen_night_on', 'screen_eve_ratio',
    'act_active_ratio', 'act_still_ratio',
    'mlight_all_mean', 'wlight_all_mean',
    'gps_moving_ratio', 'usage_late_ratio', 'usage_eve_ratio',
    'ac_presleep_charging',
    'hr_mean', 'hr_resting', 'hr_active', 'hr_rmssd',
    'hr_eve_mean', 'hr_night_mean',
]
for col in roll_cols:
    if col not in feat_all.columns:
        continue
    g = feat_all.groupby('subject_id')[col]
    feat_all[f'{col}_lag1'] = g.shift(1)
    feat_all[f'{col}_lag2'] = g.shift(2)
    feat_all[f'{col}_roll3'] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    feat_all[f'{col}_roll7'] = g.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    feat_all[f'{col}_roll14'] = g.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())

# 개인별 z-score (train 통계만 사용)
train_date_set = set(zip(train_df['subject_id'], train_df['lifelog_date']))
is_train = pd.Series(
    [(sid, d) in train_date_set for sid, d in zip(feat_all['subject_id'], feat_all['lifelog_date'])],
    index=feat_all.index
)
numeric_cols = feat_all.select_dtypes(include=[np.number]).columns.tolist()
exclude_from_norm = {
    'subject_num', 'dow', 'month', 'week', 'is_weekend',
    'is_holiday', 'is_holiday_or_weekend',
    'is_next_holiday', 'is_next_weekend', 'is_before_freeday'
}
norm_cols = [c for c in numeric_cols
             if c not in exclude_from_norm and 'lag' not in c and 'roll' not in c]
for col in norm_cols:
    stats = feat_all.loc[is_train].groupby('subject_id')[col].agg(mu='mean', sig='std')
    stats['sig'] = stats['sig'].replace(0, np.nan)
    mu = feat_all['subject_id'].map(stats['mu'])
    sig = feat_all['subject_id'].map(stats['sig'])
    feat_all[f'{col}_subj_z'] = (feat_all[col] - mu) / sig

# 데이터셋 병합
train_full = train_df.merge(feat_all, on=['subject_id', 'lifelog_date'], how='left')
train_full = train_full.merge(sleep_feats, on=['subject_id', 'sleep_date'], how='left')
test_full = test_df[['subject_id', 'lifelog_date', 'sleep_date']].merge(
    feat_all, on=['subject_id', 'lifelog_date'], how='left')
test_full = test_full.merge(sleep_feats, on=['subject_id', 'sleep_date'], how='left')

# Target encoding
all_with_labels = pd.concat([
    train_full[['subject_id', 'lifelog_date'] + TARGETS],
    test_full[['subject_id', 'lifelog_date']].assign(**{t: np.nan for t in TARGETS})
], ignore_index=True).sort_values(['subject_id', 'lifelog_date'])

enc_cols = []
for t in TARGETS:
    for w in [3, 7, 14, 21]:
        col = f'{t}_enc{w}'
        all_with_labels[col] = all_with_labels.groupby('subject_id')[t].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        enc_cols.append(col)
    col_lag = f'{t}_lag1'
    all_with_labels[col_lag] = all_with_labels.groupby('subject_id')[t].shift(1)
    enc_cols.append(col_lag)

enc_df = all_with_labels[['subject_id', 'lifelog_date'] + enc_cols]
train_full = train_full.merge(enc_df, on=['subject_id', 'lifelog_date'], how='left')
test_full = test_full.merge(enc_df, on=['subject_id', 'lifelog_date'], how='left')
feature_cols = [c for c in train_full.columns
                if c not in ['subject_id', 'lifelog_date', 'sleep_date'] + TARGETS]

X_train = train_full[feature_cols].copy()
X_test = test_full[feature_cols].copy()


# 결측치 0으로 채움
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# 로그오즈 변환
def safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# log-loss를 최소화하는 블렌딩 가중치 탐색
def objective_func(weights, preds, y_true):
    weights = np.clip(weights,  1e-5, 1)
    weights /= weights.sum()
    blend = sum(w * p for w, p in zip(weights, preds))
    blend = np.clip(blend, 1e-6, 1 - 1e-6)
    return log_loss(y_true, blend)


# 시드 / 폴드 수
SEEDS = [1, 42, 2024, 8765, 9999]
N_FOLDS = 5

# 타겟별 예측값 저장
test_preds = np.zeros((len(X_test), len(TARGETS)))
oof_preds = np.zeros((len(X_train), len(TARGETS)))

print(f"\n===== CLEAN TARGET-SPECIFIC ENSEMBLE =====")

TARGET_CONFIG = {
    "Q1": {
        "top_k": 60,
        "lgb": dict(
            n_estimators=1200, learning_rate=0.012, max_depth=3, num_leaves=7,
            min_child_samples=30, lambda_l1=5.0, lambda_l2=8.0
        ),
        "cat": dict(
            iterations=1200, learning_rate=0.012, depth=3, l2_leaf_reg=9.0
        ),
        "xgb": dict(
            n_estimators=1200, learning_rate=0.012, max_depth=3, min_child_weight=7,
            reg_alpha=5.0, reg_lambda=8.0
        ),
        "calib_C": 5.0
    },
    "S1": {
        "top_k": 110,
        "lgb": dict(
            n_estimators=1200, learning_rate=0.015, max_depth=4, num_leaves=15,
            min_child_samples=20, lambda_l1=3.0, lambda_l2=5.0
        ),
        "cat": dict(
            iterations=1200, learning_rate=0.015, depth=4, l2_leaf_reg=6.0
        ),
        "xgb": dict(
            n_estimators=1200, learning_rate=0.015, max_depth=4, min_child_weight=5,
            reg_alpha=3.0, reg_lambda=5.0
        ),
        "calib_C": 1e6
    },
    "S2": {
        "top_k": 100,
        "lgb": dict(
            n_estimators=1200, learning_rate=0.015, max_depth=4, num_leaves=15,
            min_child_samples=20, lambda_l1=2.0, lambda_l2=4.0
        ),
        "cat": dict(
            iterations=1200, learning_rate=0.015, depth=4, l2_leaf_reg=5.0
        ),
        "xgb": dict(
            n_estimators=1200, learning_rate=0.015, max_depth=4, min_child_weight=4,
            reg_alpha=2.0, reg_lambda=4.0
        ),
        "calib_C": 1e6
    },
    "S4": {
        "top_k": 100,
        "lgb": dict(
            n_estimators=1200, learning_rate=0.015, max_depth=4, num_leaves=15,
            min_child_samples=20, lambda_l1=2.0, lambda_l2=4.0
        ),
        "cat": dict(
            iterations=1200, learning_rate=0.015, depth=4, l2_leaf_reg=5.0
        ),
        "xgb": dict(
            n_estimators=1200, learning_rate=0.015, max_depth=4, min_child_weight=4,
            reg_alpha=2.0, reg_lambda=4.0
        ),
        "calib_C": 1e6
    },
    "S3": {
        "top_k": 120,
        "lgb": dict(
            n_estimators=1200, learning_rate=0.018, max_depth=4, num_leaves=15,
            min_child_samples=20, lambda_l1=2.0, lambda_l2=4.0
        ),
        "cat": dict(
            iterations=1200, learning_rate=0.018, depth=4, l2_leaf_reg=5.0
        ),
        "xgb": dict(
            n_estimators=1200, learning_rate=0.018, max_depth=4, min_child_weight=4,
            reg_alpha=2.0, reg_lambda=4.0
        ),
        "calib_C": 1e6
    }
}

DEFAULT_Q = {
    "top_k": 40,
    "lgb": dict(
        n_estimators=1000, learning_rate=0.015, max_depth=4, num_leaves=15,
        min_child_samples=25, lambda_l1=3.0, lambda_l2=5.0
    ),
    "cat": dict(
        iterations=1000, learning_rate=0.015, depth=4, l2_leaf_reg=7.0
    ),
    "xgb": dict(
        n_estimators=1000, learning_rate=0.015, max_depth=4, min_child_weight=5,
        reg_alpha=3.0, reg_lambda=5.0
    ),
    "calib_C": 1e6
}

DEFAULT_S = {
    "top_k": 100,
    "lgb": dict(
        n_estimators=1000, learning_rate=0.02, max_depth=5, num_leaves=31,
        min_child_samples=15, lambda_l1=1.0, lambda_l2=2.0
    ),
    "cat": dict(
        iterations=1000, learning_rate=0.02, depth=5, l2_leaf_reg=3.0
    ),
    "xgb": dict(
        n_estimators=1000, learning_rate=0.02, max_depth=5, min_child_weight=3,
        reg_alpha=1.0, reg_lambda=2.0
    ),
    "calib_C": 1e6
}

for ti, target in enumerate(TARGETS):

    y = train_full[target].values

    print(f"\n===== {target} =====")

    if target in TARGET_CONFIG:
        cfg = TARGET_CONFIG[target]
    elif target.startswith("Q"):
        cfg = DEFAULT_Q
    else:
        cfg = DEFAULT_S

    top_k = cfg["top_k"]

    print(f"Feature Selection Top-{top_k}")

    selector = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    selector.fit(X_train, y)

    imp = pd.Series(
        selector.feature_importances_,
        index=X_train.columns
    )

    CAUSAL_FEATURES = {
        "Q1": ["slp_screen_ratio", "slp_screen_on"],
        "S1": ["slp_screen_ratio", "slp_screen_on", "pedo_total_steps"],
        "S2": ["act_still_ratio_lag1", "pedo_total_steps"],
        "S3": ["screen_late_ratio", "screen_presleep_ratio", "screen_late_on", "ac_night_charging"],
        "S4": ["act_still_ratio_lag1", "pedo_total_steps"]
    }

    causal_must = [f for f in CAUSAL_FEATURES.get(target, []) if f in X_train.columns]
    remaining_k = top_k - len(causal_must)
    imp_filtered = imp.drop(index=[f for f in causal_must if f in imp.index], errors='ignore')
    selected_features = causal_must + imp_filtered.nlargest(remaining_k).index.tolist()

    X_train_target = X_train[selected_features]
    X_test_target = X_test[selected_features]

    oof_lgb_all = np.zeros(len(X_train_target))
    oof_cat_all = np.zeros(len(X_train_target))
    oof_xgb_all = np.zeros(len(X_train_target))

    test_lgb_all = np.zeros(len(X_test_target))
    test_cat_all = np.zeros(len(X_test_target))
    test_xgb_all = np.zeros(len(X_test_target))

    for seed in SEEDS:

        skf = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=seed
        )

        oof_lgb = np.zeros(len(X_train_target))
        oof_cat = np.zeros(len(X_train_target))
        oof_xgb = np.zeros(len(X_train_target))

        test_lgb = np.zeros(len(X_test_target))
        test_cat = np.zeros(len(X_test_target))
        test_xgb = np.zeros(len(X_test_target))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_target, y)):
            X_tr = X_train_target.iloc[tr_idx]
            y_tr = y[tr_idx]

            X_val = X_train_target.iloc[val_idx]
            y_val = y[val_idx]

            pos_count = y_tr.sum()
            neg_count = len(y_tr) - pos_count

            if target == "S3":
                dynamic_spw = np.clip(
                    neg_count / pos_count if pos_count > 0 else 1.0,
                    0.5, 3.0
                )
            else:
                dynamic_spw = np.clip(
                    neg_count / pos_count if pos_count > 0 else 1.0,
                    1.0, 3.0
                )

            lgb_m = lgb.LGBMClassifier(
                **cfg["lgb"],
                scale_pos_weight=dynamic_spw,
                random_state=seed,
                verbosity=-1
            )

            cat_m = CatBoostClassifier(
                **cfg["cat"],
                random_seed=seed,
                verbose=0,
                early_stopping_rounds=50,
                allow_writing_files=False
            )

            xgb_m = XGBClassifier(
                **cfg["xgb"],
                scale_pos_weight=dynamic_spw,
                random_state=seed,
                eval_metric='logloss',
                early_stopping_rounds=50
            )

            lgb_m.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )

            cat_m.fit(
                X_tr,
                y_tr,
                eval_set=(X_val, y_val)
            )

            xgb_m.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            oof_lgb[val_idx] = lgb_m.predict_proba(X_val)[:, 1]
            oof_cat[val_idx] = cat_m.predict_proba(X_val)[:, 1]
            oof_xgb[val_idx] = xgb_m.predict_proba(X_val)[:, 1]

            test_lgb += lgb_m.predict_proba(X_test_target)[:, 1] / N_FOLDS
            test_cat += cat_m.predict_proba(X_test_target)[:, 1] / N_FOLDS
            test_xgb += xgb_m.predict_proba(X_test_target)[:, 1] / N_FOLDS

        oof_lgb_all += oof_lgb / len(SEEDS)
        oof_cat_all += oof_cat / len(SEEDS)
        oof_xgb_all += oof_xgb / len(SEEDS)

        test_lgb_all += test_lgb / len(SEEDS)
        test_cat_all += test_cat / len(SEEDS)
        test_xgb_all += test_xgb / len(SEEDS)

    preds_list = [
        oof_lgb_all,
        oof_cat_all,
        oof_xgb_all
    ]

    res = minimize(
        objective_func,
        [1 / 3, 1 / 3, 1 / 3],
        args=(preds_list, y),
        method='L-BFGS-B',
        bounds=[(1e-5, 1), (1e-5, 1), (1e-5, 1)]
    )
    best_weights = np.clip(res.x, 1e-5, 1)
    best_weights /= best_weights.sum()

    print("Weights:", best_weights)

    meta_oof = (
            best_weights[0] * oof_lgb_all +
            best_weights[1] * oof_cat_all +
            best_weights[2] * oof_xgb_all
    )
    oof_preds[:, ti] = np.clip(meta_oof, 1e-6, 1 - 1e-6)

    meta_test = (
            best_weights[0] * test_lgb_all +
            best_weights[1] * test_cat_all +
            best_weights[2] * test_xgb_all
    )

    meta_loss = log_loss(y, np.clip(meta_oof, 1e-6, 1 - 1e-6))

    print("Blend Loss:", meta_loss)

    calibrator = LogisticRegression(
        C=cfg["calib_C"],
        max_iter=1000,
        solver="lbfgs"
    )

    calib_X = safe_logit(meta_oof).reshape(-1, 1)
    calib_test_X = safe_logit(meta_test).reshape(-1, 1)

    calibrator.fit(calib_X, y)

    calibrated_oof = calibrator.predict_proba(calib_X)[:, 1]
    calibrated_test = calibrator.predict_proba(calib_test_X)[:, 1]

    calib_loss = log_loss(y, calibrated_oof)

    print("Calibrated Loss:", calib_loss)
    threshold = 0.5
    pred_binary = (calibrated_oof > threshold).astype(int)
    f1 = f1_score(y, pred_binary)
    print(f"F1 Score (threshold=0.5): {f1:.4f}")
    print(classification_report(y, pred_binary))

    if calib_loss < meta_loss:
        final_test = calibrated_test
        print("Use calibrated")
    else:
        final_test = meta_test
        print("Use raw blend")

    clip_low = max(0.01, y.mean() * 0.10)
    clip_high = min(0.99, 1 - (1 - y.mean()) * 0.10)

    test_preds[:, ti] = np.clip(
        final_test,
        clip_low,
        clip_high
    )
    print(f"   -> Dynamic clipping: [{clip_low:.4f}, {clip_high:.4f}]")

# 최종 제출 파일 생성
oof_df = train_full[["subject_id", "sleep_date", "lifelog_date"]].copy()
for i, t in enumerate(TARGETS):
    oof_df[t] = oof_preds[:, i]
oof_path = OUTPUT_DIR / 'oof_lgbcatxgb.csv'
oof_df.to_csv(oof_path, index=False)
print(f"OOF 저장됨: {oof_path}")
submission = test_df[["subject_id", "sleep_date", "lifelog_date"]].copy()
for i, t in enumerate(TARGETS):
    submission[t] = test_preds[:, i]

output_path = OUTPUT_DIR / 'submission.csv'
submission.to_csv(output_path, index=False)
print(f"\n 제출 파일 생성됨: {output_path}")
