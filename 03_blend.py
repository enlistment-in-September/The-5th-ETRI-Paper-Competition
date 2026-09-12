import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss

from config import DATA_DIR as ITEMS_DIR, OUTPUT_DIR

BASE_DIR = OUTPUT_DIR  # STEP 2 산출물도 여기 있음
TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']

# 라벨 로드
train_df = pd.read_csv(ITEMS_DIR / 'ch2026_metrics_train.csv')
train_df['lifelog_date'] = pd.to_datetime(train_df['lifelog_date']).dt.date.astype(str)

# OOF 로드
oof_lgb = pd.read_csv(OUTPUT_DIR / 'oof_lgbcatxgb.csv')
oof_lstm = pd.read_csv(BASE_DIR / 'oof_5fold_bilstm_multitask_safe.csv')

oof_lgb['lifelog_date'] = pd.to_datetime(oof_lgb['lifelog_date']).dt.date.astype(str)
oof_lstm['lifelog_date'] = pd.to_datetime(oof_lstm['lifelog_date']).dt.date.astype(str)

# oof_lgb 기준으로 행 순서 정렬
oof_lstm = oof_lgb[['subject_id', 'lifelog_date']].merge(
    oof_lstm, on=['subject_id', 'lifelog_date'], how='left'
)

# 라벨도 같은 순서로 정렬
labels = oof_lgb[['subject_id', 'lifelog_date']].merge(
    train_df[['subject_id', 'lifelog_date'] + TARGETS],
    on=['subject_id', 'lifelog_date'], how='left'
)

def objective(weights, p1, p2, y):
    w = np.clip(weights, 0, 1)
    w = w / w.sum()
    blend = w[0] * p1 + w[1] * p2
    blend = np.clip(blend, 1e-6, 1 - 1e-6)
    return log_loss(y, blend)

final_weights = {}
for t in TARGETS:
    y = labels[t].values.astype(int)
    p_lgb = oof_lgb[t].values
    p_lstm_col = f'{t}_pred'
    p_lstm = oof_lstm[p_lstm_col].values if p_lstm_col in oof_lstm.columns else oof_lstm[t].values

    # 개별 모델 loss (출력용)
    lgb_loss = log_loss(y, np.clip(p_lgb, 1e-6, 1-1e-6))
    lstm_loss = log_loss(y, np.clip(p_lstm, 1e-6, 1-1e-6))

    # 타겟별 블렌딩 가중치 결정
    if t == "S3":
        # S3는 수동으로 가중치 고정
        w = np.array([0.5, 0.5])
    else:
        # Nelder-Mead로 탐색
        res = minimize(objective, [0.8, 0.2], args=(p_lgb, p_lstm, y), method='Nelder-Mead')
        w = np.clip(res.x, 0, 1)
        w = w / w.sum()

    final_weights[t] = w
    blend_loss = log_loss(y, np.clip(w[0]*p_lgb + w[1]*p_lstm, 1e-6, 1-1e-6))

    print(f"{t}: LGB={w[0]:.3f}(loss={lgb_loss:.4f}), LSTM={w[1]:.3f}(loss={lstm_loss:.4f}), Blend={blend_loss:.5f}")

# 테스트 예측 로드
sub_lgb = pd.read_csv(OUTPUT_DIR / 'submission.csv')
sub_lstm = pd.read_csv(BASE_DIR / 'submission_5fold_bilstm_multitask_safe.csv')

sub_blend = sub_lgb[['subject_id', 'sleep_date', 'lifelog_date']].copy()
for t in TARGETS:
    w = final_weights[t]
    sub_blend[t] = np.clip(
        w[0] * sub_lgb[t].values + w[1] * sub_lstm[t].values,
        0.01, 0.99
    )

sub_blend.to_csv(OUTPUT_DIR / 'submission_blend_lstm.csv', index=False)
print("\n저장됨!")
