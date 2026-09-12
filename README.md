# Proximal Causal Inference-Guided Feature Identification for Lifelog-Based Sleep Prediction

제5회 ETRI 휴먼이해 인공지능 논문경진대회 제출 코드.

스마트폰·스마트워치 라이프로그 12종으로 수면 관련 이진 타겟 7개(`Q1`–`Q3`, `S1`–`S4`)의
확률을 예측한다. 평가 지표는 타겟별 binary log-loss의 단순 평균이다.

## 파이프라인

| 단계 | 스크립트 | 내용 | 산출물 |
|---|---|---|---|
| STEP 1 | `01_gbm_ensemble.py` | LightGBM + CatBoost + XGBoost, 타겟별 하이퍼파라미터 · RandomForest Top-K 피처 선택 · 5-Fold × 5-Seed OOF · 로지스틱 캘리브레이션 | `outputs/oof_lgbcatxgb.csv`<br>`outputs/submission.csv` |
| STEP 2 | `02_bilstm.py` | MultiTaskBiLSTM (Attention Pooling + Subject Embedding), lookback 14일 · 5-Fold × 5-Seed OOF | `outputs/oof_5fold_bilstm_multitask_safe.csv`<br>`outputs/submission_5fold_bilstm_multitask_safe.csv` |
| STEP 3 | `03_blend.py` | OOF 기반 타겟별 blending 가중치 탐색(Nelder-Mead) 후 logit 결합 | `outputs/submission_blend_lstm.csv` |

## 실행

```bash
pip install -r requirements.txt

python 01_gbm_ensemble.py     
python 02_bilstm.py          
python 03_blend.py            
```

STEP 3은 STEP 1·2의 OOF 산출물을 입력으로 쓰므로 반드시 순서대로 실행한다.

## 데이터 배치

`data/raw/` 아래에 대회에서 제공한 아래 파일을 사전에 넣어두어야 실행이 가능.

| 파일 | 설명 |
|---|---|
| `ch2026_metrics_train.csv` | 학습 라벨 (450 participant-day) |
| `ch2026_submission_sample.csv` | 제출 양식 (250 participant-day) |
| `ch2025_mActivity.parquet` | 스마트폰 활동 |
| `ch2025_mACStatus.parquet` | 충전 상태 |
| `ch2025_mScreenStatus.parquet` | 화면 사용 |
| `ch2025_mLight.parquet` | 스마트폰 조도 |
| `ch2025_wLight.parquet` | 워치 조도 |
| `ch2025_wPedo.parquet` | 보행 계수 |
| `ch2025_wHr.parquet` | 심박수 |
| `ch2025_mUsageStats.parquet` | 앱 사용 통계 |
| `ch2025_mWifi.parquet` | Wi-Fi 스캔 |
| `ch2025_mBle.parquet` | Bluetooth 스캔 |
| `ch2025_mGps.parquet` | GPS |
| `ch2025_mAmbience.parquet` | 주변 소리 |

(데이터를 다른 위치에 두었다면 환경변수로 지정할 수 있다.)

```bash
export ETRI_DATA_DIR=/path/to/ch2025_data_items
export ETRI_OUTPUT_DIR=/path/to/outputs   # 선택
```

경로는 모두 `config.py` 한 곳에서 관리한다.

## 디렉터리

```
.
├── config.py              # 입출력 경로 설정 
├── 01_gbm_ensemble.py     # STEP 1
├── 02_bilstm.py           # STEP 2
├── 03_blend.py            # STEP 3
├── data/raw/              # 데이터 경로
├── outputs/               # OOF, 제출 파일 경로

```
