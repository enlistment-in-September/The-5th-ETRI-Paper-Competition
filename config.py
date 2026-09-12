"""
공통 경로 설정.

모든 스크립트는 이 파일에서 입출력 경로를 가져온다.
기본값은 리포지토리 내부 경로이며, 환경변수로 덮어쓸 수 있다.

ETRI_DATA_DIR : 원본 데이터(parquet/csv) 디렉터리 (기본: <repo>/data/raw)
ETRI_OUTPUT_DIR :  OOF/제출 파일 출력 디렉터리 (기본: <repo>/outputs)
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("ETRI_DATA_DIR", ROOT / "data" / "raw"))
OUTPUT_DIR = Path(os.environ.get("ETRI_OUTPUT_DIR", ROOT / "outputs"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
