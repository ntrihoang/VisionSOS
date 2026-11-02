"""
SilentSOS
Shared configuration file for the entire ML pipeline
File cau hinh dung chung cho toan bo pipeline ML
"""

"""
Scope:
where is data stored?
where will the trained models be saved?
class are there?
how should train and test sets be split?
What hyperparameters should be used for random forest?
What error threshold should abort?
"""

from pathlib import Path

# direction
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

#nguon data du nhat
FEATURES_CSV = DATA_DIR / "landmarks_1.csv"

# Models / metadata
MODEL_PATH = MODELS_DIR / "sos_rf.joblib"
MODEL_META_PATH = MODELS_DIR / "sos_rf_meta.json"

CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"

# LABELS
LABEL = [
    "OTHER",
    "OPEN_PALM",
    "THUMB_TUCKED",
    "THUMB_ENCLOSED",
]

NON_SOS_LABEL = "OTHER"

# cau truc cot CSV
LABEL_COL = "label"
PERSON_COL = "person"

FEATURE_DIM = 63
FEATURE_COLS = [f"feature_{i}" for i in range(FEATURE_DIM)]

# cot bat buoc
REQUIRED_COL = [LABEL_COL, PERSON_COL] + FEATURE_COLS

# chia tap
SPLIT_MODE = "by_person"
TEST_SIZE = 0.2

TEST_PERSON = ["nam"]

CONFIDENCE_SWEEP_PATH = REPORTS_DIR / "confidence_sweep.txt"

# Nguong VALIDATE # nguong kiem soat chuat luong data

# canh bao khi lop co it nguong
MIN_SAMPLES_PER_CLASS = 300 # moi lop it nhat 30 mau

MIN_SAMPLES_PER_PERSON = 20
# Canh bao khi 1 person co tong mau it hon nguong
MAX_CLASS_IMBALANCE_RATIO = 5.0

# Sanity check chuẩn hóa (chỉ cảnh báo, không loại dòng)
WRIST_NORM_TOL = 1e-3          # ||landmark_0|| phải ≈ 0
SCALE_TOL = 0.05               # ||landmark_9|| phải ∈ [1-tol, 1+tol]
MAX_BAD_NORM_RATIO = 0.15      # nếu tỷ lệ dòng lệch chuẩn hóa > này → dừng hỏi nhóm

#SEED
SEED = 42
# ======================
# CẤU HÌNH RANDOM FOREST
# ======================
RF_PARAMS = {
    "n_estimators": 150,
    "max_depth": 20,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
}

# nguong confidence

CONFIDENCE_SWEEP = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
