"""Configuration file for the synthetic inspector project."""

import os 
from pathlib import Path 
from dotenv import load_dotenv

load_dotenv()

# PROJECT ROOT AND DIRECTORIES 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
DEMO_DIR = PROJECT_ROOT / "demo"

# DATA PATHS (GENERATION OUTPUTS, TRAINING INPUT)
DATA_IMAGE_DIR = RAW_DIR / "images"
DATA_ANNOTATION_DIR = RAW_DIR / "annotations"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_SAMPLES_DIR = DATA_DIR / "samples"


# Model Artifact Paths 
MODEL_FP32_PATH = MODEL_DIR / "defect_classifier_fp32.pt"
MODEL_INT8_PATH = MODEL_DIR / "defect_classifier_int8.pt"
MODEL_ONNX_PATH = MODEL_DIR / "defect_classifier.onnx"
TRAINING_REPORT_PATH = MODEL_DIR / "training_report.json"

DEFECT_PROMPTS = {
    "scratch": "A metallic industrial part with a visible linear scratch on the surface, industrial lighting, high detail, photorealistic",
    "dent": "A metallic industrial part with small dark circular pits on the surface, industrial lighting, high detail, photorealistic",
    "porosity": "A metallic industrial part with small dark circular pits on the surface, industrial lighting, high detail, photorealistic",
    "crack": "A metallic industrial part with a thin irregular fracture line, industrial lighting, high detail, photorealistic",
    "misalignment": "A metallic industrial part with visibly offset edges, industrial lighting, high detail, photorealistic",
    "good" : "A perfectly smooth metallic industrial part with no defects, industrial lighting, high detail, photorealistic",
}

DEFECT_CLASSES = sorted(DEFECT_PROMPTS.keys())
NUM_CLASSES = len(DEFECT_CLASSES)
CLASS_TO_IDX = {name: i for i, name in enumerate(DEFECT_CLASSES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}

IMAGES_PER_CLASS = 2000
MAX_IMAGES = 12000
JOB_TIMEOUT_SECONDS = 900
WORKFLOW_YAML = "workflows/testing/cosmos3-text-to-image.yaml"
S3_PREFIX = "cosmos3"


# Train settings 
IMAGE_SIZE = 224
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42
NUM_WORKERS = 2
MAX_MODEL_MB = 50
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8080
CONFIDENCE_THRESHOLD = 0.5


# edge simulator settings 
LATENCY_BUDGET_MS = 100
DEFAULT_VIDEO_FPS =30
DASHBOARD_WINDOW_NAME = "Synthetic Inspector - Edge Simulator"


# Secrets from Environment 
S3_BUCKET = os.environ.get("NPA_S3_BUCKET", "")
ENDPOINT_URL = os.environ.get("ENDPOINT_URL", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

# validate_Secretes function 
def validate_secrets(required: tuple = ()) -> None:
    """
        Raise if any required secret is missing.
        Call this at the start of any script that needs a secret.
    """
    missing = [name for name in required if not globals().get(name)]
    if missing: 
        raise EnvironmentError(
            f"Missing required environment variables: {missing}."
            f"\nCheck your .env file."
        )


    