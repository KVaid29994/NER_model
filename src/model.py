import joblib
import os
from pathlib import Path
from src.logger_config import get_logger

logger = get_logger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "crf_model.pkl"

def save_crf_model(model):
    joblib.dump(model, MODEL_PATH)
    logger.info(f"CRF model saved at {MODEL_PATH}")

def load_crf_model():
    if MODEL_PATH.exists():
        logger.info(f"Loading CRF model from {MODEL_PATH}")
        return joblib.load(MODEL_PATH)
    else:
        logger.warning("CRF model not found, training a new one.")
        return None
