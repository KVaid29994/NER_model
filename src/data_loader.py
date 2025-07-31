import pandas as pd
from collections import defaultdict
from logger_config import get_logger

logger = get_logger(__name__)

def load_data(path):
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, encoding='ISO-8859-1')
    sentences = defaultdict(list)
    for _, row in df.iterrows():
        sent_id = row["Sentence #"]
        word = row["Word"]
        tag = row["Tag"]
        sentences[sent_id].append((word, tag))
    logger.info("Data loaded successfully")
    return list(sentences.values())