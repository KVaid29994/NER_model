from src.data_loader import load_data
from src.feature_extraction import prepare_data
from src.model import save_crf_model
from src.logger_config import get_logger
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logger = get_logger(__name__)

def train_pipeline():
    logger.info("Starting training pipeline")

    # Load and prepare data
    data_path = Path(__file__).parent.parent / "data" / "ner_dataset.csv"
    sentences = load_data(data_path)
    X_all, y_all = prepare_data(sentences)
    logger.info(f"Total sequences: {len(X_all)}")

    # Split: 60% train, 20% val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
    logger.info(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")

    # Train CRF
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    logger.info("Model training completed")

    # Save model
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    save_crf_model(crf)  # handles internal path logging
    logger.info("Model saved successfully")

    # Evaluation
    logger.info("Evaluating on validation set")
    val_report, val_f1, _ = evaluate_model(crf, X_val, y_val, label_list=crf.classes_, tag="val")

    logger.info("Evaluating on test set")
    test_report, test_f1, _ = evaluate_model(crf, X_test, y_test, label_list=crf.classes_, tag="test")

    return crf, test_f1, test_report


def evaluate_model(crf, X, y, label_list=None, tag="eval"):
    logger.info(f"Evaluating CRF model on {tag} set")

    y_pred = crf.predict(X)
    report = metrics.flat_classification_report(y_true=y, y_pred=y_pred, labels=label_list, digits=4)
    logger.info(f"\n{tag.upper()} Classification Report:\n{report}")

    # F1 scores
    scores = metrics.flat_f1_score(y, y_pred, average=None, labels=label_list)
    f1_per_label = dict(zip(label_list, scores))
    logger.info(f"{tag.upper()} F1 per label: {f1_per_label}")

    # Save F1 bar plot
    f1_plot_path = Path(__file__).parent.parent / "logs" / f"f1_scores_{tag}.png"
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(f1_per_label.keys()), y=list(f1_per_label.values()))
    plt.xticks(rotation=45)
    plt.title(f"F1 Score per Label - {tag.upper()} Set")
    plt.ylabel("F1 Score")
    plt.tight_layout()
    plt.savefig(f1_plot_path)
    plt.close()
    logger.info(f"{tag.upper()} F1 score plot saved at {f1_plot_path}")

    # Confusion matrix
    true_flat = [label for sent in y for label in sent]
    pred_flat = [label for sent in y_pred for label in sent]
    conf_df = pd.crosstab(pd.Series(true_flat, name='Actual'),
                          pd.Series(pred_flat, name='Predicted'))

    # Save confusion matrix heatmap
    conf_plot_path = Path(__file__).parent.parent / "logs" / f"conf_matrix_{tag}.png"
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {tag.upper()} Set")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(conf_plot_path)
    plt.close()
    logger.info(f"{tag.upper()} Confusion matrix plot saved at {conf_plot_path}")

    # Log top errors
    errors = []
    for i, (sent_x, true_tags, pred_tags) in enumerate(zip(X, y, y_pred)):
        for j, (true, pred) in enumerate(zip(true_tags, pred_tags)):
            if true != pred:
                word = sent_x[j].get('word', '?')
                errors.append((word, true, pred))
    error_df = pd.DataFrame(errors, columns=['Word', 'True Label', 'Predicted Label'])

    if not error_df.empty:
        logger.info(f"{tag.upper()} Top Misclassified Tokens:\n{error_df.value_counts().head(10)}")

    error_log_path = Path(__file__).parent.parent / "logs" / f"crf_errors_{tag}.csv"
    error_df.to_csv(error_log_path, index=False)
    logger.info(f"{tag.upper()} Error log saved at {error_log_path}")

    return report, f1_per_label, error_df