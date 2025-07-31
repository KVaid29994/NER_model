from src.data_loader import load_data
from src.pipeline import train_pipeline
from pathlib import Path

if __name__ == "__main__":
    
    project_root = Path(__file__).parent.parent  # one level above /src
    data_path = project_root / "data" / "ner_dataset.csv"
    data = load_data(data_path)
    model, f1_score, report = train_pipeline()
    print(f"F1 Score: {f1_score}")
    print(report)