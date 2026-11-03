"""Train and evaluate the handover models, then save them to models/."""
import json
from pathlib import Path

import pandas as pd

from modules.handover_prediction import save_bundle, train_and_evaluate

ROOT = Path(__file__).parent


def main():
    data_path = ROOT / "dataset" / "network_data.csv"
    if not data_path.exists():
        from generate_dataset import main as gen
        gen()
    df = pd.read_csv(data_path)
    bundle = train_and_evaluate(df)
    save_bundle(bundle, ROOT / "models" / "handover_model.pkl")
    summary = {k: bundle[k] for k in ("metrics", "cv_f1_mean", "cv_f1_std", "feature_importance",
                                      "n_train", "n_test", "seed", "trained_at")}
    (ROOT / "models" / "metrics.json").write_text(json.dumps(summary, indent=2))
    for name, m in bundle["metrics"].items():
        print(f"{name:22s} acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
              f"rec={m['recall']:.3f} f1={m['f1']:.3f}")
    print(f"Random Forest 5-fold CV F1: {bundle['cv_f1_mean']:.3f} +/- {bundle['cv_f1_std']:.3f}")
    print("Saved models/handover_model.pkl and models/metrics.json")


if __name__ == "__main__":
    main()
