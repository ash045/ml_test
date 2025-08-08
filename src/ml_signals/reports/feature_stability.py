import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_lgbm_feature_stability(imp_df: pd.DataFrame, report_dir: str, top_k: int = 30):
    """
    Build a fold-wise stability report for LightGBM feature importances.
    - Ranks features within each fold by gain (descending).
    - Computes mean rank, rank std, stability score = 1 - std(rank)/max_rank.
    - Saves CSV summary + barplot of top_k by mean rank (lowest is best).
    """
    df = imp_df.copy()
    assert {"fold","feature","gain"}.issubset(df.columns), "imp_df must have fold, feature, gain"

    # Rank within each fold (1 = best)
    df["rank"] = df.groupby("fold")["gain"].rank(ascending=False, method="average")

    # Aggregate
    agg = df.groupby("feature").agg(
        mean_gain=("gain","mean"),
        sum_gain=("gain","sum"),
        mean_rank=("rank","mean"),
        std_rank=("rank","std"),
        count_folds=("fold","nunique")
    ).reset_index()

    max_rank = df.groupby("fold")["rank"].max().mean()
    agg["stability"] = 1.0 - (agg["std_rank"] / (max_rank + 1e-12))

    # Sort by mean_rank then stability (best at top)
    agg = agg.sort_values(["mean_rank","stability","sum_gain"], ascending=[True, False, False])

    # Save CSV
    os.makedirs(report_dir, exist_ok=True)
    csv_path = os.path.join(report_dir, "lgbm_feature_stability.csv")
    agg.to_csv(csv_path, index=False)

    # Plot top_k by mean_rank
    top = agg.head(top_k)
    plt.figure(figsize=(10, max(4, int(0.3*len(top)))))
    y = np.arange(len(top))[::-1]
    plt.barh(y, top["stability"].values)
    plt.yticks(y, top["feature"].values)
    plt.xlabel("Stability (1 - rank std / max_rank)")
    plt.title("LightGBM Feature Stability (Top by Mean Rank)")
    plot_path = os.path.join(report_dir, "lgbm_feature_stability.png")
    plt.tight_layout(); plt.savefig(plot_path, dpi=120); plt.close()

    # Save JSON summary
    summary = {
        "n_features": int(agg.shape[0]),
        "top_k": int(min(top_k, agg.shape[0])),
        "csv": csv_path,
        "plot": plot_path
    }
    with open(os.path.join(report_dir, "lgbm_feature_stability.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
