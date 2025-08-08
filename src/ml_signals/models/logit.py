from sklearn.linear_model import LogisticRegression
def train_logit(X, y, sample_weight=None, C=1.0, l1_ratio=0.5, seed=42):
    clf = LogisticRegression(
        penalty="elasticnet", solver="saga",
        C=C, l1_ratio=l1_ratio, max_iter=4000, random_state=seed
    )
    clf.fit(X, y, sample_weight=sample_weight)
    return clf
