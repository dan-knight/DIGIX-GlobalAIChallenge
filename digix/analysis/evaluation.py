

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats


def evaluate_propensity_score(
    original: "pd.DataFrame",
    synthetic: "pd.DataFrame"
) -> float:
    """
    trains model to distinguish original data from synthetic
    Returns the propensity score (ROC AUC) score
    """
    # copy data
    orig = original.copy()
    synth = synthetic.copy()
    orig['source'] = 1
    synth['source'] = 0
    combined = pd.concat([orig, synth], ignore_index=True)

    y = combined['source']
    X = combined.drop(columns=['source'])

    X = X.fillna(-9999)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # calculate propensity score
    y_prob = clf.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)

    return auc_score


def evaluate_precision_recall_density(
    original: "pd.DataFrame",
    synthetic: "pd.DataFrame",
    n_neighbors: int
) -> tuple[float, float, float]:
    """
    Calculates:
    - Precision: prop of synthetic points within threshold that's derived from original df
    - Recall: prop of original points within threshold from any synthetic point
    Returns precision, recall, and PRD - which is the harmonic mean of precision and recall
    """
    nn_orig = NearestNeighbors(n_neighbors=n_neighbors)
    nn_orig.fit(original)

    distances_synth, _ = nn_orig.kneighbors(synthetic)

    nn_orig_orig = NearestNeighbors(n_neighbors=n_neighbors)
    nn_orig_orig.fit(original)
    distances_orig, _ = nn_orig_orig.kneighbors(original)
    threshold = np.median(distances_orig)


    precision = np.mean(np.min(distances_synth, axis=1) <= threshold)

    nn_synth = NearestNeighbors(n_neighbors=n_neighbors)
    nn_synth.fit(synthetic)
    distances_orig_to_synth, _ = nn_synth.kneighbors(original)
    recall = np.mean(np.min(distances_orig_to_synth, axis=1) <= threshold)

    pr_density = 2 * (precision * recall) / (precision + recall + 1e-9)

    return precision, recall, pr_density


def evaluate_privacy(original: "pd.DataFrame", synthetic: "pd.DataFrame") -> float:
    """
    mesures privacy using avg Nearest-Neighbour Distance Ratio (NNDR)
    
    NNDR = distance_to_nearest / distance_to_second_nearest
    higher values indicate high privacy   
    """
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(original)

    distances, _ = nn.kneighbors(synthetic)
    d1 = distances[:, 0]
    d2 = distances[:, 1]

    ratios = d1 / (d2 + 1e-9)
    avg_nndr = ratios.mean()
    
    return avg_nndr


def evaluate_synthetic_data_quality(
    original: "pd.DataFrame",
    synthetic: "pd.DataFrame"
) -> dict[str, float]:
    """
    Calculates:
      1. Propensity score (ROC AUC)
      2, Precision, recall, and precision-recall density
      3. Privacy metric (avg nearest neighbor distance)
    returns a dict of each eval metric
    """
    propensity_auc = evaluate_propensity_score(original, synthetic)
    orig_numeric = original.select_dtypes(include=[np.number])
    synth_numeric = synthetic.select_dtypes(include=[np.number])
    precision, recall, pr_density = evaluate_precision_recall_density(orig_numeric, synth_numeric, n_neighbors=5)
    privacy_risk = evaluate_privacy(orig_numeric, synth_numeric)

    results = {
        'propensity_score_auc': propensity_auc,
        'precision': precision,
        'recall': recall,
        'pr_density': pr_density,
        'privacy_risk_distance': privacy_risk
    }

    return results
