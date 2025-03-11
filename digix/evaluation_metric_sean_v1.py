

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats


def evaluate_propensity_score(original, synthetic):
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

    # model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # calculate propensity score
    y_prob = clf.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)

    return auc_score

def evaluate_precision_recall_density(original, synthetic, n_neighbors):
    """
    Calculates:
    - Precision: prop of synthetic points within threshold that's derived from original df
    - Recall: prop of original points within threshold from any synthetic point
    Returns precision, recall, and PRD - which is the harmonic mean of precision and recall
    """
    # 1. Fit KNN on original df
    nn_orig = NearestNeighbors(n_neighbors=n_neighbors)
    nn_orig.fit(original)

    # 2. For each synthetic sample, find distances to knn in original df
    distances_synth, _ = nn_orig.kneighbors(synthetic)

    # 3. Calc threshold from the dist of distances within the original df
    nn_orig_orig = NearestNeighbors(n_neighbors=n_neighbors)
    nn_orig_orig.fit(original)
    distances_orig, _ = nn_orig_orig.kneighbors(original)
    threshold = np.median(distances_orig)


    precision = np.mean(np.min(distances_synth, axis=1) <= threshold)

    # calc if there's a synthetic obs within the threshold
    nn_synth = NearestNeighbors(n_neighbors=n_neighbors)
    nn_synth.fit(synthetic)
    distances_orig_to_synth, _ = nn_synth.kneighbors(original)
    recall = np.mean(np.min(distances_orig_to_synth, axis=1) <= threshold)

    # calc PRD
    pr_density = 2 * (precision * recall) / (precision + recall + 1e-9)

    return precision, recall, pr_density


def evaluate_privacy(original: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    """
    a privacy metric based on the avg nearest neighbor distance
    """
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(original)
    distances, _ = nn.kneighbors(synthetic)
    avg_distance = np.mean(distances)
    
    return avg_distance


def evaluate_synthetic_data_quality(original, synthetic):
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



##############################################################################################
# Input original and synthetic data sets - numeric only
##############################################################################################

original_data = pd.concat([X_train_numeric, y_train_numeric], axis=1)


# SMOTE KNN
synthetic_data = pd.concat([X_train_resampled_knn, y_train_resampled_knn], axis=1)
results_smoteknn = evaluate_synthetic_data_quality(original_data, synthetic_data)

print("Propensity Score=", results_smoteknn['propensity_score_auc'])
print("Precision:", results_smoteknn['precision'])
print("Recall:", results_smoteknn['recall'])
print("Precision-Recall Density (F1-like):", results_smoteknn['pr_density'])
print("Privacy Risk (avg. nearest neighbor distance):", results_smoteknn['privacy_risk_distance'])



# SMOTE SVM
synthetic_data = pd.concat([X_train_resampled_svm, y_train_resampled_svm], axis=1)
results_smotesvm = evaluate_synthetic_data_quality(original_data, synthetic_data)

print("Propensity Score=", results_smotesvm['propensity_score_auc'])
print("Precision:", results_smotesvm['precision'])
print("Recall:", results_smotesvm['recall'])
print("Precision-Recall Density (F1-like):", results_smotesvm['pr_density'])
print("Privacy Risk (avg. nearest neighbor distance):", results_smotesvm['privacy_risk_distance'])



# SMOTE GMM
synthetic_data = pd.concat([pd.DataFrame(X_train_resampled_gmm), pd.DataFrame(y_train_resampled_gmm)], axis=1)
synthetic_data.columns = original_data.columns

results_smotegmm = evaluate_synthetic_data_quality(original_data, synthetic_data)

print("Propensity Score=", results_smotegmm['propensity_score_auc'])
print("Precision:", results_smotegmm['precision'])
print("Recall:", results_smotegmm['recall'])
print("Precision-Recall Density (F1-like):", results_smotegmm['pr_density'])
print("Privacy Risk (avg. nearest neighbor distance):", results_smotegmm['privacy_risk_distance'])






















