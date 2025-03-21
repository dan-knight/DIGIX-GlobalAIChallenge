
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from sklearn.mixture import GaussianMixture

from digix.analysis.evaluation import evaluate_synthetic_data_quality


smote_knn = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled_knn, y_train_resampled_knn = smote_knn.fit_resample(X_train_numeric, y_train_numeric)

print("Original Distribution:", np.bincount(y_train_numeric))
print("SMOTE-KNN Distribution:", np.bincount(y_train_resampled_knn))


smote_svm = SVMSMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled_svm, y_train_resampled_svm = smote_svm.fit_resample(X_train_numeric, y_train_numeric)

print("Original Distribution:", np.bincount(y_train_numeric))
print("SMOTE-SVM Distribution:", np.bincount(y_train_resampled_svm))


minority_data = X_train_numeric[y_train_numeric == 1]
minority_label = 1
majority_label = 0

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(minority_data)

n_synthetic_samples = sum(y_train_numeric == majority_label) - sum(y_train_numeric == minority_label)
synthetic_samples, _ = gmm.sample(n_synthetic_samples)

X_train_resampled_gmm = np.vstack([X_train_numeric, synthetic_samples])
y_train_resampled_gmm = np.hstack([y_train_numeric, [minority_label] * n_synthetic_samples])

print("Original Distribution:", np.bincount(y_train_numeric))
print("SMOTE-GMM Distribution:", np.bincount(y_train_resampled_gmm))


original_data = pd.concat([X_train_numeric, y_train_numeric], axis=1)


synthetic_data = pd.concat([X_train_resampled_knn, y_train_resampled_knn], axis=1)
results_smoteknn = evaluate_synthetic_data_quality(original_data, synthetic_data)

print("Propensity Score=", results_smoteknn['propensity_score_auc'])
print("Precision:", results_smoteknn['precision'])
print("Recall:", results_smoteknn['recall'])
print("Precision-Recall Density (F1-like):", results_smoteknn['pr_density'])
print("Privacy Risk (NNDR):", results_smoteknn['privacy_risk_distance'])


synthetic_data = pd.concat([X_train_resampled_svm, y_train_resampled_svm], axis=1)
results_smotesvm = evaluate_synthetic_data_quality(original_data, synthetic_data)

print("Propensity Score=", results_smotesvm['propensity_score_auc'])
print("Precision:", results_smotesvm['precision'])
print("Recall:", results_smotesvm['recall'])
print("Precision-Recall Density (F1-like):", results_smotesvm['pr_density'])
print("Privacy Risk (NNDR):", results_smotesvm['privacy_risk_distance'])


synthetic_data = pd.concat([pd.DataFrame(X_train_resampled_gmm), pd.DataFrame(y_train_resampled_gmm)], axis=1)
synthetic_data.columns = original_data.columns

results_smotegmm = evaluate_synthetic_data_quality(original_data, synthetic_data)

print("Propensity Score=", results_smotegmm['propensity_score_auc'])
print("Precision:", results_smotegmm['precision'])
print("Recall:", results_smotegmm['recall'])
print("Precision-Recall Density (F1-like):", results_smotegmm['pr_density'])
print("Privacy Risk (NNDR):", results_smotegmm['privacy_risk_distance'])
