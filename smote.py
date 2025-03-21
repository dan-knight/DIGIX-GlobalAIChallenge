
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from sklearn.mixture import GaussianMixture


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
