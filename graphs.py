import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import kagglehub
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_score, recall_score, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

columns_to_keep = [
    'age', 'device_size', 'app_score', 'total_impressions', 'total_clicks',
    'avg_refresh_times', 'total_dislikes', 'total_upvotes', 'unique_news_categories', 'ctr'
]

fig, axes = plt.subplots(len(columns_to_keep), 1, figsize=(10, len(columns_to_keep) * 5))


# Iterate over each feature (column) in the datasets
for i, column in enumerate(columns_to_keep):
    # Plot KDE for original data
    sns.kdeplot(original_data[column], ax=axes[i], color='blue', label='Original', fill=True)

    # Plot KDE for synthetic data
    sns.kdeplot(synthetic_data_knn[column], ax=axes[i], color='orange', label='Synthetic - KNN', fill=True)
    sns.kdeplot(synthetic_data_svm[column], ax=axes[i], color='green', label='Synthetic - SVM', fill=True)
    sns.kdeplot(synthetic_data_gmm[column], ax=axes[i], color='red', label='Synthetic - GMM', fill=True)

    # Set the title, labels, and legend
    axes[i].set_title(f'{column} - Original vs Synthetic Data')
    axes[i].set_xlabel(f'{column}')
    axes[i].set_ylabel('Density')
    axes[i].legend()

# Adjust layout for better spacing
plt.tight_layout()
plt.show()