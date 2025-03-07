
from pathlib import Path
import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


################################################################################################
# Helper Functions
################################################################################################


def merge_data(feeds: pd.DataFrame, ads: pd.DataFrame) -> pd.DataFrame:

    feeds['total_clicks_numeric'] = feeds['u_click_ca2_news'].apply(
      lambda s: sum(int(i) for i in s.split('^')) if isinstance(s, str) else 0
    )

    # aggregate news feed by user
    aggregated_feeds = feeds.groupby('u_userId').agg(
        total_impressions=('u_userId', 'count'),
        total_clicks=('total_clicks_numeric', 'sum'),
        avg_refresh_times=('u_refreshTimes', 'mean'),
        total_dislikes=('i_dislikeTimes', 'sum'),
        total_upvotes=('i_upTimes', 'sum'),
        unique_news_categories=('u_newsCatInterestsST', lambda x: len(set("^".join(x).split("^")))),
        most_common_category=('u_newsCatInterestsST', lambda x: pd.Series("^".join(x).split("^")).value_counts().index[0] if not x.empty else None)
    ).reset_index()

    aggregated_feeds.rename(columns={'u_userId': 'user_id'}, inplace=True)
    merged = ads.merge(aggregated_feeds, on='user_id', how='left')

    merged['ctr'] = merged['total_impressions'] / merged['total_clicks']
    merged['category_diversity'] = merged['unique_news_categories'] / merged['unique_news_categories'].max()

    merged['pt_d'] = pd.to_datetime(merged['pt_d'], format='%Y%m%d%H%M')
    merged['pt_d_year'] = merged['pt_d'].dt.year
    merged['pt_d_month'] = merged['pt_d'].dt.month

    # missing values for numeric cols
    merged.fillna({
        'total_clicks': 0,
        'total_impressions': 0,
        'avg_refresh_times': merged['avg_refresh_times'].median(),
        'total_dislikes': 0,
        'total_upvotes': 0,
        'category_diversity': 0,
        'ctr': 0,
        'most_common_category': 'unknown' #Add fill value
    }, inplace=True)

    return merged


def load_data(ads_path: Path, feeds_path: Path) -> pd.DataFrame:
  feeds: pd.DataFrame = pd.read_csv(feeds_path)
  ads: pd.DataFrame = pd.read_csv(ads_path)
  return merge_data(ads=ads, feeds=feeds)


def split_data(data: pd.DataFrame, split: float=0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
  selected_i = np.random.permutation(data.index)
  split_i: int = int(len(data) * split)
  return data.loc[selected_i[:split_i]], data.loc[selected_i[split_i:]]


def clean_numeric_data(X: pd.DataFrame) -> pd.DataFrame:
    # Replace infinite values with NaN 
    X = X.replace([np.inf, -np.inf], np.nan)
    # NaN values with the median of each column
    return X.fillna(X.median())  


################################################################################################
# Load in Data and Form CV sets
################################################################################################


data_path: Path = Path(kagglehub.dataset_download("xiaojiu1414/digix-global-ai-challenge"))

# Load the full dataset
training_data = load_data(
    feeds_path=data_path / "train" / "train_data_feeds.csv",
    ads_path=data_path / "train" / "train_data_ads.csv"
)

small_data = training_data.sample(frac=0.01, random_state=42).reset_index(drop=True)

# Split into training (80%) and test (20%)
training_data, test_data = train_test_split(small_data, test_size=0.2, random_state=42)

# Further split training into training (80%) and validation (20%)
training_data, validation_data = train_test_split(training_data, test_size=0.2, random_state=42)


numeric_features = training_data.select_dtypes(include='number').columns.tolist()


# training data
X_train_numeric = training_data[numeric_features].drop(columns=['label'])
y_train_numeric = training_data[numeric_features]['label']
X_train_numeric = clean_numeric_data(X_train_numeric)

# validation data
X_validation_numeric = validation_data[numeric_features].drop(columns=['label'])
y_validation_numeric = validation_data[numeric_features]['label']
X_validation_numeric = clean_numeric_data(X_validation_numeric)

# test data
X_test_numeric = test_data[numeric_features].drop(columns=['label'])
y_test_numeric = test_data[numeric_features]['label']
X_test_numeric = clean_numeric_data(X_test_numeric)




