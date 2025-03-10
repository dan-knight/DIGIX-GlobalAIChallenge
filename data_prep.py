
from pathlib import Path
import kagglehub
from sklearn.model_selection import train_test_split

from digix.utility.data import clean_numeric_data, load_data


data_path: Path = Path(kagglehub.dataset_download("xiaojiu1414/digix-global-ai-challenge"))

training_data = load_data(
    feeds_path=data_path / "train" / "train_data_feeds.csv",
    ads_path=data_path / "train" / "train_data_ads.csv"
)

small_data = training_data.sample(frac=0.01, random_state=42).reset_index(drop=True)

training_data, test_data = train_test_split(small_data, test_size=0.2, random_state=42)
training_data, validation_data = train_test_split(training_data, test_size=0.2, random_state=42)


numeric_features = training_data.select_dtypes(include='number').columns.tolist()

X_train_numeric = training_data[numeric_features].drop(columns=['label'])
y_train_numeric = training_data[numeric_features]['label']
X_train_numeric = clean_numeric_data(X_train_numeric)

X_validation_numeric = validation_data[numeric_features].drop(columns=['label'])
y_validation_numeric = validation_data[numeric_features]['label']
X_validation_numeric = clean_numeric_data(X_validation_numeric)

X_test_numeric = test_data[numeric_features].drop(columns=['label'])
y_test_numeric = test_data[numeric_features]['label']
X_test_numeric = clean_numeric_data(X_test_numeric)




