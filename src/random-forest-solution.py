from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from joblib import dump

from src.helper import create_dataframe, print_metrics, plot_accuracy

df, label_mappings = create_dataframe()

# Splitting data into features and target
X_one_hot = df.filter(like='ZIP')
X_other = df[['UPDATE_YEAR', 'AREA', 'BUILDING_YEAR', 'INCOME']]
X = pd.concat([X_one_hot, X_other], axis=1)
y = df['CATEGORY']
#
# Optionally, scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
#model = RandomForestClassifier(oob_score=True, n_estimators=3000, verbose = 1, max_depth=30, min_samples_split=2, max_features='sqrt', n_jobs=-1)
model = RandomForestClassifier(oob_score=True, n_estimators=3000, verbose = 1, n_jobs=-1)


# Fit the model to the training data
history = model.fit(X_train, y_train)

# Predict the categories of the test set
predictions = model.predict(X_test)

print_metrics(predictions, label_mappings, y_test)

print("OOB Score: ", model.oob_score_)

dump(model, '../results/models/random_forest_model-simple2.joblib')
