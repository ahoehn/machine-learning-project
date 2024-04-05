from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
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
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Step 5: Evaluate the model
predictions = knn.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

print_metrics(predictions, label_mappings, y_test)

dump(knn, '../results/models/knn_model.joblib')
