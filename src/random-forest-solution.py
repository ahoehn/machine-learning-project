from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.helper import create_dataframe, print_metrics, plot_accuracy

df, label_mappings = create_dataframe()

# Splitting data into features and target
X = df[['GBAUJ', 'FLAECHE']]
y = df['CATEGORY']
#
# Optionally, scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model to the training data
history = random_forest.fit(X_train, y_train)

# Predict the categories of the test set
predictions = random_forest.predict(X_test)

print_metrics(predictions, label_mappings, y_test)
