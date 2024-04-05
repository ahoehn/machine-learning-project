from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from joblib import dump
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from src.helper import create_dataframe, print_metrics, plot_accuracy

df, label_mappings = create_dataframe()

df = df[df['PLACE'].str.contains('Zürich', case=False, na=False)]

# Splitting data into features and target
X_one_hot = df.filter(like='ZIP')
X_other = df[['UPDATE_YEAR', 'AREA', 'BUILDING_YEAR', 'INCOME']]
X = pd.concat([X_one_hot, X_other], axis=1)
y = df['CATEGORY']
#
# Optionally, scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the parameter grid
param_grid = {
    #'n_estimators': [600, 800, 1000, 3000, 5000],  #{'n_estimators': 3000}
    #'max_depth': [None, 10, 20, 30, 40, 50],  # {'max_depth': 30}
    #'min_samples_split': [2, 5, 10, 20, 40, 80, 160],  # {'min_samples_split': 2}
    #'min_samples_leaf': [1, 2, 4, 8, 16, 31],    # Minimum number of samples required at each leaf node
    #'max_features': [None,'log2', 'sqrt'], {'max_features': 'sqrt'}
    #'bootstrap': [True, False]        # Method of selecting samples for training each tree
    'min_weight_fraction_leaf': [0.0,0.25,0.5]
}

# Initialize the classifier
rf = RandomForestClassifier(oob_score=True, n_estimators=3000, verbose = 1, max_depth=30, min_samples_split=2, max_features='sqrt', n_jobs=-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the grid search to the data (assuming you have a dataset with X features and y labels)
history = grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
