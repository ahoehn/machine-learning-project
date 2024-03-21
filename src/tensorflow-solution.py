import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from src.helper import create_dataframe, print_metrics, plot_accuracy, print_metrics_nn

df, label_mappings = create_dataframe()

# Features and target variable
X_one_hot = df.filter(like='ZIP')
X_other = df[['UPDATE_YEAR', 'AREA']]
X = pd.concat([X_one_hot, X_other], axis=1)
y = df['CATEGORY'].values

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert labels to one-hot encoding
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

model = Sequential([
    Dense(3072, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(1024, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y_train_encoded.shape[1], activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(X_train_scaled, y_train_encoded, epochs=50, validation_split=0.1, verbose=1, batch_size=500)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test_scaled, y_test_encoded, verbose=2, batch_size=500)

print('\nTest accuracy:', test_acc)

predictions = model.predict(X_test_scaled, batch_size=500)

print_metrics_nn(predictions, label_mappings, y_test)

plot_accuracy(history)

model.save('heating-predictions.keras')
