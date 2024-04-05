from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def create_dataframe():
    csv_file_path = "../data/heating_source.csv"
    df = pd.read_csv(csv_file_path)
    print("data imported from ", csv_file_path)
    print(df.head())

    df = pd.get_dummies(df, columns=['ZIP'])
    # Convert categories to numeric
    label_encoder = LabelEncoder()

    df['CATEGORY'] = label_encoder.fit_transform(df['ENERGY_SOURCE_TEXT'])
    label_mappings = {index: label for index, label in enumerate(label_encoder.classes_)}
    print(label_mappings)
    df = df.drop(columns='ENERGY_SOURCE_TEXT')

    category_counts = df['CATEGORY'].value_counts()
    print(category_counts)

    return df, label_mappings


def print_metrics_nn(predictions, label_mappings, y_test):
    predictions_labels = np.argmax(predictions, axis=1)
    print_metrics(predictions_labels, label_mappings, y_test)


def print_metrics(predictions, label_mappings, y_test):
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy}')

    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(cm)

    report = classification_report(y_test, predictions,
                                   target_names=[str(label) for label in label_mappings.values()], zero_division=0)
    print("Classification Report:")
    print(report)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_mappings.values(),
                yticklabels=label_mappings.values())
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Heating Type Confusion Matrix')
    plt.show()


def plot_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='lower right')
    plt.show()
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper right')
    plt.show()
