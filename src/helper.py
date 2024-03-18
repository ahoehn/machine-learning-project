from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def create_dataframe():
    engine = create_engine('sqlite:///../data/data.sqlite')

    query_training_data = """select DPLZ4 as ZIP,
                 GBAUJ,
                 GASTW,
                 GAREA * GASTW as FLAECHE,
                 GWAERZH1,
                 w1.CODTXTLD   as GWAERZH1TXT,
                 GENH1,
                 c1.CODTXTLD   as GENH1TXT,
                 GWAERDATH1
          from building b,
               entrance e,
               (select CECODID, CODTXTLD from codes WHERE CMERKM = 'GWAERZH1') w1,
               (select CECODID, CODTXTLD from codes WHERE CMERKM = 'GENH1') c1
          WHERE b.GABBJ == ''
            AND b.GBAUJ != ''
            AND GENH1 NOT IN ('', '7500', '7598', '7599', '7550')
            AND b.GWAERZH1 = w1.CECODID
            AND b.GENH1 = c1.CECODID
            AND e.EGID = b.EGID
            AND (GSTAT NOT in ('1005', '1007', '1008'))"""

    df = pd.read_sql_query(query_training_data, engine)

    df['ZIP'] = pd.to_numeric(df['ZIP'], errors='coerce')
    df['GBAUJ'] = pd.to_numeric(df['GBAUJ'], errors='coerce')
    df['FLAECHE'] = pd.to_numeric(df['FLAECHE'], errors='coerce')
    df['GENH1'] = pd.to_numeric(df['GENH1'], errors='coerce')
    df['GASTW'] = pd.to_numeric(df['GASTW'], errors='coerce')
    df['GWAERDATH1'] = pd.to_datetime(df['GWAERDATH1'], format='%Y-%m-%d', errors='coerce')
    df['YEAR_OF_REPLACEMENT'] = df['GWAERDATH1'].dt.year

    genh1_counts = df['GENH1TXT'].value_counts()
    print(genh1_counts)
    # merge categories
    category_mapping = {
        'Heizöl': 'Heizöl',
        'Gas': 'Gas',
        'Elektrizität': 'Elektrizität',
        'Luft': 'Luft',
        'Erdregister': 'Erdwärme',
        'Erdwärme (generisch)': 'Erdwärme',
        'Erdwärmesonde': 'Erdwärme',
        'Fernwärme (Hochtemperatur)': 'Fernwärme',
        'Fernwärme (Niedertemperatur)': 'Fernwärme',
        'Fernwärme (generisch)': 'Fernwärme',
        'Holz (Pellets)': 'Holz',
        'Holz (Schnitzel)': 'Holz',
        'Holz (Stückholz)': 'Holz',
        'Holz (generisch)': 'Holz',
        'Sonne (thermisch)': 'Sonne',
        'Wasser (Grundwasser, Oberflächenwasser, Abwasser)': 'Wasser'}
    df['GENH1TXT'] = df['GENH1TXT'].map(category_mapping)
    print(df['GENH1TXT'].unique())

    # Convert categories to numeric
    label_encoder = LabelEncoder()
    df['CATEGORY'] = label_encoder.fit_transform(df['GENH1TXT'])
    label_mappings = {index: label for index, label in enumerate(label_encoder.classes_)}
    print(label_mappings)

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
