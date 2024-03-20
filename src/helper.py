from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def create_dataframe():
    engine = create_engine('sqlite:///../data/data.sqlite')

    query_training_data = """select DPLZ4 as PLZ,
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

    query_training_data2 = """select DPLZ4 as ZIP,
       GAREA * GASTW as AREA,
       GWAERDATH1 as UPDATE_DATE,
       c1.CODTXTLD   as ENERGY_SOURCE_TEXT
       from building b,
            entrance e,
            (select CECODID, CODTXTLD from codes WHERE CMERKM = 'GENH1') c1
                where GWAERDATH1 != ''
                  AND GENH1 NOT IN ('', '7500', '7598', '7599', '7550')
                  AND e.EGID = b.EGID
                  AND b.GENH1 = c1.CECODID"""

    df = pd.read_sql_query(query_training_data2, engine)

    df['ZIP'] = pd.to_numeric(df['ZIP'], errors='coerce')
    df['AREA'] = pd.to_numeric(df['AREA'], errors='coerce')
    df['UPDATE_YEAR'] = df['UPDATE_DATE'].str.split('-').str[0]
    df = df.drop(columns='UPDATE_DATE')
    df.dropna(subset=['UPDATE_YEAR'], inplace=True)
    # drop 0 values
    df = df.replace(0, np.nan)
    df = df.dropna()

    print(df['ENERGY_SOURCE_TEXT'].value_counts())

    category_mapping = {
        'Heizöl': 'Heizöl',
        'Gas': 'Gas',
        'Elektrizität': 'Elektrizität',
        'Luft': 'Wärmepumpe',
        'Erdregister': 'Wärmepumpe',
        'Erdwärme (generisch)': 'Wärmepumpe',
        'Erdwärmesonde': 'Wärmepumpe',
        'Fernwärme (Hochtemperatur)': 'Fernwärme',
        'Fernwärme (Niedertemperatur)': 'Fernwärme',
        'Fernwärme (generisch)': 'Fernwärme',
        'Holz (Pellets)': 'Holz',
        'Holz (Schnitzel)': 'Holz',
        'Holz (Stückholz)': 'Holz',
        'Holz (generisch)': 'Holz',
        'Sonne (thermisch)': 'Wärmepumpe',
        'Wasser (Grundwasser, Oberflächenwasser, Abwasser)': 'Wärmepumpe'}

    df['ENERGY_SOURCE_TEXT'] = df['ENERGY_SOURCE_TEXT'].map(category_mapping)
    print(df['ENERGY_SOURCE_TEXT'].unique())
    df = pd.get_dummies(df, columns=['ZIP'])

    # Convert categories to numeric
    label_encoder = LabelEncoder()

    df['CATEGORY'] = label_encoder.fit_transform(df['ENERGY_SOURCE_TEXT'])
    label_mappings = {index: label for index, label in enumerate(label_encoder.classes_)}
    print(label_mappings)
    df = df.drop(columns='ENERGY_SOURCE_TEXT')

    category_counts = df['CATEGORY'].value_counts()
    print(category_counts)

    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    print(df.head())

    return df_shuffled, label_mappings


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
