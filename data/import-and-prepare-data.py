from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np

def prepare_data(csv_file_path):
    engine = create_engine('sqlite:///../data/data.sqlite')

    query_training_data2 = """select 
        DPLZ4 as ZIP,
        b.GGDENAME as PLACE,
       GAREA * GASTW as AREA,
       GBAUJ as BUILDING_YEAR,
       GWAERDATH1 as UPDATE_DATE,
       c1.CODTXTLD   as ENERGY_SOURCE_TEXT,
       j.max_median_income as INCOME
    from building b,
         entrance e,
         (select CECODID, CODTXTLD from codes WHERE CMERKM = 'GENH1') c1,
         (SELECT g.plz, MAX(i.median_income) AS max_median_income
          FROM gemeinde g
                   JOIN income i ON g.gemeinde_id = i.gemeinde_id
          GROUP BY g.plz) j
    where GWAERDATH1 != ''
      AND GBAUJ != ''
      AND GENH1 NOT IN ('', '7500', '7598', '7599', '7550')
      AND e.EGID = b.EGID
      AND b.GENH1 = c1.CECODID
      AND j.plz = DPLZ4
      AND GAREA * GASTW  > 0
    ORDER BY b.EGID"""

    df = pd.read_sql_query(query_training_data2, engine)

    df['ZIP'] = pd.to_numeric(df['ZIP'], errors='coerce')
    df['AREA'] = pd.to_numeric(df['AREA'], errors='coerce')
    df['INCOME'] = df['INCOME'].str.replace("’", "").astype(int)
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

    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    print(df.head())
    df.to_csv(csv_file_path, index=False)

engine = create_engine('sqlite:///../data/data.sqlite')

with engine.connect() as conn:
    conn.execute(text("drop table gemeinde"))
    conn.execute(text("""CREATE TABLE gemeinde (
                          plz INT,
                          gemeinde_id INT,
                          gemeinde_name VARCHAR);"""))
    conn.execute(text("drop table income"))
    conn.execute(text("""CREATE TABLE income (
                        gemeinde_id INT PRIMARY KEY,
                        gemeinde_name VARCHAR,
                        median_income INT
                );"""))

csv_file_path2 = './gemeinde-plz.csv'
data2 = pd.read_csv(csv_file_path2, sep=';')
data2 = data2[['plz', 'gemeinde_id', 'gemeinde_name']]
data2.to_sql('gemeinde', con=engine, if_exists='append', index=False)

csv_file_path2 = './gemeinde-einkommen.csv'
data2 = pd.read_csv(csv_file_path2, sep=';')
data2 = data2[['gemeinde_id', 'gemeinde_name', 'median_income']]
data2.to_sql('income', con=engine, if_exists='append', index=False)

print('Data imported successfully')
csv_file_path = "./heating_source.csv"
prepare_data(csv_file_path)
print('Data prepared successfully as ', csv_file_path)
