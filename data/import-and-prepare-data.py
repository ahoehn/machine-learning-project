from sqlalchemy import create_engine, text
import pandas as pd

from src.helper import prepare_data

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
