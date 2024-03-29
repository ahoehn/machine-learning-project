# Heating type prediction
Based on data from the Swiss Buildings & Dwellings Register (GWR), the model is designed to predict the most likely energy source of a building. The sources can be divided into 10 classes (e.g. oil, gas, wood, air, water, district heating, ...)

The data set comprises around 300,000 adjusted data which can be used as a training model and verification data.

Stretch goal: The project aims to compare different models against each other to achieve the most accurate prediction.

# Development
## Project setup
run tensorflow-solution.py or random-forest-solution.py

## Features used for the model
* Gebäudejahr
* Aktualisierungsjahr (der Heizung)
* Fläche (Gebäudefläche * Anzahl Stockwerke)
* PLZ => one hot encoding, erst als Ausbaustufe

## Original Sources
Download https://public.madd.bfs.admin.ch/ch.zip and unzip to folder /data. The files are too big for a free github repo.
Other sources are:
- https://data.geo.admin.ch/ch.swisstopo-vd.ortschaftenverzeichnis_plz/ortschaftenverzeichnis_plz/ortschaftenverzeichnis_plz_2056.csv.zip
- https://www.estv.admin.ch/dam/estv/de/dokumente/estv/steuerstatistiken/direkte-bundessteuer/statistik-dbst-np-kennzahlen-ohne-2020.xlsx.download.xlsx/statistik-dbst-np-kennzahlen-ohne-2020.xlsx
rund data import-and-prepare-data.py which creates a new heating_source.csv


