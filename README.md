# Heating type prediction
Based on data from the Swiss Buildings & Dwellings Register (GWR), the model is designed to predict the most likely energy source of a building. The sources can be divided into 10 classes (e.g. oil, gas, wood, air, water, district heating, ...)

The data set comprises around 300,000 adjusted data which can be used as a training model and verification data.

Stretch goal: The project aims to compare different models against each other to achieve the most accurate prediction.

# Development
## Project setup
Download https://public.madd.bfs.admin.ch/ch.zip and unzip to folder /data. The files are too big for a free github repo.

## Features
* Gebäudetype
* Gebäudejahr
* Aktualisierungsjahr (der Heizung)
* Fläche (Gebäudefläche * Anzahl Stockwerke)
* PLZ => one hot encoding, erst als Ausbaustufe
