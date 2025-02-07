import pandas as pd
from io import BytesIO
from zipfile import ZipFile
import kaggle

# Obtener el dataset directamente como un archivo ZIP en memoria
competition = "nlp-getting-started"
dataset = kaggle.api.competition_download_files(competition, path=".", quiet=False)

# Cargar el ZIP en memoria sin escribirlo en disco
with ZipFile("nlp-getting-started.zip") as z:
    with z.open("train.csv") as f:
        train_df = pd.read_csv(f)
    with z.open("test.csv") as f:
        test_df = pd.read_csv(f)

# Revisar las primeras filas del dataset sin almacenarlo en disco
print(train_df.head())
print(test_df.head())