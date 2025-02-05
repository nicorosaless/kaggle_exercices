# Guardar predicciones
import pandas as pd
sub = pd.read_csv("submissions.csv")
sub = sub[['Price']]
sub['id'] = range(30000, 30000 + len(sub))
sub = sub[['id', 'Price']]  # Intercambiar las posiciones de las columnas
sub.to_csv("submissions.csv", index=False)
