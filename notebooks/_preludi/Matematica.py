import numpy as np

# La matrice dei dati che le pagine danno per esistente: righe = esempi,
# colonne = feature. Nel libro il punto è l'operazione, non la provenienza.
rng = np.random.default_rng(0)
X = rng.normal(loc=[10.0, 200.0, 3.0], scale=[2.0, 40.0, 0.5], size=(50, 3))
