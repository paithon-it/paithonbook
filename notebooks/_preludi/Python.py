import pandas as pd

# Il file che la pagina dà per esistente: sei righe inventate, quel tanto che
# basta a far funzionare filtri, raggruppamenti e grafici.
pd.DataFrame({
    "nome":  ["Ada", "Bruno", "Carla", "Dario", "Elena", "Furio"],
    "eta":   [34, 28, 41, 36, 52, 23],
    "citta": ["Milano", "Torino", "Milano", "Napoli", "Milano", "Torino"],
    "spesa": [120.5, 89.0, 240.0, 55.5, 310.0, 74.9],
}).to_csv("vendite.csv", index=False)

# Il blocco dei grafici li usa senza definirli, perché nel libro il punto è
# un altro: come si chiama matplotlib, non da dove vengono i numeri.
mesi = ["gen", "feb", "mar", "apr", "mag", "giu"]
fatturato = [12_000, 13_500, 11_800, 15_200, 16_400, 15_900]
