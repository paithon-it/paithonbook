# Il file `vendite.csv` non si crea qui: lo fabbrica la pagina stessa, con un
# blocco visibile prima del `read_csv`, così ogni numero che il testo commenta
# si può rifare da soli («un blocco si costruisce i dati che usa»). Qui restano
# solo i nomi che le pagine usano di passaggio.

# Il blocco dei grafici li usa senza definirli, perché nel libro il punto è
# un altro: come si chiama matplotlib, non da dove vengono i numeri.
mesi = ["gen", "feb", "mar", "apr", "mag", "giu"]
fatturato = [12_000, 13_500, 11_800, 15_200, 16_400, 15_900]
