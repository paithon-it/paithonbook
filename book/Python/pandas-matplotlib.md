# Pandas e Matplotlib: dati e visualizzazione

Prima di addestrare qualunque modello c'è un lavoro poco glamour che occupa,
nella pratica, gran parte del tempo di chi fa machine learning: prendere dei
dati grezzi (un file esportato da un gestionale, uno storico di vendite, un
registro di sensori) e portarli in una forma pulita, ordinata, esplorabile. In
Python questo lavoro ha due strumenti quasi obbligati: **Pandas** per
manipolare le tabelle e **Matplotlib** per guardarle. Se NumPy è l'algebra,
Pandas è il foglio di calcolo programmabile e Matplotlib è la finestra da cui
osservare cosa abbiamo davvero tra le mani.

## Series e DataFrame: la tabella come oggetto

Pandas ruota attorno a due strutture. Una **Series** è una colonna: una
sequenza di valori con un'etichetta ciascuno (l'*indice*). Un **DataFrame** è
una tabella intera: tante Series affiancate che condividono lo stesso indice
di riga.

```{figure} ../figures/pandas-series-dataframe.svg
:name: fig-series-dataframe
:alt: "Schema di un DataFrame: una tabella con i nomi delle colonne in alto e l'indice di riga evidenziato sul lato sinistro. A destra, una singola colonna viene estratta dalla tabella e mostrata come Series, che conserva lo stesso indice di riga della tabella da cui proviene."
:width: 94%

Una colonna staccata da un DataFrame è una Series, e si porta dietro
l'indice. È quell'indice condiviso a permettere di riallineare i dati senza
badare all'ordine delle righe.
```

La parte da fissare in {numref}`fig-series-dataframe` è la colonna evidenziata
a sinistra. L'indice non è un numero di riga qualunque: è l'etichetta con cui
pandas riconosce ogni riga, e resta attaccata ai dati quando si filtra, si
ordina o si estrae una colonna.

`````{tab} Elementare

Immagina un foglio Excel. Ogni colonna ha un'intestazione ("età", "città",
"acquisti") e ogni riga è un cliente. Un DataFrame è esattamente questo, ma
invece di cliccare con il mouse dai istruzioni a parole:

```python
import pandas as pd

df = pd.DataFrame({
    "citta": ["Milano", "Roma", "Napoli"],
    "eta":   [34, 28, 41],
    "spesa": [120.0, 85.5, 60.0],
})
```

Ogni colonna è una Series; tutte insieme formano la tabella. Il vantaggio
rispetto a Excel è che ogni operazione è ripetibile e documentata: la scrivi
una volta e la riesegui su un milione di righe senza cambiare nulla.

`````

`````{tab} Superiore

Un `DataFrame` è una collezione di `Series` allineate su un `Index` comune.
Ogni colonna ha un proprio `dtype` omogeneo (`int64`, `float64`, `object`,
`category`, `datetime64`), il che permette a Pandas di appoggiarsi a NumPy per
le operazioni vettoriali colonna per colonna. L'indice non è un semplice
numero di riga: è una struttura etichettata (anche gerarchica, `MultiIndex`)
usata per l'allineamento automatico. Quando sommi due Series, Pandas non
allinea per posizione ma **per etichetta**, inserendo `NaN` dove le etichette
non combaciano: comportamento che evita interi errori "off-by-one" tipici
degli array grezzi.

`````

## Caricare e ispezionare i dati

Nella realtà i dati non li digiti a mano: li carichi. Il formato più comune è
il **CSV** (un semplice file di testo con i valori separati da virgole, il
formato in cui quasi ogni programma sa esportare una tabella) e la funzione
`read_csv` lo legge in una riga, riconoscendo da sola tipi e intestazioni.

```python
df = pd.read_csv("vendite.csv")

df.head()        # prime 5 righe: uno sguardo veloce
df.info()        # colonne, dtype, valori non nulli, memoria
df.describe()    # statistiche riassuntive delle colonne numeriche
```

Questi tre metodi sono il rituale d'apertura di ogni analisi. `head()` ti dice
*che aspetto* hanno i dati; `info()` ti dice *quanti* sono e se ci sono buchi
(valori mancanti); `describe()` ti dà, per ogni colonna numerica, la media, la
deviazione standard (quanto i valori si sparpagliano attorno alla media), il
minimo, il massimo e i quartili (i valori che dividono i dati in quattro fette
uguali). Prima di ogni modello, questi numeri raccontano già metà della storia.

## Selezionare, filtrare, creare colonne

Una volta caricata la tabella, la si interroga. Selezionare una colonna,
tenere solo le righe che soddisfano una condizione, calcolare una nuova
colonna a partire dalle altre: sono le tre operazioni che si ripetono
all'infinito.

```python
df["spesa"]                    # una colonna (Series)
df[df["eta"] > 30]             # filtro booleano: solo gli over 30
df["spesa_iva"] = df["spesa"] * 1.22   # nuova colonna calcolata
```

`````{tab} Elementare

La riga centrale è la più importante. `df["eta"] > 30` non restituisce un
numero: restituisce una colonna di `True`/`False`, una per riga. Mettendola
tra parentesi quadre, Pandas tiene solo le righe dove il valore è `True`. È
come applicare un colino: la condizione decide cosa passa e cosa resta fuori.
Puoi combinarne più d'una con `&` ("e") e `|` ("o"):

```python
df[(df["eta"] > 30) & (df["citta"] == "Milano")]
```

`````

`````{tab} Superiore

Il filtro booleano è *boolean masking*: la Series di condizione è un vettore
di `bool` che indicizza il DataFrame, esattamente come in NumPy. Le condizioni
si combinano con gli operatori bit a bit `&`, `|`, `~` (non con `and`/`or`
Python, che non sono vettorizzati), e le parentesi sono obbligatorie per via
della precedenza degli operatori. Per selezioni miste per etichetta e
posizione esistono gli accessor `.loc[righe, colonne]` (per etichetta) e
`.iloc[...]` (per posizione intera), che restano il modo canonico e non
ambiguo di indicizzare.

`````

## Raggruppare e aggregare

La domanda che quasi ogni analisi finisce per porsi è: *quanto vale questa
grandezza, suddivisa per categoria?* Spesa media per città, numero di ordini
per mese, errore medio per classe. È il pattern **split-apply-combine**: dividi
i dati in gruppi, applichi una funzione a ciascuno, ricomponi il risultato.

```{figure} ../figures/pandas-selezione-filtri-groupby.svg
:name: fig-split-apply-combine
:alt: "Una tabella unica viene divisa in tre gruppi secondo il valore di una colonna; su ciascun gruppo si applica la stessa funzione di aggregazione, che lo riduce a un solo valore; i tre valori vengono infine ricomposti in una tabella nuova, con una riga per gruppo."
:width: 100%

Le tre mosse in fila. La tabella finale ha una riga per gruppo, e la colonna
su cui si è diviso è diventata il suo indice.
```

L'ultimo passaggio di {numref}`fig-split-apply-combine` è quello che si tende
a dimenticare: dopo un `groupby` la colonna di raggruppamento non è più una
colonna, è l'indice. Da lì in poi si seleziona per etichetta, non per valore,
e questo spiega gran parte dei sorprendenti `KeyError` che seguono.

```python
df.groupby("citta")["spesa"].mean()    # spesa media per città
df.groupby("citta").agg(
    spesa_media=("spesa", "mean"),
    clienti=("spesa", "count"),
)
```

`````{tab} Elementare

`groupby("citta")` mette in scatole separate tutte le righe di Milano, tutte
quelle di Roma, e così via. Poi `.mean()` calcola la media dentro ogni scatola.
Il risultato è una tabellina con una riga per città: il riassunto che
cercavi. È l'equivalente delle tabelle pivot di Excel, ma in una riga di
codice e senza click.

`````

`````{tab} Superiore

Concettualmente `groupby` partiziona le righe secondo una o più chiavi e
applica a ogni gruppo $g$ una funzione di aggregazione. Per la media, sul
gruppo con valori $\{x_1,\dots,x_{n_g}\}$:

$$
\bar{x}_g = \frac{1}{n_g}\sum_{i=1}^{n_g} x_i .
$$

dove $n_g$ è la numerosità del gruppo. Oltre a `mean` sono disponibili
`sum`, `count`, `std`, `min`, `max`, `median` e funzioni arbitrarie via
`agg`/`apply`. Il metodo `agg` con argomenti nominati (*named aggregation*)
produce colonne dal nome esplicito, rendendo il risultato pronto per un
report o per un ulteriore `merge`.

`````

## I valori mancanti

I dati reali sono quasi sempre incompleti: un campo non compilato, un sensore
spento, una risposta saltata. Pandas rappresenta questi buchi con `NaN` (*Not
a Number*), e ignorarli non è un'opzione: un solo `NaN` può propagarsi e
avvelenare un intero calcolo.

```{figure} ../figures/gestire-dati-mancanti.svg
:name: fig-dati-mancanti
:alt: "La stessa tabella con alcune celle vuote, trattata in tre modi affiancati. Nel primo le righe incomplete vengono eliminate e la tabella si accorcia. Nel secondo i buchi vengono riempiti con un valore fisso, la media della colonna. Nel terzo il valore mancante viene ricostruito dal contesto, cioè dalle altre colonne della stessa riga."
:width: 100%

Tre modi di rispondere alla stessa cella vuota. Nessuno è neutro: il primo
butta anche i dati buoni della riga, il secondo inventa un valore plausibile,
il terzo lo inventa con più cura.
```

Vale la pena guardare {numref}`fig-dati-mancanti` ricordando che una casella
vuota è essa stessa un'informazione. Se manca perché il sensore era spento, è
un caso; se manca perché la domanda era imbarazzante, il fatto che manchi dice
qualcosa, e riempirla con la media cancella proprio quel qualcosa.

`````{tab} Elementare

Hai due strade. Puoi **buttare via** le righe incomplete, oppure **riempirle**
con un valore ragionevole (spesso la media o la mediana della colonna):

```python
df.isna().sum()              # quanti buchi per colonna?
df.dropna()                  # elimina le righe con valori mancanti
df["eta"].fillna(df["eta"].median())   # riempi con la mediana
```

La scelta non è tecnica ma di buonsenso: se manca il 2% dei dati puoi
scartarli; se manca il 40% di una colonna, buttarla via distruggerebbe
informazione, e conviene riempire. Non esiste una risposta valida sempre:
dipende da *perché* quel dato manca.

`````

`````{tab} Superiore

Una precisazione sul contenitore: `NaN` è la rappresentazione dei mancanti
per i `float` (e un solo `NaN` forza a `float64` una colonna di interi, come
si nota da `df.info()`); le colonne di date usano `NaT` e i dtype *nullable*
di Pandas usano `pd.NA`.

La strategia dipende dal meccanismo di mancanza (MCAR, MAR, MNAR nella
tassonomia di Rubin): se i dati mancano *completamente a caso* (MCAR) è
garantito che eliminare le righe incomplete non introduca distorsioni, e in
una regressione l'eliminazione resta lecita anche quando la mancanza dipende
solo dalle covariate e non dalla risposta. L'imputazione con media o mediana
è semplice ma comprime la varianza e ignora le correlazioni tra variabili;
alternative più fedeli sono l'imputazione tramite modello (es. $k$-NN o
regressione, `sklearn.impute.KNNImputer`) o l'imputazione multipla.
Regola d'oro: qualsiasi imputazione va **stimata solo sul training set** e poi
applicata al test set, per non far trapelare informazione (*data leakage*).

`````

## Perché guardare i dati prima di modellare

Verrebbe la tentazione di saltare direttamente al modello. È un errore, e c'è
un esempio classico che lo dimostra meglio di mille parole. Nel 1973 lo
statistico Francis Anscombe costruì quattro piccoli insiemi di dati con le
stesse statistiche riassuntive: identica media di $x$ e di $y$, identica
varianza, identica correlazione ($\approx 0{,}816$) e **la stessa retta di
regressione** $\hat{y} = 3 + 0{,}5\,x$. Sulla carta sono indistinguibili. Ma
basta disegnarli ({numref}`fig-anscombe`) per scoprire che raccontano quattro
storie completamente diverse.

```{figure} ../figures/quartetto-anscombe.svg
:name: fig-anscombe
:alt: "Quattro grafici a dispersione con la stessa retta di regressione ma nubi di punti molto diverse: una relazione lineare, una curva, una lineare con un valore anomalo, e una con i punti allineati verticalmente più un punto isolato."
:width: 90%

Il quartetto di Anscombe {cite}`anscombe1973graphs`. Stesse statistiche,
stessa retta: solo il grafico rivela che i quattro dataset non hanno nulla
in comune.
```

Il primo è davvero lineare; il secondo è una curva che una retta descrive
male; il terzo è una retta rovinata da un solo valore anomalo (un *outlier*);
il quarto ha tutti i punti su una verticale, tranne uno che da solo determina
la pendenza. Nessuna
di queste patologie emerge dai numeri riassuntivi: solo l'occhio le coglie.

Matplotlib è lo strumento per farlo. Tre grafici bastano per l'esplorazione
iniziale: la **dispersione** per due variabili, l'**istogramma** per la
distribuzione di una, la **linea** per un andamento nel tempo.

```{figure} ../figures/matplotlib-primi-grafici.svg
:name: fig-anatomia-figura
:alt: "Un grafico Matplotlib annotato con i nomi delle sue parti: la Figure è il foglio che contiene tutto, gli Axes sono l'area di disegno delimitata dai due assi, e su di essi sono marcati il titolo, le etichette degli assi, i tick con i loro valori, la legenda e le linee tracciate."
:width: 96%

I nomi delle parti. La distinzione che serve subito è fra la **Figure**, cioè
il foglio, e gli **Axes**, cioè il riquadro dove si disegna: quasi tutti i
metodi appartengono ai secondi.
```

Conviene imparare i nomi di {numref}`fig-anatomia-figura` prima di scrivere il
primo grafico, perché la documentazione di Matplotlib li dà per noti, e perché
l'errore più comune dei primi tempi (chiamare un metodo sulla Figure quando
serviva sugli Axes) diventa leggibile appena si sa che sono due oggetti
distinti.

```python
import matplotlib.pyplot as plt

plt.scatter(df["eta"], df["spesa"])   # relazione tra due variabili
plt.hist(df["spesa"], bins=20)        # distribuzione di una variabile
plt.plot(mesi, fatturato)             # andamento nel tempo
plt.xlabel("età"); plt.ylabel("spesa")
plt.show()
```

Lo scatter rivela relazioni e valori anomali; l'istogramma mostra se una
variabile è simmetrica, asimmetrica o bimodale (con due "gobbe" invece di
una): cose che una media da sola nasconde. È il modo più economico per non
costruire, sopra dati fraintesi, un modello perfetto nella forma e sbagliato
nella sostanza.

```{admonition} Da ricordare
:class: important
- Il **DataFrame** è la tabella programmabile di Pandas: colonne (`Series`)
  tipizzate, allineate su un indice etichettato.
- Il flusso tipico è **carica** (`read_csv`) → **ispeziona** (`head`, `info`,
  `describe`) → **filtra e trasforma** (maschere booleane, nuove colonne) →
  **aggrega** (`groupby`).
- I **valori mancanti** (`NaN`) vanno gestiti con criterio, imputando solo sul
  training set per evitare *data leakage*.
- **Visualizza prima di modellare**: il quartetto di Anscombe mostra che
  statistiche identiche possono nascondere dati radicalmente diversi.
```
