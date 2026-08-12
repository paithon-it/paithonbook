# Pandas e Matplotlib: dati e visualizzazione

Prima di addestrare qualunque modello c'è un lavoro poco glamour che occupa,
nella pratica, gran parte del tempo di chi fa machine learning: prendere dei
dati grezzi (l'elenco degli ordini sputato fuori dal programma con cui
un'azienda tiene la contabilità, uno storico di vendite, un registro di
sensori) e portarli in una forma pulita, ordinata, esplorabile. In
Python questo lavoro ha due strumenti quasi obbligati: **Pandas** per
manipolare le tabelle e **Matplotlib** per guardarle. Se NumPy è l'algebra,
Pandas è il foglio di calcolo programmabile e Matplotlib è la finestra da cui
osservare cosa abbiamo davvero tra le mani.

## Series e DataFrame: la tabella come oggetto

Pandas ruota attorno a due strutture. Una **Series** è una colonna: una
sequenza di valori con un'etichetta ciascuno (l'*indice*). Un **DataFrame** è
una tabella intera: tante Series affiancate che condividono lo stesso indice
di riga. Attenzione alla parola *indice*, che qui cambia mestiere rispetto alla
pagina su NumPy: là era il numero della posizione (`x[0]`, il primo), qui è
un'etichetta attaccata alla riga, che può benissimo essere una data o un nome
e che resta la stessa anche se le righe si riordinano.

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
Ogni colonna ha un proprio `dtype` omogeneo (`int64`, `float64`, `str`,
`category`, `datetime64`), il che permette a Pandas di appoggiarsi a NumPy per
le operazioni vettoriali colonna per colonna. Sul dtype del testo vale la pena
una precisazione, perché è cambiato di recente e la rete è piena di materiale
che dice il contrario: da **pandas 3.0** una colonna di testo ha dtype `str`,
sostenuto da Arrow quando `pyarrow` è installato, ed è molto più compatto e
veloce del vecchio `object`, in cui ogni cella era un oggetto Python a sé.
`object` esiste ancora, ma è diventato il dtype delle colonne che mescolano
tipi. L'indice non è un semplice
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
df = pd.read_csv("vendite.csv")   # il file va cercato dove sta girando il
                                  # programma: stessa cartella, oppure il
                                  # percorso completo ("dati/vendite.csv")

df.head()        # prime 5 righe: uno sguardo veloce
df.info()        # colonne, dtype, valori non nulli, memoria
df.describe()    # statistiche riassuntive delle colonne numeriche
```

Da qui in avanti `df` è questa tabella caricata da file, non più quella scritta
a mano poche righe fa: il nome è lo stesso perché `df` (da *dataframe*) è il
nome che quasi tutti danno alla tabella su cui stanno lavorando in quel
momento. Nel notebook compagno di questo capitolo il file `vendite.csv` viene
creato all'avvio, così le righe che seguono funzionano davvero.

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

Da qui la regola che evita l'errore più frequente del mestiere: per *leggere*
va bene qualunque forma, per **scrivere** si usa `.loc`. `df[df["eta"] > 30]`
è un oggetto nuovo, quindi `df[df["eta"] > 30]["spesa"] = 0` modifica quello e
lascia `df` com'era (pandas 3 lo segnala con un `ChainedAssignmentError`). La
forma che funziona è una sola, `df.loc[df["eta"] > 30, "spesa"] = 0`, perché
seleziona e assegna in un passo solo. Nota per chi cerca in rete: con il
Copy-on-Write, predefinito da pandas 3, il vecchio `SettingWithCopyWarning` non
esiste più e la copia non scrive **mai** sull'originale, quindi il classico
«a volte funziona» dei tutorial di due anni fa non descrive più niente.

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
colonna, è l'indice. Da lì in poi ci si riferisce a una riga con la sua
etichetta e non con il valore di una colonna, e questo spiega gran parte dei
`KeyError` che seguono: `KeyError` è l'errore con cui Python dice «questo nome
qui dentro non c'è», e chiedere `df["citta"]` dopo un raggruppamento per città
è il modo più rapido di provocarlo.

```python
df.groupby("citta")["spesa"].mean()    # spesa media per città
df.groupby("citta").agg(
    spesa_media=("spesa", "mean"),     # una colonna nuova, che chiamo io
    clienti=("spesa", "count"),        # (da quale colonna, con quale conto)
)
```

Il secondo esempio calcola due riassunti in una volta e dà a ciascuno il nome
che si vuole: a sinistra dell'uguale il nome della colonna che uscirà, a destra
la coppia «da quale colonna prendere i valori, che conto farci sopra».

`````{tab} Elementare

`groupby("citta")` mette in scatole separate tutte le righe di Milano, tutte
quelle di Roma, e così via. Poi `.mean()` calcola la media dentro ogni scatola.
Il risultato è una tabellina con una riga per città: il riassunto che
cercavi. In Excel la stessa cosa si fa con le *tabelle pivot*, trascinando
colonne con il mouse; qui è una riga di codice, che si rilegge e si riesegue.

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

Il terzo pannello di {numref}`fig-dati-mancanti` mostra una via più raffinata:
invece di mettere lo stesso valore dappertutto, si *indovina* quello che manca
guardando le altre colonne della stessa riga (conoscendo età e città di un
cliente si può stimare quanto avrebbe speso). Costa di più, e si fa con un
modello: è materia dei capitoli sul machine learning, qui basta sapere che
esiste.

`````

`````{tab} Superiore

Una precisazione sul contenitore: `NaN` è la rappresentazione dei mancanti
per i `float` (e un solo `NaN` forza a `float64` una colonna di interi, come
si nota da `df.info()`); le colonne di date usano `NaT`, i dtype *nullable* di
Pandas usano `pd.NA`, e il nuovo dtype `str` di pandas 3 continua a usare
`nan`, così `isna()` risponde come sempre.

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
statistico Francis Anscombe costruì quattro piccoli insiemi di dati che,
misurati, si somigliano fino alla seconda cifra decimale. Hanno la stessa
**media** di $x$ e di $y$ (il valore attorno a cui i dati si dispongono); la
stessa **varianza**, cioè lo stesso sparpagliamento attorno a quella media
(piccola se i valori sono tutti lì vicino, grande se sono sparsi); la stessa
**correlazione**, $\approx 0{,}816$, che è un numero fra $-1$ e $1$ e dice
quanto due grandezze crescono insieme ($0$ vuol dire che non si sa niente
dell'una sapendo l'altra); e la stessa **retta di regressione**, cioè la retta
che passa più vicino possibile a tutti i punti:

$$
\hat{y} = 3 + 0{,}5\,x .
$$

Il cappuccio sopra la $y$ è la notazione, che ritroverai in tutto il libro, per
«valore *previsto* dalla retta», da tenere distinto dal valore misurato
davvero. Costruire quattro insiemi di dati che coincidono su tutte e quattro
queste misure è un lavoro di precisione, ed è il punto: sulla carta sono
indistinguibili. Ma basta disegnarli ({numref}`fig-anscombe`) per scoprire che
raccontano quattro storie completamente diverse.

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
distribuzione di una, la **linea** per un andamento nel tempo. («Variabile»,
qui, non è la variabile di Python: in statistica è una grandezza misurata, cioè
una colonna della tabella.)

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
distinti. Nel codice qui sotto non compaiono né l'una né gli altri, e non è una
dimenticanza: le funzioni `plt.qualcosa` lavorano sulla figura *corrente*,
quella aperta in quel momento, il che va benissimo per un grafico veloce.
Quando i grafici diventano due o più, o quando li si vuole affiancare, si
prendono i due oggetti per nome, `fig, ax = plt.subplots()`, e si chiamano i
metodi su `ax` (`ax.scatter(...)`, `ax.set_xlabel(...)`): è la forma che si
trova nella documentazione e nel codice altrui.

```python
import matplotlib.pyplot as plt

mesi = ["gen", "feb", "mar", "apr", "mag", "giu"]
fatturato = [12_000, 13_500, 11_800, 15_200, 16_400, 15_900]

plt.scatter(df["eta"], df["spesa"])   # relazione tra due variabili
plt.xlabel("età")
plt.ylabel("spesa")
plt.show()                            # mostra il grafico e chiude questo foglio

plt.hist(df["spesa"], bins=20)        # distribuzione: 20 barre ("bins")
plt.show()

plt.plot(mesi, fatturato)             # andamento nel tempo
plt.show()
```

`plt.show()` non è decorativo: senza, i tre grafici finirebbero uno sopra
l'altro sullo stesso foglio, con le etichette del primo appiccicate agli altri
due. È il modo di dire «questo è finito». Quanto ai *bins* dell'istogramma,
sono le barre in cui l'intervallo dei valori viene diviso: cambiarne il numero
cambia il disegno, e vale la pena provarne due o tre, perché troppo poche
nascondono la forma e troppe la sbriciolano.

Lo scatter rivela relazioni e valori anomali; l'istogramma mostra se una
variabile è simmetrica, asimmetrica o bimodale (con due "gobbe" invece di
una): cose che una media da sola nasconde. È il modo più economico per non
costruire, sopra dati fraintesi, un modello perfetto nella forma e sbagliato
nella sostanza.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il **DataFrame** è il foglio di calcolo programmabile di Pandas: ogni colonna
  tiene valori di un solo tipo, e tutte le colonne condividono le stesse
  etichette di riga.
- L'ordine di lavoro è sempre lo stesso: **carichi** (`read_csv`), **guardi**
  (`head`, `info`, `describe`), **filtri** con il colino di una condizione e
  aggiungi colonne calcolate, **raggruppi** (`groupby`) per avere un riassunto
  per categoria.
- Una casella vuota (`NaN`) è essa stessa un'informazione: prima di buttarla o
  di riempirla, chiediti *perché* manca. E se la riempi con un valore inventato
  a partire dai dati, quel valore va calcolato solo sui dati con cui il modello
  impara, mai su quelli con cui lo si giudica: altrimenti stai facendo copiare
  il modello durante l'esame.
- **Guarda i dati prima di modellare**: il quartetto di Anscombe mostra che
  quattro insiemi di dati con gli stessi numeri riassuntivi possono essere
  completamente diversi, e che a vederlo è l'occhio, non la media.
```

`````

`````{tab} Superiore

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

`````
