# Pandas e Matplotlib: dati e visualizzazione

Prima di addestrare un modello c'è il lavoro sui dati, che nella pratica
occupa gran parte del tempo di chi fa machine learning: prendere dati grezzi
(l'esportazione degli ordini dal gestionale di un'azienda, uno storico di
vendite, il registro di alcuni sensori) e portarli in una forma pulita,
ordinata, esplorabile. In Python gli strumenti quasi obbligati sono due:
**Pandas** per manipolare le tabelle e **Matplotlib** per disegnarle. NumPy dà
l'array; Pandas ci mette sopra le etichette e le colonne, come un foglio di
calcolo programmabile; Matplotlib mostra i dati in un grafico.

## Series e DataFrame: la tabella come oggetto

Pandas ruota attorno a due strutture. Una **Series** è una colonna: una
sequenza di valori con un'etichetta ciascuno (l’*indice*). Un **DataFrame** è
una tabella intera: tante Series affiancate che condividono lo stesso indice
di riga. Attenzione alla parola *indice*, che qui cambia mestiere rispetto a
NumPy: là era il numero della posizione (`x[0]`, il primo), qui è
un'etichetta attaccata alla riga, che può benissimo essere una data o un nome
e che resta la stessa anche se le righe si riordinano.

```{figure} ../figures/pandas-series-dataframe.svg
:name: fig-series-dataframe
:alt: "A sinistra un DataFrame con sei righe e quattro colonne: nome, eta, citta, spesa, per Ada, Bruno, Carla, Dario, Elena e Furio. Sul fianco sinistro, in terracotta, la fascia delle etichette di riga, 0, 1, 2, 3, 4, 5, che pandas ha messo da sé perché il file non ne portava. Due caselle sono NaN: l'età di Bruno e la spesa di Dario. Le età si leggono 34.0, 41.0, 36.0, 52.0, 23.0, in virgola mobile. La colonna della spesa è tinta di ocra. Una freccia etichettata «si estrae la colonna» e df tra quadre spesa la porta a destra, dove la stessa colonna compare da sola come Series: sei valori, 120.5, 89.0, 240.0, NaN, 310.0, 74.9, e accanto a ciascuno la stessa etichetta di riga che aveva nella tabella, da 0 a 5."
:width: 94%

Una colonna staccata da un DataFrame è una Series, e si porta dietro
l'indice. È quell'indice condiviso a permettere di riallineare i dati senza
badare all'ordine delle righe.
```

La parte da fissare in {numref}`fig-series-dataframe` è la fascia di
etichette sul fianco sinistro, che colonna non è. L'indice fa da etichetta:
pandas riconosce ogni riga da lì, e quell'etichetta resta attaccata ai dati
quando si filtra, si ordina o si estrae una colonna.

`````{tab} Elementare

Un DataFrame è un foglio Excel fatto di codice. Ogni colonna ha
un'intestazione (`nome`, `eta`, `citta`, `spesa`, senza accenti, come si
scrivono di solito i nomi delle colonne) e ogni riga è un cliente; solo che
invece di cliccare con il mouse dai istruzioni a parole:

```python
import pandas as pd

df = pd.DataFrame({
    "nome":  ["Ada", "Carla", "Furio"],
    "eta":   [34, 41, 23],
    "citta": ["Milano", "Milano", "Torino"],
    "spesa": [120.5, 240.0, 74.9],
})
```

Quello fra le graffe è un dizionario, lo stesso delle basi del linguaggio: le
chiavi diventano i nomi delle colonne, e il valore di ciascuna è la lista dei
dati di quella colonna, dall'alto in basso. Ogni colonna è una Series; tutte
insieme formano la tabella. E dentro una colonna i valori sono di norma tutti
della stessa specie, numeri con numeri e testo con testo: è la stessa regola
dell'array di NumPy, applicata una colonna per volta, ed è quello che permette
a Pandas di affidare i conti a NumPy. Il vantaggio rispetto a Excel è che ogni
operazione è ripetibile e documentata: la scrivi una volta e la riesegui su un
milione di righe senza cambiare nulla.

L'altra differenza da Excel sta nelle etichette di riga. Pandas ragiona con
quelle: se accosti due elenchi di clienti scritti in ordine diverso, le righe
si appaiano per etichetta, come quando si confrontano due registri cercando lo
stesso nome su entrambi. E se un nome compare in un registro solo, nella
casella accanto resta un buco, non il dato di un altro cliente: un buco si
vede subito, uno scambio di persona no.

`````

`````{tab} Superiore

Un `DataFrame` è una collezione di `Series` allineate su un `Index` comune. Ogni
colonna ha un proprio `dtype` omogeneo (`int64`, `float64`, `str`, `category`,
`datetime64`), il che permette a Pandas di appoggiarsi a NumPy per le operazioni
vettoriali colonna per colonna. Il dtype del testo è cambiato di recente, e la
rete è piena di materiale che descrive ancora quello vecchio: da pandas 3.0 una
colonna di testo ha dtype `str`, che con `pyarrow` installato è sostenuto da
Arrow ed è molto più compatto e veloce del vecchio `object`, in cui ogni cella
era un oggetto Python a sé (senza `pyarrow` il dtype è lo stesso, ma i dati
restano oggetti Python). `object` esiste ancora e resta il dtype delle colonne
che mescolano tipi; non è più quello del testo. L'indice è una struttura
etichettata (anche gerarchica, `MultiIndex`) usata per l'allineamento
automatico, e il numero di riga ne è solo il caso più semplice. Quando sommi due
Series, Pandas non allinea per posizione ma per etichetta, inserendo `NaN` dove
le etichette non combaciano: comportamento che evita i disallineamenti tipici
degli array grezzi, dove due vettori si sommano per posizione anche quando le
righe non si corrispondono.

`````

## Caricare e ispezionare i dati

Nella realtà i dati non li digiti a mano: li carichi. Il formato più comune è
il **CSV** (un semplice file di testo con i valori separati da virgole, il
formato in cui quasi ogni programma sa esportare una tabella) e la funzione
`read_csv` lo legge in una riga, riconoscendo da sola tipi e intestazioni. Un
avviso da tastiera italiana: il CSV che Excel produce qui da noi usa spesso il
punto e virgola come separatore e la virgola per i decimali, e si legge
dichiarandolo, `read_csv("file.csv", sep=";", decimal=",")`. Se il file ha
lettere accentate ed è stato salvato come «CSV» semplice (non «CSV UTF-8»), è
in codifica Windows, e serve anche `encoding="cp1252"`: senza, `read_csv` si
ferma con un `UnicodeDecodeError`.

Un file su cui provare ce lo fabbrichiamo al volo, così ogni numero che segue
si può rifare:

```python
import pandas as pd

pd.DataFrame({
    "nome":  ["Ada", "Bruno", "Carla", "Dario", "Elena", "Furio"],
    "eta":   [34, None, 41, 36, 52, 23],
    "citta": ["Milano", "Torino", "Milano", "Napoli", "Milano", "Torino"],
    "spesa": [120.5, 89.0, 240.0, None, 310.0, 74.9],
}).to_csv("vendite.csv", index=False)  # index=False: le etichette di riga
                                       # qui sono 0, 1, 2..., e nel file non
                                       # servono
```

Il blocco che segue la rilegge da lì, e da quel punto in avanti `df` è questa
tabella caricata da file, e non più una scritta a mano: il nome è lo stesso
perché `df` (da *dataframe*) è il nome che quasi tutti danno alla tabella su
cui stanno lavorando in quel momento. La tabella su cui girano le righe che
seguono ha sei clienti e quattro colonne (`nome`, `eta`, `citta`, `spesa`),
con un paio di caselle lasciate vuote di proposito, perché i dati veri sono
quasi sempre così.

Tre comandi bastano per il primo sguardo, e il primo è `head()`.

```python
df = pd.read_csv("vendite.csv")   # il file va cercato dove sta girando il
                                  # programma: stessa cartella, oppure il
                                  # percorso completo ("dati/vendite.csv")

print(df.head())   # prime 5 righe: uno sguardo veloce
```

```text
    nome   eta   citta  spesa
0    Ada  34.0  Milano  120.5
1  Bruno   NaN  Torino   89.0
2  Carla  41.0  Milano  240.0
3  Dario  36.0  Napoli    NaN
4  Elena  52.0  Milano  310.0
```

Da leggere ci sono due cose oltre ai dati. La colonna senza intestazione a
sinistra, con 0, 1, 2, 3, 4, è l’indice: le etichette di riga di cui si
parlava poco fa, che qui pandas ha messo da sé perché il file non ne aveva. E
quei due `NaN` sono le caselle vuote. Sono anche il motivo per cui l'età
compare come 34.0 invece che come 34: una casella vuota non è un numero
intero, e per tenerla in colonna insieme agli altri pandas passa tutta la
colonna ai numeri con la virgola.

Gli altri due riassumono la tabella: `info()` elenca le colonne con il loro
tipo e quante caselle sono piene, `describe()` calcola le statistiche delle
colonne numeriche. Il tipo di ogni colonna si legge anche da solo:

```python
print(df.dtypes)
```

```text
nome         str
eta      float64
citta        str
spesa    float64
dtype: object
```

Con `info()` e `describe()` il primo sguardo è completo:

```python
df.info()        # colonne, tipo, quante caselle sono piene, memoria
df.describe()    # media, deviazione standard, minimo, massimo e quartili
```

Questi tre metodi aprono quasi ogni analisi. `head()` mostra che aspetto hanno
i dati; `info()` dice quanti sono e dove ci sono buchi (valori mancanti);
`describe()` dà, per ogni colonna numerica, il conteggio, la media, la
deviazione standard (quanto i valori si sparpagliano attorno alla media), il
minimo, il massimo e i quartili (i valori che dividono i dati in quattro fette
uguali). Prima di costruire un modello bastano a scoprire colonne vuote, tipi
sbagliati e valori impossibili.

## Selezionare, filtrare, creare colonne

Una volta caricata la tabella, la si interroga. Selezionare una colonna,
tenere solo le righe che soddisfano una condizione, calcolare una nuova
colonna a partire dalle altre: sono le tre operazioni che si ripetono
all'infinito. Nell'esempio la colonna nuova è la spesa con l'IVA, l'imposta
del 22 per cento: moltiplicare per 1,22 la aggiunge.

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
Puoi combinarne più d'una con `&` («e») e `|` («o»), e ogni condizione va
chiusa fra parentesi sue, altrimenti Python legge la riga in un altro modo e
risponde con un errore:

```python
df[(df["eta"] > 30) & (df["citta"] == "Milano")]
```

Quello che passa dal colino finisce in una ciotola a parte, e la ciotola non è
la pentola. Il filtro fa lo stesso: la tabella filtrata è una copia. Se scrivi
lì dentro, la tabella di partenza resta com'era, il programma non si ferma, e
la correzione riesce sul recipiente sbagliato. La riga che fa questo guaio è

```python
df[df["eta"] > 30]["spesa"] = 0       # scrive sulla ciotola, non sulla pentola
```

In NumPy la riga `x[x > 25] = 0`, con una sola coppia di quadre, scriveva
sull'originale; qui le coppie di quadre sono due, una dopo l'altra: la prima
fabbrica la ciotola, la seconda ci scrive dentro. Pandas la segnala con un
avviso, che però in mezzo a mille righe di uscita
non lo legge nessuno. Per scrivere sulla tabella originale c'è un attrezzo
apposta, `.loc`, che sceglie le righe e le cambia lì dove stanno. Fra le sue
quadre si scrive prima la condizione sulle righe e poi il nome della colonna:
`df.loc[df["eta"] > 30, "spesa"] = 0` azzera la spesa di chi ha più di
trent'anni sulla tabella vera, e non su una ciotola a parte.

`````

`````{tab} Superiore

Il filtro booleano è *boolean masking*: la Series di condizione è un vettore
di `bool` che indicizza il DataFrame, esattamente come in NumPy. Le condizioni
si combinano con gli operatori bit a bit `&`, `|`, `~` (non con `and`/`or`
Python, che non sono vettorizzati), e le parentesi sono obbligatorie per via
della precedenza degli operatori. Per selezioni miste per etichetta e
posizione esistono gli accessor `.loc[righe, colonne]` (per etichetta) e
`.iloc[...]` (per posizione intera), che restano il modo canonico e non
ambiguo di indicizzare. Con una differenza rispetto alle fette di Python e
NumPy: nelle fette per etichetta di `.loc` il secondo estremo è incluso
(`df.loc[0:2]` restituisce tre righe), mentre `.iloc[0:2]`, per posizione, ne
restituisce due.

Da qui la regola che evita l'errore più frequente del mestiere: per *leggere*
va bene qualunque forma, per scrivere si usa `.loc`. `df[df["eta"] > 30]` è un
oggetto nuovo, quindi `df[df["eta"] > 30]["spesa"] = 0` modifica quello e
lascia `df` com'era. Pandas 3 lo segnala con un `ChainedAssignmentError` che,
malgrado il nome, viene *emesso* come avviso e non *sollevato* come errore: il
programma non si ferma, tira dritto, e la modifica che credevi di aver fatto
semplicemente non c'è. In uno script che filtra gli avvisi, o in un notebook
con mille righe di output, il gesto sbagliato passa in silenzio. Chi vuole che
si fermi lo può promuovere a errore vero con `warnings.simplefilter("error",
pd.errors.ChainedAssignmentError)`, che è una riga da mettere in cima a uno
script che tratta dati veri. La forma che funziona è una sola,
`df.loc[df["eta"] > 30, "spesa"] = 0`, perché seleziona e assegna in un passo
solo. Nota per chi cerca in rete: con il Copy-on-Write, predefinito da pandas
3, il vecchio `SettingWithCopyWarning` non esiste più e la copia non scrive mai
sull'originale, quindi il classico «a volte funziona» dei tutorial di due anni
fa non descrive più niente. Con il Copy-on-Write ogni oggetto derivato si
comporta come una copia indipendente, e i dati si copiano davvero solo quando
qualcuno scrive. Per lo stesso motivo i metodi di Pandas restituiscono un
oggetto nuovo e lasciano quello di partenza com'è: `df.dropna()` e
`df["eta"].fillna(...)` non cambiano `df` finché il risultato non viene
assegnato (`df = df.dropna()`).

`````

## Raggruppare e aggregare

La domanda che quasi ogni analisi finisce per porsi è: *quanto vale questa
grandezza, suddivisa per categoria?* Spesa media per città, numero di ordini
per mese, errore medio per classe. È il pattern **split-apply-combine**: dividi
i dati in gruppi, applichi una funzione a ciascuno, ricomponi il risultato.

```{figure} ../figures/pandas-selezione-filtri-groupby.svg
:name: fig-split-apply-combine
:alt: "Le tre mosse in fila, da sinistra a destra. A sinistra la tabella di sei clienti, una riga per ciascuno, con la città e la spesa: Milano 120.5, Torino 89.0, Milano 240.0, Napoli NaN, Milano 310.0, Torino 74.9, e ogni riga tinta del colore della sua città. Tre frecce la dividono in tre gruppi: Milano con tre righe, Napoli con una sola, che porta NaN, Torino con due. Su ogni gruppo una freccia etichettata «la media» lo riduce a un numero: 223.50 per Milano, NaN per Napoli, 81.95 per Torino. Tre frecce ricompongono i tre numeri in una tabella finale di tre righe, dove la città non è più una colonna ma l'etichetta di riga."
:width: 100%

Le tre mosse in fila. La tabella finale ha una riga per gruppo, e la colonna
su cui si è diviso è diventata il suo indice.
```

```python
print(df.groupby("citta")["spesa"].mean())    # spesa media per città
```

```text
citta
Milano    223.50
Napoli       NaN
Torino     81.95
Name: spesa, dtype: float64
```

```python
df.groupby("citta").agg(
    spesa_media=("spesa", "mean"),     # una colonna nuova, che chiamo io
    clienti=("nome", "count"),         # (da quale colonna, con quale conto)
)
```

L'ultimo passaggio di {numref}`fig-split-apply-combine` è quello che si tende
a dimenticare: dopo un `groupby` la colonna di raggruppamento diventa
l'indice, e smette di essere una colonna. Da lì in poi una riga si chiama con
la sua etichetta, non con il valore di una colonna. È questo a spiegare gran
parte dei `KeyError` che arrivano dopo un raggruppamento, dove `KeyError` è
l'errore con cui Python dice «questo nome qui dentro non c'è»: chiedere la
colonna `"citta"` al risultato di un raggruppamento per città è il modo più
rapido di provocarlo. Sulla tabella di partenza, che il raggruppamento non
tocca, quella colonna c'è ancora. Per tenere la città come colonna nel
risultato si chiede `as_index=False` (`df.groupby("citta",
as_index=False)["spesa"].mean()`), oppure si richiama `.reset_index()` sul
risultato.

La prima riga è la più lunga catena di punti e quadre vista finora, e si legge
da sinistra a destra come una frase, un pezzo per volta: «prendi `df`,
raggruppalo per città, di quel che esce tieni la colonna `spesa`, e di quella
fai la media». Ogni pezzo lavora su ciò che ha prodotto il pezzo precedente, e
questo modo di incatenare le operazioni è lo stile normale di pandas.

Il secondo esempio calcola due riassunti in una volta e dà a ciascuno il nome
che si vuole: a sinistra dell'uguale il nome della colonna che uscirà, a destra
la coppia «da quale colonna prendere i valori, che conto farci sopra».

Le tre città escono in ordine alfabetico, e non nell'ordine in cui compaiono
nel file: `groupby` ordina le chiavi, a meno che non gli si dica `sort=False`.

Napoli risponde `NaN`. Le funzioni di riassunto di pandas saltano le caselle
vuote: una colonna con due numeri e un buco fa la media dei due. Ma il suo
unico cliente ha la spesa mancante, e una media senza nemmeno un valore da
mediare non esiste. Le caselle vuote meritano una sezione loro.

`````{tab} Elementare

`groupby("citta")` mette in scatole separate tutte le righe di Milano, tutte
quelle di Torino, e così via. Poi `.mean()` calcola la media dentro ogni
scatola. Il risultato è una tabellina con una riga per città: il riassunto che
cercavi. In Excel la stessa cosa si fa con le *tabelle pivot*, trascinando
colonne con il mouse; qui è una riga di codice, che si rilegge e si riesegue.

`````

`````{tab} Superiore

Concettualmente `groupby` partiziona le righe secondo una o più chiavi e
applica a ogni gruppo $g$ una funzione di aggregazione. Per la media, sul
gruppo con valori $\{x_1,\dots,x_{n_g}\}$:

$$
\bar{x}_g = \frac{1}{n_g}\sum_{i=1}^{n_g} x_i ,
$$

dove $n_g$ conta i valori presenti nel gruppo, e non le sue righe: con
`skipna=True`, che è il default, `mean` scarta i mancanti prima di sommare e
prima di dividere. Su un gruppo che non ha nemmeno un valore la somma vale zero
e il divisore pure, ed è quello zero diviso zero a dare il `NaN` di Napoli.
Oltre a `mean` sono disponibili `sum`, `count`, `std`, `min`, `max`, `median` e
funzioni arbitrarie via `agg`/`apply`. `std` di pandas divide per $n_g - 1$
(`ddof=1`, come `describe()`), mentre `np.std` divide per $n_g$ (`ddof=0`): le
due non coincidono sugli stessi dati. E le aggregazioni nominate con una
stringa (`"mean"`, `"sum"`) girano in codice compilato, mentre una funzione
Python passata ad `apply` viene chiamata una volta per gruppo, o per riga con
`axis=1`, e rimette nel ciclo l'interprete che la vettorizzazione aveva
tolto. Il metodo `agg` con argomenti nominati
(*named aggregation*) produce colonne dal nome esplicito, rendendo il risultato
pronto per un report o per un incrocio con un'altra tabella (il `merge`, il
parente pandas della `JOIN` dei database).

L'incrocio appaia le righe di due tabelle che hanno lo stesso valore in una
colonna, e `how` decide che cosa fare di quelle senza corrispondenza: `inner`
le scarta da tutte e due le parti, `left` tiene tutte le righe della tabella
di sinistra e riempie con `NaN` le colonne dell'altra, `right` fa il
contrario, `outer` tiene tutto. Con una tabella di regioni che non conosce
Napoli e conosce Roma:

```python
regioni = pd.DataFrame({"citta": ["Milano", "Torino", "Roma"],
                        "regione": ["Lombardia", "Piemonte", "Lazio"]})

for how in ["inner", "left", "right", "outer"]:
    print(how, len(df.merge(regioni, on="citta", how=how)))
```

```text
inner 5
left 6
right 6
outer 7
```

Con `inner` Dario, di Napoli, sparisce; con `left` resta, con la regione a
`NaN`; con `right` entra Roma, con il nome del cliente a `NaN`; `outer` ha sia
l'uno sia l'altro.

`````

## I valori mancanti

I dati reali sono quasi sempre incompleti: un campo non compilato, un sensore
spento, una risposta saltata. Pandas rappresenta questi buchi con `NaN` (*Not a
Number*), e ignorarli non è un'opzione. Le funzioni di riassunto di pandas, si
è visto, le caselle vuote le saltano; ma un conto fatto casella per casella
(sommare due colonne, moltiplicare per un prezzo) il buco se lo porta dietro:
dove c'era una casella vuota il risultato è di nuovo vuoto, e ogni colonna che
nasce da quella si ritrova lo stesso buco.

```{figure} ../figures/gestire-dati-mancanti.svg
:name: fig-dati-mancanti
:alt: "La stessa tabella di cinque righe e tre colonne (eta, acquisti, spesa), con due caselle vuote, gli acquisti della seconda riga e la spesa della terza, trattata in tre modi affiancati. Nel primo le due righe incomplete sono barrate: con loro se ne vanno 4 numeri buoni e restano tre righe su cinque. Nel secondo i due buchi sono riempiti con la media della loro colonna, 4,75 acquisti e 159,25 di spesa, uguale per chiunque. Nel terzo il valore si ricostruisce dalle altre colonne della stessa riga, e due frecce tratteggiate orizzontali entrano in ciascun buco partendo dalle celle che gli stanno accanto: chi ha fatto nove acquisti prende 347 di spesa invece di 159,25, e chi ha speso 240 prende 6,2 acquisti invece di 4,75. In fondo, la regola d'oro: il valore con cui si riempie si calcola solo sui dati con cui il modello impara."
:width: 100%

Tre modi di rispondere alla stessa cella vuota, su una tabella che accanto
alla spesa ha il numero di acquisti. Nessuno è neutro: il primo,
per due caselle mancanti, butta via anche quattro numeri buoni; il secondo
mette lo stesso valore in ogni buco, e la colonna si sparpaglia meno di prima;
il terzo guarda le altre colonne della stessa riga, e a chi ha fatto nove
acquisti dà 347 di spesa invece dei 159,25 della media.
```

Nei tre modi di {numref}`fig-dati-mancanti` resta fuori una cosa: che una
casella vuota è essa stessa un'informazione. Se manca perché il sensore era
spento, è un
caso; se manca perché la domanda era imbarazzante, il fatto che manchi dice
qualcosa, e riempirla con la media cancella proprio quel qualcosa.

`````{tab} Elementare

Hai due strade. Puoi buttare via le righe incomplete, oppure riempirle
con un valore ragionevole, la media della colonna o la sua mediana (il valore
che sta in mezzo quando li si mette in fila):

```python
df.isna().sum()              # quanti buchi per colonna?
df.dropna()                  # elimina le righe con valori mancanti
df["eta"].fillna(df["eta"].median())   # riempi con la mediana
```

Quale delle due strade prendere dipende da *perché* quel dato manca, e non
esiste una risposta valida sempre. La quantità dà solo un'indicazione grossa:
con il 2% dei dati mancante scartare le righe costa poco, con il 40% di una
colonna mancante buttarla via distruggerebbe informazione. Ma è la ragione
della mancanza a decidere, e la percentuale da sola non l'ha mai decisa.

Riempire ha comunque un prezzo: mettere in tanti buchi lo stesso numero rende
i dati più uniformi del vero. In una classe dove agli assenti di un compito si
assegna il voto medio dei presenti, la media non cambia, ma i voti sembrano
più simili fra loro di quanto siano.

Il terzo pannello di {numref}`fig-dati-mancanti` mostra una via più raffinata:
invece di mettere lo stesso valore dappertutto, si *indovina* quello che manca
guardando le altre colonne della stessa riga (conoscendo età e città di un
cliente si può stimare quanto avrebbe speso). Costa di più, e si fa con un
modello: è materia dei {doc}`capitoli sul machine learning
</MachineLearning/overview>`, qui basta sapere che
esiste.

C'è però una cautela che conviene conoscere fin d'ora, perché riguarda il
*quando* e non il *come*. Quando si costruisce un modello, i dati si dividono
in due mucchi: uno con cui il modello impara e uno, tenuto da parte e mai
guardato, con cui alla fine lo si giudica. È l'unico modo di sapere se ha
imparato davvero o se ha soltanto imparato a memoria gli esempi che gli
abbiamo dato. E allora anche la mediana con cui riempi i buchi va calcolata
solo sul primo mucchio: se la calcoli su tutti i dati, un pezzetto di
quello che il modello dovrà indovinare gli è già passato sotto gli occhi, e il
voto d'esame diventa più alto di quanto meriti. Il nome tecnico di questo
guaio, che ritroverai spesso, è *data leakage*.

`````

`````{tab} Superiore

Una precisazione sul contenitore: `NaN` è la rappresentazione dei mancanti
per i `float` (e un solo `NaN` forza a `float64` una colonna di interi, come
si nota da `df.dtypes`); le colonne di date usano `NaT`, i dtype *nullable* di
Pandas usano `pd.NA`, e il nuovo dtype `str` di pandas 3 continua a usare
`nan`, così `isna()` risponde come sempre.

La strategia dipende dal meccanismo di mancanza (MCAR, MAR, MNAR, nella
tradizione che nasce con Rubin {cite}`rubin1976inference` e prende questa
forma a tre nei lavori successivi), cioè da che cosa regge la probabilità che
un valore manchi. In MCAR (*missing completely at random*) non dipende né dai
valori osservati né da quelli mancanti: un sensore che salta qualche lettura
per un guasto casuale. In MAR (*missing at random*) dipende solo da quanto è
stato osservato: i sensori più vecchi, di cui l'età è registrata, saltano più
letture. In MNAR (*missing not at random*) dipende dal valore mancante stesso:
chi guadagna di più dichiara meno spesso il reddito. Se i dati sono MCAR è
garantito che eliminare le righe incomplete non introduca distorsioni (a costo
di perdere informazione), e in una regressione l'eliminazione resta lecita
anche quando la mancanza dipende solo dalle covariate e non dalla risposta.
Con MAR l'eliminazione può distorcere le stime, mentre l'imputazione multipla
e i metodi di massima verosimiglianza, se il modello è scritto bene, restano
validi; con MNAR servono ipotesi esplicite sul meccanismo, perché i dati da
soli non lo rivelano. L'imputazione con media o mediana è
semplice ma comprime la varianza e ignora le correlazioni tra variabili;
alternative più fedeli sono l'imputazione tramite modello (es. $k$-NN o
regressione, `sklearn.impute.KNNImputer`) o l'imputazione multipla. Regola
d'oro: qualsiasi imputazione va stimata solo sul training set e poi applicata
al test set, per non far trapelare informazione (*data leakage*).

`````

## Perché guardare i dati prima di modellare

Verrebbe la tentazione di saltare direttamente al modello, e c'è un esempio
famoso che spiega perché sia una cattiva idea. Nel 1973 lo statistico Francis
Anscombe mise insieme quattro piccole raccolte di undici punti ciascuna, e le
costruì apposta perché, misurate, risultassero gemelle: alcune misure
coincidono esatte, le altre a meno di qualche millesimo.

Le misure su cui risultano gemelle sono le stesse che si prendono davanti a
qualunque tabella nuova, e si guardano una per volta. Si rifanno in poche
righe (i dati sono quelli pubblicati da Anscombe nel 1973, con undici punti
per insieme):

```python
import numpy as np

x = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
x4 = np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8])
y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]

print("      media x  var x  media y   var y       r   retta")
for nome, (a, b) in {"I": (x, y1), "II": (x, y2),
                     "III": (x, y3), "IV": (x4, y4)}.items():
    b = np.array(b)
    pendenza, intercetta = np.polyfit(a, b, 1)
    r = np.corrcoef(a, b)[0, 1]
    print(f"{nome:>3}  {a.mean():7.2f} {a.var(ddof=1):6.2f} "
          f"{b.mean():8.4f} {b.var(ddof=1):7.4f} {r:7.4f}   "
          f"y = {intercetta:.3f} + {pendenza:.3f} x")
```

```text
      media x  var x  media y   var y       r   retta
  I     9.00  11.00   7.5009  4.1273  0.8164   y = 3.000 + 0.500 x
 II     9.00  11.00   7.5009  4.1276  0.8162   y = 3.001 + 0.500 x
III     9.00  11.00   7.5000  4.1226  0.8163   y = 3.002 + 0.500 x
 IV     9.00  11.00   7.5009  4.1232  0.8165   y = 3.002 + 0.500 x
```

La **media** è il valore attorno a cui i numeri si dispongono: si sommano e si
divide per quanti sono. In orizzontale ($x$) vale $9$ in tutte e quattro;
in verticale ($y$) vale $7{,}50$, e a separarle è il quarto decimale.

La **varianza** misura lo sparpagliamento attorno a quella media: piccola se i
valori stanno tutti lì vicino, grande se sono sparsi ai due estremi. In
orizzontale è identica in tutte e quattro; in verticale le quattro differiscono
al terzo decimale, che è già più in là di dove si guarda.

La **correlazione** (di Pearson) è un numero fra $-1$ e $1$ che misura quanto
le due grandezze crescono insieme lungo una retta; vale $0$ quando non c'è
nessuna tendenza lineare, il che non esclude legami di altra forma, e i
quattro disegni stanno per mostrarlo. Nelle quattro raccolte vale $0{,}816$
fino alla terza cifra decimale.

La **retta di regressione**, infine, è quella che passa più vicino possibile a
tutti i punti insieme. Ed è la stessa retta:

$$
\hat{y} = 3 + 0{,}5\,x .
$$

Il cappuccio sopra la $y$ vuol dire «valore *previsto* dalla retta», da tenere
distinto dal valore misurato davvero, e la distinzione fra i due è quella su
cui si misura ogni modello supervisionato. Un conto solo basta a fissare
l'idea:
nel primo insieme, dove $x$ vale $10$, la retta prevede
$\hat{y} = 3 + 0{,}5 \cdot 10 = 8$, mentre il punto misurato in quel posto sta
a $8{,}04$. La differenza fra i due, qui quattro centesimi, è l’**errore** su
quel punto, ed è la quantità che ogni modello cercherà di rendere piccola.

Costruire quattro insiemi di dati che coincidono su tutte e quattro queste
misure è un lavoro di precisione, ed è il punto: sono fabbricati apposta
{cite}`anscombe1973graphs`, e sulla carta restano indistinguibili. Ma basta
disegnarli ({numref}`fig-anscombe`) per scoprire che raccontano quattro storie
completamente diverse.

```{figure} ../figures/quartetto-anscombe.svg
:name: fig-anscombe
:alt: "Quattro grafici a dispersione con la stessa retta di regressione ma nubi di punti molto diverse: una relazione lineare, una curva, una lineare con un valore anomalo, e una con i punti allineati verticalmente più un punto isolato."
:width: 90%

Il quartetto di Anscombe. Le stesse statistiche a meno di qualche millesimo,
la stessa
retta: solo il grafico rivela che i quattro insiemi di dati non hanno nulla in
comune.
```

Il primo è davvero lineare; il secondo è una curva che una retta descrive
male; il terzo è una retta rovinata da un solo valore anomalo (un *outlier*);
il quarto ha tutti i punti su una verticale, tranne uno che da solo determina
la pendenza. Nessuna
di queste patologie emerge dai numeri riassuntivi: solo l'occhio le coglie.

## Matplotlib: tre grafici per guardare i dati

Per disegnare i dati serve Matplotlib. Tre grafici bastano per l'esplorazione
iniziale: la dispersione per due variabili, l’istogramma per la
distribuzione di una, la linea per un andamento nel tempo. («Variabile»,
qui, non è la variabile di Python: in statistica è una grandezza misurata, cioè
una colonna della tabella.)

```{figure} ../figures/matplotlib-anatomia-figura.svg
:name: fig-anatomia-figura
:alt: "Un grafico Matplotlib annotato con i nomi delle sue parti. Il grafico disegna il fatturato dei primi sei mesi, da gennaio a giugno, con una linea che sale, scende a marzo e risale fino a maggio. Un riquadro tratteggiato che racchiude tutto è la Figure, cioè il foglio; il riquadro pieno delimitato dai due assi sono gli Axes, cioè l'area di disegno. Cinque didascalie collegate da tratteggi indicano il titolo (ax.set_title), la legenda (ax.legend), la linea dei dati (ax.plot), le tacche degli assi con i loro numeri, e le etichette degli assi (ax.set_xlabel e ax.set_ylabel). In fondo: una Figure può contenere più Axes, e quasi tutti i metodi appartengono agli Axes."
:width: 96%

I nomi delle parti di un grafico, sull'esempio del fatturato dei primi sei
mesi.
La distinzione che serve subito è fra la Figure, cioè
il foglio, e gli Axes, cioè il riquadro dove si disegna: quasi tutti i
metodi appartengono ai secondi.
```

I nomi di {numref}`fig-anatomia-figura` si imparano prima di scrivere il primo
grafico, perché la documentazione di Matplotlib li dà per noti, e perché
l'errore più comune dei primi tempi (chiamare un metodo sulla Figure quando
serviva sugli Axes) diventa leggibile appena si sa che sono due oggetti
distinti. Nelle righe `plt.qualcosa` non compaiono né l'una né gli altri, ed è
voluto: quelle funzioni lavorano sulla figura *corrente*,
quella aperta in quel momento, il che va benissimo per un grafico veloce.
Quando i grafici diventano due o più, o quando li si vuole affiancare, si
prendono i due oggetti per nome e si chiamano i metodi su `ax`, come vedremo
subito dopo.

```python
import matplotlib.pyplot as plt
import numpy as np

mesi = ["gen", "feb", "mar", "apr", "mag", "giu"]
fatturato = [12_000, 13_500, 11_800, 15_200, 16_400, 15_900]

plt.scatter(df["eta"], df["spesa"])   # relazione tra due variabili
plt.xlabel("età")
plt.ylabel("spesa")
plt.show()                            # mostra quello che c'è sul foglio

# per un istogramma servono molti valori: con i sei della tabella si vedrebbero
# sei stecchi e nessuna forma, quindi qui ne fabbrichiamo trecento finti
spese = np.random.default_rng(0).normal(120, 30, size=300)
plt.figure()                          # foglio nuovo, o si disegna sul primo
plt.hist(spese, bins=20)              # distribuzione: 20 barre ("bins")
plt.show()

plt.figure()
plt.plot(mesi, fatturato)             # andamento nel tempo
plt.show()
```

Le due righe fanno mestieri diversi. `plt.figure()` apre un foglio nuovo: qui
non sarebbe indispensabile, perché `plt.show()` chiude i fogli che ha appena
mostrato, ma serve ogni volta che si preparano più grafici prima di mostrarli,
altrimenti i disegni finiscono tutti sullo stesso foglio. `plt.show()` mostra
i fogli aperti: in uno script fa comparire la finestra, in un notebook la
cella mostra il grafico da sola, e la riga si scrive lo stesso per abitudine e
perché in un file `.py` senza non si vedrebbe niente.

Quanto ai *bins* dell'istogramma, sono le barre in cui l'intervallo dei valori
viene diviso: cambiarne il numero cambia il disegno, e vale la prova di due o
tre valori diversi, perché troppo poche barre nascondono la forma e troppe la
sbriciolano.

Ecco infine la forma con gli oggetti presi per nome, che è quella che troverai
nella documentazione e nel codice altrui. Fa esattamente lo stesso lavoro delle
quattro righe dello scatter:

```python
fig, ax = plt.subplots()          # il foglio e il riquadro, ciascuno col suo nome
                                  # (una chiamata, due cose: la funzione le
                                  # restituisce in coppia, e i due nomi a
                                  # sinistra se le prendono in ordine)
ax.scatter(df["eta"], df["spesa"])
ax.set_xlabel("età")              # sugli Axes i metodi si chiamano set_qualcosa
ax.set_ylabel("spesa")
plt.show()
```

Lo scatter rivela relazioni e valori anomali; l'istogramma mostra se una
variabile è simmetrica, asimmetrica o bimodale (con due "gobbe" invece di
una): cose che una media da sola nasconde. È il modo più economico per non
costruire, sopra dati fraintesi, un modello perfetto nella forma e sbagliato
nella sostanza.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il DataFrame è il foglio di calcolo programmabile di Pandas: ogni colonna
  tiene di norma valori di un solo tipo, e tutte le colonne condividono le
  stesse etichette di riga.
- L'ordine di lavoro è sempre lo stesso: carichi (`read_csv`), guardi
  (`head`, `info`, `describe`), filtri con il colino di una condizione e
  aggiungi colonne calcolate, raggruppi (`groupby`) per avere un riassunto
  per categoria.
- Il colino restituisce una copia. Per cambiare i valori nella tabella vera
  c'è `.loc`, e fra le sue quadre si scrive prima la condizione sulle righe e
  poi il nome della colonna: `df.loc[df["eta"] > 30, "spesa"] = 0`. Senza, il
  programma non protesta e la modifica finisce nella ciotola sbagliata.
- Una casella vuota (`NaN`) è essa stessa un'informazione: prima di buttarla o
  di riempirla, chiediti *perché* manca. E se la riempi con un valore inventato
  a partire dai dati, quel valore va calcolato solo sui dati con cui il modello
  impara, mai su quelli con cui lo si giudica: altrimenti stai facendo copiare
  il modello durante l'esame.
- Per guardarli bastano tre grafici: la dispersione per due grandezze
  insieme, l’istogramma per la forma di una sola, la linea per un
  andamento nel tempo. Il foglio (`Figure`) e il riquadro in cui si disegna
  (`Axes`) sono due oggetti distinti, e quasi tutti i metodi appartengono al
  secondo: da lì nasce l'errore più comune dei primi tempi.
- Guarda i dati prima di modellare: il quartetto di Anscombe mostra che
  quattro insiemi di dati con gli stessi numeri riassuntivi, a meno di qualche
  millesimo, possono essere completamente diversi, e che a vederlo è l'occhio,
  non la media.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il DataFrame è la tabella programmabile di Pandas: colonne (`Series`)
  tipizzate, allineate su un indice etichettato.
- Il flusso tipico è carica (`read_csv`) → ispeziona (`head`, `info`,
  `describe`) → filtra e trasforma (maschere booleane, nuove colonne) →
  aggrega (`groupby`).
- Per leggere va bene qualunque forma, per scrivere si usa `.loc`:
  `df[maschera]["col"] = 0` assegna a una copia, e pandas 3 lo segnala con un
  `ChainedAssignmentError` che, malgrado il nome, è un avviso e lascia
  proseguire il programma.
- I valori mancanti (`NaN`) vanno gestiti secondo il meccanismo di mancanza
  (MCAR, MAR, MNAR), imputando solo sul training set per evitare *data leakage*.
- Matplotlib: `scatter`, `hist` e `plot` per l'esplorazione, e la coppia
  `Figure`/`Axes` come modello a oggetti (`fig, ax = plt.subplots()`), che è la
  forma da preferire appena i grafici sono più d'uno.
- Visualizza prima di modellare: il quartetto di Anscombe mostra che
  statistiche che coincidono a meno di qualche millesimo possono nascondere
  dati radicalmente diversi.
```

`````

Adesso i numeri stanno in un array di NumPy, le tabelle in un DataFrame di
Pandas, e prima di fidarsene li si guarda in un grafico. Il
{doc}`capitolo di matematica </Matematica/overview>` dà a queste stesse cose
il loro nome matematico: una lista di numeri diventa un vettore, una tabella
una matrice, e la domanda «questa media dice la verità?» una domanda di
statistica. Il codice per cominciare c'è già; adesso bisogna sapere che cosa
fargli calcolare.
