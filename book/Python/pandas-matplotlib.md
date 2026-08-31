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
sequenza di valori con un'etichetta ciascuno (l’*indice*). Un **DataFrame** è
una tabella intera: tante Series affiancate che condividono lo stesso indice
di riga. Attenzione alla parola *indice*, che qui cambia mestiere rispetto a
NumPy: là era il numero della posizione (`x[0]`, il primo), qui è
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
a sinistra. L'indice fa da etichetta: pandas riconosce ogni riga da lì, e
quell'etichetta resta attaccata ai dati quando si filtra, si ordina o si
estrae una colonna.

`````{tab} Elementare

Un DataFrame è un foglio Excel fatto di codice. Ogni colonna ha un'intestazione
("età", "città", "acquisti") e ogni riga è un cliente; solo che invece di
cliccare con il mouse dai istruzioni a parole:

```python
import pandas as pd

df = pd.DataFrame({
    "citta": ["Milano", "Roma", "Napoli"],
    "eta":   [34, 28, 41],
    "spesa": [120.0, 85.5, 60.0],
})
```

Quello fra le graffe è un **dizionario**, lo stesso delle basi del linguaggio: le
chiavi diventano i nomi delle colonne, e il valore di ciascuna è la lista dei
dati di quella colonna, dall'alto in basso. Ogni colonna è una Series; tutte
insieme formano la tabella. E dentro una colonna i valori sono tutti della
stessa specie, numeri con numeri e testo con testo: è la stessa regola
dell'array di NumPy, applicata una colonna per volta, ed è quello che permette
a Pandas di girare a lui i conti. Il vantaggio
rispetto a Excel è che ogni operazione è ripetibile e documentata: la scrivi
una volta e la riesegui su un milione di righe senza cambiare nulla.

L'altra differenza da Excel sta nelle etichette di riga. Pandas ragiona con
quelle: se accosti due elenchi di clienti scritti in ordine diverso, le righe
si appaiano per etichetta, come quando si confrontano due registri cercando lo
stesso nome su entrambi. E se un nome compare in un registro solo, nella
casella accanto resta un buco, non il dato di un altro cliente: un buco si
vede subito, uno scambio di persona no.

`````

`````{tab} Superiore

Un `DataFrame` è una collezione di `Series` allineate su un `Index` comune.
Ogni colonna ha un proprio `dtype` omogeneo (`int64`, `float64`, `str`,
`category`, `datetime64`), il che permette a Pandas di appoggiarsi a NumPy per
le operazioni vettoriali colonna per colonna. Il dtype del testo è cambiato di
recente, e la rete è piena di materiale che descrive ancora quello vecchio: da
**pandas 3.0** una colonna di testo ha dtype `str`,
sostenuto da Arrow quando `pyarrow` è installato, ed è molto più compatto e
veloce del vecchio `object`, in cui ogni cella era un oggetto Python a sé.
`object` esiste ancora, ma è diventato il dtype delle colonne che mescolano
tipi. L'indice è una struttura etichettata (anche gerarchica, `MultiIndex`)
usata per l'allineamento automatico, e il numero di riga ne è solo il caso più
semplice. Quando sommi due Series, Pandas non
allinea per posizione ma **per etichetta**, inserendo `NaN` dove le etichette
non combaciano: comportamento che evita interi errori "off-by-one" tipici
degli array grezzi.

`````

## Caricare e ispezionare i dati

Nella realtà i dati non li digiti a mano: li carichi. Il formato più comune è
il **CSV** (un semplice file di testo con i valori separati da virgole, il
formato in cui quasi ogni programma sa esportare una tabella) e la funzione
`read_csv` lo legge in una riga, riconoscendo da sola tipi e intestazioni. Un
avviso da tastiera italiana: il CSV che Excel produce qui da noi usa spesso il
punto e virgola come separatore e la virgola per i decimali, e si legge
dichiarandolo, `read_csv("file.csv", sep=";", decimal=",")`.

Un file su cui provare ce lo fabbrichiamo al volo, così ogni numero che segue
si può rifare:

```python
pd.DataFrame({
    "nome":  ["Ada", "Bruno", "Carla", "Dario", "Elena", "Furio"],
    "eta":   [34, None, 41, 36, 52, 23],
    "citta": ["Milano", "Torino", "Milano", "Napoli", "Milano", "Torino"],
    "spesa": [120.5, 89.0, 240.0, None, 310.0, 74.9],
}).to_csv("vendite.csv", index=False)
```

```python
df = pd.read_csv("vendite.csv")   # il file va cercato dove sta girando il
                                  # programma: stessa cartella, oppure il
                                  # percorso completo ("dati/vendite.csv")

df.head()        # prime 5 righe: uno sguardo veloce
df.info()        # colonne, dtype, valori non nulli, memoria
df.describe()    # statistiche riassuntive delle colonne numeriche
```

Da qui in avanti `df` è questa tabella caricata da file, e non più quella
scritta a mano: il nome è lo stesso perché `df` (da *dataframe*) è il
nome che quasi tutti danno alla tabella su cui stanno lavorando in quel
momento. La tabella su cui girano le righe che seguono ha sei clienti e
quattro colonne (nome, età, città, spesa), con un paio di caselle lasciate
vuote di proposito, perché i dati veri sono quasi sempre così. Ecco che cosa risponde la prima:

```text
    nome   eta   citta  spesa
0    Ada  34.0  Milano  120.5
1  Bruno   NaN  Torino   89.0
2  Carla  41.0  Milano  240.0
3  Dario  36.0  Napoli    NaN
4  Elena  52.0  Milano  310.0
```

Da leggere ci sono due cose oltre ai dati. La colonna senza intestazione a
sinistra, con 0, 1, 2, 3, 4, è l’**indice**: le etichette di riga di cui si
parlava poco fa, che qui pandas ha messo da sé perché il file non ne aveva. E
quei due `NaN` sono le caselle vuote. Sono anche il motivo per cui l'età
compare come 34.0 invece che come 34: una casella vuota non è un numero
intero, e per tenerla in colonna insieme agli altri pandas passa tutta la
colonna ai numeri con la virgola.

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
Puoi combinarne più d'una con `&` («e») e `|` («o»), e ogni condizione va
chiusa fra parentesi sue, altrimenti Python legge la riga in un altro modo e
risponde con un errore:

```python
df[(df["eta"] > 30) & (df["citta"] == "Milano")]
```

Il colino però non svuota la pentola: quello che passa finisce in una ciotola
a parte, e la pentola resta com'era. Il filtro fa lo stesso. La tabella
filtrata è una copia, e correggerne i valori non cambia la tabella di
partenza; il programma non si ferma, e il gesto riesce sul recipiente
sbagliato. Per scrivere sulla tabella originale c'è un attrezzo apposta,
`.loc`, che sceglie le righe e le cambia lì dove stanno. Fra le sue quadre si
scrive prima la condizione sulle righe e poi il nome della colonna:
`df.loc[df["eta"] > 30, "spesa"] = 0` azzera la spesa di chi ha più di trent'anni
sulla tabella vera, e non su una ciotola a parte.

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
lascia `df` com'era. Pandas 3 lo segnala con un `ChainedAssignmentError` che,
malgrado il nome, è un avviso e non un'eccezione: il programma non si
ferma, tira dritto, e la modifica che credevi di aver fatto semplicemente non
c'è. In uno script che filtra gli avvisi, o in un notebook con mille righe di
output, il gesto sbagliato passa in silenzio, ed è questo che lo rende
l'errore più frequente del mestiere. La
forma che funziona è una sola, `df.loc[df["eta"] > 30, "spesa"] = 0`, perché
seleziona e assegna in un passo solo. Nota per chi cerca in rete: con il
Copy-on-Write, predefinito da pandas 3, il vecchio `SettingWithCopyWarning` non
esiste più e la copia non scrive mai sull'originale, quindi il classico
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
a dimenticare: dopo un `groupby` la colonna di raggruppamento diventa
l'indice, e smette di essere una colonna. Da lì in poi una riga si chiama con
la sua etichetta, non con il valore di una colonna. È questo a spiegare gran
parte dei `KeyError` che arrivano dopo un raggruppamento, dove `KeyError` è
l'errore con cui Python dice «questo nome qui dentro non c'è»: chiedere la
colonna `"citta"` al risultato di un raggruppamento per città è il modo più
rapido di provocarlo. Sulla tabella di partenza, che il raggruppamento non
tocca, quella colonna c'è ancora.

```python
df.groupby("citta")["spesa"].mean()    # spesa media per città
df.groupby("citta").agg(
    spesa_media=("spesa", "mean"),     # una colonna nuova, che chiamo io
    clienti=("nome", "count"),         # (da quale colonna, con quale conto)
)
```

La prima riga è la più lunga catena di punti e quadre vista finora, e si legge
da sinistra a destra come una frase, un pezzo per volta: «prendi `df`,
raggruppalo per città, di quel che esce tieni la colonna `spesa`, e di quella
fai la media». Ogni pezzo lavora su ciò che ha prodotto il pezzo precedente, e
questo modo di incatenare le operazioni è lo stile normale di pandas.

Il secondo esempio calcola due riassunti in una volta e dà a ciascuno il nome
che si vuole: a sinistra dell'uguale il nome della colonna che uscirà, a destra
la coppia «da quale colonna prendere i valori, che conto farci sopra».

Eseguendo la prima riga sulla nostra tabella, la spesa media per città esce
così:

```text
citta
Milano    223.50
Napoli       NaN
Torino     81.95
Name: spesa, dtype: float64
```

Napoli risponde `NaN`. Le funzioni di riassunto di pandas saltano le caselle
vuote: una colonna con due numeri e un buco fa la media dei due. Ma il suo
unico cliente ha la spesa mancante, e una media senza nemmeno un valore da
mediare non esiste. È il primo incontro con le caselle vuote, che meritano una
sezione loro.

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
\bar{x}_g = \frac{1}{n_g}\sum_{i=1}^{n_g} x_i ,
$$

dove $n_g$ conta i valori **presenti** nel gruppo, e non le sue righe: con
`skipna=True`, che è il default, `mean` scarta i mancanti prima di sommare e
prima di dividere. Su un gruppo che non ha nemmeno un valore la somma resta
senza divisore, ed è da lì che viene il `NaN` di Napoli. Oltre a `mean` sono
disponibili
`sum`, `count`, `std`, `min`, `max`, `median` e funzioni arbitrarie via
`agg`/`apply`. Il metodo `agg` con argomenti nominati (*named aggregation*)
produce colonne dal nome esplicito, rendendo il risultato pronto per un
report o per un incrocio con un'altra tabella (il `merge`, il parente pandas
della `JOIN` dei database).

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

Nei tre modi di {numref}`fig-dati-mancanti` resta fuori una cosa: che una
casella vuota è essa stessa un'informazione. Se manca perché il sensore era
spento, è un
caso; se manca perché la domanda era imbarazzante, il fatto che manchi dice
qualcosa, e riempirla con la media cancella proprio quel qualcosa.

`````{tab} Elementare

Hai due strade. Puoi **buttare via** le righe incomplete, oppure **riempirle**
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
modello: è materia dei capitoli sul machine learning, qui basta sapere che
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
si nota da `df.info()`); le colonne di date usano `NaT`, i dtype *nullable* di
Pandas usano `pd.NA`, e il nuovo dtype `str` di pandas 3 continua a usare
`nan`, così `isna()` risponde come sempre.

La strategia dipende dal meccanismo di mancanza (MCAR, MAR, MNAR nella
tassonomia di Rubin {cite}`rubin1976inference`): se i dati mancano
*completamente a caso* (MCAR) è
garantito che eliminare le righe incomplete non introduca distorsioni, e in
una regressione l'eliminazione resta lecita anche quando la mancanza dipende
solo dalle covariate e non dalla risposta. L'imputazione con media o mediana
è semplice ma comprime la varianza e ignora le correlazioni tra variabili;
alternative più fedeli sono l'imputazione tramite modello (es. $k$-NN o
regressione, `sklearn.impute.KNNImputer`) o l'imputazione multipla.
Regola d'oro: qualsiasi imputazione va stimata solo sul training set e poi
applicata al test set, per non far trapelare informazione (*data leakage*).

`````

## Perché guardare i dati prima di modellare

Verrebbe la tentazione di saltare direttamente al modello, e c'è un esempio
famoso che spiega perché sia una cattiva idea. Nel 1973 lo statistico Francis
Anscombe mise insieme quattro piccole raccolte di undici punti ciascuna, e le
costruì apposta perché, misurate, risultassero gemelle fino al secondo
decimale, che è la precisione con cui una statistica si scrive.

Le misure su cui risultano gemelle sono le stesse che si prendono davanti a
qualunque tabella nuova, e si guardano una per volta.

La **media** è il valore attorno a cui i numeri si dispongono: si sommano e si
divide per quanti sono. Nelle quattro raccolte è la stessa, sia per la
grandezza in orizzontale ($x$) sia per quella in verticale ($y$).

La **varianza** misura lo sparpagliamento attorno a quella media: piccola se i
valori stanno tutti lì vicino, grande se sono sparsi ai due estremi. Coincide
anche questa.

La **correlazione** è un numero fra $-1$ e $1$ che dice quanto le due grandezze
crescono insieme lungo una retta, e a zero vuol dire che una retta fra le due
non c'è: un legame di un'altra forma può esserci eccome, e i quattro disegni
stanno per mostrarlo. Nelle quattro raccolte vale circa $0{,}816$.

La **retta di regressione**, infine, è quella che passa più vicino possibile a
tutti i punti insieme. Ed è la stessa retta:

$$
\hat{y} = 3 + 0{,}5\,x .
$$

Il cappuccio sopra la $y$ vuol dire «valore *previsto* dalla retta», da tenere
distinto dal valore misurato davvero, e la distinzione fra i due è quella su
cui poggia tutto il machine learning. Il conto si fa una volta e poi si
ricorda: nel primo insieme, dove $x$ vale $10$, la retta prevede
$\hat{y} = 3 + 0{,}5 \cdot 10 = 8$, mentre il punto misurato in quel posto sta
a $8{,}04$. La differenza fra i due, qui quattro centesimi, è l’**errore** su
quel punto, ed è la quantità che ogni modello cercherà di rendere piccola.

Costruire quattro insiemi di dati che coincidono su tutte e quattro queste
misure è un lavoro di precisione, ed è il punto: sono fabbricati apposta
{cite}`anscombe1973graphs`, e sulla carta restano indistinguibili. Ma basta disegnarli ({numref}`fig-anscombe`) per scoprire
che raccontano quattro storie completamente diverse.

```{figure} ../figures/quartetto-anscombe.svg
:name: fig-anscombe
:alt: "Quattro grafici a dispersione con la stessa retta di regressione ma nubi di punti molto diverse: una relazione lineare, una curva, una lineare con un valore anomalo, e una con i punti allineati verticalmente più un punto isolato."
:width: 90%

Il quartetto di Anscombe. Le stesse statistiche a due decimali, la stessa
retta: solo il grafico rivela che i quattro insiemi di dati non hanno nulla in
comune.
```

Il primo è davvero lineare; il secondo è una curva che una retta descrive
male; il terzo è una retta rovinata da un solo valore anomalo (un *outlier*);
il quarto ha tutti i punti su una verticale, tranne uno che da solo determina
la pendenza. Nessuna
di queste patologie emerge dai numeri riassuntivi: solo l'occhio le coglie.

Matplotlib è lo strumento per farlo. Tre grafici bastano per l'esplorazione
iniziale: la **dispersione** per due variabili, l’**istogramma** per la
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
plt.show()                            # mostra il grafico e chiude questo foglio

# per un istogramma servono molti valori: con i sei della tabella si vedrebbero
# sei stecchi e nessuna forma, quindi qui ne fabbrichiamo trecento finti
spese = np.random.default_rng(0).normal(120, 30, size=300)
plt.hist(spese, bins=20)              # distribuzione: 20 barre ("bins")
plt.show()

plt.plot(mesi, fatturato)             # andamento nel tempo
plt.show()
```

Senza `plt.show()` i tre grafici finirebbero uno sopra l'altro sullo stesso
foglio, con le etichette del primo appiccicate agli altri due: è il modo di
dire «questo è finito». In uno script serve anche a farlo comparire; in un
notebook la cella lo mostra da sé, e la riga si scrive lo stesso, per abitudine
e perché in un file `.py` senza non si vedrebbe niente.

Quanto ai *bins* dell'istogramma, sono le barre in cui l'intervallo dei valori
viene diviso: cambiarne il numero cambia il disegno, e vale la prova di due o
tre valori diversi, perché troppo poche barre nascondono la forma e troppe la
sbriciolano.

Ecco infine la forma con gli oggetti presi per nome, che è quella che troverai
nella documentazione e nel codice altrui. Fa esattamente lo stesso lavoro delle
quattro righe dello scatter:

```python
fig, ax = plt.subplots()          # il foglio e il riquadro, ciascuno col suo nome
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
- Il **DataFrame** è il foglio di calcolo programmabile di Pandas: ogni colonna
  tiene valori di un solo tipo, e tutte le colonne condividono le stesse
  etichette di riga.
- L'ordine di lavoro è sempre lo stesso: **carichi** (`read_csv`), **guardi**
  (`head`, `info`, `describe`), **filtri** con il colino di una condizione e
  aggiungi colonne calcolate, **raggruppi** (`groupby`) per avere un riassunto
  per categoria.
- Il colino restituisce una copia. Per **cambiare** i valori nella tabella vera
  c'è `.loc`, e fra le sue quadre si scrive prima la condizione sulle righe e
  poi il nome della colonna: `df.loc[df["eta"] > 30, "spesa"] = 0`. Senza, il
  programma non protesta e la modifica finisce nella ciotola sbagliata.
- Una casella vuota (`NaN`) è essa stessa un'informazione: prima di buttarla o
  di riempirla, chiediti *perché* manca. E se la riempi con un valore inventato
  a partire dai dati, quel valore va calcolato solo sui dati con cui il modello
  impara, mai su quelli con cui lo si giudica: altrimenti stai facendo copiare
  il modello durante l'esame.
- Per guardarli bastano tre grafici: la **dispersione** per due grandezze
  insieme, l’**istogramma** per la forma di una sola, la **linea** per un
  andamento nel tempo. Il foglio (`Figure`) e il riquadro in cui si disegna
  (`Axes`) sono due oggetti distinti, e quasi tutti i metodi appartengono al
  secondo: da lì nasce l'errore più comune dei primi tempi.
- **Guarda i dati prima di modellare**: il quartetto di Anscombe mostra che
  quattro insiemi di dati con gli stessi numeri riassuntivi a due decimali
  possono essere completamente diversi, e che a vederlo è l'occhio, non la
  media.
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
- Per **leggere** va bene qualunque forma, per **scrivere** si usa `.loc`:
  `df[maschera]["col"] = 0` assegna a una copia, e pandas 3 lo segnala con un
  `ChainedAssignmentError` che, malgrado il nome, è un avviso e lascia
  proseguire il programma.
- I **valori mancanti** (`NaN`) vanno gestiti secondo il meccanismo di mancanza
  (MCAR, MAR, MNAR), imputando solo sul training set per evitare *data leakage*.
- **Matplotlib**: `scatter`, `hist` e `plot` per l'esplorazione, e la coppia
  `Figure`/`Axes` come modello a oggetti (`fig, ax = plt.subplots()`), che è la
  forma da preferire appena i grafici sono più d'uno.
- **Visualizza prima di modellare**: il quartetto di Anscombe mostra che
  statistiche che a due decimali coincidono possono nascondere dati
  radicalmente diversi.
```

`````

Python adesso è un attrezzo che risponde: un array di NumPy per i numeri, un
DataFrame di Pandas per le tabelle, un grafico per guardarle prima di
fidarsene. Sono le stesse cose che il
{doc}`capitolo di matematica </Matematica/overview>` chiama con altri nomi,
perché una lista di numeri lì diventa un vettore, una tabella diventa una
matrice, e la domanda «questa media dice la verità?» diventa una domanda di
statistica. Da lì in avanti non si impara più a scrivere codice: si impara che
cosa fargli calcolare.
