# I dati come cittadini di prima classe

Apri il registro delle modifiche (la cronologia di `git`) di un sistema di
rilevamento frodi in produzione. Il codice del modello è quasi fermo: qualche
commit al mese, una libreria aggiornata, un iperparametro ritoccato. Poi
guarda i dati che quel codice ingoia: milioni di transazioni ogni giorno, mai
due uguali, con nuovi negozi, nuovi importi, nuove truffe che ieri non
esistevano. Il codice è il fiume; i dati sono la piena. Nel software
tradizionale l'oggetto che cambia (e che quindi si versiona, si testa, si
sorveglia) è il codice. Nel machine learning il codice è spesso la parte
*stabile*, e la parte viva, quella che si muove sotto i piedi, sono i dati.

Da questa asimmetria nasce un'idea che negli ultimi anni ha un nome:
**data-centric AI**. La provocazione, resa popolare da Andrew Ng intorno al
2021, è semplice: abbiamo passato un decennio a limare architetture per rubare
un decimale di accuratezza a un *benchmark*, mentre il guadagno più grande,
nei sistemi reali, si ottiene quasi sempre migliorando i *dati* (etichette più
coerenti, esempi più rappresentativi, meno rumore). Se è così, i dati non
possono restare un allegato del codice: vanno trattati come **cittadini di
prima classe**, versionati, testati e sorvegliati con la stessa disciplina. La
sezione precedente ha stabilito che riprodurre un modello richiede tre
artefatti: codice, dati, modello. Questa entra nel più grande e trascurato dei
tre, e nel sistema di tubature che lo trasporta {cite}`huyen2022designing`.

## Versionare i dati

Se non sai *quali* dati hanno prodotto un modello, non puoi riprodurlo, non
puoi capire perché una predizione è quella che è, e non puoi tornare indietro
quando un nuovo addestramento peggiora le cose. Il codice lo versiona `git` da
decenni; per i dati serve qualcosa di analogo, ma `git` da solo non basta: è
pensato per file di testo piccoli e confrontabili riga per riga, mentre un
dataset è grande, binario e opaco. Metterci dentro dieci gigabyte di immagini
lo gonfia fino a renderlo inservibile.

`````{tab} Elementare

Hai presente la cronologia di un documento condiviso: un Google Doc, una
pagina di Wikipedia? Puoi tornare a com'era martedì scorso, vedere chi ha
cambiato cosa, recuperare un paragrafo cancellato per sbaglio. Versionare i
dati è la stessa idea, applicata a una tabella o a una cartella di immagini: a
ogni «versione» del dataset resta appeso un cartellino con un codice, e
citando quel codice chiunque ritrova *esattamente* quei dati, non una loro
versione simile.

Il trucco per farlo senza duplicare montagne di file è un'impronta digitale.
Immagina di passare l'intero dataset in un tritatutto che ne ricava un codice
corto (poche decine di caratteri) con una proprietà magica: se cambi anche un
solo pixel, il codice cambia del tutto; se il dataset è identico, il codice è
identico. Così, invece di conservare cento copie dei dati dentro il progetto,
conservi solo i cartellini, e i dati veri stanno una volta sola in un
magazzino a parte. Il progetto dice «mi servono i dati con il cartellino
`a3f8…`», e il magazzino li consegna.

`````

`````{tab} Superiore

L'impronta è un **hash crittografico** del contenuto (per esempio SHA-256):
una funzione che mappa una sequenza arbitraria di byte in una stringa di
lunghezza fissa, deterministica e sensibile a ogni bit. Su questa idea si
costruisce il **content-addressable storage**: l'indirizzo di un dato *è*
l'hash del suo contenuto, non un percorso né un nome scelto a mano. Ne
discendono tre proprietà preziose. **Immutabilità**: un dato non si modifica
«sul posto», si scrive una nuova versione con un nuovo indirizzo; le vecchie
restano raggiungibili, e un esperimento passato non cambia sotto i piedi.
**Deduplicazione**: file identici hanno lo stesso hash e occupano spazio una
volta sola. **Integrità**: ricalcolare l'hash verifica che il dato non si sia
corrotto in transito.

Nel repository si versiona allora solo un *puntatore* (un piccolo file di
testo che contiene l'hash e l'indirizzo del magazzino remoto), mentre
l'artefatto vero vive in un *object store* (un bucket S3, un disco di rete).
Strumenti come DVC (*Data Version Control*) industrializzano esattamente
questo pattern, ma il concetto è indipendente dal tool: è lo stesso meccanismo
con cui `git` identifica i propri oggetti. Sopra i puntatori si costruisce
infine il **lineage** (la *provenienza*): il grafo che lega ogni modello alla
versione esatta dei dati e del codice che lo hanno generato. È ciò che
permette, mesi dopo, di rispondere alla domanda più temuta in un audit («da
dove viene questa predizione?») risalendo la catena fino al singolo dato
grezzo.

`````

## Le pipeline di dati

Un dataset pronto per l'addestramento non nasce così: è il prodotto finale di
una catena di trasformazioni. Si **estrae** il dato grezzo da una o più
sorgenti (un database, un flusso di eventi, dei file); si **pulisce** (valori
mancanti, duplicati, formati incoerenti); si costruiscono le **feature**, cioè
le variabili in cui il modello «vede» il mondo; e solo alla fine si
**addestra**. Ognuno di questi passaggi usa gli strumenti che già conosciamo
(le maschere booleane e i `groupby` di Pandas, la vettorizzazione di NumPy
visti nel capitolo su Python) ma il salto di qualità non è tecnico, è
organizzativo: la catena deve essere **riproducibile** (rieseguendola sugli
stessi dati grezzi si riottiene lo stesso dataset) e **orchestrata** (i
passaggi si succedono in un ordine dichiarato, non a mano in un notebook).

```{figure} ../figures/feature-engineering.svg
:name: fig-feature-engineering
:alt: "Catena in tre blocchi: i dati grezzi entrano nel blocco di feature engineering, che li trasforma nelle variabili con cui il modello lavora, e solo queste ultime raggiungono il modello. Il modello non vede mai i dati grezzi."
:width: 90%

Il modello non vede i dati: vede le feature. Tutto ciò che si decide nel
blocco di mezzo è il mondo dentro cui il modello dovrà cavarsela.
```

La freccia a senso unico di {numref}`fig-feature-engineering` è il motivo per
cui questa catena va versionata come si versiona il codice. Se cambia il modo
di costruire una feature, il modello addestrato prima e quello addestrato dopo
non stanno più guardando la stessa cosa, e nessun confronto fra i due punteggi
significa niente.

`````{tab} Elementare

Pensa a una catena di montaggio: la materia prima entra da un'estremità e a
ogni stazione subisce una lavorazione precisa, sempre la stessa, nell'ordine
giusto. Se un operaio salta un passaggio o li fa in ordine diverso, il pezzo
finale esce sbagliato. Una pipeline di dati è questo: stazioni collegate, ognuna
con un compito, che trasformano il dato grezzo in dato pronto senza che nessuno
intervenga a mano ogni volta.

Il pericolo ha un nome pittoresco: la **pipeline jungle**, la «giungla di
tubature». Nasce così: qualcuno aggiunge di fretta uno script per sistemare un
caso particolare, poi un altro sopra il primo per rattoppare un nuovo problema,
e mesi dopo nessuno capisce più quale tubo alimenta quale, né cosa succede se ne
tocchi uno. La cura non è un attrezzo magico ma una disciplina: passaggi
piccoli, dichiarati, ognuno ripetibile da solo, e nessuna trasformazione «a
mano» che non lasci traccia.

`````

`````{tab} Superiore

Formalmente una pipeline è un **DAG** (*directed acyclic graph*): i nodi sono
trasformazioni, gli archi le dipendenze dato-verso-dato, e l'assenza di cicli
garantisce un ordine di esecuzione ben definito. Due proprietà la rendono
governabile. L'**idempotenza**: rieseguire uno stadio sugli stessi input
produce lo stesso output, senza effetti collaterali accumulati (condizione per
poter ripartire da metà catena dopo un errore). E la **materializzazione
versionata** degli stadi intermedi, così che un cambiamento a valle non
obblighi a ricalcolare tutto da capo. Orchestratori come Airflow, Dagster o
Prefect gestiscono lo scheduling, le dipendenze e i tentativi di ripristino;
ma, come per il versionamento, lo strumento è secondario rispetto al
principio. Automatizzare l'intera catena (da dato grezzo a modello valutato)
con un comando solo è il cuore della *Continuous Delivery for Machine
Learning* {cite}`sato2019continuous`: finché un pezzo della pipeline resta un
rito manuale, l'intero sistema non è né riproducibile né rilasciabile in modo
affidabile.

`````

## Il feature store

C'è un punto della pipeline che merita un discorso a sé: le **feature**. Una
stessa feature («spesa media dell'utente negli ultimi 30 giorni», «numero di
transazioni nell'ultima ora») serve a più modelli e, soprattutto, va calcolata
in *due momenti diversi*: durante l'addestramento, su masse di dati storici, e
durante il servizio, su un singolo caso che arriva ora. Se i due calcoli
divergono anche di poco, il modello riceve in produzione qualcosa di diverso
da ciò su cui ha imparato. Il **feature store** è l'infrastruttura che risolve
questo problema centralizzando la definizione e il calcolo delle feature
{cite}`huyen2022designing`.

`````{tab} Elementare

Immagina una grande cucina con venti cuochi. Se ognuno si prepara il soffritto a
modo suo, ogni piatto esce leggermente diverso e nessuno sa più perché. La
soluzione è una **dispensa comune**: il soffritto lo prepara un reparto solo,
sempre con la stessa ricetta, e tutti i cuochi lo prendono già pronto da lì. Il
piatto di oggi è identico a quello di ieri, e un cuoco nuovo non deve reinventare
nulla.

Il feature store è quella dispensa. Le feature si definiscono una volta, in un
posto solo, e sia chi addestra il modello sia chi lo manda in produzione le
prende dallo stesso scaffale, con la garanzia che siano *le stesse* e
*fresche*, cioè aggiornate. In più, una feature preparata bene la riusano
dieci modelli diversi: si cucina una volta, si serve a tutti.

`````

`````{tab} Superiore

Un feature store ha tipicamente due volti. L'**offline store** conserva lo
storico completo delle feature, ottimizzato per letture massicce: è la sorgente
per costruire i dataset di addestramento. L'**online store** tiene invece
l'ultimo valore di ogni feature, ottimizzato per letture a bassissima latenza:
è ciò che il servizio interroga quando deve rispondere in millisecondi. La
stessa *definizione* alimenta entrambi, ed è questa condivisione a sradicare
alla radice le incoerenze tra addestramento e produzione.

Il problema tecnico più insidioso che un feature store deve garantire è la
**point-in-time correctness**. Costruendo un esempio di addestramento
etichettato al tempo $t$, le sue feature vanno calcolate con i soli dati
disponibili *prima* di $t$: usare un valore aggregato che include informazione
successiva a $t$ inietta nel modello una conoscenza del futuro che in
produzione non avrà mai (una forma di *data leakage* temporale che gonfia le
metriche in laboratorio e crolla sul campo). Un *point-in-time join* corretto
è tedioso da implementare a mano ed è una delle ragioni per cui il feature
store esiste come componente dedicato.

`````

## Training–serving skew

Arriviamo così al bug più classico e più costoso di tutta la disciplina,
quello che il feature store esiste per prevenire: il **training–serving
skew**. Si verifica quando una feature è calcolata in un modo durante
l'addestramento e in un modo *anche solo leggermente diverso* durante il
servizio. Il modello, tarato sui numeri del training, riceve in produzione
numeri che vogliono dire un'altra cosa, e sbaglia in silenzio, senza che
nessuna eccezione venga sollevata.

`````{tab} Elementare

È come tarare una bilancia in grammi e poi, senza dirlo a nessuno, pesarci
sopra in once. Nessun errore lampeggia sullo schermo: i numeri arrivano,
sembrano plausibili, e il risultato è semplicemente sbagliato. Il caso da
manuale è la **normalizzazione**. In addestramento si «centra» ogni feature
sottraendo la sua media e dividendo per la sua deviazione (per portarle tutte
a una scala confrontabile), calcolando media e deviazione sull'intero dataset.
In produzione, per una svista, qualcuno le ricalcola sul *singolo lotto* di
dati in arrivo, e un lotto tutto di importi alti, ricentrato su sé stesso,
sembra improvvisamente «normale». Vediamolo con i numeri.

`````

`````{tab} Superiore

Sia $x$ una feature e $z = (x - \mu)/\sigma$ la sua versione standardizzata,
dove $\mu$ e $\sigma$ sono media e deviazione standard. La regola vincolante è
che $\mu$ e $\sigma$ siano **statistiche del training**, stimate una volta e
*congelate*: fanno parte del modello tanto quanto i pesi. Usarle in
addestramento e ricalcolarle in produzione su un altro campione viola
l'ipotesi sotto cui il modello è stato ottimizzato. Nel codice qui sotto la
versione corretta applica $\mu,\sigma$ del training; quella bacata ricalcola
$\mu_{\text{batch}},\sigma_{\text{batch}}$ sul lotto corrente, azzerandone di
fatto la media: un batch anomalo (tutto di importi alti) viene ricondotto a
zero e il modello non lo riconosce più come anomalo.

`````

```python
import numpy as np

rng = np.random.default_rng(0)

# --- addestramento ---
# una sola feature: l'importo di una transazione, in euro
X_train = rng.normal(loc=100.0, scale=20.0, size=10_000)

# statistiche "congelate" al momento dell'addestramento
mu, sigma = X_train.mean(), X_train.std()

# modellino lineare gia' addestrato sulla feature normalizzata z = (x - mu)/sigma
# lo score passa in una sigmoide -> probabilita' che la transazione sia una frode
w, b = 1.5, -0.2


def sigmoide(t):
    return 1.0 / (1.0 + np.exp(-t))


def predici_corretto(x):
    z = (x - mu) / sigma          # normalizza con le statistiche DEL TRAINING
    return sigmoide(w * z + b)


def predici_bacato(x_batch):
    # BUG: normalizza con media/std DEL BATCH corrente, non del training
    z = (x_batch - x_batch.mean()) / x_batch.std()
    return sigmoide(w * z + b)


# in produzione arriva un batch anomalo: importi molto piu' alti del solito
X_prod = rng.normal(loc=160.0, scale=20.0, size=32)

p_ok = predici_corretto(X_prod)
p_bug = predici_bacato(X_prod)

print("prob. media di frode (corretta):", round(float(p_ok.mean()), 3))
print("prob. media di frode (bacata):  ", round(float(p_bug.mean()), 3))
print("scarto massimo sulle predizioni:", round(float(np.abs(p_ok - p_bug).max()), 3))
```

```text
prob. media di frode (corretta): 0.967
prob. media di frode (bacata):   0.454
scarto massimo sulle predizioni: 0.793
```

Lo stesso identico modello, sugli stessi identici dati, dà due risposte
opposte. La pipeline corretta riconosce il lotto come sospetto (probabilità
media di frode $0{,}97$); quella bacata, ricentrando ogni batch su sé stesso,
cancella l'anomalia e lo giudica quasi innocuo ($0{,}45$), con differenze fino
a $0{,}79$ su singole transazioni. Nessun errore, nessun avviso: solo
predizioni sbagliate. Ecco perché la definizione di una feature deve vivere in
*un posto solo*, condiviso tra addestramento e servizio: è il compito del
feature store.

## Validare i dati in ingresso

Il *training–serving skew* è un bug di *calcolo* delle feature. Ma un'intera
famiglia di guai arriva prima, dal dato grezzo stesso: un campo che cambia
unità di misura, una colonna che di colpo si riempie di valori nulli, un codice
prodotto che non era mai comparso. Nel software normale un input malformato fa
esplodere il programma, e l'errore si nota subito. Un modello, invece, un numero
lo restituisce *sempre*: dagli in pasto spazzatura e ti darà una predizione
dall'aria rispettabile. Per questo i dati in ingresso vanno **validati
esplicitamente**, come si controlla la merce alla porta del magazzino prima di
metterla a scaffale.

`````{tab} Elementare

Al ricevimento merci di un supermercato qualcuno controlla ogni bancale: è il
prodotto giusto? La quantità è quella dell'ordine? Ci sono confezioni rotte o
scadute? Chi non passa il controllo non entra. La validazione dei dati fa lo
stesso, e verifica quattro cose. Lo **schema**: ci sono tutte le colonne
attese, e del tipo giusto (un'età è un numero, non la parola «trenta»)? Il
**range**: i valori sono plausibili (un'età tra 0 e 120, un importo non
negativo)? I **valori mancanti**: quante caselle sono vuote, e possiamo
permettercelo? Le **distribuzioni**: la media, la varianza, le proporzioni delle
categorie somigliano a quelle di ieri? Le prime tre si controllano su ogni
singolo record; l'ultima solo guardando tanti record insieme.

`````

`````{tab} Superiore

La validazione dei dati è uno dei quattro assi della **ML Test Score**
{cite}`breck2017ml`, la rubrica di collaudo che misura la maturità di un
sistema di ML: include test sullo schema delle feature, sui loro intervalli e
sul fatto che ogni feature apporti davvero valore. Conviene distinguere due
livelli. I controlli **puntuali** (tipo, obbligatorietà, intervallo, assenza
di `NaN`) si applicano a ogni record isolato e sono economici: sono quelli che
implementiamo qui sotto. I controlli **distribuzionali** (la media di una
feature è slittata? la proporzione di una categoria è raddoppiata?) richiedono
di confrontare un lotto con una *baseline* di riferimento, ed è qui che la
validazione statica sfuma nel **monitoraggio** del *dataset shift*
{cite}`quinonero2009dataset`: lo abbiamo inquadrato in termini statistici
nella sezione «Quando i dati cambiano», e il suo lato operativo (sorvegliare
le distribuzioni nel tempo e decidere quando riaddestrare) avrà una sezione
dedicata. Qui restiamo al primo livello: fermare alla porta il record
palesemente malformato.

`````

```python
from math import isnan

# schema: per ogni campo, il tipo atteso, l'intervallo ammesso e se e' obbligatorio
SCHEMA = {
    "eta":     {"tipo": int,   "min": 0,   "max": 120,   "obbligatorio": True},
    "importo": {"tipo": float, "min": 0.0, "max": 1e6,   "obbligatorio": True},
    "citta":   {"tipo": str,                             "obbligatorio": True},
}


def valida(record, schema):
    """Controlla un record (dict) contro lo schema; ritorna la lista degli errori."""
    errori = []
    for campo, regole in schema.items():
        # 1) valore assente o nullo
        if campo not in record or record[campo] is None:
            if regole.get("obbligatorio"):
                errori.append(f"{campo}: valore mancante")
            continue
        valore = record[campo]
        # 2) NaN, il "buco" numerico di NumPy/Pandas
        if isinstance(valore, float) and isnan(valore):
            errori.append(f"{campo}: NaN")
            continue
        # 3) tipo sbagliato
        if not isinstance(valore, regole["tipo"]):
            atteso = regole["tipo"].__name__
            errori.append(f"{campo}: tipo {type(valore).__name__}, atteso {atteso}")
            continue
        # 4) fuori dall'intervallo ammesso
        if "min" in regole and valore < regole["min"]:
            errori.append(f"{campo}: {valore} sotto il minimo {regole['min']}")
        if "max" in regole and valore > regole["max"]:
            errori.append(f"{campo}: {valore} oltre il massimo {regole['max']}")
    return errori


records = [
    {"eta": 34, "importo": 250.0, "citta": "Milano"},          # valido
    {"eta": 200, "importo": 90.0, "citta": "Roma"},            # eta fuori range
    {"eta": 41, "importo": float("nan"), "citta": "Napoli"},   # importo NaN
    {"eta": 29, "importo": 60.0},                              # citta mancante
    {"eta": "trenta", "importo": 15.0, "citta": "Torino"},     # eta di tipo sbagliato
]

for i, r in enumerate(records):
    errori = valida(r, SCHEMA)
    print(f"record {i}:", "OK" if not errori else " | ".join(errori))
```

```text
record 0: OK
record 1: eta: 200 oltre il massimo 120
record 2: importo: NaN
record 3: citta: valore mancante
record 4: eta: tipo str, atteso int
```

Poche righe, ma è la porta blindata del sistema: ogni record che entra viene
promosso o respinto secondo regole esplicite, e i respinti finiscono in un
registro invece che, silenziosamente, dentro il modello. In produzione questo
schema si arricchisce (soglie sulla percentuale di `NaN` tollerata, controlli
di coerenza tra campi, l'aggancio ai test distribuzionali) ma l'ossatura è
questa: dichiarare cosa ci si aspetta dai dati, e verificarlo prima di
fidarsene. Trattare i dati da cittadini di prima classe significa, alla fine,
esattamente questo: dargli un contratto, e farlo rispettare.

```{admonition} Da ricordare
:class: important
- Nel ML il codice è spesso la parte stabile e i **dati** la parte viva:
  l'approccio **data-centric** sposta l'attenzione dal limare il modello al
  migliorare i dati, e li tratta come artefatti di prima classe (versionati,
  testati, sorvegliati {cite}`huyen2022designing`).
- **Versionare i dati**: `git` non basta (file grandi e binari); si usa
  l'**hash del contenuto** come indirizzo (*content-addressable storage*), da
  cui immutabilità, deduplicazione e **lineage** che lega ogni modello ai dati
  esatti che l'hanno prodotto. DVC è un esempio, il concetto è indipendente dal
  tool.
- Una **pipeline di dati** (estrazione → pulizia → feature → training) va
  resa **riproducibile e orchestrata** (un DAG di stadi idempotenti), per non
  degenerare nella *pipeline jungle*; automatizzarla per intero è il cuore della
  CD4ML {cite}`sato2019continuous`.
- Il **feature store** centralizza la definizione delle feature: stessa ricetta
  in addestramento (*offline*) e in produzione (*online*), riuso tra modelli,
  freschezza e **point-in-time correctness** contro il *leakage* temporale.
- Il **training–serving skew** è il bug silenzioso per eccellenza: una feature
  calcolata diversamente in training e in produzione (es. normalizzare col
  batch invece che con le statistiche congelate del training) sballa le
  predizioni senza sollevare alcun errore.
- **Validare i dati in ingresso** (schema, tipi, range, valori mancanti,
  distribuzioni) è un asse della **ML Test Score** {cite}`breck2017ml`. I
  controlli puntuali fermano il record malformato; quelli distribuzionali
  sfumano nel monitoraggio del *dataset shift* {cite}`quinonero2009dataset`.
```
