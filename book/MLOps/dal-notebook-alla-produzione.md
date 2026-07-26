# Dal notebook alla produzione

Il notebook ha quarantasette celle, e i contatori a lato raccontano una storia
sconfortante: `[12]`, poi `[8]`, poi `[31]`, poi di nuovo `[9]`. Le celle sono
state eseguite in ordine sparso, avanti e indietro, per giorni. Ieri sera il
modello dava un'accuratezza del 94% e l'autrice è andata a dormire soddisfatta.
Stamattina una collega apre lo stesso file, preme *Restart & Run All* — riavvia
il kernel e riesegue tutto dall'alto, come farebbe una macchina che non sa nulla
della cronologia — e metà delle celle esplode: una variabile definita in un
blocco poi cancellato, un file letto da un percorso che esiste solo su quel
portatile, uno `train_test_split` senza seme che ogni volta divide i dati in
modo diverso. La frase che chiude la giornata è la più celebre della disciplina:
«Ma sul mio computer funzionava».

Tra quel notebook e un sistema che serve previsioni agli utenti, ogni giorno,
in modo affidabile, c'è un abisso. Colmarlo è il mestiere che va sotto il nome
di **MLOps**: l'insieme di pratiche che portano un modello dalla fase di
esplorazione — dove va benissimo che le celle si eseguano in disordine — a un
software vero, riproducibile, monitorato e manutenibile nel tempo. Questa
sezione è la mappa di quell'abisso: che cosa cambia davvero quando si passa
«dal mio computer» al mondo, e con quali strumenti si attraversa.

## Il divario ricerca–produzione

Un modello che «funziona» in un notebook ha risolto una piccola parte del
problema. La parte grande — quella che riempie gli anni di lavoro di un team —
comincia dopo: farlo girare in modo affidabile, abbastanza in fretta, per tanti
utenti, e riuscire a rifare esattamente lo stesso risultato tra sei mesi, quando
nessuno ricorda più quali dati e quale versione del codice l'avevano prodotto.

`````{tab} Elementare

Pensa alla differenza tra cucinare un piatto una sera per gli amici e metterlo
nel menu di un ristorante. La prima volta puoi assaggiare, aggiustare di sale a
occhio, ripetere se viene male: nessuno ti cronometra. Nel menu è un altro
mestiere. Il piatto deve venire *identico* la centesima e la cinquecentesima
volta; deve uscire dalla cucina in otto minuti, non in tre ore; deve reggere il
sabato sera con la sala piena; e se il fornitore cambia i pomodori, qualcuno se
ne deve accorgere prima che se ne accorga il cliente. Il notebook è la cena tra
amici. La produzione è il servizio in sala, tutte le sere, per anni.

`````

`````{tab} Superiore

In un prototipo conta quasi solo un **requisito funzionale**: il modello è
accurato? In produzione dominano i **requisiti non-funzionali**, che nella fase
di ricerca sono invisibili — affidabilità, latenza, throughput, scalabilità,
riproducibilità, manutenibilità nel tempo. Uno studio-intervista con
professionisti del settore {cite}`shankar2022operationalizing` sintetizza ciò
che separa i team che ci riescono in **tre V**:

- **Velocity** — la capacità di iterare in fretta: cambiare un'idea, riaddestrare
  e valutare in ore, non in settimane. È ciò che rende l'esplorazione produttiva.
- **Validation** — testare *presto e in automatico* dati, feature e pipeline, per
  intercettare gli errori prima che raggiungano gli utenti, non dopo.
- **Versioning** — conservare le versioni di codice, dati e modelli così da poter
  tornare indietro, confrontare e riprodurre qualunque risultato passato.

Queste tre spinte sono spesso in tensione: la Velocity preme per tagliare gli
angoli, la Validation e il Versioning per non tagliarli. L'ingegneria di un
sistema di ML è, in buona misura, l'arte di bilanciarle.

`````

## Il ciclo di vita, in concreto

Il primo malinteso da smontare è che addestrare il modello sia il cuore del
lavoro. Nel codice di un sistema di ML reale, la parte di apprendimento vero e
proprio è una frazione minima; tutto il resto è raccogliere dati, ripulirli,
trasformarli, distribuire il modello e sorvegliarlo. E soprattutto: non è una
linea retta con un traguardo, ma un **ciclo** che si percorre molte volte.

Uno studio di ingegneria del software condotto in Microsoft {cite}`amershi2019software`
mette in fila le fasi ricorrenti di un progetto di machine learning:

1. **Dati** — raccolta, pulizia, etichettatura. È dove si consuma la maggior
   parte del tempo, e dove nasce la maggior parte degli errori.
2. **Feature** — costruzione delle variabili di input a partire dai dati grezzi
   (*feature engineering*): la forma in cui il modello «vede» il mondo.
3. **Training** — l'addestramento vero e proprio. È il ciclo di ottimizzazione
   che abbiamo scritto a mano nel capitolo su PyTorch (si veda
   [Il training loop](../PyTorch/addestramento.md)): qui è solo *una* delle fasi.
4. **Valutazione** — la misura onesta delle prestazioni su dati mai visti, con la
   disciplina di train/validation/test già discussa nel capitolo sul machine
   learning (si veda [Overfitting e validazione](../MachineLearning/overfitting-validazione.md)).
5. **Deploy** — mettere il modello in un servizio che risponde a richieste reali,
   dietro un'API o dentro un'applicazione.
6. **Monitoraggio** — sorvegliare il modello in esercizio: le prestazioni
   reggono? I dati in ingresso somigliano ancora a quelli di addestramento?

La freccia importante è quella che torna indietro. Il monitoraggio scopre che il
mondo è cambiato — nuovi utenti, nuove parole, nuovi prodotti — e rimanda alla
raccolta di dati freschi; la valutazione insoddisfacente rimanda al feature
engineering o all'addestramento. Un sistema di ML non si «finisce»: si coltiva.
È la ragione per cui in produzione il lavoro non cala dopo il primo rilascio,
ma comincia davvero {cite}`huyen2022designing`.

## Riproducibilità: i tre artefatti da versionare

Rifare esattamente un risultato è la competenza fondativa di tutto il resto: se
non sai riprodurre un modello, non puoi confrontarlo con un altro, non puoi
correggerlo quando sbaglia, non puoi tornare alla versione buona quando la nuova
peggiora. E la riproducibilità in ML è più difficile che nel software normale,
perché un risultato non dipende solo dal codice.

`````{tab} Elementare

Una torta viene uguale a quella di ieri solo se tre cose coincidono: la
**ricetta** (i passaggi), gli **ingredienti** (con le dosi esatte) e il **forno**
(la stessa temperatura, lo stesso tempo). Sbaglia uno solo dei tre e il risultato
cambia. Nel software tradizionale, di solito, basta congelare la ricetta: stesso
codice, stesso risultato. Nel machine learning no: lo stesso codice, addestrato
su dati anche solo un po' diversi, produce un modello diverso. Per rifare la
torta servono tutti e tre gli elementi congelati — e, in più, va segnato pure il
lancio dei dadi, perché qui dentro c'è del caso.

`````

`````{tab} Superiore

Per riprodurre un modello occorre versionare **tre artefatti** distinti, più il
contesto in cui sono stati combinati:

- **Codice** — sorgente del modello, delle trasformazioni e della pipeline. Qui
  `git` fa benissimo il suo mestiere.
- **Dati** — l'esatto insieme di addestramento e valutazione. Si versiona
  fissandone un'**impronta** (l'hash del contenuto) e conservando lo *snapshot*
  in un archivio dedicato.
- **Modello** — i pesi addestrati (in PyTorch lo `state_dict` visto nel capitolo
  PyTorch), catalogati in un **model registry** che ne traccia versione, metriche
  e provenienza.

A questi si aggiungono l'**ambiente** — versioni esatte di Python e delle
librerie, *pinnate* (`pip freeze > requirements.txt`, un lockfile, un'immagine
container) perché una minor version diversa di una libreria può cambiare i
risultati — e la **configurazione**: iperparametri e, cruciale, i **semi**
casuali. Il punto delicato è che `git` da solo non basta: è pensato per file di
testo piccoli e diffabili, mentre dati e modelli sono grandi, binari e opachi.
Versionarli dentro un repository lo gonfia e lo rende inutilizzabile; per questo
si versiona nel repository un *puntatore* (l'hash, un percorso all'archivio) e si
tiene l'artefatto vero altrove.

`````

Il seme casuale merita una riga a parte, perché è la fonte di riproducibilità
più facile da dimenticare e più economica da fissare. Divisione dei dati,
inizializzazione dei pesi, ordine dei mini-batch: tutto attinge a un generatore
di numeri pseudo-casuali, e fissarne il seme rende la sequenza ripetibile.

```python
import random
import numpy as np


def fissa_seed(seed: int = 42) -> None:
    """Rende ripetibile ogni sorgente di casualita' del programma."""
    random.seed(seed)
    np.random.seed(seed)
    # in PyTorch si aggiungerebbe: torch.manual_seed(seed)
```

Fissare il seme non garantisce da solo la riproducibilità (restano di mezzo
versioni delle librerie e non-determinismi dell'hardware), ma è il primo,
irrinunciabile passo: senza, due esecuzioni dello stesso codice danno modelli
diversi, e ogni confronto perde di significato.

## Tracciare gli esperimenti

Durante l'esplorazione si provano decine, poi centinaia di configurazioni:
learning rate diversi, architetture diverse, feature diverse. Senza un registro,
dopo una settimana nessuno ricorda *quale* combinazione aveva dato quel 94%. L'
**experiment tracking** è la pratica di registrare, per ogni esecuzione (*run*),
gli iperparametri, le metriche ottenute e gli artefatti prodotti. Strumenti
come MLflow o Weights & Biases lo industrializzano, ma l'idea è indipendente dal
tool e sta in poche righe: associare a ogni configurazione un'identità stabile e
tenerne un registro.

Il cuore è un'**impronta riproducibile** della configurazione: uno stesso
insieme di iperparametri deve produrre sempre lo stesso identificativo, così da
riconoscere quando stiamo ripetendo un esperimento già fatto. Si ottiene con una
funzione di hash sul dizionario ordinato degli iperparametri.

```python
import hashlib
import json


def hash_config(iperparametri: dict) -> str:
    """Impronta stabile della configurazione: stesso dict -> stesso hash."""
    # sort_keys rende irrilevante l'ordine con cui scriviamo le chiavi
    canonico = json.dumps(iperparametri, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonico).hexdigest()[:12]


def logga_run(registro: dict, iperparametri: dict, metriche: dict) -> str:
    """Registra un esperimento indicizzandolo per impronta di configurazione."""
    run_id = hash_config(iperparametri)
    registro[run_id] = {
        "iperparametri": iperparametri,
        "metriche": metriche,
    }
    return run_id


# --- uso: un registro in memoria, serializzabile in JSON ---
registro = {}

run_a = logga_run(
    registro,
    iperparametri={"lr": 1e-3, "batch_size": 64, "epoche": 5, "seed": 42},
    metriche={"val_accuracy": 0.973, "val_loss": 0.089},
)

# stessa configurazione, chiavi scritte in ordine diverso -> stesso identico id
run_b = hash_config({"seed": 42, "epoche": 5, "batch_size": 64, "lr": 1e-3})

print(run_a)            # e4d5dc4d91ef
print(run_a == run_b)   # True: l'impronta non dipende dall'ordine delle chiavi
```

Il registro è un semplice dizionario, salvabile in un file JSON: nulla di magico.
Il valore non sta nella tecnologia ma nella disciplina — non lanciare *mai* un
addestramento senza che iperparametri, semi e metriche finiscano da qualche parte
che sopravviva alla chiusura del notebook. È la differenza tra un laboratorio con
i quaderni e uno dove si va a memoria.

## Il debito tecnico del machine learning

C'è un'ultima verità, la più scomoda, e la enuncia il paper che ha dato il nome
al problema: *Hidden Technical Debt in Machine Learning Systems*
{cite}`sculley2015hidden`, di un gruppo di Google nel 2015. La tesi è che i
sistemi di ML accumulano **debito tecnico** — le scorciatoie di oggi che si
pagano con gli interessi domani — più in fretta e in modi più insidiosi del
software normale.

`````{tab} Elementare

Il debito tecnico è come costruire una casa di fretta: per consegnare in tempo
salti qualche fondamenta, e per un po' la casa sta in piedi. Ma ogni scorciatoia
è un prestito: prima o poi va restituito, con gli interessi, sotto forma di crepe
da riparare. Il machine learning, dice quel paper, è «la carta di credito ad alto
tasso del debito tecnico»: fa spendere pochissimo oggi — un notebook, qualche
riga di *incollaggio* per far parlare i pezzi — e il conto arriva salatissimo
dopo, quando quei pezzi vanno mantenuti per anni. Il motivo profondo è che un
modello dipende dai **dati**, non solo dal codice: e i dati cambiano da soli,
senza che nessuno tocchi una riga. Cambiare *qualsiasi* cosa può cambiare
*tutto*.

`````

`````{tab} Superiore

Sculley e colleghi catalogano le forme di debito specifiche dell'ML, tra cui:

- **Glue code** — la valanga di codice di raccordo attorno a una libreria di ML
  generica: spesso il 95% di un sistema è incollaggio e solo il 5% è
  apprendimento vero.
- **Pipeline jungles** — trasformazioni dei dati che si stratificano fino a
  diventare grovigli impossibili da modificare in sicurezza.
- **Configuration debt** — la proliferazione di manopole (iperparametri, opzioni,
  soglie) senza controllo né validazione.
- **Data dependencies** — le dipendenze dai dati sono più insidiose di quelle dal
  codice, perché sono *silenziose*: nessun compilatore si lamenta se una feature
  a monte cambia distribuzione. Da qui il principio **CACE**: *Changing Anything
  Changes Everything*.

Come si misura se un sistema è pronto per la produzione? La **ML Test Score**
{cite}`breck2017ml` è una rubrica di 28 controlli concreti su quattro aree — dati
e feature, sviluppo del modello, infrastruttura, monitoraggio — e il voto
complessivo è dettato dall'area più debole: non basta un modello brillante se il
monitoraggio è assente. All'estremo opposto della maturità c'è la **Continuous
Delivery for Machine Learning** {cite}`sato2019continuous`, che estende al ML le
pratiche di consegna continua del software: automatizzare l'intero ciclo — dati,
training, valutazione, deploy — così che qualunque modello sia riproducibile e
rilasciabile in modo affidabile, in ogni momento, con un clic invece che con un
rito manuale.

`````

Nessuno di questi strumenti è un fine in sé. Servono a una cosa sola: fare in
modo che il modello del notebook di stamattina — quello che «funzionava sul mio
computer» — continui a funzionare domani, sul computer di tutti, e che tra sei
mesi qualcuno possa capire *perché* funzionava e rifarlo daccapo. È il passaggio
dalla dimostrazione al prodotto: meno spettacolare della prima intuizione, ma è
qui che la ricerca diventa qualcosa su cui le persone possono contare.

```{admonition} Da ricordare
:class: important
- Tra un **notebook** e la **produzione** c'è un abisso: cambiano i requisiti
  non-funzionali — affidabilità, latenza, scala, riproducibilità,
  manutenibilità — invisibili nella fase di ricerca.
- I team che ci riescono bilanciano tre spinte {cite}`shankar2022operationalizing`:
  **Velocity** (iterare in fretta), **Validation** (testare presto e in
  automatico), **Versioning** (conservare le versioni).
- Il **ciclo di vita** — dati, feature, training, valutazione, deploy,
  monitoraggio {cite}`amershi2019software` — non è una retta ma un anello: il
  monitoraggio rimanda ai dati, un sistema di ML si coltiva.
- La **riproducibilità** richiede tre artefatti versionati — **codice, dati,
  modello** — più ambiente e semi casuali. `git` da solo non basta: dati e
  modelli sono grandi e binari, se ne versiona un'impronta.
- L'**experiment tracking** registra iperparametri, metriche e artefatti di ogni
  run: un'impronta riproducibile della configurazione basta a riconoscere gli
  esperimenti già fatti.
- Il ML accumula **debito tecnico** in fretta {cite}`sculley2015hidden` (glue
  code, pipeline jungle, dipendenze dai dati); la maturità si misura con rubriche
  come la **ML Test Score** {cite}`breck2017ml` e si automatizza con la CD4ML
  {cite}`sato2019continuous`.
```
