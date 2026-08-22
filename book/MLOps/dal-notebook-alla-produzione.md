# Dal notebook alla produzione

Il notebook ha quarantasette **celle**, i pezzetti in cui il programma è
spezzato e che si possono eseguire uno alla volta, in qualunque ordine. A lato
di ciascuna un numero fra parentesi quadre dice quando è stata eseguita
l'ultima volta, e quei numeri raccontano una storia sconfortante: `[12]`, poi
`[8]`, poi `[31]`, poi di nuovo `[9]`. Le celle sono state eseguite in ordine
sparso, avanti e indietro, per giorni, e il risultato di ieri sera esiste
soltanto nella memoria di quella sessione, non nel file. Ieri sera il modello
dava un'accuratezza del 94% e l'autrice è andata a dormire soddisfatta.
Stamattina una collega apre lo stesso file e preme *Restart & Run All*, che
butta via quella memoria e riesegue tutto dall'alto, come farebbe una macchina
che non sa nulla della cronologia. Metà delle celle esplode. Una cerca un
risultato che era stato calcolato in una cella poi cancellata, e che quindi
adesso non esiste più. Una cerca un file in una cartella che esiste solo su
quel portatile.

E poi c'è quella che non esplode affatto, ed è la peggiore. Divide gli esempi
fra dati di addestramento e dati di prova tirandoli a sorte
(`train_test_split`), e funziona benissimo: solo che la sorte ogni volta cade
diversa, perché nessuno le ha detto da dove partire, e il 94% di ieri sera non
si rivede più. La frase che chiude la giornata è la più celebre della
disciplina: «Ma sul mio computer funzionava».

Attenzione: il disordine di quelle celle non è un peccato in sé. Mentre si
esplora, saltare avanti e indietro è esattamente il modo giusto di lavorare, ed
è per questo che il notebook esiste. Il peccato è **consegnare** quel disordine,
cioè lasciare che il risultato buono viva soltanto nella memoria di una
sessione che qualcuno prima o poi chiuderà.

Fra quel notebook e un sistema che dà previsioni a persone vere, ogni giorno,
senza sorprese, c'è quindi un abisso. Colmarlo è il mestiere dell'**MLOps**. Che cosa cambia davvero, passando «dal
mio computer» al mondo, e con quali attrezzi si attraversa: si comincia da
qui.

## Il divario ricerca–produzione

Un modello che «funziona» in un notebook ha risolto una piccola parte del
problema. La parte grande (quella che riempie gli anni di lavoro di un team)
comincia dopo: farlo girare in modo affidabile, abbastanza in fretta, per
tanti utenti, e riuscire a rifare esattamente lo stesso risultato tra sei
mesi, quando nessuno ricorda più quali dati e quale versione del codice
l'avevano prodotto.

`````{tab} Elementare

Una sera cucini un piatto per gli amici. Puoi assaggiare, aggiustare di sale a
occhio, ripetere se viene male: nessuno ti cronometra. Metterlo nel menu di un
ristorante è un altro mestiere. Il piatto deve venire *identico* la centesima e
la cinquecentesima volta; deve uscire dalla cucina in otto minuti, non in tre
ore; deve reggere il sabato sera con la sala piena; e se il fornitore cambia i
pomodori, qualcuno se ne deve accorgere prima che se ne accorga il cliente. I
fogli delle ricette poi si tengono tutti, non l'ultimo soltanto: quando il
piatto di questa settimana piace meno di quello di un mese fa, si ripesca il
foglio di allora e si torna a com'era. Provare varianti in fretta, controllare
ogni fornitura prima di usarla e tenere in ordine il raccoglitore si contendono
le stesse ore, e la sera in cui la sala è piena una delle tre si salta: quale,
lo decide chi conduce la cucina. Il notebook è la cena tra amici. La produzione
è il servizio in sala, tutte le sere, per anni.

`````

`````{tab} Superiore

In un prototipo conta quasi solo un **requisito funzionale**: il modello è
accurato? In produzione dominano i **requisiti non-funzionali**, che nella
fase di ricerca sono invisibili: affidabilità, latenza, throughput,
scalabilità, riproducibilità, manutenibilità nel tempo. Uno studio-intervista
con professionisti del settore riassume in **tre V** ciò che separa i team che
ci riescono {cite}`shankar2022operationalizing`:

- **Velocity**, la capacità di iterare in fretta: cambiare un'idea,
  riaddestrare e valutare in ore, non in settimane. È ciò che rende
  l'esplorazione produttiva.
- **Validation**: testare *presto e in automatico* dati, feature e pipeline,
  per intercettare gli errori prima che raggiungano gli utenti, non dopo.
- **Versioning**: conservare le versioni di codice, dati e modelli così da
  poter tornare indietro, confrontare e riprodurre qualunque risultato
  passato.

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

Prima di guardare il ciclo intero, però, conviene guardare da vicino un suo
pezzo, quello che si ripete più spesso ({numref}`fig-cicd-ml`): il viaggio che
compie **una singola modifica**, da quando qualcuno la propone a quando finisce
sotto gli occhi del pubblico. Quel viaggio, preso da solo, è dritto.

```{figure} ../figures/cicd-machine-learning.svg
:name: fig-cicd-ml
:alt: "Catena automatica per il machine learning: da una proposta di modifica (una pull request) si passa alla prova automatica del codice, poi all'addestramento del modello, alla valutazione contro una soglia e infine al rilascio. Se la valutazione non supera la soglia la catena si ferma e il modello non viene rilasciato."
:width: 100%

Il percorso che una modifica compie prima di andare in pubblico. Qualcuno
propone un cambiamento, una macchina prova il codice, poi riaddestra il
modello e gli dà un voto su esempi che non ha mai visto. Solo se quel voto
supera la soglia decisa in anticipo, il cambiamento viene accettato e
pubblicato. Nella figura il voto è l’`F1`, uno dei modi di dare un numero solo
alla bravura di un classificatore, visto nel capitolo sul machine learning; va
bene qualunque altro, purché la soglia sia stata scritta *prima*. Rispetto al
software normale lo stadio in più è proprio quello del voto, ed è un cancello
che può dire di no.
```

Lo stadio aggiunto in {numref}`fig-cicd-ml` è la differenza fra il rilascio di
un programma e quello di un modello. Di un programma rotto il computer si
accorge da solo e si rifiuta di partire; un modello no: un modello risponde
comunque, con la stessa aria sicura, anche quando risponde peggio di quello di
prima. Senza una soglia scritta prima, non c'è modo automatico di
accorgersene.

Uno studio di ingegneria del software condotto in Microsoft mette in fila nove
fasi ricorrenti di un progetto di machine learning
{cite}`amershi2019software`, che qui raggruppiamo in sei momenti:

1. **Dati**: raccolta, pulizia, etichettatura. È dove si consuma la maggior
   parte del tempo, e dove nasce la maggior parte degli errori.
2. **Feature**, costruzione delle variabili di input a partire dai dati grezzi
   (*feature engineering*): la forma in cui il modello «vede» il mondo.
3. **Training**: l'addestramento vero e proprio. È il ciclo di ottimizzazione
   che abbiamo scritto a mano nel {doc}`capitolo su PyTorch </PyTorch/overview>` (si veda [Il training
   loop](../PyTorch/addestramento.md)): qui è solo *una* delle fasi.
4. **Valutazione**: la misura onesta delle prestazioni su dati mai visti, con
   la disciplina di train/validation/test già discussa nel capitolo sul
   machine learning (si veda [Overfitting e
   validazione](../MachineLearning/overfitting-validazione.md)).
5. **Deploy**: mettere il modello in un servizio che risponde a richieste
   reali, dietro un’**API** o dentro un'applicazione. Un'API è una specie di
   sportello elettronico: un indirizzo a cui un altro programma manda una
   domanda e da cui riceve la risposta, senza sapere né dover sapere che cosa
   c'è dietro.
6. **Monitoraggio**, sorvegliare il modello in esercizio: le prestazioni
   reggono? I dati in ingresso somigliano ancora a quelli di addestramento?

La freccia importante è quella che torna indietro. Il monitoraggio scopre che
il mondo è cambiato (nuovi utenti, nuove parole, nuovi prodotti) e rimanda
alla raccolta di dati freschi; la valutazione insoddisfacente rimanda al
feature engineering o all'addestramento. Un sistema di ML non si «finisce»: si
coltiva. È la ragione per cui in produzione il lavoro non cala dopo il primo
rilascio, ma comincia davvero {cite}`huyen2022designing`.

## Riproducibilità: i tre artefatti da versionare

Rifare esattamente un risultato è la competenza su cui poggia tutto il resto.
Se non sai riprodurre un modello non puoi confrontarlo con un altro, non puoi
correggerlo quando sbaglia, e non puoi tornare a quello buono quando il nuovo
peggiora. E qui è più difficile che nel software normale, perché il risultato
non dipende solo dal codice.

Le cose da conservare sono gli **artefatti** nominati nella pagina d'apertura,
e conservarne ogni versione invece dell'ultima soltanto è ciò che in gergo si
dice **versionare**: la parola torna in tutto il capitolo, e vuol dire questo e
nient'altro.

`````{tab} Elementare

Una torta viene uguale a quella di ieri solo se tre cose coincidono: la
**ricetta** (i passaggi), gli **ingredienti** (con le dosi esatte) e il
**forno** (la stessa temperatura, lo stesso tempo). Sbaglia uno solo dei tre e
il risultato cambia. Nel software tradizionale, di solito, basta congelare la
ricetta: stesso codice, stesso risultato. Nel machine learning no: lo stesso
codice, addestrato su dati anche solo un po’ diversi, produce un modello
diverso. Per rifare la torta servono tutti e tre gli elementi congelati, e, in
più, va segnato pure il lancio dei dadi, perché qui dentro c'è del caso.

Tradotta dalla cucina al mestiere, l'analogia dice così: la ricetta è il
programma, gli ingredienti sono i dati, e la torta è il modello addestrato. Il
forno è il computer con sopra le sue **librerie**, cioè i pacchi di programmi
già fatti che il nostro programma usa senza riscriverli (uno che sa fare i
conti sui numeri, uno che sa addestrare le reti); e come un forno vero, una
libreria di marca diversa, o della stessa marca ma di un altro anno, cuoce in
modo un po’ diverso.

E la torta si conserva anche lei, non solo la ricetta: rifarla identica costa
ore di forno, e chi la deve mangiare non può aspettarle. Da tenere sotto
chiave, allora, sono **codice, dati e modello**, con il forno (le librerie)
come condizione da non dimenticare. Nel quaderno delle ricette, però, non ci
stanno i sacchi di farina né le torte: chi ce li forzasse dentro avrebbe un
quaderno che non si sfoglia più. Sul quaderno va il cartellino, quel sacco lì,
di quel lotto; la roba vera sta in dispensa e in freezer. Chi scrive «farina» e
basta, un anno dopo la torta non la sa più rifare.

`````

`````{tab} Superiore

Per riprodurre un modello occorre versionare **tre artefatti** distinti, più il
contesto in cui sono stati combinati:

- **Codice**: sorgente del modello, delle trasformazioni e della pipeline. Qui
  `git` fa benissimo il suo mestiere.
- **Dati**: l'esatto insieme di addestramento e valutazione. Si versiona
  fissandone un’**impronta** (l'hash del contenuto) e conservando lo
  *snapshot* in un archivio dedicato.
- **Modello**: i pesi addestrati (in PyTorch lo `state_dict` visto nel
  capitolo PyTorch), catalogati in un **model registry** che ne traccia
  versione, metriche e provenienza.

A questi si aggiungono l’**ambiente**, versioni esatte di Python e delle
librerie, *pinnate* (`pip freeze > requirements.txt`, un lockfile, un'immagine
container), perché una minor version diversa di una libreria può cambiare i
risultati, e la **configurazione**: iperparametri e, cruciale, i **semi**
casuali. Il punto delicato è che `git` da solo non basta: è pensato per file
di testo piccoli e diffabili, mentre dati e modelli sono grandi, binari e
opachi. Versionarli dentro un repository lo gonfia e lo rende inutilizzabile;
per questo si versiona nel repository un *puntatore* (l'hash, un percorso
all'archivio) e si tiene l'artefatto vero altrove.

`````

Il seme casuale merita una riga a parte, perché è la fonte di riproducibilità
più facile da dimenticare e più economica da fissare. Il computer i dadi li
tira per finta: segue una lista di numeri preparata in anticipo, e il **seme**
è il punto della lista da cui parte. Fissarlo vuol dire far uscire sempre gli
stessi dadi. E i dadi qui si tirano almeno in tre punti: quando i dati si
dividono fra addestramento e prova, quando i pesi della rete ricevono i loro
valori iniziali, e quando gli esempi vengono mescolati prima di essere dati in
pasto al modello un gruppetto alla volta (i *mini-batch*).

```python
import random

import numpy as np
import torch


def fissa_seed(seed: int = 42) -> None:
    """Fissa le sorgenti di casualita' che il libro usa davvero."""
    random.seed(seed)
    np.random.seed(seed)      # sorgente "legacy" di NumPy
    torch.manual_seed(seed)   # pesi iniziali, dropout, DataLoader che mescola
    # i Generator moderni di NumPy ricevono il seme alla creazione:
    #   rng = np.random.default_rng(seed)
    # e un DataLoader che mescola vuole il proprio generatore, piu' un
    # worker_init_fn se num_workers > 0:
    #   DataLoader(dati, shuffle=True,
    #              generator=torch.Generator().manual_seed(seed))
```

Fissare il seme è il primo passo, non l'ultimo. Restano di mezzo le versioni
delle librerie, che cambiando cambiano i risultati, e un fatto sorprendente
dell'aritmetica dei calcolatori: **sommare gli stessi numeri in ordine diverso
non dà esattamente lo stesso totale**. Si può verificare in tre secondi con una
calcolatrice che non sia quella della mente: chiedendo a Python
`(0.1 + 0.2) + 0.3` si ottiene `0.6000000000000001`, mentre
`0.1 + (0.2 + 0.3)` dà `0.6` tondo. Gli addendi sono gli stessi, il totale no.
Il motivo è che i numeri con la virgola vengono arrotondati a ogni passaggio, e
un arrotondamento fatto prima o dopo non lascia lo stesso residuo.

Ora, l'ordine in cui una libreria combina milioni di numeri non è sempre lo
stesso: dipende da quanti esempi viaggiano insieme, da quale ricetta interna la
libreria sceglie per quella forma di dati, da quanti processori se lo dividono.
Le ultime cifre ballano. In produzione ballano di più, perché lì quanti esempi
viaggiano insieme lo decide il servizio momento per momento (la sezione sul
deployment lo chiamerà *batching dinamico*, e a questa ripetibilità rinuncia
per scelta).

Conviene allora distinguere due promesse diverse, perché costano diversamente.

La prima è la **riproducibilità bit a bit**: due esecuzioni che danno numeri
identici fino all'ultima cifra. Si può avere, ma si paga. Bisogna dare un seme
a ogni sorgente di casualità, chiedere esplicitamente alla libreria di usare
solo procedimenti che a parità di ingressi danno sempre la stessa uscita
(`torch.use_deterministic_algorithms(True)`), rinunciare a caricare i dati con
più processi in parallelo, e accettare di andare più piano.

La seconda è la **riproducibilità statistica**: i numeri non coincidono
all'ultima cifra, ma le differenze restano dentro il ballo naturale delle
ultime cifre, e le conclusioni non cambiano. È quella che serve quasi sempre,
ed è quella che seme, ambiente congelato e dati versionati consegnano davvero.
Senza nemmeno il seme, però, non si ha né l'una né l'altra: due esecuzioni
dello stesso codice danno modelli diversi, e ogni confronto perde di
significato.

## Tracciare gli esperimenti

Durante l'esplorazione non si prova una configurazione sola: se ne provano
decine, poi centinaia. Si cambia la velocità con cui il modello impara, si
cambia la forma della rete, si cambia quali informazioni le si danno in pasto.
E senza un registro, dopo una settimana, nessuno ricorda più *quale*
combinazione aveva dato quel 94%.

La cura è segnare ogni prova: che cosa si era impostato prima di lanciarla e
com'è andata dopo. In gergo si chiama **experiment tracking**, e una singola
prova è una *run*. Esistono servizi che la industrializzano, con interfacce e
grafici, ma l'idea non dipende da nessuno di loro e sta in poche righe.

`````{tab} Elementare

Cambi un'impostazione, riprovi, va meglio. Ne cambi un'altra, riprovi, va
peggio. Dopo cento giri hai un numero buono in mano e non sai più a quale
combinazione appartenga: rifarlo a memoria non funziona, perché le prove si
somigliano tutte.

La cura non è un attrezzo, è un'abitudine. Prima di lanciare una prova, scrivere
da qualche parte che cosa si è impostato; a prova finita, scrivere com'è andata.
«Da qualche parte» vuol dire in un posto che sopravviva alla chiusura del
programma, non in una cella del notebook.

Serve poi un modo per dare a ogni combinazione un nome corto e sempre uguale,
così da accorgersi di stare rifacendo una prova già fatta. Il modo è un
tritatutto: si passa dentro l'elenco delle impostazioni e ne esce un codice
corto, completamente diverso appena una cifra cambia. Le impostazioni si
mettono in fila in ordine alfabetico prima di buttarle dentro, così chi le ha
scritte in un ordine e chi in un altro ottiene lo stesso codice. Per il resto,
però, il tritatutto è letterale: legge quello che c'è scritto, non quello
che si intendeva. «5» e «5,0» sono lo stesso numero di giri e danno due codici
diversi, quindi i numeri vanno scritti sempre allo stesso modo, o si rifà una
prova credendo che sia nuova. È lo stesso attrezzo che serve a mettere un
cartellino a un intero archivio di dati.

`````

`````{tab} Superiore

Il cuore è un’**impronta** della configurazione: si serializza il dizionario
degli iperparametri in una forma canonica e se ne prende un hash, così da
riconoscere quando stiamo ripetendo un esperimento già fatto. `sort_keys=True`
è ciò che rende irrilevante l'ordine in cui le chiavi sono state scritte, ed è
la proprietà che l'esempio dimostra.

La promessa larga («stessa configurazione, stesso identificativo») non è però
quella che il codice consegna: la stabilità di quell'impronta ha un perimetro
stretto. `json.dumps` conserva la **rappresentazione** dei valori, non il loro
valore numerico: `epoche=5` ed `epoche=5.0` danno due
impronte diverse pur essendo lo stesso esperimento, una tupla e una lista si
serializzano uguali e quindi collidono, e un valore non serializzabile (un
`torch.dtype`, una classe) solleva un'eccezione. In un impianto vero i valori
si normalizzano prima di serializzarli; qui l'impronta è stabile **rispetto
all'ordine delle chiavi**, che è già sufficiente a riconoscere il duplicato più
frequente, cioè la stessa configurazione riscritta in un altro ordine.

`````

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

Le manopole che si scelgono prima di lanciare una prova hanno un nome: sono
gli **iperparametri**, cioè i numeri che decide una persona e che il modello
non impara da sé (quanti esempi per volta, con che velocità imparare, per
quanti giri).

Il registro, poi, è una semplice rubrica in memoria, che si salva su disco in un
file di testo: nulla di magico. Il valore non sta nella tecnologia ma nella
disciplina: non lanciare *mai* un addestramento senza che iperparametri, semi e
risultati finiscano da qualche parte che sopravviva alla chiusura del notebook.
È la differenza tra un laboratorio con i quaderni e uno dove si va a memoria.

## Il debito tecnico del machine learning

C'è un'ultima verità, la più scomoda, e la enuncia il paper che ha dato il
nome al problema: *Hidden Technical Debt in Machine Learning Systems*
{cite}`sculley2015hidden`, di un gruppo di Google nel 2015. La tesi è che i
sistemi di ML accumulano **debito tecnico** (le scorciatoie di oggi che si
pagano con gli interessi domani) più in fretta e in modi più insidiosi del
software normale.

`````{tab} Elementare

Il debito tecnico è come costruire una casa di fretta: per consegnare in tempo
salti qualche fondamenta, e per un po’ la casa sta in piedi. Ma ogni
scorciatoia è un prestito: prima o poi va restituito, con gli interessi, sotto
forma di crepe da riparare. Lo stesso gruppo, un anno prima, aveva intitolato
un articolo così: il machine learning è «la carta di credito ad alto tasso del
debito tecnico». Oggi basta un notebook e qualche riga scritta di fretta per
collegare fra loro i pezzi già pronti che si sono messi insieme: righe che non
fanno niente di intelligente, servono solo a far combaciare l'uscita di un
pezzo con l'ingresso del successivo, come il nastro adesivo che tiene su un
impianto. Si scrivono in un pomeriggio, poi vanno mantenute per anni, e intanto
diventano tantissime: tubi aggiunti uno sopra l'altro, con rubinetti che
nessuno sa più a che cosa servano e che nessuno osa chiudere.

E c'è un motivo più profondo: un modello dipende dai **dati**, non solo dal
codice, e dentro un modello nulla è separato da nulla. Basta cambiare una delle
informazioni che gli si danno in pasto perché il modello rifaccia i suoi
equilibri e sposti le risposte anche dove nessuno se lo aspettava. Cambiare
*qualsiasi* cosa può cambiare *tutto*. E i dati, a differenza del codice,
cambiano da soli, senza che nessuno tocchi una riga.

Chi compra una casa così manda un perito, e il perito non fa la media delle
stanze: guarda quella messa peggio. Muri perfetti e impianto elettrico fuori
norma fanno una casa fuori norma. E il sopralluogo non si fa una volta sola a
lavori finiti: si rifà a ogni modifica, e a farlo è una macchina che non si
stanca.

`````

`````{tab} Superiore

Sculley e colleghi catalogano le forme di debito specifiche dell'ML, tra cui:

- **Glue code**, la valanga di codice di raccordo attorno a una libreria di ML
  generica. Gli autori arrivano a dire che un sistema maturo può ritrovarsi con
  al massimo il 5% di codice di apprendimento e almeno il 95% di incollaggio:
  non è una misura, è l'ordine di grandezza a cui vogliono che si pensi.
- **Pipeline jungles**: trasformazioni dei dati che si stratificano fino a
  diventare grovigli impossibili da modificare in sicurezza.
- **Configuration debt**: la proliferazione di manopole (iperparametri,
  opzioni, soglie) senza controllo né validazione.
- **Data dependencies**, le dipendenze dai dati sono più insidiose di quelle
  dal codice, perché sono *silenziose*: nessun compilatore si lamenta se una
  feature a monte cambia distribuzione. Da qui il principio **CACE**:
  *Changing Anything Changes Everything*.

Come si misura se un sistema è pronto per la produzione? Una rubrica nota come
**ML Test Score** mette in fila 28 controlli concreti su quattro aree (dati e
feature, sviluppo del modello, infrastruttura, monitoraggio)
{cite}`breck2017ml`. Il voto complessivo è dettato dall'area più debole: non
basta un modello brillante se il monitoraggio è assente. All'estremo opposto
della maturità sta la **Continuous Delivery for Machine Learning**
{cite}`sato2019continuous`, che estende al ML le pratiche di consegna continua
del software: automatizzare l'intero ciclo (dati, training, valutazione,
deploy), così che qualunque modello sia riproducibile e rilasciabile in modo
affidabile, in ogni momento, con un comando invece che con un rito manuale.

`````

Nessuno di questi strumenti è un fine in sé. Servono a una cosa sola: fare in
modo che il modello del notebook di stamattina (quello che «funzionava sul mio
computer») continui a funzionare domani, sul computer di tutti, e che tra sei
mesi qualcuno possa capire *perché* funzionava e rifarlo daccapo. È il
passaggio dalla dimostrazione al prodotto: meno spettacolare della prima
intuizione, ma è qui che la ricerca diventa qualcosa su cui le persone possono
contare.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Fra il **notebook** e la **produzione** c'è un abisso, ed è quello fra
  cucinare un piatto una sera per gli amici e metterlo nel menù: in
  esplorazione conta solo che venga buono, in servizio deve venire identico la
  centesima volta, uscire in fretta e reggere il sabato sera.
- Rifare esattamente un risultato è la competenza su cui poggia tutto il resto:
  se non sai rifare un modello non puoi confrontarlo, correggerlo, né tornare
  a quello buono quando il nuovo peggiora.
- Le cose da conservare sono **tre** (il programma, i dati, il modello), e a
  queste tre vanno aggiunte due condizioni: le **librerie** con cui si è
  cucinato e il **seme**, cioè il punto da cui parte il sorteggio. Fissare il
  seme è il primo passo e non basta da solo: due
  esecuzioni possono ancora differire nelle ultime cifre, e va bene così,
  purché le conclusioni non cambino.
- **Segnare ogni prova** appena la si lancia: che cosa si era impostato e com'è
  andata, in un posto che sopravviva alla chiusura del programma. È la
  differenza fra un laboratorio con i quaderni e uno dove si va a memoria.
- Le scorciatoie prese oggi si pagano con gli interessi domani (il **debito
  tecnico**), e in questo mestiere si pagano più care, perché un modello
  dipende dai dati e i dati cambiano da soli, senza che nessuno tocchi una
  riga.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Tra un **notebook** e la **produzione** c'è un abisso: cambiano i requisiti
  non-funzionali (affidabilità, latenza, scala, riproducibilità,
  manutenibilità) invisibili nella fase di ricerca.
- I team che ci riescono bilanciano tre spinte {cite}`shankar2022operationalizing`:
  **Velocity** (iterare in fretta), **Validation** (testare presto e in
  automatico), **Versioning** (conservare le versioni).
- Il **ciclo di vita** (dati, feature, training, valutazione, deploy,
  monitoraggio {cite}`amershi2019software`) non è una retta ma un anello: il
  monitoraggio rimanda ai dati, un sistema di ML si coltiva.
- La **riproducibilità** richiede tre artefatti versionati (**codice, dati,
  modello**) più ambiente e semi casuali. `git` da solo non basta: dati e
  modelli sono grandi e binari, se ne versiona un'impronta. Distinguere la
  riproducibilità **bit a bit** (che si paga in prestazioni e che il batching
  dinamico rinuncia a dare) da quella **statistica**, che è quella che serve.
- L’**experiment tracking** registra iperparametri, metriche e artefatti di ogni
  run: un'impronta della configurazione, stabile rispetto all'ordine delle
  chiavi, riconosce la stessa configurazione riscritta in un altro ordine, non
  ogni duplicato (`epoche=5` ed `epoche=5.0` danno due impronte diverse).
- Il ML accumula **debito tecnico** in fretta {cite}`sculley2015hidden` (glue
  code, pipeline jungle, dipendenze dai dati); la maturità si misura con rubriche
  come la **ML Test Score** {cite}`breck2017ml` e si automatizza con la CD4ML
  {cite}`sato2019continuous`.
```
`````
