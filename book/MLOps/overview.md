# MLOps: mettere i modelli in produzione

C'è una figura, in un articolo del 2015, che vale da sola un capitolo intero.
La disegnano alcuni ingegneri di Google (David Sculley e colleghi) per una
conferenza di apprendimento automatico {cite}`sculley2015hidden`. Al centro
del foglio c'è un rettangolino nero, minuscolo, con dentro due parole: *codice
ML*. È l'unico pezzo di cui parlano di solito i libri, i corsi, i paper: il
modello, l'algoritmo, la rete che impara. Tutto intorno, a soffocarlo, ci sono
scatole molto più grandi: raccolta dati, verifica dei dati, estrazione delle
*feature*, gestione della configurazione, infrastruttura di *serving*,
monitoraggio, strumenti di analisi, gestione delle risorse di calcolo. La
morale della figura è brutale e onesta: addestrare il modello è la parte più
piccola del lavoro. Tutto il resto (il grosso) è il sistema che gli sta
intorno.

Non è un dettaglio da ingegneri pignoli. Nel settore si ripete spesso che una
quota altissima di modelli non arrivi mai in produzione, e che molti di quelli
che ci arrivano inciampino proprio nel passaggio finale: un'osservazione
diffusa, più aneddotica che misurata. Ma il problema di fondo è documentato:
una rassegna di casi reali {cite}`paleyes2022challenges` lo ricostruisce
progetto per progetto, mostrando quanti ostacoli costellino *ogni* tappa del
percorso che porta un modello dal prototipo al servizio (raccolta e verifica
dei dati, deployment, monitoraggio, manutenzione). Non perché i modelli
sbaglino le predizioni in laboratorio, ma perché nessuno aveva pensato a come
alimentarli, aggiornarli, sorvegliarli. Questo capitolo parla esattamente di
quel «tutto il resto». Il suo nome, ormai, è **MLOps**.

## Il modello è la punta dell'iceberg

L'errore di prospettiva è comprensibile: fino a qui, in tutto il libro, «fare
machine learning» ha significato scegliere un modello, addestrarlo, misurarne
l'accuratezza. In un notebook, sul proprio computer, il lavoro sembra finito
quando la metrica sul *test set* è buona. Ma un modello che nessuno usa non
serve a niente, e la distanza tra «funziona nel mio notebook» e «funziona per
migliaia di persone, ogni giorno, per anni» è enorme, ed è quasi tutta fuori
dal modello.

`````{tab} Elementare

Pensa a un ristorante. Il cuoco che inventa un piatto memorabile ha fatto una
cosa difficilissima e necessaria, ma non ha ancora aperto il ristorante. Per
servire quel piatto a cento clienti a sera servono una cucina attrezzata, un
magazzino con le forniture che non finiscono, camerieri, un sistema per
prendere le comande, l'igiene a norma, i conti che tornano. Se il freezer si
rompe o il fornitore non consegna, il piatto più buono del mondo non arriva al
tavolo.

Il modello addestrato è il piatto: la ricetta funziona. MLOps è tutto il resto
del ristorante. E come in un ristorante, il grosso dei guai non viene dalla
ricetta: viene dal magazzino che si svuota, dal fornitore che cambia gli
ingredienti, dalla cucina che va tenuta pulita ogni giorno.

`````

`````{tab} Superiore

Sculley e colleghi {cite}`sculley2015hidden` inquadrano il fenomeno con il
linguaggio del **debito tecnico**: come nel software tradizionale scorciatoie
prese in fretta si pagano con gli interessi più avanti, un sistema di
apprendimento automatico accumula un debito *aggiuntivo* e più insidioso,
perché nasconde la sua complessità nei **dati**, non solo nel codice. Il
sistema in produzione comprende pipeline di raccolta e validazione dei dati,
estrazione e trasformazione delle *feature*, un livello di *serving* che
espone il modello, monitoraggio, gestione della configurazione e delle
risorse: il codice di addestramento è una frazione minima del totale.

Il debito più caratteristico è l'**entanglement**, riassunto dal principio
**CACE**: *Changing Anything Changes Everything*. In un modello di ML nessuna
*feature* è davvero indipendente: cambiare la distribuzione di un solo
ingresso, aggiungerne o toglierne uno, ritoccare un iperparametro ripesa tutti
gli altri e sposta le predizioni ovunque, in modi non locali e difficili da
prevedere. È l'opposto della modularità a cui l'ingegneria del software
tradizionale ci ha abituati, ed è la ragione per cui un sistema di ML non si
governa con le sole pratiche del software classico.

`````

## Che cos'è MLOps

Il nome è un calco su **DevOps**, la cultura che nell'ingegneria del software
ha unito sviluppo (*Dev*) e gestione operativa (*Ops*) in un flusso unico e
automatizzato: controllo di versione, integrazione e distribuzione continue
(CI/CD), monitoraggio, capacità di rilasciare spesso e senza patemi. MLOps
prende quella cultura e la porta al ciclo di vita del machine learning
{cite}`kreuzberger2023machine`. Con una complicazione in più, che è il cuore
di tutto: nel software classico l'artefatto da versionare è il codice; qui gli
artefatti sono **tre**, e due di loro (i dati e il modello), non sono testo.

`````{tab} Elementare

C'è una differenza tra saper cucinare e saper mandare avanti una catena di
ristoranti. Il cuoco tiene la ricetta in testa, aggiusta a occhio, e ogni suo
piatto è un po' diverso dal precedente, ed è bellissimo così. Ma una catena
che serve mille coperti al giorno in venti città non può permetterselo: la
ricetta va scritta al grammo, gli ingredienti devono essere sempre gli stessi,
e il piatto di oggi a Milano deve essere identico a quello di ieri a Napoli.
Serve **riproducibilità**: chiunque, seguendo la procedura, riottiene lo
stesso risultato.

MLOps è il passaggio dal cuoco geniale alla catena affidabile. Non toglie
nulla alla bravura di chi inventa il modello: aggiunge la disciplina che
permette di rifarlo uguale, di aggiornarlo senza rompere niente e di
accorgersi in tempo se qualcosa va storto.

`````

`````{tab} Superiore

La definizione operativa {cite}`kreuzberger2023machine` poggia su una
tripletta versionata: **dati + codice + modello**. Rendere un esperimento
riproducibile significa poter ricostruire una predizione a partire da (a) la
versione esatta del dataset di addestramento, (b) la versione del codice e
degli iperparametri, (c) i pesi del modello che ne sono risultati. Da qui le
pratiche cardine: *data versioning* e *feature store* per gli ingressi,
*experiment tracking* per legare metriche e configurazioni, *model registry*
per i modelli, pipeline automatizzate che rieseguono l'intero percorso (da
dato grezzo a modello servito) con un comando solo.

L'obiettivo non è la sofisticazione, ma l'**automazione** e la
**tracciabilità**: ridurre il lavoro manuale che si fa a mano ogni volta,
rendere ogni rilascio ripetibile e ogni predizione riconducibile agli
artefatti che l'hanno prodotta. È la tesi di fondo dei testi che hanno
sistematizzato la disciplina, da *Designing Machine Learning Systems*
{cite}`huyen2022designing` in poi: un modello in produzione non è un risultato,
è un **processo** da tenere in vita.

`````

## Il ciclo di vita di un modello

Ed è proprio la parola «processo» a segnare la differenza più importante.
Siamo abituati a pensare al machine learning come a una linea retta: si
raccolgono i dati, si addestra, si valuta, si consegna. Fine. Ma la consegna
non è la fine: è il punto in cui il modello incontra il mondo reale, e il
mondo reale cambia. Un modello in produzione va **sorvegliato**, e prima o poi
i dati che incontra smettono di somigliare a quelli su cui è stato addestrato:
il *drift* che avevamo incontrato nella sezione «Quando i dati cambiano» del
capitolo di Machine Learning, dove lo abbiamo inquadrato in termini
statistici. Quando succede, si torna all'inizio: nuovi dati, nuovo
addestramento. Il percorso non è una linea, è un **anello**
({numref}`fig-mlops-ciclo-vita`).

```{figure} ../figures/mlops-ciclo-vita.svg
:name: fig-mlops-ciclo-vita
:alt: Cinque nodi disposti in cerchio e collegati da frecce in senso orario (Dati, Addestramento, Valutazione, Deploy, Monitoraggio) formano un anello chiuso; una freccia terracotta tratteggiata chiude il ciclo riportando dal Monitoraggio ai Dati con l'etichetta drift.
:width: 80%

Il ciclo di vita di un modello come anello chiuso. Il monitoraggio in
produzione non è un capolinea: quando rileva il *drift* rimanda ai dati, e il
ciclo ricomincia.
```

`````{tab} Elementare

Un modello non è un quadro che appendi al muro e dimentichi: è un giardino. Lo
pianti (i dati), lo fai crescere (l'addestramento), controlli che sia sano (la
valutazione), lo apri al pubblico (il deploy), e poi devi *tornarci ogni
giorno*: annaffiare, potare, accorgerti se una pianta si ammala (il
monitoraggio). Le stagioni cambiano, il terreno si impoverisce, e ciò che
andava bene a maggio non basta a novembre. Il lavoro del giardiniere non
finisce mai: ricomincia, in tondo.

L'anello della figura racconta proprio questo. Le prime quattro tappe (dati,
addestramento, valutazione, deploy) sono quelle che già conosci. La quinta, il
monitoraggio, è quella nuova: è l'occhio che sta in produzione e, quando vede
che i dati sono cambiati, tira la freccia che riporta all'inizio.

`````

`````{tab} Superiore

Uno studio di Microsoft {cite}`amershi2019software` ha formalizzato il flusso
di lavoro del ML in **nove fasi**: definizione dei requisiti del modello,
raccolta dei dati, pulizia dei dati, etichettatura, *feature engineering*,
addestramento, valutazione, deployment e monitoraggio. Il punto qualificante
dello studio non è l'elenco, ma la sua **topologia**: le fasi non formano una
catena lineare ma un grafo con molti cicli di ritorno. Il monitoraggio
retroagisce sulla raccolta dei dati (è la freccia del *drift*); una
valutazione insoddisfacente rimanda al *feature engineering* o alla raccolta;
un errore nell'etichettatura obbliga a rivedere i dati a monte.

Rispetto a DevOps ci sono due retroazioni specifiche del ML, assenti nel
software tradizionale: la dipendenza dai dati (che cambiano nel tempo e
degradano il modello senza che una riga di codice sia stata toccata) e la
necessità di **riaddestrare** come operazione ordinaria, non come eccezione. È
questa la ragione strutturale per cui il ciclo è un anello e non un segmento.

`````

## Perché non basta il notebook

Il capitolo dà per acquisito tutto ciò che serve a *costruire* un modello: la
teoria dei capitoli precedenti e gli strumenti di **PyTorch**, `torch.nn`,
`torch.optim`, il `DataLoader` (visti nel capitolo dedicato). Il notebook che
addestra una rete e ne stampa l'accuratezza è, a tutti gli effetti, il punto
di partenza di questo capitolo, non un traguardo.

Perché un notebook non basta lo si capisce elencando ciò che non fa. Non
espone il modello a chi deve usarlo, e non decide se farlo su una CPU che
risponde in un decimo di secondo o in *batch* durante la notte. Non sa dire se
i dati di oggi somigliano ancora a quelli di ieri. Non tiene traccia di quale
versione dei dati ha prodotto quali pesi, così che tra sei mesi si possa
capire *perché* una predizione è quella. Non si riaddestra da solo quando il
mondo cambia. Ognuna di queste mancanze è una sezione di questo capitolo.

## Come è organizzato il capitolo

Le cinque sezioni che seguono percorrono l'anello e ne sciolgono i nodi, uno
per uno.

- **Dal notebook alla produzione**, che cosa cambia quando si esce
  dall'ambiente di sperimentazione: riproducibilità, versionamento degli
  artefatti (dati, codice, modello), esperimenti tracciabili, il debito
  tecnico da tenere a bada.
- **Dati e pipeline**, l'ingranaggio più grande e più trascurato: raccolta,
  validazione, *feature engineering* automatizzato e ripetibile, così che il
  dato che alimenta l'addestramento e quello che alimenta le predizioni siano
  costruiti allo stesso modo.
- **Deployment e serving**, come si mette un modello *in ascolto*: servizio in
  tempo reale contro elaborazione in *batch*, latenza e *throughput*, rilasci
  graduali (*canary*, *shadow*, A/B) per non rompere niente.
- **Monitoraggio e drift**, l'occhio in produzione: sorvegliare ingressi,
  uscite ed errori, rilevare il *dataset shift* introdotto nel capitolo di
  Machine Learning e decidere *quando* riaddestrare. È il lato operativo del
  problema che lì avevamo posto in termini statistici.
- **LLMOps**, come cambiano le regole del gioco con i grandi modelli
  linguistici: modelli che non si addestrano ma si *interrogano*, valutazione
  senza una risposta giusta univoca, costi per *token*, *prompt* e recupero di
  informazioni come nuovi artefatti da versionare.

```{admonition} Da ricordare
:class: important
- Addestrare un modello è **la punta dell'iceberg**: nel sistema reale il
  «codice ML» è un rettangolino minuscolo circondato da dati, feature,
  serving, monitoraggio e configurazione {cite}`sculley2015hidden`. Portare un
  modello dal prototipo al servizio reale è un problema d'ingegneria a sé,
  costellato di ostacoli a ogni tappa {cite}`paleyes2022challenges`.
- **MLOps** porta la cultura DevOps (automazione, CI/CD, monitoraggio) al
  ciclo di vita del ML, aggiungendo i **dati** e il **modello** come artefatti
  da versionare accanto al codice {cite}`kreuzberger2023machine`.
- Il **debito tecnico** del ML è aggravato dall'**entanglement** (principio
  CACE: *Changing Anything Changes Everything*): nessuna feature è davvero
  indipendente dalle altre.
- Il ciclo di vita è un **anello**, non una linea: dati → addestramento →
  valutazione → deploy → monitoraggio → (drift) → di nuovo dati. Lo studio di
  Microsoft lo descrive in nove fasi con molte retroazioni
  {cite}`amershi2019software`.
- Un **notebook non basta**: manca il serving, il monitoraggio, la
  tracciabilità e il riaddestramento. Un modello in produzione è un
  **processo** da tenere in vita {cite}`huyen2022designing`, non un risultato
  da archiviare.
```
