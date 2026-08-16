# MLOps: mettere i modelli in produzione

C'è una figura, in un articolo del 2015, che vale da sola un capitolo intero.
La disegnano alcuni ingegneri di Google (D. Sculley e colleghi) per una
conferenza di apprendimento automatico {cite}`sculley2015hidden`. Al centro
del foglio c'è un rettangolino nero, minuscolo, con dentro due parole: *codice
ML*. È l'unico pezzo di cui parlano di solito i libri, i corsi, i paper: il
modello, l'algoritmo, la rete che impara. Tutto intorno, a soffocarlo, ci sono
scatole molto più grandi: raccogliere i dati, controllarli, ricavarne le
grandezze su cui il modello ragiona (le *feature*), tenere in ordine le
impostazioni, far girare la macchina che riceve le domande e restituisce le
risposte (in gergo il *serving*), sorvegliare, analizzare, e amministrare i
computer che fanno i conti. La morale della figura è brutale e onesta:
addestrare il modello è la parte più piccola del lavoro. Tutto il resto (il
grosso) è il sistema che gli sta intorno.

Non è un dettaglio da ingegneri pignoli. Nel settore si ripete spesso che
moltissimi modelli non arrivino mai **in produzione**, cioè non finiscano mai
davanti a persone vere che li usano ogni giorno. È un'osservazione diffusa e
più aneddotica che misurata, e non è su quella che ci appoggiamo. Il problema
di fondo, però, è documentato. Una rassegna di casi reali lo ricostruisce
progetto per progetto, e mostra quanti ostacoli costellino *ogni* tappa del
percorso che porta un modello dal prototipo al servizio: raccogliere e
verificare i dati, consegnare il modello al mondo reale (il *deployment*),
sorvegliarlo, mantenerlo {cite}`paleyes2022challenges`. Non è che i modelli
sbaglino le predizioni in laboratorio. È che nessuno aveva pensato a come
alimentarli, aggiornarli, tenerli d'occhio. Questo capitolo parla esattamente
di quel «tutto il resto». Il suo nome, ormai, è **MLOps**: *ML* per machine
learning, *Ops* per le operazioni, cioè il mestiere di tenere in funzione ciò
che è in funzione.

## Il modello è la punta dell'iceberg

L'errore di prospettiva è comprensibile: fino a qui, in tutto il libro, «fare
machine learning» ha significato scegliere un modello, addestrarlo, misurarne
l'accuratezza. E tutto questo si fa in un **notebook**, che qui non è un
computer portatile: è una pagina su cui il programma si scrive a pezzetti, uno
sotto l'altro, e ogni pezzetto lo si può far partire da solo vedendo subito che
cosa combina. È comodissimo per
provare, ed è lì che nasce quasi ogni modello. Sul proprio computer il lavoro
sembra finito quando la metrica sui dati di prova è buona.

Ma un modello che nessuno usa non serve a niente. E la distanza fra «funziona
nel mio notebook» e «funziona per migliaia di persone, ogni giorno, per anni»
è enorme, ed è quasi tutta fuori dal modello.

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

C'è poi una cosa che rende questa cucina più insidiosa di una vera. Se il
fornitore cambia i pomodori, tu ti aspetti che cambi il sapore dei piatti col
pomodoro; qui invece cambia anche il sapore di piatti che il pomodoro non lo
contengono affatto. Il motivo è che il modello non impara una regola per volta:
impara un equilibrio fra tutto quello che gli è stato dato. Cambia una sola
delle informazioni in ingresso e l'equilibrio si rifà da capo, e le risposte si
spostano anche dove non te lo aspettavi. È la ragione per cui un impianto del
genere si tiene d'occhio invece di darlo per finito.

`````

`````{tab} Superiore

Sculley e colleghi inquadrano il fenomeno con il linguaggio del **debito
tecnico** {cite}`sculley2015hidden`. Nel software tradizionale le scorciatoie
prese in fretta si pagano con gli interessi più avanti; un sistema di
apprendimento automatico ne accumula uno *aggiuntivo* e più insidioso, perché
nasconde la propria complessità nei **dati** e non solo nel codice. Il sistema
in produzione comprende pipeline di raccolta e validazione dei dati,
estrazione e trasformazione delle *feature*, un livello di *serving* che
espone il modello, monitoraggio, gestione della configurazione e delle
risorse: il codice di addestramento è una frazione minima del totale.

Il debito più caratteristico è l’**entanglement**, riassunto dal principio
**CACE**: *Changing Anything Changes Everything*. In un modello di ML nessuna
*feature* è davvero indipendente. Cambiare la distribuzione di un solo
ingresso, aggiungerne o toglierne uno, ritoccare un iperparametro: ognuna di
queste mosse ripesa tutte le altre e sposta le predizioni ovunque, in modi non
locali e difficili da prevedere. È l'opposto della modularità a cui
l'ingegneria del software tradizionale ci ha abituati, ed è la ragione per cui
un sistema di ML non si governa con le sole pratiche del software classico.

`````

## Che cos'è MLOps

Il nome ricalca **DevOps**. Chi costruisce software si divideva in due mondi
separati: chi scrive i programmi (*Dev*, lo sviluppo) e chi li tiene in
funzione (*Ops*, le operazioni). DevOps è il modo di lavorare con cui quei due
mondi hanno smesso di essere separati, e il patto è che tutto ciò che sta in
mezzo diventi automatico e tracciato. Si conserva ogni versione del programma,
non soltanto l'ultima. E a ogni modifica, senza che nessuno lo chieda, un
computer di servizio riprende il programma, gli fa passare tutte le prove e,
se le supera, lo pubblica al posto della versione di prima: quel meccanismo si
chiama **CI/CD**, che sta per «integrazione e distribuzione continue», cioè
controllo e pubblicazione che avvengono di continuo invece che una volta ogni
tanto. Infine si sorveglia ciò che è in funzione. Il risultato è che pubblicare
una versione nuova diventa un gesto ordinario invece che una notte in bianco.

MLOps prende quella cultura e la porta al ciclo di vita del machine learning
{cite}`kreuzberger2023machine`. Con una complicazione in più, che è il cuore
di tutto. Nel software classico la cosa da conservare versione per versione è
una sola, il codice; qui i pezzi che compongono il lavoro finito (gli
**artefatti**) sono **tre**: il codice, i dati e il modello addestrato. E due
dei tre non sono testo. Gli strumenti con cui il software tiene la propria
cronologia da decenni sono fatti apposta per il testo: sanno dire quale riga è
cambiata fra ieri e oggi. Su una cartella di immagini, o sul file che contiene
i **pesi** del modello (i milioni di numeri che l'addestramento ha aggiustato,
e che *sono* quello che il modello ha imparato), quel confronto non vuol dire
niente: sono file enormi, e dentro non ci sono righe da confrontare.

`````{tab} Elementare

C'è una differenza tra saper cucinare e saper mandare avanti una catena di
ristoranti. Il cuoco tiene la ricetta in testa, aggiusta a occhio, e ogni suo
piatto è un po’ diverso dal precedente, ed è bellissimo così. Ma una catena
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

La definizione operativa poggia su una tripletta versionata: **dati + codice +
modello** {cite}`kreuzberger2023machine`. Rendere un esperimento riproducibile
significa poter ricostruire una predizione a partire da (a) la versione esatta
del dataset di addestramento, (b) la versione del codice e degli
iperparametri, (c) i pesi del modello che ne sono risultati. Da qui le
pratiche cardine: *data versioning* e *feature store* per gli ingressi,
*experiment tracking* per legare metriche e configurazioni, *model registry*
per i modelli, pipeline automatizzate che rieseguono l'intero percorso (da
dato grezzo a modello servito) con un comando solo.

L'obiettivo non è la sofisticazione, ma l’**automazione** e la
**tracciabilità**: ridurre il lavoro che si rifà a mano ogni volta, rendere
ogni rilascio ripetibile e ogni predizione riconducibile agli artefatti che
l'hanno prodotta. È la tesi di fondo dei testi che hanno sistematizzato la
disciplina, *Designing Machine Learning Systems* fra i primi
{cite}`huyen2022designing`: un modello in produzione non è un risultato, è un
**processo** da tenere in vita.

`````

## Il ciclo di vita di un modello

Ed è proprio la parola «processo» a segnare la differenza più importante.
Siamo abituati a pensare al machine learning come a una linea retta: si
raccolgono i dati, si addestra, si valuta, si consegna. Fine. Ma la consegna
non è la fine: è il punto in cui il modello incontra il mondo reale, e il
mondo reale cambia. Un modello in produzione va **sorvegliato**, perché prima o
poi i dati che incontra smettono di somigliare a quelli su cui è stato
addestrato. Quello scivolamento lento ha un nome inglese che useremo sempre,
*drift*, e vuol dire deriva: è la stessa cosa di cui parla la sezione «Quando i
dati cambiano» del capitolo di Machine Learning, lì misurata con gli strumenti
della statistica, qui affrontata da chi il servizio lo deve tenere acceso.
Quando succede, si torna all'inizio: nuovi dati, nuovo addestramento. Il
percorso non è una linea, è un **anello**
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

Uno studio condotto in Microsoft ha formalizzato il flusso di lavoro del ML in
**nove fasi** {cite}`amershi2019software`: definizione dei requisiti del
modello, raccolta dei dati, pulizia dei dati, etichettatura, *feature
engineering*, addestramento, valutazione, deployment e monitoraggio. Il punto
qualificante non è l'elenco, ma la sua **topologia**: le fasi non formano una
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

Il notebook, si è detto, è dove il modello nasce. Il guaio è che di solito lo
si scambia anche per il posto dove il lavoro finisce, e in realtà è la prima
casella di cinque ({numref}`fig-cinque-tappe`): dopo di lui il modello deve
uscire dal foglio, farsi raggiungere da altri programmi, girare uguale su
computer che non sono il proprio, e poi restare sotto controllo per anni.

```{figure} ../figures/dal-notebook-alla-produzione.svg
:name: fig-cinque-tappe
:alt: "Cinque tappe in fila dal notebook alla produzione: l'esperimento nel notebook, l'estrazione in programmi di cui si conserva ogni versione, il rilascio dietro uno sportello a cui altri programmi possono rivolgersi, il confezionamento in una scatola che si comporta uguale su qualsiasi computer e infine la sorveglianza continua. Una freccia tratteggiata torna dall'ultima tappa alla prima. Solo la prima è quella che di solito si considera «il lavoro»."
:width: 100%

Il notebook è la prima delle cinque caselle, non l'ultima. Dopo di lui il
codice esce dal foglio e diventa un programma di cui si conserva ogni versione
(`Git`); il programma viene messo dietro uno sportello a cui altri programmi
possono bussare (l’*API*); lo sportello viene chiuso in una scatola che si
comporta uguale su qualsiasi computer (il *container*); e solo allora si apre
al pubblico, sorvegliato. Le quattro tappe che seguono la prima non aggiungono
intelligenza al modello: aggiungono le condizioni perché quell'intelligenza
serva a qualcuno. E la freccia che torna indietro in fondo è l'anello: da lì
si ricomincia.
```

Quattro caselle su cinque vengono *dopo* il notebook: è la proporzione di
{numref}`fig-cinque-tappe`, ed è il messaggio dell'intero capitolo. Perché
questo capitolo dà per acquisito tutto ciò che serve a *costruire* un modello,
cioè la teoria dei capitoli precedenti e gli attrezzi con cui si addestra una
rete, che sono quelli del capitolo su **PyTorch**. Il notebook che addestra una
rete e ne stampa il voto è, a tutti gli effetti, il punto di partenza di questo
capitolo, non un traguardo.

Perché un notebook non basta lo si capisce elencando ciò che non fa. Non mette
il modello a disposizione di chi deve usarlo. Non decide *come* metterlo a
disposizione: un caso per volta, mentre l'utente aspetta, oppure tutti insieme
durante la notte, un mucchio alla volta (in gergo un *batch*, e la parola
tornerà spesso). Non sa dire se i dati di oggi somigliano ancora a quelli di
ieri. Non tiene traccia di quale versione dei dati ha prodotto quali pesi, così
che tra sei mesi si possa capire *perché* una predizione è quella. Non si
riaddestra da solo quando il mondo cambia. Ognuna di queste mancanze è una
sezione di questo capitolo.

## Come è organizzato il capitolo

Le sezioni che seguono percorrono l'anello e ne sciolgono i nodi, uno per uno.
L'elenco è fitto di parole nuove: sono tutte spiegate, una per una, nella
pagina che le introduce, e nessuna va saputa già adesso.

- **Dal notebook alla produzione**, che cosa cambia quando si esce
  dall'ambiente di sperimentazione: riproducibilità, versionamento degli
  artefatti (dati, codice, modello), esperimenti tracciabili, il debito
  tecnico da tenere a bada.
- **Dati e pipeline**, l'ingranaggio più grande e più trascurato. Una
  *pipeline* è alla lettera una conduttura: la catena di stazioni che prende il
  dato grezzo, lo pulisce e lo consegna pronto al modello. Qui si vede come si
  raccoglie, come si controlla e come si fa in modo che il dato su cui il
  modello impara e quello su cui risponde siano costruiti allo stesso modo.
- **Servire un modello**, cioè come lo si mette *in ascolto*: rispondere a
  una richiesta per volta oppure a mille tutte insieme di notte, quanto si
  aspetta una risposta (la *latenza*) e quante se ne servono al secondo (il
  *throughput*), e come si sostituisce un modello con uno nuovo facendolo
  provare prima a pochi, per non rompere niente.
- **Monitoraggio e drift**, l'occhio in produzione: sorvegliare che cosa entra,
  che cosa esce e quanto si sbaglia, accorgersi della deriva di cui si diceva
  poco fa e decidere *quando* rimettere mano al modello.
- **LLMOps**, come cambiano le regole del gioco con i grandi modelli
  linguistici (in sigla **LLM**, *large language model*): modelli che non si
  addestrano ma si *interrogano*, testi da giudicare senza che esista una
  risposta giusta sola, e un conto che si paga a pezzetti di testo (i *token*).
- **Misurare un servizio**, che cosa vuol dire davvero «veloce» quando la
  risposta non arriva tutta insieme ma una parola alla volta: quanto si aspetta
  la prima, con che ritmo scorrono le altre, e perché le medie mentono.
- **Il conto in energia**, l'unica voce che non si dichiara quasi mai: dove
  finisce la corrente (nel movimento dei dati, non nei conti), come si arriva
  dall'energia ai grammi di anidride carbonica, e perché in un modello che
  resta in servizio per anni rispondere costa più che addestrare.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Addestrare un modello è **la punta dell'iceberg**: è il piatto del cuoco, e
  il ristorante è tutto il resto (le forniture, la cucina, il servizio in
  sala). Nel disegno del 2015 raccontato all'inizio, il pezzo di cui parlano i
  libri è un rettangolino, e tutto intorno ci sono scatole più grandi.
- **In produzione** vuol dire che il modello ha smesso di essere un
  esperimento: lo stanno usando persone vere, adesso, e la distanza fra le due
  cose è quasi tutta fuori dal modello.
- Ci sono **tre cose** da conservare, non una: il programma, i dati e il
  modello addestrato. Nel software normale basta il primo, ed è per questo che
  gli strumenti del software normale qui non bastano.
- Nulla è davvero separato da nulla: cambiare un ingrediente sposta il
  risultato ovunque, anche dove non te lo aspetti. È il motivo per cui un
  sistema del genere si sorveglia invece di darlo per finito.
- Il percorso non è una linea con un traguardo ma un **anello**: dati,
  addestramento, valutazione, apertura al pubblico, sorveglianza, e da lì di
  nuovo ai dati. Un modello è un giardino, non un quadro appeso.
```
`````

`````{tab} Superiore
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
- Il **debito tecnico** del ML è aggravato dall’**entanglement** (principio
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
`````
