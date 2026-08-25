# Modelli a energia

L'8 ottobre 2024 l'Accademia reale svedese delle scienze annuncia il premio
Nobel per la fisica: va a John Hopfield e Geoffrey Hinton «per scoperte e
invenzioni fondamentali che rendono possibile l'apprendimento automatico con
reti neurali artificiali». La notizia lascia interdetti parecchi addetti ai
lavori (lo stesso Hinton, raggiunto al telefono in un albergo della
California, si dice sbalordito) e per giorni rimbalza la stessa domanda: che
cosa c'entra la *fisica*? Hopfield e Hinton non hanno scoperto particelle né
misurato onde gravitazionali: hanno costruito reti neurali.

La risposta della giuria è seria, ed è la porta d'ingresso di questo capitolo.
Le reti premiate non *assomigliano* a un sistema fisico, per esempio a una
calamita: si comportano esattamente come tale. A ogni **configurazione** dei
loro neuroni, cioè a ogni modo in cui possono essere accesi e spenti, è
associato un numero, e quel numero si chiama **energia**.

Conviene fermarsi subito su quella parola, perché è una parola presa in
prestito. Qui «energia» non è la corrente che accende una lampadina né le
calorie di un piatto di pasta: non è una sostanza che la rete possiede e
consuma. È un **voto**, un numero che il modello dà a ogni risposta possibile:
basso se la risposta è sensata, alto se è assurda. Poteva chiamarsi punteggio,
o stranezza, o altezza. Si chiama energia per due motivi: la formula che lo
calcola è, lettera per lettera, quella con cui i fisici descrivono una
calamita (da dove esca lo racconta la prima sezione), e quella
parola porta con sé un'immagine comoda, le risposte buone come il fondo di una
valle.

Sarà così per tutto il capitolo: ogni volta che una parola arriva dalla
fisica il testo la scioglie sul posto. La prima è quella appena sciolta,
**energia**; le altre che contano sono **temperatura**, **partizione** e
**spin**, e arrivano in quest'ordine.

E c'è una ragione precisa per cui il fondo, e non la cima. Una pallina, nel
mondo, cade: nei punti bassi ci va da sola, mentre in cima a un monte non ci
sta ferma nessuno. Se mettiamo le risposte buone in basso non dobbiamo
*cercarle*, basta lasciar rotolare; se le mettessimo in alto qualcuno dovrebbe
spingere, e quel qualcuno saremmo noi. È qui la furbizia di tutto il capitolo:
la dinamica di queste reti fa scendere l'energia, quindi *ricordare* significa
scivolare in un minimo, e *imparare* significa scolpire il paesaggio, scavare
valli nei punti dove vogliamo che la rete vada a finire.

Quarant'anni dopo quelle reti, il linguaggio dell'energia non è un pezzo da
museo. È la lingua in cui è scritta la proposta per l'AI che verrà di Yann
LeCun, uno dei tre a cui nel 2018 è andato il premio Turing per il deep
learning; è, sotto mentite spoglie, ciò che addestra i modelli di diffusione;
ed è il modo più economico che conosciamo per rispondere a una domanda senza
essere costretti a rispondere, insieme, a tutte le altre.

Quel «senza rispondere a tutte le altre» è il seguito diretto del capitolo
precedente, e conviene dirlo perché i due si tengono per mano. Là abbiamo
visto la famiglia che la probabilità la restituisce esatta, e il prezzo che
paga per riuscirci: chi mette i dati in fila è poi costretto a generarli un
pezzetto alla volta, chi li deforma non può buttare via niente e quindi non
può comprimere. In tutti e due i casi se n'è andata la libertà di dare alla
rete la forma che si vuole. Qui si prende la strada opposta: si rinuncia in
partenza a normalizzare, si tiene un voto e basta, e quella libertà torna
indietro. Tutto il capitolo racconta come si vive senza la normalizzazione, e
quanto costa.

## Un numero al posto di una probabilità

```{figure} ../figures/oltre-il-gradiente.svg
:name: fig-paesaggio-energia
:alt: "Un paesaggio di energia con più valli di profondità diversa. Una pallina che segue soltanto la discesa resta intrappolata nella prima valle che incontra, poco profonda. Una traiettoria tratteggiata mostra invece un percorso che accetta di risalire ogni tanto e riesce così a raggiungere la valle più profonda."
:width: 92%

Il paesaggio che dà il nome al capitolo: in basso le risposte sensate, in alto
quelle assurde, e in orizzontale (nel disegno, «spazio delle soluzioni») tutte
le risposte possibili messe in fila. La pallina di sinistra si limita a
scendere, e si ferma nella prima conca che trova; quella tratteggiata ogni
tanto accetta di risalire, e così cambia valle. Sono due mosse per due
problemi diversi, e il capitolo le incontra in quest'ordine.
```

Le due mosse del disegno portano i loro nomi inglesi, e conviene scioglierli
subito perché nel disegno sembrano dire il contrario di quel che fanno. La
prima si chiama *hill climbing*, «scalata della collina», e il nome viene da
dove è nata, cioè da chi cercava il punto più *alto*: qui il paesaggio è
rovesciato e quella stessa mossa scende, ma il nome le è rimasto addosso. La
seconda si chiama *simulated annealing*, «ricottura simulata»: si scuote il
paesaggio, forte all'inizio e poi sempre più piano, e finché la scossa è forte
la pallina salta fuori anche dalle conche in cui si era infilata per sbaglio.
«Ricottura» è quello che fa il fabbro quando scalda un pezzo di metallo e lo
lascia raffreddare adagio invece di buttarlo nell'acqua: raffreddando piano,
gli atomi hanno tempo di sistemarsi bene. È la mossa su cui è costruita la
seconda sezione, quella delle macchine di Boltzmann.

Ed ecco la seconda parola presa in prestito dalla fisica, quella che nel
disegno è la «T»: **temperatura**. Qui non c'è niente di caldo e non c'è
nessun termometro. «Temperatura» vuol dire soltanto *quanto forte stiamo
scuotendo*: alta quando la pallina salta dappertutto, bassa quando resta nei
fondovalle, zero quando può solo scendere. La seconda sezione la riprende per
esteso, e le fa fare un mestiere in più: trasformare le altezze del paesaggio
in percentuali.

Quella seconda mossa, la scossa che accetta di far risalire, anticipa una
differenza di mentalità che attraversa quasi tutto il capitolo. Un
classificatore o un regressore *ottimizzano*: cercano la risposta migliore e
si fermano lì. La prima rete che incontreremo, quella di Hopfield, fa lo
stesso. Ma dalla seconda sezione in poi i modelli di questo capitolo
**campionano**, cioè producono una risposta alla volta, e la pescano in modo
che a lungo andare le risposte buone escano
spesso, quelle mediocri ogni tanto e quelle assurde quasi mai: è la frequenza
che il paesaggio prescrive. Per riuscirci accettano di peggiorare per un
tratto, perché è l'unico modo di uscire da una valle e vederne un'altra. Non è
una novità assoluta, e sarebbe scorretto farla passare per tale: i generatori
dei due capitoli precedenti campionano anche loro, e quello di diffusione lo
fa con una mossa che è parente stretta di quella che si incontra qui.

Un modello probabilistico, per dire quanto è verosimile una risposta, deve
tenere il conto di tutte le risposte possibili: le percentuali che dà a tutte
le risposte, sommate, devono fare cento, e quel cento è un vincolo che tiene
insieme il mondo intero. Un modello a energia rinuncia al vincolo.
Assegna a ogni **configurazione** (cioè a ogni risposta possibile: un'immagine,
una frase, uno stato della rete) un numero, l'energia, e si limita a pretendere
che le configurazioni sensate stiano in basso e le altre in alto. Nessuna
somma da chiudere, nessun totale da rispettare: solo un paesaggio.

`````{tab} Elementare

Passa un dito su una carta geografica in rilievo e senti le valli e le cime.
Ogni punto di quella carta è una risposta possibile alla tua domanda: una
faccia, una frase, il fotogramma che verrà. L'altezza del punto è la sua
energia, e dice quanto la risposta è insensata: le risposte buone stanno nelle
valli, quelle assurde in cima ai monti. Energia e altezza sulla carta sono la
stessa cosa. Rispondere significa lasciar rotolare una pallina e guardare dove
si ferma; imparare significa scavare il paesaggio finché le valli non stanno
nei punti giusti.

Le percentuali, dalla carta, si ricavano. Scuotila e lascia girare la pallina
per un'ora. Tocca un po’ tutti i punti, ma in quelli bassi si trattiene molto
più a lungo che in cima, e quel «molto più a lungo» è una percentuale. Altezze e
frequenze dicono la stessa cosa in due lingue. La traduzione, però, si paga. Se
ti chiedessi «quante probabilità ci sono che dietro l'angolo ci sia un gatto?»
e volessi una percentuale onesta, dovrei aver messo in conto tutto quello che
un gatto non è: cani, biciclette, cassonetti, qualunque cosa esista. È il
prezzo del cento per cento: per dire «70%» su una cosa devi aver pesato tutte
le altre, e una carta grande così nessuno riesce a misurarla tutta.

Se invece ti chiedo soltanto «gatto o cassonetto, quale delle due torna di
più?», ti basta confrontare due altezze sulla carta. Il paesaggio non ti
obbliga mai a fare il giro del mondo per rispondere a una domanda locale.

`````

`````{tab} Superiore

Un modello a energia (*energy-based model*, EBM) è una funzione scalare
$E_\theta(\mathbf{x})$ (o $E_\theta(\mathbf{x}, y)$ quando le variabili osservate $\mathbf{x}$ e quelle
da predire $y$ vanno distinte) con parametri $\theta$: bassa dove i dati sono
plausibili, alta altrove. L'inferenza è un'ottimizzazione,

$$
\hat{y} = \arg\min_{y \in \mathcal{Y}} E_\theta(\mathbf{x}, y),
$$

dove $\hat{y}$ è la risposta predetta e $\mathcal{Y}$ l'insieme delle
risposte ammissibili: nessuna somma su $\mathcal{Y}$, solo una ricerca del
minimo.

Il legame con la probabilità esiste, ed è la distribuzione di Boltzmann–Gibbs:

$$
p_\theta(\mathbf{x}) = \frac{e^{-E_\theta(\mathbf{x})}}{Z(\theta)},
\qquad
Z(\theta) = \int e^{-E_\theta(\mathbf{x}')}\, d\mathbf{x}',
$$

dove $Z(\theta)$ è la **funzione di partizione**, l'integrale (o la somma, nel
caso discreto) su *tutto* lo spazio delle configurazioni. Ogni energia per
cui quell'integrale è finito definisce una densità, e ogni densità
strettamente positiva si riscrive come
energia, $E_\theta(\mathbf{x}) = -\log p_\theta(\mathbf{x}) + \text{cost.}$: le due descrizioni
sono equivalenti *sulla carta*. Non lo sono nei conti. $Z(\theta)$ è il
termine che nessuno sa calcolare quando $\mathbf{x}$ è un'immagine, e metà di questo
capitolo è dedicata a ciò che si può fare senza di lui, e all'osservazione,
tutt'altro che ovvia, che moltissimi compiti non ne hanno mai avuto bisogno.

`````

## Perché un capitolo a sé

Perché lo stesso oggetto continua a riaffiorare sotto nomi diversi, e finché
lo si incontra un pezzo per volta non lo si riconosce.

Prendiamo di nuovo i modelli di diffusione. Partono da un'immagine tutta
sporca di rumore e arrivano, mille passi più tardi, a
un'immagine pulita; e ogni passo ha il suo paesaggio: all'inizio liscio, con
poche valli larghe, poi via via più dettagliato. Quello che quei modelli
imparano, punto per punto, è la **pendenza** di quei paesaggi: da che parte si
scende e quanto ripido. In inglese quella pendenza si chiama *score*, ed è la
parola che si incontra nei loro articoli. Attraversare quella successione di
paesaggi, dal più liscio al più dettagliato, *è* generare.

Il {doc}`capitolo sui world model </WorldModels/overview>` racconta i modelli che, invece di ridisegnare il
mondo, ne confrontano due riassunti: si chiamano **JEPA** (*Joint-Embedding
Predictive Architecture*). Anche loro sono energie mai trasformate in
percentuali: giudicano quanto un pezzo di mondo osservato e uno da predire
stiano bene insieme. Le reti di Hopfield «moderne», poi, richiamano un ricordo
con lo stesso conto con cui un modello di linguaggio decide a quali parole
guardare {cite}`ramsauer2021hopfield`. E LeCun chiude da anni le sue
conferenze con lo stesso elenco: quattro cose a cui il campo dovrebbe
rinunciare, ciascuna con la sua alternativa (l'ultima sezione le guarda una
per una). La seconda dice «abbandonare il modello probabilistico in favore dei
modelli a energia» {cite}`lecun2022path`.

Diffusione, JEPA, Hopfield moderne, il programma di LeCun: sembrano quattro
argomenti distinti. Sono lo stesso, e questo capitolo lo guarda in faccia.

## Dal paesaggio all'energia

Cinque tappe, in salita dolce. Si comincia da dove l'idea è nata: la **memoria
associativa** di Hopfield {cite}`hopfield1982neural`, venticinque neuroni che
ricostruiscono un ricordo rovinato rotolando in fondo a una valle, con il
codice per vederlo accadere. Poi la **macchina di Boltzmann**
{cite}`ackley1985learning`, che aggiunge temperatura e neuroni nascosti,
trasforma il paesaggio in percentuali e proprio per questo incontra il muro
contro cui va a sbattere metà del capitolo: per dire che una risposta vale il
30% bisogna aver pesato tutte le altre, cioè aver misurato il paesaggio
intero.

Quel conto porta un nome che spaventa più di quel che vale, ed è la terza
parola presa in prestito: si chiama **funzione di partizione**. «Partizione»
qui non ha niente a che vedere con il dividere un insieme in parti né con le
partizioni di un disco fisso: il conto dice come il cento per cento si
*ripartisce* fra tutte le configurazioni possibili, e il nome viene da lì. La
lettera con cui i libri lo indicano è $Z$, che in italiano non è l'iniziale di
niente: arriva dal tedesco *Zustandssumme*, «somma su tutti gli stati», che è
esattamente quello che quel conto fa.

La terza tappa è quel muro. **Oltre la partizione** mette in fila le tre
strade che l'hanno aggirato. La prima manda esploratori a caso nel paesaggio e si
accontenta di
quello che riportano: è il campionamento, e la ricetta che si usa porta il
nome di Langevin. La seconda rinuncia alle percentuali e impara soltanto la
pendenza: si chiama *score matching* {cite}`hyvarinen2005estimation`, e nella
sua forma su dati sporcati apposta {cite}`vincent2011connection` è quella che,
un decennio dopo, è finita dentro i modelli di diffusione. La terza cambia
domanda e chiede «questo viene dai dati o l'ho fabbricato io?»: è la stima
contrastiva col rumore {cite}`gutmann2010noise`.

La quarta tappa è il gesto di LeCun: il lungo articolo didattico del 2006
{cite}`lecun2006tutorial` che rilegge quasi ogni modello di apprendimento come
un giudizio di **compatibilità** fra una domanda e una risposta. Ha una
promessa (il paesaggio non va mai misurato tutto) e un pericolo: che il
modello impari a dire di sì a qualunque cosa, e in gergo si chiama
**collasso**. L'ultima tappa guarda al presente: i modelli a energia
addestrati sulle immagini vere {cite}`du2019implicit`, il classificatore che
era un modello a energia senza saperlo {cite}`grathwohl2020your`, e le quattro
rinunce che LeCun ripete nelle sue conferenze, discusse per quello che sono:
un programma di ricerca, non un verdetto.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un **modello a energia** è una carta geografica in rilievo di tutte le
  risposte possibili: ogni risposta ha la sua altezza, bassa se è sensata e
  alta se è assurda. Rispondere significa lasciar rotolare una pallina e
  guardare in che valle si ferma; imparare significa scavare le valli nei
  punti giusti.
- Altezze e percentuali dicono la stessa cosa in due lingue, ma la traduzione
  costa cara, perché per dire onestamente «70% gatto» bisogna aver pesato
  tutto quello che gatto non è. Il paesaggio non lo chiede mai: per
  sapere quale di due risposte torna di più bastano due altezze messe a
  confronto. Quel conto di tutto il resto del mondo, che nessuno riesce a
  fare, si chiama **funzione di partizione**, ed è l'ostacolo contro cui si
  scontra metà del capitolo.
- Il **premio Nobel per la fisica del 2024** a Hopfield e Hinton ha ricordato
  a tutti che questo modo di ragionare non se n'è mai andato: i generatori di
  immagini a diffusione ripuliscono il rumore seguendo la pendenza di un
  paesaggio, e le reti di Hopfield di oggi richiamano un ricordo con lo
  stesso conto con cui i modelli di linguaggio decidono a quali parole
  guardare.
- Nelle prossime pagine, cinque: la memoria che si ripara da sola, le reti che
  imparano scaldandosi e raffreddandosi, i modi di girare intorno al conto
  impossibile, il giudizio a coppie («questa risposta sta bene con questa
  domanda?») e i paesaggi che si usano oggi.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Un **modello a energia** assegna un numero a ogni configurazione (basso se
  plausibile, alto se no) e risponde cercando il minimo:
  $\hat{y} = \arg\min_y E_\theta(\mathbf{x}, y)$. Niente probabilità da far
  sommare a uno.
- Energia e probabilità sono legate dalla distribuzione di Boltzmann–Gibbs,
  $p_\theta(\mathbf{x}) = e^{-E_\theta(\mathbf{x})}/Z(\theta)$. Il ponte si
  paga con la **funzione di partizione** $Z(\theta)$, intrattabile in alta
  dimensione: è il personaggio contro cui si scontra metà del capitolo.
- Il premio **Nobel per la fisica 2024** a Hopfield e Hinton ha riportato
  alla luce un filone che non se n'era mai andato: lo *score* della
  diffusione è $-\nabla_{\mathbf{x}} E_t$, una pendenza per ogni livello di
  rumore; la JEPA è un'energia non normalizzata; l'aggiornamento delle Hopfield
  moderne è la *scaled dot-product attention*, a meno della proiezione dei
  value.
- Nel resto del capitolo: memoria associativa, macchine di Boltzmann e
  contrastive divergence, i modi di aggirare $Z$, la cornice
  dell’*energy-based learning* e i modelli a energia di oggi.
```
`````
