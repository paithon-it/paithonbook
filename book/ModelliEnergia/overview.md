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
Le reti premiate non *assomigliano* a sistemi fisici: si comportano
esattamente come tali. A ogni configurazione dei loro neuroni è associato un
numero, e quel numero si chiama **energia**.

Conviene fermarsi subito su quella parola, perché è una parola presa in
prestito. Qui «energia» non è la corrente che accende una lampadina né le
calorie di un piatto di pasta: non è una sostanza che la rete possiede e
consuma. È un **voto**, un numero che il modello dà a ogni risposta possibile:
basso se la risposta è sensata, alto se è assurda. Poteva chiamarsi punteggio,
o stranezza, o altezza. Si chiama energia per due motivi: la formula che lo
calcola è, lettera per lettera, quella con cui i fisici descrivono una
calamita (la prima sezione dice quale, e da dove viene), e quella
parola porta con sé un'immagine comoda, le risposte buone come il fondo di una
valle.

E c'è una ragione precisa per cui il fondo, e non la cima. Una pallina, nel
mondo, cade: nei punti bassi ci va da sola, mentre in cima a un monte non ci
sta ferma nessuno. Se mettiamo le risposte buone in basso non dobbiamo
*cercarle*, basta lasciar rotolare; se le mettessimo in alto qualcuno dovrebbe
spingere, e quel qualcuno saremmo noi. È qui la furbizia di tutto il capitolo:
la dinamica di queste reti fa scendere l'energia, quindi *ricordare* significa
scivolare in un minimo, e *imparare* significa scolpire il paesaggio, scavare
valli nei punti dove vogliamo che la rete vada a finire.

Quarant'anni dopo quelle reti, il linguaggio dell'energia non è un pezzo da
museo. È la lingua in cui è scritta la proposta di Yann LeCun per l'AI che
verrà; è, sotto mentite spoglie, ciò che addestra i modelli di diffusione del
capitolo precedente; ed è il modo più economico che conosciamo per rispondere
a una domanda senza essere costretti a rispondere, insieme, a tutte le altre.

## Un numero al posto di una probabilità

```{figure} ../figures/oltre-il-gradiente.svg
:name: fig-paesaggio-energia
:alt: "Un paesaggio di energia con più valli di profondità diversa. Una pallina che segue soltanto la discesa resta intrappolata nella prima valle che incontra, poco profonda. Una traiettoria tratteggiata mostra invece un percorso che accetta di risalire ogni tanto e riesce così a raggiungere la valle più profonda."
:width: 92%

Il paesaggio che dà il nome al capitolo: in basso le risposte sensate, in alto
quelle assurde, e in orizzontale (nel disegno, «spazio delle soluzioni») tutte
le risposte possibili messe in fila. Chi scende soltanto si ferma nella conca
più vicina, e quando le conche sono i ricordi è esattamente quello che si
vuole: nel disegno la mossa si chiama *hill climbing*, «scalata della
collina», perché il nome è nato dove si cercava il punto più alto, e qui il
paesaggio è rovesciato. Chi invece ogni tanto accetta di risalire (*simulated
annealing*, la ricottura simulata, ed è la scossa del fabbro che riscalda il
metallo: la «T» del disegno è la temperatura, ne parla la seconda sezione) può
cambiare valle. Sono due mosse per due problemi diversi, e il capitolo le
incontra in quest'ordine.
```

La seconda mossa di {numref}`fig-paesaggio-energia`, quella che accetta di
risalire, anticipa una differenza di mentalità che attraversa tutto il
capitolo. Dove un classificatore o un regressore *ottimizzano* (cercano la
risposta migliore e si fermano lì), i modelli di questo capitolo
**campionano**: ne producono una alla volta, pescandola con la frequenza
giusta, e per riuscirci accettano di peggiorare per un tratto, perché è
l'unico modo di uscire da una valle e vederne un'altra. Non è una novità
assoluta, e sarebbe scorretto farla passare per tale: i generatori dei due
capitoli precedenti campionano anche loro, e quello di diffusione lo fa con
una mossa che è parente stretta di quella che si incontra qui.

Un modello probabilistico, per dire quanto è verosimile una risposta, deve
tenere il conto di tutte le risposte possibili: le probabilità sommano a uno,
e quell'uno è un vincolo globale. Un modello a energia rinuncia al vincolo.
Assegna a ogni **configurazione** (cioè a ogni risposta possibile: un'immagine,
una frase, uno stato della rete) un numero, l'energia, e si limita a pretendere
che le configurazioni sensate stiano in basso e le altre in alto. Nessuna
somma da chiudere, nessun totale da rispettare: solo un paesaggio.

`````{tab} Elementare

Immagina una carta geografica in rilievo, con valli e montagne. Ogni punto
della carta è una risposta possibile alla tua domanda: una faccia, una frase,
il fotogramma che verrà. L'altezza di quel punto *è* l'energia di cui sopra, e
dice quanto la risposta è insensata: le risposte buone stanno nelle valli,
quelle assurde in cima ai monti. Da qui in avanti, ogni volta che leggi
«energia», puoi leggere «altezza sulla carta»: sono la stessa cosa.
Rispondere significa lasciar rotolare una pallina e guardare dove si ferma;
imparare significa scavare il paesaggio finché le valli non stanno nei punti
giusti.

Il vantaggio si vede confrontandolo con l'altro modo di fare, quello delle
probabilità. Se ti chiedessi «quante probabilità ci sono che dietro l'angolo
ci sia un gatto?» e volessi una percentuale onesta, dovrei aver messo in
conto tutto quello che *non* è un gatto: cani, biciclette, cassonetti,
qualunque cosa esista. È il prezzo del cento per cento: per dire «70%» su una
cosa devi aver pesato tutte le altre. Se invece ti chiedo soltanto «gatto o
cassonetto, quale delle due torna di più?», ti basta confrontare due altezze
sulla carta. Il paesaggio non ti obbliga mai a fare il giro del mondo per
rispondere a una domanda locale.

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

Prendiamo i modelli di diffusione del capitolo precedente. Partono da
un'immagine tutta sporca di rumore e arrivano, mille passi più tardi, a
un'immagine pulita; e ogni passo ha il suo paesaggio: all'inizio liscio, con
poche valli larghe, poi via via più dettagliato. Quello che quei modelli
imparano, punto per punto, è la **pendenza** di quei paesaggi: da che parte si
scende e quanto ripido. Ha un nome tecnico, *score*, ed è la pendenza del
paesaggio di energia di quel passo. Attraversare quella successione di
paesaggi, dal più liscio al più dettagliato, *è* generare.

Le architetture **JEPA** del prossimo capitolo (*Joint-Embedding Predictive
Architecture*, cioè architettura predittiva a incorporamento congiunto: sono i
modelli che invece di ridisegnare il mondo ne confrontano
due riassunti) sono energie che nessuno ha mai
trasformato in percentuali: giudicano quanto un pezzo di mondo osservato e uno
da predire stiano bene insieme. Le reti di Hopfield «moderne» richiamano un
ricordo con lo stesso conto con cui i Transformer prestano attenzione
{cite}`ramsauer2021hopfield`. E il programma che LeCun ripete da anni in fondo
alle sue conferenze contiene, come seconda delle quattro rinunce, «abbandonare
il modello probabilistico in favore dei modelli a energia»
{cite}`lecun2022path`.

Quattro cose che sembravano quattro. Sono una sola, e questo capitolo la
guarda in faccia.

## Come è organizzato il capitolo

Cinque tappe, in salita dolce. Si comincia da dove l'idea è nata: la **memoria
associativa** di Hopfield {cite}`hopfield1982neural`, venticinque neuroni che
ricostruiscono un ricordo rovinato rotolando in fondo a una valle, con il
codice per vederlo accadere. Poi la **macchina di Boltzmann**
{cite}`ackley1985learning`, che aggiunge temperatura e neuroni nascosti,
trasforma il paesaggio in una distribuzione di probabilità e incontra per la
prima volta il muro: la funzione di partizione.

La terza tappa è quel muro. **Oltre la partizione** mette in fila le tre
strade che l'hanno aggirato: mandare esploratori a caso nel paesaggio e
accontentarsi di quello che riportano (il campionamento, con la dinamica di
Langevin); rinunciare alle percentuali e imparare soltanto la pendenza (lo
*score matching* {cite}`hyvarinen2005estimation` e la sua forma denoising
{cite}`vincent2011connection`); oppure cambiare domanda, e chiedere «questo
viene dai dati o l'ho fabbricato io?» (la stima contrastiva col rumore
{cite}`gutmann2010noise`). La seconda strada è quella che, un decennio dopo, è
finita dentro i modelli di diffusione. I nomi tecnici sono qui perché tu li
riconosca quando torneranno, non perché tu li sappia già: ciascuno ha la sua
sezione.

La quarta tappa è il gesto di LeCun: il tutorial del 2006
{cite}`lecun2006tutorial` che rilegge quasi ogni modello di apprendimento come
un giudizio di **compatibilità** fra una domanda e una risposta, con la sua
promessa (niente misura del continente) e il suo pericolo (il collasso).
L'ultima guarda al presente: i modelli a energia addestrati sulle immagini
vere {cite}`du2019implicit`, il classificatore che era un modello a energia
senza saperlo
{cite}`grathwohl2020your`, e le quattro rinunce con cui LeCun chiude le sue
conferenze, discusse per quello che sono: un programma di ricerca, non un
verdetto.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un **modello a energia** è una carta geografica in rilievo di tutte le
  risposte possibili: ogni risposta ha la sua altezza, bassa se è sensata e
  alta se è assurda. Rispondere significa lasciar rotolare una pallina e
  guardare in che valle si ferma; imparare significa scavare le valli nei
  punti giusti.
- Le percentuali costano care: per dire onestamente «70% gatto» bisogna aver
  pesato tutto quello che gatto non è. Il paesaggio non lo chiede mai: per
  sapere quale di due risposte torna di più bastano due altezze messe a
  confronto. Quel conto di tutto il resto del mondo, che nessuno riesce a
  fare, è l'ostacolo con cui si scontra metà del capitolo.
- Il **premio Nobel per la fisica del 2024** a Hopfield e Hinton ha ricordato
  a tutti che questo modo di ragionare non se n'è mai andato: i generatori di
  immagini del capitolo precedente ripuliscono il rumore seguendo la pendenza
  di un paesaggio, e le reti di Hopfield di oggi richiamano un ricordo con lo
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
  dell'*energy-based learning* e i modelli a energia di oggi.
```
`````
