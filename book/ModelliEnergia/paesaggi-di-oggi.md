# Paesaggi di oggi

Le reti di Hopfield non si usano più, le RBM quasi neanche. Sarebbe facile
archiviare l'energia come un capitolo di storia del deep learning, e sarebbe
sbagliato: il linguaggio è rimasto, e tre luoghi diversi lo parlano oggi
correntemente. Uno è dichiarato, gli altri due no.

## Il ritorno dichiarato: modelli a energia sulle immagini

Il ritorno esplicito arriva nel 2019, quando Yilun Du e Igor Mordatch mostrano
che un modello a energia (in inglese *energy-based model*, che nella
letteratura si abbrevia sempre in **EBM**) si può addestrare su immagini vere,
e lo fanno con gli attrezzi della sezione sulla partizione, senza inventarne
di nuovi {cite}`du2019implicit`. Il paesaggio è una rete convoluzionale come
quelle del capitolo sulla visione. Le risposte sbagliate su cui alzare il
terreno se le fabbrica il modello stesso, lasciando rotolare qualche pallina
con la ricetta di Langevin. E le palline non ripartono da capo ogni volta: si
tengono in un serbatoio e riprendono da dove erano arrivate, che è parola per
parola l'idea che Tieleman aveva proposto nel 2008 per le macchine di
Boltzmann ristrette {cite}`tieleman2008training`, undici anni prima e su
piccole immagini in bianco e nero invece che su fotografie a colori
(CIFAR-10). La qualità delle immagini generate supera quella degli altri
modelli che imparano dalle probabilità e si avvicina, senza raggiungerla, a
quella delle GAN dell'epoca. Ma il valore del lavoro è un altro: mostrare che
la famiglia è viva, e mettere in luce la proprietà che le è tipica. Un solo
modello serve a generare, a completare immagini a cui manca un pezzo, a
segnalare quello che è fuori posto e a mescolare concetti sommando i loro
paesaggi, perché tutte queste cose sono la stessa cosa: cercare un punto
basso, con vincoli diversi.

L'anno dopo arriva l'osservazione che ribalta la prospettiva. Will Grathwohl
e colleghi notano che **un classificatore è già un modello a energia**, e
nessuno se n'era accorto {cite}`grathwohl2020your`.

`````{tab} Elementare

Una rete che classifica immagini produce, per ogni immagine, un pugno di
numeri: uno per classe, tanto più alto quanto più la rete crede in quella
classe. Di solito quei numeri si normalizzano e si legge la classe vincente,
buttando via il resto. Ma dentro c'è di più: se sommi in modo opportuno tutti
i punteggi di un'immagine, ottieni una misura di quanto *quell'immagine* sia
tipica nel suo insieme: non quale classe sia, ma se sia un'immagine
plausibile. Cioè, esattamente, un'energia.

Il seguito è la parte interessante. Addestrando la stessa rete a fare bene
tutte e due le cose (riconoscere la classe *e* dare energia bassa alle
immagini plausibili) si ottiene un classificatore che sbaglia con più
prudenza: quando è incerto lo dice, riconosce di trovarsi davanti a qualcosa
che non ha mai visto, ed è più difficile da ingannare con immagini manipolate.
La misura di quanto una cosa è plausibile, che sembrava un lusso per generare,
si rivela utile per non prendere abbagli.

`````

`````{tab} Superiore

Sia $f_\theta(\mathbf{x}) \in \mathbb{R}^K$ il vettore dei logit di un classificatore
a $K$ classi. La lettura usuale è
$p_\theta(y \mid \mathbf{x}) = \operatorname{softmax}(f_\theta(\mathbf{x}))_y$. Grathwohl e
colleghi osservano che gli stessi logit definiscono anche una densità
congiunta, se si pone $E_\theta(\mathbf{x}, y) = -f_\theta(\mathbf{x})[y]$:

$$
p_\theta(\mathbf{x}, y) = \frac{e^{f_\theta(\mathbf{x})[y]}}{Z(\theta)},
\qquad
p_\theta(\mathbf{x}) = \sum_{y} p_\theta(\mathbf{x}, y)
= \frac{e^{\operatorname{logsumexp}_y f_\theta(\mathbf{x})[y]}}{Z(\theta)},
$$

da cui l'energia marginale
$E_\theta(\mathbf{x}) = -\operatorname{logsumexp}_y f_\theta(\mathbf{x})[y]$
{cite}`grathwohl2020your`. La softmax è invariante alla traslazione dei logit,
quindi questa informazione (il livello assoluto, non le differenze) è
precisamente ciò che l'addestramento standard *butta via*. JEM (*Joint
Energy-based Model*) la recupera addestrando la rete sulla fattorizzazione
$\log p_\theta(\mathbf{x}, y) = \log p_\theta(y \mid \mathbf{x}) + \log p_\theta(\mathbf{x})$: il primo
termine è la solita cross-entropy, il secondo è un EBM addestrato con Langevin
e serbatoio, come in {cite}`du2019implicit`. Il risultato riportato è un
classificatore con calibrazione migliore, rilevamento di fuori distribuzione
più affidabile e maggiore robustezza agli attacchi avversari, al prezzo di un
addestramento più fragile, che è il difetto ereditario di tutta la famiglia.

`````

## I due ritorni non dichiarati

Il primo lo abbiamo già incontrato nella sezione sulla partizione, e vale la
pena ripeterlo perché è il ponte più solido di tutto il capitolo: **i modelli
di diffusione sono modelli a energia che hanno smesso di dirlo**. Il compito
con cui si addestrano è quello della seconda via, «indovina il rumore che ti ho
aggiunto» {cite}`vincent2011connection`. Quello che imparano, però, non è la
pendenza di un paesaggio solo. Un modello
di diffusione parte da un'immagine di puro rumore e attraversa mille gradi di
sporco decrescente, e a ogni grado corrisponde un paesaggio diverso: quando
l'immagine è ancora tutta rumore il paesaggio è liscio, con poche valli larghe
in cui è difficile sbagliare direzione; scendendo di grado diventa più
dettagliato, con valli più strette, quelle che distinguono un volto
dall'altro. Quello che il modello impara, per ogni grado di sporco, è la
pendenza del paesaggio corrispondente, cioè lo *score* della sezione
precedente; e generare è una **discesa rumorosa** lungo quella successione di
paesaggi, parente stretta della dinamica di Langevin {cite}`song2021score`.
«Rumorosa» va preso alla lettera, ed è il punto che il capitolo sulla
diffusione misura: a ogni passo si rimette dentro rumore fresco in quantità
molto maggiore di quanta ne tolga il passo di discesa. Non è quindi una
ripulitura progressiva, ed è esattamente ciò che dice anche il nome di
Langevin: la pallina non viene accompagnata a valle, viene spinta a valle e
scossa insieme.

Attraversarli in fila, invece di rotolare in uno solo, è anche il rimedio al
guaio della prima via: una pallina lasciata subito nel paesaggio più
dettagliato si fermerebbe nella prima conca che incontra, mentre partire da
quello liscio la porta prima nella regione giusta e solo dopo nei particolari.
La differenza con un modello a energia dichiarato, l'abbiamo detta: qui si
imparano direttamente le frecce della discesa, senza passare per il paesaggio
di cui sarebbero la pendenza.

Il secondo è più sorprendente, e chiude un cerchio con il capitolo sui
Transformer. Le reti di Hopfield di oggi non sono quelle del 1982: i neuroni
non sono più soltanto accesi o spenti, e soprattutto la formula dell'energia è
stata riscritta. Dmitry Krotov e lo stesso Hopfield, nel 2016, la riscrivono
in modo da moltiplicare la capienza {cite}`krotov2016dense`; Mete Demircigil e
colleghi, l'anno dopo, spingono la stessa idea fino a una capienza che cresce
in modo esponenziale {cite}`demircigil2017model`; e Hubert Ramsauer e colleghi
portano il tutto ai valori continui, trovando la cosa che nessuno si aspettava
{cite}`ramsauer2021hopfield`. Il loro articolo si intitola, non a caso,
*Hopfield Networks is All You Need*: il conto con cui una di queste memorie
richiama un ricordo è, a meno di un passaggio, lo stesso con cui un
Transformer presta attenzione {cite}`vaswani2017attention`. La domanda che si
fa alla memoria è la *query*, i ricordi in archivio sono le *key*, e un passo
di attenzione è un passo di discesa verso il ricordo più compatibile.

«A meno di un passaggio» non è una formula di cortesia, e vale la pena
spendere due righe, perché è il punto in cui la battuta del titolo va presa
meno alla lettera di quanto si direbbe. L'identità vale a tre condizioni: che
si faccia **un solo** passo di aggiornamento invece di iterare fino in fondo
come farebbe una rete di Hopfield normale; che un parametro di scala sia
fissato esattamente al valore che i Transformer usano; e che quello che la
memoria restituisce venga fatto passare per un'ulteriore trasformazione, che è
poi quella che nell'attenzione produce i *value*. I ricordi, nella
corrispondenza, sono le key: i value sono i ricordi già trasformati.

C'è poi un risultato che questo capitolo tiene volentieri, perché è più
interessante della battuta. Puntando questa lente sulle teste di attenzione
dei Transformer veri, gli autori trovano che nei primi strati la maggior parte
di esse non sta richiamando nessun ricordo singolo: sta facendo una media di
tutti quanti. La discesa c'è sempre; il punto d'arrivo è un ricordo preciso
solo quando i ricordi sono ben separati fra loro, e altrimenti è una media. La
memoria del 1982 e il meccanismo che regge i modelli di linguaggio parlano
dunque la stessa lingua, e la parentela dice sull'attenzione qualcosa di più
sfumato, e più informativo, di «è un richiamo di memoria».

## Le quattro rinunce

Chi ha seguito le conferenze di Yann LeCun degli ultimi anni conosce la sua
slide di chiusura: quattro righe, ciascuna una rinuncia, ciascuna con la sua
alternativa. Vale la pena metterle in fila, perché sono la mappa del programma
di ricerca in cui questo capitolo si inserisce, e perché tre delle quattro
toccano cose che il libro ha già trattato.

```{figure} ../figures/quattro-rinunce.svg
:name: fig-quattro-rinunce
:alt: Quattro righe affiancate. A sinistra, in terracotta e barrate, le cose a cui rinunciare: modelli generativi, modelli probabilistici, metodi contrastivi, reinforcement learning. A destra, in teal, le alternative proposte: architetture a incorporamento congiunto, modelli a energia, metodi regolarizzati, controllo predittivo basato su modello. Sotto, in piccolo, il rimando ai capitoli del libro che trattano ciascuna coppia.
:width: 92%

Le quattro rinunce con cui LeCun chiude le sue conferenze, ridisegnate. In
parole: via i modelli che rifanno il dato pezzo per pezzo, meglio reti che si
limitano a confrontare due riassunti; via le probabilità, meglio l'energia;
via il mostrare al modello anche gli esempi sbagliati perché impari a
respingerli, meglio costruirlo in modo che non possa dire di sì a tutto; via
l'imparare per tentativi, meglio pianificare dentro un modello del mondo.
L'argomento esteso è in *A Path Towards Autonomous Machine Intelligence*
{cite}`lecun2022path`; la seconda riga è la tesi di questo capitolo.
```

Le quattro righe di {numref}`fig-quattro-rinunce` non hanno tutte lo stesso
peso, e conviene prenderle una per una.

La **seconda** riga (abbandonare il modello probabilistico in favore dei
modelli a energia) è quella su cui l'argomento tecnico è più solido, ed è
tutto in questo capitolo: $Z$ non è cara, è impossibile, e moltissimi compiti
non l'hanno mai richiesta. La **terza** (abbandonare i metodi contrastivi in
favore di quelli regolarizzati) è la scelta discussa nella sezione precedente,
ed è una questione di ricerca aperta con risultati da entrambe le parti:
l'apprendimento contrastivo ha prodotto sistemi che funzionano molto bene, e i
metodi regolarizzati sono la scommessa che quei sistemi non reggeranno alla
dimensionalità del video. La **quarta** (sostituire il reinforcement learning
con il controllo predittivo basato su modello) dice, nel lessico dei capitoli
sull'apprendimento per rinforzo, di pianificare dentro un modello del mondo
invece di imparare per tentativi, e di ricorrere al RL solo per correggere il
modello quando la previsione sbaglia.

La **prima** riga è la più contestata, e non conviene nasconderlo. Rinunciare
ai modelli generativi in favore delle architetture a incorporamento congiunto
è una tesi sul modo giusto di costruire un *modello del mondo*, non un
verdetto sulla generazione in quanto tale: mentre la slide circolava, i
modelli generativi hanno prodotto i migliori generatori di immagini che
conosciamo (di diffusione, cioè, come questo capitolo ha mostrato, modelli a
energia) e i modelli linguistici che hanno cambiato il dibattito pubblico.
L'argomento di LeCun non è che quei sistemi non funzionino; è che predire ogni
pixel spende capacità sull'imprevedibile, e che per prevedere il mondo
convenga prevedere non i pixel ma il *riassunto* che la rete se ne fa:
confrontare due riassunti, invece di ridisegnare ogni pixel. È una previsione sul
futuro della ricerca, e come tutte le previsioni va tenuta distinta dai
risultati che abbiamo in mano. Il capitolo che segue la prende sul serio
proprio perché la tratta così: come una scommessa argomentata, con i suoi
risultati e i suoi limiti, non come una profezia.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un paesaggio si può scavare anche su **immagini vere**, e allora un solo
  modello fa quattro mestieri: genera, completa un'immagine a cui manca un
  pezzo, segnala quello che è fuori posto e mescola concetti. Sono la stessa
  cosa: cercare il punto più basso, con vincoli diversi.
- Un **classificatore è già un modello a energia** senza saperlo: sommando nel
  modo giusto i punteggi che dà alle classi si ottiene quanto quell'immagine
  è plausibile, non quale classe sia. Addestrarlo a fare bene anche questo lo
  rende più prudente: dice quando è incerto, riconosce le cose mai viste ed è
  più difficile da ingannare.
- I **modelli di diffusione** sono modelli a energia che non lo dichiarano:
  imparano la pendenza di un paesaggio per ogni grado di sporco, e ripulire
  un'immagine è scendere lungo quella fila di paesaggi, dal più liscio (dove
  è difficile sbagliare direzione) al più dettagliato.
- Le **reti di Hopfield di oggi** tengono in memoria molti più ricordi di
  quelle del 1982, e il modo in cui li richiamano è, a un passaggio di
  distanza, l'attenzione dei Transformer: la domanda fa da indizio, e in
  memoria ci sono i ricordi. Con una sorpresa: guardando dentro i Transformer
  veri con questa lente, la maggior parte delle teste non richiama un ricordo
  solo, ne fa una media.
- Le **quattro rinunce** di Yann LeCun: via i modelli che rifanno il dato
  pezzo per pezzo (meglio reti che si limitano a confrontare due riassunti,
  invece di ridisegnare ogni pixel), via le probabilità (meglio l'energia), via il
  mostrare al modello anche gli esempi sbagliati perché impari a respingerli
  (meglio costruirlo in modo che non possa dire di sì a tutto, come stringere
  la porta invece di istruire il buttafuori), via l'imparare per tentativi
  (meglio pianificare dentro un modello del mondo). La seconda è la tesi di
  questo capitolo; la prima resta una scommessa, e il capitolo sui modelli del
  mondo la discute per quello che è.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Gli **EBM sulle immagini** {cite}`du2019implicit` addestrano una rete come
  $E_\theta$ con campioni negativi da Langevin e un serbatoio persistente: un
  solo modello genera, completa, rileva anomalie e compone concetti.
- **JEM** {cite}`grathwohl2020your`: un classificatore è già un EBM, con
  $E_\theta(\mathbf{x}) = -\operatorname{logsumexp}_y f_\theta(\mathbf{x})[y]$.
  Addestrarlo anche come tale migliora calibrazione, rilevamento del fuori
  distribuzione e robustezza.
- I **modelli di diffusione** sono modelli a energia che non lo dichiarano:
  loss di denoising score matching, campo dello score
  $-\nabla_{\mathbf{x}} E_t(\mathbf{x})$ a ogni livello di rumore $t$,
  campionamento parente di Langevin lungo la successione di paesaggi, dal più
  liscio al più dettagliato.
- Le **Hopfield moderne** hanno energia riprogettata {cite}`krotov2016dense`,
  capacità esponenziale {cite}`demircigil2017model` e, agli stati continui,
  una regola di aggiornamento che è la *scaled dot-product attention*
  {cite}`ramsauer2021hopfield`: con $\beta = 1/\sqrt{d_k}$, un solo passo di
  aggiornamento e una proiezione dei pattern sui value. Sulle teste vere il
  punto fisso è di solito uno stato metastabile (una media di più ricordi),
  non un ricordo singolo.
- Le **quattro rinunce** di LeCun {cite}`lecun2022path`: generativo →
  incorporamento congiunto, probabilistico → energia, contrastivo →
  regolarizzato, RL → controllo predittivo. La seconda è la tesi di questo
  capitolo; la prima resta una scommessa, e il capitolo sui world model la
  discute per quello che è.
```
`````
