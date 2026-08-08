# Paesaggi di oggi

Le reti di Hopfield non si usano più, le RBM quasi neanche. Sarebbe facile
archiviare l'energia come un capitolo di storia del deep learning, e sarebbe
sbagliato: il linguaggio è rimasto, e tre luoghi diversi lo parlano oggi
correntemente. Uno è dichiarato, gli altri due no.

## Il ritorno dichiarato: EBM sulle immagini

Il ritorno esplicito arriva nel 2019, quando Yilun Du e Igor Mordatch mostrano
che un modello a energia si può addestrare su immagini vere con gli strumenti
della sezione sulla partizione: una rete convoluzionale nel ruolo di
$E_\theta$, campioni negativi prodotti per **dinamica di Langevin**, e un
serbatoio di campioni da cui le catene ripartono invece di ricominciare da
zero: la persistent contrastive divergence che Tieleman aveva proposto nel
2008 per le macchine di Boltzmann ristrette {cite}`tieleman2008training`,
undici anni dopo e su CIFAR-10 {cite}`du2019implicit`. La qualità dei
campioni supera quella degli altri modelli a verosimiglianza e si avvicina
(senza raggiungerla) a quella delle GAN dell'epoca; ma il valore del lavoro è
mostrare che la famiglia è viva, e mettere in luce la proprietà che le è
tipica: un solo modello serve a generare, a completare immagini parziali, a
segnalare anomalie e a comporre concetti sommando energie, perché tutte queste
cose sono la stessa cosa: cercare un minimo, con vincoli diversi.

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

Sia $f_\theta(x) \in \mathbb{R}^K$ il vettore dei logit di un classificatore
a $K$ classi. La lettura usuale è
$p_\theta(y \mid x) = \operatorname{softmax}(f_\theta(x))_y$. Grathwohl e
colleghi osservano che gli stessi logit definiscono anche una densità
congiunta, se si pone $E_\theta(x, y) = -f_\theta(x)[y]$:

$$
p_\theta(x, y) = \frac{e^{f_\theta(x)[y]}}{Z(\theta)},
\qquad
p_\theta(x) = \sum_{y} p_\theta(x, y)
= \frac{e^{\operatorname{logsumexp}_y f_\theta(x)[y]}}{Z(\theta)},
$$

da cui l'energia marginale
$E_\theta(x) = -\operatorname{logsumexp}_y f_\theta(x)[y]$
{cite}`grathwohl2020your`. La softmax è invariante alla traslazione dei logit,
quindi questa informazione (il livello assoluto, non le differenze) è
precisamente ciò che l'addestramento standard *butta via*. JEM (*Joint
Energy-based Model*) la recupera addestrando la rete sulla fattorizzazione
$\log p_\theta(x, y) = \log p_\theta(y \mid x) + \log p_\theta(x)$: il primo
termine è la solita cross-entropy, il secondo è un EBM addestrato con Langevin
e serbatoio, come in {cite}`du2019implicit`. Il risultato riportato è un
classificatore con calibrazione migliore, rilevamento di fuori distribuzione
più affidabile e maggiore robustezza agli attacchi avversari, al prezzo di un
addestramento più fragile, che è il difetto ereditario di tutta la famiglia.

`````

## I due ritorni non dichiarati

Il primo lo abbiamo già incontrato nella sezione sulla partizione, e vale la
pena ripeterlo perché è il ponte più solido di tutto il capitolo: **i modelli
di diffusione sono modelli a energia che hanno smesso di dirlo**. La loss con
cui si addestrano è denoising score matching {cite}`vincent2011connection`.
Quello che imparano, però, non è la pendenza di un paesaggio solo. Un modello
di diffusione parte da un'immagine di puro rumore e la ripulisce un poco alla
volta, e a ogni grado di sporco corrisponde un paesaggio diverso: quando
l'immagine è ancora tutta rumore il paesaggio è liscio, con poche valli larghe
in cui è difficile sbagliare direzione; man mano che si pulisce diventa più
dettagliato, con valli più strette, quelle che distinguono un volto
dall'altro. In formule, per ogni livello di rumore $t$ il campo imparato è
$\nabla_x \log p_t(x) = -\nabla_x E_t(x)$, e generare (togliere rumore un
passo alla volta) è una discesa rumorosa lungo quella successione di paesaggi,
parente stretta della dinamica di Langevin {cite}`song2021score`.

Attraversarli in fila, invece di rotolare in uno solo, è anche il rimedio al
guaio della prima via: una pallina lasciata subito nel paesaggio più
dettagliato si fermerebbe nella prima conca che incontra, mentre partire da
quello liscio la porta prima nella regione giusta e solo dopo nei particolari.
La differenza con un modello a energia dichiarato, l'abbiamo detta: qui si
impara il campo vettoriale direttamente, senza passare da una $E_\theta$
scalare.

Il secondo è più sorprendente, e chiude un cerchio con il capitolo sui
Transformer. Le reti di Hopfield **moderne** (stati continui, energia
riprogettata) hanno capacità esponenziale nel numero di neuroni e una regola
di aggiornamento che coincide, formula alla mano, con la *scaled dot-product
attention* {cite}`ramsauer2021hopfield`. Il paper si intitola, non a caso,
*Hopfield Networks is All You Need*, e rilegge l'attenzione di
{cite}`vaswani2017attention` come il richiamo di una memoria associativa: le
query sono indizi, le coppie key–value i ricordi, e un passo di attenzione è
un passo di discesa verso il ricordo più compatibile. La memoria del 1982 e il
meccanismo che regge i modelli di linguaggio parlano, matematicamente, la
stessa lingua.

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

Le quattro rinunce con cui LeCun chiude le sue conferenze, ridisegnate.
L'argomento esteso è in *A Path Towards Autonomous Machine Intelligence*
{cite}`lecun2022path`; la seconda riga è la tesi di questo capitolo.
```

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
convenga predire nello spazio delle rappresentazioni. È una previsione sul
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
  quelle del 1982, e il modo in cui li richiamano è, conto alla mano,
  l'attenzione dei Transformer: la domanda fa da indizio, e in memoria ci sono
  i ricordi.
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
- Le **Hopfield moderne** {cite}`ramsauer2021hopfield` hanno capacità
  esponenziale e una regola di aggiornamento identica all'attenzione dei
  Transformer.
- Le **quattro rinunce** di LeCun {cite}`lecun2022path`: generativo →
  incorporamento congiunto, probabilistico → energia, contrastivo →
  regolarizzato, RL → controllo predittivo. La seconda è la tesi di questo
  capitolo; la prima resta una scommessa, e il capitolo sui world model la
  discute per quello che è.
```
`````
