# Paesaggi di oggi

Le reti di Hopfield non si usano più, le RBM quasi neanche. Sarebbe facile
archiviare l'energia come un pezzo di storia del deep learning, e sarebbe
sbagliato: il linguaggio è rimasto, e tre luoghi diversi lo parlano oggi
correntemente. Nel primo lo si parla dichiarandolo, e sono i ricercatori che
scrivono «modello a energia» nel titolo; negli altri due no, e sono i
generatori di immagini e le memorie associative di oggi, che quel linguaggio
lo usano senza nominarlo.

## Il ritorno dichiarato: modelli a energia sulle immagini

Il ritorno esplicito arriva nel 2019, quando Yilun Du e Igor Mordatch mostrano
che un modello a energia (in inglese *energy-based model*, che nella
letteratura si abbrevia in **EBM**) si può addestrare su immagini vere,
e lo fanno con gli attrezzi della sezione sulla partizione, senza inventarne
di nuovi {cite}`du2019implicit`. Il paesaggio è una rete convoluzionale come
quelle del capitolo sul deep learning. Le risposte sbagliate su cui alzare il
terreno se le fabbrica il modello stesso, lasciando rotolare qualche pallina
con la ricetta di Langevin. E le palline non ripartono da capo ogni volta: si
tengono in un serbatoio e riprendono da dove erano arrivate, che è parola per
parola l'idea della sezione precedente, quella con cui si faceva proseguire il
sogno invece di rifarlo da capo {cite}`tieleman2008training`. Undici anni
separano le due cose, e le separa anche la taglia del problema: allora
minuscole cifre in bianco e nero, qui fotografie a colori di animali e mezzi
di trasporto (l'archivio si chiama CIFAR-10), e più su fino alle immagini di
ImageNet.

La qualità delle immagini generate supera quella degli altri
modelli che imparano dalle probabilità e si avvicina, senza raggiungerla, a
quella delle GAN dell'epoca. Ma il valore del lavoro è un altro: mostrare che
la famiglia è viva, e mettere in luce la proprietà che le è tipica. Un solo
modello serve a generare, a completare immagini a cui manca un pezzo, a
segnalare quello che è fuori posto e a mescolare concetti sommando i loro
paesaggi (sommare due paesaggi vuol dire che un punto sta in basso solo se sta
in basso in tutti e due, e allora le valli che sopravvivono sono quelle che le
due richieste hanno in comune: «giovane» più «sorridente» lascia in piedi le
facce giovani e sorridenti). Tutte queste cose sono la stessa cosa: cercare un
punto basso, con vincoli diversi.

L'anno dopo arriva l'osservazione che ribalta la prospettiva. Will Grathwohl
e colleghi notano che **un classificatore è già un modello a energia**, e
nessuno se n'era accorto {cite}`grathwohl2020your`.

`````{tab} Elementare

Una rete che classifica immagini produce, per ogni immagine, un pugno di
numeri: uno per classe, tanto più alto quanto più la rete crede in quella
classe. Di solito quei numeri si riscalano in percentuali che sommano a cento
e si legge la classe vincente, buttando via il resto. Ma dentro c'è di più.
Prendi il punteggio più alto, e poi correggilo un poco verso l'alto se anche
gli altri sono alti: quello che ottieni è una specie di «quanto forte grida
questa immagine», ed è una misura di quanto *quell'immagine* sia tipica nel
suo insieme, non di quale classe sia. Cioè, esattamente, un'energia.

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

dove $\operatorname{logsumexp}_y f[y] = \log \sum_y e^{f[y]}$ è il massimo
«ammorbidito» dei logit (vale sempre almeno quanto il più grande, e un po' di
più quando anche gli altri sono grandi), da cui l'energia marginale
$E_\theta(\mathbf{x}) = -\operatorname{logsumexp}_y f_\theta(\mathbf{x})[y]$
{cite}`grathwohl2020your`. Perché non la si veda mai, in un classificatore
normale, è questione di gradienti e non di assenza: la softmax è invariante
alla traslazione dei logit, quindi la cross-entropy **non vincola** il loro
livello assoluto, che resta libero di andare dove capita. L'informazione è lì
(tanto che il logsumexp di una rete addestrata alla maniera solita si usa così
com'è per riconoscere il fuori distribuzione); semplicemente nessuno le ha mai
chiesto di essere sensata. JEM (*Joint
Energy-based Model*) gliela chiede, massimizzando la log-verosimiglianza
congiunta nella forma
$\log p_\theta(\mathbf{x}, y) = \log p_\theta(y \mid \mathbf{x}) + \log p_\theta(\mathbf{x})$: il primo
termine è la solita cross-entropy cambiata di segno, il secondo è un EBM
addestrato con Langevin
e serbatoio, come in {cite}`du2019implicit`. Da notare, in un capitolo che ha
dedicato una sezione a $Z$: quel $Z(\theta)$ esiste solo se
$\int \sum_y e^{f_\theta(\mathbf{x})[y]}\, d\mathbf{x}$ converge, che per una rete
convoluzionale qualunque nessuno garantisce. La lettura «un classificatore è
un'energia» è sempre vera; la densità che ne segue, sotto condizione. Il risultato riportato è un
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
pendenza del paesaggio corrispondente, cioè lo *score* della seconda via;
e generare è una **discesa rumorosa** lungo quella successione di
paesaggi, parente stretta della dinamica di Langevin {cite}`song2021score`.
«Rumorosa» va preso alla lettera, ed è il punto che il capitolo sulla
diffusione misura: a ogni passo si rimette dentro rumore fresco in quantità
molto maggiore di quanta ne tolga il passo di discesa. Il livello di sporco
cala lo stesso, ma di pochissimo per volta, e come saldo fra tre mosse che
tirano in versi diversi: nessuno sta togliendo un velo alla volta. È
esattamente ciò che dice anche il nome di Langevin: la pallina non viene
accompagnata a valle, viene spinta a valle e scossa insieme.

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
stata riscritta. Dmitry Krotov e lo stesso Hopfield, nel 2016, sostituiscono
al prodotto fra due neuroni una potenza di grado $n$, e la capienza smette di
crescere in proporzione ai neuroni per crescere come $N^{n-1}$: da lineare a
polinomiale {cite}`krotov2016dense`. Mete Demircigil e
colleghi, l'anno dopo, spingono la stessa idea fino a una capienza che cresce
in modo esponenziale nel numero di neuroni {cite}`demircigil2017model`: dove
la rete del 1982 guadagna un ricordo ogni sette neuroni in più, questa
raddoppia il numero di ricordi ogni due. E Hubert Ramsauer e colleghi portano
il tutto ai valori continui, dove un neurone non è più acceso o spento ma
porta un numero qualsiasi, trovando la cosa che nessuno si aspettava
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
:alt: Quattro righe affiancate, sotto le intestazioni «abbandonare» e «in favore di». A sinistra, in terracotta e precedute da un simbolo di divieto, le cose a cui rinunciare: modelli generativi, modelli probabilistici, metodi contrastivi, reinforcement learning. A destra, in teal e raggiunte da una freccia, le alternative proposte: architetture a incorporamento congiunto, modelli a energia, metodi regolarizzati, controllo predittivo su modello. Sotto ciascuna alternativa, in piccolo e in grigio, una riga che dice dove il libro la tratta o in che cosa consiste.
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
(le JEPA nominate in apertura di capitolo)
è una tesi sul modo giusto di costruire un *modello del mondo*, non un
verdetto sulla generazione in quanto tale: mentre la slide circolava, i
modelli generativi hanno prodotto i generatori di immagini del capitolo
precedente (di diffusione, cioè, come questo capitolo ha mostrato, modelli a
energia) e i modelli linguistici che hanno cambiato il dibattito pubblico.
L'argomento di LeCun non è che quei sistemi non funzionino; è che predire ogni
pixel costringe la rete a spendere i suoi neuroni e il suo addestramento su
dettagli che nessuno potrebbe indovinare (la forma esatta di una foglia mossa
dal vento), e che per prevedere il mondo
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
  loss di denoising score matching (riponderata per livello di rumore), campo
  dello score $-\nabla_{\mathbf{x}} E_t(\mathbf{x})$ a ogni livello $t$,
  campionamento parente di Langevin lungo la successione di paesaggi, dal più
  liscio al più dettagliato. Con la riserva detta nella sezione sulla
  partizione: imparano le frecce, e che siano la pendenza di una superficie
  vera nessuno lo garantisce.
- Le **Hopfield moderne** hanno energia riprogettata {cite}`krotov2016dense`,
  capienza esponenziale {cite}`demircigil2017model` e, agli stati continui,
  una regola di aggiornamento che è la *scaled dot-product attention*
  {cite}`ramsauer2021hopfield`: con $\beta = 1/\sqrt{d_k}$ (la temperatura
  inversa coincide col fattore di scala dell'attenzione), un solo passo di
  aggiornamento e una proiezione dei pattern sui value. Sulle teste vere il
  punto fisso è raramente un ricordo singolo: nei primi strati è una media di
  tutti i pattern, più in alto uno stato metastabile, cioè la media di un
  sottoinsieme.
- Le **quattro rinunce** di LeCun {cite}`lecun2022path`: generativo →
  incorporamento congiunto, probabilistico → energia, contrastivo →
  regolarizzato, RL → controllo predittivo. La seconda è la tesi di questo
  capitolo; la prima resta una scommessa, e il capitolo sui world model la
  discute per quello che è.
```
`````
