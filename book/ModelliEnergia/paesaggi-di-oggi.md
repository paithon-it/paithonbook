# Paesaggi di oggi

Le reti di Hopfield non si usano più, e nemmeno le macchine di Boltzmann
ristrette, le RBM della seconda sezione. Sarebbe facile archiviare l'energia come
un pezzo di storia del deep learning, e sarebbe sbagliato: il modo di
ragionare è rimasto, e tre luoghi diversi lo praticano oggi correntemente. Nel
primo lo si dichiara, e sono i ricercatori che scrivono «modello a energia»
nel titolo; negli altri due no, e sono i generatori di immagini e le memorie
associative di oggi, che quel linguaggio lo usano senza nominarlo.

## Il ritorno dichiarato: modelli a energia sulle immagini

Il ritorno esplicito arriva nel 2019, quando Yilun Du e Igor Mordatch mostrano
che un modello a energia (in inglese *energy-based model*, che nella
letteratura si abbrevia in **EBM**) si può addestrare su immagini vere,
e lo fanno con gli attrezzi della sezione dedicata alla funzione di
partizione, senza inventarne di nuovi {cite}`du2019implicit`.

A calcolare l'altezza del paesaggio, cioè a dare il voto a ogni immagine, c'è
una rete convoluzionale come quelle del {doc}`capitolo sul deep learning </DeepLearning/overview>`: il
paesaggio non è disegnato da nessuna parte, esiste solo nel senso che quella
rete, per ogni immagine che le si dà, sa dire quanto in alto sta. Le risposte
sbagliate su cui alzare il terreno se le fabbrica il modello stesso, lasciando
rotolare qualche pallina con la ricetta di Langevin, cioè scendendo lungo la
pendenza con addosso un po’ di rumore. E le palline non ripartono da capo ogni
volta: si tengono in un serbatoio e riprendono da dove erano arrivate, che è
parola per parola l'idea della sezione sulle macchine di Boltzmann, quella con
cui si faceva proseguire il sogno invece di rifarlo da capo
{cite}`tieleman2008training`. Undici anni separano le due cose, e le separa
anche la taglia del problema: allora minuscole cifre in bianco e nero, qui
fotografie a colori di animali e mezzi di trasporto (l'archivio si chiama
CIFAR-10), e più su fino alle immagini di ImageNet.

Gli autori riportano immagini di qualità superiore a quella degli altri
modelli che imparano dalle probabilità, e vicina, senza raggiungerla, a quella
delle GAN di allora. Ma il valore del lavoro è un altro: mostrare che
la famiglia è viva, e mettere in luce la proprietà che le è tipica. Un solo
modello serve a generare, a completare immagini a cui manca un pezzo, a
segnalare quello che è fuori posto e a mescolare concetti sommando i loro
paesaggi.

Quest'ultima merita due numeri, perché è la più bella e la meno ovvia. Si
prende il paesaggio di «giovane» e quello di «sorridente» e si sommano le due
altezze punto per punto. Una faccia giovane e imbronciata sta a 3 nel primo
paesaggio e a 10 nel secondo: sommati, 13, ed è una cima. Una faccia giovane e
sorridente sta a 3 e a 2: sommati, 5, ed è ancora una valle. Sopravvivono
insomma soltanto le conche che le due richieste hanno in comune, e cercare il
punto più basso del paesaggio somma vuol dire cercare una faccia giovane *e*
sorridente. E il riferimento rispetto a cui 13 è una cima e 5 una valle è il
paesaggio stesso: contano i confronti fra punti, non i numeri presi da soli.

Tutte queste cose, generare e completare e segnalare e mescolare, sono poi la
stessa cosa: andare a stare in basso. Non «nel punto più basso di tutti», che
sarebbe la stessa risposta ogni volta: si scende con addosso il rumore, come
nella prima delle tre vie, e ogni volta si finisce in una valle diversa. A
cambiare, da un mestiere all'altro, è solo il vincolo con cui si scende.

L'anno dopo arriva l'osservazione che ribalta la prospettiva. Will Grathwohl
e colleghi notano che **un classificatore è già un modello a energia**, e
nessuno se n'era accorto {cite}`grathwohl2020your`.

`````{tab} Elementare

Una rete che classifica immagini produce, per ogni immagine, un pugno di
numeri: uno per classe, tanto più alto quanto più la rete crede in quella
classe. Di solito quei numeri si trasformano in percentuali che sommano a
cento, si legge la classe vincente e il resto si butta via.

Si butta via più di quel che sembra. Il passaggio alle percentuali guarda
solo le *differenze* fra i punteggi, non quanto sono grandi: i punteggi 8, 2,
1 e i punteggi 9, 3, 2 danno esattamente le stesse percentuali (99,7%, 0,2%,
0,1%), perché sono gli stessi numeri spostati tutti in su di uno. Ma i secondi
sono più forti dei primi, e quella forza si perde per strada.

Recuperarla costa poco. La ricetta è: si prende il punteggio più alto e gli si
aggiunge una correzione che dipende da quanto gli altri gli stanno vicino, e
che vale tanto meno quanto più sono staccati. Con 8, 2, 1 gli altri due sono
lontanissimi e la correzione è quasi zero: viene 8,003, cioè il massimo tale e
quale. Con 8, 7, 7 gli altri due sono a un passo e la correzione conta: viene
8,55. Tre voci che gridano insieme fanno più chiasso di una sola.
Chiamiamolo «quanto forte grida questa immagine».

Perché dovrebbe dirci se l'immagine è *tipica*? Perché la rete quei punteggi
li ha imparati sulle immagini vere, e davanti a quelle grida forte. Davanti a
qualcosa che non ha mai visto, invece, nessuna delle sue classi si accende
davvero, e i punteggi restano fiacchi tutti quanti. Il grido misura dunque quanto
l'immagine somiglia a ciò che la rete conosce, e non quale classe sia. E
siccome nel nostro paesaggio le cose sensate stanno *in basso*, l'energia è
quel grido con il segno cambiato: chi grida forte sta in fondo a una valle,
chi non grida sta in cima.

Il seguito è la parte interessante. Addestrando la stessa rete a fare bene
tutte e due le cose (riconoscere la classe *e* dare energia bassa alle
immagini plausibili) si ottiene un classificatore che sbaglia con più
prudenza: quando è incerto lo dice, riconosce di trovarsi davanti a qualcosa
che non ha mai visto, ed è più difficile da ingannare con immagini manipolate.
La misura di quanto una cosa è plausibile, che sembrava un lusso per generare,
si rivela utile per non prendere abbagli.

Il conto però va chiuso, perché una contropartita c'è. Il secondo mestiere si
insegna con le palline che rotolano della prima delle tre vie, e quelle ogni
tanto scappano: chiedere le due cose insieme rende l'addestramento fragile, e va
messo in preventivo. È il difetto di famiglia di tutti i modelli a energia di
questo capitolo, non di questo in particolare.

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
«ammorbidito» dei logit (vale sempre almeno quanto il più grande, e un po’ di
più quando anche gli altri sono grandi), da cui l'energia marginale
$E_\theta(\mathbf{x}) = -\operatorname{logsumexp}_y f_\theta(\mathbf{x})[y]$
{cite}`grathwohl2020your`. Perché non la si veda mai, in un classificatore
normale, è questione di gradienti e non di assenza: la softmax è invariante
alla traslazione dei logit, quindi la cross-entropy **non vincola** il loro
livello assoluto, che resta libero di andare dove capita. L'informazione è lì
(tanto che il logsumexp di una rete addestrata alla maniera solita si usa così
com'è per riconoscere il fuori distribuzione, come mostrano Weitang Liu e colleghi nel 2020 {cite}`liu2020energy`); semplicemente nessuno le ha mai
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

Il primo lo abbiamo già incontrato nella sezione sulla funzione di partizione,
e conviene ripeterlo perché è il ponte più solido di tutto il capitolo: **i
modelli di diffusione sono modelli a energia che hanno smesso di dirlo**. Il
compito con cui si addestrano è quello della seconda delle tre vie, quella che
rinuncia alle percentuali e impara soltanto la pendenza: «indovina il rumore
che ti ho aggiunto» {cite}`vincent2011connection`.

Quello che imparano, però, non è la pendenza di un paesaggio solo. Un modello
di diffusione parte da un'immagine di puro rumore e attraversa mille gradi di
sporco decrescente, e a ogni grado corrisponde un paesaggio diverso: quando
l'immagine è ancora tutta rumore il paesaggio è liscio, con poche valli larghe
in cui è difficile sbagliare direzione; scendendo di grado diventa più
dettagliato, con valli più strette, quelle che distinguono un volto
dall'altro. Quello che il modello impara, per ogni grado di sporco, è la
pendenza del paesaggio corrispondente; e generare è una **discesa rumorosa**
lungo quella successione di paesaggi, parente stretta della dinamica di
Langevin {cite}`song2021score`.

«Rumorosa» va preso alla lettera, ed è il punto che il capitolo sulla
diffusione misura. A ogni passo si fanno tre mosse: si cancella una scheggia
del disturbo, quella che la rete indica; si moltiplica tutta l'immagine per un
numero appena sopra uno, che ingrandisce insieme il disegno e lo sporco che
lo copre; e si getta sopra una manciata di rumore appena sorteggiato.
La terza mossa mette dentro molto più disturbo di quanto la prima ne tolga, e
viene da chiedersi come faccia lo sporco a calare. La risposta è in due pezzi.
Il primo: la scheggia cancellata è **mirata**, punta sempre dalla stessa
parte giro dopo giro, mentre le manciate di rumore sono sorteggiate ogni volta
in una direzione diversa e a lungo andare si disfano fra loro. Piccola e
costante batte grande e a casaccio, purché si ripeta abbastanza. Il secondo:
la mossa di mezzo, quella che ingrandisce, tira su il disegno e lo sporco
insieme, e messa in fila con le altre due fa sì che il disegno cresca mentre il
livello di sporco cala. I numeri di quel saldo li misura il capitolo sulla
diffusione. Nessuno, comunque, sta togliendo un velo alla
volta, ed è esattamente ciò che dice anche la ricetta di Langevin: la pallina
non viene accompagnata a valle, viene spinta a valle e scossa insieme.

Attraversare i paesaggi in fila, invece di rotolare in uno solo, è anche il
rimedio al guaio della prima via, quella del campionamento: nel paesaggio più
dettagliato le valli sono strette e le creste alte, e una pallina lasciata lì
resterebbe prigioniera nei dintorni di dove è caduta, per quanto la si scuota.
Partire da quello liscio la porta prima nella regione giusta, dove le valli
sono larghe e si passa da una all'altra, e solo dopo nei particolari.

Resta una differenza con un modello a energia dichiarato. Un
modello a energia impara l'altezza di ogni punto, e le frecce della discesa si
ricavano da quella; un modello di diffusione impara direttamente le frecce, e
niente garantisce che esista davvero un paesaggio di cui quelle frecce siano
la pendenza (sono le quattro frecce in tondo lungo il bordo di un quadrato,
che sembrano un pendio e non lo sono).

Il secondo è più sorprendente, e chiude un cerchio con il
{doc}`capitolo sui Transformer </Transformers/overview>`, cioè con
l'architettura su cui sono costruiti i modelli di
linguaggio. Le reti di Hopfield di oggi non sono quelle del 1982: i neuroni
non sono più soltanto accesi o spenti, e soprattutto la formula dell'energia è
stata riscritta.

Il primo a riscriverla è Dmitry Krotov con lo stesso Hopfield, nel 2016
{cite}`krotov2016dense`. Nella formula del 1982 ogni ricordo contribuisce all'energia con la sua
somiglianza allo stato della rete, contata al quadrato; loro alzano quella
somiglianza a una potenza più alta, e più alta è la potenza più la memoria è
capiente. Con l'esponente due si ritrova la rete di sempre, in cui la capienza
cresce in proporzione al numero di neuroni: raddoppiando i neuroni si
raddoppiano i ricordi. Con l'esponente tre cresce come il *quadrato* dei
neuroni:
raddoppiandoli, i ricordi diventano quattro volte tanti.

Mete Demircigil e colleghi, l'anno dopo, spingono la stessa idea fino in fondo
{cite}`demircigil2017model`, e la capienza cambia proprio modo di crescere.
La rete del 1982 tiene circa il 14% dei neuroni, cioè guadagna un ricordo ogni
sette neuroni in più; questa raddoppia il numero di ricordi ogni due.

E Hubert Ramsauer e colleghi portano il tutto ai valori continui, dove un
neurone non è più acceso o spento ma porta un numero qualsiasi, trovando la
cosa che nessuno si aspettava {cite}`ramsauer2021hopfield`. Il loro articolo
si intitola *Hopfield Networks is All You Need*, «le reti di Hopfield sono
tutto ciò che serve», ed è una citazione: il titolo dell'articolo del 2017 che
ha introdotto i Transformer era *Attention Is All You Need*
{cite}`vaswani2017attention`. La battuta è che si può dire la stessa frase con
l'altro nome, perché il conto con cui una di queste memorie richiama un
ricordo è, a meno di un passaggio, lo stesso con cui un Transformer presta
attenzione. La domanda che si fa alla memoria è quella che nell'attenzione si
chiama *query*, i ricordi in archivio sono le *key*, e un passo di attenzione
è un passo di discesa verso il ricordo più compatibile.

«A meno di un passaggio» non è una formula di cortesia, e conviene spendere
due righe, perché è il punto in cui la battuta del titolo va presa meno alla
lettera di quanto si direbbe. L'identità vale a tre condizioni. La prima: che
si faccia **un solo** passo di aggiornamento, invece di ripetere il passo fino
in fondo come farebbe una rete di Hopfield normale. La seconda: che la
temperatura della memoria, cioè quanto forte la si scuote, sia fissata
esattamente al valore che i Transformer usano per dividere i loro punteggi
prima di confrontarli, cioè la radice quadrata della lunghezza dei vettori in
gioco. La terza: che quello che la memoria restituisce venga fatto passare per
un'ultima moltiplicazione, la stessa che nell'attenzione trasforma i ricordi
in ciò che poi viene davvero letto, e che si chiama *value*. I ricordi grezzi,
nella corrispondenza, sono le key; i value sono quegli stessi ricordi dopo
quella moltiplicazione.

C'è poi un risultato che questo capitolo tiene volentieri, perché è più
interessante della battuta. L'attenzione di un Transformer non è un blocco
solo: dentro ogni strato ce ne sono parecchie copie che lavorano in parallelo,
e ciascuna copia si chiama **testa**. Puntando questa lente sulle teste dei
Transformer veri, gli autori trovano che nei primi strati la maggior parte di
esse non sta richiamando nessun ricordo singolo: sta facendo una media su
moltissimi. La discesa c'è sempre; il punto d'arrivo è un ricordo preciso
solo quando i ricordi sono ben separati fra loro, e altrimenti è una media. La
memoria del 1982 e il meccanismo che regge i modelli di linguaggio parlano
dunque la stessa lingua, e la parentela dice sull'attenzione qualcosa di più
sfumato, e più informativo, di «è un richiamo di memoria».

## Le quattro rinunce

Chi ha seguito le conferenze di Yann LeCun degli ultimi anni conosce la sua
slide delle raccomandazioni: quattro righe, ciascuna una rinuncia, ciascuna
con la sua alternativa. Conviene metterle in fila, perché sono la mappa del
programma di ricerca in cui questo capitolo si inserisce, e perché tre delle
quattro toccano cose che il libro ha già trattato.

```{figure} ../figures/quattro-rinunce.svg
:name: fig-quattro-rinunce
:alt: Quattro righe affiancate, sotto le intestazioni «abbandonare» e «in favore di». A sinistra, in terracotta e precedute da un simbolo di divieto, le cose a cui rinunciare: modelli generativi, modelli probabilistici, metodi contrastivi, reinforcement learning. A destra, in teal e raggiunte da una freccia, le alternative proposte: architetture a incorporamento congiunto, modelli a energia, metodi regolarizzati, controllo predittivo su modello. Sotto ciascuna alternativa, in piccolo e in grigio, una riga che dice dove il libro la tratta o in che cosa consiste.
:width: 92%

Le quattro rinunce che ricorrono nelle conferenze di LeCun, ridisegnate. In
parole: via i modelli che rifanno il dato pezzo per pezzo, meglio reti che si
limitano a confrontare due riassunti; via le probabilità, meglio l'energia;
via il mostrare al modello anche gli esempi sbagliati perché impari a
respingerli, meglio costruirlo in modo che non possa dire di sì a tutto; via
l'imparare per tentativi, meglio pianificare dentro un modello del mondo.
L'argomento esteso è in *A Path Towards Autonomous Machine Intelligence*,
il documento di posizione di LeCun del 2022; la seconda riga è la tesi di
questo capitolo.
```

Le quattro righe di {numref}`fig-quattro-rinunce` non hanno tutte lo stesso
peso, e conviene prenderle una per una, in ordine di solidità: prima la
seconda, poi la terza, poi la quarta, e la prima per ultima perché è quella su
cui si litiga di più.

La **seconda** riga, abbandonare il modello probabilistico in favore dei
modelli a energia, è quella su cui l'argomento tecnico è più solido, ed è
tutto in questo capitolo: misurare il paesaggio intero, il conto che serve per
trasformare le altezze in percentuali, non è caro, è impossibile, e moltissimi
compiti non l'hanno mai richiesto.

La **terza**, abbandonare i metodi contrastivi in favore di quelli
regolarizzati, è la scelta discussa nella sezione precedente: mostrare al
modello dei controesempi, oppure costruirlo in modo che non possa dire di sì a
tutto. È una questione di ricerca aperta con risultati da entrambe le parti.
L'apprendimento contrastivo ha prodotto sistemi che funzionano molto bene, e
la scommessa di chi sta dall'altra parte è che non reggeranno al video, dove
il numero di risposte possibili è tale che nessuna quantità di controesempi
basterebbe a puntellarlo.

La **quarta**, sostituire il reinforcement learning con il controllo
predittivo basato su modello, dice, nel lessico dei capitoli
sull'apprendimento per rinforzo: invece di imparare per tentativi ed errori,
costruirsi un modello di come va il mondo e pianificare dentro quello,
ricorrendo ai tentativi soltanto per correggere il modello (o il giudice che
valuta le mosse) quando la previsione sbaglia.

Il *perché* di quella riga non sta nella diapositiva, e conviene anticiparlo
perché è un conto, non un'antipatia. Quando un sistema impara per tentativi,
la correzione che riceve alla fine di un tentativo è una quantità sola: è
andata bene oppure male. Quando impara guardando, la correzione è grande
quanto il pezzo di mondo che stava provando a indovinare, e contata in bit è
dell'ordine di centomila volte tanto. È l'argomento che LeCun riassume dicendo
che l'apprendimento per rinforzo è la «ciliegina sulla torta»
{cite}`lecun2016cake`, e il capitolo sull'auto-supervisione lo misura per
intero, quel conto compreso, insieme alle obiezioni di chi non ci sta.

La **prima** riga è la più contestata, e non conviene nasconderlo. Rinunciare
ai modelli generativi in favore delle architetture a incorporamento congiunto,
cioè delle JEPA nominate in apertura di capitolo, quelle che confrontano due
riassunti del mondo invece di ridisegnarlo, è una tesi sul modo giusto di
costruire un modello del mondo. Non è un verdetto sulla generazione in quanto
tale, e i fatti lo mostrano: mentre la slide circolava, i modelli generativi
hanno prodotto i generatori di immagini a diffusione e i modelli linguistici
che hanno cambiato il dibattito pubblico.

L'argomento di LeCun non è che quei sistemi non funzionino. È che predire ogni
pixel costringe la rete a spendere i suoi neuroni e il suo addestramento su
dettagli che nessuno potrebbe indovinare, la forma esatta di una foglia mossa
dal vento, e che per prevedere il mondo convenga prevedere non i pixel ma il
*riassunto* che la rete se ne fa. È una previsione sul futuro della ricerca, e
come tutte le previsioni va tenuta distinta dai risultati che abbiamo in mano.
Il {doc}`capitolo sui world model </WorldModels/overview>` la prende sul serio proprio perché la tratta così:
come una scommessa argomentata, con i suoi risultati e i suoi limiti, non come
una profezia.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un paesaggio si può scavare anche su **immagini vere**, e allora un solo
  modello fa quattro mestieri: genera, completa un'immagine a cui manca un
  pezzo, segnala quello che è fuori posto e mescola concetti. Sono la stessa
  cosa: cercare il punto più basso, con vincoli diversi.
- Un **classificatore è già un modello a energia** senza saperlo: dal più alto
  dei punteggi che dà alle classi, corretto un poco verso l'alto quando anche
  gli altri sono alti, si ottiene quanto quell'immagine è plausibile, non
  quale classe sia. Addestrarlo a fare bene anche questo lo
  rende più prudente: dice quando è incerto, riconosce le cose mai viste ed è
  più difficile da ingannare.
- I **modelli di diffusione** sono modelli a energia che non lo dichiarano:
  imparano la pendenza di un paesaggio per ogni grado di sporco, e generare
  un'immagine è scendere, con addosso il rumore, lungo quella fila di
  paesaggi, dal più liscio (dove è difficile sbagliare direzione) al più
  dettagliato.
- Le **reti di Hopfield di oggi** tengono in memoria molti più ricordi di
  quelle del 1982, e il modo in cui li richiamano è, a un passaggio di
  distanza, l'attenzione dei Transformer: la domanda che si fa alla memoria è
  la stessa cosa che nell'attenzione decide a quali parole guardare. Con una
  sorpresa: guardando dentro i Transformer
  veri con questa lente, la maggior parte delle teste non richiama un ricordo
  solo, ne fa la media di moltissimi.
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
  punto fisso è raramente un ricordo singolo: nei primi strati è una media su
  moltissimi pattern, negli strati intermedi compaiono stati metastabili
  stretti, fino al quasi-richiamo di un ricordo solo, e negli ultimi prevalgono
  stati metastabili di ampiezza media.
- Le **quattro rinunce** di LeCun {cite}`lecun2022path`: generativo →
  incorporamento congiunto, probabilistico → energia, contrastivo →
  regolarizzato, RL → controllo predittivo. La seconda è la tesi di questo
  capitolo; la prima resta una scommessa, e il {doc}`capitolo sui world model </WorldModels/overview>` la
  discute per quello che è.
```
`````

L'energia, più che un modello, è una lente, ed è come lente che conviene
tenerla: un punteggio di compatibilità fra due cose, basso quando stanno bene
insieme, e nessun obbligo di trasformarlo in una probabilità. Resta però in
sospeso la domanda che questo capitolo ha incontrato a ogni pagina, da dove
arrivano gli esempi che tengono alto il resto del paesaggio.
«Auto-supervisione» la prende dall'altro capo e chiede da dove venga il segnale
di addestramento quando nessuno ha etichettato niente, che è poi la domanda da
cui dipende se un modello del genere impara qualcosa o impara a rispondere
sempre la stessa cosa.
