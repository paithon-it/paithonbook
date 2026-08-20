# Reti neurali su grafo

Nella Königsberg del primo Settecento (oggi Kaliningrad, in Russia) c'era un
passatempo cittadino. Il fiume Pregel divideva la città prussiana in quattro
lembi di terra, due sponde e due isole, cuciti insieme da sette ponti; e la
domanda che circolava era se esistesse una passeggiata che attraversasse ogni
ponte **una e una sola volta**. Nessuno ci riusciva, ma nessuno sapeva dire se
fosse davvero impossibile o solo difficile.

A rispondere fu il matematico svizzero Leonhard Euler. La sua mossa geniale
non fu camminare di più, ma buttare via la mappa. Le distanze, la forma delle
isole, la lunghezza dei ponti: niente di tutto questo contava. Contava solo
*quale lembo di terra fosse collegato a quale*. Euler ridusse allora la città
a quattro punti e sette linee (quelli che oggi chiamiamo **nodi** e **archi**)
e su quello scheletro dimostrò che la passeggiata non poteva esistere. Il suo
articolo, *Solutio Problematis ad Geometriam Situs Pertinentis*, è considerato
l'atto di nascita della **teoria dei grafi**: la matematica delle cose
collegate tra loro.

Quasi tre secoli dopo, quella stessa astrazione fa cose che Euler non avrebbe
immaginato. Nel 2020, un gruppo del MIT guidato da Jonathan Stokes e James
Collins pubblica su *Cell* la scoperta di un nuovo antibiotico: hanno
addestrato una rete neurale a leggere le molecole come grafi (atomi nei nodi,
legami chimici negli archi) e a prevedere quali fermassero la crescita dei
batteri. Le hanno poi fatto passare al setaccio un archivio di migliaia di
sostanze già preparate, e la rete ne ha segnalata una che nessuno associava
agli antibiotici; nei topi ha curato anche un'infezione da *Acinetobacter
baumannii* resistente a tutti gli antibiotici provati. L'hanno chiamata
**halicin**, in omaggio a HAL 9000, il computer di *2001: Odissea nello
spazio*. Il filo che unisce i sette ponti di Königsberg a un
antibiotico del XXI secolo è proprio l'oggetto di questo capitolo: le **reti
neurali su grafo** (*Graph Neural Networks*, GNN).

## Perché un capitolo dedicato

Fin qui il libro ha lavorato su due forme di dato molto ordinate. Le reti
convoluzionali del {doc}`capitolo sul deep learning </DeepLearning/overview>` suppongono una **griglia**: i
pixel di un'immagine hanno vicini fissi, sopra-sotto-destra-sinistra, sempre lo
stesso numero. Le reti ricorrenti suppongono una **sequenza**: le parole di una
frase arrivano in un ordine, una dopo l'altra. Sono ipotesi comode, e per
immagini e testo sono anche giuste.

Ma moltissimi dati del mondo non sono né una griglia né una sequenza. Sono
fatti di cose e dei legami fra quelle cose: si dice che sono dati
**relazionali**, perché quel che conta non sono i pezzi ma le relazioni.

- Una **molecola** è un insieme di atomi tenuti insieme da legami.
- Un **social network** è un insieme di persone tenute insieme da amicizie.
- Una **mappa stradale** è un insieme di incroci tenuti insieme da strade.
- Una rete di **transazioni** collega conti che si scambiano denaro; un
  **knowledge graph** collega concetti («Roma» (*è capitale di*) «Italia»).

Sotto la superficie, tutte queste cose hanno la stessa struttura: nodi e
archi. È esattamente ciò che Euler aveva capito guardando i ponti.

```{figure} ../figures/grafo-esempi.svg
:name: fig-grafo-esempi
:alt: Tre pannelli (una molecola, una rete sociale, una mappa stradale) condividono la stessa struttura di cinque nodi e sei archi; sotto, il grafo astratto comune con nodi e archi spogliati.
:width: 90%

Molecola, rete sociale e mappa stradale sembrano cose lontanissime, ma
condividono lo stesso scheletro di nodi e archi (in basso). La GNN lavora su
quello scheletro, qualunque cosa rappresenti.
```

Come mostra {numref}`fig-grafo-esempi`, se spogliamo questi oggetti delle loro
apparenze resta la stessa figura astratta. E qui nasce il problema tecnico: un
grafo è ostico da dare in pasto a una rete neurale ordinaria, per tre motivi.

`````{tab} Elementare

Immagina di dover elencare gli invitati a una cena e chi è amico di chi. In
che **ordine** scrivi i nomi? Non c'è un ordine giusto: Anna prima o dopo
Bruno è indifferente, l'importante è *chi conosce chi*. Se una rete neurale
si accorgesse dell'ordine in cui le passi i nomi, imparerebbe una sciocchezza,
perché quell'ordine non significa nulla. Una griglia di pixel e una frase, al
contrario, un ordine ce l'hanno eccome: il pixel in alto a sinistra è sempre
in alto a sinistra, la prima parola è sempre la prima.

E non è finita: a una cena ognuno ha un **numero diverso di amici** (c'è chi
ne ha due e chi dieci), mentre in un'immagine ogni pixel ha sempre lo stesso
numero di vicini. E ogni cena ha un **numero diverso di invitati**, mentre le
foto le possiamo tagliare tutte alla stessa misura. Ordine che non conta,
vicini in numero variabile, dimensione variabile: ecco perché un grafo non è
né una griglia né una sequenza, e serve un'idea nuova.

`````

`````{tab} Superiore

Formalmente un grafo è una coppia $G = (V, E)$, dove $V$ è l'insieme dei nodi e
$E \subseteq V \times V$ quello degli archi. Lo si rappresenta spesso con la
**matrice di adiacenza** $\mathbf{A} \in \{0,1\}^{N \times N}$ (con $N = |V|$),
in cui $A_{ij} = 1$ se esiste l'arco $(i,j)$. Ma questa rappresentazione
nasconde un'insidia: $\mathbf{A}$ dipende dall’**ordine** con cui numeriamo i
nodi. Rietichettare i nodi con una permutazione $\mathbf{P}$ trasforma
$\mathbf{A}$ in $\mathbf{P} \mathbf{A} \mathbf{P}^\top$ senza cambiare il
grafo. Un modello sensato deve quindi essere **invariante alla permutazione**
(se produce un'etichetta per l'intero grafo, non deve cambiare quando
rinumeriamo i nodi) o **equivariante** (se produce un'etichetta per ogni nodo,
le etichette devono seguire la permutazione). Questa è la simmetria che le CNN
hanno per la traslazione e che le GNN devono avere per la permutazione.

A ciò si aggiungono due irregolarità che rompono le architetture a griglia: il
**grado** dei nodi è variabile ($\deg(v)$ diverso da nodo a nodo), quindi non
esiste un «vicinato di dimensione fissa» su cui far scorrere un filtro; e la
dimensione $N$ del grafo cambia da esempio a esempio, quindi il modello deve
funzionare su grafi di taglia qualunque con gli **stessi** parametri
$\theta$.

`````

## L'idea in una frase

La soluzione è tanto semplice da enunciare quanto potente nelle conseguenze:

> dare a ogni nodo (o a ogni arco, o all'intero grafo) una **fila di numeri**
> che lo descrive, e costruirla facendo **circolare l'informazione lungo gli
> archi**: ogni nodo ascolta i suoi vicini e si aggiorna, e si ricomincia.

Quella fila di numeri si chiama **rappresentazione** del nodo, ed è la parola
che in questo capitolo torna più spesso: vuol dire sempre questo, la fila di
numeri con cui il modello descrive un nodo in un certo momento. All'inizio non
contiene niente di speciale, sono le informazioni che sul nodo abbiamo già noi
(per una persona l'età, per un atomo il tipo di elemento); a ogni giro di
ascolto diventa qualcosa di più.

Il resto è la macchina di sempre. Le operazioni che compongono un giro sono
tutte di quelle di cui si sa calcolare la derivata, e quindi la rete si
addestra dall'ingresso all'uscita con la stessa discesa del gradiente di ogni
altro modello di questo libro. Non serve un modo nuovo di imparare: serve solo
un modo di far parlare i nodi fra loro.

`````{tab} Elementare

Pensa a come ti fai un'idea di una persona che non conosci: guardi le
compagnie che frequenta. «Dimmi con chi vai e ti dirò chi sei.» Una rete su
grafo fa esattamente questo, a giri. All'inizio ogni nodo sa solo di sé; poi,
a ogni giro, ciascun nodo **guarda i suoi vicini**, raccoglie quello che sanno
e aggiorna la propria idea di sé. Dopo un giro, ogni nodo ha assorbito
qualcosa dagli amici diretti; dopo due giri, anche dagli amici degli amici; e
così l'informazione si diffonde per la rete come una voce che circola. Alla
fine, la fila di numeri di ogni nodo non descrive più solo il nodo, ma il nodo
*immerso nel suo pezzo di mondo*. Questo passaparola tra vicini ha un nome
(**message passing**, «scambio di messaggi») ed è il cuore del capitolo.

`````

`````{tab} Superiore

A ogni nodo $v$ si associa un vettore di stato $\mathbf{h}_v^{(k)}$, che parte
dalle sue *feature* iniziali $\mathbf{h}_v^{(0)} = \mathbf{x}_v$ e viene
raffinato per $K$ iterazioni. Ogni iterazione ha la stessa forma: raccogliere
($\mathrm{AGGREGATE}$) i vettori dei vicini $\mathcal{N}(v)$ e fonderli con il
proprio ($\mathrm{UPDATE}$),

$$
\mathbf{h}_v^{(k)} = \mathrm{UPDATE}^{(k)}\!\Big(
  \mathbf{h}_v^{(k-1)},\;
  \mathrm{AGGREGATE}^{(k)}\big(\{\, \mathbf{h}_u^{(k-1)} : u \in \mathcal{N}(v) \,\}\big)
\Big),
$$

dove $\mathcal{N}(v)$ è l'insieme dei vicini di $v$ e $\mathrm{AGGREGATE}$ è
un'operazione **invariante all'ordine** dei vicini (una somma, una media, un
massimo): proprio perché i vicini non hanno un ordine canonico. Dopo $K$ passi,
$\mathbf{h}_v^{(K)}$ riassume l'informazione contenuta nel sottografo a
distanza $K$ da $v$. Le funzioni $\mathrm{AGGREGATE}$ e $\mathrm{UPDATE}$ sono
reti neurali con parametri $\theta$ condivisi da tutti i nodi e tutti i grafi:
è questa condivisione a garantire l'equivarianza alla permutazione e a rendere
il modello indipendente dalla taglia del grafo. La sezione sul message passing
sviscera questo schema e ne ricava la sua incarnazione più celebre, la *Graph
Convolutional Network* (GCN).

`````

L'idea non è nuova, e ha una storia in buona parte italiana da datare bene. La
prima forma è di fine anni Novanta, con i lavori di Alessandro Sperduti e
Antonina Starita (1997) e di Paolo Frasconi, Marco Gori e Sperduti (1998);
reggevano però soltanto grafi in cui, seguendo le frecce, non si torna mai al
punto di partenza. Il primo modello che regge un grafo qualunque, giri chiusi
compresi, è del gruppo di Siena di Franco Scarselli e Marco Gori, proposto a
metà anni Duemila e pubblicato in forma estesa nel 2009
{cite}`scarselli2009graph`. Centrale, l'idea, lo è diventata solo nell'ultimo
decennio, quando si è capito come farla girare in fretta anche su grafi enormi
{cite}`hamilton2020graph`.

Una volta che ogni nodo ha la sua rappresentazione, che cosa ce ne facciamo?
Le domande che si possono fare a un grafo sono di tre tipi soltanto, e conviene
distinguerli fin da subito, perché ogni problema reale ricade in una di queste
tre caselle.

`````{tab} Elementare

Il primo tipo di domanda riguarda un **singolo nodo**: «questo account è un
bot?», «questo utente a quale categoria appartiene?». Il secondo riguarda un
**arco** che ancora non c'è: «queste due persone diventeranno amiche?», «a
questo cliente piacerà questo prodotto?»; è la domanda che sta dietro ai
suggerimenti di amicizia e alle raccomandazioni, e indovinare un collegamento
che non c'è ancora si chiama *link prediction*, «previsione dei
collegamenti». Il terzo riguarda l’**intero grafo** preso come un tutt'uno:
«questa molecola è tossica?», «questo composto uccide i batteri?», ed eccoci di
nuovo ad halicin. Nodo, arco, grafo intero: la stessa macchina, tre domande
diverse.

`````

`````{tab} Superiore

I tre livelli corrispondono ad altrettante famiglie di compiti. A livello di
**nodo**: classificazione o regressione dei nodi (per esempio l'assegnazione di
una categoria a partire dallo stato finale $\mathbf{h}_v^{(K)}$), tipicamente
in regime *semi-supervisionato* (pochi nodi etichettati, il grafo intero come
contesto). A livello di **arco**: *link prediction*, cioè stimare la
probabilità che esista un arco $(u,v)$ a partire dalla coppia
$(\mathbf{h}_u^{(K)}, \mathbf{h}_v^{(K)})$. A livello di **grafo**: si
aggregano ($\mathrm{READOUT}$) tutti i vettori dei nodi in un unico vettore del
grafo, su cui fare classificazione o regressione; qui il regime è di norma
*induttivo*, perché ogni esempio è un grafo a sé e a test se ne incontrano di
mai visti in addestramento.

`````

## La mappa del capitolo

Il capitolo procede dal dato all'architettura.

- **Il mondo come grafo**. Come si mette un problema «in forma di grafo»: cosa
  sono nodi e archi, e che cosa c'è scritto su ciascuno (le loro
  caratteristiche, in gergo le *feature*); collegamenti con e senza verso, con
  e senza peso, di più tipi; e i tre tipi di domanda appena visti, con esempi
  concreti.
- **Message passing**. Il meccanismo di propagazione vicino-per-vicino nella
  sua forma generale, e da lì, passo dopo passo, la *Graph Convolutional
  Network* (**GCN**): il modello che ha fatto delle GNN uno strumento pratico,
  presentato nel 2017 da Thomas Kipf e Max Welling.
- **I knowledge graph**, cioè i grafi di fatti. Che cosa cambia quando ogni
  arco porta scritto sopra un verbo, e la coppia di nodi con il verbo in mezzo
  è un fatto sul mondo: («Roma», *è capitale di*, «Italia»). Come si mettono in
  numeri quei fatti, perché un arco che manca vuol dire «non lo so» e non «è
  falso», e perché a una domanda si può rispondere **camminando sul grafo** da
  un fatto all'altro invece di cercare la pagina che la contiene.
- **Oltre la GCN: GraphSAGE, GAT e applicazioni**. Le due varianti che hanno
  reso le GNN utilizzabili su scala reale: *GraphSAGE*, che guarda solo un
  campione dei vicini e regge così grafi enormi, e la *Graph Attention
  Network* (GAT), che pesa i vicini con l'attenzione incontrata nel capitolo
  sui Transformer. Poi come si passa dai nodi a un verdetto sull'intero grafo, e
  la sorpresa che ne viene fuori: esistono coppie di grafi diversi che nessuna
  rete di questa famiglia riuscirà mai a distinguere. Una carrellata di
  applicazioni, dalla chimica alla frode, dalle mappe ai sistemi
  di raccomandazione; i limiti; e infine i *Graph Transformer*, che rifanno la
  strada in senso inverso: non una rete su grafo che assomiglia a un
  Transformer, ma un Transformer messo a lavorare su un grafo.

## Tre fili che tornano

Il primo è la **convoluzione**. Nel {doc}`capitolo sul deep learning </DeepLearning/overview>` abbiamo visto
un filtro scorrere su una griglia di pixel, combinando ogni pixel con i suoi
vicini. Il message passing è la stessa idea (combinare un elemento con i suoi
vicini) liberata dal vincolo della griglia: i «vicini» non sono più i quattro
pixel adiacenti, ma i nodi collegati da un arco, in numero variabile. In questo
senso la rete su grafo **generalizza** la rete convoluzionale, la CNN dei
capitoli sulle immagini, a dati che una griglia non la formano.

Su questo c'è un modo di guardare le cose da conoscere, e si chiama *geometric
deep learning* {cite}`bronstein2021geometric`. Parte da una domanda sola: che
cosa si può fare a un dato senza cambiarne il significato? Un'immagine
spostata di un pixel contiene sempre lo stesso gatto; un grafo con i nodi
rinumerati è sempre lo stesso grafo. Queste trasformazioni che non cambiano la
risposta si chiamano **simmetrie** del dato, e una volta elencate dicono come
deve essere fatta la rete che ci lavora sopra. Da quel punto di vista reti
convoluzionali, reti ricorrenti e reti su grafo non sono tre invenzioni
separate: sono la stessa ricetta applicata a tre elenchi di simmetrie diversi.

Il secondo filo porta ai **sistemi di raccomandazione**. Lì il dato è, per sua
natura, un grafo: da un lato gli utenti, dall'altro gli oggetti, e un arco
ogni volta che un utente interagisce con un oggetto. Un grafo fatto così, con i
nodi divisi in due squadre e archi solo fra una squadra e l'altra (mai fra due
utenti, mai fra due prodotti), si dice **bipartito**, e la parola tornerà più
volte nel capitolo. La *link prediction* su questo grafo è, letteralmente, il
problema della raccomandazione: prevedere gli archi che ancora non ci sono.
Non è un caso che le GNN siano oggi il motore dei sistemi di raccomandazione
dei grandi servizi. Il capitolo dedicato le riprenderà da vicino.

Il terzo filo è il meno ovvio dei tre e il più utile, perché porta a un
capitolo che sembrava parlare d'altro: quello sui **Transformer**, i modelli
che leggono e scrivono il linguaggio.

`````{tab} Elementare

Nel capitolo sui Transformer si è visto come un modello legge una frase. Anche
lì ogni parola ha la sua fila di numeri, la sua rappresentazione. Per farsi
un'idea di una parola il modello non guarda solo quella: la confronta con tutte
le altre della frase e decide quanto ciascuna conta, dando a ognuna un peso;
poi la nuova rappresentazione di quella parola è il miscuglio di quelle delle
altre, dosato secondo quei pesi. È il meccanismo che lì si chiama
**attenzione**.

Adesso rileggi la stessa cosa con le parole di questo capitolo. «Ogni parola
guarda tutte le altre» vuol dire: c'è un grafo in cui ogni parola è un nodo, e
ogni nodo è collegato a tutti gli altri (un grafo così si chiama **completo**).
«Decide quanto ciascuna conta» vuol dire: ogni collegamento porta un peso. E
«la nuova rappresentazione è il miscuglio delle altre» è, parola per parola, il
passaparola fra vicini descritto qui sopra. Un Transformer, insomma, sta già
facendo message passing: solo che il grafo non glielo dà nessuno, se lo
fabbrica collegando tutti con tutti.

Il che ha un vantaggio e un prezzo, e conviene vederli in coppia perché
tornano per tutto il capitolo. Il vantaggio è che non gli serve sapere niente
su chi è collegato a chi: funziona anche quando i collegamenti veri nessuno li
conosce. Il prezzo è che collegare tutti con tutti costa, e il conto è presto
fatto: ogni parola va confrontata con ogni parola, quindi con dieci parole sono
dieci per dieci, cento confronti, e con mille parole un milione. Da qui una
cosa che a prima vista non c'entra niente: buona parte della ricerca su come
rendere l'attenzione meno costosa consiste, alla lettera, nel **togliere
archi** da quel grafo completo.

`````

`````{tab} Superiore

Nel Transformer ogni token calcola la propria nuova rappresentazione come
**somma pesata di quelle di tutti gli altri**, con pesi appresi. Detto così è
una frase su un'architettura per il linguaggio; riletta con il vocabolario di
questo capitolo è la definizione di un passo di message passing, con l'unica
particolarità che il grafo è **completo**: ogni token è collegato a ogni altro,
e i coefficienti di attenzione sono i pesi degli archi.

Non è un'analogia costruita a posteriori. Le rassegne che hanno unificato il
campo lo dicono esplicitamente: il quadro delle *message passing neural
network* {cite}`gilmer2017neural` copre le GNN, quello delle *non-local neural
network* {cite}`wang2018non` copre i metodi «in stile self-attention» a partire
dal Transformer, e le *graph network* di Battaglia e colleghi
{cite}`battaglia2018relational` li tengono insieme in un'unica formulazione,
elencando fra i metodi coperti anche la GAT che incontreremo fra poco. Ne
discende una lettura che vale in entrambe le direzioni: la GAT è la
self-attention applicata a un grafo sparso invece che completo; e un
Transformer è una GNN che ha rinunciato alla struttura, pagando in costo
quadratico la libertà di non doverla conoscere. Da lì si capisce anche perché
tanta ricerca sull'efficienza dell'attenzione somigli a teoria dei grafi:
renderla sparsa vuol dire, letteralmente, togliere archi.

`````

Questo terzo filo tornerà due volte: quando incontreremo la GAT, che pesa i
vicini con l'attenzione, e alla fine del capitolo, dove si prova a fare la
strada al contrario e a mettere un Transformer su un grafo qualunque.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Moltissimi dati sono fatti di **cose collegate fra loro** (molecole, reti di
  amicizie, mappe stradali, pagamenti, fatti sul mondo): tutti hanno la stessa
  struttura di **puntini e linee**, cioè **nodi e archi**, l'astrazione che
  Euler inventò nel 1736 sui ponti di Königsberg.
- Un grafo non è né una griglia di pixel né una fila di parole, e per tre
  motivi: **l'ordine in cui si elencano i nodi non conta**, ogni nodo ha **un
  numero diverso di vicini** e ogni grafo ha **un numero diverso di nodi**.
  Serve un modello a cui, se riordini l'elenco, non cambi la risposta.
- L'idea delle **GNN** è dare a ogni nodo una fila di numeri che lo descrive, e
  costruirla facendo **circolare l'informazione lungo i collegamenti**: a ogni
  giro ogni nodo ascolta i vicini e si aggiorna. Il meccanismo si chiama
  **message passing**, «dimmi con chi vai e ti dirò chi sei».
- Le domande sono di **tre tipi**: su un nodo, su un collegamento che ancora non
  c'è (prevederlo si chiama *link prediction*), sull'intero grafo.
- Le GNN fanno per i grafi quello che le reti convoluzionali fanno per le
  immagini, e sono il motore dei moderni **sistemi di raccomandazione**, dove
  il grafo ha gli utenti da una parte e i prodotti dall'altra.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Moltissimi dati sono **relazionali** (molecole, social network, mappe,
  transazioni, knowledge graph) e hanno tutti la stessa struttura di **nodi e
  archi**, l'astrazione che Euler inventò nel 1736 sui ponti di Königsberg.
- Un grafo non è né una griglia (come per le CNN) né una sequenza (come per le
  RNN): i nodi non hanno **ordine canonico**, hanno **grado variabile**, e il
  grafo ha **dimensione variabile**. Serve un modello **invariante alla
  permutazione**.
- L'idea delle **GNN** è imparare una rappresentazione di nodi/archi/grafo
  facendo **propagare l'informazione lungo gli archi**, in modo differenziabile
  ed end-to-end. Il meccanismo si chiama **message passing**.
- I compiti sono a **tre livelli**: nodo (classificazione), arco (*link
  prediction*), grafo intero (classificazione o regressione).
- Le GNN **generalizzano la convoluzione** a domini non a griglia e alimentano
  i moderni **sistemi di raccomandazione** (grafo bipartito utente-prodotto).
```

`````
