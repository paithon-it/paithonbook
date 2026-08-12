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
batteri. Passata al setaccio una libreria di composti, la rete ha segnalato
una molecola che nessuno associava agli antibiotici; funziona anche su un
ceppo di *Acinetobacter baumannii* resistente a tutti gli antibiotici provati.
L'hanno chiamata **halicin**, in omaggio a HAL
9000. Il filo che unisce i sette ponti di Königsberg a un antibiotico del XXI
secolo è proprio l'oggetto di questo capitolo: le **reti neurali su grafo**
(*Graph Neural Networks*, GNN).

## Perché un capitolo dedicato

Fin qui il libro ha lavorato su due forme di dato molto ordinate. Le reti
convoluzionali del capitolo sul deep learning suppongono una **griglia**: i
pixel di un'immagine hanno vicini fissi, sopra-sotto-destra-sinistra, sempre lo
stesso numero. Le reti ricorrenti suppongono una **sequenza**: le parole di una
frase arrivano in un ordine, una dopo l'altra. Sono ipotesi comode, e per
immagini e testo sono anche giuste.

Ma moltissimi dati del mondo non sono né una griglia né una sequenza. Sono
**relazionali**: fatti di entità e delle relazioni tra loro.

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
nasconde un'insidia: $\mathbf{A}$ dipende dall'**ordine** con cui numeriamo i
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

> imparare una **rappresentazione** (una fila di numeri) per ogni nodo, per
> ogni arco o per l'intero grafo, facendo **propagare l'informazione lungo gli
> archi**; e farlo con operazioni di cui si sa calcolare la derivata, così che
> la macchina intera si addestri in un colpo solo, dall'ingresso all'uscita,
> con la stessa discesa del gradiente di ogni altra rete di questo libro.

L'idea di far girare una rete neurale direttamente su un grafo non è nuova, e
vale la pena datarla bene. Le reti ricorsive su strutture di Alessandro
Sperduti e Antonina Starita (1997) e di Paolo Frasconi, Marco Gori e Sperduti
(1998) sono la prima forma, e trattavano però soltanto grafi aciclici diretti.
Il modello del gruppo di Siena di Franco Scarselli e Marco Gori, proposto a
metà anni Duemila e pubblicato in forma estesa nel 2009
{cite}`scarselli2009graph`, è il primo a coprire i grafi qualunque, cicli
compresi. L'idea è diventata poi centrale nell'ultimo decennio, quando si è
capito come renderla efficiente e scalabile {cite}`hamilton2020graph`.

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

Le rappresentazioni così ottenute rispondono a domande a **tre livelli**, e
conviene tenerle distinte fin da subito perché ogni compito reale ricade in una
di queste tre caselle.

`````{tab} Elementare

Su un grafo puoi fare tre tipi di domanda. Su un **singolo nodo**: «questo
account è un bot?», «questo utente a quale categoria appartiene?». Su un
**arco** che ancora non c'è: «queste due persone diventeranno amiche?», «a
questo cliente piacerà questo prodotto?»; è la domanda che sta dietro ai
suggerimenti di amicizia e alle raccomandazioni. Oppure sull'**intero grafo**
preso come un tutt'uno: «questa molecola è tossica?», «questo composto uccide
i batteri?», ed eccoci di nuovo ad halicin. Nodo, arco, grafo intero: la
stessa macchina, tre domande diverse.

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
grafo, su cui fare classificazione o regressione; il regime *induttivo*, in cui
a test si incontrano grafi mai visti in addestramento.

`````

## La mappa del capitolo

Il capitolo procede dal dato all'architettura.

- **Il mondo come grafo**. Come si mette un problema «in forma di grafo»: cosa
  sono nodi, archi e le loro *feature*; grafi diretti e non diretti, pesati,
  con più tipi di relazione; e i tre livelli di compito (nodo, arco, grafo
  intero) appena introdotti, con esempi concreti.
- **Message passing**. Il meccanismo di propagazione vicino-per-vicino nella
  sua forma generale, e la derivazione della *Graph Convolutional Network*: il
  modello che ha fatto delle GNN uno strumento pratico, presentato nel 2017 da
  Thomas Kipf e Max Welling.
- **I knowledge graph**. Che cosa cambia quando gli archi hanno un'etichetta e
  sono **fatti**: le triple, l'assunzione di mondo aperto (un arco che manca
  vuol dire «non lo so»), le entità come punti e le relazioni come frecce, e a
  che serve poter rispondere **navigando** invece che recuperando.
- **Oltre la GCN: GraphSAGE, GAT e applicazioni**. Le varianti che hanno reso
  le GNN utilizzabili su scala reale, *GraphSAGE*, che campiona i vicini per
  scalare a grafi enormi, e la *Graph Attention Network* (GAT), che pesa i
  vicini con l'attenzione incontrata nel capitolo sui Transformer, e una
  carrellata di applicazioni, dalla chimica alla frode, dalle mappe ai sistemi
  di raccomandazione.

## Tre fili che tornano

Vale la pena legare esplicitamente questo capitolo a tre che il libro ha già
percorso.

Il primo è la **convoluzione**. Nel capitolo sul deep learning abbiamo visto
un filtro scorrere su una griglia di pixel, combinando ogni pixel con i suoi
vicini. Il message passing è la stessa idea (combinare un elemento con i suoi
vicini) liberata dal vincolo della griglia: i «vicini» non sono più i quattro
pixel adiacenti, ma i nodi collegati da un arco, in numero variabile. In
questo senso la GNN **generalizza** la CNN a domini irregolari; è la
prospettiva del *geometric deep learning*, che legge CNN, reti ricorrenti e
GNN come casi particolari di uno stesso principio: sfruttare le simmetrie del
dominio del dato {cite}`bronstein2021geometric`.

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

Nel capitolo sui Transformer si è visto come un modello legge una frase. Per
farsi un'idea di una parola non guarda solo quella: la confronta con tutte le
altre della frase e decide quanto ciascuna conta, dando a ognuna un peso; poi
la nuova descrizione di quella parola è il miscuglio delle descrizioni delle
altre, dosato secondo quei pesi. È il meccanismo che lì si chiama
**attenzione**.

Adesso rileggi la stessa cosa con le parole di questo capitolo. «Ogni parola
guarda tutte le altre» vuol dire: c'è un grafo in cui ogni parola è un nodo, e
ogni nodo è collegato a tutti gli altri (un grafo così si chiama **completo**).
«Decide quanto ciascuna conta» vuol dire: ogni collegamento porta un peso. E
«la nuova descrizione è il miscuglio delle altre» è, parola per parola, il
passaparola fra vicini descritto qui sopra. Un Transformer, insomma, sta già
facendo message passing: solo che il grafo non glielo dà nessuno, se lo
fabbrica collegando tutti con tutti.

Il che ha un vantaggio e un prezzo, e conviene vederli in coppia perché
tornano per tutto il capitolo. Il vantaggio è che non gli serve sapere niente
su chi è collegato a chi: funziona anche quando i collegamenti veri nessuno li
conosce. Il prezzo è che collegare tutti con tutti costa: con dieci parole sono
cento collegamenti, con mille parole un milione. Da qui una cosa che a prima
vista non c'entra niente: buona parte della ricerca su come rendere
l'attenzione meno costosa consiste, alla lettera, nel **togliere archi** da
quel grafo completo.

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
