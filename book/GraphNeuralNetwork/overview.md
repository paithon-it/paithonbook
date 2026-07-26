# Reti neurali su grafo

Nell'estate del 1736 la città prussiana di Königsberg (oggi Kaliningrad, in
Russia) aveva un passatempo cittadino. Il fiume Pregel la divideva in quattro
lembi di terra, due sponde e due isole, cuciti insieme da sette ponti; e la
domanda che circolava era se esistesse una passeggiata che attraversasse ogni
ponte **una e una sola volta**, tornando al punto di partenza. Nessuno ci
riusciva, ma nessuno sapeva dire se fosse davvero impossibile o solo
difficile.

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
una molecola che nessuno associava agli antibiotici; funziona contro ceppi
resistenti a ogni farmaco noto. L'hanno chiamata **halicin**, in omaggio a HAL
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

Formalmente un grafo è una coppia $G = (V, E)$, dove $V$ è l'insieme dei nodi
e $E \subseteq V \times V$ quello degli archi. Lo si rappresenta spesso con la
**matrice di adiacenza** $A \in \{0,1\}^{n \times n}$ (con $n = |V|$), in cui
$A_{ij} = 1$ se esiste l'arco $(i,j)$. Ma questa rappresentazione nasconde
un'insidia: $A$ dipende dall'**ordine** con cui numeriamo i nodi. Rietichettare
i nodi con una permutazione $P$ trasforma $A$ in $P A P^\top$ senza cambiare il
grafo. Un modello sensato deve quindi essere **invariante alla permutazione**
(se produce un'etichetta per l'intero grafo, non deve cambiare quando
rinumeriamo i nodi) o **equivariante** (se produce un'etichetta per ogni nodo,
le etichette devono seguire la permutazione). Questa è la simmetria che le CNN
hanno per la traslazione e che le GNN devono avere per la permutazione.

A ciò si aggiungono due irregolarità che rompono le architetture a griglia: il
**grado** dei nodi è variabile ($\deg(v)$ diverso da nodo a nodo), quindi non
esiste un «vicinato di dimensione fissa» su cui far scorrere un filtro; e la
dimensione $n$ del grafo cambia da esempio a esempio, quindi il modello deve
funzionare su grafi di taglia qualunque con gli **stessi** parametri
$\theta$.

`````

## L'idea in una frase

La soluzione è tanto semplice da enunciare quanto potente nelle conseguenze:

> imparare una **rappresentazione** (un vettore di numeri) per ogni nodo, per
> ogni arco o per l'intero grafo, facendo **propagare l'informazione lungo gli
> archi**: in modo differenziabile, così che l'intera cosa si addestri >
end-to-end con la discesa del gradiente.

L'idea di far girare una rete neurale direttamente su un grafo non è nuova:
risale al modello di Franco Scarselli e colleghi, ricercatori italiani, del
2009 {cite}`scarselli2009graph`. È diventata però centrale nell'ultimo decennio,
quando si è capito come renderla efficiente e scalabile
{cite}`hamilton2020graph`.

`````{tab} Elementare

Pensa a come ti fai un'idea di una persona che non conosci: guardi le
compagnie che frequenta. «Dimmi con chi vai e ti dirò chi sei.» Una rete su
grafo fa esattamente questo, a giri. All'inizio ogni nodo sa solo di sé; poi,
a ogni giro, ciascun nodo **guarda i suoi vicini**, raccoglie quello che sanno
e aggiorna la propria idea di sé. Dopo un giro, ogni nodo ha assorbito
qualcosa dagli amici diretti; dopo due giri, anche dagli amici degli amici; e
così l'informazione si diffonde per la rete come una voce che circola. Alla
fine, il vettore di ogni nodo non descrive più solo il nodo, ma il nodo
*immerso nel suo pezzo di mondo*. Questo passaparola tra vicini ha un nome
(**message passing**, «scambio di messaggi») ed è il cuore del capitolo.

`````

`````{tab} Superiore

A ogni nodo $v$ si associa un vettore di stato $h_v^{(k)}$, che parte dalle
sue *feature* iniziali $h_v^{(0)} = x_v$ e viene raffinato per $K$ iterazioni.
Ogni iterazione ha la stessa forma: raccogliere ($\mathrm{AGGREGATE}$) i
vettori dei vicini $\mathcal{N}(v)$ e fonderli con il proprio
($\mathrm{UPDATE}$),

$$
h_v^{(k)} = \mathrm{UPDATE}^{(k)}\!\Big(
  h_v^{(k-1)},\;
  \mathrm{AGGREGATE}^{(k)}\big(\{\, h_u^{(k-1)} : u \in \mathcal{N}(v) \,\}\big)
\Big),
$$

dove $\mathcal{N}(v)$ è l'insieme dei vicini di $v$ e $\mathrm{AGGREGATE}$ è
un'operazione **invariante all'ordine** dei vicini (una somma, una media, un
massimo): proprio perché i vicini non hanno un ordine canonico. Dopo $K$
passi, $h_v^{(K)}$ riassume l'informazione contenuta nel sottografo a distanza
$K$ da $v$. Le funzioni $\mathrm{AGGREGATE}$ e $\mathrm{UPDATE}$ sono reti
neurali con parametri $\theta$ condivisi da tutti i nodi e tutti i grafi: è
questa condivisione a garantire l'equivarianza alla permutazione e a rendere
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
**nodo**: classificazione o regressione dei nodi (per esempio l'assegnazione
di una categoria a partire dallo stato finale $h_v^{(K)}$), tipicamente in
regime *semi-supervisionato* (pochi nodi etichettati, il grafo intero come
contesto). A livello di **arco**: *link prediction*, cioè stimare la
probabilità che esista un arco $(u,v)$ a partire dalla coppia
$(h_u^{(K)}, h_v^{(K)})$. A livello di **grafo**: si aggregano
($\mathrm{READOUT}$) tutti i vettori dei nodi in un unico vettore del grafo,
su cui fare classificazione o regressione; il regime *induttivo*, in cui a
test si incontrano grafi mai visti in addestramento.

`````

## La mappa del capitolo

Il capitolo procede in tre tappe, dal dato all'architettura.

- **Il mondo come grafo**. Come si mette un problema «in forma di grafo»: cosa
  sono nodi, archi e le loro *feature*; grafi diretti e non diretti, pesati,
  con più tipi di relazione; e i tre livelli di compito (nodo, arco, grafo
  intero) appena introdotti, con esempi concreti.
- **Message passing**. Il meccanismo di propagazione vicino-per-vicino nella
  sua forma generale, e la derivazione della *Graph Convolutional Network*: il
  modello che ha fatto delle GNN uno strumento pratico, presentato nel 2017 da
  Thomas Kipf e Max Welling.
- **Oltre la GCN: GraphSAGE, GAT e applicazioni**. Le varianti che hanno reso
  le GNN utilizzabili su scala reale, *GraphSAGE*, che campiona i vicini per
  scalare a grafi enormi, e la *Graph Attention Network* (GAT), che pesa i
  vicini con l'attenzione incontrata nel capitolo sui Transformer, e una
  carrellata di applicazioni, dalla chimica alla frode, dalle mappe ai sistemi
  di raccomandazione.

## Due fili che tornano

Vale la pena legare esplicitamente questo capitolo a due che il libro ha già
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
ogni volta che un utente interagisce con un oggetto (un grafo *bipartito*
utente-item). La *link prediction* su questo grafo è, letteralmente, il
problema della raccomandazione: prevedere gli archi che ancora non ci sono.
Non è un caso che le GNN siano oggi il motore dei sistemi di raccomandazione
dei grandi servizi. Il capitolo dedicato le riprenderà da vicino.

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
  i moderni **sistemi di raccomandazione** (grafo bipartito utente-item).
```
