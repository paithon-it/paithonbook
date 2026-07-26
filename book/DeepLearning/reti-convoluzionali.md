# Reti convoluzionali (CNN)

Prendi una foto di un gatto e spostalo di dieci pixel a destra: per te è
ancora, banalmente, un gatto. Per una rete fatta come quelle del capitolo
scorso, invece, è diventato un input completamente diverso. Questo scarto tra
come *noi* vediamo un'immagine e come la vede una rete neurale ordinaria è il
problema che le **reti convoluzionali** (Convolutional Neural Networks, CNN)
sono nate per risolvere. L'idea affonda le radici nella biologia (gli studi di
Hubel e Wiesel sulla corteccia visiva del gatto negli anni Sessanta) e prende
forma nel *Neocognitron* di Fukushima {cite}`fukushima1980neocognitron` e
nella LeNet-5 di Yann LeCun (1998), che leggeva le cifre scritte a mano sugli
assegni. Nel 2012 AlexNet, di Krizhevsky, Sutskever e Hinton, vince la
competizione ImageNet con un margine tale da riaccendere l'intero campo del
deep learning.

## Perché uno strato denso non basta

`````{tab} Elementare

Uno strato **denso** (il "completamente connesso" che conosciamo dal capitolo
sulle reti neurali) collega *ogni* pixel a *ogni* neurone. Sembra generoso, ma
è uno spreco. Una foto a colori di 256×256 pixel sono quasi 200.000 numeri: un
solo strato denso con 1000 neuroni avrebbe quasi 200 milioni di pesi da
imparare (solo per il primo strato). Troppi.

E c'è un problema più profondo. Se la rete impara a riconoscere un occhio
quando compare in alto a sinistra, non sa nulla dello stesso occhio in basso
a destra: per lei sono due cose diverse, perché occupano posizioni (e quindi
pesi) diversi. Manca l'idea che *un motivo è lo stesso ovunque appaia*.

`````

`````{tab} Superiore

Uno strato *fully-connected* su un input $\mathbf{x}\in\mathbb{R}^{d}$ con
$h$ unità richiede $h\cdot d$ pesi. Per un'immagine RGB $256\times256$ si ha
$d = 256\cdot256\cdot3 \approx 1{,}97\times10^{5}$: con $h=1000$ servono circa
$2\times10^{8}$ parametri, un invito all'*overfitting*.

Soprattutto, lo strato denso non possiede **invarianza alla traslazione**: un
pattern spostato di un vettore $\Delta$ attiva pesi diversi, perché l'indice
della componente cambia. Le CNN reintroducono questa struttura via due vincoli
architetturali (connettività locale e condivisione dei pesi) che riducono i
parametri e rendono la risposta *equivariante* alla traslazione (sposti
l'input, si sposta l'attivazione).

`````

## La convoluzione: un filtro che scorre

Il cuore della rete è la **convoluzione**: un piccolo filtro (o *kernel*),
tipicamente $3\times3$, che scivola su tutta l'immagine. In ogni posizione
sovrappone il filtro alla porzione di immagine sottostante, moltiplica valore
per valore e somma il tutto in *un* numero. Quel numero misura quanto bene
quella porzione somiglia al motivo che il filtro cerca: un bordo verticale,
una macchia di colore, una texture.

```{figure} ../figures/convoluzione.svg
:name: fig-convoluzione
:alt: Un kernel 3x3 evidenziato su una porzione di una griglia di input 5x5; una freccia collega la regione alla singola cella corrispondente nella feature map di output.
:width: 85%

Il kernel $3\times3$ copre una finestra dell'input; la somma pesata dei nove
valori produce **una** cella della feature map. Facendo scorrere la finestra
si riempie l'intera mappa.
```

Come mostra {numref}`fig-convoluzione`, il filtro guarda solo una finestra per
volta, ma quella *stessa* finestra visita ogni angolo dell'immagine.

`````{tab} Elementare

Immagina uno stampino traforato con nove caselle, che fai scorrere su una
pagina a quadretti. In ogni punto guardi i nove quadretti sotto lo stampino,
li combini secondo una ricetta fissa e scrivi il risultato su un foglio nuovo.
Lo stampino non cambia mai mentre scorre: se è bravo a trovare un bordo, lo
trova ovunque nell'immagine, in alto come in basso. È questo il trucco che
mancava allo strato denso.

`````

`````{tab} Superiore

In due dimensioni, con input $I$ e kernel $K$, l'operazione (tecnicamente una
*cross-correlazione*, come da convenzione nelle librerie di deep learning) è

$$
S(i,j) = \sum_{m}\sum_{n} I(i+m,\, j+n)\; K(m,n).
$$

Qui $S(i,j)$ è il valore in posizione $(i,j)$ della mappa di uscita, mentre
$m,n$ scorrono sulle celle del kernel. Con più canali in ingresso (es. RGB) e
un bias $b$, seguiti da una non linearità $\sigma$ (di solito la ReLU):

$$
a_{i,j} = \sigma\!\left( b + \sum_{c}\sum_{m}\sum_{n}
K_{c,m,n}\; I_{c,\,i+m,\,j+n} \right).
$$

I simboli: $c$ indicizza i canali d'ingresso, $K_{c,m,n}$ è il peso del filtro
per canale $c$ e posizione $(m,n)$, $a_{i,j}$ l'attivazione risultante.

`````

```{figure} ../figures/convoluzione.gif
:name: fig-convoluzione-animata
:alt: "Animazione: una finestra 3x3 scorre sulle nove posizioni di un'immagine 5x5 che contiene una barra verticale; a ogni posizione si riempie la cella corrispondente della mappa 3x3, con valori -3 sulla colonna di sinistra, 0 al centro e +3 a destra."
:width: 90%

La stessa operazione in movimento, con un kernel che cerca **bordi verticali**:
la mappa risponde $-3$ dove la barra comincia, $+3$ dove finisce e $0$ nel
mezzo, dove non c'è nessun bordo da trovare.
```

Vale la pena leggere i numeri della {numref}`fig-convoluzione-animata`: il
filtro non "vede" la barra: vede i suoi **bordi**, e li segna con segno opposto
a seconda del verso. È già un abbozzo di ciò che i primi strati di una CNN
imparano da soli.

## Campi recettivi locali e pesi condivisi

Due principi rendono tutto ciò possibile. Il **campo recettivo locale**: ogni
neurone convoluzionale guarda solo una piccola finestra dell'input, non
l'intera immagine. La **condivisione dei pesi**: lo stesso kernel viene usato
in ogni posizione, quindi i pochi pesi che lo compongono sono riutilizzati
migliaia di volte.

`````{tab} Elementare

Un filtro $3\times3$ su un'immagine a colori ha solo $3\times3\times3+1 = 28$
numeri da imparare. Con 32 filtri diversi arrivi a meno di mille parametri,
contro i milioni dello strato denso. Pochi pesi, riusati ovunque: la rete
impara *cosa* cercare, non *dove*.

`````

`````{tab} Superiore

Un layer convoluzionale con $F$ filtri, kernel $k\times k$ e $C$ canali in
ingresso ha $F\,(k^2 C + 1)$ parametri, **indipendentemente** dalla
risoluzione dell'immagine. Il costo per pixel è disaccoppiato dal numero di
parametri: è la condivisione dei pesi (*weight tying*) che impone
l'equivarianza traslazionale come *prior* strutturale, riducendo drasticamente
lo spazio delle ipotesi e quindi il rischio di overfitting.

`````

L'uscita di un filtro è una **feature map**: una mappa che segna, punto per
punto, *dove* nell'immagine è presente il motivo cercato. Un layer produce una
pila di feature map, una per filtro; i primi layer imparano motivi elementari
(bordi, angoli), i più profondi li combinano in parti sempre più astratte
(occhi, ruote, volti).

## Il pooling: ridurre per generalizzare

Dopo la convoluzione si applica quasi sempre il **pooling**, che rimpicciolisce
le feature map tenendo solo l'informazione saliente. Il più comune è il
**max pooling**: su ogni finestra (di solito $2\times2$) conserva il valore
massimo.

`````{tab} Elementare

Il max pooling è come chiedere, per ogni quadratino $2\times2$: "il motivo qui
intorno c'è, sì o no?", tenendo solo la risposta più forte. Dimezza larghezza
e altezza, quindi alleggerisce il calcolo, e regala un po' di tolleranza: se il
motivo si sposta di un pixel, il massimo della zona resta lo stesso.

`````

`````{tab} Superiore

Per una regione $\mathcal{R}_{i,j}$ della feature map,

$$
y_{i,j} = \max_{(m,n)\,\in\,\mathcal{R}_{i,j}} x_{m,n}.
$$

Non ha parametri da apprendere. Sottocampionando, aumenta il campo recettivo
effettivo dei layer successivi e introduce una modesta **invarianza a piccole
traslazioni** locali, complementare all'equivarianza della convoluzione.

`````

## L'architettura tipica

Lo schema classico alterna blocchi **conv → ReLU → pool**, ripetuti alcune
volte, e chiude con uno o più strati densi che trasformano le feature astratte
in una decisione (la classe dell'immagine). Due manopole governano lo
scorrimento del filtro: lo **stride** $s$, di quanti pixel salta la finestra a
ogni passo (stride 2 dimezza la risoluzione), e il **padding** $p$, la cornice
di zeri aggiunta ai bordi per non perdere i pixel di frontiera. La dimensione
di uscita lungo un asse è

$$
o = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1,
$$

dove $n$ è la dimensione d'ingresso e $k$ quella del kernel. Un esempio con i
numeri del codice qui sotto, ingresso $n=28$, kernel $k=3$, padding $p=1$,
stride $s=1$: $o = \lfloor(28 + 2 - 3)/1\rfloor + 1 = 28$, la risoluzione non
cambia. Con `padding="same"` si sceglie appunto $p$ affinché $o=n$ (a stride
1): l'uscita conserva la risoluzione dell'ingresso.

In PyTorch l'intera architettura sta in poche righe:

```python
from torch import nn

# input: un batch di immagini in scala di grigi, shape (N, 1, 28, 28)
model = nn.Sequential(
    # blocco 1: 32 filtri 3x3, mappe grandi come l'input
    nn.Conv2d(1, 32, 3, padding="same"), nn.ReLU(),
    nn.MaxPool2d(2),               # 28x28 -> 14x14
    # blocco 2: piu' filtri man mano che le mappe rimpiccioliscono
    nn.Conv2d(32, 64, 3, padding="same"), nn.ReLU(),
    nn.MaxPool2d(2),               # 14x14 -> 7x7
    nn.Flatten(),                  # srotola in un vettore di 64*7*7 = 3136
    nn.Linear(64 * 7 * 7, 10),     # 10 classi (logit)
)
```

Nota il ritmo ricorrente: le mappe si restringono (28 → 14 → 7), mentre il
numero di filtri cresce (32 → 64). La rete scambia risoluzione spaziale con
ricchezza semantica, finché le poche feature rimaste bastano allo strato denso
per decidere. Un dettaglio tutto di PyTorch: lo strato `nn.Linear` finale
vuole sapere esattamente quanti numeri riceve (qui
$64 \cdot 7 \cdot 7 = 3136$) e il conto delle dimensioni resta a chi progetta
la rete, non alla libreria.

```{admonition} Da ricordare
:class: important
- Gli strati densi falliscono sulle immagini per **troppi parametri** e
  **nessuna invarianza alla traslazione**.
- La **convoluzione** fa scorrere un piccolo kernel che evidenzia un motivo
  ovunque compaia; l'uscita è una **feature map**.
- **Campo recettivo locale** + **pesi condivisi** = pochi parametri, motivi
  riconosciuti indipendentemente dalla posizione.
- Il **max pooling** riduce la risoluzione e aggiunge tolleranza a piccoli
  spostamenti; l'architettura tipica alterna **conv–pool** e chiude con strati
  densi.
```
