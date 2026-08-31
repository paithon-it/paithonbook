# Reti convoluzionali (CNN)

Per un computer una fotografia è una tabella di numeri: uno per ogni pixel, o
tre se la foto è a colori. Prendi la foto di un gatto e spostalo di dieci pixel
a destra: per te è ancora, banalmente, un gatto; per la tabella, invece, quasi
nessun numero è rimasto dov'era, e una rete fatta come quelle del capitolo sulle
reti neurali si ritrova davanti un ingresso completamente nuovo.

Questo scarto tra come *noi* vediamo un'immagine e come la vede una rete neurale
ordinaria è il problema che le **reti convoluzionali** (Convolutional Neural
Networks, CNN) sono nate per risolvere. L'idea affonda le radici negli stessi
esperimenti di Hubel e Wiesel sulla corteccia del gatto: prende forma nel
*Neocognitron* di Fukushima {cite}`fukushima1980neocognitron`, una rete a strati
che imita proprio quella catena di rivelatori, e arriva a maturazione nella
LeNet-5 di Yann LeCun (1998), che leggeva le cifre scritte a mano sugli assegni
bancari.

## Perché uno strato denso non basta

Prima di costruire qualcosa di nuovo conviene capire perché il pezzo che
abbiamo già, lo strato in cui ogni neurone riceve tutti i numeri che escono dallo strato di
sotto, quello con cui in {doc}`Reti neurali </RetiNeurali/overview>` erano
fatte tutte le reti e che qui chiameremo strato **denso**, sulle immagini non
funziona. Le ragioni sono due, e
nessuna delle due è un dettaglio: il numero di pesi da imparare, che diventa
ingestibile, e il fatto che una rete fatta così tratti la stessa forma come due
cose diverse a seconda di *dove* si trova nell'immagine.

`````{tab} Elementare

Uno strato denso collega *ogni* pixel a *ogni* neurone. Sembra generoso, ma
è uno spreco. Una foto a colori di 256×256 pixel non sono 65.536 numeri ma il
triplo: ogni pixel ne porta tre, uno per il rosso, uno per il verde e uno
per il blu, e $256 \times 256 \times 3$ fa 196.608, quasi 200.000. Uno strato
denso con 1000 neuroni ha un peso per ogni coppia (numero in ingresso,
neurone): $196.608 \times 1000$, cioè quasi **200 milioni** di pesi da
imparare, solo per il primo strato. Troppi. Con tanti numeri da regolare, la
rete ha modo di imparare a memoria le foto che le mostri invece di quello che
hanno in comune: sulle foto già viste va benissimo, sulla prima foto nuova
sbanda.

E c'è un problema più profondo. Se la rete impara a riconoscere un occhio
quando compare in alto a sinistra, non sa nulla dello stesso occhio in basso
a destra: per lei sono due cose diverse, perché occupano posizioni (e quindi
pesi) diversi. Manca l'idea che *un motivo è lo stesso ovunque appaia*.

`````

`````{tab} Superiore

Uno strato *fully-connected* su un input $\mathbf{x}\in\mathbb{R}^{D}$ con
$h$ unità richiede $h\cdot D$ pesi. Per un'immagine RGB $256\times256$ si ha
$D = 256\cdot256\cdot3 \approx 1{,}97\times10^{5}$: con $h=1000$ servono circa
$2\times10^{8}$ parametri, un invito all’*overfitting*.

Soprattutto, lo strato denso non è **equivariante alla traslazione**: un
pattern spostato di un vettore $\boldsymbol{\Delta}$ attiva pesi diversi,
perché l'indice
della componente cambia. Le CNN recuperano l'equivarianza grazie a due vincoli
architetturali (connettività locale e condivisione dei pesi) che riducono
anche i parametri: sposti l'input, e l'attivazione si sposta con lui.
Attenzione a non chiamarla invarianza, che è un'altra proprietà (la risposta
non cambia affatto) e la convoluzione non la dà: semmai la porta la testa
della rete, con il *global average pooling* che incontreremo parlando di NiN.

`````

## La convoluzione: un filtro che scorre

Il cuore della rete è la **convoluzione**: un piccolo filtro (o *kernel*),
tipicamente $3\times3$, che scivola su tutta l'immagine. In ogni posizione
sovrappone il filtro alla porzione di immagine sottostante, moltiplica valore
per valore e somma il tutto in *un* numero. Quel numero misura quanto bene
quella porzione somiglia al **motivo** che il filtro cerca, cioè al disegno
ricorrente che lo interessa: un bordo verticale, una macchia di colore, una
trama.

I nove numeri di cui un filtro $3\times3$ è fatto non li scrive nessuno a mano:
sono pesi come tutti gli altri della rete, e sono esattamente ciò che
l'addestramento aggiusta. Un filtro, in altre parole, è una domanda che la rete
impara a formulare da sé.

```{figure} ../figures/convoluzione.svg
:name: fig-convoluzione
:alt: Un filtro 3x3 evidenziato su una porzione di una griglia di ingresso 5x5; una freccia collega la regione alla singola casella corrispondente della mappa dei risultati.
:width: 85%

Il filtro (o *kernel*) di $3\times3$ copre nove caselle dell'immagine per
volta: i nove valori vengono moltiplicati ciascuno per il proprio peso e poi
sommati, e il totale diventa **una** casella del foglio dei risultati, che si
chiama *feature map*. Facendo scorrere la finestra si riempie l'intera mappa.
```

Come mostra {numref}`fig-convoluzione`, il filtro guarda solo una finestra per
volta, ma quella *stessa* finestra visita ogni angolo dell'immagine.

`````{tab} Elementare

Uno stampino traforato con nove caselle scorre su una pagina a quadretti. In
ogni punto guardi i nove quadretti che si affacciano dai buchi, moltiplichi
ciascuno per il numero scritto sul suo buco e sommi: viene un totale solo, e
quello scrivi su un foglio nuovo. I nove numeri dello stampino sono la ricetta,
e sono quelli che la rete impara: all'inizio sono presi a caso e non trovano
niente, e a forza di esempi diventano un cercatore di bordi o di macchie.

Il punto è che lo stampino non cambia mai mentre scorre: la ricetta è la stessa
in tutti i punti della pagina. Se è brava a trovare un bordo, lo trova ovunque
nell'immagine, in alto come in basso. E ogni totale finisce sul foglio nuovo
nel punto corrispondente: sposta il bordo di due quadretti e si sposta di due
quadretti anche il segno che lo segnala. È questo il trucco che mancava allo
strato denso.

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

La stessa operazione in movimento, con un filtro che cerca **bordi verticali**.
Sotto i tre riquadri, la regola scritta in simboli, che dice questo: moltiplica
i nove valori sotto la finestra per i nove pesi del filtro, e somma tutto. La
barra è spessa
un pixel, quindi in ogni posizione finisce sotto una sola colonna del filtro: la
mappa risponde $-3$ quando cade sotto la colonna destra, $+3$ quando cade sotto
la sinistra e $0$ quando cade sotto quella centrale, i cui pesi valgono zero.
```

Conviene rifare i conti della {numref}`fig-convoluzione-animata`. Il filtro è
fatto di tre righe uguali, ciascuna con i pesi $1$, $0$, $-1$; la barra vale
$1$ e lo sfondo $0$. Quando la barra finisce sotto la colonna destra del
filtro, ogni riga contribuisce $-1$ e le tre righe insieme danno $-3$; quando
finisce sotto la colonna sinistra, $+3$; quando è al centro, il peso che la
moltiplica è $0$ e le altre due colonne vedono solo sfondo. Il filtro non
misura quanto la barra è chiara: misura il **contrasto** tra il lato destro e
il lato sinistro della propria finestra, e il segno dice da che parte sta il
chiaro. È già un abbozzo di ciò che i primi strati di una CNN imparano da
soli.

## Campi recettivi locali e pesi condivisi

Due principi rendono tutto ciò possibile, e conviene dar loro un nome perché
tornano dappertutto. Il primo è il **campo recettivo locale**: ogni casella
della mappa dei risultati (che è poi un neurone come quelli del capitolo sulle
reti neurali, solo con pochissimi ingressi) guarda una finestra piccola, non
l'immagine intera. Il secondo è la **condivisione dei pesi**: lo stesso filtro
si usa in ogni posizione, quindi i pochi numeri che lo compongono vengono
riutilizzati migliaia di volte.

`````{tab} Elementare

Un filtro $3\times3$ su un'immagine a colori guarda nove caselle, e di ogni
casella i tre numeri del colore: $3\times3\times3 = 27$ pesi.

Più un ultimo numero, sempre lo stesso, che si somma al risultato in ogni
posizione. Alza o abbassa in blocco l'intera mappa, e serve a regolare quanto
forte debba essere la somiglianza prima che il filtro dica «l'ho trovato»: se
vale $-2$, una somiglianza da 1 non basta più a produrre un risultato positivo.
Si chiama *bias*, e fa $27+1 = 28$ numeri da imparare per filtro.

Con 32 filtri diversi sono $28 \times 32 = 896$ pesi, meno di mille, contro i
milioni dello strato denso. Con così pochi numeri da regolare resta molto meno
spazio per imparare le foto a memoria. E restano 896 anche su una foto con
quattro volte i pixel: lo stampino non si allarga, fa solo più giri, mentre lo
strato denso avrebbe preteso quattro volte i pesi. Pochi pesi, riusati ovunque:
la rete impara *cosa* cercare, non *dove*.

`````

`````{tab} Superiore

Un layer convoluzionale con $F$ filtri, kernel $k\times k$ e $C$ canali in
ingresso ha $F\,(k^2 C + 1)$ parametri, e nel conto la risoluzione
dell'immagine non compare: su una foto con quattro volte i pixel il lavoro da
fare quadruplica, i pesi da imparare restano quelli. È la condivisione dei pesi
(*weight tying*) che impone l'equivarianza traslazionale come *prior*
strutturale, riducendo drasticamente
lo spazio delle ipotesi e quindi il rischio di overfitting.

`````

L'uscita di un filtro è una **feature map**: una mappa che segna, punto per
punto, *dove* nell'immagine è presente il motivo cercato. Uno strato convoluzionale, in inglese *layer*, produce una pila di feature
map, una per filtro; i primi strati imparano motivi
elementari (bordi, angoli), i più profondi li combinano in parti sempre più
astratte (occhi, ruote, volti).

## Il pooling: mappe più piccole, e cosa si guadagna

Dopo la convoluzione si applica quasi sempre il **pooling**, che rimpicciolisce
le feature map tenendo di ogni zona solo il valore più forte. Il più comune è
il **max pooling**: su ogni finestra (di solito $2\times2$) conserva il massimo.
Su un quadratino che contiene $1$, $7$, $3$ e $2$, esce $7$, e gli altri tre
numeri si perdono.

`````{tab} Elementare

Il max pooling è come chiedere, per ogni quadratino $2\times2$: "il motivo qui
intorno c'è, sì o no?", tenendo solo la risposta più forte. Non c'è nessun peso
da imparare: la regola è una sola, «tieni il più grande». Dimezza larghezza e
altezza, quindi tutto quello che viene dopo lavora su un quarto dei numeri.
E ogni casella rimasta ne riassume quattro,
così il filtro dello strato successivo, pur restando di nove caselle, arriva a
coprire una fetta di immagine larga il doppio. Sono questi i guadagni sicuri.

Ce n'è un altro, ma più piccolo di come lo si racconta di solito. Se il motivo
si sposta di un pixel e resta dentro lo stesso quadratino, il massimo di quel
quadratino non cambia, e dopo il pooling la mappa è identica: lo spostamento è
stato assorbito. Se invece scavalca il confine fra due quadratini, cambia
eccome. Su una mappa piena di valori diversi uno spostamento di un solo pixel
altera quasi sempre il risultato.

È un baratto, non un regalo: si guadagnano leggerezza e un po’ di tolleranza,
si perde precisione su dove le cose stanno.

`````

`````{tab} Superiore

Per una regione $\mathcal{R}_{i,j}$ della feature map,

$$
y_{i,j} = \max_{(m,n)\,\in\,\mathcal{R}_{i,j}} x_{m,n}.
$$

Non ha parametri da apprendere. Sottocampionando, aumenta il campo recettivo
effettivo dei layer successivi; e in cambio dell'equivarianza esatta, che con
finestre prese a passo 2 (lo stride, cioè di quanti pixel avanza la finestra a
ogni scatto) sopravvive solo per gli spostamenti pari, offre una modesta
tolleranza alle traslazioni di un pixel. Modesta è la
parola giusta: su un picco isolato spostato di un pixel la mappa risultante
resta identica circa una volta su due, su feature map dense di valori diversi
praticamente mai. È un baratto e non un'aggiunta gratuita.

`````

## L'architettura tipica

Lo schema classico alterna blocchi **conv → ReLU → pool**, ripetuti alcune
volte, e chiude con uno o più strati densi che trasformano le feature astratte
in una decisione, cioè nella classe dell'immagine: gatto, cane, tazza da caffè.

La ReLU sta in mezzo, fra il filtro e il pooling, e non è un ornamento: è lei a
rendere davvero diversi due strati impilati. Una convoluzione è fatta di
moltiplicazioni e somme, e applicarne una al risultato di un'altra, senza niente
in mezzo, darebbe ancora moltiplicazioni e somme: una convoluzione sola, un po’
più larga. È il piegare i numeri (buttare via i negativi) a far sì che il
secondo strato veda qualcosa che il primo non poteva produrre da solo.

Due manopole governano poi lo scorrimento del filtro: lo **stride**, di quanti
pixel salta la finestra a ogni passo, e il **padding**, la cornice di zeri
aggiunta ai bordi per non perdere i pixel di frontiera.

`````{tab} Elementare

Quanto viene grande la mappa che esce? Se la finestra avanza di un
pixel per volta (stride 1) e all'immagine si aggiunge attorno una cornice
spessa uno (padding 1), un filtro $3\times3$ restituisce una mappa grande
esattamente quanto l'immagine di partenza: da $28\times28$ pixel escono
$28\times28$ risultati. Senza quella cornice ne uscirebbero $26\times26$, e il
conto si fa a mente: la finestra è larga 3, può cominciare dal primo pixel e
deve finire entro il ventottesimo, quindi le posizioni buone sono
$28 - 3 + 1 = 26$, una in meno per lato. Se invece si tiene la cornice e si
porta il passo a 2, la finestra salta una posizione ogni volta e la mappa esce
dimezzata, $14\times14$.

Col passo a 2 il conto non sempre torna in pieno, e allora tocca scegliere. Su
una striscia larga 61 quadretti, una finestra da 2 che avanza di 2 ne copre 60
e ne lascia fuori uno: o si butta via l'ultima colonna, e i risultati sono 30,
oppure la si tiene con la finestra mezza fuori dal foglio, e sono 31. Due
persone che scelgono in modo diverso si ritrovano con mappe di dimensione
diversa, convinte di aver costruito la stessa rete.

C'è anche il modo di allargare la finestra senza aggiungere caselle: le nove
caselle che legge si distanziano, una ogni due quadretti invece che attaccate.
Legge sempre nove numeri, ma la fetta che copre è larga cinque quadretti invece
di tre, e con la solita cornice da uno la mappa esce $26\times26$ invece di
$28\times28$: guarda più lontano e mangia più bordo.

Il ritmo con cui le mappe si rimpiccioliscono man mano che si sale nella rete è
una scelta di chi la progetta: più le mappe si stringono, meno conti ci sono da
fare, ma meno dettaglio resta.

`````

`````{tab} Superiore

La dimensione di uscita lungo un asse è

$$
o = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1,
$$

dove $n$ è la dimensione d'ingresso, $k$ quella del kernel, $p$ il padding, $s$
lo stride e $\lfloor \cdot \rfloor$ la parte intera inferiore. Un esempio con i
numeri di una rete che lavora su immagini $28\times28$: ingresso $n=28$, kernel
$k=3$, padding $p=1$, stride $s=1$, e
$o = \lfloor(28 + 2 - 3)/1\rfloor + 1 = 28$, la risoluzione non cambia. Con
`padding="same"` si sceglie appunto $p$ affinché $o=n$ (a stride 1): l'uscita
conserva la risoluzione dell'ingresso.

Due precisazioni, perché la formula così com'è nasconde altrettante ipotesi.

La prima: a stride 1 la parte intera non serve, perché dividere per uno dà
sempre un intero. Conta da stride 2 in su, dove le posizioni in cui la
finestra sporgerebbe dal bordo semplicemente non si contano. E lì va saputo che
arrotondare per difetto è una convenzione, non l'unica possibile:
`nn.MaxPool2d(..., ceil_mode=True)` arrotonda per eccesso, tenendo anche
l'ultima finestra incompleta. Con $n=61$ e $k=s=2$ la prima dà $30$ e la
seconda $31$, e due reti che si credono uguali si ritrovano con mappe di
dimensione diversa.

La seconda: la formula assume **dilatazione 1**, cioè un filtro i cui pesi
guardano pixel adiacenti. Con dilatazione $d$ i pesi si distanziano fra loro e
il filtro copre $d(k-1)+1$ pixel invece di $k$, quindi

$$
o = \left\lfloor \frac{n + 2p - d(k-1) - 1}{s} \right\rfloor + 1 .
$$

Con $n=28$, $k=3$, $p=1$, $s=1$ e $d=2$ l'uscita è $26$, non $28$: un $3\times3$
dilatato di due è largo cinque pixel e mangia i bordi come una $5\times5$.

`````

In PyTorch l'intera architettura sta in poche righe:

```python
from torch import nn

# input: un batch di immagini in scala di grigi, shape (N, 1, 28, 28)
model = nn.Sequential(
    # blocco 1: 32 filtri 3x3, mappe grandi come l'input
    nn.Conv2d(1, 32, 3, padding="same"), nn.ReLU(),
    nn.MaxPool2d(2),               # 28x28 -> 14x14
    # blocco 2: più filtri man mano che le mappe rimpiccioliscono
    nn.Conv2d(32, 64, 3, padding="same"), nn.ReLU(),
    nn.MaxPool2d(2),               # 14x14 -> 7x7
    nn.Flatten(),                  # srotola in un vettore di 64*7*7 = 3136
    nn.Linear(64 * 7 * 7, 10),     # 10 classi (logit)
)
```

Nota il ritmo ricorrente: le mappe si restringono (28 → 14 → 7), mentre il
numero di filtri cresce (32 → 64). È uno scambio: la rete rinuncia a sapere
*dove* le cose stanno con precisione, e in cambio si porta dietro più tipi
diversi di cose trovate, finché le poche rimaste bastano allo strato denso per
decidere.

Un dettaglio tutto di PyTorch: `nn.Flatten()` srotola la pila di mappe in
un'unica fila di numeri, e lo strato `nn.Linear` finale vuole sapere
esattamente quanti ne riceve (qui $64 \cdot 7 \cdot 7 = 3136$). Quel conto resta
a chi progetta la rete, non alla libreria.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Uno strato denso sull'immagine intera ha **troppi pesi** da imparare (quasi
  200 milioni per una sola foto a colori e mille neuroni), e per di più non ha
  nessuna idea che *un motivo sia lo stesso ovunque appaia*.
- La **convoluzione** fa scorrere sull'immagine un filtro piccolo e scrive su
  un foglio nuovo, punto per punto, quanto quel motivo c'è: quel foglio si
  chiama *feature map*.
- Pochi pesi, riusati in ogni punto: la rete impara **cosa** cercare, non
  **dove**. Se il motivo si sposta, si sposta con lui anche il segnale che lo
  indica.
- Il **max pooling** tiene, di ogni quadratino, solo il valore più forte:
  rimpicciolisce le mappe e dà un po’ di tolleranza agli spostamenti minimi, in
  cambio della precisione su dove le cose stanno. L'architettura tipica alterna
  convoluzione e pooling, e chiude con gli strati densi che decidono la classe.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Gli strati densi falliscono sulle immagini per **troppi parametri** e perché
  non sono **equivarianti alla traslazione**.
- La **convoluzione** fa scorrere un piccolo kernel che evidenzia un motivo
  ovunque compaia; l'uscita è una **feature map**.
- **Campo recettivo locale** + **pesi condivisi** = pochi parametri e risposta
  **equivariante**: il motivo è riconosciuto ovunque compaia, e l'attivazione
  si sposta insieme a lui. L'invarianza è un'altra proprietà, e arriva semmai
  dalla testa della rete (pooling globale), non dalla convoluzione.
- Il **max pooling** riduce la risoluzione e baratta l'equivarianza esatta con
  una tolleranza ai piccoli spostamenti; l'architettura tipica alterna conv e
  pool e chiude con strati densi.
```
`````
