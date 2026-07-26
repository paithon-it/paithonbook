# Le architetture che hanno fatto la storia

Ogni autunno, tra il 2010 e il 2017, i laboratori di visione artificiale di
mezzo mondo si sfidavano su **ImageNet**: oltre un milione di fotografie da
classificare in mille categorie, dal cane pastore alla tazza da caffè. C'era
una classifica, e quella classifica racconta una storia. Nel 2011 l'errore del
sistema migliore era intorno al 26%; nel 2015 era sceso sotto il 4%, meglio di
un essere umano messo alla prova sullo stesso compito. In quattro anni un
problema considerato durissimo è stato quasi chiuso.

Dietro quel crollo non ci sono soltanto "più dati e computer più potenti". C'è
una manciata di **architetture** (modi diversi di impilare gli strati di una
rete) ognuna delle quali ha spostato la frontiera. Ripercorriamole in ordine,
perché conoscerle significa capire come si progetta una rete profonda.

## LeNet-5: dove tutto comincia

Molto prima di ImageNet, alla fine degli anni '90, **Yann LeCun** e colleghi
ai Bell Labs progettavano reti per leggere i codici di avviamento postale e le
cifre scritte a mano sugli assegni bancari. Il loro modello, **LeNet-5**
{cite}`lecun1998gradient`, è la prima rete convoluzionale davvero
funzionante su un compito reale.

`````{tab} Elementare
Immagina una piccola lente che scorre sull'immagine di una cifra, un pezzetto
alla volta, cercando tratti semplici: un bordo, una curva, un angolo. Poi una
seconda lente combina questi tratti in forme più grandi, finché la rete decide
quale numero da 0 a 9 sta guardando. LeNet-5 fa esattamente questo, e ha
imparato a leggere le cifre scritte a mano meglio di qualsiasi programma
scritto a regole.
`````

`````{tab} Superiore
LeNet-5 alterna strati **convoluzionali** (che condividono i pesi su tutta
l'immagine) e strati di **sottocampionamento** (l'antenato del *pooling*),
seguiti da strati *fully-connected* per la classificazione finale, con
attivazioni $\tanh$. Ha circa $60\,000$ parametri (minuscola per gli standard
odierni) ed è addestrata con la *backpropagation* sul dataset di cifre
manoscritte **MNIST**. Introduce già i tre principi delle CNN: connettività
locale, condivisione dei pesi e invarianza approssimata alla traslazione.
`````

## AlexNet: la notte in cui il deep learning vinse

Per oltre un decennio le CNN restarono una curiosità. La svolta arriva nel
2012, quando **AlexNet** {cite}`krizhevsky2012imagenet` vince la sfida
ImageNet con un margine imbarazzante: un errore *top-5* (la risposta giusta
non compare nemmeno tra le prime cinque proposte) del 15,3%, contro il 26% del
secondo classificato, che usava ancora tecniche "artigianali". Fu il momento
in cui il resto del campo capì che il deep learning funzionava.

`````{tab} Elementare
AlexNet è, in fondo, una LeNet cresciuta: molti più strati, molte più
"lenti", e per la prima volta addestrata su schede grafiche (GPU) invece che su
normali processori. Con più muscoli e qualche trucco nuovo per non "imparare a
memoria", ha imparato a distinguere migliaia di oggetti diversi in fotografie
vere, sfocate e disordinate come quelle che scattiamo tutti i giorni.
`````

`````{tab} Superiore
Otto strati con pesi (cinque convoluzionali, tre *fully-connected*), circa
$60$ milioni di parametri, addestrata su due GPU. Le tre scelte decisive:
attivazioni **ReLU** al posto di $\tanh$ (gradienti che non saturano,
addestramento molto più rapido), **dropout** negli strati densi per contenere
l'overfitting, e **data augmentation** aggressiva (ritagli, riflessioni,
perturbazioni di colore). Non concetti nuovi in assoluto, ma messi insieme alla
scala giusta sul dataset giusto.
`````

## Network in Network: una piccola rete dentro il filtro

Nel 2013, mentre il mondo digeriva la lezione di AlexNet, tre ricercatori
della National University of Singapore (Min Lin, Qiang Chen e Shuicheng Yan)
pubblicano un articolo dal titolo quasi ricorsivo: **Network in Network**
(NiN) {cite}`lin2013network`. Non vince nessuna classifica, ma contiene due
idee destinate a diventare equipaggiamento standard di quasi tutte le reti
venute dopo.

`````{tab} Elementare
In ogni punto dell'immagine la rete accumula, strato dopo strato, tante
"opinioni" diverse: una segnala un bordo, un'altra una macchia di colore,
un'altra ancora una trama. La prima idea di NiN è una lente piccolissima che
guarda un solo punto alla volta, ma legge *tutte* le opinioni raccolte su quel
punto e le fonde in un giudizio più maturo: una piccola riunione di esperti
convocata pixel per pixel. La seconda idea riguarda il finale. Invece di
collegare tutto a un enorme "ufficio" di neuroni che decide la classe (dove le
reti dell'epoca concentravano quasi tutti i loro collegamenti) NiN prepara una
mappa per ogni categoria e guarda quanto ciascuna mappa si accende in media:
la più accesa vince. Milioni di collegamenti sostituiti da una media.
`````

`````{tab} Superiore
Una **convoluzione $1\times 1$** con $C_{\text{in}}$ canali in ingresso e
$C_{\text{out}}$ in uscita applica a ogni pixel la stessa trasformazione
lineare del suo vettore di canali, seguita dalla non-linearità: non tocca la
struttura spaziale ma ricombina i canali, con appena
$C_{\text{in}} \cdot C_{\text{out}}$ pesi. Farne seguire una o due a una
convoluzione ordinaria equivale a far scorrere sull'immagine un piccolo
percettrone multistrato al posto di un filtro lineare: da qui il nome *network
in network*. All'altro capo della rete, il **global average pooling** elimina
gli strati *fully-connected* finali: l'ultimo strato convoluzionale produce
una mappa di attivazione per classe, ogni mappa viene ridotta alla propria
media spaziale e il vettore risultante va dritto alla softmax. Zero parametri
dove AlexNet ne concentrava oltre il 90% (nei tre strati densi), e meno
overfitting.
`````

Tenete a mente entrambe le idee: le ritroveremo tra poco, dentro la rete che
vincerà ImageNet l'anno successivo.

## VGG: la profondità con mattoncini piccoli

Nel 2014 il gruppo di Oxford (Simonyan e Zisserman
{cite}`simonyan2015very`) pone una domanda semplice:
e se usassimo sempre e solo filtri piccolissimi, $3\times 3$, ma ne
impilassimo tanti? Nasce **VGG**, elegante nella sua monotonia.

`````{tab} Elementare
Invece di una lente grande che guarda molto in una volta sola, VGG usa tante
lenti piccole, una dopo l'altra. Due lenti piccole in fila "vedono" quanto una
lente più grande, ma la rete diventa più profonda e impara pattern più ricchi
usando meno pezzi. È il principio del mattoncino Lego: pochi pezzi uguali,
combinati in tanti strati.
`````

`````{tab} Superiore
Due convoluzioni $3\times 3$ in serie hanno lo stesso *campo recettivo* di una
$5\times 5$, tre in serie di una $7\times 7$, ma con più non-linearità
intermedie e **meno parametri**. Su $C$ canali, un filtro $5\times 5$ costa
$25C^2$ pesi, due filtri $3\times 3$ solo $2\cdot 9C^2 = 18C^2$. VGG-16 e
VGG-19 spingono la profondità a 16–19 strati e diventano il punto di
riferimento per il *transfer learning* degli anni successivi.
`````

## GoogLeNet e i moduli Inception

Lo stesso anno Google vince la classificazione di ImageNet con **GoogLeNet**
{cite}`szegedy2015going`, costruita a partire da un blocco ingegnoso (il
**modulo Inception**) che mette a frutto proprio le convoluzioni $1\times 1$
di NiN.

`````{tab} Elementare
Quanto è grande la cosa che stiamo cercando? Un dettaglio minuscolo o un
oggetto che riempie l'inquadratura? Inception non sceglie: guarda lo stesso
punto contemporaneamente con lenti di misure diverse e poi mette insieme tutto
ciò che ha visto. Come avere occhiali per vicino e per lontano nello stesso
istante.
`````

`````{tab} Superiore
Ogni modulo esegue in **parallelo** convoluzioni $1\times 1$, $3\times 3$,
$5\times 5$ e un *pooling*, poi concatena le uscite lungo i canali,
elaborando così l'immagine a **più scale** simultaneamente. Le convoluzioni
$1\times 1$ fungono da collo di bottiglia che riduce i canali prima delle
convoluzioni costose. Risultato: 22 strati ma solo $\approx 5$ milioni di
parametri, un dodicesimo di AlexNet, con accuratezza persino superiore.
`````

Anche la seconda idea di NiN è all'appello: GoogLeNet rinuncia ai grandi
strati *fully-connected* di AlexNet e VGG e chiude con un *global average
pooling* seguito da un unico strato lineare. È in buona parte per questo che
i suoi parametri sono così pochi.

## ResNet: insegnare alle reti a non dimenticare l'input

Restava un muro. Impilando strati oltre una certa soglia, le reti non solo
smettevano di migliorare: peggioravano, e non per overfitting; sbagliavano di
più anche sui dati di addestramento. Questo **problema di degradazione** viene
risolto da **ResNet** {cite}`he2016deep` di He, Zhang, Ren e Sun, che porta la
profondità a 152 strati e l'errore su ImageNet al 3,57%.

```{figure} ../figures/residuo-skip-connection.svg
:name: fig-skip-connection
:alt: L'input x di un blocco attraversa due strati peso; una connessione laterale porta lo stesso x, non modificato, fino a un nodo somma che lo aggiunge all'uscita degli strati prima dell'attivazione finale.
:width: 55%

La connessione residua: l'input $\mathbf{x}$ salta gli strati interni e viene
sommato alla loro uscita $\mathcal{F}(\mathbf{x})$ prima dell'attivazione.
```

`````{tab} Elementare
L'idea è quasi banale e per questo geniale ({numref}`fig-skip-connection`).
Invece di chiedere a un blocco di strati di ricostruire da capo tutto il
segnale, gli si affianca una "scorciatoia" che porta l'input intatto fino
all'uscita, dove viene ri-sommato. Così il blocco deve imparare solo la
**correzione** da apportare, non l'intera risposta. E se non serve correggere
nulla, può lasciar passare l'input senza rovinarlo: aggiungere strati non fa
più danni.
`````

`````{tab} Superiore
Se $\mathcal{F}(\mathbf{x})$ è la trasformazione dei due strati interni, il
blocco residuo calcola

$$
\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x},
$$

dove $\mathbf{x}$ è l'input del blocco, $\{W_i\}$ i suoi pesi e $\mathbf{y}$
l'uscita (a cui si applica poi la non-linearità). Il blocco apprende il
**residuo** $\mathcal{F} = \mathcal{H} - \mathbf{x}$ rispetto alla mappa
desiderata $\mathcal{H}$: azzerare $\mathcal{F}$ per ottenere l'identità è
facile, ricostruire l'identità da zero no. In più, il termine additivo
$\mathbf{x}$ apre una via diretta al gradiente durante la
*backpropagation*, mitigando la sua scomparsa e rendendo addestrabili reti di
centinaia di strati.
`````

## DenseNet: se una scorciatoia funziona, prendetele tutte

La connessione residua apre una strada, e nel 2017 un gruppo tra Cornell,
Tsinghua e Facebook AI Research (Gao Huang, Zhuang Liu, Laurens van der Maaten
e Kilian Weinberger) la percorre fino in fondo con **DenseNet**
{cite}`huang2017densely`, premiata come miglior articolo, a pari merito con un
altro lavoro, alla conferenza CVPR. Se ResNet *somma* l'input all'uscita
($\mathbf{x} + \mathcal{F}(\mathbf{x})$), DenseNet *concatena*: dentro un
blocco denso ogni strato riceve le feature di **tutti** gli strati precedenti,
affiancate una all'altra.

```{figure} ../figures/blocco-denso.svg
:name: fig-blocco-denso
:alt: "Blocco denso con tre strati: archi di concatenazione portano l'input e l'uscita di ogni strato a tutti i nodi di concatenazione successivi, dove le feature vengono affiancate lungo i canali prima di entrare nello strato seguente."
:width: 100%

Il blocco denso: ogni strato riceve, affiancate lungo i canali, le feature di
tutti gli strati precedenti (non una sola scorciatoia come nel blocco residuo,
ma tutte).
```

`````{tab} Elementare
Se una scorciatoia che porta l'input intatto fino all'uscita funziona così
bene, perché fermarsi a una? In un blocco denso ogni strato riceve non solo il
risultato dello strato precedente, ma *tutto quello che è stato prodotto
prima*, messo semplicemente uno accanto all'altro
({numref}`fig-blocco-denso`). È come una chat di gruppo in cui ogni nuovo
messaggio ha sotto gli occhi l'intera conversazione: nessuna informazione va
riassunta o ricostruita, basta consultarla. E proprio perché può contare su
tutto il lavoro già fatto, ogni strato aggiunge poco di suo (poche mappe nuove
per volta) e la rete resta sorprendentemente snella. Il rovescio della
medaglia è lo stesso delle chat: la cronologia cresce, e tenerla tutta aperta
occupa parecchia memoria.
`````

`````{tab} Superiore
Lo strato $\ell$-esimo di un blocco denso calcola

$$
\mathbf{x}_\ell = H_\ell\!\left([\mathbf{x}_0, \mathbf{x}_1, \dots,
\mathbf{x}_{\ell-1}]\right),
$$

dove $[\cdot]$ indica la concatenazione lungo i canali, $\mathbf{x}_0$ è
l'input del blocco e $H_\ell$ una sequenza batch normalization → ReLU →
convoluzione $3\times 3$. Ogni strato produce solo $k$ mappe nuove (il
**growth rate**, tipicamente $k=12$ o $k=32$), così lo strato $\ell$ riceve
$k_0 + k(\ell-1)$ canali, dove $k_0$ sono quelli dell'input. Il **riuso delle
feature** rende la rete efficiente nei parametri (a parità di accuratezza su
ImageNet, all'incirca la metà di una ResNet comparabile) e la concatenazione
apre a ogni strato un percorso diretto verso il gradiente della loss. Poiché
concatenare richiede mappe della stessa dimensione spaziale, i blocchi densi
sono separati da *strati di transizione* (convoluzione $1\times 1$ e pooling),
che dimezzano la risoluzione. Il conto da pagare è la **memoria** in
addestramento: in un'implementazione ingenua le attivazioni concatenate
crescono col quadrato della profondità del blocco, ed esistono varianti più
parsimoniose, ma una ResNet di pari accuratezza resta in genere più leggera da
addestrare.
`````

## Progettare architetture: dall'artigianato al metodo

Vista da vicino, la stagione 2012–2016 è stata artigianato d'alta scuola:
intuizioni individuali, provate e riprovate su ImageNet a colpi di settimane
di GPU. Poi il mestiere si è fatto metodo, in due direzioni. La prima:
smettere di chiedersi soltanto *quale* rete disegnare, e chiedersi *come
farla crescere*. È la domanda di **EfficientNet** {cite}`tan2019efficientnet`,
che nel 2019 le dà una risposta precisa, il *compound scaling*.

`````{tab} Elementare
Per fare una torta doppia non si raddoppia solo la farina: si aumentano tutti
gli ingredienti in proporzione, o il risultato è immangiabile. Una rete può
crescere in tre modi: più strati (profondità), strati più larghi (larghezza),
immagini d'ingresso più grandi (risoluzione). Invece di puntare tutto su uno
solo, EfficientNet li fa crescere insieme, in proporzioni fisse trovate una
volta per tutte: a ogni raddoppio del budget di calcolo, la profondità aumenta
di circa il 20%, la larghezza del 10% e la risoluzione del 15%. Ripetendo la
ricetta si ottiene una famiglia di reti, dalla più piccola (adatta a un
telefono) alla più grande.
`````

`````{tab} Superiore
Il *compound scaling* fissa i fattori di crescita come

$$
d = \alpha^{\phi}, \qquad w = \beta^{\phi}, \qquad r = \gamma^{\phi},
\qquad \text{con } \alpha \cdot \beta^{2} \cdot \gamma^{2} \approx 2,
$$

dove $d$, $w$ e $r$ sono i moltiplicatori di profondità, larghezza e
risoluzione rispetto alla rete di partenza, $\phi$ è il *coefficiente
composto* che fissa il budget di calcolo, e le costanti
$\alpha=1{,}2$, $\beta=1{,}1$, $\gamma=1{,}15$ sono trovate con una piccola
grid search sulla rete base. Il vincolo fa sì che ogni incremento unitario di
$\phi$ raddoppi all'incirca i FLOPs (che crescono come $\alpha \cdot \beta^2
\cdot \gamma^2$ elevato a $\phi$). Dalla rete base EfficientNet-B0, il
compound scaling genera la famiglia B1–B7, che a parità di accuratezza su
ImageNet usa fino a quasi un ordine di grandezza di parametri in meno
rispetto alle CNN precedenti.
`````

La seconda direzione è automatizzare la ricerca stessa: nella **neural
architecture search** un algoritmo esplora lo spazio delle possibili
architetture e seleziona le più promettenti; la rete base di EfficientNet è
stata trovata proprio così, non disegnata a mano. E la storia non è finita:
dal 2020 i **Vision Transformer** {cite}`dosovitskiy2021image`, reti basate
sull'attenzione e nate per il linguaggio, hanno dimostrato di poter competere
con le CNN quando i dati abbondano. Oggi in visione artificiale le due
famiglie convivono e si scambiano idee; ne riparleremo nel capitolo dedicato
ai Transformer.

## L'architettura conta quanto i dati

Nessuna di queste reti ha vinto solo con più esempi o più GPU. LeNet ha dato
la convoluzione, AlexNet ha dimostrato che scala, NiN ha ripensato il filtro e
il finale della rete, VGG ha reso la profondità sistematica, Inception ha
aggiunto la multi-scala, ResNet ha reso la profondità finalmente addestrabile,
DenseNet ha portato il riuso delle feature alle estreme conseguenze. Ogni
salto è nato da un'**idea strutturale** su come far scorrere l'informazione
dentro la rete. I dati sono il carburante, ma la forma del motore decide
quanto lontano si arriva, ed è una lezione che vale ancora oggi, dai
Transformer in poi.

```{admonition} Da ricordare
:class: important
- **LeNet-5** (1998): la convoluzione che legge le cifre (connettività locale,
  pesi condivisi, pooling).
- **AlexNet** (2012): ReLU, dropout e GPU portano le CNN a vincere ImageNet.
- **NiN** (2013): convoluzioni $1\times 1$ come mini-rete pixel per pixel e
  *global average pooling* al posto degli strati densi.
- **VGG** (2014): profondità con soli filtri $3\times 3$ impilati.
- **Inception/GoogLeNet** (2014): elaborare a più scale in parallelo, con
  colli di bottiglia $1\times 1$.
- **ResNet** (2015): la connessione residua $\mathbf{y}=\mathcal{F}(\mathbf{x})+\mathbf{x}$
  rende addestrabili reti di centinaia di strati.
- **DenseNet** (2017): ogni strato riceve, concatenate, le feature di
  **tutti** i precedenti (pochi parametri, molta memoria).
- Dopo l'artigianato, il metodo: il **compound scaling** di EfficientNet fa
  crescere insieme profondità, larghezza e risoluzione; la *neural
  architecture search* automatizza il progetto.
```
