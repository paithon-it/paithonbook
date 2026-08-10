# Le architetture che hanno fatto la storia

Ogni autunno, tra il 2010 e il 2017, i laboratori di visione artificiale di
mezzo mondo si sfidavano su **ImageNet**: oltre un milione di fotografie da
classificare in mille categorie, dal cane pastore alla tazza da caffè. C'era
una classifica, e quella classifica racconta una storia. Nel 2011 l'errore del
sistema migliore era intorno al 26%; nel 2015 era sceso sotto il 4%, meglio
del 5,1% sbagliato da una persona che si era allenata a fare esattamente lo
stesso lavoro, cioè a mettere l'etichetta giusta sulle stesse
fotografie[^annotatore]. In quattro anni un problema considerato durissimo è
stato quasi chiuso.

Dietro quel crollo non ci sono soltanto "più dati e computer più potenti". C'è
una manciata di **architetture** (modi diversi di impilare gli strati di una
rete) ognuna delle quali ha spostato la frontiera. Ripercorriamole in ordine,
perché conoscerle significa capire come si progetta una rete profonda.

## LeNet-5: dove tutto comincia

Molto prima di ImageNet, a partire dalla fine degli anni '80, **Yann LeCun**
e colleghi ai Bell Labs progettavano reti per leggere i codici di avviamento
postale e le cifre scritte a mano sugli assegni bancari. Una loro rete più
semplice, costruita nel 1989, leggeva già i codici postali scritti a mano
sulle buste della posta americana[^zip1989], ed è il primo caso in cui una
rete di questo tipo ha davvero funzionato su un lavoro vero e non su un
esercizio da laboratorio. **LeNet-5** {cite}`lecun1998gradient` è quella arrivata dopo, la
più matura della serie, ed è la versione che si studia ancora oggi.

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

```{figure} ../figures/alexnet-2012.svg
:name: fig-alexnet
:alt: "In alto, lo stack di AlexNet: otto strati addestrabili, prima i convolutivi poi i densi, distribuiti su due GPU che lavorano in parallelo, con la nota di due schede GTX 580 per circa sei giorni di addestramento. In basso, il confronto dell'errore top-5 su ImageNet: 26,2% per i metodi costruiti a mano, 15,3% per AlexNet."
:width: 96%

Profondità più GPU. Il salto dell'errore in basso è la parte che fece
notizia; la riga sopra, due schede da gioco per sei giorni, è quella che rese
l'esperimento ripetibile da chiunque.
```

Il dettaglio di {numref}`fig-alexnet` che vale più della classifica è la
divisione fra due GPU. Non era una scelta di eleganza ma di necessità (la
memoria di una scheda sola non bastava) e per farla gli autori dovettero
spezzare la rete a metà: il primo caso in cui l'hardware disponibile ha
disegnato l'architettura, e non l'ultimo.

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
profondità a 152 strati e vince l'edizione 2015 di ImageNet. Una singola
ResNet-152 scende intorno al 4,5% di errore top-5; il record che chiude la
competizione, il 3,57%, non lo fa una rete sola ma un gruppo di sei reti
residue interrogate tutte insieme, che poi mettono ai voti le loro risposte
(si chiama *ensemble*, ed è un trucco che funziona quasi sempre: reti
addestrate in modo leggermente diverso sbagliano su immagini diverse, e la
media degli errori è più bassa di ciascuno).

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

## Separare lo spazio dai canali: la convoluzione che sta in un telefono

Fin qui la corsa è stata verso l'alto: più strati, più connessioni, più
accuratezza, e pazienza per il costo. Attorno al 2016 una parte della ricerca
gira la domanda: **a parità di accuratezza, quanto poco si può spendere?** Non
è una curiosità da risparmiatori, è la condizione perché la visione artificiale
esca dai centri di calcolo ed entri in un telefono, in una telecamera, in
un'automobile. La risposta più fruttuosa nasce da un'osservazione sulla
convoluzione stessa.

`````{tab} Elementare
Una convoluzione ordinaria fa due lavori in una volta sola, e non ce ne
accorgiamo perché li fa insieme. Il primo è **guardarsi intorno**: prendere un
quadratino di $3\times3$ pixel e cercarci una forma. Il secondo è **mettere
d'accordo i canali**: combinare quello che dicono tutte le opinioni raccolte in
quel punto (il bordo, il colore, la trama) in una nuova opinione.

L'idea è di smettere di farli insieme. Prima si guarda intorno, ma **un canale
per volta**: ogni opinione viene esaminata nel suo quadratino, per conto suo,
senza mescolarsi con le altre. Poi, separatamente, si mettono d'accordo i
canali con una lente che guarda un solo punto: è la convoluzione $1\times1$ di
*Network in Network*, che abbiamo incontrato poco fa e che qui trova il suo
impiego più importante.

Il risultato ha la stessa forma di prima, ma costa **quasi nove volte meno**. E
il motivo per cui costa meno è semplice: nella versione ordinaria ogni
combinazione «quale pixel del quadratino» per «quale canale di partenza» per
«quale canale di arrivo» ha il suo peso, e quei tre elenchi si moltiplicano fra
loro. Separando, due dei tre si **sommano** invece di moltiplicarsi.

È il tipo di risparmio che non si ottiene tagliando qualcosa, ma accorgendosi
che si stava pagando due volte. Su questo mattone sono costruite quasi tutte le
reti che girano sui telefoni.
`````

`````{tab} Superiore
Una convoluzione standard $k \times k$ da $C_{\text{in}}$ a $C_{\text{out}}$
canali ha $k^2 C_{\text{in}} C_{\text{out}}$ pesi e costa, per pixel d'uscita,
altrettante moltiplicazioni-accumulo. La **convoluzione separabile in
profondità** (*depthwise separable*) la fattorizza in due passi:

1. **depthwise**: una convoluzione $k \times k$ applicata **a ciascun canale
   indipendentemente** (in PyTorch, `groups=C_in`), con $k^2 C_{\text{in}}$
   pesi. Filtra nello spazio senza mescolare i canali;
2. **pointwise**: una convoluzione $1 \times 1$ da $C_{\text{in}}$ a
   $C_{\text{out}}$, con $C_{\text{in}} C_{\text{out}}$ pesi. Mescola i canali
   senza guardare i vicini.

Il rapporto fra i due costi è

$$
\frac{k^2 C_{\text{in}} C_{\text{out}}}
{k^2 C_{\text{in}} + C_{\text{in}} C_{\text{out}}}
= \frac{k^2 C_{\text{out}}}{k^2 + C_{\text{out}}}
\;\xrightarrow[\;C_{\text{out}} \to \infty\;]{}\; k^2 ,
$$

cioè il risparmio tende a $k^2$ ($9$ per i filtri $3\times3$ e si avvicina già
molto a quel limite con qualche decina di canali in uscita). Da notare che è la
**fattorizzazione** a produrre il guadagno, non un taglio: il tensore d'uscita
ha esattamente la stessa forma, e ciò che si perde è l'espressività delle
combinazioni spazio-canale congiunte, che l'esperienza mostra essere in gran
parte ridondanti.

L'idea circolava da tempo, ma è **MobileNet** {cite}`howard2017mobilenets` a
farne l'ossatura di una famiglia di reti pensate per il calcolo su dispositivo,
con due manopole esplicite (un moltiplicatore di larghezza e uno di
risoluzione) per scendere lungo la curva costo-accuratezza. **Xception**
{cite}`chollet2017xception` porta la stessa idea al limite dentro un'architettura
in stile Inception, leggendola come l'ipotesi estrema che correlazioni spaziali
e correlazioni fra canali si possano trattare del tutto separatamente.

**MobileNetV2** {cite}`sandler2018mobilenetv2` aggiunge il pezzo che manca e
che è arrivato fino a oggi: il **residuo invertito** con **collo di bottiglia
lineare**. Il blocco *espande* i canali con una $1\times1$, applica la
depthwise nello spazio espanso, poi *ricomprime* con un'altra $1\times1$
**senza non-linearità finale** (perché una ReLU su uno spazio a poche
dimensioni distrugge informazione che non si recupera), e la connessione
residua collega i due estremi stretti anziché quelli larghi, che è l'opposto di
ResNet e serve a tenere basso il consumo di memoria. Quel blocco si chiama
**MBConv**, ed è il mattone di cui è fatta la rete base di EfficientNet, che
incontriamo qui sotto.

Una variazione sul tema merita una riga, perché mostra che la stessa economia
si può ottenere altrimenti: il *Fire module* di **SqueezeNet**
{cite}`iandola2016squeezenet` alterna uno strato di *squeeze* a $1\times1$ che
strozza i canali e uno di *expand* che li riapre con un misto di $1\times1$ e
$3\times3$, arrivando all'accuratezza di AlexNet con cinquanta volte meno
parametri.
`````

Il conto si verifica in poche righe, ed è utile farlo perché il numero che ne
esce è meno ovvio della formula.

```python
import torch
import torch.nn as nn

C_IN, C_OUT, K, H, W = 64, 128, 3, 56, 56
x = torch.randn(1, C_IN, H, W)

# convoluzione ordinaria: ogni filtro guarda tutti i canali in una volta sola
ordinaria = nn.Conv2d(C_IN, C_OUT, K, padding=1, bias=False)

# separabile: prima la parte spaziale, un filtro per canale (groups=C_IN),
# poi la parte fra i canali, una 1x1 che li rimescola
separabile = nn.Sequential(
    nn.Conv2d(C_IN, C_IN, K, padding=1, groups=C_IN, bias=False),   # depthwise
    nn.Conv2d(C_IN, C_OUT, 1, bias=False),                          # pointwise
)

def parametri(m):
    return sum(p.numel() for p in m.parameters())

print("stessa forma in uscita:", ordinaria(x).shape == separabile(x).shape,
      tuple(separabile(x).shape))
print(f"parametri, ordinaria : {parametri(ordinaria):>8,}")
print(f"parametri, separabile: {parametri(separabile):>8,}")
print(f"risparmio            : {parametri(ordinaria) / parametri(separabile):.2f}x")

teorico = (K * K * C_OUT) / (K * K + C_OUT)
print(f"previsto dalla formula: {teorico:.2f}x   (limite: {K * K}x)")
```

Da $73\,728$ pesi a $8\,768$, cioè **$8{,}41$ volte meno**, con il tensore
d'uscita di forma identica. E il numero misurato coincide con quello previsto
dalla formula fino all'ultima cifra, perché qui non c'è niente di empirico: è
aritmetica.

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

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- **LeNet-5** (1998): la piccola lente che scorre sull'immagine un pezzetto
  alla volta, sempre la stessa in ogni punto, e impara a leggere le cifre
  scritte a mano.
- **AlexNet** (2012): la stessa idea cresciuta (molti più strati, molte più
  lenti), addestrata su schede grafiche e con qualche trucco per non imparare
  a memoria; nel 2012 vince ImageNet e convince tutti.
- **NiN** (2013): una lente che guarda un solo punto ma legge tutte le
  opinioni raccolte lì e le fonde; e un finale che, invece di un enorme
  ufficio di neuroni, tiene una mappa per categoria e premia la più accesa.
- **VGG** (2014): il principio del mattoncino Lego, tante lenti piccole e
  uguali impilate una dopo l'altra al posto di poche lenti grandi.
- **Inception/GoogLeNet** (2014): guardare lo stesso punto con lenti di misure
  diverse nello stesso istante, tenendo basso il conto grazie alla lente che
  guarda un punto solo.
- **ResNet** (2015): la scorciatoia che porta l'input intatto fino all'uscita,
  dove viene ri-sommato; al blocco resta da imparare solo la correzione, e
  così si addestrano reti di centinaia di strati.
- **DenseNet** (2017): non una scorciatoia ma tutte, come una chat di gruppo
  in cui ogni strato ha sotto gli occhi l'intera conversazione (pochi pesi,
  molta memoria).
- La **convoluzione separabile** (MobileNet, 2017) smette di fare due lavori
  insieme: prima guarda intorno un canale per volta, poi mette d'accordo i
  canali con la lente che guarda un punto solo. Stessa forma in uscita, quasi
  nove volte meno pesi: è il mattone delle reti che stanno in un telefono.
- Dopo l'artigianato, il metodo: EfficientNet fa crescere insieme profondità,
  larghezza e dimensione delle immagini come gli ingredienti di una torta; e
  la ricerca automatica delle architetture disegna la rete al posto nostro.
```
`````

`````{tab} Superiore
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
- La **convoluzione separabile in profondità** (MobileNet, Xception) fattorizza
  la convoluzione in *depthwise* ($k^2 C_{\text{in}}$ pesi) più *pointwise*
  $1\times1$ ($C_{\text{in}}C_{\text{out}}$): il costo scende di un fattore
  $k^2 C_{\text{out}} / (k^2 + C_{\text{out}}) \to k^2$. MobileNetV2 vi
  aggiunge il **residuo invertito** con collo di bottiglia lineare
  (**MBConv**), che è il blocco base di EfficientNet.
- Dopo l'artigianato, il metodo: il **compound scaling** di EfficientNet fa
  crescere insieme profondità, larghezza e risoluzione; la *neural
  architecture search* automatizza il progetto.
```
`````

[^zip1989]: Il sistema riconosceva le cifre dei codici postali ritagliate
    dalle buste dal servizio postale statunitense
    {cite}`lecun1989backpropagation`.

[^annotatore]: Il 5,1% è la prova di una persona sola, Andrej Karpathy, che
    nel 2014 si addestrò al compito e si misurò contro le reti. Un
    esperimento serio ma con un solo partecipante: va letto come ordine di
    grandezza, non come misura della "prestazione umana" in generale.
