---
titolo_ricerca: "Architetture storiche: da LeNet a ResNet"
descrizione: "Le architetture che hanno segnato il deep learning, da LeNet-5 e
  AlexNet a VGG, Inception, ResNet e DenseNet, e che cosa ciascuna ha cambiato."
---
# Le architetture che hanno fatto la storia

Ogni anno, tra il 2010 e il 2017, i laboratori di visione artificiale di mezzo
mondo si sfidavano su ImageNet, la gara che AlexNet vinse nel 2012: oltre un
milione di fotografie da classificare in mille categorie, dal cane pastore alla
tazza da caffè. Le classifiche di quegli anni, lette una dopo l'altra,
raccontano la storia delle architetture.

Ogni programma poteva proporre cinque etichette per fotografia, e la risposta
era giusta se fra quelle cinque c'era quella vera. La regola ha una ragione: in
una fotografia di solito c'è più di una cosa, e fra mille categorie ce ne sono
di vicinissime (il pastore tedesco e il pastore belga sono due voci
distinte), quindi pretendere un'unica risposta avrebbe misurato anche la
fortuna. Il punteggio si chiama **errore top-5**.

Nel 2011 l'errore del sistema migliore era intorno al 26%, cioè su cento
fotografie in ventisei nemmeno una delle cinque proposte era quella giusta. Nel
2015 lo stesso errore era sceso sotto il 4%, meglio del 5,1% sbagliato da una
persona che si era allenata a fare lo stesso lavoro sulle stesse
fotografie[^annotatore]. In quattro anni un problema considerato
durissimo è stato quasi chiuso.

Dietro quel crollo non ci sono soltanto più dati e computer più potenti, ma
anche una manciata di **architetture** (modi diversi di impilare gli strati di
una rete), ciascuna delle quali ha spostato la frontiera. Seguirne la
successione mostra come si progetta una rete profonda.

## LeNet-5: dove tutto comincia

Molto prima di ImageNet, a partire dalla fine degli anni '80, Yann LeCun
e colleghi ai Bell Labs progettavano reti per leggere i codici di avviamento
postale e le cifre scritte a mano sugli assegni bancari. Una loro rete più
semplice, costruita nel 1989, leggeva già i codici postali scritti a mano
sulle buste della posta americana[^zip1989], uno dei primi casi in cui una
rete di questo tipo ha funzionato su un lavoro vero e non su un esercizio da
laboratorio.

**LeNet-5** {cite}`lecun1998gradient` è la rete arrivata dopo, la quinta della
serie (LeNet-1 risaliva al 1989), la più matura e la versione che si studia
ancora oggi. Le cifre su cui fu messa alla prova sono quelle di MNIST, le
stesse che la rete del {doc}`capitolo su PyTorch </PyTorch/moduli>` imparava
a riconoscere; lo stesso articolo che presenta LeNet-5 presenta anche quella
raccolta.

`````{tab} Elementare
Una piccola lente scorre sull'immagine di una cifra, un pezzetto alla volta,
cercando tratti semplici: un bordo, una curva, un angolo. È lo stampino che
scorreva sul foglio a quadretti, e [la rete ne usa la stessa copia in ogni
punto, quindi c'è da impararne una sola invece di una diversa per ogni
posizione]{.pt-matita}.

Dopo ogni passata la rete riassume. Divide il risultato in quadretti e di
ciascuno tiene un numero solo. In questo modo resta scritto che il tratto
c'era, e si perde dove stava di preciso. Una cifra scritta un po’ più in alto,
o un po’ più storta, lascia allora la stessa traccia. Poi una seconda lente
combina questi tratti in forme più grandi, finché la rete decide quale numero
da 0 a 9 sta guardando.

Così lavora LeNet-5, che ha imparato a leggere le cifre scritte a mano meglio
dei programmi scritti a regole con cui fu confrontata.
`````

`````{tab} Superiore
LeNet-5 alterna strati convoluzionali (che condividono i pesi su tutta
l'immagine) e strati di sottocampionamento (l'antenato del *pooling*), seguiti
da strati *fully-connected* per la classificazione finale, con attivazioni
$\tanh$. Ha $60\,000$ parametri (minuscola per gli standard odierni) ed è
addestrata con la *backpropagation* sul dataset di cifre manoscritte MNIST.
[Introduce già i tre principi delle CNN]{.pt-matita}: connettività locale,
condivisione dei pesi (da cui l'equivarianza alla traslazione) e
sottocampionamento, che aggiunge una tolleranza approssimata a spostamenti e
piccole deformazioni.

Vista da vicino, però, non è la CNN dei manuali di oggi. Il sottocampionamento
di S2 e S4, ad esempio, somma i quattro valori di una finestra $2\times2$, li
moltiplica per un coefficiente addestrabile, aggiunge un bias e passa il
risultato per una sigmoide, quindi non è né il massimo né la media semplice.
Lo strato C5 è una convoluzione $5\times5$ su mappe $5\times5$, cioè uno
strato denso scritto come convoluzione.

L'attivazione è $f(a) = A\tanh(Sa)$ con $A = 1{,}7159$, e in uscita, al posto
della softmax, ci sono dieci unità a base radiale, una per cifra, ciascuna con
un prototipo scelto a mano e tenuto fisso; vince la cifra la cui unità misura
la distanza minima. I parametri addestrabili sono, strato per strato, $156$,
$12$, $1\,516$, $32$, $48\,120$ e $10\,164$, che sommati fanno $60\,000$.

Infine, lo strato convoluzionale C3 non collega ogni mappa in uscita a tutte
quelle in ingresso, ma segue una tabella di connessione sparsa, scritta a mano.
LeCun ne dà due ragioni, tenere il numero di connessioni entro limiti
ragionevoli e rompere la simmetria, così che mappe diverse imparino cose
diverse.
`````

## AlexNet: quando il deep learning vinse ImageNet

Per oltre un decennio le reti convoluzionali restarono una tecnica di nicchia.
Leggevano cifre e caratteri, anche in produzione (il sistema degli assegni ne
leggeva milioni al giorno), ma la visione artificiale in generale usava altri
metodi. La svolta arriva nel 2012, quando AlexNet
{cite}`krizhevsky2012imagenet` vince la sfida ImageNet con un distacco di quasi
undici punti: errore top-5 del 15,3%, contro il 26,2% del secondo classificato,
che usava ancora tecniche costruite a mano.

Quel 15,3%, però, è il punteggio di ciò che il gruppo ha consegnato alla gara,
e non è una rete sola. Sono sette reti fatte lavorare insieme: per
ogni fotografia ciascuna assegna un punteggio a ciascuna delle mille categorie,
i sette punteggi si mediano categoria per categoria, e le cinque categorie col
totale più alto sono la risposta. Due delle sette avevano in più un
allenamento preliminare su un archivio di immagini più di dodici volte maggiore
(quindici milioni di immagini contro 1,2 milioni).

La singola rete descritta nell'articolo si ferma al 18,2% sull'insieme di
validazione. [Il salto fu comunque tale da convincere il resto del campo che il
deep learning funzionava]{.pt-matita}. La stessa distinzione fra la rete e la
squadra tornerà con ResNet.

```{figure} ../figures/alexnet-2012.svg
:name: fig-alexnet
:alt: "A sinistra, gli otto strati addestrabili di AlexNet divisi in due colonne, una per ciascuna delle due GPU: cinque convolutivi in teal e tre densi in ocra. Fra il primo e il secondo strato, e fra il terzo, il quarto e il quinto, le due colonne restano separate; dopo il secondo strato, poi all'ingresso del primo strato denso e su ogni passaggio successivo, le connessioni incrociano da una GPU all'altra, disegnate come linee a X in terracotta. A destra, il confronto dell'errore top-5 delle sottomissioni a ImageNet: 26,2% per i metodi costruiti a mano, 15,3% per AlexNet."
:width: 96%

Profondità più GPU. Il salto dell'errore, a destra, fece notizia; a sinistra
la rete in due metà, una per scheda, che comunicano soltanto agli incroci in
terracotta.
```

Le due schede grafiche da videogioco su cui girò l'addestramento sono la storia
con cui si apre la {doc}`sezione sulle prestazioni di PyTorch
</PyTorch/prestazioni>`. Per l'architettura, però, conta il vincolo che
imposero. La memoria di una scheda sola (3 GB) non bastava a contenere la rete,
quindi gli autori la divisero in due metà, una per scheda, che si scambiano
informazione soltanto agli incroci segnati in terracotta nella
{numref}`fig-alexnet`: dopo il secondo strato convolutivo, all'ingresso del
primo strato denso e da lì a ogni passaggio fino all'uscita.

Il costo del calcolo aveva già disegnato una rete. In LeNet-5, quattordici anni
prima, LeCun non collegava ogni mappa di uno strato a tutte quelle dello strato
precedente, anche per tenere basso il numero di collegamenti. In AlexNet il
vincolo cambia la forma dell'intera rete, e non soltanto i collegamenti di uno
strato.

`````{tab} Elementare
AlexNet è, in fondo, una LeNet cresciuta: molti più strati, molte più
"lenti", e l'addestramento su schede grafiche (le GPU, le schede nate per
far girare i videogiochi) invece che su normali processori.

Cambia anche il modo in cui ogni neurone decide quanto accendersi. La regola
vecchia schiacciava i valori grandi verso un tetto, e una volta appiccicati al
tetto si somigliavano tutti, quindi la rete non capiva più in che verso
correggersi, e imparava pianissimo. La regola nuova è sbrigativa: sotto lo
zero spegne, sopra lo zero lascia passare il numero com'è. Niente tetto,
nessuna zona piatta, addestramento molto più rapido. È la ReLU, il primo dei
tre accorgimenti della {doc}`panoramica del capitolo </DeepLearning/overview>`.

Gli altri due servono contro il difetto peggiore, quello di "imparare a
memoria" le fotografie dell'addestramento. Uno spegne a caso una parte dei
neuroni a ogni passata (il *dropout*), così nessuno può contare sempre sugli
stessi compagni per dare la sua risposta. L'altro non mostra mai la stessa
fotografia due volte uguale: la ritaglia in un punto diverso, la specchia, le
sposta un po’ i colori (la *data augmentation*).

Con più muscoli e quegli accorgimenti, AlexNet ha imparato a distinguere mille
categorie di oggetti in fotografie vere, sfocate e disordinate come quelle che
scattiamo tutti i giorni.
`````

`````{tab} Superiore
AlexNet ha otto strati con pesi (cinque convoluzionali, tre *fully-connected*)
e circa $60$ milioni di parametri, ed è addestrata su due GPU. Il numero
esatto dipende da quale versione si conta, perché ne circolano tre.
$60\,965\,224$ è la rete dell'articolo, con tre convoluzioni che guardano solo
la metà dei canali perché l'altra metà sta sull'altra scheda. $62\,378\,344$ è
la stessa architettura senza quella divisione, cioè con ogni filtro che vede
tutti i canali.

$61\,100\,840$, infine, è ciò che dichiara
`torchvision.models.alexnet()`, che non implementa nessuna delle due ma la
variante più snella che Krizhevsky ridisegnò nel 2014 in un secondo articolo,
quello sulla parallelizzazione (i primi due strati passano da 96 e 256 filtri a
64 e 192, il quarto da 384 a 256). È la versione che quasi tutti eseguono
credendo di eseguire quella dell'articolo del 2012.

Le scelte decisive sono tre: le attivazioni ReLU al posto di $\tanh$ (perché i
gradienti non saturano e l'addestramento è molto più rapido), il dropout negli
strati densi per contenere l'overfitting, e una data augmentation aggressiva
(ritagli, riflessioni, perturbazioni di colore). Non sono concetti nuovi in
assoluto, ma qui sono messi insieme alla scala giusta sul dataset giusto.
`````

## Network in Network: una piccola rete dentro il filtro

Nel 2013, mentre il mondo digeriva la lezione di AlexNet, tre ricercatori
della National University of Singapore (Min Lin, Qiang Chen e Shuicheng Yan)
pubblicano un articolo dal titolo che si morde la coda: **Network in Network**
(NiN) {cite}`lin2013network`. La rete non compare nella classifica, perché
non partecipa a ImageNet; contiene però due idee che diventeranno
equipaggiamento standard di quasi tutte le reti successive.

`````{tab} Elementare
Ogni lente che scorre sull'immagine lascia dietro di sé un foglio di numeri,
uno per punto: quanto, in quel punto, somigliava al motivo cercato. Sono le
feature map di {doc}`Reti convoluzionali <reti-convoluzionali>`, e la rete ne
accumula tante, strato dopo strato. Se guardi un singolo punto dell'immagine,
quella pila di mappe è una collezione di opinioni su quel punto: una dice che
lì c'è un bordo, un'altra una macchia di colore, un'altra ancora una trama.

La prima idea di NiN è una lente piccolissima che guarda un solo punto alla
volta, ma legge *tutte* le opinioni raccolte lì e le fonde in un giudizio più
maturo: una piccola riunione di esperti convocata pixel per pixel.

La seconda idea riguarda il finale. Invece di collegare tutto agli enormi
strati densi finali che decidono la categoria (dove le reti dell'epoca
concentravano quasi tutti i loro collegamenti), NiN prepara una mappa per ogni
categoria e ne fa la media, cioè guarda quanto quella mappa ha risposto forte
nell'insieme dell'immagine. Vince la categoria con la media più alta, e in
questo modo [una media prende il posto di milioni di collegamenti]{.pt-matita}.
`````

`````{tab} Superiore
Una **convoluzione $1\times 1$** con $C_{\text{in}}$ canali in ingresso e
$C_{\text{out}}$ in uscita applica a ogni pixel la stessa trasformazione
lineare del suo vettore di canali, seguita dalla non-linearità. In altri
termini, non tocca la struttura spaziale ma ricombina i canali, con appena
$C_{\text{in}} \cdot C_{\text{out}}$ pesi. Farne seguire una o due a una
convoluzione ordinaria equivale a far scorrere sull'immagine un piccolo
percettrone multistrato al posto di un filtro lineare: da qui il nome *network
in network*.

All'altro capo della rete, il **global average pooling** elimina gli strati
*fully-connected* finali. L'ultimo strato convoluzionale produce una mappa di
attivazione per classe, ogni mappa viene ridotta alla propria media spaziale e
il vettore risultante passa direttamente alla softmax. [Il pooling globale non
ha parametri, mentre negli strati densi che sostituisce AlexNet ne concentra il
94%]{.pt-matita}; con meno parametri la rete sovradatta meno.

Il calcolo si fa a mano. Le tre matrici dense sono $9216\times4096$,
$4096\times4096$ e $4096\times1000$, cioè $58\,621\,952$ pesi, che con i
rispettivi bias fanno $58\,631\,144$ parametri, il 94% dei $62\,378\,344$
della versione non divisa (e il 96% dei $60\,965\,224$ di quella
dell'articolo). Tutto il resto, cioè i cinque strati convoluzionali che fanno
il lavoro, sta in quel poco che resta.
`````

Le due idee tornano entrambe in GoogLeNet, la rete che vince ImageNet l'anno
successivo.

## VGG: la profondità con filtri piccoli

Nel 2014 il Visual Geometry Group di Oxford (Simonyan e Zisserman
{cite}`simonyan2015very`) pone una domanda semplice: e se usassimo sempre e
solo filtri piccolissimi, $3\times 3$, ma ne impilassimo tanti? Nasce così
**VGG**, una rete elegante nella sua monotonia, che prende il nome dalle
iniziali del gruppo. Alla gara di quell'anno arriva seconda, con il 7,3% di
errore top-5, anche lei con una squadra di sette reti.

`````{tab} Elementare
Invece di una lente grande che guarda molto in una volta sola, come quella da
undici pixel per undici con cui cominciava AlexNet, VGG usa tante lenti
piccole, una dopo l'altra.

[Due lenti piccole in successione "vedono" quanto una lente più
grande]{.pt-matita}, come mostra un conto a mente. La prima lente guarda tre
pixel per volta. La seconda guarda tre risultati della prima, e ciascuno di
quei tre riassumeva tre pixel: il primo i pixel da 1 a 3, il secondo da 2 a 4,
il terzo da 3 a 5. Messi insieme, la seconda lente sta guardando i pixel da 1 a
5, cioè quanto una lente da cinque. Allo stesso modo, tre lenti in successione
arrivano a sette.

Il guadagno è doppio. Fra una lente e l'altra la rete piega i numeri, quindi ci
sono più passaggi in cui può succedere qualcosa di interessante; e i numeri da
imparare sono meno, perché due quadratini da tre per tre hanno diciotto caselle
e uno da cinque per cinque ne ha venticinque. È il principio del mattoncino
Lego: pochi pezzi uguali, combinati in tanti strati.

Il risparmio, però, riguarda le lenti, e il peso di VGG sta altrove. In fondo
alla rete restano gli enormi strati densi che decidono la categoria,
quelli che NiN aveva appena mostrato come togliere. Lì dentro stanno quasi nove
decimi dei centotrentotto milioni di numeri che VGG deve imparare, cioè più del
doppio di quanti ne aveva AlexNet. I mattoncini piccoli danno profondità con
pochi parametri, mentre gli strati densi finali ne usano moltissimi.
`````

`````{tab} Superiore
[Due convoluzioni $3\times 3$ in serie hanno lo stesso *campo recettivo* di una
$5\times 5$]{.pt-matita}, tre in serie di una $7\times 7$, ma con più non-linearità
intermedie e meno parametri. Su $C$ canali, ad esempio, un filtro $5\times 5$
costa $25C^2$ pesi, due filtri $3\times 3$ solo $2\cdot 9C^2 = 18C^2$. VGG-16 e
VGG-19 spingono la profondità a 16–19 strati e diventano il punto di
riferimento per il *transfer learning* degli anni successivi.

Il campo recettivo di una pila di strati si calcola con una ricorrenza:

$$
r_\ell = r_{\ell-1} + (k_\ell - 1)\prod_{i<\ell} s_i ,
$$

dove $r_\ell$ è il lato della regione di immagine che un neurone dello strato
$\ell$ riesce a vedere, $k_\ell$ è il lato del suo kernel e $s_i$ gli stride
degli strati che lo precedono. In particolare, con soli $3\times3$ a stride
$1$ il prodotto vale sempre $1$ e la successione è $r_0=1$, poi $3$, $5$, $7$;
due strati arrivano quindi a $5$, come una $5\times5$, e tre a $7$. È il
calcolo che giustifica l'intera scelta di VGG.

Il peso di VGG-16, però, sta dove non ci si aspetta. La rete ha
$138\,357\,544$ parametri, 2,3 volte AlexNet, e l’$89{,}4\%$ è ancora nei tre
strati densi finali, il difetto che *Network in Network* aveva appena mostrato
come evitare e che VGG non raccoglie.

Gli stessi 138 milioni sono 2,3 volte anche i parametri di una ResNet-152
($60\,192\,808$), che ha centocinquantadue strati contro sedici. Il numero di
strati, quindi, non basta a prevedere quello dei parametri. Incide invece sul
numero di operazioni: He e colleghi contano le
moltiplicazioni-accumulo per immagine, $11{,}3$ miliardi per ResNet-152, il
triplo dei $3{,}8$ di ResNet-50 ma meno dei $15{,}3$ di VGG-16
{cite}`he2016deep`.
`````

## GoogLeNet e i moduli Inception

Lo stesso anno Google vince la classificazione di ImageNet con **GoogLeNet**
{cite}`szegedy2015going`, al 6,67% (anche qui una squadra di sette reti). La
rete è costruita a partire da un blocco ingegnoso, detto **modulo Inception**,
che mette a frutto le convoluzioni $1\times 1$ di NiN. Il nome, con la L
maiuscola, è un omaggio dichiarato a LeNet-5.

`````{tab} Elementare
Quanto è grande il motivo che stiamo cercando? Un dettaglio minuscolo o un
oggetto che riempie l'inquadratura? Inception non sceglie: guarda lo stesso
punto contemporaneamente con lenti di misure diverse, e poi tiene tutti i
risultati, uno accanto all'altro. Come avere occhiali per vicino e per lontano
nello stesso istante.

Usare tutte le lenti insieme, però, darebbe troppi numeri da imparare. Una
lente, infatti, deve leggere tutte le opinioni raccolte in quel punto, e in
mezzo alla rete le opinioni sono centinaia. I numeri da imparare sono le
caselle del quadratino moltiplicate per le opinioni in entrata e ancora per
quelle in uscita, e la lente grande ha molte più caselle.

Davanti a ogni lente grande, allora, il modulo mette la lente di NiN, quella
che guarda un solo punto: [prima riassume le opinioni, poi la lente grande
legge il riassunto]{.pt-matita}. Nel primo modulo di GoogLeNet, ad esempio, la
lente da cinque per cinque avrebbe $192$ opinioni da leggere e ne produce $32$,
cioè venticinque caselle per $192$ per $32$, che fanno $153\,600$ numeri.

Il riassunto gliene passa $16$ invece di $192$. La lente grande ha allora
venticinque per $16$ per $32$ numeri, cioè $12\,800$, e il riassunto stesso,
che legge $192$ opinioni e ne scrive $16$, ne ha $192$ per $16$, cioè
$3\,072$. In tutto, per quel ramo, sono $15\,872$, un decimo di prima.
`````

`````{tab} Superiore
Ogni modulo esegue in parallelo convoluzioni $1\times 1$, $3\times 3$, $5\times
5$ e un *pooling*, poi concatena le uscite lungo i canali, elaborando così
l'immagine a più scale simultaneamente. In particolare, [le convoluzioni
$1\times 1$ fungono da collo di bottiglia che riduce i canali prima delle
convoluzioni costose]{.pt-matita}. Il risultato è una rete di 22 strati con
pesi e, come dichiara l'articolo, dodici volte meno parametri di AlexNet (cioè
circa $5$ milioni), con un'accuratezza persino superiore.

Quella cifra è un rapporto, non un conteggio. L'articolo non scrive un totale,
ma la sua Tabella 1 ha una colonna dei parametri strato per strato, che
sommata dà $6{,}80$ milioni. Ricostruendo la rete dai canali della stessa
tabella se ne contano $6{,}99$, perché la colonna arrotonda e in qualche riga
non torna (per il primo modulo, ad esempio, dichiara 159 mila parametri,
mentre i canali ne danno 163 mila).

A sua volta, `torchvision.models.googlenet(aux_logits=False)` ne dichiara
$6\,624\,904$, e la differenza sta nel ramo $5\times5$, che `torchvision`
realizza con un kernel $3\times3$. Attivando i due **classificatori
ausiliari** (i rami intermedi che durante l'addestramento iniettano un segnale
di supervisione a metà rete) si arriva a $13\,004\,888$, quasi il doppio. Il
rapporto con i $60\,965\,224$ parametri di AlexNet è quindi fra 8,7 e 9,2
volte, non dodici. Resta un risparmio di quasi un ordine di grandezza, ma il
numero preciso dipende da che cosa si conta.
`````

GoogLeNet riprende anche la seconda idea di NiN, cioè rinuncia agli strati
densi finali (*fully-connected*) con cui AlexNet e VGG chiudevano. Al loro
posto calcola la media di ogni mappa (il *global average pooling* di NiN) e la
passa a un solo strato lineare che sceglie la classe. Per questo ha così
pochi parametri, circa nove volte meno di AlexNet, e sbaglia meno.

## ResNet: insegnare alle reti a non dimenticare l'input

Restava un muro: oltre una certa profondità le reti non solo smettevano di
migliorare, ma peggioravano. Il sospetto ovvio, che stessero imparando a
memoria le fotografie di addestramento (l’*overfitting*, che si riconosce
perché la rete migliora su quelle e peggiora su tutte le altre), non spiega il
fenomeno. Le reti, infatti, sbagliavano di più anche sulle fotografie di
addestramento, cioè su quelle che avevano sotto gli occhi. Questo **problema di
degradazione** viene risolto da **ResNet** {cite}`he2016deep` di He, Zhang, Ren
e Sun, che porta la profondità a 152 strati e vince l'edizione 2015 di
ImageNet.

Anche qui il numero della classifica descrive una squadra. Una singola
ResNet-152 fa il 4,49% di errore top-5 sulle fotografie di validazione; il
3,57% con cui si vince, sulle fotografie della prova, viene da sei reti di
profondità diversa, due delle quali ResNet-152, combinate insieme. Una squadra
di questo tipo si chiama *ensemble*, e di solito funziona perché reti diverse
sbagliano su immagini diverse, così la loro combinazione sbaglia meno di
ciascuna; ma solo finché i loro errori sono poco legati fra loro.

La competizione va avanti fino al 2017 e l'errore scende ancora, fino al 2,25%
dell'ultima edizione. Intanto la domanda cambia, da quanto in alto si può
arrivare a quanto poco basta, e a rispondere è soprattutto la convoluzione
separabile, nata qualche anno prima e usata da MobileNet per reti intere.

```{figure} ../figures/residuo-skip-connection.svg
:name: fig-skip-connection
:alt: L'input x di un blocco attraversa due strati peso; una connessione laterale porta lo stesso x, non modificato, fino a un nodo somma che lo aggiunge all'uscita degli strati prima dell'attivazione finale.
:width: 55%

La connessione residua. L'ingresso del blocco (la $\mathbf{x}$ in cima)
scavalca i due strati interni lungo la linea di destra e viene sommato al loro
risultato, che il disegno chiama $\mathcal{F}(\mathbf{x})$: è la scorciatoia.
Solo dopo la somma i numeri passano dalla ReLU.
```

`````{tab} Elementare
L'idea è semplice ({numref}`fig-skip-connection`). Invece di chiedere a un
blocco di strati di ricostruire da capo tutto il segnale, gli si affianca una
"scorciatoia" che porta l'input intatto fino all'uscita, dove viene ri-sommato.
Così [il blocco deve imparare solo la correzione da apportare, non l'intera
risposta]{.pt-matita}; se non serve correggere nulla, può lasciar passare
l'input senza rovinarlo, e aggiungere strati non fa più danni.

Perché prima li facesse, quei danni, non è del tutto chiaro nemmeno oggi. Si sa
però che la soluzione buona esisteva anche nella rete profonda, e che la rete
non riusciva a trovarla: la scorciatoia gliela mette a portata di mano.

La somma, però, chiede una condizione: che i due pezzi abbiano la stessa forma,
tanti numeri di qua quanti di là, disposti allo stesso modo. Finché il blocco
restituisce un'uscita della forma dell'ingresso, la somma si può fare.

Ogni tanto la rete cambia formato (raddoppia le opinioni raccolte in ogni
punto, oppure rimpicciolisce le mappe), e lì i due pezzi hanno misure diverse,
quindi sommarli è impossibile. La scorciatoia porta allora una versione
riadattata dell'input, che di numeri da imparare ne ha anche lei, e
l'argomento «se non serve correggere nulla, non si guasta niente» non vale
più, perché lasciar passare tutto liscio ha un costo.

Nelle reti più profonde cambia anche l'interno del blocco. Con 256 opinioni per
punto, due lenti da tre per tre consecutive hanno più di un milione di numeri
da imparare.

Il blocco allora riassume prima le opinioni a 64, con la lente che guarda un
solo punto, fa il lavoro vero su quelle poche e alla fine le riporta a 256:
stessa forma in uscita, diciassette volte meno numeri. Per questo una rete di
centocinquantadue strati ha meno della metà dei numeri da imparare di VGG, che
di strati ne ha sedici.
`````

`````{tab} Superiore
Se $\mathcal{F}(\mathbf{x})$ è la trasformazione dei due strati interni, il
**blocco residuo** calcola

$$
\mathbf{y} = \mathcal{F}(\mathbf{x}, \{\mathbf{W}_i\}) + \mathbf{x},
$$

dove $\mathbf{x}$ è l'input del blocco, $\{\mathbf{W}_i\}$ i suoi pesi e
$\mathbf{y}$ l'uscita (a cui si applica poi la non-linearità). Il blocco
apprende quindi il residuo $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) -
\mathbf{x}$ rispetto alla mappa desiderata $\mathcal{H}$. Osserviamo che
[azzerare $\mathcal{F}$ per ottenere l'identità è facile, mentre ricostruire
l'identità da zero non lo è]{.pt-matita}.

Inoltre il termine additivo apre una via diretta al gradiente: lo jacobiano
del blocco è $\mathbf{I} + \partial\mathcal{F}/\partial\mathbf{x}$, quindi
$\partial\mathcal{L}/\partial\mathbf{x} = \partial\mathcal{L}/\partial\mathbf{y} + (\partial\mathcal{F}/\partial\mathbf{x})^{\top}\,\partial\mathcal{L}/\partial\mathbf{y}$,
e il primo addendo arriva intatto qualunque cosa faccia $\mathcal{F}$. Il
calcolo, però, vale se fra una somma e la successiva non c'è altro: la ReLU
applicata dopo l'addizione, come nella forma originale, interrompe la via
diretta.

Per questo la versione successiva degli stessi autori (*Identity
Mappings in Deep Residual Networks*, 2016) {cite}`he2016identity` sposta
normalizzazione e attivazione dentro il ramo $\mathcal{F}$, nella forma detta
*pre-activation*. Lungo una pila di blocchi vale allora
$\mathbf{x}_L = \mathbf{x}_\ell + \sum_{i=\ell}^{L-1}\mathcal{F}(\mathbf{x}_i)$,
e il gradiente in $\mathbf{x}_\ell$ contiene sempre
$\partial\mathcal{L}/\partial\mathbf{x}_L$ senza nessun fattore moltiplicativo.

Quella somma ha però una precondizione che l'equazione nasconde:
$\mathcal{F}(\mathbf{x})$ e $\mathbf{x}$ devono avere la stessa forma.
Quando un blocco raddoppia i canali o dimezza la risoluzione con uno stride,
sommarli dà un errore di dimensione (`RuntimeError`), non
un'approssimazione. L'articolo lo prevede nella sua equazione (2), dove la
scorciatoia porta una proiezione lineare,

$$
\mathbf{y} = \mathcal{F}(\mathbf{x}, \{\mathbf{W}_i\}) + \mathbf{W}_s\mathbf{x},
$$

realizzata come una convoluzione $1\times1$ con lo stesso stride; in
`torchvision` è il modulo `downsample`, che compare solo sul primo blocco di
uno stage, e solo dove serve.

In una ResNet-50 il primo blocco del primo stage ce l'ha, perché il collo di
bottiglia quadruplica i canali da 64 a 256; in una ResNet-18 lo stesso blocco
non ce l'ha, perché lì la forma non cambia affatto. Dove il `downsample` c'è,
l'argomento «azzerare $\mathcal{F}$ dà l'identità gratis» smette di valere,
poiché la scorciatoia non è l'identità ma una trasformazione con pesi da
imparare.

Le versioni profonde, poi, non impilano quel blocco così com'è. Da ResNet-50 in
su si usa il blocco a **collo di bottiglia**: tre convoluzioni al posto di due,
una $1\times1$ che riduce i canali, una $3\times3$ che lavora nello spazio
ridotto e una $1\times1$ che li riporta al numero di partenza.

La convoluzione $1\times1$ di NiN serve qui a un secondo scopo, ridurre il
numero di pesi. Con 256 canali in ingresso, per esempio, due $3\times3$ da
$256$ a $256$ costano $2\cdot 9\cdot 256^2 = 1\,179\,648$ pesi, mentre una
$1\times1$ ($256\to64$), una $3\times3$ ($64\to64$) e una $1\times1$
($64\to256$) ne costano $16\,384 + 36\,864 + 16\,384 = 69\,632$, diciassette
volte meno. Per questo una rete di 152 strati sta in $60$ milioni di
parametri, meno della metà di VGG-16, che ha sedici strati.

Usare la via diretta al gradiente come spiegazione della degradazione è
l'inversione di causa più diffusa su ResNet. Gli autori, infatti, la escludono
espressamente, perché le reti *plain* (senza scorciatoie) con cui fanno il
confronto erano addestrate con batch normalization e i loro gradienti
all'indietro avevano norme sane. Il problema che le connessioni residue
risolvono è quindi di ottimizzazione, non di gradiente che svanisce, e il
meccanismo preciso per cui funzionano resta oggetto di studio.
`````

## DenseNet: se una scorciatoia funziona, prendile tutte

La connessione residua apre una strada, e nel 2017 Gao Huang e colleghi, fra
Cornell, Tsinghua e Facebook AI Research, la percorrono fino in fondo con
**DenseNet** {cite}`huang2017densely`, premiata (a pari merito con un altro
lavoro) come miglior articolo della CVPR, il congresso principale della visione
artificiale.

Se ResNet somma l'input all'uscita del blocco, DenseNet li affianca. Dentro un
**blocco denso** ogni strato riceve le feature di tutti gli strati precedenti,
messe una accanto all'altra invece che sommate (l'operazione si chiama
*concatenazione*). *Denso*, qui, si riferisce ai collegamenti fra
gli strati, ognuno con tutti i precedenti; lo strato denso delle reti
convoluzionali è un'altra cosa, con ogni neurone collegato a tutti gli
ingressi.

```{figure} ../figures/blocco-denso.svg
:name: fig-blocco-denso
:alt: "Blocco denso con tre strati: archi di concatenazione portano l'input e l'uscita di ogni strato a tutti i nodi di concatenazione successivi, dove le feature vengono affiancate lungo i canali prima di entrare nello strato seguente."
:width: 100%

Il blocco denso: ogni strato riceve, affiancate, le feature di tutti gli
strati precedenti, mentre nel blocco residuo c'è una sola scorciatoia. Le mappe
che uno strato produce, una per filtro, sono i suoi *canali*: affiancarle è
concatenare i canali invece di sommarli, e nel disegno ogni strato ne aggiunge
alla pila un numero fisso, sempre lo stesso, che si chiama $k$.
```

`````{tab} Elementare
Se una scorciatoia che porta l'input intatto fino all'uscita funziona così
bene, perché fermarsi a una? In un blocco denso ogni strato riceve tutte le
mappe prodotte prima, messe semplicemente una accanto all'altra
({numref}`fig-blocco-denso`). È come una chat di gruppo in cui ogni nuovo
messaggio ha sotto gli occhi l'intera conversazione: nessuna informazione va
riassunta o ricostruita, basta consultarla.

Poiché può contare su tutto il lavoro già fatto, [ogni strato aggiunge poco di
suo (poche mappe nuove per volta) e la rete resta sorprendentemente
snella]{.pt-matita}.

Due mappe, però, si possono affiancare solo se hanno la stessa misura, quindi
la conversazione non prosegue all'infinito: ogni tanto la rete la chiude,
rimpicciolisce tutte le mappe e ne comincia una nuova. Il rovescio della
medaglia è lo stesso delle chat: la cronologia cresce, ogni strato se ne tiene
una copia sua, e la memoria occupata sale molto più in fretta del numero di
strati.
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
$k_0 + k(\ell-1)$ canali, dove $k_0$ sono quelli dell'input.

[Il riuso delle feature rende la rete efficiente nei parametri]{.pt-matita} (a
parità di accuratezza su ImageNet, all'incirca la metà di una ResNet
comparabile), e la concatenazione apre a ogni strato un percorso diretto verso
il gradiente della loss. Poiché concatenare richiede mappe della stessa
dimensione spaziale, i blocchi densi sono separati da **strati di transizione**
(convoluzione $1\times 1$ e pooling), che dimezzano la risoluzione.

Il costo è la memoria in addestramento. In un'implementazione ingenua, infatti,
le attivazioni concatenate crescono col quadrato della profondità del blocco, e
a parità di accuratezza una ResNet ne consuma un po' meno. Le implementazioni
parsimoniose ricalcolano le concatenazioni invece di conservarle, riportano la
crescita a lineare e ribaltano il confronto; il risparmio di memoria, però,
richiede più tempo di calcolo.
`````

## Separare lo spazio dai canali: la convoluzione che sta in un telefono

Fin qui la corsa è andata verso l'alto: più strati, più connessioni, più
accuratezza, senza badare troppo al costo. Attorno al 2016, però, una parte
della ricerca rovescia la domanda: a parità di accuratezza, quanto calcolo si
può risparmiare? Da questo dipende se la visione artificiale esce dai centri di
calcolo ed entra in un telefono, in una telecamera, in un'automobile. La
risposta più fruttuosa viene da un'osservazione sulla convoluzione stessa.

`````{tab} Elementare
Una convoluzione ordinaria fa due lavori in una volta sola, e non ce ne
accorgiamo perché li fa insieme. Il primo è guardarsi intorno: prendere un
quadratino di $3\times3$ pixel e cercarci una forma. Il secondo è mettere
d'accordo i canali: combinare quello che dicono tutte le opinioni raccolte in
quel punto (il bordo, il colore, la trama) in una nuova opinione.

L'idea è di smettere di farli insieme. Prima si guarda intorno, ma un canale
per volta: ogni opinione viene esaminata separatamente nel suo quadratino,
senza mescolarsi con le altre. Poi si mettono d'accordo i canali con una lente
che guarda un solo punto: è la convoluzione $1\times1$ di *Network in
Network*, che qui trova il suo impiego più importante.

Il risultato ha la stessa forma di prima, ma con molti meno numeri da imparare.
Con un quadratino di tre per tre, 64 opinioni in entrata e 128 in uscita, ad
esempio, si passa da $73\,728$ numeri a $8\,768$, cioè poco più di otto volte
di meno.

Più sono le opinioni in uscita, più ci si avvicina a nove volte, ma
nove non si supera mai. La convoluzione ordinaria, infatti, ha nove volte i
numeri della sola parte che mette d'accordo le opinioni ($64 \times 128$),
mentre quella separata ha quella parte più un piccolo resto ($9 \times 64$), e
il resto non sparisce.

La ragione sta nel modo in cui i pesi si contano. Nella versione ordinaria ogni
combinazione di «quale pixel del quadratino», «quale opinione di partenza» e
«quale opinione di arrivo» ha il suo peso, e i tre elenchi si moltiplicano fra
loro: $9 \times 64 \times 128$. [Separando, due dei tre si sommano invece di
moltiplicarsi]{.pt-matita}: $9 \times 64$, più $64 \times 128$.

Qualcosa si perde, però. Le combinazioni che mescolavano in un colpo solo il
posto nel quadratino e l'opinione di partenza adesso vanno ottenute in due
tempi, e in due tempi non vengono tutte uguali. All'atto pratico erano quasi
tutte ripetizioni, quindi l'accuratezza ne risente pochissimo. Su questo
mattone, a partire da una famiglia di reti di Google che si chiama MobileNet,
sono costruite molte delle reti che girano sui telefoni.
`````

`````{tab} Superiore
Una convoluzione standard $k \times k$ da $C_{\text{in}}$ a $C_{\text{out}}$
canali ha $k^2 C_{\text{in}} C_{\text{out}}$ pesi e costa, per pixel d'uscita,
altrettante moltiplicazioni-accumulo. La **convoluzione separabile in
profondità** (*depthwise separable*) la fattorizza in due passi:

1. **depthwise**: una convoluzione $k \times k$ applicata a ciascun canale
   indipendentemente (in PyTorch, `groups=C_in`), con $k^2 C_{\text{in}}$
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

cioè [il risparmio tende a $k^2$, che per i filtri $3\times3$ fa
$9$]{.pt-matita}. La convergenza, però, è lenta: con $C_{\text{out}}=32$ il
rapporto vale $7{,}0$, cioè il 78% del limite; a $128$ canali $8{,}4$; a $256$
ancora soltanto $8{,}7$. In altri termini, il «quasi nove volte» è la promessa
asintotica, e con i 128 canali in uscita dell'esempio numerico si sta
misurabilmente al di sotto.

Il guadagno viene da una fattorizzazione del kernel. La separabile in
profondità, infatti, è la convoluzione piena con il kernel vincolato alla forma
$K_{f,c,m,n} = P_{f,c}\,D_{c,m,n}$ (con gli indici della
{doc}`sezione sulle reti convoluzionali <reti-convoluzionali>`: $f$ il filtro,
$c$ il canale, $(m,n)$ la posizione nel kernel), cioè un filtro spaziale $D_c$
per ogni canale e una matrice di mescolamento $\mathbf{P}$ uguale in tutte le
posizioni.

Il tensore d'uscita ha la stessa forma, ma si perdono le combinazioni
spazio-canale che non si fattorizzano così. Howard e colleghi misurano che
cosa si perde: su ImageNet la MobileNet separabile fa il 70,6% di accuratezza
contro il 71,7% della stessa rete con convoluzioni piene, con otto volte e
mezzo meno moltiplicazioni-accumulo {cite}`howard2017mobilenets`.

L'idea viene da Laurent Sifre, che la sviluppò nel 2013 durante un tirocinio a
Google Brain e la provò su AlexNet; Inception V1 e V2 la usavano già come primo
strato {cite}`chollet2017xception`. È **MobileNet**
{cite}`howard2017mobilenets` a farne l'ossatura di una famiglia di reti pensate
per il calcolo su dispositivo, con due iperparametri espliciti (un
moltiplicatore di larghezza e uno di risoluzione) per scendere lungo la curva
costo-accuratezza.

**Xception** {cite}`chollet2017xception` porta la stessa idea al limite dentro
un'architettura in stile Inception, leggendola come l'ipotesi estrema che
correlazioni spaziali e correlazioni fra canali si possano trattare del tutto
separatamente.

**MobileNetV2** {cite}`sandler2018mobilenetv2` aggiunge un elemento che è
arrivato fino a oggi: il **residuo invertito** con collo di bottiglia lineare.
Il blocco *espande* i canali con una $1\times1$, applica la depthwise nello
spazio espanso, poi *ricomprime* con un'altra $1\times1$ senza non-linearità
finale (perché una ReLU su uno spazio a poche dimensioni distrugge informazione
che non si recupera), e la connessione residua collega i due estremi stretti
anziché quelli larghi, cioè all'opposto di ResNet, per tenere basso il consumo
di memoria. Questo blocco, detto **MBConv**, è il mattone di cui è fatta la
rete base di EfficientNet.

La stessa economia si può ottenere anche in un altro modo. Il *Fire module* di
**SqueezeNet** {cite}`iandola2016squeezenet` alterna uno strato di *squeeze* a
$1\times1$, che riduce i canali, e uno di *expand*, che li riespande con un
misto di $1\times1$ e $3\times3$, arrivando all'accuratezza di AlexNet con
cinquanta volte meno parametri.
`````

Possiamo rifare il calcolo in PyTorch, contando i parametri di una
convoluzione ordinaria e di una separabile con le stesse misure.

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

```text
stessa forma in uscita: True (1, 128, 56, 56)
parametri, ordinaria :   73,728
parametri, separabile:    8,768
risparmio            : 8.41x
previsto dalla formula: 8.41x   (limite: 9x)
```

I pesi scendono quindi da $73\,728$ a $8\,768$, cioè $8{,}41$ volte meno, e il
risultato ha la stessa forma di prima. Il numero stampato coincide fino
all'ultima cifra con il calcolo sulla carta, perché qui non c'è niente di
sperimentale: è aritmetica.

L'errore più comune di chi progetta reti per il telefono è dimenticare che il
fattore nove sta nei pesi e nelle moltiplicazioni, non nei secondi. La
convoluzione che filtra un canale per volta, infatti, esegue pochissime
operazioni per ogni dato che deve leggere dalla memoria, quindi il tempo se ne
va nel trasferire i dati più che nel calcolarli, e su una macchina vera il
guadagno misurato è una frazione di quello teorico, a volte nullo. Per questo
le architetture per dispositivo si valutano cronometrandole, non contando le
operazioni.

## Progettare architetture: dall'artigianato al metodo

Dal 2012 al 2016 le architetture sono state in gran parte artigianato d'alta
scuola, fatto di intuizioni individuali provate e riprovate su ImageNet a colpi
di settimane di GPU. Poi la progettazione diventa metodo, in due direzioni. La
prima sposta la domanda da *quale* rete disegnare a *come farla crescere*, ed è
la domanda di **EfficientNet** {cite}`tan2019efficientnet`, che nel 2019 le
dà una risposta precisa, il *compound scaling*.

`````{tab} Elementare
Per fare una torta doppia non si raddoppia solo la farina, ma si aumentano
tutti gli ingredienti in proporzione, o il risultato è immangiabile. Una
rete può crescere in tre modi: più strati (la profondità), più
lenti per strato (la larghezza), immagini d'ingresso più grandi (la
risoluzione).

Invece di puntare tutto su uno solo, [EfficientNet li fa crescere insieme, in
proporzioni fisse trovate una volta per tutte]{.pt-matita}: ogni volta che si
accetta di spendere il doppio in conti, la profondità aumenta di circa il 20%,
la larghezza del 10% e la risoluzione del 15%.

Sembrano aumenti minuscoli per un raddoppio, e invece bastano: i tre si
moltiplicano fra loro, e due contano al quadrato. La risoluzione perché
l'immagine cresce in larghezza e in altezza; le lenti perché con più lenti
ciascuna ha anche più roba da leggere. Moltiplicando 1,2 per 1,1 per 1,1 per
1,15 per 1,15 si ottiene 1,92, poco meno di due, come si voleva. Ripetendo la
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
grid search sulla rete base.

Il vincolo fa sì che [ogni incremento unitario di $\phi$ raddoppi all'incirca i
FLOPs]{.pt-matita} (che crescono come $\alpha \cdot \beta^2 \cdot \gamma^2$
elevato a $\phi$). Dalla rete base EfficientNet-B0, il compound scaling genera
la famiglia B1–B7, che a parità di accuratezza su ImageNet usa fino a quasi un
ordine di grandezza di parametri in meno rispetto alle CNN precedenti.
`````

La seconda direzione è automatizzare la ricerca stessa: nella **neural
architecture search** è un algoritmo a disegnare la rete. L'algoritmo non
prova tutte le reti possibili, perché sarebbe impossibile, ma ne prova alcune,
guarda quali vanno meglio e da quelle ricava le prossime da provare. La rete
base di EfficientNet, ad esempio, è stata trovata così.

La storia, però, non è chiusa. Dal 2020 i **Vision Transformer**
{cite}`dosovitskiy2021image`, reti basate sull’attenzione (è il nome di un
meccanismo preciso, non la parola di tutti i giorni) e nate per il linguaggio,
hanno dimostrato di poter competere con le CNN quando i dati abbondano. Oggi in
visione artificiale le due famiglie convivono e si scambiano idee; ne
riparleremo nella {doc}`sezione sui Vision Transformer e i modelli multimodali
</Transformers/multimodalita>`.

## L'architettura decide dove passa l'informazione

Nessuna di queste reti ha vinto soltanto con più esempi o più GPU. [Ogni salto
è nato da un'idea su come far circolare l'informazione dentro la
rete]{.pt-matita}: dove farla passare, dove farla saltare, dove riunirla con sé
stessa.

Parecchie di quelle idee, inoltre, rispondevano a un vincolo di costo, cioè non
a «come faccio a metterci più roba» ma a «come faccio a spendere meno»: i
collegamenti risparmiati fra uno strato e l'altro di LeNet, la rete divisa fra
le due schede di AlexNet, il *global average pooling* di NiN, il collo di
bottiglia che riduce i canali prima della convoluzione grande, in Inception e
in ResNet, la convoluzione separata in due tempi di MobileNet.

A parità di dati e di calcolo, l'architettura sposta l'accuratezza che si
riesce a raggiungere. Quanto l'architettura pesi rispetto a più dati o più
calcolo è un'altra domanda, quella delle leggi di scala della {doc}`sezione sui
grandi modelli di linguaggio </Transformers/llm>`. Secondo quelle leggi, dentro
una stessa famiglia di modelli, e in limiti ragionevoli, l'errore dipende
soprattutto dalla scala (parametri, dati, calcolo) e poco dai dettagli della
forma, come il rapporto fra profondità e larghezza {cite}`kaplan2020scaling`.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- LeNet-5 (1998): la piccola lente che scorre sull'immagine un pezzetto
  alla volta, sempre la stessa in ogni punto, e impara a leggere le cifre
  scritte a mano.
- AlexNet (2012): la stessa idea cresciuta (molti più strati, molte più
  lenti), addestrata su schede grafiche, con la regola sbrigativa che sotto lo
  zero spegne e sopra lascia passare il numero com'è (la ReLU) e con due
  accorgimenti per non imparare a memoria: spegnere neuroni a caso a ogni
  passata, e non mostrare mai la stessa fotografia due volte uguale. Nel 2012
  vince ImageNet e convince tutti.
- NiN (2013): una lente che guarda un solo punto ma legge tutte le
  opinioni raccolte lì e le fonde; e un finale che, invece di enormi strati
  densi, tiene una mappa per categoria e premia la più accesa.
- VGG (2014): il principio del mattoncino Lego, tante lenti piccole e
  uguali impilate una dopo l'altra al posto di poche lenti grandi. Il risparmio
  però riguarda le lenti: quasi nove decimi dei suoi centotrentotto milioni di
  numeri stanno negli enormi strati densi con cui la rete chiude, quelli che
  NiN aveva appena mostrato come togliere.
- Inception/GoogLeNet (2014): guardare lo stesso punto con lenti di misure
  diverse nello stesso istante, limitando i numeri da imparare grazie alla
  lente che guarda un punto solo.
- ResNet (2015): la scorciatoia che porta l'input intatto fino all'uscita,
  dove viene ri-sommato; al blocco resta da imparare solo la correzione, e
  così si addestrano reti di centinaia di strati. Dove la rete cambia formato i
  due pezzi non si sommano, e la scorciatoia porta una versione riadattata
  dell'input, con numeri da imparare anche lei.
- DenseNet (2017): non una scorciatoia ma tutte, come una chat di gruppo
  in cui ogni strato ha sotto gli occhi l'intera conversazione (pochi pesi,
  molta memoria).
- La convoluzione separabile smette di fare due lavori insieme: prima guarda
  intorno un canale per volta, poi mette d'accordo i canali con la lente che
  guarda un punto solo. Stessa forma in uscita e poco più di otto volte meno
  pesi (nove volte è il tetto, e ci si avvicina solo con moltissime opinioni in
  uscita): è il mattone delle reti che stanno in un telefono, a partire da
  MobileNet (2017).
- Dopo l'artigianato, il metodo: EfficientNet fa crescere insieme profondità,
  larghezza e dimensione delle immagini come gli ingredienti di una torta; e
  la ricerca automatica delle architetture disegna la rete al posto nostro.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- LeNet-5 (1998): la convoluzione che legge le cifre (connettività locale,
  pesi condivisi, pooling).
- AlexNet (2012): ReLU, dropout e GPU portano le CNN a vincere ImageNet.
- NiN (2013): convoluzioni $1\times 1$ come mini-rete pixel per pixel e
  *global average pooling* al posto degli strati densi.
- VGG (2014): profondità con soli filtri $3\times 3$ impilati (campo
  recettivo $r_\ell = r_{\ell-1} + (k_\ell-1)\prod_{i<\ell}s_i$), ma
  $138$ milioni di parametri, l’$89{,}4\%$ nei tre densi finali.
- Inception/GoogLeNet (2014): elaborare a più scale in parallelo, con
  colli di bottiglia $1\times 1$.
- ResNet (2015): la connessione residua
  $\mathbf{y}=\mathcal{F}(\mathbf{x})+\mathbf{x}$ rende addestrabili reti di
  centinaia di strati; dove le forme non coincidono la scorciatoia porta una
  proiezione $\mathbf{W}_s\mathbf{x}$, e dalla ResNet-50 in su il blocco è a
  collo di bottiglia ($1\times1$, $3\times3$, $1\times1$).
- DenseNet (2017): ogni strato riceve, concatenate, le feature di
  tutti i precedenti (pochi parametri, molta memoria).
- La convoluzione separabile in profondità (MobileNet, Xception) fattorizza
  la convoluzione in *depthwise* ($k^2 C_{\text{in}}$ pesi) più *pointwise*
  $1\times1$ ($C_{\text{in}}C_{\text{out}}$): il costo scende di un fattore
  $k^2 C_{\text{out}} / (k^2 + C_{\text{out}}) \to k^2$. MobileNetV2 vi
  aggiunge il residuo invertito con collo di bottiglia lineare
  (MBConv), che è il blocco base di EfficientNet.
- Dopo l'artigianato, il metodo: il compound scaling di EfficientNet fa
  crescere insieme profondità, larghezza e risoluzione; la *neural
  architecture search* automatizza il progetto.
```
`````

Il tronco convoluzionale di queste reti non serve soltanto a ImageNet: gli
stessi strati possono servire più compiti insieme, nell’{doc}`apprendimento
multi-compito <multi-compito>`.

[^zip1989]: Il sistema riconosceva le cifre dei codici postali ritagliate
    dalle buste dal servizio postale statunitense
    {cite}`lecun1989backpropagation`.

[^annotatore]: Il 5,1% è la prova di Andrej Karpathy, che nel 2014 si
    addestrò al compito su 500 fotografie e poi ne classificò 1500
    {cite}`russakovsky2015imagenet`. Un secondo annotatore, addestrato su 100
    fotografie e provato su 258, sbagliò il 12,0%. Due persone sono troppo
    poche per una misura della "prestazione umana" in generale: il 5,1% dà
    un ordine di grandezza.
