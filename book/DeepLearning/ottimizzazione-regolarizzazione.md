# Far funzionare le reti profonde

Per molto tempo una rete con tanti strati è stata più un'idea che una pratica.
Negli anni Novanta e nei primi anni Duemila impilare più livelli spesso
*peggiorava* le cose: la loss (il numero che misura quanto la rete sbaglia) non
scendeva, l'addestramento si arenava dopo poche passate sui dati. Non era solo
questione di potenza di calcolo. Mancavano gli accorgimenti che rendono stabile
l'apprendimento quando la rete è profonda.

Sono arrivati fra il 2010 e il 2015, e sono una manciata: inizializzazioni più
accorte, la *batch normalization* {cite}`ioffe2015batch`, il *dropout*
{cite}`srivastava2014dropout`, gli optimizer adattivi come Adam
{cite}`kingma2015adam`. Insieme hanno trasformato le reti profonde da promessa
fragile a strumento affidabile. Questa sezione li mette in fila: prima il
problema, poi i rimedi.

## Quando il segnale svanisce (o esplode)

Una rete impara per correzioni: risponde, si vede dire di quanto ha sbagliato e
aggiusta i propri pesi. Il numero che dice a ciascun peso in che verso e di
quanto muoversi si chiama **gradiente**, e in questa sezione lo chiameremo
spesso, più alla buona, il *segnale di correzione*. La ricetta che lo calcola è
la **retropropagazione** (*backpropagation*): parte dall'uscita della rete e
risale verso l'ingresso, uno strato alla volta.

A ogni passo indietro quel segnale viene moltiplicato per i pesi dello strato e
per le **derivate** delle attivazioni. La derivata è il numero che dice quanto
l'uscita di un neurone reagisce a una piccola variazione del suo ingresso: è
grande dove il neurone è reattivo, e piccola dove è pigro, cioè dove muovere
l'ingresso non cambia quasi niente. Ed è qui che nasce il guaio: moltiplicare
tante volte per numeri piccoli.

```{figure} ../figures/gradiente-svanisce.svg
:name: fig-gradiente-svanisce
:alt: "Animazione: due file di barre, sigmoide e ReLU. Risalendo dall'uscita verso l'ingresso le barre della sigmoide si accorciano di un fattore quattro per strato, quelle della ReLU restano intere."
:width: 90%

Quanta parte del segnale di correzione sopravvive al passaggio di ogni strato,
con due attivazioni diverse: la **sigmoide**, la curva a S che schiaccia
qualunque numero dentro l'intervallo fra 0 e 1, e la **ReLU**, che lascia
passare i positivi come sono e azzera i negativi. Con la sigmoide ogni strato lo
moltiplica per un numero che non supera mai $0{,}25$, cioè lo divide almeno per
quattro: dal sesto strato al primo, che sono cinque passaggi, resta un
millesimo scarso di
quello di partenza ($0{,}25^5 = 0{,}00098$, l'ultima barra). Con la ReLU quel
numero vale $1$ dove il neurone è acceso, cioè dove il numero che gli entra è
positivo, e il prodotto non si consuma. *(Le barre
sono disegnate in scala logaritmica: la lunghezza cresce di un tratto uguale
ogni volta che il valore si moltiplica per dieci, e il disegno ne copre quattro
di questi tratti. In scala normale le ultime barre non si vedrebbero, che poi è
il punto.)*
```

I numeri della {numref}`fig-gradiente-svanisce` sono il caso **migliore** per
la sigmoide: $0{,}25$ è il massimo della sua derivata, e lo raggiunge in un
punto solo, quello in cui il numero che entra nel neurone vale zero. Altrove è
molto più piccolo, e il crollo è più rapido.

Verrebbe da concludere che con la ReLU il problema sia chiuso, e non lo è. Le
derivate sono solo metà della storia: a ogni passo indietro il segnale viene
moltiplicato **anche** per i pesi dello strato, e quelli, all'inizio, non li ha
ancora sistemati nessuno. È la ragione per cui la sezione continua.

`````{tab} Elementare

Immagina di fotocopiare un foglio scritto a matita, e poi di fotocopiare la
fotocopia, e poi la fotocopia della fotocopia. A ogni giro il grigio sbiadisce.
Se a ogni passaggio ne resta un decimo, dopo dieci copie di quello che c'era
scritto è rimasto un decimo di miliardesimo: un foglio bianco.

È ciò che succede al segnale di correzione che, dall'uscita della rete, deve
tornare fino ai primi strati: attraversando molti livelli si assottiglia fino a
sparire. Gli strati vicini all'ingresso non ricevono quasi nessuna indicazione
su come cambiare, e di fatto smettono di imparare. Il difetto opposto è pari e
contrario: se ogni passaggio *ingrandisce* invece di sbiadire, dopo poche
copie il segnale esplode in numeri enormi e l'addestramento va in tilt.

`````

`````{tab} Superiore

Il gradiente rispetto ai pesi di uno strato profondo è un prodotto di molti
fattori. Chiamiamo $\mathbf{z}_k$ gli ingressi dei neuroni dello strato $k$
prima dell'attivazione, e $\boldsymbol{\delta}_k = \partial\mathcal{L}/\partial
\mathbf{z}_k$ il gradiente della loss rispetto a quegli ingressi. La
retropropagazione è la ricorsione $\boldsymbol{\delta}_{k-1} =
\operatorname{diag}\!\big(\sigma'(\mathbf{z}_{k-1})\big)\,\mathbf{W}_k^\top
\boldsymbol{\delta}_k$, e svolgendola dalla cima fino allo strato $\ell$ in una
rete di $L$ strati si ottiene

$$
\boldsymbol{\delta}_\ell = \left[\;\prod_{k=\ell+1}^{L}
\operatorname{diag}\!\big(\sigma'(\mathbf{z}_{k-1})\big)\,
\mathbf{W}_k^\top \right] \boldsymbol{\delta}_L ,
$$

dove $\mathbf{W}_k$ è la matrice dei pesi dello strato $k$ e
$\sigma'(\mathbf{z}_{k-1})$ la derivata dell'attivazione calcolata negli
ingressi dello strato precedente. Il prodotto è **ordinato**: i fattori vanno
scritti da sinistra a destra per $k$ crescente e non si possono scambiare,
perché il prodotto di matrici non commuta. Se i fattori hanno modulo tipico
minore di $1$, il prodotto tende a $0$ esponenzialmente in $L$ (**vanishing
gradient**); se maggiore di $1$, diverge (**exploding gradient**).

Una nota sul simbolo, perché in questa sezione lavora troppo. $\sigma$ qui è
l’**attivazione generica**, qualunque essa sia, e non la sigmoide, che ne è
solo un caso particolare e che viene sempre nominata per esteso; più avanti,
nella batch normalization, $\sigma_{\mathcal{B}}$ sarà invece una deviazione
standard, come vuole la tradizione statistica. Tre mestieri per una lettera
sola: il contesto li distingue, ma è meglio saperlo prima che accorgersene.

La sigmoide aggrava il primo caso: la sua derivata non supera $0{,}25$ in
nessun punto, quindi il solo fattore di attivazione riduce il gradiente di
almeno quattro volte a ogni strato. Rimedi complementari:
attivazioni non saturanti come la ReLU ($\sigma'=1$ per input positivi),
*gradient clipping* per l'esplosione, e (soprattutto) una scelta accurata
della scala iniziale dei pesi.

`````

## Partire col piede giusto: l'inizializzazione

Se il prodotto di tanti fattori decide se il segnale svanisce o esplode, il
punto di partenza conta enormemente. Inizializzare i pesi con la scala
sbagliata condanna la rete prima ancora del primo aggiornamento.

Attenzione a una parola che da qui in avanti cambia mestiere. **«Attivazione»
indica due cose diverse**: la *funzione* che ogni neurone applica al proprio risultato (la
ReLU, la sigmoide) e i *numeri* che escono da uno strato dopo che quella
funzione è stata applicata. «La derivata dell'attivazione» è la prima cosa;
«normalizzare le attivazioni», che è ciò di cui parleremo tra poco, sono i
secondi. Il contesto basta a distinguerle, ma conviene saperlo in anticipo
invece di inciampare.

```{figure} ../figures/inizializzazione-pesi.svg
:name: fig-inizializzazione
:alt: "Grafico della varianza del segnale strato dopo strato, con tre curve. Con pesi iniziali troppo grandi la varianza cresce di strato in strato ed esplode; con pesi troppo piccoli collassa verso lo zero; con la scala calibrata resta costante lungo tutta la profondità della rete."
:width: 92%

Tre inizializzazioni, tre destini, e nessun addestramento ancora avvenuto. In
verticale c'è la **varianza** del segnale, cioè quanto sono sparpagliati i
numeri che escono da uno strato: grande vuol dire valori forti e distanti fra
loro, vicina a zero vuol dire valori tutti appiccicati, cioè un segnale ormai
spento. La curva piatta è l'obiettivo: quell'ampiezza deve attraversare la rete
senza gonfiarsi né spegnersi.
```

Il fatto che le tre curve di {numref}`fig-inizializzazione` divergano *prima*
del primo aggiornamento è ciò che rende l'inizializzazione un problema a sé. Non
è una scorciatoia per arrivare più in fretta: con la scala sbagliata la rete non
arriva affatto, perché il segnale che dovrebbe correggerla è già rovinato al
primo passaggio.

`````{tab} Elementare

L'idea è tenere costante il "volume" del segnale (l'ampiezza che la curva del
disegno qui sopra chiama varianza) mentre attraversa gli strati: né più forte
né più debole. La regola è semplice: più ingressi somma un neurone, più piccoli
devono essere i suoi pesi iniziali, così che la somma non gli scappi di mano.

Le ricette collaudate sono due, e si distinguono per quale attivazione hanno in
mente. **Xavier/Glorot** (un nome solo scritto in due modi: Xavier è il nome di
battesimo di Glorot) è tarata sulle curve che scendono sotto lo zero tanto
quanto salgono sopra: la più usata è la stessa curva a S della sigmoide, ma
centrata sullo zero, fra $-1$ e $+1$, e si chiama *tanh*. **He** è tarata sulla
ReLU: siccome la ReLU azzera i numeri negativi, cioè butta via metà di quello
che riceve, i pesi partono più grandi per compensare.

Si legge spesso che ormai una delle due è l'impostazione predefinita e che non
serve toccarla: attenzione, perché non è così. Uno strato creato in PyTorch
senza dire niente non nasce con i pesi di He: ne ha con una varianza sei volte
più piccola, per una scelta ereditata dalle prime versioni della libreria e mai
più cambiata. E nemmeno con i **bias** a zero: il bias è quel numero fisso che
ogni neurone somma sempre al proprio risultato, e che dovrebbe partire da zero
perché all'inizio non c'è nessuna ragione di preferire un verso all'altro;
PyTorch invece lo estrae a caso come i pesi.

Su una rete di pochi strati la differenza si assorbe e non la nota nessuno. Su una pila di quaranta il segnale di correzione che arriva al primo strato è
un milionesimo di miliardesimo di miliardesimo di quello che ci arriverebbe
partendo da He, un numero che si scrive con diciassette zeri dopo la virgola: non è ancora zero, ma tanto vale. E su una pila di sessanta
diventa zero per davvero, non per modo di dire: è un numero così piccolo che il
computer non ha più cifre per scriverlo, e scrive zero. Scegliere
l'inizializzazione è un passo del mestiere, non un dettaglio da lasciare alla
libreria.

`````

`````{tab} Superiore

L'obiettivo è preservare la varianza delle attivazioni (e dei gradienti) da
uno strato all'altro. Con $n_{\text{in}}$ ingressi e $n_{\text{out}}$ uscite,
l'inizializzazione di **Glorot** {cite}`glorot2010understanding` campiona i
pesi con varianza

$$
\operatorname{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}},
$$

adatta ad attivazioni simmetriche attorno a zero (tanh). Per la ReLU, che
azzera metà degli ingressi, **He** {cite}`he2015delving` raddoppia la scala
usando solo il fan-in:

$$
\operatorname{Var}(w) = \frac{2}{n_{\text{in}}}.
$$

In entrambi i casi $w$ si estrae da una normale (o da una uniforme con
supporto equivalente) e i bias si pongono a $0$. La regola pratica: **He** con
ReLU e varianti, **Glorot** con tanh e sigmoide.

Quello che queste due ricette **non** sono è il comportamento predefinito di
PyTorch, ed è un equivoco che costa poco credere e parecchio pagare.
`nn.Linear` e `nn.Conv2d` inizializzano i pesi con
`kaiming_uniform_(a=math.sqrt(5))`, che nonostante il nome produce varianza

$$
\operatorname{Var}(w) = \frac{1}{3\,n_{\text{in}}},
$$

cioè **un sesto** di He. Su `nn.Linear(100, 100)` la varianza misurata dei pesi
è $3{,}34\times10^{-3}$ contro i $2{,}00\times10^{-2}$ di He. E i bias non sono
nulli: escono da una uniforme $\pm 1/\sqrt{n_{\text{in}}}$, la stessa scala dei
pesi.

Su reti poco profonde la differenza si assorbe. Su una pila di quaranta blocchi
`Linear(100, 100)` + ReLU no. Il protocollo, perché la misura si possa rifare:
ingresso $64\times100$ da una normale standard, loss l'errore quadratico medio
dell'uscita contro zero, si legge la norma del gradiente sui pesi del **primo**
strato, mediana su cinque semi. I **bias** seguono ciascuno la propria ricetta:
azzerati con He e con Glorot, come le due prescrivono, e lasciati come li mette
PyTorch nel caso del default, che è appunto quel che si ottiene senza toccare
niente. Non è un dettaglio: azzerando anche quelli del default, si arriva a
zero esatto già a quaranta blocchi invece che a sessanta, perché quei valori
uniformi sono l'unica cosa che tiene in vita il segnale quando i pesi lo
spengono. Viene $4{,}2\times10^{-1}$ inizializzando alla
He, $5{,}5\times10^{-13}$ alla Glorot e $9{,}7\times10^{-18}$ con il default di
PyTorch: più di sedici ordini di grandezza fra la prima e l'ultima, e
l'addestramento non è ancora cominciato. La dispersione fra semi è ampia (con
He il singolo seme va da $1{,}4\times10^{-1}$ a $3{,}8$), il divario fra le tre
no. Portando la pila a **sessanta** blocchi il default arriva esattamente a
$0{,}0$, in cinque semi su cinque, e lì lo zero non è un modo di dire ma un
underflow in `float32`. Che la questione sia nota a chi scrive le
librerie lo dicono le librerie stesse: la ResNet di `torchvision` non si fida
del default e reinizializza ogni convoluzione con
`nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")`.

`````

## Normalizzare mentre si impara: la batch normalization

Anche partendo bene, i numeri che circolano dentro la rete cambiano scala di
continuo mentre si impara: ogni aggiornamento modifica ciò che arriva allo
strato successivo. La *batch normalization* interviene qui, rimettendo in riga
i numeri che escono da ogni strato, e in pratica rende l'addestramento molto
più rapido e stabile.

```{figure} ../figures/batch-normalization-2015.svg
:name: fig-batch-norm
:alt: "Tre riquadri in fila. Nel primo, tre distribuzioni delle attivazioni provenienti da batch diversi, spostate l'una rispetto all'altra e di larghezza diversa. Una freccia marcata BN porta al secondo riquadro, dove resta una sola curva centrata sullo zero, di media zero e varianza uno: le tre sono diventate indistinguibili. Una seconda freccia, marcata gamma e beta, porta al terzo riquadro, dove la curva è di nuovo spostata e riallargata, stavolta come decide la rete."
:width: 96%

Da tre **distribuzioni** che vagano a una sola. Una distribuzione è la gobba che
si ottiene segnando quanti numeri cadono in ciascun punto: alta dove i numeri si
addensano, bassa dove sono rari, e spostata a destra o a sinistra a seconda di
dove sta il grosso. Nel riquadro di mezzo le tre non sono sparite: sono
diventate indistinguibili, che è esattamente il punto, e per
questo se ne disegna una. Il terzo riquadro è la coda dell'operazione: due
manopole, chiamate $\gamma$ (gamma) e $\beta$ (beta), restituiscono alla rete la
libertà che la normalizzazione le ha appena tolto, la prima riallargando o
restringendo i numeri, la seconda spostandoli in su o in giù.
```

La coda di {numref}`fig-batch-norm` è la parte che spesso si salta e che
invece conta. Normalizzare e basta imporrebbe a ogni strato un'ampiezza decisa
da noi; $\gamma$ e $\beta$ sono due numeri che la rete impara come tutti gli
altri, e le permettono di riallargare e spostare il risultato se le conviene,
invece di subire la scala che abbiamo scelto noi.

`````{tab} Elementare

Le reti non si addestrano un esempio alla volta: si prende un gruppetto di
esempi, di solito da qualche decina a qualche centinaio, si guarda quanto la
rete sbaglia su tutti insieme e si fa un'unica correzione. Quel gruppetto si
chiama **mini-batch**, o batch per brevità, ed è l'unità di misura
dell'addestramento.

La batch normalization fa questo: a ogni strato ricentra le attivazioni perché
abbiano media zero e ampiezza regolare, e la media e l'ampiezza le misura sul
mini-batch corrente. È come rimettere in scala i numeri a ogni passo, così che
nessuno strato debba adattarsi a ingressi che cambiano scala di continuo. In
pratica accelera molto l'addestramento, permette learning rate (il passo di
correzione dei pesi) più aggressivi e ha un lieve effetto di regolarizzazione
(cioè frena l'imparare a memoria), perché ogni gruppetto ha statistiche un po’
diverse dal precedente e quella variabilità fa da rumore utile.

`````

`````{tab} Superiore

Sul mini-batch $\mathcal{B}$ si calcolano media $\mu_{\mathcal{B}}$ e varianza
$\sigma_{\mathcal{B}}^2$, si normalizza e si riscala con due parametri appresi
$\gamma$ (scala) e $\beta$ (shift):

$$
\hat{x} = \frac{x - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}},
\qquad
y = \gamma\,\hat{x} + \beta.
$$

Il termine $\epsilon$ evita la divisione per zero.

**Su che cosa** si calcolino quelle statistiche è la domanda che decide quanti
parametri ha lo strato, e la risposta dipende dal tipo di dato. In una rete
densa la coppia $(\gamma,\beta)$ è per unità. In una rete convoluzionale
sarebbe assurdo trattare i pixel come feature diverse, visto che la mappa è
prodotta dallo stesso filtro in ogni punto: media e varianza si calcolano
**per canale**, su tutte le posizioni e tutti gli esempi del batch insieme (gli
assi $(N,H,W)$), e la coppia $(\gamma,\beta)$ è una per canale. Per questo
`nn.BatchNorm2d(16)` ha $32$ parametri appresi e non $2\,C\,H\,W$: sedici
$\gamma$ e sedici $\beta$, e la risoluzione delle mappe non c'entra.

Su che cosa restituiscano $\gamma$ e $\beta$ conviene essere precisi, perché la
formula promette meno di come viene raccontata di solito. Rispetto alle
statistiche **fisse** usate in inferenza, sì: con $\gamma=\sqrt{\sigma^2+
\epsilon}$ e $\beta=\mu$ lo strato torna esattamente l'identità. Lotto per
lotto no: $(\gamma,\beta)$ sono due costanti apprese, mentre
$(\mu_{\mathcal{B}},\sigma_{\mathcal{B}})$ cambiano a ogni batch, e nessuna
coppia di costanti può annullare una normalizzazione che si muove.

In inferenza infatti le statistiche del batch non si usano: al loro posto va
una **media mobile esponenziale** aggiornata durante l'addestramento,
$\hat{\mu} \leftarrow (1-m)\,\hat{\mu} + m\,\mu_{\mathcal{B}}$, non la media di
tutto ciò che si è visto. Attenzione al nome del parametro: il `momentum` di
`nn.BatchNorm2d` (default $0{,}1$) è il peso del **dato nuovo**, cioè
l'opposto del $\beta_1$ di Adam, dove $0{,}9$ è il peso della **storia**.

Ioffe e Szegedy la
introdussero per contrastare l’*internal covariate shift*, cioè lo spostarsi
della distribuzione degli input di ogni strato durante l'addestramento, ma
quella spiegazione è stata contestata: Santurkar e colleghi
{cite}`santurkar2018batchnorm` mostrano che la stabilità distribuzionale
c'entra poco con il successo della batch normalization (si può iniettare
rumore *dopo* la normalizzazione, aumentando lo shift, senza perderne i
benefici) e propongono che l'effetto vero sia un panorama della loss più
liscio e percorribile. Anche questa resta un'ipotesi: il meccanismo per cui la
BN funziona è tuttora aperto.

`````

**Perché la batch normalization funzioni così bene non lo sa ancora nessuno
con certezza.** Quello che fa è fuori discussione: sottrae la media, divide per la
dispersione, e lascia alla rete due manopole per rimettere le cose a modo suo.
Il perché no.

I suoi autori dicevano che serve a impedire alla distribuzione dei numeri di
spostarsi sotto i piedi di ogni strato mentre la rete impara. Un lavoro
successivo ha mostrato che quella spiegazione non regge: si può rimescolare
apposta i numeri *dopo* la normalizzazione, cioè far spostare la distribuzione
ancora di più, e i benefici restano tutti. La spiegazione che ha preso il posto
della prima (che la normalizzazione renda più regolare il modo in cui l'errore
cambia quando si toccano i pesi, e quindi più facile capire in che direzione
conviene muoversi) è a sua volta un'ipotesi, non una dimostrazione.

Non è un motivo per non usarla, è un motivo per diffidare delle spiegazioni
troppo pulite: nel deep learning capita spesso che una tecnica sia solidissima
in pratica e ancora senza una teoria che regga.

## Spegnere neuroni a caso: il dropout

La batch normalization regolarizza un po’ come effetto collaterale. Il dropout
lo fa per scelta esplicita, ed è uno dei modi più semplici per combattere
l’*overfitting*, cioè il caso in cui la rete impara a memoria gli esempi che le
sono stati mostrati e su quelli nuovi sbaglia.

```{figure} ../figures/dropout.svg
:name: fig-dropout
:alt: "Animazione: una rete con due strati nascosti; a ogni mini-batch si spegne un sottoinsieme diverso dei neuroni nascosti, estratto a caso e in media grande la metà, insieme alle sue connessioni, mentre input e output restano sempre attivi."
:width: 90%

Quattro mini-batch consecutivi con $p = 0{,}5$: ogni volta la rete che viene
davvero addestrata è **un'altra**. Ogni neurone nascosto se la gioca a testa o
croce per conto proprio, quindi il numero di quelli spenti cambia da un passo
all'altro. Input e output non si spengono mai.
```

`````{tab} Elementare

Durante l'addestramento, a ogni passo, spegniamo a caso una frazione dei
neuroni. La rete non può più affidarsi a un singolo neurone "specialista":
deve distribuire la conoscenza, perché quel neurone potrebbe non esserci al
prossimo giro. È come allenare ogni volta una squadra leggermente diversa: il
risultato è un modello più robusto, che generalizza meglio su dati nuovi.
Quando poi la rete lavora sul serio, cioè quando risponde invece di allenarsi
(si dice *a inferenza*), tutti i neuroni tornano attivi.

Verrebbe da chiedersi se a quel punto i numeri non raddoppino, visto che
raddoppiano i neuroni che li producono. Non succede, perché il conto è già stato
pareggiato prima: durante l'allenamento, quando metà dei neuroni è spenta, i
sopravvissuti vengono raddoppiati sul posto, così la somma che esce ha la taglia
giusta fin da subito. A rete piena non resta niente da aggiustare.

`````

`````{tab} Superiore

Con probabilità di spegnimento $p$, la convenzione di `nn.Dropout(p)` in
PyTorch, si applica alle attivazioni una maschera binaria
$\mathbf{m} \sim \text{Bernoulli}(1-p)$:

$$
\tilde{\mathbf{h}} = \frac{1}{1-p}\,(\mathbf{m} \odot \mathbf{h}),
$$

dove $\odot$ è il prodotto elemento per elemento. Il fattore $1/(1-p)$
(*inverted dropout*) mantiene invariato il valore atteso di ciascuna
attivazione, e a inferenza si usa direttamente la rete piena senza
riscalature. Attenzione a cosa questo garantisce: l'uscita della rete piena
**non** è la media dell'ensemble di sotto-reti, perché attraversare una
non-linearità non conserva il valore atteso. Ne è un'approssimazione (della
media geometrica delle distribuzioni predette), esatta solo per modelli senza
unità nascoste non lineari e per il resto giustificata dalla sola evidenza
empirica, che però è schiacciante. Valori tipici: $p \in [0{,}2,\ 0{,}5]$. Non
si combina bene con la batch normalization sullo stesso strato: spesso si
sceglie l'una o l'altro.

`````

Adesso si capisce anche perché il dropout venga descritto come un *ensemble*
implicito, cioè come una squadra di reti al posto di una sola. I neuroni che si
possono spegnere sono quelli **nascosti**, cioè quelli in mezzo, che non
ricevono i dati e non danno la risposta finale: se sono $n$, le combinazioni
possibili di acceso e spento sono $2^n$, che già con dieci neuroni fa 1024 reti
diverse e con venti più di un milione. Ogni passo di addestramento ne allena una
presa a caso. Quelle reti però non hanno pesi
propri: sono tutte ritagliate dallo stesso insieme di pesi, e un peso migliorato
adesso lo ritroveranno migliorato tutte le innumerevoli sotto-reti che lo
contengono. Non si allenano una alla volta: si allenano tutte, un pezzetto per
volta.

## Scendere bene: gli optimizer moderni

Tutti questi accorgimenti stabilizzano il *segnale*; resta da decidere *come*
muoversi una volta che lo si è ricevuto.

Serve prima un'immagine, che accompagnerà tutto il resto della sezione. Immagina
di poter disegnare, per ogni possibile scelta dei pesi, quanto la rete sbaglia
con quei pesi: ne viene fuori un paesaggio, con alture dove sbaglia molto e
conche dove sbaglia poco. Addestrare vuol dire camminare in quel paesaggio
cercando il fondo di una conca, e il gradiente è la pendenza sotto i piedi, che
dice da che parte si scende. È il **panorama della loss**, e la rete ci si muove
al buio: della pendenza nel punto in cui si trova sa tutto, del resto del
paesaggio niente.

Gli algoritmi che decidono come fare il passo si chiamano **ottimizzatori**, o
*optimizer*. La discesa del gradiente pura fa un passo proporzionale alla
pendenza e basta. In una valle stretta e allungata questo significa rimbalzare
da parete a parete invece di scivolare verso il fondo
({numref}`fig-momentum`).

```{figure} ../figures/discesa-momentum.svg
:name: fig-momentum
:alt: Due traiettorie di discesa in una valle stretta ed allungata; senza momentum la traiettoria oscilla da parete a parete, con momentum procede più diretta verso il minimo.
:width: 85%

In una valle stretta la discesa del gradiente senza momentum (terracotta)
oscilla tra le pareti ripide. Il momentum (teal) accumula velocità lungo la
direzione utile e smorza le oscillazioni, arrivando più dritto al minimo.
```

`````{tab} Elementare

Il **momentum** dà alla discesa un po’ di inerzia, come una pallina che rotola
in una valle: accumula velocità nella direzione giusta e si lascia dietro i
rimbalzi laterali. **Adagrad** aggiunge un'idea in più: dare a ogni parametro
(i pesi, più i bias: tutti i numeri che la rete regola) un passo su misura, più
corto dove il terreno è ripido e più lungo dove è piatto. Per farlo tiene il
conto di tutta la strada già percorsa, parametro per parametro: quelli corretti
di continuo rallentano, quelli toccati di rado conservano passi generosi.

Toccati di rado capita più spesso di quanto sembri. Quando una rete lavora su un
testo, per esempio, ogni parola del vocabolario viene trasformata in una fila di
numeri, e quei numeri sono parametri come gli altri: la rete li aggiusta solo
quando quella parola compare. Se compare in una pagina su mille, riceve una
correzione una volta su mille, ed è ragionevole che quando arriva sia grande. Il
difetto di Adagrad è che quel conto non si azzera mai:
passo dopo passo la falcata si accorcia, finché la discesa semplicemente si
ferma. **RMSProp** rimedia guardando solo al passato recente invece che
all'intera storia, così il passo non muore mai del tutto. **Adam** combina
questo passo adattivo con l'inerzia del momentum, e per questo è oggi la
scelta predefinita in gran parte delle reti.

`````

`````{tab} Superiore

Con momentum si accumula una media mobile dei gradienti $\mathbf{g}_t$ (che
sono vettori, uno per parametro, e come tali vanno in grassetto; $\theta$ resta
tondo, come tutte le greche dei parametri):

$$
\mathbf{v}_t = \beta\,\mathbf{v}_{t-1} + (1-\beta)\,\mathbf{g}_t,
\qquad
\theta_t = \theta_{t-1} - \eta\,\mathbf{v}_t,
$$

tipicamente $\beta = 0{,}9$.[^momentum-pytorch] **Adagrad** {cite}`duchi2011adaptive` normalizza
per la scala di ciascun parametro sommando i quadrati di tutti i gradienti
visti finora:

$$
\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2,
\qquad
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t} + \epsilon}\,\mathbf{g}_t,
$$

dove le operazioni sono elemento per elemento: ogni parametro riceve un
learning rate effettivo $\eta/(\sqrt{\mathbf{s}_t}+\epsilon)$ tutto suo. La
normalizzazione premia le feature sparse (i parametri aggiornati di rado
conservano passi ampi) ma $\mathbf{s}_t$ cresce monotonicamente, quindi il
passo effettivo tende a zero e prima o poi l'addestramento si arena.
**RMSProp** {cite}`tieleman2012rmsprop` rimedia sostituendo la somma con una
media mobile esponenziale, che dimentica il passato remoto:

$$
\mathbf{s}_t = \rho\,\mathbf{s}_{t-1} + (1-\rho)\,\mathbf{g}_t^2,
\qquad
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t}+\epsilon}\,\mathbf{g}_t,
$$

dove $\rho \in (0,1)$ è il coefficiente della media mobile: quanto più è
vicino a uno, tanto più lungo è il passato che $\mathbf{s}_t$ tiene in conto,
e tanto più lentamente il passo si riadatta. La formulazione originale usa
$0{,}9$; `torch.optim.RMSprop` chiama questo coefficiente `alpha` e lo lascia
a $0{,}99$.

Sulla stessa idea, **Adadelta** {cite}`zeiler2012adadelta` accumula una media
mobile anche degli aggiornamenti, eliminando di fatto la scelta di $\eta$.
**Adam** unisce momentum e passo adattivo (i coefficienti delle due medie
mobili si ribattezzano $\beta_1$ e $\beta_2$) con correzione del bias iniziale
$\hat{\mathbf{v}}_t = \mathbf{v}_t/(1-\beta_1^t)$ e
$\hat{\mathbf{s}}_t = \mathbf{s}_t/(1-\beta_2^t)$:

$$
\theta_t = \theta_{t-1}
- \eta\,\frac{\hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t}+\epsilon}.
$$

I default $\beta_1=0{,}9$, $\beta_2=0{,}999$, $\epsilon=10^{-8}$ funzionano in
un'enorme varietà di casi.[^adam-convergenza]

`````

C'è un punto in cui gli ottimizzatori incontrano la lotta contro
l'overfitting: il parametro `weight_decay` di `torch.optim`, che serve a
tenere i pesi piccoli. L'idea è che una rete costretta a lavorare con numeri
modesti non può affidarsi a pochi valori enormi per imparare a memoria. Il
nome però promette una cosa e il codice ne fa un'altra, e conviene disfare
l'equivoco subito, perché è quello da cui nasce AdamW.

`````{tab} Elementare

Tenere piccoli i pesi si può fare in due modi, che sembrano lo stesso e non lo
sono. Il primo è aggiungere al conto dell'errore una **multa** proporzionale a
quanto i pesi sono grandi: la rete, cercando di pagare meno multa, li tiene
bassi da sé. Il secondo è più diretto: a ogni passo si accorciano un pochino
tutti i pesi, e basta.

Con la discesa più spoglia che ci sia, quella senza inerzia, i due modi
finiscono per fare la stessa cosa. Basta però aggiungere l'inerzia (il
*momentum*, la pallina che rotola) e non è più vero: la multa entra nella
spinta accumulata e continua a farsi sentire nei passi successivi, invece di
esaurirsi in quello in cui è stata data. E l'inerzia c'è quasi sempre.

Si misura in due righe. Prendi un peso che vale $1$, mettilo in un punto dove il
terreno è perfettamente piatto (così l'unica cosa che lo muove è la multa) e
fagli fare quaranta passi. Senza inerzia i due modi lo lasciano tutti e due a
$0{,}818$: identici, come promesso. Con l'inerzia il primo lo porta a $0{,}061$
e il secondo resta a $0{,}818$, tredici volte più in alto. Nessuno ha cambiato
l'importo della multa: è cambiato solo che adesso si accumula.

Con Adam la differenza è di un altro tipo ancora, perché Adam ridimensiona ogni
correzione in base a quanto quel peso è stato corretto di recente, e insieme
alla correzione ridimensiona anche la multa. Risultato: la multa arriva forte
su certi pesi e debole su altri, e non perché qualcuno l'abbia deciso.
**AdamW** tiene le due cose separate, accorcia i pesi per conto suo senza
passare dalla bilancia di Adam, ed è per questo che è diventato la scelta
abituale.

`````

`````{tab} Superiore

Il parametro `weight_decay` di `torch.optim` (in `SGD` come in `Adam`) non
implementa il decadimento dei pesi: somma al gradiente il termine
$\lambda\theta$, cioè la **regolarizzazione L2** (una penalità nella loss
proporzionale al quadrato dei pesi). A gradiente nullo l'aggiornamento diventa
$\theta \leftarrow \theta(1-\eta\lambda)$: il peso si accorcia, ma la
sforbiciata passa dal learning rate.

Questa forma coincide con il decadimento vero solo nel caso più spoglio, la
discesa **senza momentum**. Basta accendere il momentum e le due divergono,
perché il termine $\lambda\theta$ entra nel buffer della velocità e si accumula
come farebbe un gradiente qualunque: la penalità non agisce più una volta per
passo, ma con la coda di tutti i passi precedenti.

Il conto, col protocollo perché si possa rifare: un parametro solo che parte da
$\theta_0 = 1$, gradiente identicamente nullo (così si misura la sola penalità),
$\eta = 0{,}1$, $\lambda = 0{,}05$, quaranta passi di `torch.optim.SGD`. Senza
momentum, `weight_decay=0.05` lascia $0{,}8183$, che è esattamente
$(1-\eta\lambda)^{40}$, e il decadimento applicato a mano fuori
dall'ottimizzatore lascia lo stesso identico $0{,}8183$. Con `momentum=0.9` il
decadimento a mano non cambia di una cifra, resta $0{,}8183$, mentre
`weight_decay` porta il peso a $0{,}0606$: **tredici volte più piccolo**, senza
che nessuno abbia toccato $\lambda$. È una differenza che conta, perché l'SGD
che si usa davvero è quello con il momentum.

Con i passi adattivi di Adam la questione cambia natura, e non nel senso che il
decadimento si indebolisce: il termine $\lambda\theta$ finisce **dentro** la
normalizzazione, quindi i parametri con gradienti tipicamente grandi vengono
regolarizzati **meno** e quelli con gradienti piccoli **di più**. Il difetto è
la disomogeneità, non la debolezza. **AdamW** {cite}`loshchilov2019decoupled`
lo *disaccoppia* dall'aggiornamento adattivo, applicandolo direttamente ai
pesi.

Su *come* lo applichi conviene essere precisi, perché l'articolo e il codice
non dicono la stessa cosa. Nell'articolo il decadimento è $\theta \leftarrow
\theta(1-\lambda)$, indipendente dal learning rate; `torch.optim.AdamW` esegue
invece `param.mul_(1 - lr * weight_decay)`, cioè $\theta \leftarrow
\theta(1-\eta\lambda)$, **lo stesso identico fattore** dell'L2 di SGD. Il conto:
con $\eta=0{,}5$, $\lambda=0{,}1$ e gradiente nullo, un passo porta un peso da
$1{,}0$ a $0{,}95$ sia con `SGD(weight_decay=0.1)` sia con `AdamW`; con `Adam`
lo porta a $0{,}50$. Quello, e non un fattore dieci fra le due formule, è ciò
che AdamW corregge: non la scala del decadimento, ma il fatto che passi dalla
bilancia adattiva.

`````

AdamW è oggi la scelta abituale per addestrare i Transformer, la famiglia di
reti a cui il libro dedica un capitolo più avanti, e in PyTorch si usa
esattamente come Adam:
`optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)`.

## Regolare il passo nel tempo

Un ultimo dettaglio spesso decisivo: il **learning rate** $\eta$, cioè la
lunghezza del passo con cui si correggono i pesi, non deve restare costante.

```{figure} ../figures/learning-rate.svg
:name: fig-learning-rate
:alt: "Animazione: tre parabole affiancate con lo stesso punto di partenza. Con un passo piccolo il punto striscia lungo il fianco senza arrivare al minimo; con quello giusto ci arriva in pochi passi; con quello troppo grande rimbalza da una parete all'altra allontanandosi."
:width: 90%

Sei passi di discesa lungo la parabola $f(x)=x^2$, sempre a partire dallo
stesso punto, con tre lunghezze di passo diverse. Nelle scritte, la lettera
greca $\eta$ («eta») è la lunghezza del passo e $f'(x)$ è la pendenza del
terreno nel punto in cui ci si trova. Il passo governa tutto: troppo corto e
non si arriva, troppo lungo e si scappa.
```

Il terzo pannello non è una licenza grafica: dietro c'è un conto esatto, e sta
in una riga. Sulla parabola del disegno ogni passo moltiplica la distanza dal
minimo per $1-2\eta$. Con $\eta = 0{,}4$ quel fattore vale $0{,}2$: a ogni passo
la distanza si riduce a un quinto, e in sei passi non resta quasi niente. Con
$\eta = 1{,}05$ vale $-1{,}1$, e il segno meno dice solo che si finisce
dall'altra parte del minimo; quello che conta è che in valore assoluto sia
maggiore di uno, perché vuol dire che a ogni passo si atterra più lontano di
dove si era partiti. Oltre una certa lunghezza il passo non rallenta la discesa,
la fa scappare.

Quella lunghezza dipende da quanto la conca è ripida e stretta, e le conche non
sono tutte uguali. Camminando, la rete passa da un tratto di paesaggio a un
altro, e il passo tarato sul primo è quello sbagliato sul secondo: non è il
paesaggio che si muove, è che se ne sta attraversando un pezzo nuovo.

`````{tab} Elementare

Il passo giusto non è lo stesso all'inizio e alla fine. All'inizio conviene
grande: la rete è lontana da qualunque soluzione decente e serve coprire
strada. Alla fine conviene piccolo, altrimenti si continua a scavalcare il
punto in cui ci si voleva fermare, come chi cerca di infilare la chiave nella
toppa muovendo la mano dieci centimetri per volta.

La ricetta che regola il passo mentre l'addestramento procede si chiama
**schedule** (è l'inglese per "programma"). Ce ne sono diverse e si assomigliano
tutte: ridurre il passo un po’ a ogni passata sui dati, dimezzarlo a scalini
ogni tot, oppure farlo scendere lungo una curva morbida che parte quasi piatta,
cala in fretta a metà strada e si riappiattisce alla fine (quella curva è il
coseno, ed è la scelta più comune oggi).

`````

`````{tab} Superiore

Su $f(x)=x^2$
l'aggiornamento è $x \leftarrow x(1-2\eta)$, quindi $x_k = x_0\,(1-2\eta)^k$:
si converge se e solo se $|1-2\eta| < 1$, cioè $0 < \eta < 1$. Il terzo
pannello usa $\eta = 1{,}05$: fattore $-1{,}1$, e ogni passo scavalca il
minimo più lontano del precedente. Su una funzione qualunque la soglia dipende
dalla curvatura, ed è proprio questo che uno schedule insegue mentre la
curvatura cambia. Un passo grande all'inizio esplora in fretta; lo stesso passo
verso la fine fa oscillare attorno al minimo senza mai stabilizzarsi. Il
**learning rate schedule** riduce progressivamente $\eta$: per esempio con
decadimento inverso $\eta_t = \eta_0/(1+kt)$, a gradini, o con andamento a
coseno.

`````

In PyTorch gli *scheduler* vivono accanto all'ottimizzatore e si aggiornano
dentro il ciclo di addestramento. Quello qui sotto è di un quarto tipo rispetto
ai tre appena elencati: invece di seguire una curva decisa in partenza, tiene
d'occhio l'errore su un gruppo di esempi messi da parte apposta (la
**validazione**, che serve a misurare la rete su dati che non ha usato per
imparare) e dimezza il passo quando quell'errore smette di scendere.

```{code-block} python
:class: pt-non-eseguibile

from torch import nn, optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)   # passo iniziale

# dimezza il learning rate quando la loss di validazione smette di scendere
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 factor=0.5, patience=3)

for epoca in range(50):        # un'«epoca» è una passata su tutti i dati
    addestra_una_epoca(model, train_loader, criterion, optimizer)
    loss_val = valuta(model, val_loader, criterion)   # loss di validazione
    scheduler.step(loss_val)                          # decide se ridurre il passo
```

C'è però un pezzo dello schedule che non sta alla fine ma **all'inizio**, e che
si incontra in ogni ricetta di addestramento moderna: il **warmup**. Invece di
partire subito da $\eta_0$, si sale da (quasi) zero fino a $\eta_0$ nell'arco
dei primi qualche centinaio o migliaio di passi, e solo dopo comincia il
decadimento.

`````{tab} Elementare

Sembra un capriccio, e ha una ragione precisa. Gli ottimizzatori moderni non
usano il gradiente grezzo: lo confrontano con una media di quelli visti finora,
per capire quanto quel gradiente sia affidabile e quanto grande fare il passo.
Ma **all'inizio quella media è fatta di due o tre numeri**, quindi è rumorosa,
e la stima può risultare sballata di parecchio.

Il guaio è che i primi passi sono anche i più pericolosi: la rete è ancora
disordinata, e un passo troppo lungo in una direzione sbagliata può portarla in
una regione da cui non si riprende: i neuroni finiti tutti nel tratto piatto
della propria curva, dove non reagiscono più a niente, oppure i pesi diventati
enormi. Partire piano è un modo di **non prendere decisioni importanti mentre
si è ignoranti**: si fanno passetti finché le statistiche non si assestano, e
poi si va.

`````

`````{tab} Superiore

Adam normalizza il gradiente per la radice della stima del secondo momento,
$\hat{\mathbf{s}}_t$ nella notazione fissata qui sopra (Kingma e Ba, e con loro
buona parte della letteratura, chiamano $\mathbf{m}_t$ il primo momento e
$\mathbf{v}_t$ il secondo). Nei primi passi quella stima è calcolata su
pochissimi campioni ed è ad alta varianza, quindi il rapporto
$\hat{\mathbf{v}}_t/\sqrt{\hat{\mathbf{s}}_t}$ può
assumere valori molto più grandi del previsto: il passo effettivo è
enormemente più variabile di $\eta$. La correzione del bias di Adam sistema la
media ma non la **varianza** della stima: è la diagnosi proposta da Liu e
colleghi {cite}`liu2020radam`, che leggono il warmup come un riduttore di
varianza nella fase iniziale. Non è l'ultima parola. Ma e Yarats
{cite}`ma2021adequacy` la contestano e riconducono il fenomeno alla dimensione
del passo, e il fatto che il warmup serva anche a SGD con momentum, che di
stime adattive non ne ha, dice che di una spiegazione sola non si tratta.

Si somma a due fattori che agiscono nella stessa direzione. Con batch grandi il
learning rate viene scalato verso l'alto (la regola lineare di
{cite}`goyal2017accurate`), e quel valore alto è
proprio ciò che all'inizio si vuole evitare. E nei Transformer *post-LN* la
norma è dopo il blocco residuo, il che produce gradienti molto grandi negli
strati alti a inizio addestramento {cite}`xiong2020layer`; è il motivo per cui
la ricetta originale prevedeva warmup obbligatorio, e per cui l'adozione del
*pre-LN* lo ha reso meno critico ma non inutile.

La forma standard è lineare crescente per $T_w$ passi e poi coseno decrescente:

$$
\eta_t = \begin{cases}
\eta_0 \dfrac{t}{T_w}, & t \le T_w,\\[2ex]
\eta_{\min} + \dfrac{\eta_0-\eta_{\min}}{2}
\left(1 + \cos\dfrac{\pi (t-T_w)}{T-T_w}\right), & t > T_w,
\end{cases}
$$

dove $t$ è il passo di addestramento, $T$ il numero totale di passi previsti,
$T_w$ la durata del warmup, $\eta_0$ il learning rate di picco (quello che si
tocca esattamente alla fine del warmup) ed $\eta_{\min}$ il valore su cui il
coseno si appoggia a fine corsa, spesso zero. Le due righe si saldano senza
scalini: in $t = T_w$ la prima dà $\eta_0$ e la seconda pure, perché
$\cos 0 = 1$; in $t = T$ resta $\eta_{\min}$, perché $\cos \pi = -1$. Ed è
quella che si trova, con nomi diversi, in quasi ogni configurazione di
addestramento su larga scala.

`````

Messe in fila, queste tecniche non sono un elenco di trucchi indipendenti:
rispondono a due domande sole. La prima è come far arrivare un segnale sensato
dall'uscita fino ai primi strati, e riguarda l'inizializzazione, la scelta
dell'attivazione e la batch normalization. La seconda è come camminare in quel
paesaggio una volta ricevuto il segnale: gli ottimizzatori, la lunghezza del
passo e il modo in cui cambia nel tempo. Il dropout e la multa sui pesi grandi
stanno un po’ di traverso rispetto a entrambe, perché non servono a far
imparare la rete ma a impedirle di imparare *troppo* quello che ha davanti; e
compaiono qui perché in pratica si montano nello stesso punto, dentro lo stesso
ciclo di addestramento.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Il segnale di correzione che torna indietro **si assottiglia o esplode**,
  perché attraversando gli strati viene moltiplicato tante volte: se ogni
  strato lo riduce di quattro volte, dopo cinque strati ne resta un millesimo.
- Partire con i pesi della **scala giusta** (e non è quella che la libreria
  mette da sé: va bene per le reti corte e affonda quelle lunghe), rimettere in
  riga i numeri a ogni strato (**batch normalization**) e spegnere neuroni a
  caso (**dropout**) sono i tre accorgimenti che rendono l'addestramento
  stabile e la rete meno incline a imparare a memoria.
- **Adam** è il punto di partenza sensato: mette insieme l'inerzia della
  pallina che rotola e un passo su misura per ogni peso. **AdamW** se si
  vogliono anche tenere piccoli i pesi.
- La lunghezza del passo non resta la stessa per tutto l'addestramento: prima
  **sale** da quasi zero (è il *warmup*: non si prendono decisioni importanti
  mentre si è ignoranti), poi **cala** man mano che ci si avvicina, di solito
  lungo la curva del coseno.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- I gradienti **svaniscono o esplodono** perché la backpropagation moltiplica
  tanti fattori: profondità e attivazioni saturanti sono i colpevoli.
- **Inizializzazione** giusta (He per ReLU, Glorot per tanh) e scelta a mano,
  perché il default di PyTorch è un sesto di He e i suoi bias non sono nulli;
  **batch normalization**, che in una CNN normalizza per canale, e **dropout**
  rendono l'addestramento stabile e generalizzabile; perché la batch
  normalization funzioni è però ancora una questione aperta.
- **Adam** (momentum + passo adattivo) è il punto di partenza sensato,
  **AdamW** se si usa il weight decay (che in `torch.optim` è una penalità L2,
  la quale coincide col decadimento vero solo per l'SGD senza momentum); un
  **learning rate schedule** che decade nel tempo rifinisce la convergenza.
- Lo schedule comincia però **salendo**: il **warmup** porta il learning rate
  da quasi zero a $\eta_0$ nei primi passi, quando le statistiche dei
  momenti di Adam sono stimate su pochissimi campioni e il passo effettivo ha
  varianza altissima, proprio mentre la rete è più fragile. Poi si decade, di
  norma a coseno.
```
`````

[^momentum-pytorch]: Attenzione a trasferire la formula nel codice:
    `torch.optim.SGD` usa la convenzione classica
    $\mathbf{v}_t = \beta\,\mathbf{v}_{t-1} + \mathbf{g}_t$,
    senza il fattore $(1-\beta)$. La sua $\mathbf{v}_t$ è quindi $1/(1-\beta)$ volte la
    nostra, e le due forme producono la stessa traiettoria solo passando a
    `lr` il valore $\eta\,(1-\beta)$, cioè **dividendo** il learning rate per
    $10$ quando $\beta=0{,}9$: a parità di learning rate, il passo di PyTorch
    è dieci volte più lungo. Qui adottiamo la media mobile esponenziale perché è la
    stessa che ritroveremo tra poco in Adam.

[^adam-convergenza]: Sulle garanzie teoriche conviene essere espliciti, perché
    i default che funzionano non sono un teorema. La dimostrazione di
    convergenza dell'articolo originale di Adam è **errata**: Reddi, Kale e
    Kumar {cite}`reddi2018convergence` esibiscono un problema convesso in una
    sola variabile su cui Adam converge al punto peggiore del dominio invece
    che all'ottimo. Il rimedio che propongono, AMSGrad, tiene il massimo
    storico del secondo momento ed è disponibile come
    `torch.optim.Adam(amsgrad=True)`; in pratica però Adam resta il default,
    cioè funziona senza che si sappia dimostrare che debba.
