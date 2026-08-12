# Far funzionare le reti profonde

Per molto tempo una rete con tanti strati è stata più un'idea che una pratica.
Negli anni '90 e nei primi 2000 impilare più livelli spesso *peggiorava* le
cose: la loss (il numero che misura quanto la rete sbaglia) non scendeva,
l'addestramento si arenava dopo poche passate sui dati. Non
era solo questione di potenza di calcolo. Mancavano gli accorgimenti che
rendono stabile l'apprendimento quando la rete è profonda. Tra il 2010 e il
2015 una manciata di idee, inizializzazioni pensate meglio, la *batch
normalization* {cite}`ioffe2015batch`, il *dropout*
{cite}`srivastava2014dropout`, gli optimizer adattivi come Adam
{cite}`kingma2015adam`: hanno trasformato le reti profonde da promessa fragile
a strumento affidabile. Questo capitolo le mette in fila: prima il problema,
poi i rimedi.

## Quando il segnale svanisce (o esplode)

La *backpropagation* calcola il gradiente della loss strato per strato,
partendo dall'uscita e risalendo verso l'ingresso. A ogni passo indietro il
segnale viene moltiplicato per i pesi di quello strato e per le **derivate**
delle attivazioni: la derivata è il numero che dice quanto l'uscita di un
neurone reagisce a una piccola variazione del suo ingresso, ed è piccola dove
il neurone è pigro, cioè dove muovere l'ingresso non cambia quasi niente. Ed è
qui che nasce il guaio.

```{figure} ../figures/gradiente-svanisce.svg
:name: fig-gradiente-svanisce
:alt: "Animazione: due file di barre, sigmoide e ReLU. Risalendo dall'uscita verso l'ingresso le barre della sigmoide si accorciano di un fattore quattro per strato, quelle della ReLU restano intere."
:width: 90%

Quanta parte del segnale di correzione sopravvive al passaggio di ogni strato,
con due attivazioni diverse. Con la sigmoide ogni strato lo moltiplica per un
numero che non supera mai $0{,}25$, cioè lo divide almeno per quattro: dal
sesto strato al primo, che sono cinque passaggi, resta un millesimo scarso di
quello di partenza ($0{,}25^5 = 0{,}00098$, l'ultima barra). Con la ReLU quel
numero vale $1$ sulla parte attiva, e il prodotto non si consuma. *(Le barre
sono disegnate in scala logaritmica, cioè ogni tacca vale dieci volte la
precedente: in scala normale le ultime non si vedrebbero, che poi è il punto.)*
```

I numeri della {numref}`fig-gradiente-svanisce` sono il caso **migliore** per
la sigmoide: $0{,}25$ è il massimo della sua derivata, raggiunto solo in
$z=0$. Lontano dall'origine è molto più piccolo, e il crollo è più rapido.

`````{tab} Elementare

Immagina una catena di ingranaggi in cui ogni ruota trasmette solo una
frazione del movimento alla successiva: diciamo un decimo. Dopo dieci ruote,
del movimento iniziale non resta quasi nulla: un decimo di miliardesimo. È ciò
che succede al segnale di correzione che, dall'uscita della rete, deve tornare
fino ai primi strati: attraversando molti livelli si assottiglia fino a
sparire. Gli strati vicini all'ingresso non ricevono quasi nessuna indicazione
su come cambiare, e di fatto smettono di imparare. Il difetto opposto è pari e
contrario: se ogni ruota *amplifica* il movimento, dopo pochi passi il segnale
esplode in numeri enormi e l'addestramento va in tilt.

`````

`````{tab} Superiore

Il gradiente rispetto ai pesi di uno strato profondo è un prodotto di molti
fattori. Schematicamente, per una rete di $L$ strati, il gradiente che
raggiunge lo strato $\ell$ contiene un termine

$$
\prod_{k=\ell}^{L-1} \mathbf{W}_k^\top \,
\operatorname{diag}\!\big(\sigma'(\mathbf{z}_k)\big),
$$

dove $\mathbf{W}_k$ è la matrice dei pesi e $\sigma'(\mathbf{z}_k)$ la derivata
dell'attivazione. Se i fattori hanno modulo tipico minore di $1$, il prodotto
tende a $0$ esponenzialmente in $L$ (**vanishing gradient**); se maggiore di
$1$, diverge (**exploding gradient**). La sigmoide aggrava il primo caso:
$\sigma'(z)\le 0{,}25$ ovunque, quindi il solo fattore di attivazione riduce
il gradiente di almeno quattro volte a ogni strato. Rimedi complementari:
attivazioni non saturanti come la ReLU ($\sigma'=1$ per input positivi),
*gradient clipping* per l'esplosione, e (soprattutto) una scelta accurata
della scala iniziale dei pesi.

`````

## Partire col piede giusto: l'inizializzazione

Se il prodotto di tanti fattori decide se il segnale svanisce o esplode, il
punto di partenza conta enormemente. Inizializzare i pesi con la scala
sbagliata condanna la rete prima ancora del primo aggiornamento.

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
del primo aggiornamento è ciò che rende l'inizializzazione un problema a sé.
Non è una scorciatoia per convergere prima: con la scala sbagliata la rete non
converge affatto, perché il gradiente che dovrebbe correggerla è già
degenerato al primo passaggio.

`````{tab} Elementare

L'idea è tenere costante il "volume" del segnale (l'ampiezza che la curva del
disegno qui sopra chiama varianza) mentre attraversa gli strati:
né più forte né più debole. Se un neurone somma tanti ingressi, i suoi pesi
iniziali devono essere piccoli in proporzione, così che la somma non diventi
troppo grande o troppo piccola. Due ricette calibrano questa scala in base a
quanti ingressi ha ciascun neurone: **Xavier/Glorot**, pensata per attivazioni
simmetriche come la tanh, e **He**, tarata per la ReLU. Sono l'impostazione
predefinita nei framework moderni: raramente serve toccarle a mano.

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

`````

## Normalizzare mentre si impara: la batch normalization

Anche partendo bene, i numeri che circolano dentro la rete cambiano scala di
continuo mentre si impara: ogni aggiornamento modifica ciò che arriva allo
strato successivo. La *batch normalization* interviene qui, rimettendo in riga
le attivazioni di ogni strato, e in pratica rende l'addestramento molto più
rapido e stabile. Conviene dirlo subito: **perché funzioni così bene è ancora
materia di discussione**, e la spiegazione data dai suoi autori non ha retto
alla prova dei fatti.

```{figure} ../figures/batch-normalization-2015.svg
:name: fig-batch-norm
:alt: "A sinistra tre distribuzioni delle attivazioni provenienti da batch diversi, spostate l'una rispetto all'altra e di larghezza diversa. Al centro una stazione di batch normalization sottrae la media e divide per la deviazione standard, poi riapplica i parametri appresi gamma e beta. A destra le tre distribuzioni escono centrate in zero e con la stessa dispersione."
:width: 96%

Da tre distribuzioni che vagano a tre distribuzioni sovrapposte. Le due
manopole in coda, chiamate $\gamma$ (gamma) e $\beta$ (beta), servono a
restituire alla rete la libertà che la normalizzazione le ha appena tolto: la
prima riallarga o restringe i numeri, la seconda li sposta in su o in giù.
```

La coda di {numref}`fig-batch-norm` è la parte che spesso si salta e che
invece conta. Normalizzare e basta imporrebbe a ogni strato un'ampiezza decisa
da noi; $\gamma$ e $\beta$ sono due numeri che la rete impara come tutti gli
altri, e le permettono di riallargare e spostare il risultato se le conviene,
anche fino ad annullare del tutto la normalizzazione.

`````{tab} Elementare

A ogni strato ricentriamo le attivazioni perché abbiano media zero e ampiezza
regolare, calcolate sul mini-batch corrente. È come rimettere in scala i
numeri a ogni passo, così che nessuno strato debba adattarsi a input che
cambiano scala di continuo. In pratica accelera molto l'addestramento,
permette learning rate (il passo di correzione dei pesi) più aggressivi e ha
un lieve effetto di regolarizzazione (cioè frena l'imparare a memoria), perché
la statistica del batch introduce un po' di rumore utile.

`````

`````{tab} Superiore

Per ogni attivazione, sul mini-batch $\mathcal{B}$ si calcolano media
$\mu_{\mathcal{B}}$ e varianza $\sigma_{\mathcal{B}}^2$, si normalizza e si
riscala con due parametri appresi $\gamma$ (scala) e $\beta$ (shift):

$$
\hat{x} = \frac{x - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}},
\qquad
y = \gamma\,\hat{x} + \beta.
$$

Il termine $\epsilon$ evita la divisione per zero; $\gamma$ e $\beta$
restituiscono alla rete la libertà di rappresentare qualsiasi scala, inclusa
l'identità. In inferenza si usano media e varianza mobili accumulate durante
il training, non le statistiche del singolo batch. Ioffe e Szegedy la
introdussero per contrastare l'*internal covariate shift*, cioè lo spostarsi
della distribuzione degli input di ogni strato durante l'addestramento, ma
quella spiegazione è stata contestata: Santurkar e colleghi
{cite}`santurkar2018batchnorm` mostrano che la stabilità distribuzionale
c'entra poco con il successo della batch normalization (si può iniettare
rumore *dopo* la normalizzazione, aumentando lo shift, senza perderne i
benefici) e propongono che l'effetto vero sia un panorama della loss più
liscio e percorribile. Anche questa resta un'ipotesi: il meccanismo per cui la
BN funziona è tuttora aperto.

`````

## Spegnere neuroni a caso: il dropout

La batch normalization regolarizza un po' come effetto collaterale. Il
dropout lo fa per scelta esplicita, ed è uno dei modi più semplici per
combattere l'*overfitting*.

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
implicito, cioè come una squadra di reti al posto di una sola: con $n$ neuroni
nascosti le combinazioni possibili di acceso e spento sono $2^n$, ogni passo di
addestramento ne allena una presa a caso, e tutte quelle reti condividono gli
stessi pesi, quindi allenarne una fa progredire anche le altre.

## Scendere bene: gli optimizer moderni

Tutti questi accorgimenti stabilizzano il *segnale*; resta da decidere *come*
muoversi nel panorama della loss. Gli algoritmi che lo decidono si chiamano
**optimizer**, e in italiano *ottimizzatori*: le due parole vogliono dire la
stessa cosa e nel libro si alternano. La discesa del gradiente pura fa un passo
proporzionale al gradiente e basta. In una valle stretta e allungata questo
significa rimbalzare da parete a parete invece di scivolare verso il fondo
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

Il **momentum** dà alla discesa un po' di inerzia, come una pallina che rotola
in una valle: accumula velocità nella direzione giusta e si lascia dietro i
rimbalzi laterali. **Adagrad** aggiunge un'idea in più: dare a ogni parametro
un passo su misura, più corto dove il terreno è ripido e più lungo dove è
piatto. Per farlo tiene il conto di tutta la strada già percorsa: i parametri
corretti di continuo rallentano, quelli toccati di rado (capita spesso quando
molti ingressi sono quasi sempre zero, come le parole rare in un testo),
conservano passi generosi. Il difetto è che quel conto non si azzera mai:
passo dopo passo la falcata si accorcia, finché la discesa semplicemente si
ferma. **RMSProp** rimedia guardando solo al passato recente invece che
all'intera storia, così il passo non muore mai del tutto. **Adam** combina
questo passo adattivo con l'inerzia del momentum, e per questo è oggi la
scelta predefinita in gran parte delle reti.

`````

`````{tab} Superiore

Con momentum si accumula una media mobile dei gradienti $g_t$:

$$
v_t = \beta\,v_{t-1} + (1-\beta)\,g_t,
\qquad
\theta_t = \theta_{t-1} - \eta\,v_t,
$$

tipicamente $\beta = 0{,}9$.[^momentum-pytorch] **Adagrad** {cite}`duchi2011adaptive` normalizza
per la scala di ciascun parametro sommando i quadrati di tutti i gradienti
visti finora:

$$
G_t = G_{t-1} + g_t^2,
\qquad
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{G_t} + \epsilon}\,g_t,
$$

dove le operazioni sono elemento per elemento: ogni parametro riceve un
learning rate effettivo $\eta/(\sqrt{G_t}+\epsilon)$ tutto suo. La
normalizzazione premia le feature sparse (i parametri aggiornati di rado
conservano passi ampi) ma $G_t$ cresce monotonicamente, quindi il passo
effettivo tende a zero e prima o poi l'addestramento si arena. **RMSProp**
{cite}`tieleman2012rmsprop` rimedia sostituendo la somma con una media mobile
esponenziale, che dimentica il passato remoto:

$$
s_t = \rho\,s_{t-1} + (1-\rho)\,g_t^2,
\qquad
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{s_t}+\epsilon}\,g_t.
$$

Sulla stessa idea, **Adadelta** {cite}`zeiler2012adadelta` accumula una media
mobile anche degli aggiornamenti, eliminando di fatto la scelta di $\eta$.
**Adam** unisce momentum e passo adattivo (i coefficienti delle due medie
mobili si ribattezzano $\beta_1$ e $\beta_2$) con correzione del bias iniziale
$\hat{v}_t = v_t/(1-\beta_1^t)$ e $\hat{s}_t = s_t/(1-\beta_2^t)$:

$$
\theta_t = \theta_{t-1}
- \eta\,\frac{\hat{v}_t}{\sqrt{\hat{s}_t}+\epsilon}.
$$

I default $\beta_1=0{,}9$, $\beta_2=0{,}999$, $\epsilon=10^{-8}$ funzionano in
un'enorme varietà di casi.[^adam-convergenza]

`````

C'è un punto in cui gli ottimizzatori incontrano la lotta contro
l'overfitting: il parametro `weight_decay` di `torch.optim`, che serve a
tenere i pesi piccoli. L'idea è che una rete costretta a lavorare con numeri
modesti non può affidarsi a pochi valori enormi per imparare a memoria. Il
nome però promette una cosa e il codice ne fa un'altra, e vale la pena
disfare l'equivoco subito, perché è quello da cui nasce AdamW.

`````{tab} Elementare

Tenere piccoli i pesi si può fare in due modi, che sembrano lo stesso e non lo
sono. Il primo è aggiungere al conto dell'errore una **multa** proporzionale a
quanto i pesi sono grandi: la rete, cercando di pagare meno multa, li tiene
bassi da sé. Il secondo è più diretto: a ogni passo si accorciano un pochino
tutti i pesi, e basta.

Con la discesa semplice i due modi finiscono per fare la stessa cosa. Con Adam
no, perché Adam ridimensiona ogni correzione in base a quanto quel peso è
stato corretto di recente, e insieme alla correzione ridimensiona anche la
multa. Risultato: la multa arriva forte su certi pesi e debole su altri, e non
perché qualcuno l'abbia deciso. **AdamW** tiene le due cose separate, accorcia
i pesi per conto suo senza passare dalla bilancia di Adam, ed è per questo che
è diventato la scelta abituale.

`````

`````{tab} Superiore

Il parametro `weight_decay` di `torch.optim` (in `SGD` come in `Adam`) non
implementa il decadimento dei pesi: somma al gradiente il termine
$\lambda\theta$, cioè la **regolarizzazione L2** (una penalità nella loss
proporzionale al quadrato dei pesi). Per la discesa semplice le due forme
coincidono a meno del learning rate: L2 dà
$\theta \leftarrow \theta(1-\eta\lambda)$, il decadimento disaccoppiato
$\theta \leftarrow \theta(1-\lambda)$, e con $\eta = 0{,}1$ è un fattore dieci
a parità di `weight_decay`.

Con i passi adattivi di Adam l'equivalenza si rompe, e non nel senso che il
decadimento si indebolisce: il termine $\lambda\theta$ finisce **dentro** la
normalizzazione, quindi i parametri con gradienti tipicamente grandi vengono
regolarizzati **meno** e quelli con gradienti piccoli **di più**. Il difetto è
la disomogeneità, non la debolezza. **AdamW** {cite}`loshchilov2019decoupled`
lo *disaccoppia* dall'aggiornamento adattivo, applicandolo direttamente ai
pesi.

`````

AdamW è oggi il default de facto per addestrare i Transformer, e in PyTorch si
usa esattamente come Adam:
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

Il terzo pannello non è una licenza grafica: dietro c'è un conto esatto, e dice
che oltre una certa lunghezza il passo non rallenta la discesa, la fa scappare.
Quella lunghezza dipende da quanto è ripida e stretta la valle, e siccome la
valle cambia forma man mano che si scende, il passo giusto oggi è quello
sbagliato domani.

`````{tab} Elementare

Il passo giusto non è lo stesso all'inizio e alla fine. All'inizio conviene
grande: la rete è lontana da qualunque soluzione decente e serve coprire
strada. Alla fine conviene piccolo, altrimenti si continua a scavalcare il
punto in cui ci si voleva fermare, come chi cerca di infilare la chiave nella
toppa muovendo la mano dieci centimetri per volta.

La ricetta che regola il passo mentre l'addestramento procede si chiama
**schedule** (è l'inglese per "programma"). Ce ne sono diverse e si assomigliano
tutte: ridurre il passo un po' a ogni passata sui dati, dimezzarlo a scalini
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

In PyTorch gli *scheduler* vivono accanto all'ottimizzatore e si aggiornano nel
training loop:

```{code-block} python
:class: pt-non-eseguibile

from torch import nn, optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)   # passo iniziale

# dimezza il learning rate quando la loss di validazione smette di scendere
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 factor=0.5, patience=3)

for epoca in range(50):
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
una regione da cui non si riprende, con le attivazioni saturate o i pesi troppo
grandi. Partire piano è un modo di **non prendere decisioni importanti mentre
si è ignoranti**: si fanno passetti finché le statistiche non si assestano, e
poi si va.

`````

`````{tab} Superiore

Adam normalizza il gradiente per la radice della stima del secondo momento,
$\hat{s}_t$ nella notazione fissata qui sopra (Kingma e Ba, e con loro buona
parte della letteratura, chiamano $m_t$ il primo momento e $v_t$ il secondo).
Nei primi passi quella stima è calcolata su pochissimi campioni ed
è ad alta varianza, quindi il rapporto $\hat{v}_t/\sqrt{\hat{s}_t}$ può
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

ed è quella che si trova, con nomi diversi, in quasi ogni configurazione di
addestramento su larga scala.

`````

Messe in fila, queste tecniche non sono un elenco di trucchi indipendenti:
rispondono tutte alla stessa domanda, cioè come far arrivare un segnale sensato
dall'uscita fino ai primi strati, e come muoversi una volta arrivati senza
rovinare quello che si è capito.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Il segnale di correzione che torna indietro **si assottiglia o esplode**,
  perché attraversando gli strati viene moltiplicato tante volte: se ogni
  strato lo riduce di quattro volte, dopo cinque strati ne resta un millesimo.
- Partire con i pesi della **scala giusta**, rimettere in riga i numeri a ogni
  strato (**batch normalization**) e spegnere neuroni a caso (**dropout**)
  sono i tre accorgimenti che rendono l'addestramento stabile e la rete meno
  incline a imparare a memoria.
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
- **Inizializzazione** giusta (He per ReLU, Glorot per tanh), **batch
  normalization** e **dropout** rendono l'addestramento stabile e
  generalizzabile; perché la batch normalization funzioni è però ancora una
  questione aperta.
- **Adam** (momentum + passo adattivo) è il punto di partenza sensato,
  **AdamW** se si usa il weight decay (che in `torch.optim` è una penalità L2,
  non un decadimento disaccoppiato); un **learning rate schedule** che
  decade nel tempo rifinisce la convergenza.
- Lo schedule comincia però **salendo**: il **warmup** porta il learning rate
  da quasi zero a $\eta_0$ nei primi passi, quando le statistiche dei
  momenti di Adam sono stimate su pochissimi campioni e il passo effettivo ha
  varianza altissima, proprio mentre la rete è più fragile. Poi si decade, di
  norma a coseno.
```
`````

[^momentum-pytorch]: Attenzione a trasferire la formula nel codice:
    `torch.optim.SGD` usa la convenzione classica $v_t = \beta\,v_{t-1} + g_t$,
    senza il fattore $(1-\beta)$. La sua $v_t$ è quindi $1/(1-\beta)$ volte la
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
