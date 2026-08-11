# Far funzionare le reti profonde

Per molto tempo una rete con tanti strati è stata più un'idea che una pratica.
Negli anni '90 e nei primi 2000 impilare più livelli spesso *peggiorava* le
cose: la loss non scendeva, l'addestramento si arenava dopo poche epoche. Non
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
gradiente viene moltiplicato per i pesi e per le derivate delle attivazioni.
Ed è qui che nasce il guaio.

```{figure} ../figures/gradiente-svanisce.svg
:name: fig-gradiente-svanisce
:alt: "Animazione: due file di barre, sigmoide e ReLU. Risalendo dall'uscita verso l'ingresso le barre della sigmoide si accorciano di un fattore quattro per strato, quelle della ReLU restano intere."
:width: 90%

Il fattore che sopravvive al passaggio di ogni strato. Con la sigmoide, la cui
derivata non supera mai $0{,}25$, dopo sei strati resta meno di un millesimo
del gradiente di partenza; con la ReLU, che sulla parte attiva ha derivata
$1$, il prodotto non si consuma. *(Le altezze sono in scala logaritmica: in
scala lineare le ultime barre non si vedrebbero, che poi è il punto.)*
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
\prod_{k=\ell}^{L-1} W_k^\top \, \operatorname{diag}\!\big(\sigma'(z_k)\big),
$$

dove $W_k$ è la matrice dei pesi e $\sigma'(z_k)$ la derivata
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

Tre inizializzazioni, tre destini, e nessun addestramento ancora avvenuto. La
curva piatta è l'obiettivo: la varianza del segnale deve attraversare la rete
senza gonfiarsi né spegnersi.
```

Il fatto che le tre curve di {numref}`fig-inizializzazione` divergano *prima*
del primo aggiornamento è ciò che rende l'inizializzazione un problema a sé.
Non è una scorciatoia per convergere prima: con la scala sbagliata la rete non
converge affatto, perché il gradiente che dovrebbe correggerla è già
degenerato al primo passaggio.

`````{tab} Elementare

L'idea è tenere costante il "volume" del segnale mentre attraversa gli strati:
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
l'inizializzazione di **Glorot** (2010) campiona i pesi con varianza

$$
\operatorname{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}},
$$

adatta ad attivazioni simmetriche attorno a zero (tanh). Per la ReLU, che
azzera metà degli ingressi, **He** (2015) raddoppia la scala usando solo il
fan-in:

$$
\operatorname{Var}(W) = \frac{2}{n_{\text{in}}}.
$$

In entrambi i casi $W$ si estrae da una normale (o da una uniforme con
supporto equivalente) e i bias si pongono a $0$. La regola pratica: **He** con
ReLU e varianti, **Glorot** con tanh e sigmoide.

`````

## Normalizzare mentre si impara: la batch normalization

Anche partendo bene, durante l'addestramento la distribuzione delle
attivazioni si sposta di continuo: ogni aggiornamento cambia gli input dello
strato successivo, che deve inseguire un bersaglio mobile. La *batch
normalization* stabilizza questo bersaglio.

```{figure} ../figures/batch-normalization-2015.svg
:name: fig-batch-norm
:alt: "A sinistra tre distribuzioni delle attivazioni provenienti da batch diversi, spostate l'una rispetto all'altra e di larghezza diversa. Al centro una stazione di batch normalization sottrae la media e divide per la deviazione standard, poi riapplica i parametri appresi gamma e beta. A destra le tre distribuzioni escono centrate in zero e con la stessa dispersione."
:width: 96%

Da tre distribuzioni che vagano a tre distribuzioni sovrapposte. I parametri
$\gamma$ e $\beta$ in coda servono a restituire alla rete la libertà che la
normalizzazione le ha appena tolto.
```

La coda di {numref}`fig-batch-norm` è la parte che spesso si salta e che
invece conta. Normalizzare e basta imporrebbe a ogni strato una distribuzione
decisa da noi; $\gamma$ e $\beta$ sono addestrabili, e permettono alla rete di
spostare e riscalare il risultato se le conviene, anche fino ad annullare la
normalizzazione.

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
introdussero per contrastare l'*internal covariate shift*; l'effetto pratico è
un panorama della loss più liscio e percorribile
{cite}`santurkar2018batchnorm`.

`````

## Spegnere neuroni a caso: il dropout

La batch normalization regolarizza un po' come effetto collaterale. Il
dropout lo fa per scelta esplicita, ed è uno dei modi più semplici per
combattere l'*overfitting*.

```{figure} ../figures/dropout.svg
:name: fig-dropout
:alt: "Animazione: una rete con due strati nascosti; a ogni mini-batch una metà diversa dei neuroni nascosti si spegne insieme alle sue connessioni, mentre input e output restano sempre attivi."
:width: 90%

Quattro mini-batch consecutivi con $p = 0{,}5$: ogni volta la rete che viene
davvero addestrata è **un'altra**. Input e output non si spengono mai.
```

Guardando la {numref}`fig-dropout` si capisce anche perché il dropout venga
descritto come un *ensemble implicito*: una rete con $n$ neuroni nascosti
nasconde $2^n$ sottoreti, e l'addestramento ne campiona una a ogni passo,
condividendo i pesi fra tutte.

`````{tab} Elementare

Durante l'addestramento, a ogni passo, spegniamo a caso una frazione dei
neuroni. La rete non può più affidarsi a un singolo neurone "specialista":
deve distribuire la conoscenza, perché quel neurone potrebbe non esserci al
prossimo giro. È come allenare ogni volta una squadra leggermente diversa: il
risultato è un modello più robusto, che generalizza meglio su dati nuovi. A
inferenza tutti i neuroni tornano attivi.

`````

`````{tab} Superiore

Con probabilità di spegnimento $p$, la convenzione di `nn.Dropout(p)` in
PyTorch, si applica alle attivazioni una maschera binaria
$m \sim \text{Bernoulli}(1-p)$:

$$
\tilde{h} = \frac{1}{1-p}\,(m \odot h),
$$

dove $\odot$ è il prodotto elemento per elemento. Il fattore $1/(1-p)$
(*inverted dropout*) mantiene invariato il valore atteso, così a inferenza si
usa la rete piena senza riscalature. Interpretazione: il dropout addestra
implicitamente un *ensemble* esponenziale di sotto-reti che condividono i
pesi. Valori tipici: $p \in [0{,}2,\ 0{,}5]$. Non si combina bene con la
batch normalization sullo stesso strato: spesso si sceglie l'una o l'altro.

`````

## Scendere bene: gli optimizer moderni

Tutti questi accorgimenti stabilizzano il *segnale*; resta da decidere *come*
muoversi nel panorama della loss. La discesa del gradiente pura fa un passo
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
rimedia sostituendo la somma con una media mobile esponenziale, che dimentica
il passato remoto:

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
un'enorme varietà di casi.

`````

C'è un punto in cui gli ottimizzatori incontrano la lotta contro
l'overfitting. Il parametro `weight_decay` di `torch.optim` applica il
**decadimento dei pesi**: a ogni aggiornamento i pesi vengono leggermente
riportati verso lo zero, così la rete non può affidarsi a valori enormi per
imparare a memoria. È la stessa cosa della **regolarizzazione L2** (aggiungere
alla loss una penalità proporzionale al quadrato dei pesi) almeno finché la
discesa è quella semplice. Con i passi adattivi di Adam, però, l'equivalenza
si rompe: il decadimento viene riscalato insieme al gradiente e perde parte
dell'effetto. **AdamW** {cite}`loshchilov2019decoupled` lo *disaccoppia*
dall'aggiornamento adattivo, applicandolo direttamente ai pesi, ed è oggi il
default de facto per addestrare i Transformer:
`optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)`.

## Regolare il passo nel tempo

Un ultimo dettaglio spesso decisivo: il learning rate $\eta$ non deve restare
costante.

```{figure} ../figures/learning-rate.svg
:name: fig-learning-rate
:alt: "Animazione: tre parabole affiancate con lo stesso punto di partenza. Con un passo piccolo il punto striscia lungo il fianco senza arrivare al minimo; con quello giusto ci arriva in pochi passi; con quello troppo grande rimbalza da una parete all'altra allontanandosi."
:width: 90%

Sei passi di discesa su $f(x)=x^2$ a partire dallo stesso punto. Il passo
governa tutto: troppo corto e non si arriva, troppo lungo e si scappa.
```

La {numref}`fig-learning-rate` ha anche un conto esatto dietro. Su $f(x)=x^2$
l'aggiornamento è $x \leftarrow x(1-2\eta)$, quindi $x_k = x_0\,(1-2\eta)^k$:
si converge se e solo se $|1-2\eta| < 1$, cioè $0 < \eta < 1$. Il terzo
pannello usa $\eta = 1{,}05$: fattore $-1{,}1$, e ogni passo scavalca il
minimo più lontano del precedente. Non è una licenza grafica: è la
disuguaglianza che si rompe. Su una funzione qualunque la soglia dipende dalla
curvatura, ed è proprio questo che uno schedule insegue mentre la curvatura
cambia. Un passo grande all'inizio esplora in fretta; lo stesso passo verso la
fine fa oscillare attorno al minimo senza mai stabilizzarsi. Il **learning
rate schedule** riduce progressivamente $\eta$: per esempio con decadimento
inverso $\eta_t = \eta_0/(1+kt)$, a gradini, o con andamento a coseno. In
PyTorch gli *scheduler* vivono accanto all'ottimizzatore e si aggiornano nel
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
$\hat{v}_t$. Nei primi passi quella stima è calcolata su pochissimi campioni ed
è ad alta varianza, quindi il rapporto $\hat{m}_t/\sqrt{\hat{v}_t}$ può
assumere valori molto più grandi del previsto: il passo effettivo è
enormemente più variabile di $\eta$. La correzione del bias di Adam sistema la
media ma non la **varianza** della stima, ed è questa la diagnosi che motiva il
warmup come riduttore di varianza nella fase iniziale.

Si somma a due fattori che agiscono nella stessa direzione. Con batch grandi il
learning rate viene scalato verso l'alto (regola lineare), e quel valore alto è
proprio ciò che all'inizio si vuole evitare. E nei Transformer *post-LN* la
norma è dopo il blocco residuo, il che produce gradienti molto grandi negli
strati alti a inizio addestramento; è il motivo per cui la ricetta originale
prevedeva warmup obbligatorio, e per cui l'adozione del *pre-LN* lo ha reso
meno critico ma non inutile.

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

```{admonition} Da ricordare
:class: important
- I gradienti **svaniscono o esplodono** perché la backpropagation moltiplica
  tanti fattori: profondità e attivazioni saturanti sono i colpevoli.
- **Inizializzazione** giusta (He per ReLU, Glorot per tanh), **batch
  normalization** e **dropout** rendono l'addestramento stabile e
  generalizzabile.
- **Adam** (momentum + passo adattivo) è il punto di partenza sensato,
  **AdamW** se si usa il weight decay; un **learning rate schedule** che
  decade nel tempo rifinisce la convergenza.
- Lo schedule comincia però **salendo**: il **warmup** porta il learning rate
  da quasi zero a $\eta_0$ nei primi passi, perché lì le statistiche dei
  momenti di Adam sono stimate su pochissimi campioni e il passo effettivo ha
  varianza altissima, proprio quando la rete è più fragile. Poi si decade, di
  norma a coseno.
```

[^momentum-pytorch]: Attenzione a trasferire la formula nel codice:
    `torch.optim.SGD` usa la convenzione classica $v_t = \beta\,v_{t-1} + g_t$,
    senza il fattore $(1-\beta)$. La sua $v_t$ è quindi $1/(1-\beta)$ volte la
    nostra, e le due forme producono la stessa traiettoria solo passando a
    `lr` il valore $\eta\,(1-\beta)$, cioè **dividendo** il learning rate per
    $10$ quando $\beta=0{,}9$: a parità di learning rate, il passo di PyTorch
    è dieci volte più lungo. Qui adottiamo la media mobile esponenziale perché è la
    stessa che ritroveremo tra poco in Adam.
