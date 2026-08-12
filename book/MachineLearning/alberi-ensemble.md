# Alberi decisionali e metodi ensemble

C'è un gioco da tavolo, *Indovina chi?*, in cui si scopre il personaggio
misterioso dell'avversario a forza di domande sì/no: «Porta gli occhiali?»,
«Ha i capelli neri?». Ogni risposta abbatte a metà i sospetti, finché non ne
resta uno solo. È esattamente così che ragiona un **albero decisionale**: una
catena di domande sulle caratteristiche di un esempio, ciascuna scelta per
dividere i casi nel modo più netto possibile, fino a una risposta.

Nelle sezioni precedenti abbiamo incontrato modelli che separano i dati con un
taglio **dritto**: la regressione lineare traccia una retta che segue i punti,
la logistica una retta che li divide, e con più colonne quella retta diventa
l'equivalente in più dimensioni di un piano (il nome tecnico, che tornerà nella
sezione sulle SVM, è *iperpiano*). Gli alberi appartengono a
un'altra famiglia, e sono i re incontrastati di un terreno preciso: i **dati
tabellari**, quelli a righe e colonne di un foglio di calcolo, dove ogni
colonna è una caratteristica di natura diversa (un'età, un reddito, una
categoria). Su questo terreno gli algoritmi che vedremo (foreste casuali e
gradient boosting) restano, ancora oggi, difficili da battere.

C'è poi una ragione in più per studiarli: sono **interpretabili**. Un albero
si può leggere, stampare, seguire domanda per domanda. È la differenza tra un
modello «scatola bianca» (*white box*), di cui capiamo la logica, e una
«scatola nera» (*black box*) come una grande rete neurale, che dà la risposta
giusta senza dirci perché. Quando la decisione conta (un prestito negato, una
diagnosi), poter spiegare il *perché* non è un lusso.

## L'albero decisionale: dividere per domande

L'algoritmo classico per costruire questi alberi si chiama **CART**
(*Classification And Regression Trees*), introdotto nel 1984 da Leo Breiman e
colleghi {cite}`breiman1984classification`. L'idea è cercare, a ogni nodo, la
domanda che separa meglio i dati, e ripeterla, ricorsivamente, su ciascuno dei
due gruppi che ne risultano.

L'albero è fatto di **nodi**, e ogni nodo è una domanda: si parte da quello in
cima (la **radice**), si scende a destra o a sinistra secondo la risposta, e si
finisce in un nodo che non ha più domande sotto di sé (una **foglia**), dove
sta la risposta finale. Che poi la stessa procedura si applichi identica a ogni
sottogruppo che si forma, all'infinito finché c'è qualcosa da dividere, è ciò
che si intende dicendo che l'albero si costruisce **ricorsivamente**.

Ogni domanda è una soglia su una caratteristica: «reddito < 25 000 €?», «età <
30?». Una risposta manda l'esempio a sinistra, l'altra a destra. Ricordando che
ogni colonna è una direzione e ogni esempio un punto, questo taglia lo **spazio
delle caratteristiche** (il foglio su cui abbiamo disegnato i punti) in
**rettangoli** con i lati paralleli agli assi
({numref}`fig-albero-decisionale`): una domanda sul reddito è una riga
orizzontale, una sull'età una riga verticale, e non c'è modo di ottenere un
taglio in diagonale. Ogni foglia dell'albero
è una di quelle regioni, e a tutti i punti che vi cadono l'albero assegna la
stessa risposta.

```{figure} ../figures/albero-decisionale.svg
:name: fig-albero-decisionale
:alt: "A sinistra un piano cartesiano con asse orizzontale «età» e asse verticale «reddito», partizionato da due tagli ortogonali agli assi in tre regioni rettangolari colorate; i punti di due classi cadono ciascuno nella propria regione. A destra lo schema ad albero corrispondente: il nodo radice chiede «età minore di 30?», i rami portano a un secondo nodo che chiede «reddito minore di 25 mila?» e a tre foglie colorate come le regioni del piano."
:width: 100%

Un albero decisionale partiziona lo spazio in rettangoli. A sinistra il piano
età–reddito tagliato da due split; a destra l'albero corrispondente. Ogni
foglia colorata è una regione del piano, e a tutta la regione l'albero assegna
la stessa classe.
```

Ma cosa vuol dire, in numeri, «separare meglio»? Serve una misura di quanto un
gruppo è **impuro**, cioè mescolato tra classi diverse. Un gruppo tutto di una
classe è puro (impurità zero); un gruppo metà e metà è il più impuro possibile.
La domanda migliore è quella che, dopo lo split, lascia i due gruppi il più
puri possibile.

`````{tab} Elementare

La misura più usata è l'**indice di Gini**: la probabilità di sbagliare se
tirassimo a indovinare la classe di un esempio pescando a caso dal gruppo, con
le stesse proporzioni del gruppo. Un gruppo puro non ci fa mai sbagliare (Gini
= 0); un gruppo bilanciato ci fa sbagliare spesso (Gini alto).

Da quella frase alla formula ci si arriva in due righe, e vale la pena farle
perché così il Gini smette di essere una regola calata dall'alto. Pesca due
volte dal gruppo, rimettendo dentro: se in quel gruppo la classe «compra» è una
frazione $p$, la probabilità di pescarla due volte di fila è $p \cdot p = p^2$.
Sommando i quadrati di tutte le classi ottieni la probabilità di pescare **due
volte la stessa** classe, cioè di indovinare; e siccome le due possibilità
esauriscono i casi, la probabilità di pescarne due **diverse**, cioè di
sbagliare, è $1$ meno quella somma. Ecco da dove viene l'«uno meno la somma dei
quadrati».

Facciamo i conti su un esempio. Un negozio online ha 10 clienti, e vogliamo
prevedere chi comprerà: 5 comprano (sì), 5 no. Il gruppo è metà e metà, il più
mescolato possibile:

$$
\text{Gini}_\text{padre} = 1 - \left(\tfrac{5}{10}\right)^2 - \left(\tfrac{5}{10}\right)^2
= 1 - 0{,}25 - 0{,}25 = 0{,}5 .
$$

Proviamo a dividerli con la domanda «ha visitato il sito almeno 3 volte?».
I 4 che rispondono «sì» comprano tutti e 4; dei 6 che rispondono «no», solo 1
compra e 5 no. Calcoliamo l'impurità dei due gruppi figli:

$$
\text{Gini}_\text{sì} = 1 - \left(\tfrac{4}{4}\right)^2 - \left(\tfrac{0}{4}\right)^2 = 0 ,
\qquad
\text{Gini}_\text{no} = 1 - \left(\tfrac{1}{6}\right)^2 - \left(\tfrac{5}{6}\right)^2
= 1 - \tfrac{26}{36} \approx 0{,}278 .
$$

Il gruppo dei «sì» è puro; quello dei «no» è quasi puro. L'impurità *dopo* lo
split è la media pesata sui due gruppi (4 clienti da una parte, 6 dall'altra):

$$
\tfrac{4}{10}\cdot 0 + \tfrac{6}{10}\cdot 0{,}278 \approx 0{,}167 .
$$

Il **guadagno** è quanto abbiamo ridotto l'impurità:
$0{,}5 - 0{,}167 = 0{,}333$. Un bel taglio! L'algoritmo prova tutte le domande
possibili su tutte le caratteristiche e sceglie quella dal guadagno più alto:
poi ricomincia su ciascun gruppo.

«Tutte le domande possibili» sembra impossibile, visto che le soglie su un
reddito sarebbero infinite: minore di 25 000, di 25 001, di 25 002... Ma le
soglie che cambiano davvero qualcosa sono poche, ed è facile vedere quali. Se
nei dati nessuno guadagna fra 24 000 e 26 000 euro, tutte le soglie in quel
buco dividono i clienti nello stesso identico modo: sono la stessa domanda
scritta in mille modi. Basta allora ordinare i valori che compaiono davvero nei
dati e provare una soglia in mezzo a ogni coppia di valori consecutivi: con
diecimila clienti sono al massimo novemilanovecentonovantanove prove per
colonna, non infinite.

`````

`````{tab} Superiore

Sia $p_k$ la frazione di esempi di classe $k$ in un nodo. Le due misure di
impurità classiche sono l'**indice di Gini** e l'**entropia**:

$$
G = 1 - \sum_{k=1}^{K} p_k^2 ,
\qquad
H = - \sum_{k=1}^{K} p_k \log_2 p_k ,
$$

dove $K$ è il numero di classi. Entrambe valgono $0$ su un nodo puro ($p_k = 1$
per una sola classe) e sono massime sulla distribuzione uniforme. La qualità di
uno split che manda una frazione $w_L$ degli esempi nel figlio sinistro e $w_R
= 1 - w_L$ nel destro si misura con l'**information gain**, la riduzione attesa
di impurità:

$$
\Delta = I_\text{padre} - \big(w_L\, I_L + w_R\, I_R\big) ,
$$

dove $I$ è l'impurità scelta (Gini o entropia) e $w_L, w_R$ pesano i figli per
la loro numerosità. CART sceglie, tra tutte le coppie (caratteristica, soglia),
quella che massimizza $\Delta$, e procede in modo ricorsivo e *greedy*: nessun
passo indietro, ogni split è ottimo solo localmente.

Riprendendo l'esempio numerico dell'altro livello con l'**entropia**: il nodo
padre bilanciato ha $H_\text{padre} = -\tfrac{1}{2}\log_2\tfrac{1}{2} -
\tfrac{1}{2}\log_2\tfrac{1}{2} = 1$ bit. Il figlio «sì» è puro ($H = 0$); il
figlio «no» ha

$$
H_\text{no} = -\tfrac{1}{6}\log_2\tfrac{1}{6} - \tfrac{5}{6}\log_2\tfrac{5}{6}
\approx 0{,}431 + 0{,}219 = 0{,}650 \text{ bit} .
$$

L'entropia media dopo lo split è $\tfrac{4}{10}\cdot 0 + \tfrac{6}{10}\cdot
0{,}650 = 0{,}390$ bit, e l'information gain vale $1 - 0{,}390 = 0{,}610$ bit.
Gini ed entropia danno in pratica alberi quasi identici; Gini è un po' più
veloce (niente logaritmi) ed è la scelta di default in scikit-learn.

`````

Per **predire**, un esempio nuovo scende lungo l'albero rispondendo alle
domande, fino a una foglia: la sua classe è quella di maggioranza tra gli
esempi di addestramento finiti in quella foglia. Lo stesso meccanismo serve la
**regressione**: basta cambiare cosa contiene la foglia e come si misura
l'impurità. La foglia non predice più una classe ma la **media** dei valori
$y$ degli esempi che vi cadono, e al posto di Gini si cerca lo split che rende
i valori dentro ciascun gruppo il più simili possibile alla loro media. Il modo
di misurarlo è lo stesso già usato per giudicare la retta di best fit (la
distanza fra valore vero e valore previsto, elevata al quadrato e mediata:
l'**errore quadratico medio**, o **MSE** dall'inglese *mean squared error*).
Qui però il modello, invece di
una retta, produce una funzione «a scalini», costante su ogni rettangolo.

## Il tallone d'Achille: alta varianza

Un albero lasciato crescere senza freni continua a dividere finché ogni foglia
contiene un solo esempio: a quel punto classifica alla perfezione i dati di
addestramento, e generalizza malissimo. È l'**overfitting** che abbiamo
studiato nella sezione sull'overfitting e la validazione, nella sua forma più
estrema.

`````{tab} Elementare

Un albero profondo è come lo studente che impara a memoria: costruisce una
domanda su misura per ogni singolo esempio, rumore compreso. Il problema è che
è anche **instabile**. Cambia appena qualche dato di addestramento (togline
dieci, aggiungine altri dieci) e l'albero può risultare completamente diverso:
uno split scelto in cima cambia, e tutto ciò che ci sta sotto cambia con lui.

Nella sezione sul compromesso bias-varianza avevamo dato un nome a questa
irrequietezza: si chiama **varianza**. Un singolo albero profondo ha bias basso
(sa adattarsi a qualsiasi forma) ma varianza alta (dipende troppo dal
particolare campione di dati). Ridurre quella varianza senza perdere la
flessibilità: è tutto il problema che gli *ensemble* risolvono.

`````

`````{tab} Superiore

Nel linguaggio del compromesso bias-varianza, un albero cresciuto a fondo è un
modello a **bias basso, varianza alta**: la sua espressività gli permette di
approssimare frontiere di decisione arbitrarie, ma la scelta greedy degli split
è estremamente sensibile alle fluttuazioni del campione. Piccole perturbazioni
dei dati si propagano dalla radice alle foglie, cambiando l'intera struttura.

Lo si può limitare con la **potatura** (*pruning*) o vincolando la crescita
(profondità massima, numero minimo di esempi per foglia) ma questi freni
scambiano varianza con bias, e un solo albero raramente compete con i modelli
migliori. La strada vincente è un'altra: tenere alberi flessibili (bias basso)
e abbattere la varianza **combinandone molti**. È il principio degli ensemble,
oggetto del resto della sezione.

`````

## Ensemble: la saggezza della folla

Chiedi a una sola persona quante caramelle ci sono nel barattolo e sbaglierà di
parecchio. Chiedilo a mille persone e fai la media delle risposte: il numero
sarà sorprendentemente vicino al vero. Gli errori individuali, se indipendenti,
tendono a elidersi. È la **saggezza della folla**, e i metodi *ensemble* la
mettono al lavoro: invece di un modello solo, ne addestrano molti e ne
combinano le risposte.

```{figure} ../figures/ensemble-modelli-deboli.svg
:name: fig-voto-di-maggioranza
:alt: "Cinque modelli deboli, ciascuno appena migliore del caso, ricevono lo stesso esempio e votano; alcuni sbagliano, ma i loro errori cadono su risposte diverse mentre i corretti convergono sulla stessa. Il voto di maggioranza produce la risposta giusta."
:width: 92%

Perché il voto funzioni servono errori *diversi*. I tre che azzeccano
concordano; i due che sbagliano sbagliano in due modi differenti, e da soli
non fanno maggioranza.
```

La condizione nascosta in {numref}`fig-voto-di-maggioranza` è quella che tutto
il resto della sezione cerca di ottenere. Se i cinque modelli sbagliassero
sugli stessi esempi e nello stesso modo, la media non correggerebbe niente:
riprodurrebbe l'errore comune con più sicurezza di prima.

Ci sono due strategie profondamente diverse per farlo, e conviene tenerle
distinte fin da subito ({numref}`fig-bagging-vs-boosting`): il **bagging**
addestra i modelli *in parallelo*, indipendenti l'uno dall'altro, e ne fa la
media; il **boosting** li addestra *in sequenza*, ognuno per correggere gli
errori del precedente.

```{figure} ../figures/bagging-vs-boosting.svg
:name: fig-bagging-vs-boosting
:alt: "Due schemi affiancati. A sinistra il bagging: un dataset genera tre campioni bootstrap diversi, ciascuno addestra in parallelo un proprio albero; le tre risposte confluiscono in un blocco di voto o media che produce la predizione finale. A destra il boosting: tre alberi piccoli disposti in sequenza da sinistra a destra, ciascuno collegato al successivo da una freccia etichettata «residui», e tutti e tre confluiscono in una somma pesata che dà la predizione finale."
:width: 100%

Le due grandi famiglie di ensemble. Nel bagging (sinistra) gli alberi sono
addestrati in parallelo su campioni diversi e votano alla pari. Nel boosting
(destra) sono addestrati in sequenza, ognuno sugli errori del precedente, e
sommati con pesi.
```

### Bagging: mediare per ridurre la varianza

Il **bagging** (da *bootstrap aggregating*, proposto da Breiman nel 1996
{cite}`breiman1996bagging`) attacca direttamente il problema della varianza
degli alberi. Il trucco è generare tanti dataset di addestramento leggermente
diversi a partire da uno solo, e su ciascuno addestrare un albero.

`````{tab} Elementare

Come si ottengono dataset diversi avendone uno solo? Con il **bootstrap**: si
pesca a caso dal dataset, *rimettendo* ogni volta l'esempio pescato nel
mucchio. Così alcuni esempi capitano più volte, altri restano fuori, e ogni
campione è una versione un po' diversa dell'originale, come rifare la spesa
prendendo a caso dagli scaffali: la lista somiglia sempre a sé stessa, ma non
è mai identica.

Su ognuno di questi campioni si addestra un albero. Ne escono, poniamo, 100
alberi tutti diversi. Per classificare un esempio nuovo, li si interpella tutti
e si fa **votare** a maggioranza (o, in regressione, la media delle loro
risposte). Il singolo albero è nervoso e sbaglia in modo imprevedibile; ma se
gli errori dei 100 alberi non sono tutti uguali, mediando si annullano a
vicenda, e la risposta collettiva è molto più stabile. Il bias resta quello di
un albero (basso), la varianza crolla.

`````

`````{tab} Superiore

Perché mediare abbassa la varianza si vede con un conto. Siano
$\hat{f}_1, \dots, \hat{f}_B$ le predizioni di $B$ alberi, ciascuna con varianza
$\sigma^2$ e correlazione a due a due $\rho$. La varianza della loro media è

$$
\operatorname{Var}\!\left(\frac{1}{B}\sum_{b=1}^{B}\hat{f}_b\right)
= \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2 ,
$$

dove $\sigma^2$ è la varianza del singolo modello e $\rho$ la correlazione tra
due modelli distinti. Il secondo termine svanisce all'aumentare di $B$: con
molti alberi resta solo $\rho\,\sigma^2$. La media riduce dunque la varianza
tanto più quanto i modelli sono **decorrelati** (cioè quanto $\rho$ è
piccolo). Qui sta il limite del bagging puro: alberi addestrati su campioni
bootstrap dello stesso dataset restano abbastanza correlati; se una
caratteristica è molto predittiva, quasi tutti gli alberi la scelgono in cima
e finiscono per somigliarsi. È il collo di bottiglia che la foresta casuale
rimuove.

`````

## Random Forest: decorrelare gli alberi

La **foresta casuale** (*random forest*), sempre di Breiman, nel 2001
{cite}`breiman2001random`, aggiunge al bagging una seconda dose di casualità,
mirata proprio ad abbassare quella correlazione $\rho$ che frena il bagging.

```{figure} ../figures/random-forest.svg
:name: fig-foresta-voto
:alt: "Molti alberi di decisione affiancati, ciascuno cresciuto su un campione diverso dei dati e su un sottoinsieme diverso delle feature; ognuno emette la propria predizione, e le predizioni confluiscono in un voto di maggioranza che produce il verdetto finale."
:width: 96%

La foresta al lavoro. La diversità qui non è un caso fortunato: è costruita
apposta, dando a ogni albero dati diversi e feature diverse fra cui scegliere.
```

Il secondo sorteggio illustrato in {numref}`fig-foresta-voto`, quello sulle
feature, è il contributo specifico di Breiman. Senza, tutti gli alberi
sceglierebbero per prima la stessa colonna dominante e si somiglierebbero
troppo; togliendogliela a turno, sono costretti a scoprire strade diverse.

`````{tab} Elementare

L'idea è tanto semplice quanto efficace: a ogni split, invece di lasciar
scegliere all'albero la domanda migliore tra *tutte* le caratteristiche,
gliene mostriamo solo un sottoinsieme casuale (poche, estratte a caso ogni
volta). Se la caratteristica dominante non è tra quelle proposte, l'albero è
costretto a guardare altrove.

È come chiedere a una giuria di esperti di votare, ma bendando ogni giurato su
aspetti diversi del caso: nessuno può basarsi sempre sull'indizio più ovvio, e i
loro pareri diventano davvero indipendenti. Alberi più diversi tra loro, media
più efficace, varianza ancora più bassa. Il singolo albero diventa un po' meno
bravo (gli abbiamo nascosto delle carte), ma l'insieme diventa molto più forte.

`````

`````{tab} Superiore

A ogni nodo, la ricerca dello split migliore è ristretta a un sottoinsieme
casuale di $m$ caratteristiche estratte dalle $n$ totali (una scelta comune è
$m = \sqrt{n}$ per la classificazione, $m = n/3$ per la regressione). Questo
vincolo abbassa la correlazione $\rho$ tra gli alberi: nella formula della
varianza della media, è esattamente la leva che fa scendere il termine
dominante $\rho\,\sigma^2$. Si accetta un lieve aumento del bias e della
varianza del singolo albero in cambio di una riduzione netta della varianza
dell'ensemble. Una variante ancora più aggressiva, gli **Extra-Trees**
(*Extremely Randomized Trees*), estrae a caso anche le soglie di split invece
di ottimizzarle, guadagnando velocità e ulteriore decorrelazione.

`````

La foresta casuale porta in dote due strumenti pratici molto amati.

Il primo è l'**errore out-of-bag** (OOB). Ricordiamo che ogni albero è
addestrato su un campione bootstrap: si pesca $m$ volte da un mucchio di $m$
esempi, rimettendo dentro ogni volta. Un esempio preciso ha una probabilità di
$1 - 1/m$ di non essere pescato al primo colpo, e di scampare tutte e $m$ le
pescate ha probabilità $(1 - 1/m)^m$, che già con qualche centinaio di esempi
vale circa $0{,}37$ (per la precisione tende a $1/e$). Ecco perché **in media
circa un terzo** degli esempi resta *fuori* da ogni campione: sono gli esempi
*out-of-bag*. Per ciascun esempio
possiamo raccogliere il voto dei soli alberi che *non* l'hanno visto in
addestramento: è una stima dell'errore di generalizzazione ottenuta gratis,
senza mettere da parte un validation set separato.

Il secondo è la **feature importance**. Sommando, su tutti gli alberi, di
quanto ciascuna caratteristica ha ridotto l'impurità nei suoi split, si
ottiene una classifica di quanto ogni caratteristica «conta» per il modello. È
un'informazione preziosa per capire i dati, con un'avvertenza che scikit-learn
stessa segnala: questa misura tende a gonfiare l'importanza delle
caratteristiche con molti valori distinti, e va letta con prudenza (una stima
più affidabile è la *permutation importance*).

## Boosting: correggere gli errori, uno alla volta

Il **boosting** ribalta la logica del bagging. Invece di addestrare tanti
alberi forti in parallelo e mediarli, ne addestra molti **deboli** (alberi
minuscoli, spesso profondi appena uno o due livelli) ma **in sequenza**, dove
ognuno si concentra sugli errori commessi da chi lo precede. La somma di tanti
correttori mediocri, ciascuno che ripara un pezzetto, diventa un modello molto
accurato.

`````{tab} Elementare

Immagina uno studente che ripassa per un esame. Fa un primo giro di esercizi,
sbaglia alcuni tipi di problema. Al secondo giro si concentra proprio su quelli
che ha sbagliato. Al terzo, su ciò che ancora non gli riesce. Ogni ripasso non
riparte da zero: aggiusta il tiro là dove serve. Alla fine padroneggia
l'insieme, un errore corretto per volta.

Il primo algoritmo di questo tipo, **AdaBoost** (Freund e Schapire, 1997
{cite}`freund1997decision`), fa proprio così con dei **pesi**: dopo ogni albero,
gli esempi classificati male ricevono un peso maggiore, così l'albero successivo
è spinto a occuparsi soprattutto di loro. Gli alberi che nel complesso sbagliano
meno pesano di più nel voto finale. Il risultato è un comitato in cui ciascuno è
specializzato sui casi difficili lasciati aperti dai colleghi precedenti.

`````

`````{tab} Superiore

Il **gradient boosting** (Friedman, 2001 {cite}`friedman2001greedy`) generalizza
l'idea di AdaBoost e la inquadra come una **discesa del gradiente nello spazio
delle funzioni**. Il modello è additivo, costruito passo dopo passo:

$$
F_B(\mathbf{x}) = F_0 + \sum_{t=1}^{B} \nu\, h_t(\mathbf{x}) ,
$$

dove $B$ è il numero di alberi (la stessa lettera del bagging), $F_0$ è la
costante che da sola minimizza la loss sui dati (per la loss
quadratica, la media dei target), $h_t$ è l'albero aggiunto al passo $t$ e
$\nu \in (0,1]$ è il **learning rate**. A ogni passo si vorrebbe muovere la
funzione corrente $F_{t-1}$ nella direzione che riduce di più la loss
$\mathcal{L}$; quella direzione, valutata su ciascun esempio, è l'opposto del
gradiente

$$
r_i^{(t)} = -\left[\frac{\partial \mathcal{L}(y_i, F(\mathbf{x}_i))}
{\partial F(\mathbf{x}_i)}\right]_{F = F_{t-1}} ,
$$

detto **pseudo-residuo**. Il nuovo albero $h_t$ viene addestrato per
approssimare proprio questi pseudo-residui. Nel caso della loss quadratica
$\mathcal{L} = \tfrac{1}{2}(y - F)^2$ il gradiente si riduce a $r_i = y_i -
F_{t-1}(\mathbf{x}_i)$: cioè, semplicemente, l'**errore residuo** ancora da
spiegare (al primo passo, lo scarto dalla media $F_0$).
Detto a parole: ogni albero fitta ciò che i precedenti hanno sbagliato. AdaBoost
è il caso particolare che si ottiene scegliendo la *exponential loss*.

`````

Nel mondo reale, due implementazioni del gradient boosting dominano le
competizioni sui dati tabellari: **XGBoost** (Chen e Guestrin, 2016
{cite}`chen2016xgboost`) e **LightGBM** (Ke e colleghi, 2017
{cite}`ke2017lightgbm`). Non sono idee nuove, ma ingegnerizzazioni del gradient
boosting molto più veloci e robuste, e vale la pena sapere perché vincono:

- **Regolarizzazione esplicita**. XGBoost aggiunge alla loss una penalità sulla
  complessità di ogni albero (numero di foglie, ampiezza dei valori nelle
  foglie), nello spirito del rasoio di Occam già visto per Ridge e Lasso. Questo
  tiene a bada l'overfitting, il vero rischio del boosting.
- **Non solo la pendenza, anche la curvatura**. Per decidere dove tagliare,
  XGBoost non guarda soltanto in che direzione la loss cala (il gradiente) ma
  anche quanto in fretta quella pendenza sta cambiando: è come scendere dalla
  collina sapendo non solo che si scende, ma anche se il pendio sta per
  spianarsi. Tecnicamente è uno sviluppo di Taylor al secondo ordine, cioè
  l'uso della derivata seconda accanto alla prima, e serve a fare un passo più
  informato.
- **Istogrammi e velocità**. Entrambi raggruppano i valori continui delle
  caratteristiche in poche decine di intervalli (*bin*): trovare lo split
  migliore diventa scorrere un istogramma invece di ordinare tutti i valori.
  Entrambi, inoltre, sanno gestire da soli i **valori mancanti**, imparando per
  ogni split da che parte conviene mandare le righe con la casella vuota (in
  XGBoost è lo *sparsity-aware split finding* del paper del 2016). La
  differenza vera di LightGBM è un'altra: la crescita *leaf-wise*, cioè
  espandere sempre la foglia più promettente invece di completare un livello
  per volta, che è ciò che lo rende particolarmente rapido sui dataset grandi.

## Bagging o boosting? Varianza contro bias

Le due strategie curano mali opposti, e questo dice quando preferire l'una o
l'altra.

Il **bagging** (e la sua incarnazione migliore, la random forest) parte da
alberi a varianza alta e la abbatte mediando. È robusto, poco sensibile agli
iperparametri, difficile da mandare in overfitting: aggiungere alberi non
peggiora quasi mai. Ottima scelta di default, specie quando si vuole un modello
solido con poca messa a punto, e si parallelizza banalmente (gli alberi sono
indipendenti).

Il **boosting** parte da alberi deboli a bias alto e lo abbatte correggendo
gli errori in sequenza. Tipicamente raggiunge l'accuratezza più alta sui dati
tabellari, ma è più delicato: siccome ogni albero rincorre gli errori del
precedente, **può andare in overfitting** se lo si lascia correre troppo. I
due freni principali sono il **learning rate** (la lunghezza del passo, nelle
formule $\nu$): qui il passo non è quello della discesa del gradiente sui pesi
di una retta, ma quanto di ogni nuovo albero si somma al modello, e la logica è
la stessa vista con la collina nella nebbia. Passi piccoli
rendono l'apprendimento più lento ma più stabile, e di solito si abbina un
passo piccolo a molti alberi. Il secondo freno è l'**early stopping**, cioè
fermarsi quando l'errore
su un validation set smette di migliorare, come abbiamo visto nella sezione
sugli iperparametri. Il boosting inoltre è **sequenziale** per costruzione:
non si parallelizza sugli alberi come il bagging.

In sintesi: se cerchi robustezza con poco sforzo, parti dalla random forest; se
cerchi l'ultimo punto di accuratezza e sei disposto a mettere a punto learning
rate ed early stopping, passa al gradient boosting.

## Combinare modelli diversi: voto e stacking

Bagging e boosting combinano **molte copie dello stesso tipo di modello**.
Resta la domanda che si pone chiunque abbia provato tre algoritmi diversi e li
veda arrivare a punteggi simili: si possono mettere insieme *quelli*?

`````{tab} Elementare

Sì, e in due modi, che si distinguono per chi decide come pesare i pareri.

Il primo è il **voto**. Ogni modello dice la sua e vince la maggioranza. C'è
una variante che quasi sempre funziona meglio: invece di contare i voti secchi
si mediano le **probabilità**, così un modello sicurissimo pesa più di uno che
era incerto. Contare i voti butta via l'informazione più utile, cioè quanto
ciascuno ci credeva.

Il secondo è lo **stacking**, e l'idea è più ambiziosa: invece di decidere noi
come pesare i modelli, **si addestra un modello a farlo**. Sopra i predittori
di base si mette un ultimo modello, di solito semplicissimo, che riceve in
ingresso le loro predizioni e impara quando fidarsi di chi. Può scoprire che il
primo è affidabile sui casi facili e il secondo sui casi rari, cosa che una
media fissa non può fare.

C'è una regola che sembra un dettaglio tecnico ed è invece tutto il punto: il
combinatore va addestrato su **predizioni che i modelli di base non hanno mai
visto in addestramento**. Se gli si danno le predizioni sui dati con cui quei
modelli si sono allenati, lui vedrà tutti bravissimi, e si fiderà proprio di
chi ha imparato a memoria. È lo stesso principio del test che non si tocca,
applicato un piano più in su.

E la condizione perché tutto questo serva a qualcosa: i modelli devono
**sbagliare in modi diversi**. Tre modelli che sbagliano sugli stessi casi non
si correggono a vicenda, e combinarli non porta nulla. È la stessa ragione per
cui una random forest decorrela gli alberi invece di limitarsi a fare la media.

`````

`````{tab} Superiore

Il **voting** aggrega $M$ modelli eterogenei $f_1,\dots,f_M$. Nella forma
*hard* si prende la moda delle etichette predette; nella forma *soft* la media
(eventualmente pesata) delle probabilità,
$\hat{p}(y\mid \mathbf{x}) = \frac{1}{M}\sum_m \hat{p}_m(y\mid \mathbf{x})$,
seguita da un $\arg\max$. Il *soft voting* domina di norma perché conserva la
confidenza, che nel voto duro viene scartata: ma richiede probabilità
**comparabili** fra i modelli, e modelli mal calibrati possono peggiorarlo.

Lo **stacking** {cite}`wolpert1992stacked` sostituisce la regola fissa con un
**meta-modello** $g$ addestrato su $\hat{\mathbf{z}} = (f_1(\mathbf{x}), \dots,
f_M(\mathbf{x}))$. La regola critica è che le $\hat{\mathbf{z}}$ di
addestramento siano **fuori campione**: si genera una matrice di predizioni per
*cross-validation* (per ogni fold, i modelli di base sono addestrati sugli
altri fold e predicono su quello tenuto fuori), e su quella si addestra $g$.
Senza questa precauzione il meta-modello osserva le predizioni *in-sample* dei
modelli di base, che sono ottimisticamente buone in misura proporzionale a
quanto ciascuno sovradatta, e impara a pesare la memorizzazione.

Come meta-modello si sceglie tipicamente qualcosa di **semplice** (una
regressione logistica, spesso regolarizzata): la capacità serve sotto, non
sopra, e un combinatore flessibile sovradatta la matrice delle predizioni, che
ha poche colonne e forte collinearità.

La condizione di efficacia si legge nella scomposizione **ambiguità-errore** di
Krogh e Vedelsby {cite}`krogh1995neural`, che è un'identità esatta **per la
loss quadratica** sulla media dell'ensemble: l'errore della media è l'errore
medio dei membri meno la loro **diversità**,
$E_{\text{ens}} = \bar{E} - \bar{A}$ con $\bar{A} \ge 0$. Combinare aiuta nella
misura in cui i modelli sono decorrelati negli errori, e non aiuta affatto se
sono d'accordo anche quando sbagliano.

Due cautele, che il paragrafo qui sotto mette alla prova con i numeri. La
prima: l'identità **non regge tutti gli ensemble**, perché per la loss 0-1
(cioè per il voto di maggioranza) una scomposizione additiva analoga non
esiste, e gli effetti della diversità dipendono dalla distribuzione delle
etichette {cite}`wood2023unified`. La seconda, più insidiosa: da
$\bar{A}\ge0$ segue che l'ensemble non è mai peggiore del membro **medio**,
il che non dice nulla sul confronto con il membro **migliore**. Un ensemble
peggiore del suo componente più bravo non contraddice affatto Krogh e Vedelsby.

`````

Vale la pena vedere che cosa succede davvero, perché il risultato non è quello
che ci si aspetta. Nell'esperimento che segue i tre modelli di base sono una
foresta casuale, un k-NN e un terzo che non abbiamo ancora incontrato, il
**Bayes ingenuo**: un classificatore elementare che guarda le colonne **una per
una**, calcola per ciascuna quanto è probabile il valore osservato in ognuna
delle due classi, e poi moltiplica tutte quelle probabilità come se le colonne
fossero indipendenti fra loro. «Ingenuo» è proprio quell'ipotesi, quasi sempre
falsa (reddito e quartiere non sono indipendenti), ed è la ragione per cui qui
sarà nettamente il più debole dei tre. Il capitolo sul linguaggio naturale lo
riprende per esteso, dove invece funziona benissimo.

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import (RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

X, y = make_classification(n_samples=3000, n_features=20, n_informative=8,
                           class_sep=0.7, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

# tre modelli che sbagliano in modi DIVERSI: è questa la condizione
base = [("foresta", RandomForestClassifier(n_estimators=200, random_state=0)),
        ("vicini",  KNeighborsClassifier(n_neighbors=15)),
        ("bayes",   GaussianNB())]

for nome, m in base:
    print(f"{nome:<12} {m.fit(X_tr, y_tr).score(X_te, y_te):.4f}")

duro = VotingClassifier(base, voting="hard").fit(X_tr, y_tr)
morbido = VotingClassifier(base, voting="soft").fit(X_tr, y_tr)
# il combinatore si addestra su predizioni FUORI CAMPIONE (cv=5): senza,
# imparerebbe a fidarsi di chi ha memorizzato il training set
pila = StackingClassifier(base, final_estimator=LogisticRegression(),
                          cv=5).fit(X_tr, y_tr)

print(f"{'voto duro':<12} {duro.score(X_te, y_te):.4f}")
print(f"{'voto morbido':<12} {morbido.score(X_te, y_te):.4f}")
print(f"{'stacking':<12} {pila.score(X_te, y_te):.4f}")
```

I singoli arrivano a $0{,}8933$ (foresta), $0{,}8889$ (vicini) e $0{,}8156$
(Bayes ingenuo). Poi:

- il **voto duro** dà $0{,}8867$ e quello **morbido** $0{,}8822$: entrambi
  **peggio del miglior singolo modello**;
- lo **stacking** dà $0{,}9089$, cioè un punto e mezzo sopra il migliore.

La differenza ha un nome, ed è il terzo modello. Il Bayes ingenuo è nettamente
il più debole, e in una media a pesi fissi conta quanto gli altri: il voto lo
tratta alla pari con la foresta. Attenzione a che cosa questo *non* smentisce:
la media dei tre membri sta a $0{,}8659$, e il voto morbido la batte
($0{,}8822$), esattamente come l'identità ambiguità-errore promette. Quello che
l'identità non promette è di battere il **migliore** dei tre, e infatti non lo
batte. Lo stacking invece **impara** che di quel modello ci si può fidare poco,
e gli assegna un peso piccolo: è il vantaggio strutturale di far decidere i
pesi ai dati invece che fissarli a priori, e la ragione per cui, in un ensemble
eterogeneo, la media semplice è una scommessa sulla qualità uniforme dei
membri.

Non è però un invito a impilare tutto: il guadagno qui è di un punto e mezzo,
pagato con quattro modelli da addestrare e da mantenere, e una
cross-validation interna. In produzione quel conto va fatto.

## In pratica, con scikit-learn

L'interfaccia `fit`/`predict` è la stessa vista per gli altri modelli
supervisionati; per una guida applicativa estesa a questi metodi rimandiamo al
manuale di Géron {cite}`geron2022hands`. I quattro protagonisti di questa
sezione stanno in poche righe:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Un solo albero: interpretabile, ma ad alta varianza.
# max_depth frena la crescita per non memorizzare i dati.
albero = DecisionTreeClassifier(max_depth=4, criterion="gini")
albero.fit(X_train, y_train)

# Random forest: 300 alberi in parallelo, split su un sottoinsieme di feature.
# oob_score chiede la stima out-of-bag dell'errore, gratis.
foresta = RandomForestClassifier(
    n_estimators=300, max_features="sqrt", oob_score=True, n_jobs=-1,
    random_state=0)   # senza seme, OOB e importanze cambiano a ogni esecuzione
foresta.fit(X_train, y_train)
print("accuratezza OOB:", foresta.oob_score_)
print("importanza feature:", foresta.feature_importances_)

# Gradient boosting: alberi piccoli in sequenza.
# learning_rate basso + molti alberi = più stabile.
gb = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=3)
gb.fit(X_train, y_train)
```

Per il gradient boosting «da competizione» si usa di norma la libreria
dedicata, con la stessa interfaccia e l'early stopping integrato:

```{code-block} python
:class: pt-lento

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# La validazione si stacca dal training, mai dal test: serve a decidere
# quando fermarsi, e un test usato per decidere non misura più niente.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=0)

# eval_set + early_stopping_rounds: si ferma quando la validazione
# smette di migliorare, evitando l'overfitting del boosting.
xgb = XGBClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=4,
    subsample=0.8, early_stopping_rounds=30)
xgb.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
print("alberi usati:", xgb.best_iteration + 1, "su 1000")
```

Un consiglio pratico, che riassume tutta la sezione: su un problema tabellare
nuovo, una random forest con i parametri di default è quasi sempre la prima
cosa da provare; è la linea di base onesta contro cui misurare tutto il resto.
Se serve spremere di più, si passa a XGBoost o LightGBM con learning rate
basso ed early stopping. Per il *deep learning* (che affronteremo con PyTorch
nei capitoli successivi), il turno arriva sui dati non tabellari: immagini,
testo, audio, dove queste stesse foreste e questi boosting cedono il passo
alle reti.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un **albero decisionale** è *Indovina chi?*: una catena di domande sì/no,
  ciascuna scelta perché divide i casi nel modo più netto possibile, fino a una
  risposta. Sul foglio dei punti, ogni domanda è un taglio dritto, e l'albero
  ritaglia rettangoli.
- Si legge e si spiega («perché mi hai negato il prestito?»), ed è il suo
  pregio più raro. Ma un albero lasciato crescere impara a memoria ed è
  **instabile**: cambia dieci dati e viene fuori un albero diverso.
- Il rimedio è **non fidarsi di uno solo**. Se ne addestrano tanti su versioni
  leggermente diverse degli stessi dati e si fanno votare: gli errori, se sono
  errori diversi, si annullano a vicenda. La **foresta casuale** aggiunge il
  colpo di genio di nascondere a ogni albero alcune colonne, come una giuria in
  cui ogni giurato è bendato su aspetti diversi, così i pareri diventano
  davvero indipendenti.
- L'altra strada è metterli **in fila** invece che in parallelo: ogni nuovo
  modello si occupa solo di ciò che i precedenti hanno sbagliato, come lo
  studente che al secondo giro ripassa gli esercizi andati male. È il
  **boosting**, oggi il più accurato sui dati in tabella, ma va frenato: passi
  corti e stop appena smette di migliorare.
- Per combinare modelli **di tipo diverso** si può votare, oppure far decidere
  a un ultimo modello quanto fidarsi di ciascuno (lo **stacking**). Il secondo
  vince quando uno dei modelli è più debole degli altri, perché impara a
  pesarlo poco; una media a pesi fissi, invece, se lo porta appresso.
- La condizione perché combinare serva è sempre la stessa: i modelli devono
  **sbagliare in modi diversi**. Combinarne tre che sbagliano insieme non
  corregge niente, ripete l'errore con più sicurezza.
- Su un problema nuovo in tabella: prima una foresta casuale con le impostazioni
  di fabbrica, ed è già una linea di partenza onesta contro cui misurare tutto
  il resto.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un **albero decisionale** (CART) classifica per domande sì/no che partizionano
  lo spazio in rettangoli; sceglie a ogni nodo lo split che riduce di più
  l'impurità (**Gini** o **entropia**), massimizzando l'**information gain**.
  In regressione la foglia predice la media e si minimizza l'**MSE**.
- Gli alberi sono **interpretabili** («scatola bianca») ma ad **alta varianza**:
  un albero profondo memorizza i dati ed è instabile.
- Il **bagging** addestra molti alberi in parallelo su campioni **bootstrap** e
  li fa votare: mediando modelli decorrelati, abbatte la **varianza**.
- La **random forest** aggiunge il campionamento casuale delle feature a ogni
  split per **decorrelare** gli alberi; offre gratis l'errore **out-of-bag** e la
  **feature importance**.
- Il **boosting** addestra alberi deboli **in sequenza**, ognuno sui residui del
  precedente: dal riequilibrio dei pesi di **AdaBoost** alla discesa del
  gradiente nello spazio delle funzioni del **gradient boosting**. **XGBoost** e
  **LightGBM** lo rendono veloce e regolarizzato: dominano i dati tabellari.
- **Bagging vs boosting**: il primo cura la **varianza** (robusto, difficile da
  overfittare); il secondo cura il **bias** (più accurato, ma va frenato con
  **learning rate** basso ed **early stopping**).
- Per combinare modelli **di tipo diverso** ci sono il **voto** (meglio quello
  *morbido*, che media le probabilità invece di contare le etichette) e lo
  **stacking**, che addestra un meta-modello a pesare i predittori di base. Il
  meta-modello va addestrato su predizioni **fuori campione**, altrimenti
  impara a fidarsi di chi ha memorizzato.
- La condizione perché un ensemble serva è che i membri **sbaglino in modo
  diverso**: per la loss quadratica l'errore della media è l'errore medio dei
  membri meno la loro **diversità** (Krogh–Vedelsby), quindi un ensemble non è
  mai peggiore del membro **medio**; nulla vieta che sia peggiore del membro
  **migliore**, ed è ciò che accade al voto a pesi fissi con un componente
  debole. Lo stacking regge perché impara a pesarlo poco.
```

`````
