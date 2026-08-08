# Alberi decisionali e metodi ensemble

C'è un gioco da tavolo, *Indovina chi?*, in cui si scopre il personaggio
misterioso dell'avversario a forza di domande sì/no: «Porta gli occhiali?»,
«Ha i capelli neri?». Ogni risposta abbatte a metà i sospetti, finché non ne
resta uno solo. È esattamente così che ragiona un **albero decisionale**: una
catena di domande sulle caratteristiche di un esempio, ciascuna scelta per
dividere i casi nel modo più netto possibile, fino a una risposta.

Nelle sezioni precedenti abbiamo incontrato modelli che tracciano rette e
iperpiani: regressione, k-NN, regressione logistica. Gli alberi appartengono a
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

Ogni domanda è una soglia su una caratteristica: «reddito < 25 000 €?», «età <
30?». Una risposta manda l'esempio a sinistra, l'altra a destra. Geometricamente
questo taglia lo spazio delle caratteristiche in **rettangoli** con lati
paralleli agli assi ({numref}`fig-albero-decisionale`): ogni foglia dell'albero
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
$y$ degli esempi che vi cadono, e al posto di Gini si minimizza l'**MSE**
interno ai figli, cioè si cerca lo split che rende i valori dentro ciascun
gruppo il più simili possibile alla loro media. L'errore quadratico medio è la
stessa loss vista per la regressione lineare; qui però il modello, invece di
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
addestrato su un campione bootstrap: in media circa un terzo degli esempi
resta *fuori* da quel campione (gli esempi *out-of-bag*). Per ciascun esempio
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
F_M(x) = F_0 + \sum_{t=1}^{M} \nu\, h_t(x) ,
$$

dove $F_0$ è la costante che da sola minimizza la loss sui dati (per la loss
quadratica, la media dei target), $h_t$ è l'albero aggiunto al passo $t$ e
$\nu \in (0,1]$ è il **learning rate**. A ogni passo si vorrebbe muovere la
funzione corrente $F_{t-1}$ nella direzione che riduce di più la loss
$\mathcal{L}$; quella direzione, valutata su ciascun esempio, è l'opposto del
gradiente

$$
r_i^{(t)} = -\left[\frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{t-1}} ,
$$

detto **pseudo-residuo**. Il nuovo albero $h_t$ viene addestrato per
approssimare proprio questi pseudo-residui. Nel caso della loss quadratica
$\mathcal{L} = \tfrac{1}{2}(y - F)^2$ il gradiente si riduce a $r_i = y_i -
F_{t-1}(x_i)$: cioè, semplicemente, l'**errore residuo** ancora da spiegare
(al primo passo, lo scarto dalla media $F_0$).
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
- **Approssimazione al secondo ordine**. Invece del solo gradiente, XGBoost usa
  anche la derivata seconda della loss (uno sviluppo di Taylor al secondo
  ordine) per scegliere gli split: un passo più informato, come usare non solo
  la pendenza ma anche la curvatura.
- **Istogrammi e velocità**. Entrambi raggruppano i valori continui delle
  caratteristiche in poche decine di intervalli (*bin*): trovare lo split
  migliore diventa scorrere un istogramma invece di ordinare tutti i valori. È
  la mossa che rende LightGBM particolarmente rapido sui dataset grandi, insieme
  alla crescita *leaf-wise* (espande la foglia più promettente, non un livello
  per volta) e alla gestione nativa dei valori mancanti.

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
due freni principali sono il **learning rate** $\nu$ (passi piccoli, che
rendono l'apprendimento più lento ma più stabile: di solito si abbina un $\nu$
piccolo a molti alberi) e l'**early stopping**, cioè fermarsi quando l'errore
su un validation set smette di migliorare, come abbiamo visto nella sezione
sugli iperparametri. Il boosting inoltre è **sequenziale** per costruzione:
non si parallelizza sugli alberi come il bagging.

In sintesi: se cerchi robustezza con poco sforzo, parti dalla random forest; se
cerchi l'ultimo punto di accuratezza e sei disposto a mettere a punto learning
rate ed early stopping, passa al gradient boosting.

## In pratica, con scikit-learn

L'interfaccia `fit`/`predict` è la stessa vista per gli altri modelli
supervisionati; per una guida applicativa estesa a questi metodi rimandiamo al
manuale di Géron {cite}`geron2019hands`. I quattro protagonisti di questa
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
    n_estimators=300, max_features="sqrt", oob_score=True, n_jobs=-1)
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
```
