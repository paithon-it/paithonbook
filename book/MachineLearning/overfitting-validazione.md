# Overfitting, bias-varianza e validazione

C'è un modo infallibile per andare male a un esame: imparare a memoria le
soluzioni dei compiti degli anni scorsi. Chi lo fa risponde alla perfezione
alle domande già viste e va nel panico davanti a un esercizio anche solo
leggermente diverso. Ha memorizzato, non capito. Un modello di machine
learning può cadere esattamente nella stessa trappola, e distinguere la
memoria dalla comprensione è, in fondo, il problema centrale di tutta la
disciplina.

Nelle sezioni precedenti abbiamo detto che la posta in gioco non è riprodurre
gli esempi già visti, ma **generalizzare** a input nuovi. Un modello che azzecca
ogni risposta sui dati di addestramento e sbaglia su quelli veri non ha imparato
nulla di utile. Questo capitolo parla di come accorgersene e come porvi rimedio.

## Imparare o memorizzare: overfitting e underfitting

Ci sono due modi opposti di sbagliare, e conviene tenerli davanti agli occhi
insieme ({numref}`fig-overfitting`).

```{figure} ../figures/overfitting-tre-fit.svg
:name: fig-overfitting
:alt: Tre pannelli con la stessa nube di punti a forma di collina. A sinistra una retta quasi orizzontale la ignora (underfitting); al centro una curva morbida la segue bene (buon fit); a destra una curva contorta passa per ogni punto oscillando in mezzo (overfitting).
:width: 95%

Lo stesso insieme di punti, tre modelli. Il modello troppo semplice non coglie
l'andamento; quello troppo flessibile lo ricalca fin dentro il rumore. In mezzo,
il buon compromesso.
```

`````{tab} Elementare

Da una parte c'è chi generalizza troppo poco: un modello **troppo semplice**,
come una retta costretta a descrivere dei dati chiaramente curvi. Sbaglia già
sugli esempi che ha visto in addestramento, figurarsi sui nuovi. Si chiama
**underfitting**: il modello è troppo rigido per il problema.

Dall'altra parte c'è chi generalizza troppo: un modello **troppo flessibile**
che si contorce per passare esattamente su ogni singolo punto, rumore compreso.
Sul foglio d'esame degli esempi visti prende dieci e lode, ma ha imparato anche
gli errori di misura, gli accidenti, il caso. Su un dato nuovo crolla. Si chiama
**overfitting**: il modello ha memorizzato invece di capire.

`````

`````{tab} Superiore

Formalmente, l'errore che ci interessa è quello su dati **non** visti in
addestramento, l'*errore di generalizzazione*. Confrontarlo con l'errore
sull'insieme di training rivela il regime in cui ci troviamo:

- **Underfitting**: errore di training *alto* e vicino a quello di test. Il
  modello è troppo poco espressivo: non riesce a catturare la struttura dei
  dati (*bias* alto).
- **Overfitting**: errore di training *molto basso* ma errore di test *alto*.
  Il modello ha capacità in eccesso e adatta $f_\theta$ anche alle
  fluttuazioni casuali del campione (*varianza* alta).

Il divario tra i due errori, $\text{err}_{\text{test}} -
\text{err}_{\text{train}}$, è la spia dell'overfitting: quando si allarga,
il modello sta memorizzando.

`````

## Il compromesso bias-varianza

Underfitting e overfitting sono le due facce di un'unica tensione, che ha un nome
classico: il **compromesso bias-varianza** (*bias-variance tradeoff*).

```{figure} ../figures/bias-varianza.svg
:name: fig-bias-varianza
:alt: "Quattro bersagli disposti in una griglia due per due, con le colonne per varianza bassa e alta e le righe per bias basso e alto. Con bias e varianza bassi i colpi sono raccolti al centro; con varianza alta e bias basso sono sparsi ma centrati in media; con bias alto e varianza bassa sono raccolti ma spostati dal centro; con entrambi alti sono sparsi e spostati."
:width: 78%

I quattro casi sul bersaglio. Il bias è di quanto si è spostato il gruppo dei
colpi; la varianza è quanto il gruppo è largo. Sono due difetti diversi e si
correggono in modi opposti.
```

Il bersaglio in basso a sinistra di {numref}`fig-bias-varianza`, colpi
raccolti ma tutti fuori centro, è il più insidioso: un modello del genere è
molto *coerente*, dà quasi sempre la stessa risposta, e la coerenza si scambia
facilmente per affidabilità. Aggiungere dati non lo aggiusta, perché il
problema non è l'incertezza ma la mira.

`````{tab} Elementare

Immagina di ripetere l'esperimento: raccogli molte volte un nuovo campione di
dati e riaddestri il modello ogni volta.

- Un modello **rigido** (la retta) darà sempre più o meno la stessa risposta,
  campione dopo campione: è *stabile* ma *sistematicamente storto*, perché la
  forma giusta non è una retta. Questo errore sistematico è il **bias**.
- Un modello **flessibile** (la curva contorta) cambierà parecchio a ogni nuovo
  campione, inseguendo il rumore di turno. È *senza pregiudizi* sulla forma, ma
  *instabile*: questa irrequietezza è la **varianza**.

Semplice = molto bias, poca varianza. Complesso = poco bias, molta varianza. Il
bravo modellista cerca il punto di mezzo.

`````

`````{tab} Superiore

Per un target $y = f(x) + \varepsilon$ con rumore di varianza $\sigma^2$,
l'errore quadratico atteso di una previsione $\hat{f}(x)$, mediato sui possibili
insiemi di addestramento e sul rumore del punto di test, si decompone in tre
termini:

$$
\mathbb{E}\big[(y-\hat{f}(x))^2\big]
= \underbrace{\big(\mathbb{E}[\hat{f}(x)]-f(x)\big)^2}_{\text{Bias}^2}
+ \underbrace{\mathbb{E}\big[(\hat{f}(x)-\mathbb{E}[\hat{f}(x)])^2\big]}_{\text{Varianza}}
+ \underbrace{\sigma^2}_{\text{irriducibile}} .
$$

Il **bias** misura quanto la previsione media si scosta dalla verità $f(x)$; la
**varianza** quanto $\hat{f}(x)$ oscilla al variare del campione; $\sigma^2$ è il
rumore intrinseco, che nessun modello può eliminare. Aumentando la complessità il
bias cala ma la varianza cresce: l'errore di test ha la classica forma a **U**, e
il minimo è il modello ottimale.

`````

## Train, validation e test: perché il test non si tocca

Per accorgersi dell'overfitting bisogna misurare l'errore su dati che il modello
**non ha usato** per imparare. Da qui la regola d'oro: si divide il dataset in
tre parti, ciascuna con un compito distinto.

```{figure} ../figures/train-test-split-scaling-outlier.svg
:name: fig-split-e-scaler
:alt: "Il dataset viene diviso in una parte di training e una di test. Lo scaler viene tarato soltanto sulla parte di training, calcolandone media e deviazione standard, e poi applicato a entrambe le parti. Una freccia barrata segnala l'errore da evitare: tarare lo scaler sull'intero dataset prima della divisione."
:width: 96%

La freccia barrata è l'errore che non si vede. Se lo scaler guarda anche il
test per calcolare la media, un pezzo di informazione del test è già entrato
nell'addestramento.
```

Quella di {numref}`fig-split-e-scaler` è la forma più insidiosa di **data
leakage**, perché non produce nessun errore e nessun avviso: produce solo un
punteggio un po' più alto del vero. La regola pratica che ne discende è secca:
qualunque cosa impari dai dati (uno scaler, un'imputazione, una selezione di
feature) va tarata dentro il training e applicata al resto, mai prima della
divisione.

- **Training set**: i dati su cui il modello impara i parametri $\theta$.
- **Validation set**, i dati su cui si scelgono gli *iperparametri*, cioè le
  scelte di contorno che non si imparano dai dati: quanto complesso può essere
  il modello, quanto forte il freno alla memorizzazione che vedremo tra poco.
- **Test set**: i dati che si guardano **una sola volta**, alla fine, per
  stimare onestamente le prestazioni nel mondo reale.

`````{tab} Elementare

Il test set è il compito d'esame vero. Se lo sbirci mentre studi e aggiusti le
tue scelte in base a quello, il voto finale non dice più nulla: hai imparato a
memoria *quell'* esame. Per questo il test si tiene chiuso in un cassetto e si
apre soltanto alla fine. Ogni volta che usi il test per decidere qualcosa, lo
"consumi", e il numero che ti restituisce diventa troppo ottimista.

`````

`````{tab} Superiore

Usare il test per selezionare modelli introduce una forma sottile di *data
leakage*: si finisce per fare overfitting sul test stesso, e la stima
dell'errore di generalizzazione diventa distorta verso il basso. Il validation
set esiste proprio per assorbire tutte le decisioni intermedie e preservare
l'imparzialità del test. Suddivisioni tipiche: $60/20/20$ o $80/10/10$. La
selezione dei modelli avviene su training + validation; il test resta un
osservatore neutrale che entra in scena solo a giochi fatti.

`````

## La cross-validation

Mettere da parte un validation set fisso ha un difetto: con pochi dati, la stima
dipende troppo da *quali* esempi sono finiti nel validation. La
**k-fold cross-validation** aggira il problema riutilizzando i dati con
intelligenza.

```{figure} ../figures/cross-validation-il-test-che-non-bara.svg
:name: fig-cross-validation
:alt: "Cinque righe, una per giro. In ciascuna, i dati sono divisi in cinque blocchi: uno fa da test e gli altri quattro da training, e il blocco di test scorre di una posizione a ogni riga, dal primo al quinto. A destra di ogni riga il punteggio ottenuto in quel giro. In fondo, il risultato è la media dei cinque punteggi con la loro deviazione standard."
:width: 92%

Il blocco di test ruota. Alla fine ogni esempio ha fatto da test esattamente
una volta, e il risultato non è un numero ma un numero con la sua
variabilità.
```

La riga finale di {numref}`fig-cross-validation` è la parte che si tende a
buttare via: la deviazione standard fra i cinque giri. Se due modelli
distano meno di quella, la classifica fra loro dipende da come sono caduti i
blocchi, non da quale sia migliore.

`````{tab} Elementare

Dividi i dati di addestramento in $k$ blocchi uguali (di solito $k=5$ o $10$).
A turno, ogni blocco fa da validation mentre gli altri $k-1$ addestrano il
modello. Ottieni così $k$ misure di errore, ognuna su un pezzo diverso di dati,
e ne fai la **media**. È come far correggere il compito a cinque professori
diversi invece che a uno solo: il giudizio finale è più affidabile e meno
soggetto al caso.

`````

`````{tab} Superiore

Partizionato il training in $k$ fold $D_1,\dots,D_k$, per ogni $i$ si addestra su
$D\setminus D_i$ e si valuta su $D_i$. La stima cross-validata è la media degli
errori di validazione:

$$
\text{CV}_{k} = \frac{1}{k}\sum_{i=1}^{k}
\mathcal{L}\big(f_\theta^{(-i)},\, D_i\big),
$$

dove $f_\theta^{(-i)}$ è il modello addestrato escludendo il fold $i$-esimo. Il
caso estremo $k=m$ (un fold per esempio) è la *leave-one-out*: quasi non
distorta ma costosa. Valori $k=5$ o $k=10$ offrono il miglior compromesso tra
costo computazionale e stabilità della stima.

`````

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge

# il test resta da parte fin dall'inizio, non lo tocchiamo più
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

modello = Ridge(alpha=1.0)              # alpha = quanto frena il modello (v. sotto)
scores = cross_val_score(modello, X_train, y_train, cv=5,
                         scoring="neg_mean_squared_error")  # 5-fold CV
print(-scores.mean())                  # errore medio di validazione
```

## Mettere un freno: regolarizzazione L1 e L2

Un modo diretto per contrastare l'overfitting è impedire al modello di diventare
troppo "estremo". La **regolarizzazione** aggiunge alla loss un termine di
penalità che cresce con la grandezza dei parametri: il modello paga un prezzo
ogni volta che alza troppo i pesi, e quindi lo fa solo se ne vale davvero la pena.

```{figure} ../figures/regolarizzazione-l1-l2.svg
:name: fig-l1-l2
:alt: "Due piani con i pesi w1 e w2 sugli assi. A sinistra la L1: la regione ammessa è un rombo con i vertici sugli assi, e le curve di livello dell'errore lo toccano proprio in un vertice, dove w1 è esattamente zero. A destra la L2: la regione è un cerchio, e il punto di contatto cade in una posizione qualsiasi del bordo, dove entrambi i pesi sono piccoli ma nessuno è zero."
:width: 96%

La differenza sta negli spigoli. Il rombo della L1 ha i vertici sugli assi, e
un vertice è il punto che una curva di livello incontra per primo: da lì i
pesi esattamente nulli. Il cerchio della L2 non ha spigoli, e non privilegia
nessuna direzione.
```

{numref}`fig-l1-l2` spiega con la geometria quello che di solito si impara
come una regola da mandare a memoria («la L1 fa selezione di variabili, la L2
no»). Non è una proprietà misteriosa della norma: è la forma della regione
ammessa, e il fatto che un rombo tocchi gli assi mentre un cerchio li sfiora
solo per caso.

`````{tab} Elementare

Pensa alla regolarizzazione come a un budget di spesa sui pesi del modello.
Senza limiti, per passare su ogni punto il modello gonfia i suoi coefficienti
a dismisura: è così che nasce la curva contorta. Imponendo un tetto alla spesa
totale, lo costringi a essere sobrio, e le curve sobrie sono più morbide,
quindi generalizzano meglio. Due modi di contare la spesa:

- **Ridge (L2)**: penalizza la *somma dei quadrati* dei pesi. Li rimpicciolisce
  tutti dolcemente, senza azzerarne nessuno.
- **Lasso (L1)**: penalizza la *somma dei valori assoluti*. Tende a spingere a
  **zero netto** i pesi delle feature inutili, di fatto selezionandole.

`````

`````{tab} Superiore

Si aggiunge alla loss $\mathcal{L}(\theta)$ un termine di penalità pesato da un
iperparametro $\lambda \ge 0$ che regola l'intensità del freno. Per la
regressione **Ridge** (norma $\ell_2$):

$$
\mathcal{L}_{\text{Ridge}}(\theta)
= \frac{1}{m}\sum_{i=1}^{m}\big(\hat{y}^{(i)}-y^{(i)}\big)^2
+ \lambda\sum_{j=1}^{n}\theta_j^{2},
$$

per il **Lasso** (norma $\ell_1$):

$$
\mathcal{L}_{\text{Lasso}}(\theta)
= \frac{1}{m}\sum_{i=1}^{m}\big(\hat{y}^{(i)}-y^{(i)}\big)^2
+ \lambda\sum_{j=1}^{n}|\theta_j| .
$$

Con $\lambda \to 0$ si torna al modello non regolarizzato (varianza alta); con
$\lambda$ grande i pesi sono schiacciati verso zero (bias alto): $\lambda$ è
la manopola del compromesso bias-varianza, e la si sceglie per
cross-validation. La geometria spigolosa della norma $\ell_1$ è ciò che rende
*sparse* le soluzioni del Lasso, annullando interi coefficienti: un selettore
automatico di feature.

`````

## Il rasoio di Occam

Sotto tutto questo c'è un principio antico. Nel XIV secolo il frate francescano
**Guglielmo di Occam** enunciò quello che oggi chiamiamo il *rasoio*: *entia non
sunt multiplicanda praeter necessitatem*, non moltiplicare le entità oltre il
necessario. Tradotto per noi: **a parità di capacità di spiegare i dati, scegli
il modello più semplice**.

La regolarizzazione non è altro che il rasoio di Occam scritto in formule: quel
termine $\lambda$ è il prezzo che facciamo pagare alla complessità, così che il
modello la compri solo quando serve davvero. La curva morbida del pannello
centrale vince non perché sia la più elaborata, ma perché è la più semplice tra
quelle che rendono conto dei dati. La semplicità, in machine learning, non è
eleganza estetica: è ciò che permette di generalizzare.

## Quando la U non basta: la doppia discesa

C'è un punto in cui il quadro appena disegnato entra in tensione con la
pratica del deep learning, e vale la pena affrontarlo invece di ignorarlo. La
curva a U dice: oltre una certa capacità l'errore di test risale. Eppure i
modelli linguistici moderni hanno miliardi di parametri, molti più degli
esempi di addestramento, memorizzano perfettamente il training set, e
generalizzano.

```{figure} ../figures/double-descent.svg
:name: fig-double-descent
:alt: "Grafico con la capacità del modello, cioè il numero di parametri, in ascissa e l'errore in ordinata. L'errore di training scende e resta a zero. L'errore di test disegna prima la classica U del regime classico, con un minimo, poi risale fino a un picco in corrispondenza della soglia di interpolazione, e infine riscende in una seconda discesa nel regime sovraparametrizzato."
:width: 96%

La U non è sbagliata: è solo il primo tratto. Oltre il picco, dove il modello
ha appena abbastanza capacità per memorizzare tutto, la curva riscende invece
di continuare a salire.
```

Il punto interessante di {numref}`fig-double-descent` è il **picco**, non le
discese. Sta dove il modello ha esattamente la capacità necessaria per
interpolare i dati e nessuna di più: è costretto a una sola soluzione, e
quella soluzione è pessima. Con più capacità le soluzioni tornano a essere
tante, e fra tante l'addestramento ne sceglie una regolare.

`````{tab} Elementare

Qualcuno ha fatto la cosa che i manuali sconsigliavano: ha continuato a
ingrandire il modello *oltre* il punto in cui impara a memoria ogni esempio. E
l'errore sul test, dopo essere risalito come previsto, **è tornato a scendere**.
Non un caso fortunato: un fenomeno riproducibile, chiamato **doppia discesa**.

La curva, insomma, non è una U ma una U seguita da una seconda discesa. Il
picco sta esattamente nel punto di **interpolazione**: quando il modello ha
giusto i parametri sufficienti per azzerare l'errore di training. Lì è
costretto a passare per tutti i punti, rumore compreso, nell'unico modo che
gli riesce: contorcendosi. È il momento peggiore.

Oltre quel punto, però, di soluzioni che passano per tutti i dati ce ne sono
infinite, e l'addestramento non ne sceglie una a caso: la discesa del gradiente
tende verso le soluzioni «più lisce» fra quelle disponibili. Avere parametri in
eccesso non è più libertà di sbagliare, è **libertà di scegliere una soluzione
gentile**.

Un'immagine: far passare una curva per venti punti. Con un polinomio di grado
esattamente venti la curva è obbligata, e fra un punto e l'altro impazzisce. Con
molta più libertà puoi scegliere, fra le infinite curve che passano per quei
punti, la meno tormentata.

`````

`````{tab} Superiore

Il fenomeno è stato descritto sistematicamente da Belkin e colleghi (2019) e poi
in ambito neurale da Nakkiran e colleghi (2021). Tre precisazioni che evitano
di trarne la conclusione sbagliata.

**Non è solo la taglia del modello.** La doppia discesa si osserva anche
rispetto al *tempo di addestramento* (epoch-wise) e alla *quantità di dati*, e
in quest'ultimo caso produce l'effetto contro-intuitivo per cui, vicino al
punto di interpolazione, **aggiungere dati può peggiorare** il test error.

**Il rasoio di Occam non è confutato, è misurato male.** Il numero di
parametri è un pessimo proxy della complessità di una rete: la quantità che
conta è una misura di norma della soluzione trovata, non di capacità
dell'ipotesi. La discesa del gradiente ha un *bias implicito* verso soluzioni
a norma piccola, e in quel senso continua a scegliere la spiegazione più
semplice: solo che «semplice» non si conta in parametri.

**La regolarizzazione appiana il picco.** Con regolarizzazione adeguata la gobba
attorno all'interpolazione si attenua o sparisce: la doppia discesa è più
marcata proprio dove non si regolarizza.

Resta parecchio da capire: quali architetture e quali regimi la mostrino, e
perché il bias implicito abbia la forma che ha. Il consiglio operativo non è
cambiato, ma la sua motivazione sì: **non fermarsi al primo minimo della curva
di validazione solo perché il modello sembra troppo grande.**

`````

## Il biglietto vincente: a che serve tutta quella capacità

La doppia discesa dice *che* le reti sovradimensionate generalizzano. Resta la
domanda su *perché*, e c'è un risultato che offre una risposta diversa e
sorprendentemente concreta.

```{figure} ../figures/lottery-ticket-hypothesis.svg
:name: fig-biglietto-vincente
:alt: "A sinistra una rete densa con tutte le sue connessioni disegnate in grigio. A destra la stessa rete con evidenziato un sottoinsieme molto più piccolo di connessioni e nodi, il biglietto vincente, che addestrato da solo a partire dalla propria inizializzazione originale raggiunge la stessa accuratezza della rete intera."
:width: 96%

Dentro la rete grande ce n'è una piccola che basta. Il punto non è che si può
potare a posteriori: è che quella sottorete funziona solo se riparte dai *suoi*
pesi iniziali, quelli che aveva nella rete grande.
```

La condizione in coda a {numref}`fig-biglietto-vincente` è ciò che rende
l'ipotesi interessante invece che ovvia. Se si riprende la stessa sottorete e
la si inizializza da capo a caso, non impara altrettanto bene: il biglietto
non è la forma della sottorete, è la coppia fra la forma e i numeri con cui è
nata. Sovradimensionare, in questa lettura, serve a comprare molti biglietti.

`````{tab} Elementare

Il punto di partenza è un paradosso noto da tempo. Prendi una rete addestrata,
elimina i pesi più piccoli: puoi buttarne via il $90\%$ senza quasi perdere
accuratezza. Ma se poi provi a costruire da zero una rete piccola *con quella
stessa struttura* e ad addestrarla, impara peggio. La potatura funziona solo
**dopo** l'addestramento, e nessuno spiegava bene perché.

Frankle e Carbin (2019) hanno provato una cosa diversa. Dopo aver potato, invece
di ripartire con pesi casuali nuovi, hanno **riavvolto** i pesi sopravvissuti ai
valori casuali che avevano *all'inizio*, prima di qualsiasi addestramento. Quella
sottorete minuscola, riaddestrata da sola, raggiunge l'accuratezza della rete
piena.

Il dettaglio decisivo è che se reinizializzi con *altri* numeri casuali, non
funziona più. Quindi non conta solo quali connessioni sopravvivono: conta che
partano da quei valori lì. La rete grande, insomma, non serve perché serva tutta
quella capacità: serve perché, fra le sue milioni di connessioni inizializzate a
caso, ne contiene per fortuna un sottoinsieme già disposto bene per il compito.
Il resto è impalcatura. Da qui il nome: **biglietto vincente**, e la rete grande
come un mazzo di biglietti comprati tutti insieme.

`````

`````{tab} Superiore

La procedura è l'*iterative magnitude pruning*: si annota l'inizializzazione
$\theta_0$, si addestra, si elimina una frazione dei pesi più piccoli, si
riportano i sopravvissuti ai valori in $\theta_0$, si riaddestra, si ripete. Le
sottoreti trovate pesavano spesso meno del $10$–$20\%$ della rete di partenza, e
raggiungevano l'accuratezza piena in un numero comparabile di iterazioni.

Due avvertenze onestà, perché il risultato è più fragile di come viene spesso
citato.

**Alla scala grande la ricetta va corretta.** Su reti profonde e dataset seri il
riavvolgimento a $\theta_0$ smette di funzionare; si riavvolge invece a un
$\theta_k$ dopo qualche iterazione di addestramento (*rewinding* tardivo). Il
biglietto, quindi, non è del tutto presente all'inizializzazione: si forma nelle
prime fasi.

**Non è un metodo di compressione pratico.** Per *trovare* il biglietto
bisogna addestrare la rete piena, più volte. Il valore è conoscitivo (dice
qualcosa su cosa fa la sovraparametrizzazione) non computazionale. Per
comprimere davvero si usano la potatura strutturata e la quantizzazione del
capitolo su MLOps.

Il filo con la sezione precedente è comunque lo stesso: il numero di parametri
misura male la complessità. La doppia discesa lo mostra dall'esterno, guardando
la curva d'errore; il biglietto vincente dall'interno, guardando cosa la rete
usa davvero.

`````

```{admonition} Da ricordare
:class: important
- **Underfitting**: modello troppo semplice, sbaglia già sul training (bias
  alto). **Overfitting**: modello troppo flessibile, memorizza il rumore ed è
  ottimo sul training ma pessimo sui dati nuovi (varianza alta).
- L'errore di test ha forma a **U** nella complessità: il minimo è il
  compromesso **bias-varianza**. Oltre il punto di interpolazione, però, la
  curva può **riscendere** (doppia discesa): il numero di parametri è un
  pessimo proxy della complessità di una rete.
- Il **biglietto vincente** guarda lo stesso fatto dall'interno: una rete grande
  contiene una sottorete piccola già ben inizializzata, e il resto è
  impalcatura.
- Si divide in **train / validation / test**. Il **test non si tocca**: si apre
  una sola volta, alla fine, o la stima diventa ottimista.
- La **k-fold cross-validation** media $k$ validazioni su fold diversi: stima
  più stabile quando i dati sono pochi.
- La **regolarizzazione** (Ridge $\ell_2$, Lasso $\ell_1$) frena la complessità
  con una penalità $\lambda$ sui pesi; il Lasso azzera le feature inutili.
- Tutto obbedisce al **rasoio di Occam**: a parità di adattamento, vince il
  modello più semplice.
```
