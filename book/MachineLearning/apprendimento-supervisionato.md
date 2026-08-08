# Apprendimento supervisionato: regressione e classificazione

Immagina di affiancare per una settimana un agente immobiliare esperto. Non ti
spiega nessuna formula: ti mostra centinaia di case già vendute (metri quadri,
numero di stanze, quartiere) e accanto a ciascuna il prezzo finale. Dopo un
po', davanti a un appartamento mai visto, sai già sparare una cifra
ragionevole. Hai imparato **dagli esempi etichettati**. È, in una frase, ciò
che fa l'apprendimento supervisionato: mostragli abbastanza coppie
*domanda–risposta* e imparerà a rispondere da solo.

## Imparare una funzione dagli esempi

```{figure} ../figures/feature-label-tipi-dati-ml.svg
:name: fig-tipi-di-feature
:alt: "I tre tipi di feature affiancati con un esempio ciascuno: numerica, come i metri quadri, su cui hanno senso somme e differenze; categorica, come il quartiere, dove i valori sono nomi senza ordine; ordinale, come la classe energetica, dove esiste un ordine ma non una distanza."
:width: 96%

Tre tipi di colonna, tre cose che si possono fare con i numeri. Confonderli è
l'errore che porta un modello a calcolare la media fra «Milano» e «Roma».
```

La distinzione di {numref}`fig-tipi-di-feature` è la prima decisione di ogni
progetto e non la prende il modello: la prende chi prepara i dati. Una colonna
categorica codificata come intero diventa, per il modello, una colonna
numerica a tutti gli effetti, con un ordine e delle distanze che nessuno
intendeva metterci.

Riprendiamo il vettore delle caratteristiche del capitolo di algebra lineare:
ogni appartamento è un vettore $X$ (metri quadri, stanze, piano), e a ciascuno
associamo un'etichetta $y$ (il prezzo). Il "supervisore" è proprio quella
$y$ nota: qualcuno, in passato, ha già registrato la risposta giusta.

`````{tab} Elementare

Abbiamo tante coppie *(descrizione, risposta)*: la descrizione è la nostra $X$,
la risposta è la $y$. L'obiettivo è trovare una **regola** che, data una nuova
descrizione, indovini la risposta. Chiamiamo questa regola $f$:

$$
\hat{y} = f(X)
$$

Il cappello su $\hat{y}$ ricorda che è una *previsione*, non la verità: è la
migliore ipotesi del modello. Imparare significa scegliere la $f$ che sbaglia
il meno possibile sugli esempi che già conosciamo, sperando che se la cavi bene
anche su quelli nuovi.

`````

`````{tab} Superiore

Partiamo da un insieme di addestramento di $m$ esempi etichettati,

$$
\mathcal{D} = \{(X^{(i)}, y^{(i)})\}_{i=1}^{m}, \qquad X^{(i)}\in\mathbb{R}^n,
$$

e cerchiamo una funzione $f:\mathcal{X}\to\mathcal{Y}$ che approssimi la
relazione ignota tra ingressi e uscite, con $\hat{y}=f(X)$. La qualità di $f$
si misura con una **funzione di costo** (o *loss*) $\mathcal{L}$, e
l'addestramento è il problema di ottimizzazione

$$
\theta^\star = \arg\min_{\theta}\ \frac{1}{m}\sum_{i=1}^{m}
\mathcal{L}\big(f_\theta(X^{(i)}),\, y^{(i)}\big),
$$

dove $\theta$ sono i parametri del modello. La natura di $\mathcal{Y}$
distingue i due problemi cardine: continuo per la regressione, discreto per la
classificazione.

`````

## Due domande, due problemi

Ciò che cambia tutto è il *tipo* di risposta. "Quanto costa questa casa?"
chiede un numero su una scala continua: è **regressione**. "Questa email è
spam, sì o no?" chiede un'etichetta da un insieme finito: è
**classificazione**. Stesso impianto, imparare $f$ da coppie $(X, y)$, due
geometrie diverse, come mostra {numref}`fig-regr-classif`: a sinistra
cerchiamo una linea che *segua* i punti, a destra una linea che li *separi*.

```{figure} ../figures/regressione-vs-classificazione.svg
:name: fig-regr-classif
:alt: Due pannelli affiancati. A sinistra, uno scatter di punti attraversato da una retta di regressione che ne segue l'andamento crescente. A destra, due nuvole di punti di colore diverso separate da una retta tratteggiata che funge da confine di decisione.
:width: 95%

Due volti dello stesso problema. Nella regressione (sinistra) la retta
*approssima* i dati; nella classificazione (destra) la retta *separa* le classi.
```

## La regressione lineare: la retta di best fit

Il modello più semplice, e sorprendentemente utile, ipotizza che la risposta
sia una combinazione lineare delle caratteristiche: raddoppia i metri quadri e,
grosso modo, il prezzo sale in proporzione.

`````{tab} Elementare

Con una sola caratteristica (i metri quadri) la regola è una **retta**:

$$
\hat{y} = w\,x + b
$$

Qui $w$ è la pendenza (di quanto sale il prezzo per ogni metro quadro in più) e
$b$ è il punto di partenza. Un esempio con i numeri: se ogni metro quadro vale
$w = 2\,000$ € e il punto di partenza è $b = 50\,000$ €, un appartamento di
$80$ m² viene stimato $\hat{y} = 2\,000 \cdot 80 + 50\,000 = 210\,000$ €. Fra
tutte le rette possibili scegliamo quella che passa "più in mezzo" ai punti:
la *retta di best fit*. Come misuriamo quanto è
buona? Guardiamo, per ogni casa, la distanza in verticale tra il prezzo vero e
quello previsto, la eleviamo al quadrato (così gli errori non si annullano tra
loro) e facciamo la media. Più questo numero è piccolo, migliore è la retta.

`````

`````{tab} Superiore

Con $n$ caratteristiche il modello diventa un prodotto scalare più un bias:

$$
\hat{y} = W^\top X + b .
$$

I parametri $W\in\mathbb{R}^n$ e $b\in\mathbb{R}$ si stimano minimizzando
l'**errore quadratico medio** (*Mean Squared Error*):

$$
\mathcal{L}(W, b) = \frac{1}{m}\sum_{i=1}^{m}\big(\hat{y}^{(i)} - y^{(i)}\big)^2
= \frac{1}{m}\sum_{i=1}^{m}\big(W^\top X^{(i)} + b - y^{(i)}\big)^2 .
$$

$\mathcal{L}$ è convessa in $(W,b)$: niente minimi locali in cui restare
intrappolati. Se le colonne della matrice dei dati, insieme alla colonna
costante del bias, sono linearmente indipendenti, il minimo è anche unico e si
raggiunge in forma chiusa con le equazioni normali; con feature collineari (una
feature costante basta, perché replica la colonna del bias), o con meno esempi
che feature, i punti di minimo diventano infiniti (un intero sottospazio, tutti
con lo stesso valore della loss) e le equazioni normali degenerano. Su grandi
dataset, in ogni caso, si preferisce la discesa del gradiente. Elevare al
quadrato penalizza fortemente gli errori grossi e rende la loss differenziabile
ovunque: due proprietà che tornano comode.

`````

## La regressione logistica: dal numero alla probabilità

Per la classificazione la retta da sola non basta: un prezzo può valere
$310\,000$, ma "spam sì/no" vive solo tra $0$ e $1$. La **regressione
logistica** (che, malgrado il nome, classifica) risolve il problema
schiacciando l'uscita lineare dentro l'intervallo $(0,1)$.

```{figure} ../figures/regressione-logistica.svg
:name: fig-sigmoide-soglia
:alt: "La curva sigmoide che sale da zero a uno, con una linea orizzontale tratteggiata alla soglia di 0,5. I punti che cadono sotto la soglia sono assegnati alla classe 0, quelli sopra alla classe 1; vicino alla soglia la curva è ripida, agli estremi si appiattisce."
:width: 84%

Dalla retta alla probabilità. La sigmoide non decide: produce un numero fra
zero e uno, e la decisione arriva dopo, quando si sceglie dove tagliare.
```

La separazione fra i due gesti in {numref}`fig-sigmoide-soglia` conta più di
quanto sembri, e tornerà nel capitolo sulle metriche. Il modello produce una
probabilità; la soglia a $0{,}5$ è una convenzione, non un risultato, e
spostarla è il modo più economico di scambiare falsi positivi con falsi
negativi senza riaddestrare niente.

`````{tab} Elementare

Prima calcoliamo un punteggio lineare, come nella regressione. Poi lo facciamo
passare in una funzione a forma di "S", la **sigmoide**, che comprime qualsiasi
numero in un valore tra $0$ e $1$: lo leggiamo come una *probabilità*. Un
punteggio molto positivo esce vicino a $1$ ("quasi certo spam"), uno molto
negativo vicino a $0$, e lo zero cade esattamente a $0{,}5$: è la linea di
confine, il punto in cui il modello è indeciso e cambia idea.

`````

`````{tab} Superiore

Sia $z = W^\top X + b$ il punteggio lineare. La sigmoide (o logistica) è

$$
\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1),
$$

e interpretiamo $\hat{y} = \sigma(z)$ come $P(y=1 \mid X)$. La previsione di
classe si ottiene con una soglia a $0{,}5$, che equivale a $z = 0$: l'insieme

$$
\{X : W^\top X + b = 0\}
$$

è il **confine di decisione**, un iperpiano che divide lo spazio in due regioni.
I parametri si stimano minimizzando la *cross-entropy* invece dell'MSE, perché
si accorda con l'interpretazione probabilistica e mantiene la loss convessa.

`````

## k-NN: chiedi ai vicini

Non tutti i modelli imparano dei parametri. Alcuni si limitano a *ricordare*.

```{figure} ../figures/knn-classificare-per-somiglianza.svg
:name: fig-knn
:alt: "Un piano con punti di due classi già etichettati. Un punto nuovo, di classe ignota, è al centro di un cerchio che racchiude i suoi cinque vicini più prossimi: tre appartengono a una classe e due all'altra, e il punto nuovo riceve l'etichetta della maggioranza."
:width: 80%

Nessun addestramento, solo un conteggio. La classe del punto nuovo è quella
che vince fra i suoi $k$ vicini, e cambiare $k$ può cambiare il verdetto.
```

Il cerchio disegnato in {numref}`fig-knn` è tutta la scelta di progetto:
allargandolo si includono vicini più lontani, e la decisione diventa più
stabile ma meno sensibile alle strutture locali. Con $k=1$ il modello ripete
il vicino più prossimo, rumore compreso; con $k$ pari al numero di esempi,
risponde sempre la classe più frequente.

`````{tab} Elementare

L'idea dei **k-nearest neighbors** è quasi banale, e proprio per questo istruttiva:
per classificare una casa nuova, cerca le $k$ case più simili tra quelle che
già conosci e lascia che *votino*. Se i $5$ vicini più prossimi sono per lo più
"quartiere costoso", lo sarà anche lei. Non c'è addestramento vero e proprio:
il modello tiene in memoria tutti gli esempi e decide solo al momento della
domanda. Per questo si dice **non parametrico**: non riassume i dati in pochi
numeri, li usa tutti.

`````

`````{tab} Superiore

Dato un punto $X$, si ordinano gli esempi di addestramento per distanza,
tipicamente euclidea, $\lVert X - X^{(i)}\rVert_2$, e si prendono i $k$ più
vicini. In classificazione si assegna la classe di maggioranza; in regressione
si fa la media dei loro $y^{(i)}$. Non esiste una fase di ottimizzazione: il
costo si sposta interamente sulla previsione, $O(m)$ per query nella versione
ingenua. Il valore di $k$ regola il compromesso: $k$ piccolo segue il rumore,
$k$ grande liscia troppo. La distanza euclidea, inoltre, impone di
normalizzare le feature, altrimenti quella con la scala più ampia domina il
conto.

Due raffinamenti sono già in scikit-learn. Il **voto pesato**
(`weights="distance"`) fa contare di più i vicini più prossimi invece di dare
a tutti e $k$ lo stesso peso. Le **strutture di indicizzazione** (KD-tree,
ball-tree) abbattono il costo per query da $O(m)$ a circa $O(\log m)$
partizionando lo spazio in anticipo. Proprio quegli indici, però, smettono di
essere utili oltre poche decine di dimensioni, per la ragione dell'avvertenza
qui sotto.

`````

```{warning}
**k-NN ha un nemico naturale: le troppe dimensioni.** Tutto il metodo poggia
sull'idea che «vicino» voglia dire «simile». In alta dimensione quell'idea si
sgretola: distribuendo punti a caso, le distanze fra tutte le coppie diventano
quasi identiche, e lo scarto fra il vicino più prossimo e il più lontano si
assottiglia fino a sparire. È come chiedere a qualcuno di indicare il migliore
amico in una folla dove tutti stanno esattamente alla stessa distanza: la
domanda perde senso, e il voto dei $k$ vicini diventa un voto casuale.

Per questo k-NN va quasi sempre preceduto da una riduzione della dimensionalità
o da una selezione seria delle feature. Il fenomeno, con i conti, è la
**maledizione della dimensionalità**, ed è il punto di partenza del capitolo su
riduzione e clustering.
```

## Un'ombra all'orizzonte: l'overfitting

C'è un tranello in agguato. Un modello abbastanza flessibile può imparare *a
memoria* gli esempi di addestramento (rumore compreso) e poi fallire su dati
nuovi, come uno studente che ripete le soluzioni senza aver capito il metodo.
È l'**overfitting**, il problema centrale del machine learning applicato: lo
affrontiamo nella sezione dedicata, insieme all'idea di tenere sempre da parte
dati che il modello non ha mai visto per misurarne l'onestà.

## In pratica, con scikit-learn

In Python i tre modelli sono tre righe, con la stessa interfaccia `fit`/`predict`:

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Regressione: prevede un valore continuo (es. il prezzo)
reg = LinearRegression().fit(X_train, y_prezzo)
prezzo_stimato = reg.predict(X_nuovo)

# Classificazione lineare: prevede una probabilità, poi una classe
clf = LogisticRegression().fit(X_train, y_spam)      # y_spam in {0, 1}
prob_spam = clf.predict_proba(X_nuovo)[:, 1]         # P(spam) in (0, 1)

# k-NN: nessun parametro, "vota" con i 5 vicini più simili
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_spam)
etichetta = knn.predict(X_nuovo)
```

La stessa forma (`fit` per imparare, `predict` per rispondere) vale per quasi
tutti i modelli della libreria: è la grammatica comune che ci porteremo dietro
per tutto il resto del libro.

```{admonition} Da ricordare
:class: important
- Supervisionato significa imparare $f:X\to y$ da **esempi già etichettati**.
- **Regressione** = uscita continua (MSE, retta di best fit); **classificazione**
  = uscita discreta (sigmoide, confine di decisione).
- **k-NN** è non parametrico: non stima parametri, ricorda i dati e li fa votare.
- Attenzione all'**overfitting**: imparare a memoria non è capire. Ne parliamo
  nella sezione dedicata.
```
