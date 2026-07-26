# Support Vector Machine: il margine massimo

Disegna due nuvole di punti su un foglio — pallini blu a sinistra, quadratini
rossi a destra — e traccia una retta che li separi. Facile. Ora traccia
*un'altra* retta che li separi lo stesso, e poi un'altra ancora: se le due
nuvole sono ben distinte, di rette buone ce ne sono infinite. Quale scegliere?
La regressione logistica del capitolo sull'apprendimento supervisionato ne
sceglie una, il *perceptron* un'altra, e nessuno dei due si pone davvero la
domanda. La **Support Vector Machine** (SVM) sì, e con una risposta di netta
eleganza geometrica: tra tutte le rette che separano, scegli la più
*prudente* — quella che lascia il corridoio più largo possibile tra le due
classi.

È un'idea nata nei primi anni Novanta nei laboratori Bell da Vladimir Vapnik e
colleghi, formalizzata da Boser, Guyon e Vapnik nel 1992
{cite}`boser1992training` e completata da Cortes e Vapnik nel 1995
{cite}`cortes1995support`. Per un decennio, prima dell'ondata del deep
learning, le SVM sono state il classificatore di riferimento: matematicamente
solide, sorprendentemente efficaci su dataset di dimensioni medie, e ancora
oggi una scelta sensata quando gli esempi sono poche migliaia.

## La retta più prudente

Immagina il confine tra due quartieri di case. Potresti tracciarlo rasente al
muro dell'ultima villetta di uno dei due — ma basterebbe una casa nuova, un
metro più in là, per trovarti dalla parte sbagliata. La scelta prudente è
tirare il confine *nel mezzo del prato*, il più lontano possibile dalle case
di entrambi i lati. Così hai il massimo respiro: piccole variazioni non ti
fanno sbagliare. Questo respiro, in gergo, è il **margine**, e la SVM lo rende
il più largo possibile.

```{figure} ../figures/svm-margine.svg
:name: fig-svm-margine
:alt: Due classi di punti nel piano, cerchi teal e quadrati terracotta, separate da un iperpiano nero e da due rette di margine parallele tratteggiate che delimitano un corridoio. I punti che toccano le rette di margine, i vettori di supporto, sono cerchiati in ocra; una doppia freccia misura la larghezza del corridoio, pari a due su norma di w.
:width: 80%

Tra le infinite rette che separano le due classi, la SVM sceglie quella che
massimizza il **margine**: la larghezza del corridoio (in ocra) tra i punti
più vicini. Solo quei punti — i **vettori di supporto**, cerchiati — determinano
la soluzione.
```

Come mostra {numref}`fig-svm-margine`, la soluzione poggia su pochissimi punti:
quelli che toccano i bordi del corridoio. Tutti gli altri, per quanto numerosi,
sono irrilevanti — potresti spostarli o cancellarli e il confine non si
muoverebbe di un millimetro. Sono i punti sul bordo a «reggere» l'iperpiano, e
per questo si chiamano **vettori di supporto**. È una differenza sostanziale
rispetto alla regressione logistica, la cui frontiera dipende (sia pur poco) da
*tutti* i dati.

## Il classificatore a massimo margine

Formalizziamo la geometria del corridoio. La frontiera è un **iperpiano**: una
retta in due dimensioni, un piano in tre, un oggetto piatto di dimensione
$n-1$ in uno spazio a $n$ dimensioni.

`````{tab} Elementare

L'iperpiano è l'insieme dei punti $X$ che soddisfano l'equazione

$$
W^\top X + b = 0,
$$

la stessa che descriveva il confine di decisione della regressione logistica:
$W$ è il vettore che dà l'orientamento della frontiera e $b$ la sposta avanti o
indietro. Un punto nuovo si classifica guardando il *segno* di $W^\top X + b$:
positivo di qua, negativo di là. La novità della SVM non è questa formula — è
il criterio con cui sceglie $W$ e $b$: non una frontiera qualsiasi, ma quella
che lascia il vuoto più ampio attorno a sé. Più il corridoio è largo, più il
classificatore è robusto.

`````

`````{tab} Superiore

Fissiamo la scala imponendo che gli esempi più vicini alla frontiera
soddisfino $W^\top X + b = \pm 1$: sono le due rette di margine. Con la
convenzione $y_i \in \{-1, +1\}$ per le due classi, chiedere che ogni punto
stia dalla parte giusta *e fuori dal corridoio* si scrive in un colpo solo:

$$
y_i\,(W^\top X_i + b) \ge 1, \qquad i = 1, \dots, m.
$$

La distanza di un punto sul margine dall'iperpiano è $1/\lVert W\rVert$, quindi
la larghezza totale del corridoio — da un bordo all'altro — è

$$
\text{margine} = \frac{2}{\lVert W\rVert}.
$$

Massimizzare il margine equivale allora a *minimizzare* $\lVert W\rVert$, o più
comodamente il suo quadrato (differenziabile ovunque). Il problema del
**margine rigido** (*hard margin*) è

$$
\min_{W,\,b}\ \tfrac{1}{2}\lVert W\rVert^2
\quad\text{soggetto a}\quad y_i\,(W^\top X_i + b) \ge 1 \ \ \forall i,
$$

dove $W$ è il vettore dei pesi, $b$ il termine di bias, $X_i$ l'$i$-esimo
esempio e $y_i \in \{-1,+1\}$ la sua etichetta. È un problema di
programmazione quadratica *convesso*: ha un'unica soluzione, senza minimi
locali in cui restare intrappolati.

`````

### Un esempio con i numeri

Mettiamo in fila numeri concreti in due dimensioni. Prendiamo quattro punti:

| punto | coordinate | classe $y_i$ |
|-------|------------|--------------|
| $X_1$ | $(0,\,0)$  | $-1$ |
| $X_2$ | $(2,\,2)$  | $+1$ |
| $X_3$ | $(-1,-1)$  | $-1$ |
| $X_4$ | $(3,\,3)$  | $+1$ |

I punti stanno tutti sulla diagonale. Per simmetria l'iperpiano di massimo
margine è perpendicolare alla diagonale e passa a metà strada tra $X_1$ e
$X_2$, cioè per il punto $(1,1)$. La soluzione è

$$
W = (0{,}5,\ 0{,}5), \qquad b = -1.
$$

Verifichiamo che i due punti più interni, $X_1$ e $X_2$, cadano esattamente
sulle rette di margine, cioè soddisfino $y_i(W^\top X_i + b) = 1$:

- $X_1=(0,0)$, classe $-1$: $\;W^\top X_1 + b = 0{,}5\cdot 0 + 0{,}5\cdot 0 - 1 = -1$, e $\;y_1(-1) = (-1)(-1) = 1$. ✓
- $X_2=(2,2)$, classe $+1$: $\;W^\top X_2 + b = 0{,}5\cdot 2 + 0{,}5\cdot 2 - 1 = 1$, e $\;y_2(1) = (+1)(1) = 1$. ✓

Sono loro i **vettori di supporto**. Gli altri due stanno più lontani, oltre il
margine (il vincolo vale con la disuguaglianza *stretta*):

- $X_3=(-1,-1)$: $\;W^\top X_3 + b = -0{,}5 - 0{,}5 - 1 = -2$, e $\;y_3(-2) = (-1)(-2) = 2 \ge 1$. ✓
- $X_4=(3,3)$: $\;W^\top X_4 + b = 1{,}5 + 1{,}5 - 1 = 2$, e $\;y_4(2) = (+1)(2) = 2 \ge 1$. ✓

La larghezza del corridoio è

$$
\frac{2}{\lVert W\rVert} = \frac{2}{\sqrt{0{,}5^2 + 0{,}5^2}}
= \frac{2}{\sqrt{0{,}5}} = \frac{2}{0{,}707} \approx 2{,}83 = 2\sqrt{2}.
$$

Ed è esattamente la distanza euclidea tra $X_1=(0,0)$ e $X_2=(2,2)$, che vale
$\sqrt{2^2+2^2}=\sqrt{8}=2\sqrt{2}$: i due vettori di supporto, uno per classe,
si affacciano sui bordi opposti dello stesso corridoio. Nota il punto cruciale:
cancellare $X_3$ e $X_4$ non cambia nulla — la soluzione dipende solo dai due
punti sul bordo.

## Quando i dati non sono perfetti: il margine morbido

Il margine rigido ha due difetti gemelli: pretende che i dati siano
*perfettamente* separabili, e basta un solo punto fuori posto — un outlier, un
errore di misura — per stravolgere la soluzione o renderla impossibile. Nel
mondo reale le classi si sovrappongono quasi sempre. La risposta di Cortes e
Vapnik {cite}`cortes1995support` è il **margine morbido** (*soft margin*):
concedere qualche violazione, pagandola.

`````{tab} Elementare

Torniamo al confine tra i due quartieri. Se una singola villetta isolata sconfina
nel prato dell'altro, non ha senso stravolgere tutto il confine per accontentarla:
meglio tracciare comunque un bel corridoio largo e mettere in conto quella
manciata di eccezioni. La SVM a margine morbido fa proprio questo. Una
manopola, chiamata $C$, decide quanto è severa:

- $C$ **grande** = «non tollero errori»: il corridoio si stringe pur di far
  stare quasi tutti dalla parte giusta. Rischio di inseguire il rumore, cioè di
  **overfitting**.
- $C$ **piccolo** = «accetto qualche sbavatura»: il corridoio si allarga, più
  robusto, anche a costo di qualche punto dentro la fascia.

È lo stesso compromesso bias-varianza della sezione sull'overfitting, con la
manopola girata al contrario rispetto alla regolarizzazione: qui $C$ *grande*
significa freno *debole*.

`````

`````{tab} Superiore

Si introduce per ogni esempio una **variabile di slack** $\xi_i \ge 0$ che
misura di quanto quel punto viola il proprio margine, e la si somma nella
funzione obiettivo:

$$
\min_{W,\,b,\,\xi}\ \tfrac{1}{2}\lVert W\rVert^2 + C\sum_{i=1}^{m}\xi_i
\quad\text{soggetto a}\quad
y_i\,(W^\top X_i + b) \ge 1 - \xi_i,\ \ \xi_i \ge 0,
$$

dove $\xi_i$ è la violazione dell'$i$-esimo punto e $C > 0$ regola il
compromesso tra «corridoio largo» ($\lVert W\rVert$ piccolo) e «poche
violazioni» ($\sum\xi_i$ piccolo). Eliminando i vincoli, il problema si
riscrive come minimizzazione della **hinge loss** più un termine di
regolarizzazione:

$$
\min_{W,\,b}\ \sum_{i=1}^{m}\max\!\big(0,\ 1 - y_i\,(W^\top X_i + b)\big)
+ \frac{1}{2C}\lVert W\rVert^2 .
$$

La hinge loss $\max(0,\,1 - y_i f(X_i))$ è nulla per i punti ben classificati e
fuori dal margine, e cresce *linearmente* per quelli dentro la fascia o dalla
parte sbagliata: è l'analogo, per la SVM, di ciò che la log-loss è per la
regressione logistica — con la differenza che, essendo piatta oltre il margine,
ignora del tutto i punti «facili» e dà alla SVM la sua sparsità in vettori di
supporto. In questa forma si legge chiaramente il ruolo di $C$: il coefficiente
della penalità $\lVert W\rVert^2$ è $1/(2C)$, quindi **$C$ è l'inverso della
forza di regolarizzazione**. $C$ grande → penalità debole → margine stretto,
varianza alta; $C$ piccolo → penalità forte → margine largo, bias più alto. È
la stessa manopola $\lambda$ della sezione sull'overfitting, letta al contrario:
$\lambda \approx 1/(2C)$.

`````

## Il kernel trick: separare l'inseparabile

Fin qui, però, la SVM traccia solo iperpiani: frontiere *diritte*. E se le due
classi sono intrecciate in modo che nessuna retta le separi — pensa a un
bersaglio, con una classe al centro e l'altra tutt'intorno ad anello? Qui entra
in gioco l'idea più affascinante di tutta la storia delle SVM, quella che le ha
rese celebri: il **kernel trick**.

`````{tab} Elementare

Prendi il bersaglio: cerchio interno di una classe, anello esterno dell'altra.
Sul foglio, piatto, nessuna retta li separa. Ma immagina di *sollevare* ogni
punto in aria di un'altezza pari a quanto è lontano dal centro: i punti del
cerchio interno, vicini al centro, restano bassi; quelli dell'anello, lontani,
salgono in alto. Ora le due classi stanno a quote diverse, e un semplice
*piano orizzontale* — una lastra di vetro infilata a mezz'aria — le separa
nettamente. Non abbiamo cambiato i punti: li abbiamo guardati in uno spazio con
una dimensione in più, e lì il problema è diventato lineare.

`````

`````{tab} Superiore

L'idea è mappare ogni esempio in uno spazio di dimensione maggiore con una
funzione $\phi$, e cercare l'iperpiano *lì*. Per il bersaglio basta aggiungere
la feature $r^2 = x_1^2 + x_2^2$:

$$
\phi\,(x_1, x_2) = \big(x_1,\ x_2,\ x_1^2 + x_2^2\big).
$$

Nello spazio a tre dimensioni la classe interna (piccolo $r^2$) e quella esterna
(grande $r^2$) sono separate da un piano orizzontale a un'altezza-soglia. Il
problema, così com'è, sembra però costoso: se $\phi$ manda in uno spazio a
migliaia di dimensioni, calcolare e conservare tutti quei $\phi(X_i)$ diventa
proibitivo.

Il **kernel trick** è l'osservazione che salva tutto: nella formulazione duale
della SVM, gli esempi compaiono *solo* attraverso prodotti scalari
$\phi(X)^\top\phi(Z)$. Se esiste una funzione $k$ che calcola quel prodotto
scalare direttamente dalle coordinate originali,

$$
k(X, Z) = \phi(X)^\top \phi(Z),
$$

allora non serve mai costruire $\phi$: si lavora nello spazio ad alta dimensione
*senza mai visitarlo*. La funzione $k$ è il **kernel**
{cite}`scholkopf2002learning`. I più usati:

$$
\begin{aligned}
&\text{lineare:} && k(X,Z) = X^\top Z, \\
&\text{polinomiale:} && k(X,Z) = (X^\top Z + c)^{d}, \\
&\text{RBF / gaussiano:} && k(X,Z) = \exp\!\big(-\gamma\,\lVert X - Z\rVert^2\big),
\end{aligned}
$$

dove $d$ è il grado del polinomio, $c \ge 0$ un termine costante e $\gamma > 0$
l'ampiezza del kernel gaussiano. Il kernel RBF corrisponde a uno spazio $\phi$
di dimensione *infinita*: sarebbe impossibile da costruire, eppure $k$ si
calcola in una riga.

`````

```{figure} ../figures/svm-kernel-trick.svg
:name: fig-svm-kernel
:alt: "A sinistra, in due dimensioni, un disco di punti teal circondato da un anello di punti terracotta, non separabili da una retta; una freccia phi al centro. A destra, dopo la mappa, gli stessi punti giacciono su una parabola: gli interni in basso, gli esterni in alto, e una retta orizzontale tratteggiata li separa."
:width: 100%

Il kernel trick. A sinistra due classi non separabili da una retta nel piano.
Aggiungendo l'altezza $r^2 = x^2 + y^2$ (freccia $\phi$), a destra i punti si
«sollevano» e un semplice piano orizzontale li separa. Il kernel calcola i
prodotti scalari in quello spazio senza costruirlo mai.
```

Come illustra {numref}`fig-svm-kernel`, ciò che era un anello inseparabile
diventa, dopo la mappa, un problema lineare banale. Il parametro $\gamma$ del
kernel RBF merita un commento: è il **raggio d'influenza** di ogni punto.
Vediamolo con i numeri, scegliendo $\gamma = 0{,}5$:

- due punti *vicini*, $X=(2,2)$ e $Z=(3,3)$, distano
  $\lVert X-Z\rVert^2 = 1^2 + 1^2 = 2$, quindi
  $k(X,Z) = e^{-0{,}5\cdot 2} = e^{-1} \approx 0{,}37$: si «vedono» bene;
- due punti *lontani*, $X=(2,2)$ e $Z=(0,0)$, distano
  $\lVert X-Z\rVert^2 = 2^2 + 2^2 = 8$, quindi
  $k(X,Z) = e^{-0{,}5\cdot 8} = e^{-4} \approx 0{,}018$: quasi si ignorano.

Con $\gamma$ grande la campana si stringe, ogni punto influenza solo i vicinissimi
e la frontiera si fa frastagliata (varianza alta, rischio overfitting); con
$\gamma$ piccolo la campana si allarga, l'influenza è a lungo raggio e la
frontiera si liscia. Insieme a $C$, il parametro $\gamma$ è l'altra manopola da
tarare per validazione.

## Non solo classificare: la regressione con le SVM

Lo stesso principio si ribalta per la **regressione** (SVR, *Support Vector
Regression*). Nella classificazione la SVM vuole il corridoio più largo *tra*
le classi; nella regressione vuole un tubo che contenga *quanti più punti
possibile*.

`````{tab} Elementare

Invece di penalizzare ogni piccolo scarto tra previsione e valore vero — come
fa la regressione lineare classica — la SVR disegna un «tubo» di tolleranza
attorno alla curva: finché un punto ci sta dentro, l'errore conta *zero*.
Vengono penalizzati solo i punti che sporgono dal tubo, e solo per quanto
sporgono. È un modo indulgente di adattare i dati: non insegue le piccole
oscillazioni, si preoccupa solo degli scostamenti seri.

`````

`````{tab} Superiore

Si fissa una tolleranza $\epsilon > 0$ e si usa la **loss $\epsilon$-insensitive**,
nulla dentro il tubo e lineare fuori:

$$
L_\epsilon\big(y,\, f(X)\big) = \max\!\big(0,\ |y - f(X)| - \epsilon\big).
$$

Gli errori entro $\pm\epsilon$ non vengono penalizzati; oltre, la penalità
cresce linearmente. Il parametro $\epsilon$ fissa l'ampiezza del tubo, mentre
$C$ regola come sempre il compromesso tra piattezza del modello e violazioni.
Anche qui vale il kernel trick, così la SVR può adattare curve non lineari
esattamente come la SVM classifica frontiere non lineari.

`````

## Una classe sola: novelty e anomaly detection

C'è un'ultima variante, e risponde a una domanda diversa: e se avessimo esempi
di *una sola* classe? Vogliamo imparare com'è fatto il «normale» — transazioni
regolari, macchinari sani, traffico di rete legittimo — per poi accorgerci di
ciò che se ne discosta. È il problema della **novelty detection** (riconoscere
il nuovo) e dell'**anomaly detection** (riconoscere il guasto), e si lega a
quel tema dei dati fuori distribuzione toccato nella sezione sui dati che
cambiano: individuare gli input troppo lontani da ciò che il modello ha visto,
invece di predire con finta sicurezza.

`````{tab} Elementare

Immagina di aver visto migliaia di transazioni oneste con la carta di credito e
nemmeno una frode. Non puoi addestrare un classificatore «onesto contro frode»:
la seconda classe non ce l'hai. La **one-class SVM** ribalta il problema:
impara a disegnare, attorno ai dati normali, il «recinto» più stretto che li
racchiude tutti. Da quel momento, ogni nuova transazione che cade *fuori* dal
recinto è sospetta — non perché somigli a una frode nota, ma perché non somiglia
a nulla di normale. Una manopola, $\nu$, dice più o meno quale frazione di dati
ci aspettiamo che finisca fuori (le anomalie tollerate). Serve per rilevare
frodi, guasti di macchinari, intrusioni informatiche, difetti in una linea di
produzione: ovunque gli esempi «anomali» siano rari o non ancora visti.

`````

`````{tab} Superiore

La one-class SVM di Schölkopf e colleghi {cite}`scholkopf2001estimating`
adatta l'idea del margine al caso non supervisionato: mappati i dati nello
spazio delle feature con un kernel (di solito RBF), cerca l'iperpiano che
separa i punti dall'**origine** con il massimo margine. Ricondotto allo spazio
originale, questo equivale a racchiudere i dati normali in una regione compatta;
ciò che cade fuori è novità/anomalia. Il parametro $\nu \in (0,1]$ ha un doppio
significato preciso: è un limite *superiore* alla frazione di esempi di
addestramento classificati come anomali (i *margin error*) e un limite
*inferiore* alla frazione di vettori di supporto. La distingue dalla
classificazione binaria un'assenza: in addestramento **non** c'è la classe
«anomalo»: si impara solo la forma del normale. Un parente stretto è la
**Support Vector Data Description** (SVDD) di Tax e Duin, che invece della
separazione dall'origine cerca la *ipersfera* minima che racchiude i dati; e
tra le alternative non-kernel ci sono l'**Isolation Forest** — che isola le
anomalie con partizioni casuali, ereditando la scalabilità degli alberi della
sezione sugli ensemble — e il *Local Outlier Factor* basato sulla densità
locale.

`````

## In pratica, con scikit-learn

In scikit-learn la famiglia SVM vive nel modulo `sklearn.svm`: `SVC` per la
classificazione con kernel, `LinearSVC` per la versione lineare veloce, `SVR`
per la regressione, `OneClassSVM` per la novelty detection. Due avvertenze
valgono per tutte, e non sono opzionali.

**Standardizzare sempre le feature.** La SVM misura distanze: una feature con
scala molto più ampia domina il conto e schiaccia le altre, esattamente come
succede al k-NN. Si antepone quindi sempre uno `StandardScaler` in una
`Pipeline`, come nel pattern «In pratica, con scikit-learn» delle sezioni
precedenti.

**Attenzione ai numeri grandi.** L'addestramento di una SVM con kernel costa
circa tra $O(m^2)$ e $O(m^3)$ nel numero di esempi $m$: ottima da poche
centinaia a qualche decina di migliaia di punti, diventa proibitiva su milioni.
Per i dataset molto grandi si ripiega su modelli lineari (`LinearSVC`,
`SGDClassifier`, che scalano circa come $O(m)$) o sugli alberi in boosting della
sezione precedente {cite}`geron2019hands`.

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC, SVR, OneClassSVM

X, y = make_moons(n_samples=200, noise=0.20, random_state=0)

# Classificazione con kernel RBF: standardizzare SEMPRE (la SVM e' sensibile alla scala)
clf = make_pipeline(StandardScaler(),
                    SVC(kernel="rbf", C=1.0, gamma="scale"))
clf.fit(X, y)

# Variante lineare, veloce su molti esempi: niente kernel trick, costo ~O(m)
lin = make_pipeline(StandardScaler(), LinearSVC(C=1.0))
lin.fit(X, y)

# Regressione: il tubo epsilon-insensitive ignora gli scarti piccoli
reg = make_pipeline(StandardScaler(),
                    SVR(kernel="rbf", C=10.0, epsilon=0.1))
reg.fit(X, y.astype(float))

# One-class SVM: impara la regione dei dati "normali";
# nu ~ frazione di anomalie attese
normali = X[y == 0]                      # fingiamo di avere solo la classe "normale"
det = make_pipeline(StandardScaler(),
                    OneClassSVM(kernel="rbf", nu=0.05, gamma="scale"))
det.fit(normali)
esito = det.predict(X)                   # +1 = normale, -1 = anomalia
print("anomalie segnalate:", int(np.sum(esito == -1)))
```

La solita grammatica `fit`/`predict` regge anche qui. Per la SVM con kernel la
coppia di iperparametri da tarare per validazione è $(C, \gamma)$: una ricerca
su griglia con la cross-validation della sezione sull'overfitting è la prassi.

```{admonition} Da ricordare
:class: important
- La **SVM** sceglie, tra le infinite frontiere che separano due classi, quella
  a **margine massimo**: il corridoio $2/\lVert W\rVert$ più largo. La soluzione
  dipende solo dai **vettori di supporto**, i pochi punti sul bordo.
- Il **margine morbido** ammette violazioni $\xi_i$ pagate dal parametro $C$,
  l'**inverso** della forza di regolarizzazione: $C$ grande → margine stretto
  (overfitting), $C$ piccolo → margine largo. La perdita è la **hinge loss**,
  parente della log-loss ma piatta oltre il margine.
- Il **kernel trick** rende non lineare la SVM: mappa i dati in uno spazio più
  ampio dove diventano separabili, calcolando i prodotti scalari con un
  **kernel** $k(X,Z)=\phi(X)^\top\phi(Z)$ senza costruirlo. Kernel principali:
  lineare, polinomiale, RBF (parametro $\gamma$ = raggio d'influenza).
- La **SVR** regredisce con un tubo $\epsilon$-insensitive; la **one-class SVM**
  ($\nu$ = frazione di anomalie attese) impara la regione dei dati normali per
  la **novelty/anomaly detection**, senza vedere esempi anomali.
- In pratica: **standardizzare sempre** le feature; il costo $O(m^2)$–$O(m^3)$
  sconsiglia la SVM con kernel oltre le decine di migliaia di esempi.
```
