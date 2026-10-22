# Curve al posto di rette: spline e modelli additivi

«*A spline is a simple mechanical device for drawing smooth curves*»: uno
strumento meccanico, di quelli che stanno su un tavolo. Lo scrive nel 1946 il
matematico Isaac Jacob Schoenberg, in un lavoro di approssimazione numerica
{cite}`schoenberg1946contributions`. Lo strumento è un listello di legno lungo
e flessibile, che si posa sul foglio e si tiene fermo in qualche punto con dei
pesi, chiamati nel testo *dogs* o *rats*, finché prende la forma che si vuole
tracciare. Chi disegnava scafi di navi, fusoliere e carrozzerie ha lavorato
così fino agli anni Sessanta.

Fra un peso e l'altro il listello non si piega a caso, ma assume la curva che
richiede meno energia di flessione, perché il legno si oppone alla piega. Il
lavoro di Schoenberg parte da qui: finché le pendenze restano piccole
(l'approssimazione che lui stesso dichiara), la curva che il legno disegna e
quella che la statistica calcola sono la stessa.

Finora ai dati si sono adattate delle rette: una che li segue, nella
regressione, e una che li divide, nella classificazione. Quando i dati non
stanno su una retta, si può spezzare la linea a gradini, come fanno gli
{doc}`alberi decisionali
<alberi-ensemble>`, oppure tracciare la retta in uno spazio di caratteristiche
trasformate, come fanno le {doc}`macchine a vettori di supporto con il kernel
<svm-kernel>`. La strada più diretta, quella che verrebbe in mente a chiunque
abbia disegnato un grafico su un foglio, è tenere la retta e piegarla.

## Perché non basta alzare il grado

Il modo più immediato di piegare una retta è la **regressione polinomiale**,
che aggiunge alla matrice dei dati le colonne $x^2, x^3, \dots, x^p$ e risolve
lo stesso problema ai minimi quadrati della retta (in scikit-learn,
`PolynomialFeatures` davanti a `LinearRegression`). La potenza più alta, $p$, è
il **grado**: ad esempio, $2$ dà una parabola e $3$ una cubica. Finché il grado
resta basso il metodo funziona, ma al crescere del grado l'errore esplode
vicino agli estremi dell'intervallo, anche quando al centro la curva è
perfetta. È il **fenomeno di Runge**, descritto da Carl Runge nel 1901
{cite}`runge1901empirische`.

`````{tab} Elementare

Un palazzo di ventun appartamenti, una caldaia sola in cantina, e l'acqua calda
che parte da lì e passa per tutti i termosifoni. Ogni inquilino ha la sua idea
di temperatura giusta, e le idee sono ventuno. Chi regola l'impianto ha pochi
comandi, e ognuno agisce sul palazzo intero. Apre per il quinto piano e bolle
anche il terzo; allora strozza la colonna per rimediare, e si raffredda il
secondo; ogni rimedio ne chiede un altro.

Ai piani di mezzo i rimedi si accavallano e finiscono per compensarsi, e lì la
temperatura viene giusta al grado. Ai due capi della colonna, invece, in cima e
a pianterreno, non c'è più niente che compensi. Uno gela e l'altro bolle, e non
di un grado o due. In cima si sta col cappotto. Più si pretende di azzeccare la
temperatura di ognuno, peggio stanno quei due; e se un solo inquilino cambia un
radiatore, tutta la regolazione va rifatta da capo.

Nel polinomio gli inquilini sono i punti, i comandi sono i coefficienti, e ogni
coefficiente cambia la curva dappertutto, come ogni comando cambia l'acqua di
tutto il palazzo. È il fenomeno di Runge.

Ad esempio, prendi una curva a campana molto stretta, piazza dei punti a
distanze uguali lungo l'asse, e chiedi a un polinomio di passare per tutti i
punti. Al centro il polinomio è impeccabile, ma ai bordi impazzisce, e più
punti gli dai peggio va: con cinque punti sbaglia al massimo di $0{,}4$, con
ventuno sbaglia di quasi $60$, su una curva che non supera mai $1$.

Per questa campana basta infittire i punti verso i bordi, invece di tenerli a
distanze uguali. Nei dati veri, però, non si decide dove cadono i punti, e il
problema di fondo resta: una formula sola vale per tutto il campo, e ogni
coefficiente in più piega la curva dappertutto, anche dove andava bene
com'era. L'errore esplode ai bordi, dove i punti stanno tutti da una parte sola
e niente tiene ferma la curva dall'altra.

`````

`````{tab} Superiore

La regressione polinomiale $y = \theta_0 + \theta_1 x + \theta_2 x^2 + \dots$ è
ancora un modello lineare, perché è lineare nei *parametri* $\theta_j$ e non
in $x$; resta quindi un problema ai minimi quadrati con una matrice di disegno
più larga. Le difficoltà sono due, e distinte. La prima è che la base dei
monomi è mal condizionata (il numero di condizione della matrice di disegno
cresce esponenzialmente col grado, e si cura cambiando base, per esempio con i
polinomi di Čebyšëv); la seconda è che i polinomi hanno supporto globale,
qualunque sia la base.

L'interpolazione polinomiale su nodi equispaziati non converge uniformemente per
ogni funzione continua. L'esempio canonico di Runge è
$f(x) = 1/(1+25x^2)$ su $[-1,1]$. Detto $p_{n}$ il polinomio di grado $n-1$ che
interpola $f$ su $n$ nodi equispaziati, si ha

$$
\lim_{n \to \infty} \max_{x \in [-1,1]} \bigl|f(x) - p_n(x)\bigr| = \infty ,
$$

con la divergenza concentrata vicino a $x = \pm 1$. Il codice calcola l'errore
massimo del polinomio interpolante e, accanto, quello della spline cubica
naturale sugli stessi nodi (la definizione di spline sta nella sezione sugli
archi cubici).

```python
import numpy as np
from scipy.interpolate import CubicSpline

def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

xf = np.linspace(-1, 1, 2001)
for nodi in (5, 9, 15, 21):
    xn = np.linspace(-1, 1, nodi)                       # nodi equispaziati
    c = np.polyfit(xn, runge(xn), nodi - 1)             # interpolazione esatta
    err = np.abs(np.polyval(c, xf) - runge(xf))
    spline = CubicSpline(xn, runge(xn), bc_type="natural")
    err_s = np.abs(spline(xf) - runge(xf))
    print(f"nodi={nodi:3d}  errore max: polinomio = {err.max():8.3f}  "
          f"spline cubica naturale = {err_s.max():.4f}")
```

```text
nodi=  5  errore max: polinomio =    0.438  spline cubica naturale = 0.2793
nodi=  9  errore max: polinomio =    1.045  spline cubica naturale = 0.0561
nodi= 15  errore max: polinomio =    7.195  spline cubica naturale = 0.0025
nodi= 21  errore max: polinomio =   59.822  spline cubica naturale = 0.0032
```

Osserviamo che il problema di interpolazione su nodi equispaziati è mal
condizionato. La costante di Lebesgue $\Lambda_n$ dell'interpolazione di grado
$n$, cioè il fattore massimo con cui l'interpolazione amplifica un errore sui
valori, cresce infatti esponenzialmente,
$\Lambda_n \sim 2^{n+1}/(e\,n\log n)$, mentre per i nodi di Čebyšëv
$\cos(k\pi/n)$, che si addensano ai bordi, cresce come $\tfrac{2}{\pi}\log n$.

Che per la funzione di Runge l'errore diverga, e non soltanto si amplifichi,
dipende anche dalla sua analiticità: i poli in $\pm i/5$ cadono fuori dalla
regione del piano complesso in cui l'interpolazione equispaziata converge, e il
polinomio diverge per $|x| \gtrsim 0{,}726$ {cite}`trefethen2013approximation`.

In definitiva, cambiare la posizione dei nodi cura questo esempio, e cambiare
base cura il condizionamento, ma nessuno dei due cura il problema di fondo: un
polinomio ha **supporto globale**, quindi ogni coefficiente influenza la curva
su tutto il dominio, e la stima ai minimi quadrati non ha modo di localizzare
la flessibilità.

`````

Con dati rumorosi la misura si fa con la cross-validation, cioè su punti che il
modello non ha visto in addestramento. L'errore stampato è quello commesso al
centro dell'intervallo, nel quinto esterno (i bordi) e su tutti i punti. Le
spline, indicate con il numero dei loro nodi, fanno da termine di paragone: più
nodi, curva più flessibile.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

rng = np.random.default_rng(0)
x = np.sort(rng.uniform(-1, 1, 120))
y = 1.0 / (1.0 + 25.0 * x**2) + rng.normal(0, 0.05, x.size)   # la curva di Runge
X = x.reshape(-1, 1)

bordo = np.abs(x) > 0.8          # il quinto esterno del campo
pieghe = KFold(n_splits=5, shuffle=True, random_state=0)

def errori(modello):
    """MSE fuori campione: al centro, ai due bordi, e su tutti i punti."""
    res = np.empty_like(y)
    for tr, te in pieghe.split(X):
        res[te] = modello.fit(X[tr], y[tr]).predict(X[te])
    e = (res - y) ** 2
    return e[~bordo].mean(), e[bordo].mean(), e.mean()

print(f"{'modello':<26}{'centro':>10}{'bordi':>10}{'tutti':>10}")
for g in (9, 15, 21):
    c, b, t = errori(make_pipeline(PolynomialFeatures(g), LinearRegression()))
    print(f"polinomio grado {g:<10d}{c:10.4f}{b:10.4f}{t:10.4f}")
for n in (8, 12, 20):
    c, b, t = errori(make_pipeline(SplineTransformer(n_knots=n, degree=3),
                                   LinearRegression()))
    print(f"spline cubica {n:2d} nodi{'':<6}{c:10.4f}{b:10.4f}{t:10.4f}")
print(f"\npunti al bordo: {bordo.sum()} su {len(x)}; rumore vero: {0.05**2:.4f}")
```

```text
modello                       centro     bordi     tutti
polinomio grado 9             0.0088    0.0054    0.0078
polinomio grado 15            0.0034    0.0049    0.0038
polinomio grado 21            0.0034    0.0280    0.0104
spline cubica  8 nodi          0.0096    0.0029    0.0077
spline cubica 12 nodi          0.0036    0.0029    0.0034
spline cubica 20 nodi          0.0033    0.0031    0.0032

punti al bordo: 34 su 120; rumore vero: 0.0025
```

Il numero da guardare è lo $0{,}0280$ della terza riga. Passando dal grado 15
al grado 21 il polinomio non peggiora affatto al centro ($0{,}0034$ tutte e due
le volte) e peggiora di quasi sei volte ai bordi, arrivando a undici volte il
rumore presente nei dati. La spline, invece, con lo stesso aumento di
flessibilità (da 12 a 20 nodi), resta stabile al centro e ai bordi.

Al grado 15 il polinomio va bene, ma la fascia dei gradi accettabili è
stretta, e appena la si supera gli errori grandi compaiono ai bordi. Si noti
che nella media su tutti i punti il peggioramento si vede (da $0{,}0038$ a
$0{,}0104$), ma solo la separazione fra centro e bordi dice dove avviene.

## Tanti archi cubici invece di una formula sola

Il rimedio è rinunciare a una formula sola. Si divide l'asse con dei punti di
giunzione, detti **nodi**, si usa un polinomio di grado basso fra un nodo e
l'altro, e si raccordano i pezzi nei nodi. La curva che ne esce è una
**spline**, nome che viene dal listello del tavolo da disegno, che fra un peso
e l'altro si dispone in tanti archi cubici raccordati.

`````{tab} Elementare

Sul tavolo da disegno il listello poggia su dei pesi, piantati come paletti, e
fra un paletto e il successivo forma un arco. Il paletto che separa due archi è
il nodo. Ogni arco si piega come una cubica, la curva più semplice capace di
avere un massimo, un minimo e un punto di flesso, quanto basta per fare una
gobba e cambiare idea. Se gli archi fossero listelli separati, appoggiati uno di
seguito all'altro, resterebbero indipendenti e sotto ogni peso si vedrebbe lo
scalino.

La giuntura si collauda col dito. Lo passi sopra il peso e non deve sentire
niente: nessun gradino (i due pezzi si toccano), nessuno spigolo (arrivano con
la stessa pendenza), nessuno scatto (piegano dalla stessa parte e con la stessa
forza). Tre pretese su ogni peso, e la giuntura sparisce anche alla vista:
l'occhio non sa dire dove finisce un arco e comincia l'altro. Sul foglio resta
una spline cubica.

Una gobba in più in fondo a destra si ottiene aggiungendo un peso a destra, e
gli archi di sinistra restano quasi dov'erano, mentre con una formula sola per
tutto il campo si muoveva tutto.

Quanta libertà resti alla curva si conta con le dita. Tre pesi fanno quattro
tratti; ogni tratto è una cubica, e una cubica per stare ferma vuole quattro
numeri, quindi sedici numeri in tutto. Ogni peso ne blocca tre, uno per pretesa,
e i pesi sono tre: nove bloccati, sette liberi. Restano sette gradi di libertà,
e ne viene uno in più per ogni peso che si aggiunge.

Il legno rispetta le tre pretese e il grado tre senza che nessuno gliele
imponga. Schoenberg le ricava dalla teoria elementare della trave elastica,
nella pagina stessa in cui descrive lo strumento: un'asta appoggiata su dei
pesi si dispone in archi cubici, che si raccordano con la stessa pendenza e la
stessa curvatura, e i raccordi cadono sotto i pesi. I paletti sul foglio sono i
suoi pesi.

Resta il pezzo di listello che sporge oltre l'ultimo peso. Là il foglio è vuoto,
non ci sono dati e niente lo trattiene: un listello libero in quel tratto non
ha nulla che lo pieghi, e prosegue dritto come una retta oltre gli estremi. Un
listello montato così si dice **naturale**; è la scelta di riferimento perché
rende noiosi i bordi, dove i modelli fanno i danni.

Sul foglio di un modello, però, non c'è il legno a raddrizzare la punta, quindi
la rettilineità va scritta come condizione: due condizioni a ogni capo, e dei
sette gradi di libertà ne restano tre, uno per paletto.

`````

`````{tab} Superiore

Fissati i nodi interni $\xi_1 < \dots < \xi_K$ nel dominio, una **spline di
grado $q$** è una funzione che su ogni intervallo fra due nodi consecutivi è un
polinomio di grado $\le q$, e che in ogni nodo è continua insieme alle derivate
fino alla $(q-1)$-esima. In particolare, per $q = 3$ (il caso quasi universale)
le condizioni di raccordo sono continuità di $f$, $f'$ e $f''$.

Il caso $q=3$ viene direttamente dal listello. Schoenberg, infatti, deriva la
definizione dalla meccanica dell'asta appoggiata, con una linearizzazione
dichiarata: se la curva è quasi parallela all'asse $x$ si trascura $y'$, e la
curvatura $1/R = y''/(1+y'^2)^{3/2}$ si riduce a $y''$. Ne conclude che la
forma assunta è una
linea composta di archi cubici raccordati con derivata prima e seconda
continue, con i raccordi dove poggiano i pesi. La definizione formale che segue
nel suo articolo è la generalizzazione di quel fatto meccanico a un grado
qualunque.

Il conteggio dei gradi di libertà rende la costruzione trasparente. Con $K$ nodi
interni ci sono $K+1$ intervalli, ciascuno con una cubica a 4 coefficienti, per
un totale di $4(K+1)$ parametri liberi. Poiché ogni nodo impone 3 vincoli e i
nodi sono $K$, restano

$$
4(K+1) - 3K = K + 4
$$

gradi di libertà. Per questo motivo una spline cubica con $K$ nodi si scrive
come combinazione di $K+4$ funzioni di base. La spline cubica naturale aggiunge
il vincolo $f'' = f''' = 0$ oltre i due nodi estremi, cioè 2 vincoli per
estremo, e scende a $K$ gradi di libertà {cite}`hastie2009elements`.

La base che si usa in pratica non è quella dei polinomi troncati $(x -
\xi_k)_+^3$, numericamente pessima, ma quella delle **B-spline**, le funzioni
di base a **supporto locale** costruite da Schoenberg: ogni $B_{k,3}$ è diversa
da zero solo su quattro intervalli consecutivi.

Da qui deriva tutto il vantaggio sul polinomio globale, nel quale ogni base è
diversa da zero ovunque: la matrice di disegno è a banda, la base resta ben
condizionata al crescere di $K$ (il suo numero di condizione dipende dal grado
e non dal numero di nodi), e un coefficiente sposta la curva solo nel suo
pezzo di dominio.

La matrice di disegno dipende però anche dai dati: ha rango pieno se e solo se
fra i dati si possono scegliere $K+4$ punti in ordine crescente, l’$i$-esimo
nel supporto della $i$-esima funzione di base (la condizione di Schoenberg e
Whitney). Di conseguenza, con nodi in zone senza dati, o più nodi che dati, il
problema ai minimi quadrati torna singolare.

In scikit-learn questa base si ottiene con `SplineTransformer`, che trasforma
una colonna in $K + q + 1$ colonne di B-spline valutate sui dati, cioè i $K+4$
gradi di libertà della spline cubica (si noti che il suo `n_knots` conta anche
i due nodi di bordo, quindi con `n_knots=8` e grado 3 le colonne sono dieci).
Quelle colonne sommano a $1$ in ogni punto, perché le B-spline sono una
partizione dell'unità; la costante è quindi inclusa, e l'intercetta del modello
lineare successivo è ridondante. A parte questo, resta un ordinario problema ai
minimi quadrati.

`````

Con le spline, quindi, la scelta del grado diventa la scelta dei nodi: quanti e
dove. Per il «dove», l'uso è metterli ai quantili della variabile, cioè più
fitti dove ci sono più dati (`SplineTransformer(knots="quantile")`; il default
li mette equispaziati); quanto al «quanti», è un iperparametro come gli
altri, che si sceglie con la cross-validation di
{doc}`Overfitting e validazione <overfitting-validazione>`. C'è anche una
terza via, che evita questa scelta.

## Smoothing spline: la manopola che irrigidisce il legno

Invece di decidere quanti nodi usare, se ne mette uno per dato e si penalizza
la curvatura. È l'idea della **smoothing spline**. Il listello dei cantieri ne
è il caso limite: tenuto fermo dai pesi passa per tutti i punti, e la sua
rigidità non cambia la forma; il compromesso fra fedeltà ai dati e curvatura è
quello di un listello agganciato ai punti con degli elastici, e il rapporto fra
la rigidità del legno e quella degli elastici è il $\lambda$ della smoothing
spline.

Ancora Schoenberg, nel 1964, dimostra che il minimo di quel compromesso
è una spline {cite}`schoenberg1964graduation`; tre anni dopo,
Christian Reinsch la rende un algoritmo che gira {cite}`reinsch1967smoothing`.

`````{tab} Elementare

Adesso si tolgono i pesi che inchiodano il listello ai punti, e in ogni punto lo
si aggancia con un elastico. Il listello fa allora due cose in contrasto fra
loro. Da un lato deve passare vicino ai punti, e ogni elastico lo tira dalla
sua parte, tanto più forte quanto più è teso. Dall'altro il legno si oppone
alla piega: piegarlo richiede fatica, tanto più quanto più stretta è la piega.

La curva che il listello disegna è il compromesso fra questi due tiri, cioè
quella che spende meno fatica in totale. Con i pesi la rigidità non conterebbe
niente: il listello passerebbe per tutti i punti comunque, duro o morbido che
sia.

Nessuno deve dire al listello che forma prendere: non sa che cosa siano i tratti
e i paletti, eppure ne esce una curva a tratti raccordati, con i raccordi in
corrispondenza dei punti, e ne esce una sola, perché fra tutte le forme
possibili ce n'è una di minima fatica.

La smoothing spline scrive quel compromesso in una formula sola, con una
manopola in mezzo: quanto è rigido il legno rispetto agli elastici.

- Gira la manopola verso il morbido e ottieni uno **spago**: passa per tutti i
  punti, uno per uno, e fra un punto e l'altro fa quello che vuole. Ha imparato
  anche il rumore.
- Gira la manopola verso il rigido e ottieni un **righello di acciaio**: non si
  piega affatto, e dei punti si limita a stare in mezzo. È tornato a essere la
  retta di partenza.

La posizione della manopola si traduce in un numero, come per i paletti: i
gradi di libertà che restano alla curva. Verso il morbido quel numero resta
alto, mentre verso il rigido scende a $2$, che è la libertà di una retta: uno
per dire quanto è alta, uno per dire quanto pende.

`````

`````{tab} Superiore

Fra tutte le funzioni $f$ due volte derivabili, la smoothing spline è quella che
minimizza

$$
\mathcal{L}(f) = \sum_{i=1}^{m} \bigl(y_i - f(x_i)\bigr)^2
\;+\; \lambda \int f''(t)^2 \, dt ,
$$

dove il primo termine è la fedeltà ai dati e il secondo la penalità di
curvatura, pesata da $\lambda \ge 0$. Il risultato, non ovvio, è che il
minimo di questo problema su uno spazio di funzioni di dimensione infinita
esiste, è unico ed è una spline cubica naturale con un nodo in ogni $x_i$
distinto: la forma non si impone, si ottiene.

I due estremi si leggono nella formula. Per $\lambda \to 0$ la penalità sparisce
e resta l'interpolazione; per $\lambda \to \infty$ l'unico modo di tenere finito
il costo è $f'' \equiv 0$, cioè $f$ affine, e il primo termine sceglie fra le
rette quella dei minimi quadrati.

Poiché i nodi sono fissati ai dati, la complessità non si governa contandoli ma
misurandola. Osserviamo che la stima è lineare nei dati, $\hat{\mathbf{y}} =
\mathbf{S}_{\lambda}\, \mathbf{y}$ con $\mathbf{S}_{\lambda}$ la *matrice di
lisciamento*, e si definiscono gradi di libertà effettivi

$$
\mathrm{df}(\lambda) = \operatorname{tr}(\mathbf{S}_{\lambda}) ,
$$

che interpolano con continuità fra $m$ (interpolazione) e $2$ (retta). Sono la
generalizzazione naturale del «numero di parametri»: per un modello lineare
ordinario con $d$ colonne la matrice di proiezione ha traccia pari a $d$,
quindi la definizione non è una convenzione nuova ma la stessa di sempre,
letta in un caso in cui i parametri non si contano.

La stessa linearità rende quasi gratuita la scelta di $\lambda$. L'errore
*leave-one-out*, infatti, non richiede $m$ adattamenti, perché vale l'identità

$$
\mathrm{CV}(\lambda) = \frac{1}{m}\sum_{i=1}^{m}\left(\frac{y_i - \hat{f}_\lambda(x_i)}{1 - S_{\lambda,ii}}\right)^2 ,
$$

dove $S_{\lambda,ii}$ è l’$i$-esimo elemento diagonale di $\mathbf{S}_\lambda$
{cite}`hastie2009elements`; la *validazione incrociata generalizzata* (GCV)
sostituisce ogni $S_{\lambda,ii}$ con la media $\mathrm{df}(\lambda)/m$, che è
anche il criterio con cui `make_smoothing_spline` di SciPy sceglie $\lambda$
quando non glielo si passa. Infine, nella base B-spline il sistema da risolvere
è a banda, quindi il costo è lineare in $m$ e un nodo per dato diventa
praticabile.

`````

Il codice adatta la smoothing spline per sei valori di $\lambda$ e stampa, per
ciascuno, i gradi di libertà effettivi $\mathrm{df}(\lambda) =
\operatorname{tr}(\mathbf{S}_\lambda)$ e l'errore quadratico medio rispetto alla
curva vera, che su dati simulati si conosce. La traccia si calcola un dato per
volta: la spline adattata a un vettore che vale $1$ nel solo dato $j$ e $0$
altrove, letta in $x_j$, è l'elemento $S_{\lambda,jj}$, cioè di quanto si
sposta la curva in $x_j$ se quel dato, e solo quello, si sposta di uno.
($10^{-8}$, un centomilionesimo, si stampa `1e-08`.)

```python
import numpy as np
from scipy.interpolate import make_smoothing_spline

rng = np.random.default_rng(0)
x = np.sort(rng.uniform(-1, 1, 120))
vera = 1.0 / (1.0 + 25.0 * x**2)
y = vera + rng.normal(0, 0.05, x.size)

def gdl(lam):
    """Gradi di libertà effettivi: la traccia della matrice che manda y in ŷ."""
    tr = 0.0
    for j in range(len(x)):
        e = np.zeros(len(x)); e[j] = 1.0
        tr += make_smoothing_spline(x, e, lam=lam)(x[j])
    return tr

print(f"{'lambda':>10}{'gdl effettivi':>16}{'errore vs curva vera':>24}")
for lam in (1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e3):
    s = make_smoothing_spline(x, y, lam=lam)
    print(f"{lam:10.0e}{gdl(lam):16.1f}{np.mean((s(x) - vera)**2):24.5f}")

# l'ultima riga da vicino: quella curva quanto e' davvero una retta?
retta = np.polyval(np.polyfit(x, y, 1), x)
scarto = np.abs(make_smoothing_spline(x, y, lam=1e3)(x) - retta).max()
print(f"\na lambda=1e3 i gdl valgono {gdl(1e3):.4f}, e la curva si scosta dalla")
print(f"retta dei minimi quadrati al massimo di {scarto/np.ptp(y):.5f} "
      f"dell'ampiezza dei dati")
```

```text
    lambda   gdl effettivi    errore vs curva vera
     1e-08            97.0                 0.00217
     1e-06            53.7                 0.00102
     1e-04            20.2                 0.00031
     1e-02             7.2                 0.00312
     1e+00             3.0                 0.03214
     1e+03             2.0                 0.07755

a lambda=1e3 i gdl valgono 2.0026, e la curva si scosta dalla
retta dei minimi quadrati al massimo di 0.00076 dell'ampiezza dei dati
```

La prima colonna è l'iperparametro $\lambda$: piccolo in alto (legno
morbido), grande in fondo (legno rigido). La seconda è il numero che dice
quanto è flessibile la curva che ne esce, e ha un nome, **gradi di libertà
effettivi**.

Nell'ultima riga, con il legno più rigido, i gradi di libertà arrivano a
$2{,}0$, quelli di una retta: uno per l'altezza, uno per la pendenza. Il valore
esatto delle due righe in fondo è $2{,}0026$ e non $2$, perché la retta si
raggiunge solo quando l'iperparametro tende all'infinito, e a ogni valore
finito resta un filo di flessibilità. La curva si scosta dalla retta dei minimi
quadrati, nel punto peggiore, di otto decimillesimi dell'ampiezza dei dati,
meno del tratto di matita con cui la si disegnerebbe.

L'errore rispetto alla curva vera non scende sempre: tocca il minimo a
$\lambda = 10^{-4}$ e risale da tutte e due le parti. È la U del compromesso
bias-varianza, già vista in {doc}`Overfitting e validazione
<overfitting-validazione>`, disegnata qui da un iperparametro continuo invece
che da un numero intero di nodi.

Fra $\lambda = 10^{-8}$ e $\lambda = 10^{-4}$ i gradi di libertà passano da $97$
a $20$, un taglio di quasi quattro quinti, e l'errore migliora di sette volte.
In altri termini, quei $77$ gradi di libertà stavano descrivendo il rumore.

La {numref}`fig-spline-tavolette` mette a confronto il polinomio, la spline
cubica naturale e la smoothing spline.

```{figure} ../figures/spline-tavola-e-tavolette.svg
:name: fig-spline-tavolette
:alt: "Tre pannelli. Nei primi due gli stessi quindici punti equispaziati presi su una curva a campana, disegnata tenue sotto. Nel primo il polinomio di grado quattordici che passa per tutti e quindici i punti: al centro segue la campana, ma verso le due estremita oscilla sempre piu forte e finisce fuori dal riquadro, come segnalano due frecce. Nel secondo la spline cubica naturale sugli stessi punti resta aderente alla campana per tutta la sua lunghezza. Il terzo pannello ha altri dati, quaranta punti rumorosi, e mostra la stessa curva a tre rigidita crescenti: una in ocra passa quasi per ogni punto e ondeggia, una in terracotta segue la campana, la terza tratteggiata e indistinguibile da una retta."
:width: 100%

A sinistra e al centro, lo stesso compito e la stessa scala: far
passare una curva per quindici punti presi su una campana (in grigio tenue). Il
polinomio ci riesce, e nel farlo scappa fuori dal riquadro alle due estremità
(errore massimo $7{,}19$); la spline cubica naturale resta aderente per tutta la
lunghezza (errore massimo $0{,}0025$, quasi tremila volte meno). A destra
servono dati rumorosi, perché è su dati così che ha senso regolare la rigidità,
dallo spago che insegue ogni punto al righello che non si piega.
```

Il pannello di destra di {numref}`fig-spline-tavolette` mostra una
regolarizzazione molto vicina alla Ridge di {doc}`Overfitting e validazione
<overfitting-validazione>`. Infatti, scritta nella base B-spline,
$f = \sum_j \theta_j B_j$ (i $\theta_j$ sono i pesi delle funzioni di base), la
penalità di curvatura diventa una forma quadratica nei pesi, $\int f''^2 =
\boldsymbol{\theta}^\top\boldsymbol{\Omega}\,\boldsymbol{\theta}$ con
$\Omega_{jk} = \int B_j''\,B_k''$. Si tratta quindi di una Ridge
*generalizzata* {cite}`hastie2009elements`, che al posto di
$\lVert\boldsymbol{\theta}\rVert^2$ penalizza
$\boldsymbol{\theta}^\top\boldsymbol{\Omega}\,\boldsymbol{\theta}$. La Ridge
tira tutti i pesi verso lo zero, mentre la penalità di curvatura lascia liberi
quelli che descrivono una retta, su cui vale zero, e frena solo le combinazioni
che piegano la curva.

(sec-regressione-a-nucleo)=
## Una media che scivola: la regressione a nucleo

La smoothing spline si ottiene con un'unica ottimizzazione su tutti i dati. La
funzione, però, si può anche stimare punto per punto. Per stimarla in $x_0$ si
fa la media pesata delle $y_i$, con pesi che decrescono con la distanza
$\lvert x_i - x_0\rvert$ secondo una funzione a campana centrata in $x_0$, così
che i vicini contino molto e i lontani quasi niente. È la **regressione a
nucleo**, proposta indipendentemente da Èlizbar Nadaraya e da Geoffrey Watson
nel 1964 {cite}`nadaraya1964estimating,watson1964smooth`.

La campana si chiama **nucleo**, in inglese *kernel*, ed è parente stretta del
kernel che tornerà nella {doc}`sezione sul kernel trick <svm-kernel>`. La sua
larghezza, $h$, prende il nome di **larghezza di banda** (*bandwidth*), e
regola la flessibilità come nella smoothing spline faceva $\lambda$.

`````{tab} Elementare

Per stimare quanto vale una casa di 80 metri quadri si guardano le case vendute.
Quelle di 79 o di 81 metri contano quasi come se fossero la casa da stimare;
quelle di 70 un po’ meno; quelle di 150 non dicono niente. La stima è la media
dei prezzi, dove ogni prezzo pesa tanto quanto la sua casa somiglia per
superficie a quella da stimare: ogni prezzo si moltiplica per il suo peso, e il
totale si divide per la somma dei pesi invece che per il numero delle case.

Ad esempio, con due case da 79 e 81 metri vendute a 200 e 210 mila euro, che
contano per intero (arrotondando il loro «quasi»), e una da 70 metri venduta a
180 mila, che conta per metà, il calcolo dà
$(200 + 210 + 0{,}5 \times 180) / (1 + 1 + 0{,}5) = 500 /
2{,}5 = 200$ mila euro. Poi si fa lo stesso per 81 metri, per 82, e così via: la
finestra dei «simili» scivola lungo i metri quadri, e la stima disegna una curva
({numref}`fig-media-che-scivola`). È una finestra dai bordi sfumati, e nella
figura il peso che cala ha la forma di una campana.

Tutta la questione è quanto fare larga quella finestra. Se si considerano
simili solo le case identiche, ogni stima si appoggia su una o due vendite e
salta qua e là a ogni affare fortunato. Se si considerano simili tutte le
case fino a cento metri di differenza, la stima diventa la media di tutto il
quartiere, la stessa dappertutto, e perde la forma che si cercava.

In mezzo c'è una larghezza che sbaglia meno delle altre. È lo stesso
compromesso fra una risposta nervosa e una troppo piatta che nel listello
regolava la manopola della rigidità. Per trovarla non serve conoscere il
prezzo giusto: si nasconde a turno ogni casa venduta, la si stima con le altre,
e si tiene la larghezza con cui quelle stime cadono più vicino ai prezzi veri.

Con più vendite la finestra si può stringere, perché anche stretta ne trova
abbastanza, e la stima migliora, ma più lentamente di quanto farebbe se si
sapesse già che il prezzo sale in linea retta con i metri quadri.

C'è un punto in cui la media inganna: il bordo. Per la casa più piccola del
quartiere tutte le vicine sono più grandi, e di solito più care; la finestra le
prende solo da una parte, e la media tira la stima verso l'alto, tanto più
quanto più è larga la finestra e ripida la discesa dei prezzi verso le case
piccole.

Il rimedio è tracciare, dentro la finestra, un piccolo pezzo di retta che passi
il più vicino possibile ai prezzi, contando di più le case più simili, e
leggere la stima sulla retta, all'altezza della casa da stimare. La retta,
infatti, si accorge che i prezzi vanno giù, e lo tiene in conto anche dove i
vicini stanno tutti da un lato.

Questa correzione vale però solo se la finestra non è troppo stretta: una retta
tirata su due o tre case sole punta dove vuole, e al bordo sbaglia più della
media che doveva correggere.

La finestra, infine, non basta più quando le caratteristiche sono tante. Case
simili per superficie ce ne sono sempre; case simili per superficie, piano,
anno, quartiere, esposizione e dieci altre cose insieme non ce ne sono quasi
mai, e la finestra resta vuota.

`````

`````{tab} Superiore

Lo stimatore di Nadaraya-Watson, con un nucleo $K$ (qui la gaussiana,
$K(u) = e^{-u^2/2}/\sqrt{2\pi}$) e larghezza di banda $h$, è

$$
\hat f(x_0) = \frac{\sum_{i=1}^{m} K_h(x_0 - x_i)\, y_i}{\sum_{i=1}^{m} K_h(x_0 - x_i)},
\qquad K_h(u) = \frac{1}{h}K\!\left(\frac{u}{h}\right),
$$

cioè la soluzione di $\min_{c}\sum_i K_h(x_0 - x_i)(y_i - c)^2$: un modello
costante adattato con pesi locali. Sostituendo la costante con una retta,
$\min_{a,b}\sum_i K_h(x_0 - x_i)\,\big(y_i - a - b(x_i - x_0)\big)^2$ e $\hat
f(x_0) = \hat a$, si ottiene la **regressione lineare locale** (la base del
LOWESS di Cleveland {cite}`cleveland1979robust`).

All'interno la distorsione di entrambe è $O(h^2)$ e la varianza
$O\big(1/(mh)\big)$, quindi la banda ottima va come $m^{-1/5}$ e l'errore
quadratico come $m^{-4/5}$, più lento del $m^{-1}$ di un modello parametrico
corretto. Sono risultati asintotici, per $h \to 0$ e $mh \to
\infty$, con $f$ due volte derivabile con continuità, densità degli $x_i$
positiva attorno a $x_0$ e rumore di varianza finita; il termine della
distorsione proporzionale a $h\,f'(x_0)$ all'interno sparisce perché il nucleo è
simmetrico. Più in dettaglio, con $p$ la densità degli $x_i$, $\mu_2 = \int
u^2K(u)\,du$ e $R(K) = \int K^2$, all'interno valgono

$$
\begin{gathered}
\operatorname{bias}_{\mathrm{NW}}(x_0) \approx h^2\mu_2\Bigl(\tfrac12 f''(x_0) + \frac{f'(x_0)\,p'(x_0)}{p(x_0)}\Bigr),
\\
\operatorname{bias}_{\mathrm{LL}}(x_0) \approx \tfrac12 h^2\mu_2 f''(x_0),
\qquad
\operatorname{Var}(x_0) \approx \frac{\sigma^2 R(K)}{m\,h\,p(x_0)},
\end{gathered}
$$

dove LL è la retta locale e $\sigma^2$ la varianza del rumore, la stessa per i
due stimatori. Il termine $f'p'/p$, detto **distorsione di progetto**, c'è
anche all'interno quando i dati non sono distribuiti in modo uniforme: la media
pesata la subisce, la retta locale no, e per questo la regressione lineare
locale è preferibile anche lontano dai bordi {cite}`fan1992design`.

Al bordo, invece, i dati stanno da una parte sola, il termine in $h\,f'(x_0)$
non si cancella più, e la distorsione di Nadaraya-Watson sale a $O(h)$ (nel
punto estremo, con nucleo gaussiano e dati uniformi, circa
$0{,}8\,h\,\lvert f'(x_0)\rvert$), mentre quella lineare locale resta
$O(h^2)$: la retta corregge al primo ordine l'asimmetria del vicinato
{cite}`hastie2009elements`.

In cambio cresce la varianza, che nel punto estremo, con le stesse ipotesi, è
circa tripla di quella della media pesata: con una banda stretta la retta
poggia su pochi punti tutti da un lato, e al bordo sbaglia più della media che
doveva correggere.

La stessa idea, massimizzare una verosimiglianza pesata attorno a $x_0$, dà la
*verosimiglianza locale*, che porta la regressione logistica o di Poisson in
versione locale.

Come la smoothing spline, lo stimatore è lineare nelle risposte,
$\hat{\mathbf{y}} = \mathbf{S}_h\mathbf{y}$ con
$S_{h,ij} = K_h(x_i - x_j)/\sum_l K_h(x_i - x_l)$, quindi la banda si sceglie
con la stessa scorciatoia *leave-one-out*, che qui è esatta, e
$\operatorname{tr}(\mathbf{S}_h)$ ne misura i gradi di libertà effettivi. A
differenza della spline, però, lo stimatore tiene con sé tutti gli esempi, e
ogni stima costa $O(m)$ nella forma diretta.

Più in generale, in $d$ dimensioni la varianza diventa
$O\big(1/(m h^d)\big)$ e la velocità $m^{-4/(4+d)}$, che per funzioni due
volte derivabili nessuno stimatore può battere {cite}`stone1982optimal`. È la
maledizione della dimensionalità del
{doc}`k-NN <apprendimento-supervisionato>`, che è a sua volta una media locale
con un nucleo uniforme e una banda che si adatta per contenere $k$ punti.

Il nucleo gaussiano fa anche da ponte verso l'architettura dei Transformer. Se
le chiavi $\mathbf{k}_i$ hanno tutte la stessa norma,
$\exp\big(-\lVert\mathbf{q}-\mathbf{k}_i\rVert^2/(2h^2)\big) \propto
\exp(\mathbf{q}^\top\mathbf{k}_i/h^2)$ (la norma di $\mathbf{q}$ non conta,
perché il fattore $e^{-\lVert\mathbf{q}\rVert^2/(2h^2)}$ è comune a tutti i
pesi e la normalizzazione lo cancella), e la media pesata $\sum_i
\mathrm{softmax}_i(\mathbf{q}^\top\mathbf{k}_i/h^2)\,\mathbf{v}_i$ è lo
stimatore di Nadaraya-Watson con il punto $\mathbf{q}$, gli esempi
$\mathbf{k}_i$ e le risposte $\mathbf{v}_i$.

In particolare, con $h^2 = \sqrt{d_k}$, dove $d_k$ è la dimensione di query e
chiavi (il numero delle loro componenti), è
l’{doc}`attenzione </Transformers/attenzione>` dei Transformer: il suo fattore
$1/\sqrt{d_k}$ fa la parte della larghezza della campana, e la similarità, a
differenza della campana, si impara, attraverso le proiezioni che producono
$\mathbf{q}$ e $\mathbf{k}_i$.

`````

```{figure} ../figures/media-che-scivola.svg
:name: fig-media-che-scivola
:alt: "Animazione: cento punti sparsi attorno a un'onda sinusoidale tratteggiata. Una campana ocra scorre da sinistra a destra lungo l'asse orizzontale, e mentre avanza si traccia dietro di lei una curva terracotta, la media dei punti pesata dalla campana. La curva segue l'onda all'interno e se ne stacca ai due estremi, dove la campana ha punti da una parte sola: all'estremo sinistro resta sopra l'onda, all'estremo destro sotto."
:width: 92%

La campana ocra decide quanto pesa ogni punto nella media, e mentre scorre la
media disegna la curva terracotta. All'interno la curva segue l'onda vera
(tratteggiata); ai due estremi la campana ha punti da una parte sola, e la curva
se ne stacca: a sinistra i vicini stanno tutti più in alto e la tirano su, a
destra stanno tutti più in basso e la tirano giù.
```

Il codice misura il compromesso sulla larghezza e il difetto del bordo
ripetendo la prova duecento volte, ogni volta con cento esempi rumorosi nuovi
presi dalla stessa onda (la funzione $\sin 2\pi x$, che fra $0$ e $1$ compie un
periodo intero, con un massimo e un minimo, più un rumore di deviazione
standard $0{,}3$). Su una
prova sola, infatti, il rumore dei pochi punti ai bordi basta a far vincere,
per caso, il metodo che in media sbaglia di più.

Per ogni stimatore il codice stampa la radice dell'errore quadratico medio,
cioè di quanto, tipicamente, la stima si scosta dall'onda vera, sui
duecentouno punti in cui la si calcola: in tutto, all'interno e nel 10% vicino
agli estremi. I numeri da confrontare sono, riga per riga, quello dell'interno
e quello dei bordi.

```python
import numpy as np

rng = np.random.default_rng(0)
vera = lambda x: np.sin(2 * np.pi * x)            # la funzione da ritrovare
griglia = np.linspace(0, 1, 201)                   # dove si stima
bordo = (griglia < 0.05) | (griglia > 0.95)        # il 10% vicino agli estremi

def pesi(x, h):
    """Una campana larga h attorno a ogni punto della griglia."""
    return np.exp(-0.5 * ((griglia[:, None] - x[None, :]) / h) ** 2)

def nadaraya_watson(x, y, h):
    w = pesi(x, h)
    return (w * y).sum(axis=1) / w.sum(axis=1)     # media pesata delle y

def retta_locale(x, y, h):
    """Minimi quadrati pesati di una retta attorno a ogni punto, valutata lì."""
    w, d = pesi(x, h), x[None, :] - griglia[:, None]
    s0, s1, s2 = w.sum(1), (w * d).sum(1), (w * d * d).sum(1)
    t0, t1 = (w * y).sum(1), (w * d * y).sum(1)
    return (s2 * t0 - s1 * t1) / (s0 * s2 - s1 ** 2)

errori = {}
for _ in range(200):                               # duecento campioni
    x = np.sort(rng.random(100))
    y = vera(x) + rng.normal(0, 0.3, size=100)
    stime = [(f"media pesata, h = {h}", nadaraya_watson(x, y, h))
             for h in (0.005, 0.02, 0.05, 0.1, 0.3)]
    stime += [(f"retta locale, h = {h}", retta_locale(x, y, h))
              for h in (0.02, 0.05)]
    for nome, stima in stime:
        errori.setdefault(nome, []).append((stima - vera(griglia)) ** 2)
for nome, e in errori.items():
    e = np.array(e)
    print(f"{nome:<23}: errore {np.sqrt(e.mean()):.3f}  (dentro"
          f" {np.sqrt(e[:, ~bordo].mean()):.3f}, ai bordi {np.sqrt(e[:, bordo].mean()):.3f})")
```

```text
media pesata, h = 0.005: errore 0.229  (dentro 0.228, ai bordi 0.239)
media pesata, h = 0.02 : errore 0.125  (dentro 0.120, ai bordi 0.159)
media pesata, h = 0.05 : errore 0.107  (dentro 0.089, ai bordi 0.202)
media pesata, h = 0.1  : errore 0.181  (dentro 0.147, ai bordi 0.357)
media pesata, h = 0.3  : errore 0.459  (dentro 0.466, ai bordi 0.392)
retta locale, h = 0.02 : errore 0.138  (dentro 0.123, ai bordi 0.231)
retta locale, h = 0.05 : errore 0.091  (dentro 0.082, ai bordi 0.146)
```

L'errore scende e poi risale al crescere della larghezza: $0{,}229$ con la
finestra strettissima, che insegue il rumore; $0{,}107$ con $h = 0{,}05$;
$0{,}459$ con una campana larga $0{,}3$, quasi un terzo dell'intervallo, che
appiattisce l'onda (e che ai bordi sbaglia meno che all'interno solo perché lì
l'onda vale zero, come la curva appiattita).

Alla migliore delle cinque larghezze la media pesata sbaglia ai bordi più del
doppio che all'interno ($0{,}202$ contro $0{,}089$), e la retta locale porta il
bordo a $0{,}146$ migliorando un poco anche l'interno. Con la finestra stretta
di $h = 0{,}02$, invece, al bordo la retta sbaglia più della media ($0{,}231$
contro $0{,}159$), perché pochi punti tutti da un lato non bastano a fissarne
la pendenza.

## Da una curva a molte: i modelli additivi

Spline e regressione a nucleo danno il meglio con una variabile sola. Con dieci
colonne una funzione flessibile di tutte e dieci insieme chiederebbe una
quantità di dati che nessuno ha. È la maledizione della dimensionalità,
incontrata a proposito del {doc}`k-NN <apprendimento-supervisionato>` e appena
ritrovata, come velocità, nella regressione a nucleo. In numeri: una griglia a
$10$ caselle per lato ne ha $10$ in una dimensione, $100$ in due, $1000$ in tre
e $10^{10}$ in dieci, e per un esempio per casella ne servirebbero dieci
miliardi.

La via d'uscita è rinunciare alle interazioni. Il **modello additivo
generalizzato** (*Generalized Additive Model*, GAM), proposto da Trevor Hastie
e Robert Tibshirani nel 1986 {cite}`hastie1986generalized`, scrive la risposta
come una somma di funzioni di una colonna ciascuna, $\theta_0 + f_1(x_1) +
\dots + f_d(x_d)$, e stima ogni $f_j$ con una spline («generalizzato» perché la
somma può passare per una funzione di collegamento, come nella regressione
logistica).

`````{tab} Elementare

L'idea è rinunciare a una sola cosa, sapendo a quale. Invece di chiedere
«come varia il risultato al variare di tutte le colonne insieme», si chiede
«come varia al variare di ciascuna colonna presa da sola», e poi si sommano
le risposte. Con dieci colonne le curve da disegnare sono dieci, e dieci curve
si disegnano anche con i pochi dati che uno ha.

Ad esempio, per il prezzo di un affitto una curva dice come cambia al crescere
dei metri quadri (sale, ma sempre meno: i primi cinquanta metri quadri valgono
più dei cinquanta successivi). Un'altra dice come cambia con la distanza dal
centro (scende in fretta nei primi chilometri e poi si appiattisce). Una terza
mostra l'effetto dell'anno di costruzione, e non è affatto monotona: le case
degli anni Sessanta valgono meno sia delle nuove sia di quelle d'epoca. Ogni
curva la disegna una spline, e il prezzo stimato è la somma delle tre.

Le tre curve dicono uno scostamento, cioè di quanto si sale o si scende rispetto
a un prezzo di partenza, che è scritto una volta sola per tutte e tre.
Questa convenzione serve a poterle leggere. Senza, infatti, si potrebbe alzare
di cento euro la curva dei metri quadri e abbassare di cento quella della
distanza, ottenendo gli stessi identici prezzi con curve diverse, e nessuna
delle tre vorrebbe più dire niente.

Come si trovano, quelle tre curve? Una per volta, a giri. Si prende il prezzo
di ogni casa, gli si toglie la parte che le altre due curve spiegano già, e quel
che resta lo si affida alla curva dei metri quadri, che si adatta a seguirlo.
Poi tocca alla distanza, tenendo ferme le altre due, poi all'anno di
costruzione. Si ricomincia il giro, e a ogni giro le curve si muovono meno,
finché smettono di muoversi.

Il guadagno che si vede subito è che quelle curve si possono guardare. Un
modello che restituisce un numero solo non dice da dove viene quel numero; qui
si disegnano tre grafici e si legge la forma di ciascun effetto, gobbe comprese,
senza aver dovuto decidere in anticipo che forma avesse.

La rinuncia è che gli effetti non si parlano fra loro. Il modello non
può dire «i metri quadri contano di più se sei in centro»: quello è un effetto
delle due colonne insieme, e in una somma di curve separate non c'è posto per
scriverlo. Se nei dati veri quel legame c'è, il GAM lo manca, e lo manca del
tutto, non per un pelo.

`````

`````{tab} Superiore

Un GAM sostituisce la parte lineare di un {doc}`modello lineare generalizzato
</MachineLearning/apprendimento-supervisionato>` con una somma di funzioni
univariate lisce:

$$
g\bigl(\mathbb{E}[y \mid \mathbf{x}]\bigr) = \theta_0 + \sum_{j=1}^{d} f_j(x_j),
$$

dove $d$ è il numero di colonne, $g$ è la funzione di collegamento, e ogni
$f_j$ è una funzione liscia stimata dai dati, tipicamente una spline. La
funzione $g$ è l'identità per la regressione e la logit per la classificazione
(in quest'ultimo caso il modello è la regressione logistica con le $\theta_j
x_j$ promosse a $f_j(x_j)$). Per identificabilità si impone $\sum_{i}
f_j(x_{ij}) = 0$ per ogni $j$, dove $i$ scorre sugli esempi e $x_{ij}$ è la
$j$-esima colonna dell’$i$-esimo; in questo modo il livello medio si scarica su
$\theta_0$.

La stima classica è il **backfitting**: si aggiorna ciclicamente

$$
\mathbf{f}_j \leftarrow \mathcal{S}_j \Bigl( \mathbf{y} - \theta_0\mathbf{1} - \textstyle\sum_{k \ne j} \mathbf{f}_k \Bigr),
\qquad \mathbf{f}_k = \bigl(f_k(x_{1k}), \dots, f_k(x_{mk})\bigr)^{\top},
$$

cioè si liscia il residuo parziale rispetto alla sola $x_j$, e si ripete fino a
convergenza. È lo schema «alterna, tenendo fermo tutto il resto» che torna nel
$k$-means e nell'EM della {doc}`sezione su riduzione e clustering
<riduzione-clustering>`. Se ogni $\mathcal{S}_j$ è una proiezione, il
procedimento converge alla soluzione dei minimi quadrati vincolati allo spazio
additivo {cite}`buja1989linear`.

Costruire un GAM con B-spline non richiede altro che quanto già visto: si
espande ogni colonna nella sua base e si risolve un unico problema lineare
regolarizzato. La complessità sale quindi linearmente con il numero $d$ di
colonne, non esponenzialmente: è la maledizione della dimensionalità evitata
per costruzione, e in cambio lo spazio delle ipotesi perde tutti i termini di
interazione $f_{jk}(x_j, x_k)$.

La garanzia che giustifica la rinuncia è di Stone {cite}`stone1985additive`: se
il modello additivo è vero e ogni $f_j$ è due volte derivabile, l'errore
quadratico di stima converge come $m^{-4/5}$, la velocità di una variabile sola,
qualunque sia $d$, contro l’$m^{-4/(4+d)}$ della regressione a nucleo su tutte
le colonne insieme.

Se invece il modello non è additivo, il GAM converge alla proiezione della
funzione vera sullo spazio delle somme additive, e perde per intero la
varianza della parte che resta fuori. Nell'esperimento con
l'interazione, con $x_1, x_2$ indipendenti e uniformi su $[-2, 2]$, il prodotto
$1{,}5\,x_1x_2$ resta fuori tutto ($\mathbb{E}[x_1x_2 \mid x_1] = 0$), e la sua
varianza vale $1{,}5^2\cdot(4/3)^2 = 4$.

`````

Il codice costruisce un GAM con gli strumenti già visti (una spline per
colonna, sommate da una regressione lineare) e lo mette alla prova due volte:
su dati che sono una somma di effetti separati, e poi sugli stessi dati con
dentro un'interazione. Il confronto è con un modello lineare da una parte e un
gradient boosting dall'altra (alberi messi in fila), cioè con il più rigido e
il più libero fra i modelli vicini.

```python
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, (800, 3))
def forma(X):
    return np.sin(3*X[:, 0]) + 0.5*X[:, 1]**2 - 0.8*X[:, 2]
y = forma(X) + rng.normal(0, 0.2, 800)

def gam(nodi=8):
    """Un GAM: una spline per colonna, sommate da una regressione lineare."""
    per_colonna = ColumnTransformer(
        [(f"s{j}", SplineTransformer(n_knots=nodi, degree=3), [j])
         for j in range(X.shape[1])])
    return make_pipeline(per_colonna, Ridge(alpha=1e-3))

pieghe = KFold(5, shuffle=True, random_state=0)
def mse(m, bersaglio):
    return -cross_val_score(m, X, bersaglio, cv=pieghe,
                            scoring="neg_mean_squared_error").mean()

print(f"lineare            MSE = {mse(LinearRegression(), y):.4f}")
print(f"GAM (spline)       MSE = {mse(gam(), y):.4f}")
print(f"boosting           MSE = {mse(HistGradientBoostingRegressor(random_state=0), y):.4f}")
print(f"rumore vero              {0.2**2:.4f}")

# ora con un'interazione, che un GAM per costruzione non può vedere
y2 = forma(X) + 1.5*X[:, 0]*X[:, 1] + rng.normal(0, 0.2, 800)
print()
print("con interazione x1*x2:")
print(f"GAM (spline)       MSE = {mse(gam(), y2):.4f}")
print(f"boosting           MSE = {mse(HistGradientBoostingRegressor(random_state=0), y2):.4f}")
```

```text
lineare            MSE = 0.7841
GAM (spline)       MSE = 0.0427
boosting           MSE = 0.0866
rumore vero              0.0400

con interazione x1*x2:
GAM (spline)       MSE = 4.4078
boosting           MSE = 0.2904
```

Sui dati additivi il GAM arriva a $0{,}0427$ contro un rumore vero di
$0{,}0400$, quindi la parte riducibile dell'errore è quasi esaurita. Il
gradient boosting, pur più flessibile, si ferma a $0{,}0866$, il doppio, perché
deve ricostruire a gradini una struttura additiva che il GAM ha già nella
propria forma. Chi si aspetta una classifica di potenza trova il contrario.

Con l'interazione il GAM passa da $0{,}0427$ a $4{,}4078$, cento volte peggio,
mentre il boosting passa da $0{,}0866$ a $0{,}2904$. L'interazione, infatti, è
costruita in modo che il GAM non ne colga niente: con $x_1$ e $x_2$
indipendenti e simmetrici attorno allo zero, il prodotto $x_1x_2$ non ha
nessuna componente additiva.

In generale, un modello con un'ipotesi forte sbaglia così: perde tutto ciò che
sta fuori dall'ipotesi, poco o tanto che sia, e prima di usarne uno bisogna
sapere quale ipotesi si sta assumendo.

## Dove stanno, in pratica

Spline e GAM stanno fra la retta e i modelli del tutto liberi, e non sono
soltanto «modelli in più».

Sono la risposta giusta quando servono tre cose insieme: che l'effetto di una
variabile sia curvo e non se ne conosca la forma; che quell'effetto vada
mostrato a qualcuno, in un grafico che si legge senza sapere di statistica; e
che i dati non siano tantissimi. La medicina e l'epidemiologia li usano da
decenni per questa ragione, e per lo stesso motivo la {doc}`sezione sui
modelli trasparenti </Interpretabilita/modelli-trasparenti-e-importanza>` ci
torna sopra: un GAM si guarda direttamente, senza bisogno che qualcuno venga
dopo a spiegarlo.

Sono la risposta sbagliata quando le interazioni sono il cuore del problema, e
quando i dati sono immagini, suono o testo, perché là non ci sono
colonne con un significato proprio di cui abbia senso disegnare l'effetto, e
servono altri modelli.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un polinomio unico piegato troppo ondeggia, e i danni li fa ai bordi, dove
  niente lo tiene fermo dall'altra parte: è il fenomeno di Runge, e al grado 21
  il polinomio sbaglia ai bordi undici volte il rumore che c'è nei dati,
  restando impeccabile al centro.
- Una spline è la curva che prende il listello di legno dei cantieri navali
  quando le pendenze sono piccole: tante cubiche corte, giuntate in modo che sui
  paletti (i nodi) non si vedano né gradini né spigoli né scatti di curvatura.
- Aggiungere flessibilità a una spline la aggiunge dove serve, un tratto per
  volta; aggiungerla a un polinomio la aggiunge ovunque.
- La versione naturale obbliga la curva a proseguire dritta oltre gli
  estremi, cioè rende noiosi i bordi, che sono il posto dove i modelli fanno i
  danni.
- La smoothing spline aggancia il listello ai punti con degli elastici e mette
  una manopola sulla rigidità del legno: da spago che passa per ogni punto a
  righello che non si piega. Girata tutta verso il rigido dà una retta, e il
  conto lo mostra: la flessibilità che resta è $2{,}0026$ contro il $2$ tondo di
  una retta vera.
- La regressione a nucleo (la media che scivola) stima il valore in un punto con
  la media dei vicini, pesati da una campana. La larghezza della campana è la
  manopola (stretta insegue il rumore, larga appiattisce), e ai bordi, dove i
  vicini stanno da una parte sola, la media si storce: una retta locale al posto
  della media ne toglie gran parte, purché la finestra prenda abbastanza case.
- Un GAM somma una curva per colonna: si vede la forma di ogni effetto, uno
  per uno. La rinuncia dichiarata è che gli effetti non si parlano fra loro: se
  nei dati due colonne contano insieme, il GAM non se ne accorge.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- L'interpolazione polinomiale su nodi equispaziati può divergere (Runge): la
  costante di Lebesgue cresce esponenzialmente e i polinomi hanno supporto
  globale, quindi ogni coefficiente agisce ovunque. Cambiare base cura il
  condizionamento, non la globalità.
- Una spline cubica con $K$ nodi è cubica a tratti con continuità di $f$,
  $f'$, $f''$ nei nodi: $4(K+1) - 3K = K+4$ gradi di libertà, che diventano $K$
  imponendo la naturalità ($f'' = f''' = 0$ fuori dai nodi estremi).
- La base operativa sono le B-spline, a supporto locale (quattro intervalli):
  matrice di disegno a banda e ben condizionata, di rango pieno se e solo se
  i dati permettono di scegliere $K+4$ punti in ordine crescente, l’$i$-esimo nel
  supporto della $i$-esima funzione di base (condizione di Schoenberg e Whitney).
- La smoothing spline minimizza $\sum_i (y_i - f(x_i))^2 + \lambda \int f''^2$;
  il minimo su uno spazio di dimensione infinita è una spline cubica naturale
  con nodi nei dati. La complessità si misura con i gradi di libertà effettivi
  $\operatorname{tr}(\mathbf{S}_\lambda)$, che vanno da $m$ a $2$. Nella base
  B-spline è una Ridge generalizzata, che non frena le rette.
- La regressione a nucleo di Nadaraya-Watson è una media pesata, con pesi dati
  da una campana di larghezza $h$ centrata in $x_0$: distorsione $O(h^2)$ e
  varianza $O\big(1/(mh)\big)$ danno $h \propto m^{-1/5}$ ed errore quadratico
  $m^{-4/5}$. Al bordo la distorsione sale a $O(h)$; la regressione lineare
  locale la riporta a $O(h^2)$, pagandola in varianza (circa il triplo nel punto
  estremo), e all'interno toglie la distorsione di progetto $f'p'/p$. Con nucleo
  gaussiano, chiavi di norma comune e $h^2 = \sqrt{d_k}$ è l'attenzione dei
  Transformer, che in più impara le proiezioni da cui escono query e chiavi.
- Un GAM pone $g(\mathbb{E}[y \mid \mathbf{x}]) = \theta_0 + \sum_j f_j(x_j)$:
  costo lineare nel numero di colonne invece che esponenziale, ma senza
  interazioni. Si stima con il backfitting, cioè lisciando i
  residui parziali una variabile per volta; se il modello è vero l'errore va
  come $m^{-4/5}$ qualunque sia $d$ (Stone).
- Il compromesso è misurabile in tutte e due le direzioni: su dati additivi il
  GAM batte un gradient boosting (MSE $0{,}0427$ contro $0{,}0866$, con rumore
  irriducibile $0{,}0400$); aggiunta un'interazione senza componente additiva,
  sale a $4{,}4078$ perché ne perde tutta la varianza, mentre il boosting passa
  da $0{,}0866$ a $0{,}2904$.
```

`````

C'è un filo che tiene insieme spline e GAM con gli alberi e le macchine a
vettori di supporto: la domanda su quanta struttura mettere nel modello prima di
guardare i dati. La retta ne mette troppa e non si piega; il polinomio ne toglie
troppa e si piega dove non deve; la spline la rimette al posto giusto, dicendo
che la curva deve essere dolce ma non che forma abbia, e la regressione a nucleo
chiede la stessa dolcezza per un'altra strada, pesando i vicini invece di
penalizzare le pieghe. Il GAM fa lo stesso un gradino più su, sulle colonne. Gli
{doc}`alberi decisionali <alberi-ensemble>` rispondono alla stessa domanda con
lo strumento opposto, spezzando la linea a gradini invece di piegarla.
