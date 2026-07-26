# Processi gaussiani: prevedere con l'incertezza

C'è una differenza sottile ma decisiva tra due previsioni del tempo. «Domani
24 gradi» è una cifra secca: sembra sicura, ma non dice nulla su quanto
fidarsi. «Domani tra 21 e 27» dice di meno e comunica di più: oltre alla
stima, dichiara *quanto il modello non sa*. Tutti i modelli visti finora in
questo capitolo — la retta di regressione, la logistica, il k-NN — rispondono
alla prima maniera: un numero, prendere o lasciare. In questa sezione
incontriamo un modello che risponde alla seconda: il **processo gaussiano**.

L'idea ha radici minerarie. Nel 1951 Danie Krige, un giovane ingegnere
sudafricano, affrontava il problema più costoso delle miniere d'oro del
Witwatersrand: ogni carotaggio — un pozzo di assaggio per misurare la
concentrazione del minerale — costava una fortuna, e i punti campionati erano
per forza pochi e sparsi. Come stimare quanto oro c'è *tra* un pozzo e
l'altro? Krige propose di usare medie pesate dei campioni vicini, con pesi
scelti in modo statistico — e con una misura esplicita di quanto ogni stima
fosse affidabile. Nei primi anni Sessanta il matematico francese Georges
Matheron formalizzò il metodo e lo battezzò *kriging*, in suo onore. Oggi la
stessa matematica, generalizzata e ribattezzata processi gaussiani, è uno
degli strumenti più eleganti del machine learning
{cite}`rasmussen2006gaussian`.

## Un fascio di curve, non una sola

La regressione lineare dell'inizio del capitolo impara *una* curva: la retta
di best fit, e basta. Il processo gaussiano fa una scelta più ambiziosa e più
onesta: invece di impegnarsi su una sola curva, tiene in mano **tutte le curve
compatibili con i dati**, ciascuna con il suo grado di plausibilità.

`````{tab} Elementare

Immagina un fascio di fili elastici tesi sopra un tavolo: ognuno è una
possibile curva "vera", un modo in cui il mondo potrebbe comportarsi. Ogni
misura che facciamo è un chiodo piantato nel tavolo: da quel momento tutti i
fili devono passare lì vicino, quasi toccarlo. Vicino ai chiodi il fascio è
costretto, i fili quasi si sovrappongono; lontano dai chiodi si riapre a
ventaglio, perché nulla lo vincola. La previsione del processo gaussiano è
doppia: *dove passa in media il fascio* (la stima) e *quanto è largo lì* (l'incertezza).
È esattamente la previsione «tra 21 e 27»: stretta dove abbiamo
misurato, larga dove stiamo tirando a indovinare.

`````

`````{tab} Superiore

Un processo gaussiano è una distribuzione di probabilità **sulle funzioni**:

$$
f \sim \mathcal{GP}\big(m(x),\, k(x, x')\big),
$$

dove $m(x)$ è la funzione media (spesso posta a zero dopo aver centrato i
dati) e $k(x, x')$ è la funzione di covarianza, o **kernel**. La proprietà
che lo definisce: per *qualunque* insieme finito di punti $x_1, \dots, x_n$,
il vettore dei valori $\big(f(x_1), \dots, f(x_n)\big)$ ha distribuzione
gaussiana multivariata, con medie $m(x_i)$ e covarianze $k(x_i, x_j)$. È un
*prior* sulle funzioni: prima di vedere i dati, tutte le curve coerenti con il
kernel sono possibili; condizionare sulle osservazioni — lo vedremo tra poco —
restringe il fascio, e il risultato è ancora un processo gaussiano
{cite}`rasmussen2006gaussian`.

`````

## Il kernel: chi è vicino si somiglia

Che cosa tiene insieme il fascio? Da dove sa, il modello, che le curve devono
essere lisce e non impazzite? Tutta la "personalità" di un processo gaussiano
sta in un unico ingrediente, il **kernel**: una regola che dice quanto i
valori in due punti devono somigliarsi.

`````{tab} Elementare

La regola del kernel è il buon senso del geometra: **punti vicini hanno valori
simili**. Se a Modena ci sono 24 gradi, a Bologna — quaranta chilometri — mi
aspetto quasi la stessa temperatura; ad Ancona, duecento chilometri più in là,
la mia misura modenese dice ormai poco. Il kernel trasforma questa intuizione
in un numero tra 0 e 1: vicini quasi gemelli valgono quasi 1, lontani
estranei valgono quasi 0. E ha una manopola fondamentale, il **raggio di
influenza**: fin dove arriva l'effetto di una misura? Con un raggio corto ogni
osservazione parla solo del suo vicinato e le curve possono zigzagare; con un
raggio lungo ogni misura si fa sentire lontano e le curve escono morbide e
distese. Per dare un'idea con i numeri: con raggio 1, due punti a distanza 1
si somigliano 0,61; a distanza 2 solo 0,14; a distanza 3 appena 0,01 —
l'influenza svanisce in fretta.

`````

`````{tab} Superiore

Il kernel più usato è l'**RBF** (*Radial Basis Function*, o gaussiano):

$$
k(x, x') = \sigma^2 \exp\!\left(-\frac{\lVert x - x'\rVert^2}{2\ell^2}\right),
$$

dove $\sigma^2$ è la varianza di segnale (l'ampiezza tipica delle
oscillazioni del fascio) e $\ell$ è la **lunghezza-scala** (*lengthscale*): la
distanza oltre la quale due valori diventano, di fatto, indipendenti. Con
$\ell = 1$ due punti a distanza $1$ hanno correlazione
$e^{-0{,}5} \approx 0{,}61$; a distanza $3$, $e^{-4{,}5} \approx 0{,}01$. Una
$\ell$ piccola produce funzioni nervose che dimenticano in fretta; una $\ell$
grande, funzioni lisce e a lungo raggio. Il kernel RBF genera funzioni
infinitamente derivabili — un'ipotesi di regolarità forte, non sempre
realistica. I suoi iperparametri $(\sigma, \ell)$ non si fissano a mano: si
stimano massimizzando la verosimiglianza marginale dei dati, cosa che
`scikit-learn` fa da sola durante il `fit`.

`````

## La previsione: media e incertezza insieme

Vediamo ora il momento in cui i chiodi entrano nel tavolo: come si passa dal
fascio libero (il prior) al fascio inchiodato ai dati (il posteriore), cioè
alla previsione vera e propria.

`````{tab} Elementare

Ogni punto osservato *stringe* il fascio lì vicino: le curve che non passano
nei paraggi vengono scartate, quelle che restano sono quasi d'accordo tra
loro, e la banda d'incertezza si riduce a un filo. Lontano dai punti — tra un
dato e l'altro, o fuori dalla zona esplorata — sopravvivono curve molto
diverse, e la banda si riapre. Il risultato, per ogni punto in cui vogliamo
una previsione, sono due numeri: la **media** delle curve sopravvissute (la
stima migliore) e la **larghezza** del fascio (quanto fidarsi). Se la stima
è 24 gradi e la banda va da 21 a 27, il modello sta dicendo: «quasi
certamente il valore è lì in mezzo». Una banda larghissima non è un difetto:
è il modello che alza la mano e ammette di non avere dati per rispondere.

`````

`````{tab} Superiore

Siano $X$ gli $n$ punti di addestramento con osservazioni rumorose $y$, e
$X_*$ gli $n_*$ punti dove vogliamo predire. Il posteriore è gaussiano con
media e covarianza in forma chiusa {cite}`rasmussen2006gaussian`:

$$
\mu_* = K_*^\top \big(K + \sigma_n^2 I\big)^{-1} y,
\qquad
\Sigma_* = K_{**} - K_*^\top \big(K + \sigma_n^2 I\big)^{-1} K_*,
$$

dove $K \in \mathbb{R}^{n \times n}$ è la matrice del kernel tra i punti di
addestramento ($K_{ij} = k(x_i, x_j)$), $K_* \in \mathbb{R}^{n \times n_*}$
quella tra addestramento e punti nuovi, $K_{**}$ quella tra i punti nuovi,
$\sigma_n^2$ la varianza del rumore di misura e $I$ la matrice identità. Le
due formule si leggono bene. La media $\mu_*$ è una **combinazione pesata
delle osservazioni** $y$, con pesi dettati dal kernel: il kriging di Krige,
appunto. La covarianza $\Sigma_*$ è la varianza del prior ($K_{**}$) *meno*
ciò che i dati spiegano: vicino ai dati la sottrazione mangia quasi tutto e
l'incertezza crolla; lontano non sottrae nulla e si torna all'incertezza del
prior. La banda al 95% è $\mu_* \pm 2\sqrt{\operatorname{diag}(\Sigma_*)}$.

`````

La {numref}`fig-processo-gaussiano` mostra tutto il meccanismo in un colpo
d'occhio: la banda si stringe sui punti osservati fin quasi a toccarli — quasi,
perché il rumore di misura $\sigma_n^2$ impedisce la certezza assoluta anche
lì — e si riapre nel buco centrale e ai bordi, dove i dati mancano.

```{figure} ../figures/processo-gaussiano.svg
:name: fig-processo-gaussiano
:alt: Grafico di una regressione con processo gaussiano, con sei punti osservati, la curva media a posteriori, due curve campione plausibili e una banda di incertezza che si stringe in prossimità dei punti e si allarga dove mancano dati.
:width: 90%

La previsione di un processo gaussiano: la banda d'incertezza si stringe sui
punti osservati e si riapre dove i dati mancano.
```

## In pratica, con scikit-learn

Proviamo su un caso da manuale: pochi punti rumorosi di una sinusoide, come
fossero otto esperimenti costosi.

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Otto misure "costose" di una sinusoide, con rumore
rng = np.random.default_rng(0)
X_train = rng.uniform(0, 6, size=(8, 1))
y_train = np.sin(X_train).ravel() + rng.normal(0, 0.1, size=8)

# Kernel RBF; alpha è la varianza del rumore (il sigma_n^2 delle formule)
kernel = 1.0 * RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1**2,
                              n_restarts_optimizer=5)
gp.fit(X_train, y_train)          # stima anche sigma e l dai dati

# Previsione CON incertezza: media e deviazione standard
X_test = np.array([[1.5], [3.0], [8.0]])
media, dev_std = gp.predict(X_test, return_std=True)

for x, mu, s in zip(X_test.ravel(), media, dev_std):
    print(f"x = {x:.1f}  ->  f(x) = {mu:+.2f} ± {2 * s:.2f}")
```

La riga chiave è `return_std=True`: accanto a ogni previsione arriva la sua
deviazione standard, e stampiamo l'intervallo $\mu \pm 2\sigma$ (circa il 95%
di confidenza). Il risultato racconta la storia della figura: a $x = 1{,}5$ e
$x = 3{,}0$, dentro la zona dei dati, l'intervallo è stretto; a $x = 8{,}0$,
fuori da tutto ciò che il modello ha visto, la banda si spalanca fin quasi
all'ampiezza del prior. Il modello non finge di sapere: allarga le braccia.

## Il conto da pagare, e dove conviene

Tanta eleganza ha un prezzo, e va detto senza giri di parole: il processo
gaussiano **non scala**.

`````{tab} Elementare

Per ogni previsione, il processo gaussiano non consulta un riassunto: riapre
*tutto* l'archivio delle osservazioni e le confronta tra loro, come un medico
che a ogni visita rileggesse le cartelle di tutti i pazienti mai avuti. Con
cento pazienti funziona benissimo; con un milione è impensabile. E il costo
non cresce piano: raddoppiare i dati moltiplica il lavoro per otto, passare da
mille a diecimila punti lo moltiplica per mille. È il motivo per cui non
addestreremo mai un processo gaussiano sulle foto di tutto internet.

`````

`````{tab} Superiore

Il collo di bottiglia è l'inversione (in pratica, la fattorizzazione di
Cholesky) di $K + \sigma_n^2 I$: costo $O(n^3)$ in tempo e $O(n^2)$ in
memoria. Oltre qualche decina di migliaia di punti il metodo esatto diventa
proibitivo. Esistono approssimazioni *sparse* — si riassume il dataset con
$p \ll n$ punti "induttori", scendendo a $O(n p^2)$ — ma pagano in fedeltà
proprio sulla merce di casa: la qualità delle incertezze. A ciò si aggiunge la
sensibilità alla scelta del kernel, che incorpora ipotesi forti (con l'RBF, la
regolarità infinita) da verificare sul problema reale.

`````

Il suo territorio, allora, è l'opposto del big data: **pochi dati costosi**.
Esperimenti di laboratorio dove ogni misura vale una giornata di lavoro,
simulazioni ingegneristiche da ore di calcolo l'una, prove sul campo che non
si possono ripetere. E il caso che abbiamo già incontrato: l'**ottimizzazione
bayesiana degli iperparametri** {cite}`snoek2012practical`, dove ogni "dato" è
un intero addestramento e il processo gaussiano fa da mappa — stima più
incertezza — per decidere quale configurazione provare dopo. La sezione sugli
iperparametri di questo capitolo racconta proprio quel meccanismo: qui abbiamo
aperto il cofano del suo motore.

```{admonition} Da ricordare
:class: important
- Un processo gaussiano non impara una curva sola: mantiene una distribuzione
  su **tutte le curve compatibili con i dati** e per ogni punto restituisce
  una media e un'incertezza — «tra 21 e 27», non «24 e basta».
- Il **kernel** codifica la somiglianza ("punti vicini hanno valori simili");
  la lunghezza-scala $\ell$ decide fin dove arriva l'influenza di
  un'osservazione.
- La **banda d'incertezza** si stringe sui punti osservati e si riapre dove i
  dati mancano: il modello dichiara quanto non sa.
- Costo $O(n^3)$: improponibile sui grandi dataset, perfetto con **pochi dati
  costosi** — esperimenti, simulazioni, ottimizzazione bayesiana degli
  iperparametri.
```
