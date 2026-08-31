# Processi gaussiani: prevedere con l'incertezza

C'è una differenza sottile ma decisiva tra due previsioni del tempo. «Domani
24 gradi» è una cifra secca: sembra sicura, ma non dice nulla su quanto
fidarsi. «Domani tra 21 e 27» dice di meno e comunica di più: oltre alla
stima, dichiara *quanto il modello non sa*. Quasi tutti i modelli visti finora
(la retta di best fit, la regressione logistica, il k-NN)
rispondono alla prima maniera: un numero, prendere o lasciare. Ne esiste
uno che risponde alla seconda: il **processo
gaussiano**.

Il nome, per una volta, si spiega in una riga. **Gaussiano** perché tutto ciò
che il modello dice ha la forma della curva a campana di Gauss, quella con un
valore centrale e un margine attorno. E **processo** non nel senso del
tribunale né del tempo che scorre: è il termine con cui in statistica si
indica un'intera famiglia di quantità imparentate fra loro, qui i valori che la
curva vera può assumere in ogni punto.

L'idea ha radici minerarie. Nel 1951 Danie Krige, un giovane ingegnere
sudafricano, affrontava il problema più costoso delle miniere d'oro del
Witwatersrand: ogni carotaggio (un pozzo di assaggio per misurare la
concentrazione del minerale) costava una fortuna, e i punti campionati erano
per forza pochi e sparsi. Come stimare quanto oro c'è *tra* un pozzo e
l'altro? Krige propose di usare medie pesate dei campioni vicini, con pesi
scelti in modo statistico, e con una misura esplicita di quanto ogni stima
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

Un fascio di fili elastici tesi sopra un tavolo, ognuno una possibile curva
"vera", un modo in cui il mondo potrebbe comportarsi. Prima di misurare
qualsiasi cosa i fili si affollano attorno alla stessa altezza in ogni punto
del tavolo, moltissimi vicini a quella quota, pochi scostati parecchio,
pochissimi in cima o in fondo. Punta il dito su un punto qualunque del tavolo e
guarda soltanto le altezze dei fili lì sopra. La loro forma è la campana.
Puntane tre insieme, prendi da ogni filo la terna di altezze, e ritrovi la
stessa storia. Vale per una manciata qualsiasi di punti, ed è la regolarità che
dà il nome al metodo.

Ogni misura che facciamo è un chiodo piantato nel tavolo: da quel momento i
fili devono passare lì vicino, quasi toccarlo, e chi passa lontano esce di
scena. Vicino ai chiodi il fascio è costretto, i fili quasi si sovrappongono;
lontano dai chiodi si riapre a ventaglio, perché nulla lo vincola. I chiodi
però non cambiano la natura del fascio. Anche dopo, in ogni
punto del tavolo, le altezze si affollano attorno a un centro con la loro
campana; solo che il centro si è spostato sui dati e la campana si è
ristretta.

La previsione del processo gaussiano è doppia: *dove passa in media il fascio*
(la stima) e *quanto è largo lì* (l'incertezza). È esattamente la previsione
«tra 21 e 27»: stretta dove abbiamo misurato, larga dove stiamo tirando a
indovinare.

`````

`````{tab} Superiore

Un processo gaussiano è una distribuzione di probabilità **sulle funzioni**:

$$
f \sim \mathcal{GP}\big(\mu(\mathbf{x}),\, k(\mathbf{x}, \mathbf{x}')\big),
$$

dove $\mu(\mathbf{x})$ è la funzione media (spesso posta a zero dopo aver
centrato i dati) e $k(\mathbf{x}, \mathbf{x}')$ è la funzione di covarianza, o
**kernel**. La proprietà che
lo definisce: per *qualunque* insieme finito di $q$ punti
$\mathbf{x}_1, \dots, \mathbf{x}_q$, il vettore dei valori
$\big(f(\mathbf{x}_1), \dots, f(\mathbf{x}_q)\big)$ ha distribuzione
gaussiana multivariata, con medie $\mu(\mathbf{x}_i)$ e covarianze
$k(\mathbf{x}_i, \mathbf{x}_j)$. È un
*prior* sulle funzioni: prima di vedere i dati, tutte le curve coerenti con il
kernel sono possibili; condizionare sulle osservazioni
restringe il fascio, e il risultato è ancora un processo gaussiano
{cite}`rasmussen2006gaussian`.

`````

## Il kernel: chi è vicino si somiglia

Che cosa tiene insieme il fascio? Da dove sa, il modello, che le curve devono
essere lisce e non impazzite? Tutta la "personalità" di un processo gaussiano
sta in un unico ingrediente, il **kernel**: una regola che dice quanto i
valori in due punti devono somigliarsi.

`````{tab} Elementare

Quaranta chilometri separano Modena da Bologna. Se a Modena il termometro segna
24 gradi, a Bologna ci aspettiamo quasi la stessa temperatura; ad Ancona,
duecento chilometri più giù, quella lettura ci dice ormai poco. Il kernel mette
la faccenda in numeri fra 0 e 1: quasi 1 per due città a un passo, quasi 0 per
due lontanissime.

Fin dove arriva una lettura lo decide una manopola, il **raggio d'influenza**.
Corto, Modena non impegna Bologna, che resta libera di segnare qualunque cosa,
e fra un termometro e l'altro la curva zigzaga. Lungo, Modena tiene stretta
Bologna e Bologna tiene Ferrara: una catena del genere non fa scatti bruschi, e
ne escono curve morbide e distese.

Le distanze si contano in raggi, e con un raggio di quaranta chilometri Bologna
sta a distanza $1$. La somiglianza cala come una campana, col quadrato della
distanza: a Bologna vale $0{,}61$, a ottanta chilometri (dove il quadrato è
quattro volte più grande) $0{,}14$, a centoventi (nove volte) appena $0{,}01$,
e Ancona è fuori. A un raggio pieno siamo appena sopra la metà, quindi «quasi
gemelli» vuol dire molto più vicini di quaranta chilometri: l'influenza di un
termometro sparisce di colpo.

Una seconda manopola, l'ampiezza, dice quanto ballano le letture: un grado di
scarto o dieci. Nessuna delle due la giriamo a mano; la posizione la scelgono i
termometri che abbiamo già, provando e dando un voto.

Il voto tira in due versi. Uno premia chi indovina le nostre letture. L'altro
conta quante altre tabelle di temperature, tutte diverse dalla nostra, quella
posizione avrebbe spiegato altrettanto bene, e più ne sono più toglie. Col
raggio cortissimo ogni città è libera dalle altre, e va bene qualsiasi tabella,
anche quaranta gradi a Modena e zero a Bologna: chi accetta tutto non ha
indovinato niente. Col raggio lunghissimo mezza Italia segna la stessa cifra, e
le nostre letture dicono di no. Vince una posizione di mezzo.

Le posizioni buone però sono più di una, e chi gira la manopola sempre nello
stesso verso si ferma sulla prima. Per questo il giro si rifà cinque volte, da
posizioni sorteggiate, tenendo il voto più alto.

Sotto tutto c'è un'ipotesi, che il tempo cambi sempre dolcemente da una città
all'altra. Su un valico, sulla costa, sul bordo di un temporale non è vero:
cinque gradi se ne vanno in due chilometri, e quel salto il fascio non lo sa
fare. Ci passa in mezzo con una rampa, e non avverte. La banda si stringe dove
i termometri sono fitti, non dove le loro letture ci hanno sorpreso: sopra il
valico, con due misure lì accanto, resta stretta e sbagliata.

`````

`````{tab} Superiore

Il kernel più usato è l’**RBF** (*Radial Basis Function*, o gaussiano):

$$
k(\mathbf{x}, \mathbf{x}') = \sigma^2
\exp\!\left(-\frac{\lVert \mathbf{x} - \mathbf{x}'\rVert^2}{2\ell^2}\right),
$$

dove $\sigma^2$ è la varianza di segnale (l'ampiezza tipica delle oscillazioni
del fascio) e $\ell$ è la **lunghezza-scala** (*lengthscale*): la distanza
oltre la quale due valori diventano, di fatto, indipendenti. Con $\ell = 1$
due punti a distanza $1$ hanno correlazione $e^{-0{,}5} \approx 0{,}61$; a
distanza $3$, $e^{-4{,}5} \approx 0{,}01$. Una $\ell$ piccola produce funzioni
nervose che dimenticano in fretta; una $\ell$ grande, funzioni lisce e a lungo
raggio. Il kernel RBF genera funzioni infinitamente derivabili: un'ipotesi di
regolarità forte, non sempre realistica.

I suoi iperparametri $(\sigma, \ell)$
non si fissano a mano: si stimano massimizzando la **verosimiglianza
marginale** dei dati, cosa che `scikit-learn` fa da sola durante il `fit`. È il
pezzo di matematica più elegante dei processi gaussiani:

$$
\log p(\mathbf{y} \mid \mathbf{X}) =
-\tfrac{1}{2} \mathbf{y}^\top
\big(\mathbf{K} + \sigma_n^2\mathbf{I}\big)^{-1} \mathbf{y}
-\tfrac{1}{2} \log\big\lvert \mathbf{K} + \sigma_n^2\mathbf{I} \big\rvert
-\tfrac{m}{2}\log 2\pi ,
$$

dove $\mathbf{K}$ è la matrice del kernel fra i punti di addestramento
($K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$), $\sigma_n^2$ la varianza del
**rumore di misura** (da non confondere con la $\sigma^2$ di segnale del
kernel), $\mathbf{I}$ la matrice identità e $m$ il numero di esempi.

Il primo termine premia l'aderenza ai dati, il secondo (il logaritmo del
determinante) penalizza i kernel «capaci», quelli che ammettono troppe funzioni
diverse. È il **rasoio di Occam scritto dentro il criterio**: qui non serve un
validation set per punire la complessità, ci pensa la formula. Con una
avvertenza pratica: quella funzione **non è concava** negli iperparametri e ha
massimi locali {cite}`rasmussen2006gaussian`, ed è la ragione per cui
l'ottimizzazione si fa ripartire cinque volte da inizializzazioni sorteggiate
(`n_restarts_optimizer=5` in `scikit-learn`, che di suo non ne fa nessuna: il
default è zero) tenendo il massimo più alto,
esattamente nello spirito della sezione sugli iperparametri.

`````

## La previsione: media e incertezza insieme

Vediamo ora il momento in cui i chiodi entrano nel tavolo: come si passa dal
fascio libero, cioè tutto quello che il modello ritiene possibile *prima* di
vedere una sola misura (in statistica si chiama **prior**, «ciò che viene
prima»), al fascio inchiodato ai dati, cioè quello che resta possibile *dopo*
averle viste (il **posteriore**). È la previsione vera e propria.

`````{tab} Elementare

Ogni punto osservato *stringe* il fascio lì vicino: le curve che non passano
nei paraggi vengono scartate, quelle che restano sono quasi d'accordo tra
loro, e la banda d'incertezza si riduce a un filo. Lontano dai punti (tra un
dato e l'altro, o fuori dalla zona esplorata) sopravvivono curve molto
diverse, e la banda si riapre. Il risultato, per ogni punto in cui vogliamo una
previsione, sono due numeri, e nessuno dei due si cerca per tentativi. Piantati
i chiodi e fissate le manopole, escono da un conto diretto.

Il primo numero è la stima, e si ottiene come faceva Krige nelle miniere: una
media delle misure che abbiamo, in cui ognuna pesa quanto è vicina al punto che
ci interessa. Per prevedere a Bologna, il termometro di Modena conta quasi da
solo; quello di Ancona entra nel conto con un peso troppo piccolo per spostare
la virgola.

Il secondo numero è la larghezza del fascio, e si ottiene per sottrazione. Si
parte da quanto eravamo ignoranti prima di misurare, cioè dall'apertura del
ventaglio libero, e si toglie quello che le misure hanno già spiegato. Accanto
a un chiodo la sottrazione porta via quasi tutto e resta un filo; lontano non
c'è niente da togliere, e si torna all'apertura di partenza.

Se la stima è 24 gradi e la banda va da 21 a 27, il modello sta dicendo: «quasi
certamente la temperatura vera è lì in mezzo». Quella banda risponde alla
domanda «quanti gradi fa davvero adesso a Bologna». C'è una domanda vicina,
«quanto segnerà il termometro che ci piazzo domani», e la banda che le risponde
è più larga, perché porta con sé anche lo sbaglio dello strumento. La
differenza si vede nel caso estremo. Su una città dove abbiamo cento misure la
banda sulla temperatura vera si assottiglia fino quasi a sparire, mentre quella
sulla lettura di domani non scende mai sotto l'errore del termometro.

Una banda larghissima è il modello che alza la mano e ammette
di non avere dati per rispondere.

`````

`````{tab} Superiore

Siano $\mathbf{X}$ gli $m$ punti di addestramento
$\mathbf{x}_1, \dots, \mathbf{x}_m$ con osservazioni rumorose $\mathbf{y}$ (la
solita $m$ del capitolo: il numero di esempi), e $\mathbf{X}_*$ gli $m_*$ punti
dove vogliamo predire. Il posteriore è gaussiano con media e covarianza in
forma chiusa {cite}`rasmussen2006gaussian`:

$$
\boldsymbol{\mu}_* = \mathbf{K}_*^\top
\big(\mathbf{K} + \sigma_n^2 \mathbf{I}\big)^{-1} \mathbf{y},
\qquad
\boldsymbol{\Sigma}_* = \mathbf{K}_{**} - \mathbf{K}_*^\top
\big(\mathbf{K} + \sigma_n^2 \mathbf{I}\big)^{-1} \mathbf{K}_*,
$$

dove $\mathbf{K} \in \mathbb{R}^{m \times m}$ è la matrice del kernel tra i
punti di addestramento ($K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$),
$\mathbf{K}_* \in \mathbb{R}^{m \times m_*}$ quella tra addestramento e punti
nuovi, $\mathbf{K}_{**}$ quella tra i punti nuovi,
$\sigma_n^2$ la varianza del rumore di misura, $\mathbf{y}$ il vettore delle
osservazioni e $\mathbf{I}$ la matrice identità. Le
due formule si leggono bene. La media $\boldsymbol{\mu}_*$ è una
**combinazione pesata delle osservazioni** $\mathbf{y}$, con pesi dettati dal
kernel: il kriging di Krige, appunto. La covarianza $\boldsymbol{\Sigma}_*$ è
la varianza del prior ($\mathbf{K}_{**}$)
*meno*
ciò che i dati spiegano: vicino ai dati la sottrazione mangia quasi tutto e
l'incertezza crolla; lontano non sottrae nulla e si torna all'incertezza del
prior.

La banda al 95% **sulla funzione** è $\boldsymbol{\mu}_* \pm
2\sqrt{\operatorname{diag}(\boldsymbol{\Sigma}_*)}$, ed è quella che
`scikit-learn` restituisce con `return_std=True`. Attenzione a non confonderla
con l'intervallo su una **nuova osservazione**, che è un'altra cosa: lì al
posteriore sulla funzione va aggiunto il rumore di misura, cioè
$\boldsymbol{\mu}_* \pm 2\sqrt{\operatorname{diag}(\boldsymbol{\Sigma}_*) +
\sigma_n^2}$. La differenza non è cosmetica: dove le misure si infittiscono la
prima si assottiglia fino quasi a sparire, mentre la seconda non scende mai
sotto $\sigma_n$. Se la domanda è «che valore misurerò domani» serve la
seconda; se è «quanto vale la grandezza vera», la prima.

`````

La {numref}`fig-processo-gaussiano` mostra tutto il meccanismo in un colpo
d'occhio: la banda si stringe sui punti osservati fin quasi a toccarli (quasi,
perché anche le misure sbagliano un po’, e quel margine d'errore non si può
eliminare) e si riapre nel buco centrale e ai bordi, dove i dati mancano.

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

# Kernel RBF; alpha è la varianza del rumore delle osservazioni
kernel = 1.0 * RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1**2,
                              n_restarts_optimizer=5)
gp.fit(X_train, y_train)          # stima anche sigma e l dai dati

# Previsione CON incertezza: media e deviazione standard
X_test = np.array([[1.5], [3.0], [8.0]])
media, dev_std = gp.predict(X_test, return_std=True)

print("i punti misurati:", np.sort(X_train.ravel()).round(2))
for x, mu, s in zip(X_test.ravel(), media, dev_std):
    print(f"x = {x:.1f}  ->  f(x) = {mu:+.2f} ± {2 * s:.2f}")
```

```text
i punti misurati: [0.1  0.25 1.62 3.64 3.82 4.38 4.88 5.48]
x = 1.5  ->  f(x) = +0.83 ± 0.20
x = 3.0  ->  f(x) = +0.07 ± 0.36
x = 8.0  ->  f(x) = +0.12 ± 1.38
```

La riga chiave è `return_std=True`: accanto a ogni previsione arriva la sua
**deviazione standard**, cioè di quanto il valore vero, tipicamente, si scosta
dalla stima. Nella stampa la raddoppiamo, e non a caso: in una curva a campana,
fra due deviazioni standard sotto la media e due sopra cade circa il $95\%$ dei
casi. È una proprietà della campana, non una scelta nostra, ed è la ragione per
cui un intervallo largo due deviazioni standard per parte si legge come «quasi
certamente il valore sta lì dentro».

Le tre righe stampate raccontano la storia della figura in tre gradini, non
in due. A $x = 1{,}5$, accanto a un dato osservato, la banda è strettissima
($\pm 0{,}20$). A $x = 3{,}0$ siamo ancora *dentro* l'intervallo esplorato, ma
in mezzo a un buco: fra $1{,}62$ e $3{,}64$ la prima riga stampata non ha
nessun punto, e $3{,}0$ sta proprio in quel vuoto. La banda si
allarga già a $\pm 0{,}36$, quasi il doppio, pur restando utile. A $x = 8{,}0$,
fuori da tutto ciò che il modello ha visto, si spalanca a $\pm 1{,}38$, cioè
quasi quanto era prima di vedere qualsiasi dato. È la lezione della sezione:
l'incertezza non
distingue «dentro» da «fuori», distingue **vicino a un dato** da **lontano da
un dato**. E il modello non finge di sapere: allarga le braccia.

## Il conto da pagare, e dove conviene

Tanta eleganza ha un prezzo, e va detto senza giri di parole: il processo
gaussiano **regge male i dati tanti**.

`````{tab} Elementare

Il processo gaussiano non si costruisce un riassunto dei dati da consultare
poi: tiene *tutte* le osservazioni e le confronta a due a due, come un medico
che a ogni visita rileggesse le cartelle di tutti i pazienti mai avuti. Con
qualche centinaio di pazienti funziona benissimo; verso le decine di migliaia
comincia a non stare più in piedi; con un milione non se ne parla.

E il conto è peggiore di quanto l'immagine suggerisca. Confrontare tutte le
coppie sarebbe già un lavoro che cresce col **quadrato** del numero di
pazienti: raddoppiandoli, le coppie quadruplicano. Ma non basta guardarle una
per una: quelle somiglianze vanno risolte tutte insieme, come un sistema di
equazioni in cui ogni riga tira le altre, e questo aggiunge un fattore. Il
risultato è che il lavoro cresce col **cubo**: raddoppiare i dati lo moltiplica
per otto ($2 \times 2 \times 2$), e passare da mille a diecimila punti lo
moltiplica per mille. È il motivo per cui non
addestreremo mai un processo gaussiano sulle foto di tutto internet.

Una scorciatoia esiste, ed è proprio il riassunto che il metodo si rifiutava di
fare. Invece di tenere tutte le cartelle se ne scelgono un centinaio, casi
rappresentativi a cui ricondurre gli altri, e il conto torna abbordabile. Il
prezzo si paga sull'ingrediente per cui si era scelto questo modello: le stime
reggono, i margini di fiducia diventano meno affidabili. E resta in piedi la
scommessa di partenza, la regola di somiglianza che abbiamo adottato, che è
un'ipotesi sul mondo e sul mondo va controllata.

`````

`````{tab} Superiore

Il collo di bottiglia è l'inversione (in pratica, la fattorizzazione di
Cholesky) di $\mathbf{K} + \sigma_n^2 \mathbf{I}$: costo $O(m^3)$ in tempo e
$O(m^2)$ in memoria, il caso peggiore dell’$O(m^2)$–$O(m^3)$ visto per la SVM
con kernel, e qui senza sconti. Oltre qualche decina di migliaia di punti il metodo
esatto diventa proibitivo. Esistono approssimazioni *sparse*, si riassume il
dataset con $p \ll m$ punti "induttori", scendendo a $O(m p^2)$, ma pagano in
fedeltà proprio sulla merce di casa: la qualità delle incertezze. A ciò si
aggiunge la sensibilità alla scelta del kernel, che incorpora ipotesi forti
(con l'RBF, la regolarità infinita) da verificare sul problema reale.

`````

Il suo territorio, allora, è l'opposto del big data: **pochi dati costosi**.
Esperimenti di laboratorio dove ogni misura vale una giornata di lavoro,
simulazioni ingegneristiche da ore di calcolo l'una, prove sul campo che non
si possono ripetere. E il caso che abbiamo già incontrato: l’**ottimizzazione
bayesiana degli iperparametri** {cite}`snoek2012practical`, dove ogni "dato" è
un intero addestramento e il processo gaussiano fa da mappa (stima più
incertezza) per decidere quale configurazione provare dopo.
{doc}`Trovare gli iperparametri <iperparametri>` racconta quel meccanismo dal
lato di chi lo usa, e il processo gaussiano ne è il motore.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un processo gaussiano non sceglie **una** curva: tiene in mano tutte quelle
  che i dati non hanno ancora escluso, e per ogni punto risponde con due
  numeri, la stima e quanto fidarsene. «Domani tra 21 e 27», non «domani 24».
- L'ingrediente che tiene insieme il fascio è la regola del buon senso: **punti
  vicini hanno valori simili**. Quanto lontano arrivi l'effetto di una misura
  lo decide una sola manopola, il raggio d'influenza: corto, curve nervose;
  lungo, curve morbide.
- La **banda d'incertezza** si stringe accanto ai dati e si riapre dove
  mancano, compresi i buchi *in mezzo* alle misure. A distinguere una
  previsione affidabile da una azzardata è avere o non avere un dato vicino,
  più che stare dentro o fuori dall'intervallo esplorato.
- Una banda larghissima vale come ammissione: il modello dichiara di non
  sapere, e pochi altri metodi lo fanno.
- Il prezzo è che **non scala**: a ogni previsione riapre l'archivio di tutte
  le misure e le confronta fra loro, e raddoppiare i dati moltiplica il lavoro
  per otto. È perfetto quando i dati sono **pochi e costosi** (un esperimento,
  una simulazione, un addestramento intero da provare) e impensabile quando
  sono milioni.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un processo gaussiano non impara una curva sola: mantiene una distribuzione
  su **tutte le curve compatibili con i dati** e per ogni punto restituisce
  una media e un'incertezza; «tra 21 e 27», non «24 e basta».
- Il **kernel** codifica la somiglianza ("punti vicini hanno valori simili");
  la lunghezza-scala $\ell$ decide fin dove arriva l'influenza di
  un'osservazione. I suoi iperparametri si stimano massimizzando la
  **verosimiglianza marginale**, che contiene già il rasoio di Occam ma non è
  concava: da qui le ripartenze multiple.
- La **banda d'incertezza** si stringe sui punti osservati e si riapre dove i
  dati mancano, buchi interni compresi: il modello dichiara quanto non sa.
  $\boldsymbol{\mu}_* \pm
  2\sqrt{\operatorname{diag}(\boldsymbol{\Sigma}_*)}$ è la banda **sulla
  funzione**; per una nuova osservazione va aggiunto $\sigma_n^2$.
- Il costo cresce come il **cubo del numero di esempi**: raddoppiare i dati
  costa otto volte il tempo. Improponibile sui grandi dataset, perfetto con
  **pochi dati costosi** (esperimenti, simulazioni, ottimizzazione bayesiana
  degli iperparametri).
```

`````

Fin qui la forma del modello l'abbiamo scelta noi, una per problema: una retta,
un albero, un confine largo, un fascio di curve, dei gruppi trovati senza
etichette. Cambiava il problema e si cambiava attrezzo, e cambiava anche il
metro, perché dove una risposta giusta non esiste il voto si dichiara invece di
calcolarlo. Resta uguale la pretesa di un conto onesto, e va portata intatta
in {doc}`Reti neurali </RetiNeurali/overview>`, dove invece l'attrezzo è uno
solo e prende la forma che serve impilando pezzi tutti uguali.
