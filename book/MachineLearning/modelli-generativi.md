# Classificare descrivendo: analisi discriminante e naive Bayes

Chiedi a un ornitologo come distingua una cornacchia da una gazza e non ti
risponderà con un confine. Ti dirà **com'è fatta una cornacchia**: grigia e nera,
tozza, coda corta, becco robusto. E com'è fatta una gazza: bianca e nera, più
snella, con quella coda lunghissima che non si può sbagliare. Il confine fra le
due specie non lo ha mai tracciato; ce l'ha in testa come conseguenza di due
descrizioni.

Tutti i classificatori visti finora fanno il contrario. La regressione logistica
cerca la retta che separa meglio; l'albero cerca la sequenza di domande che
separa meglio; la SVM cerca il corridoio più largo fra le due classi. Nessuno di
loro sa **com'è fatta** una classe: sanno solo dove finisce una e comincia
l'altra, che è un'informazione più povera e, spesso, più difficile da ottenere.

Questa sezione racconta l'altra strada. Si impara a descrivere ciascuna classe,
una per volta, e il confine si ricava dopo, con una riga di conto. La famiglia
si chiama dei modelli **generativi**, e i suoi tre membri classici (analisi
discriminante lineare, quadratica, e naive Bayes) hanno tutti più di
cinquant'anni e sono tutti ancora in uso.

## Due modi di rispondere alla stessa domanda

Alla fine ogni classificatore deve produrre la stessa cosa: dato un esempio,
quanto è probabile che appartenga a ciascuna classe. Cambia da dove ci si arriva.

`````{tab} Elementare

Una moneta raccolta per terra, da uno o da due euro? Sul tavolo ci sono una
bilancia, un calibro e un barattolo di mille monete già identificate.

Versi il barattolo sul foglio e segni ogni moneta come un punto, il peso in
orizzontale e il diametro in verticale. Poi tiri la riga che tiene i due mucchi
più separati. Adesso pesi la moneta nuova, la misuri, guardi da che parte cade.
Della moneta da un euro non hai imparato niente in particolare. Hai imparato
dove finisce.

Oppure il barattolo lo dividi in due mucchi. Delle monete da un euro calcoli
peso medio, diametro medio, e di quanto le singole si scostano da quelle medie;
poi rifai tutto sul mucchio da due. Adesso hai due **descrizioni**, e alla
moneta nuova fai due domande. Quanto sarebbe strana fra quelle da un euro? E fra
quelle da due? Vince chi la trova meno strana, con una correzione che nel
barattolo si legge: settecento delle mille erano da un euro e trecento da due,
quindi a parità di stranezza la dai da un euro.

Con una descrizione in mano fai una cosa che con la riga non si poteva fare.
Sorteggi un peso e un diametro che le stiano dentro, ed ecco sul foglietto una
moneta da un euro credibile che nel barattolo non c'era. Il nome **generativo**
viene da qui, perché da quello che ha imparato il metodo sa tirare fuori
esemplari nuovi.

Il giorno che nel barattolo finiscono i cinquanta centesimi, ne calcoli media e
scostamenti e la terza descrizione è pronta; le prime due restano quelle di
ieri, mentre la riga andrebbe ritracciata da capo. E la moneta col bordo
ammaccato, che nel calibro non entra dritta, la giudichi col solo peso, perché
una descrizione si può usare a pezzi.

Togli dal barattolo tutto tranne venti monete, e di ognuna misura anche
spessore, colore del bordo e usura. Il peso medio delle monete da un euro, e
quanto quel peso varia, escono lo stesso, perché li calcoli sulle monete da un
euro e basta. La riga invece deve accontentare tutti i punti in una volta sola,
e quando i punti sono pochi ne basta uno fuori posto a farla girare.

Poi sul tavolo arriva un gettone del luna park, o una moneta straniera, o un
falso fatto male. Sta lontano da tutti e due i centri, e le due descrizioni lo
trovano stranissimo tutte e due. La riga quella parola non ce l'ha: qualunque
cosa le metti sopra cade a destra o a sinistra, e il gettone esce come una
moneta da due euro con la stessa disinvoltura di una vera.

Il conto si paga quando la descrizione che ti sei dato è sbagliata. Hai dato per
buono che ogni taglio faccia un mucchio solo, tondo e compatto, e invece le
monete da due euro sono di due serie, una più pesante e una più leggera, cioè
due mucchietti staccati. La tua descrizione ne fa la media e finisce a metà
strada, dove monete da due euro non ce ne sono, e da lì sbagli anche pezzi che
una riga tirata a occhio avrebbe messo dalla parte giusta. Raccontare com'è
fatta ogni specie vuol dire pagare ogni dettaglio raccontato male, compresi
quelli che alla domanda non servivano.

`````

`````{tab} Superiore

La distinzione è quella fra classificatori **discriminativi** e **generativi**.

Un classificatore discriminativo modella direttamente la posteriore
$p(y \mid \mathbf{x})$ (regressione logistica, alberi, reti) o addirittura solo il
confine di decisione senza probabilità (SVM, percettrone). Un classificatore
generativo modella la **congiunta** $p(\mathbf{x}, y) = p(\mathbf{x} \mid y)\,p(y)$,
cioè la distribuzione dei dati **dentro ciascuna classe** più la frequenza delle
classi, e ricava la posteriore con il teorema di Bayes:

$$
p(y = k \mid \mathbf{x}) =
\frac{p(\mathbf{x} \mid y = k)\; \pi_k}
     {\sum_{j} p(\mathbf{x} \mid y = j)\; \pi_j},
\qquad \pi_k = p(y = k).
$$

Il nome «generativo» viene da una proprietà che il discriminativo non ha:
avendo $p(\mathbf{x} \mid y)$ si possono **campionare esempi nuovi** di una
classe. Il modello non riassume i dati, li sa rifare, ed è la stessa parola che
il libro userà per i modelli di generazione di immagini e testo, dove la
famiglia è la stessa e cambia solo quanto è espressiva la $p(\mathbf{x} \mid y)$.

Le conseguenze pratiche di modellare $p(\mathbf{x}\mid y)$ invece di
$p(y \mid \mathbf{x})$ sono quattro, e tornano tutte più avanti nel libro:

1. si ottiene una **densità**, quindi il rilevamento di anomalie e la
   rilevazione di input fuori distribuzione vengono in regalo;
2. i parametri si stimano in **forma chiusa**, ciascuno da tutti i dati della
   sua classe e non tutti insieme dentro un'unica ottimizzazione, e questo si
   sente quando gli esempi sono pochi rispetto alle feature;
3. le classi si stimano **una alla volta e indipendentemente**: aggiungere una
   classe non richiede di riaddestrare le altre, e i dati mancanti si trattano
   marginalizzando invece che imputando;
4. se il modello di $p(\mathbf{x}\mid y)$ è sbagliato, l'errore si paga anche
   dove non serviva: il generativo spende capacità a descrivere aspetti dei dati
   che non contano per la decisione.

`````

Resta da dire di quale descrizione stiamo parlando. Il caso classico è il più
semplice possibile: ogni classe è una **campana gaussiana**, cioè una collina di
probabilità con un **centro** (dove sta il tipico esemplare della classe) e una
**forma** (quanto e in quali direzioni gli esemplari se ne allontanano). Due
ingredienti, e si calcolano con due medie: la media dei punti della classe dà il
centro, la media dei loro scarti dà la forma. Fine.

È un caso fortunato, e conviene dire subito perché. Quando le etichette non ci
sono, gli stessi due ingredienti vanno **indovinati** insieme all'appartenenza
di ciascun punto, e ci vuole una procedura iterativa: è quello che la sezione
sul clustering farà con le misture gaussiane e l'algoritmo EM. Qui le etichette
ci sono, quindi non c'è niente da indovinare.

## Analisi discriminante: lineare o quadratica

Ronald Fisher affronta il problema nel 1936, su dei fiori
{cite}`fisher1936use`. Il botanico Edgar Anderson aveva misurato lunghezza e
larghezza di petali e sepali di centocinquanta iris, cinquanta per ciascuna di
tre specie; la domanda era se quelle quattro misure bastassero a distinguerle.
Conviene riportare l'avvertenza che Fisher mette nel suo stesso articolo,
perché quasi nessuno di quelli che riusano questi dati la conosce: due delle
tre specie vengono dalla penisola di Gaspé, in Québec, mentre la terza, *Iris
virginica*, «differisce dagli altri due campioni per non essere stata raccolta
nella stessa colonia naturale», il che «potrebbe alterare parecchio sia le
medie sia le loro variabilità». Fisher cercava la combinazione delle quattro
che separasse al meglio le specie, e il metodo che ne uscì porta il suo nome.
È lo stesso `iris` che il capitolo sull'interpretabilità darà in pasto a un
albero, ed è probabilmente il dataset più riusato della storia della
statistica.[^eugenics]

[^eugenics]: L'articolo esce sugli *Annals of Eugenics*, che è il nome della
    rivista fino al 1954, e Fisher ne fu a lungo redattore. È un fatto
    bibliografico e non un dettaglio da nascondere: la statistica inferenziale
    del primo Novecento nasce in buona parte dentro quel programma di ricerca,
    e i metodi che ne uscirono sono validi indipendentemente da esso. Il
    capitolo sull'AI responsabile torna sul rapporto fra strumenti statistici e
    usi che se ne fanno.

`````{tab} Elementare

Torniamo alle monete, e mettiamo che i due tagli abbiano la stessa forma, nel
senso che variano allo stesso modo (chi è più pesante è anche un po’ più largo,
nella stessa misura per tutte e due), e che a distinguerli sia solo dove sta il
centro.

In questo caso capire da quale specie viene una moneta nuova è quasi come
chiedersi a quale dei due centri sono più vicino. Il «quasi» sta in due
accortezze. La prima è misurare la distanza nella forma giusta: se le monete
variano molto in peso e poco in diametro, un grammo di differenza conta meno di
un millimetro. La seconda è la solita correzione per quanto sono comuni le due
specie, che non sparisce nemmeno qui. Il confine che ne esce è una retta, ed è il
metodo di Fisher, l'analisi discriminante **lineare**, LDA per gli amici.

Se invece le due specie hanno forme diverse (una varia tanto in peso, l'altra
tanto in diametro) la vicinanza al centro da sola inganna. Una specie molto
variabile trova poco strano qualunque valore, e a lasciarla fare si prenderebbe
tutte le monete dubbie; quindi dal suo giudizio si toglie tanto più quanto più
quella specie è larga. Con una forma sola quello sconto sarebbe stato identico
per le due specie e non avrebbe spostato il confine di un millimetro; con due
forme diverse decide. Impari una forma per ciascuna, e il confine che ne esce si
incurva. È l'analisi discriminante **quadratica**, QDA.

Sembra che convenga sempre la seconda, visto che può fare tutto quello che fa la
prima. Non è così, ed è il compromesso bias-varianza in una delle sue forme più
nitide, perché imparare una forma per ciascuna classe vuol dire stimare il
doppio dei numeri con gli stessi dati, quindi stimarli peggio. Con classi che
davvero hanno la stessa forma, la QDA spende parametri per scoprire una cosa che
era già vera e ci rimette; con classi di forma diversa, la LDA non ha proprio
modo di accorgersene.

C'è anche una via di mezzo, e si prende quando le monete a disposizione sono
poche. Si stimano le due forme separate e poi le si tira verso la forma unica,
tenendo un po’ di ciascuna, e quanto tirare è una manopola da girare.

`````

`````{tab} Superiore

Si assume $p(\mathbf{x} \mid y = k) = \mathcal{N}(\mathbf{x} \mid
\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$. Sostituendo in Bayes e prendendo il
logaritmo, la regola di decisione confronta le **funzioni discriminanti**

$$
\delta_k(\mathbf{x}) = -\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_k)^{\!\top}
\boldsymbol{\Sigma}_k^{-1}(\mathbf{x}-\boldsymbol{\mu}_k)
-\tfrac{1}{2}\log\lvert\boldsymbol{\Sigma}_k\rvert + \log \pi_k ,
$$

e assegna $\mathbf{x}$ alla classe con $\delta_k$ massima. Il primo termine è la
**distanza di Mahalanobis** al quadrato, cioè la distanza euclidea misurata
nella metrica che la covarianza della classe induce: è la formalizzazione di
«misurare la distanza nella forma giusta».

Il confine fra due classi è $\delta_k(\mathbf{x}) = \delta_\ell(\mathbf{x})$, e
la sua forma dipende da una sola ipotesi.

**Covarianze diverse (QDA).** I termini
$\mathbf{x}^{\!\top}\boldsymbol{\Sigma}_k^{-1}\mathbf{x}$ non si cancellano, e il
confine è una quadrica (iperbole, ellisse, parabola secondo il caso).

**Covarianze uguali (LDA), $\boldsymbol{\Sigma}_k = \boldsymbol{\Sigma}$
per ogni $k$.** Il termine quadratico è lo **stesso** nelle due funzioni
discriminanti e sparisce nella differenza. Sviluppando:

$$
\delta_k(\mathbf{x}) = \mathbf{x}^{\!\top} \boldsymbol{\Sigma}^{-1}
\boldsymbol{\mu}_k
- \tfrac{1}{2} \boldsymbol{\mu}_k^{\!\top} \boldsymbol{\Sigma}^{-1}
\boldsymbol{\mu}_k + \log \pi_k
+ \underbrace{\bigl(-\tfrac{1}{2}\mathbf{x}^{\!\top}\boldsymbol{\Sigma}^{-1}
\mathbf{x} - \tfrac{1}{2}\log\lvert\boldsymbol{\Sigma}\rvert\bigr)}_{
\text{uguale per ogni } k},
$$

dove il termine raccolto dalla graffa non è costante in $\mathbf{x}$ (è proprio
quello quadratico), ma è lo stesso per tutte le classi, quindi sparisce nella
**differenza** $\delta_k - \delta_\ell$, che è ciò da cui il confine dipende.
Tolto quello, quel che resta è **affine in $\mathbf{x}$**: il confine è un
iperpiano. È anche l'enunciato che il collaudo numerico verifica, dato che
`decision_function` restituisce proprio quella differenza.

Da qui il nome: l'analisi discriminante di Fisher si chiama *lineare* perché
lineare le viene il confine, e la linearità non è un'ipotesi imposta ma una
conseguenza dell'aver condiviso la covarianza.

Il conto dei parametri spiega il compromesso. Con $d$ feature e $K$ classi, la
LDA stima $K$ medie più **una** covarianza, cioè $Kd + d(d+1)/2$ numeri; la QDA
ne stima $K$, cioè $Kd + K\,d(d+1)/2$. Per $d = 20$ e $K = 2$ sono $250$ contro
$460$: quasi il doppio, e la parte che raddoppia è quella difficile, perché
stimare una covarianza è stimare $d^2/2$ numeri da dati che ne informano poco.
Da qui la **regularized discriminant analysis** di Friedman
{cite}`friedman1989regularized`, che interpola fra le due mescolando
$\boldsymbol{\Sigma}_k$ con la covarianza comune. In scikit-learn quella
interpolazione non c'è (`QuadraticDiscriminantAnalysis(reg_param=...)` fa solo
l'altra metà, cioè tira ciascuna $\boldsymbol{\Sigma}_k$ verso l'identità), e
c'è invece un rimedio diverso e complementare,
`LinearDiscriminantAnalysis(shrinkage=...)`, che tira la covarianza **comune**
verso un multiplo dell'identità:
$(1-\alpha)\hat{\boldsymbol{\Sigma}} +
\alpha\,\frac{\operatorname{tr}\hat{\boldsymbol{\Sigma}}}{d}\mathbf{I}$.
Cura cioè il rumore della stima, non la differenza fra le classi (e vuole
`solver="lsqr"` o `"eigen"`: con il solver predefinito il parametro solleva un
errore invece di essere ignorato, che è il modo giusto di comportarsi).

Due parentele, con la regressione logistica e con le misture gaussiane. La LDA
produce una posteriore che, per due classi, è esattamente una **sigmoide di una
funzione affine**, cioè la stessa forma funzionale della regressione logistica; e
il legame con le misture gaussiane è ancora più stretto, perché **la LDA è una
mistura gaussiana a covarianza condivisa in cui le variabili latenti sono
osservate**. All'EM della sezione sul clustering la seconda parentela si legge al
contrario: il passo E, che là dovrà stimare le responsabilità, qui è dato
(valgono $0$ e $1$, e le sanno tutti), e resta il solo passo M, che sono le due
medie e la covarianza comune, eseguito una volta.

`````

Le due situazioni, misurate: due classi con la stessa forma e due classi con
forme diverse, gli stessi $200$ esempi di addestramento, la stessa prova su
ventimila esempi mai visti.

```python
import numpy as np
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.linear_model import LogisticRegression

def genera(n, forma_uguale, seme):
    """Due classi gaussiane, con la stessa forma oppure con forme diverse."""
    r = np.random.default_rng(seme)
    C0 = np.array([[2.0, 1.2], [1.2, 1.0]])
    C1 = C0 if forma_uguale else np.array([[0.6, -0.5], [-0.5, 2.2]])
    y = r.integers(0, 2, n)
    return (np.where(y[:, None] == 0,
                     r.multivariate_normal([0, 0], C0, n),
                     r.multivariate_normal([1.6, 1.2], C1, n)), y)

print(f"{'':30} {'LDA':>13} {'QDA':>13} {'logistica':>13}")
for uguale in (True, False):
    Xte, yte = genera(20_000, uguale, 999)
    col = []
    for M in (LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
              LogisticRegression):
        # venti addestramenti da 200 esempi: la deviazione dice quanto ballano
        s = [M().fit(*genera(200, uguale, 10 + k)).score(Xte, yte) for k in range(20)]
        col.append(f"{np.mean(s):.3f} ±{np.std(s):.3f}")
    etichetta = "stessa forma, due classi" if uguale else "forme diverse"
    print(f"{etichetta:30} {col[0]:>13} {col[1]:>13} {col[2]:>13}")
```

```text
                                         LDA           QDA     logistica
stessa forma, due classi        0.728 ±0.003  0.726 ±0.003  0.728 ±0.003
forme diverse                   0.813 ±0.005  0.855 ±0.002  0.810 ±0.007
```

La colonna del $\pm$ è la deviazione standard fra venti addestramenti, cioè
quanto quel numero balla se si ripete tutto: senza di lei il resto della
tabella non si legge. Nella prima riga i tre valori stanno dentro un $\pm 0{,}003$
l'uno dall'altro: sono lo **stesso** numero scritto tre volte, e la conclusione è
che quando le classi hanno la stessa forma non c'è niente da guadagnare a
imparare due forme (né a passare a un discriminativo). Nella seconda riga la QDA
sta quattro punti sopra le altre due, con la deviazione più piccola di tutte: lì
la differenza è reale.

Notare anche chi resta indietro insieme a chi: LDA e regressione logistica si
muovono appaiate in tutte e due le righe, che è quello che il paragrafo formale
prevede, perché tracciano lo stesso tipo di confine e differiscono solo su come
ne stimano la posizione.

E che il confine della LDA sia davvero una retta non è una cosa da credere
sulla parola. Il collaudo è questo: si prende il punteggio con cui il modello
decide, si cerca la retta che meglio lo imita, e si guarda **di quanto** i due
si scostano nel punto peggiore. Se il punteggio è una retta lo scarto deve
venire zero.

```python
X, y = genera(4000, True, 1)
lda = LinearDiscriminantAnalysis().fit(X, y)
qda = QuadraticDiscriminantAnalysis().fit(X, y)

r = np.random.default_rng(0)
P = r.normal(0, 3, (500, 2))          # cinquecento punti a caso nel piano
base = np.c_[P, np.ones(len(P))]      # la piu' generale funzione affine del piano

def scarto_dall_affine(decisione):
    """Quanto la funzione di decisione si scosta dalla piu' vicina retta."""
    coef = np.linalg.lstsq(base, decisione(P), rcond=None)[0]
    return np.abs(decisione(P) - base @ coef).max()

print(f"LDA, scarto dall'affine: {scarto_dall_affine(lda.decision_function):.2e}")
print(f"QDA, scarto dall'affine: {scarto_dall_affine(qda.decision_function):.2e}")
```

```text
LDA, scarto dall'affine: 8.88e-15
QDA, scarto dall'affine: 7.21e+00
```

Lo scarto della LDA è $10^{-15}$, cioè un miliardesimo di miliardesimo: non è
«quasi zero», è zero, e quel che resta sono gli arrotondamenti del calcolatore.
Il punteggio della LDA **è** una retta, non le somiglia. La QDA se ne scosta di
$7{,}2$, e nessuna retta la approssima. È la stessa cosa che il paragrafo
formale ottiene con l'algebra, dove i termini al quadrato si cancellano fra le
due classi perché sono identici.

```{figure} ../figures/lda-qda-naive-bayes.svg
:name: fig-lda-qda
:alt: "Tre pannelli sugli stessi due gruppi di punti, teal e terracotta, che si sovrappongono in parte. Nel primo, LDA, le due classi hanno la stessa identica ellisse di forma in due posizioni diverse, e il confine fra loro e una retta. Nel secondo, QDA, ogni classe ha la sua ellisse, con orientamenti diversi, e il confine e una curva. Nel terzo, naive Bayes gaussiano, le ellissi hanno gli assi obbligatoriamente paralleli agli assi del grafico, perche l ipotesi di indipendenza vieta le diagonali, e il confine e una curva diversa dalla precedente."
:width: 100%

Le tre ipotesi, disegnate. L'ovale che circonda ciascun gruppo (un’**ellisse**)
è la forma che quel metodo si concede per descrivere la classe, e da sola decide
la forma del confine: una **sola forma**, la stessa per le due classi in due
posizioni diverse, dà una retta; **due forme diverse** danno una curva. Nessuno
dei confini è stato disegnato: sono tutti conseguenze delle ellissi. Il terzo
pannello anticipa il metodo del paragrafo che segue, che le ellissi le obbliga a
stare dritte, con gli assi paralleli a quelli del grafico.
```

## Naive Bayes: l'ipotesi sfacciata che funziona

Il terzo membro della famiglia si ottiene da una semplificazione che, detta ad
alta voce, sembra insostenibile.

`````{tab} Elementare

Descrivere una classe con centro e forma costa: la forma dice anche come le
caratteristiche vanno insieme (se le monete più pesanti sono anche più larghe), e
con venti caratteristiche le coppie da guardare sono $20 \times 19 / 2$, cioè
centonovanta. Tante da imparare.

Il **naive Bayes** taglia corto e dichiara che quelle relazioni non esistono:
dentro una classe, ogni caratteristica va per conto suo. Peso e diametro non si
sanno l'uno dell'altro. Così di ogni classe basta imparare, una caratteristica
per volta, una media e una dispersione. Poi la moneta nuova si giudica su ogni
caratteristica separatamente, e i giudizi che ne escono si moltiplicano fra loro.

L'ipotesi è quasi sempre falsa, e non un po’: in un'email le parole «offerta» e
«gratis» si tirano dietro a vicenda, in una moneta peso e diametro pure. Il nome
lo ammette: *naive* vuol dire ingenuo.

Il fatto strano è che funziona lo stesso, e la ragione è che al classificatore
non serve avere ragione sulle probabilità, gli serve **mettere in classifica** le
classi nel giusto ordine. Può sbagliare di brutto sul «quanto» (dirà $0{,}999$
dove il vero è $0{,}7$, perché contando due volte prove che erano la stessa prova
si convince troppo) e azzeccare comunque il «quale». Domingos e Pazzani hanno
studiato proprio questo nel 1997 {cite}`domingos1997optimality`, mostrando che
l'insieme dei casi in cui il naive Bayes è ottimo è molto più grande di quello in
cui la sua ipotesi è vera.

Il corollario pratico tocca chi usa questi modelli. Le probabilità del naive
Bayes non si usano come probabilità. Come classifica sono buone, come numeri no.
Se servono probabilità di cui fidarsi (una soglia da tarare, un costo da
calcolare) vanno ricalibrate.

`````

`````{tab} Superiore

L'ipotesi è l'indipendenza condizionata delle feature **data la classe**:

$$
p(\mathbf{x} \mid y = k) = \prod_{j=1}^{d} p(x_j \mid y = k).
$$

Nel caso gaussiano equivale a imporre $\boldsymbol{\Sigma}_k$ **diagonale**, e i
parametri di covarianza crollano da $K\,d(d+1)/2$ a $Kd$: per $d = 20$ e
$K = 2$, da $420$ a $40$. Geometricamente, le ellissi di livello hanno gli assi paralleli
agli assi coordinati (nessuna rotazione), che è ciò che mostra il terzo pannello
di {numref}`fig-lda-qda`.

Il risultato di Domingos e Pazzani {cite}`domingos1997optimality` è che la
regione di ottimalità del naive Bayes sotto perdita $0$–$1$ è **strettamente più
ampia** di quella in cui vale l'indipendenza condizionata: l'errore
sull’*ordinamento* delle posteriori è un evento più raro dell'errore sulle
posteriori stesse, perché la funzione $\arg\max$ è invariante a un'ampia classe
di distorsioni monotone. Le stime restano però mal **calibrate**, tipicamente
sovrasicure, perché feature correlate contribuiscono evidenza ripetuta al
prodotto: chi ha bisogno delle probabilità e non solo dell'etichetta ricalibri
(Platt scaling, isotonica).

Nel caso discreto (conteggi di parole) il modello prende il nome di naive Bayes
**multinomiale**, e con lo smoothing di Laplace è la base storica della
classificazione dei testi. Il capitolo sul *Natural Language Processing* lo
tratta in quella veste, con il conto dello smoothing: qui interessa come membro
della famiglia generativa, non come classificatore di documenti.

`````

## Quando il generativo vince: pochi dati

Resta la domanda che decide se questa famiglia serve ancora, e ha una risposta
misurabile. Andrew Ng e Michael Jordan la formulano nel 2001
{cite}`ng2001discriminative`, e il confronto che scelgono è il più pulito
possibile: naive Bayes contro regressione logistica, cioè due modelli che
arrivano alla **stessa identica formula** per decidere, e differiscono solo su
come ne ricavano i numeri dai dati.

Il risultato ha la forma di una gara con due tempi. Il generativo parte meglio:
con pochi esempi è già vicino al meglio che sa fare. Il discriminativo parte
peggio, ma il meglio che sa fare è **più alto**, e con abbastanza esempi ci
arriva. Su pochi dati vince il primo; su tanti, il secondo.

L'esperimento qui sotto rifà quel confronto, con una colonna in più che cambia le
conclusioni.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

D = 40
rng = np.random.default_rng(0)
MU = rng.choice([-1, 1], D) * 0.35     # le due classi differiscono in ogni feature

def dati(n, r):
    """Due classi gaussiane a feature indipendenti: l'ipotesi naive qui e' VERA."""
    y = r.integers(0, 2, n)
    return r.normal(0, 1, (n, D)) + np.outer(y, MU), y

X_test, y_test = dati(20_000, np.random.default_rng(999))

print(f"{'n':>6} {'naive Bayes':>12} {'logistica (default)':>21} {'logistica nuda':>16}")
for n in (20, 40, 80, 200, 600, 2000):
    a, b, c = [], [], []
    for s in range(15):
        r = np.random.default_rng(100 + s)
        X, y = dati(n, r)
        if len(np.unique(y)) < 2:
            continue
        a.append(GaussianNB().fit(X, y).score(X_test, y_test))
        b.append(LogisticRegression(max_iter=5000).fit(X, y).score(X_test, y_test))
        c.append(LogisticRegression(C=1e6, max_iter=5000).fit(X, y).score(X_test, y_test))
    print(f"{n:6d} {np.mean(a):12.3f} {np.mean(b):21.3f} {np.mean(c):16.3f}")
```

```text
     n  naive Bayes   logistica (default)   logistica nuda
    20        0.640                 0.708            0.663
    40        0.710                 0.746            0.693
    80        0.778                 0.771            0.722
   200        0.828                 0.817            0.796
   600        0.851                 0.847            0.846
  2000        0.860                 0.859            0.859
```

La terza colonna è la logistica **senza regolarizzazione**, che è quella del
confronto originale, e su di lei il fenomeno si vede intero: a $n = 80$ il naive
Bayes sta a $0{,}778$ e lei a $0{,}722$, cinque punti e mezzo sotto; a $n = 200$
sono ancora tre punti; da $n = 600$ in poi si raggiungono e restano insieme. Con
quaranta caratteristiche e ottanta esempi la logistica ha due esempi per
parametro, e con così poco non impara; il naive Bayes ne stima anche di più
(centosessanta: una media e una varianza per classe e per colonna), ma li stima
**uno alla volta**, ciascuno con tutti i dati della sua classe, e se la cava.

La seconda colonna è la logistica **come la si usa oggi**, cioè col
`penalty="l2"` che scikit-learn applica per default, ed è il motivo per cui
questo esperimento conviene rifarlo invece di citarlo. Quel freno rimedia
quasi tutto lo svantaggio: a $n = 20$ e $n = 40$ la logistica regolarizzata
sta davanti al naive Bayes, e da $n = 80$ in poi le tre colonne si
assottigliano fino a coincidere. Il fenomeno del 2001 è reale e si riproduce;
ma il rimedio che gli si oppone oggi è acceso per impostazione predefinita, e
chi confronta i due modelli con i default della libreria non lo vede.

Una precisazione onesta su questa tabella, perché è quella che le dà il suo
limite. I dati qui sono stati fabbricati **a feature indipendenti**, cioè nel
mondo in cui l'ipotesi del naive Bayes è vera. Questo rende visibile il primo
tempo della gara, la partenza rapida del generativo, e rende **invisibile il
secondo**: se il modello del naive Bayes è quello giusto, i due metodi hanno lo
stesso tetto, e quel tetto è il minimo teorico del problema ($0{,}866$, che si
calcola). Infatti l'ultima riga li dà appaiati a $0{,}860$ e $0{,}859$, e nessuno
dei due supera mai l'altro: l'asintoto più alto del discriminativo si vede solo
quando l'ipotesi naive è **falsa**, che è il caso qui sotto. Rifacendo tutto con
feature correlate a $0{,}25$ il naive Bayes resta indietro a ogni numerosità,
perché al vantaggio di stimare poco si somma il costo di un'ipotesi falsa. Il
vantaggio dei pochi dati è reale; non è un salvacondotto.

## In pratica

```python
from scipy.special import logsumexp

X, y = genera(2000, True, 0)
lda = LinearDiscriminantAnalysis(store_covariance=True).fit(X, y)
print("accuratezza LDA:", round(lda.score(*genera(20_000, True, 999)), 3))

# la LDA ha imparato due gaussiane: da quelle si ricava anche p(x), non solo la
# classe. E' l'unica cosa che un discriminativo non puo' dare.
inversa = np.linalg.inv(lda.covariance_)
_, logdet = np.linalg.slogdet(lda.covariance_)

def log_densita(P):
    """log p(x): quanto e' verosimile un punto, per il modello gia' addestrato."""
    per_classe = [-0.5*np.einsum("ij,jk,ik->i", P - m, inversa, P - m)
                  - 0.5*logdet - np.log(2*np.pi) + np.log(q)
                  for m, q in zip(lda.means_, lda.priors_)]
    return logsumexp(per_classe, axis=0)

fuori = np.array([[14.0, -11.0]])          # un punto che non c'entra niente
print(f"log p(x) di un punto in mezzo ai dati: {log_densita(X[:1])[0]:9.2f}")
print(f"log p(x) di un punto lontanissimo    : {log_densita(fuori)[0]:9.2f}")
print(f"e sullo stesso punto lontano si dichiara sicuro al "
      f"{lda.predict_proba(fuori).max():.2%}")
```

```text
accuratezza LDA: 0.731
log p(x) di un punto in mezzo ai dati:     -2.36
log p(x) di un punto lontanissimo    :   -711.59
e sullo stesso punto lontano si dichiara sicuro al 99.99%
```

Le ultime tre righe sono il gettone fra le monete, misurato. Il punto
$(14, -11)$ non ha niente a che vedere con questi dati, e la **densità** lo dice
senza esitazioni. I due numeri stampati sono logaritmi, e la differenza fra loro
è di settecentonove: non «settecento volte meno probabile», ma un rapporto di
$10^{308}$, cioè un $1$ seguito da trecentotto zeri. Quel punto, per il modello,
semplicemente non capita. La **classificazione** dello stesso punto, invece,
esce al $99{,}99\%$ di sicurezza, perché una volta scelto da che parte della
retta si trova non c'è altro da dire.

I due numeri vengono dallo stesso modello, addestrato una volta sola, e sono
la ragione per cui conviene avere in casa un generativo: la sicurezza di un
classificatore è sempre relativa alle classi che conosce, e da sola non
distingue «è certamente una gazza» da «non ho idea di cosa sia, ma se devo
scegliere dico gazza». La densità quella distinzione la fa.

Quando conviene prenderli in considerazione, in concreto:

- **come riferimento di partenza**: la LDA non ha iperparametri, si addestra in
  un istante su qualunque tabella, e dà un numero contro cui misurare tutto il
  resto. Se il modello elaborato non la batte, il problema è nei dati o
  nell'impostazione, non nel modello;
- **con pochi esempi e molte colonne**, che è la situazione tipica dei dati
  clinici e sperimentali: è la riga $n = 80$ della tabella;
- **quando serve una densità**, cioè quando bisogna accorgersi degli esempi che
  non somigliano a niente di visto. La sezione sui dati che cambiano userà
  proprio questo;
- **per schiacciare i dati in poche dimensioni senza perdere le classi**: la
  LDA li proietta su al più $K-1$ direzioni, scelte apposta perché su quelle le
  classi si distinguano. È la cugina supervisionata dell'analisi delle
  componenti principali, che la sezione sul clustering costruirà: quella cerca
  le direzioni in cui i dati **variano** di più, questa quelle in cui le
  **classi** si distinguono di più, e le due possono benissimo non coincidere.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Ci sono due modi di classificare. Imparare **dove passa il confine** (la
  logistica, gli alberi, le SVM) e imparare **com'è fatta ogni classe**, per poi
  ricavarne il confine. Il secondo è quello dell'ornitologo, e si chiama
  **generativo**.
- Chi sa com'è fatta ogni classe sa anche riconoscere quello che **non somiglia
  a nessuna**: un gettone fra le monete. Chi ha imparato solo il confine no.
- **LDA**: una sola forma condivisa dalle due classi, e il confine viene una
  retta. **QDA**: una forma per classe, e il confine si incurva.
- Non conviene sempre la più flessibile: con classi che hanno davvero la stessa
  forma i tre metodi danno lo stesso numero ($0{,}728$, $0{,}726$, $0{,}728$, e
  ballano di $\pm 0{,}003$), mentre con forme diverse la QDA sta quattro punti
  sopra.
- Il **naive Bayes** dichiara che dentro una classe le caratteristiche non si
  parlano fra loro. È quasi sempre falso, e funziona lo stesso, perché per
  scegliere la classe basta l’**ordine**, non il valore esatto. Le sue
  probabilità però non vanno usate come probabilità: sono troppo sicure di sé.
- Il generativo dà il meglio **con pochi dati**: a ottanta esempi e quaranta
  colonne il naive Bayes sta cinque punti e mezzo sopra la logistica non
  regolarizzata. A seicento esempi il vantaggio è finito.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- **Discriminativo**: si modella $p(y \mid \mathbf{x})$. **Generativo**: si
  modella $p(\mathbf{x} \mid y)\,p(y)$ e si applica Bayes. Il secondo dà in più
  una densità (anomalie), la stima classe per classe e un migliore
  comportamento a pochi dati.
- La regola di decisione confronta
  $\delta_k(\mathbf{x}) = -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_k)^\top
  \boldsymbol{\Sigma}_k^{-1}(\mathbf{x}-\boldsymbol{\mu}_k)
  -\frac{1}{2}\log\lvert\boldsymbol{\Sigma}_k\rvert + \log\pi_k$: distanza di
  **Mahalanobis**, più il termine di volume e la priore.
- Con $\boldsymbol{\Sigma}_k = \boldsymbol{\Sigma}$ il termine quadratico si
  cancella nella differenza e $\delta_k$ diventa **affine**: è la LDA, e la
  linearità è una conseguenza, non un'ipotesi. Verificato numericamente: scarto
  dalla più vicina funzione affine $8{,}9 \cdot 10^{-15}$ per la LDA contro
  $7{,}2$ per la QDA.
- Parametri (medie **più** covarianze): LDA $Kd + d(d+1)/2$, QDA
  $Kd + K\,d(d+1)/2$, naive Bayes gaussiano $2Kd$ (una media e una varianza per
  classe e per feature, niente termini incrociati). È il compromesso
  bias-varianza sul modello di $p(\mathbf{x}\mid y)$; la *regularized
  discriminant analysis* e lo `shrinkage` interpolano.
- **Naive Bayes**: $p(\mathbf{x}\mid y) = \prod_j p(x_j \mid y)$. La regione di
  ottimalità sotto perdita $0$–$1$ è più ampia di quella in cui l'ipotesi vale
  {cite}`domingos1997optimality`, perché conta l’$\arg\max$ e non il valore; le
  posteriori restano **sovrasicure** e vanno ricalibrate.
- **Ng e Jordan** {cite}`ng2001discriminative`: il generativo converge al
  proprio asintoto molto prima, ma quello del discriminativo è più alto **quando
  l'ipotesi del generativo è falsa**. La tabella qui sopra misura solo la prima
  metà, perché è costruita a feature indipendenti e lì i due asintoti coincidono
  (col tasso di Bayes, $0{,}866$). Il confronto è contro la logistica **non
  regolarizzata**; con l’$\ell_2$ di default il divario quasi sparisce, cioè il
  fenomeno è del 2001 e i default di oggi lo mascherano.
- La LDA è anche una **riduzione di dimensionalità supervisionata** su al più
  $K-1$ direzioni, ed è la mistura gaussiana della sezione sul clustering con le
  variabili latenti **osservate**: resta il solo passo M, eseguito una volta.
```

`````

Con questa sezione il capitolo ha chiuso il giro dei classificatori classici, e
lo ha chiuso tornando al punto di partenza da dietro: la regressione logistica
con cui tutto era cominciato e la LDA di Fisher tracciano lo stesso confine e non
sono lo stesso metodo, perché una guarda il confine e l'altra guarda le classi.
La prossima sezione abbandona del tutto le etichette e chiede ai dati di
raggrupparsi da soli, che è l'unica domanda a cui nessuno dei modelli visti
finora sa rispondere.
