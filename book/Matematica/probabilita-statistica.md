# Probabilità e statistica: convivere con l'incertezza

Nell'estate del 1654 due matematici francesi, Blaise Pascal e Pierre de
Fermat, si scambiano alcune lettere su un problema apparentemente frivolo:
come dividere la posta di un gioco d'azzardo interrotto prima della fine. Da
quella corrispondenza nasce, di fatto, la teoria della probabilità: la
disciplina che insegna a ragionare quando non sappiamo *con certezza* cosa
accadrà. È la stessa situazione del machine learning. Un modello non "sa" se
un'email è spam: stima *quanto è probabile* che lo sia. Misurare l'incertezza,
e aggiornarla quando arrivano nuovi dati, è metà del mestiere.

## Lo spazio delle possibilità

Ogni volta che c'è del caso in gioco, si parte elencando cosa può succedere.

`````{tab} Elementare

Lancia un dado. I risultati possibili sono $\{1,2,3,4,5,6\}$: è lo **spazio
campionario**. Un **evento** è un gruppo di questi risultati, per esempio "esce
un numero pari" $=\{2,4,6\}$. Se il dado è onesto, la probabilità di un evento è
semplicemente *casi favorevoli su casi possibili*: $P(\text{pari}) = 3/6 = 0{,}5$.

Tre regole di buon senso governano tutto: una probabilità sta sempre fra $0$
(impossibile) e $1$ (certo); l'evento "esce qualcosa" ha probabilità $1$; e se
due eventi non possono capitare insieme, la probabilità che ne capiti uno *o*
l'altro è la somma delle due.

`````

`````{tab} Superiore

Formalizziamo con lo spazio campionario $\Omega$ e gli eventi come suoi
sottoinsiemi $A\subseteq\Omega$, presi in una famiglia $\mathcal{F}$ chiusa
per complementare e unione numerabile (una $\sigma$-algebra): quando $\Omega$
non è numerabile non tutti i sottoinsiemi si possono misurare, e la
restrizione serve appena si passa alle variabili continue. Una misura di
probabilità soddisfa gli **assiomi di Kolmogorov** (1933):

$$
P(A)\ge 0, \qquad P(\Omega)=1, \qquad
P\!\Big(\bigcup_i A_i\Big)=\sum_i P(A_i),
$$

dove l'ultima uguaglianza vale per eventi $A_i$ a due a due disgiunti. Da questi
tre assiomi discende tutto il resto: $P(\varnothing)=0$, la probabilità del
complementare $P(A^c)=1-P(A)$, e la regola di inclusione-esclusione
$P(A\cup B)=P(A)+P(B)-P(A\cap B)$.

`````

## Variabili aleatorie

Spesso non ci interessa l'esito grezzo, ma un numero che gli associamo.

`````{tab} Elementare

Una **variabile aleatoria** è un numero il cui valore dipende dal caso. Se lancio
dieci monete e conto le teste, quel conteggio è una variabile aleatoria. Ne
esistono due specie. Quelle **discrete** contano cose separate ($0, 1, 2, \dots$
teste): a ciascun valore assegniamo una probabilità, e la somma fa $1$. Quelle
**continue** misurano grandezze che scorrono senza salti (un'altezza, un tempo di
attesa): qui la probabilità non si concentra in un singolo punto ma si spalma su
una curva, la **densità**, e le probabilità diventano *aree* sotto quella curva.

`````

`````{tab} Superiore

Una variabile aleatoria è una funzione $X:\Omega\to\mathbb{R}$. Se è discreta la
descrive la *funzione di massa* $p(x)=P(X=x)$, con $\sum_x p(x)=1$. Se è continua
la descrive la *densità* $f(x)\ge 0$ con $\int_{-\infty}^{+\infty} f(x)\,dx=1$, e

$$
P(a\le X\le b)=\int_a^b f(x)\,dx .
$$

In entrambi i casi la *funzione di ripartizione* $F(x)=P(X\le x)$ raccoglie
l'informazione cumulata fino a $x$.

`````

La stessa curva si può disegnare in due modi, e conviene saperli riconoscere
tutti e due, perché nel resto del libro compaiono entrambi.

```{figure} ../figures/variabili-aleatorie-momenti-percentili.svg
:name: fig-densita-percentili
:alt: "Due grafici affiancati sulla stessa distribuzione asimmetrica. A sinistra la densità di probabilità, intitolata PDF, con media e mediana segnate da due linee tratteggiate e la coda oltre il novantacinquesimo percentile riempita di colore. A destra la funzione di ripartizione, intitolata CDF, che sale da zero a uno, e su cui lo stesso percentile si legge entrando dall'altezza 0,95 e scendendo sull'asse orizzontale."
:width: 100%

La stessa variabile, due modi di guardarla. A sinistra la **densità** appena
descritta, che disegna quali valori escono spesso e quali di rado: lì la
probabilità è l'*area* sotto la curva, e infatti la zona colorata è il cinque
per cento dei casi che cadono oltre il valore segnato. A destra una seconda
curva, che per ogni valore risponde alla domanda «quanta parte dei casi sta
sotto questo numero?» e perciò sale da $0$ a $1$: lì la stessa probabilità è
l'*altezza* della curva, cioè un punto da leggere invece di un'area da
calcolare. Il nome tecnico della seconda è **funzione di ripartizione**. (Le
due sigle in cima ai grafici sono le abbreviazioni inglesi con cui si trovano
ovunque: *PDF* per la densità, *CDF* per la ripartizione.)
```

La corrispondenza fra i due grafici di {numref}`fig-densita-percentili` non è
un vezzo: risponde alla domanda che si fa più spesso su una misura, cioè
«sotto quale valore cade il novantacinque per cento dei casi?». Quel valore si
chiama **novantacinquesimo percentile**, e i percentili si leggono sul grafico
di destra perché lì basta partire dall'altezza $0{,}95$ e scendere a leggere il
numero corrispondente. Sul grafico di sinistra la stessa domanda vorrebbe che
si calcolasse un'area, che è un conto e non una lettura. È l'abitudine dei
tecnici che sorvegliano un servizio online: invece di dire «in media il sito
risponde in mezzo secondo» dicono «il novantacinquesimo percentile del tempo
di risposta è due secondi», che è un modo di parlare non della giornata
tipica, ma di quanto vanno male le giornate storte.

## Il centro e la larghezza

Il quadro completo di come si comporta una grandezza casuale (quali valori
escono, e con che frequenza ciascuno) si chiama la sua **distribuzione**: è la
parola che d'ora in poi useremo per la curva o per l'elenco di probabilità
appena visti, e ritorna in ogni pagina del libro. Quasi sempre non serve tutto
il quadro: bastano due numeri, dove sta il centro della distribuzione e quanto
è dispersa attorno a esso.

```{figure} ../figures/media-mediana-moda-varianza-numpy.svg
:name: fig-centro-larghezza
:alt: "Una distribuzione asimmetrica, con la coda allungata a destra, annotata con tre indicatori di centro che cadono in punti diversi: la moda sul picco, la mediana poco più a destra e la media ancora più a destra, tirata dalla coda. Sotto, un segmento marca la deviazione standard attorno alla media."
:width: 92%

Tre centri per la stessa distribuzione. La **moda** sta sul picco (il valore
più frequente), la **mediana** taglia i casi a metà, la **media** è quella
scolastica. Su una curva simmetrica coinciderebbero; qui no, perché la curva
ha una **coda**, cioè si allunga da un lato con valori rari ma molto grandi, e
la media è quella che la coda sposta di più: pochi stipendi altissimi alzano
lo stipendio medio senza spostare quello di mezzo.
```

L'ordine in cui i tre indicatori compaiono in {numref}`fig-centro-larghezza`
non è casuale ed è un buon promemoria pratico: quando media e mediana si
allontanano molto, la distribuzione ha una coda, e riassumerla con la sola
media descrive un valore tipico che quasi nessuno osserva.

`````{tab} Elementare

Il **valore atteso** è la media dei valori possibili, ciascuno pesato dalla sua
probabilità: è il risultato medio che ci aspettiamo "a lungo andare". Si scrive
$\mathbb{E}[X]$, e si legge «valore atteso di X» (la E doppia sta per
*expected*, atteso, e $X$ è il nome della grandezza che stiamo guardando). Per
un dado onesto vale $\tfrac{1+2+3+4+5+6}{6}=3{,}5$: un numero che non uscirà
mai in un singolo lancio, ma attorno al quale si assesta la media di tanti
lanci.

La **varianza**, che si scrive $\mathrm{Var}(X)$ e si legge «varianza di X»,
misura invece quanto tipicamente ci si allontana da quel centro: piccola se i
valori sono raccolti, grande se sono sparpagliati. Il conto è meno spaventoso
del nome, e sul dado si fa a mano in un minuto. Si guarda di quanto ciascuna
faccia dista dal centro $3{,}5$ (sono $-2{,}5$, $-1{,}5$, $-0{,}5$, $0{,}5$,
$1{,}5$, $2{,}5$), si eleva ogni scarto al quadrato (per togliere di mezzo i
segni: $6{,}25$, $2{,}25$, $0{,}25$, $0{,}25$, $2{,}25$, $6{,}25$) e si fa la
media: $17{,}5$ diviso $6$ fa circa $2{,}92$. Quella è la varianza.

Il quadrato però ha gonfiato tutto, e per tornare a parlare in punti di dado si
fa la radice: $\sqrt{2{,}92} \approx 1{,}7$. Questa è la **deviazione
standard**, ed è la forma leggibile delle due: dice che rispetto al centro
$3{,}5$ i risultati si sparpagliano tipicamente di un punto e mezzo o due, che
guardando le facce di un dado è proprio quello che ci si aspetta.

`````

`````{tab} Superiore

Per una variabile discreta, con $\mu=\mathbb{E}[X]$:

$$
\mathbb{E}[X]=\sum_x x\,p(x), \qquad
\mathrm{Var}(X)=\mathbb{E}\big[(X-\mu)^2\big]=\mathbb{E}[X^2]-\mu^2 ;
$$

per una continua le somme diventano integrali. La **deviazione standard**
$\sigma=\sqrt{\mathrm{Var}(X)}$ riporta la dispersione nelle stesse unità di $X$.
Il valore atteso è lineare, $\mathbb{E}[aX+b]=a\,\mathbb{E}[X]+b$, proprietà
che useremo di continuo. Per il dado:
$\mathbb{E}[X^2]=\tfrac{91}{6}\approx 15{,}17$, quindi
$\mathrm{Var}(X)=15{,}17-3{,}5^2\approx 2{,}92$ e $\sigma\approx 1{,}71$.

**Una distinzione che il resto del libro dà per fatta.** $\mathrm{Var}(X)$ è
una proprietà della *distribuzione*, e nei conti di sopra la si calcola perché
la distribuzione è nota. Sui dati veri non lo è: la si **stima** da un
campione $x_1,\dots,x_n$, ed è un'altra cosa, che si scrive $s^2$ e non
$\mathrm{Var}(X)$. La media degli scarti quadratici
$\frac{1}{n}\sum_i (x_i-\bar x)^2$ **sottostima** sistematicamente $\sigma^2$,
perché $\bar x$ è stata calcolata sugli stessi dati e si adagia su di essi;
dividere per $n-1$ (**correzione di Bessel**) rende lo stimatore non distorto:

$$
s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2 .
$$

Il divisore $n$ è quello che esce dalla massima verosimiglianza di
§«Dalla probabilità all'apprendimento», ed è esattamente il motivo per cui
quello stimatore è distorto. Le librerie scelgono default diversi, e conviene
saperlo prima di confrontare due numeri: `np.var` e `StandardScaler` di
scikit-learn dividono per $n$ (`ddof=0`), `torch.var` divide per $n-1$
(`correction=1`). Su otto osservazioni la differenza è del $14\%$; su
diecimila è invisibile.

`````

## Due distribuzioni ovunque

Di distribuzioni ne esistono molte, ma due tornano di continuo. La **Bernoulli**
descrive ogni singola prova "sì/no"; la **normale** (o gaussiana) descrive tutto
ciò che si accumula attorno a un valore medio.

```{figure} ../figures/curva-normale.svg
:name: fig-curva-normale
:alt: Curva a campana della distribuzione normale, con la media al centro e le bande a una e due deviazioni standard evidenziate.
:width: 85%

La distribuzione normale, che si abbrevia $\mathcal{N}(\mu,\sigma^2)$: le due
lettere greche sono il centro della campana ($\mu$, «mu») e il suo scarto
tipico ($\sigma$, «sigma»), cioè di quanto in genere i valori si allontanano
dal centro. Nella sigla il secondo numero è scritto al quadrato perché per
tradizione si elenca la varianza e non lo scarto; la larghezza che si vede nel
disegno resta $\sigma$. Circa il $68\%$ della probabilità cade entro
$\mu\pm\sigma$ e circa il $95\%$ entro $\mu\pm 2\sigma$.
```

`````{tab} Elementare

Una Bernoulli è una moneta, magari truccata: esce $1$ (successo) con probabilità
$p$ e $0$ con probabilità $1-p$. Serve a modellare qualunque esito binario:
click/non click, spam/non spam.

La normale, che si chiama anche **gaussiana** dal nome del matematico Carl
Friedrich Gauss (i due nomi indicano la stessa identica cosa e nel libro si
alternano), è la celebre **curva a campana** ({numref}`fig-curva-normale`):
simmetrica, con la maggior parte dei valori stretti attorno alla media $\mu$ e
code che si assottigliano ai lati. Vale una regola pratica utilissima, la
"68–95": circa il $68\%$ dei casi cade entro una deviazione standard dalla media
($\mu\pm\sigma$), circa il $95\%$ entro due ($\mu\pm 2\sigma$). Altezze, errori di
misura, rumore: in natura la campana è dappertutto.

**Perché dappertutto?** C'è un risultato che lo spiega, e ha un nome
importante: il **teorema del limite centrale**. Dice una cosa sorprendente:
ogni volta che una grandezza è la *somma* di tanti piccoli contributi casuali
e indipendenti fra loro, il risultato assomiglia a una campana, e questo
succede **qualunque sia la forma dei singoli contributi**. L'altezza di una
persona è la somma di centinaia di piccoli effetti (geni, alimentazione,
salute da bambino): campana. L'errore di uno strumento è la somma di tante
imprecisioni minuscole: campana. Il totale di tre dadi è la somma di tre
numeri che, presi da soli, non hanno niente di campanulare (in un dado tutte
le facce sono ugualmente probabili, il disegno sarebbe piatto): eppure il
totale, sì.

Attenzione a cosa il teorema promette e cosa no. Promette che, sommando
abbastanza pezzi, la campana arriva. Non promette *quanto in fretta*: con i
dadi bastano tre addendi, ma con ingredienti molto storti (per esempio un
evento che capita una volta su cento) nemmeno trenta bastano, e la campana non
si vede ancora.

`````

`````{tab} Superiore

La Bernoulli$(p)$ ha $P(X{=}1)=p$, valore atteso $\mathbb{E}[X]=p$ e varianza
$\mathrm{Var}(X)=p(1-p)$. La normale $\mathcal{N}(\mu,\sigma^2)$ ha densità

$$
f(x)=\frac{1}{\sigma\sqrt{2\pi}}\;
\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right),
$$

dove $\mu$ è la media (il centro della campana) e $\sigma$ la deviazione
standard (la sua larghezza). Perché è così onnipresente? Per il **teorema del
limite centrale** (de Moivre, Laplace, poi Lyapunov): se $X_1,\dots,X_n$ sono
i.i.d. con media $\mu$ e varianza $\sigma^2$ finita e non nulla, posto
$S_n = X_1 + \dots + X_n$, la somma standardizzata
$(S_n - n\mu)/(\sigma\sqrt{n})$ converge in distribuzione a
$\mathcal{N}(0,1)$, *quale che sia* la distribuzione di partenza; in pratica,
per $n$ grande, $S_n$ si approssima bene con $\mathcal{N}(n\mu,\,n\sigma^2)$.
(Per variabili indipendenti ma non identicamente distribuite servono
condizioni in più, come quella di Lindeberg o di Lyapunov.) È il motivo per
cui gli errori di misura si modellano gaussiani, e per cui in una rete neurale
larga le pre-attivazioni di uno strato, che sono somme di molti contributi,
tendono a essere gaussiane qualunque sia la distribuzione dei pesi. (Non è
invece il motivo per cui si inizializzano i pesi in un modo o nell'altro: gli
schemi standard derivano la **varianza** dell'inizializzazione, per conservare
la scala delle attivazioni fra uno strato e l'altro, e non la famiglia, tanto
che il default di PyTorch campiona da un'uniforme e non da una normale.)

Il teorema dice **che** si converge, non **quanto in fretta**, e la seconda
domanda ha una risposta separata. Una garanzia grossolana la dà la
disuguaglianza di Berry–Esseen,
$\sup_z |F_n(z)-\Phi(z)| \le C\,\rho_3 / (\sigma^3\sqrt{n})$ con $\rho_3 =
\mathbb{E}|X-\mu|^3$: la distanza cala come $1/\sqrt{n}$, e davanti c'è un
momento terzo **assoluto**, che non sa distinguere una coda lunga da un lato
sola da due code lunghe simmetriche. A separarle è lo sviluppo di Edgeworth, il
cui primo termine correttivo è proporzionale all'**asimmetria** e si annulla
quando la distribuzione di partenza è simmetrica: è quella, e non la
"stranezza" percepita, a governare la velocità. Misurato con la statistica di
Kolmogorov–Smirnov sulla somma standardizzata: partendo da un dado uniforme
(asimmetria nulla) la distanza dalla normale è già $0{,}069$ con $n=3$;
partendo da una Bernoulli$(0{,}01)$ resta $0{,}45$ a $n=30$, e da una
lognormale$(0;\,2)$ resta $0{,}27$. Il caso «piatto» è il più gentile di tutti,
non il più ostile.

`````

Vale la pena vedere il teorema del limite centrale **succedere**, perché
detto a parole sembra una promessa e guardato sembra un trucco.

```{figure} ../figures/limite-centrale.gif
:name: fig-limite-centrale
:alt: "Animazione: a sinistra le sei facce di un dado, tutte della stessa altezza; a destra un istogramma delle somme di tre dadi che si riempie lotto dopo lotto e assume la forma di una campana, su cui compare la curva normale prevista dalla teoria."
:width: 90%

Un dado è **piatto**: nessuna faccia è più probabile di un'altra. Eppure la
somma di tre dadi, ripetuta seicento volte, si dispone da sé lungo la campana.
La curva sovrapposta non è stata disegnata sopra le barre a cose fatte: è la
campana che il teorema prevede prima ancora di tirare i dadi, quella centrata
sulla somma media dei tre e larga di conseguenza, e i dadi le danno ragione.
```

Due cose valgono più della formula, nella {numref}`fig-limite-centrale`. La
prima è che la distribuzione di partenza non ha **niente** della campana:
nessuna faccia è privilegiata, il disegno di un dado singolo sarebbe una
fila di barre tutte uguali, e la campana arriva lo stesso. È questo il senso
di «qualunque sia la forma dei singoli contributi».

La seconda va detta con più cautela di quanto si faccia di solito. Qui bastano
tre addendi, ma non è un merito del teorema: è un merito del dado, che è
**simmetrico**, cioè non ha una coda più lunga dell'altra. È proprio la
simmetria a far arrivare in fretta la campana, ed è per questo che il
caso più «piatto» è anche il più facile. Con una distribuzione di partenza
molto sbilanciata (un evento che capita una volta su cento, un valore che ogni
tanto è enorme) la campana non si vede nemmeno dopo trenta addendi. Il teorema
promette l'arrivo, non il tempo di percorrenza.

Lo stesso fenomeno si vede con le monete, ed è il caso che ricorre più spesso
nel libro: contare i successi in una serie di prove «sì/no» indipendenti.

```{figure} ../figures/gaussiana-ovunque-distribuzioni-ai.svg
:name: fig-conteggio-successi
:alt: "Istogramma del numero di teste su venti lanci di una moneta equa: una serie di barre discrete, più alte al centro attorno a dieci e via via più basse verso gli estremi. Sovrapposta alle barre, la curva normale con la stessa media e la stessa varianza le segue quasi esattamente."
:width: 90%

Barre discrete, curva continua. Il conteggio dei successi può valere solo
numeri interi, eppure la campana continua lo approssima già bene con venti
prove.
```

L'accordo mostrato in {numref}`fig-conteggio-successi` è ciò che autorizza
un'abitudine diffusa. Per sapere quanto vale un modello lo si prova su un
gruppo di esempi tenuti da parte, mai visti durante l'addestramento (il **test
set**), e si conta la percentuale di risposte esatte: quella percentuale si
chiama **accuratezza**. Ma un conteggio di risposte esatte su prove
indipendenti è la stessa cosa del conteggio di teste appena disegnato, quindi
lo si può trattare come una campana. È l'approssimazione su cui poggiano gli
intervalli di confidenza che vedremo fra poco.

## Aggiornare le credenze: il teorema di Bayes

Fin qui abbiamo calcolato probabilità "in avanti". Ma spesso il problema è
rovesciato: osserviamo un effetto e vogliamo risalire alla causa. È il regno di
**Thomas Bayes**, il cui saggio fu pubblicato postumo nel 1763.

`````{tab} Elementare

Immagina un test per una malattia rara, che colpisce $1$ persona su $100$. Il
test è buono: individua il $99\%$ dei malati e sbaglia solo nel $5\%$ dei sani. Ti
arriva un risultato **positivo**: quanto devi preoccuparti? L'istinto dice
"molto", ma i conti dicono altro. Su $10\,000$ persone, $100$ sono malate (e $99$
risultano positive); ma dei $9\,900$ sani ben $495$ risultano positivi *per
errore*. I positivi totali sono $99+495=594$, e solo $99$ sono davvero malati:
circa il **17%**. Con una malattia rara, la maggior parte dei positivi sono falsi
allarmi.

`````

`````{tab} Superiore

Il teorema lega la probabilità condizionata nei due versi:

$$
P(A\mid B)=\frac{P(B\mid A)\,P(A)}{P(B)} .
$$

Con $A=\text{"malato"}$ e $B=\text{"positivo"}$, prevalenza $P(A)=0{,}01$,
sensibilità $P(B\mid A)=0{,}99$ e tasso di falsi positivi $P(B\mid A^c)=0{,}05$,
il denominatore è $P(B)=0{,}99\cdot 0{,}01+0{,}05\cdot 0{,}99=0{,}0594$, da cui

$$
P(A\mid B)=\frac{0{,}99\cdot 0{,}01}{0{,}0594}\approx 0{,}167 .
$$

La prevalenza $P(A)$ (il *prior*) domina il risultato: nessun classificatore
va giudicato dalla sola accuratezza quando le classi sono fortemente
sbilanciate.

`````

Che a comandare sia quanto la malattia è diffusa, e non quanto il test è
buono, si vede meglio spostando proprio quel dato. Teniamo lo stesso test
(individua il $99\%$ dei malati e sbaglia sul $5\%$ dei sani) e rendiamo la
malattia dieci volte più rara, una persona su mille invece di una su cento.

```{figure} ../figures/teorema-bayes-test-medici.svg
:name: fig-bayes-test
:alt: "Diagramma ad albero su una popolazione di 10.000 persone sottoposte a screening, con prevalenza dello 0,1%: si dividono in 10 malate e 9.990 sane. Delle 10 malate, il 99% risulta positivo, circa 10 positivi veri e nessun falso negativo. Delle 9.990 sane, il 5% risulta positivo per errore, circa 500 falsi positivi, e le restanti 9.490 negative. In basso il conto: su circa 510 positivi totali, i malati veri sono 10, cioè circa il 2%."
:width: 96%

Lo stesso test su una malattia dieci volte più rara. I falsi positivi non
cambiano di molto (vengono dai sani, che sono tanti); i positivi veri sì, e la
probabilità di essere davvero malati crolla dal $17\%$ al $2\%$. Nel disegno
compaiono due termini del mestiere: **prevalenza** è quanto la malattia è
diffusa nella popolazione (qui una persona su mille, cioè lo $0{,}1\%$), e
**falso negativo** è il caso opposto del falso allarme, cioè un malato a cui il
test dice che sta bene.
```

Il test non è peggiorato di una virgola fra {numref}`fig-bayes-test` e
l'esempio di prima: sono cambiati i malati. È la ragione per cui la stessa
identica prova diagnostica va interpretata diversamente in uno screening di
massa e in un reparto, dove chi arriva è già stato selezionato dai sintomi.

## La legge dei grandi numeri: perché servono tanti dati

Un casinò perde in continuazione. A ogni tavolo di roulette qualcuno vince, e
il banco paga. Eppure nessun casinò è mai fallito per la roulette: su una
singola puntata il margine è minuscolo e il caso domina, su milioni di puntate
il caso si spegne e resta solo il margine.

```{figure} ../figures/grandi-numeri-tanti-dati.svg
:name: fig-legge-grandi-numeri
:alt: "Grafico della media osservata al crescere del numero di prove: nelle prime decine oscilla ampiamente sopra e sotto il valore vero, poi le oscillazioni si restringono progressivamente e la curva si assesta sulla linea orizzontale del valore atteso, senza però toccarla esattamente. Due linee punteggiate a imbuto racchiudono le oscillazioni e portano l'etichetta «errore tipico circa uno diviso radice di n»."
:width: 92%

Le oscillazioni non spariscono: si restringono. È una differenza che conta,
perché nessun numero di prove rende la media *uguale* al valore vero.
```

La forma a imbuto di {numref}`fig-legge-grandi-numeri` è la stessa che governa
quanti dati servono per addestrare o per valutare un modello, e l'imbuto si
stringe più lentamente di quanto verrebbe da sperare: non in proporzione al
numero di prove, ma alla sua **radice quadrata**. Il conto è quello: con cento
prove l'incertezza vale un certo tanto, e per dimezzarla non basta arrivare a
duecento, bisogna arrivare a quattrocento, perché quello che deve raddoppiare è
la radice, e la radice di quattrocento è il doppio della radice di cento
($20$ contro $10$). Quattro volte i dati per metà dell'errore: è una tassa che
si paga in tutto il libro.

`````{tab} Elementare

Se ripeti tante volte lo stesso esperimento casuale, la media delle
osservazioni si avvicina sempre di più al valore atteso. Su $10$ lanci di una
moneta equa, $7$ teste non stupiscono nessuno. Su $10\,000$ lanci, il $70\%$
di teste è talmente improbabile che concluderesti (ragionevolmente) che la
moneta è truccata.

Attenzione a un equivoco diffuso: **il caso non si corregge, si diluisce.**
Dopo dieci teste di fila la moneta non "deve" croce, il lancio successivo resta
50 e 50. Quello che succede è che le dieci teste iniziali pesano sempre meno
man mano che i lanci si accumulano, finché diventano irrilevanti nella media.

Due condizioni non sono pignoleria, e conviene tenerle distinte perché si
rompono in modi diversi. La prima è che le osservazioni siano
**indipendenti**, cioè che nessuna influenzi le altre: se le prime recensioni
di un ristorante sono entusiaste, chi scrive dopo le ha lette e si adegua, e
mille voti così non valgono mille pareri raccolti separatamente. La seconda è
che vengano tutte dalla **stessa distribuzione**, cioè che si stia misurando
sempre la stessa cosa: mille recensioni scritte dagli amici del ristoratore
misurano l'amicizia, non la cucina, e sommarle a quelle dei clienti fa una
media di due cose diverse. Se salta l'una o l'altra, la garanzia non vale più,
ed è la ragione per cui una raccolta di dati fatta male non migliora
aggiungendone altra raccolta allo stesso modo.

`````

`````{tab} Superiore

Siano $X_1,\dots,X_n$ variabili i.i.d. con media $\mu=\mathbb{E}[X]$ finita. La media
campionaria $\bar{X}_n=\frac{1}{n}\sum_i X_i$ converge a $\mu$: in probabilità
(legge **debole**) e quasi certamente (legge **forte**).

La velocità è il punto che interessa il machine learning. Se
$\mathrm{Var}(X)=\sigma^2$, allora

$$
\mathrm{Var}(\bar{X}_n)=\frac{\sigma^2}{n},
\qquad
\text{deviazione standard} = \frac{\sigma}{\sqrt{n}} .
$$

L'errore cala come $1/\sqrt{n}$, non come $1/n$: **per dimezzare l'incertezza
servono quattro volte i dati**. È la ragione strutturale per cui i guadagni di
prestazione diventano sempre più costosi, e per cui raddoppiare un dataset
raramente raddoppia la qualità.

L'ipotesi i.i.d. è quella che si rompe più spesso nella pratica: dati correlati
nel tempo, esempi duplicati, campioni raccolti da una sola fonte. Quando salta,
$n$ conta molto meno di quanto dica il conteggio delle righe.

`````

## Quanto fidarsi di una stima: gli intervalli di confidenza

Un modello nuovo raggiunge l'$87{,}2\%$ di accuratezza sul test set, il vecchio
si fermava all'$86{,}8\%$. Il team festeggia. Ma se il test set ha $500$
esempi, quattro decimi di punto sono **due risposte esatte in più**. Due.

```{figure} ../figures/intervalli-di-confidenza.svg
:name: fig-intervalli-confidenza
:alt: "Tre modelli su un asse dell'accuratezza da 80 a 92 per cento, ciascuno con la sua stima puntuale e una barra d'errore. Il modello A sta all'87,2% e il modello B all'86,8%: le loro barre si sovrappongono quasi per intero, e una nota dice che con intervalli sovrapposti non c'è un vincitore. Il modello C sta all'81,0%, con la barra molto più in basso, che con le altre due non si sovrappone."
:width: 90%

Le stesse tre misure, con l'incertezza disegnata. A e B non sono
distinguibili: la differenza fra loro è più piccola di quanto la misura possa
risolvere. C invece è davvero peggiore.
```

{numref}`fig-intervalli-confidenza` mostra cosa si compra con una barra
d'errore: la capacità di dire «non lo so». Senza, una classifica si può
sempre stilare, e sembrerà informativa anche quando ordina rumore; con la
barra, alcune coppie restano semplicemente a pari merito, che è la risposta
corretta.

`````{tab} Elementare

L'accuratezza sul test set non è *la* prestazione del modello: è una **stima**
della prestazione vera, calcolata su un **campione**, cioè su un gruppo
limitato di casi pescati dal mucchio di tutti quelli possibili. Il numero che
interessa davvero (quanto il modello risponderebbe bene su tutti i casi
possibili) non lo conosceremo mai.

L'analogia giusta è il sondaggio elettorale. Nessun istituto serio titola "il
candidato A è al $47{,}2\%$" senza il margine d'errore: con mille intervistati
quel numero significa "da qualche parte fra il $44\%$ e il $50\%$". Valutare un
modello su $500$ esempi è come sondare $500$ elettori.

Con $500$ esempi e un'accuratezza dell'$87\%$, il margine al $95\%$ è di circa
**$\pm 3$ punti**: la stima onesta è "fra l'$84\%$ e il $90\%$". Il modello
vecchio sta fra l'$83{,}8\%$ e l'$89{,}8\%$. I due intervalli si sovrappongono
quasi per intero: quel miglioramento non è distinguibile dal **rumore**, che è
il nome che si dà alla parte di un risultato dovuta al caso e non al merito
(niente a che vedere con i suoni: è un'immagine presa dalle
telecomunicazioni).

Quel numero non piove dal cielo, e una regola pratica basta a rifarlo a mente,
in due mosse. La prima: quando il modello è sul $50\%$, il margine in punti
percentuali è circa $100/\sqrt{n}$, e con $n=500$ fa circa $4{,}5$ punti
($\sqrt{500}$ è poco più di $22$). La seconda: un modello più bravo del $50\%$
ha anche meno modo di variare (chi sbaglia raramente non può sbagliare in tanti
modi diversi), quindi quel margine va ridotto, e di quanto lo dice un fattore
che dipende solo dall'accuratezza. Vale $1$ al $50\%$, cioè non riduce niente;
vale circa $0{,}8$ all'$80\%$, circa due terzi all'$87\%$, circa la metà al
$93\%$. Ecco i tre punti: $4{,}5$ moltiplicato per due terzi fa $3$.

Anche il «$95\%$» ha un significato preciso:
non è la fiducia che riponiamo nel singolo numero, è la percentuale di volte
in cui una procedura del genere, ripetuta su tanti campioni diversi, contiene
davvero il valore giusto. Una volta su venti sbaglia, e questo è messo in
conto.

Il margine si stringe solo aumentando gli esempi, e lentamente, perché a
comandare è la radice quadrata: servono $5\,000$ esempi per scendere a circa
$\pm 1$ punto, $50\,000$ per arrivare a $\pm 0{,}3$.

`````

`````{tab} Superiore

Per una proporzione $\hat{p}$ stimata su $n$ prove indipendenti, l'errore
standard è $\sqrt{\hat{p}(1-\hat{p})/n}$ e l'intervallo di Wald al livello
$95\%$ è

$$
\hat{p} \pm 1{,}96\,\sqrt{\frac{\hat{p}(1-\hat{p})}{n}} .
$$

Con $\hat{p}=0{,}872$ e $n=500$: errore standard $\approx 1{,}49$ punti,
margine $\approx 2{,}93$ punti, intervallo $[84{,}3\%,\ 90{,}1\%]$.

**Cosa significa davvero "95%".** Non che il valore vero abbia il $95\%$ di
probabilità di stare in *questo* intervallo: il valore vero è un numero fisso,
o ci sta o non ci sta. Significa che la *procedura*, ripetuta su molti campioni,
produce intervalli che contengono il valore vero nel $95\%$ dei casi. È una
proprietà del metodo, non di questo singolo risultato.

Due avvertenze pratiche. L'intervallo di Wald è inaffidabile con $n$ piccolo o
$\hat{p}$ vicino a $0$ o $1$: lì si usano Wilson o Clopper–Pearson. E quando si
confrontano **due** modelli sullo stesso test set gli errori sono appaiati: il
test corretto è quello di McNemar sui disaccordi, non il confronto fra due
intervalli, che è conservativo e può nascondere differenze reali.

`````

## Correlazione non è causalità

Nei mesi estivi aumentano le vendite di gelato. Negli stessi mesi aumentano gli
annegamenti. Le due curve salgono e scendono insieme con regolarità quasi
imbarazzante, e un modello addestrato su questi dati imparerebbe la relazione
senza esitare. Nessuno però propone di vietare il gelato: dietro entrambe le
curve c'è una terza cosa, il caldo.

```{figure} ../figures/correlazione-non-causalita.svg
:name: fig-confonditore
:alt: "Grafo causale con tre nodi. Dal caldo estivo, indicato come confonditore Z, partono due frecce di causa: una verso i gelati venduti (X) e una verso gli annegamenti (Y). Fra X e Y non c'è alcuna freccia, ma una linea tratteggiata etichettata correlazione spuria. In basso la chiave di lettura: X e Y si muovono insieme solo perché Z muove entrambi."
:width: 82%

Il **confondente**, disegnato (nel disegno è etichettato «confonditore Z»: i
due nomi indicano la stessa cosa, e nel libro si usa il primo). Fra gelati e
annegamenti non passa nessuna freccia: il legame che si misura nei dati è
tutto riflesso di quello che ciascuno dei due ha con il caldo.
```

Vale la pena notare cosa *non* c'è in {numref}`fig-confonditore`: la freccia
fra $X$ e $Y$, cioè fra i gelati e gli annegamenti (nel disegno le due lettere
stanno per le due grandezze che si misurano, e $Z$ per la causa comune). I
dati da soli non la disegnano né la cancellano, perché
correlazione e causalità lasciano sui numeri la stessa traccia. A distinguerle
serve qualcosa che nei dati non c'è: un intervento (cambiare $X$ e guardare
$Y$) oppure una conoscenza del dominio che dica quale freccia è plausibile.

`````{tab} Elementare

La correlazione misura il **co-movimento**: quanto due grandezze tendono a
salire e scendere insieme. Ed è simmetrica e cieca.

*Simmetrica*: la correlazione fra gelati e annegamenti è identica a quella fra
annegamenti e gelati (il numero non contiene alcuna informazione su chi
influenzi chi).

*Cieca*: non distingue fra una relazione diretta, una mediata da altro e una
pura coincidenza. È come notare che due colleghi arrivano sempre insieme in
ufficio: potrebbero viaggiare insieme, o semplicemente prendere lo stesso treno
perché abitano nello stesso quartiere.

Il caldo dell'esempio si chiama **confondente**: una causa comune che spiega
entrambi gli effetti. Il guaio è che noi il caldo lo sospettiamo per buon
senso; un modello no. Impara la scorciatoia che funziona sui dati che ha visto
e la usa finché il mondo non cambia: poi sbaglia, e sbaglia in modo
inspiegabile.

`````

`````{tab} Superiore

Il coefficiente di **Pearson** fra $X$ e $Y$ è

$$
\rho_{XY} = \frac{\mathrm{Cov}(X,Y)}{\sigma_X\,\sigma_Y} \in [-1,1],
$$

e misura la sola dipendenza **lineare**: $\rho=0$ non implica indipendenza;
se $X$ è distribuita in modo simmetrico attorno allo zero, $Y=X^2$ ha
correlazione nulla con $X$ e dipendenza perfetta. (La simmetria è essenziale:
per $X$ uniforme su $[0,1]$ la stessa parabola dà una correlazione vicina a
$0{,}97$.)

Le strutture da tenere distinte:

- **confondente**: $Z \to X$ e $Z \to Y$ producono correlazione fra $X$ e $Y$
  senza alcun legame causale diretto (il caldo);
- **catena**: $X \to Z \to Y$, causalità reale ma mediata;
- **collider**: $X \to Z \leftarrow Y$, dove condizionare su $Z$ *crea*
  correlazione fra $X$ e $Y$ che non esisteva. È il meccanismo dietro molti
  paradossi di selezione del campione.

Dai soli dati osservativi le tre sono indistinguibili: servono un intervento
(esperimento randomizzato) o assunzioni causali esplicite. Nel machine
learning la conseguenza ha un nome, *shortcut learning*: il modello aggancia
la correlazione più comoda del dataset (lo sfondo invece dell'animale, il
marcatore dell'ospedale invece della patologia) e crolla appena la
distribuzione cambia. La correlazione basta per **predire** dentro la stessa
distribuzione; non basta per **decidere** un intervento.

`````

### Tre gradini: vedere, fare, immaginare

Le tre strutture appena elencate si sistemano dentro una cornice più larga, che
vale la pena avere in testa perché rimette in fila cose che questo libro
incontra in capitoli lontanissimi fra loro. La propone Judea Pearl, e la chiama
**scala della causalità** {cite}`pearl2018book`.

`````{tab} Elementare

I gradini sono tre, e ognuno risponde a un tipo di domanda che il gradino
sotto non sa nemmeno formulare.

**Vedere.** «Fra chi compra il gelato, quanti annegano?» È una domanda che si
risolve guardando i dati e contando. Tutto ciò che si fa con una tabella, un
grafico o un modello che prevede sta qui: si osserva quello che è successo e si
cercano le regolarità.

**Fare.** «Se vietassi il gelato, gli annegamenti calerebbero?» Non è la stessa
domanda, e non si risponde con gli stessi dati. Osservare chi compra il gelato
non è come *far comprare* il gelato a qualcuno scelto a caso: nel secondo caso
si è tagliata la freccia che va dal caldo alla decisione d'acquisto. È il
motivo per cui esistono gli esperimenti randomizzati.

**Immaginare.** «Quel bagnante che è annegato, si sarebbe salvato se non fosse
uscito quel giorno?» Riguarda un caso singolo e un mondo che non è accaduto.
Non basta nemmeno l'esperimento, perché l'esperimento dice cosa succede in
media, non cosa sarebbe successo a *lui*.

I modelli addestrati sui dati stanno quasi tutti sul primo gradino, e ci stanno
benissimo. Pearl lo dice con un'immagine che vale la pena riportare: «la
civetta è un buon cacciatore senza capire perché il topo vada da A a B». La
civetta ha visto migliaia di topi e sa dove sarà questo fra un secondo, il che
le basta per prenderlo; delle ragioni per cui il topo si sposta non sa niente,
e non le servono. Predire, insomma, non richiede capire. Il punto è sapere
quale domanda si sta facendo, perché salire un gradino richiede sempre qualcosa
che nei dati non c'è.

`````

`````{tab} Superiore

I tre livelli si distinguono formalmente dall'oggetto probabilistico che
sanno esprimere.

1. **Associazione**: $P(y \mid x)$, cioè condizionare. È tutto ciò che si
   ottiene osservando, e include correlazione, regressione e ogni modello
   predittivo di questo libro.
2. **Intervento**: $P(y \mid do(x))$, dove l'operatore $do$ denota
   l'imposizione di $X = x$ dall'esterno, che nel grafo causale corrisponde a
   **recidere tutti gli archi entranti** in $X$. In generale
   $P(y \mid do(x)) \neq P(y \mid x)$, e la differenza è esattamente il
   contributo dei confondenti.
3. **Controfattuale**: $P(y_x \mid x', y')$, la probabilità che $Y$ sarebbe
   stato $y$ sotto $X = x$, **dato che** in realtà si è osservato $x'$ e $y'$.
   Richiede un modello strutturale completo, non solo il grafo.

Il risultato che rende la scala operativa e non solo tassonomica è che
condizioni grafiche esplicite (il *criterio di backdoor*, il *do-calculus*)
dicono **quando** una quantità di livello 2 è calcolabile a partire da soli
dati osservativi di livello 1, e con quale aggiustamento. Non sempre lo è: se i
confondenti rilevanti non sono osservati, nessuna quantità di dati basta, ed è
una impossibilità di principio, non un limite di campione.

La conseguenza per chi costruisce sistemi è che il gradino di una domanda
determina che dati servono. Chiedere «quali clienti abbandoneranno» è
livello 1 e un classificatore basta; chiedere «quali clienti abbandoneranno
*se non li chiamiamo*» è livello 2, e un classificatore addestrato su dati in
cui le chiamate erano decise da qualcuno risponde alla domanda sbagliata con
grande sicurezza.

`````

Non è una gerarchia di merito, è una gerarchia di **cosa serve avere** per
rispondere, e il libro la attraversa tutta. Sul primo gradino, quello di chi
guarda i dati e conta, sta la gran parte di quello che leggeremo. Sul secondo,
quello di chi il mondo lo tocca invece di limitarsi a guardarlo, stanno i
**test A/B** (si mostrano due versioni di un prodotto a due gruppi di utenti
scelti a caso e si confrontano i risultati) e l'apprendimento per rinforzo,
dove un programma i dati non li riceve, se li produce agendo. Sul terzo stanno
le domande su ciò che non è successo, tipo «a questo cliente il prestito è
stato negato; glielo avrebbero dato con mille euro di reddito in più?»: si
chiamano **spiegazioni controfattuali**. Ognuno dei tre ha il suo capitolo più
avanti, e non serve andarlo a cercare adesso.

## Dalla probabilità all'apprendimento

Resta un'ultima domanda: cosa c'entra tutto questo con l'*addestrare* un modello?
Il ponte si chiama **massima verosimiglianza** (*maximum likelihood estimation*,
MLE).

```{figure} ../figures/massima-verosimiglianza.svg
:name: fig-verosimiglianza
:alt: "Due curve di verosimiglianza sovrapposte, in funzione della probabilità p di ottenere testa. La prima viene da 7 teste su 10 lanci ed è larga e piatta; la seconda da 70 teste su 100 ed è molto più stretta. Entrambe hanno il massimo nello stesso punto, p uguale a 0,7."
:width: 90%

Stessa proporzione, due quantità di prove. Il punto più alto non si sposta; a
cambiare è quanto la curva sia stretta, cioè quanto ci si può contare.
```

Le due curve di {numref}`fig-verosimiglianza` dicono ciò che la sola stima non
dice. Con dieci lanci molti valori di $p$ spiegano l'osservazione quasi
altrettanto bene; con cento, no. La stima puntuale è identica, ma la seconda è
un'affermazione molto più forte, e a misurare la differenza è la larghezza
della curva.

`````{tab} Elementare

Abbiamo dei dati e vogliamo scegliere i parametri che li rendono *meno
sorprendenti possibile*. Lancio una moneta 10 volte e vedo 7 teste: quale
valore di $p$ (la probabilità che esca testa) spiega meglio quel che ho
osservato?

Il modo di rispondere è provarli tutti e tenere il migliore. Per ogni valore di
$p$ si calcola quanto sarebbe probabile vedere proprio 7 teste su 10 (il conto
esatto qui non lo facciamo: è quello che dà la probabilità di un certo numero
di successi in un certo numero di prove, e in questo capitolo lo prendiamo per
buono). I risultati sono questi. Se la moneta fosse equa, $p=0{,}5$, quella
probabilità è circa il $12\%$: possibile, non entusiasmante. Se fosse $p=0{,}6$
sale al $21\%$, se fosse $p=0{,}7$ arriva al $27\%$, e se fosse $p=0{,}8$
ridiscende al $20\%$. Il massimo cade a $0{,}7$, cioè esattamente sulla
proporzione osservata, ed è questo che si intende quando si dice che $0{,}7$ è
il valore «che rende l'osservazione più probabile».

Quel «quanto è probabile l'osservazione, se il parametro valesse così» ha un
nome, ed è il nome che compare nel titolo di questa sezione, nella figura e
nel riquadro finale: si chiama **verosimiglianza**. La curva della figura è
proprio questa: per ogni valore di $p$ sull'asse orizzontale, quanto sarebbe
verosimile ciò che ho visto.

"Imparare", per un modello, è spesso esattamente questo: girare le manopole
dei parametri finché i dati osservati diventano i più plausibili.

`````

`````{tab} Superiore

Dati esempi indipendenti $x^{(1)},\dots,x^{(m)}$, la **verosimiglianza** dei
parametri $\theta$ è

$$
L(\theta)=\prod_{i=1}^{m} p\big(x^{(i)};\theta\big),
$$

e la stima di massima verosimiglianza è

$$
\hat{\theta}=\arg\max_{\theta}\ \sum_{i=1}^{m}\log p\big(x^{(i)};\theta\big),
$$

Il passaggio dal prodotto alla somma dei logaritmi è lecito perché il
logaritmo è **strettamente crescente**: applicarlo non sposta il punto di
massimo, e i due problemi hanno lo stesso $\arg\max$. Che poi sommare sia
anche numericamente più stabile che moltiplicare mille numeri piccoli è un
secondo vantaggio, non la giustificazione.

(Scriviamo $L$ e non $\mathcal{L}$: la $\mathcal{L}$ calligrafica in questo
libro è la loss, che si minimizza; la verosimiglianza si massimizza, e le due
si incontrano nella log-verosimiglianza negativa,
$\mathcal{L}(\theta)=-\log L(\theta)$ a meno di costanti.) Il punto cruciale:
per un modello di regressione che descrive $y$ dato $x$ come una gaussiana
centrata sulla predizione,
$y \mid x \sim \mathcal{N}(\hat{y},\sigma^2)$
con varianza fissa, massimizzare la log-verosimiglianza equivale a
**minimizzare l'errore quadratico medio**; sotto ipotesi di
Bernoulli/categoriche equivale a minimizzare la **cross-entropy**. Le loss
$\mathcal{L}$ che incontreremo nel resto del libro non sono scelte arbitrarie:
sono verosimiglianze travestite.

`````

## In pratica, con NumPy

Poche righe traducono in codice tutto ciò che abbiamo visto: media e varianza di
una distribuzione discreta, e l'aggiornamento bayesiano del test diagnostico.

```python
import numpy as np

# Valore atteso e varianza di un dado onesto
valori = np.arange(1, 7)
p = np.full(6, 1/6)
mu  = (valori * p).sum()                # 3.5
var = ((valori - mu)**2 * p).sum()      # ~2.9167
print(mu, var, np.sqrt(var))            # 3.5  2.9167  1.7078

# Teorema di Bayes: test per una malattia rara
prevalenza  = 0.01     # P(malato)
sensibilita = 0.99     # P(positivo | malato)
falsi_pos   = 0.05     # P(positivo | sano)

p_pos     = sensibilita * prevalenza + falsi_pos * (1 - prevalenza)
posterior = sensibilita * prevalenza / p_pos
print(posterior)       # ~0.167: solo il 17% dei positivi e' davvero malato
```

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- La probabilità misura l'incertezza; una **variabile aleatoria** le dà un
  numero (le teste su dieci lanci), e per riassumerla bastano quasi sempre due
  cose: il risultato medio che ci si aspetta a lungo andare e quanto
  tipicamente ci si allontana da quel centro.
- La **curva a campana** compare ovunque perché tante piccole cause casuali che
  si sommano la producono da sé, anche partendo da dadi in cui nessuna faccia è
  favorita. Regola pratica: circa il $68\%$ dei casi cade entro uno scarto
  tipico dalla media, circa il $95\%$ entro due.
- Un risultato positivo va sempre letto insieme a quanto la cosa cercata è
  rara: con una malattia che colpisce una persona su cento, anche un test molto
  buono produce più falsi allarmi che malati veri (solo il $17\%$ circa dei
  positivi è davvero malato). È il **teorema di Bayes** al lavoro.
- La media di tante osservazioni si assesta sul valore vero, ma le oscillazioni
  si stringono con calma: per dimezzare l'incertezza non basta il doppio dei
  dati, ne servono quattro volte tanti.
- Un'accuratezza è una **stima**, come un sondaggio elettorale: su $500$ esempi
  vale circa $3$ punti in più o in meno, e differenze più piccole di così sono
  rumore, non progresso.
- Due grandezze che salgono e scendono insieme (i gelati e gli annegamenti)
  bastano a **prevedere** finché il mondo resta com'è, non a **decidere** un
  intervento: dietro le due curve c'è una causa comune, il caldo, e vietare il
  gelato non salverebbe nessuno.
- Imparare, per un modello, è girare le manopole dei parametri finché i dati
  osservati diventano i meno sorprendenti possibile: le funzioni di costo più
  usate sono questa stessa idea, scritta in un altro modo.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- La probabilità misura l'incertezza; una **variabile aleatoria** le dà un
  numero, e $\mathbb{E}[X]$ e $\mathrm{Var}(X)$ ne riassumono centro e
  dispersione. Attenzione a non confondere la varianza *della distribuzione*
  con la sua **stima dal campione**: quest'ultima è non distorta solo
  dividendo per $n-1$ (correzione di **Bessel**), e le librerie scelgono
  default diversi.
- La **normale** compare ovunque per il teorema del limite centrale; la regola
  68–95 lega la deviazione standard $\sigma$ alle probabilità.
- Il **teorema di Bayes** aggiorna le credenze alla luce dei dati: con classi
  rare, occhio ai falsi positivi.
- La **legge dei grandi numeri** garantisce la convergenza della media, ma solo
  come $1/\sqrt{n}$: dimezzare l'incertezza costa quattro volte i dati.
- Un'accuratezza è una **stima**: su $500$ esempi il margine al $95\%$ è di
  circa $\pm 3$ punti, e differenze più piccole sono rumore.
- La **correlazione** basta per predire dentro la stessa distribuzione, non per
  decidere un intervento: confondenti e collider producono correlazioni senza
  causalità.
- La **massima verosimiglianza** è il ponte fra probabilità e apprendimento:
  minimizzare MSE o cross-entropy è massimizzare una verosimiglianza.
```
`````
