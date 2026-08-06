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
sottoinsiemi $A\subseteq\Omega$. Una misura di probabilità soddisfa gli
**assiomi di Kolmogorov** (1933):

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

```{figure} ../figures/variabili-aleatorie-momenti-percentili.svg
:name: fig-densita-percentili
:alt: "Due grafici affiancati sulla stessa distribuzione asimmetrica. A sinistra la densità di probabilità, con media e mediana segnate e l'area sotto la curva fino al novantacinquesimo percentile evidenziata. A destra la funzione di ripartizione, che sale da zero a uno, e su cui lo stesso percentile si legge come l'ascissa a cui la curva raggiunge 0,95."
:width: 100%

La stessa variabile, due modi di guardarla. Sulla densità un percentile è
un'area; sulla ripartizione è un punto, e questo è il motivo per cui i
percentili si leggono sulla seconda.
```

La corrispondenza fra i due grafici di {numref}`fig-densita-percentili` è
comoda da avere in testa quando si passerà alle metriche di servizio. Dire
«il novantacinquesimo percentile della latenza» significa scegliere
un'ordinata sulla curva di destra e leggere l'ascissa: una domanda che sulla
densità richiederebbe di integrare.

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

## Il centro e la larghezza

Due numeri riassumono una distribuzione: dove sta il suo centro e quanto è
dispersa attorno ad esso.

```{figure} ../figures/media-mediana-moda-varianza-numpy.svg
:name: fig-centro-larghezza
:alt: "Una distribuzione asimmetrica, con la coda allungata a destra, annotata con tre indicatori di centro che cadono in punti diversi: la moda sul picco, la mediana poco più a destra e la media ancora più a destra, tirata dalla coda. Sotto, un segmento marca la deviazione standard attorno alla media."
:width: 92%

Tre centri per la stessa distribuzione. Su una curva simmetrica coinciderebbero;
su una asimmetrica no, e la media è quella che la coda sposta di più.
```

L'ordine in cui i tre indicatori compaiono in {numref}`fig-centro-larghezza`
non è casuale ed è un buon promemoria pratico: quando media e mediana si
allontanano molto, la distribuzione ha una coda, e riassumerla con la sola
media descrive un valore tipico che quasi nessuno osserva.

`````{tab} Elementare

Il **valore atteso** $E[X]$ è la media dei valori possibili, ciascuno pesato
dalla sua probabilità: è il risultato medio che ci aspettiamo "a lungo
andare". Per un dado onesto vale $\tfrac{1+2+3+4+5+6}{6}=3{,}5$: un numero che
non uscirà mai in un singolo lancio, ma attorno al quale si assesta la media
di tanti lanci. La **varianza** $\mathrm{Var}(X)$ misura invece quanto
tipicamente ci si allontana dal centro: piccola se i valori sono raccolti,
grande se sono sparpagliati. La sua radice quadrata, la **deviazione
standard**, esprime quello scarto tipico nelle stesse unità dei valori: per il
dado vale circa $1{,}7$ punti (in media, un lancio cade a un paio di punti dal
centro $3{,}5$).

`````

`````{tab} Superiore

Per una variabile discreta, con $\mu=E[X]$:

$$
E[X]=\sum_x x\,p(x), \qquad
\mathrm{Var}(X)=E\big[(X-\mu)^2\big]=E[X^2]-\mu^2 ;
$$

per una continua le somme diventano integrali. La **deviazione standard**
$\sigma=\sqrt{\mathrm{Var}(X)}$ riporta la dispersione nelle stesse unità di $X$.
Il valore atteso è lineare, $E[aX+b]=a\,E[X]+b$, proprietà che useremo di
continuo. Per il dado: $E[X^2]=\tfrac{91}{6}\approx 15{,}17$, quindi
$\mathrm{Var}(X)=15{,}17-3{,}5^2\approx 2{,}92$ e $\sigma\approx 1{,}71$.

`````

## Due distribuzioni ovunque

Di distribuzioni ne esistono molte, ma due tornano di continuo. La **Bernoulli**
descrive ogni singola prova "sì/no"; la **normale** (o gaussiana) descrive tutto
ciò che si accumula attorno a un valore medio.

```{figure} ../figures/curva-normale.svg
:name: fig-curva-normale
:alt: Curva a campana della distribuzione normale, con la media al centro e le bande a una e due deviazioni standard evidenziate.
:width: 85%

La distribuzione normale $\mathcal{N}(\mu,\sigma^2)$. Circa il $68\%$ della
probabilità cade entro $\mu\pm\sigma$ e circa il $95\%$ entro $\mu\pm 2\sigma$.
```

`````{tab} Elementare

Una Bernoulli è una moneta, magari truccata: esce $1$ (successo) con probabilità
$p$ e $0$ con probabilità $1-p$. Serve a modellare qualunque esito binario:
click/non click, spam/non spam.

La normale è la celebre **curva a campana** ({numref}`fig-curva-normale`):
simmetrica, con la maggior parte dei valori stretti attorno alla media $\mu$ e
code che si assottigliano ai lati. Vale una regola pratica utilissima, la
"68–95": circa il $68\%$ dei casi cade entro una deviazione standard dalla media
($\mu\pm\sigma$), circa il $95\%$ entro due ($\mu\pm 2\sigma$). Altezze, errori di
misura, rumore: in natura la campana è dappertutto.

`````

`````{tab} Superiore

La Bernoulli$(p)$ ha $P(X{=}1)=p$, valore atteso $E[X]=p$ e varianza
$\mathrm{Var}(X)=p(1-p)$. La normale $\mathcal{N}(\mu,\sigma^2)$ ha densità

$$
f(x)=\frac{1}{\sigma\sqrt{2\pi}}\;
\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right),
$$

dove $\mu$ è la media (il centro della campana) e $\sigma$ la deviazione
standard (la sua larghezza). Perché è così onnipresente? Per il **teorema del
limite centrale** (de Moivre, Laplace, poi Lyapunov): la somma (o la media) di
tante variabili aleatorie indipendenti e a varianza finita, *quale che sia* la
loro distribuzione di partenza, tende a una normale al crescere del numero di
termini. È il motivo per cui gli errori di misura si modellano gaussiani e per
cui il rumore e l'inizializzazione dei pesi nelle reti neurali sono spesso
normali.

`````

Vale la pena vedere il teorema del limite centrale **succedere**, perché
detto a parole sembra una promessa e guardato sembra un trucco.

```{figure} ../figures/limite-centrale.gif
:name: fig-limite-centrale
:alt: "Animazione: a sinistra le sei facce di un dado, tutte della stessa altezza; a destra un istogramma delle somme di tre dadi che si riempie lotto dopo lotto e assume la forma di una campana, su cui compare la curva normale prevista dalla teoria."
:width: 90%

Un dado è **piatto**: nessuna faccia è più probabile di un'altra. Eppure la
somma di tre dadi, ripetuta seicento volte, si dispone da sé lungo la campana,
e la curva sovrapposta non è adattata ai dati: è la
$\mathcal{N}(n\mu,\, n\sigma^2)$ che il teorema prevede prima di tirare.
```

Due cose valgono più della formula, nella {numref}`fig-limite-centrale`. La
prima è che la distribuzione di partenza è **la più piatta possibile**, e la
campana arriva lo stesso: è questo il senso di *quale che sia*. La seconda è
che $n=3$ (tre soli addendi) basta già a renderla riconoscibile.

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
un'abitudine diffusa: trattare come gaussiana l'accuratezza misurata su un
test set, che è esattamente un conteggio di successi su prove indipendenti.
È l'approssimazione su cui poggiano gli intervalli di confidenza che vedremo
fra poco.

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

Che il prior domini si vede meglio spostandolo. Teniamo lo stesso test
($99\%$ di sensibilità, $5\%$ di falsi positivi) e rendiamo la malattia dieci
volte più rara, una persona su mille invece di una su cento.

```{figure} ../figures/teorema-bayes-test-medici.svg
:name: fig-bayes-test
:alt: "Diagramma ad albero su una popolazione di 10.000 persone sottoposte a screening, con prevalenza dello 0,1%: si dividono in 10 malate e 9.990 sane. Delle 10 malate, il 99% risulta positivo, circa 10 positivi veri e nessun falso negativo. Delle 9.990 sane, il 5% risulta positivo per errore, circa 500 falsi positivi, e le restanti 9.490 negative. In basso il conto: su circa 510 positivi totali, i malati veri sono 10, cioè circa il 2%."
:width: 96%

Lo stesso test su una malattia dieci volte più rara. I falsi positivi non
cambiano di molto (vengono dai sani, che sono tanti); i positivi veri sì, e la
probabilità di essere davvero malati crolla dal $17\%$ al $2\%$.
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
:alt: "Grafico della media osservata al crescere del numero di prove: nelle prime decine oscilla ampiamente sopra e sotto il valore vero, poi le oscillazioni si restringono progressivamente e la curva si assesta sulla linea orizzontale del valore atteso, senza però toccarla esattamente."
:width: 92%

Le oscillazioni non spariscono: si restringono. È una differenza che conta,
perché nessun numero di prove rende la media *uguale* al valore vero.
```

La forma a imbuto di {numref}`fig-legge-grandi-numeri` è la stessa che governa
quanti dati servono per addestrare o per valutare un modello. Le oscillazioni
si stringono come $1/\sqrt{n}$: per dimezzare l'incertezza di una misura non
bastano il doppio dei dati, ne servono quattro volte tanti.

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

Due condizioni non sono pignoleria: le osservazioni devono essere
**indipendenti** e provenire dalla **stessa distribuzione**. Se si influenzano
a vicenda, o se la distribuzione cambia strada facendo, la garanzia salta:
mille recensioni scritte dagli amici del ristoratore non valgono mille
recensioni di clienti qualsiasi. È la stessa ragione per cui un dataset
raccolto male non migliora aggiungendone altro raccolto allo stesso modo.

`````

`````{tab} Superiore

Siano $X_1,\dots,X_n$ variabili i.i.d. con media $\mu=E[X]$ finita. La media
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
:alt: "Tre modelli su un asse dell'accuratezza da 80 a 92 per cento, ciascuno con la sua stima puntuale e una barra d'errore. Il modello A sta all'87,2% e il modello B all'86,8%: le loro barre si sovrappongono ampiamente, e una nota dice che con intervalli sovrapposti non c'è un vincitore. Il modello C sta all'81,0%, con la barra nettamente staccata dalle altre due."
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
della prestazione vera, calcolata su un campione finito. Il numero che
interessa davvero (quanto il modello risponderebbe bene su tutti i casi
possibili) non lo conosceremo mai.

L'analogia giusta è il sondaggio elettorale. Nessun istituto serio titola "il
candidato A è al $47{,}2\%$" senza il margine d'errore: con mille intervistati
quel numero significa "da qualche parte fra il $44\%$ e il $50\%$". Valutare un
modello su $500$ esempi è come sondare $500$ elettori.

Con $500$ esempi e un'accuratezza dell'$87\%$, il margine al $95\%$ è di circa
**$\pm 3$ punti**: la stima onesta è "fra l'$84\%$ e il $90\%$". Il modello
vecchio sta fra l'$83{,}8\%$ e l'$89{,}8\%$. I due intervalli si sovrappongono
quasi per intero: quel miglioramento non è distinguibile dal rumore.

Il margine si stringe solo aumentando gli esempi, e lentamente: servono
$5\,000$ esempi per scendere a circa $\pm 1$ punto, $50\,000$ per arrivare a
$\pm 0{,}3$.

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

Il confonditore, disegnato. Fra gelati e annegamenti non passa nessuna
freccia: il legame che si misura nei dati è tutto riflesso di quello che
ciascuno dei due ha con il caldo.
```

Vale la pena notare cosa *non* c'è in {numref}`fig-confonditore`: la freccia
fra $X$ e $Y$. I dati da soli non la disegnano né la cancellano, perché
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
una relazione a parabola ha correlazione nulla e dipendenza perfetta.

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
sorprendenti possibile*. Lancio una moneta 10 volte e vedo 7 teste: quale valore
di $p$ spiega meglio quel che ho osservato? Il valore $p=0{,}7$, perché è quello
che rende l'osservazione più probabile. "Imparare", per un modello, è spesso
esattamente questo: girare le manopole dei parametri finché i dati osservati
diventano i più plausibili.

`````

`````{tab} Superiore

Dati esempi indipendenti $x^{(1)},\dots,x^{(m)}$, la **verosimiglianza** dei
parametri $\theta$ è

$$
\mathcal{L}(\theta)=\prod_{i=1}^{m} p\big(x^{(i)};\theta\big),
$$

e la stima di massima verosimiglianza è

$$
\hat{\theta}=\arg\max_{\theta}\ \sum_{i=1}^{m}\log p\big(x^{(i)};\theta\big),
$$

dove il logaritmo trasforma il prodotto in somma, numericamente più stabile. Il
punto cruciale: massimizzare la log-verosimiglianza sotto ipotesi gaussiana
equivale a **minimizzare l'errore quadratico medio**, e sotto ipotesi di
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

```{admonition} Da ricordare
:class: important
- La probabilità misura l'incertezza; una **variabile aleatoria** le dà un
  numero, e $E[X]$ e $\mathrm{Var}(X)$ ne riassumono centro e dispersione.
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
