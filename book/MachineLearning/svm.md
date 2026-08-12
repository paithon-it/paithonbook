# Support Vector Machine: il margine massimo

Disegna due nuvole di punti su un foglio (pallini blu a sinistra, quadratini
rossi a destra) e traccia una retta che li separi. Facile. Ora traccia
*un'altra* retta che li separi lo stesso, e poi un'altra ancora: se le due
nuvole sono ben distinte, di rette buone ce ne sono infinite. Quale scegliere?
La regressione logistica del capitolo sull'apprendimento supervisionato ne
sceglie una, e nemmeno si pone la domanda. La **Support Vector Machine** (SVM)
invece sì, e con una risposta di netta
eleganza geometrica: tra tutte le rette che separano, scegli la più *prudente*
(quella che lascia il corridoio più largo possibile tra le due classi).

L'idea del margine massimo è vecchia, e non nasce dove si crede: la formularono
in Unione Sovietica **Vladimir Vapnik e Alexey Chervonenkis**, che dal 1962
lavoravano all'Istituto di Problemi di Controllo di Mosca e la pubblicarono nel
1963-64 col nome di *metodo del ritratto generalizzato*
{cite}`vapnik1963pattern`. Quello che nasce trent'anni dopo nei laboratori Bell
sono i due innesti che la rendono praticabile, e sono anche le due sezioni che
seguono: il **kernel trick** (Boser, Guyon e Vapnik, 1992
{cite}`boser1992training`) e il **margine morbido** (Cortes e Vapnik, 1995
{cite}`cortes1995support`). Un algoritmo che ha aspettato trent'anni due idee.

Per un decennio, prima dell'ondata del deep
learning, le SVM sono state il classificatore di riferimento: matematicamente
solide, sorprendentemente efficaci su dataset di dimensioni medie, e ancora
oggi una scelta sensata quando gli esempi sono poche migliaia.

## La retta più prudente

Immagina il confine tra due quartieri di case. Potresti tracciarlo rasente al
muro dell'ultima villetta di uno dei due, ma basterebbe una casa nuova, un
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
più vicini. Solo quei punti (i **vettori di supporto**, cerchiati) determinano
la soluzione.
```

Come mostra {numref}`fig-svm-margine`, la soluzione poggia su pochissimi
punti: quelli che toccano i bordi del corridoio. Tutti gli altri, per quanto
numerosi, sono irrilevanti: potresti spostarli o cancellarli e il confine non
si muoverebbe di un millimetro. Sono i punti sul bordo a «reggere»
l'iperpiano, e per questo si chiamano **vettori di supporto**. È una
differenza sostanziale rispetto alla regressione logistica, la cui frontiera
dipende (sia pur poco) da *tutti* i dati.

## Il classificatore a massimo margine

Formalizziamo la geometria del corridoio. La frontiera è un **iperpiano**: una
retta in due dimensioni, un piano in tre, un oggetto piatto di dimensione
$n-1$ in uno spazio a $n$ dimensioni.

`````{tab} Elementare

L'iperpiano è l'insieme dei punti $\mathbf{x}$ che soddisfano l'equazione

$$
\mathbf{w}^\top \mathbf{x} + b = 0,
$$

ed è lo stesso conto che faceva la regressione logistica prima di schiacciare
il risultato nella curva a S: moltiplica ogni caratteristica del punto per il
suo peso, somma tutto, aggiungi il numero di partenza $b$, e guarda che segno
ha il risultato. Positivo di qua, negativo di là, e zero esattamente sul
confine. Nella scrittura, $\mathbf{w}$ è l'elenco dei pesi (che dà
l'orientamento della frontiera), $\mathbf{x}$ l'elenco delle caratteristiche
del punto, e la scrittura $\mathbf{w}^\top \mathbf{x}$ è solo un modo compatto
di dire «moltiplica a coppie e somma»; $b$ sposta la frontiera avanti o
indietro. La novità della SVM non è
questa formula, è il criterio con cui sceglie $\mathbf{w}$ e $b$: non una
frontiera qualsiasi, ma quella che lascia il vuoto più ampio attorno a sé. Più
il corridoio è largo, più il classificatore è robusto.

`````

`````{tab} Superiore

Fissiamo la scala imponendo che gli esempi più vicini alla frontiera
soddisfino $\mathbf{w}^\top \mathbf{x} + b = \pm 1$: sono le due rette di
margine. Con la convenzione $y_i \in \{-1, +1\}$ per le due classi, chiedere
che ogni punto stia dalla parte giusta *e fuori dal corridoio* si scrive in un
colpo solo:

$$
y_i\,(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1, \qquad i = 1, \dots, m.
$$

La distanza di un punto sul margine dall'iperpiano è $1/\lVert \mathbf{w}\rVert$,
quindi la larghezza totale del corridoio (da un bordo all'altro) è

$$
\text{margine} = \frac{2}{\lVert \mathbf{w}\rVert}.
$$

Massimizzare il margine equivale allora a *minimizzare*
$\lVert \mathbf{w}\rVert$, o più comodamente il suo quadrato (differenziabile
ovunque). Il problema del
**margine rigido** (*hard margin*) è

$$
\min_{\mathbf{w},\,b}\ \tfrac{1}{2}\lVert \mathbf{w}\rVert^2
\quad\text{soggetto a}\quad
y_i\,(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 \ \ \forall i,
$$

dove $\mathbf{w}$ è il vettore dei pesi, $b$ il termine di bias,
$\mathbf{x}_i$ l'$i$-esimo esempio e $y_i \in \{-1,+1\}$ la sua etichetta. È
un problema di programmazione quadratica *convesso*: ha un'unica soluzione, senza minimi
locali in cui restare intrappolati.

`````

### Un esempio con i numeri

Mettiamo in fila numeri concreti, in due dimensioni: quattro punti, due per
classe, tutti su una diagonale.

`````{tab} Elementare

Quattro case lungo una strada in diagonale: due del quartiere blu, a $(0,0)$ e
$(-1,-1)$, e due del quartiere rosso, a $(2,2)$ e $(3,3)$. Le due case più
vicine fra loro, una per quartiere, sono $(0,0)$ e $(2,2)$: sono loro a
decidere tutto. Il corridoio più largo possibile è quello che va dall'una
all'altra, e il confine passa esattamente a metà strada, per il punto
$(1,1)$, messo di traverso rispetto alla diagonale. La larghezza del corridoio
è la distanza fra le due case, circa $2{,}8$.

E le altre due case, quelle più arretrate? Prova a cancellarle dal foglio: il
confine non si sposta di un millimetro, perché non toccano il corridoio. Le
due case sul bordo sono i **vettori di supporto**: reggono da sole l'intera
soluzione.

`````

`````{tab} Superiore

Prendiamo quattro punti:

| punto | coordinate | classe $y_i$ |
|-------|------------|--------------|
| $\mathbf{x}_1$ | $(0,\,0)$  | $-1$ |
| $\mathbf{x}_2$ | $(2,\,2)$  | $+1$ |
| $\mathbf{x}_3$ | $(-1,-1)$  | $-1$ |
| $\mathbf{x}_4$ | $(3,\,3)$  | $+1$ |

I punti stanno tutti sulla diagonale. Per simmetria l'iperpiano di massimo
margine è perpendicolare alla diagonale e passa a metà strada tra $\mathbf{x}_1$ e
$\mathbf{x}_2$, cioè per il punto $(1,1)$: questo fissa la *direzione* di
$\mathbf{w}$, che è quella della diagonale $(1,1)$. La *scala* la fissa la
convenzione $\mathbf{w}^\top \mathbf{x} + b = \pm 1$ sui punti di margine, e i
due vincoli (in $\mathbf{x}_1$ e in $\mathbf{x}_2$) danno

$$
\mathbf{w} = (0{,}5,\ 0{,}5), \qquad b = -1.
$$

Verifichiamo che i due punti più interni, $\mathbf{x}_1$ e $\mathbf{x}_2$,
cadano esattamente sulle rette di margine, cioè soddisfino
$y_i(\mathbf{w}^\top \mathbf{x}_i + b) = 1$:

- $\mathbf{x}_1=(0,0)$, classe $-1$:
  $\;\mathbf{w}^\top \mathbf{x}_1 + b = 0{,}5\cdot 0 + 0{,}5\cdot 0 - 1 = -1$,
  e $\;y_1(-1) = (-1)(-1) = 1$. ✓
- $\mathbf{x}_2=(2,2)$, classe $+1$:
  $\;\mathbf{w}^\top \mathbf{x}_2 + b = 0{,}5\cdot 2 + 0{,}5\cdot 2 - 1 = 1$,
  e $\;y_2(1) = (+1)(1) = 1$. ✓

Sono loro i **vettori di supporto**. Gli altri due stanno più lontani, oltre il
margine (il vincolo vale con la disuguaglianza *stretta*):

- $\mathbf{x}_3=(-1,-1)$:
  $\;\mathbf{w}^\top \mathbf{x}_3 + b = -0{,}5 - 0{,}5 - 1 = -2$,
  e $\;y_3(-2) = (-1)(-2) = 2 \ge 1$. ✓
- $\mathbf{x}_4=(3,3)$:
  $\;\mathbf{w}^\top \mathbf{x}_4 + b = 1{,}5 + 1{,}5 - 1 = 2$,
  e $\;y_4(2) = (+1)(2) = 2 \ge 1$. ✓

La larghezza del corridoio è

$$
\frac{2}{\lVert \mathbf{w}\rVert} = \frac{2}{\sqrt{0{,}5^2 + 0{,}5^2}}
= \frac{2}{\sqrt{0{,}5}} = \frac{2}{0{,}707} \approx 2{,}83 = 2\sqrt{2}.
$$

Ed è esattamente la distanza euclidea tra $\mathbf{x}_1=(0,0)$ e
$\mathbf{x}_2=(2,2)$, che vale
$\sqrt{2^2+2^2}=\sqrt{8}=2\sqrt{2}$: i due vettori di supporto, uno per
classe, si affacciano sui bordi opposti dello stesso corridoio. Nota il punto
cruciale: cancellare $\mathbf{x}_3$ e $\mathbf{x}_4$ non cambia nulla; la
soluzione dipende solo dai due punti sul bordo.

`````

## Quando i dati non sono perfetti: il margine morbido

Il margine rigido ha due difetti gemelli: pretende che i dati siano
*perfettamente* separabili, e basta un solo punto fuori posto (un outlier, un
errore di misura) per stravolgere la soluzione o renderla impossibile. Nel
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
\min_{\mathbf{w},\,b,\,\xi}\ \tfrac{1}{2}\lVert \mathbf{w}\rVert^2
+ C\sum_{i=1}^{m}\xi_i
\quad\text{soggetto a}\quad
y_i\,(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 - \xi_i,\ \ \xi_i \ge 0,
$$

dove $\xi_i$ è la violazione dell'$i$-esimo punto e $C > 0$ regola il
compromesso tra «corridoio largo» ($\lVert \mathbf{w}\rVert$ piccolo) e «poche
violazioni» ($\sum\xi_i$ piccolo). Eliminando i vincoli, il problema si
riscrive come minimizzazione della **hinge loss** più un termine di
regolarizzazione:

$$
\min_{\mathbf{w},\,b}\ \sum_{i=1}^{m}
\max\!\big(0,\ 1 - y_i\,(\mathbf{w}^\top \mathbf{x}_i + b)\big)
+ \frac{1}{2C}\lVert \mathbf{w}\rVert^2 .
$$

La hinge loss $\max(0,\,1 - y_i f(\mathbf{x}_i))$ è nulla per i punti ben
classificati
e fuori dal margine, e cresce *linearmente* per quelli dentro la fascia o
dalla parte sbagliata: è l'analogo, per la SVM, di ciò che la log-loss è per
la regressione logistica, con la differenza che, essendo piatta oltre il
margine, ignora del tutto i punti «facili» e dà alla SVM la sua sparsità in
vettori di supporto. In questa forma si legge chiaramente il ruolo di $C$: il
coefficiente della penalità $\lVert \mathbf{w}\rVert^2$ è $1/(2C)$, quindi **$C$ è
l'inverso della forza di regolarizzazione**. $C$ grande → penalità debole →
margine stretto, varianza alta; $C$ piccolo → penalità forte → margine largo,
bias più alto. È la stessa manopola $\lambda$ della sezione sull'overfitting,
letta al contrario; con un'avvertenza di normalizzazione: la loss Ridge era
*mediata* sugli $m$ esempi, la hinge qui è *sommata*, e a parità di
convenzione l'identificazione esatta è $\lambda = 1/(2Cm)$, cioè a meno di un
fattore pari alla taglia del dataset.

`````

### Il problema duale, e perché i vettori di supporto sono pochi

Fin qui abbiamo scritto il problema nella forma **primale**, cioè cercando
direttamente $\mathbf{w}$ e $b$. C'è una seconda scrittura equivalente, il
problema **duale**, che serve a due cose: spiega perché la soluzione dipenda
solo da una manciata di punti (un fatto finora asserito e mai dimostrato) e
apre la porta al kernel trick della prossima sezione.

`````{tab} Elementare

Torna al confine fra i due quartieri e alla frase più sorprendente di tutta la
sezione: le case lontane dal confine non contano nulla, e potresti cancellarle
senza spostarlo di un millimetro. Perché?

Immagina il confine come un muro elastico teso fra le due parti, e ogni casa
come una mano che spinge il muro per allontanarlo da sé. Le case addossate al
corridoio spingono davvero: se togli una di loro, il muro scivola. Le case
arretrate non toccano il muro, quindi la loro spinta è **esattamente zero**, e
zero moltiplicato per qualunque cosa resta zero: nella soluzione, il loro
contributo non c'è proprio. È il motivo per cui la SVM, alla fine, si porta
appresso solo pochi punti (quelli che spingono) e può dimenticare tutti gli
altri, anche se erano un milione.

`````

`````{tab} Superiore

Introducendo i moltiplicatori di Lagrange $\alpha_i \ge 0$ per i vincoli di
margine e eliminando $\mathbf{w}$ e $b$, il problema a margine morbido diventa

$$
\max_{\alpha}\ \sum_{i=1}^{m}\alpha_i
- \tfrac{1}{2}\sum_{i,j}\alpha_i\alpha_j\, y_i y_j\,
k(\mathbf{x}_i, \mathbf{x}_j)
\quad\text{con}\quad
0 \le \alpha_i \le C,\ \ \sum_{i=1}^{m}\alpha_i y_i = 0,
$$

e la soluzione primale si ricostruisce come
$\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$ (qui $k$ è il prodotto scalare;
la prossima sezione lo sostituirà con un kernel, ed è tutta lì la comodità di
questa forma: **gli esempi compaiono solo dentro prodotti scalari**).

La sparsità in vettori di supporto è una conseguenza, non un'osservazione
empirica. Le condizioni di Karush–Kuhn–Tucker impongono la
**complementarità**

$$
\alpha_i\big[\,y_i(\mathbf{w}^\top\mathbf{x}_i + b) - 1 + \xi_i\,\big] = 0 ,
$$

quindi per ogni punto strettamente fuori dal margine (dove la parentesi quadra
è diversa da zero) deve essere $\alpha_i = 0$, e quel punto sparisce dalla
somma che ricostruisce $\mathbf{w}$. Restano solo i punti *sul* margine o
dentro la fascia: i **vettori di supporto**.

Sui quattro punti dell'esempio numerico i conti si chiudono in una riga:
$\alpha_1 = \alpha_2 = 0{,}25$ e $\alpha_3 = \alpha_4 = 0$, da cui
$\mathbf{w} = 0{,}25\,(2,2) = (0{,}5;\,0{,}5)$, esattamente la soluzione
trovata per via geometrica, con $\sum_i \alpha_i y_i = 0$ rispettato.

`````

## Il kernel trick: separare l'inseparabile

Fin qui, però, la SVM traccia solo iperpiani: frontiere *diritte*. E se le due
classi sono intrecciate in modo che nessuna retta le separi: pensa a un
bersaglio, con una classe al centro e l'altra tutt'intorno ad anello? Qui
entra in gioco l'idea più affascinante di tutta la storia delle SVM, quella
che le ha rese celebri: il **kernel trick**.

`````{tab} Elementare

Prendi il bersaglio: cerchio interno di una classe, anello esterno dell'altra.
Sul foglio, piatto, nessuna retta li separa. Ma immagina di *sollevare* ogni
punto in aria di un'altezza pari a quanto è lontano dal centro: i punti del
cerchio interno, vicini al centro, restano bassi; quelli dell'anello, lontani,
salgono in alto. Ora le due classi stanno a quote diverse, e un semplice
*piano orizzontale* (una lastra di vetro infilata a mezz'aria) le separa
nettamente. Non abbiamo cambiato i punti: li abbiamo guardati in uno spazio
con una dimensione in più, e lì il problema è diventato lineare.

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
migliaia di dimensioni, calcolare e conservare tutti quei
$\phi(\mathbf{x}_i)$ diventa proibitivo.

Il **kernel trick** è l'osservazione che salva tutto: nel problema duale
appena scritto, gli esempi compaiono *solo* attraverso prodotti scalari
$\phi(\mathbf{x})^\top\phi(\mathbf{z})$. Se esiste una funzione $k$ che
calcola quel prodotto scalare direttamente dalle coordinate originali,

$$
k(\mathbf{x}, \mathbf{z}) = \phi(\mathbf{x})^\top\phi(\mathbf{z}),
$$

allora non serve mai costruire $\phi$: si lavora nello spazio ad alta dimensione
*senza mai visitarlo*. La funzione $k$ è il **kernel**
{cite}`scholkopf2002learning`. I più usati:

$$
\begin{aligned}
&\text{lineare:} && k(\mathbf{x},\mathbf{z}) = \mathbf{x}^\top \mathbf{z}, \\
&\text{polinomiale:} && k(\mathbf{x},\mathbf{z})
   = (\mathbf{x}^\top \mathbf{z} + c)^{d}, \\
&\text{RBF / gaussiano:} && k(\mathbf{x},\mathbf{z})
   = \exp\!\big(-\gamma\,\lVert \mathbf{x} - \mathbf{z}\rVert^2\big),
\end{aligned}
$$

dove $d$ è il grado del polinomio, $c \ge 0$ un termine costante e $\gamma > 0$
il parametro di ampiezza del kernel gaussiano, che è l'**inverso** della
larghezza della campana (la deviazione standard equivalente vale
$1/\sqrt{2\gamma}$): $\gamma$ grande, campana stretta. Il kernel RBF
corrisponde a uno spazio $\phi$ di dimensione *infinita*: sarebbe impossibile
da costruire, eppure $k$ si calcola in una riga.

Un'ultima clausola, quella che fa del trucco un teorema invece che una
speranza. La frase «se esiste una funzione $k$ che calcola quel prodotto
scalare» rovescia l'ordine dei fatti: in pratica non si parte da $\phi$ per
cercare $k$, si **sceglie** $k$ e si spera che un $\phi$ esista. Esiste se e
solo se $k$ è simmetrica e **semidefinita positiva**, cioè se ogni matrice di
Gram $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ ha autovalori non negativi: è il
teorema di Mercer {cite}`scholkopf2002learning`, ed è la ragione per cui i
kernel non si inventano a piacere. Se $k$ non lo è, il duale smette di essere
concavo e il solutore sta risolvendo un problema diverso da quello che si
crede. Non ogni «misura di somiglianza» è un kernel, ed è l'errore più comune
di chi prova a scriversene uno.

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

`````{tab} Elementare

Prima di tutto, il collegamento con la pagina precedente, perché sembrano due
storie diverse e sono la stessa. Là abbiamo sollevato i punti in aria per
separarli con una lastra di vetro; qui diciamo che il kernel misura quanto due
punti si somigliano. Il ponte è questo: **il kernel è il prodotto scalare dei
punti già sollevati**, cioè misura quanto due punti si somigliano *nel nuovo
spazio*, senza che nessuno debba costruirlo. Sollevamento e somiglianza sono la
stessa cosa vista da due lati: si sceglie la somiglianza, e il sollevamento
viene dietro gratis.

Pensa allora a ogni punto come a un lampione acceso di notte: illumina bene chi
gli sta accanto, sempre meno chi si allontana, per niente chi è lontano. Il
kernel misura proprio questa «luce» su una scala da $0$ a $1$, dove $1$ vuol
dire «stesso punto». Chiamiamo *portata* del lampione la distanza alla quale la
luce è scesa a poco più di un terzo, cioè $0{,}37$: mettiamo un metro e mezzo.
Chi sta a un metro e mezzo si vede ancora. E chi sta al doppio, a tre metri?
Non riceve la metà della luce, e nemmeno un terzo: la formula fa scendere la
luce con il **quadrato** della distanza, e a distanza doppia il conto è
$0{,}37$ elevato alla quarta, cioè appena $0{,}018$, meno di due centesimi. Per
lui è già buio. Ecco perché il raggio d'influenza di un punto finisce così
bruscamente.

La manopola $\gamma$ decide quanto è stretto il cono di luce, ed è la portata
**al contrario**: $\gamma$ grande, luce corta, e la frontiera viene frastagliata
perché ogni punto comanda solo nel suo cortile (rischio di imparare il rumore);
$\gamma$ piccolo, luce lunga, e la frontiera esce morbida.

`````

`````{tab} Superiore

Vediamolo con i numeri, scegliendo $\gamma = 0{,}5$:

- due punti *vicini*, $\mathbf{x}=(2,2)$ e $\mathbf{z}=(3,3)$, distano
  $\lVert \mathbf{x}-\mathbf{z}\rVert^2 = 1^2 + 1^2 = 2$, quindi
  $k(\mathbf{x},\mathbf{z}) = e^{-0{,}5\cdot 2} = e^{-1} \approx 0{,}37$:
  si «vedono» bene;
- due punti *lontani*, $\mathbf{x}=(2,2)$ e $\mathbf{z}=(0,0)$, distano
  $\lVert \mathbf{x}-\mathbf{z}\rVert^2 = 2^2 + 2^2 = 8$, quindi
  $k(\mathbf{x},\mathbf{z}) = e^{-0{,}5\cdot 8} = e^{-4} \approx 0{,}018$:
  quasi si ignorano.

Con $\gamma$ grande la campana si stringe, ogni punto influenza solo i vicinissimi
e la frontiera si fa frastagliata (varianza alta, rischio overfitting); con
$\gamma$ piccolo la campana si allarga, l'influenza è a lungo raggio e la
frontiera si liscia.

`````

Insieme a $C$, il parametro $\gamma$ è l'altra manopola da tarare per
validazione.

## Non solo classificare: la regressione con le SVM

Lo stesso principio si ribalta per la **regressione** (SVR, *Support Vector
Regression*). Nella classificazione la SVM vuole il corridoio più largo *tra*
le classi; nella regressione vuole un tubo che contenga *quanti più punti
possibile*.

`````{tab} Elementare

Invece di penalizzare ogni piccolo scarto tra previsione e valore vero (come
fa la regressione lineare classica) la SVR disegna un «tubo» di tolleranza
attorno alla curva: finché un punto ci sta dentro, l'errore conta *zero*.
Vengono penalizzati solo i punti che sporgono dal tubo, e solo per quanto
sporgono. È un modo indulgente di adattare i dati: non insegue le piccole
oscillazioni, si preoccupa solo degli scostamenti seri.

`````

`````{tab} Superiore

Si fissa una tolleranza $\epsilon > 0$ e si usa la **loss
$\epsilon$-insensitive**, nulla dentro il tubo e lineare fuori:

$$
L_\epsilon\big(y,\, f(\mathbf{x})\big)
= \max\!\big(0,\ |y - f(\mathbf{x})| - \epsilon\big).
$$

Gli errori entro $\pm\epsilon$ non vengono penalizzati; oltre, la penalità
cresce linearmente. Il parametro $\epsilon$ fissa l'ampiezza del tubo, mentre
$C$ regola come sempre il compromesso tra piattezza del modello e violazioni.
Anche qui vale il kernel trick, così la SVR può adattare curve non lineari
esattamente come la SVM classifica frontiere non lineari.

`````

## Una classe sola: novelty e anomaly detection

C'è un'ultima variante, e risponde a una domanda diversa: e se avessimo esempi
di *una sola* classe? Vogliamo imparare com'è fatto il «normale» (transazioni
regolari, macchinari sani, traffico di rete legittimo) per poi accorgerci di
ciò che se ne discosta. È il problema della **novelty detection** (riconoscere
il nuovo) e dell'**anomaly detection** (riconoscere il guasto), e si lega a
quel tema dei dati fuori distribuzione toccato nella sezione sui dati che
cambiano: individuare gli input troppo lontani da ciò che il modello ha visto,
invece di predire con finta sicurezza.

`````{tab} Elementare

Immagina di aver visto migliaia di transazioni oneste con la carta di credito
e nemmeno una frode. Non puoi addestrare un classificatore «onesto contro
frode»: la seconda classe non ce l'hai. La **one-class SVM** ribalta il
problema: impara a disegnare, attorno ai dati normali, il «recinto» più
stretto che li racchiude tutti. Da quel momento, ogni nuova transazione che
cade *fuori* dal recinto è sospetta: non perché somigli a una frode nota, ma
perché non somiglia a nulla di normale. Una manopola, $\nu$, dice più o meno
quale frazione di dati ci aspettiamo che finisca fuori (le anomalie
tollerate). Serve per rilevare frodi, guasti di macchinari, intrusioni
informatiche, difetti in una linea di produzione: ovunque gli esempi «anomali»
siano rari o non ancora visti.

`````

`````{tab} Superiore

La one-class SVM di Schölkopf e colleghi {cite}`scholkopf2001estimating`
adatta l'idea del margine al caso non supervisionato: mappati i dati nello
spazio delle feature con un kernel (di solito RBF), cerca l'iperpiano che
separa i punti dall'**origine** con il massimo margine. Ricondotto allo spazio
originale, questo equivale a racchiudere i dati normali in una regione
compatta; ciò che cade fuori è novità/anomalia. Il parametro $\nu \in (0,1]$
ha un doppio significato preciso: è un limite *superiore* alla frazione di
esempi di addestramento classificati come anomali (i *margin error*) e un
limite *inferiore* alla frazione di vettori di supporto. La distingue dalla
classificazione binaria un'assenza: in addestramento **non** c'è la classe
«anomalo»: si impara solo la forma del normale. Un parente stretto è la
**Support Vector Data Description** (SVDD) di Tax e Duin, che invece della
separazione dall'origine cerca la *ipersfera* minima che racchiude i dati; e
tra le alternative non-kernel ci sono l'**Isolation Forest** (che isola le
anomalie con partizioni casuali, ereditando la scalabilità degli alberi della
sezione sugli ensemble) e il *Local Outlier Factor* basato sulla densità
locale.

`````

## In pratica, con scikit-learn

In scikit-learn la famiglia SVM vive nel modulo `sklearn.svm`: `SVC` per la
classificazione con kernel, `LinearSVC` per la versione lineare veloce, `SVR`
per la regressione, `OneClassSVM` per la novelty detection. Due avvertenze
valgono per tutte, e non sono opzionali.

**Standardizzare sempre le feature**, cioè riportare tutte le colonne alla
stessa scala prima di dare i dati al modello: si sottrae a ogni colonna la sua
media e la si divide per la sua ampiezza tipica, così che i metri quadri (che
valgono decine) e il numero di stanze (che vale unità) contino allo stesso
modo. La SVM misura distanze, e senza questa operazione una colonna con numeri
grandi domina il conto e schiaccia le altre, esattamente come
succede al k-NN. Si antepone quindi sempre uno `StandardScaler` in una
`Pipeline`, come nel pattern «In pratica, con scikit-learn» delle sezioni
precedenti.

**Attenzione ai numeri grandi.** Il costo di addestramento cresce assai più in
fretta del numero di esempi: raddoppiando gli esempi il lavoro non raddoppia,
si moltiplica per quattro o per otto. Nella notazione con cui si scrivono
queste crescite (si legge «ordine di») è circa fra $O(m^2)$ e $O(m^3)$ nel
numero di esempi $m$: ottima da poche
centinaia a qualche decina di migliaia di punti, diventa proibitiva su milioni.
Per i dataset molto grandi si ripiega su modelli lineari (`LinearSVC`,
`SGDClassifier`, che scalano circa come $O(m)$) o sugli alberi in boosting della
sezione precedente {cite}`geron2022hands`.

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

# Regressione: qui serve un target CONTINUO, non le classi 0/1 di sopra.
# Fabbrichiamone uno: una sinusoide della prima coordinata, con un po' di rumore.
rng = np.random.default_rng(0)
y_reg = np.sin(3 * X[:, 0]) + rng.normal(0, 0.1, size=len(X))

# il tubo epsilon-insensitive ignora gli scarti piccoli
reg = make_pipeline(StandardScaler(),
                    SVR(kernel="rbf", C=10.0, epsilon=0.1))
reg.fit(X, y_reg)

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

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- La **SVM** sceglie, tra le infinite linee che separano due classi, la più
  prudente: quella che lascia il **corridoio più largo** possibile fra i due
  quartieri. A reggere il confine sono solo i pochi punti che ne toccano i
  bordi, i **vettori di supporto**: gli altri si possono cancellare dal foglio
  e il confine non si sposta di un millimetro.
- Il **margine morbido** mette in conto qualche sconfinamento, e una manopola
  decide quanto è severa: severa, il corridoio si stringe pur di accontentare
  quasi tutti (e si rischia di inseguire il rumore); indulgente, il corridoio
  si allarga ed è più robusto. I punti già comodamente fuori dal corridoio non
  pesano affatto: a reggere il confine restano sempre e solo le poche case sul
  bordo.
- Il **kernel trick** rende curva la frontiera: gli stessi punti si guardano in
  uno spazio con una dimensione in più, e lì tornano separabili da un taglio
  dritto. È il bersaglio sollevato in aria, ogni punto tanto più in alto quanto
  più è lontano dal centro, finché una lastra di vetro orizzontale divide il
  centro dall'anello: i punti non sono cambiati, è cambiato il posto da cui li
  guardiamo. Il modo più usato di misurare quanto due punti si somigliano è
  quello «a lampione», dove ogni punto illumina i vicini: luce corta, frontiera
  frastagliata; luce lunga, frontiera morbida.
- La stessa idea serve anche a **prevedere numeri** (un tubo di tolleranza
  attorno alla curva: finché il punto ci sta dentro, l'errore conta zero) e a
  **riconoscere le anomalie** (un recinto attorno ai dati normali, e chi cade
  fuori è sospetto, senza aver mai visto una frode).
- In pratica: portare **sempre** tutte le caratteristiche alla stessa scala,
  perché la SVM ragiona per distanze; e ricordare che il conto cresce assai più
  in fretta del numero di esempi, tanto che oltre le decine di migliaia la SVM
  con kernel diventa impraticabile.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- La **SVM** sceglie, tra le infinite frontiere che separano due classi, quella
  a **margine massimo**: il corridoio $2/\lVert \mathbf{w}\rVert$ più largo.
  L'idea è di Vapnik e Chervonenkis (1963-64); dai laboratori Bell arrivano
  trent'anni dopo il kernel trick e il margine morbido.
- La soluzione dipende solo dai **vettori di supporto**, e non è
  un'osservazione ma un corollario: la complementarità KKT
  $\alpha_i[y_i(\mathbf{w}^\top\mathbf{x}_i+b)-1+\xi_i]=0$ annulla $\alpha_i$
  per ogni punto fuori dal margine, e $\mathbf{w}=\sum_i\alpha_iy_i\mathbf{x}_i$
  non lo contiene.
- Il **margine morbido** ammette violazioni $\xi_i$ pagate dal parametro $C$,
  l'**inverso** della forza di regolarizzazione: $C$ grande → margine stretto
  (overfitting), $C$ piccolo → margine largo. La perdita è la **hinge loss**,
  parente della log-loss ma piatta oltre il margine.
- Il **kernel trick** rende non lineare la SVM: mappa i dati in uno spazio più
  ampio dove diventano separabili, calcolando i prodotti scalari con un
  **kernel** $k(\mathbf{x},\mathbf{z})=\phi(\mathbf{x})^\top\phi(\mathbf{z})$
  senza costruirlo: funziona perché nel **duale** gli esempi compaiono solo
  dentro prodotti scalari, e vale se e solo se $k$ è simmetrica e semidefinita
  positiva (Mercer). Kernel principali: lineare, polinomiale, RBF, dove
  $\gamma$ è l'**inverso** della larghezza della campana.
- La **SVR** regredisce con un tubo $\epsilon$-insensitive; la **one-class SVM**
  ($\nu$ = frazione di anomalie attese) impara la regione dei dati normali per
  la **novelty/anomaly detection**, senza vedere esempi anomali.
- In pratica: **standardizzare sempre** le feature; il costo $O(m^2)$–$O(m^3)$
  sconsiglia la SVM con kernel oltre le decine di migliaia di esempi.
```

`````
