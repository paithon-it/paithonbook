# Support Vector Machine: il margine massimo

Disegna due nuvole di punti su un foglio (pallini blu a sinistra, quadratini
rossi a destra) e traccia una retta che li separi. Facile. Ora traccia
*un'altra* retta che li separi lo stesso, e poi un'altra ancora: se le due
nuvole sono ben distinte, di rette buone ce ne sono infinite. Quale scegliere?
La regressione logistica della sezione sull'apprendimento supervisionato ne
sceglie una, e nemmeno si pone la domanda. La **Support Vector Machine** (SVM)
invece sì, e con una risposta di netta
eleganza geometrica: tra tutte le rette che separano, scegli la più *prudente*
(quella che lascia il corridoio più largo possibile tra le due classi).

L'idea del margine massimo è vecchia, e non nasce dove si crede: nasce a Mosca,
all'Istituto di Problemi di Controllo, dove dal 1962 **Vladimir Vapnik** ne
discuteva con Alexander Lerner e **Alexey Chervonenkis**. Il nome era *metodo
del ritratto generalizzato*, e lo pubblicano Vapnik e Lerner nel 1963
{cite}`vapnik1963pattern`; l'anno dopo Vapnik e Chervonenkis ne ricavano il
classificatore lineare a margine rigido. Quello che nasce trent'anni dopo nei
laboratori Bell sono i due innesti che la rendono praticabile, e sono anche le
due sezioni che seguono: il **kernel trick** (Boser, Guyon e Vapnik, 1992
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
si muoverebbe di un millimetro. Sono i punti sul bordo a «reggere» la
frontiera, e per questo si chiamano **vettori di supporto**. Il «supporto» è
questo, e il «vettore» viene da come li scriviamo: un esempio è un elenco
ordinato di numeri, uno per colonna, e un elenco del genere si chiama vettore
(lo abbiamo incontrato nella sezione sull'apprendimento supervisionato). Un
vettore di supporto, insomma, è semplicemente **uno dei pochi esempi appoggiati
al bordo del corridoio**.

È una
differenza sostanziale rispetto alla regressione logistica, la cui frontiera
dipende (sia pur poco) da *tutti* i dati.

## Il classificatore a massimo margine

Adesso mettiamo in numeri la geometria del corridoio. La frontiera è quello che
in matematica si chiama un **iperpiano**, e la parola spaventa più della cosa:
in due dimensioni è una retta, in tre un piano, e in cento dimensioni è
l'oggetto che fa lo stesso mestiere, cioè taglia lo spazio in due metà, solo
che non lo possiamo disegnare. Sta sempre una dimensione sotto lo spazio che
divide: una retta (una dimensione) in un piano (due), un piano (due) in una
scatola (tre).

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

Il conto che porta a questa formula occupa due righe e sta più avanti, nella
sezione sull'approccio della strada più larga, dove il problema viene ricavato
da capo per intero. Massimizzare il margine equivale allora a *minimizzare*
$\lVert \mathbf{w}\rVert$, o più comodamente il suo quadrato (differenziabile
ovunque). Il problema del
**margine rigido** (*hard margin*) è

$$
\min_{\mathbf{w},\,b}\ \tfrac{1}{2}\lVert \mathbf{w}\rVert^2
\quad\text{soggetto a}\quad
y_i\,(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 \ \ \forall i,
$$

dove $\mathbf{w}$ è il vettore dei pesi, $b$ il termine di bias,
$\mathbf{x}_i$ l’$i$-esimo esempio e $y_i \in \{-1,+1\}$ la sua etichetta. È
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
$(1,1)$, messo di traverso rispetto alla diagonale.

Quanto è largo il corridoio? Qui è la distanza fra le due case, e si calcola
con Pitagora: da $(0,0)$ a $(2,2)$ ci sono $2$ passi in orizzontale e $2$ in
verticale, quindi $\sqrt{2^2 + 2^2} = \sqrt{8} \approx 2{,}8$. Attenzione però,
qui va bene perché le due case sono messe proprio l'una di fronte all'altra
attraverso la strada; se fossero sfalsate, la distanza fra loro conterebbe
anche un pezzo di cammino *lungo* la strada, che con la larghezza non c'entra.
Più avanti vedremo come togliere quel pezzo di troppo.

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

dove $\xi_i$ è la violazione dell’$i$-esimo punto e $C > 0$ regola il
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

## L'approccio della strada più larga

Fin qui abbiamo chiesto fiducia su due cose e non ne abbiamo dimostrata
nessuna: che si sappia calcolare **quanto è larga** la strada (ci serve, visto
che vogliamo la più larga di tutte), e che a reggere la frontiera siano
soltanto pochi punti, i vettori di supporto.

Adesso le dimostriamo tutte e due, e lungo il percorso salterà fuori una terza
cosa che nessuno aveva cercato: il problema si può riscrivere in una forma in
cui gli esempi non compaiono più uno per uno, ma **soltanto a coppie**, e di
ogni coppia serve un numero solo. Sembra un dettaglio contabile ed è il perno di
tutto il resto della sezione, kernel trick compreso.

La strada per arrivarci è corta, sta in una pagina di algebra, e lungo il
percorso succede due volte una cosa che all'inizio non era prevedibile.

Il percorso che segue è quello che Patrick Winston chiamava *the widest street
approach*, l'approccio della strada più larga, nella sedicesima lezione del
corso 6.034 del MIT {cite}`winston2010svm`. Prendiamo in prestito anche il suo
vocabolario, che è più concreto del nostro: il corridoio diventa **una strada**,
le due rette di margine diventano **i marciapiedi**, e quello che cerchiamo è la
strada più larga che si riesca a far passare fra i più e i meno.

Winston apriva quella lezione con un avvertimento che è anche la ragione per cui
questa sezione esiste. Quando una derivazione così la si legge in un libro,
finita e ordinata, viene da pensare che Vapnik l'abbia tirata fuori tutta
insieme un sabato pomeriggio in cui il tempo era troppo brutto per uscire. Non è
così che succede. Succede in un altro modo, e alla fine della sezione vedremo
quale.

### Primo passo: da che parte della strada sei

`````{tab} Elementare

Piantiamo una freccia perpendicolare alla strada e chiamiamola $\mathbf{w}$. Di
lei sappiamo una cosa sola: la direzione, di traverso alla strada. Quanto sia
lunga non lo sappiamo ancora, e la cosa tornerà utile.

Arriva un punto nuovo, di cui non conosciamo la classe. Come decidiamo da che
parte sta? Gli facciamo fare l’**ombra sulla freccia**. Immagina la freccia
appoggiata per terra, con la coda nell'origine, e una luce che arriva
perpendicolare a lei: l'ombra del punto cade sulla freccia, in un certo punto,
e quel punto lo possiamo misurare come una distanza dalla coda. Ne esce un
numero solo. Se supera una certa soglia, il
punto ha attraversato la strada ed è un più; se resta al di qua, è un meno.

Vale la pena notare che cosa abbiamo appena buttato via. Della posizione del
punto *lungo* la strada non ci importa niente, perché camminando lungo la strada
non si cambia mai lato: l'ombra la ignora, ed è esattamente ciò che vogliamo.
Conta solo di quanto la strada la si attraversa. Quell'operazione (fare l'ombra
di un punto su una direzione e leggerne un numero solo) si chiama **prodotto
scalare**, ed è il mattone di tutto ciò che segue. È la stessa cosa del
«moltiplica a coppie e somma» di poco fa: si moltiplica ogni coordinata del
punto per la coordinata corrispondente della freccia e si sommano i risultati.
Due descrizioni molto diverse, un unico conto, ed è proprio questa doppia
natura, geometrica e aritmetica insieme, che alla fine della sezione farà il
miracolo.

`````

`````{tab} Superiore

Sia $\mathbf{w}$ un vettore perpendicolare all'asse della strada, di lunghezza
per ora indeterminata, e sia $\mathbf{u}$ un punto di classe ignota. La
proiezione di $\mathbf{u}$ su $\mathbf{w}$ è $\mathbf{w}^\top\mathbf{u}$, e la
decisione si prende confrontandola con una soglia $c$:

$$
\mathbf{w}^\top\mathbf{u} \ge c
\quad\Longleftrightarrow\quad
\mathbf{w}^\top\mathbf{u} + b \ge 0 \;\Rightarrow\; \text{classe } +1,
$$

dove si è posto $b = -c$. Questa è la **regola di decisione**, ed è il primo dei
cinque pezzi che dobbiamo mettere in fila.

Il guaio è che così com'è non serve a niente: non conosciamo $b$, e di
$\mathbf{w}$ conosciamo la direzione ma non la lunghezza. Un vettore
perpendicolare alla strada può essere lungo un metro o un chilometro, e a ogni
lunghezza corrisponde una soglia $b$ diversa che dà però la stessa frontiera. Ci
sono infinite coppie $(\mathbf{w}, b)$ che descrivono la stessa retta: mancano
vincoli, e i prossimi due passi servono a metterne abbastanza da fissarne una
sola.

`````

### Secondo passo: due vincoli che diventano uno

`````{tab} Elementare

La regola del primo passo dice solo «da che parte», e non basta. Di una casa che
conosciamo già non vogliamo sapere soltanto che sta dal lato giusto: vogliamo
che stia anche *lontana* dalla strada, non appiccicata al bordo. Alziamo quindi
l'asticella. A una casa dei più chiediamo che l'ombra superi la soglia di almeno
una tacca; a una dei meno, che le resti sotto di almeno una tacca.

Quanto vale una tacca? Lo decidiamo noi, e possiamo dire che vale $1$. Non è un
atto di fede: ricordi che la lunghezza della freccia $\mathbf{w}$ era rimasta
libera? Allungandola o accorciandola tutte le ombre si riscalano insieme, quindi
fissare la tacca a $1$ non è un'ipotesi sui dati, è la scelta del righello. Uno
dei due gradi di libertà che avanzavano lo abbiamo appena speso.

Restano due regole, una per i più e una per i meno, e portarsene dietro due è
scomodo. Il trucco è dare a ogni casa un'etichetta numerica che vale $+1$ se è
un più e $-1$ se è un meno, e moltiplicare la regola per quell'etichetta. Sui
più non cambia nulla; sui meno si moltiplicano per un numero negativo entrambi i
lati, il verso della disuguaglianza si rovescia, e le due regole diventano la
stessa identica riga. Nessuna necessità matematica lo imponeva: è comodità
dichiarata, e metà della matematica applicata è fatta di comodità dichiarate.

`````

`````{tab} Superiore

Con le etichette $y_i \in \{-1,+1\}$ già introdotte, i due vincoli separati

$$
\mathbf{w}^\top\mathbf{x}_i + b \ge +1 \ \ (y_i = +1),
\qquad
\mathbf{w}^\top\mathbf{x}_i + b \le -1 \ \ (y_i = -1)
$$

si fondono, moltiplicando ciascuno per il proprio $y_i$, nell'unica scrittura

$$
y_i\,(\mathbf{w}^\top\mathbf{x}_i + b) - 1 \ge 0,
\qquad i = 1,\dots,m .
$$

A questo aggiungiamo la condizione che completa il secondo pezzo: per gli
esempi che stanno **sul marciapiede** la disuguaglianza vale con l'uguale,

$$
y_i\,(\mathbf{w}^\top\mathbf{x}_i + b) - 1 = 0
\qquad\text{per } \mathbf{x}_i \text{ sul margine.}
$$

Il valore $1$ non è una costante fisica: è la normalizzazione che spende il
grado di libertà residuo sulla scala di $(\mathbf{w}, b)$. Con quella fissata,
alla coppia resta un solo grado di libertà da determinare, ed è la lunghezza di
$\mathbf{w}$: il terzo passo mostra che è proprio lei a misurare la strada.

`````

### Terzo passo: quanto è larga la strada

Adesso la domanda vera. Sappiamo scrivere i vincoli, ma non sappiamo ancora
misurare la quantità che vogliamo rendere massima.

`````{tab} Elementare

Prendiamo una casa che tocca il marciapiede di sinistra e una che tocca quello
di destra, e congiungiamole con una freccia. Quella freccia **non** è la
larghezza della strada, perché va di sghembo: parte in un punto della strada e
arriva in un altro, e quindi contiene sia l'attraversamento sia un pezzo di
cammino lungo la strada, che a noi non interessa.

Ma sappiamo già come buttare via il pezzo che non interessa: l'ombra. Facciamo
fare a quella freccia l'ombra sulla direzione perpendicolare alla strada, e
quello che resta è esattamente la larghezza, come mostra
{numref}`fig-svm-larghezza`. Una accortezza sola: la freccia $\mathbf{w}$ ci
serve qui come direzione e non come lunghezza, quindi se ne fa una **copia**
lunga esattamente $1$, dividendone tutte le coordinate per la lunghezza
dell'originale. Una freccia di lunghezza $1$ indica una direzione e basta;
l'originale però resta dov'è, e fra due righe torna utile proprio con la sua
lunghezza.

Il conto è di due righe e sta nella scheda accanto, ma il risultato si può
raccontare, perché è sorprendente. Viene fuori che la larghezza della strada
vale sempre $2$ diviso la lunghezza della freccia $\mathbf{w}$ di partenza,
quella che non abbiamo accorciato. Sempre: delle
due case, che erano il punto di partenza, non resta traccia. Il $2$ non è un
numero magico ma la conseguenza di come abbiamo fissato l'asticella al passo
precedente, cioè a $+1$ da una parte e $-1$ dall'altra: fra i due c'è appunto
una distanza di $2$, ed è quella che riemerge qui.

E allora il problema, che era «trova la strada più larga», è diventato: **rendi
$\mathbf{w}$ più corta che puoi**, senza infrangere le regole del secondo passo.
Sono quelle regole a impedire la risposta stupida (una freccia lunga zero, cioè
nessuna frontiera).

`````

`````{tab} Superiore

Siano $\mathbf{x}_+$ un esempio positivo sul proprio marciapiede e
$\mathbf{x}_-$ un esempio negativo sul suo. La larghezza della strada è la
proiezione della loro differenza sul versore normale $\mathbf{w}/\lVert
\mathbf{w}\rVert$:

$$
\text{larghezza}
= (\mathbf{x}_+ - \mathbf{x}_-)^\top \frac{\mathbf{w}}{\lVert \mathbf{w}\rVert}.
$$

Sembra dipendere dai due punti scelti, e invece no, perché quei due punti stanno
sul margine e lì vale l'uguaglianza del secondo passo. Per $\mathbf{x}_+$ (con
$y=+1$) si ha $\mathbf{w}^\top\mathbf{x}_+ = 1 - b$; per $\mathbf{x}_-$ (con
$y=-1$) il vincolo $-(\mathbf{w}^\top\mathbf{x}_- + b) = 1$ dà
$\mathbf{w}^\top\mathbf{x}_- = -1 - b$. Sostituendo, il termine $b$ si elide:

$$
(\mathbf{x}_+ - \mathbf{x}_-)^\top\mathbf{w} = (1-b) - (-1-b) = 2,
\qquad\text{quindi}\qquad
\text{larghezza} = \frac{2}{\lVert \mathbf{w}\rVert}.
$$

È la formula che avevamo enunciato senza prova. Massimizzare
$2/\lVert\mathbf{w}\rVert$ equivale a massimizzare $1/\lVert\mathbf{w}\rVert$, e
quindi a minimizzare $\lVert\mathbf{w}\rVert$, e infine, per pura comodità,

$$
\min_{\mathbf{w},\,b}\ \tfrac{1}{2}\lVert \mathbf{w}\rVert^2
\qquad\text{con}\qquad
y_i(\mathbf{w}^\top\mathbf{x}_i + b) - 1 \ge 0 \ \ \forall i .
$$

Il quadrato toglie di mezzo la radice quadrata nascosta nella norma e rende la
funzione differenziabile ovunque, compresa l'origine; il fattore $1/2$ serve solo
a far sparire il $2$ quando deriveremo. Nessuna delle due mosse cambia il punto
di minimo, perché elevare al quadrato è monotono sui numeri non negativi. Terzo
pezzo sistemato.

`````

```{figure} ../figures/svm-larghezza-strada.svg
:name: fig-svm-larghezza
:alt: Una strada inclinata delimitata da due marciapiedi tratteggiati e dalla mediana continua. Un cerchio teal sul marciapiede superiore, x meno, e un quadrato terracotta su quello inferiore, x più, sono uniti da una freccia obliqua etichettata x più meno x meno. Da x meno scende un segmento a doppia freccia perpendicolare ai marciapiedi, etichettato due su norma di w, che arriva al marciapiede inferiore ed è la larghezza della strada. Dal piede di quel segmento un tratto grigio spesso corre lungo il marciapiede fino a x più, ed è la componente che la proiezione scarta.
:width: 90%

La larghezza della strada, ricavata in due righe. La freccia che unisce i due
esempi appoggiati ai marciapiedi va di sghembo, e si scompone in due pezzi:
quello che attraversa la strada, che vale esattamente
$2/\lVert\mathbf{w}\rVert$ (cioè $2$ diviso la lunghezza di $\mathbf{w}$), e
quello che corre lungo la strada, in grigio, che l'ombra butta via.
```

```{admonition} Il primo caffè
:class: tip
È il punto della lezione in cui Winston si ferma. Facciamo una pausa? Andiamo a
prendere un caffè? Peccato che qui non si possa, dice alla classe, ma se si
potesse lo faremmo. E aggiunge la frase per cui vale la pena raccontare tutto
questo: «sono sicuro che quando Vapnik è arrivato a questo punto, è uscito a
prendere un caffè».

Sembra una battuta buttata lì per far respirare l'aula, e in parte lo è. Ma è
anche l'unico modo onesto di segnalare una cosa che le derivazioni scritte
nascondono sistematicamente: qui finisce un pezzo e ne comincia un altro, e fra
i due c'è un salto che nessuna riga di algebra spiega. Abbiamo una funzione da
minimizzare e dei vincoli da rispettare, e la mossa successiva non discende da
quella precedente. Bisogna averla vista da un'altra parte, o inventarla.

Teniamo il conto: caffè numero uno.
```

### Quarto passo: Lagrange, e la prima sorpresa

`````{tab} Elementare

Il minimo di una funzione libera si sa trovare da tre secoli: si cerca il punto
dove la pendenza si annulla, perché sul fondo di una conca il terreno è piatto.
Ma noi liberi non siamo: ci sono i paletti del secondo passo, e il fondo della
conca cade fuori dal recinto. Il minimo che ci interessa sta *appoggiato* a un
paletto, e lì il terreno non è piatto per niente.

La ricetta per uscirne ha più di due secoli e porta il nome di Joseph-Louis Lagrange,
che era nato a Torino nel 1736 e si chiamava Giuseppe Lodovico Lagrangia. L'idea
è di trasformare i divieti in prezzi. A ogni paletto si attacca un numero,
$\alpha_i$ (si legge «alfa i-esimo», e la $i$ dice soltanto di quale paletto
stiamo parlando), e alla quota del terreno si somma quanto ciascun paletto fa
pagare a chi lo tocca. Nella funzione nuova, quota più pedaggi, i paletti non
compaiono più.

E il recinto? Non c'è più nemmeno lui, ed è esattamente il punto: la mossa non
serve a impedirci fisicamente di uscire, serve a **rendere sconveniente**
uscire. Fuori dal recinto il pedaggio cresce a dismisura, tanto da mangiarsi
qualunque guadagno di quota; e allora possiamo cercare il minimo come se si
fosse liberi di andare dove si vuole, sicuri che il minimo cadrà dentro. Un
problema con le regole è diventato un problema senza regole, ed è tutto quello
che serviva.

Restano da scegliere i prezzi, e la cosa notevole è che si scelgono da soli:
sono i valori che rendono quel pedaggio il più caro possibile, e per ogni
paletto la matematica dice qual è. Il caso che ci interessa è quello dei
paletti che non stiamo nemmeno sfiorando: quelli non hanno nessuna ragione di
farci pagare qualcosa, e il loro prezzo viene **zero**.
Vale la pena tenere a mente questa frase, perché fra poco diventa la
dimostrazione del fatto che i vettori di supporto sono pochi.

Fatto questo, cerchiamo dove la pendenza si annulla, e succede la prima cosa non
prevedibile. Viene fuori che la freccia $\mathbf{w}$, cioè la frontiera stessa,
si ottiene **sommando fra loro le case**, ciascuna moltiplicata per il proprio
prezzo. (Sommare due case vuol dire sommare i loro elenchi di numeri, coordinata
per coordinata: come si sommano due frecce mettendole in fila una dopo l'altra.)
Non doveva andare
così: da conti di questo tipo può uscire di tutto, e spesso esce qualcosa di
intrattabile. Invece è venuta fuori una somma, e questo vuol dire che la
frontiera non è un oggetto estraneo appoggiato sui dati: è fatta dei dati. E
siccome molti prezzi valgono zero, è fatta di **pochi** dati.

`````

`````{tab} Superiore

Si costruisce la **lagrangiana**, associando a ogni vincolo un moltiplicatore
$\alpha_i \ge 0$:

$$
\mathcal{L}(\mathbf{w}, b, \alpha)
= \tfrac{1}{2}\lVert \mathbf{w}\rVert^2
- \sum_{i=1}^{m} \alpha_i\Big[\,y_i(\mathbf{w}^\top\mathbf{x}_i + b) - 1\,\Big],
$$

dove il primo termine è la quantità da minimizzare, la parentesi quadra è
esattamente il vincolo del secondo passo, quello che vale zero sui punti di
margine, e $\alpha_i$ è il moltiplicatore associato al vincolo $i$-esimo. Un
avvertimento sul simbolo, perché qui il libro fa un'eccezione: in queste pagine
$\mathcal{L}$ è la **lagrangiana**, non la funzione di costo che la stessa
lettera indica ovunque altrove. È la notazione consolidata di questa
derivazione, e vale fino alla fine della sezione. Il segno meno, con
$\alpha_i \ge 0$, non è arbitrario: se un vincolo è
violato la parentesi diventa negativa, il termine $-\alpha_i[\,\cdot\,]$ diventa
positivo e si può far crescere quanto si vuole alzando $\alpha_i$, quindi la
violazione si paga; se invece il vincolo è rispettato con margine, il valore di
$\alpha_i$ che conviene è **zero**. La sparsità è già lì, in nuce.

Ora si annullano le derivate. Rispetto a $\mathbf{w}$, che è un vettore, si
deriva componente per componente e il risultato ha la stessa forma del caso
scalare (è la convenzione di layout dichiarata nel capitolo di matematica):

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}}
= \mathbf{w} - \sum_{i=1}^{m}\alpha_i y_i \mathbf{x}_i = \mathbf{0}
\qquad\Longrightarrow\qquad
\mathbf{w} = \sum_{i=1}^{m}\alpha_i y_i \mathbf{x}_i .
$$

Rispetto a $b$, l'unico termine che lo contiene è quello dentro la sommatoria:

$$
\frac{\partial \mathcal{L}}{\partial b}
= -\sum_{i=1}^{m}\alpha_i y_i = 0
\qquad\Longrightarrow\qquad
\sum_{i=1}^{m}\alpha_i y_i = 0 .
$$

La prima delle due è la sorpresa, e conviene dirla per esteso: il vettore dei
pesi $\mathbf{w}$ è una **combinazione lineare degli esempi di addestramento**,
con coefficienti $\alpha_i y_i$. Guardando il problema di partenza non c'era
nulla che lo annunciasse, e invece la soluzione vive nello **spazio generato
dagli esempi**: la frontiera si scrive con i dati e con nient'altro. La seconda
equazione, $\sum_i \alpha_i y_i = 0$, sembra per ora solo una condizione di
bilanciamento fra le due classi; fra un attimo cancellerà da sola un termine
intero.

`````

E qui, alla lavagna, Winston si concede il secondo caffè. Poi aggiunge la frase
che dà senso anche al primo: «a proposito, queste pause caffè durano mesi». Si
sta lì a guardare il risultato, si lavora ad altro, ci si preoccupa degli esami,
ogni tanto ci si ripensa. Prima o poi si torna dal caffè e si fa il passo
successivo. Caffè numero due: quanto siano durate davvero, quelle due pause, lo
dice l'ultima parte della sezione.

### Quinto passo: il duale, e la seconda sorpresa

Abbiamo scoperto come è fatta la freccia $\mathbf{w}$. La mossa che resta è
ovvia e faticosa: rimettere quella scoperta dentro la funzione «quota più
pedaggi» del passo precedente, e vedere che aspetto prende il problema quando
$\mathbf{w}$ non compare più.

`````{tab} Elementare

È solo algebra, e la scheda accanto la svolge per intero: sostituzioni e
raccoglimenti, niente idee nuove. Quello che conta è il
risultato, che è la seconda cosa non prevedibile del percorso.

Dopo la sostituzione, dei dati non restano né le coordinate né le distanze dalla
frontiera. Resta una cosa sola: per **ogni coppia** di case, l'ombra dell'una
sulla direzione dell'altra. Il problema da risolvere e la regola per classificare
un punto nuovo si scrivono entrambi usando soltanto quelle ombre a due a due.

Detto altrimenti: della mappa del quartiere si può buttare via tutto, tenendo
solo una tabella che per ogni coppia di case dice quanto si «vedono». Con quella
tabella si costruisce la frontiera, e senza la mappa. È un fatto che al momento
sembra soltanto elegante; è invece la porta di quello che viene fra poco in
questa stessa pagina, perché se i dati entrano **solo** attraverso quelle
ombre, allora nessuno ci vieta di cambiare il modo di calcolarle.

`````

`````{tab} Superiore

Sostituiamo $\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$ nei tre pezzi della
lagrangiana. Il termine quadratico diventa

$$
\tfrac{1}{2}\lVert\mathbf{w}\rVert^2
= \tfrac{1}{2}\Big(\sum_i \alpha_i y_i \mathbf{x}_i\Big)^{\!\top}
        \Big(\sum_j \alpha_j y_j \mathbf{x}_j\Big)
= \tfrac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j\,
  \mathbf{x}_i^\top\mathbf{x}_j ,
$$

il termine $\sum_i \alpha_i y_i \mathbf{w}^\top\mathbf{x}_i$ è la stessa doppia
somma **senza** il fattore $1/2$, e il termine
$\sum_i \alpha_i y_i b = b\sum_i \alpha_i y_i$ si annulla per la seconda
condizione del quarto passo. Rimane il $+\sum_i\alpha_i$ che veniva dal $-1$
dentro la parentesi quadra. Sommando i primi due (uno con $1/2$, l'altro intero,
di segno opposto) resta metà del secondo, cambiata di segno:

$$
\max_{\alpha}\ \mathcal{L}(\alpha)
= \sum_{i=1}^{m}\alpha_i
- \tfrac{1}{2}\sum_{i,j}\alpha_i\alpha_j\, y_i y_j\,
  \mathbf{x}_i^\top\mathbf{x}_j
\qquad\text{con}\qquad
\alpha_i \ge 0,\ \ \sum_{i=1}^{m}\alpha_i y_i = 0 .
$$

È il **problema duale**: non contiene più né $\mathbf{w}$ né $b$, solo gli $m$
moltiplicatori. Resta una programmazione quadratica, ma con un dettaglio che
vale tutta la fatica. Riscriviamo anche la regola di decisione del primo passo
sostituendoci $\mathbf{w}$:

$$
\sum_{i=1}^{m}\alpha_i y_i\, \mathbf{x}_i^\top\mathbf{u} + b \ \ge\ 0
\;\Longrightarrow\; \text{classe } +1 .
$$

Ecco il dettaglio: in **entrambe** le formule gli esempi compaiono soltanto
dentro un prodotto scalare, $\mathbf{x}_i^\top\mathbf{x}_j$ nel problema da
ottimizzare e $\mathbf{x}_i^\top\mathbf{u}$ nella regola di decisione. Non
servono le coordinate, non serve la dimensione dello spazio, non serve nemmeno
sapere che cosa siano gli $\mathbf{x}_i$: serve una tabella di prodotti scalari.
Da questa osservazione, e da nient'altro, nasce il kernel trick.

Due note a margine. La prima: la funzione obiettivo del duale è **concava** e
il dominio è convesso, quindi ogni massimo locale è anche globale e non ci sono
ottimi locali in cui restare intrappolati, al contrario di quanto succede
addestrando una rete neurale. La seconda: passando al
margine morbido cambia una riga sola, il vincolo $\alpha_i \ge 0$ diventa
$0 \le \alpha_i \le C$, cioè il parametro $C$ mette un **tetto** a quanto può
spingere un singolo punto. Tutto il resto, kernel compreso, resta identico.

`````

### Perché i vettori di supporto sono pochi

Che i punti che contano siano pochi lo abbiamo detto fin dalla prima pagina, e
finora era una promessa. Adesso si dimostra in due righe, e discende dal
percorso appena fatto: non è un fatto osservato provando, è una conseguenza.

`````{tab} Elementare

Torniamo alla frase più sorprendente della sezione: le case lontane dal confine
non contano nulla, e potresti cancellarle senza spostarlo di un millimetro.
Adesso sappiamo perché.

Immagina il confine come un muro elastico teso fra le due parti, e ogni casa
come una mano che spinge il muro per allontanarlo da sé. Le case addossate al
corridoio spingono davvero: se ne togli una, il muro scivola. Le case arretrate
non toccano il muro, e la loro spinta è **esattamente zero**. Sono i prezzi
$\alpha_i$ del quarto passo: chi non tocca il proprio paletto non paga pedaggio,
e il suo prezzo è zero. Ma $\mathbf{w}$, l'abbiamo appena scoperto, è la somma
delle case *pesata con quei prezzi*, e zero moltiplicato per qualunque cosa resta
zero: nella soluzione, il contributo delle case arretrate non c'è proprio.

È il motivo per cui la SVM, alla fine, si porta appresso solo i pochi punti che
spingono e può dimenticare tutti gli altri, anche se erano un milione.

`````

`````{tab} Superiore

Le condizioni di Karush–Kuhn–Tucker, che completano il metodo di Lagrange nel
caso di vincoli di disuguaglianza, impongono la **complementarità**: per ogni
$i$, il prodotto fra il moltiplicatore e il proprio vincolo è nullo,

$$
\alpha_i\Big[\,y_i(\mathbf{w}^\top\mathbf{x}_i + b) - 1 + \xi_i\,\Big] = 0 ,
$$

(qui già nella forma a margine morbido, con la variabile di slack $\xi_i$; per
il margine rigido basta porre $\xi_i = 0$). Se un punto è strettamente fuori dal
margine, la parentesi quadra è diversa da zero, e allora deve essere
$\alpha_i = 0$: quel punto sparisce dalla somma
$\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$ che ricostruisce la soluzione.
Restano solo i punti *sul* margine o dentro la fascia, cioè i **vettori di
supporto**.

Sui quattro punti dell'esempio numerico i conti si chiudono in una riga. La
condizione $\sum_i \alpha_i y_i = 0$ e la ricostruzione di $\mathbf{w}$ danno
$\alpha_1 = \alpha_2 = 0{,}25$ e $\alpha_3 = \alpha_4 = 0$, da cui

$$
\mathbf{w} = 0{,}25\,(2,2) = (0{,}5;\ 0{,}5),
$$

esattamente la soluzione trovata per via geometrica. Come controprova vale
l'identità $\sum_i \alpha_i = \lVert\mathbf{w}\rVert^2$, che discende
dall'uguaglianza fra primale e duale all'ottimo: qui $0{,}25 + 0{,}25 = 0{,}5$ e
$\lVert\mathbf{w}\rVert^2 = 0{,}25 + 0{,}25 = 0{,}5$, e il margine
$2/\lVert\mathbf{w}\rVert = 2/\sqrt{0{,}5} \approx 2{,}83$ torna con il conto
svolto per via geometrica all'inizio di questa pagina.

`````

### Quanto dura davvero una pausa caffè

Il passo successivo, nella storia vera, arrivò trent'anni dopo.

Vapnik aveva scritto l'idea della strada più larga nella sua tesi a Mosca,
all'inizio degli anni Sessanta, e per un pezzo non se ne fece niente. Non perché
l'idea fosse debole, racconta Winston: perché non c'erano macchine su cui
provarla. Negli anni successivi, sempre con Chervonenkis, costruì la teoria che
stabilisce *quando* un modello che ha imparato bene sugli esempi visti
continuerà a funzionare su quelli nuovi {cite}`vapnik1971uniform`: è ciò che
oggi si chiama teoria dell'apprendimento statistico, e la sua misura di
complessità, la dimensione VC, porta le iniziali dei due. In Occidente, per
vent'anni, quel lavoro non lo lesse quasi nessuno.

Nel 1990 Vapnik emigra negli Stati Uniti e finisce ai laboratori Bell di
Holmdel, nel New Jersey, dove si lavorava al riconoscimento delle cifre scritte
a mano. È lì che, nel 1992, nasce l'idea del paragrafo che segue: si prende la
tabella delle ombre a due a due appena trovata e si cambia il modo di
riempirla, ed è il **kernel** {cite}`boser1992training`. Winston fa notare per
inciso il vantaggio di
studiare cose fatte da gente ancora viva: a Fourier non si può telefonare per
chiedergli come gli sia venuta, a Vapnik sì. E il seguito lo racconta così. Gli
articoli mandati alla conferenza NIPS quell'anno furono respinti tutti. Vapnik
aveva un'opinione bassissima delle reti neurali, e scommise una cena con un
collega che le SVM le avrebbero battute sulla scrittura a mano. Fu il collega a
mettersi alla prova: usò un kernel appena appena curvo (un polinomio di grado
due, cioè la più timida delle frontiere non dritte) e funzionò al primo colpo,
facendo vincere la cena a chi l'aveva sfidato. A Napoleone si attribuisce l'osservazione che un
soldato si batte a lungo e con ferocia per un pezzetto di nastro colorato; ecco,
commenta Winston, questo è il pezzetto di nastro, ed era una cena.

Il kernel Vapnik ce l'aveva già nella tesi. Non aveva mai pensato che fosse
importante, e furono i risultati sulle cifre a fargli cambiare idea. Fra il
momento in cui aveva capito i kernel e il momento in cui ne capì l'importanza
passarono trent'anni.

È qui che la battuta del caffè smette di essere una battuta. La derivazione che
abbiamo appena percorso sta in una pagina e si legge in un quarto d'ora; le due
pause in mezzo, per chi la faceva per la prima volta, sono durate una carriera.
Le grandi idee, chiude Winston, sono seguite da lunghi periodi in cui non
succede niente, e poi da un momento in cui l'idea di partenza si rivela
potentissima con appena mezzo giro di vite. E da lì in avanti il mondo non si
volta più indietro.

## Il kernel trick: separare l'inseparabile

Fin qui, però, la SVM traccia solo frontiere *diritte*. E se le due classi sono
intrecciate in modo che nessuna retta le separi? Pensa a un
bersaglio, con una classe al centro e l'altra tutt'intorno ad anello: nessuna
riga tirata su quel foglio le divide. Qui
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

Il guaio è che sollevare i punti costa. Nei casi utili le dimensioni da
aggiungere non sono una ma migliaia, a volte infinite, e nessun calcolatore
può reggerle. E qui sta il trucco: il passo precedente ci ha lasciato una
frontiera che si calcola **usando soltanto le ombre a due a due**, cioè un
numero per ogni coppia di punti. Se sappiamo produrre direttamente quei numeri
*come sarebbero dopo il sollevamento*, il sollevamento non serve più farlo.

La regola che li produce si chiama **kernel**, ed è tutto ciò che serve. Si
sceglie il kernel, e lo spazio sollevato resta un'idea: non lo si costruisce
mai.

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

Il **kernel trick** è l'osservazione che salva tutto, ed è la ragione per cui
valeva la pena percorrere il duale passo per passo: là dentro, e nella regola
di decisione, gli esempi compaiono *solo* attraverso prodotti scalari
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
il parametro di ampiezza del kernel gaussiano, che **stringe** la campana al
crescere. La larghezza è la deviazione standard equivalente $\sigma$, e vale
$\sigma = 1/\sqrt{2\gamma}$, cioè $\gamma = 1/(2\sigma^2)$: $\gamma$ è quindi
l'inverso del *quadrato* della larghezza, e per dimezzare la campana va
quadruplicato, non raddoppiato. $\gamma$ grande, campana stretta. Il kernel RBF
corrisponde a uno spazio $\phi$ di dimensione *infinita*: sarebbe impossibile
da costruire, eppure $k$ si calcola in una riga.

Un'ultima clausola, quella che fa del trucco un teorema invece che una
speranza. La frase «se esiste una funzione $k$ che calcola quel prodotto
scalare» rovescia l'ordine dei fatti: in pratica non si parte da $\phi$ per
cercare $k$, si **sceglie** $k$ e si spera che un $\phi$ esista. Esiste se e
solo se $k$ è simmetrica e **semidefinita positiva**, cioè se ogni matrice di
Gram $\mathbf{K}$, quella di elementi $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$,
ha autovalori non negativi: è il
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

Il kernel trick. A sinistra due classi che nessuna retta separa: un disco al
centro e un anello intorno. A destra gli stessi punti dopo il sollevamento, con
l'altezza pari alla distanza dal centro elevata al quadrato: i punti del disco
restano in basso, quelli dell'anello salgono, e una retta orizzontale basta a
dividerli.
```

Come illustra {numref}`fig-svm-kernel`, ciò che era un anello inseparabile
diventa, dopo il sollevamento, un problema lineare banale.

Fra i kernel c'è un preferito, e si chiama **RBF** (sono le iniziali di *radial
basis function*, «funzione a base radiale»: radiale perché guarda solo la
distanza fra due punti, in qualunque direzione). Ha una manopola sola, che nelle
formule si chiama $\gamma$, «gamma», e vale la pena capire che cosa fa, perché è
il **raggio d'influenza** di ogni punto.

`````{tab} Elementare

Il kernel RBF si racconta meglio con i lampioni. Pensa a ogni punto come a un
lampione acceso di notte: illumina bene chi
gli sta accanto, sempre meno chi si allontana, per niente chi è lontano, e il
numero che il kernel restituisce per due punti è quanta luce dell'uno arriva
all'altro. Sta fra $0$ e $1$, e vale $1$ solo per il lampione stesso.

Chiamiamo *portata* del lampione la distanza alla quale la
luce è scesa a poco più di un terzo, cioè a $0{,}37$: mettiamo un metro e mezzo.
Chi sta a un metro e mezzo si vede ancora. E chi sta al doppio, a tre metri?
Non riceve la metà della luce, e nemmeno un terzo. Il motivo è che a decidere
non è la distanza ma il suo **quadrato**, e raddoppiando la distanza il
quadrato si moltiplica per quattro: è come se quel lampione, per lui, fosse
lontano quattro portate invece di una. La luce che gli arriva è quindi $0{,}37$
elevato alla quarta, cioè circa $0{,}018$: meno di due
centesimi, praticamente buio. Ecco perché il raggio d'influenza di un punto
finisce così bruscamente.

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
il nuovo) e dell’**anomaly detection** (riconoscere il guasto), e si lega a
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
separa i punti dall’**origine** con il massimo margine. Ricondotto allo spazio
originale, questo equivale a racchiudere i dati normali in una regione
compatta; ciò che cade fuori è novità/anomalia. Il parametro $\nu \in (0,1]$
ha un doppio significato preciso: è un limite *superiore* alla frazione di
esempi di addestramento classificati come anomali (i *margin error*) e un
limite *inferiore* alla frazione di vettori di supporto. La distingue dalla
classificazione binaria un'assenza: in addestramento **non** c'è la classe
«anomalo»: si impara solo la forma del normale. Un parente stretto è la
**Support Vector Data Description** (SVDD) di Tax e Duin, che invece della
separazione dall'origine cerca la *ipersfera* minima che racchiude i dati; e
tra le alternative non-kernel ci sono l’**Isolation Forest** (che isola le
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
`SGDClassifier`, che scalano circa come $O(m)$) o sugli alberi in boosting
della sezione sugli ensemble {cite}`geron2022hands`.

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
# Fabbrichiamone uno: una sinusoide della prima coordinata, con un po’ di rumore.
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

```text
anomalie segnalate: 104
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
- La **strada più larga** si ricava in cinque passi, e due volte lungo il
  percorso salta fuori qualcosa che nessuno aveva chiesto: che la frontiera è
  fatta dei dati stessi, sommati con un peso, e che i dati entrano nel conto
  **solo a coppie**, ognuno attraverso l'ombra che fa sull'altro. È la seconda
  cosa ad aprire la porta al kernel trick.
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
- La derivazione («l'approccio della strada più larga») va dalla regola di
  decisione $\mathbf{w}^\top\mathbf{u}+b\ge 0$ ai vincoli
  $y_i(\mathbf{w}^\top\mathbf{x}_i+b)-1\ge 0$, da lì a
  $\text{larghezza}=2/\lVert\mathbf{w}\rVert$ per proiezione di
  $\mathbf{x}_+-\mathbf{x}_-$ sulla normale, e infine, via Lagrange, a
  $\mathbf{w}=\sum_i\alpha_iy_i\mathbf{x}_i$ e $\sum_i\alpha_iy_i=0$. Sostituendo
  si ottiene il **duale**
  $\sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_jy_iy_j\,\mathbf{x}_i^\top\mathbf{x}_j$,
  concavo, in cui gli esempi compaiono **solo** dentro prodotti scalari: da qui,
  e da nient'altro, il kernel trick.
- La soluzione dipende solo dai **vettori di supporto**, e non è
  un'osservazione ma un corollario: la complementarità KKT
  $\alpha_i[y_i(\mathbf{w}^\top\mathbf{x}_i+b)-1+\xi_i]=0$ annulla $\alpha_i$
  per ogni punto fuori dal margine, e $\mathbf{w}=\sum_i\alpha_iy_i\mathbf{x}_i$
  non lo contiene.
- Il **margine morbido** ammette violazioni $\xi_i$ pagate dal parametro $C$,
  l’**inverso** della forza di regolarizzazione: $C$ grande → margine stretto
  (overfitting), $C$ piccolo → margine largo. La perdita è la **hinge loss**,
  parente della log-loss ma piatta oltre il margine.
- Il **kernel trick** rende non lineare la SVM: mappa i dati in uno spazio più
  ampio dove diventano separabili, calcolando i prodotti scalari con un
  **kernel** $k(\mathbf{x},\mathbf{z})=\phi(\mathbf{x})^\top\phi(\mathbf{z})$
  senza costruirlo: funziona perché nel **duale** gli esempi compaiono solo
  dentro prodotti scalari, e vale se e solo se $k$ è simmetrica e semidefinita
  positiva (Mercer). Kernel principali: lineare, polinomiale, RBF, dove
  $\gamma$ **stringe** la campana al crescere ($\gamma = 1/(2\sigma^2)$).
- La **SVR** regredisce con un tubo $\epsilon$-insensitive; la **one-class SVM**
  ($\nu$ = frazione di anomalie attese) impara la regione dei dati normali per
  la **novelty/anomaly detection**, senza vedere esempi anomali.
- In pratica: **standardizzare sempre** le feature; il costo $O(m^2)$–$O(m^3)$
  sconsiglia la SVM con kernel oltre le decine di migliaia di esempi.
```

`````
