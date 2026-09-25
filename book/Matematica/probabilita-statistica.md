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

### Contare i casi

«Casi favorevoli su casi possibili» funziona finché i casi si lasciano contare,
e contarli è meno ovvio di quanto sembri. Il problema della posta interrotta,
da cui era partita la corrispondenza fra Pascal e Fermat, si risolve così.
Fermat contò tutti i modi in cui sarebbero potuti andare i turni rimasti,
giocandoli fino in fondo anche quando la partita sarebbe già stata decisa, e
Pascal ne scrisse la soluzione generale con il triangolo aritmetico, quello che
in Italia si chiama di Tartaglia. Le regole per contare formano il **calcolo
combinatorio**, e il numero che ne esce più spesso ha un simbolo suo, il
**coefficiente binomiale** $\binom{n}{k}$: quanti gruppi diversi di $k$
oggetti si formano da $n$, senza badare all'ordine. Serve ogni volta che si
contano coppie, squadre o gruppi: quante coppie si formano con mille esempi, in
quanti modi una giuria di sette può trovarsi in maggioranza dalla parte giusta,
che è il conto dietro il {doc}`teorema della giuria di Condorcet
</SistemiMultiAgente/protocolli-e-consenso>`.

`````{tab} Elementare

Dieci amici devono scegliere un capitano e un vice. Per il capitano ci sono
dieci possibilità, e per ciascuna ne restano nove per il vice: $10\cdot 9 = 90$
coppie capitano-vice. È la regola che regge tutto il resto: quando si sceglie
una cosa dopo l'altra, le possibilità si moltiplicano.

Se invece devono formare una coppia per il doppio a ping-pong, dove nessuno
comanda, il conto di prima conta ogni coppia due volte. Anna capitano con Bruno
vice e Bruno capitano con Anna vice, al tavolo, sono la stessa coppia. Le coppie
sono quindi $90/2 = 45$.

Con una squadra di tre i doppioni sono di più. Scegliendoli in fila si hanno
$10\cdot 9\cdot 8 = 720$ terne ordinate, ma ogni squadra compare tante volte
quanti sono i modi di mettere in fila i suoi tre membri: tre per il primo posto,
due per il secondo, uno per l'ultimo, $3\cdot 2\cdot 1 = 6$ (Anna, Bruno, Carlo
è la stessa squadra di Anna, Carlo, Bruno, di Bruno, Anna, Carlo, e così via).
Le squadre sono $720/6 = 120$. Il prodotto $3\cdot 2\cdot 1$ si scrive $3!$ e si
legge «tre fattoriale»; il numero delle squadre si scrive $\binom{10}{3}$ e si
legge «dieci su tre», anche se non è una divisione. (Scegliere i tre che giocano
è come scegliere i sette che restano in panchina, e infatti anche le panchine da
sette sono $120$.)

Si può contare anche in un altro modo, ed è un buon controllo. Anna o è nella
squadra, o non c'è. Se c'è, restano da scegliere due compagni fra gli altri
nove, $9\cdot 8/2 = 36$ squadre; se non c'è, servono tre persone fra quei nove,
$9\cdot 8\cdot 7/6 = 84$. E $36 + 84 = 120$.

Adesso ciascuno lancia una moneta. Le file possibili di testa e croce sono
$2^{10} = 1024$ (due possibilità per ciascuno, moltiplicate dieci volte), tutte
ugualmente probabili se le monete sono oneste. Quante hanno esattamente tre
teste? Tante quante i modi di scegliere *chi* fa testa, cioè ancora $120$. La
probabilità è $120/1024$, circa l’$11{,}7\%$: il conto delle squadre e quello
delle monete sono lo stesso conto.

La ricetta però vale solo se i casi contati sono ugualmente probabili, e
sbagliare è facile. Nel 1754 Jean-Baptiste Le Rond d'Alembert, uno dei
matematici più celebri del suo secolo, sostenne che lanciando due volte una
moneta la probabilità di vedere almeno una testa fosse $2/3$. Pensava al gioco
che si ferma appena esce testa: i casi, diceva, sono tre (testa al primo lancio,
e il secondo non serve; croce e poi testa; due croci), e due su tre vanno bene.
Ma quei tre casi non si equivalgono, perché «testa subito» raccoglie due file,
testa-testa e testa-croce: il secondo lancio, anche se non lo si fa, sarebbe
potuto andare in due modi. Contando le file, che sono quattro e ugualmente
probabili, quelle con almeno una testa sono tre, e la probabilità è $3/4$.
Pascal e Fermat, un secolo prima, avevano evitato proprio questo errore: per
dividere la posta giocavano sulla carta anche i turni che non servivano più.

E i casi crescono in fretta. I modi di mettere in fila i dieci amici sono già
$10! = 3\,628\,800$, e quelli di mescolare un mazzo da 52 carte formano un
numero di sessantotto cifre. Quando i casi sono troppi per elencarli, se ne
guarda una parte estratta a caso, un campione, e si stima: lo si vedrà fare, per
esempio, per dividere il merito di una previsione fra le informazioni su cui si
è basata, nelle {doc}`spiegazioni locali
</Interpretabilita/spiegazioni-locali>`.

`````

`````{tab} Superiore

Il **principio del prodotto** dice che se una prima scelta si fa in $a$ modi e,
per ciascuno, una seconda in $b$ modi, le coppie sono $ab$ anche quando le
seconde scelte possibili dipendono dalla prima: se $A$ è l'insieme delle prime e
$B_x$ quello delle seconde dopo la scelta $x$, con $|A| = a$ e $|B_x| = b$, le
coppie sono $\sum_{x\in A}|B_x| = ab$; quando $B_x = B$ per ogni $x$ è la
cardinalità del prodotto cartesiano, $|A\times B| = |A|\,|B|$. Iterandolo, le
sequenze di $k$ elementi distinti presi da $n$ (le *disposizioni semplici*, per
$0\le k\le n$) sono

$$
n(n-1)\cdots(n-k+1) = \frac{n!}{(n-k)!},
$$

con $n! = n(n-1)\cdots 1$ e $0! = 1$ (un solo modo di mettere in fila zero
oggetti, ed è la scelta che fa valere la formula anche per $k = n$); per $k = n$
sono le $n!$ permutazioni. Se invece lo stesso elemento si può riprendere, le
sequenze sono $n^k$, le *disposizioni con ripetizione*: le file di $n$ lanci di
una moneta sono $2^n$. Ogni sottoinsieme di taglia $k$ corrisponde a esattamente
$k!$ disposizioni, i suoi ordinamenti, quindi i sottoinsiemi sono

$$
\binom{n}{k} = \frac{n!}{k!\,(n-k)!},
$$

simmetrico per $k \leftrightarrow n-k$ (scegliere chi entra è scegliere chi
resta fuori). Vale la **regola di Pascal**, quella del triangolo (ogni numero è
la somma dei due che gli stanno sopra),

$$
\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k},
$$

valida per $n\ge 1$ con la convenzione $\binom{n}{k} = 0$ fuori da
$0\le k\le n$, e con una dimostrazione biiettiva che non chiede conti: fissato
un elemento, i sottoinsiemi di taglia $k$ si dividono in quelli che lo
contengono (e scelgono gli altri $k-1$ fra i restanti $n-1$) e quelli che non lo
contengono (e li scelgono tutti e $k$ fra i restanti). Il nome viene dal teorema
del binomio, $(x+y)^n = \sum_{k=0}^{n}\binom{n}{k}x^k y^{n-k}$, dove il
coefficiente di $x^k y^{n-k}$ conta da quali $k$ degli $n$ fattori si prende la
$x$; con $x = y = 1$ dà $\sum_k \binom{n}{k} = 2^n$, il numero dei sottoinsiemi.
Le scelte in cui conta solo quante volte compare ciascun elemento, le
*combinazioni con ripetizione*, sono $\binom{n+k-1}{k}$ ($k$ segni e $n-1$
separatori messi in fila, e si sceglie dove stanno i segni): tanti quanti i
monomi di grado $k$ in $n$ variabili, cioè le coordinate di grado $k$ dello
spazio che il {doc}`kernel polinomiale </MachineLearning/svm-kernel>` usa senza
calcolarle.

Per $n$ lanci indipendenti di una moneta onesta le $2^n$ sequenze sono
equiprobabili, e quelle con $k$ teste corrispondono una a una ai sottoinsiemi
di $k$ lanci, quelli che fanno testa: $P(k \text{ teste}) = \binom{n}{k}/2^n$.
Con una moneta che dà testa con probabilità $p$ e lanci indipendenti, ogni
sequenza con $k$ teste ha probabilità $p^k(1-p)^{n-k}$, e sommando sulle
$\binom{n}{k}$ sequenze il numero $X$ di teste ha

$$
P(X = k) = \binom{n}{k}\,p^k (1-p)^{n-k}, \qquad k = 0,\dots,n,
$$

che è la **distribuzione binomiale**: il teorema del binomio con $x = p$ e
$y = 1-p$ la normalizza a uno. È la distribuzione del conteggio di successi su
cui poggiano il test binomiale esatto e l'intervallo di Clopper-Pearson.

Il punto debole della definizione classica, $P(A) = |A|/|\Omega|$, è l'ipotesi
che gli esiti elementari siano equiprobabili, e da come si sceglie lo spazio
dipende il risultato. D'Alembert, nella voce «Croix ou pile» dell’*Encyclopédie*
(1754), diede $2/3$ alla probabilità di almeno una testa in due lanci perché
prese come equiprobabili gli esiti del gioco che si ferma alla prima testa ($T$,
$CT$, $CC$) invece delle quattro sequenze complete; la risposta è $3/4$. È
l'obiezione che Roberval aveva mosso al metodo di Fermat, e che Pascal riferisce
nella lettera del 24 agosto 1654: la partita vera si ferma appena qualcuno ha
vinto. Giocare anche i turni che non servono non cambia il vincitore, e rende
equiprobabili le sequenze. Si conta su uno spazio in cui la simmetria del
meccanismo rende gli esiti equiprobabili, e solo dopo si passa alle probabilità.

Resta la crescita. La formula di Stirling, $n! \sim \sqrt{2\pi n}\,(n/e)^n$
(il rapporto fra i due lati tende a $1$, e a $n = 52$ la sottostima è appena
dello $0{,}16\%$), dà per il coefficiente centrale $\binom{2m}{m} \sim
4^m/\sqrt{\pi m}$: con $n = 2m$, i soli sottoinsiemi di taglia $n/2$ sono $2^n$
diviso appena $\sqrt{\pi n/2}$. Enumerare i sottoinsiemi di $n$ elementi costa
dunque $2^n$ anche fermandosi a una taglia sola, e smette di essere praticabile
già per qualche decina di elementi. Le grandezze definite come somme su tutti i
sottoinsiemi o su tutti gli ordinamenti, come i valori di Shapley delle
{doc}`spiegazioni locali </Interpretabilita/spiegazioni-locali>`, si calcolano
quindi esatte solo per $n$ piccolo, e altrimenti si stimano campionando
sottoinsiemi o ordinamenti.

`````

I conti delle squadre, delle monete e di d'Alembert si possono rifare senza
formule, elencando i casi uno per uno. L'elenco però non sceglie lo spazio: a
partire dai tre casi di d'Alembert restituisce di nuovo $2/3$. A dire quali casi
si equivalgono è il gioco stesso, e lo si interroga giocandolo: su centomila
partite simulate, fermandosi alla prima testa, almeno una testa esce tre volte
su quattro. Il blocco confronta anche $52!$ con la formula approssimata di
Stirling, che lo stima senza fare tutte le moltiplicazioni.

```python
import random
from itertools import combinations, permutations, product
from math import comb, e, factorial, pi, sqrt

amici = "ABCDEFGHIL"                      # dieci amici, "A" è Anna
print("capitano e vice:", len(list(permutations(amici, 2))))
print("coppie per il doppio:", len(list(combinations(amici, 2))))
print("terne ordinate:", len(list(permutations(amici, 3))),
      "| squadre da tre:", len(list(combinations(amici, 3))), "=", comb(10, 3),
      "| panchine da sette:", len(list(combinations(amici, 7))))
con_anna = sum("A" in s for s in combinations(amici, 3))
senza_anna = sum("A" not in s for s in combinations(amici, 3))
print("squadre con Anna:", con_anna, "=", comb(9, 2),
      "| senza Anna:", senza_anna, "=", comb(9, 3))

file = list(product("TC", repeat=10))    # tutte le file di dieci lanci
tre_teste = sum(f.count("T") == 3 for f in file)
print(f"file di dieci lanci: {len(file)}, con tre teste: {tre_teste}"
      f" ({tre_teste / len(file):.1%})")

# d'Alembert: l'elenco restituisce la risposta dello spazio che riceve
due = list(product("TC", repeat=2))
print("almeno una testa, file complete:",
      sum("T" in f for f in due), "su", len(due))
print("almeno una testa, casi di d'Alembert:",
      sum("T" in c for c in ("T", "CT", "CC")), "su 3")
# il gioco vero: si lancia finché esce testa, al più due volte
sorte = random.Random(0)
partite = 100_000
vinte = sum(sorte.random() < 0.5 or sorte.random() < 0.5
            for _ in range(partite))
print(f"in {partite} partite, almeno una testa nel {vinte / partite:.0%}")

print("10! =", factorial(10))
n = 52
stirling = sqrt(2 * pi * n) * (n / e)**n
print(f"52! ha {len(str(factorial(n)))} cifre;"
      f" Stirling la sottostima dello {1 - stirling / factorial(n):.2%}")
```

```text
capitano e vice: 90
coppie per il doppio: 45
terne ordinate: 720 | squadre da tre: 120 = 120 | panchine da sette: 120
squadre con Anna: 36 = 36 | senza Anna: 84 = 84
file di dieci lanci: 1024, con tre teste: 120 (11.7%)
almeno una testa, file complete: 3 su 4
almeno una testa, casi di d'Alembert: 2 su 3
in 100000 partite, almeno una testa nel 75%
10! = 3628800
52! ha 68 cifre; Stirling la sottostima dello 0.16%
```

## Variabili aleatorie

Lanciare dieci monete e guardare come sono venute è una cosa; contare quante
teste sono uscite è un'altra, ed è quasi sempre quella che interessa. Il salto
è piccolo e cambia tutto: da un elenco di casi si passa a un numero, e sui
numeri si possono fare medie, grafici, confronti.

`````{tab} Elementare

Una **variabile aleatoria** è un numero il cui valore dipende dal caso. Se lancio
dieci monete e conto le teste, quel conteggio è una variabile aleatoria. Ne
esistono due specie. Quelle discrete contano cose separate ($0, 1, 2, \dots$
teste): a ciascun valore assegniamo una probabilità, e la somma fa $1$. Quelle
continue misurano grandezze che scorrono senza salti (un'altezza, un tempo di
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
tutti e due.

```{figure} ../figures/variabili-aleatorie-momenti-percentili.svg
:name: fig-densita-percentili
:alt: "Due grafici affiancati sulla stessa distribuzione asimmetrica. A sinistra la densità di probabilità, intitolata PDF, con media e mediana segnate da due linee tratteggiate e la coda oltre il novantacinquesimo percentile riempita di colore. A destra la funzione di ripartizione, intitolata CDF, che sale da zero a uno, e su cui lo stesso percentile si legge entrando dall'altezza 0,95 e scendendo sull'asse orizzontale."
:width: 100%

La stessa variabile, due modi di guardarla. A sinistra la densità appena
descritta, che disegna quali valori escono spesso e quali di rado: lì la
probabilità è l’*area* sotto la curva, e infatti la zona colorata è il cinque
per cento dei casi che cadono oltre il valore segnato. A destra una seconda
curva, che per ogni valore risponde alla domanda «quanta parte dei casi sta
sotto questo numero?» e perciò sale da $0$ a $1$: lì la stessa probabilità è
l’*altezza* della curva, cioè un punto da leggere invece di un'area da
calcolare. Il nome tecnico della seconda è **funzione di ripartizione**. (Le
due sigle in cima ai grafici sono le abbreviazioni inglesi con cui si trovano
ovunque: *PDF* per la densità, *CDF* per la ripartizione.)
```

I due grafici di {numref}`fig-densita-percentili` si corrispondono punto per
punto, e la seconda curva esiste per rispondere alla domanda che si fa più
spesso su una misura: «sotto quale valore cade il novantacinque per cento dei
casi?». Quel valore si
chiama novantacinquesimo percentile, e i percentili si leggono sul grafico
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
appena visti. Quasi sempre non serve tutto
il quadro: bastano due numeri, dove sta il centro della distribuzione e quanto
è dispersa attorno a esso.

```{figure} ../figures/media-mediana-moda-varianza-numpy.svg
:name: fig-centro-larghezza
:alt: "Una distribuzione asimmetrica, con la coda allungata a destra, annotata con tre indicatori di centro che cadono in punti diversi: la moda sul picco, la mediana poco più a destra e la media ancora più a destra, tirata dalla coda. Sotto, un segmento marca la deviazione standard attorno alla media."
:width: 92%

Tre centri per la stessa distribuzione. La moda sta sul picco (il valore
più frequente), la mediana taglia i casi a metà, la media è quella
scolastica. Su una curva simmetrica coinciderebbero; qui no, perché la curva
ha una coda, cioè si allunga da un lato con valori rari ma molto grandi, e
la media è quella che la coda sposta di più: pochi stipendi altissimi alzano
lo stipendio medio senza spostare quello di mezzo.
```

L'ordine in cui i tre indicatori compaiono in {numref}`fig-centro-larghezza`
è un buon promemoria pratico: quando media e mediana si
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

La varianza, che si scrive $\mathrm{Var}(X)$ e si legge «varianza di X»,
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

Sul dado il conto fila liscio perché le facce e le loro probabilità si
conoscono in partenza, centro compreso. Sui dati veri il centro non lo conosce
nessuno: lo si ricava dagli stessi numeri di cui poi si misura lo
sparpagliamento, e così la misura esce un po’ più piccola del giusto. Tre
freccette su un bersaglio: se come centro prendi il punto di mezzo dei tuoi
tre tiri, quel punto insegue i tiri, e gli scarti misurati da lì vengono, nel
complesso, più corti di quelli dal centro vero del bersaglio. Il rimedio è
dividere la somma degli scarti al quadrato per uno in meno, per due invece che
per tre: con pochi dati la differenza si sente, con migliaia sparisce.

`````

`````{tab} Superiore

Per una variabile discreta, con $\mu=\mathbb{E}[X]$:

$$
\mathbb{E}[X]=\sum_x x\,p(x), \qquad
\mathrm{Var}(X)=\mathbb{E}\big[(X-\mu)^2\big]=\mathbb{E}[X^2]-\mu^2 ;
$$

per una continua le somme diventano integrali. La deviazione standard
$\sigma=\sqrt{\mathrm{Var}(X)}$ riporta la dispersione nelle stesse unità di $X$.
Il valore atteso è lineare, $\mathbb{E}[aX+b]=a\,\mathbb{E}[X]+b$, proprietà
che useremo di continuo. Per il dado:
$\mathbb{E}[X^2]=\tfrac{91}{6}\approx 15{,}17$, quindi
$\mathrm{Var}(X)=15{,}17-3{,}5^2\approx 2{,}92$ e $\sigma\approx 1{,}71$.

Una distinzione che d'ora in poi è data per fatta. $\mathrm{Var}(X)$ è
una proprietà della *distribuzione*, e nei conti di sopra la si calcola perché
la distribuzione è nota. Sui dati veri non lo è: la si stima da un
campione $x_1,\dots,x_n$, ed è un'altra cosa, che si scrive $s^2$ e non
$\mathrm{Var}(X)$. La media degli scarti quadratici
$\frac{1}{n}\sum_i (x_i-\bar x)^2$ sottostima sistematicamente $\sigma^2$,
perché $\bar x$ è stata calcolata sugli stessi dati e si adagia su di essi;
dividere per $n-1$ (**correzione di Bessel**) rende lo stimatore non distorto:

$$
s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2 .
$$

Il linguaggio è quello degli **stimatori**: una funzione $\hat\theta$ del
campione che serve a indovinare un parametro $\theta$ della distribuzione. La
sua distorsione è
$\operatorname{bias}(\hat\theta)=\mathbb{E}[\hat\theta]-\theta$, la sua
varianza dice quanto cambia da un campione all'altro, e l'errore quadratico
medio si spezza nelle due,
$\mathbb{E}[(\hat\theta-\theta)^2]=\operatorname{bias}(\hat\theta)^2+\mathrm{Var}(\hat\theta)$.
Per campioni i.i.d. il conto di Bessel sta in una riga: da $\sum_i(x_i-\bar
x)^2=\sum_i(x_i-\mu)^2-n(\bar x-\mu)^2$ segue $\mathbb{E}\big[\sum_i(x_i-\bar
x)^2\big]=n\sigma^2-n\,\mathrm{Var}(\bar x)=(n-1)\sigma^2$. Due avvertenze. Non
distorto non vuol dire migliore: su dati gaussiani il divisore $n$ ha errore
quadratico medio più piccolo di $n-1$, e $n+1$ più piccolo ancora. E la non
distorsione non attraversa le funzioni non lineari: $s=\sqrt{s^2}$ sottostima
$\sigma$, per la disuguaglianza di Jensen.

Il divisore $n$ è quello della massima verosimiglianza gaussiana (il criterio
che sceglie i parametri che rendono i dati più probabili): annullando le
derivate di
$-\frac{n}{2}\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}\sum_i(x_i-\mu)^2$ si
trovano $\hat\mu=\bar x$ e $\hat\sigma^2=\frac1n\sum_i(x_i-\bar x)^2$. La
massima verosimiglianza, dunque, non promette stimatori non distorti. Le
librerie scelgono default diversi, e conviene saperlo prima di confrontare due
numeri: `np.var` e `StandardScaler` di scikit-learn dividono per $n$
(`ddof=0`), `torch.var` divide per $n-1$ (`correction=1`). Su otto osservazioni
la differenza è del $14\%$; su diecimila è invisibile.

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

La normale, che si chiama anche gaussiana dal nome del matematico Carl
Friedrich Gauss (i due nomi indicano la stessa identica cosa e si usano l'uno
per l'altro), è la celebre curva a campana ({numref}`fig-curva-normale`):
simmetrica, con la maggior parte dei valori stretti attorno alla media $\mu$ e
code che si assottigliano ai lati. Vale una regola pratica utilissima, la
"68–95": circa il $68\%$ dei casi cade entro una deviazione standard dalla media
($\mu\pm\sigma$), circa il $95\%$ entro due ($\mu\pm 2\sigma$). Altezze, errori di
misura, rumore: in natura la campana è dappertutto.

Perché dappertutto? C'è un risultato che lo spiega, e ha un nome importante: il
**teorema del limite centrale**. Dice una cosa sorprendente: ogni volta che una
grandezza è la *somma* di tanti contributi casuali e indipendenti fra loro,
nessuno dei quali possa da solo essere enormemente più grande di tutti gli
altri messi insieme, il risultato assomiglia a una campana, e questo succede
qualunque sia la forma dei singoli contributi. L'altezza di una persona è la
somma di centinaia di piccoli effetti (geni, alimentazione, salute da bambino):
campana. L'errore di uno strumento è la somma di tante imprecisioni minuscole:
campana. Il totale di tre dadi è la somma di tre numeri che, presi da soli, non
hanno niente di campanulare (in un dado tutte le facce sono ugualmente
probabili, il disegno sarebbe piatto): eppure il totale, sì.

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
standard (la sua larghezza). Perché è così onnipresente? Per il teorema del
limite centrale (de Moivre, Laplace, poi Lyapunov): se $X_1,\dots,X_n$ sono
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
schemi standard derivano la varianza dell'inizializzazione, per conservare
la scala delle attivazioni fra uno strato e l'altro, e non la famiglia, tanto
che il default di PyTorch campiona da un'uniforme e non da una normale.)

Il teorema dice che si converge, non quanto in fretta, e la seconda domanda ha
una risposta separata. Una garanzia grossolana la dà la disuguaglianza di
Berry–Esseen, $\sup_z |F_n(z)-\Phi(z)| \le C\,\rho_3 / (\sigma^3\sqrt{n})$,
dove $F_n$ è la funzione di ripartizione della somma standardizzata, $\Phi$
quella della normale standard, $C$ una costante universale minore di mezzo che
non dipende dalla distribuzione di partenza, e $\rho_3 = \mathbb{E}|X-\mu|^3$.
La distanza cala dunque come $1/\sqrt{n}$. Davanti, però, c'è un momento terzo
assoluto, che non distingue una sola coda lunga da due code lunghe simmetriche.
A separarle è lo sviluppo di Edgeworth, il cui primo termine correttivo è
proporzionale all’asimmetria e si annulla quando la distribuzione di partenza è
simmetrica: è lei, a parità di tutto il resto, a governare il termine
dominante.

La distanza fra due distribuzioni si misura con la statistica di
Kolmogorov–Smirnov, applicata qui alla somma standardizzata contro la
normale. Partendo da un dado uniforme (asimmetria nulla) la distanza è già
$0{,}069$ con $n=3$, e il conto è esatto perché i $216$ esiti si enumerano
tutti; partendo da una lognormale$(0;\,2)$, asimmetrica e continua,
resta $0{,}27$ a $n=30$, su quattrocentomila somme simulate. Il confronto con
una Bernoulli$(0{,}01)$, che a $n=30$ dà $0{,}45$, non è dello stesso tipo: lì
il numero non misura l'asimmetria ma la discretezza, perché la somma mette
il $74\%$ della massa su un valore solo e quel salto, da solo, vale $0{,}449$.
Il caso «piatto» resta il più gentile di tutti, non il più ostile.

`````

Conviene vedere il teorema del limite centrale succedere, perché detto a
parole sembra una promessa e guardato sembra un trucco.

```{figure} ../figures/limite-centrale.gif
:name: fig-limite-centrale
:alt: "Animazione: a sinistra le sei facce di un dado, tutte della stessa altezza; a destra un istogramma delle somme di tre dadi che si riempie lotto dopo lotto e assume la forma di una campana, su cui compare la curva normale prevista dalla teoria."
:width: 90%

Un dado è piatto: nessuna faccia è più probabile di un'altra. Eppure la
somma di tre dadi, ripetuta seicento volte, si dispone da sé lungo la campana.
La curva sovrapposta è la campana che il teorema prevede prima ancora di
tirare i dadi, e non un disegno fatto sopra le barre a cose fatte: è centrata
sulla somma media dei tre e larga di conseguenza, e i dadi le danno ragione.
```

Due cose valgono più della formula, nella {numref}`fig-limite-centrale`. La
prima è che la distribuzione di partenza non ha niente della campana:
nessuna faccia è privilegiata, il disegno di un dado singolo sarebbe una
fila di barre tutte uguali, e la campana arriva lo stesso. È questo il senso
di «qualunque sia la forma dei singoli contributi».

La seconda va detta con più cautela di quanto si faccia di solito. Qui bastano
tre addendi, ma il merito è del dado più che del teorema: è simmetrico,
cioè non ha una coda più lunga dell'altra. È proprio la simmetria a far
arrivare in fretta la campana, ed è per questo che il caso più «piatto» è anche
il più facile.

Lo stesso fenomeno si vede con le monete, ed è il caso che ricorre più spesso
nel machine learning: contare i successi in una serie di prove «sì/no»
indipendenti.

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
gruppo di esempi tenuti da parte, mai visti durante l'addestramento (il test
set), e si conta la percentuale di risposte esatte: quella percentuale si
chiama **accuratezza**. Ma un conteggio di risposte esatte su prove
indipendenti è la stessa cosa del conteggio di teste appena disegnato, quindi
lo si può trattare come una campana. È l'approssimazione su cui poggiano gli
intervalli di confidenza.

## Aggiornare le credenze: il teorema di Bayes

Fin qui abbiamo calcolato probabilità "in avanti". Ma spesso il problema è
rovesciato: osserviamo un effetto e vogliamo risalire alla causa. È il regno di
Thomas Bayes, il cui saggio fu pubblicato postumo nel 1763.

`````{tab} Elementare

Un test cerca una malattia rara, che colpisce $1$ persona su $100$. Il
test è buono: individua il $99\%$ dei malati e sbaglia solo nel $5\%$ dei sani. Ti
arriva un risultato positivo: quanto devi preoccuparti? L'istinto dice
"molto", ma i conti dicono altro. Su $10\,000$ persone, $100$ sono malate (e $99$
risultano positive); ma dei $9\,900$ sani ben $495$ risultano positivi *per
errore*. I positivi totali sono $99+495=594$, e solo $99$ sono davvero malati:
circa il 17%. Con una malattia rara, la maggior parte dei positivi sono falsi
allarmi.

`````

`````{tab} Superiore

Per $P(B)>0$ la probabilità di $A$ dato $B$ è $P(A\mid B)=P(A\cap B)/P(B)$,
la probabilità ricalcolata dentro il mondo ristretto a $B$. Due eventi sono
**indipendenti** se $P(A\cap B)=P(A)\,P(B)$, cioè se sapere che $B$ è
successo non sposta $A$; per le variabili aleatorie si chiede la stessa
fattorizzazione per ogni coppia di valori, e «i.i.d.» vuol dire indipendenti
e con la stessa distribuzione. Se $B_1,\dots,B_k$ partizionano lo spazio degli
esiti, vale la legge della probabilità totale
$P(A)=\sum_j P(A\mid B_j)\,P(B_j)$. Scrivendo $P(A\cap B)$ nei due modi si
ottiene il teorema di Bayes, che lega la probabilità condizionata nei due
versi:

$$
P(A\mid B)=\frac{P(B\mid A)\,P(A)}{P(B)} .
$$

Con $A$ = «malato» e $B$ = «positivo», prevalenza $P(A)=0{,}01$,
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
l'esempio di prima: a cambiare sono stati i malati. È la ragione per cui la stessa
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
numero di prove, ma alla sua radice quadrata. Il conto è quello: con cento
prove l'incertezza vale un certo tanto, e per dimezzarla non basta arrivare a
duecento, bisogna arrivare a quattrocento, perché quello che deve raddoppiare è
la radice, e la radice di quattrocento è il doppio della radice di cento
($20$ contro $10$). Quattro volte i dati per metà dell'errore, e la tassa si
paga ogni volta che si misura qualcosa su un campione.

`````{tab} Elementare

Se ripeti tante volte lo stesso esperimento casuale, la media delle
osservazioni si avvicina sempre di più al valore atteso. Su $10$ lanci di una
moneta equa, $7$ teste non stupiscono nessuno. Su $10\,000$ lanci, il $70\%$
di teste è talmente improbabile che concluderesti (ragionevolmente) che la
moneta è truccata.

Attenzione a un equivoco diffuso: il caso non si corregge, si diluisce.
Dopo dieci teste di fila la moneta non "deve" croce, il lancio successivo resta
50 e 50. Quello che succede è che le dieci teste iniziali pesano sempre meno
man mano che i lanci si accumulano, finché diventano irrilevanti nella media.

Due condizioni reggono la garanzia, e conviene tenerle distinte perché si
rompono in modi diversi. La prima è che le osservazioni siano
indipendenti, cioè che nessuna influenzi le altre: se le prime recensioni
di un ristorante sono entusiaste, chi scrive dopo le ha lette e si adegua, e
mille voti così non valgono mille pareri raccolti separatamente. La seconda è
che vengano tutte dalla stessa distribuzione, cioè che si stia misurando
sempre la stessa cosa: mille recensioni scritte dagli amici del ristoratore
misurano l'amicizia, non la cucina, e sommarle a quelle dei clienti fa una
media di due cose diverse. Se salta l'una o l'altra, la garanzia non vale più,
ed è la ragione per cui una raccolta di dati fatta male non migliora
aggiungendone altra raccolta allo stesso modo.

`````

`````{tab} Superiore

Siano $X_1,\dots,X_n$ variabili i.i.d. con media $\mu=\mathbb{E}[X]$ finita. La
media campionaria $\bar{X}_n=\frac{1}{n}\sum_i X_i$ converge a $\mu$: in
probabilità (legge debole) e quasi certamente (legge forte).

La velocità è il punto che interessa il machine learning. Se
$\mathrm{Var}(X)=\sigma^2$, allora

$$
\mathrm{Var}(\bar{X}_n)=\frac{\sigma^2}{n},
\qquad
\text{deviazione standard} = \frac{\sigma}{\sqrt{n}} .
$$

L'errore cala come $1/\sqrt{n}$, non come $1/n$: per dimezzare l'incertezza
servono quattro volte i dati. Vale per qualunque grandezza che si stimi come
media di osservazioni indipendenti, e quindi per ogni metrica che sia una media
sugli esempi di un test set, come l'accuratezza. L'errore di un modello
addestrato segue invece curve di apprendimento proprie, con esponenti che
dipendono dal compito e in genere sono più piccoli di $1/2$
{cite}`kaplan2020scaling`, anche se la tendenza ai rendimenti decrescenti è la
stessa.

L'ipotesi i.i.d. è quella che si rompe più spesso nella pratica: dati correlati
nel tempo, esempi duplicati, campioni raccolti da una sola fonte. Quando salta,
$n$ conta molto meno di quanto dica il conteggio delle righe.

`````

## Quanto fidarsi di una stima: gli intervalli di confidenza

Un modello nuovo raggiunge l’$87{,}2\%$ di accuratezza sul test set, il vecchio
si fermava all’$86{,}8\%$. Il team festeggia. Ma se il test set ha $500$
esempi, quattro decimi di punto sono due risposte esatte in più. Due.

```{figure} ../figures/intervalli-di-confidenza.svg
:name: fig-intervalli-confidenza
:alt: "Tre modelli su un asse dell'accuratezza da 72 a 92 per cento, ciascuno con la sua stima puntuale e una barra d'errore. Il modello A sta all'87,2% e il modello B all'86,8%: le loro barre si sovrappongono quasi per intero, e una nota dice che con intervalli sovrapposti non c'è un vincitore. Il modello C sta al 78,0%, con la barra molto più in basso, che con le altre due non si sovrappone."
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

Un istituto intervista mille persone e trova il candidato A al $47{,}2\%$.
Nessun giornale serio pubblica quel numero da solo, perché accanto ci va il
margine d'errore, e con mille intervistati quel $47{,}2\%$ vuol dire «da
qualche parte fra il $44\%$ e il $50\%$». L'istituto non ha contato i voti di
tutti. Ne ha presi mille a caso, cioè un **campione**, e da quelli ha ricavato
una stima del numero vero, che si saprà solo la sera dello spoglio.

Provare un modello su $500$ esempi è sondare $500$ elettori. L’$87\%$ di
risposte esatte del modello nuovo è il suo sondaggio, e l'elezione (come se la
caverebbe su tutti i casi possibili) nessuno la vedrà mai.

Il margine, un sondaggista se lo calcola a mente in due mosse. La prima vale
per un candidato al $50\%$, dove gli intervistati sono più divisi e la stima
balla di più, e dà circa $100/\sqrt{n}$ punti percentuali con $n$ intervistati;
$\sqrt{500}$ è poco più di $22$, quindi circa $4{,}5$ punti. La seconda
accorcia quel numero, perché un candidato lontano dal $50\%$ fa ballare meno la
stima (quando quasi tutti la pensano allo stesso modo, due campioni diversi si
somigliano). Il conto si fa in tre passi, con la quota dell’$87\%$. Si
moltiplica la quota per la quota opposta: $0{,}87 \times 0{,}13 = 0{,}113$. Se
ne fa la radice: $0{,}34$. Si divide per $0{,}5$, che è quanto lo stesso conto
dà al $50\%$: $0{,}34$ diviso $0{,}5$ fa circa due terzi. All’$80\%$ viene
$0{,}8$, al $93\%$ circa la metà. Quindi $4{,}5$ punti moltiplicati per due
terzi, e il margine è di tre punti.

Ecco la stima onesta del modello nuovo, fra l’$84\%$ e il $90\%$. Quella del
modello vecchio va dall’$83{,}8\%$ all’$89{,}8\%$. I due intervalli si
sovrappongono quasi per intero, e quei quattro decimi di punto di vantaggio
sono **rumore**, il nome che si dà alla parte di un risultato che viene dal
caso e non dal merito.

Il conto con la radice vale per i sondaggi normali. In un paesino di poche
decine di elettori, o quando un candidato è dato al $99\%$ o all’$1\%$, quel
conto sbaglia, e chi fa sondaggi passa a formule fatte apposta.

Anche il $95\%$ stampato accanto al margine parla dell'istituto e non della
singola tornata di telefonate. Il risultato dell'urna è già deciso, e o sta
dentro quel margine o non ci sta. Il $95\%$ conta quante volte l'istituto ci
prende lavorando così: di venti sondaggi, diciannove contengono il valore vero
e uno lo manca, e quale sia lo sbagliato non lo dice nessuno. Quella volta su
venti è messa in conto.

Per stringere il margine si intervista più gente, e ne serve molta, perché
sotto c'è una radice quadrata. Un test set di $5\,000$ esempi porta a circa
$\pm 1$ punto, uno di $50\,000$ a $\pm 0{,}3$.

Quando due candidati restano a pari merito, l'istituto torna dalle stesse
persone e chiede a ciascuna quale dei due preferisce. Chi li apprezza entrambi,
o non sopporta né l'uno né l'altro, non sposta niente, e la partita si gioca su
chi li divide. Due modelli provati sugli stessi $500$ esempi rispondono uguale
quasi ovunque, e a dividerli sono gli esempi in cui uno risponde giusto e
l'altro no. Se quei casi si spartiscono quasi a metà, il pari merito è
confermato. Se uno la spunta quasi sempre, il vantaggio è reale anche con i
margini sovrapposti, perché due margini messi a confronto sono prudenti per
costruzione e qualche differenza vera la lasciano in ombra.

`````

`````{tab} Superiore

Per una proporzione $\hat{p}$ stimata su $n$ prove indipendenti, l'errore
standard è $\sqrt{\hat{p}(1-\hat{p})/n}$ e l'intervallo di Wald al livello
$95\%$ è

$$
\hat{p} \pm 1{,}96\,\sqrt{\frac{\hat{p}(1-\hat{p})}{n}} .
$$

Con $\hat{p}=0{,}872$ e $n=500$: errore standard $\approx 1{,}49$ punti,
margine $\approx 2{,}93$ punti, intervallo $[84{,}2\%,\ 90{,}2\%]$.

Cosa significa davvero "95%". Non che il valore vero abbia il $95\%$ di
probabilità di stare in *questo* intervallo: il valore vero è un numero fisso,
o ci sta o non ci sta. Significa che la *procedura*, ripetuta su molti campioni,
produce intervalli che contengono il valore vero nel $95\%$ dei casi. È una
proprietà del metodo, non di questo singolo risultato.

Due avvertenze pratiche. L'intervallo di Wald è inaffidabile con $n$ piccolo o
$\hat{p}$ vicino a $0$ o $1$: lì si usano Wilson o Clopper–Pearson. E quando si
confrontano due modelli sullo stesso test set gli errori sono appaiati: il
test corretto è quello di McNemar sui disaccordi, non il confronto fra due
intervalli, che è conservativo e può nascondere differenze reali.

`````

## Escludere il caso: l'ipotesi nulla e il p-value

Un intervallo di confidenza sa dire «non lo so», ed è già molto. Ma prima o poi
qualcuno chiede una risposta secca: questo modello è meglio dell'altro, sì o no?
Questa serie di dati è cambiata rispetto al mese scorso, sì o no? Una risposta
secca si può dare, a patto di dichiarare quanto spesso si è disposti a
sbagliarla.

`````{tab} Elementare

Cento lanci di una moneta, sessanta teste. È truccata?

La mossa che la statistica fa, e che sorprende chi la incontra la prima volta,
è cominciare dalla risposta che si vorrebbe scartare. Si suppone la moneta
onesta, e si guarda quanto sarebbe strano un sessanta contro quaranta se lo
fosse davvero. Quella supposizione si chiama **ipotesi nulla**. Non si scrive
perché ci si creda: si scrive perché è l'unica su cui si sappiano fare i conti.
«Onesta» dice esattamente che numeri aspettarsi; «truccata» non dice niente,
perché truccata al cinquantacinque per cento e truccata al novanta sono due
mondi diversi.

Il conto a mente si può fare, con la regola 68–95. Su cento lanci di una
moneta onesta le teste si accumulano attorno a cinquanta, e la larghezza
tipica di quella campana è la radice di un quarto dei lanci: un quarto di
cento fa venticinque, la cui radice è cinque teste. (È lo stesso conto del
margine dei sondaggi: cento diviso la radice di cento fa dieci punti, che là
erano due larghezze, cioè il margine al $95\%$, mentre qui la larghezza è una
sola.) Sessanta sta dunque due larghezze sopra il centro, e fuori da due
larghezze si finisce circa cinque volte su cento. «Circa cinque su cento» però
non basta a decidere, perché la soglia è proprio cinque su cento, e la campana
è un'approssimazione di un conto che si può fare esatto, contando quante file
di cento testa-o-croce hanno sessanta teste o più. Fatto esatto: uno
sbilanciamento di sessanta e più (o di quaranta e meno) capita a una moneta
onesta $57$ volte su mille, cioè $0{,}057$. La regola a mente dava il paese
giusto, la decisione la prende la cifra. Quel numero, cioè quanto spesso il
caso da solo produrrebbe una stranezza almeno pari a quella vista, si chiama
$p$, o $p$-value. Attenzione alla lettera: questa $p$ è una proprietà della
*prova* appena fatta, e non ha niente a che vedere con la $p$ con cui poco fa
si indicava la probabilità che esca testa, che è una proprietà *della moneta*.

Si contano anche i quaranta e meno perché la domanda era «è truccata», e non «è
truccata a favore di testa». Chi decide il verso dopo aver visto il risultato ha
due possibilità di gridare al trucco invece di una, e quelle possibilità in più
se le prende anche quando il trucco non c'è.

Chi la prova sul serio decide prima di lanciare quanto piccolo debba essere
$p$ perché si smetta di credere all'ipotesi nulla. Per convenzione cinque su
cento, e quel numero non dice quanto è forte la prova: dice quanto spesso si
accetta di accusare una moneta onesta. Qui $p$ vale $0{,}057$, appena sopra la
soglia, e la moneta non viene accusata: per accusarla ci sarebbero volute
sessantuno teste.

Il verdetto ha due facce, e sono due errori diversi. Si può accusare una moneta
onesta, e quanto spesso capita lo si è deciso a tavolino. Oppure si può lasciar
passare una moneta truccata, e questo capita tanto più spesso quanto meno lanci
si sono fatti: con venti lanci e dodici teste $p$ vale $0{,}503$, cioè con una
prova così corta la stessa proporzione di teste non dice più niente. Non
accusare non è assolvere. E i due errori si scambiano: alzare l'asticella per
accusare vuol dire lasciar passare più monete truccate, e viceversa. Con la
stessa prova in mano, l'unico modo di stringerli tutti e due insieme è
lanciare di più.

E c'è un modo di leggere $p$ che è sbagliato, ed è quello che viene in mente per
primo: $p$ non è la probabilità che la moneta sia onesta. Per dire quella
servirebbe sapere quante monete truccate girano in giro, che è il passaggio del
teorema di Bayes visto prima. $p$ risponde a una domanda sola: se fosse
onesta, quanto spesso vedrei una cosa così?

`````

`````{tab} Superiore

Si fissa un'ipotesi nulla $H_0$ (la moneta è onesta, $\pi = 1/2$), un'alternativa
$H_1$, e una statistica $T$ calcolata sui dati. Il **$p$-value** è

$$
p = \Pr\big(T \text{ almeno estrema quanto } T_{\text{oss}} \;\big|\; H_0\big),
$$

dove $T_{\text{oss}}$ è il valore osservato e «almeno estrema» va nel verso
dell'alternativa (bilaterale, se conta lo sbilanciamento in entrambi i sensi).
Con $n = 100$ e $60$ successi, la binomiale esatta dà $p = 0{,}057$ (bilaterale
come raddoppio della coda minore, la convenzione più diffusa; la variante che
somma tutte le probabilità non superiori a quella osservata qui coincide, perché
la binomiale con $\pi = 1/2$ è simmetrica).

La discretezza costa qualcosa, e va detto perché si vede nei conti. I valori
ottenibili di $p$ sono un insieme finito, quindi il livello nominale $\alpha$
non è quasi mai raggiunto: con $n = 100$ il più piccolo rifiuto ammesso è a $61$
successi, e la taglia effettiva del test è $0{,}035$ invece di $0{,}05$. Un test
esatto su una statistica discreta è conservativo, e un conto di falsi
positivi fatto con $\alpha$ nominale ne prevede più di quanti ne arrivino.

Il livello $\alpha$ si fissa prima, e Neyman e Pearson
{cite}`neyman1933problem` gli danno il significato che ancora si usa: è la
frequenza con cui la procedura rifiuta $H_0$ quando $H_0$ è vera, cioè l'errore
di prima specie. L'errore di seconda specie $\beta$ è non rifiutare quando
$H_1$ è vera, e $1-\beta$ è la potenza. A test fissato i due si scambiano;
per stringerli insieme serve $n$.

La dualità con gli intervalli di confidenza vale a parità di famiglia:
rifiutare $H_0: \theta = \theta_0$ al livello $\alpha$ equivale a non trovare
$\theta_0$ dentro l'intervallo di confidenza al $1-\alpha$ costruito con lo
stesso metodo. L'intervallo di Wald e il test binomiale esatto non sono duali
fra loro, e chi li mescola trova casi in cui uno rifiuta e l'altro contiene, e
con $n = 100$ succede proprio alle sessanta teste da cui questo conto è
partito. Le coppie giuste: Clopper-Pearson con il test binomiale esatto, Wald
con il test $z$ che stima l'errore standard dalla proporzione osservata,
Wilson con il test $z$ che lo calcola sotto $H_0$. Test e intervallo dicono la
stessa cosa in due modi, e l'intervallo dice in più *quali* valori restano
compatibili.

Sull'interpretazione l'American Statistical Association ha ritenuto necessario
un comunicato {cite}`wasserstein2016asa`, sei principi. Il primo dice che cosa
il $p$-value fa: indica quanto i dati siano incompatibili con un modello
statistico specificato. Un altro dice che cosa non fa: non misura la
probabilità che l'ipotesi studiata sia vera, né la probabilità che i dati
siano stati prodotti dal solo caso. Un terzo non parla del $p$-value ma di chi
lo legge: nessuna conclusione dovrebbe reggersi soltanto sul fatto che un
$p$-value superi o non superi una soglia. Il divieto del secondo è il
passaggio da $\Pr(\text{dati} \mid H_0)$ a $\Pr(H_0 \mid \text{dati})$, che
senza una probabilità a priori non si fa.

Il punto di rottura è a monte del conto. Il $p$-value ha il significato dichiarato
solo se la statistica, il verso e la soglia sono stati scelti prima di
guardare i dati. Chi prova più definizioni di errore, più finestre temporali, più
sottogruppi e riporta la migliore ha misurato la propria ostinazione, ed è la
stessa porta da cui entra il guaio della molteplicità.

`````

### Mille domande insieme: Bonferroni e le false scoperte

Il guaio comincia quando la domanda non è una sola. Un programma che sorveglia
un modello al lavoro tiene d'occhio trecento grandezze e per ciascuna chiede «è
cambiata?»; uno studio di genetica misura ventimila geni; un banco di prova
confronta sessanta modi di regolare la stessa rete. La soglia che regge una
domanda non regge trecento.

`````{tab} Elementare

Una scatola con mille monete, novecento oneste e cento truccate, e da fuori non
si vede quali. Le truccate escono testa il sessantacinque per cento delle volte:
al sessanta, come si è appena visto, in cento lanci non si distinguerebbero. Le
si lancia cento volte ciascuna e si accusa quella che sbilancia troppo da una
parte o dall'altra, con la solita soglia di cinque su cento: da sessantuno
teste in su, o da trentanove in giù.

Il guaio si vede prima ancora di aprire la scatola. La soglia prometteva cinque
accuse ingiuste ogni cento monete oneste, che con novecento farebbero
quarantacinque; saranno una trentina, perché le teste sono un numero intero e
la soglia non si può centrare esattamente, ma trenta o quarantacinque il punto
non cambia. In un elenco di accusate lungo un centinaio, decine sono lì per
caso, e chi legge l'elenco non ha modo di sapere quali.

La prima riparazione alza l'asticella in proporzione al numero di domande: con
mille monete si chiede $0{,}05$ diviso mille, cioè cinque su centomila. Il conto
che la giustifica è di una riga: ogni moneta onesta viene accusata cinque volte
su centomila, le monete sono mille, e mille per cinque su centomila fa cinque
centesimi di accusa ingiusta a scatola. Cioè si accusa un innocente in una
scatola su venti: di nuovo cinque su cento, ma stavolta riferito all’intera
scatola invece che a ogni singola moneta. Si chiama correzione di
**Bonferroni**, ed è prudentissima. Il prezzo lo si vede subito: con
un'asticella così in alto smettono di essere accusate anche quasi tutte le
monete davvero truccate.

La seconda riparazione cambia la promessa, e si chiama **Benjamini-Hochberg**.
Invece di «quasi certamente non accuso nemmeno una moneta onesta», si promette
«fra le monete che accuso, in media non più del cinque per cento sono oneste»,
e anzi qualcosa meno, tanto meno quante più truccate ci sono nella scatola. In
media su tante scatole, perché su una singola scatola nessuno può garantire
niente; ed è una promessa più debole, che proprio per questo permette di
accusarne molte di più.

La ricetta si mette in una riga. Si allineano le mille monete dalla più
sbilanciata alla meno, e alla moneta che sta al posto numero $k$ si chiede la
soglia divisa per mille diviso $k$: alla prima la soglia divisa per mille,
come faceva Bonferroni; alla seconda divisa per cinquecento; alla decima
divisa per cento; alla centesima divisa per dieci. Si scende fino all'ultima
che ce la fa, si segna quel posto, e si accusano tutte le monete da lì in su.
Sì: nel gruppo finisce anche qualcuna che da sola non ce l'avrebbe fatta, ed è
voluto. Il nome della cosa che si tiene sotto controllo è **tasso di false
scoperte**, e «scoperta» è il nome che si dà a un'accusa quando il colpevole
non è una moneta ma un gene o un guasto.

Ed è anche la ragione per cui la promessa più debole permette di accusarne di
più. Bonferroni giudica ogni moneta da sola, come se fosse l'unica; qui una
moneta un po' sospetta viene creduta anche per la compagnia in cui si trova,
perché sopra di lei nella fila ce ne sono decine ancora più sbilanciate, e
decine di monete sbilanciate tutte insieme sono una cosa che il caso da solo non
produce.

Quale delle due serva dipende da che cosa succede dopo l'accusa. Se ogni
segnalazione fa scattare qualcosa di costoso, e una segnalazione sbagliata si
paga cara, la prudenza di Bonferroni è quella giusta. Se invece la segnalazione
apre soltanto un controllo, e di controlli se ne possono fare a decine, tenere
bassa la quota di controlli inutili serve molto più che evitarne uno a tutti i
costi.

`````

`````{tab} Superiore

Con $m$ test simultanei, sia $V$ il numero di ipotesi nulle vere rifiutate
(falsi positivi) e $R$ il numero totale di rifiuti. Le due grandezze da
controllare sono

$$
\mathrm{FWER} = \Pr(V \ge 1),
\qquad
\mathrm{FDR} = \mathbb{E}\!\left[\frac{V}{R}\right]
$$

(con $V/R$ posto a $0$ quando $R = 0$). La prima è la probabilità di sbagliare
almeno una volta in tutta la famiglia; la seconda è la quota attesa di
errori fra le scoperte annunciate.

La correzione di Bonferroni confronta ogni $p$-value con $\alpha/m$
{cite}`dunn1961multiple`. Che controlli la FWER segue dalla disuguaglianza di
Boole, $\Pr(\bigcup_i A_i) \le \sum_i \Pr(A_i)$, e quindi non chiede
indipendenza: è la sua forza e la ragione della sua prudenza. Una cosa la
chiede, e va detta perché è quella che si rompe più spesso: che ogni $p$-value
sia valido di per sé, cioè $\Pr(p_i \le t \mid H_0) \le t$. Con $p$-value
ottimisti Bonferroni fallisce anche senza nessuna dipendenza. Un fratello
maggiore gratuito esiste: la procedura di Holm ordina i $p$-value e allenta la
soglia man mano, controlla la stessa FWER sotto le stesse ipotesi e rifiuta
sempre almeno quanto Bonferroni.

La procedura di Benjamini e Hochberg {cite}`benjamini1995controlling`
controlla invece la FDR a un livello $q$: si ordinano i $p$-value
$p_{(1)} \le \dots \le p_{(m)}$, si cerca il più grande $k$ tale che

$$
p_{(k)} \le \frac{k}{m}\,q,
$$

e si rifiutano le prime $k$ ipotesi. Sotto indipendenza, e sotto la dipendenza
positiva che Benjamini e Yekutieli chiamano PRDS
{cite}`benjamini2001control`, la garanzia è $\mathrm{FDR} \le q\,m_0/m$, dove
$m_0$ è il numero di ipotesi nulle vere: con il $90\%$ di nulle vere e
$q = 0{,}05$, il limite effettivo è $4{,}5\%$. Fuori di lì la garanzia cade e
serve la versione conservativa $q/\sum_{j \le m} 1/j$. E come Bonferroni,
anche questa presuppone $p$-value validi: se sotto $H_0$ sono conservativi,
come lo sono i $p$-value binomiali delle monete, la FDR misurata resta sotto il
limite per quella ragione e non per merito della procedura.

Il punto di rottura sta nella parola *attesa*. La FDR è una media su ripetizioni:
sul singolo insieme di dati la quota di falsi fra le scoperte può essere ben
più alta di $q$, e nessuna procedura può garantire il contrario. La scelta fra
le due grandezze è quindi una scelta di mestiere: confermare una singola
affermazione chiede la FWER, esplorare ventimila geni chiede la FDR.

`````

I due $p$ delle monete singole e la differenza fra le tre strade si misurano
nello stesso blocco. Per la scatola: mille monete, cento delle quali escono
testa il $65\%$ delle volte e le altre novecento oneste, cento lanci a testa, e
il $p$ calcolato contando le combinazioni; il tutto ripetuto trecento volte con
scatole sorteggiate in modo diverso, perché una quota media si guarda su molte
prove.

```python
import numpy as np
from math import comb

M, TRUCCATE, LANCI = 1000, 100, 100
SBILANCIO, ALFA, PROVE = 0.65, 0.05, 300

def massa_binomiale(n):
    return [comb(n, i) / 2**n for i in range(n + 1)]

def tabella_p(n):
    """Quanto spesso una moneta onesta sbilancia almeno quanto k teste,
    contando tutti e due i versi."""
    massa = massa_binomiale(n)
    return [min(1.0, 2 * min(sum(massa[:k+1]), sum(massa[k:]))) for k in range(n + 1)]

tavola = tabella_p(LANCI)
massa = massa_binomiale(LANCI)
prima = next(k for k in range(LANCI // 2 + 1, LANCI + 1) if tavola[k] <= ALFA)
taglia = sum(massa[k] for k in range(LANCI + 1) if tavola[k] <= ALFA)
print(f"larghezza tipica della campana dei {LANCI} lanci: {(LANCI * 0.25)**0.5:.1f} teste")
print(f"p di 60 teste su 100 lanci: {tavola[60]:.3f}")
print(f"p di 12 teste su  20 lanci: {tabella_p(20)[12]:.3f}")
print(f"prima accusata a {prima} teste; soglia dichiarata {ALFA}, "
      f"quota vera di oneste accusate {taglia:.3f}")
print()

truccata = np.zeros(M, dtype=bool)
truccata[M - TRUCCATE:] = True

def una_scatola(seme):
    rng = np.random.default_rng(seme)
    teste = rng.binomial(LANCI, np.where(truccata, SBILANCIO, 0.5))
    p = np.array([tavola[k] for k in teste])
    ordine = np.argsort(p)                       # Benjamini-Hochberg
    sotto = np.nonzero(p[ordine] <= ALFA * np.arange(1, M + 1) / M)[0]
    bh = np.zeros(M, dtype=bool)
    if len(sotto):
        bh[ordine[:sotto[-1] + 1]] = True
    return {"nessuna correzione": p < ALFA, "Bonferroni": p < ALFA / M, "Benjamini-Hochberg": bh}

conti = {nome: [] for nome in ("nessuna correzione", "Bonferroni", "Benjamini-Hochberg")}
for seme in range(PROVE):
    for nome, accusate in una_scatola(seme).items():
        n = accusate.sum()
        conti[nome].append((n, (accusate & ~truccata).sum() / n if n else 0.0,
                            (accusate & truccata).sum()))

for nome, righe in conti.items():
    n, quota, trovate = (np.mean([r[i] for r in righe]) for i in range(3))
    print(f"{nome:20} accusate {n:6.1f}   oneste fra le accusate {100*quota:5.1f}%"
          f"   truccate trovate {trovate:5.1f} su {TRUCCATE}")
```

```text
larghezza tipica della campana dei 100 lanci: 5.0 teste
p di 60 teste su 100 lanci: 0.057
p di 12 teste su  20 lanci: 0.503
prima accusata a 61 teste; soglia dichiarata 0.05, quota vera di oneste accusate 0.035

nessuna correzione   accusate  115.3   oneste fra le accusate  27.9%   truccate trovate  83.0 su 100
Bonferroni           accusate   12.7   oneste fra le accusate   0.2%   truccate trovate  12.7 su 100
Benjamini-Hochberg   accusate   48.6   oneste fra le accusate   3.5%   truccate trovate  46.9 su 100
```

I numeri delle ultime tre righe sono medie su trecento scatole, ed è per questo
che hanno la virgola. Senza correzione si annunciano centoquindici scoperte e
più di una su quattro è rumore: la soglia del cinque per cento sta facendo il
suo mestiere una domanda alla volta, e chi legge l'elenco intero non ha nessuna
delle garanzie che crede di avere. Bonferroni porta i falsi quasi a zero e trova
tredici monete truccate su cento; Benjamini-Hochberg ne trova quarantasette, e la
quota di oneste fra le accusate resta al $3{,}5\%$, sotto il $4{,}5\%$ promesso
(che è il cinque per cento moltiplicato per la quota di monete oneste nella
scatola). L'ultima colonna, quante truccate si trovano su cento, ha un nome: è
la **potenza** della procedura. Le tre righe descrivono tre promesse diverse,
non una classifica, e la domanda giusta è quale delle tre serva a chi legge il
risultato.

Un conto va rifatto, perché non torna, e la ragione è istruttiva. Senza
correzione le accusate sono $115$ e le truccate trovate $83$: le oneste accusate
sono quindi una trentina, mentre il cinque per cento di novecento ne farebbe
prevedere quarantacinque. La spiegazione sta nella quarta riga stampata. Le
teste sono un numero intero, e fra sessanta e sessantuno non c'è niente: a
sessanta $p$ vale $0{,}057$ e non basta, a sessantuno vale $0{,}035$ e basta. La
soglia dichiarata è cinque su cento, quella che si ottiene davvero è tre e mezzo
su cento, e il test è più prudente di quanto prometta. Da lì viene anche il
margine fra il $3{,}5\%$ misurato e il $4{,}5\%$ promesso: con $p$-value
continui quel margine sparirebbe. È il primo posto in cui guardare quando un
conto sui $p$-value non torna.


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

Il confondente, disegnato (nel disegno è etichettato «confonditore Z»: i
due nomi indicano la stessa cosa, e qui si usa il primo). Fra gelati e
annegamenti non passa nessuna freccia: il legame che si misura nei dati è
tutto riflesso di quello che ciascuno dei due ha con il caldo.
```

In {numref}`fig-confonditore` conta soprattutto ciò che *non* c'è: la freccia fra
$X$ e $Y$, cioè fra i gelati e gli annegamenti (nel disegno le due lettere
stanno per le due grandezze che si misurano, e $Z$ per la causa comune). I
dati da soli non la disegnano né la cancellano, perché correlazione e
causalità lasciano sui numeri la stessa traccia. A distinguerle serve qualcosa
che nei dati non c'è: un intervento (cambiare $X$ e guardare $Y$) oppure una
conoscenza del dominio che dica quale freccia è plausibile.

`````{tab} Elementare

La correlazione misura il co-movimento: quanto due grandezze tendono a
salire e scendere insieme. Ed è simmetrica e cieca.

*Simmetrica*: la correlazione fra gelati e annegamenti è identica a quella fra
annegamenti e gelati (il numero non contiene alcuna informazione su chi
influenzi chi).

*Cieca*: non distingue fra una relazione diretta, una mediata da altro e una
pura coincidenza. È come notare che due colleghi arrivano sempre insieme in
ufficio: potrebbero viaggiare insieme, o semplicemente prendere lo stesso treno
perché abitano nello stesso quartiere.

Il caldo dell'esempio si chiama **confondente**: una causa comune che spiega
entrambi gli effetti. Accanto alla causa comune vivono altre due strutture. La
**catena** è un legame vero che fa scalo: il caldo fa venire sete, la sete fa
vendere bibite; la causalità c'è, ma passa per un anello intermedio. Il
**collider** nasce quando si sceglie chi guardare: fra i ristoranti che
sopravvivono, quelli nei vicoli senza passaggio cucinano meglio della media,
perché lì resta aperto solo chi cucina davvero bene. Fra tutti i ristoranti
quel legame non esiste; lo ha creato la selezione dei sopravvissuti.

Il guaio è che noi il caldo lo sospettiamo per buon senso; un modello no.
Impara la scorciatoia che funziona sui dati che ha visto e la usa finché il
mondo non cambia: poi sbaglia, e sbaglia in modo inspiegabile.

`````

`````{tab} Superiore

Il coefficiente di Pearson fra $X$ e $Y$ è

$$
\rho_{XY} = \frac{\mathrm{Cov}(X,Y)}{\sigma_X\,\sigma_Y} \in [-1,1],
$$

dove $\mathrm{Cov}(X,Y)=\mathbb{E}\big[(X-\mu_X)(Y-\mu_Y)\big]$ è la
**covarianza**, cioè la media del prodotto dei due scarti dalla propria media
(positiva se le due grandezze stanno di solito dalla stessa parte del proprio
centro, negativa se da parti opposte), e $\sigma_X,\sigma_Y$ sono le due
deviazioni standard, che servono a togliere di mezzo le unità di misura.

Con più variabili le covarianze si raccolgono in una matrice. Per un vettore
aleatorio $\mathbf{x}\in\mathbb{R}^d$ con media
$\boldsymbol{\mu}=\mathbb{E}[\mathbf{x}]$, la **matrice di covarianza** è
$\boldsymbol{\Sigma}=\mathbb{E}\big[(\mathbf{x}-\boldsymbol{\mu})(\mathbf{x}-\boldsymbol{\mu})^\top\big]\in\mathbb{R}^{d\times
d}$, con $\Sigma_{ij}=\mathrm{Cov}(x_i,x_j)$ e le varianze sulla diagonale. È
simmetrica e semidefinita positiva, perché
$\mathbf{v}^\top\boldsymbol{\Sigma}\mathbf{v}=\mathrm{Var}(\mathbf{v}^\top\mathbf{x})\ge
0$ per ogni $\mathbf{v}$: la varianza lungo una direzione unitaria è una forma
quadratica, e massimizzarla porta all'autovettore dominante di
$\boldsymbol{\Sigma}$, cioè alla PCA della {doc}`sezione sull'algebra lineare
<algebra-lineare>`. Se $\boldsymbol{\Sigma}$ è definita positiva, la normale
multivariata ha densità

$$
f(\mathbf{x})=\frac{1}{(2\pi)^{d/2}(\det\boldsymbol{\Sigma})^{1/2}}
\exp\!\Big(-\tfrac12(\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\Big),
$$

e il suo $(\det\boldsymbol{\Sigma})^{1/2}$ è proporzionale al volume
dell'ellissoide di covarianza della {doc}`sezione sul determinante
<determinante-e-volume>`.

Il coefficiente misura la sola dipendenza lineare: $\rho=0$ non implica
indipendenza; se $X$ è distribuita in modo simmetrico attorno allo zero,
$Y=X^2$ ha correlazione nulla con $X$ e dipendenza perfetta. (La simmetria è
essenziale: per $X$ uniforme su $[0,1]$ la stessa parabola dà una correlazione
vicina a $0{,}97$.)

Le strutture da tenere distinte:

- confondente: $Z \to X$ e $Z \to Y$ producono correlazione fra $X$ e $Y$
  senza alcun legame causale diretto (il caldo);
- catena: $X \to Z \to Y$, causalità reale ma mediata;
- collider: $X \to Z \leftarrow Y$, dove condizionare su $Z$ *crea*
  correlazione fra $X$ e $Y$ che non esisteva. È il meccanismo dietro molti
  paradossi di selezione del campione.

Dai soli dati osservativi la catena e il confondente sono indistinguibili:
tutti e due rendono $X$ e $Y$ dipendenti, e tutti e due le rendono
indipendenti dato $Z$. Il collider lascia invece un'impronta propria, $X$ e
$Y$ indipendenti finché non si condiziona su $Z$, ed è questa struttura a v
che gli algoritmi di scoperta causale riescono a orientare dai dati, sotto
l'ipotesi che ogni indipendenza osservata venga dal grafo (fedeltà). Il verso
di una catena e un confondente non osservato chiedono invece un intervento
(esperimento randomizzato) o assunzioni causali esplicite. Nel machine
learning la conseguenza ha un nome, *shortcut learning*: il modello aggancia
la correlazione più comoda del dataset (lo sfondo invece dell'animale, il
marcatore dell'ospedale invece della patologia) e crolla appena la
distribuzione cambia. La correlazione basta per predire dentro la stessa
distribuzione; non basta per decidere un intervento.

`````

### Tre gradini: vedere, fare, immaginare

Le tre strutture appena elencate si sistemano dentro una cornice più larga, da
avere in testa perché rimette in fila cose che tornano in capitoli
lontanissimi fra loro. La propone Judea Pearl, e la chiama **scala
della causalità** {cite}`pearl2018book`.

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
benissimo. Pearl lo dice con un'immagine: «la civetta può essere un buon
cacciatore senza capire perché il topo vada sempre da A a B». La
civetta ha visto migliaia di topi e sa dove sarà questo fra un secondo, il che
le basta per prenderlo; delle ragioni per cui il topo si sposta non sa niente,
e non le servono. Predire, insomma, non richiede capire. Il punto è sapere
quale domanda si sta facendo, perché salire un gradino richiede sempre qualcosa
che nei dati non c'è.

`````

`````{tab} Superiore

I tre livelli si distinguono formalmente dall'oggetto probabilistico che
sanno esprimere.

1. Associazione: $P(y \mid x)$, cioè condizionare. È tutto ciò che si
   ottiene osservando, e include correlazione, regressione e ogni modello
   puramente predittivo.
2. Intervento: $P(y \mid do(x))$, dove l'operatore $do$ denota
   l'imposizione di $X = x$ dall'esterno, che nel grafo causale corrisponde a
   recidere tutti gli archi entranti in $X$. In generale
   $P(y \mid do(x)) \neq P(y \mid x)$, e la differenza è esattamente il
   contributo dei confondenti.
3. Controfattuale: $P(y_x \mid x', y')$, la probabilità che $Y$ sarebbe
   stato $y$ sotto $X = x$, dato che in realtà si è osservato $x'$ e $y'$.
   Richiede un modello strutturale completo, non solo il grafo.

Il risultato che rende la scala operativa e non solo tassonomica è che
condizioni grafiche esplicite (il *criterio di backdoor*, il *do-calculus*)
dicono quando una quantità di livello 2 è calcolabile a partire da soli
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

I tre gradini si distinguono per che cosa serve avere prima di poter
rispondere, non per quanto siano nobili le domande. Sul primo gradino, quello
di chi guarda i dati e conta, sta la gran parte di quello che leggeremo. Sul
secondo, quello di chi il mondo lo tocca invece di limitarsi a guardarlo,
stanno i test A/B (si mostrano due versioni di un prodotto a due gruppi di
utenti scelti a caso e si confrontano i risultati) e l'apprendimento per
rinforzo, dove un programma i dati non li riceve, se li produce agendo. Sul
terzo stanno le domande su ciò che non è successo, tipo «a questo cliente il
prestito è stato negato; glielo avrebbero dato con mille euro di reddito in
più?»: si chiamano spiegazioni controfattuali. Il secondo gradino ha il suo
capitolo nel {doc}`reinforcement learning </ReinforcementLearning/overview>`;
le spiegazioni controfattuali stanno nel capitolo sull’{doc}`interpretabilità
</Interpretabilita/overview>`.

## Dalla probabilità all'apprendimento

Resta un'ultima domanda: cosa c'entra tutto questo con l’*addestrare* un modello?
Il ponte si chiama **massima verosimiglianza** (*maximum likelihood estimation*,
MLE).

```{figure} ../figures/massima-verosimiglianza.svg
:name: fig-verosimiglianza
:alt: "Due curve di verosimiglianza sovrapposte, in funzione della probabilità p di ottenere testa. La prima viene da 7 teste su 10 lanci ed è larga e piatta; la seconda da 70 teste su 100 ed è molto più stretta. Entrambe hanno il massimo nello stesso punto, p uguale a 0,7."
:width: 90%

Stessa proporzione, due quantità di prove. Il punto più alto non si sposta; a
cambiare è quanto la curva sia stretta, cioè quanto ci si può contare. Il
cappello su $\hat p$, nel disegno, vuol dire «ricavato dai dati», e serve a
tenerlo distinto dal valore vero, che nessuno conosce.
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
$p$ si calcola quanto sarebbe probabile vedere proprio 7 teste su 10. Il conto
ha due pezzi. Il primo: una sequenza precisa, per dire TTTTTTTCCC, ha
probabilità $p^7(1-p)^3$, cioè sette volte $p$ per tre volte $1-p$. Il secondo:
di sequenze con sette teste ce ne sono $120$, tutte ugualmente probabili,
quindi si moltiplica per $120$. Con $p=0{,}5$ viene $120 \cdot 0{,}5^{10}
\approx 0{,}12$, cioè il $12\%$: possibile, non entusiasmante. Se fosse
$p=0{,}6$ si salirebbe al $21\%$, con $p=0{,}7$ si arriverebbe al $27\%$, con
$p=0{,}8$ si ridiscenderebbe al $20\%$. Il massimo cade a $0{,}7$, cioè
esattamente sulla proporzione osservata, ed è questo che si intende quando si
dice che $0{,}7$ è il valore «che rende l'osservazione più probabile».

Quel «quanto è probabile l'osservazione, se il parametro valesse così» ha un
nome: si chiama **verosimiglianza**. La curva della figura è
proprio questa: per ogni valore di $p$ sull'asse orizzontale, quanto sarebbe
verosimile ciò che ho visto.

"Imparare", per un modello, è spesso esattamente questo: girare le manopole
dei parametri finché i dati osservati diventano i più plausibili.

`````

`````{tab} Superiore

Dati esempi indipendenti $x^{(1)},\dots,x^{(m)}$, la verosimiglianza dei
parametri $\theta$ è

$$
L(\theta)=\prod_{i=1}^{m} p\big(x^{(i)};\theta\big),
$$

e la stima di massima verosimiglianza è

$$
\hat{\theta}=\arg\max_{\theta}\ \sum_{i=1}^{m}\log p\big(x^{(i)};\theta\big).
$$

Il passaggio dal prodotto alla somma dei logaritmi è lecito perché il
logaritmo è strettamente crescente: applicarlo non sposta il punto di
massimo, e i due problemi hanno lo stesso $\arg\max$. Che poi sommare sia
anche numericamente più stabile che moltiplicare mille numeri piccoli è un
secondo vantaggio, non la giustificazione.

(Scriviamo $L$ e non $\mathcal{L}$: la $\mathcal{L}$ calligrafica indica la
loss, che si minimizza; la verosimiglianza si massimizza, e le due
si incontrano nella log-verosimiglianza negativa,
$\mathcal{L}(\theta)=-\log L(\theta)$ a meno di costanti.) Il punto cruciale:
per un modello di regressione che descrive $y$ dato $x$ come una gaussiana
centrata sulla predizione,
$y \mid x \sim \mathcal{N}(\hat{y},\sigma^2)$
con varianza fissa, massimizzare la log-verosimiglianza equivale a
minimizzare l'errore quadratico medio; sotto ipotesi di
Bernoulli/categoriche equivale a minimizzare la cross-entropy. Le loss
$\mathcal{L}$ sono verosimiglianze travestite, non scelte arbitrarie: a
svestirle una per una, e a farne una procedura applicabile a una
distribuzione qualsiasi, è
{doc}`Da dove viene la loss </RetiNeurali/da-dove-viene-la-loss>`.

`````

### Con un'opinione di partenza: la stima bayesiana

La massima verosimiglianza tratta il parametro come un numero da trovare e
basta. Il teorema di Bayes della sezione «Aggiornare le credenze» permette di
fare di più: dare al parametro $\theta$ (per la moneta, la probabilità di testa)
una **distribuzione a priori** $p(\theta)$, in breve la *priore*, che dice che
cosa se ne pensa prima dei dati, e ottenere con i dati $\mathcal{D}$ la
**distribuzione a posteriori** $p(\theta \mid \mathcal{D})$, la *posteriore*,
dove la barra si legge «sapendo che». Da lì escono la stima **MAP** (*maximum a
posteriori*, il punto più alto della posteriore) e, quando la priore è
**coniugata** alla verosimiglianza, una posteriore della stessa famiglia della
priore, che si aggiorna sommando dei conteggi.
{numref}`fig-credenza-che-si-stringe` mostra l'aggiornamento sulla moneta, un
lancio dopo l'altro.

```{figure} ../figures/credenza-che-si-stringe.svg
:name: fig-credenza-che-si-stringe
:alt: Animazione su un asse orizzontale che va da 0 a 1, la probabilità di testa, con una linea verticale tratteggiata sul valore vero 0,7. Una curva terracotta bassa e larga, la credenza prima dei lanci, lascia il posto tappa dopo tappa a curve sempre più alte e strette, dopo 1, 3, 10, 30 e 100 lanci; ognuna resta come traccia sbiadita, e l'ultima, piena, sta stretta attorno al valore vero. Una scritta in alto a sinistra dice quanti lanci e quante teste si sono visti.
:width: 92%

La credenza sulla probabilità di testa, prima dei lanci e dopo 1, 3, 10, 30 e
100 lanci di una moneta che fa testa il 70% delle volte: a ogni tappa la curva
si alza e si stringe, e alla fine sta attorno al valore vero.
```

`````{tab} Elementare

Chi prende in mano una moneta qualunque, prima di lanciarla, un'idea ce l'ha:
non sa se fa testa esattamente metà delle volte, ma non scommetterebbe che la
faccia nove volte su dieci. Quell'idea si può scrivere come lanci immaginati,
per esempio come se si fossero già viste due teste e due croci. Poi arrivano i
lanci veri, e si sommano a quelli immaginati: dopo 7 teste su 10, la stima (il
baricentro della curva di cui si dice subito sotto) diventa
$(2 + 7)/(4 + 10) = 9/14 \approx 0{,}64$, un po' più vicina a metà del $0{,}7$
dei soli lanci veri, perché i quattro immaginati tirano verso il centro.

Il risultato non è un numero ma una curva, che dice quanto è credibile ogni
valore della moneta. All'inizio è bassa e larga; con i lanci si alza e si
stringe, e i lanci immaginati pesano sempre meno: dopo 70 teste su 100 la stima
è $(2 + 70)/(4 + 100) = 72/104 \approx 0{,}69$, quasi la proporzione osservata.
Una curva si può riassumere in due modi, con il suo punto più alto o con il suo
baricentro, e con pochi lanci i due differiscono un poco; con tanti coincidono.
E il punto più alto è la massima verosimiglianza con in più una spinta verso
l'opinione di partenza: la {doc}`regolarizzazione
</MachineLearning/overfitting-validazione>`, nel capitolo sul Machine Learning,
userà proprio questo gesto per trattenere un modello che esagera.

Il prezzo è l'opinione di partenza, che bisogna scegliere. Con quattro lanci
immaginati conta poco e presto; con mille lanci immaginati tutti a metà, cento
teste di fila sposterebbero la stima soltanto da $0{,}50$ a $0{,}55$ (cioè
$600/1100$), e sarebbe la testardaggine di chi l'ha scelta, non la prudenza.

`````

`````{tab} Superiore

Per il teorema di Bayes,
$p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta)\, p(\theta)$,
e la stima MAP è

$$
\hat{\theta}_{\text{MAP}} = \arg\max_{\theta}\ \Big[\log p(\mathcal{D} \mid \theta) + \log p(\theta)\Big]:
$$

la massima verosimiglianza più un termine che dipende solo dal parametro. Con
una priore gaussiana $\theta_j \sim \mathcal{N}(0, \tau^2)$ quel termine è
$-\sum_j \theta_j^2 / (2\tau^2)$, che corrisponde, a meno di un fattore
positivo, alla penalità $\ell_2$ del Ridge della {doc}`regolarizzazione
</MachineLearning/overfitting-validazione>`; con una priore di Laplace è la
$\ell_1$ del Lasso. La stima bayesiana completa, invece, tiene tutta la
posteriore: ne ricava la media, un intervallo di credibilità, la predittiva
$p(x \mid \mathcal{D}) = \int p(x \mid \theta)\, p(\theta \mid \mathcal{D})\, d\theta$.

Una priore è coniugata a una verosimiglianza quando la posteriore sta nella sua
stessa famiglia. Per $k$ teste su $n$ lanci di Bernoulli la coniugata è la Beta:
da $\mathrm{Beta}(a, b)$ si passa a $\mathrm{Beta}(a + k,\, b + n - k)$, con
moda $(a + k - 1)/(a + b + n - 2)$ e media $(a + k)/(a + b + n)$, e $a$, $b$ si
leggono come conteggi fittizi per la media (per la moda i conteggi sono $a - 1$
e $b - 1$). Per la media $\mu$ di una gaussiana con varianza $\sigma^2$ nota, la
coniugata è gaussiana: con priore $\mathcal{N}(\mu_0, \tau_0^2)$ e $n$
osservazioni di media $\bar{x}$,

$$
\frac{1}{\tau_n^2} = \frac{1}{\tau_0^2} + \frac{n}{\sigma^2}, \qquad
\mu_n = \tau_n^2 \left( \frac{\mu_0}{\tau_0^2} + \frac{n \bar{x}}{\sigma^2} \right),
$$

una media pesata con le precisioni, in cui il peso della priore resta fisso e
quello dei dati cresce con $n$: per $n \to \infty$ la media tende a $\bar{x}$ e
la varianza a posteriori scende come $\sigma^2/n$. I limiti: la coniugazione
esiste per le famiglie esponenziali con priori scelte apposta, e altrove la
posteriore si approssima, campionando con le catene di Markov della
{doc}`sezione sul MCMC </Matematica/catene-di-markov>` o con metodi
variazionali; con pochi dati la priore pesa, e va giustificata; e la MAP, a
differenza della massima verosimiglianza, dipende da come si parametrizza
$\theta$, perché una densità cambia forma sotto un cambio di variabile e il suo
massimo si sposta.

`````

Il blocco calcola sulla moneta, con la priore $\mathrm{Beta}(2, 2)$, la massima
verosimiglianza, la MAP, la media a posteriori e l'intervallo al 95% (la fascia
centrale che contiene il 95% della credenza, detto intervallo di credibilità)
dopo 7 teste su 10 e dopo 70 su 100; poi ricostruisce la prima posteriore su una
griglia, moltiplicando verosimiglianza e priore, e la confronta con la formula
della coniugata.

```python
import numpy as np
from scipy.stats import beta

a0, b0 = 2, 2                                   # la credenza di partenza: Beta(2, 2)
for teste, lanci in [(7, 10), (70, 100)]:
    a, b = a0 + teste, b0 + lanci - teste       # la coniugata: ai parametri si sommano i conteggi
    mle = teste / lanci
    map_ = (a - 1) / (a + b - 2)                # il punto più alto della posteriore
    media = a / (a + b)                         # il suo baricentro
    basso, alto = beta.ppf([0.025, 0.975], a, b)
    print(f"{teste:2}/{lanci:3}: massima verosimiglianza {mle:.3f}, MAP {map_:.3f}, "
          f"media {media:.3f}, intervallo al 95% da {basso:.3f} a {alto:.3f}")

# la stessa posteriore senza la formula: verosimiglianza per prior, su una griglia
p = np.linspace(0.0005, 0.9995, 1000)
posteriore = p**7 * (1 - p)**3 * beta.pdf(p, a0, b0)
posteriore /= posteriore.sum() * (p[1] - p[0])
print("coincide con la Beta(9, 5):", np.allclose(posteriore, beta.pdf(p, 9, 5), rtol=1e-3))
```

```text
 7/ 10: massima verosimiglianza 0.700, MAP 0.667, media 0.643, intervallo al 95% da 0.386 a 0.861
70/100: massima verosimiglianza 0.700, MAP 0.696, media 0.692, intervallo al 95% da 0.601 a 0.777
coincide con la Beta(9, 5): True
```

Con dieci lanci la priore si sente: la MAP scende da $0{,}700$ a $0{,}667$ e la
media a $0{,}643$, e l'intervallo va da $0{,}386$ a $0{,}861$, largo quasi metà
dell'asse. Con cento lanci le tre stime stanno fra $0{,}692$ e $0{,}700$ e
l'intervallo si è ristretto a poco più di un sesto dell'asse, da $0{,}601$ a
$0{,}777$. La griglia, che non sa niente di coniugate, dà la stessa
$\mathrm{Beta}(9, 5)$ della formula.

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
print(mu, var, np.sqrt(var))            # 3.5  ~2.9167  ~1.7078

# Teorema di Bayes: test per una malattia rara
prevalenza  = 0.01     # P(malato)
sensibilita = 0.99     # P(positivo | malato)
falsi_pos   = 0.05     # P(positivo | sano)

p_pos     = sensibilita * prevalenza + falsi_pos * (1 - prevalenza)
posterior = sensibilita * prevalenza / p_pos
print(posterior)       # ~0.167: solo il 17% dei positivi è davvero malato
```

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- La probabilità misura l'incertezza; una variabile aleatoria le dà un
  numero (le teste su dieci lanci), e per riassumerla bastano quasi sempre due
  cose: il risultato medio che ci si aspetta a lungo andare e quanto
  tipicamente ci si allontana da quel centro.
- Contare i casi vuol dire moltiplicare le scelte e dividere per i modi di
  mettere in fila che non contano: dieci amici formano $\binom{10}{3} = 120$
  squadre da tre. Il conto diventa una probabilità solo se i casi contati sono
  ugualmente probabili, ed è lì che sbagliò d'Alembert.
- La curva a campana compare ovunque perché tante piccole cause casuali che
  si sommano la producono da sé, anche partendo da dadi in cui nessuna faccia è
  favorita. Regola pratica: circa il $68\%$ dei casi cade entro uno scarto
  tipico dalla media, circa il $95\%$ entro due.
- Un risultato positivo va sempre letto insieme a quanto la cosa cercata è
  rara: con una malattia che colpisce una persona su cento, anche un test molto
  buono produce più falsi allarmi che malati veri (solo il $17\%$ circa dei
  positivi è davvero malato). È il teorema di Bayes al lavoro.
- La media di tante osservazioni si assesta sul valore vero, ma le oscillazioni
  si stringono con calma: per dimezzare l'incertezza non basta il doppio dei
  dati, ne servono quattro volte tanti.
- Un'accuratezza è una stima, come un sondaggio elettorale: su $500$ esempi
  vale circa $3$ punti in più o in meno, e differenze più piccole di così sono
  rumore, non progresso.
- Per una risposta secca («è cambiato qualcosa, sì o no?») si parte dalla
  risposta che si vuole scartare: si suppone che sia stato il caso e si
  guarda quanto spesso il caso, da solo, produrrebbe una stranezza come quella
  vista. Quel «quanto spesso» è il $p$, e la soglia sotto cui lo si accetta
  (per convenzione cinque su cento) dice quante volte si è disposti ad accusare
  un innocente, non quanto è forte la prova. Non accusare non è assolvere: con
  pochi dati non si distingue una moneta truccata da una onesta.
- E $p$ non è la probabilità che l'ipotesi sia vera: per quella servirebbe
  anche sapere quanto la cosa cercata è rara, cioè di nuovo Bayes.
- Facendo mille domande insieme con la solita soglia, decine di risposte
  positive arriverebbero per puro caso anche se non ci fosse assolutamente
  niente da trovare. Bonferroni stringe la soglia
  dividendola per il numero di domande, così è improbabile sbagliarne anche una
  sola, e in cambio quasi non trova più niente; Benjamini-Hochberg promette
  invece che fra le scoperte annunciate la quota di errori resti bassa in
  media, e ne trova molte di più. La scelta dipende da quanto costa una
  segnalazione sbagliata.
- Due grandezze che salgono e scendono insieme (i gelati e gli annegamenti)
  bastano a prevedere finché il mondo resta com'è, non a decidere un
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
- La probabilità misura l'incertezza; una variabile aleatoria le dà un
  numero, e $\mathbb{E}[X]$ e $\mathrm{Var}(X)$ ne riassumono centro e
  dispersione. Attenzione a non confondere la varianza *della distribuzione*
  con la sua stima dal campione: quest'ultima è non distorta solo
  dividendo per $n-1$ (correzione di Bessel), e le librerie scelgono
  default diversi.
- $\binom{n}{k} = n!/(k!\,(n-k)!)$ conta i sottoinsiemi di taglia $k$, e
  $\binom{n}{k}p^k(1-p)^{n-k}$, la binomiale, è la probabilità di $k$ successi
  in $n$ prove indipendenti con la stessa probabilità $p$; la probabilità
  classica «favorevoli su possibili» vale solo su uno spazio di esiti
  equiprobabili.
- La normale compare ovunque per il teorema del limite centrale; la regola
  68–95 lega la deviazione standard $\sigma$ alle probabilità.
- Il teorema di Bayes aggiorna le credenze alla luce dei dati: con classi
  rare, occhio ai falsi positivi.
- La legge dei grandi numeri garantisce la convergenza della media, ma solo
  come $1/\sqrt{n}$: dimezzare l'incertezza costa quattro volte i dati.
- Un'accuratezza è una stima: su $500$ esempi il margine al $95\%$ è di
  circa $\pm 3$ punti, e differenze più piccole sono rumore.
- Il $p$-value è $\Pr(T \text{ almeno estrema} \mid H_0)$, e il livello
  $\alpha$ (errore di prima specie) si fissa prima: rifiutare a livello
  $\alpha$ equivale a non trovare il valore nullo nell'intervallo di confidenza
  al $1-\alpha$ costruito con lo stesso metodo (Wald con il test $z$,
  Clopper-Pearson con il binomiale esatto). Su una statistica discreta il
  livello nominale non è raggiunto e il test è conservativo: con $n=100$ e
  $\alpha = 0{,}05$ la taglia vera è $0{,}035$. Non misura $\Pr(H_0 \mid \text{dati})$, e perde ogni
  significato se statistica e soglia sono scelte dopo aver visto i dati.
- Con $m$ test simultanei si controlla la FWER $= \Pr(V \ge 1)$ con
  Bonferroni ($\alpha/m$, valida senza ipotesi di indipendenza per la
  disuguaglianza di Boole) oppure la FDR $= \mathbb{E}[V/R]$ con
  Benjamini-Hochberg (il più grande $k$ con $p_{(k)} \le kq/m$), che sotto
  indipendenza o dipendenza positiva (PRDS) garantisce
  $\mathrm{FDR} \le q\,m_0/m$. Entrambe presuppongono $p$-value validi, e la
  FDR è una media su ripetizioni, non una promessa su questo insieme di dati.
- La correlazione basta per predire dentro la stessa distribuzione, non per
  decidere un intervento: confondenti e collider producono correlazioni senza
  causalità.
- La massima verosimiglianza è il ponte fra probabilità e apprendimento:
  minimizzare MSE o cross-entropy è massimizzare una verosimiglianza.
```
`````

La media di un campione è la stima che torna più spesso in tutto quello che
segue, e il limite centrale ne dice l'errore tipico quando gli esempi sono
tanti. Quanto possa sbagliare con un numero finito di esempi, e con quale
probabilità, lo dice la {doc}`sezione sulla concentrazione <concentrazione>`.
