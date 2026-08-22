# Algebra lineare: vettori, matrici, prodotti

Tutto, nel machine learning, comincia con l'idea di mettere i numeri in fila.
Un'email è una lista di parole, una foto è una griglia di pixel, un cliente è
una scheda di attributi (età, acquisti, città). In tutti questi casi facciamo
la stessa mossa: impilare i numeri in un **vettore**, e impilare i vettori in
una **matrice**. L'algebra lineare è la grammatica di queste pile.

## Vettori: dati e direzioni

```{figure} ../figures/vettori-e-matrici.svg
:name: fig-somma-vettori
:alt: "Piano cartesiano con due vettori disegnati come frecce che partono dall'origine. La loro somma è una terza freccia, e un parallelogramma tratteggiato costruito sui primi due mostra che il vertice opposto all'origine è proprio l'estremo della somma."
:width: 80%

Il vettore ha due letture, e la figura le tiene insieme: una coppia di numeri
e una freccia. Sommare due liste di numeri voce per voce ($[3,2]$ più $[1,3]$
fa $[4,5]$) e fare prima uno spostamento e poi l'altro sono la stessa
operazione: si arriva nello stesso punto.
```

La doppia lettura di {numref}`fig-somma-vettori` è il motivo per cui l'algebra
lineare serve al machine learning. I dati arrivano come liste di numeri, e
appena li si guarda come frecce diventano disponibili parole che sui numeri
non avevano senso: direzione, distanza, angolo.

`````{tab} Elementare

Un vettore è semplicemente una lista ordinata di numeri. Se descrivo un
appartamento con tre numeri (metri quadri, numero di stanze, piano), ho già un
vettore:

$$
\mathbf{x} = (75,\ 3,\ 2)
$$

L'ordine conta: il primo posto è sempre "metri quadri", il secondo "stanze",
e così via. Possiamo immaginare un vettore di due o tre numeri come una
**freccia** che parte dall'origine e punta verso quel punto: ci dice una
*direzione* e una *lunghezza*.

Con i vettori si fanno tre gesti, e sono tutti conti che sapresti fare a
mente.

*Sommarli*: si sommano i numeri che occupano lo stesso posto. Se
$\mathbf{x} = (75, 3, 2)$ e $\mathbf{y} = (10, 1, 0)$, allora
$\mathbf{x} + \mathbf{y} = (85, 4, 2)$. Sulla freccia vuol dire
fare prima uno spostamento e poi l'altro.

*Sottrarli*: si sottraggono voce per voce,
$\mathbf{y} - \mathbf{x} = (-65, -2, -2)$. È
l'operazione che serve per misurare uno scarto, e quindi un errore: la
differenza fra quello che il modello ha previsto e quello che è successo
davvero.

*Moltiplicarli per un numero*: si moltiplica ogni voce per quel numero,
$2\mathbf{x} = (150, 6, 4)$. La freccia resta sulla stessa retta e cambia solo
lunghezza (e se il numero è negativo, si volta dall'altra parte).

Quanti numeri contiene la lista si chiama la sua **dimensione**: l'appartamento
di sopra è un vettore di dimensione tre. Non ha niente a che vedere con le
dimensioni di un mobile, è solo un conteggio di caselle.

Con più di tre numeri la freccia non la disegniamo più, ma i tre gesti restano
identici, perché sono conti sulle liste e non sul disegno: si sommano
ottantaquattro numeri con ottantaquattro numeri esattamente come se ne
sommavano tre con tre. Prendi una fotografia in bianco e nero di ventotto
puntini per ventotto: ogni puntino, che si chiama **pixel**, per il calcolatore
è un numero, e dice quanto quel punto è chiaro o scuro. Mettendoli in fila una
riga dopo l'altra viene una lista di $28 \times 28 = 784$ numeri, ed è un
esempio che tornerà spesso. Quando si dirà
«direzione» in uno spazio a 784 dimensioni, la cosa da tenere in mente non è
un'immagine ma questa: le operazioni sono le stesse, e le parole «vicino»,
«lontano», «dalla stessa parte» continuano a voler dire qualcosa perché si
calcolano, non perché si vedono.

`````

`````{tab} Superiore

Un vettore è un elemento di uno spazio vettoriale, tipicamente $\mathbb{R}^n$.
Lo scriviamo come colonna di $n$ componenti reali:

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}
\in \mathbb{R}^n .
$$

Le due operazioni fondanti sono la **somma** componente per componente e la
**moltiplicazione per uno scalare** $\alpha \in \mathbb{R}$:

$$
(\mathbf{x}+\mathbf{y})_i = x_i + y_i,
\qquad
(\alpha\,\mathbf{x})_i = \alpha\,x_i .
$$

In machine learning $\mathbf{x}$ è tipicamente il vettore delle
**caratteristiche** (*feature*) di un singolo esempio, e $n$ è la
dimensionalità dello spazio delle feature. Un'immagine $28\times 28$ in scala
di grigi, "srotolata", è un vettore di $\mathbb{R}^{784}$.

`````

## Il prodotto scalare: quanto due vettori "vanno d'accordo"

È l'operazione più importante di tutto il libro: un **neurone artificiale**,
in fondo, non fa altro che calcolare un prodotto scalare. Il nome arriva dal
capitolo sulle reti neurali e qui basta sapere cosa indica: il mattone
elementare di cui una rete è fatta, un pezzetto di calcolo che riceve una
lista di numeri, la confronta con una lista di numeri propri e restituisce un
numero solo.

Conviene rendersi conto della scala. Ogni volta che un modello linguistico
genera una singola parola, da qualche parte in un datacenter vengono eseguiti
miliardi di esemplari della stessa operazione: prendere due liste di numeri,
moltiplicarle voce per voce, sommare tutto. Due vettori entrano, un numero solo
esce. Occupa mezza riga in un libro di matematica ed è, con ogni probabilità,
l'operazione aritmetica più eseguita sul pianeta in questo momento.

```{figure} ../figures/prodotto-scalare.svg
:name: fig-prodotto-scalare
:alt: "Proiezione geometrica: il vettore a proiettato sul vettore b, con l'angolo theta e la perpendicolare tratteggiata; la lunghezza della proiezione evidenziata in terracotta"
:width: 85%

La proiezione di $\mathbf{a}$ su $\mathbf{b}$ è l’"ombra" che $\mathbf{a}$
getta sulla direzione di $\mathbf{b}$. Il prodotto scalare è la lunghezza di
quest'ombra moltiplicata per la lunghezza di $\mathbf{b}$, e l'angolo fra le
due frecce (nel disegno la lettera greca $\theta$, si legge «theta») decide se
il risultato è grande, nullo o negativo. Nella formula scritta accanto, le
stanghette $\lvert\mathbf{a}\rvert$ vogliono dire «lunghezza di $\mathbf{a}$»
e $\cos\theta$, il *coseno* dell'angolo, è semplicemente un numero fra $-1$ e
$1$ che misura l'accordo fra le due direzioni: vale $1$ se puntano dalla
stessa parte, $0$ se sono perpendicolari, $-1$ se sono opposte. Non serve
saperlo calcolare per leggere il resto: qui il coseno è solo il nome di quel
numero.
```

`````{tab} Elementare

Alla cassa del supermercato si incontrano due liste. Nel carrello ci sono 4
chili di patate e 2 vaschette di fragole; le patate stanno a 1 euro al chilo e
le fragole a 3 euro l'una. La cassa moltiplica ogni quantità per il suo prezzo
e somma, $4\cdot 1 + 2\cdot 3 = 4 + 6 = 10$ euro. Quel conto è il **prodotto
scalare** delle due liste $\mathbf{a}=(4,2)$ e $\mathbf{b}=(1,3)$, e in
generale si scrive

$$
\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + \dots + a_n b_n .
$$

I numerini in basso dicono il posto nella fila. $a_1$ è la prima voce dello
scontrino, $a_2$ la seconda, $n$ quante sono in tutto. Letta a voce, la riga
dice *primo per primo, più secondo per secondo, e così via fino alla fine*. Con
mille voci nel carrello le moltiplicazioni da sommare diventano mille, e
qualsiasi totale pesato che tu abbia mai fatto era uno di questi. Mille
quantità e mille prezzi entrano, e ne esce un numero solo, che in matematica si
chiama *scalare* e dà il nome all'operazione: la spesa intera in una cifra,
confrontabile con quella di domenica scorsa.

Le stesse due liste, disegnate come frecce, rispondono a un'altra domanda.
Puntano dalla stessa parte? La strada dritta davanti a te è $\mathbf{b}$; il
bastone piantato obliquo all'inizio della strada è $\mathbf{a}$, e il sole è a
picco ({numref}`fig-prodotto-scalare`). L'ombra sull'asfalto è la parte di
bastone che va dove va la strada, e il prodotto scalare è la sua lunghezza
moltiplicata per quella della strada. Se il bastone pende anche di lato l'ombra
cade di sbieco, e nel conto entra soltanto quanto ne avanza lungo la
carreggiata.

Bastone quasi sdraiato sulla strada, ombra lunga, numero grande e positivo.
Dritto in piedi, ombra ridotta a un punto, zero. Inclinato all'indietro, ombra
dall'altra parte, e allora la sua lunghezza la contiamo col segno meno, perché
il segno dice da che lato l'ombra è caduta. Tre pendenze, tre verdetti:
concordi, indifferenti, opposti.

Allunga la strada di un chilometro e il totale cresce, senza che il bastone si
sia mosso di un grado, perché quel numero mescola le due lunghezze con
l'accordo. Il segno invece non si muove, comunque si allunghino strada e
bastone.

Per confrontare le sole pendenze si divide il prodotto scalare per le due
lunghezze, cioè per il massimo che potrebbe raggiungere, quello che si tocca
col bastone sdraiato esattamente lungo la strada. La lunghezza di una freccia
la dà il teorema di Pitagora, si chiama **norma** e avrà più avanti una sezione
tutta sua. Le nostre due frecce misurano
$\sqrt{4^2+2^2}=\sqrt{20}\approx 4{,}47$ e
$\sqrt{1^2+3^2}=\sqrt{10}\approx 3{,}16$; il prodotto scalare faceva $10$, e
$10 / (4{,}47 \cdot 3{,}16) \approx 0{,}71$. Quel numero sta fra $-1$ e $1$,
non cambia se allunghi bastone e strada, ed è il coseno dell'angolo fra le due
frecce, l'accordo puro fra le direzioni, la **similarità del coseno**.

`````

`````{tab} Superiore

Per $\mathbf{a},\mathbf{b}\in\mathbb{R}^n$ il prodotto scalare (o *interno*) è

$$
\mathbf{a}^\top \mathbf{b} = \sum_{i=1}^{n} a_i b_i
= \lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert\cos\theta,
$$

dove $\theta$ è l'angolo tra i due vettori. La seconda uguaglianza è la
chiave: il prodotto scalare misura l'allineamento. Da qui la **similarità del
coseno**, onnipresente nel NLP per confrontare *embedding*:

$$
\cos\theta = \frac{\mathbf{a}^\top \mathbf{b}}
{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}\in[-1,1].
$$

Due vettori sono **ortogonali** quando $\mathbf{a}^\top\mathbf{b}=0$. Un
singolo neurone artificiale calcola esattamente $\mathbf{w}^\top\mathbf{x}+b$:
il prodotto scalare tra i pesi $\mathbf{w}$ e l'input $\mathbf{x}$, più un
termine di bias.

`````

## Matrici: trasformazioni di interi insiemi di dati

Un vettore descrive un appartamento; ma di appartamenti ce ne sono cento, e
prima o poi bisogna metterli tutti insieme. Basta impilare le liste una sotto
l'altra e viene fuori una tabella. Da lì nasce la seconda cosa che si fa con
una tabella di numeri, meno ovvia della prima e molto più importante: non solo
tenere fermi i dati, ma trasformarli.

`````{tab} Elementare

Una matrice è una tabella di numeri: righe e colonne. Ci serve per due cose.
Primo, **impilare tanti esempi**: se ho 100 appartamenti descritti da 3 numeri
ciascuno, ottengo una tabella $100\times 3$. Secondo, **trasformare** i dati,
ed è qui che serve rallentare, perché «moltiplicare una lista di numeri
per una tabella di numeri» sembra un'operazione misteriosa e non lo è: è il
prodotto scalare di poco fa, ripetuto una volta per ogni riga della tabella.

Torniamo all'appartamento, $\mathbf{x} = (75,\ 3,\ 2)$: metri quadri, stanze,
piano.
Voglio ricavarne due numeri nuovi, un punteggio di «ampiezza» e uno di
«comodità». Scrivo due liste di pesi, una per punteggio, e le impilo:

$$
\begin{array}{lccc}
 & \text{mq} & \text{stanze} & \text{piano}\\
\text{ampiezza} & 2 & 10 & 0\\
\text{comodità} & 0 & 1 & 5
\end{array}
$$

Ogni riga si combina con l'appartamento come uno scontrino: quantità per
prezzi, e si somma.

- ampiezza: $2\cdot 75 + 10\cdot 3 + 0\cdot 2 = 150 + 30 + 0 = 180$;
- comodità: $0\cdot 75 + 1\cdot 3 + 5\cdot 2 = 0 + 3 + 10 = 13$.

Il risultato è la lista $(180,\ 13)$. Tre numeri sono entrati, due ne sono
usciti, e non sono gli stessi tre riordinati: sono due misture nuove,
ciascuna decisa da una riga della tabella. Questo è tutto ciò che significa
moltiplicare i dati per una matrice, e lo si può ripetere: la lista che esce
può entrare in un'altra tabella. È così che ogni strato di una rete neurale
"riscrive" ciò che riceve prima di passarlo allo strato dopo, e i pesi delle
righe sono esattamente ciò che l'addestramento va a regolare.

Il conto appena fatto porta con sé due avvertenze. Ogni riga di pesi deve
avere un numero per ciascuna voce dell'appartamento, né uno di più né uno di
meno: una riga scritta per quattro voci non si combina con una lista di tre,
come uno scontrino con un prezzo a cui non corrisponde nessuna quantità, e il
totale non si chiude. E quando le tabelle si mettono in fila, l'ordine conta:
passare i dati prima in una tabella e poi nell'altra, o al contrario, dà quasi
sempre risultati diversi, come vestirsi (le calze prima delle scarpe funziona,
nell'ordine inverso no).

`````

`````{tab} Superiore

Una matrice $\mathbf{A}\in\mathbb{R}^{m\times n}$ ha $m$ righe e $n$ colonne.
Il **prodotto matrice-vettore** $\mathbf{A}\mathbf{x}$ produce un vettore di
$\mathbb{R}^m$ le cui componenti sono i prodotti scalari tra le righe di
$\mathbf{A}$ e $\mathbf{x}$:

$$
(\mathbf{A}\mathbf{x})_i = \sum_{j=1}^{n} A_{ij}\,x_j .
$$

(Il grassetto distingue l'oggetto intero dai suoi elementi: $\mathbf{A}$ è la
matrice, $A_{ij}$ il numero che sta all'incrocio fra riga $i$ e colonna $j$;
maiuscolo grassetto per le matrici, minuscolo grassetto per i vettori, tondo
per i numeri singoli.)

Il **prodotto tra matrici** $\mathbf{C} = \mathbf{A}\mathbf{B}$, con
$\mathbf{A}\in\mathbb{R}^{m\times k}$ e $\mathbf{B}\in\mathbb{R}^{k\times n}$,
dà $\mathbf{C}\in\mathbb{R}^{m\times n}$ con

$$
C_{ij} = \sum_{r=1}^{k} A_{ir} B_{rj} .
$$

Non è commutativo ($\mathbf{A}\mathbf{B}\neq\mathbf{B}\mathbf{A}$ in generale)
e le dimensioni "interne" devono combaciare. Uno strato *fully-connected* di
una rete non è altro che
$\mathbf{h} = \sigma(\mathbf{W}\mathbf{x}+\mathbf{b})$: una moltiplicazione
per la matrice dei pesi $\mathbf{W}$, seguita da una non linearità $\sigma$.
Il fatto che tante operazioni si riducano a prodotti tra matrici è ciò che
rende le GPU (nate per moltiplicare matrici in grafica) così efficaci nel
deep learning.

`````

Un caso merita di essere guardato da vicino: quello in cui i numeri che escono
sono **tanti quanti quelli che entrano**, e in particolare due e due, perché
allora si può disegnare. L'ingresso è una freccia sul foglio, l'uscita è
un'altra freccia sullo stesso foglio, e la matrice diventa un gesto: prende
ogni punto del piano e lo sposta altrove. Applicandola a molte frecce insieme
si vede che cosa fa davvero quella tabella di numeri, e di solito fa due cose
in una: **gira** le frecce e le **stira**, allungandone alcune e accorciandone
altre.

Ma non tutte le direzioni vengono girate: alcune resistono.

```{figure} ../figures/autovettori.svg
:name: fig-autovettori
:alt: "Animazione: sedici frecce tutte lunghe uno, che partono dall'origine in sedici direzioni diverse, vengono trasformate da una matrice; quasi tutte cambiano direzione, mentre le due sulle diagonali restano sulla propria retta, una allungandosi di tre volte e l'altra senza muoversi."
:width: 85%

La matrice $\mathbf{A}=\begin{pmatrix}2&1\\1&2\end{pmatrix}$ applicata a
sedici frecce tutte lunghe uno (si dicono *vettori unitari*: puntano in
direzioni diverse ma hanno tutte la stessa lunghezza, così l'unica cosa che
cambia fra loro è la direzione). Quasi tutte ruotano; le due sulle diagonali
(in terracotta) restano sulla propria retta: una si allunga di $3$ volte,
l'altra non si muove affatto. La riga scritta nel disegno,
$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$, si legge: «applicare la matrice a
quella freccia dà la stessa freccia moltiplicata per un numero», e quel numero
è il $3$ o l’$1$.
```

Il $3$ e l’$1$ non sono da prendere per buoni, sono due conti come quelli
dell'appartamento. Nella didascalia la tabella è scritta stretta fra due
parentesi, che è il modo consueto di scriverla: la prima riga è $2$ e $1$, la
seconda è $1$ e $2$. E la diagonale che sale è la freccia $(1,1)$, cioè quella
che avanza di un passo verso destra e di uno verso l'alto, così che i due
numeri restino uguali.

Applicarle la tabella vuol dire fare due volte lo scontrino: prima riga,
$2\cdot 1 + 1\cdot 1 = 3$; seconda riga, $1\cdot 1 + 2\cdot 1 = 3$. Ne esce
$(3,3)$, che è la stessa freccia moltiplicata per tre. La diagonale che scende
è $(1,-1)$, un passo a destra e uno in basso: prima riga,
$2\cdot 1 + 1\cdot(-1) = 1$; seconda riga, $1\cdot 1 + 2\cdot(-1) = -1$. Ne
esce $(1,-1)$, cioè sé stessa. Provando invece una freccia qualsiasi, per dire
$(1,0)$, si ottiene $(2,1)$, che punta da un'altra parte: quella è stata
girata.

## Autovalori e autovettori: le direzioni che resistono

Quelle due direzioni sono gli **autovettori** di $\mathbf{A}$, e i fattori $3$
e $1$ i suoi **autovalori**.

`````{tab} Elementare

Una tavola di legno si spacca con niente lungo la venatura, e resiste di
traverso. Le venature stanno nella tavola, decise dall'albero molto prima che
qualcuno la segasse, e non si spostano a seconda di dove batti. A cambiare è
soltanto *come* la tavola risponde.

La matrice è la tavola, e la freccia a cui la applichiamo è il punto in cui
batti. Le venature di una matrice si chiamano autovettori e stanno nella
tabella di numeri una volta per tutte. Lungo una venatura la trasformazione si
riduce a moltiplicare la freccia per un numero, e lo stirare di qua e il
comprimere di là spariscono. La freccia può allungarsi, accorciarsi, perfino
ribaltarsi, ma dalla sua retta non esce.

Quel fattore di allungamento si chiama autovalore e si scrive con la lettera
greca $\lambda$ (si legge «lambda»). Un $\lambda$ sopra $1$ allunga tutto ciò
che punta di lì, e in {numref}`fig-autovettori` una venatura ha $\lambda = 3$
e triplica, mentre l'altra ha $\lambda = 1$ e lascia le frecce come le trova.
Fra $0$ e $1$ le accorcia. Sotto zero le volta dall'altra parte, tanto più
lunghe quanto più grande è quel numero preso senza il segno.

Certe tavole una venatura non ce l'hanno. Una tabella può far girare tutte le
frecce dello stesso angolo, come le lancette di un orologio, e allora nessuna
resta sulla propria retta e sul foglio non c'è niente da indicare. I conti una
risposta la danno lo stesso, con numeri di un'altra specie che qui non servono.
Le tabelle che il machine learning incontra più spesso sono venate, e con le
venature perpendicolari fra loro.

Il guaio comincia con cento tavole tagliate dallo stesso tronco, venate tutte
uguali, che il colpo attraversa una dopo l'altra, come un segnale che ripassa
cento volte per lo stesso strato. Gli allungamenti si moltiplicano fra loro. Un
$1{,}1$ ripetuto cento volte fa $13\,781$, un $0{,}9$ fa $0{,}000027$, numeri
enormi da un capo e indistinguibili da zero dall'altro.

Una rete profonda è una pila di cento tavole incollate una sull'altra, che è
poi il modo in cui si fa il compensato. All'andata il colpo entra dalla prima e
attraversa tutta la pila, in fondo si guarda quanto il pezzo è venuto storto, e
la correzione risale tavola per tavola dicendo a ognuna come dovrà essere
venata la prossima volta. Quel messaggio di ritorno si chiama *gradiente*, ed è
l'argomento della prossima sezione. Moltiplicato a ogni tavola per poco più di
uno, alla prima arriva gonfiato a dismisura; per poco meno di uno non ci arriva
affatto. In gergo i gradienti esplodono o svaniscono, ed è fra le prime cose
che si guardano quando un addestramento smette di migliorare.

Le cento tavole, però, non hanno la stessa venatura, ed è qui che l'immagine
viene usata a sproposito più spesso. Ogni strato ha la sua matrice, e quello
che si accumula non è un solo $\lambda$ elevato a cento ma il prodotto di cento
allungamenti diversi. Resta vera l'idea generale, che in una catena lunga basta
poco perché cento moltiplicazioni portino lontanissimo; il numero da guardare
non è l'autovalore di una singola matrice, ed è un conto che il capitolo sulle
reti neurali rifà per esteso.

`````

`````{tab} Superiore

Dato $\mathbf{A}\in\mathbb{R}^{n\times n}$, un vettore non nullo $\mathbf{v}$
è un **autovettore** con **autovalore** $\lambda$ se

$$
\mathbf{A}\mathbf{v} = \lambda\mathbf{v}.
$$

Riscrivendo come $(\mathbf{A}-\lambda \mathbf{I})\mathbf{v}=\mathbf{0}$, dove
$\mathbf{I}$ è la **matrice identità** (uno sulla diagonale e zero altrove:
quella che moltiplicando non cambia niente), la soluzione non banale esiste
solo se $\mathbf{A}-\lambda \mathbf{I}$ è **singolare**, cioè non invertibile,
cioè se

$$
\det(\mathbf{A} - \lambda \mathbf{I}) = 0 .
$$

È l’**equazione caratteristica**, un polinomio di grado $n$ in $\lambda$: una
matrice $n\times n$ ha quindi $n$ autovalori nel campo complesso, contati con
molteplicità. Per la matrice della figura,
$\det\!\begin{pmatrix}2-\lambda&1\\1&2-\lambda\end{pmatrix}
=(2-\lambda)^2-1=0$ dà $\lambda_1=3$ e $\lambda_2=1$.

Il caso che ricorre di più nel machine learning è quello **simmetrico**
($\mathbf{A}=\mathbf{A}^\top$): il *teorema spettrale* garantisce che gli
autovalori siano reali e che esista una base ortonormale di autovettori, cioè
$\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ con $\mathbf{Q}$
ortogonale e $\boldsymbol{\Lambda}$ diagonale. Le matrici di covarianza sono
simmetriche, ed è precisamente questa decomposizione che la **PCA** calcola:
gli autovettori danno le direzioni di massima varianza, gli autovalori quanta
varianza c'è lungo ciascuna.

```{figure} ../figures/pca-in-pratica.svg
:name: fig-pca-assi-principali
:alt: "Una nube di punti allungata in diagonale, con sovrapposti i due assi principali: il primo orientato lungo la direzione in cui i punti si sparpagliano di più, il secondo perpendicolare a esso e molto più corto, lungo la direzione di varianza minima."
:width: 82%

Gli autovettori della covarianza, disegnati. Il primo asse non è né
l'orizzontale né la verticale: è la direzione che i dati stessi indicano.
```

{numref}`fig-pca-assi-principali` mostra perché la PCA sia una faccenda di
autovettori e non di scelta fra le colonne originali. Le direzioni buone
quasi mai coincidono con gli assi in cui i dati sono stati registrati; il
teorema spettrale garantisce che esistano, siano ortogonali fra loro, e si
possano ordinare per quanta varianza catturano.

**La decomposizione ai valori singolari.** Il teorema spettrale chiede la
simmetria, e quindi anche la quadratura. Esiste una decomposizione che non
chiede niente, e che servirà più spesso: ogni matrice
$\mathbf{A}\in\mathbb{R}^{m\times n}$, rettangolare o quadrata, singolare o
no, si scrive come

$$
\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top ,
$$

con $\mathbf{U}\in\mathbb{R}^{m\times m}$ e $\mathbf{V}\in\mathbb{R}^{n\times
n}$ ortogonali e $\boldsymbol{\Sigma}$ "diagonale" con elementi
$\sigma_1\ge\sigma_2\ge\dots\ge 0$, i **valori singolari**. È la
decomposizione ai valori singolari (*singular value decomposition*, SVD).
Letta a destra-verso-sinistra dice che ogni trasformazione lineare è una
rotazione, seguita da una dilatazione lungo assi ortogonali, seguita da
un'altra rotazione. I due valori estremi hanno un significato immediato:
$\sigma_1 = \max_{\lVert\mathbf{x}\rVert=1}\lVert\mathbf{A}\mathbf{x}\rVert$ è
di quanto al massimo la matrice allunga un vettore, $\sigma_{\min}$ di quanto
al minimo (per una matrice con almeno tante righe quante colonne: altrimenti
c'è sempre una direzione che viene annullata), e il **rango** è il numero dei
$\sigma_i$ non nulli. Per una matrice simmetrica i valori singolari sono i
moduli degli autovalori; per una matrice quadrata qualsiasi le due famiglie non
coincidono, ma non sono nemmeno estranee: il prodotto dei moduli è lo stesso
(entrambe le famiglie danno $|\det\mathbf{A}|$) e soprattutto
$\sigma_{\max}\ge|\lambda|_{\max}$, cioè l'allungamento massimo non è mai
inferiore al **modulo** dell'autovalore più grande. È una
disuguaglianza che può essere larghissima, e proprio in quella distanza sta
il fenomeno più interessante. Tornerà nella sezione
di analisi numerica (dove il numero di condizionamento è il rapporto
$\sigma_{\max}/\sigma_{\min}$), nel capitolo sulle reti neurali (la
"grandezza" di una Jacobiana) e in quello sui sistemi di raccomandazione
(l'approssimazione di rango basso di una matrice di valutazioni).

L'iterazione chiarisce il resto: $\mathbf{A}^k\mathbf{v} =
\lambda^k\mathbf{v}$, quindi il comportamento asintotico di un sistema che
applica *sempre la stessa* matrice è governato dall'autovalore di modulo
massimo (il *raggio spettrale* $\rho(\mathbf{A})$). Se $\rho(\mathbf{A})<1$
ogni vettore collassa a zero; se $\rho(\mathbf{A})>1$ diverge ogni vettore
generico, cioè con componente non nulla lungo l'autodirezione dominante. Lo
stesso argomento, applicato alla matrice dei link del web (resa stocastica per
colonne e corretta con il *teletrasporto* del fattore di smorzamento), è il
**PageRank**: sotto quelle ipotesi il teorema di Perron–Frobenius garantisce
che l'autovalore $1$ sia semplice e dominante, e l'ordinamento delle pagine è
il suo autovettore, l'unica direzione che la trasformazione lascia esattamente
com'è.

Diciamo subito dove questo argomento non arriva, perché è il
punto in cui viene applicato più spesso a sproposito. Il gradiente che
attraversa una rete profonda non è $\mathbf{A}^k$: è un prodotto di matrici
**diverse** (le Jacobiane dei singoli strati), e gli autovalori di un prodotto
non si ricavano dagli autovalori dei fattori. Di più: anche con una
sola matrice, $\rho(\mathbf{A})<1$ garantisce soltanto il comportamento
*asintotico*. Per $\mathbf{A} = \begin{pmatrix}0{,}9 & 100\\ 0 &
0{,}9\end{pmatrix}$, che ha $\rho = 0{,}9$, un vettore unitario arriva a norma
$387$ dopo dieci applicazioni prima di cominciare a scendere: le matrici non
**normali** ($\mathbf{A}\mathbf{A}^\top \neq \mathbf{A}^\top\mathbf{A}$) hanno
un transitorio che il raggio spettrale non vede, e una rete di qualche decina
di strati vive tutta lì dentro. La grandezza giusta per i gradienti che
esplodono o svaniscono è quindi la norma del prodotto, cioè i valori
singolari, ed è il conto che rifà il capitolo sulle reti neurali.

`````

## Norme: misurare lunghezze ed errori

È la promessa lasciata in sospeso parlando del prodotto scalare: quanto è
lunga una freccia. La risposta è più utile di quanto sembri, perché la stessa
domanda, fatta alla freccia che va dalla risposta giusta alla risposta del
modello, misura di quanto il modello ha sbagliato.

`````{tab} Elementare

La **norma** di un vettore è la sua "lunghezza". Per una freccia nel piano è
il buon vecchio teorema di Pitagora: radice quadrata della somma dei quadrati
delle componenti. La freccia $(3, 4)$, per esempio, è lunga
$\sqrt{3^2 + 4^2} = \sqrt{25} = 5$. Ci serve soprattutto per misurare
quanto un modello sbaglia: se la risposta giusta è un vettore e la
previsione è un altro vettore, si fa la loro differenza (quella voce per
voce di poco fa) e se ne misura la lunghezza. Quel numero è l'errore.

La stessa mossa dà anche la **distanza** fra due vettori qualsiasi: quanto
sono lontani due punti è la lunghezza della freccia che va dall'uno all'altro,
cioè la norma della loro differenza. È il conto che si fa ogni volta che si
dice che due parole, due canzoni o due clienti "si somigliano".

Un'ultima avvertenza sui conti veri. Al posto della lunghezza si usa quasi
sempre il suo quadrato, cioè la lunghezza moltiplicata per sé stessa, perché
toglie la radice di mezzo e i conti vengono più comodi. Il quadrato però
cambia la scala: un errore lungo $5$ pesa $25$, uno lungo $10$ pesa $100$,
quindi un errore doppio pesa quattro volte tanto. Comodo, finché lo si tiene
a mente: quel numero non è più una distanza, e gli errori grandi contano
molto più che in proporzione.

`````

`````{tab} Superiore

La norma euclidea (o $\ell_2$) di $\mathbf{x}\in\mathbb{R}^n$ è

$$
\lVert\mathbf{x}\rVert_2 = \sqrt{\sum_{i=1}^{n} x_i^2}
= \sqrt{\mathbf{x}^\top\mathbf{x}} .
$$

Accanto a essa usiamo spesso la norma $\ell_1$ (somma dei valori assoluti,
$\lVert\mathbf{x}\rVert_1=\sum_i |x_i|$), che nella regolarizzazione promuove
soluzioni *sparse*.

Da una norma si ricava una **distanza**:

$$
d(\mathbf{x},\mathbf{y}) = \lVert \mathbf{x}-\mathbf{y}\rVert_2 .
$$

È ciò che si misura quando si dice che due *embedding* sono vicini, ed è una
distanza in senso proprio: non negativa, nulla solo se $\mathbf{x}=\mathbf{y}$,
simmetrica, e soggetta alla disuguaglianza triangolare. Il **quadrato** della
norma, comodo perché deriva bene e toglie la radice, distanza non è:
raddoppiando lo spostamento quadruplica, e la disuguaglianza triangolare
salta.

Proprio il quadrato è però ciò che compare nella celebre loss dell’**errore
quadratico medio**:

$$
\mathcal{L} = \frac{1}{m}\sum_{i=1}^{m}
\lVert \hat{\mathbf{y}}^{(i)} - \mathbf{y}^{(i)} \rVert_2^2 .
$$

Qui $m$ è il numero di esempi (poco sopra la stessa lettera contava le righe
di una matrice), $\mathbf{y}^{(i)}$ è l'uscita vera dell'esempio $i$-esimo e
$\hat{\mathbf{y}}^{(i)}$ quella predetta dal modello.

(Scritta così la media è sui soli $m$ esempi. Con uscite vettoriali le
librerie mediano anche sulle componenti: `mean_squared_error` di scikit-learn
e `nn.MSELoss` con `reduction='mean'` dividono per $m\cdot d$, dove $d$ è
quante componenti ha ciascuna uscita, quindi il loro numero differisce da
questo per un fattore $d$. Il minimo è lo stesso, il
valore stampato no.)

Norme e prodotti scalari sono legati da $\lVert\mathbf{x}\rVert_2^2 =
\mathbf{x}^\top\mathbf{x}$: misurare una lunghezza è fare il prodotto scalare
di un vettore con sé stesso.

`````

## In pratica, con NumPy

In Python l'algebra lineare vive in **NumPy**, la cassetta di funzioni già
pronte di cui parla il {doc}`capitolo su Python </Python/overview>` (in gergo una cassetta così si chiama
*libreria*). Qui basta il richiamo: le operazioni di sopra sono una riga
ciascuna.

Chi non ha mai programmato può leggere solo i commenti, cioè il testo dopo il
cancelletto: dicono in italiano quello che ogni riga fa.

```python
import numpy as np

a = np.array([4, 2])
b = np.array([1, 3])

a @ b                     # prodotto scalare -> 10
np.linalg.norm(a)         # norma euclidea di a -> 4.472...

W = np.array([[0.2, 0.8],
              [-0.5, 0.1]])
W @ a                     # prodotto matrice-vettore -> array([2.4, -1.8])
```

L'operatore `@` è il prodotto matriciale: la stessa notazione vale per
prodotto scalare, matrice-vettore e matrice-matrice, perché per NumPy sono
tutti casi della stessa operazione.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un **vettore** è una lista ordinata di numeri che descrive un esempio
  (l'appartamento: metri quadri, stanze, piano); una **matrice** è una tabella
  che impila tanti esempi oppure li trasforma, ed è la mossa che ogni strato di
  una rete ripete sui dati che riceve.
- Il **prodotto scalare** moltiplica due liste voce per voce e somma tutto,
  come lo scontrino della spesa (le quantità per i prezzi, e viene fuori il
  totale). Il numero che ne esce dice se i due vettori vanno d'accordo: grande
  se puntano dalla stessa parte, zero se sono perpendicolari, negativo se
  opposti. È il conto che fa un singolo neurone.
- Gli **autovettori** sono le venature del legno di una matrice: le direzioni
  che la trasformazione non devia, e lungo cui si limita ad allungare o
  accorciare di un fattore fisso, l’**autovalore** $\lambda$. Applicando cento
  volte la *stessa* matrice quel fattore si moltiplica per sé stesso, e basta
  poco perché il risultato scappi via: un fattore appena sopra l'uno fa
  esplodere tutto, uno appena sotto lo fa svanire. In una rete vera le matrici
  sono diverse a ogni strato, quindi va tenuta l'idea (gli effetti si
  moltiplicano lungo la catena) e non il numero.
- La **norma** è la lunghezza di una freccia (il teorema di Pitagora sulle sue
  componenti) e serve soprattutto a misurare l’**errore** di un modello:
  quanto è lunga la differenza fra la risposta giusta e la previsione. La
  stessa lunghezza, applicata alla differenza di due vettori qualsiasi, è la
  loro **distanza**: quanto si somigliano.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Un vettore rappresenta un esempio (le sue *feature*); una matrice impila
  esempi oppure trasforma i dati.
- Il **prodotto scalare** misura l'allineamento tra due vettori ed è il cuore
  del singolo neurone: $\mathbf{w}^\top\mathbf{x}+b$.
- Gli **autovettori** sono le direzioni che una matrice non devia
  ($\mathbf{A}\mathbf{v}=\lambda\mathbf{v}$), e i loro **autovalori** dicono
  di quanto le allunga. Iterando *la stessa* matrice
  ($\mathbf{A}^k\mathbf{v}=\lambda^k\mathbf{v}$) comanda il raggio spettrale;
  per un prodotto di matrici **diverse**, come le Jacobiane di una rete, il
  raggio spettrale non basta e la grandezza da guardare è la norma.
- Ogni matrice, anche rettangolare, si decompone come
  $\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ (**SVD**): i
  **valori singolari** $\sigma_i$ dicono di quanto la matrice allunga al
  massimo e al minimo, e quanti sono i non nulli è il rango.
- La **norma** $\lVert\mathbf{x}\rVert_2=\sqrt{\mathbf{x}^\top\mathbf{x}}$
  misura lunghezze e, soprattutto, l’**errore** di un modello; la
  **distanza** $\lVert\mathbf{x}-\mathbf{y}\rVert_2$ che ne discende è ciò
  che si intende quando si dice che due embedding sono vicini.
```
`````
