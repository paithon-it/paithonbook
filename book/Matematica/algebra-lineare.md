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
riga dopo l'altra viene una lista di $28 \times 28 = 784$ numeri, ed è
l'esempio che ritorna spesso in questo libro. Quando si dirà
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

Prendi due vettori e chiediti: *puntano più o meno nella stessa direzione?*
Il prodotto scalare risponde con un numero solo. Lo calcoli moltiplicando le
componenti che occupano lo stesso posto e sommando tutto:

$$
\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + \dots + a_n b_n .
$$

I numerini in basso sono solo un modo di dire «il posto»: $a_1$ è il primo
numero della lista $\mathbf{a}$, $a_2$ il secondo, e $n$ è quanti sono in
tutto. La riga, letta a voce, dice: *primo per primo, più secondo per secondo,
e così via fino alla fine*.

Per esempio, con $\mathbf{a}=(4,2)$ e $\mathbf{b}=(1,3)$:

$$
\mathbf{a} \cdot \mathbf{b} = 4\cdot 1 + 2\cdot 3 = 4 + 6 = 10 .
$$

Un'immagine quotidiana: **lo scontrino della spesa**. Un vettore contiene le
quantità (2 chili di pane, 1 litro di latte), l'altro i prezzi al chilo o al
litro; il prodotto scalare è il totale da pagare. Qualsiasi "totale pesato"
che hai mai calcolato era un prodotto scalare sotto mentite spoglie. Con mille
dimensioni invece di due non cambia niente: le moltiplicazioni da sommare
diventano mille.

Il nome viene dall'esito. La somma di due vettori restituisce un vettore,
questa operazione restituisce uno *scalare*, un numero singolo. È una
compressione radicale (da duemila numeri a uno) e proprio per questo è
preziosa: condensa la relazione fra due oggetti complicati in una cifra
confrontabile.

**Perché quel numero dice qualcosa sulla direzione?** Serve un'immagine
geometrica ({numref}`fig-prodotto-scalare`). Immagina il vettore $\mathbf{b}$
come una strada dritta, e il vettore $\mathbf{a}$ come un bastone piantato
obliquo all'inizio della strada, col sole a picco. Il bastone proietta un'ombra
sull'asfalto: quell'ombra è la parte di $\mathbf{a}$ che va nella stessa
direzione della strada. Il
prodotto scalare è la lunghezza dell'ombra moltiplicata per la lunghezza della
strada. (Se il bastone pende anche di lato, l'ombra non cade più tutta sulla
carreggiata ma un po’ di sbieco: quel che conta è solo quanto ne avanza *lungo*
la strada, e il pezzo di traverso non entra nel conto. È la stessa cosa che
facciamo quando diciamo che di un viaggio verso sud-est «tanti chilometri sono
verso sud».)

Se il bastone è quasi sdraiato lungo la strada, l'ombra è lunga e il risultato
è **grande e positivo**. Se è perpendicolare, l'ombra si riduce a un punto:
**zero**. Se punta all'indietro, l'ombra cade sull'altro lato del bastone, e
allora contiamo la sua lunghezza col segno meno, perché quel che ci interessa
non è quanto è lunga l'ombra ma da che parte cade: il numero diventa
**negativo**. Tre casi, tre verdetti: concordi, indifferenti, opposti.

Attenzione a non chiedere all'immagine più di quel che dà. Il prodotto scalare
dipende da due cose insieme: da **quanto** i due vettori sono lunghi e da
**quanto sono d'accordo**. Allungando la strada il totale cresce, senza che
nessuno abbia cambiato direzione. Il segno, invece, dipende solo dall'accordo,
e resta lo stesso comunque si allunghino le due frecce.

Quando si vogliono confrontare *solo* le direzioni, allora, si divide il
prodotto scalare per le lunghezze delle due frecce. È la stessa mossa che si fa
in classe per passare da un punteggio a una percentuale: si divide per il
massimo possibile, così quello che resta non dipende più da quanto era grande
il totale. Qui il massimo possibile è proprio il prodotto delle due lunghezze
(lo si tocca quando le frecce puntano esattamente nella stessa direzione), e
dividere per quello lascia solo l'accordo.

Quanto è lunga una freccia lo si calcola col teorema di Pitagora, ed è la
**norma** di cui parla l'ultima sezione di questa pagina; qui basta il
risultato. Con $\mathbf{a}=(4,2)$ e $\mathbf{b}=(1,3)$, che sono le due liste
di poco fa, le lunghezze valgono $\sqrt{4^2+2^2}=\sqrt{20}\approx 4{,}47$ e
$\sqrt{1^2+3^2}=\sqrt{10}\approx 3{,}16$; il prodotto scalare faceva $10$, e
diviso per le due lunghezze dà $10 / (4{,}47 \cdot 3{,}16) \approx 0{,}71$.
Quel numero sta fra $-1$ e $1$ e non cambia se si allungano le frecce: è il
coseno della figura, cioè l'accordo puro fra le due direzioni, ed è la
quantità che nel resto del libro si chiama **similarità del coseno**.

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

`````{tab} Elementare

Una matrice è una tabella di numeri: righe e colonne. Ci serve per due cose.
Primo, **impilare tanti esempi**: se ho 100 appartamenti descritti da 3 numeri
ciascuno, ottengo una tabella $100\times 3$. Secondo, **trasformare** i dati,
ed è qui che vale la pena rallentare, perché «moltiplicare una lista di numeri
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
matrice, $A_{ij}$ il numero che sta all'incrocio fra riga $i$ e colonna $j$.
È la convenzione che il libro segue ovunque: maiuscolo grassetto per le
matrici, minuscolo grassetto per i vettori, tondo per i numeri singoli.)

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

L'analogia più onesta è la **venatura del legno**. Una tavola si comporta in
modo diverso a seconda della direzione: lungo la venatura si spacca con niente,
di traverso resiste. E la venatura appartiene alla tavola: è una proprietà di
quel pezzo di legno, decisa dall'albero molto prima che qualcuno lo lavorasse,
e non cambia a seconda di dove provi a colpire. Quello che cambia, a seconda di
dove provi, è soltanto *come risponde*.

La matrice è la tavola, e la freccia a cui la applichiamo è il punto in cui
provi. Le direzioni buone sono già dentro la tabella di numeri, una volta per
tutte; la freccia non le crea, le incontra o non le incontra. Gli autovettori
sono le venature di una matrice: le direzioni lungo cui la trasformazione
agisce nel modo più semplice possibile.

Semplice quanto? Applicare la matrice a un vettore che sta su una venatura
equivale a moltiplicarlo per un numero. Tutta la complessità del gesto (stira
di qua, comprime di là) lungo quella direzione collassa in una
moltiplicazione. Il vettore può allungarsi, accorciarsi, perfino ribaltarsi,
ma **non lascia la sua retta** ({numref}`fig-autovettori`).

Quel fattore di allungamento ha un nome e un simbolo: si chiama **autovalore**
e si scrive con la lettera greca $\lambda$ (si legge «lambda»). Nella figura
di sopra $\lambda$ vale $3$ per una delle due venature e $1$ per l'altra.
Sapere quanto vale è una specie di oroscopo per quella direzione:

- $\lambda > 1$: tutto ciò che punta di lì viene **amplificato**;
- $\lambda = 1$: resta **identico**, è il caso della seconda venatura del
  disegno;
- $0 < \lambda < 1$: viene **attenuato**, cioè accorciato;
- $\lambda < 0$: viene **ribaltato** dalla parte opposta, e per giunta
  allungato o accorciato a seconda di quanto quel numero sia grande in valore
  assoluto.

Il punto diventa serio quando è **sempre la stessa** matrice ad applicarsi
cento volte, come a un segnale che ripassa cento volte per lo stesso strato.
L'effetto si accumula: $\lambda$ moltiplicato per sé stesso cento volte. Un
$1{,}1$ diventa $13\,781$, un $0{,}9$ diventa $0{,}000027$. Da un capo si
finisce con numeri enormi, dall'altro con numeri indistinguibili da zero.

E perché sia un guaio bisogna anticipare una cosa sola sulle prossime pagine.
Addestrare un modello è un viaggio di andata e ritorno: all'andata i numeri
attraversano gli strati e ne esce una risposta, al ritorno la correzione rifà
la stessa strada al contrario, dall'ultimo strato fino al primo, per dire a
ogni manopola di quanto girare. Quel segnale di ritorno si chiama *gradiente*,
ed è l'argomento della prossima sezione. Se lungo il percorso viene
moltiplicato cento volte per un numero poco più grande di uno arriva ai primi
strati gonfiato a dismisura; se il numero è poco più piccolo di uno arriva
ridotto a zero. In gergo si dice che «i gradienti esplodono o svaniscono», ed è
una delle prime cose che si vanno a guardare quando un addestramento smette di
migliorare.

Una cautela conviene portarsela dietro fin d'ora, perché è il punto in cui
questa immagine viene usata più spesso a sproposito. In una rete profonda la
matrice non è sempre la stessa: ogni strato ha la sua, e quello che si accumula
non è un solo $\lambda$ elevato a cento, ma il prodotto di cento allungamenti
diversi. Da tenere a mente, quindi, è l'idea, non il numero: gli effetti si
moltiplicano lungo la catena, e basta poco perché cento moltiplicazioni portino
lontanissimo. Quale numero vada guardato davvero, invece, non è l'autovalore di
una singola matrice, ed è un conto che il capitolo sulle reti neurali rifà per
esteso.

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
chiede niente, e nel resto del libro serve più spesso: ogni matrice
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
disuguaglianza che può essere larghissima, e fra poche righe si vedrà che
proprio in quella distanza sta il fenomeno più interessante. Tornerà nella sezione
di analisi numerica (dove il numero di condizionamento è il rapporto
$\sigma_{\max}/\sigma_{\min}$), nel capitolo sulle reti neurali (la
"grandezza" di una Jacobiana) e in quello sui sistemi di raccomandazione
(l'approssimazione di rango basso di una matrice di valutazioni).

L'iterazione chiarisce il resto: $\mathbf{A}^k\mathbf{v} =
\lambda^k\mathbf{v}$, quindi il comportamento asintotico di un sistema che
applica **sempre la stessa** matrice è governato dall'autovalore di modulo
massimo (il *raggio spettrale* $\rho(\mathbf{A})$). Se $\rho(\mathbf{A})<1$
ogni vettore collassa a zero; se $\rho(\mathbf{A})>1$ diverge ogni vettore
generico, cioè con componente non nulla lungo l'autodirezione dominante. Lo
stesso argomento, applicato alla matrice dei link del web (resa stocastica per
colonne e corretta con il *teletrasporto* del fattore di smorzamento), è il
**PageRank**: sotto quelle ipotesi il teorema di Perron–Frobenius garantisce
che l'autovalore $1$ sia semplice e dominante, e l'ordinamento delle pagine è
il suo autovettore, l'unica direzione che la trasformazione lascia esattamente
com'è.

Diciamo subito dove questo argomento **non** arriva, perché è il
punto in cui viene applicato più spesso a sproposito. Il gradiente che
attraversa una rete profonda non è $\mathbf{A}^k$: è un prodotto di matrici
**diverse** (le Jacobiane dei singoli strati), e gli autovalori di un prodotto
non hanno alcun rapporto con gli autovalori dei fattori. Di più: anche con una
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

`````{tab} Elementare

La **norma** di un vettore è la sua "lunghezza". Per una freccia nel piano è
il buon vecchio teorema di Pitagora: radice quadrata della somma dei quadrati
delle componenti. La freccia $(3, 4)$, per esempio, è lunga
$\sqrt{3^2 + 4^2} = \sqrt{25} = 5$. Ci serve soprattutto per misurare
**quanto un modello sbaglia**: se la risposta giusta è un vettore e la
previsione è un altro vettore, si fa la loro differenza (quella voce per
voce di poco fa) e se ne misura la lunghezza. Quel numero è l'errore.

La stessa mossa dà anche la **distanza** fra due vettori qualsiasi: quanto
sono lontani due punti è la lunghezza della freccia che va dall'uno all'altro,
cioè la norma della loro differenza. È il conto che si fa ogni volta che si
dice che due parole, due canzoni o due clienti "si somigliano".

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

Da una norma si ricava una **distanza**, ed è il ponte che il resto del libro
dà per costruito:

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

Qui $m$ è il numero di esempi (in questa pagina la stessa lettera contava le
righe di una matrice: da qui in avanti, e nel resto del libro, conta gli
esempi), $\mathbf{y}^{(i)}$ è l'uscita vera dell'esempio $i$-esimo e
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
pronte di cui parla il capitolo su Python (in gergo una cassetta così si chiama
*libreria*). Qui basta il richiamo: le operazioni di sopra sono una riga
ciascuna.

Ogni sezione di questo capitolo si chiude con un blocco di codice come questo,
e vale per tutti la stessa avvertenza: **chi non ha mai programmato può leggere
solo i commenti**, cioè il testo dopo il cancelletto, e tirare dritto senza
perdersi niente. Il codice serve a chi vuole rifare i conti da sé.

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
