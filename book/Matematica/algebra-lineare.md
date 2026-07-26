# Algebra lineare: vettori, matrici, prodotti

Tutto, nel machine learning, comincia con l'idea di mettere i numeri in fila.
Un'email è una lista di parole, una foto è una griglia di pixel, un cliente è
una scheda di attributi (età, acquisti, città). In tutti questi casi facciamo
la stessa mossa: impilare i numeri in un **vettore**, e impilare i vettori in
una **matrice**. L'algebra lineare è la grammatica di queste pile.

## Vettori: dati e direzioni

`````{tab} Elementare

Un vettore è semplicemente una lista ordinata di numeri. Se descrivo un
appartamento con tre numeri — metri quadri, numero di stanze, piano — ho già
un vettore:

$$
x = (75,\ 3,\ 2)
$$

L'ordine conta: il primo posto è sempre "metri quadri", il secondo "stanze",
e così via. Possiamo immaginare un vettore di due o tre numeri come una
**freccia** che parte dall'origine e punta verso quel punto: ci dice una
*direzione* e una *lunghezza*. Con più di tre numeri la freccia non la
disegniamo più, ma l'idea di "direzione nello spazio" resta la stessa.

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

È l'operazione più importante di tutto il libro: un neurone artificiale, in
fondo, non fa altro che calcolare un prodotto scalare.

Vale la pena rendersi conto della scala. Ogni volta che un modello linguistico
genera una singola parola, da qualche parte in un datacenter vengono eseguiti
miliardi di esemplari della stessa operazione: prendere due liste di numeri,
moltiplicarle voce per voce, sommare tutto. Due vettori entrano, un numero solo
esce. Occupa mezza riga in un libro di matematica ed è, con ogni probabilità,
l'operazione aritmetica più eseguita sul pianeta in questo momento.

```{figure} ../figures/prodotto-scalare.svg
:name: fig-prodotto-scalare
:alt: "Proiezione geometrica: il vettore a proiettato sul vettore b, con l'angolo theta e la perpendicolare tratteggiata; la lunghezza della proiezione evidenziata in terracotta"
:width: 85%

La proiezione di $a$ su $b$ è l'"ombra" che $a$ getta sulla direzione di $b$.
Il prodotto scalare è la lunghezza di quest'ombra moltiplicata per la lunghezza
di $b$: l'angolo $\theta$ decide se il risultato è grande, nullo o negativo.
```

`````{tab} Elementare

Prendi due vettori e chiediti: *puntano più o meno nella stessa direzione?*
Il prodotto scalare risponde con un numero solo. Lo calcoli moltiplicando le
componenti che occupano lo stesso posto e sommando tutto:

$$
a \cdot b = a_1 b_1 + a_2 b_2 + \dots + a_n b_n .
$$

Per esempio, con $a=(4,2)$ e $b=(1,3)$:

$$
a \cdot b = 4\cdot 1 + 2\cdot 3 = 4 + 6 = 10 .
$$

Un'immagine quotidiana: **lo scontrino della spesa**. Un vettore contiene le
quantità (2 chili di pane, 1 litro di latte), l'altro i prezzi al chilo o al
litro; il prodotto scalare è il totale da pagare. Qualsiasi "totale pesato"
che hai mai calcolato era un prodotto scalare sotto mentite spoglie. Con mille
dimensioni invece di due non cambia niente: le moltiplicazioni da sommare
diventano mille.

Il nome viene dall'esito. La somma di due vettori restituisce un vettore,
questa operazione restituisce uno *scalare*, un numero singolo. È una
compressione radicale — da duemila numeri a uno — e proprio per questo è
preziosa: condensa la relazione fra due oggetti complicati in una cifra
confrontabile.

**Perché quel numero dice qualcosa sulla direzione?** Serve un'immagine
geometrica ({numref}`fig-prodotto-scalare`). Immagina il vettore $b$ come una
strada dritta, e il vettore $a$ come un bastone piantato obliquo all'inizio
della strada, col sole a picco. Il bastone proietta un'ombra sull'asfalto:
quell'ombra è la parte di $a$ che va nella stessa direzione della strada. Il
prodotto scalare è la lunghezza dell'ombra moltiplicata per la lunghezza della
strada.

Se il bastone è quasi sdraiato lungo la strada, l'ombra è lunga e il risultato
è **grande e positivo**. Se è perpendicolare, l'ombra si riduce a un punto:
**zero**. Se punta all'indietro, l'ombra cade dalla parte opposta e il numero
diventa **negativo**. Tre casi, tre verdetti: concordi, indifferenti, opposti.

Il prodotto scalare non misura quanto due vettori sono grandi: misura quanto
sono d'accordo.

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
ciascuno, ottengo una tabella $100\times 3$. Secondo, **trasformare** i dati:
moltiplicare i dati per una matrice significa mescolarli e rimapparli in un
nuovo spazio — è così che ogni strato di una rete neurale "riscrive" ciò che
riceve prima di passarlo allo strato dopo.

`````

`````{tab} Superiore

Una matrice $A\in\mathbb{R}^{m\times n}$ ha $m$ righe e $n$ colonne. Il
**prodotto matrice-vettore** $A\mathbf{x}$ produce un vettore di
$\mathbb{R}^m$ le cui componenti sono i prodotti scalari tra le righe di $A$ e
$\mathbf{x}$:

$$
(A\mathbf{x})_i = \sum_{j=1}^{n} A_{ij}\,x_j .
$$

Il **prodotto tra matrici** $C = AB$, con $A\in\mathbb{R}^{m\times k}$ e
$B\in\mathbb{R}^{k\times n}$, dà $C\in\mathbb{R}^{m\times n}$ con

$$
C_{ij} = \sum_{r=1}^{k} A_{ir} B_{rj} .
$$

Non è commutativo ($AB\neq BA$ in generale) e le dimensioni "interne" devono
combaciare. Uno strato *fully-connected* di una rete non è altro che
$\mathbf{h} = \sigma(W\mathbf{x}+\mathbf{b})$: una moltiplicazione per la
matrice dei pesi $W$, seguita da una non linearità $\sigma$. Il fatto che
tante operazioni si riducano a prodotti tra matrici è ciò che rende le GPU —
nate per moltiplicare matrici in grafica — così efficaci nel deep learning.

`````

Una matrice, dunque, gira e stira lo spazio. Ma non tutte le direzioni vengono
girate: alcune resistono.

```{figure} ../figures/autovettori.svg
:name: fig-autovettori
:alt: "Animazione: sedici vettori unitari vengono trasformati da una matrice; quasi tutti cambiano direzione, mentre i due sulle diagonali restano sulla propria retta, uno allungandosi di tre volte e l'altro senza muoversi."
:width: 85%

La matrice $A=\begin{pmatrix}2&1\\1&2\end{pmatrix}$ applicata a sedici
vettori unitari. Quasi tutti ruotano; i due sulle diagonali (in
terracotta) restano sulla propria retta: uno si allunga di $3$ volte, l'altro
non si muove affatto.
```

## Autovalori e autovettori: le direzioni che resistono

Quelle due direzioni sono gli **autovettori** di $A$, e i fattori $3$ e $1$ i
suoi **autovalori**.

`````{tab} Elementare

L'analogia più onesta è la **venatura del legno**. Un'asse si comporta in modo
diverso a seconda della direzione: lungo la venatura si piega e si spacca
facilmente, di traverso resiste. La venatura è una proprietà del materiale, non
del coltello che usi. Gli autovettori sono le venature di una matrice: le
direzioni lungo cui la trasformazione agisce nel modo più semplice possibile.

Semplice quanto? Applicare la matrice a un vettore che sta su una venatura
equivale a moltiplicarlo per un numero. Tutta la complessità del gesto — stira
di qua, comprime di là — lungo quella direzione collassa in una moltiplicazione.
Il vettore può allungarsi, accorciarsi, perfino ribaltarsi, ma **non lascia la
sua retta** ({numref}`fig-autovettori`).

Il numero $\lambda$ è una specie di oroscopo per quella direzione:

- $\lambda > 1$: tutto ciò che punta di lì viene **amplificato**;
- $0 < \lambda < 1$: viene **attenuato**;
- $\lambda < 0$: viene **ribaltato**.

Il punto diventa serio quando la matrice si applica non una volta ma cento —
come a un segnale che attraversa cento strati di una rete. L'effetto si
accumula: $\lambda$ elevato alla centesima. Un $1{,}1$ diventa $13\,781$, un
$0{,}9$ diventa $0{,}000027$. È da qui che nascono i gradienti che esplodono o
svaniscono, e per questo si guardano gli autovalori quando un addestramento
non converge.

`````

`````{tab} Superiore

Dato $A\in\mathbb{R}^{n\times n}$, un vettore non nullo $\mathbf{v}$ è un
**autovettore** con **autovalore** $\lambda$ se

$$
A\mathbf{v} = \lambda\mathbf{v}.
$$

Riscrivendo come $(A-\lambda I)\mathbf{v}=\mathbf{0}$: la soluzione non banale
esiste solo se $A-\lambda I$ è singolare, cioè se

$$
\det(A - \lambda I) = 0 .
$$

È l'**equazione caratteristica**, un polinomio di grado $n$ in $\lambda$: una
matrice $n\times n$ ha quindi $n$ autovalori nel campo complesso, contati con
molteplicità. Per la matrice della figura,
$\det\!\begin{pmatrix}2-\lambda&1\\1&2-\lambda\end{pmatrix}
=(2-\lambda)^2-1=0$ dà $\lambda_1=3$ e $\lambda_2=1$.

Il caso che ricorre di più nel machine learning è quello **simmetrico**
($A=A^\top$): il *teorema spettrale* garantisce che gli autovalori siano reali
e che esista una base ortonormale di autovettori, cioè $A = Q\Lambda Q^\top$
con $Q$ ortogonale e $\Lambda$ diagonale. Le matrici di covarianza sono
simmetriche, ed è precisamente questa decomposizione che la **PCA** calcola:
gli autovettori danno le direzioni di massima varianza, gli autovalori quanta
varianza c'è lungo ciascuna.

L'iterazione chiarisce il resto: $A^k\mathbf{v} = \lambda^k\mathbf{v}$, quindi
il comportamento asintotico di un sistema iterato è governato dall'autovalore
di modulo massimo (il *raggio spettrale*). Se $\rho(A)>1$ le componenti
divergono, se $\rho(A)<1$ collassano a zero — la lettura in una riga dei
gradienti che esplodono o svaniscono nelle reti profonde. Lo stesso argomento,
applicato alla matrice dei link del web, è il **PageRank**: l'ordinamento delle
pagine è l'autovettore associato all'autovalore $1$, l'unica direzione che
quella trasformazione lascia esattamente com'è.

`````

## Norme: misurare lunghezze ed errori

`````{tab} Elementare

La **norma** di un vettore è la sua "lunghezza". Per una freccia nel piano è
il buon vecchio teorema di Pitagora: radice quadrata della somma dei quadrati
delle componenti. La freccia $(3, 4)$, per esempio, è lunga
$\sqrt{3^2 + 4^2} = \sqrt{25} = 5$. Ci serve soprattutto per misurare
**quanto un modello sbaglia**: se la risposta giusta è un vettore e la
previsione è un altro vettore, la lunghezza della loro differenza è l'errore.

`````

`````{tab} Superiore

La norma euclidea (o $\ell_2$) di $\mathbf{x}\in\mathbb{R}^n$ è

$$
\lVert\mathbf{x}\rVert_2 = \sqrt{\sum_{i=1}^{n} x_i^2}
= \sqrt{\mathbf{x}^\top\mathbf{x}} .
$$

Accanto a essa usiamo spesso la norma $\ell_1$ (somma dei valori assoluti,
$\lVert\mathbf{x}\rVert_1=\sum_i |x_i|$), che nella regolarizzazione promuove
soluzioni *sparse*. La distanza tra previsione $\hat{\mathbf{y}}$ e target
$\mathbf{y}$ misurata con la norma $\ell_2$ al quadrato è la celebre loss
dell'**errore quadratico medio**:

$$
\mathcal{L} = \frac{1}{m}\sum_{i=1}^{m}
\lVert \hat{\mathbf{y}}^{(i)} - \mathbf{y}^{(i)} \rVert_2^2 .
$$

Norme e prodotti scalari sono legati da $\lVert\mathbf{x}\rVert_2^2 =
\mathbf{x}^\top\mathbf{x}$: misurare una lunghezza è fare il prodotto scalare
di un vettore con sé stesso.

`````

## In pratica, con NumPy

In Python l'algebra lineare vive nella libreria **NumPy**, che vedremo in
dettaglio nel capitolo su Python. Qui basti l'assaggio: le operazioni di
sopra sono una riga ciascuna.

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

```{admonition} Da ricordare
:class: important
- Un vettore rappresenta un esempio (le sue *feature*); una matrice impila
  esempi oppure trasforma i dati.
- Il **prodotto scalare** misura l'allineamento tra due vettori ed è il cuore
  del singolo neurone: $\mathbf{w}^\top\mathbf{x}+b$.
- Gli **autovettori** sono le direzioni che una matrice non devia, e i loro
  **autovalori** dicono di quanto le allunga: iterati, spiegano perché i
  gradienti esplodono o svaniscono.
- La **norma** misura lunghezze e, soprattutto, l'**errore** di un modello.
```
