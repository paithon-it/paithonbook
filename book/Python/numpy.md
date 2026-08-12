# NumPy: calcolo numerico vettorizzato

Nel 2005 Travis Oliphant unisce due librerie rivali (*Numeric* e *Numarray*)
in un solo progetto, che l'anno seguente rilascia come **NumPy 1.0**. È una di
quelle scelte silenziose che cambiano un intero campo: da allora quasi ogni
pezzo dell'ecosistema scientifico di Python (Pandas, scikit-learn, PyTorch,
TensorFlow) poggia, direttamente o no, sulla struttura dati che NumPy
introduce. Quella struttura è l'**array N-dimensionale**, l'`ndarray`: una
griglia di numeri che può essere una semplice fila, una tabella, o una pila di
tabelle. Capirlo bene è il prerequisito pratico a tutto il resto del libro: è
il ponte tra la matematica dei vettori e delle matrici e il codice che
addestra i modelli. Quanto a *vettorizzato*, che sta nel titolo, è la parola
che questa pagina spiega a metà strada: per ora vuol dire fare un conto su un
blocco intero di numeri in una volta sola, invece che su un numero per volta.

## L'ndarray: perché non basta una lista

Python ha già le liste. Perché inventare un altro contenitore di numeri?

```{figure} ../figures/numpy-array-vs-liste.svg
:name: fig-array-vs-lista
:alt: "Due rappresentazioni della memoria a confronto. In alto una lista Python: una sequenza di rimandi, ciascuno dei quali indica un oggetto numerico collocato altrove, sparso nella memoria. In basso un array NumPy: i valori sono scritti uno dopo l'altro in un blocco contiguo, senza intermediari."
:width: 92%

Dove stanno davvero i numeri. Nella lista ogni numero è un pacchetto a sé,
sparso nella memoria, e per raggiungerlo si segue un rimando per volta;
nell'array i valori stanno di fila, e il processore può leggerli a blocchi.
```

Quel che {numref}`fig-array-vs-lista` mostra è il motivo per cui l'array esiste:
scorrere valori messi di fila è l'operazione per cui un processore è costruito,
mentre inseguire un rimando alla volta è quella che gli riesce peggio. Vale la
pena essere precisi su chi ne beneficia, perché è una confusione facile e la
riprenderemo alla fine della pagina: la compattezza serve al codice in C che
attraversa l'array tutto insieme, non a un ciclo scritto in Python.

Una nota di passaggio, perché è la domanda che viene subito: in Python il `+`
fra due liste non somma i numeri, attacca la seconda in coda alla prima
(`[1, 2] + [3]` fa `[1, 2, 3]`). Per sommarle davvero, valore per valore,
servirebbe un ciclo scritto a mano, ed è una delle ragioni per cui NumPy
esiste.

`````{tab} Elementare

Una lista Python è un contenitore *generico*: può tenere insieme un numero, una
stringa e perfino un'altra lista. Questa flessibilità si paga. Per il computer
ogni elemento è una scatola separata sparsa nella memoria, e per raddoppiare un
milione di numeri deve visitarle una a una, chiedendosi ogni volta "che cos'è
questo?".

L'`ndarray` fa il patto opposto: tutti gli elementi sono dello **stesso tipo** e
stanno **uno accanto all'altro** in un blocco compatto di memoria. Perde la
libertà di mescolare tipi diversi, ma in cambio le operazioni sui numeri
diventano corte da scrivere e molto più veloci da eseguire.

`````

`````{tab} Superiore

Un `ndarray` è una **vista tipizzata su un blocco di memoria**: un `dtype`
omogeneo (per esempio `float64` o `int32`), una *forma* (`shape`) e un insieme
di *stride* che dicono di quanti byte spostarsi per passare all'elemento
successivo lungo ogni asse. Quando gli stride sono esattamente quelli che si
ricavano dalla forma, l'array è **contiguo** (in ordine C per righe, in ordine
Fortran per colonne) e un ciclo in C può scorrerlo di fila; la contiguità è
quindi una proprietà del modo in cui l'array guarda il buffer, non una sua
definizione, ed è verificabile con `.flags` (`C_CONTIGUOUS`, `F_CONTIGUOUS`).
La distinzione conta: `M[:, 0]`, una delle selezioni di questa pagina, è un
`ndarray` perfettamente legittimo e **non** contiguo, e `M.T` è F-contigua ma
non C-contigua. Questa struttura permette due cose.
Primo: slice e trasposizione sono sempre *viste*, ricalcoli di stride a costo
zero senza copia dei dati (ed è proprio perché la contiguità non è garantita
che gli stride esistono); `reshape` è una vista quando la disposizione in
memoria lo consente, altrimenti copia. Secondo: le operazioni
elemento-per-elemento sono delegate a cicli in C compilati e vettorizzati
(istruzioni SIMD), che
saltano l'*overhead* dell'interprete su ogni iterazione; l'algebra lineare
vera e propria (i prodotti tra matrici) passa invece per librerie BLAS
ottimizzate. È la differenza tra `float` scatolati sparsi nella heap e un
array C nudo.

`````

## Creare un array

Ci sono pochi modi ricorrenti per far nascere un array; li useremo ovunque.

```python
import numpy as np

np.array([1, 2, 3])          # da una lista Python
np.zeros((2, 3))             # tabella di zeri con 2 righe e 3 colonne
np.ones(4)                   # vettore di 1: array([1., 1., 1., 1.])
np.arange(0, 10, 2)          # come range, ma array: [0 2 4 6 8]
np.linspace(0, 1, 5)         # 5 punti equispaziati tra 0 e 1 inclusi

rng = np.random.default_rng(0)   # generatore con seme, per risultati riproducibili
rng.normal(size=(2, 2))          # matrice 2x2 di numeri casuali "a campana"
```

La prima riga contiene una parolina che vale la pena guardare: `as`. `import
numpy as np` vuol dire «importa numpy e, qui dentro, chiamalo `np`»: è un
soprannome, e da quel momento ogni strumento della libreria si scrive
`np.qualcosa`. Il soprannome lo sceglie chi scrive (funzionerebbe anche
`numpy.array(...)`, o `as npy`), ma `np` per NumPy, `pd` per Pandas e `plt` per
Matplotlib sono convenzioni così universali che cambiarle rende il codice
illeggibile agli altri.

Attenzione alle parentesi di `np.zeros((2, 3))`, che sono due: la funzione
vuole *una sola* cosa, la forma dell'array, e la forma è una coppia
(righe, colonne), che si scrive fra le sue parentesi. Con un array a una
dimensione la forma è un numero solo e le parentesi doppie non servono, da cui
`np.ones(4)`. L'ordine è sempre quello: prima le righe, poi le colonne.

Due dettagli importanti: `arange` è pensato per interi e passi, `linspace` per
dividere un intervallo in un numero *esatto* di punti (è quello giusto per
disegnare curve). E il generatore casuale moderno si costruisce con
`default_rng(seme)`: fissare il seme rende l'esperimento ripetibile, requisito
minimo di ogni lavoro scientifico serio. Non è una contraddizione: un computer
non sa fare niente a caso, e quei numeri li calcola con una formula che li fa
*sembrare* casuali (si dicono infatti *pseudo-casuali*). Il seme è il numero da
cui la formula parte: stesso seme, stessa sequenza, oggi e fra un anno; seme
diverso, sequenza diversa. Zero non ha niente di speciale, è solo il primo
numero che viene in mente.

## Indicizzazione e slicing

Su un array si "affonda la mano" con le stesse parentesi quadre delle liste
(`numeri[0]` è il primo elemento), ma con più potenza: si indicizzano più
**assi** insieme, separati da virgola, dove gli assi sono le direzioni lungo
cui l'array si estende (in una tabella: le righe e le colonne).

```{figure} ../figures/numpy-indexing-reshape-vettoriale.svg
:name: fig-slicing-numpy
:alt: "Un array di quattro righe per sei colonne disegnato come griglia, con tre selezioni evidenziate su copie affiancate: un'intera riga, un'intera colonna e una sottomatrice rettangolare presa incrociando un intervallo di righe e uno di colonne."
:width: 96%

Lo slicing visto sulla griglia. Il primo indice sceglie fra le righe, il
secondo fra le colonne, e i due tagli si incrociano: quel che resta è la
selezione.
```

Conviene tenere a mente, guardando {numref}`fig-slicing-numpy`, che nessuna
delle tre selezioni copia i dati. Sono **viste** sullo stesso array in
memoria, e scriverci dentro modifica l'originale: è la differenza più
insidiosa rispetto allo slicing delle liste, che invece copia.

```python
x = np.array([10, 20, 30, 40, 50])
x[0]        # np.int64(10)  il primo (si conta da zero)
x[-1]       # np.int64(50)  l'ultimo: gli indici negativi contano dalla fine
x[1:4]      # array([20, 30, 40])  slice: da 1 incluso a 4 escluso

M = np.arange(12).reshape(3, 4)   # i numeri 0..11 ridisposti in 3 righe e 4 colonne
M[1, 2]     # np.int64(6)  seconda riga, terza colonna: gli indici sono 1 e 2
M[:, 0]     # tutte le righe, colonna 0 -> array([0, 4, 8])
M[0]        # prima riga intera -> array([0, 1, 2, 3])
```

Tre convenzioni, in tre righe. Si conta **da zero**, quindi l'indice `1` è il
secondo elemento e `M[1, 2]` sta nella seconda riga, terza colonna. Un indice
**negativo** conta dalla fine, e `x[-1]` è l'ultimo qualunque sia la lunghezza.
E in una *slice* il secondo estremo è **escluso**: `x[1:4]` dà tre elementi,
non quattro. Sembra una scortesia, ed è la scelta che fa quadrare i conti: la
lunghezza del pezzo è la differenza dei due numeri ($4-1=3$), e due fette
scritte di seguito, `x[0:3]` e `x[3:6]`, si incastrano senza sovrapporsi e
senza buchi.

Sulla forma di ciò che viene stampato: `np.int64(10)` non è un numero
strano, è il modo in cui NumPy 2 *mostra* un suo numero intero quando lo si
scrive all'interprete, per dire di che tipo è (`x[0]` è a tutti gli effetti il
numero 10, e `print(x[0])` stampa proprio `10`). Chi arriva da un tutorial
scritto per NumPy 1, dove usciva `10` e basta, si trova la differenza qui e in
tutti i punti in cui da un array si estrae un valore singolo.

C'è poi un'indicizzazione che in Python puro richiederebbe un ciclo con `if`.

`````{tab} Elementare

L'**indicizzazione booleana** seleziona gli elementi in base a una condizione.
Scrivi la domanda ("quali sono maggiori di 25?") e NumPy ti restituisce solo
quelli:

```python
x = np.array([10, 20, 30, 40, 50])
x > 25            # array([False, False,  True,  True,  True])
x[x > 25]         # array([30, 40, 50])  tieni solo i "True"

y = x.copy()      # una copia, per non rovinare x
y[y > 25] = 0     # ...oppure riscrivili tutti in un colpo -> [10 20 0 0 0]
```

È il modo naturale per filtrare dati: "prendi solo i clienti sopra i 25 anni",
"azzera i valori negativi".

`````

`````{tab} Superiore

Una condizione come `x > 25` produce una **maschera booleana**, un array di
`bool` della stessa forma. Usata come indice, `x[mask]` estrae gli elementi
dove la maschera è `True`, restituendo un array 1-D (una *copia*, non una
vista). La stessa maschera funziona in assegnazione, `x[mask] = 0`, e si
compone con gli operatori logici *bitwise* `&`, `|`, `~` (non `and`/`or`, che
su array sono ambigui), con ciascun confronto tra parentesi:

```python
x = np.array([10, 20, 30, 40, 50])
x[(x > 15) & (x < 45)]    # elementi in (15, 45) -> array([20, 30, 40])
```

Questa indicizzazione booleana è il pane quotidiano della pulizia dati e
sostituisce interi cicli con un'unica espressione dichiarativa.

`````

## Broadcasting: sommare forme diverse senza cicli

Cosa succede se provi a sommare una riga e una colonna di dimensioni diverse?
In quasi ogni linguaggio, un errore. In NumPy, il **broadcasting**: le forme
"più piccole" vengono espanse virtualmente finché combaciano
({numref}`fig-broadcasting`).

```{figure} ../figures/broadcasting-numpy.svg
:name: fig-broadcasting
:alt: Una riga 1x4 si ripete verso il basso e una colonna 3x1 verso destra, sommandosi in una matrice 3x4.
:width: 90%

Broadcasting: una riga $(1\times 4)$ e una colonna $(3\times 1)$ si espandono
virtualmente ciascuna lungo la dimensione mancante e si sommano in una matrice
$(3\times 4)$. Nessun dato viene davvero copiato in memoria.
```

`````{tab} Elementare

Immagina una tabella da riempire: hai i prezzi base di 4 prodotti (una riga) e 3
sovrapprezzi regionali (una colonna). Vuoi tutte le combinazioni. Invece di un
doppio ciclo, allinei riga e colonna e NumPy ripete l'una lungo le righe e
l'altra lungo le colonne, calcolando la griglia intera:

```python
a = np.array([10, 20, 30, 40])    # riga: 4 prezzi base
b = np.array([[1], [2], [3]])     # colonna: 3 sovrapprezzi
a + b                             # tabella 3x4, senza un solo for
# array([[11, 21, 31, 41],
#        [12, 22, 32, 42],
#        [13, 23, 33, 43]])
```

Le quadre dentro le quadre di `b` non sono un vezzo: `[1, 2, 3]` sarebbe una
*riga* di tre numeri, mentre `[[1], [2], [3]]` è fatto di tre righe da un
numero ciascuna, cioè una **colonna**. Le parentesi esterne racchiudono la
tabella, quelle interne una riga per volta.

La regola pratica: se una delle due forme ha $1$ dove l'altra ha $n$, quel lato
viene "steso" a $n$. Il ripetersi è solo apparente: serve a far tornare i conti,
non consuma memoria.

`````

`````{tab} Superiore

Il broadcasting allinea le forme **da destra**. Due assi sono compatibili se
sono uguali oppure se uno dei due vale $1$: quel lato viene esteso senza copia.
Con $a$ di forma $(4,)$ e $b$ di forma $(3,1)$:

$$
(4,) \;\oplus\; (3,1) \;\to\; (1,4)\;\oplus\;(3,1)\;\to\;(3,4).
$$

L'asse mancante di $a$ viene inserito a sinistra come $1$, poi ogni asse-$1$ è
trasmesso lungo l'altra dimensione. Il risultato è equivalente a
$C_{ij}=a_j+b_i$ ma è calcolato in C, senza materializzare le copie: gli stride
del lato "trasmesso" sono posti a $0$, così lo stesso dato viene riletto più
volte. È il meccanismo che permette, per esempio, di sottrarre la media di
colonna da un'intera matrice di dati con `X - X.mean(axis=0)`.

`````

## Vettorizzazione: quanto conta davvero

Il motivo per cui tutto questo esiste è la velocità. "Vettorizzare" significa
sostituire un ciclo Python con un'operazione sull'intero array.

```python
import numpy as np

x = np.random.default_rng(0).random(1_000_000)

def raddoppia_loop(v):          # la versione "a mano"
    out = np.empty_like(v)
    for i in range(len(v)):
        out[i] = 2 * v[i]
    return out

%timeit raddoppia_loop(x)       # ~100 millisecondi
%timeit 2 * x                   # ~0,2 millisecondi
```

Le ultime due righe non sono Python: `%timeit` è un comando dei notebook (una
*magic* di IPython) che cronometra un'istruzione ripetendola molte volte e
riportando **media e deviazione standard** dei tempi (`mean ± std. dev.`),
insieme al numero di ripetizioni. In un normale file `.py` non funziona: lì si
usa il modulo `timeit` della libreria standard. Su una macchina condivisa la
media è rumorosa e il minimo è una misura più onesta: `%timeit -o` restituisce
l'oggetto del risultato, il cui campo `.best` è il tempo migliore.

`````{tab} Elementare

Le due misure riguardano la stessa cosa (raddoppiare un milione di numeri) ma
la seconda strada è tipicamente **centinaia di volte più veloce**. La ragione
non è che i numeri stanno vicini: è che nel secondo caso il ciclo **sparisce**.
Il ciclo Python paga un piccolo pedaggio a ogni giro, un milione di volte;
`2 * x` è una sola richiesta, e a scorrere il blocco è il motore in C, che quel
pedaggio non lo paga. La regola d'oro con NumPy: *se stai scrivendo un `for` su
un array, quasi sempre esiste un modo per non scriverlo*.

`````

`````{tab} Superiore

Il divario è di due o tre ordini di grandezza (sull'esempio qui sopra: circa
$100$ ms il ciclo, circa $0{,}2$ ms la forma vettorizzata) e nasce
dall'*overhead* dell'interprete: ogni iterazione in Python comporta controllo
di tipo, allocazione di oggetti e dispatch dinamico. La forma vettorizzata
sposta il ciclo dentro codice C compilato che opera su memoria contigua, con
buona località di cache e, dove disponibile, vettorizzazione SIMD.

Vale la pena essere espliciti su quale dei due fattori pesa, perché la
conclusione sbagliata è a portata di mano: il guadagno non viene dal
*contenitore*, viene dalla sparizione del ciclo. La prova è misurabile e va nel
verso opposto all'intuizione: un ciclo Python che indicizza un `ndarray` è più
lento dello stesso ciclo su una lista (sul milione di elementi qui sopra, circa
$102$ ms contro $41$ ms di tempo di CPU), perché ogni `v[i]` deve *incartare* il
numero grezzo in un oggetto `np.float64` che nella lista esiste già. La
contiguità serve al ciclo in C, non a quello in Python: un `ndarray` non è
veloce perché è un `ndarray`, è veloce quando lo si tocca tutto in una volta.
Chi "ottimizza" un ciclo Python convertendo la lista in array lo rallenta.

Non è comunque gratis all'infinito: la vettorizzazione può aumentare l'uso di
memoria (array temporanei intermedi) e non copre bene ogni algoritmo
intrinsecamente sequenziale, ma per l'algebra dei dati è quasi sempre la scelta
giusta.

`````

## Algebra lineare, in una riga

Qui i conti del prossimo capitolo (quello di matematica) diventano codice. Se
termini come *prodotto scalare* o *matrice inversa* non ti dicono ancora
nulla, nessun problema: verranno spiegati lì, e potrai tornare a rileggere
queste righe. Per ora conta una cosa sola: ogni operazione è una riga.
Prodotto scalare, prodotto matrice-vettore e prodotto tra matrici sono tutti
l'operatore `@`; `np.linalg` raccoglie il resto. (Sì, è lo stesso simbolo dei
decoratori: là sta da solo sopra una funzione, qui sta fra due array, e i due
mestieri non hanno niente in comune se non il segno.)

```python
A = np.array([[1., 2.],
              [3., 4.]])
v = np.array([1., 1.])

A @ v                  # prodotto matrice-vettore -> array([3., 7.])
A @ A                  # prodotto matrice-matrice
v @ v                  # prodotto scalare -> np.float64(2.0)

np.linalg.norm(v)      # norma euclidea
np.linalg.inv(A)       # inversa
np.linalg.solve(A, v)  # risolve A z = v  (piu' stabile dell'inversa)
```

Un'avvertenza che torna spesso: per risolvere un sistema $A z = v$ si usa
`np.linalg.solve`, non `inv(A) @ v`. Il primo è più preciso e più veloce;
calcolare l'inversa esplicita è quasi sempre uno spreco. Con questi mattoni
(array, broadcasting, vettorizzazione, algebra lineare), abbiamo il
vocabolario per esprimere in poche righe ciò che un modello, sotto, fa milioni
di volte.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- In un `ndarray` i valori sono tutti dello **stesso tipo** e stanno **uno
  accanto all'altro**: è da qui che viene la sua velocità rispetto a una lista.
- `array`, `zeros`, `ones`, `arange`, `linspace`, `default_rng` creano array; le
  parentesi quadre ne scelgono un pezzo, e una **condizione fra le quadre**
  (`x[x > 25]`) fa da colino, tenendo solo gli elementi che la soddisfano.
- Il **broadcasting** permette di sommare forme diverse: dove una delle due ha
  un solo elemento, quel lato viene steso quanto serve, senza copiare niente.
- **Vettorizzare** vuol dire sostituire un `for` con un'operazione su tutto
  l'array: il codice è più corto e da cento a mille volte più veloce, perché il
  ciclo sparisce.
- Il segno `@` fa i prodotti fra vettori e matrici, `np.linalg` il resto. Per
  risolvere un sistema si usa `solve`.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- L'`ndarray` è una **vista tipizzata** su un blocco di memoria, contigua
  quando gli stride sono quelli della forma: da qui la sua velocità rispetto
  alle liste Python, e da qui il fatto che una slice resti una vista.
- `array`, `zeros`, `ones`, `arange`, `linspace`, `default_rng` creano array;
  slicing e **indicizzazione booleana** li selezionano senza cicli (lo slicing
  dà una vista, la maschera booleana una copia).
- Il **broadcasting** allinea le forme da destra ed espande gli assi di
  dimensione $1$: somma forme diverse senza copiare dati.
- **Vettorizzare** (sostituire un `for` con un'operazione sull'array) rende il
  codice più corto e da cento a mille volte più veloce (due o tre ordini di
  grandezza). Il guadagno sta nel ciclo che sparisce, non nel contenitore: un
  `for` su un `ndarray` è più lento dello stesso `for` su una lista.
- Prodotti e algebra lineare vivono in `@` e `np.linalg`: per i sistemi usa
  `solve`, non l'inversa.
```

`````
