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
addestra i modelli.

## L'ndarray: perché non basta una lista

Python ha già le liste. Perché inventare un altro contenitore di numeri?

```{figure} ../figures/numpy-array-vs-liste.svg
:name: fig-array-vs-lista
:alt: "Due rappresentazioni della memoria a confronto. In alto una lista Python: una sequenza di puntatori, ciascuno che rimanda a un oggetto numerico collocato altrove, sparso nella memoria. In basso un array NumPy: i valori sono scritti uno dopo l'altro in un blocco contiguo, senza intermediari."
:width: 92%

Dove stanno davvero i numeri. La lista tiene indirizzi e insegue un oggetto
per volta; l'array tiene i valori di fila, e il processore può leggerli a
blocchi.
```

La differenza di {numref}`fig-array-vs-lista` non è di eleganza ma di
velocità, e spiega da sola quasi tutto ciò che segue. Sommare due liste
significa saltare da un capo all'altro della memoria a ogni elemento; sommare
due array significa scorrere due nastri affiancati, che è esattamente
l'operazione per cui il processore è costruito.

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

Un `ndarray` è un blocco di memoria **contiguo, omogeneo e tipizzato**
(`dtype`, per esempio `float64` o `int32`), corredato da una *forma* (`shape`)
e da un insieme di *stride* che dicono di quanti byte spostarsi per passare
all'elemento successivo lungo ogni asse. Questa struttura permette due cose.
Primo: molte viste (*slice*, `reshape`, trasposizione) sono ricalcoli di
stride a costo zero, senza copiare i dati. Secondo: le operazioni
elemento-per-elemento sono delegate a cicli in C compilati e vettorizzati
(istruzioni SIMD), che saltano l'*overhead* dell'interprete su ogni
iterazione; l'algebra lineare vera e propria (i prodotti tra matrici) passa
invece per librerie BLAS ottimizzate. È la differenza tra `float` scatolati
sparsi nella heap e un array C nudo.

`````

## Creare un array

Ci sono pochi modi ricorrenti per far nascere un array; li useremo ovunque.

```python
import numpy as np

np.array([1, 2, 3])          # da una lista Python
np.zeros((2, 3))             # matrice 2x3 di zeri (utile per inizializzare)
np.ones(4)                   # vettore di 1: array([1., 1., 1., 1.])
np.arange(0, 10, 2)          # come range, ma array: [0 2 4 6 8]
np.linspace(0, 1, 5)         # 5 punti equispaziati tra 0 e 1 inclusi

rng = np.random.default_rng(0)   # generatore con seme, per risultati riproducibili
rng.normal(size=(2, 2))          # matrice 2x2 di numeri casuali "a campana"
```

Due dettagli importanti: `arange` è pensato per interi e passi, `linspace` per
dividere un intervallo in un numero *esatto* di punti (è quello giusto per
disegnare curve). E il generatore casuale moderno si costruisce con
`default_rng(seme)`: fissare il seme rende l'esperimento ripetibile, requisito
minimo di ogni lavoro scientifico serio.

## Indicizzazione e slicing

Su un array si "affonda la mano" con le stesse parentesi quadre delle liste, ma
con più potenza: si indicizzano più assi insieme, separati da virgola.

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
x[0]        # 10  (il primo)
x[-1]       # 50  (l'ultimo)
x[1:4]      # array([20, 30, 40])  slice: da 1 incluso a 4 escluso

M = np.arange(12).reshape(3, 4)   # matrice 3x4 con 0..11
M[1, 2]     # riga 1, colonna 2 -> 6
M[:, 0]     # tutte le righe, colonna 0 -> array([0, 4, 8])
M[0]        # prima riga intera -> array([0, 1, 2, 3])
```

C'è poi un'indicizzazione che in Python puro richiederebbe un ciclo con `if`.

`````{tab} Elementare

L'**indicizzazione booleana** seleziona gli elementi in base a una condizione.
Scrivi la domanda ("quali sono maggiori di 25?") e NumPy ti restituisce solo
quelli:

```python
x = np.array([10, 20, 30, 40, 50])
x > 25            # array([False, False,  True,  True,  True])
x[x > 25]         # array([30, 40, 50])  tieni solo i "True"
x[x > 25] = 0     # ...oppure riscrivili tutti in un colpo
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
su array sono ambigui) ciascun confronto tra parentesi:

```python
x[(x > 15) & (x < 45)]    # elementi in (15, 45)
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
b = np.array([[1], [2], [3]])      # colonna: 3 sovrapprezzi
a + b                              # matrice 3x4, senza un solo for
```

La regola pratica: se una delle due forme ha $1$ dove l'altra ha $n$, quel lato
viene "steso" a $n$. Il ripetersi è solo apparente: serve a far tornare i conti,
non consuma memoria.

`````

`````{tab} Superiore

Il broadcasting allinea le forme **da destra**. Due assi sono compatibili se
sono uguali oppure se uno dei due vale $1$: quel lato viene esteso senza copia.
Con $a$ di forma $(4,)$ e $b$ di forma $(3,1)$:

$$
(3,1) \;\oplus\; (4,) \;\to\; (3,1)\;\oplus\;(1,4)\;\to\;(3,4).
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

%timeit raddoppia_loop(x)       # ~centinaia di millisecondi
%timeit 2 * x                   # ~pochi millisecondi
```

`````{tab} Elementare

Le due righe fanno la stessa cosa (raddoppiare un milione di numeri) ma la
seconda è tipicamente **decine o centinaia di volte più veloce** (`%timeit` è
il cronometro dei notebook: misura quanto impiega un'istruzione). Il ciclo
Python paga un piccolo pedaggio a ogni giro; `2 * x` fa lavorare direttamente
il motore in C su tutto il blocco. La regola d'oro con NumPy: *se stai
scrivendo un `for` su un array, quasi sempre esiste un modo per non
scriverlo*.

`````

`````{tab} Superiore

Il divario è di uno o due ordini di grandezza e nasce dall'*overhead*
dell'interprete: ogni iterazione in Python comporta controllo di tipo,
allocazione di oggetti e dispatch dinamico. La forma vettorizzata sposta il
ciclo dentro codice C compilato che opera su memoria contigua, con buona
località di cache e, dove disponibile, vettorizzazione SIMD. Non è gratis
all'infinito: la vettorizzazione può aumentare l'uso di memoria (array
temporanei intermedi) e non copre bene ogni algoritmo intrinsecamente
sequenziale, ma per l'algebra dei dati è quasi sempre la scelta giusta.

`````

## Algebra lineare, in una riga

Qui i conti del prossimo capitolo (quello di matematica) diventano codice. Se
termini come *prodotto scalare* o *matrice inversa* non ti dicono ancora
nulla, nessun problema: verranno spiegati lì, e potrai tornare a rileggere
queste righe. Per ora conta una cosa sola: ogni operazione è una riga.
Prodotto scalare, prodotto matrice-vettore e prodotto tra matrici sono tutti
l'operatore `@`; `np.linalg` raccoglie il resto.

```python
A = np.array([[1., 2.],
              [3., 4.]])
v = np.array([1., 1.])

A @ v                  # prodotto matrice-vettore -> array([3., 7.])
A @ A                  # prodotto matrice-matrice
v @ v                  # prodotto scalare -> 2.0

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

```{admonition} Da ricordare
:class: important
- L'`ndarray` è un blocco di memoria **contiguo e tipizzato**: da qui la sua
  velocità rispetto alle liste Python.
- `array`, `zeros`, `ones`, `arange`, `linspace`, `default_rng` creano array;
  slicing e **indicizzazione booleana** li selezionano senza cicli.
- Il **broadcasting** allinea le forme da destra ed espande gli assi di
  dimensione $1$: somma forme diverse senza copiare dati.
- **Vettorizzare** (sostituire un `for` con un'operazione sull'array) rende il
  codice più corto e uno o due ordini di grandezza più veloce.
- Prodotti e algebra lineare vivono in `@` e `np.linalg`: per i sistemi usa
  `solve`, non l'inversa.
```
