# Tensori e autograd

Dentro PyTorch ogni cosa è un tensore: l'immagine da classificare, la frase
tradotta in numeri, ogni peso della rete. Il nome stesso della libreria lo
porta scritto: *Torch* accende i tensori, *Py* li porta in Python. E accanto
ai tensori vive il secondo protagonista, più discreto ma decisivo:
**autograd**, il meccanismo che osserva i calcoli mentre avvengono e sa
risalire alle derivate. Capire questi due oggetti significa capire il motore
su cui gira tutto il deep learning moderno.

## Che cos'è un tensore

Scalari, vettori e matrici li abbiamo già incontrati nel capitolo di algebra
lineare. Il tensore è semplicemente il passo successivo: la stessa idea di
"mettere i numeri in fila" portata a un numero qualunque di dimensioni.

```{figure} ../figures/tensori-scala.svg
:name: fig-tensori-scala
:alt: "Quattro oggetti in fila: un singolo quadrato (scalare, rank 0), una riga di quattro celle (vettore, rank 1), una griglia 3x3 (matrice, rank 2) e una pila di tre griglie (tensore 3D, rank 3)."
:width: 90%

La scala dei tensori. Salendo di *rank* si aggiunge ogni volta un asse: da un
numero solo, a una fila, a una griglia, a una pila di griglie.
```

`````{tab} Elementare

Un tensore è un contenitore di numeri con un certo numero di "assi", cioè di
direzioni lungo cui si estende ({numref}`fig-tensori-scala`):

- un **numero solo** (per esempio la temperatura, $23{,}5$) è un tensore a zero
  assi: uno *scalare*;
- una **fila di numeri** (i voti di uno studente) è un tensore a un asse: un
  *vettore*;
- una **tabella** di numeri (i pixel di una foto in scala di grigi) ha due
  assi: una *matrice*;
- una **pila di tabelle** (una foto a colori: una griglia per il rosso, una
  per il verde, una per il blu) ha tre assi.

Il numero di assi si chiama **rank**; le lunghezze lungo ciascun asse formano
la **shape** (la "forma"). Una foto RGB $256 \times 256$ è, in PyTorch, un
tensore di shape $(3, 256, 256)$, prima i tre colori, poi le due dimensioni
della griglia: rank 3.

`````

`````{tab} Superiore

Formalmente un tensore di rank $r$ è un array multidimensionale i cui elementi
si indicizzano con $r$ indici, $T_{i_1 i_2 \dots i_r}$. Lo caratterizzano due
attributi:

- il **rank** (o numero di assi), $r$;
- la **shape** $(n_1, n_2, \dots, n_r)$, la dimensione lungo ciascun asse.

Scalare, vettore e matrice sono i casi $r = 0, 1, 2$. Nel deep learning si
lavora costantemente con rank più alti: in PyTorch un *batch* di immagini RGB
è un tensore $(N, C, H, W)$ (numero di esempi, canali, altezza, larghezza)
quindi rank 4 (l'ordine *channels-first*, diverso dal $(N, H, W, C)$ di altre
librerie). A differenza dell'oggetto matematico "tensore" (che porta con sé
regole di trasformazione tra sistemi di coordinate), qui il termine indica
soltanto la struttura dati: un array $n$-dimensionale con un `dtype` omogeneo
(`float32`, `int64`, …).

`````

## Creare tensori e farci i conti

Un `torch.Tensor` si crea da liste Python, con le funzioni di fabbrica, o
direttamente da un array NumPy: le due librerie sono parenti strette.

```python
import torch

s = torch.tensor(3.14)                     # scalare, rank 0
v = torch.tensor([1.0, 2.0, 3.0])          # vettore, rank 1
M = torch.tensor([[1., 2.], [3., 4.]])     # matrice, rank 2

M.shape        # torch.Size([2, 2])
M.dtype        # torch.float32

torch.zeros(2, 3)        # matrice 2x3 di zeri
torch.ones(5)            # vettore di uno
torch.randn(3, 3)        # numeri casuali dalla normale standard
torch.arange(0, 10, 2)   # tensor([0, 2, 4, 6, 8])
```

Sui tensori valgono le operazioni dell'algebra lineare che già conosciamo
(somma, prodotto per scalare, prodotto matriciale) e ogni riga viene eseguita
nell'istante in cui la scrivi, come in NumPy:

```python
a = torch.tensor([1., 2., 3.])
b = torch.tensor([10., 20., 30.])

a + b            # tensor([11., 22., 33.])
a * b            # prodotto elemento per elemento
a.sum()          # tensor(6.)  -> calcolato SUBITO
a @ b            # prodotto scalare: 1·10 + 2·20 + 3·30 = tensor(140.)
a.reshape(3, 1)  # nuova forma: gli stessi numeri in colonna, 3x1
```

`````{tab} Elementare

Due cose da notare. Primo: il risultato compare all'istante, con i numeri già
dentro (puoi controllare ogni passaggio come su una calcolatrice, senza
"avviare" nulla). Secondo: il **broadcasting**, che conosciamo già da NumPy,
funziona identico. Se scrivi `a + 5`, PyTorch capisce da solo che vuoi sommare
$5$ a *ciascuno* dei tre numeri, come un insegnante che alza di un punto tutti
i voti della classe senza bisogno di scrivere la regola tre volte: il
risultato è `tensor([6., 7., 8.])`.

`````

`````{tab} Superiore

L'interoperabilità con NumPy è alla pari: `torch.from_numpy(arr)` e
`t.numpy()` convertono nei due sensi **condividendo la memoria** (nessuna
copia: modificare l'uno modifica l'altro). Le regole di broadcasting sono le
stesse di NumPy, gli assi si allineano da destra e le dimensioni compatibili
(uguali, o pari a 1) si espandono virtualmente: una matrice $(3, 4)$ più un
vettore $(4,)$ produce una $(3, 4)$. Il prodotto matriciale è `@` (ovvero
`torch.matmul`), con la stessa semantica di NumPy anche sui vettori rank-1:
`a @ b` tra due vettori è direttamente il prodotto scalare, senza bisogno di
reshape. A proposito di `reshape`: restituisce una **vista** (stessa memoria,
solo un modo diverso di leggerla) quando la disposizione dei dati lo permette,
e altrimenti copia in silenzio; `t.view(...)` la vista la pretende, e su un
tensore non contiguo solleva un errore invece di copiare. Molte operazioni
esistono in variante *in-place* col suffisso underscore (`t.add_(1)`,
`t.zero_()`): risparmiano memoria ma, come vedremo, vanno evitate sui tensori
tracciati da autograd.

`````

## Lo stesso codice su CPU e GPU

Ogni tensore vive su un *device*. Di default è la CPU; spostarlo su una GPU
(se c'è) è una chiamata a `.to()`, e tutto il resto del codice non cambia.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

M = torch.randn(1000, 1000)
M = M.to(device)          # ora vive sulla GPU (se disponibile)
prodotto = M @ M          # calcolato dove vive il tensore
```

`````{tab} Elementare

La regola è una sola: **i conti avvengono dove stanno i numeri**. Se il
tensore è sulla CPU, calcola la CPU; se lo sposti sulla scheda grafica,
calcola lei, e per le moltiplicazioni tra matrici grandi può essere decine o
centinaia di volte più veloce, perché una GPU è nata per fare migliaia di
piccoli conti in parallelo (in origine, i pixel dei videogiochi). L'unica
attenzione: due tensori possono lavorare insieme solo se stanno sullo stesso
dispositivo (non puoi sommare un numero che sta in cucina con uno che sta in
garage senza prima spostarne uno).

`````

`````{tab} Superiore

Il pattern idiomatico è definire `device` una volta all'inizio e spostare
modello e batch con `.to(device)`; su Apple Silicon il device si chiama
`"mps"`. Operazioni tra tensori su device diversi sollevano un errore
esplicito (nessun trasferimento implicito, che nasconderebbe costi: il
passaggio CPU↔GPU attraversa il bus PCIe ed è lento rispetto al calcolo). Le
stesse moltiplicazioni tra matrici che scriviamo qui sono, sull'hardware,
migliaia di prodotti scalari eseguiti in parallelo dai kernel CUDA/cuDNN: è il
motivo per cui le GPU (nate per la grafica) sono diventate lo strumento del
deep learning. Il codice resta identico; cambia solo la velocità.

`````

## Autograd: la derivata calcolata da sola

Addestrare una rete significa girare le sue tante manopole interne (i
**parametri**, cioè i pesi) finché l'errore che commette, quello che il
mestiere chiama *loss* e scrive $\mathcal{L}$, non diventa piccolo. Per sapere
da che parte girare ciascuna manopola serve il **gradiente**: la derivata
dell'errore rispetto a ogni parametro, che dice se aumentandolo l'errore sale
o scende, e di quanto. Calcolarlo a mano per una rete con milioni di pesi è
impensabile: qui entra la **differenziazione automatica** (*autodiff*), il
vero cuore di PyTorch.

```{figure} ../figures/extra-backpropagation-spiegata.svg
:name: fig-autograd-due-passate
:alt: "Una rete a tre strati percorsa in due sensi. Le frecce di andata vanno dall'ingresso all'uscita e calcolano le attivazioni; le frecce di ritorno vanno dall'uscita all'ingresso e propagano il gradiente, riusando i valori memorizzati durante l'andata."
:width: 96%

I due sensi di marcia. L'andata calcola e *ricorda*; il ritorno riusa quei
valori memorizzati, ed è per questo che il gradiente costa all'incirca quanto
una seconda passata e non quanto milioni di derivate separate.
```

La memoria implicita in {numref}`fig-autograd-due-passate` spiega un
comportamento di PyTorch che altrimenti sorprende: perché un forward pass
consumi RAM crescente con la profondità, e perché `torch.no_grad()` la
liberi. Se non chiederete mai il gradiente, quei valori intermedi non serve
tenerli.

```python
x = torch.tensor(3.0, requires_grad=True)   # "osserva questo tensore"

y = x**2 + 2*x            # y = x² + 2x: il grafo si costruisce da solo
y.backward()              # passata all'indietro

x.grad                    # dy/dx = 2x + 2  ->  tensor(8.)
```

`````{tab} Elementare

Immagina un **registratore** acceso mentre fai i conti: annota ogni singola
operazione eseguita a partire dai tensori "osservati". Alla fine gli chiedi:
"com'è cambiato il risultato al variare di questo ingresso?" e lui,
riavvolgendo il nastro all'indietro, ti risponde con la derivata esatta.

È ciò che fa autograd: dichiarando `requires_grad=True` accendi il
registratore su `x`; ogni calcolo successivo viene annotato; e la chiamata
`y.backward()` riavvolge il nastro, depositando la derivata in `x.grad`. Nel
codice sopra $y = x^2 + 2x$, la cui derivata è $2x + 2$; valutata in $x = 3$
dà $8$, esattamente il numero restituito. Non abbiamo scritto nessuna formula
di derivata: l'ha ricostruita la libreria.

`````

`````{tab} Superiore

Autograd implementa la differenziazione automatica in modalità *reverse*
(*reverse-mode autodiff*). Ogni operazione su tensori con
`requires_grad=True` aggiunge un nodo al grafo dinamico delle computazioni;
`y.backward()` percorre il grafo a ritroso applicando la **regola della
catena** vista nel capitolo di analisi e ottimizzazione:

$$
\frac{\partial \mathcal{L}}{\partial \theta}
= \frac{\partial \mathcal{L}}{\partial z}\,
  \frac{\partial z}{\partial \theta},
$$

dove $z$ è una quantità intermedia. Composta lungo tutto il grafo, questa
regola è precisamente l'algoritmo di **backpropagation** del capitolo
precedente: un'unica passata all'indietro calcola il gradiente rispetto a
*tutti* i parametri in tempo proporzionale a quello della passata in avanti.

Tre dettagli operativi che incontreremo di continuo. I gradienti si
**accumulano**: una `backward()` successiva, su un nuovo forward, somma in
`x.grad` invece di sovrascrivere (ripetere la *stessa* chiamata sullo stesso
grafo, invece, solleva un errore, salvo `retain_graph=True`), per questo il
training loop azzera i gradienti a ogni passo. Il
blocco `with torch.no_grad():` sospende la registrazione; indispensabile in
valutazione, quando i gradienti non servono e il grafo sarebbe solo memoria
sprecata. Infine `t.detach()` restituisce una vista del tensore staccata dal
grafo, e le operazioni in-place sui tensori tracciati vanno evitate perché
possono invalidare i valori salvati per la passata a ritroso.

`````

```{admonition} Da ricordare
:class: important
- Un **tensore** generalizza scalari, vettori e matrici a un numero qualunque
  di assi: lo descrivono **rank** (numero di assi) e **shape** (forma). In
  PyTorch le immagini sono *channels-first*: $(N, C, H, W)$.
- L'esecuzione è immediata e le regole (broadcasting compreso) sono quelle di
  NumPy, con cui i tensori si scambiano dati senza copie.
- Ogni tensore vive su un **device** (`"cpu"`, `"cuda"`, `"mps"`): i conti
  avvengono dove stanno i numeri, e il codice non cambia.
- **Autograd** calcola i gradienti da solo: `requires_grad=True` accende il
  registratore, `.backward()` riavvolge il nastro; è la backpropagation, cioè
  la regola della catena applicata a ritroso sul grafo dei calcoli.
```
