# Kernel: dare ordini a migliaia di thread

Scrivi `c = a + b` su due tensori PyTorch che vivono sulla GPU, e sembra
l'operazione più banale del mondo — la stessa somma che faresti su due numeri.
Ma se `a` e `b` hanno un milione di elementi, dietro quella riga innocua è
appena partito un piccolo programma, lanciato in un colpo solo su un milione di
minuscoli esecutori che sommano ognuno la propria coppia di numeri, in
parallelo. Quel programma ha un nome: **kernel**. È il vero
protagonista di questo capitolo — l'unità di lavoro che gira davvero sulla GPU
— e finora l'abbiamo solo nominato. Nella sezione sull'architettura abbiamo
visto *chi* esegue (gli Streaming Multiprocessor, i warp da 32 thread); in
quella sulla memoria, *da dove* arrivano i dati. Qui vediamo *cosa* eseguono:
il kernel, appunto, e come lo si scrive.

## Un programma solo, un milione di esecutori

La cosa spiazzante, la prima volta, è che un kernel non descrive cosa fa la GPU
sull'intero array. Descrive cosa fa **un singolo thread** su *un pezzetto* di
dato. Poi lo lanci su una griglia, e l'hardware lo replica identico su tutti i
thread. Ognuno esegue lo stesso codice, ma su dati diversi — e per sapere *su
quali* dati, ogni thread comincia col calcolare il proprio indice.

`````{tab} Elementare

Immagina di dover consegnare a mano un milione di volantini, uno per cassetta
della posta, e di avere a disposizione un esercito. Non scrivi un milione di
ordini diversi. Ne scrivi **uno solo**, che vale per tutti: «guarda il numero
cucito sulla tua divisa, va' alla cassetta con quel numero, infila il
volantino». Poi lo leggi ad alta voce una volta, e l'intero esercito parte. Il
soldato numero 0 va alla cassetta 0, il numero 41.999 alla cassetta 41.999,
tutti insieme. L'ordine è identico per ognuno; l'unica cosa che cambia è quel
numero sulla divisa, che ciascuno legge da sé per capire di quale cassetta
occuparsi. Un kernel è esattamente quell'ordine unico: una manciata di righe,
scritte pensando a *un* esecutore, che la GPU fa eseguire in parallelo a
un'intera folla. La riga «leggi il tuo numero» è la più importante di tutte:
senza, i soldati si accalcherebbero tutti sulla stessa cassetta.

`````

`````{tab} Superiore

Questo stile si chiama **SPMD**, *Single Program, Multiple Data*: un unico
programma, tante copie in esecuzione su porzioni diverse dei dati. Sull'hardware
NVIDIA si concretizza nel modello **SIMT** già visto nell'architettura (i 32
thread di un warp eseguono la stessa istruzione in lockstep). Nel modello CUDA
{cite}`nickolls2008scalable` il kernel è una funzione — marcata `__global__` nel
C per GPU — che riceve implicitamente le coordinate del thread che la sta
eseguendo, dentro la gerarchia griglia → blocco → thread già introdotta. Tre
variabili predefinite bastano a orientarsi:

- `threadIdx` — la posizione del thread *dentro* il suo blocco;
- `blockIdx` — la posizione del blocco *dentro* la griglia;
- `blockDim` — quanti thread ha ogni blocco.

Da queste, la prima riga di quasi ogni kernel ricostruisce l'**indice globale**
del thread — la sua identità univoca nell'intera griglia:

$$
i = \text{blockIdx} \cdot \text{blockDim} + \text{threadIdx},
$$

dove $i$ è l'indice dell'elemento di cui *questo* thread si occupa
({numref}`fig-kernel-indice`). Con blocchi da 4 thread, il thread `threadIdx=2`
del blocco `blockIdx=1` lavora sull'elemento $1 \cdot 4 + 2 = 6$. Da lì in poi
il kernel è codice ordinario — legge `x[i]`, calcola, scrive `y[i]` — con la
sola avvertenza che l'ultimo blocco può sforare la fine dell'array (se la
lunghezza non è un multiplo esatto della dimensione del blocco), e allora serve
un controllo `i < n` per non scrivere fuori dai bordi.

`````

```{figure} ../figures/kernel-griglia-indice.svg
:name: fig-kernel-indice
:alt: "Un array di otto elementi indicizzati da 0 a 7; sotto, otto thread raggruppati in due blocchi da quattro, ciascuno collegato da una freccia all'elemento dell'array di cui si occupa. Il thread con threadIdx 2 del blocco 1 è evidenziato in terracotta: la formula i = blockIdx per blockDim piu threadIdx dà 1 per 4 piu 2, cioè 6, l'elemento anch'esso evidenziato."
:width: 90%

Ogni thread calcola il proprio indice globale $i$ dalle coordinate
(`blockIdx`, `threadIdx`) e lo usa per scegliere il proprio elemento. È l'unica
riga che distingue un esecutore dall'altro: il resto del kernel è identico per
tutti.
```

## Un kernel in Python: Triton

Scrivere kernel nel C per GPU è potente ma ostico — indici a mano, gestione
della memoria, dettagli dell'hardware. Nel 2019 Philippe Tillet ha proposto
un'alternativa che ha cambiato le carte in tavola: **Triton**
{cite}`tillet2019triton`, un linguaggio per kernel *dentro* Python. Il motivo
per cui ci riguarda da vicino è dichiarato nella sezione «Prestazioni e scala»:
Triton è il linguaggio in cui `torch.compile`, tramite il suo compilatore
TorchInductor, **genera** i kernel fusi. Impararne la forma significa vedere,
dal di dentro, cosa fabbrica quella riga di `torch.compile`.

Ecco un kernel che calcola in un colpo solo $y = \max(0,\; a x + b)$ — una
moltiplicazione per uno scalare, una somma e una ReLU, il genere di catena che
ricorre ovunque nelle reti:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def fused_kernel(x_ptr, out_ptr, a, b,
                 n_elementi, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                    # indice del blocco di programma
    inizio = pid * BLOCK_SIZE
    offsets = inizio + tl.arange(0, BLOCK_SIZE)    # gli indici che questo blocco elabora
    mask = offsets < n_elementi                    # non uscire dal bordo dell'array
    x = tl.load(x_ptr + offsets, mask=mask)        # UNA lettura dalla memoria
    y = tl.maximum(a * x + b, 0.0)                  # a*x + b e poi ReLU, tutto insieme
    tl.store(out_ptr + offsets, y, mask=mask)      # UNA scrittura in memoria


def fused_relu(x, a, b):
    out = torch.empty_like(x)
    n = out.numel()
    # quanti blocchi di programma servono per coprire tutti gli elementi
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    fused_kernel[grid](x, out, a, b, n, BLOCK_SIZE=1024)
    return out
```

Riga per riga: il decoratore `@triton.jit` dice a Triton di compilare la
funzione in un kernel per GPU. `tl.program_id(axis=0)` è l'analogo del
`blockIdx` di prima — l'identità di *questa* istanza del programma. Da lì
`offsets` costruisce, con `tl.arange`, l'elenco degli indici di cui l'istanza si
occupa; `mask` marca quelli validi (gli altri, oltre la fine dell'array,
verranno ignorati); `tl.load` legge dalla memoria solo le posizioni valide,
`tl.maximum(a * x + b, 0.0)` fa tutti i conti *sui dati appena caricati*, e
`tl.store` scrive il risultato. La funzione `fused_relu` sotto è il **lancio**:
alloca l'uscita, calcola quante istanze servono (`triton.cdiv`, la divisione
arrotondata per eccesso) e invoca il kernel con la sintassi
`fused_kernel[grid](...)`.

`````{tab} Elementare

C'è una differenza di *taglia* rispetto all'esercito di prima, e vale la pena
notarla. Nel modello CUDA ogni soldato si occupa di una sola cassetta. In
Triton dài l'ordine non al singolo soldato ma a un'intera **squadra**: «voi
mille, occupatevi delle cassette da 1000 a 1999». Come le mille persone si
spartiscano il lavoro dentro la squadra non è più affar tuo — lo decide Triton,
che sa come tenere occupati i thread della GPU meglio di quanto faresti a mano.
Tu ragioni a blocchi di lavoro; il compilatore scende ai dettagli. È per questo
che un kernel Triton sta in dieci righe di Python leggibile invece che in una
pagina di codice di basso livello.

`````

`````{tab} Superiore

Il salto di astrazione è preciso: un *program instance* di Triton (un `pid`) non
è un thread, ma elabora un intero **blocco** di `BLOCK_SIZE` elementi. Il
programmatore lavora su vettori e tessere (`offsets` è un vettore di indici,
`x` un vettore di valori); il compilatore Triton mappa da sé quel lavoro sui
thread e sui warp dell'SM, sceglie il layout dei dati e sintetizza gli accessi
coalescenti alla memoria discussi nella sezione precedente. È un livello sopra
CUDA — dove invece scriveresti esplicitamente cosa fa *un* thread — e un livello
sotto PyTorch. `BLOCK_SIZE` è un `tl.constexpr`, cioè una costante nota a tempo
di compilazione: Triton la usa per generare codice specializzato (srotolare
cicli, dimensionare i registri), ed è uno dei pomelli su cui l'autotuning cerca
il valore migliore. Il codice qui sopra è illustrativo: per girare servono una
GPU e Triton installato, e in pratica lo si lascia scrivere a TorchInductor.

`````

## Ogni lancio si paga: perché fondere

Perché prendersi la briga di scrivere un kernel fuso come quello, invece di tre
righe PyTorch pulite `y = torch.relu(a * x + b)`? Perché quelle tre righe, in
esecuzione *eager*, non sono affatto una cosa sola: sono tre kernel distinti,
lanciati uno dopo l'altro, e ogni lancio ha un prezzo.

`````{tab} Elementare

Ogni volta che lanci un kernel è come fare una telefonata per piazzare un
ordine: c'è un costo fisso di «comporre il numero e spiegarsi» che paghi
uguale, che l'ordine sia grande o minuscolo. Scrivere `relu(a * x + b)` in modo
ingenuo sono **tre** telefonate: una per la moltiplicazione, una per la somma,
una per la ReLU. E c'è di peggio del costo delle chiamate. A ogni telefonata,
l'intero array viene tirato su dalla memoria, gli si fa un solo, misero
conticino, e lo si rispedisce indietro — per poi ritirarlo su di nuovo alla
telefonata dopo. Tre viaggi di andata e ritorno per un milione di numeri, per
fare un lavoro che si poteva fare in un viaggio solo. **Fondere** i kernel vuol
dire proprio questo: una telefonata sola, i dati salgono una volta, si fanno
tutti e tre i conti mentre sono lì a portata di mano, e si riscrive una volta.

`````

`````{tab} Superiore

Ci sono due costi sovrapposti. Il primo è il **launch overhead**: ogni
invocazione di kernel richiede alla CPU di preparare e inviare il lancio alla
GPU, un costo dell'ordine dei microsecondi che, moltiplicato per una catena di
molte operazioni leggere, diventa visibile. Il secondo, più pesante, è il
traffico di memoria. Le operazioni *elemento-per-elemento* hanno intensità
aritmetica bassissima: come calcolato nel roofline della sezione precedente,
una somma vettoriale fa circa $1$ FLOP ogni $12$ byte spostati — profondamente
*memory-bound*. Tre op separate leggono e riscrivono l'array tre volte; il
kernel fuso una sola. A parità di FLOP, tagliare i byte alza l'intensità
aritmetica e sposta l'operazione verso destra sul roofline, dal tetto di banda
verso quello di calcolo. È esattamente ciò che fa la **kernel fusion** di
`torch.compile`, descritta in «Prestazioni e scala»: TorchInductor riconosce le
catene di operazioni fondibili e ne sintetizza un unico kernel Triton, così che
la memoria venga letta e scritta una volta invece di $k$. Il guadagno cresce
con quanto sei memory-bound — cioè, per gran parte delle operazioni non-matmul,
parecchio.

`````

## Da PyTorch al kernel: `eager` contro `compile`

Con questo in mente, si capisce cosa succede *davvero* sotto ogni riga di
PyTorch — e perché `torch.compile` sposti il cronometro. Sono due strade
diverse dal codice ai kernel.

In modalità **eager**, quella di default, ogni operazione tensoriale viene
smistata (*dispatch*) al proprio kernel già compilato, uno per uno, nell'ordine
in cui la scrivi. Le operazioni pesanti non le esegue PyTorch con kernel
propri: le delega a librerie specializzate di NVIDIA — **cuBLAS** per le
moltiplicazioni tra matrici, **cuDNN** per le convoluzioni — kernel scritti e
ottimizzati a mano dal produttore dell'hardware (il GEMM tiled che ci sta
dentro è il tema della prossima sezione). Tutto il resto — somme, ReLU,
normalizzazioni — passa per i kernel *elementwise* di PyTorch, uno per
operazione. È flessibile e immediato da debuggare, ma paga i lanci e i viaggi
in memoria appena visti, uno per ogni riga.

In modalità **compile**, la catena cambia forma. Come descritto in «Prestazioni
e scala», TorchDynamo cattura la sequenza di operazioni in un grafo e
TorchInductor la ricompila: le operazioni pesanti restano affidate a cuBLAS e
cuDNN, ma le lunghe catene elementwise che le circondano — quelle che in eager
sarebbero stati dieci kernel e dieci viaggi in memoria — vengono **fuse** in
pochi kernel Triton generati al volo. Meno lanci, meno traffico sulla HBM, la
GPU meglio sfamata. La riga `model = torch.compile(model)` non è magia: è
questa fabbrica di kernel fusi che si mette in moto, e i kernel che sforna sono
scritti nel linguaggio che abbiamo appena letto.

```{admonition} Da ricordare
:class: important
- Un **kernel** è il programma che gira sulla GPU: descrive cosa fa *un* thread
  su un pezzo di dato, e la GPU lo replica su tutta una griglia di thread
  (stile **SPMD/SIMT**). Ogni thread calcola il proprio **indice globale**
  $i = \text{blockIdx} \cdot \text{blockDim} + \text{threadIdx}$ per scegliere
  il dato su cui lavorare {cite}`nickolls2008scalable`.
- **Triton** {cite}`tillet2019triton` permette di scrivere kernel in Python
  ragionando a *blocchi* di lavoro (non a singoli thread): è il linguaggio in
  cui `torch.compile` (via TorchInductor) genera i suoi kernel fusi.
- Ogni **lancio di kernel** ha un costo fisso, e ogni operazione
  elemento-per-elemento rilegge e riscrive l'intero array: una catena di op è
  tanti kernel e tanti viaggi in memoria — *memory-bound*.
- La **kernel fusion** unisce più operazioni in un kernel solo (una lettura,
  una scrittura): alza l'intensità aritmetica e sposta l'operazione verso il
  tetto di calcolo del roofline.
- In **eager** ogni op è un kernel a sé (cuBLAS/cuDNN per matmul e convoluzioni,
  kernel elementwise per il resto); con `torch.compile` le catene elementwise
  vengono **fuse** in kernel Triton, riducendo lanci e traffico di memoria.
```
