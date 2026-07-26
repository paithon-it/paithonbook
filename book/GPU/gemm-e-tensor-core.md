# GEMM: la moltiplicazione di matrici, spremuta

Prendi un addestramento qualunque (una CNN, un Transformer, un banale
percettrone multistrato) mettici sopra un profiler e guarda dove finisce il
tempo. In cima alla lista trovi quasi sempre la stessa voce, e non ha un nome
altisonante: si chiama **GEMM**, *GEneral Matrix Multiply*. È il nome che la
moltiplicazione tra due matrici porta da decenni nelle librerie BLAS, lo
standard del calcolo numerico, ed è probabilmente la singola routine più
ottimizzata della storia dell'informatica: a ogni generazione di hardware
qualcuno la riscrive da capo per spremerne l'ultima goccia.

Non è un caso. Nella sezione «Prestazioni e scala» del capitolo su PyTorch
abbiamo detto che una rete neurale, vista dall'hardware, è quasi soltanto una
cosa: moltiplicazioni tra matrici. Dallo strato di AlexNet
{cite}`krizhevsky2012imagenet` in poi, il grosso dei FLOP di qualunque rete è
GEMM. E la sezione «La memoria: il vero collo di bottiglia» ha lasciato in
sospeso una promessa; mostrare *per esteso, proprio qui,* il trucco con cui la
si rende veloce, il **tiling**. È il momento di mantenerla.

## Il conto, e il problema della versione ingenua

Moltiplicare $C = A \, B$, con $A$ di forma $(M, K)$ e $B$ di forma $(K, N)$,
significa riempire una matrice $C$ di forma $(M, N)$. Il conto in sé è
semplice e i FLOP si contano a occhio; il problema, come sempre su una GPU, non
è *quanti conti* fai, ma *quanti byte* muovi per farli.

`````{tab} Elementare
Ogni casella del risultato è una somma di prodotti: per riempirla prendi una
riga della prima matrice e una colonna della seconda, le moltiplichi elemento
per elemento e sommi. Facile. Ma immagina di farlo *una casella alla volta*,
andando ogni volta a ripescare la riga e la colonna dal magazzino lontano: la
memoria globale della sezione precedente. Due caselle vicine sulla stessa riga
usano la *stessa* riga della prima matrice: eppure, alla cieca, vai a
riprenderla da capo per ognuna. È come cucinare cento piatti identici correndo
in dispensa a prendere gli stessi ingredienti cento volte. I conti sono pochi,
i viaggi in dispensa tantissimi: e sappiamo dalla sezione sulla memoria che
sono i viaggi, non i conti, a decidere il tempo. La versione ingenua della
moltiplicazione tra matrici è, in una parola, *affamata di banda*.
`````

`````{tab} Superiore
Il prodotto costa circa $2 M N K$ operazioni in virgola mobile: per ognuno
degli $M \times N$ elementi di $C$, una somma di $K$ prodotti, cioè $K$
moltiplicazioni e $K$ addizioni. La versione ingenua assegna un thread a ogni
elemento di uscita e legge dalla memoria globale, per calcolare
$C_{ij} = \sum_{k} A_{ik} B_{kj}$, un'intera riga di $A$ ($K$ valori) e
un'intera colonna di $B$ ($K$ valori). Sommando su tutte le uscite sono
$2 M N K$ letture di elementi, a fronte di $2 M N K$ FLOP: in `float32`
(4 byte per elemento) l'**intensità aritmetica** vale

$$
I_\text{naive} = \frac{2 M N K}{4 \cdot 2 M N K} = \frac{1}{4} \ \text{FLOP/byte},
$$

*indipendente dalla taglia delle matrici*. Sul roofline della sezione «La
memoria: il vero collo di bottiglia» è un punto incollato in basso a sinistra:
profondamente memory-bound. La radice dello spreco è la ri-lettura: la stessa
riga di $A$ torna dalla HBM per ognuna delle $N$ colonne di $C$, la stessa
colonna di $B$ per ognuna delle $M$ righe. Si spostano montagne di byte per
rileggere all'infinito gli stessi numeri.
`````

## Tiling: portare gli ingredienti sul tavolo una volta sola

La cura è quella già anticipata nella sezione sulla memoria: **caricare una
volta, riusare in tanti**. Invece di calcolare $C$ una casella alla volta, la
si spezza in **tessere** (i *tile*); per ogni tessera si portano i blocchi
corrispondenti di $A$ e di $B$ nella shared memory (il ripiano condiviso della
scrivania) *una sola volta*, e da lì si riusano per tutti i prodotti della
tessera ({numref}`fig-gemm-tiling`).

```{figure} ../figures/gemm-tiling.svg
:name: fig-gemm-tiling
:alt: La matrice A sta a sinistra della matrice C e ne condivide le righe; la matrice B sta sopra C e ne condivide le colonne. Una tessera di C è evidenziata; nasce dal prodotto della banda-riga di A che le sta a sinistra per la banda-colonna di B che le sta sopra. Le bande si scorrono a blocchi lungo la dimensione K; un blocco di A e uno di B, in ocra, rappresentano i dati caricati una volta nella shared memory e riusati per tutta la tessera, con frecce che li collegano alla tessera di C.
:width: 90%

Il tiling del GEMM: una tessera di $C$ nasce dalla banda-riga di $A$ per la
banda-colonna di $B$. Le bande si scorrono lungo $K$ a blocchi; ogni blocco,
caricato una volta in shared memory (in ocra), serve tutti i prodotti della
tessera prima di essere scartato.
```

`````{tab} Elementare
Torniamo ai cento piatti identici. La mossa intelligente non è correre in
dispensa per ogni piatto: è portare *una cassetta* di ingredienti sul tavolo,
all'inizio, e da lì cucinare un'intera infornata di piatti. Il viaggio in
dispensa lo paghi una volta, non cento. Il tiling fa esattamente questo con la
moltiplicazione tra matrici: prende un blocchetto della prima matrice e uno
della seconda, li porta sul ripiano vicino ai calcolatori (la shared memory) e
li tiene lì finché ha finito di usarli per tutta la tessera del risultato che
sta calcolando. Ogni numero, caricato una volta, viene riusato molte volte
prima di essere buttato. Più grande è la tessera, più prodotti spremi da ogni
viaggio in dispensa, con un limite: sul ripiano ci sta solo una manciata di
pagine, e la tessera non può crescere oltre.
`````

`````{tab} Superiore
Spezziamo $C$ in tessere $T \times T$. Per calcolare una tessera si scorre la
dimensione $K$ a blocchi: a ogni passo si caricano *una volta* un blocco
$T \times T$ di $A$ e uno di $B$ nella shared memory, e si accumulano nella
tessera tutti i $T \times T$ prodotti che quei due blocchi generano, prima di
passare al blocco $K$ successivo. Ogni valore caricato serve $T$
moltiplicazioni invece di una: il **fattore di riuso** (la grandezza già
incontrata nella sezione sulla memoria) è $T$. Le letture dalla HBM scendono
da $2 M N K$ a circa $2 M N K / T$ elementi, e l'intensità aritmetica sale da
$\tfrac14$ a

$$
I_\text{tiled} \approx \frac{T}{4} \ \text{FLOP/byte}.
$$

Con $T = 32$ sono $8$ FLOP/byte: trentadue volte l'intensità della versione
ingenua. Sul roofline la tessera scivola verso destra, dal tetto di banda verso
quello di calcolo. Il tetto teorico resta l'$n/6$ della sezione precedente (con
$M=N=K=n$), e ci si avvicina **impilando il tiling su più livelli**: una tessera
grande in shared memory, una tessera più piccola nei registri di ogni thread.
È proprio l'architettura dei GEMM industriali.
`````

Il modo più limpido di vedere la struttura del tiling è scriverla, senza GPU,
in puro NumPy. Il codice qui sotto non è veloce (NumPy fa già i suoi prodotti
in modo ottimizzato) ma rende visibile *il nido di cicli*: si scorre $C$ a
tessere, e per ogni tessera si sommano i contributi dei blocchi lungo $K$.
Ogni `a` e `b` è un blocco «caricato in shared memory»; `a @ b` è il lavoro
che lo riusa. Il risultato coincide con la moltiplicazione diretta `A @ B`.

```python
import numpy as np

def matmul_a_blocchi(A, B, T=32):
    """Moltiplica A (M,K) per B (K,N) lavorando a tessere T×T.
    Stesso risultato di A @ B, ma esplicita il riuso: ogni blocco di A
    e di B, caricato una volta, serve tutti i prodotti della tessera."""
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N))
    for i in range(0, M, T):              # scorre le tessere di righe di C
        for j in range(0, N, T):          # scorre le tessere di colonne di C
            acc = np.zeros((min(T, M - i), min(T, N - j)))
            for k in range(0, K, T):      # somma sui blocchi lungo K
                a = A[i:i+T, k:k+T]       # blocco di A -> "shared memory"
                b = B[k:k+T, j:j+T]       # blocco di B -> "shared memory"
                acc += a @ b              # riuso: un blocco, molti prodotti
            C[i:i+T, j:j+T] = acc
    return C

A = np.random.randn(96, 80)
B = np.random.randn(80, 64)
print(np.allclose(matmul_a_blocchi(A, B), A @ B))   # True
```

Il triplo ciclo su tessere è lo scheletro di *ogni* GEMM veloce, dalla CPU alla
GPU: cambia solo chi fa i blocchi (i thread di un blocco CUDA) e dove vive la
tessera (la shared memory dell'SM). La logica è questa.

## I tensor core: un intero prodotto in un colpo

Il tiling risolve il problema dei byte. Resta quello dei conti: anche
saturando la banda, ogni moltiplicazione la deve pur fare qualcuno. Dal 2017
quel qualcuno non è più il generico CUDA core, ma un'unità costruita apposta
per il prodotto tra matrici: il **tensor core**.

`````{tab} Elementare
Immagina un ragioniere che deve compilare una piccola tabellina, quattro righe
per quattro colonne. Cella per cella, a mano, è un lavoro noioso. Ora immagina
un timbro speciale che stampa un intero blocco della tabellina in un colpo
solo: appoggi, premi, fatto. Il tensor core è quel timbro. Là dove un
calcolatore normale fa una moltiplicazione alla volta, il tensor core ne
compie in blocco un'intera manciata a ogni battito di clock. E c'è di più:
lavora con i numeri «arrotondati» a metà precisione (quelli della sezione
«Prestazioni e scala», più corti e più svelti da leggere) per fare le
moltiplicazioni, ma tiene il totale che va sommando in precisione piena, così
che la lunga somma non perda per strada le cifre che contano. È lo stesso
spirito del pesare gli ingredienti al grammo mentre si tiene il conto esatto
della spesa.
`````

`````{tab} Superiore
Introdotti con l'architettura **Volta** (la GPU V100, 2017), i tensor core
calcolano in un solo colpo di clock un piccolo prodotto-matrice con accumulo,
della forma

$$
D = A \, B + C,
$$

su tessere dell'ordine di $4 \times 4$ (l'operazione è esposta al
programmatore, a livello di warp, su tessere $16 \times 16$). Il cuore è la
**precisione mista** {cite}`micikevicius2018mixed`: gli ingressi $A$ e $B$
sono a 16 bit (`float16` sulla V100; le architetture successive, da Ampere in
poi, aggiungono anche `bfloat16`), ma l'accumulo di $C$ e $D$ resta a
`float32`, così la somma di molti prodotti non degrada. È, non a caso, la
forma «generale» del GEMM delle BLAS ($C \leftarrow \alpha A B + \beta C$,
moltiplica *e* accumula) cablata nel silicio. Il guadagno è di circa un ordine
di grandezza sul throughput di matmul rispetto ai CUDA core normali: è
l'innalzamento di $P_\text{picco}$ che, come notava il roofline, sposta il
ginocchio verso destra e rende la banda ancora più decisiva. Non li programmi
tu direttamente: **cuBLAS** e **cuDNN** li usano dietro le quinte ogni volta
che una `nn.Linear` o una convoluzione girano su una GPU recente in mezza
precisione.
`````

## In pratica: forme «tonde» e mezza precisione

Chiudiamo con un'onestà dovuta. Quasi certamente non scriverai mai un GEMM a
mano: librerie come cuBLAS e cuDNN, e generatori come **CUTLASS** o **Triton**
{cite}`tillet2019triton` (che incontreremo nella sezione sui kernel) lo fanno
meglio di quanto potrebbe chiunque, sfruttando tiling multilivello e tensor
core in modi che cambiano a ogni architettura. Perché allora capire il tiling?
Perché spiega due regole pratiche che spostano davvero il cronometro, e che
altrimenti sembrerebbero magia:

- **Dai alle matrici forme «tonde».** Se le dimensioni sono multiple della
  tessera (e dei blocchi che i tensor core divorano, tipicamente multipli di 8
  o 16) le tessere si riempiono senza avanzi, e nessun thread resta a
  processare un bordo incompleto. Portare la dimensione nascosta di un
  modello, o la taglia del vocabolario, al multiplo di 8 più vicino è spesso
  un guadagno gratuito. Una forma «storta» lascia i tensor core mezzi vuoti.
- **Usa la mezza precisione.** I tensor core esistono per lei: senza `autocast`
  (o `bfloat16`) girano a una frazione della loro potenza. Le quattro righe
  viste in «Prestazioni e scala» non sono un vezzo da datacenter, sono
  l'interruttore che accende il pezzo di silicio più veloce che hai.

Il tiling, insomma, è il ponte tra i due limiti del roofline: abbatte i byte
(riuso in shared memory) e mette al lavoro i FLOP (tensor core). La stessa
idea (riorganizzare un calcolo per non rileggere mai dalla HBM ciò che si può
tenere vicino) è precisamente ciò che, applicato all'attenzione, dà la
**FlashAttention** {cite}`dao2022flashattention` della prossima sezione. Il
GEMM è il primo, e più puro, esempio di una lezione che tornerà a ogni pagina.

```{admonition} Da ricordare
:class: important
- **GEMM** (*GEneral Matrix Multiply*) è il cuore di calcolo di ogni rete: il
  prodotto $C = A B$ con $A$ di forma $(M,K)$ e $B$ di forma $(K,N)$ costa circa
  $2 M N K$ FLOP.
- La versione **ingenua** rilegge dalla HBM le stesse righe e colonne
  all'infinito: intensità aritmetica fissa a $\tfrac14$ FLOP/byte,
  indipendente dalla taglia, profondamente **memory-bound**.
- Il **tiling** spezza $C$ in tessere e carica i blocchi di $A$ e $B$ in
  **shared memory** una volta sola, riusandoli: con tessere $T \times T$
  l'intensità sale a circa $T/4$ FLOP/byte, e sul roofline il GEMM scivola verso
  il regime **compute-bound**.
- I **tensor core** (dal 2017, Volta) eseguono un piccolo prodotto-matrice con
  accumulo $D = AB + C$ per colpo di clock, in **precisione mista**
  {cite}`micikevicius2018mixed` (ingressi 16 bit, accumulo 32 bit): circa un
  ordine di grandezza di throughput in più. `cuBLAS`/`cuDNN` li usano da soli.
- Raramente scriverai un GEMM a mano, ma capire il tiling spiega perché le
  forme «tonde» (multipli di 8/16) e la mezza precisione vanno più veloci.
- Riuso in shared memory + tensor core: la stessa ricetta tornerà, applicata
  all'attenzione, in **FlashAttention** {cite}`dao2022flashattention`.
```
