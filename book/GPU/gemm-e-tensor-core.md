# GEMM: la moltiplicazione di matrici, spremuta

Prendi un addestramento qualunque (una CNN, un Transformer, un banale
percettrone multistrato) mettici sopra un *profiler* (uno strumento che
cronometra il programma pezzo per pezzo e dice quanto tempo se ne va in
ciascuno) e guarda dove finisce il
tempo. In cima alla lista trovi quasi sempre la stessa voce, e non ha un nome
altisonante: si chiama **GEMM**, *GEneral Matrix Multiply*. È il nome che la
moltiplicazione tra due matrici porta da decenni nelle librerie BLAS, lo
standard del calcolo numerico, ed è probabilmente la singola routine più
ottimizzata della storia dell'informatica: a ogni generazione di hardware
qualcuno la riscrive da capo per spremerne l'ultima goccia.

Non è un caso. Nella sezione «Prestazioni e scala» del capitolo su PyTorch
abbiamo detto che una rete neurale, vista dall'hardware, è quasi soltanto una
cosa: moltiplicazioni tra matrici. Dallo strato di AlexNet
{cite}`krizhevsky2012imagenet` in poi, il grosso dei **FLOP** di qualunque rete
è GEMM (FLOP, come si è detto nella sezione sull'architettura, sta per
*floating-point operation*: un conto elementare su un numero con la virgola,
una moltiplicazione o una somma). E la sezione «La memoria: il vero collo di bottiglia» ha lasciato in
sospeso una promessa; mostrare *per esteso, proprio qui,* il trucco con cui la
si rende veloce, il **tiling**. È il momento di mantenerla.

## Il conto, e il problema della versione ingenua

Moltiplicare due matrici vuol dire riempire una tabella di risultati, e ogni
casella del risultato è una somma di prodotti: si prende una riga della prima
tabella e una colonna della seconda, le si moltiplica numero per numero e si
somma il tutto. In simboli, $\mathbf{C} = \mathbf{A}\,\mathbf{B}$, con
$\mathbf{A}$ di forma $(M, K)$ (cioè con $M$ righe e $K$ colonne) e
$\mathbf{B}$ di forma $(K, N)$, dà una matrice $\mathbf{C}$ di forma $(M, N)$.
Il conto in sé è semplice e i FLOP si contano a occhio; il problema, come
sempre su una GPU, non è *quanti conti* fai, ma *quanti byte* muovi per farli.

`````{tab} Elementare
Una riga per una colonna, moltiplica e somma: facile. Ma immagina di farlo
davvero *una casella alla volta*,
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
degli $M \times N$ elementi di $\mathbf{C}$, una somma di $K$ prodotti, cioè $K$
moltiplicazioni e $K$ addizioni. La versione ingenua assegna un thread a ogni
elemento di uscita e legge dalla memoria globale, per calcolare
$C_{ij} = \sum_{k} A_{ik} B_{kj}$, un'intera riga di $\mathbf{A}$ ($K$ valori)
e un'intera colonna di $\mathbf{B}$ ($K$ valori). Sommando su tutte le uscite sono
$2 M N K$ letture di elementi, a fronte di $2 M N K$ FLOP: in `float32`
(4 byte per elemento) l'**intensità aritmetica** vale

$$
I_\text{naive} = \frac{2 M N K}{4 \cdot 2 M N K} = \frac{1}{4} \ \text{FLOP/byte},
$$

*indipendente dalla taglia delle matrici*. Il conto assume il modello più
crudo: ogni lettura emessa viene servita dalla HBM, senza cache di mezzo.
Nella realtà la L2 e il broadcast dentro il warp recuperano una parte del
riuso, e il kernel ingenuo fa un po' meglio di $\tfrac14$; ma è un riuso
*sperato*, affidato alla cache, mentre il tiling che segue lo rende
*garantito* dal programma. Sul roofline della sezione «La
memoria: il vero collo di bottiglia» è comunque un punto incollato in basso a
sinistra: profondamente memory-bound. La radice dello spreco è la ri-lettura:
la stessa riga di $\mathbf{A}$ torna dalla HBM per ognuna delle $N$ colonne di
$\mathbf{C}$, la stessa colonna di $\mathbf{B}$ per ognuna delle $M$ righe. Si
spostano montagne di byte per rileggere all'infinito gli stessi numeri.
`````

## Tiling: portare gli ingredienti sul tavolo una volta sola

La cura è quella già anticipata nella sezione sulla memoria: **caricare una
volta, riusare in tanti**. Invece di calcolare $\mathbf{C}$ una casella alla
volta, la si spezza in **tessere** (i *tile*); per ogni tessera si portano i
blocchi corrispondenti di $\mathbf{A}$ e di $\mathbf{B}$ nella shared memory
(il ripiano condiviso della scrivania) *una sola volta*, e da lì si riusano per
tutti i prodotti della tessera ({numref}`fig-gemm-tiling`).

```{figure} ../figures/gemm-tiling.svg
:name: fig-gemm-tiling
:alt: La matrice A sta a sinistra della matrice C e ne condivide le righe; la matrice B sta sopra C e ne condivide le colonne. Una tessera di C è evidenziata; nasce dal prodotto della banda-riga di A che le sta a sinistra per la banda-colonna di B che le sta sopra. Le bande si scorrono a blocchi lungo la dimensione K; un blocco di A e uno di B, in ocra, rappresentano i dati caricati una volta nella shared memory e riusati per tutta la tessera, con frecce che li collegano alla tessera di C.
:width: 90%

Il tiling del GEMM: un quadratino del risultato nasce dalla striscia di righe
che ha a sinistra per la striscia di colonne che ha sopra. Le strisce si
scorrono a blocchi; ogni blocco, portato una volta sul tavolo di lavoro vicino
ai calcolatori (in ocra), serve tutti i prodotti del quadratino prima di essere
scartato.
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
Spezziamo $\mathbf{C}$ in tessere $T \times T$. Per calcolare una tessera si
scorre la dimensione $K$ a blocchi: a ogni passo si caricano *una volta* un
blocco $T \times T$ di $\mathbf{A}$ e uno di $\mathbf{B}$ nella shared memory,
e si accumulano nella tessera tutti i $T \times T$ prodotti che quei due
blocchi generano, prima di passare al blocco $K$ successivo. Ogni valore
caricato serve $T$ moltiplicazioni invece di una: il **fattore di riuso** (la
grandezza già incontrata nella sezione sulla memoria) è $T$. Le letture dalla
HBM scendono da $2 M N K$ a circa $2 M N K / T$ elementi, e l'intensità
aritmetica sale da $\tfrac14$ a

$$
I_\text{tiled} \approx \frac{T}{4} \ \text{FLOP/byte}.
$$

Con $T = 32$ sono $8$ FLOP/byte: trentadue volte l'intensità della versione
ingenua, e sul roofline la tessera scivola di parecchio verso destra.

Vale però la pena fermarsi a fare il confronto che questo capitolo ha insegnato
a fare, perché la conclusione non è quella che ci si aspetta: **otto FLOP/byte
non bastano ancora**. Il ginocchio della sezione precedente sta a $\approx 10$
con i CUDA core in `float32` e a $\approx 161$ con i tensor core in `float16`
su A100: otto è a sinistra di entrambi, quindi questa tessera, da sola, è
ancora memory-bound. E il tiling in shared memory non può cavarsela da sé:
servirebbe $T \ge 41$ per superare il primo ginocchio e, in `float16` (dove
$I \approx T/2$), $T \ge 323$ per superare il secondo, cioè una tessera che
occuperebbe oltre 400 KB contro i 164 KB di shared memory configurabile di una
A100. E non è questione di scegliere meglio la taglia: la tessera più grande
che in quei 164 KB ci sta è $204 \times 204$ (due blocchi da $T^2$ elementi a
2 byte l'uno), e dà $I \approx 102$. **Nessuna** tessera in shared memory
arriva a 161, e quella $128 \times 128$ dei GEMM industriali si ferma a 64.

La domanda diventa allora chi li salvi davvero, quei GEMM, dato che veloci lo
sono. La risposta sta un piano più giù ed è la **cache L2**. Il modello usato
finora (ogni lettura che esce dall'SM arriva fino alla HBM) è lo stesso modello
crudo del kernel ingenuo, e sbaglia allo stesso modo: le tessere di
$\mathbf{A}$ e di $\mathbf{B}$ che blocchi diversi si portano sul tavolo sono
*le stesse*, e a servirle è la L2 senza disturbare la memoria. Il conto, su un
GEMM $8192 \times 8192 \times 8192$ in `float16` con tessere $128 \times 128$:
nel modello crudo dalla HBM escono $2MNK/T$ elementi, cioè circa **17 GB**, che
a $1{,}935$ TB/s valgono $8{,}9$ ms contro i $3{,}5$ ms di calcolo dei tensor
core; con il riuso in L2 dalla HBM $\mathbf{A}$ e $\mathbf{B}$ escono una volta
sola e $\mathbf{C}$ ci rientra una volta sola, cioè circa **400 MB** e
$0{,}21$ ms. Stesso kernel, stessi FLOP: nel primo conto è bloccato dalla
memoria, nel secondo dai tensor core.

E allora a che serve il **secondo livello di tessere nei registri** che i GEMM
veri impilano davvero sotto il primo? Non a spostare il punto sul roofline
della HBM: lì il traffico lo decide soltanto la tessera in shared memory, e la
micro-tessera nei registri non ne cambia un byte. Serve a un problema diverso e
altrettanto reale, un piano più su: la **banda della shared memory**. Ogni SM
la serve con 32 banchi da 4 byte per colpo di clock, cioè 128 byte per ciclo,
che su una A100 fanno circa 19 TB/s aggregati (la stessa cifra che il paper di
FlashAttention attribuisce alla SRAM on-chip). Per alimentare 312 TFLOP/s di
tensor core servono dunque **16 FLOP per ogni byte letto dalla shared**, e un
thread che vada a prendersi i due operandi di ogni singolo prodotto ne fa
$0{,}5$: due FLOP ogni quattro byte in `float16`. Con una micro-tessera
$R \times R$ tenuta nei registri, invece, ogni thread legge $2R$ valori e ne
ricava $R^2$ prodotti, cioè $R/2$ FLOP per byte. È la stessa aritmetica del
tiling, con la shared al posto della HBM e i registri al posto della shared: il
riuso si ricompra un piano più giù, ed è possibile per la ragione vista nella
sezione sulla memoria, che il register file di un SM è il banco on-chip più
capiente che ci sia.

Ed è questa, più della gerarchia in sé, la morale che il capitolo si è
guadagnato: **c'è un roofline per ogni livello della piramide**, ciascuno con
la sua banda e il suo ginocchio, e ogni livello di tiling esiste per superare
il proprio. Quanto al tetto, l'$n/6$ della sezione precedente è un *ideale* che
richiederebbe le tre matrici intere on-chip, e per fortuna non serve
raggiungerlo: l'intensità realmente raggiungibile non cresce con $n$, ma con la
**radice** della memoria veloce disponibile (è il risultato classico di Hong e
Kung sulla complessità di I/O {cite}`hongkung1981io`, che dà
$\Omega(n^3/\sqrt{M_\text{chip}})$ trasferimenti e quindi
$I = O(\sqrt{M_\text{chip}})$, dove $M_\text{chip}$ è la memoria veloce
disponibile e non va confusa con le $M$ righe di $\mathbf{A}$).
`````

Chi programma può vedere la struttura del tiling scritta per esteso, senza GPU,
in puro NumPy. Il codice qui sotto non è veloce (NumPy fa già i suoi prodotti
in modo ottimizzato) ma rende visibile *il nido di cicli*: si scorre il
risultato a tessere, e per ogni tessera si sommano i contributi dei blocchi
lungo $K$. Ogni `a` e `b` è un blocco «caricato in shared memory»; `a @ b` è il
lavoro che lo riusa, e alla fine il risultato coincide con la moltiplicazione
diretta `A @ B`. Chi non programma può guardarlo da lontano cogliendone la
forma: tre cicli uno dentro l'altro, e in mezzo la riga in cui il blocco
portato sul tavolo viene usato. Quella forma è tutto il messaggio, e il
paragrafo dopo il codice la riassume in una riga.

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
calcolatore normale fa una moltiplicazione a ogni battito di clock, il tensor
core ne compie **sessantaquattro**, e alla fine, su una scheda intera, le
moltiplicazioni fra matrici vanno una quindicina di volte più veloci. E c'è di
più: per fare le moltiplicazioni lavora con i numeri «arrotondati» a metà
precisione (quelli della sezione «Prestazioni e scala», più corti e più svelti
da leggere), ma tiene il totale che va sommando in precisione piena, così che
la lunga somma non perda per strada le cifre che contano. È il gesto di chi
pesa gli ingredienti a occhio, perché tanto non cambia il piatto, ma il conto
della spesa lo tiene all'ultimo centesimo, perché lì gli errori si sommano.
`````

`````{tab} Superiore
Introdotti con l'architettura **Volta** (la GPU V100, 2017), i tensor core
calcolano in un solo colpo di clock un piccolo prodotto-matrice con accumulo,
della forma

$$
\mathbf{D} = \mathbf{A} \, \mathbf{B} + \mathbf{C},
$$

su tessere $4 \times 4$, cioè 64 moltiplicazioni-accumulo per colpo di clock
per unità (l'operazione è esposta al programmatore, a livello di warp, su
tessere $16 \times 16$). Le forme sono quelle di Volta: le generazioni
successive ne usano di più grandi, e da Hopper l'unità che emette l'istruzione
non è più il singolo warp ma un gruppo di quattro. Il cuore è la
**precisione mista** {cite}`micikevicius2018mixed`: gli ingressi $\mathbf{A}$ e
$\mathbf{B}$
sono a 16 bit (`float16` sulla V100; le architetture successive, da Ampere in
poi, aggiungono anche `bfloat16`), mentre l'accumulo di $\mathbf{C}$ e
$\mathbf{D}$ può restare
a `float32`: è la modalità dell'addestramento in precisione mista, così la
somma di molti prodotti non degrada (il silicio offre anche l'accumulo a 16
bit, usato talvolta in inferenza). È, non a caso, la
forma «generale» del GEMM delle BLAS
($\mathbf{C} \leftarrow \alpha \mathbf{A}\mathbf{B} + \beta \mathbf{C}$,
moltiplica *e* accumula) cablata nel silicio. Il guadagno è di circa un ordine
di grandezza sul throughput di matmul rispetto ai CUDA core normali (un fattore
8 sulla V100, 16 sull'A100, 15 sulla H100): è
l'innalzamento di $P_\text{picco}$ che, come notava il roofline, sposta il
ginocchio verso destra e rende la banda ancora più decisiva. Non li programmi
tu direttamente: **cuBLAS** e **cuDNN** li usano dietro le quinte ogni volta
che una `nn.Linear` o una convoluzione girano su una GPU recente in mezza
precisione.
`````

## L'altra strada: far scorrere i dati invece dei conti

Il tiling e i tensor core sono la risposta della GPU a una domanda che si può
affrontare anche in un modo completamente diverso, e vale la pena vederlo
perché mette in prospettiva tutto il capitolo. La domanda è sempre quella: come
si moltiplicano due matrici muovendo il meno possibile.

`````{tab} Elementare

Nella GPU il dato sta fermo in memoria e i calcolatori vanno a prenderlo. Il
tiling serve a ridurre i viaggi, ma i viaggi ci sono.

L'idea opposta è tenere fermi i calcolatori e far **scorrere** i dati
attraverso di loro, come su una catena di montaggio. Si dispone una griglia di
piccole unità, ognuna capace di una sola mossa: «moltiplica, e aggiungi il
risultato al totale che ti sta passando davanti».

Nella variante che ha avuto più fortuna il lavoro comincia **caricando** una
delle due matrici nella griglia, un numero per postazione, dove resterà ferma
per tutto il tempo: quei numeri non attraversano niente, aspettano. Poi i
numeri dell'altra matrice entrano da sinistra e scorrono lungo le righe. E
dall'alto verso il basso, di postazione in postazione, scende il **totale che
si sta formando**: ogni unità lo riceve, gli aggiunge il proprio prodotto, e lo
passa alla postazione sotto. In fondo alla colonna il totale esce già finito,
ed è la casella del risultato.

Il guadagno è tutto lì, in quel passaggio da vicino a vicino: un numero letto
una volta sola dalla memoria attraversa un'intera fila di postazioni,
servendole tutte senza essere mai riletto, e il totale si costruisce
camminando, invece di essere ripescato ogni volta da qualche parte. È
letteralmente la catena di montaggio: il pezzo scende lungo la linea e a ogni
postazione qualcuno gli aggiunge un componente. Si chiama **array sistolico**
perché i dati pulsano attraverso la griglia come il sangue spinto dal cuore, e
il nome lo scelse chi lo inventò, alla Carnegie Mellon, alla fine degli anni
Settanta.

`````

`````{tab} Superiore

Un **array sistolico** {cite}`kung1982why` è una griglia di elementi di
elaborazione identici, ciascuno collegato soltanto ai vicini immediati e capace
di una sola operazione: moltiplicare due ingressi, sommare il prodotto a un
valore che gli arriva, e propagare ai vicini al ciclo successivo. Non c'è
memoria condivisa, non c'è arbitraggio, non c'è un file di registri da
indirizzare: il movimento dei dati è cablato nella topologia. Che cosa stia
fermo e che cosa scorra, però, non è unico: è la scelta di **dataflow**, e
cambia la macchina.

Nella variante *output stationary*, quella classica di Kung, è il totale a
restare nell'elemento (un accumulatore interno) mentre entrambi gli operandi
scorrono. Nella variante *weight stationary*, che è quella adottata dagli
acceleratori più noti, l'elemento non tiene un totale: tiene un **peso**,
precaricato e fermo. Le attivazioni entrano da sinistra e attraversano le
righe; le **somme parziali** scendono di riga in riga raccogliendo un prodotto
per volta, e i totali completi escono in fondo all'array, in una memoria di
accumulatori posta sotto di esso. In entrambi i casi ogni valore letto una
volta dalla memoria esterna viene riusato lungo tutta una dimensione
dell'array, e il riuso non è ottenuto da una cache che *spera* di essere
colpita, ma dalla geometria.

La prima TPU di Google {cite}`jouppi2017datacenter` è la realizzazione più nota
del secondo schema: un array $256 \times 256$, cioè $65\,536$
moltiplicazioni-accumulo per ciclo di clock in una sola unità (le generazioni
successive usano più unità $128 \times 128$), con i pesi precaricati dall'alto,
i dati che entrano da sinistra e 4 MiB di accumulatori a 32 bit sotto la
matrice, che raccolgono una somma parziale da 256 elementi per ciclo. Il
confronto con la GPU non è «chi calcola di più»
ma «chi si muove di meno», ed è per questo che gli acceleratori dedicati
guadagnano soprattutto in **energia per operazione**, un tema che il capitolo
su MLOps riprende quando si tratta di pagare la bolletta.

Il prezzo della specializzazione è la rigidità. Un array sistolico è bravo
esattamente a una cosa. Se la matrice è più piccola dell'array, gran parte
degli elementi calcola zeri; se l'operazione non è un GEMM (una convoluzione
sparsa, un gather irregolare, un'operazione elemento per elemento), la
struttura non serve e bisogna uscire dall'array. La GPU, con la sua gerarchia
di memoria programmabile e i suoi CUDA core generici, perde in efficienza di
picco e guadagna in tutto il resto: è la stessa tensione fra lepre e formicaio
della prima sezione, spostata di un livello.

`````

## In pratica: forme «tonde» e mezza precisione

Chiudiamo con un'onestà dovuta. Quasi certamente non scriverai mai un GEMM a
mano: librerie come cuBLAS e cuDNN, e generatori come **CUTLASS** o **Triton**
{cite}`tillet2019triton` (che abbiamo incontrato nella sezione sui kernel) lo fanno
meglio di quanto potrebbe chiunque, sfruttando tiling multilivello e tensor
core in modi che cambiano a ogni architettura. Perché allora capire il tiling?
Perché spiega due regole pratiche che spostano davvero il cronometro, e che
altrimenti sembrerebbero magia:

- **Dai alle matrici forme «tonde»**, cioè misure che siano multipli di numeri
  come 8, 16 o 64 invece di misure qualsiasi. Se le dimensioni sono multiple
  della tessera (e dei blocchi che i tensor core divorano) le tessere si
  riempiono senza avanzi, e nessun lavoratore resta a processare un bordo
  incompleto. È il motivo per cui conviene portare al multiplo di 8 più vicino
  la *dimensione nascosta* di un modello (quanti numeri usa per rappresentare
  al proprio interno una parola o un'immagine) o la *taglia del vocabolario*
  (quante parole diverse conosce): un guadagno spesso gratuito. Una forma
  «storta» lascia i tensor core mezzi vuoti.
- **Usa la mezza precisione**, cioè numeri scritti con la metà delle cifre
  (16 bit invece di 32): occupano metà spazio e si leggono in metà tempo. I
  tensor core esistono per lei: senza `autocast`
  (o `bfloat16`) girano a una frazione della loro potenza. Le quattro righe
  viste in «Prestazioni e scala» non sono un vezzo da datacenter, sono
  l'interruttore che accende il pezzo di silicio più veloce che hai.

Tutte e due queste regole promettono un guadagno quasi gratuito, e tutte e due
capita che non lo diano. Le ragioni sono due, e nessuna delle due è quella che
si sospetta: le schede qui sotto le spiegano.

`````{tab} Elementare
Succede di arrotondare le misure e di non vedere cambiare niente, e la prima
ragione è che non conta solo *quante* caselle ha una riga: conta **dove la riga
comincia**. Una tabella con le misure giuste ma appoggiata di sghembo, che
comincia mezzo passo più in là di dove il timbro se l'aspetta, costringe a
leggere di traverso, e la strada veloce si chiude lo stesso.

La seconda ragione con le tessere non c'entra niente: c'entra con **quante
officine restano ferme all'ultimo giro**. Se i quadratini da calcolare sono un
po' più di un multiplo esatto delle officine disponibili, l'ultimo giro ne
impegna due o tre e tutte le altre stanno a guardare, e il tempo quasi
raddoppia. È il motivo per cui certe taglie «tonde» vanno peggio di taglie
vicine, e chi non conosce questo secondo effetto dà la colpa alla tessera, che
non c'entra.
`````

`````{tab} Superiore
La prima precisazione: il requisito vero non è sulle dimensioni logiche ma
sull'**allineamento in byte** (multipli di 16 byte sulla dimensione
principale), quindi una matrice con $M$, $N$ e $K$ multipli di 8 ma con un
*passo di riga* storto (lo *stride*, cioè la distanza in memoria fra l'inizio
di una riga e l'inizio della successiva, che in una vista o in una fetta non
coincide con la larghezza) esce comunque dalla strada veloce.

La seconda: esiste una quantizzazione gemella che non dipende dalla tessera ma
dal **numero di SM**, la *wave quantization*. Se il numero di tessere da
calcolare supera di poco un multiplo degli SM disponibili (108 su A100),
l'ultimo giro impegna pochissime officine e tutte le altre restano ferme: il
tempo raddoppia quasi. È per questo che certe taglie di batch «tonde» vanno
peggio di taglie vicine, e chi non conosce questo secondo effetto lo attribuisce
alla tessera, che non c'entra.
`````

Il tiling, insomma, è il ponte tra i due limiti del roofline: abbatte i byte
(riuso in shared memory) e mette al lavoro i FLOP (tensor core). La stessa
idea (riorganizzare un calcolo per non rileggere mai dalla HBM ciò che si può
tenere vicino) è precisamente ciò che, applicato all'attenzione, dà la
**FlashAttention** {cite}`dao2022flashattention` della prossima sezione. Il
GEMM è il primo, e più puro, esempio di una lezione che tornerà a ogni pagina.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Una rete neurale, vista dall'hardware, è quasi soltanto **moltiplicazioni fra
  tabelle di numeri**. Quella routine ha un nome che si incontra ovunque,
  **GEMM**, ed è probabilmente il pezzo di codice più ottimizzato della storia
  dell'informatica.
- Farla nel modo ovvio, una casella del risultato alla volta, vuol dire correre
  in dispensa a riprendere gli stessi ingredienti centinaia di volte. I conti
  sono pochi, i viaggi tantissimi: si finisce bloccati dal magazzino.
- La cura è il **tiling**: portare sul tavolo di lavoro un blocchetto di
  ciascuna tabella e usarlo per tutti i prodotti che può servire prima di
  buttarlo. Un viaggio invece di cento. Più grande il blocchetto, meglio è, ma
  sul tavolo ci sta poco: per questo i programmi veri fanno la stessa cosa
  **due volte**, con blocchetti grandi sul tavolo e blocchetti piccolissimi in
  mano a ciascun lavoratore.
- I **tensor core** sono il timbro che stampa un pezzo intero di tabellina in
  un colpo solo, sessantaquattro moltiplicazioni per battito, con i numeri
  arrotondati ma il totale tenuto preciso. È il pezzo di silicio più veloce di
  una GPU, e si accende dicendo a PyTorch di usare la mezza precisione.
- Ci si può muovere anche dall'altro capo: invece di andare a prendere i dati,
  si può far **scorrere** i dati fra postazioni vicine, come su una catena di
  montaggio, dove il totale scende lungo la colonna raccogliendo un pezzo per
  postazione. Si chiama **array sistolico** {cite}`kung1982why` ed è la scelta
  della TPU di Google {cite}`jouppi2017datacenter`: bravissima a fare questa
  cosa, inadatta a tutto il resto.
- Le due regole pratiche che restano, e che valgono anche a chi non scriverà
  mai un kernel: dare alle tabelle misure **tonde** (multipli di 8 o 16) e
  usare i **numeri corti**. Sono spesso guadagni gratuiti.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- **GEMM** (*GEneral Matrix Multiply*) è il cuore di calcolo di ogni rete: il
  prodotto $\mathbf{C} = \mathbf{A}\mathbf{B}$ con $\mathbf{A}$ di forma
  $(M,K)$ e $\mathbf{B}$ di forma $(K,N)$ costa circa
  $2 M N K$ FLOP.
- La versione **ingenua** rilegge dalla HBM le stesse righe e colonne
  all'infinito: nel modello senza cache l'intensità aritmetica resta ferma a
  $\tfrac14$ FLOP/byte, indipendente dalla taglia (nella realtà la L1 e la L2
  recuperano qualcosa, ma è riuso *sperato*, non garantito dal programma). In
  ogni caso, profondamente **memory-bound**.
- Il **tiling** spezza $\mathbf{C}$ in tessere e carica i blocchi di
  $\mathbf{A}$ e $\mathbf{B}$ in **shared memory** una volta sola, riusandoli:
  con tessere $T \times T$ l'intensità sale a circa $T/4$ FLOP/byte. Con
  $T = 32$ fa 8, che è ancora **a sinistra** di ogni ginocchio (10 con i CUDA
  core, 161 con i tensor core su A100), e nessuna tessera che stia nei 164 KB di
  shared di una A100 ci arriva: la più grande è $204 \times 204$, cioè
  $I \approx 102$. A tenere i GEMM veri lontani dal muro della HBM è la **cache
  L2**, che serve le tessere che i blocchi si ripassano (su un GEMM $8192^3$ in
  `float16` il traffico scende da 17 GB a circa 400 MB, e il tempo da $8{,}9$ a
  $0{,}21$ ms contro $3{,}5$ ms di calcolo). Il **secondo** livello di tessere,
  quello nei registri, risolve un problema diverso: la **banda della shared
  memory**, circa 19 TB/s su A100 contro i 16 FLOP/byte che i tensor core
  pretendono. C'è un roofline per ogni livello della piramide, e ogni tiling
  supera il proprio. L'$n/6$ è un tetto ideale; il raggiungibile cresce come
  $\sqrt{M_\text{chip}}$, non come $n$ {cite}`hongkung1981io`.
- I **tensor core** (dal 2017, Volta) eseguono un piccolo prodotto-matrice con
  accumulo $\mathbf{D} = \mathbf{A}\mathbf{B} + \mathbf{C}$ per colpo di clock
  (64 FMA per unità, su tessere $4\times4$ nella forma Volta), in **precisione
  mista** {cite}`micikevicius2018mixed` (ingressi 16 bit, accumulo 32 bit):
  circa un ordine di grandezza di throughput in più. `cuBLAS`/`cuDNN` li usano
  da soli.
- L'**array sistolico** {cite}`kung1982why` risolve lo stesso problema
  dall'altro capo: invece di andare a prendere i dati, li fa **scorrere** fra
  unità adiacenti, e il riuso è nella geometria invece che in una cache. La TPU
  {cite}`jouppi2017datacenter` usa la variante *weight stationary*: pesi
  precaricati e fermi, attivazioni da sinistra, somme parziali che scendono
  verso gli accumulatori sotto l'array. Massima efficienza sul GEMM, rigidità
  su tutto il resto.
- Raramente scriverai un GEMM a mano, ma capire il tiling spiega perché le
  forme «tonde» (allineamento a 16 byte, più che multipli logici di 8/16) e la
  mezza precisione vanno più veloci; e perché esiste una *wave quantization*
  legata al numero di SM, che le forme tonde non curano.
- Riuso in shared memory + tensor core: la stessa ricetta tornerà, applicata
  all'attenzione, in **FlashAttention** {cite}`dao2022flashattention`.
```
`````
