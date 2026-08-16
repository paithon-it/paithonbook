# Kernel: dare ordini a migliaia di thread

Scrivi `c = a + b` su due **tensori** PyTorch che vivono sulla GPU, e sembra
l'operazione più banale del mondo: la stessa somma che faresti su due numeri.
Ma se `a` e `b` hanno un milione di elementi ciascuno, dietro quella riga
innocua è appena partito un piccolo programma, lanciato in un colpo solo su un
milione di minuscoli esecutori che sommano ognuno la propria coppia di numeri,
tutti insieme. Quel programma ha un nome: **kernel**.

(Un tensore, se serve un ripasso, è la scatola in cui il deep learning tiene i
numeri: una lunga fila di valori, o una tabella, o una pila di tabelle,
comunque tanti numeri raccolti sotto un nome solo. `a + b` somma i due mucchi
posizione per posizione.) È il vero protagonista di
questo capitolo (l'unità di lavoro che gira davvero sulla GPU) e finora
l'abbiamo solo nominato. Nella sezione sull'architettura abbiamo visto *chi*
esegue (gli Streaming Multiprocessor, i warp da 32 thread); in quella sulla
memoria, *da dove* arrivano i dati. Qui vediamo *cosa* eseguono: il kernel,
appunto, e come lo si scrive.

## Un programma solo, un milione di esecutori

La cosa spiazzante, la prima volta, è che un kernel **non descrive il lavoro
intero**. Descrive quello di *un solo* esecutore, un thread, su un pezzetto di
dato: come una ricetta scritta per una porzione, che poi viene consegnata a
migliaia di cuochi in una volta sola.

Facciamo prima un po’ d'ordine sul pezzetto di dato. La fila di numeri su cui
un kernel lavora, messi in ordine uno dopo l'altro e ciascuno con la sua
posizione, si chiama **array**: è la forma più semplice di tensore, e nelle
prossime pagine le due parole si alterneranno. Il thread numero 7 si occuperà
del numero in posizione 7 dell'array, e così via.

Il kernel, dunque, si scrive per uno e si lancia su tutti. «Lanciare», qui, è
il verbo tecnico: si passa alla GPU il programmino e le si dice su quanti
esecutori farlo partire. Quell'insieme di esecutori è la **griglia** (in
inglese *grid*) della sezione precedente, cioè l'operazione intera, tutte le
squadre messe insieme. Ognuno esegue lo stesso codice su dati diversi, e per
sapere *su quali*, comincia col ricavare il proprio numero.

`````{tab} Elementare

Immagina di dover consegnare a mano un milione di volantini, uno per cassetta
della posta, e di avere a disposizione un esercito. Non scrivi un milione di
ordini diversi. Ne scrivi **uno solo**, che vale per tutti: «guarda il numero
cucito sulla tua divisa, va’ alla cassetta con quel numero, infila il
volantino». Poi lo leggi ad alta voce una volta, e l'intero esercito parte. Il
soldato numero 0 va alla cassetta 0, il soldato numero 999.999 alla cassetta
999.999, tutti insieme. L'ordine è identico per ognuno; l'unica cosa che cambia
è quel numero, che ciascuno ricava da sé per capire di quale cassetta
occuparsi. E lo ricava proprio come farebbe un esercito vero: sulla divisa non
c'è scritto «999.999», c'è scritto a quale squadra appartiene e che posto
occupa in fila, e da quei due il soldato si calcola il proprio numero. È la
figura qui sotto.

Un kernel è esattamente quell'ordine unico: una manciata di righe, scritte
pensando a *un* esecutore, che la GPU fa eseguire in parallelo a un'intera
folla. La riga «calcola il tuo numero» è la più importante di tutte: senza, i
soldati si accalcherebbero tutti sulla stessa cassetta.

`````

`````{tab} Superiore

Questo stile si chiama **SPMD**, *Single Program, Multiple Data*: un unico
programma, tante copie in esecuzione su porzioni diverse dei dati.
Sull'hardware NVIDIA si concretizza nel modello **SIMT** già visto
nell'architettura: i 32 thread di un warp ricevono la stessa istruzione nello
stesso momento, ed è l'hardware a raggruppare per l'emissione quelli che si
trovano allo stesso punto del programma (dal 2017, come si è visto, ciascuno ha
il proprio program counter, quindi non è più un avanzamento in blocco per
costruzione). Nel modello CUDA {cite}`nickolls2008scalable` il kernel è una
funzione (marcata `__global__` nel C per GPU) che riceve implicitamente le
coordinate del thread che la sta eseguendo, dentro la gerarchia griglia →
blocco → thread già introdotta. Tre variabili predefinite bastano a
orientarsi:

- `threadIdx`, la posizione del thread *dentro* il suo blocco;
- `blockIdx`, la posizione del blocco *dentro* la griglia;
- `blockDim`: quanti thread ha ogni blocco.

Da queste, la prima riga di quasi ogni kernel ricostruisce l’**indice
globale** del thread, la sua identità univoca nell'intera griglia:

$$
i = \text{blockIdx} \cdot \text{blockDim} + \text{threadIdx},
$$

dove $i$ è l'indice dell'elemento di cui *questo* thread si occupa
({numref}`fig-kernel-indice`). Con blocchi da 4 thread, il thread
`threadIdx=2` del blocco `blockIdx=1` lavora sull'elemento
$1 \cdot 4 + 2 = 6$. Da lì in poi il kernel è codice ordinario (legge `x[i]`,
calcola, scrive `y[i]`) con la sola avvertenza che l'ultimo blocco può sforare
la fine dell'array (se la lunghezza non è un multiplo esatto della dimensione
del blocco), e allora serve un controllo `i < n` per non scrivere fuori dai
bordi.

`````

```{figure} ../figures/kernel-griglia-indice.svg
:name: fig-kernel-indice
:alt: "Un array di otto elementi indicizzati da 0 a 7; sotto, otto thread raggruppati in due blocchi da quattro, ciascuno collegato da una freccia all'elemento dell'array di cui si occupa. Il thread con threadIdx 2 del blocco 1 è evidenziato in terracotta: la formula i = blockIdx per blockDim piu threadIdx dà 1 per 4 piu 2, cioè 6, l'elemento anch'esso evidenziato."
:width: 90%

Il numero cucito sulla divisa, disegnato. Ogni esecutore sa due cose, in quale
squadra è e che posto occupa dentro la squadra, e da quelle due ricava il
proprio numero unico in tutta l'operazione: qui il terzo della seconda squadra
(le squadre sono da quattro, e si conta da zero) trova
$1 \cdot 4 + 2 = 6$, e va a occuparsi dell'elemento numero 6. È l'unica riga
che distingue un esecutore dall'altro: il resto del kernel è identico per
tutti.
```

## Un kernel in Python: Triton

Un kernel, storicamente, si scrive in **C**, che è il linguaggio di
programmazione con cui si parla alle macchine quando si vuole controllare tutto:
potente, e faticoso. Chi lo usa deve calcolarsi gli indici a mano, decidere in
quale memoria mettere ogni numero, tenere a mente i dettagli della scheda che
ha davanti. Nel 2019 Philippe Tillet ha proposto un'alternativa che ha cambiato
le carte in tavola: **Triton** {cite}`tillet2019triton`, un modo di scrivere
kernel *dentro* Python.

Il motivo per cui ci riguarda da vicino è che Triton non serve solo a chi
scrive kernel a mano. Quando in «Prestazioni e scala» si chiedeva a PyTorch di
riscriversi il programma in forma più efficiente, la lingua in cui PyTorch se
lo riscrive è proprio questa: guardare un kernel Triton significa vedere che
cosa quella riga fabbrica.

Ecco un kernel che calcola in un colpo solo $y = \max(0,\; a x + b)$. In
parole povere: prendi ogni numero della lista, moltiplicalo per $a$, aggiungi
$b$ e, se il risultato viene negativo, sostituiscilo con uno zero. Con $a = 2$
e $b = 1$: da $3$ esce $7$; da $-4$ uscirebbe $-7$, che diventa $0$.
Quell'ultima mossa («se è sotto zero, metti zero») è la ReLU incontrata nel
capitolo sulle reti neurali, e la catena moltiplica-somma-ReLU ricorre ovunque
nelle reti. Che cosa calcola il kernel, insomma, lo abbiamo appena detto senza
simboli; il codice si può anche solo guardare da lontano, cogliendone la taglia:
il kernel vero e proprio sono le sette righe di conti in alto, il resto è il
modo di lanciarlo. La scheda Elementare qui sotto spiega proprio perché un kernel
Triton riesca a stare in così poche righe; la lettura riga per riga sta nella
scheda Superiore.

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

`````{tab} Elementare

C'è una differenza di *taglia* rispetto all'esercito di prima, e vale la pena
notarla. Con CUDA (il modo di programmare le GPU aperto da NVIDIA, quello di
cui parlava l'apertura del capitolo) l'ordine si dà al singolo soldato, che si
occupa di una cassetta sola. In Triton lo si dà a un'intera **squadra**: «voi
della seconda squadra, occupatevi delle cassette dalla 1024 alla 2047». (Le
squadre qui sono da 1024, ed è la riga `BLOCK_SIZE=1024` del codice: si sceglie
un multiplo di 32 perché i lavoratori marciano in plotoni da 32, e una squadra
di taglia diversa lascerebbe l'ultimo plotone mezzo vuoto.) Come
le mille e passa persone si spartiscano il lavoro dentro la squadra non è più
affar tuo: lo decide Triton, che sa come tenere occupati i lavoratori della GPU
meglio di quanto faresti a mano. Tu ragioni a squadre; il **compilatore**, cioè
il programma che traduce quello che scrivi in istruzioni per la macchina,
scende ai dettagli. È per questo che un kernel Triton sta in dieci righe di
Python leggibile invece che in una pagina di C.

`````

`````{tab} Superiore

Riga per riga: il decoratore `@triton.jit` dice a Triton di compilare la
funzione in un kernel per GPU. `tl.program_id(axis=0)` è l'analogo del
`blockIdx` di prima: l'identità di *questa* istanza del programma. Da lì
`offsets` costruisce, con `tl.arange`, l'elenco degli indici di cui l'istanza
si occupa; `mask` marca quelli validi (gli altri, oltre la fine dell'array,
verranno ignorati); `tl.load` legge dalla memoria solo le posizioni valide,
`tl.maximum(a * x + b, 0.0)` fa tutti i conti *sui dati appena caricati*, e
`tl.store` scrive il risultato. La funzione `fused_relu` sotto è il
**lancio**: alloca l'uscita, calcola quante istanze servono (`triton.cdiv`, la
divisione arrotondata per eccesso) e invoca il kernel con la sintassi
`fused_kernel[grid](...)`.

Il salto di astrazione è preciso: un *program instance* di Triton (un `pid`)
non è un thread, ma elabora un intero **blocco** di `BLOCK_SIZE` elementi. Il
programmatore lavora su vettori e tessere (`offsets` è un vettore di indici,
`x` un vettore di valori); il compilatore Triton mappa da sé quel lavoro sui
thread e sui warp dell'SM, sceglie il layout dei dati e sintetizza gli accessi
coalescenti alla memoria discussi nella sezione precedente. È un livello sopra
CUDA (dove invece scriveresti esplicitamente cosa fa *un* thread) e un livello
sotto PyTorch. `BLOCK_SIZE` è un `tl.constexpr`, cioè una costante nota a
tempo di compilazione: Triton la usa per generare codice specializzato
(srotolare cicli, dimensionare i registri), ed è uno dei pomelli su cui
l'autotuning cerca il valore migliore.

E il codice qui sopra non è illustrativo: gira. Non serve nemmeno una GPU per
guardarlo lavorare, perché con la variabile d'ambiente `TRITON_INTERPRET=1`
Triton esegue il kernel in un interprete sulla CPU, un thread per volta: con
$a = 2$ e $b = 1$ da $3$ esce $7$ e da $-4$ esce $0$, cioè esattamente i numeri
promessi qualche riga più su. E se si vuole vedere che cosa il compilatore ne
fa, `triton.compile` lo traduce nel **PTX** (l'assembly delle GPU NVIDIA) per
un'architettura scelta a tavolino, `sm_90` per esempio, senza che
quell'architettura sia presente. Vale la pena guardarci dentro, perché c'è la
morale della sezione scritta in linguaggio macchina: la moltiplicazione e la
somma non compaiono come istruzioni separate, al loro posto c'è una sola
`fma.rn.f32` (*fused multiply-add*), la fusione già avvenuta dentro una singola
istruzione. Quello per cui una GPU vera serve davvero è misurare quanto va
veloce, non sapere che cosa calcola.

`````

## Ogni lancio si paga: perché fondere

Perché prendersi la briga di scrivere un kernel fuso come quello, invece della
riga PyTorch pulita `y = torch.relu(a * x + b)`? Perché quella riga contiene
**tre** operazioni (moltiplica, somma, azzera i negativi) e nel modo di
eseguire di partenza, che si chiama *eager*, «impaziente», non sono affatto una
cosa sola: sono tre kernel distinti, lanciati uno dopo l'altro, e ogni lancio
ha un prezzo.

`````{tab} Elementare

Ogni volta che lanci un kernel è come fare una telefonata per piazzare un
ordine: c'è un costo fisso di «comporre il numero e spiegarsi» che paghi
uguale, che l'ordine sia grande o minuscolo. Scrivere `relu(a * x + b)` in
modo ingenuo sono **tre** telefonate: una per la moltiplicazione, una per la
somma, una per la ReLU. E c'è di peggio del costo delle chiamate. A ogni
telefonata, l'intero array viene tirato su dalla memoria, gli si fa un solo,
misero conticino, e lo si rispedisce indietro, per poi ritirarlo su di nuovo
alla telefonata dopo. Tre viaggi di andata e ritorno per un milione di numeri,
per fare un lavoro che si poteva fare in un viaggio solo. **Fondere** i kernel
vuol dire proprio questo: una telefonata sola, i dati salgono una volta, si
fanno tutti e tre i conti mentre sono lì a portata di mano, e si riscrive una
volta.

`````

`````{tab} Superiore

Ci sono due costi sovrapposti. Il primo è il **launch overhead**: ogni
invocazione di kernel richiede alla CPU di preparare e inviare il lancio alla
GPU, un costo dell'ordine dei microsecondi che, moltiplicato per una catena di
molte operazioni leggere, diventa visibile. Il secondo, più pesante, è il
traffico di memoria. Le operazioni *elemento-per-elemento* hanno intensità
aritmetica bassissima: come calcolato nel roofline della sezione precedente,
una somma vettoriale fa circa $1$ FLOP ogni $12$ byte spostati (profondamente
*memory-bound*). Tre op separate leggono e riscrivono l'array tre volte; il
kernel fuso una sola. A parità di FLOP, tagliare i byte alza l'intensità
aritmetica e sposta l'operazione verso destra sul roofline, dal tetto di banda
verso quello di calcolo. È esattamente ciò che fa la **kernel fusion** di
`torch.compile`, descritta in «Prestazioni e scala»: TorchInductor riconosce
le catene di operazioni fondibili e ne sintetizza un unico kernel Triton, così
che la memoria venga letta e scritta una volta invece di $k$. Il guadagno
cresce con quanto sei memory-bound, cioè, per gran parte delle operazioni
non-matmul, parecchio.

`````

## Da PyTorch al kernel: `eager` contro `compile`

Con questo in mente, si capisce cosa succede *davvero* sotto ogni riga di
PyTorch, e perché `torch.compile` sposti il cronometro. Sono due strade
diverse dal codice ai kernel.

`````{tab} Elementare

Le telefonate di poco fa erano il modo di partenza, quello *eager*: PyTorch
esegue il programma un'operazione alla volta, e ognuna è una telefonata. È
comodissimo, perché vedi il risultato di ogni passo appena lo scrivi e, se
qualcosa va storto, capisci subito quale riga è stata. Ma paghi il conto appena
visto: una telefonata e un viaggio in memoria per ogni riga.

La seconda strada è quella che si accende con la riga `torch.compile`: invece
di telefonare un ordine alla volta, consegni la lista intera. PyTorch se la
legge tutta *prima* di cominciare, riconosce le voci che si possono chiedere in
un colpo solo e le riscrive da sé come un ordine unico, che è esattamente il
lavoro di fusione di questa sezione. Le richieste davvero impegnative restano
affidate agli specialisti (kernel scritti a mano dal costruttore della GPU);
tutto il contorno di operazioni piccole viene accorpato. Meno telefonate, meno
viaggi, stesso identico risultato.

`````

`````{tab} Superiore

In modalità **eager**, quella di default, ogni operazione tensoriale viene
smistata (*dispatch*) al proprio kernel già compilato, uno per uno,
nell'ordine in cui la scrivi. Le operazioni pesanti non le esegue PyTorch con
kernel propri: le delega a librerie specializzate di NVIDIA (**cuBLAS** per le
moltiplicazioni tra matrici, **cuDNN** per le convoluzioni) kernel scritti e
ottimizzati a mano dal produttore dell'hardware (il GEMM tiled che ci sta
dentro è il tema della prossima sezione). Tutto il resto (somme, ReLU,
normalizzazioni) passa per i kernel *elementwise* di PyTorch, uno per
operazione. È flessibile e immediato da debuggare, ma paga i lanci e i viaggi
in memoria appena visti, uno per ogni riga.

In modalità **compile**, la catena cambia forma. Come descritto in
«Prestazioni e scala», TorchDynamo cattura la sequenza di operazioni in un
grafo e TorchInductor la ricompila: le operazioni pesanti restano affidate a
cuBLAS e cuDNN, ma le lunghe catene elementwise che le circondano (quelle che
in eager sarebbero stati dieci kernel e dieci viaggi in memoria) vengono
**fuse** in pochi kernel Triton generati al volo. Meno lanci, meno traffico
sulla HBM, la GPU meglio sfamata. La riga `model = torch.compile(model)` non è
magia: è questa fabbrica di kernel fusi che si mette in moto, e i kernel che
sforna sono scritti nel linguaggio che abbiamo appena letto.

`````

Con questo il quadro è completo: sappiamo *chi* esegue (le officine e i plotoni
da 32), *da dove* arrivano i dati (la piramide della memoria) e *che cosa* si
esegue (il kernel).

Tre sezioni hanno però lasciato per strada parecchi mestieri, e conviene
metterli in fila una volta per tutte, perché sono la stessa cosa vista da
angoli diversi. La formica, il rilevatore del censimento, il soldato con il
numero sulla divisa e il lavoratore alla scrivania sono tutti la stessa cosa:
un **thread**. Con l'avvertenza dell'architettura, che nelle analogie si perde:
il thread è il *compito*, non chi lo esegue; se qui sembrano coincidere è
perché ogni lavoratore ha esattamente un compito. Il plotone da 32 e la squadra
al tavolo comune sono il **warp** e il **blocco**: il primo è il gruppetto che
marcia insieme, il secondo la squadra più grande che condivide il ripiano.
L'officina con il caposquadra è lo **Streaming Multiprocessor**. La dispensa,
il magazzino e l'armadio dall'altra parte della stanza sono sempre la stessa
cosa, la memoria grande della scheda.

Resta la domanda che tiene insieme le tre risposte: com'è fatto il kernel su
cui una rete neurale spende quasi tutto il suo tempo, quello che moltiplica fra
loro due tabelloni di numeri. È la prossima sezione.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un **kernel** è il programmino che gira sulla GPU. La cosa spiazzante è che
  non descrive il lavoro intero: descrive quello di **un solo** esecutore su un
  pezzetto di dato, e la GPU lo fa eseguire identico a un'intera folla.
- La riga più importante di un kernel è quella in cui ogni esecutore **legge il
  proprio numero** e capisce di quale pezzetto occuparsi: è il numero cucito
  sulla divisa dei soldati che consegnano i volantini. Senza, si
  accalcherebbero tutti sulla stessa cassetta.
- **Triton** {cite}`tillet2019triton` è un modo di scrivere questi programmini
  direttamente in Python, dando l'ordine a una squadra invece che al singolo
  esecutore. È anche la lingua in cui PyTorch, quando gli si chiede di
  ottimizzare, si scrive da sé i propri kernel.
- Ogni volta che si lancia un kernel si paga una **telefonata**: un costo fisso
  che c'è sia per un ordine grande sia per uno minuscolo. E a ogni telefonata i
  dati fanno un viaggio di andata e ritorno dalla memoria.
- **Fondere** più operazioni in un kernel solo vuol dire fare una telefonata al
  posto di tre e un viaggio al posto di tre: stesso risultato, molto meno
  tempo. È il grosso di quello che fa quella riga di `torch.compile` vista nel
  capitolo su PyTorch.
```
`````

`````{tab} Superiore
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
  tanti kernel e tanti viaggi in memoria (*memory-bound*).
- La **kernel fusion** unisce più operazioni in un kernel solo (una lettura,
  una scrittura): alza l'intensità aritmetica e sposta l'operazione verso il
  tetto di calcolo del roofline.
- In **eager** ogni op è un kernel a sé (cuBLAS/cuDNN per matmul e convoluzioni,
  kernel elementwise per il resto); con `torch.compile` le catene elementwise
  vengono **fuse** in kernel Triton, riducendo lanci e traffico di memoria.
```
`````
