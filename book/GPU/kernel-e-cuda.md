# Kernel: dare ordini a migliaia di thread

Scrivi `c = a + b` su due tensori PyTorch che vivono sulla GPU, e sembra
l'operazione più banale del mondo: la stessa somma che faresti su due numeri.
Ma se `a` e `b` hanno un milione di elementi ciascuno, dietro quella riga
innocua è appena partito un piccolo programma, lanciato in un colpo solo su un
milione di minuscoli esecutori che sommano ognuno la propria coppia di numeri,
tutti insieme. Quel programma ha un nome: **kernel**.

(Un tensore, se serve un ripasso, è la scatola in cui il deep learning tiene i
numeri: una lunga fila di valori, o una tabella, o una pila di tabelle,
comunque tanti numeri raccolti sotto un nome solo. `a + b` somma i due mucchi
posizione per posizione.) Il kernel è l'unità di lavoro che gira davvero sulla
GPU, e finora l'abbiamo solo nominata. Nella sezione sull'architettura abbiamo
visto *chi* esegue (gli Streaming Multiprocessor, i warp da 32 thread); in
quella sulla memoria, *da dove* arrivano i dati. Qui vediamo *cosa* eseguono:
il kernel, appunto, e come lo si scrive.

## Un programma solo, un milione di esecutori

La cosa spiazzante, la prima volta, è che un kernel non descrive il lavoro
intero. Descrive quello di *un solo* esecutore, un thread, su un pezzetto di
dato: come una ricetta scritta per una porzione, che poi viene consegnata a
migliaia di cuochi in una volta sola.

Facciamo prima un po’ d'ordine sul pezzetto di dato. La fila di numeri su cui
un kernel lavora, messi in ordine uno dopo l'altro e ciascuno con la sua
posizione, si chiama **array**: è la forma più semplice di tensore, e nelle
prossime pagine le due parole si alterneranno. Il thread numero 7 si occuperà
del numero in posizione 7 dell'array, e così via.

Il kernel, dunque, si scrive per uno e si lancia su tutti. «Lanciare», qui, è
il verbo tecnico: si passa alla GPU il programmino e le si dice su quanti
esecutori farlo partire. Quell'insieme di esecutori è la griglia (in
inglese *grid*) vista nell'architettura, cioè l'operazione intera, tutti i
blocchi messi insieme. Ognuno esegue lo stesso codice su dati diversi, e per
sapere *su quali*, comincia col ricavare il proprio numero.

`````{tab} Elementare

Un milione di volantini da consegnare a mano, uno per cassetta della posta, e
un esercito a disposizione. Non scrivi un milione di ordini diversi. Ne scrivi
uno solo, che vale per tutti: «guarda il numero cucito sulla tua divisa, va’
alla cassetta con quel numero, infila il volantino». Poi lo leggi ad alta voce
una volta, e l'intero esercito parte. Il soldato numero 0 va alla cassetta 0,
il soldato numero 999.999 alla cassetta 999.999, tutti insieme. L'ordine è
identico per ognuno; l'unica cosa che cambia è quel numero, che ciascuno ricava
da sé per capire di quale cassetta occuparsi.

E lo ricava come farebbe un esercito vero. Sulla divisa non c'è scritto
«999.999»: c'è scritto a quale squadra appartiene e che posto occupa in fila.
Le squadre sono tutte della stessa misura, e quanto siano grandi lo sa ognuno.
Da queste tre cose il conto viene da sé: quante squadre ho davanti,
moltiplicato per quante persone stanno in una squadra, più il posto che occupo
io. Squadre da quattro, e si conta partendo da zero: chi sta al posto 2 della
squadra 1 ha davanti una squadra intera, cioè quattro cassette, e da lì conta
altri due posti, quindi la sua è la cassetta 6. È quello che disegna la
{numref}`fig-kernel-indice`.

Un kernel è esattamente quell'ordine unico: una manciata di righe, scritte
pensando a *un* esecutore, che la GPU fa eseguire in parallelo a un'intera
folla. La riga «calcola il tuo numero» è la più importante di tutte: senza, i
soldati si accalcherebbero tutti sulla stessa cassetta.

All'ordine manca ancora una riga, e serve perché le squadre sono tutte uguali
mentre il numero delle cassette non si lascia dividere così docilmente. Dieci
cassette e squadre da quattro: due squadre ne coprono otto, per le ultime due
ne serve una terza, e così partono dodici persone per dieci cassette. Le due
che avanzano andrebbero a cercare la cassetta 10 e la cassetta 11, che nel
palazzo non ci sono; e chi non trova la propria cassetta lascia comunque il
volantino da qualche parte, sotto una porta o nella buca del vicino, cioè dove
non andava. Perciò l'ordine finisce così: «se il tuo numero supera l'ultima
cassetta, fermati e non consegnare».

`````

`````{tab} Superiore

Questo stile si chiama **SPMD**, *Single Program, Multiple Data*: un unico
programma, tante copie in esecuzione su porzioni diverse dei dati.
Sull'hardware NVIDIA si concretizza nel modello SIMT già visto
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
({numref}`fig-kernel-indice`). Le tre variabili sono in realtà terne, con le
componenti `.x`, `.y` e `.z`: griglia e blocchi si possono disporre su una, due
o tre dimensioni, e su un array a una dimensione si usa la sola `x`, cioè
`blockIdx.x * blockDim.x + threadIdx.x`. Con blocchi da 4 thread, il thread
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

Un kernel, storicamente, si scrive in C, che è il linguaggio di
programmazione con cui si parla alle macchine quando si vuole controllare tutto:
potente, e faticoso. Chi lo usa deve calcolarsi gli indici a mano, decidere in
quale memoria mettere ogni numero, tenere a mente i dettagli della scheda che
ha davanti. Nel 2019 Philippe Tillet ha proposto un'alternativa che ha cambiato
le carte in tavola, **Triton** {cite}`tillet2019triton`: un linguaggio che
ragiona a *tessere*, cioè a riquadri di dati di forma fissa, invece che al
singolo esecutore. Quella prima versione era ancora un dialetto del C, e la
riscrittura *dentro* Python, quella con cui i kernel Triton si scrivono oggi,
arriva nel 2021, quando Tillet la pubblica da OpenAI
{cite}`tillet2021triton`.

Il motivo per cui ci riguarda da vicino è che Triton non serve solo a chi
scrive kernel a mano. Quando la sezione {doc}`«Prestazioni e scala»
</PyTorch/prestazioni>` chiedeva a PyTorch di riscriversi il programma in forma
più efficiente, con la riga `torch.compile`, la lingua in cui PyTorch se lo
riscrive è proprio questa: guardare un kernel Triton significa vedere che cosa
fabbrica quella riga.

Ecco un kernel che calcola in un colpo solo
$\mathbf{y} = \max(0,\; a \mathbf{x} + b)$. In parole povere: prendi ogni
numero della lista, moltiplicalo per $a$, aggiungi $b$ e, se il risultato
viene negativo, sostituiscilo con uno zero. Con $a = 2$
e $b = 1$: da $3$ esce $7$; da $-4$ uscirebbe $-7$, che diventa $0$.
Quell'ultima mossa («se è sotto zero, metti zero») è la ReLU incontrata fra le
{doc}`funzioni di attivazione </RetiNeurali/funzioni-attivazione>` delle reti
neurali, e la catena moltiplica-somma-ReLU ricorre ovunque nelle reti. Che cosa
calcola il kernel, insomma, lo abbiamo appena detto senza simboli; nel codice
il kernel vero e proprio sono le sette righe di conti in alto, il resto è il
modo di lanciarlo.

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

C'è una differenza di *taglia* rispetto all'esercito di prima. Con CUDA (il
modo di programmare le GPU aperto da NVIDIA) l'ordine si dà al singolo soldato,
che si occupa di una cassetta sola. In Triton lo si dà a un'intera squadra:
«voi della seconda squadra, occupatevi delle cassette dalla 1024 alla 2047».
Il lotto che tocca a ogni squadra è di 1024 cassette, molto più delle quattro
di poco fa, ed è la riga `BLOCK_SIZE=1024` del codice. Quante persone abbia la
squadra, invece, in quell'ordine non c'è scritto: se non lo dici lo decide
Triton, e le cassette restano comunque più delle persone, perché ognuna ne
lavora parecchie.

La misura del lotto non si sceglie a piacere: dev'essere una potenza di due,
256, 512, 1024. Le ragioni sono due, una per ciascuno dei due mestieri. I
lavoratori marciano in plotoni da 32, quindi un lotto che non sia un multiplo
di 32 lascerebbe l'ultimo plotone con delle corsie vuote; e chi traduce
l'ordine sa spezzare in parti uguali solo le taglie che si dimezzano fino in
fondo, e su una taglia come 96 (che pure di 32 è multiplo) si ferma e protesta
invece di provarci. Quale potenza di due, invece, non si sa a tavolino: dipende
dalla scheda che si ha davanti e dal conto che le si sta chiedendo, e il modo
di trovarlo è provarne qualcuna e cronometrare. Quel numero però va scritto
nell'ordine prima che l'ordine parta, non deciso per strada: chi traduce
l'ordine vuole saperlo in anticipo, così prepara istruzioni tagliate apposta
per squadre di quella taglia.

Poi si conta quanti lotti servono, e come sempre qualcosa avanza. Un milione
di cassette in lotti da 1024 fa 976 lotti pieni e un resto di 576 cassette:
di squadre ne partono 977, e l'ultima ha in mano un lotto in cui 448 cassette
non esistono. Per quelle vale la riga di prima, chi supera l'ultima cassetta si
ferma, ed è la riga `mask` del codice, quella che marca quali indici sono
buoni.

Come le persone della squadra si spartiscano poi il lotto non è
più affar tuo: lo decide Triton, che tiene occupati i lavoratori della GPU
quasi sempre come farebbe a mano un esperto. Tu ragioni a lotti; il
**compilatore**, cioè il programma che traduce quello che scrivi in istruzioni
per la macchina, scende ai dettagli. È per questo che un kernel Triton si
scrive in Python leggibile, senza toccare né l'indice del singolo esecutore né
la memoria in cui appoggiare i numeri.

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
non è un thread, ma elabora un intero blocco di `BLOCK_SIZE` elementi. Il
programmatore lavora su vettori e tessere (`offsets` è un vettore di indici,
`x` un vettore di valori); il compilatore Triton mappa da sé quel lavoro sui
thread e sui warp dell'SM, sceglie il layout dei dati e sintetizza gli accessi
coalescenti alla memoria discussi nella sezione precedente. È un livello sopra
CUDA (dove invece scriveresti esplicitamente cosa fa *un* thread) e un livello
sotto PyTorch. `BLOCK_SIZE` è un `tl.constexpr`, cioè una costante nota a
tempo di compilazione: Triton la usa per generare codice specializzato
(srotolare cicli, dimensionare i registri), ed è uno dei pomelli su cui
l’*autotuning* cerca il valore migliore, provandone diversi al primo lancio e
tenendo il più veloce.

E quel kernel non è illustrativo: gira. Non serve nemmeno una GPU per
guardarlo lavorare, perché con la variabile d'ambiente `TRITON_INTERPRET=1`
Triton esegue il kernel in un interprete sulla CPU, un'istanza di programma
per volta: con $a = 2$ e $b = 1$ da $3$ esce $7$ e da $-4$ esce $0$, cioè
esattamente i numeri promessi. E se si vuole vedere che cosa il compilatore ne
fa, `triton.compile` lo traduce nel **PTX** (la lingua intermedia in cui NVIDIA
descrive un programma per GPU, che il driver traduce poi nelle istruzioni della
scheda che si ha davanti) per un'architettura scelta a tavolino, `sm_90` per
esempio, senza che quell'architettura sia presente. Lì dentro la
moltiplicazione e la somma non compaiono come istruzioni separate: al posto
delle due c'è una `fma.rn.f32` (*fused multiply-add*). È una fusione di un
altro genere, dentro una singola istruzione invece che fra kernel: non
risparmia né un lancio né un viaggio in memoria, e in cambio arrotonda una
volta sola invece di due, quindi il numero che esce non è bit per bit quello
di una moltiplicazione seguita da una somma. Quello per cui una GPU vera serve
davvero è misurare quanto va veloce, non sapere che cosa calcola.

`````

## Ogni lancio si paga: perché fondere

Perché prendersi la briga di scrivere un kernel fuso come quello, invece della
riga PyTorch pulita `y = torch.relu(a * x + b)`? Perché quella riga contiene
tre operazioni (moltiplica, somma, azzera i negativi) e nel modo di eseguire
di partenza sono tre kernel distinti e non una cosa sola, lanciati uno dopo
l'altro, e ogni lancio ha un prezzo. Quel modo si chiama *eager*, «impaziente»,
perché esegue ogni operazione appena la incontra, senza aspettare di aver letto
il resto del programma.

`````{tab} Elementare

Ogni volta che lanci un kernel è come fare una telefonata per piazzare un
ordine: c'è un costo fisso di «comporre il numero e spiegarsi» che paghi
uguale, che l'ordine sia grande o minuscolo. Scrivere `relu(a * x + b)` in
modo ingenuo sono tre telefonate: una per la moltiplicazione, una per la
somma, una per la ReLU. E c'è di peggio del costo delle chiamate, perché a
ogni telefonata parte anche un camion, e quello si paga a merce trasportata.
L'intero array viene tirato su dalla memoria, gli si fa un solo, misero
conticino, e lo si rispedisce indietro, per poi ritirarlo su di nuovo alla
telefonata dopo. Tre viaggi di andata e ritorno per un milione di numeri,
per fare un lavoro che si poteva fare in un viaggio solo. **Fondere** i kernel
vuol dire proprio questo: una telefonata sola, i dati salgono una volta, si
fanno tutti e tre i conti mentre sono lì a portata di mano, e si riscrive una
volta.

Quanto si guadagni dipende dal rapporto fra il trasporto e il lavoro, e qui il
rapporto è impietoso: su ogni numero c'è da fare una moltiplicazione, una somma
e un confronto con lo zero, tre gesti che durano molto meno del viaggio che li
ha portati a destinazione. Con i viaggi che scendono da tre a uno, il tempo
scende quasi nella stessa proporzione. Se invece su ogni numero ci fosse
mezz'ora di conti da fare, le telefonate e i viaggi sarebbero un rumore di
fondo, e fondere non cambierebbe niente di misurabile. Si guadagna dove il
trasporto pesa più del conto, ed è il caso di quasi tutto quello che una rete
fa sui numeri uno per uno. Portata all'estremo, poi, la cura si esaurisce da
sé: a forza di togliere viaggi si arriva al punto in cui il trasporto smette di
essere il freno e a comandare il tempo comincia a essere il conto, e da lì in
poi fondere ancora non rende più niente.

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
verso quello di calcolo. È esattamente ciò che fa la kernel fusion di
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
un colpo solo e le riscrive da sé come un ordine unico: fonde le telefonate al
posto tuo, senza che tu debba scrivere niente. Le richieste davvero
impegnative restano affidate agli specialisti (kernel scritti a mano dal
costruttore della GPU); tutto il contorno di operazioni piccole viene
accorpato. Meno telefonate, meno viaggi, e lo stesso risultato salvo
l'ultima cifra: facendo i conti tutti di fila si arrotonda meno volte, e in
virgola mobile ogni arrotondamento si sente.

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

In modalità **compile**, la catena cambia forma. Come descritto in «Prestazioni
e scala», TorchDynamo cattura la sequenza di operazioni in un grafo e
TorchInductor la ricompila: le operazioni pesanti restano affidate a cuBLAS e
cuDNN, ma le lunghe catene elementwise che le circondano (quelle che in eager
sarebbero stati dieci kernel e dieci viaggi in memoria) vengono fuse in
pochi kernel Triton generati al volo. Meno lanci, meno traffico sulla HBM, la
GPU meglio sfamata. Dietro la riga `model = torch.compile(model)` c'è questa
fabbrica di kernel fusi che si mette in moto, e i kernel che sforna sono
scritti nel linguaggio che abbiamo appena letto.

`````

Con questo il quadro è completo: sappiamo *chi* esegue (le officine e i plotoni
da 32), *da dove* arrivano i dati (la piramide della memoria) e *che cosa* si
esegue (il kernel).

Tre sezioni hanno però lasciato per strada parecchi mestieri, e conviene
metterli in fila una volta per tutte. Il thread è l'unità di lavoro più
piccola, ed è il *compito*, non il pezzo di silicio che lo lavora: i compiti
che una scheda ha in carico sono molti più delle postazioni vere, ed è proprio
quell'eccedenza a tenerla sempre occupata. Il warp è il gruppetto di 32 thread
che avanzano insieme, il blocco è il gruppo più grande che condivide la memoria
veloce, lo Streaming Multiprocessor è l'officina che esegue i blocchi, e la
memoria grande della scheda è quella da cui i dati arrivano e a cui tornano.

Resta la domanda che tiene insieme le tre risposte: com'è fatto il kernel su
cui una rete neurale spende gran parte del suo tempo, quello che moltiplica fra
loro due tabelloni di numeri. È la prossima sezione.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Il rilevatore del censimento, il soldato con il numero sulla divisa e il
  lavoratore alla scrivania erano sempre la stessa cosa, un thread; la dispensa
  e il magazzino sempre la stessa, la memoria grande della scheda.
- Un kernel è il programmino che gira sulla GPU. La cosa spiazzante è che
  non descrive il lavoro intero: descrive quello di un solo esecutore su un
  pezzetto di dato, e la GPU lo fa eseguire identico a un'intera folla.
- La riga più importante di un kernel è quella in cui ogni esecutore legge il
  proprio numero e capisce di quale pezzetto occuparsi: è il numero cucito
  sulla divisa dei soldati che consegnano i volantini. Senza, si
  accalcherebbero tutti sulla stessa cassetta.
- Triton {cite}`tillet2019triton`, dal 2021 scrivibile dentro Python, dà
  l'ordine a una squadra invece che al singolo esecutore. È anche la lingua in
  cui PyTorch, quando gli si chiede di ottimizzare, si scrive da sé i propri
  kernel.
- Ogni volta che si lancia un kernel si paga una telefonata: un costo fisso
  che c'è sia per un ordine grande sia per uno minuscolo. E a ogni telefonata i
  dati fanno un viaggio di andata e ritorno dalla memoria.
- Fondere più operazioni in un kernel solo vuol dire fare una telefonata al
  posto di tre e un viaggio al posto di tre, e molto meno tempo. Si guadagna
  però solo dove il trasporto pesa più del conto: se su ogni numero ci fosse
  molto da calcolare, fondere non cambierebbe niente. È il grosso di quello
  che fa quella riga di `torch.compile` vista nella sezione
  {doc}`«Prestazioni e scala» </PyTorch/prestazioni>`.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Un kernel è il programma che gira sulla GPU: descrive cosa fa *un* thread
  su un pezzo di dato, e la GPU lo replica su tutta una griglia di thread
  (stile SPMD/SIMT). Ogni thread calcola il proprio indice globale
  $i = \text{blockIdx} \cdot \text{blockDim} + \text{threadIdx}$ per scegliere
  il dato su cui lavorare {cite}`nickolls2008scalable`.
- Triton {cite}`tillet2019triton`, dal 2021 scrivibile in Python, permette di
  ragionare a *tessere* di dati invece che a singoli thread: è il linguaggio
  in cui `torch.compile` (via TorchInductor) genera i suoi kernel fusi su GPU,
  mentre su CPU Inductor emette C++.
- Ogni lancio di kernel ha un costo fisso, e ogni operazione
  elemento-per-elemento rilegge e riscrive l'intero array: una catena di op è
  tanti kernel e tanti viaggi in memoria (*memory-bound*).
- La kernel fusion unisce più operazioni in un kernel solo (una lettura,
  una scrittura): alza l'intensità aritmetica e sposta l'operazione verso il
  tetto di calcolo del roofline.
- In eager ogni op è un kernel a sé (cuBLAS/cuDNN per matmul e convoluzioni,
  kernel elementwise per il resto); con `torch.compile` le catene elementwise
  vengono fuse in kernel Triton, riducendo lanci e traffico di memoria.
```
`````
