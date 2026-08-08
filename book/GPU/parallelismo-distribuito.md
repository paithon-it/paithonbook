# Oltre una GPU: parallelismo distribuito

Facciamo un conto che mette le cose in prospettiva. Un modello come GPT-3, con
i suoi 175 miliardi di parametri, tenuti in mezza precisione a 2 byte l'uno,
occupa già 350 GB solo per i pesi. Ma addestrare è un'altra faccenda:
l'ottimizzatore Adam, nella versione a precisione mista, si porta dietro per
ogni parametro una copia dei pesi in `float32`, più due statistiche (momento e
varianza), più gradienti (in tutto circa 16 byte per parametro). Fanno **2,8
terabyte**. Una GPU da datacenter di fascia alta, una H100, ne ha 80 di
memoria: ci vorrebbero trentacinque schede *soltanto per contenere lo stato
dell'addestramento*, prima ancora di parlare di velocità. Alcuni modelli non
stanno in una GPU sola, né per memoria, né per tempo.

Nella sezione «Prestazioni e scala» del capitolo su PyTorch abbiamo già visto
la strategia più comune per usare più schede: il **parallelismo dati** con
`DistributedDataParallel`, con l'analogia degli insegnanti che si spartiscono
i compiti e poi mediano le correzioni. Qui la riprendiamo in due righe, ne
mettiamo a fuoco il limite, e poi andiamo *oltre*: una tassonomia dei modi di
dividere il lavoro quando un modello è troppo grande perché una GPU basti a
sé. Non è materia da tutti i giorni (quasi nessun lettore avrà un cluster
sotto mano) ma è esattamente così che vengono addestrati i modelli di
frontiera, e capirne la mappa spiega molto di come funziona l'AI moderna.

## Ripasso: il parallelismo dati e il suo limite

Il parallelismo dati non divide il modello: divide i *dati*. Ogni GPU riceve
una **copia identica** della rete e una fetta diversa del mini-batch, calcola i
propri gradienti, e alla fine tutte si mettono d'accordo mediando i gradienti
con un'operazione chiamata **all-reduce**. Dopo la media, le repliche applicano
lo stesso aggiornamento e restano perfettamente uguali. È il primo dei tre
pannelli in {numref}`fig-parallelismo-strategie`.

```{figure} ../figures/parallelismo-strategie.svg
:name: fig-parallelismo-strategie
:alt: "Tre pannelli affiancati. DATI: due GPU con lo stesso modello replicato ma fette di dati diverse (dati A e dati B), i gradienti mediati con un all-reduce. TENSOR: una singola matrice di pesi W tagliata a metà, colonna sinistra su GPU 0 e destra su GPU 1, i due risultati ricomposti con un all-gather. PIPELINE: GPU 0 tiene gli strati 1-2 e GPU 1 gli strati 3-4, i micro-batch scorrono come su una catena di montaggio."
:width: 100%

Tre modi di dividere il lavoro su più GPU. **Dati**: si replica il modello e si
spartiscono gli esempi. **Tensor**: si taglia una singola matrice di pesi tra
le GPU. **Pipeline**: si mettono strati diversi su GPU diverse.
```

Il punto delicato è come avviene quella media senza intasare la rete. Il modo
ingenuo (tutte le GPU spediscono i gradienti a una sola, che somma e
rispedisce) trasforma quella GPU in un imbuto. La soluzione elegante ha un
nome preciso.

`````{tab} Elementare

Immagina le GPU disposte in cerchio, come persone attorno a un tavolo, ognuna
con la propria lista di numeri da sommare a quella delle altre. Invece di
gridare tutti verso una persona sola (che non riuscirebbe mai a stare dietro a
tutti) ciascuno parla **solo col vicino di destra**: gli passa un pezzetto
della somma parziale, riceve un pezzetto dal vicino di sinistra, e si va
avanti finché il giro non si chiude e tutti hanno la somma completa. Il bello
è che nessuno è mai sovraccarico: il lavoro di comunicazione è spalmato in
modo uniforme, e resta lo stesso che ci siano quattro persone o quaranta.
Questa danza si chiama *ring all-reduce* ed è il motivo per cui il
parallelismo dati scala bene. Il limite è un altro, e nasce dalla parola
«copia»: se ogni persona attorno al tavolo deve tenere in tasca l'intero
elenco telefonico, quando l'elenco diventa enorme non c'è tasca che tenga.

`````

`````{tab} Superiore

Con $K$ repliche, la libreria di *collettive* di NVIDIA, **NCCL** (*NVIDIA
Collective Communications Library*), sceglie fra più algoritmi in base a
taglia del messaggio e topologia; quello classico è lo schema
**ring all-reduce**: le GPU formano un anello logico e ogni GPU comunica
soltanto con i due vicini, in due fasi (*reduce-scatter* e poi *all-gather*).
Il volume di dati che ciascuna GPU trasmette è $2\frac{K-1}{K}$ volte la
dimensione del gradiente: al crescere di $K$ tende a una costante, cioè è
**ottimale in banda** (cresce solo la latenza, non il traffico per GPU). È
proprio quella latenza, proporzionale a $K$, il motivo per cui a molti nodi
NCCL abbandona l'anello per schemi ad albero (*double binary tree*), che la
contengono senza sacrificare la banda. Come
già ricordato nella sezione «Prestazioni e scala», in
`DistributedDataParallel` questo all-reduce è eseguito *durante* il
`backward`, a pacchetti (*bucket*), così che la comunicazione si sovrapponga
al calcolo e sparisca dietro di esso.

Il limite è strutturale, non implementativo: ogni replica deve contenere
l'**intero** modello, più i suoi gradienti, più gli stati dell'ottimizzatore.
Se questo pacchetto non entra nella memoria di una singola GPU, il
parallelismo dati (per quanto ben implementato), non serve a nulla. Da qui le
strategie che seguono, che invece di replicare **spezzano**.

`````

## Spezzare la matrice: il tensor parallelism

La prima idea è tagliare il modello dove è più grosso: le sue matrici di pesi.
Invece di replicare l'intera matrice su ogni GPU, se ne mette **un pezzo** su
ciascuna, e ognuna calcola la propria fetta del prodotto. È il secondo pannello
di {numref}`fig-parallelismo-strategie`, ed è l'idea alla base di **Megatron-LM**
{cite}`shoeybi2019megatron`.

`````{tab} Elementare

Due contabili devono sommare le colonne di un registro gigantesco. Copiare
l'intero registro a entrambi sarebbe uno spreco: meglio strapparlo a metà per
il lungo (le prime colonne a uno, le ultime all'altro). Ciascuno somma la sua
metà, in parallelo, e alla fine si scambiano i due risultati parziali per
rimetterli insieme. Nessuno dei due ha mai avuto l'intero registro in mano:
sta metà in una testa e metà nell'altra. Le reti neurali sono fatte in gran
parte di queste tabelloni di numeri (le matrici dei pesi) e tagliarle così
permette di far girare uno strato che, intero, non entrerebbe in una scheda
sola. Il prezzo è che i due contabili devono parlarsi in continuazione, a ogni
strato: conviene solo se sono seduti vicini, con una linea diretta velocissima
tra loro.

`````

`````{tab} Superiore

Consideriamo il prodotto $Y = XW$, cuore di ogni strato lineare. Spezzando la
matrice dei pesi per colonne, $W = [\,W_1 \; W_2\,]$, si ottiene
$Y = [\,XW_1 \; XW_2\,]$: la GPU 0 calcola $XW_1$, la GPU 1 calcola $XW_2$, e
i due blocchi si concatenano. Megatron sfrutta questa libertà con un'eleganza
particolare nel blocco *feed-forward* $Z = \mathrm{GeLU}(XA)\,B$: spezza $A$
per **colonne** (la GeLU è elemento-per-elemento, quindi ogni GPU può
applicarla alla propria fetta senza consultare le altre) e $B$ per **righe**,
così che

$$
Z = \mathrm{GeLU}(XA_1)\,B_1 + \mathrm{GeLU}(XA_2)\,B_2,
$$

dove $A_i$ e $B_i$ sono le porzioni assegnate alla GPU $i$. La somma dei due
addendi richiede **una sola** collettiva (un all-reduce) in avanti e una
all'indietro, per blocco. Nell'attenzione multi-testa il taglio è ancora più
naturale: teste diverse su GPU diverse. Il costo è la comunicazione: le
collettive sulle *attivazioni* si ripetono a ogni blocco, sono sincrone e
stanno sul cammino critico (il calcolo non può proseguire finché non
finiscono, quindi non si nascondono dietro di esso, come invece fa
l'all-reduce dei gradienti). Per questo il
tensor parallelism vive di norma **dentro un singolo nodo**, dove le GPU sono
collegate da NVLink a centinaia di GB/s, e non tra nodi diversi.

`````

## La catena di montaggio: il pipeline parallelism

Se il tensor parallelism taglia il modello *in larghezza* (dentro ogni
strato), il **pipeline parallelism** lo taglia *in profondità*: strati diversi
su GPU diverse. GPU 0 tiene gli strati 1–8, GPU 1 i 9–16, e così via: il terzo
pannello di {numref}`fig-parallelismo-strategie`. È l'idea di **GPipe**
{cite}`huang2019gpipe`.

`````{tab} Elementare

È una catena di montaggio. La prima postazione monta il telaio, lo passa alla
seconda che aggiunge il motore, poi alla terza per la carrozzeria. C'è però un
problema evidente: se in fabbrica entra **una sola** automobile, mentre la
prima postazione lavora le altre due stanno con le mani in mano, e quando
l'auto arriva in fondo la prima è già ferma. Tre operai, ma quasi sempre uno
solo lavora. Il rimedio è non mandare un'auto sola, ma un flusso continuo:
appena la prima postazione ha finito il telaio di un'auto e l'ha passato
avanti, comincia subito quello dell'auto successiva. Presto tutte e tre le
postazioni lavorano insieme, ciascuna su un'auto diversa. In una rete neurale
le «auto» sono pezzetti del mini-batch (i **micro-batch**) che si fanno
scorrere lungo gli strati. Il tempo iniziale in cui le postazioni si
riempiono, e quello finale in cui si svuotano, è tempo sprecato: si chiama
**bolla**, e più micro-batch si mandano in fila, più diventa trascurabile.

`````

`````{tab} Superiore

Gli strati sono partizionati in $p$ **stadi**, uno per GPU, disposti in
sequenza: l'output di uno stadio è l'input del successivo. Con un solo batch,
l'utilizzo è disastroso: a ogni istante lavora un solo stadio su $p$. GPipe
spezza allora ogni mini-batch in $m$ micro-batch che entrano nella pipeline
uno dopo l'altro. La frazione di tempo sprecata nel riempimento e nello
svuotamento (la **bolla**) vale

$$
\frac{p-1}{m+p-1},
$$

dove $p$ è il numero di stadi e $m$ il numero di micro-batch: con $p=4$ stadi
e $m=1$ la bolla è i tre quarti del tempo; con $m=32$ scende sotto il 9%. Il
compromesso è che micro-batch più piccoli usano peggio ogni singola GPU (meno
lavoro per lancio) e che vanno conservate le attivazioni intermedie di tutti i
micro-batch in volo per il `backward`. Il pipeline parallelism scambia poche
attivazioni ai *confini* tra stadi (molto meno del tensor parallelism) ed è
quindi adatto anche a collegamenti tra nodi più lenti dell'NVLink.

`````

## Non replicare, spartire: ZeRO e FSDP

Torniamo al difetto del parallelismo dati: con $K$ GPU ci sono $K$ copie
identiche di tutto (pesi, gradienti, stati dell'ottimizzatore). Una montagna
di memoria sprecata a ripetere le stesse cose. E se, invece di replicare, si
**spartisse**? Ogni GPU custodisce solo una $K$-esima parte dello stato, e
quando le serve un pezzo che non ha, se lo fa passare al volo dalla collega
che lo tiene. È l'idea di **ZeRO** {cite}`rajbhandari2020zero`, che in PyTorch
prende il nome di **FSDP**: *Fully Sharded Data Parallel*
{cite}`zhao2023pytorchfsdp`.

`````{tab} Elementare

Torniamo agli insegnanti che correggono i compiti. Nel parallelismo dati
ognuno teneva in tasca una fotocopia *completa* della griglia di valutazione:
comodo, ma se la griglia è un tomo di mille pagine, tenerne una copia intera a
testa è uno spreco enorme di zaini. L'alternativa: si strappa il tomo in
$K$ fascicoli e ogni insegnante ne porta uno solo. Quando arriva la domanda la
cui regola sta a pagina 700, l'insegnante che non ce l'ha la chiede al collega
che tiene quel fascicolo, se la fa fotocopiare *giusto per quella correzione*, e
appena finito butta la fotocopia. Un po' più di viavai tra colleghi, in cambio
di zaini $K$ volte più leggeri. È così che modelli enormi riescono a girare su
GPU «normali»: nessuna scheda tiene mai il modello intero, solo la sua fetta,
radunando i pezzi che servono un attimo prima di usarli e liberandoli subito
dopo.

`````

`````{tab} Superiore

ZeRO elimina la ridondanza del parallelismo dati in tre stadi cumulativi:
partiziona tra le GPU prima gli **stati dell'ottimizzatore** (stadio 1), poi
anche i **gradienti** (stadio 2), infine anche i **parametri** (stadio 3).
FSDP è l'implementazione PyTorch dell'idea dello stadio 3: a regime, ogni GPU
detiene solo $1/K$ dei parametri di ciascuna unità del modello. Prima di
eseguire il `forward` di quell'unità, un **all-gather** ricostruisce
temporaneamente i pesi completi; subito dopo l'uso, la GPU li *ri-spartisce*
(scarta i pezzi non suoi), liberando memoria; nel `backward` la stessa cosa
avviene per i gradienti, ridistribuiti con un *reduce-scatter*. Si baratta più
comunicazione per molta meno memoria: spesso un baratto vantaggioso, perché la
comunicazione si sovrappone al calcolo. In codice, FSDP somiglia molto a DDP:
si lancia con `torchrun` e il training loop resta identico.

```python
# SCHEMA (come DDP), si lancia con: torchrun --nproc_per_node=4 addestra.py
import functools
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

dist.init_process_group("nccl")
rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)

# La auto_wrap_policy dice a FSDP quali sotto-moduli trattare come unità
# separate; senza, il modello è un'unica unità e l'all-gather ricomporrebbe
# TUTTI i pesi insieme, vanificando il risparmio di memoria. Qui l'unità è il
# singolo blocco Transformer (BloccoTransformer è la classe del tuo modello).
policy = functools.partial(transformer_auto_wrap_policy,
                           transformer_layer_cls={BloccoTransformer})

# invece di REPLICARE il modello (come DDP), FSDP ne SPARTISCE i parametri.
model = FSDP(model, device_id=rank, auto_wrap_policy=policy)

# training loop IDENTICO: FSDP raduna (all-gather) i pesi di ogni blocco
# appena prima di usarlo, e li ri-spartisce subito dopo, in automatico.
```

`````

## Il quadro d'insieme

Nella pratica queste strategie non si scelgono a esclusione: si **combinano**.
Addestrare un modello di frontiera significa quasi sempre impilare tre assi,
il cosiddetto **3D parallelism**: parallelismo dati *fra* i nodi, tensor
parallelism *dentro* ogni nodo (dove l'NVLink regge collettive frequenti e
sincrone, che stanno sul cammino critico e non si nascondono dietro il
calcolo), e pipeline parallelism a spezzare gli strati lungo i gruppi di
nodi. A questi se ne aggiungono altri due, più specialistici: il **sequence
parallelism**, che spartisce la sequenza lungo la sua lunghezza per
alleggerire la memoria delle attivazioni, e l'**expert parallelism** dei
modelli *Mixture of Experts*, che distribuisce su GPU diverse i vari «esperti»
tra cui il modello smista ogni token.

Dietro tutta questa ingegneria c'è una tensione di fondo che vale la pena
nominare: il **memory wall**. La dimensione dei modelli è cresciuta molto più
in fretta della memoria che si riesce a mettere su una singola GPU: è per
questo che *spartire* lo stato, e non solo replicarlo, è diventato
inevitabile, e che FSDP è oggi la via pratica per addestrare modelli grandi su
un numero ragionevole di schede.

Un'ultima onestà, nello stesso spirito della sezione «Prestazioni e scala»:
quasi nessun lettore di questo libro avrà un cluster su cui provare tutto
questo, e va benissimo così. Ma la tassonomia (dati, tensor, pipeline,
sharding) non è folklore da datacenter: è la mappa che spiega *come* nascono i
modelli di cui leggiamo i nomi ogni settimana. E la prima delle strategie,
FSDP, è alla portata già di due GPU: se un giorno vi troverete con qualche
scheda e un modello che non entra in una sola, saprete da che parte guardare.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Alcuni modelli non stanno in una GPU sola: per un modello da 175 miliardi di
  parametri, tenere durante l'addestramento i pesi, le correzioni e i conti di
  servizio dell'ottimizzatore richiede qualche migliaio di gigabyte, contro le
  decine di una singola scheda. Da qui il lavoro spartito fra più schede.
- Il **parallelismo dati** (già visto nella sezione «Prestazioni e scala») dà a
  ogni scheda una copia del modello e una fetta diversa degli esempi, poi le
  schede mediano le correzioni parlando **solo con il vicino**, in un giro
  attorno al tavolo, così che nessuna faccia da imbuto. Il limite sta nella
  parola «copia»: ogni scheda deve tenere in tasca l'elenco telefonico intero.
- **Tagliare le matrici per il lungo** (i paragrafi qui sopra la chiamano con il
  nome inglese, *tensor parallelism*, perché una traduzione italiana non ha mai
  preso piede) {cite}`shoeybi2019megatron`: è il registro strappato a metà, ogni
  scheda tiene mezzo tabellone di numeri, calcola la sua parte e poi si scambia i
  risultati parziali con le altre. Le schede però devono parlarsi a ogni strato e
  aspettarsi a vicenda, quindi conviene solo fra schede vicine, unite da una
  linea velocissima.
- **Mettere strati diversi su schede diverse** (in inglese *pipeline
  parallelism*) {cite}`huang2019gpipe`: è la catena di montaggio, con pezzetti
  del mini-batch (i **micro-batch**) che scorrono in fila. Il tempo in cui le
  postazioni si riempiono e si svuotano è sprecato (si chiama **bolla**) e si
  assottiglia mandando più micro-batch di seguito.
- **Spartire invece di fotocopiare** {cite}`rajbhandari2020zero`
  {cite}`zhao2023pytorchfsdp`: il tomo si strappa in fascicoli, uno per scheda, e
  il pezzo che manca si fa passare dal collega che lo tiene giusto un attimo
  prima di usarlo, per poi buttare la fotocopia. Nessuna scheda tiene mai il
  modello intero, solo la sua fetta, ed è così che oggi si addestrano i modelli
  grandi. I due nomi che si incontrano sono sigle inglesi e dicono esattamente
  questa cosa: ZeRO sta per «ottimizzatore senza copie ripetute», FSDP per
  «parallelismo dati con tutto spartito».
- Nella realtà le strategie si **combinano** (dati, tensor e pipeline insieme),
  più le varianti che spartiscono la lunghezza del testo o i vari «esperti» di
  un modello. La ragione di fondo: i modelli crescono più in fretta della
  memoria che si riesce a mettere su una scheda, e allora spartire non è più
  un'opzione.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Alcuni modelli non stanno in una GPU sola: lo **stato di addestramento** (pesi
  + gradienti + Adam) di un modello da 175 miliardi di parametri è dell'ordine
  dei terabyte, contro le decine di GB di una GPU. Da qui il parallelismo su più
  schede.
- Il **parallelismo dati** (già visto nella sezione «Prestazioni e scala»)
  replica il modello e media i gradienti con un **all-reduce** (via NCCL; lo
  schema classico è il **ring**, ottimale in banda); il suo limite è che ogni
  GPU deve contenere il modello *intero*.
- Il **tensor parallelism** {cite}`shoeybi2019megatron` taglia le singole
  matrici di pesi tra GPU (Megatron-LM), ricomponendo con una collettiva; le
  collettive sono frequenti e sul cammino critico, quindi vive dentro un nodo
  (NVLink).
- Il **pipeline parallelism** {cite}`huang2019gpipe` mette strati diversi su GPU
  diverse e fa scorrere **micro-batch** in catena di montaggio; la **bolla**
  $\frac{p-1}{m+p-1}$ si riduce aumentando i micro-batch.
- **ZeRO/FSDP** {cite}`rajbhandari2020zero` {cite}`zhao2023pytorchfsdp` non
  replicano ma **spartiscono** parametri, gradienti e stati dell'ottimizzatore,
  ricomponendoli al volo (all-gather) solo quando servono: la via pratica per i
  modelli grandi. In PyTorch: `FullyShardedDataParallel`.
- Nella realtà si **combinano** (3D parallelism: dati × tensor × pipeline),
  più sequence ed expert parallelism. Il **memory wall** (modelli che crescono
  più in fretta della memoria per GPU) è la ragione per cui lo sharding conta.
```
`````
