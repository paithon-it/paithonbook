# Oltre una GPU: parallelismo distribuito

I modelli di cui leggiamo i nomi ogni settimana non nascono su una scheda: ne
servono centinaia, a volte migliaia, tutte sullo stesso compito. Quasi nessun
lettore di questo libro avrà un impianto del genere sotto mano, e va benissimo
così. Ma il modo in cui quelle schede si spartiscono il lavoro è la cosa che
spiega meglio *come* l'intelligenza artificiale di oggi viene costruita, e si
racconta con quattro immagini: un tavolo di persone che si passano i conti, un
registro strappato a metà, una catena di montaggio e un manuale diviso in
fascicoli.

Prima però vale la pena vedere *perché* una scheda non basta, e vederlo con un
conto vero invece che a parole.

Un modello come GPT-3 ha 175 miliardi di **parametri**: sono i numeri che la
rete impara, quelli che il libro chiama anche **pesi** (le due parole si usano
l'una per l'altra, e faremo lo stesso). Scritti nel formato corto che la sezione
«Prestazioni e scala» chiama *mezza precisione*, ciascuno occupa 2 byte invece
di 4. In tutto $175 \cdot 10^9 \times 2 = 350$ miliardi di byte, cioè
**350 GB** di soli pesi.

Ma addestrare è un'altra faccenda, perché durante l'addestramento in memoria
non ci sono solo i pesi. C'è anche la contabilità di chi guida l'apprendimento,
cioè di chi a ogni passo decide di quanto spostare ciascun peso: quel «chi» si
chiama **ottimizzatore**, e il più usato, **Adam** {cite}`kingma2015adam`, non
guarda soltanto la correzione del momento, si tiene anche un po’ di memoria di
come quel peso si è mosso di recente (il capitolo sul deep learning gli dedica
una sezione). Quella memoria va conservata numero per numero, per tutta la
durata dell'addestramento. L'inventario completo, per **ogni** parametro, è
questo:

- i **pesi** in mezza precisione, 2 byte, quelli che il modello usa per
  calcolare;
- i **gradienti**, cioè le correzioni da applicare, altri 2 byte;
- una copia dei pesi in precisione piena, 4 byte. Serve perché in un numero
  corto le cifre sono poche, e sommargli una correzione minuscola non lo cambia
  affatto: come aggiungere un millimetro a una misura presa al metro, il
  risultato torna arrotondato uguale a prima e la correzione sparisce invece di
  accumularsi;
- le due **statistiche** di Adam, 4 byte l'una e quindi 8 in tutto: la media
  delle correzioni recenti (per non farsi sballottare dal singolo esempio) e una
  misura di quanto quelle correzioni ballano, che serve a fare passi corti dove
  il terreno è accidentato e passi lunghi dove è liscio.

Sommando, e contando due volte l'ultima voce perché le statistiche sono due:
$2 + 2 + 4 + 4 + 4 = 16$ byte per parametro, dei quali i 350 GB di prima sono
soltanto il primo addendo. Per 175 miliardi di parametri fanno
$175 \cdot 10^9 \times 16 = 2800$ miliardi di byte, cioè **2,8 terabyte**.
Tutto questo insieme (pesi, correzioni, copia precisa e statistiche) ha un nome
che tornerà spesso: è lo **stato dell'addestramento**, e deve stare in memoria
dal primo esempio all'ultimo. Una GPU da datacenter di fascia alta, una
H100, ha 80 GB di memoria: $2800 / 80 = 35$, ci vorrebbero trentacinque schede
soltanto per *contenerlo*, prima ancora di parlare di velocità. Alcuni modelli
non stanno in una GPU sola, né per memoria, né per tempo.

Nella sezione «Prestazioni e scala» del capitolo su PyTorch abbiamo già visto
la strategia più comune per usare più schede: il **parallelismo dati**, che in
PyTorch si accende con una riga (`DistributedDataParallel`) e che lì era
raccontato con l'analogia degli insegnanti che si spartiscono i compiti da
correggere e poi mediano le correzioni. Qui la riprendiamo in due righe, ne
mettiamo a fuoco il limite, e poi andiamo *oltre*: una mappa dei modi di
dividere il lavoro quando un modello è troppo grande perché una GPU basti a
sé.

Due parole da fissare subito, perché tornano in ogni pagina di questa sezione.
Un **nodo** è un singolo computer, con dentro le sue schede, di solito quattro
o otto, collegate fra loro da connessioni interne molto veloci. Un **cluster**
è un insieme di nodi collegati in rete, che lavorano allo stesso compito: fra
due schede dello stesso nodo i dati volano, fra due nodi diversi devono passare
per la rete, che è molto più lenta. Quasi tutte le scelte che seguono nascono
da questa differenza. (Attenzione a una parola che in questa pagina fa due
mestieri: quando si dice «rete» in questo senso si intendono i cavi che
collegano i computer, non la rete neurale. Per quest'ultima, qui, diremo sempre
«il modello».)

## Ripasso: il parallelismo dati e il suo limite

Il parallelismo dati non divide il modello: divide i *dati*. Ogni GPU riceve
una **copia identica** del modello e una fetta diversa del mini-batch (il
mazzetto di esempi che si guardano in una volta sola prima di aggiornare i
pesi), calcola le proprie correzioni, e alla fine tutte si mettono d'accordo
facendone la media. L'operazione con cui si mettono d'accordo si chiama
**all-reduce**: ogni scheda mette dentro i propri numeri, si sommano, e alla
fine tutte quante hanno in mano lo stesso risultato. Dopo la media, le repliche
applicano lo stesso aggiornamento e restano perfettamente uguali. È il primo
dei tre pannelli in {numref}`fig-parallelismo-strategie`, che vale la pena
guardare adesso solo nella sua prima colonna: le altre due sono le strategie
delle sezioni che seguono, e le si capisce meglio quando ci si arriva.

```{figure} ../figures/parallelismo-strategie.svg
:name: fig-parallelismo-strategie
:alt: "Tre pannelli affiancati. DATI: due GPU con lo stesso modello replicato ma fette di dati diverse (dati A e dati B), i gradienti mediati con un all-reduce. TENSOR: una singola matrice di pesi W tagliata a metà, colonna sinistra su GPU 0 e destra su GPU 1, i due risultati ricomposti con un all-gather. PIPELINE: GPU 0 tiene gli strati 1-2 e GPU 1 gli strati 3-4, i micro-batch scorrono come su una catena di montaggio."
:width: 100%

Tre modi di dividere il lavoro su più GPU, uno per sezione. **Dati**: si
replica il modello e si spartiscono gli esempi (è il pannello di questa
sezione). **Tensor**: si taglia in due un singolo tabellone di numeri del
modello. **Pipeline**: si mettono strati diversi su GPU diverse.
```

Il punto delicato è come avviene quella media senza intasare la rete. Il modo
ingenuo (tutte le GPU spediscono i gradienti a una sola, che somma e
rispedisce) trasforma quella GPU in un imbuto. La soluzione elegante ha un
nome preciso.

`````{tab} Elementare

Immagina le schede disposte in cerchio, come persone attorno a un tavolo,
ognuna con la propria lista di numeri da sommare a quella delle altre. Invece
di gridare tutti verso una persona sola (che non riuscirebbe mai a stare dietro
a tutti) ciascuno parla **solo col vicino di destra**: gli passa un pezzetto
della somma parziale, riceve un pezzetto dal vicino di sinistra, e così via
finché il giro non si chiude. A quel punto ciascuno ha finito **un pezzo** del
totale, non il totale intero: perché tutti abbiano tutto se ne fa un secondo
giro, uguale al primo, in cui i pezzi già finiti si passano di mano in mano.
Due giri, quindi, non uno: chi vuole tenere il conto di quanti dati si
spediscono deve ricordarsi di contarli due volte.

Il bello è che nessuno è mai sovraccarico, e che aggiungere gente non peggiora
le cose. Il motivo si vede con un conto in testa: se le persone raddoppiano, la
lista viene tagliata in pezzetti che sono la metà, e ciascuno ne passa il
doppio; il numero di passaggi cresce, la quantità di roba che ciascuno spedisce
resta quella. Con quattro persone o con quaranta, ogni scheda manda in giro
all'incirca due volte la propria lista, e non di più.

Questa danza si chiama *ring all-reduce* ed è il motivo per cui il
parallelismo dati regge bene su tante schede. Il limite è un altro, e nasce
dalla parola «copia»: nell'analogia, oltre ai numeri da sommare, ciascuno ha
davanti a sé anche una copia intera del manuale con cui lavora, cioè il
modello. Finché è un opuscolo va bene. Quando diventa un elenco telefonico da
mille pagine, non c'è tasca che tenga.

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
contengono senza sacrificare la banda.

Vale la pena dire da che cosa l'anello ha preso il posto, perché il confronto
spiega la sua fortuna. Lo schema precedente era il **parameter server**
{cite}`li2014parameterserver`: uno o
più nodi dedicati custodiscono i pesi, tutti gli altri ci mandano i gradienti e
ne rileggono i pesi aggiornati. È semplice, sopporta bene i lavoratori lenti
(in versione asincrona non si aspetta nessuno) ed è tuttora sensato quando i
nodi sono eterogenei o inaffidabili. Ma il traffico che attraversa il server
cresce **linearmente con $K$**, e con esso il tempo, perché quel nodo è un
collo di bottiglia di banda che non si può allargare aggiungendo macchine.
L'anello non ha un centro: nessun nodo vede più traffico degli altri, e il
costo per GPU smette di dipendere da quante sono. È la stessa ragione per cui,
nei sistemi distribuiti, si preferisce un protocollo fra pari a uno che passa
da un coordinatore, ogni volta che il coordinatore non serve. Come
già ricordato nella sezione «Prestazioni e scala», in
`DistributedDataParallel` questo all-reduce è eseguito *durante* il
`backward`, a pacchetti (*bucket*), così che la comunicazione si sovrapponga
al calcolo e sparisca dietro di esso.

Il limite è strutturale, non implementativo: ogni replica deve contenere
l’**intero** modello, più i suoi gradienti, più gli stati dell'ottimizzatore.
Se questo pacchetto non entra nella memoria di una singola GPU, il
parallelismo dati (per quanto ben implementato), non serve a nulla. Da qui le
strategie che seguono, che invece di replicare **spezzano**.

`````

## Spezzare la matrice: il tensor parallelism

La prima idea è tagliare il modello dove è più grosso. Un modello, dentro, è
fatto in buona parte di grandi tabelloni di numeri: uno per strato, e farlo
lavorare vuol dire moltiplicare i numeri che entrano per i numeri del
tabellone. Un tabellone così, in matematica, si chiama **matrice**, ed è
l'oggetto di cui parla il capitolo sulla matematica. Invece di tenerne una
copia intera su ogni GPU, se ne mette **un pezzo** su ciascuna, e ognuna
calcola la propria fetta del risultato. È il secondo pannello di
{numref}`fig-parallelismo-strategie`, ed è l'idea alla base di **Megatron-LM**
{cite}`shoeybi2019megatron`.

`````{tab} Elementare

Due contabili devono sommare le colonne di un registro gigantesco. Copiare
l'intero registro a entrambi sarebbe uno spreco: meglio strapparlo a metà per
il lungo, le prime colonne a uno e le ultime all'altro. Ciascuno somma la sua
metà, in parallelo, e poi i due si scambiano i risultati parziali per rimetterli
insieme. Nessuno dei due ha mai avuto l'intero registro in mano: sta metà in
una testa e metà nell'altra. È così che si fa girare uno strato che, intero,
non entrerebbe in una scheda sola.

Il punto delicato è che quello scambio non avviene una volta sola alla fine. Il
totale di ogni pagina serve per cominciare la pagina dopo, quindi i due
contabili devono fermarsi, mettere insieme i pezzi e ripartire **a ogni
pagina**: e un modello di pagine ne ha decine, una per strato. Finché i due
sono seduti allo stesso tavolo, passarsi un foglio non costa quasi niente. Se
sono in due edifici diversi, quel fermarsi continuo diventa la voce di spesa
principale. Per questo tagliare i tabelloni conviene solo fra schede vicine,
unite da una linea diretta velocissima.

`````

`````{tab} Superiore

Consideriamo il prodotto $\mathbf{Y} = \mathbf{X}\mathbf{W}$, cuore di ogni
strato lineare. Spezzando la matrice dei pesi per colonne,
$\mathbf{W} = [\,\mathbf{W}_1 \; \mathbf{W}_2\,]$, si ottiene
$\mathbf{Y} = [\,\mathbf{X}\mathbf{W}_1 \; \mathbf{X}\mathbf{W}_2\,]$: la GPU 0
calcola $\mathbf{X}\mathbf{W}_1$, la GPU 1 calcola $\mathbf{X}\mathbf{W}_2$, e
i due blocchi si concatenano. Megatron sfrutta questa libertà con un'eleganza
particolare nel blocco *feed-forward*
$\mathbf{Z} = \mathrm{GeLU}(\mathbf{X}\mathbf{A})\,\mathbf{B}$: spezza
$\mathbf{A}$ per **colonne** (la GeLU è elemento-per-elemento, quindi ogni GPU
può applicarla alla propria fetta senza consultare le altre) e $\mathbf{B}$ per
**righe**, così che

$$
\mathbf{Z} = \mathrm{GeLU}(\mathbf{X}\mathbf{A}_1)\,\mathbf{B}_1
           + \mathrm{GeLU}(\mathbf{X}\mathbf{A}_2)\,\mathbf{B}_2,
$$

dove $\mathbf{A}_i$ e $\mathbf{B}_i$ sono le porzioni assegnate alla GPU $i$.
La somma dei due
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
postazioni lavorano insieme, ciascuna su un'auto diversa. Nel modello le
«auto» sono pezzetti del mazzetto di esempi, i **micro-batch**, che si fanno
scorrere lungo gli strati.

Il tempo iniziale in cui le postazioni si riempiono, e quello finale in cui si
svuotano, è tempo sprecato. Ha un nome che viene dal disegno con cui si
rappresentano queste cose (una riga per postazione, il tempo che scorre in
orizzontale, il lavoro come una barra piena): all'inizio e alla fine restano
due sacche vuote, e quelle sacche si chiamano **bolla**. Quanto costa la bolla
si conta: con quattro postazioni e una macchina sola se ne va in fumo il 75%
del tempo, con trentadue macchine in fila si scende sotto il 9%. Più
micro-batch si mandano di seguito, più la bolla si assottiglia.

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
e $m=1$ la bolla è i tre quarti del tempo; con $m=32$ scende sotto il 9%.
L'articolo dà anche la regola pratica per leggerla: la bolla è già trascurabile
con $m \ge 4p$, cioè con almeno quattro micro-batch per stadio.

Il compromesso è che micro-batch più piccoli usano peggio ogni singola GPU
(meno lavoro per lancio) e che vanno conservate, per ogni micro-batch in volo,
le attivazioni **ai confini fra stadi**. Le attivazioni *interne* a uno stadio,
invece, GPipe non le conserva affatto: le **ricalcola** nel `backward`, ed è il
secondo contributo dell'articolo accanto ai micro-batch, quello che abbassa il
picco di memoria da «tutte le attivazioni di tutti gli strati» a «quelle di uno
stadio solo, per un micro-batch solo». È lo stesso baratto calcolo-per-memoria
del *gradient checkpointing* e della sezione precedente su FlashAttention: la
stessa mossa, sotto tre nomi diversi. Vale infine la pena sapere che i sistemi
di oggi non usano più lo scheduling di GPipe ma quello **1F1B** (un `forward` e
un `backward` alternati, da PipeDream e Megatron), che a parità di bolla tiene
in volo $p$ micro-batch invece di $m$, e quindi ne conserva anche meno.

Quel che attraversa la rete, comunque, sono solo le attivazioni ai confini fra
stadi: molto meno di quanto scambi il tensor parallelism, ed è per questo che il
pipeline parallelism regge anche su collegamenti tra nodi più lenti
dell'NVLink.

`````

## Non replicare, spartire: ZeRO e FSDP

Torniamo al difetto del parallelismo dati: con otto GPU ci sono otto copie
identiche di tutto, cioè otto volte lo stato dell'addestramento del conto di
apertura (pesi, correzioni e le due statistiche che l'ottimizzatore si tiene per
ogni peso). Una montagna di memoria sprecata a ripetere le stesse cose. E se,
invece di replicare, si **spartisse**? Ogni GPU custodisce solo la propria
fetta, e quando le serve un pezzo che non ha, se lo fa passare al volo dalla
collega che lo tiene.

I due nomi con cui questa idea si incontra sono sigle inglesi, e conviene
scioglierle subito perché dicono esattamente quello che fanno: **ZeRO**
{cite}`rajbhandari2020zero` sta per «ottimizzatore senza copie ripetute»
(*Zero Redundancy Optimizer*), e la sua realizzazione in PyTorch si chiama
**FSDP**, «parallelismo dati con tutto spartito» (*Fully Sharded Data
Parallel*) {cite}`zhao2023pytorchfsdp`. Il verbo inglese per «spartire», qui,
è *shard*, e da lì viene lo **sharding** che si incontra ovunque si parli di
questa famiglia di tecniche.

`````{tab} Elementare

Torniamo agli insegnanti che correggono i compiti. Nel parallelismo dati
ognuno teneva in tasca una fotocopia *completa* della griglia di valutazione:
comodo, ma se la griglia è un tomo di mille pagine, tenerne una copia intera a
testa è uno spreco enorme di zaini. L'alternativa: se gli insegnanti sono otto,
si strappa il tomo in otto fascicoli e ognuno ne porta uno solo. Quando arriva
la domanda la cui regola sta a pagina 700, l'insegnante che non ce l'ha la
chiede al collega che tiene quel fascicolo, se la fa fotocopiare *giusto per
quella correzione*, e appena finito butta la fotocopia. Un po’ più di viavai tra
colleghi, in cambio di zaini otto volte più leggeri. È così che modelli enormi
riescono a girare su GPU «normali»: nessuna scheda tiene mai il modello intero,
solo la sua fetta, radunando i pezzi che servono un attimo prima di usarli e
liberandoli subito dopo.

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
avviene per i gradienti, ridistribuiti con un *reduce-scatter*.

Sul prezzo in comunicazione va detta la cosa che di solito si dà per scontata
al contrario: i **primi due stadi sono gratis**. Spartire gli stati
dell'ottimizzatore e i gradienti «incurs no additional communication» rispetto
al parallelismo dati puro, scrive l'articolo, a fronte di un risparmio di
memoria fino a otto volte. Non c'è quindi un motivo
per non accenderli. A pagare è solo il terzo stadio, quello di FSDP, e paga una
cifra precisa: $1{,}5\times$ la comunicazione del parallelismo dati, perché
all'all-reduce dei gradienti si aggiungono gli all-gather dei parametri in
avanti e all'indietro. Anche così è quasi sempre un buon affare, perché quella
comunicazione si sovrappone al calcolo. In codice, FSDP somiglia molto a DDP:
si lancia con `torchrun` e il training loop resta identico.

```{code-block} python
:class: pt-non-eseguibile

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
Addestrare un modello di frontiera significa quasi sempre impilarne tre
insieme, e siccome sono tre tagli in tre direzioni diverse (fra gli esempi,
dentro un tabellone, lungo la fila degli strati) l'uso ha chiamato la
combinazione **3D parallelism**, come i tre assi di uno spazio.

A decidere quale strategia va dove c'è una domanda sola: **quanto spesso le
schede devono fermarsi a parlarsi**. Prima però va detta la cosa che rende
quella domanda sensata, perché è la stessa mossa dell'occupancy vista
all'inizio del capitolo: mandare dati e fare conti sono due attività distinte e
possono avvenire *insieme*, quindi una comunicazione che parte mentre la scheda
sta ancora calcolando non si paga quasi. Si paga invece quella che obbliga
tutti a incrociare le braccia finché non è finita.

Da qui la distribuzione. Il tensor parallelism sta **dentro** ogni nodo: a ogni
strato obbliga le schede a mettere insieme un risultato e ad aspettarsi a
vicenda, e quell'attesa è di quelle che nessuno può coprire con altro lavoro,
perché il calcolo dopo dipende proprio da lei. Serve quindi il collegamento più
veloce che esista, quello interno al nodo (si chiama **NVLink**). Il pipeline
parallelism spezza gli strati fra gruppi di nodi, e si scambia poca roba. Il
parallelismo dati sta **fra** i nodi, dove c'è la rete lenta: gli basta un giro
di medie a ogni passo, cioè a ogni mazzetto di esempi.

A questi se ne aggiungono altri due, più specialistici, e sono due modi di
tagliare che i primi tre non coprono. Il **sequence parallelism**
{cite}`korthikanti2023activation` divide il testo in tratti e ne dà uno per
scheda (le prime mille parole a una, le seconde mille a un'altra), per
alleggerire la memoria che si mangiano le **attivazioni**, cioè i risultati
intermedi che ogni strato produce e che vanno conservati fino al passaggio
all'indietro, quello in cui il modello impara dai propri errori.
L’**expert parallelism** riguarda i modelli *Mixture of Experts*, quelli in cui
il modello non è uno solo ma un mazzo di modelli specializzati fra cui un
selettore smista ogni parola in arrivo: lì si mettono esperti diversi su schede
diverse.

Dietro tutta questa ingegneria c'è una tensione di fondo che vale la pena
nominare: il **memory wall**, il muro della memoria. La dimensione dei modelli
è cresciuta molto più in fretta della memoria che si riesce a mettere su una
singola GPU: è per questo che *spartire* lo stato, e non solo replicarlo, è
diventato inevitabile, e che FSDP è oggi la via pratica per addestrare modelli
grandi su un numero ragionevole di schede.

Un'ultima onestà, nello stesso spirito della sezione «Prestazioni e scala»:
quasi nessun lettore di questo libro avrà un cluster su cui provare tutto
questo, e va benissimo così. Ma le quattro strategie (spartire gli esempi,
spartire i tabelloni, spartire gli strati, spartire lo stato) non sono folklore
da datacenter: sono la mappa che spiega *come* nascono i modelli di cui
leggiamo i nomi ogni settimana. E l'ultima, FSDP, è alla portata già di **due
schede infilate nello stesso computer**. Perché proprio lei: il parallelismo
dati su due schede è facilissimo da accendere ma non risolve il problema del
modello che non ci sta, visto che ogni scheda ne tiene una copia intera;
spezzare i tabelloni o gli strati risolve, ma vuole che si rimetta mano a come
il modello è scritto. FSDP risolve *e* costa una riga di codice al posto di
un'altra. Se un giorno vi troverete con un modello che in una scheda sola non
entra, è da lì che si comincia.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Alcuni modelli non stanno in una GPU sola: per un modello da 175 miliardi di
  parametri, tenere durante l'addestramento i pesi, le correzioni e i conti di
  servizio dell'ottimizzatore richiede qualche migliaio di gigabyte, contro le
  decine di una singola scheda. Da qui il lavoro spartito fra più schede.
- Il **parallelismo dati** (già visto nella sezione «Prestazioni e scala») dà a
  ogni scheda una copia del modello e una fetta diversa degli esempi, poi le
  schede mediano le correzioni parlando **solo con il vicino**, in due giri
  attorno al tavolo (uno per completare i pezzi, uno per distribuirli), così che
  nessuna faccia da imbuto. Il limite sta nella parola «copia»: ogni scheda deve
  tenere in tasca l'elenco telefonico intero.
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
  $\frac{p-1}{m+p-1}$ è già trascurabile con $m \ge 4p$. Conserva le attivazioni
  ai *confini* fra stadi, e quelle interne le **ricalcola**: stesso baratto
  calcolo-per-memoria di FlashAttention e del *gradient checkpointing*.
- **ZeRO/FSDP** {cite}`rajbhandari2020zero` {cite}`zhao2023pytorchfsdp` non
  replicano ma **spartiscono** parametri, gradienti e stati dell'ottimizzatore,
  ricomponendoli al volo (all-gather) solo quando servono: la via pratica per i
  modelli grandi. I primi due stadi non costano **nulla** in comunicazione (e
  vanno quindi accesi sempre); solo il terzo, quello di FSDP, paga $1{,}5\times$.
  In PyTorch: `FullyShardedDataParallel`.
- Nella realtà si **combinano** (3D parallelism: dati × tensor × pipeline),
  più sequence ed expert parallelism. Il **memory wall** (modelli che crescono
  più in fretta della memoria per GPU) è la ragione per cui lo sharding conta.
```
`````
