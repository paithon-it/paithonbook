# Prestazioni e scala: spremere l'hardware

Nell'autunno del 2012 la gara mondiale di riconoscimento di immagini,
ImageNet, fu vinta con un distacco mai visto da una rete neurale, **AlexNet**:
60 milioni di parametri addestrati su 1,2 milioni di fotografie
{cite}`krizhevsky2012imagenet`. Il dettaglio che colpisce, riletto oggi, è
l'attrezzatura: non un supercomputer, ma due schede grafiche da videogiocatori
(GeForce GTX 580, circa 500 dollari l'una) montate in un normale PC, che
macinarono il dataset per cinque-sei giorni. E già nell'introduzione gli
autori scrissero, con disarmante franchezza, che i risultati sarebbero
migliorati "semplicemente aspettando GPU più veloci e dataset più grandi".
Avevano ragione: da allora il deep learning e l'hardware sono co-evoluti,
ognuno trainando l'altro; reti più grandi giustificano chip più potenti, chip
più potenti rendono pensabili reti più grandi.

Il capitolo finora ha costruito il *cosa*: tensori, moduli, training loop.
Questa sezione guarda al *quanto in fretta*: perché la GPU è lo strumento
giusto, come dimezzare i byte per (quasi) raddoppiare la velocità, cosa fa
`torch.compile`, come si addestra su più schede, e, per onestà, cosa conta
davvero quando di scheda ce n'è una sola, o nessuna. Il capitolo successivo,
«GPU e calcolo parallelo», apre poi il cofano dell'hardware: com'è fatta una
GPU, cos'è davvero un *kernel*, da dove nasce la velocità di una
moltiplicazione tra matrici, e come si divide un modello che in una scheda
sola non ci sta.

## Perché la GPU: tanti operai semplici

Nella sezione sui tensori abbiamo visto il gesto: `.to(device)`, e solo
accennato al perché. Eccolo per esteso. Il punto di partenza è che una rete
neurale, vista dall'hardware, è quasi soltanto una cosa: **moltiplicazioni tra
matrici**, milioni di prodotti e somme tutti uguali e tutti indipendenti tra
loro.

`````{tab} Elementare
Immagina due squadre a cui affidare un lavoro. La prima è la CPU: otto operai
straordinariamente qualificati, capaci di qualunque compito complicato
(decisioni, eccezioni, lavori sempre diversi). La seconda è la GPU: decine di
migliaia di manovali che sanno fare solo operazioni elementari, ma tutti
insieme, nello stesso istante. Se il lavoro è "prendi questi due numeri,
moltiplicali, somma il risultato" ripetuto milioni di volte, la squadra dei
manovali stravince: non serve intelligenza, serve manodopera. E le reti
neurali sono esattamente quel lavoro. Un esempio con i numeri: uno strato che
collega 1000 neuroni ad altri 1000, su un vassoio di 64 esempi, richiede
$64 \times 1000 \times 1000$, cioè 64 milioni di moltiplicazioni, per un
*singolo strato*, a ogni passo. Una CPU le smaltisce in fila; una GPU moderna
ne esegue decine di migliaia di miliardi al secondo, perché era nata per fare
la stessa cosa con i pixel dei videogiochi: tanti piccoli conti identici,
tutti in parallelo.
`````

`````{tab} Superiore
Il prodotto tra una matrice $(n, m)$ e una $(m, p)$ costa circa $2nmp$
operazioni in virgola mobile, tutte indipendenti a livello di prodotto
scalare: parallelismo perfetto. Le GPU adottano un'architettura *throughput
oriented* (migliaia di unità di calcolo semplici, modello SIMT: stessa
istruzione su dati diversi), mentre le CPU sono *latency oriented* (pochi core
complessi, ottimizzati per il singolo flusso di istruzioni). Dal 2017 le GPU
NVIDIA aggiungono i **tensor core**, unità dedicate proprio al prodotto tra
piccole matrici. Il collo di bottiglia, più spesso del calcolo, è il movimento
dei dati: la banda di memoria interna della GPU e, peggio ancora, il bus PCIe
che separa CPU e GPU; è il motivo per cui `.to(device)` va fatto una volta per
batch, non tensore per tensore, e per cui vedremo che tenere la GPU
*rifornita* conta quanto la GPU stessa.
`````

## Metà dei byte, quasi doppia velocità: la precisione mista

Ogni numero di un tensore `float32` occupa 4 byte. Ma servono davvero tutti?
L'idea della **precisione mista** {cite}`micikevicius2018mixed` è usare numeri
a 16 bit (metà spazio, metà traffico in memoria) nei punti dove la precisione
piena non serve, conservando il `float32` dove invece è indispensabile. C'è
anche un secondo guadagno, e riguarda i **tensor core**: dal 2017 le schede
NVIDIA hanno, accanto alle unità di calcolo normali, dei circuiti costruiti
apposta per moltiplicare piccole matrici di numeri corti, e quelli entrano in
funzione solo se i numeri sono corti davvero. Non sono un lusso da datacenter:
li hanno anche le schede da videogiocatori.

`````{tab} Elementare
Per pesare le patate non serve il bilancino del farmacista: la bilancia da
cucina basta, ed è più sbrigativa. Un numero in precisione piena porta con sé
circa 7 cifre significative; uno in mezza precisione circa 3. Per la maggior
parte dei conti di una rete (attivazioni, moltiplicazioni), tre cifre bastano,
e scrivere numeri lunghi la metà significa spostare metà dei byte: quasi il
doppio della velocità, gratis. C'è però un'insidia: i numeri piccolissimi.
Certi gradienti sono così minuscoli che, arrotondati a 16 bit, diventano zero
spaccato, e un gradiente a zero è una lezione persa: quel peso non impara più.
Il rimedio è una lente d'ingrandimento: prima di calcolare i gradienti si
moltiplica l'errore per un fattore grande (diciamo per $65\,536$), così anche
i gradienti minuscoli restano visibili; subito prima di aggiornare i pesi, si
divide per lo stesso fattore e tutto torna alla scala giusta. In PyTorch la
lente si chiama `GradScaler`, e si regola da sola.
`````

`````{tab} Superiore
`float32` ha 1 bit di segno, 8 di esponente, 23 di mantissa; `float16`
rispettivamente 1, 5, 10, quindi non solo meno precisione, ma anche un
intervallo dinamico molto più stretto: i gradienti sotto la soglia dei
denormali vanno in *underflow* a zero. Il **loss scaling** di
{cite}`micikevicius2018mixed` moltiplica $\mathcal{L}$ per un fattore $s$
prima del backward, i gradienti scalano linearmente,
$\nabla(s\mathcal{L}) = s\nabla\mathcal{L}$, e divide per $s$ prima dello
`step`; `GradScaler` adatta $s$ dinamicamente e salta l'aggiornamento se trova
`inf`/`NaN`. I pesi del modello restano in `float32` (`autocast` li converte
al volo solo dentro le singole operazioni), perché aggiornamenti piccoli su
pesi a 16 bit si perderebbero per arrotondamento (è l'idea della copia
*master* del paper). L'alternativa moderna è **bfloat16** (1, 8, 7): stesso
esponente del `float32`, quindi stesso intervallo dinamico e niente scaler, al
prezzo di una mantissa più corta; è il formato preferito sulle GPU NVIDIA da
Ampere in poi e sulle TPU, dov'è nato.
`````

Nel training loop della sezione precedente, la precisione mista sono cinque
righe in più: il `GradScaler` all'inizio, il blocco `autocast` attorno alle due
righe del forward, e tre righe al posto delle solite `backward()` e `step()`.
Il blocco qui sotto è scritto per girare **davvero**, anche senza GPU, quindi
si costruisce i suoi pezzi e stampa qualcosa a ogni giro: il ciclo che non
stampa niente è un ciclo che non è mai entrato, ed è un guasto che altrimenti
non si vede.

```python
import torch
from torch import nn

dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
# su GPU si usa float16 con la "lente"; su CPU l'autocast lavora in bfloat16,
# che non ne ha bisogno, e lo scaler si disattiva da sé
mezza = torch.float16 if dispositivo == "cuda" else torch.bfloat16

model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(),
                      nn.Linear(128, 10)).to(dispositivo)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_loader = [(torch.randn(16, 1, 28, 28), torch.randint(0, 10, (16,)))
                for _ in range(3)]               # tre batch finti, per far girare il ciclo

scaler = torch.amp.GradScaler(dispositivo)       # la "lente" per i gradienti

for i, (X, y) in enumerate(train_loader):
    X, y = X.to(dispositivo), y.to(dispositivo)
    optimizer.zero_grad()
    with torch.autocast(dispositivo, dtype=mezza):
        y_pred = model(X)                        # forward in mezza precisione
        loss = criterion(y_pred, y)
    scaler.scale(loss).backward()                # loss amplificata, poi backward
    scaler.step(optimizer)                       # gradienti riportati in scala
    scaler.update()                              # ricalibra il fattore di scala
    print(f"batch {i}: loss {loss.item():.4f} | tipo interno {y_pred.dtype}")
```

`autocast` sceglie da solo, operazione per operazione, dove i 16 bit sono
sicuri (le moltiplicazioni tra matrici) e dove no (somme lunghe, softmax): la
riga stampata mostra che l'uscita del modello è davvero in mezza precisione
anche se i pesi restano interi a 32 bit.

C'è poi un formato alternativo, il **bfloat16**, che sulle GPU recenti si usa
al posto del `float16` scrivendo `dtype=torch.bfloat16`, e che permette di
togliere del tutto il `GradScaler`. La ragione è che il problema dei gradienti
minuscoli, quello per cui serviva la lente d'ingrandimento, con il bfloat16 non
si pone: dei sedici bit ne spende di più per dire *quanto è grande* il numero e
meno per dire *con quante cifre*, quindi arriva agli stessi estremi del
`float32` (dal miliardesimo di miliardesimo in su) e i gradienti piccoli non
diventano zero. Si perdono cifre decimali, che qui non servono, e si guadagna
la semplicità di due righe in meno.

## Misurare davvero: la coda asincrona

Prima di ottimizzare qualcosa bisogna saperlo misurare, e qui c'è una trappola
in cui cade praticamente chiunque la prima volta.

`````{tab} Elementare

Quando scrivi un'operazione su GPU, Python **non aspetta che venga eseguita**.
La mette in coda e prosegue subito con la riga successiva. È il motivo per cui
la GPU riesce a stare occupata: mentre lavora su un'operazione, il programma le
sta già preparando le prossime.

La conseguenza è che un cronometro attorno a un pezzo di codice misura il tempo
di **accodare** le operazioni, non quello di eseguirle. È così che nascono i
confronti assurdi, del tipo «PyTorch è mille volte più veloce di NumPy»: non è
veloce, è che non ha ancora fatto niente.

Per misurare sul serio bisogna dire esplicitamente «fermati qui finché la GPU
non ha finito». Vale anche al contrario, quando si legge un risultato: se una
riga sembra lentissima, spesso non è lei a essere lenta, è la prima che ha
avuto bisogno del risultato e ha dovuto aspettare tutta la coda accumulata
prima.

`````

`````{tab} Superiore

Le chiamate CUDA sono **asincrone rispetto all'host**: vengono inserite in uno
*stream* e ritornano immediatamente. La sincronizzazione avviene solo in punti
precisi, e conviene conoscerli perché sono anche i punti dove il codice
rallenta senza motivo apparente: un `.item()`, un `.cpu()`, una `print` del
tensore, un `if` che dipende da un valore calcolato sulla GPU. Ognuno di questi
è una barriera implicita, ed è il motivo per cui loggare la loss a ogni passo
può costare parecchio.

Per cronometrare correttamente serve `torch.cuda.synchronize()` **prima** di
far partire il cronometro (per svuotare la coda pregressa) e **dopo** il blocco
da misurare. In alternativa si usano i `torch.cuda.Event`, che si registrano
nello stream e misurano sul lato GPU senza bloccare l'host, ed è ciò che fanno
i profiler seri.

Da qui anche un pattern utile: `tensore.to(device, non_blocking=True)`
sovrappone il trasferimento al calcolo, ma **solo** se la memoria sorgente è
*pinned* (bloccata in pagine non swappabili), che è ciò che fa
`pin_memory=True` nel `DataLoader`. Senza quella condizione l'opzione non ha
effetto, ed è una delle micro-ottimizzazioni più spesso copiate senza le sue
premesse.

`````

```python
import time
import torch

dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
A = torch.randn(1024, 1024, device=dispositivo)

def cronometra(sincronizza):
    """Il parametro decide se aspettare la GPU alla fine: True sì, False no."""
    if dispositivo == "cuda":
        torch.cuda.synchronize()          # parti da una coda vuota
    t0 = time.perf_counter()
    for _ in range(10):
        B = A @ A
    if sincronizza and dispositivo == "cuda":
        torch.cuda.synchronize()          # aspetta che la GPU abbia finito DAVVERO
    return time.perf_counter() - t0

for _ in range(3):                        # riscaldamento, fuori dal cronometro
    A @ A

print(f"dispositivo: {dispositivo}")
senza = min(cronometra(False) for _ in range(3))   # il minimo, non la prima misura
con   = min(cronometra(True)  for _ in range(3))
print(f"senza synchronize: {senza * 1000:8.2f} ms")
print(f"con synchronize  : {con   * 1000:8.2f} ms")
if dispositivo == "cpu":
    print(f"(su CPU non c'è coda asincrona: i due numeri sono dello stesso "
          f"ordine, qui a {abs(con - senza) / senza:.0%} di distanza)")
```

Due precauzioni nel codice meritano una riga, perché sono il modo giusto di
cronometrare qualunque cosa e non solo questo. La prima è il **riscaldamento**:
le prime chiamate pagano costi che le successive non pagano (l'avvio dei thread
di calcolo, la memoria che si scalda), quindi si fanno girare a vuoto e non si
misurano. La seconda è prendere il **minimo** di più ripetizioni invece della
prima misura: il minimo è il giro in cui il computer è stato meno disturbato da
altro, ed è la statistica meno rumorosa che si possa usare su una macchina
condivisa.

Su CPU, con queste due precauzioni, i due numeri si equivalgono a meno del
rumore di misura (qualche punto percentuale, in un senso o nell'altro), perché
non c'è nessuna coda: è il controllo negativo, e serve a mostrare che la
differenza non è un artefatto del modo di misurare. Su una GPU, invece, la
prima riga stampa un tempo assurdamente piccolo e la seconda quello vero. Se si
esegue questo capitolo su Colab con l'acceleratore attivo, è il primo
esperimento da rifare.

## Una riga per compilare: `torch.compile`

Il paradigma define-by-run visto nell'apertura del capitolo ha un costo:
eseguire il modello un'operazione alla volta significa che ogni operazione
paga il viaggio verso la memoria della GPU. Da PyTorch 2.0 (2023) esiste il
rimedio, e sta in una riga:

```python
model = torch.compile(model)   # tutto qui: il resto del codice non cambia
```

`````{tab} Elementare
È la differenza tra un cuoco che legge la ricetta una riga alla volta, apre il
frigo, prende il burro, chiude il frigo; apre il frigo, prende le uova..., e
uno che la legge tutta in anticipo e si organizza: un solo viaggio al frigo
con tutto l'occorrente. `torch.compile` legge il tuo modello per intero, si
accorge che tre operazioni consecutive possono diventare una sola, e riscrive
i passaggi in una versione ottimizzata. Il patto è chiaro: la prima esecuzione
è *più lenta*, perché studiare la ricetta costa; le successive sono più
veloci. Conviene quindi quando lo stesso piatto va cucinato migliaia di volte
(un addestramento lungo su un modello grande) e non conviene per un assaggio:
su un esperimento di due minuti il tempo di compilazione mangia tutto il
guadagno.
`````

`````{tab} Superiore
Sotto la riga lavorano due componenti: **TorchDynamo** intercetta il bytecode
Python e ne estrae un grafo di operazioni; **TorchInductor** genera kernel
fusi (su GPU, in Triton). La **kernel fusion** è il guadagno principale: tre
operazioni elemento-per-elemento in sequenza diventano un kernel unico che
legge e scrive la memoria una volta invece di tre; decisivo perché molte reti
sono *memory bound*, limitate dalla banda più che dal calcolo. Il grafo è
protetto da *guard*: se cambiano forme dei tensori o rami del control flow, si
ricompila (altro overhead). Nei benchmark ufficiali su GPU A100 il guadagno
medio in addestramento è attorno al 40%, ma la varianza è alta: modelli grandi
e statici guadagnano di più, modelli piccoli o dalle forme variabili poco,
nulla, o **meno di zero**: il pavimento non è la parità. Misurato sulla MLP di
MNIST su CPU: decine di secondi solo per compilare (da 23 a 75 in due prove con
carichi diversi, contro un addestramento che dura minuti) e, a regime, un
compilato più lento dell'eager, di una frazione o di parecchie volte a seconda
di quanto la macchina è carica. Su CPU l'inductor gioca in trasferta e il
fattore non è trasferibile a una GPU, ma il segno sì. La regola pratica:
attivalo quando l'addestramento dura ore, misura, e tienilo solo se il
cronometro dà ragione.
`````

## Più GPU, un solo modello: il parallelismo dati

Quando una GPU non basta, la strategia più comune non divide il *modello*:
divide i *dati*. Ogni GPU riceve una copia identica della rete e una fetta
del mini-batch; alla fine, le copie si rimettono d'accordo
({numref}`fig-parallelismo-dati`).

```{figure} ../figures/parallelismo-dati.svg
:name: fig-parallelismo-dati
:alt: Un mini-batch si divide in tre parti che vanno a tre GPU, ognuna con una replica identica del modello; i tre gradienti locali confluiscono in un nodo di all-reduce che ne calcola la media e la restituisce a tutte le GPU, che applicano lo stesso aggiornamento dei pesi.
:width: 90%

Parallelismo dati: ogni GPU calcola i gradienti sulla propria fetta di batch;
l'all-reduce ne fa la media e la restituisce a tutte, che restano così copie
identiche.
```

`````{tab} Elementare
Trecento verifiche da correggere, tre insegnanti, una sola griglia di
valutazione. Ognuno prende cento compiti e una *fotocopia* della griglia, e
correggendo annota le modifiche che farebbe: "questa domanda va pesata di
più", "qui l'errore è meno grave". A fine pila i tre si riuniscono, fanno la
**media** delle proposte e la applicano tutti e tre, identica, alla propria
fotocopia. Risultato: hanno corretto in un terzo del tempo, e le tre griglie
sono ancora perfettamente uguali, come se avesse corretto una persona sola, ma
tre volte più in fretta. Le GPU fanno lo stesso: ognuna elabora la sua fetta
di esempi, poi tutte mettono in comune i gradienti, ne fanno la media e si
aggiornano allo stesso modo. La riunione ha un nome tecnico, *all-reduce*, e
un costo: se gli insegnanti passano più tempo a riunirsi che a correggere, il
gioco non vale la candela.
`````

`````{tab} Superiore
Con $K$ repliche e il batch spartito in fette, ogni replica $k$ calcola
$\nabla_\theta \mathcal{L}_k$ sulla propria fetta; l'**all-reduce** calcola

$$
\nabla_\theta \mathcal{L} = \frac{1}{K} \sum_{k=1}^{K} \nabla_\theta \mathcal{L}_k,
$$

dove $\mathcal{L}_k$ è la loss media sulla fetta della replica $k$. Se le
fette hanno la stessa dimensione, questa media è esattamente il gradiente
sull'intero batch: matematicamente non cambia nulla, cambia solo chi fa i
conti. Lo standard è `DistributedDataParallel` (DDP): un processo per GPU,
comunicazione NCCL, e l'all-reduce eseguito *durante* il backward, a pacchetti
(bucket), così che comunicazione e calcolo si sovrappongano. Il batch efficace
diventa $K$ volte quello per replica: al crescere di $K$ va spesso ritoccato
il learning rate. Il vecchio `DataParallel` (processo unico, multi-thread)
sopravvive nei tutorial ma è sconsigliato dalla documentazione stessa: il GIL
di Python (un thread alla volta esegue bytecode, come si è visto nel capitolo
su Python) e lo sbilanciamento sulla GPU 0 ne fanno un reperto storico. DDP
gira un processo per GPU proprio per questo: un GIL a testa.
`````

In codice, DDP è uno schema più che una libreria da imparare, e chi ha una
scheda sola (cioè quasi tutti) può guardarlo di sfuggita: quello che c'era da
capire l'hanno detto la figura e le due schede qui sopra. Lo snippet che
segue non si lancia con `python`: serve un *launcher*, `torchrun`, che avvia
un processo per GPU e assegna a ciascuno la propria identità.

```{code-block} python
:class: pt-non-eseguibile

# SCHEMA, si lancia con: torchrun --nproc_per_node=4 addestra.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

dist.init_process_group("nccl")                 # collega i 4 processi
rank = int(os.environ["LOCAL_RANK"])            # chi sono io? (0, 1, 2 o 3)
torch.cuda.set_device(rank)

model = DDP(model.to(rank), device_ids=[rank])  # replica sincronizzata

sampler = DistributedSampler(train_data)        # a ognuno la sua fetta
loader = DataLoader(train_data, batch_size=64, sampler=sampler)

for epoca in range(epoche):
    sampler.set_epoch(epoca)                    # rimescola in modo coordinato
    for X, y in loader:
        ...                                     # training loop IDENTICO:
                                                # l'all-reduce avviene da solo
                                                # dentro loss.backward()
dist.destroy_process_group()
```

Il punto notevole sono gli ultimi commenti: il training loop non cambia di una
riga. DDP intercetta il `backward()` e ci innesta la media dei gradienti;
tutto il resto (loss, ottimizzatore, precisione mista) è il codice che già
conosci.

## Partire col piede giusto: `nn.init`

C'è un'ottimizzazione che non riguarda la velocità dell'hardware ma quella
dell'*apprendimento*: da quali valori partono i pesi.

Che la cosa conti non è ovvio, e mezza riga di spiegazione la merita. Prima di
imparare qualunque cosa, i pesi di una rete sono numeri a caso, e la domanda è
*quanto* grandi. Se sono troppo piccoli, il segnale che entra da una parte si
smorza attraversando gli strati e dall'ultimo esce quasi zero: non c'è niente
da correggere, e la rete non parte. Se sono troppo grandi succede il contrario,
il segnale si gonfia strato dopo strato e i numeri esplodono. Le due ricette
classiche, **Xavier/Glorot** e **He**, sono due modi di scegliere quella scala
in funzione di quanti ingressi ha ciascun neurone, la prima pensata per
attivazioni simmetriche come la tanh, la seconda per la ReLU. Il capitolo sul
deep learning ne darà la ragione per esteso; qui vediamo il gesto con cui si
applicano. I default
di PyTorch sono ragionevoli (per `nn.Linear`, una variante uniforme mantenuta
per compatibilità storica col vecchio Torch), ma quando si vuole il controllo
esplicito il modulo `torch.nn.init` offre le ricette pronte, con la solita
convenzione dell'underscore finale per le operazioni *in-place* vista nella
sezione sui tensori:

```python
from torch import nn

def inizializza(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")  # ricetta He
        nn.init.zeros_(m.bias)

model.apply(inizializza)   # applica la funzione a ogni sotto-modulo
```

`apply()` visita ricorsivamente tutti i sotto-moduli e passa ciascuno alla
funzione; l'`if isinstance` fa da filtro, così solo gli strati `nn.Linear`
ricevono la ricetta He (`kaiming`, dal nome di Kaiming He). Lo stesso schema
serve per qualunque intervento mirato sui pesi di una rete già costruita.

## E chi non ha otto GPU?

Onestà: quasi nessun lettore di questo libro addestrerà su un cluster, e va
benissimo così. La ricerca che conta si fa anche con una GPU sola: AlexNet, da
cui siamo partiti, ne aveva due. Per chi lavora su una macchina normale, le
leve che spostano davvero il cronometro sono più modeste e più vicine:

- **Riempi la GPU**: alza il `batch_size` finché la memoria regge (quando non
  regge più, PyTorch protesta con un *out of memory*: si abbassa e si
  riprova). Una GPU mezza vuota è il modo più comune di sprecarla.
- **Rifornisci la GPU**: se l'utilizzo della scheda langue, il collo di
  bottiglia è quasi sempre la catena dei dati, non il calcolo. Nel
  `DataLoader`, `num_workers=4` (o quanti core hai) prepara i batch in
  parallelo mentre la GPU lavora, e `pin_memory=True` accelera il
  trasferimento.
- **Precisione mista anche in piccolo**: i tensor core non sono un lusso da
  datacenter; li hanno tutte le GeForce RTX. Le cinque righe di `autocast`
  viste sopra sono spesso il singolo guadagno più grande disponibile su una
  GPU consumer.
- **Nessuna GPU?** Google Colab ne offre una gratis (con limiti di tempo), e
  tutto il codice di questo capitolo ci gira senza modifiche.

E prima di ogni ottimizzazione, la regola che vale a ogni scala: **misura**.
Un cronometro attorno a un'epoca (`time.time()` basta) e un'occhiata
all'utilizzo della scheda (`nvidia-smi`) dicono in trenta secondi dove va il
tempo; ottimizzare senza misurare è potare un albero al buio.

Con questa sezione il capitolo si chiude: dal primo tensore alla macchina
tenuta a regime. Il capitolo successivo apre il cofano dell'hardware.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Una rete neurale, vista dall'hardware, è quasi soltanto
  **moltiplicazioni di tabelle di numeri**: milioni di conti identici e
  indipendenti. È il lavoro perfetto per una scheda grafica, che è fatta di
  migliaia di operai semplici invece che di pochi bravissimi.
- La **precisione mista** usa numeri corti dove bastano e lunghi dove
  servono: quasi il doppio della velocità, quasi gratis, su qualunque scheda
  moderna. L'unica insidia sono i numeri piccolissimi, e c'è una lente
  d'ingrandimento apposta.
- `torch.compile` legge il modello tutto insieme e riorganizza i passaggi:
  conviene sugli addestramenti lunghi, non sugli assaggi, perché studiare la
  ricetta costa e su un modello minuscolo può costare più di quanto rende.
- Se le schede sono più d'una, ognuna prende **una fetta del vassoio**, e alla
  fine tutte mettono in comune le correzioni e ne fanno la media. Il giro di
  addestramento non cambia di una riga.
- Anche da quali numeri si parte conta: troppo piccoli e il segnale si spegne
  attraversando la rete, troppo grandi e esplode. Ci sono ricette pronte.
- Su una macchina sola le leve che spostano il cronometro sono tre: riempire
  la scheda, rifornirla di dati, usare i numeri corti. E prima di tutto,
  **misurare**: ottimizzare senza misurare è potare un albero al buio.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Le reti neurali sono soprattutto **moltiplicazioni di matrici**: milioni di
  conti identici e indipendenti, il lavoro perfetto per le migliaia di core
  semplici di una GPU. Il deep learning moderno nasce da questo incontro
  {cite}`krizhevsky2012imagenet`.
- La **precisione mista** {cite}`micikevicius2018mixed` usa 16 bit dove
  basta e 32 dove serve: `autocast` + `GradScaler` (o `bfloat16` senza
  scaler) per un guadagno quasi gratuito su qualunque GPU con tensor core.
- `torch.compile` (PyTorch 2.0) fonde i kernel in una riga: paga su modelli
  grandi e addestramenti lunghi, non sugli esperimenti brevi, dove il
  compilato può risultare **più lento** dell'eager.
- Il **parallelismo dati** replica il modello su ogni GPU, spartisce il
  batch e media i gradienti con l'**all-reduce**: lo standard è
  `DistributedDataParallel`, lanciato con `torchrun`; il training loop resta
  identico.
- `nn.init` con `apply()` applica le inizializzazioni Xavier/He
  (`kaiming_normal_`) che il capitolo sul deep learning motiverà.
- Su una macchina sola contano `batch_size`, `num_workers`, la precisione
  mista, e il cronometro prima di tutto: riscaldamento fuori dalla misura e
  minimo di più ripetizioni, non la prima.
```
`````
