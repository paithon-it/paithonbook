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
Avevano ragione: da allora il deep learning e l'hardware sono cresciuti
insieme, ciascuno tirandosi dietro l'altro. Reti più grandi giustificano chip
più potenti, chip più potenti rendono pensabili reti più grandi.

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
neurale, vista dall'hardware, è quasi soltanto una cosa: **moltiplicazioni fra
tabelle di numeri** (le matrici dell'algebra lineare), cioè milioni di prodotti
e somme tutti uguali fra loro e, soprattutto, tutti indipendenti: nessuno di
essi ha bisogno del risultato di un altro.

`````{tab} Elementare
Le squadre a cui affidarlo sono due. La prima è la CPU: otto operai
straordinariamente qualificati, capaci di qualunque compito complicato
(decisioni, eccezioni, lavori sempre diversi). La seconda è la GPU: decine di
migliaia di manovali che sanno fare solo operazioni elementari, ma tutti
insieme, nello stesso istante. Se il lavoro è "prendi questi due numeri,
moltiplicali, somma il risultato" ripetuto milioni di volte, la squadra dei
manovali stravince: non serve intelligenza, serve manodopera. E le reti neurali
sono esattamente quel lavoro.

Un esempio con i numeri. Uno strato che collega 1000 neuroni ad altri 1000:
ciascuno dei mille di destra deve raccogliere un contributo da ciascuno dei
mille di sinistra, quindi sono un milione di moltiplicazioni per una sola
immagine. Su un vassoio di 64 esempi diventano 64 milioni, per un *singolo
strato*, a ogni passo dell'addestramento. Sono conti che una CPU macina in
fila, uno dopo l'altro, mentre una scheda grafica li fa a migliaia nello stesso
istante, perché era nata per fare esattamente questo con i pixel dei
videogiochi.

C'è un modo di sprecare i manovali, ed è il più comune di tutti. Basta
lasciarli senza materiale. Mille braccia ferme in attesa che arrivi il camion
valgono quanto un operaio solo, e un camion che fa mille viaggi con un mattone
per volta è molto peggio di uno che ne porta un bancale. La strada che arriva
al cantiere, poi, è stretta rispetto al piazzale, e per buona parte della
giornata quello che decide quanti muri si tirano su è quanto materiale riesce
ad arrivare, non quante braccia ci sono ad aspettarlo.
`````

`````{tab} Superiore
Il prodotto tra una matrice $(M, K)$ e una $(K, N)$ costa circa $2MNK$
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

Ogni numero di un tensore `float32` occupa 4 byte, cioè 32 bit (un byte sono
otto bit, e il nome `float32` viene da lì). Ma servono davvero tutti? L'idea
della **precisione mista** {cite}`micikevicius2018mixed` è usare numeri da 16
bit, cioè lunghi la metà, nei punti dove la precisione piena non serve, e
tenere il `float32` dove invece è indispensabile.

Perché scrivere numeri più corti faccia andare più veloce non è ovvio, e
conviene dirlo prima di andare avanti: in una rete moderna il tempo se ne va
soprattutto a **spostare** i numeri fra la memoria e le unità di calcolo, non
a farci sopra i conti. Le unità di calcolo, per la maggior parte del tempo,
aspettano. Dimezzare la lunghezza dei numeri dimezza il traffico, e il tempo
scende quasi come lui.

C'è poi un secondo guadagno, e riguarda i **tensor core**: dal 2017 le schede
NVIDIA hanno, accanto alle unità di calcolo normali, dei circuiti costruiti
apposta per moltiplicare piccole tabelle di numeri corti, e quelli entrano in
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
moltiplica la loss per un fattore grande, così anche i gradienti minuscoli
restano visibili; subito prima di aggiornare i pesi, si divide per lo stesso
fattore e tutto torna alla scala giusta. In PyTorch la lente si chiama
`GradScaler`, e il fattore non lo devi scegliere tu. Lo raddoppia finché tutto
fila, e quando ha ingrandito troppo, cioè quando qualche numero esce dai
margini del foglio, butta via quel giro di correzioni e riprende con la metà
dell'ingrandimento. Vedendo i valori che sceglie salterà
all'occhio che sono sempre potenze di due ($65\,536$ è il più comune, cioè
$2^{16}$), e c'è una ragione: moltiplicare per una potenza di due sposta un
numero in virgola mobile senza alterarne nemmeno una cifra. La lente
ingrandisce e non sporca.

Corti però non sono tutti i numeri. Chi pesa segna man mano su un quaderno, e
il totale sul quaderno lo tiene con tutte le cifre. Se sul quaderno c'è
$0{,}512$ e la correzione del giro vale $0{,}0002$, tre cifre non bastano a
farla entrare, e il totale segna ancora $0{,}512$; con tutte le cifre la
correzione entra e si somma alle altre. Diecimila correzioni di quella misura
spostano il totale di due unità intere, oppure non lo spostano affatto, e la
differenza sta tutta in quante cifre il quaderno tiene. Per questo le
pesate si fanno sbrigative e la contabilità no, e i pesi della rete restano
lunghi anche mentre i conti di passaggio viaggiano corti.
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

Nel giro di addestramento visto nella sezione sul
[training loop](addestramento.md), la precisione mista sono cinque righe in
più. Una all'inizio, che crea la lente d'ingrandimento (si chiama
`GradScaler`). Poi `autocast`, che è il comando con cui si dice «da qui a qui
lavora in sedici bit» e che si scrive attorno alle due righe della previsione e
dell'errore: sceglierà lui, operazione per operazione, dove i sedici bit sono
sicuri. E infine tre righe che prendono il posto delle solite `backward()` e
`step()`, per ingrandire prima e rimpicciolire dopo.
Il blocco qui sotto è scritto per girare **davvero**, anche senza GPU, quindi
si costruisce i suoi pezzi e stampa qualcosa a ogni giro: il ciclo che non
stampa niente è un ciclo che non è mai entrato, ed è un guasto che altrimenti
non si vede. Una sola avvertenza sulla prima riga: di formati corti ne esistono
due, `float16` e `bfloat16`, e il codice sceglie il secondo quando gira senza
scheda grafica. La differenza fra i due la vediamo subito dopo il blocco; per
ora basta sapere che sono due modi di scrivere un numero in sedici bit.

```python
import torch
from torch import nn

dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
# su GPU si usa float16 con la "lente"; su CPU l'autocast lavora in bfloat16,
# che della lente non ha bisogno (vedi il testo dopo il blocco)
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

Su CPU stampa tre righe come questa:

```text
batch 0: loss 2.2945 | tipo interno torch.bfloat16
```

Dove `autocast` accetta i sedici bit sono le moltiplicazioni fra tabelle, cioè
il grosso del lavoro; dove li rifiuta sono le somme molto lunghe, in cui gli
arrotondamenti si accumulerebbero. La riga stampata è la prova che sta
funzionando: l'uscita del modello è davvero in mezza precisione, e questo
mentre i pesi, sul disco e in memoria, sono rimasti tutti quanti a 32 bit. Le
conversioni avvengono al volo, dentro le singole operazioni, e il modello non
lo tocca nessuno.

Ed eccola, la differenza fra i due formati corti. Un numero, dentro un
computer, si scrive in due pezzi: uno dice **quanto è grande** (l'ordine di
grandezza: miliardi, oppure miliardesimi) e l'altro dice **con quante cifre**
lo si conosce. I sedici bit si possono spartire fra i due pezzi in modi
diversi, e i due formati corti fanno appunto scelte diverse. Il `float16` tiene
più cifre e meno grandezza; il **bfloat16** fa il contrario, e arriva agli
stessi estremi del `float32`, cioè in giù fino a numeri con trentasette zeri
dopo la virgola, e altrettanto in su.

La conseguenza pratica è che con il bfloat16 il problema dei gradienti
minuscoli, quello per cui serviva la lente d'ingrandimento, non si pone
proprio: nessun gradiente finisce a zero perché era troppo piccolo per essere
scritto. Si perdono cifre decimali, che qui non servono, e si guadagna la
semplicità: il `GradScaler` si può togliere del tutto. Sulle GPU recenti si
sceglie scrivendo `dtype=torch.bfloat16`.

(Nel codice qui sopra lo scaler c'è comunque, perché quel blocco deve girare in
tutti e due i casi. Attenzione a non trarne la conclusione sbagliata, che si
legge spesso: **non** si spegne da solo quando il formato è il bfloat16. Resta
acceso e continua a moltiplicare per $65\,536$, semplicemente non fa né bene né
male, perché moltiplicare e poi dividere per una potenza di due restituisce i
numeri identici a com'erano. In un programma scritto per il solo bfloat16 quelle
righe si tolgono.)

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
non ha finito», e lo si dice due volte, una prima di far partire il cronometro,
perché nella coda può esserci ancora il lavoro di poco fa, e una alla fine,
prima di leggere il tempo. Vale anche al contrario, quando si legge un
risultato: se una riga sembra lentissima, spesso non è lei a essere lenta, è la
prima che ha avuto bisogno del risultato e ha dovuto aspettare tutta la coda
accumulata prima.

La coda però è anche un'occasione. Il vassoio di esempi successivo può mettersi
in viaggio verso la scheda mentre quella sta ancora lavorando su quello di
adesso, e il viaggio finisce per non costare niente, perché avviene nel
frattempo. C'è una condizione, che si dimentica quasi sempre. Il vassoio deve
stare in un punto fisso, non su uno scaffale che ogni tanto viene riordinato;
se qualcuno nel frattempo lo ha spostato, chi va a prenderlo deve prima
cercarlo, e allora il viaggio comincia quando la scheda è già ferma ad
aspettare. Chiedere la partenza anticipata senza la sua condizione non fa
guadagnare niente.

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

Ecco un programma che misura la stessa identica cosa in due modi, una volta
aspettando che la scheda abbia finito e una volta no, e stampa i due tempi
affiancati.

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
lì la coda non c'è: la CPU esegue e basta. Questa è la prova in bianco, quella
che si fa apposta dove il fenomeno **non** deve comparire, e serve a
dimostrare che la differenza che vedremo sulla GPU è del fenomeno e non del
modo di misurare. Su una GPU, invece, la prima riga stampa un tempo
assurdamente piccolo e la seconda quello vero. Conviene rifare questo
esperimento su una scheda vera appena se ne ha una sottomano: se non la si ha,
la presta gratis Google Colab, che è un servizio con cui si eseguono notebook
dal browser su macchine altrui, purché si ricordi di chiedere l'acceleratore
prima di partire.

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

E il piano vale finché il servizio resta quello. Cambiano la pentola o il
numero di coperti, e la ricetta studiata non torna più, così il cuoco si
rimette a leggere da capo e paga di nuovo lo studio. Per accorgersene in tempo,
prima di ogni piatto dà un'occhiata di controllo, e anche quell'occhiata costa.
Su un piatto da due ingredienti costa più di quanto la riorganizzazione faccia
risparmiare, e allora la cucina organizzata resta più lenta di quella
disordinata anche dopo il primo giro. Si può finire in perdita e restarci,
altro che pareggiare.
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
del mini-batch; alla fine le copie si rimettono d'accordo, e quel rimettersi
d'accordo ha un nome che si incontrerà ovunque: **all-reduce**, cioè «ognuno dà
il suo a tutti e tutti ne fanno la media»
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
tre volte più in fretta.

Le pile però devono essere uguali, e per questo si contano prima. Se uno ne
corregge centocinquanta e un altro cinquanta, la media delle tre proposte pesa
i cinquanta compiti quanto i centocinquanta, e la griglia che ne esce non è
quella che sarebbe uscita correggendo tutta la pila di seguito. Ogni ritocco,
poi, nasce ora da trecento compiti e non dai cento che vede un correttore da
solo, quindi il caso ci mette meno del suo e lo si può fare un po’ più deciso.

Le GPU fanno lo stesso: ognuna elabora la sua fetta di esempi, poi tutte
mettono in comune i gradienti, ne fanno la media e si aggiornano allo stesso
modo. La riunione ha un nome tecnico, *all-reduce*, e
un costo: se gli insegnanti passano più tempo a riunirsi che a correggere, il
gioco non vale la candela.
`````

`````{tab} Superiore
Con $R$ repliche e il batch spartito in fette, ogni replica $r$ calcola
$\nabla_\theta \mathcal{L}_r$ sulla propria fetta; l’**all-reduce** calcola

$$
\nabla_\theta \mathcal{L} = \frac{1}{R} \sum_{r=1}^{R} \nabla_\theta \mathcal{L}_r,
$$

dove $\mathcal{L}_r$ è la loss media sulla fetta della replica $r$. Se le
fette hanno la stessa dimensione, questa media è esattamente il gradiente
sull'intero batch: matematicamente non cambia nulla, cambia solo chi fa i
conti. Lo standard è `DistributedDataParallel` (DDP): un processo per GPU,
comunicazione NCCL, e l'all-reduce eseguito *durante* il backward, a pacchetti
(bucket), così che comunicazione e calcolo si sovrappongano. Il batch efficace
diventa $R$ volte quello per replica: al crescere di $R$ va spesso ritoccato
il learning rate. Il vecchio `DataParallel` (processo unico, multi-thread)
sopravvive nei tutorial ma è sconsigliato dalla documentazione stessa: il GIL
di Python (un thread alla volta esegue bytecode, come si è visto nel capitolo
su Python) e lo sbilanciamento sulla GPU 0 ne fanno un reperto storico. DDP
gira un processo per GPU proprio per questo: un GIL a testa.
`````

In codice questo si chiama **DDP**, da `DistributedDataParallel`, ed è più uno
schema che una libreria da imparare; chi ha una scheda sola (cioè quasi tutti)
può guardarlo di sfuggita, perché quello che c'era da capire l'ha già detto la
figura. Il pezzo di codice che segue, per giunta, non si lancia con `python`:
serve un programma di avvio, `torchrun`, che fa partire un processo per ogni
GPU e dice a ciascuno chi è.

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
dell’*apprendimento*: da quali valori partono i pesi.

Che la cosa conti non è ovvio, e mezza riga di spiegazione la merita. Prima di
imparare qualunque cosa, i pesi di una rete sono numeri a caso, e la domanda è
*quanto* grandi. Se sono troppo piccoli, il segnale che entra da una parte si
smorza attraversando gli strati e dall'ultimo esce quasi zero: non c'è niente
da correggere, e la rete non parte. Se sono troppo grandi succede il contrario,
il segnale si gonfia strato dopo strato e i numeri esplodono.

Le due ricette classiche si chiamano **Xavier** (o Glorot, dal cognome di chi
la propose) e **He**, e sono due modi di scegliere quella scala a partire da
quanti ingressi ha ciascun neurone: più ingressi, più piccoli i pesi, perché
tanti contributi piccoli sommati fanno comunque un numero della misura giusta.
Delle due, la prima è pensata per le reti in cui il segnale passa per intero,
positivi e negativi trattati allo stesso modo; la seconda per la ReLU, che i
negativi li schiaccia a zero e quindi ne lascia passare circa metà, e per
compensare vuole pesi un po’ più grandi. Il {doc}`capitolo sul deep
learning </DeepLearning/overview>` ne darà la ragione per esteso; qui vediamo il gesto con cui si
applicano.

I default di PyTorch sono ragionevoli, e conviene sapere quali sono, perché si
legge spesso che i framework moderni usino Xavier o He e per `nn.Linear` non è
vero: il default è una variante uniforme ereditata dal vecchio Torch, che
sorteggia i pesi fra $-1/\sqrt{d}$ e $+1/\sqrt{d}$, con $d$ il numero di
ingressi. Sullo strato da mille ingressi dell'esempio di poco fa,
$\sqrt{1000}$ fa circa $31{,}6$, quindi i pesi nascono tutti fra $-0{,}032$ e
$+0{,}032$: piccoli, come previsto. Quando si vuole il controllo esplicito,
`torch.nn.init` offre le ricette pronte, con la solita convenzione
dell'underscore finale per le operazioni che modificano sul posto, vista nella
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
funzione; l’`if isinstance` fa da filtro, così solo gli strati `nn.Linear`
ricevono la ricetta He (`kaiming`, dal nome di Kaiming He). Lo stesso schema
serve per qualunque intervento mirato sui pesi di una rete già costruita.

## E chi non ha otto GPU?

Onestà: quasi nessun lettore di questo libro addestrerà su un cluster, e va
benissimo così. La ricerca che conta si fa anche su una macchina normale:
AlexNet, da cui siamo partiti, girava su due schede da videogiocatori dentro un
PC, e non su un supercomputer. Per chi ne ha una, o nessuna, le leve che
spostano davvero il cronometro sono più modeste e più vicine:

- **Riempi la GPU**: alza il `batch_size` finché la memoria regge (quando non
  regge più, PyTorch protesta con un *out of memory*: si abbassa e si
  riprova). Una GPU mezza vuota è il modo più comune di sprecarla.
- **Rifornisci la GPU**: se l'utilizzo della scheda langue, il collo di
  bottiglia è quasi sempre la catena dei dati, non il calcolo. Nel
  `DataLoader`, `num_workers=4` (o quanti sono i core del processore, cioè le
  unità di calcolo indipendenti che ha dentro: `os.cpu_count()` le conta)
  prepara i batch in parallelo mentre la GPU lavora, e `pin_memory=True`
  accelera il trasferimento.
- **Precisione mista anche in piccolo**: i tensor core non sono un lusso da
  datacenter; li hanno tutte le GeForce RTX. Le cinque righe di `autocast`
  viste sopra sono spesso il singolo guadagno più grande disponibile su una
  GPU consumer.
- **Nessuna GPU?** Google Colab ne offre una gratis (con limiti di tempo), e
  tutto il codice di questo capitolo ci gira senza modifiche.

E prima di ogni ottimizzazione, la regola che vale a ogni scala: **misura**. Un
cronometro attorno a un'epoca (`time.time()` basta) e un'occhiata all'utilizzo
della scheda dicono in trenta secondi dove va il tempo. Per la seconda si usa
`nvidia-smi`, un comando che si scrive nel terminale mentre l'addestramento
gira e che stampa, fra le altre cose, quale percentuale della scheda è
effettivamente occupata. Ottimizzare senza misurare è potare un albero al buio.

Con questa sezione il capitolo si chiude: dal primo tensore alla macchina
tenuta a regime.

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
  batch e media i gradienti con l’**all-reduce**: lo standard è
  `DistributedDataParallel`, lanciato con `torchrun`; il training loop resta
  identico.
- `nn.init` con `apply()` applica le inizializzazioni Xavier/He
  (`kaiming_normal_`) che il {doc}`capitolo sul deep learning </DeepLearning/overview>` motiverà.
- Su una macchina sola contano `batch_size`, `num_workers`, la precisione
  mista, e il cronometro prima di tutto: riscaldamento fuori dalla misura e
  minimo di più ripetizioni, non la prima.
```
`````

Una riga, in tutto il capitolo, l'abbiamo usata senza aprirla: quella che
sposta il modello e i dati sulla scheda grafica. Funziona, cambia i tempi di
un addestramento, e finora non ha spiegato niente di sé. Il {doc}`capitolo su GPU </GPU/overview>` e
calcolo parallelo apre quella scatola, e da lì in poi «lento» smette di essere
un'impressione e diventa qualcosa che si sa dove andare a cercare.
