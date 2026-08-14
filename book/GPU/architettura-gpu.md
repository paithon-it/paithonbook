# Dentro la GPU: come è fatta e come esegue

Nel 1999 NVIDIA lanciò la GeForce 256 e la vendette come «la prima GPU al
mondo»: da lì in poi **Graphics Processing Unit** è il nome con cui chiamiamo
questi chip. Il compito era disegnare i mondi dei
videogiochi: milioni di pixel e di poligoni da illuminare, ruotare, colorare
sessanta volte al secondo. È un lavoro fatto di un solo gesto ripetuto
all'infinito: prendi un vertice, applicagli la stessa trasformazione, passa al
prossimo. Nessuno di questi conti è difficile; sono soltanto tantissimi, e
indipendenti l'uno dall'altro. Per farli in fretta non serve un cervello
raffinato che ragiona su un problema alla volta: serve una folla che lavora
tutta insieme.

Da qui una scommessa di progettazione opposta a quella delle CPU. Invece di
costruire pochi processori velocissimi sul singolo flusso di istruzioni, i
progettisti delle GPU ne misero migliaia, ciascuno lento e limitato ma tutti
attivi nello stesso istante. Per anni fu una scommessa confinata alla grafica.
Poi, nel 2006-2007, NVIDIA aprì quei chip al calcolo generico con **CUDA**
{cite}`nickolls2008scalable`, e qualcuno si accorse che la stessa folla di
manovali brava a colorare pixel era perfetta anche per un altro lavoro fatto
di conti identici e indipendenti: addestrare reti neurali. Nel 2012 AlexNet
riscosse la scommessa vincendo ImageNet su due schede da videogiocatori
{cite}`krizhevsky2012imagenet`, e da allora hardware e deep learning non si
sono più lasciati.

Nella sezione «Prestazioni e scala» del capitolo su PyTorch abbiamo già
incontrato l'immagine della GPU come squadra di operai semplici. Qui apriamo
il cofano: com'è fatta dentro, e (soprattutto) *come esegue* il codice. Perché
il segreto delle prestazioni non è che ogni operaio sia veloce (non lo è), ma
come sono organizzati, a squadre, e come si coprono a vicenda i tempi morti.

## Due filosofie: la lepre e il formicaio

CPU e GPU risolvono lo stesso problema di fondo (far girare istruzioni su
dati) partendo da due domande diverse. La CPU chiede: «come faccio a finire
*questo* compito il prima possibile?». La GPU chiede: «come faccio a finire
*il maggior numero* di compiti per unità di tempo?». Le due domande hanno un
nome ciascuna, e li useremo per tutto il capitolo: la prima è la **latenza**,
cioè quanto si aspetta perché una singola cosa sia pronta; la seconda è il
**throughput**, cioè quante cose vengono fuori in un'ora. Ottimizzare l'una
spesso significa sacrificare l'altro.

`````{tab} Elementare

Immagina di dover consegnare dei pacchi. La lepre è velocissima: prende un
pacco, sfreccia, lo consegna, torna indietro, ne prende un altro. Se hai *un*
pacco urgente, la lepre è imbattibile. Ma se ne hai diecimila, quella corsa
avanti e indietro non basta più. Il formicaio funziona all'opposto: ogni
formica è lenta, ma sono migliaia e partono tutte insieme. Il primo pacco
arriverà un po' più tardi che con la lepre (nessuna formica è veloce) ma nello
stesso tempo ne arrivano diecimila. La lepre ha la **latenza** più bassa (il
singolo pacco arriva prestissimo), il formicaio il **throughput** più alto
(nella giornata ne arrivano molti di più): sono le due parole del paragrafo
qui sopra, ed è così che il capitolo le userà. La CPU è la lepre: pochi
processori potentissimi, pensati per finire in fretta il singolo compito. La
GPU è il formicaio: tante unità lente, pensate per smaltire una montagna di
compiti tutti insieme. Per aprire un file o rispondere a un clic vuoi la
lepre; per fare sessanta milioni di moltiplicazioni identiche (un solo strato
di una rete), vuoi il formicaio.

`````

`````{tab} Superiore

Una CPU è *latency-oriented*: pochi core (dell'ordine della decina), ma
complessi, con grandi cache per tenere i dati vicini, predizione dei salti ed
esecuzione fuori ordine per non fermarsi mai su un singolo flusso di
istruzioni. Gran parte del silicio è spesa in logica di controllo e memoria,
non in unità di calcolo. Una GPU è *throughput-oriented*: rovescia il
bilancio. Il silicio va quasi tutto in **ALU** (le unità aritmetiche, i «CUDA
core»), pochissimo in controllo e cache per core. Il singolo thread è lento e
non ha trucchi per nascondere le proprie attese; la GPU nasconde la latenza in
un altro modo, statisticamente: tiene *moltissimi* thread pronti e, quando uno
si ferma in attesa di un dato dalla memoria, ne fa partire un altro già
pronto. Non accorcia l'attesa del singolo: la *copre* con il lavoro degli
altri. È una scelta sensata solo se il problema offre parallelismo a valanga,
ed è esattamente il caso delle reti neurali: come richiamato nella sezione
«Prestazioni e scala», il prodotto di due matrici $(n,m)$ e $(m,p)$ costa
circa $2nmp$ operazioni tutte indipendenti.

`````

## Lo Streaming Multiprocessor: la GPU come federazione di officine

Vista da lontano, una GPU sembra un blocco unico. Da vicino è un insieme di
unità quasi autonome, gli **Streaming Multiprocessor** (SM): sono le officine in
cui il lavoro viene davvero eseguito. Una GPU moderna ne ha da qualche decina a
oltre un centinaio, e la
sua potenza cresce, prima di tutto, moltiplicando gli SM.

Ogni SM è una piccola macchina completa, con i suoi calcolatori, il suo
caposquadra e i suoi ripiani di lavoro. Contiene:

- molte **ALU**, le unità aritmetiche che fanno materialmente i conti (una
  moltiplicazione, una somma): sono i «CUDA core», gli operai della folla di
  cui si parlava sopra. Sulle schede recenti accanto a esse ci sono anche i
  **tensor core**, unità costruite apposta per il prodotto tra matrici, che
  vedremo in una sezione dedicata;
- uno o più **warp scheduler**, i caposquadra: decidono, momento per momento,
  quale gruppo di lavoratori far avanzare;
- un grande **register file**, il taccuino: la memoria velocissima dove ogni
  lavoratore tiene i numeri su cui sta operando in questo istante;
- un blocco di **shared memory**, il tavolo comune: una memoria di lavoro
  condivisa fra i lavoratori della stessa squadra. La incontreremo in dettaglio
  nella prossima sezione, perché è la chiave delle prestazioni.

I «lavoratori» di questo elenco hanno un nome che ricorrerà in tutto il
capitolo: si chiamano **thread**, e la sezione qui sotto racconta come sono
organizzati (in squadre, e dentro le squadre in plotoni da 32 detti *warp*).

Gli ordini di grandezza aiutano a fissare le proporzioni, senza inseguire il
numero esatto di un modello specifico (che cambia a ogni generazione): **da
parecchie decine a oltre un centinaio di SM** per GPU, per un totale di
**migliaia o decine di migliaia di CUDA core**, capaci
di tenere in volo **centinaia di migliaia di thread** contemporaneamente.
Questa scala è la ragione dei numeri visti nel capitolo su PyTorch: le decine
di migliaia di miliardi di **operazioni in virgola mobile** al secondo (in
virgola mobile vuol dire «sui numeri con la virgola», quelli con cui una rete
neurale lavora; l'inglese le chiama *floating-point operations*, e da lì la
sigla **FLOP**, che nel resto del capitolo indicherà appunto un conto
elementare, una moltiplicazione o una somma). È anche il motivo per cui la GPU
divora le moltiplicazioni tra matrici delle reti neurali.

## La gerarchia dei thread: griglia, blocchi, warp

Con migliaia di unità di calcolo, la vera domanda diventa organizzativa: come
si dice a decine di migliaia di lavoratori *chi fa cosa*, senza scrivere
decine di migliaia di istruzioni diverse? La risposta di CUDA
{cite}`nickolls2008scalable` è una gerarchia a tre livelli, illustrata in
{numref}`fig-gpu-esecuzione`.

```{figure} ../figures/gpu-gerarchia-esecuzione.svg
:name: fig-gpu-esecuzione
:alt: "In alto la scomposizione logica da sinistra a destra; una griglia (grid) di blocchi, un blocco (CTA) fatto di più warp, un warp di 32 thread, il singolo thread che elabora un dato. In basso l'hardware: una GPU come insieme di Streaming Multiprocessor; una freccia tratteggiata collega un blocco a uno SM, a indicare che l'hardware assegna ogni blocco a uno SM che lo esegue a warp di 32 thread."
:width: 100%

La gerarchia logica dei thread (in alto) e la sua esecuzione fisica (in
basso). Il programmatore lancia una griglia di blocchi; l'hardware assegna
ogni blocco a uno Streaming Multiprocessor, che lo esegue in gruppi da 32
thread: i warp.
```

`````{tab} Elementare

Pensa a un grande censimento da svolgere in una città. Il **thread** è il
singolo rilevatore, che si occupa di *una* casa. Per non impazzire, i
rilevatori si organizzano in **squadre** (i «blocchi»): quelli di una squadra
lavorano nello stesso quartiere, si passano informazioni e si coordinano tra
loro. Tutte le squadre insieme formano l'**operazione cittadina** (la
«griglia»), che copre l'intera città. Il capo del censimento non dà ordini a
ogni singolo rilevatore: dice «voglio una griglia di 100 squadre da 256
rilevatori l'una», e lascia che l'organizzazione si dispieghi da sola. C'è poi
un dettaglio che viene dall'hardware: dentro ogni squadra i rilevatori
marciano in **plotoni da 32**, che ricevono l'ordine tutti nello stesso
istante. Questo plotone da 32 ha un nome che ci accompagnerà per tutto il
capitolo: **warp**. Ricordalo, perché quel numero 32 non è un dettaglio: è il
battito del cuore della GPU.

`````

`````{tab} Superiore

Il modello di programmazione CUDA espone tre livelli:

- **thread**: l'unità elementare, esegue il *kernel* (il programma della GPU,
  argomento della prossima sezione) su un proprio pezzo di dato;
- **block** (o *CTA*, Cooperative Thread Array): un gruppo di thread che
  condividono la shared memory dell'SM e possono sincronizzarsi tra loro;
- **grid**: l'insieme di tutti i blocchi lanciati per un kernel.

Dalle GPU **Hopper** in poi (compute capability 9.0) fra griglia e blocco c'è
un quarto livello, facoltativo: il **thread block cluster**, un gruppetto di
blocchi che l'hardware garantisce residenti su SM vicini e che possono leggere
e scrivere la shared memory l'uno dell'altro (*distributed shared memory*). Non
è un dettaglio esotico: è il livello su cui poggiano le tecniche di
FlashAttention di oggi, che la sezione dedicata riprende in fondo.

Il programmatore sceglie forma e dimensione di griglia e blocchi al momento del
lancio; l'hardware assegna ciascun blocco a uno SM e lo tiene lì fino alla fine.
Un SM può ospitare più blocchi in parallelo, se le risorse (registri, shared
memory) bastano; i blocchi che non entrano restano in coda e partono quando un
SM si libera. Questo rende un programma CUDA **scalabile in modo trasparente**:
lo stesso codice gira su una GPU con 20 SM o con 120, distribuendo gli stessi
blocchi su più o meno officine, senza cambiare una riga.

Sotto il blocco c'è un quarto livello, che il programmatore non specifica ma non
può ignorare: l'hardware esegue i thread di un blocco in **warp** da 32. Un
blocco da 256 thread è, fisicamente, 8 warp. Il warp è l'unità di
*schedulazione*: il warp scheduler non muove un thread alla volta, muove un
warp intero.

`````

## SIMT: stessa mossa, dati diversi

Perché proprio i warp da 32? Perché un chip ha una superficie di silicio
limitata, e ogni millimetro speso a decidere *cosa* fare è un millimetro non
speso a *fare*: far condividere a 32 lavoratori un solo apparato di comando
libera spazio per altre unità di calcolo, e più unità di calcolo vuol dire più
conti al secondo a parità di chip. Tutti e 32 ricevono la stessa istruzione
nello stesso momento e la eseguono insieme, ognuno sul proprio dato. Il numero
32, invece, non ha niente di necessario: è la scelta di chi ha progettato
queste GPU (AMD per anni ne ha usati 64) ed è rimasta identica da vent'anni,
tanto che ormai conviene trattarla come una costante dell'hardware. NVIDIA
chiama questo modello **SIMT**: *Single Instruction, Multiple Threads*.

`````{tab} Elementare

Torniamo al plotone da 32. Il sergente grida un solo ordine («fai un passo
avanti!») e tutti e trentadue lo eseguono insieme, ciascuno sul proprio pezzo
di strada. Un solo ordine, trentadue esecuzioni: è efficientissimo, finché
tutti devono fare la stessa cosa. Il guaio nasce a un bivio. Immagina l'ordine
«se il tuo numero è pari vai a destra, se è dispari vai a sinistra». Il
plotone non può separarsi: il sergente dà *un* ordine alla volta. Allora fa
marciare a destra i pari mentre i dispari stanno fermi ad aspettare; poi fa
marciare a sinistra i dispari mentre i pari aspettano. I due gruppi hanno
percorso strade diverse, ma in fila invece che insieme, impiegando il doppio
del tempo. Morale: sulla GPU i «se... allora... altrimenti...» in cui i 32
compagni di plotone prendono strade diverse costano cari. Il codice più veloce
è quello in cui tutti fanno la stessa mossa.

`````

`````{tab} Superiore

In un warp, i 32 thread condividono il fetch e il decode dell'istruzione: una
sola istruzione, emessa una volta, guida trentadue percorsi di dati. È un
compromesso a metà strada tra il puro SIMD (una istruzione su un vettore di
dati, senza thread distinti) e il multithreading indipendente: da qui il nome
SIMT. Il costo si paga sulla **warp divergence**. Quando un ramo condizionale
manda thread dello stesso warp su percorsi diversi, l'hardware non può
eseguirli davvero in parallelo: *serializza* i rami, attivando di volta in
volta solo i thread che seguono quel ramo e mascherando gli altri. Nel caso
peggiore (32 percorsi distinti) un warp divergente costa fino a 32 volte un
warp coerente.

Qui è d'obbligo una distinzione storica, perché il meccanismo è cambiato e
molte spiegazioni in giro descrivono ancora la macchina di prima. **Fino a
Pascal** (2016) il warp aveva un *unico* program counter condiviso dai 32
thread, più una maschera di attivazione che diceva quali fossero vivi in quel
momento: i thread di un warp, letteralmente, non potevano trovarsi in due punti
diversi del programma. **Da Volta** (2017) ogni thread ha program counter e
stack di chiamata **propri** (*independent thread scheduling*), e a raggruppare
per l'emissione i thread che eseguono la stessa istruzione pensa uno *schedule
optimizer*. L'esecuzione resta SIMT e la divergenza costa ancora, perché rami
diversi non possono essere emessi nello stesso ciclo; ciò che cambia è che
thread divergenti dello stesso warp possono ora sincronizzarsi e scambiarsi
dati (ed è possibile scrivere lock e schemi produttore-consumatore dentro un
warp). Il rovescio della medaglia riguarda chi scrive kernel: i vecchi kernel
«warp-sincroni», che davano per scontato l'avanzamento in blocco senza
sincronizzarsi esplicitamente, non sono più corretti, e servono le primitive
come `__syncwarp()`.

La regola pratica, invece, non cambia: tieni i rami condizionali allineati alla
granularità del warp, così che i 32 thread restino il più possibile *coerenti*
e nessuno resti a mascherare tempo.

`````

## Nascondere la latenza: l'occupancy

Resta la domanda cruciale. Ogni thread, prima o poi, chiede un dato alla
memoria e deve aspettare: centinaia di **cicli**, un'eternità per un
processore. Un ciclo è un battito del metronomo di cui si parlava all'inizio
del capitolo, e una GPU ne fa più di un miliardo al secondo: centinaia di cicli
sono meno di un milionesimo di secondo, ma per la GPU sono come per noi restare
fermi mezz'ora davanti a una porta chiusa. Se si fermasse a ogni attesa, tutta
la sua potenza sarebbe sprecata. La mossa che la salva non è aspettare meno, ma
**avere sempre qualcos'altro da fare**.

`````{tab} Elementare

Torniamo in cucina, ma con i ruoli chiari: stavolta il cuoco non è la GPU
intera, è il caposquadra di *una sola* officina, e le pentole sono i plotoni da
32 che ha in carico.

Immagina dunque un cuoco che gestisce dieci pentole sui fornelli invece di una.
La pasta della prima deve bollire dieci minuti: un cuoco con una pentola sola
se ne starebbe lì a fissare l'acqua. Il nostro invece, mentre la prima bolle,
gira a mescolare la seconda, assaggia la terza, impiatta la quarta. Quando
torna alla prima, i dieci minuti sono passati «gratis», coperti dal lavoro
sulle altre. La GPU fa esattamente questo con i warp. Mentre un warp aspetta
un dato dalla memoria, il suo scheduler mette al lavoro un altro warp già
pronto, e poi un altro ancora. L'attesa del singolo non si accorcia: si
*nasconde* dietro il lavoro degli altri. Ma questo funziona a una condizione:
che di pentole sul fuoco ce ne siano abbastanza. Se il cuoco ne avesse solo
due, passerebbe comunque gran parte del tempo a fissare l'acqua.

Quante pentole ha sul fuoco un'officina in un dato momento, in rapporto a
quante potrebbe averne, è proprio la parola del titolo: si chiama
**occupancy**, e tenerla alta vuol dire tenere il caposquadra sempre con
qualcosa da fare. Con una precisazione che vale la pena portarsi dietro: ciò
che conta davvero non è il numero di pentole, è il numero di **richieste in
viaggio** verso il magazzino. Poche pentole che chiedono ciascuna una cassetta
intera tengono la strada piena quanto molte pentole che chiedono un pacchetto
per volta.

`````

`````{tab} Superiore

La misura di «quanti warp l'SM tiene in volo» si chiama **occupancy**: il
rapporto tra i warp attivi su un SM e il massimo che potrebbe ospitarne. Il
passaggio da un warp all'altro è a **costo zero**, perché a differenza della
CPU la GPU non salva e ripristina il contesto: i registri di *tutti* i warp
residenti restano allocati contemporaneamente nel register file dell'SM. Ecco
perché il register file è così grande. Ma è anche una risorsa finita, e da qui
il compromesso: più registri usa ogni thread, meno thread (quindi meno warp)
entrano insieme nell'SM; lo stesso vale per la shared memory consumata da ogni
blocco. Un'alta occupancy dà allo scheduler tanti warp tra cui scegliere e
nasconde bene la latenza di memoria; un'occupancy troppo bassa lascia lo
scheduler a corto di lavoro e l'SM inattivo durante le attese. Attenzione
però: l'occupancy massima non è un fine in sé (spesso un kernel ben scritto
rende di più con occupancy moderata ma buon riuso dei dati) ma un'occupancy
troppo bassa è quasi sempre un sintomo di potenza sprecata.

Il «dipende» diventa una regola usabile se si guarda alla quantità giusta, che
non è il numero di warp ma il numero di **accessi in volo**: per saturare la
banda servono byte in viaggio pari a banda × latenza (è la legge di Little).
Su una A100 da 80 GB, con $1{,}935$ TB/s e una latenza HBM dell'ordine dei
$400$ ns, fanno circa $770$ KB su tutto il chip, cioè circa 7 KB per ciascuno
dei 108 SM. Se ogni thread legge 4 byte, un warp in volo ne porta 128 e servono
una cinquantina di richieste pendenti per SM, cioè quasi tutti i warp residenti:
occupancy alta. Se invece ogni thread legge un `float4` (16 byte, cioè 512 byte
per warp) ne bastano una quindicina, e la banda si satura con meno di un quarto
dei warp. Stesso kernel, stessa occupancy nominale, due regimi opposti: quel che
va tenuto alto sono le richieste in volo, e ci si arriva sia con tanti warp sia
con pochi warp che leggono largo.

`````

Abbiamo così il quadro dell'*esecuzione*: una GPU è una federazione di SM,
ogni SM macina warp da 32 thread in stile SIMT, e nasconde le attese tenendo
in volo tanti warp insieme. Ma quelle attese (l'abbiamo nominate a ogni passo)
di cosa sono attese? Di **dati che arrivano dalla memoria**. È qui il vero
collo di bottiglia: molto più spesso di quanto si creda, una GPU non è lenta
perché calcola poco, ma perché resta a corto di dati da calcolare. Come è
organizzata la memoria della GPU, e come tenerla rifornita, è il tema della
prossima sezione.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- CPU e GPU sono la **lepre e il formicaio**. La CPU ha pochi calcolatori
  velocissimi e finisce in fretta il singolo compito (bassa *latenza*, cioè
  poca attesa); la GPU ne ha migliaia lenti e smaltisce montagne di conti
  uguali (alto *throughput*, cioè tanta roba per ora).
- Una GPU non è un blocco unico: è una federazione di **officine** (gli
  *Streaming Multiprocessor*), da qualche decina a oltre un centinaio, ognuna
  con i propri calcolatori, il proprio caposquadra e il proprio tavolo di
  lavoro. Insieme tengono al lavoro centinaia di migliaia di lavoratori.
- I lavoratori si chiamano **thread** e sono organizzati come in un censimento:
  l'operazione intera (la *griglia*), le squadre di quartiere (i *blocchi*) e,
  dentro ogni squadra, i **plotoni da 32** (i *warp*). Il 32 è il battito del
  cuore della GPU: se lo ricordi, ricordi metà del capitolo.
- Il sergente dà **un ordine solo** a tutto il plotone. Efficientissimo finché
  tutti fanno la stessa mossa; a un bivio («se pari a destra, se dispari a
  sinistra») il plotone si divide e le due strade si percorrono una dopo
  l'altra, in doppio del tempo. Nel codice per GPU i «se... allora...» che
  dividono i compagni di plotone costano cari.
- L'attesa non si accorcia, si **nasconde**: come il cuoco con dieci pentole,
  il caposquadra manda avanti un altro plotone mentre il primo aspetta i dati.
  Quanti plotoni ha pronti si chiama **occupancy**, e tenerla decente è quello
  che tiene l'officina occupata.
- Le attese che si nascondono così sono attese di **dati che arrivano dalla
  memoria**: è il vero collo di bottiglia, ed è l'argomento della sezione
  successiva.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- CPU e GPU incarnano due filosofie opposte: la CPU è *latency-oriented*
  (pochi core complessi, finisce in fretta il singolo compito), la GPU è
  *throughput-oriented* (migliaia di ALU semplici, smaltisce montagne di
  conti identici e indipendenti).
- La GPU è una federazione di **Streaming Multiprocessor** (SM): da parecchie
  decine a oltre un centinaio di SM, per un totale di migliaia o decine di
  migliaia di CUDA core e centinaia di migliaia di thread residenti. Ogni SM ha
  ALU, warp scheduler, un grande register file e la shared memory.
- Il programmatore lancia una **griglia** di **blocchi** di **thread**;
  l'hardware assegna ogni blocco a uno SM e lo esegue in **warp da 32
  thread**: l'unità di schedulazione. Lo stesso codice scala su GPU con più o
  meno SM {cite}`nickolls2008scalable`.
- In stile **SIMT** i 32 thread di un warp eseguono la stessa istruzione su
  dati diversi. I rami condizionali che li mandano su strade diverse (**warp
  divergence**) vengono serializzati: costano. Fino a Pascal il warp aveva un
  program counter unico; **da Volta** ogni thread ha PC e stack propri
  (*independent thread scheduling*), il costo della divergenza resta ma i
  thread di un warp possono sincronizzarsi fra loro.
- La GPU nasconde la latenza della memoria con l'**occupancy**: tanti warp
  residenti, così che mentre uno aspetta un altro lavora. Il cambio di warp è a
  costo zero perché i registri restano tutti allocati. Ciò che va davvero
  tenuto alto sono gli **accessi in volo** (banda × latenza): tanti warp, o
  pochi warp che leggono largo.
- Le attese che l'occupancy nasconde sono attese di **memoria**: il collo di
  bottiglia più frequente, e l'argomento della sezione successiva.
```
`````
