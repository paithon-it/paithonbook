# GPU e calcolo parallelo

Per trent'anni i programmatori hanno goduto di un privilegio che sembrava una
legge di natura: bastava aspettare. Un programma lento oggi sarebbe stato
veloce l'anno seguente, senza toccare una riga di codice, perché i processori
alzavano la loro **frequenza di clock** a ritmo regolare. Il clock è il
metronomo del chip: un *tic* che scandisce le operazioni una dopo l'altra, e
che in un processore moderno batte qualche miliardo di volte al secondo. Farlo
battere più in fretta faceva andare più in fretta ogni programma già scritto:
il «pasto gratis». Poi, intorno al 2004, il pasto finì. La regola empirica che
per decenni aveva permesso di rimpicciolire i **transistor** (i minuscoli
interruttori di cui un chip è fatto: più sono piccoli, più ne stanno nello
stesso quadratino di silicio) e alzare il clock a consumi costanti (lo
*scaling di Dennard*) smise di valere: spingere ancora la frequenza
significava dissipare troppo calore, e i chip si schiantarono contro un muro
fisico, il *power wall*. Nel 2004 Intel rinunciò ai processori che avrebbero
dovuto correre a frequenze sempre più alte; nel 2005 Herb Sutter mise la cosa
nero su bianco in un saggio dal titolo diventato celebre, *The Free Lunch Is
Over*. La morale era semplice e spiazzante: da lì in avanti, per andare più
veloci non si sarebbe più potuto contare su un **core** più rapido, ma solo su
*più core che lavorano insieme*. Un core è un calcolatore completo in
miniatura, capace di eseguire un'istruzione alla volta: fino a quel momento i
chip ne avevano uno, o pochi, e li facevano correre sempre di più; da lì in
avanti ne avrebbero messi tanti, ciascuno alla velocità di prima. Il futuro,
scriveva Sutter, era parallelo ({numref}`fig-free-lunch`).

```{figure} ../figures/free-lunch-parallelismo.svg
:name: fig-free-lunch
:alt: Grafico schematico in scala logaritmica con l'anno sull'asse orizzontale. La linea ocra del numero di transistor sale in modo costante fino al 2020; la linea terracotta della frequenza di clock sale insieme a essa fino a circa il 2005, poi si appiattisce. Un marcatore verticale segna il 2005 come la fine del free lunch; un riquadro spiega che i transistor in più diventano più core, e il futuro è parallelo.

Andamento schematico: il numero di transistor per chip ha continuato a
raddoppiare, ma intorno al 2005 la frequenza di clock (cioè la velocità di un
singolo core) si è fermata. I transistor «in più» hanno smesso di rendere
veloce un core e hanno iniziato a fornire *più core*: la ragione per cui il
calcolo è diventato parallelo.
```

Un chip costruito fin dall'inizio proprio su quel principio (tanti piccoli
esecutori invece di uno solo velocissimo) però esisteva già, e faceva un
mestiere che con l'intelligenza artificiale sembrava non c'entrare nulla:
disegnare i videogiochi. La *Graphics Processing Unit* era nata per la
**rasterizzazione**, trasformare triangoli in milioni di pixel colorati decine
di volte al secondo: un lavoro fatto di conti identici e indipendenti, il
paradiso del parallelismo. Nel 2006-2007 NVIDIA aprì quei chip al calcolo
generico con **CUDA** {cite}`nickolls2008scalable`, e la stessa folla di esecutori che
coloriva pixel si rivelò perfetta per un altro compito fatto di conti identici
e indipendenti: addestrare reti neurali. Nel 2012 la conferma arrivò a
sorpresa da una gara di riconoscimento di immagini, ImageNet, vinta da
**AlexNet** con un margine mai visto: addestrata su normali schede da
videogiocatori, come racconta la sezione «Prestazioni e scala»
{cite}`krizhevsky2012imagenet`. Da allora hardware parallelo e deep learning
non si sono più lasciati.

Questo capitolo è, appunto, l'approfondimento «sotto il cofano» di quella
sezione. Lì abbiamo imparato i *gesti*, cioè le poche righe di codice da
scrivere: spostare i dati sulla scheda (`.to(device)`), far lavorare la rete
con numeri più corti e più svelti da leggere (la precisione mista di
`autocast`), lasciare che PyTorch riscriva da sé il programma in una forma più
efficiente (`torch.compile`), spartire gli esempi fra più schede
(`DistributedDataParallel`). Li abbiamo giustificati a grandi linee, con
l'analogia della GPU come squadra di operai semplici. Qui apriamo il cofano:
*perché* quei gesti funzionano, cosa succede davvero nel silicio quando una
rete gira, e fin dove si può spingere l'hardware. Non serve saper programmare
una GPU per usarla (PyTorch lo fa per noi) ma capire come è fatta dentro
spiega quasi tutto ciò che separa un addestramento veloce da uno lento.

Due idee attraversano tutte le sezioni che seguono, ed è bene tenerle a mente
fin da subito.

## Molti, non veloci: la scommessa del parallelismo

La prima idea è la scommessa architettonica che rovescia quella della CPU.
Invece di costruire pochi processori velocissimi, la GPU ne mette migliaia,
ciascuno lento e limitato, ma tutti attivi nello stesso istante.

```{figure} ../figures/deep-learning-gpu.svg
:name: fig-hardware-e-modelli
:alt: "Due linee del tempo parallele dal 1958 al 2020. Sopra, l'evoluzione dell'hardware disponibile, dai primi calcolatori alle GPU programmabili. Sotto, l'evoluzione dei modelli, dal percettrone alle reti profonde: ogni salto dei modelli segue di poco un salto dell'hardware."
:width: 100%

Due storie che si inseguono. Le idee di rete profonda erano quasi tutte già
scritte quando l'hardware non c'era: a cambiare, e a cambiare tutto, è stato
il secondo binario.
```

L'allineamento di {numref}`fig-hardware-e-modelli` è il motivo per cui questo
capitolo esiste in un libro di machine learning. Non si tratta di curiosità
sistemistica: per lunghi tratti della storia dell'AI il limite non è stato
capire cosa fare, ma poterlo calcolare.

`````{tab} Elementare
Puoi affidare un lavoro a un genio solitario, capace di risolvere in fretta
qualunque problema difficile, oppure a una folla di persone comuni, ognuna
capace di fare solo un conticino elementare ma tutte insieme, nello stesso
momento. Per un problema che cambia di continuo (decisioni, eccezioni,
imprevisti), vince il genio: è la CPU. Ma per una montagna di conti tutti
uguali e indipendenti vince la folla, perché non serve intelligenza, serve
manodopera. Un solo strato di una rete neurale su un vassoio di esempi (il
*mini-batch*: il mazzetto di esempi che la rete guarda in una volta sola) sono
decine di milioni di moltiplicazioni identiche: il lavoro perfetto per la
folla. La GPU è quella folla, e la sezione sull'architettura entra nel
dettaglio di come è organizzata, in squadre che si coprono i tempi morti a
vicenda.
`````

`````{tab} Superiore
È il contrasto fra un'architettura *latency-oriented* e una
*throughput-oriented*. La CPU spende il suo silicio in logica di controllo e
grandi cache per finire in fretta un singolo flusso di istruzioni; la GPU lo
spende quasi tutto in unità aritmetiche, e nasconde la latenza in modo
statistico: quando un gruppo di thread si ferma in attesa di un dato, ne fa
partire un altro già pronto. Non accorcia l'attesa del singolo: la *copre* con
il lavoro degli altri. È una scommessa che paga solo se il problema offre
parallelismo a valanga, ed è esattamente il caso delle reti neurali: come
ricordava la sezione «Prestazioni e scala», il prodotto di due matrici $(n,m)$
e $(m,p)$ costa circa $2nmp$ operazioni, scomponibili in prodotti scalari
indipendenti l'uno dall'altro. La sezione sull'architettura scioglie i
dettagli di questo modello: Streaming Multiprocessor, warp, SIMT, occupancy.
`````

## Il collo di bottiglia è muovere i dati

La seconda idea è meno intuitiva, e proprio per questo va detta subito: il limite
di una GPU, molto più spesso di quanto si creda, non è *quanti conti* sa fare, ma
*quanti byte* le arrivano da calcolare. Un **byte** è la scatoletta in cui sta
un pezzetto di informazione: uno dei numeri che una rete neurale macina ne
occupa due o quattro, un miliardo di byte fa un gigabyte, e la memoria di una
scheda si misura in decine di gigabyte. I byte sono la stoffa di cui i dati
sono fatti, e portarli fin sotto ai core costa tempo. Le migliaia di core sono
la parte facile; tenerle rifornite è l'ingegneria vera.

`````{tab} Elementare
Immagina un cuoco fulmineo che potrebbe sfornare cento piatti al minuto, se
solo avesse gli ingredienti sotto mano. Ma la dispensa è in fondo a un lungo
corridoio, e per ogni piatto qualcuno deve andare a prendere ciò che serve e
riportarlo indietro. Il cuoco, per quanto veloce, passa la giornata ad
aspettare: non è lento lui, è lento il *rifornimento*. Una GPU è spesso così
(una bestia affamata più che un mostro di calcolo) e quasi tutte le tecniche
di questo capitolo servono a una cosa sola: fare più conti con ogni carico di
ingredienti prima di rimandare qualcuno in dispensa. La sezione sulla memoria
racconta com'è fatta questa «dispensa» e perché la sua velocità di consegna
(la *banda*) decide il destino di tanti programmi.
`````

`````{tab} Superiore
La memoria di una GPU è una piramide di livelli, ciascuno un compromesso
diverso fra velocità e capienza: registri velocissimi ma minuscoli, shared
memory on-chip, cache, e la grande HBM off-chip dove vivono pesi e
attivazioni. I gruppi di thread nascondono la *latenza* di ogni accesso, ma la
**banda** (quanti byte al secondo la memoria consegna davvero) è finita, ed è
lei il vero muro. Lo strumento che formalizza tutto questo è il modello
**roofline** {cite}`williams2009roofline`: mette a confronto l'*intensità
aritmetica* di un calcolo (quanti conti fai per ogni byte spostato) con i due
tetti dell'hardware, la banda e il picco di calcolo, e dice se un programma è
*memory-bound* o *compute-bound*. Da qui un filo conduttore che ritroverai in
ogni sezione, accessi coalescenti, riuso in shared memory (*tiling*), fusione
dei kernel, fino alla **FlashAttention** {cite}`dao2022flashattention`:
variazioni sullo stesso tema, fare più conti per ogni byte e tenere il byte il
più vicino possibile ai core.
`````

## Come è organizzato il capitolo

Le sei sezioni scendono, un gradino alla volta, dal modello di esecuzione della
GPU fino a come si addestrano i modelli che non entrano in una scheda sola. I
nomi tecnici qui sotto non vanno capiti adesso: ciascuno ha accanto, fra
parentesi, la cosa che significa, ed è quella la promessa della sezione.

- **Dentro la GPU: come è fatta e come esegue**. La scommessa opposta a quella
  della CPU; gli **Streaming Multiprocessor** (le officine autonome in cui la
  GPU è divisa), la gerarchia griglia–blocco–**warp** (l'organizzazione dei
  lavoratori in operazione, squadre e plotoni da 32), il modello **SIMT** (un
  ordine solo, trentadue esecuzioni) e l'**occupancy** (quanti gruppi la GPU
  tiene pronti per coprire le attese).
- **La memoria: il vero collo di bottiglia**. La piramide dai registri alla
  **HBM** (la memoria grande della scheda), gli accessi **coalescenti**
  (chiedere i dati in fila invece che sparsi), il riuso in shared memory e il
  modello **roofline**, il grafico che dice se un calcolo è limitato dai byte
  o dai conti.
- **Kernel: dare ordini a migliaia di thread**. Che cos'è un *kernel* (il
  programmino che gira sulla GPU) e come lo si scrive, con un mini-esempio in
  **Triton** (un modo di scriverlo in Python); e perché **fondere** più
  operazioni in un kernel solo taglia i viaggi in memoria.
- **GEMM: la moltiplicazione di matrici, spremuta**. La routine su cui ogni
  rete spende il grosso del tempo: il **tiling** (portare i dati sul tavolo di
  lavoro una volta sola) che la rende veloce, i **tensor core** che eseguono
  un intero prodotto tra piccole matrici per colpo di clock, e l'**array
  sistolico**, cioè la scelta opposta fatta dagli acceleratori dedicati.
- **Flash Attention: l'attenzione che non spreca memoria**. L'attenzione dei
  Transformer riorganizzata per non scrivere mai la grande tabella dei
  confronti fra parole: di nuovo **tiling**, più la **online softmax** (fare
  le percentuali a rate, senza aver visto tutti i numeri), fino alla frontiera
  dei kernel di oggi.
- **Oltre una GPU: parallelismo distribuito**. Quando un modello non entra in
  una scheda sola: spartire gli esempi, le matrici o gli strati fra più
  schede, lo *sharding* di **ZeRO/FSDP** (nessuna scheda tiene il modello
  intero) e come queste strategie si combinano nei modelli di frontiera.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Il **«pasto gratis» è finito**: da metà anni Duemila un singolo calcolatore
  in miniatura (un *core*) non diventa più veloce da solo, e per correre
  bisogna metterne tanti a lavorare insieme. La GPU è il chip fatto così: nata
  per disegnare i videogiochi, aperta ai conti di ogni tipo da CUDA
  {cite}`nickolls2008scalable`, e sposata al deep learning quando AlexNet vinse
  ImageNet su due schede da videogiocatore {cite}`krizhevsky2012imagenet`.
- **Il genio contro la folla**: la GPU rinuncia ad avere pochi esecutori
  velocissimi e ne mette moltissimi lenti. È un pessimo affare per un lavoro
  che cambia a ogni passo, ed è l'affare perfetto per milioni di conti tutti
  uguali, che è esattamente ciò di cui una rete neurale è fatta.
- Il vero collo di bottiglia non è quasi mai fare i conti: è **portare i
  numeri** dalla memoria fin sotto ai calcolatori. Il cuoco è veloce, la
  dispensa è lontana.
- Da qui il filo conduttore di tutto il capitolo, che tornerà con nomi diversi
  in ogni sezione: **fare più conti con ogni carico di ingredienti**, e tenere
  gli ingredienti il più vicino possibile a chi cucina.
- Questo capitolo è il «sotto il cofano» della sezione «Prestazioni e scala»:
  non serve saper programmare una GPU per usarla (PyTorch lo fa al posto tuo),
  ma sapere come è fatta spiega perché un addestramento va veloce o lento.
- Quando **una scheda non basta**, il lavoro si spartisce fra più schede: è
  così che nascono i modelli di cui leggiamo i nomi ogni settimana.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Il **«free lunch» è finito**: da metà anni 2000 un core non diventa più
  veloce da solo, e per correre serve il parallelismo. La GPU è il chip
  parallelo per eccellenza: nato per i videogiochi, aperto al calcolo generico
  da **CUDA** {cite}`nickolls2008scalable`, sposato al deep learning da
  AlexNet {cite}`krizhevsky2012imagenet`.
- **Throughput contro latenza**: la GPU baratta la velocità del singolo core con
  il numero di core, ed è perfetta per i conti identici e indipendenti di una rete
  neurale (le moltiplicazioni di matrici).
- Il vero collo di bottiglia è quasi sempre il **movimento dei dati**, non il
  calcolo: la banda di memoria è il muro. Il **roofline**
  {cite}`williams2009roofline` distingue i carichi *memory-bound* da quelli
  *compute-bound*.
- Un unico filo conduttore lega tutto il capitolo, coalescenza, **tiling**,
  **kernel fusion**, **FlashAttention** {cite}`dao2022flashattention`: fare
  più conti per ogni byte spostato, e tenere il byte vicino ai core.
- Questo capitolo è il **«sotto il cofano»** della sezione «Prestazioni e
  scala»: non serve programmare una GPU per usarla (PyTorch lo fa) ma sapere
  come funziona spiega perché un addestramento va veloce o lento.
- Quando **una GPU non basta**, il lavoro si divide su più schede (parallelismo
  dati, tensor, pipeline, sharding): è così che nascono i modelli di frontiera.
```
`````
