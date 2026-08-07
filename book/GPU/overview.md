# GPU e calcolo parallelo

Per trent'anni i programmatori hanno goduto di un privilegio che sembrava una
legge di natura: bastava aspettare. Un programma lento oggi sarebbe stato
veloce l'anno seguente, senza toccare una riga di codice, perché i processori
alzavano la loro frequenza di clock a ritmo regolare: il «pasto gratis». Poi,
intorno al 2004, il pasto finì. La regola empirica che per decenni aveva
permesso di rimpicciolire i transistor e alzare il clock a consumi costanti
(lo *scaling di Dennard*) smise di valere: spingere ancora la frequenza
significava dissipare troppo calore, e i chip si schiantarono contro un muro
fisico, il *power wall*. Nel 2004 Intel rinunciò ai processori che avrebbero
dovuto correre a frequenze sempre più alte; nel 2005 Herb Sutter mise la cosa
nero su bianco in un saggio dal titolo diventato celebre, *The Free Lunch Is
Over*. La morale era semplice e spiazzante: da lì in avanti, per andare più
veloci non si sarebbe più potuto contare su un core più rapido, ma solo su
*più core che lavorano insieme*. Il futuro, scriveva Sutter, era parallelo
({numref}`fig-free-lunch`).

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
paradiso del parallelismo. Nel 2007 NVIDIA aprì quei chip al calcolo generico
con **CUDA** {cite}`nickolls2008scalable`, e la stessa folla di esecutori che
coloriva pixel si rivelò perfetta per un altro compito fatto di conti identici
e indipendenti: addestrare reti neurali. Nel 2012 la conferma arrivò a
sorpresa da una gara di riconoscimento di immagini, ImageNet, vinta da
**AlexNet** con un margine mai visto: addestrata su normali schede da
videogiocatori, come racconta la sezione «Prestazioni e scala»
{cite}`krizhevsky2012imagenet`. Da allora hardware parallelo e deep learning
non si sono più lasciati.

Questo capitolo è, appunto, l'approfondimento «sotto il cofano» di quella
sezione. Lì abbiamo imparato i *gesti*: `.to(device)`, la precisione mista con
`autocast`, la riga di `torch.compile`, il parallelismo dati con
`DistributedDataParallel`, e li abbiamo giustificati a grandi linee, con
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
manodopera. Un solo strato di una rete neurale su un vassoio di esempi sono
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
*quanti byte* le arrivano da calcolare. Le migliaia di core sono la parte facile;
tenerle rifornite è l'ingegneria vera.

```{figure} ../figures/gpu-cloud-vs-on-premise.svg
:name: fig-pareggio-cloud
:alt: "Grafico con le ore di GPU usate al mese in ascissa e il costo mensile in ordinata. La retta del cloud parte da zero e sale proporzionalmente all'uso; quella dell'hardware proprio parte alta, per l'acquisto, e poi cresce poco. Le due si incrociano in un punto di pareggio, oltre il quale conviene possedere la scheda."
:width: 90%

Due rette e un incrocio. Il cloud non è più caro né più economico in assoluto:
lo diventa a seconda di quante ore al mese la scheda resta accesa.
```

Il punto di pareggio in {numref}`fig-pareggio-cloud` è un conto che conviene
fare prima di affezionarsi a una risposta. Per un uso saltuario la retta piatta
dell'hardware proprio è quasi tutta costo fisso sprecato; per un
addestramento continuo, il noleggio diventa la voce di spesa dominante.

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
GPU fino a come si addestrano i modelli che non entrano in una scheda sola.

- **Dentro la GPU: come è fatta e come esegue** (la scommessa opposta a quella
  della CPU, gli **Streaming Multiprocessor**, la gerarchia
  griglia–blocco–**warp**, il modello **SIMT** e l'**occupancy** con cui la
  GPU copre le attese tenendo in volo migliaia di thread).
- **La memoria: il vero collo di bottiglia**; la piramide dai registri alla
  HBM, gli accessi **coalescenti**, il riuso in shared memory e il modello
  **roofline** che dice se un calcolo è limitato dai byte o dai conti.
- **Kernel: dare ordini a migliaia di thread**, che cos'è un *kernel* e come
  lo si scrive (un mini-esempio in **Triton**), il modello CUDA/SIMT e perché
  **fondere** più operazioni in un kernel solo taglia i viaggi in memoria.
- **GEMM: la moltiplicazione di matrici, spremuta**, la routine su cui ogni
  rete spende il grosso del tempo: il **tiling** che la rende veloce, i
  **tensor core** che eseguono un intero prodotto tra piccole matrici per
  colpo di clock, e l'**array sistolico**, cioè la scelta opposta fatta dagli
  acceleratori dedicati.
- **Flash Attention: l'attenzione che non spreca memoria**, l'attenzione
  riorganizzata per non scrivere mai la matrice $N \times N$: **tiling** e
  **online softmax**, l'esempio più limpido di calcolo *IO-aware*, fino alla
  frontiera dei kernel di oggi.
- **Oltre una GPU: parallelismo distribuito**, quando un modello non entra in
  una scheda sola: parallelismo dati, tensor e pipeline, lo *sharding* di
  **ZeRO/FSDP**, e come si combinano nei modelli di frontiera.

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
