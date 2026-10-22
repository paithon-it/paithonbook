# Dentro la GPU: come è fatta e come esegue

Nel 1999 NVIDIA lanciò la GeForce 256 e la vendette come «la prima GPU al
mondo», e da allora **Graphics Processing Unit** è il nome con cui chiamiamo
questi chip. Il loro compito era disegnare i mondi dei videogiochi, con milioni
di punti da spostare e milioni di pixel da colorare, sessanta volte al secondo.

Se il personaggio gira la testa, ad esempio, ogni singolo punto della sua
sagoma va spostato dove la rotazione lo manda: è lo stesso calcolo, per
centinaia di migliaia di punti, ciascuno indipendente dagli altri. Nessuno di
quei calcoli è difficile; sono soltanto tantissimi.

Da qui nasce la scommessa di progetto opposta a quella delle CPU. Una CPU è
fatta per finire in fretta *un* programma, cioè per correre lungo un'unica fila
di istruzioni, e per questo i suoi core sono pochi e complicati. I progettisti
delle GPU, invece, ne misero migliaia, ciascuno lento e limitato, tutti attivi
nello stesso istante.

La sezione {doc}`«Prestazioni e scala» </PyTorch/prestazioni>` descriveva la
GPU come una squadra di operai semplici. Vediamo ora com'è fatta dentro e come
esegue il codice. Il singolo operaio è lento, quindi la velocità dipende da
come gli operai sono organizzati in squadre e da come si coprono a vicenda i
tempi morti.

## Due filosofie: latenza e throughput

CPU e GPU risolvono lo stesso problema di fondo (far girare istruzioni su
dati) partendo da due domande diverse. La CPU chiede: «come faccio a finire
*questo* compito il prima possibile?». La GPU chiede: «come faccio a finire
*il maggior numero* di compiti per unità di tempo?».

Le due domande hanno un nome ciascuna: la prima è la **latenza**, cioè
quanto si aspetta perché una singola cosa sia pronta; la seconda è il
**throughput**, cioè quante cose vengono fuori in un'ora. Ottimizzare l'una
spesso significa sacrificare l'altro.

`````{tab} Elementare

Una lepre e un formicaio devono consegnare dei pacchi. La lepre è velocissima:
prende un pacco, sfreccia, lo consegna, torna indietro, ne prende un altro. Se
hai *un* pacco urgente, la lepre è imbattibile. Con diecimila pacchi, però,
quella corsa avanti e indietro non basta più.

Il formicaio funziona all'opposto: ogni formica è lenta, ma sono migliaia e
partono tutte insieme. Il primo pacco arriverà un po’ più tardi che con la lepre
(nessuna formica è veloce), ma nello stesso tempo ne arrivano diecimila. La
lepre ha la latenza più bassa (il singolo pacco arriva
prestissimo), il formicaio il throughput più alto (nella giornata ne arrivano
molti di più).

Perché allora non si tengono diecimila lepri? Perché una lepre è ingombrante.
Le servono le gambe lunghe, la memoria di tutte le scorciatoie, una borsa di
attrezzi per quando la strada è chiusa, tutta roba da portarsi dietro e da
nutrire. Una formica non ha niente di tutto questo, e nello stesso formicaio ce
ne stanno migliaia. Lo spazio è quello, e si spende in un modo o nell'altro:
poche lepri attrezzate, oppure una folla di operaie spoglie, capaci di un
lavoro solo.

Il formicaio ha poi un vantaggio che si vede solo quando qualcosa va storto.
Una formica che trova una pozzanghera si ferma ad aspettare che scoli, ma
nessuno la aspetta: le altre le passano avanti, la fila continua, e a sera i
pacchi consegnati sono più o meno gli stessi. Il tempo perso da quella
formica c'è tutto, ma sul totale della giornata non si vede.

Davanti alla stessa pozzanghera, la lepre tira fuori i suoi attrezzi e prova a
passare lo stesso; quando non le riesce, si ferma la consegna, perché di lepri
ce n'è una.

La CPU è la lepre: pochi processori potentissimi, pensati per finire in fretta
il singolo compito. La GPU è il formicaio: tante unità lente, pensate per
smaltire una montagna di compiti tutti insieme. Per aprire un file o rispondere
a un clic vuoi la lepre; per fare i centoventotto milioni di conti tutti
uguali di un solo strato di una rete, vuoi il formicaio.

`````

`````{tab} Superiore

Una CPU è *latency-oriented*: ha pochi core ma complessi, con grandi cache per
tenere i dati vicini, predizione dei salti ed esecuzione fuori ordine per non
fermarsi mai su un singolo flusso di istruzioni. I core vanno da qualche unità
fino a qualche centinaio nei processori da server, ciascuno con unità vettoriali
SIMD fino a 512 bit. Gran parte del silicio è spesa in logica di controllo e
memoria, non in unità di calcolo.

Una GPU è *throughput-oriented* e rovescia il bilancio: il silicio va quasi
tutto in ALU (le unità aritmetiche, i «CUDA core») e pochissimo in controllo e
cache per core. Il singolo thread è lento e non ha trucchi per nascondere le
proprie attese; la GPU nasconde la latenza in un altro modo, statisticamente.
Tiene *moltissimi* thread pronti e, quando uno si ferma in attesa di un dato
dalla memoria, ne fa partire un altro già pronto. In altri termini, non
accorcia l'attesa del singolo, ma la *copre* con il lavoro degli altri.

È una scelta sensata solo se il problema offre parallelismo in abbondanza, come
le reti neurali. Il prodotto di due matrici $(M,K)$ e $(K,N)$, ad esempio,
costa circa $2MNK$ operazioni, raccolte in $MN$ prodotti scalari indipendenti
l'uno dall'altro (dentro ciascuno le $K$ somme restano una catena, che si
accorcia a profondità $\log_2 K$ solo sommando ad albero).

`````

## Lo Streaming Multiprocessor: le unità in cui la GPU è divisa

Vista da lontano, una GPU sembra un blocco unico. Da vicino è un insieme di
unità quasi autonome, gli **Streaming Multiprocessor** (SM), ciascuna delle
quali esegue in autonomia una parte del lavoro. Una GPU moderna ne ha da
qualche decina a oltre un centinaio, e la sua potenza cresce, prima di tutto,
moltiplicando gli SM.

Prima di aprirne uno serve una distinzione. Le **ALU** (le unità aritmetiche,
quelle che eseguono materialmente una moltiplicazione o una somma) sono
circuiti stampati nel silicio, in numero fisso (migliaia per chip).

Un **thread**, invece, è un *compito*, ad esempio «occupati tu del numero in
posizione 4173». La parola inglese vuol dire «filo», nel senso di un filo di
lavoro da sbrogliare, non di qualcosa che si possa toccare. Una GPU ha migliaia
di ALU e centinaia di migliaia di thread in carico, cioè qualche decina di
compiti per ogni ALU, e questa eccedenza permette di tenere la macchina sempre
occupata.

Quella parola, però, è già comparsa con un altro significato. Nelle {doc}`basi
di Python
</Python/basi>` i thread erano i cuochi che si contendono un coltello solo: una
manciata, ciascuno con la propria fila di istruzioni, e la morale era che per i
calcoli scritti in Python non servivano a niente. Sulla GPU i thread sono
centinaia di migliaia, ricevono la fila di istruzioni a gruppi di trentadue, e
calcolare è l'unica cosa che fanno. Con quelli di Python, quindi, hanno in
comune solo il nome; cambiano quanti sono, come ricevono gli ordini e a che
cosa servono.

Ogni SM è una piccola macchina completa, e contiene:

- le ALU, che NVIDIA chiama «CUDA core» anche se sono soltanto i circuiti in cui
  avviene il calcolo, non core completi come quelli di una CPU; sulle schede
  recenti, accanto a esse, ci sono anche i tensor core, unità costruite apposta
  per moltiplicare fra loro due piccole matrici, a cui è dedicata la
  {doc}`sezione sul GEMM <gemm-e-tensor-core>`;
- uno o più **warp scheduler**, che a ogni ciclo scelgono quale gruppo di 32
  thread far avanzare (il gruppo si chiama *warp*, e il modello SIMT spiega il
  perché dei 32);
- un grande **register file** (in inglese *file* è anche lo schedario, non
  il documento), cioè la memoria più veloce dell'SM, dove ogni thread tiene i
  numeri su cui sta lavorando in quell'istante;
- la shared memory, una memoria di lavoro condivisa dai thread di uno stesso
  *blocco*, il gruppo in cui CUDA li organizza, e gestita da chi scrive il
  programma; la {doc}`sezione sulla memoria <gerarchia-memoria>` la tratta per
  esteso, perché è la chiave delle prestazioni.

Gli ordini di grandezza aiutano a fissare le proporzioni, senza inseguire il
numero esatto di un modello specifico (che cambia a ogni generazione). Una GPU
ha da qualche decina a oltre un centinaio di SM, e dentro ogni SM da
sessantaquattro a centoventotto ALU in virgola mobile a 32 bit, per un totale
di migliaia o decine di migliaia sul chip.

Da questi numeri si ricava il **picco di calcolo** che compare nelle schede
tecniche. Si moltiplica il numero di SM per le unità di moltiplicazione-e-somma
che ciascuno contiene, per due (ogni unità fa una moltiplicazione e una somma a
ogni ciclo, la cosiddetta FMA, *fused multiply-add*) e per la frequenza di
clock. In simboli, $P_\text{picco} =
n_\text{SM} \cdot n_\text{FMA} \cdot 2 \cdot f$, dove $n_\text{SM}$ è il numero
di SM, $n_\text{FMA}$ le unità di moltiplicazione-e-somma per SM e $f$ la
frequenza.

Ad esempio, su una NVIDIA A100 (una scheda da datacenter del 2020) si ottengono
$108 \cdot 64 \cdot 2 \cdot 1{,}41 \cdot 10^9 \approx 19{,}5 \cdot 10^{12}$
operazioni al secondo sui numeri `float32` (quelli da quattro byte), cioè quasi
ventimila miliardi. È l'ordine di grandezza che serve alle moltiplicazioni fra
matrici di cui una rete neurale è fatta.

Sono operazioni in virgola mobile, cioè sui numeri con la virgola, di cui è
fatta una rete neurale; l'inglese le chiama *floating-point operations* e da lì
viene la sigla **FLOP**. Un FLOP è *una* operazione elementare, una
moltiplicazione o una somma. Quando serve dire quante se ne fanno al secondo si
scrive FLOP/s, e le due cose non vanno confuse, come non si confondono i
chilometri con i chilometri all'ora.

## La gerarchia dei thread: griglia, blocchi, warp

Con centinaia di migliaia di thread, la vera domanda diventa organizzativa: come
si dice a ciascuno *che cosa* fare, senza scrivere centinaia di migliaia di
istruzioni diverse? La risposta di CUDA {cite}`nickolls2008scalable` è
organizzarli su tre livelli. Tutti i thread lanciati insieme formano la
**griglia**, la griglia è divisa in **blocchi**, e ogni blocco è fatto di
thread. La {numref}`fig-gpu-esecuzione` li mette in fila.

`````{tab} Elementare

In una città si fa il censimento, e bisogna bussare a tutte le porte. Il
thread è il singolo rilevatore, che si occupa di *una* casa. Per non
impazzire, i rilevatori si organizzano in squadre (i «blocchi»): quelli di
una squadra lavorano nello stesso quartiere, si passano informazioni e si
coordinano tra loro. Tutte le squadre insieme formano l’operazione
cittadina (la «griglia»), che copre l'intera città. Il capo del censimento
non dà ordini a ogni singolo rilevatore: dice «voglio una griglia di 100
squadre da 256 rilevatori l'una», e lascia che l'organizzazione si dispieghi da
sola.

Le squadre, però, non scelgono dove appoggiarsi: c'è chi assegna ognuna a
un ufficio di zona, e gli uffici di zona sono gli Streaming Multiprocessor. La
squadra resta lì fino a rilevazione finita, perché ci tiene le sue carte
e ci si ritrova a confrontarle.

Gli uffici aperti sono quelli che sono, e in ognuno ci sta qualche squadra per
volta, non di più. Se gli uffici sono venti e le squadre cento, ad esempio,
parte chi ci sta e le altre aspettano che si liberi posto. Il capo non ha
bisogno di saperlo, perché lo stesso ordine («100 squadre da 256») funziona in
un paese con due uffici e in una metropoli che ne ha centoventi, e a cambiare è
quanto ci si mette, non il piano.

Sulle schede più recenti c'è un livello in più, facoltativo, fra la squadra e
l'operazione intera: alcune squadre vengono messe apposta in uffici confinanti,
e possono guardare le carte l'una dell'altra, cosa che con un ufficio
dall'altra parte della città non si può fare.

C'è poi un dettaglio che viene dall'hardware: dentro ogni squadra i rilevatori
marciano in plotoni da 32 (i warp), che ricevono l'ordine tutti nello stesso
istante. Una squadra da 256 rilevatori, quindi, sono otto plotoni, e i
conti tornano sempre così: le squadre si scelgono di una taglia che sia un
multiplo di 32, altrimenti l'ultimo plotone parte mezzo vuoto. Ricordati quel
32, perché è il battito del cuore della GPU.

`````

`````{tab} Superiore

Il modello di programmazione CUDA espone tre livelli:

- thread: l'unità elementare, esegue il *kernel* (il programma che la GPU
  manda in esecuzione) su un pezzo di dato;
- block (o *CTA*, Cooperative Thread Array): un gruppo di thread che
  condividono la shared memory dell'SM e possono sincronizzarsi tra loro;
- grid: l'insieme di tutti i blocchi lanciati per un kernel.

Dalle GPU **Hopper** in poi (compute capability 9.0) fra griglia e blocco c'è
un quarto livello, facoltativo: il **thread block cluster**, un piccolo gruppo
di blocchi che l'hardware garantisce residenti su SM vicini e che possono
leggere e scrivere la shared memory l'uno dell'altro (*distributed shared
memory*).

Il programmatore sceglie forma e dimensione di griglia e blocchi al momento del
lancio; l'hardware assegna ciascun blocco a uno SM e lo tiene lì fino alla fine.
Un SM può ospitare più blocchi in parallelo, se le risorse (registri, shared
memory) bastano; i blocchi che non entrano restano in coda e partono quando un
SM si libera. Questo rende un programma CUDA scalabile in modo trasparente:
lo stesso codice gira su una GPU con 20 SM o con 120, distribuendo gli stessi
blocchi su più o meno SM, senza cambiare una riga.

Sotto il blocco c'è un livello ulteriore, che il programmatore non specifica ma
non può ignorare: l'hardware esegue i thread di un blocco in warp da 32. Un
blocco da 256 thread, ad esempio, è fisicamente fatto di 8 warp. Il warp è
l'unità di *schedulazione*, cioè il warp scheduler non muove un thread alla
volta, ma un warp intero.

`````

```{figure} ../figures/gpu-gerarchia-esecuzione.svg
:name: fig-gpu-esecuzione
:alt: "In alto la scomposizione logica da sinistra a destra; una griglia (grid) di blocchi, un blocco (CTA) fatto di più warp, un warp di 32 thread, il singolo thread che elabora un dato. In basso l'hardware: una GPU come insieme di Streaming Multiprocessor; una freccia tratteggiata collega un blocco a uno SM, a indicare che l'hardware assegna ogni blocco a uno SM che lo esegue a warp di 32 thread."
:width: 100%

Gli stessi tre livelli in figura, dalla griglia al singolo thread. In alto
come li pensa chi scrive il programma: una griglia di blocchi (NVIDIA chiama il
blocco anche CTA, ed è la sigla che compare nel disegno), ogni blocco fatto di
warp, ogni warp fatto di 32 thread, ognuno sul proprio dato. In basso come li
esegue la macchina: ogni blocco finisce su uno degli Streaming Multiprocessor,
e lì avanza un warp per volta.
```

## SIMT: stessa istruzione, dati diversi

Perché i thread avanzano a gruppi, e non ciascuno per conto proprio? Perché
dentro un chip, oltre alle unità che *calcolano*, c'è un'altra parte,
altrettanto ingombrante, che a ogni passo va a prendere l'istruzione seguente,
la decifra e dice alle unità di calcolo cosa devono fare: è la
**logica di controllo**.

Il silicio è una superficie limitata, e ogni millimetro speso nel controllo è
un millimetro non speso a calcolare. Far condividere la logica di controllo a
32 thread, invece di darne una a testa, libera quindi spazio per altre unità di
calcolo, e più unità di calcolo danno più operazioni al secondo a parità di
chip. In questo modo tutti e 32 ricevono la stessa istruzione nello stesso
momento e la eseguono insieme, ognuno sul proprio dato. NVIDIA chiama questo
modello **SIMT** (*Single Instruction, Multiple Threads*), cioè una istruzione
sola per molti thread.

Il gruppo ha un nome, **warp**. Perché 32? Non c'è una ragione profonda. È la
scelta di chi ha progettato queste GPU, e AMD, l'altro grande
produttore, sulle proprie ne ha usati a lungo 64. NVIDIA però non la cambia da
vent'anni e il codice di mezzo mondo è tarato su di essa, quindi va trattata
come una costante dell'hardware, anche se è arbitraria.

`````{tab} Elementare

Un ordine, trentadue esecuzioni. Il sergente grida «fai un passo avanti!» e
tutti e trentadue lo eseguono insieme, ciascuno sul proprio pezzo di strada. È
efficientissimo, finché tutti devono fare lo stesso passo.

Il problema nasce a un bivio. In ogni programma esistono istruzioni della forma
«*se* è vero questo fai una cosa, *altrimenti* fanne un'altra», cioè quelle che
fanno prendere al programma strade diverse a seconda dei dati; senza di esse il
programma non saprebbe fare niente di interessante.

Arriva allora l'ordine «se il tuo numero è pari vai a destra, se è dispari vai
a sinistra». Il sergente può gridare un ordine per volta, e ogni ordine
vale solo per quelli che nomina. Così fa marciare a destra i pari mentre i
dispari stanno fermi ad aspettare; poi fa marciare a sinistra i dispari mentre
i pari aspettano.

I due gruppi hanno percorso strade diverse, ma uno dopo l'altro invece che
insieme, impiegando il doppio del tempo. Se il bivio fosse così fine da mandare
ognuno dei trentadue da una parte sua, si andrebbe avanti uno alla volta:
trentadue turni per un passo solo.

Sui plotoni di oggi, poi, c'è una libertà in più. Fino alle schede del 2016 il
plotone aveva un segnalibro solo, e i trentadue si trovavano sempre allo stesso
punto della strada; adesso ognuno tiene il proprio, e due che si trovano in
punti diversi possono perfino aspettarsi a un incrocio e scambiarsi una parola.

Gli ordini, però, escono ancora uno per volta, e chi non è nominato sta fermo,
quindi il bivio pesa come prima. Il sergente che dava per scontato di trovarli
sempre tutti allineati, ora, prima di farli parlare fra loro deve fare
l'appello e aspettare che rispondano tutti.

Sulla GPU i bivi che separano i 32 compagni di plotone rallentano molto il
lavoro, e il codice più veloce è quello in cui tutti compiono lo stesso passo.

`````

`````{tab} Superiore

In un warp, i 32 thread condividono il fetch e il decode dell'istruzione, per
cui una sola istruzione, emessa una volta, guida trentadue percorsi di dati. È
un compromesso a metà strada tra il puro SIMD (una istruzione su un vettore di
dati, senza thread distinti) e il multithreading indipendente, e da qui viene
il nome SIMT.

Il limite di questo schema è la **warp divergence**. Quando un ramo
condizionale manda thread dello stesso warp su percorsi diversi, l'hardware non
può eseguirli davvero in parallelo, e quindi *serializza* i rami, attivando di
volta in volta solo i thread che seguono quel ramo e mascherando gli altri. Nel
caso peggiore (32 percorsi distinti) un warp divergente impiega fino a 32 volte
il tempo di un warp coerente.

Il meccanismo, però, è cambiato, e molti testi descrivono ancora la macchina di
prima. Fino a Pascal (2016) il warp aveva un *unico* program counter condiviso
dai 32 thread, più una maschera di attivazione che diceva quali fossero vivi in
quel momento. Di conseguenza i thread di un warp, letteralmente, non potevano
trovarsi in due punti diversi del programma. Da Volta (2017) ogni thread ha
program counter e stack di chiamata propri (*independent thread scheduling*), e
uno *schedule optimizer* raggruppa per l'emissione i thread che eseguono la
stessa istruzione.

L'esecuzione resta SIMT e la divergenza pesa ancora, perché rami diversi non
possono essere emessi nello stesso ciclo. Ciò che cambia è che thread
divergenti dello stesso warp possono ora sincronizzarsi e scambiarsi dati (ed è
possibile scrivere lock e schemi produttore-consumatore dentro un warp). Il
rovescio della medaglia riguarda chi scrive kernel: i vecchi kernel
«warp-sincroni», che davano per scontato l'avanzamento in blocco senza
sincronizzarsi esplicitamente, non sono più corretti, e servono le primitive
come `__syncwarp()`.

La regola pratica, invece, non cambia: i rami condizionali vanno tenuti
allineati alla granularità del warp, così che i 32 thread restino il più
possibile *coerenti* e nessuno sprechi tempo restando mascherato.

Warp, SM, shared memory e tensor core sono nomi di NVIDIA. Sulle GPU AMD lo SM
si chiama *compute unit* (CU), il warp *wavefront* (64 thread sulle schede da
calcolo della famiglia CDNA, 32 o 64 su quelle RDNA), la shared memory *local
data share* (LDS), e i tensor core *matrix core*, le unità MFMA; Triton, il
linguaggio della {doc}`sezione sui kernel <kernel-e-cuda>`, compila per
entrambe.

`````

La {numref}`fig-plotone-si-divide` mostra il costo di quel bivio
contando le caselle.

```{figure} ../figures/plotone-si-divide.svg
:name: fig-plotone-si-divide
:alt: "Due blocchi di caselle, una casella per ciascuno dei trentadue thread di un warp e una riga per ogni passo di esecuzione. Nel blocco di sopra tutti prendono la stessa strada: tre righe piene, trentadue caselle accese ciascuna. In quello di sotto il warp si divide a un bivio: sei righe invece di tre, e in ognuna solo sedici caselle sono accese mentre le altre sedici restano vuote, perché prima si esegue un ramo e poi l'altro. Le caselle accese sono novantasei in tutti e due i blocchi: stesso lavoro, tempo doppio."
:width: 100%

Una casella per thread, una riga per passo. Il lavoro utile è lo stesso nei
due casi, novantasei caselle accese, ma sotto ci vogliono sei passi invece di
tre, perché a ogni passo metà del warp sta ferma. È tutto qui il costo di
un ramo condizionale che divide i thread di un warp: non si fa più lavoro, si
occupa più tempo per farne altrettanto.
```

## Nascondere la latenza: l'occupancy

Ogni thread, prima o poi, chiede un dato alla memoria e deve aspettare per
centinaia di **cicli**, un'eternità per un processore. Un ciclo è un battito
del clock, e una GPU ne fa più di un miliardo al secondo. Centinaia di cicli,
quindi, sono meno di un milionesimo di secondo, ma per la GPU è come se noi,
che di calcoli ne facciamo uno al secondo, restassimo fermi otto minuti davanti
a una porta chiusa. Se la GPU si fermasse a ogni attesa, tutta la sua potenza
sarebbe sprecata; evita lo spreco tenendo sempre qualcos'altro da fare.

`````{tab} Elementare

Dieci pentole sui fornelli, e un capocuoco solo a girarci intorno, che a ogni
momento sceglie a quale pentola dare un giro. Il capocuoco è il warp scheduler
dello SM, l'officina in cui lavora; ogni pentola è un warp.

La pasta della prima deve bollire dieci minuti, e chi ne avesse una sola se ne
starebbe lì a fissare l'acqua. Il nostro invece, mentre la prima bolle, mescola
la seconda, assaggia la terza, impiatta la quarta. Quando torna alla prima, i
dieci minuti sono passati «gratis», coperti dal lavoro sulle altre. L'attesa c'è
stata tutta, ma sul totale della serata non si vede. Bastano però due pentole
sul fuoco invece di dieci, e il capocuoco torna a fissare l'acqua.

Quante pentole ha sul fuoco un'officina, in rapporto a quante potrebbe averne,
si chiama **occupancy**, che in italiano suonerebbe «riempimento».

Girare da una pentola all'altra è gratis, per una ragione precisa. Ogni
pentola ha il suo tagliere fuori sul ripiano, col coltello e gli ingredienti
già pronti, e nessuno li toglie mai, così lui si sposta e trova tutto dov'era.
Dove invece si sparecchia a ogni cambio, mettere via e rimettere fuori dura
più della mescolata. Quel ripiano è il register file dell'officina, dove ogni
warp tiene i numeri con cui sta lavorando finché non ha finito. Ecco perché in
una GPU è grande fuori misura.

Il ripiano però è largo quel tanto, e quel tanto decide quante pentole stiano
sul fuoco. Una ricetta ingombrante, che pretende mezzo ripiano per sé, lascia
posto a due o tre pentole; una ricetta sobria ne fa stare dieci. Vale allo
stesso modo per il pezzo di tavolo comune che ogni squadra si tiene occupato.
Chi cucina se ne accorge da un sintomo strano: cambi una riga della ricetta, e
all'improvviso i fornelli si svuotano.

Più delle pentole accese, però, conta il corridoio che porta gli ingredienti
dalla dispensa. Il corridoio consegna il suo massimo solo se è sempre pieno di
roba in cammino, e a riempirlo è la roba chiesta e non ancora arrivata. Otto
pentole che chiedono un cucchiaio per volta mettono in strada otto cucchiai.
Due pentole che chiedono una cassa da quattro cucchiai l'una ne mettono in
strada otto anche loro, e il corridoio è altrettanto pieno con quattro volte
meno pentole.

Riempire i fornelli resta il modo più semplice per riempire il corridoio, e
un'officina con due pentole accese che chiedono un cucchiaio per volta spreca il
suo capocuoco; ma a chi si fa portare una cassa per volta di pentole ne bastano
poche.

`````

`````{tab} Superiore

La misura di «quanti warp l'SM tiene in volo» si chiama **occupancy**, cioè il
rapporto tra i warp attivi su un SM e il massimo che potrebbe ospitarne. Il
passaggio da un warp all'altro è a costo zero, perché a differenza della CPU la
GPU non salva e ripristina il contesto: i registri di *tutti* i warp residenti
restano allocati contemporaneamente nel register file dell'SM. Ecco perché il
register file è così grande.

Il register file, però, è una risorsa finita, e da qui nasce un compromesso.
Più registri usa ogni thread, meno thread (quindi meno warp) entrano insieme
nell'SM; lo stesso vale per la shared memory consumata da ogni blocco. Un'alta
occupancy dà allo scheduler tanti warp tra cui scegliere e nasconde bene la
latenza di memoria, mentre un'occupancy troppo bassa lascia lo scheduler a
corto di lavoro e l'SM inattivo durante le attese.

L'occupancy massima non è un fine in sé. Un kernel può rendere di più con
occupancy moderata, se ogni thread tiene in volo più richieste di memoria
indipendenti e riusa di più i dati nei registri {cite}`volkov2010better`.
Tuttavia, un'occupancy troppo bassa senza quel parallelismo dentro il thread è
quasi sempre un sintomo di potenza sprecata.

Questo «dipende» diventa una regola usabile se si guarda alla quantità giusta,
che non è il numero di warp ma il numero di **accessi in volo**. Per saturare
la banda, infatti, servono byte in viaggio pari a banda × latenza (è la legge
di Little).

Consideriamo una A100 80 GB PCIe, la cui banda è di $1{,}935$ TB/s. I
microbenchmark misurano la latenza della HBM, che NVIDIA non pubblica, in circa
466 cicli, cioè circa $330$ ns a $1{,}41$ GHz (sulla versione da 40 GB, con lo
stesso chip {cite}`luo2024hopper`). Il prodotto banda × latenza vale quindi
circa $640$ KB su tutto il chip, cioè circa 6 KB per ciascuno dei 108 SM.

Se ogni thread legge 4 byte, un warp in volo ne porta 128, e servono una
quarantina abbondante di richieste pendenti per SM, più di due terzi dei 64
warp che un SM può ospitare: occupancy alta. Se invece ogni thread
legge un `float4` (16 byte, cioè 512 byte per warp), ne bastano una dozzina, e
la banda si satura con meno di un quinto dei warp. Cambia solo la larghezza
della lettura, e le occupancy necessarie sono opposte: le richieste in volo
vanno tenute alte, e ci si arriva sia con tanti warp sia con pochi warp che
leggono largo.

`````

In sintesi, sul piano dell’*esecuzione* una GPU è un insieme di SM, ogni SM
esegue warp da 32 thread in stile SIMT, e nasconde le attese tenendo in volo
tanti warp insieme. Quelle attese, però, sono attese di dati che arrivano dalla
memoria, e questo sposta il problema: quando una GPU è lenta, spesso è perché
i dati arrivano più piano di quanto le unità di calcolo li consumino. La
{doc}`sezione sulla memoria <gerarchia-memoria>` mostra come è organizzata la
memoria e quando è lei a fissare il limite.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- CPU e GPU sono la lepre e il formicaio. La CPU ha pochi calcolatori
  velocissimi e finisce in fretta il singolo compito (bassa *latenza*, cioè
  poca attesa); la GPU ne ha migliaia lenti e smaltisce montagne di conti
  uguali (alto *throughput*, cioè tanta roba per ora).
- Una GPU è divisa in officine quasi autonome, gli *Streaming Multiprocessor*
  (SM), da qualche decina a oltre un centinaio, ognuna con le proprie unità di
  calcolo, un warp scheduler (il capocuoco che sceglie a chi dare un
  giro) e la propria memoria di lavoro.
- Un thread è un *compito*, «occupati tu di questo numero», non un pezzo di
  silicio: una GPU ne tiene in carico centinaia di migliaia, qualche decina per
  ogni unità di calcolo vera. Non sono i thread di Python, che erano una
  manciata e, per i calcoli scritti in Python, non servivano a niente; qui
  calcolare è l'unica cosa che fanno. Sono organizzati come in un censimento:
  l'operazione intera (la *griglia*), le squadre di quartiere (i *blocchi*),
  ciascuna assegnata a uno SM, e dentro ogni squadra i plotoni da 32 (i
  *warp*). Il 32 è una scelta arbitraria, ma NVIDIA non la cambia da
  vent'anni.
- Il sergente dà un ordine solo a tutto il plotone. Efficientissimo finché
  tutti fanno lo stesso passo; a un bivio («se pari a destra, se dispari a
  sinistra») il plotone si divide e le due strade si percorrono una dopo
  l'altra, nel doppio del tempo. Nel codice per GPU i «se... allora...» che
  dividono i compagni di plotone pesano.
- L'attesa non si accorcia, si nasconde: il warp scheduler manda avanti un
  altro warp mentre il primo aspetta i dati, come il capocuoco che gira fra
  dieci pentole. Quanti warp ha pronti, in rapporto a quanti potrebbe averne,
  si chiama occupancy. Tenerla decente è il modo più semplice per non lasciare
  l'officina a mani vuote; ma il corridoio
  verso la memoria dev'essere sempre pieno di roba in viaggio, e ci si arriva
  anche con pochi warp che chiedono molto per volta.
- Le attese che si nascondono così sono attese di dati che arrivano dalla
  memoria, e spesso sono loro il vero collo di bottiglia; ne parla la
  {doc}`sezione sulla memoria <gerarchia-memoria>`.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- CPU e GPU incarnano due filosofie opposte: la CPU è *latency-oriented*
  (pochi core complessi, finisce in fretta il singolo compito), la GPU è
  *throughput-oriented* (migliaia di ALU semplici, smaltisce montagne di
  calcoli identici e indipendenti).
- La GPU è un insieme di Streaming Multiprocessor (SM), da parecchie
  decine a oltre un centinaio, per un totale di migliaia o decine di
  migliaia di CUDA core e centinaia di migliaia di thread residenti. Ogni SM ha
  ALU, warp scheduler, un grande register file e la shared memory.
- Il programmatore lancia una griglia di blocchi di thread; l'hardware assegna
  ogni blocco a uno SM e lo esegue in warp da 32 thread: l'unità di
  schedulazione. Lo stesso codice scala su GPU con più o meno SM
  {cite}`nickolls2008scalable`.
- In stile SIMT i 32 thread di un warp eseguono la stessa istruzione su dati
  diversi. I rami condizionali che li mandano su strade diverse (warp
  divergence) vengono serializzati e allungano l'esecuzione. Fino a Pascal il
  warp aveva un program counter unico, mentre da Volta ogni thread ha PC e
  stack propri (*independent thread scheduling*); il costo della divergenza
  resta, ma i thread di un warp possono sincronizzarsi fra loro.
- La GPU nasconde la latenza della memoria con l’occupancy: tanti warp
  residenti, così che mentre uno aspetta un altro lavora. Il cambio di warp è a
  costo zero perché i registri restano tutti allocati. Gli accessi in volo
  (banda × latenza, la legge di Little) vanno tenuti alti, con tanti warp
  o con pochi warp che leggono largo e tengono in volo più richieste
  indipendenti; per questo l'occupancy massima non è un fine in sé
  {cite}`volkov2010better`.
- Le attese che l'occupancy nasconde sono attese di memoria: spesso il collo
  di bottiglia, e l'argomento della sezione sulla memoria.
```
`````
