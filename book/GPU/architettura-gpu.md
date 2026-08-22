# Dentro la GPU: come è fatta e come esegue

Nel 1999 NVIDIA lanciò la GeForce 256 e la vendette come «la prima GPU al
mondo»: da lì in poi **Graphics Processing Unit** è il nome con cui chiamiamo
questi chip. Il compito era disegnare i mondi dei videogiochi: milioni di punti
da spostare e milioni di pixel da colorare, sessanta volte al secondo. Ed è un
lavoro fatto di un solo gesto ripetuto all'infinito. Se il personaggio gira la
testa, ogni singolo punto della sua sagoma va spostato dove la rotazione lo
manda: lo stesso conto, per centinaia di migliaia di punti, uno indipendente
dall'altro. Nessuno di quei conti è difficile; sono soltanto tantissimi. Per
farli in fretta non serve un cervello raffinato che ragiona su un problema alla
volta: serve una folla che lavora tutta insieme.

Da qui la scommessa costruttiva opposta a quella delle CPU. Una CPU è fatta per
finire in fretta *un* programma, cioè per correre lungo un'unica fila di
istruzioni, e per questo i suoi core sono pochi e complicati. I progettisti
delle GPU ne misero migliaia, ciascuno lento e limitato, tutti attivi nello
stesso istante. Per anni fu una scommessa confinata alla grafica; poi CUDA
aprì quei chip a conti di ogni tipo, AlexNet vinse ImageNet nel 2012 su due
schede da videogiocatori, e le due storie non si sono più separate: è il
racconto con cui si è aperto il capitolo.

Nella sezione «Prestazioni e scala» del {doc}`capitolo su PyTorch </PyTorch/overview>` abbiamo già
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

Una lepre e un formicaio devono consegnare dei pacchi. La lepre è velocissima:
prende un pacco, sfreccia, lo consegna, torna indietro, ne prende un altro. Se
hai *un* pacco urgente, la lepre è imbattibile. Ma se ne hai diecimila, quella
corsa avanti e indietro non basta più. Il formicaio funziona all'opposto: ogni
formica è lenta, ma sono migliaia e partono tutte insieme. Il primo pacco
arriverà un po’ più tardi che con la lepre (nessuna formica è veloce) ma nello
stesso tempo ne arrivano diecimila. La lepre ha la **latenza** più bassa (il
singolo pacco arriva prestissimo), il formicaio il **throughput** più alto
(nella giornata ne arrivano molti di più).

Perché allora non si tengono diecimila lepri? Perché una lepre costa. Le
servono le gambe lunghe, la memoria di tutte le scorciatoie, una borsa di
attrezzi per quando la strada è chiusa: roba da portarsi dietro e da nutrire.
Una formica non ha niente di tutto questo, e proprio per questo nello stesso
formicaio ce ne stanno migliaia. Lo spazio è quello, e si spende in un modo o
nell'altro: o poche lepri attrezzate, o una folla di operaie spoglie che sanno
fare una cosa sola.

C'è un secondo guadagno, e si vede quando qualcosa va storto. Una formica che
trova una pozzanghera si ferma ad aspettare che scoli, ma nessuno la aspetta:
le altre le passano avanti, la fila continua, e a sera i pacchi consegnati sono
più o meno gli stessi. Il tempo perso da quella formica c'è tutto, ma nel conto
della giornata non si vede. La lepre davanti alla stessa pozzanghera tira
fuori i suoi attrezzi e prova a passare lo stesso; quando non le riesce, si
ferma la consegna, perché di lepri ce n'è una.

La CPU è la lepre: pochi processori potentissimi, pensati per finire in fretta
il singolo compito. La GPU è il formicaio: tante unità lente, pensate per
smaltire una montagna di compiti tutti insieme. Per aprire un file o rispondere
a un clic vuoi la lepre; per fare i sessantaquattro milioni di moltiplicazioni
tutte uguali di un solo strato di una rete, vuoi il formicaio.

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
«Prestazioni e scala», il prodotto di due matrici $(M,K)$ e $(K,N)$ costa
circa $2MNK$ operazioni tutte indipendenti.

`````

## Lo Streaming Multiprocessor: la GPU come federazione di officine

Vista da lontano, una GPU sembra un blocco unico. Da vicino è un insieme di
unità quasi autonome, gli **Streaming Multiprocessor** (SM): sono le officine in
cui il lavoro viene davvero eseguito. Una GPU moderna ne ha da qualche decina a
oltre un centinaio, e la
sua potenza cresce, prima di tutto, moltiplicando gli SM.

Prima di aprirne una, però, va tolto di mezzo un equivoco che rovina tutto il
resto del capitolo se resta in piedi. Nelle pagine che seguono si parla di
«lavoratori» a migliaia, e viene naturale immaginarli come pezzi di ferro. Non
lo sono. I pezzi di ferro che fanno i conti si chiamano **ALU** (le unità
aritmetiche, quelle che eseguono materialmente una moltiplicazione o una somma)
e sono in numero fisso, stampati nel silicio; il lavoratore è invece un
**thread**, cioè
un *compito*: «occupati tu del numero in posizione 4173». La parola inglese
vuol dire «filo», ed è un filo di lavoro da sbrogliare, non qualcosa che si
possa toccare. Questo spiega il numero che altrimenti non tornerebbe: le ALU di
una GPU sono migliaia, i thread che ha in carico sono centinaia di migliaia. Ce
ne sono molti più che postazioni, ed è proprio da lì che verrà, in fondo a
questa sezione, il trucco che tiene la macchina sempre occupata.

Fatta questa premessa, ogni SM è una piccola macchina completa, con i suoi
calcolatori, il suo caposquadra e i suoi ripiani di lavoro. Contiene:

- molte **ALU**, le unità aritmetiche che fanno materialmente i conti (una
  moltiplicazione, una somma): NVIDIA le chiama «CUDA core», anche se non sono
  calcolatori completi come i core di una CPU, ma proprio soltanto le
  postazioni dove il conto avviene. Sulle schede recenti accanto a esse ci sono
  anche i **tensor core**, unità costruite apposta per moltiplicare fra loro
  due tabelloni di numeri, che vedremo in una sezione dedicata;
- uno o più **warp scheduler**, i caposquadra: decidono, momento per momento,
  quale gruppetto di thread far avanzare (il gruppetto si chiama *warp*, e la
  sezione qui sotto dice perché);
- un grande **register file**, il taccuino: la memoria velocissima dove ogni
  thread tiene i numeri su cui sta operando in questo istante. In inglese
  *file* qui non vuol dire documento, vuol dire schedario;
- un blocco di **shared memory**, il tavolo comune: una memoria di lavoro
  condivisa fra i thread della stessa squadra. La incontreremo in dettaglio
  nella prossima sezione, perché è la chiave delle prestazioni.

Gli ordini di grandezza aiutano a fissare le proporzioni, senza inseguire il
numero esatto di un modello specifico (che cambia a ogni generazione). Da
parecchie decine a oltre un centinaio di **SM** per GPU; dentro ogni SM un
centinaio di **ALU**, per un totale di migliaia o decine di migliaia sul chip;
e sopra tutte queste postazioni, **centinaia di migliaia di thread** che la
GPU tiene in carico contemporaneamente, cioè qualche decina di compiti per ogni
postazione.

Da lì si ricavano i numeri che si leggono sulle schede tecniche, e il conto è
di quelli che si fanno a mente: diecimila postazioni, ciascuna un conto per
battito, e un metronomo che batte più di un miliardo di volte al secondo, fanno
più di diecimila miliardi di conti al secondo. È il motivo per cui una GPU
divora le moltiplicazioni fra tabelloni di numeri di cui una rete neurale è
fatta.

Quei conti hanno un nome che ricorrerà per tutto il capitolo. Sono
**operazioni in virgola mobile**, cioè conti sui numeri con la virgola, che
sono quelli di cui una rete neurale è fatta; l'inglese le chiama *floating-point
operations* e da lì viene la sigla **FLOP**. Un FLOP è *un* conto elementare,
una moltiplicazione o una somma: quando serve dire quanti se ne fanno al
secondo si scrive FLOP/s, e le due cose non vanno confuse, come non si
confondono i chilometri con i chilometri all'ora.

## La gerarchia dei thread: griglia, blocchi, warp

Con migliaia di unità di calcolo, la vera domanda diventa organizzativa: come
si dice a decine di migliaia di lavoratori *chi fa cosa*, senza scrivere
decine di migliaia di istruzioni diverse? La risposta di CUDA
{cite}`nickolls2008scalable` è organizzarli su tre livelli, come si organizza
un'operazione che coinvolge molta gente: l'operazione intera, le squadre in cui
è divisa, i singoli. I loro nomi tecnici sono **griglia**, **blocco** e
**thread**, e la {numref}`fig-gpu-esecuzione` li mette in fila.

`````{tab} Elementare

In una città si fa il censimento, e bisogna bussare a tutte le porte. Il
**thread** è il singolo rilevatore, che si occupa di *una* casa. Per non
impazzire, i rilevatori si organizzano in **squadre** (i «blocchi»): quelli di
una squadra lavorano nello stesso quartiere, si passano informazioni e si
coordinano tra loro. Tutte le squadre insieme formano l’**operazione
cittadina** (la «griglia»), che copre l'intera città. Il capo del censimento
non dà ordini a ogni singolo rilevatore: dice «voglio una griglia di 100
squadre da 256 rilevatori l'una», e lascia che l'organizzazione si dispieghi da
sola.

Dove appoggiarsi, però, le squadre non lo scelgono. C'è chi le assegna a un
ufficio di zona, e la squadra assegnata a un ufficio resta lì fino a
rilevazione finita: è lì che tiene le sue carte e si ritrova a confrontarle.
Gli uffici aperti sono quelli che sono, e in ognuno ci sta qualche squadra per
volta, non di più: se gli uffici sono venti e le squadre cento, parte chi ci sta
e le altre aspettano che si liberi posto. Il capo non ha bisogno di saperlo: lo
stesso ordine («100 squadre da 256») funziona in un paese con due uffici e in
una metropoli che ne ha centoventi, e a cambiare è quanto ci si mette, non il
piano. Sulle organizzazioni più recenti
c'è mezzo gradino in più: due o tre squadre finite in uffici confinanti possono
guardare le carte l'una dell'altra, cosa che con un ufficio dall'altra parte
della città non si può fare.

C'è poi un dettaglio che viene dall'hardware: dentro ogni squadra i
rilevatori marciano in **plotoni da 32**, che ricevono l'ordine tutti nello
stesso istante. Una squadra da 256 rilevatori, quindi, sono otto plotoni, e i
conti tornano sempre così: le squadre si scelgono di una taglia che sia un
multiplo di 32, altrimenti l'ultimo plotone parte mezzo vuoto. Ricordati quel
32, perché è il battito del cuore della GPU.

`````

`````{tab} Superiore

Il modello di programmazione CUDA espone tre livelli:

- **thread**: l'unità elementare, esegue il *kernel* (il programma che la GPU
  manda in esecuzione) su un proprio pezzo di dato;
- **block** (o *CTA*, Cooperative Thread Array): un gruppo di thread che
  condividono la shared memory dell'SM e possono sincronizzarsi tra loro;
- **grid**: l'insieme di tutti i blocchi lanciati per un kernel.

Dalle GPU **Hopper** in poi (compute capability 9.0) fra griglia e blocco c'è
un quarto livello, facoltativo: il **thread block cluster**, un gruppetto di
blocchi che l'hardware garantisce residenti su SM vicini e che possono leggere
e scrivere la shared memory l'uno dell'altro (*distributed shared memory*). È
il livello su cui poggiano le tecniche di FlashAttention di oggi.

Il programmatore sceglie forma e dimensione di griglia e blocchi al momento del
lancio; l'hardware assegna ciascun blocco a uno SM e lo tiene lì fino alla fine.
Un SM può ospitare più blocchi in parallelo, se le risorse (registri, shared
memory) bastano; i blocchi che non entrano restano in coda e partono quando un
SM si libera. Questo rende un programma CUDA **scalabile in modo trasparente**:
lo stesso codice gira su una GPU con 20 SM o con 120, distribuendo gli stessi
blocchi su più o meno officine, senza cambiare una riga.

Sotto il blocco c'è un livello ulteriore, che il programmatore non specifica ma
non può ignorare: l'hardware esegue i thread di un blocco in **warp** da 32. Un
blocco da 256 thread è, fisicamente, 8 warp. Il warp è l'unità di
*schedulazione*: il warp scheduler non muove un thread alla volta, muove un
warp intero.

`````

```{figure} ../figures/gpu-gerarchia-esecuzione.svg
:name: fig-gpu-esecuzione
:alt: "In alto la scomposizione logica da sinistra a destra; una griglia (grid) di blocchi, un blocco (CTA) fatto di più warp, un warp di 32 thread, il singolo thread che elabora un dato. In basso l'hardware: una GPU come insieme di Streaming Multiprocessor; una freccia tratteggiata collega un blocco a uno SM, a indicare che l'hardware assegna ogni blocco a uno SM che lo esegue a warp di 32 thread."
:width: 100%

Gli stessi tre livelli in figura, dall'operazione intera al singolo. In alto
come li pensa chi scrive il programma: una **griglia** di **blocchi**, ogni
blocco fatto di **warp** da 32, ogni warp fatto di 32 **thread**, ognuno su un
proprio dato. In basso come li esegue la macchina: ogni blocco finisce su una
delle officine (gli Streaming Multiprocessor) e lì avanza un warp per volta.
```

## SIMT: stessa mossa, dati diversi

Perché i thread avanzano a gruppetti, e non ciascuno per conto proprio? Perché
dentro un chip non c'è solo la parte che *calcola*: ce n'è un'altra, altrettanto
ingombrante, che a ogni passo va a prendere l'istruzione seguente, la decifra e
dice alle unità di calcolo cosa devono fare. Chiamiamola l'apparato di comando.
Il silicio è una superficie limitata, e ogni millimetro speso a comandare è un
millimetro non speso a calcolare: far condividere quell'apparato a 32
lavoratori, invece di darne uno a testa, libera spazio per altre unità di
calcolo, e più unità di calcolo vuol dire più conti al secondo a parità di
chip. Tutti e 32 ricevono così la stessa istruzione nello stesso momento e la
eseguono insieme, ognuno sul proprio dato. NVIDIA chiama questo modello
**SIMT**: *Single Instruction, Multiple Threads*, una istruzione sola per molti
thread.

Il gruppetto ha un nome, **warp**, ed è quello che nel resto del capitolo
chiameremo il plotone da 32. Sul perché siano proprio 32 la risposta onesta è
che non c'è una ragione profonda: è la scelta di chi ha progettato queste GPU,
e AMD sulle proprie ne ha usati a lungo 64. Ma è una scelta che NVIDIA non
cambia da vent'anni, e su cui è tarato il codice di mezzo mondo: conviene
trattarla come una costante dell'hardware, ed è per questo che il 32 va
ricordato anche se è arbitrario.

`````{tab} Elementare

Un ordine, trentadue esecuzioni. Il sergente grida «fai un passo avanti!» e
tutti e trentadue lo eseguono insieme, ciascuno sul proprio pezzo di strada. È
efficientissimo, finché tutti devono fare la stessa cosa.

Il guaio nasce a un bivio. In ogni programma esistono istruzioni della forma
«*se* è vero questo fai una cosa, *altrimenti* fanne un'altra»: sono quelle che
fanno prendere al programma strade diverse a seconda dei dati, e senza di esse
non saprebbe fare niente di interessante. Arriva allora l'ordine «se il tuo
numero è pari vai a destra, se è dispari vai a sinistra». Il sergente ne può
gridare uno per volta, e ogni ordine vale solo per quelli che nomina: fa marciare a
destra i pari mentre i dispari stanno fermi ad aspettare; poi fa marciare a
sinistra i dispari mentre i pari aspettano. I due gruppi hanno percorso strade
diverse, ma in fila invece che insieme, impiegando il doppio del tempo. E se il
bivio fosse così fine da mandare ognuno dei trentadue da una parte sua, si
andrebbe avanti uno alla volta: trentadue turni per un passo solo.

Ognuno intanto tiene il segno di dove è arrivato lungo la propria strada, e due
che si trovano in punti diversi possono anche scambiarsi una parola e
aspettarsi a un incrocio. Quello che nessuno può fare è marciare su due ordini
diversi nello stesso istante: gli ordini escono uno per volta, e chi non è
nominato sta fermo.

Morale: sulla GPU i bivi in cui i 32 compagni di plotone prendono strade
diverse costano cari, e il codice più veloce è quello in cui tutti fanno la
stessa mossa.

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
sono meno di un milionesimo di secondo, ma per la GPU è come se noi, che di
conti ne facciamo uno al secondo, restassimo fermi otto minuti davanti a una
porta chiusa. Se si fermasse a ogni attesa, tutta
la sua potenza sarebbe sprecata. La mossa che la salva non è aspettare meno, ma
**avere sempre qualcos'altro da fare**.

`````{tab} Elementare

Dieci pentole sui fornelli, e un cuoco solo a girarci intorno. Quel cuoco è il
caposquadra di una sola officina, e ogni pentola è un plotone da 32.

La pasta della prima deve bollire dieci minuti, e un cuoco con una pentola sola
se ne starebbe lì a fissare l'acqua. Il nostro invece, mentre la prima bolle,
mescola la seconda, assaggia la terza, impiatta la quarta. Quando torna alla
prima, i dieci minuti sono passati «gratis», coperti dal lavoro sulle altre.
L'attesa c'è stata tutta, e nel conto della serata non si vede. Ma bastano due
pentole sul fuoco invece di dieci, e il cuoco torna a fissare l'acqua.

Quante pentole ha sul fuoco un'officina, in rapporto a quante potrebbe averne,
si chiama **occupancy**, che in italiano suonerebbe «riempimento».

Girare da una pentola all'altra non costa niente, e c'è una ragione. Ogni
pentola ha il suo tagliere fuori sul ripiano, col coltello e gli ingredienti
già pronti, e il cuoco non li toglie mai, così si sposta e trova tutto dov'era.
Dove invece si sparecchia a ogni cambio, mettere via e rimettere fuori costa
più della mescolata. Quel ripiano è il taccuino dell'officina, dove ogni
plotone tiene i numeri con cui sta lavorando finché non ha finito. Ecco perché
in una GPU è grande fuori misura.

Il ripiano però è largo quel tanto, ed è lui a decidere quante pentole stiano
sul fuoco. Una ricetta ingombrante, che pretende mezzo ripiano per sé, lascia
posto a due o tre pentole; una ricetta sobria ne fa stare dieci. Vale allo
stesso modo per il pezzo di tavolo comune che ogni squadra si tiene occupato.
Chi cucina se ne accorge da un sintomo strano: cambi una riga della ricetta, e
all'improvviso i fornelli si svuotano.

Il conto vero non si fa ai fornelli. Si fa nel corridoio fra la cucina e la
dispensa, dove si forma la coda, e la roba in viaggio si conta. Dieci pentole
che chiedono un cucchiaio per volta mettono in strada dieci cucchiai. Due
pentole che chiedono una cassa da cinque cucchiai l'una ne mettono in strada
dieci anche loro, e il corridoio è pieno uguale con cinque volte meno pentole.
Riempire i fornelli resta il modo più semplice per riempire il corridoio, e
un'officina con due pentole accese spreca il suo cuoco; ma a chi si fa portare
una cassa per volta, o rimacina quello che ha già sul tagliere, di pentole ne
bastano poche.

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

Abbiamo così il quadro dell’*esecuzione*: una GPU è una federazione di SM,
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
- I lavoratori si chiamano **thread** e non sono pezzi di ferro: un thread è un
  *compito*, «occupati tu di questo numero», e ce n'è qualche decina per ogni
  postazione di lavoro vera. Sono organizzati come in un censimento:
  l'operazione intera (la *griglia*), le squadre di quartiere (i *blocchi*) e,
  dentro ogni squadra, i **plotoni da 32** (i *warp*). Il 32 è arbitrario ma
  non cambia da vent'anni: se lo ricordi, ricordi metà del capitolo.
- Il sergente dà **un ordine solo** a tutto il plotone. Efficientissimo finché
  tutti fanno la stessa mossa; a un bivio («se pari a destra, se dispari a
  sinistra») il plotone si divide e le due strade si percorrono una dopo
  l'altra, nel doppio del tempo. Nel codice per GPU i «se... allora...» che
  dividono i compagni di plotone costano cari.
- L'attesa non si accorcia, si **nasconde**: come il cuoco con dieci pentole,
  il caposquadra manda avanti un altro plotone mentre il primo aspetta i dati.
  Quanti plotoni ha pronti, in rapporto a quanti potrebbe averne, si chiama
  **occupancy**. Tenerla decente è il modo più semplice per non lasciare
  l'officina a mani vuote; quello che conta davvero, però, è che il corridoio
  verso la memoria sia sempre pieno di roba in viaggio, e ci si arriva anche
  con pochi plotoni che chiedono molto per volta.
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
- La GPU nasconde la latenza della memoria con l’**occupancy**: tanti warp
  residenti, così che mentre uno aspetta un altro lavora. Il cambio di warp è a
  costo zero perché i registri restano tutti allocati. Ciò che va davvero
  tenuto alto sono gli **accessi in volo** (banda × latenza): tanti warp, o
  pochi warp che leggono largo.
- Le attese che l'occupancy nasconde sono attese di **memoria**: il collo di
  bottiglia più frequente, e l'argomento della sezione successiva.
```
`````
