# Oltre una GPU: parallelismo distribuito

I modelli di cui leggiamo i nomi ogni settimana non nascono su una scheda: ne
servono centinaia, a volte migliaia, tutte sullo stesso compito. Il modo in cui
quelle schede si spartiscono il lavoro è la cosa che spiega meglio *come*
l'intelligenza artificiale di oggi viene costruita, e si racconta con quattro
immagini: un tavolo di persone che si passano i conti, un registro strappato a
metà, una catena di montaggio e un manuale diviso in fascicoli.

Prima però conviene vedere *perché* una scheda non basta, e vederlo con un
conto vero invece che a parole.

Un modello come GPT-3 ha 175 miliardi di parametri: sono i numeri che la rete
impara, e si chiamano anche pesi (le due parole si usano l'una per l'altra, e
faremo lo stesso). Scritti in *mezza precisione*, il formato corto, ciascuno
occupa 2 byte invece di 4. In tutto $175 \cdot 10^9 \times 2 = 350$ miliardi di
byte, cioè 350 GB di soli pesi.

Ma addestrare è un'altra faccenda, perché durante l'addestramento in memoria
non ci sono solo i pesi. C'è anche la contabilità di chi guida l'apprendimento,
cioè di chi a ogni passo decide di quanto spostare ciascun peso: quel «chi» si
chiama ottimizzatore, e il più usato, **Adam** {cite}`kingma2015adam`, non
guarda soltanto la correzione di adesso, si tiene anche un po’ di memoria di
come quel peso si è mosso di recente, come racconta per esteso la sezione sugli
{doc}`optimizer moderni </DeepLearning/ottimizzazione-regolarizzazione>`. Quella
memoria va conservata numero per numero, per tutta la durata
dell'addestramento. L'inventario che segue è quello dell'addestramento a
precisione mista nella forma in cui lo fanno le librerie di scala: del modello
si tengono due copie, una corta per calcolare e una lunga per aggiornare.
Parametro per parametro, è questo:

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

Sommando i quattro punti, con le due statistiche che entrano una per una:
$2 + 2 + 4 + 4 + 4 = 16$ byte per parametro, dei quali i 350 GB di prima sono
soltanto il primo addendo. Per 175 miliardi di parametri fanno
$175 \cdot 10^9 \times 16 = 2800$ miliardi di byte, cioè 2,8 terabyte.
Tutto questo insieme (pesi, correzioni, copia precisa e statistiche) ha un nome
che tornerà spesso: è lo **stato dell'addestramento**, e deve stare in memoria
dal primo esempio all'ultimo. Una GPU da datacenter di fascia alta, una
H100, ha 80 GB di memoria: $2800 / 80 = 35$, ci vorrebbero trentacinque schede
soltanto per *contenerlo*, prima ancora di parlare di velocità. Alcuni modelli
non stanno in una GPU sola, né per memoria, né per tempo.

La sezione {doc}`«Prestazioni e scala» </PyTorch/prestazioni>` ha già visto la
strategia più comune per usare più schede: il **parallelismo dati**, che in
PyTorch si accende con una riga (`DistributedDataParallel`) e che lì era
raccontato con l'analogia degli insegnanti che si spartiscono i compiti da
correggere e poi mediano le correzioni. Qui la riprendiamo in due righe, ne
mettiamo a fuoco il limite, e poi andiamo *oltre*: una mappa dei modi di
dividere il lavoro quando un modello è troppo grande perché una GPU basti a
sé.

Due parole da fissare subito, perché da qui in avanti tornano di continuo.
Un **nodo** è un singolo computer, con dentro le sue schede, di solito quattro
o otto, collegate fra loro da connessioni interne molto veloci. Un **cluster**
è un insieme di nodi collegati in rete, che lavorano allo stesso compito: fra
due schede dello stesso nodo i dati volano, fra due nodi diversi devono passare
per la rete, che è molto più lenta. Quasi tutte le scelte che seguono nascono
da questa differenza. (Attenzione a una parola che qui fa due
mestieri: quando si dice «rete» in questo senso si intendono i cavi che
collegano i computer, non la rete neurale. Per quest'ultima, qui, diremo sempre
«il modello».)

## Ripasso: il parallelismo dati e il suo limite

Il parallelismo dati non divide il modello: divide i *dati*. Ogni GPU riceve
una copia identica del modello e una fetta diversa del mini-batch (il
mazzetto di esempi che si guardano in una volta sola prima di aggiornare i
pesi), calcola le proprie correzioni, e alla fine tutte si mettono d'accordo
facendone la media. L'operazione con cui si mettono d'accordo si chiama
all-reduce: ogni scheda mette dentro i propri numeri, si sommano, e alla
fine tutte quante hanno in mano lo stesso risultato. Dopo la media, le
repliche applicano lo stesso aggiornamento e restano perfettamente uguali. È
il primo dei tre pannelli in {numref}`fig-parallelismo-strategie`, da guardare
adesso solo nella sua prima colonna: le altre due sono le strategie delle
sezioni che seguono, e le si capisce meglio quando ci si arriva.

```{figure} ../figures/parallelismo-strategie.svg
:name: fig-parallelismo-strategie
:alt: "Tre pannelli affiancati. DATI: due GPU con lo stesso modello replicato ma fette di dati diverse (dati A e dati B), i gradienti mediati con un all-reduce. TENSOR: una singola matrice di pesi W tagliata a metà, colonna sinistra su GPU 0 e destra su GPU 1, i due risultati ricomposti con un all-gather. PIPELINE: GPU 0 tiene gli strati 1-2 e GPU 1 gli strati 3-4, i micro-batch scorrono come su una catena di montaggio."
:width: 100%

Tre modi di dividere il lavoro su più GPU, uno per sezione. Dati: si
replica il modello e si spartiscono gli esempi (è il pannello di questa
sezione). Tensor: si taglia in due un singolo tabellone di numeri del
modello. Pipeline: si mettono strati diversi su GPU diverse.
```

Il punto delicato è come avviene quella media senza intasare la rete. Il modo
ingenuo (tutte le GPU spediscono i gradienti a una sola, che somma e
rispedisce) trasforma quella GPU in un imbuto. La soluzione elegante ha un
nome preciso, e {numref}`fig-anello-somma` ne segue i due giri.

```{figure} ../figures/anello-somma.svg
:name: fig-anello-somma
:alt: "Quattro schede disposte su un anello tratteggiato con le frecce che dicono il verso, ciascuna con la propria lista tagliata in quattro celle. Nel primo giro un pezzo per volta passa al vicino di destra e si somma al suo, e le celle si scuriscono via via che raccolgono i contributi; dopo tre passaggi ogni scheda ha una sola cella piena, e sono quattro celle diverse. Nel secondo giro quelle celle piene fanno il giro com'erano, e dopo altri tre passaggi tutte le celle di tutte le schede sono piene. In basso il conto: ogni scheda spedisce sei pezzi su quattro, cioè una volta e mezza la propria lista, e il conto non cambia se le schede sono di più."
:width: 88%

Quattro schede, ognuna con la propria lista tagliata in quattro pezzi. Al primo
giro ogni pezzo passa al vicino di destra e si somma al suo, finché ciascuna
scheda ne ha uno finito; al secondo, quei pezzi finiti fanno il giro com'erano,
e alla fine tutte hanno tutto. Nessuna ha spedito più di una volta e mezza la
propria lista, e nessuna ha fatto da imbuto.
```

`````{tab} Elementare

Otto persone attorno a un tavolo, una per scheda, ognuna con la propria lista
di numeri: sono le correzioni che ha ricavato dalla sua fetta di esempi, cioè
i gradienti. Alla fine il totale deve stare in mano a tutti.

Il modo ovvio è dettare le liste a una persona sola, che somma e ridetta il
risultato agli altri. Quella persona diventa un imbuto. Più gente siede al
tavolo, più roba le arriva addosso, e gli altri aspettano il proprio turno di
parola.

L'altra via non ha nessuno al centro. Le persone si dispongono in cerchio e
ciascuna parla solo col vicino di destra. Gli passa un pezzetto di somma,
riceve un pezzetto da sinistra, lo somma al proprio e lo manda avanti. Chiuso
il giro, ognuno ha finito un pezzo solo del totale, e allora se ne fa un
secondo, che gira allo stesso modo ma non somma più niente: i pezzi già pronti
passano di mano finché tutti li hanno tutti. Due giri, non uno, e in ciascuno
ogni passaggio è roba spedita sul filo.

Nessuno però è mai sovraccarico, e il conto si fa a mente riducendo il tavolo a
quattro. Ciascuno taglia la propria lista in quattro pezzetti e nei due giri ne
spedisce tre più tre (tre e non quattro: il pezzetto che alla fine resta suo
non lo spedisce a nessuno), cioè sei quarti, una volta e mezza la lista intera.
In otto, sette più sette, 1,75 volte. In quaranta, trentanove più trentanove,
1,95 volte. Raddoppiando il tavolo i pezzetti si dimezzano mentre i passaggi
raddoppiano, e le due cose si annullano. Chiunque sieda al tavolo manda in giro
poco meno di due volte la propria lista, mai di più. I pezzetti poi partono
appena sono pronti, mentre si sta ancora sommando il resto, e il tempo del giro
si nasconde dentro il tempo del conto.

A pagare è il giro in sé. In quaranta il foglio cambia di mano quasi ottanta
volte, e prima di ogni consegna c'è l'attimo in cui uno alza la testa e cerca
il vicino. Al tavolo da otto quell'attimo non si nota. In quaranta pesa più
della roba spedita, e il cerchio si scioglie. Le persone si dispongono ad
albero, ognuna riceve da due e passa a una sola, e le mani da attraversare
diventano una manciata invece di ottanta.

Un difetto il cerchio se lo porta dietro sempre, e va al passo del più lento.
Se uno si alza a rispondere al telefono, chi viene dopo resta col foglio in
mano e dietro si ferma tutta la fila. La persona sola al centro questo difetto
non ce l'ha, purché si accontenti: rifà il totale con le liste che le sono
arrivate, chi tarda entra nel totale dopo, e intanto gli altri vanno avanti con
una somma un po’ vecchia. Con qualcuno molto più lento degli altri, o che ogni
tanto si alza, tenere una persona al centro ha ancora senso.

Questa danza si chiama *ring all-reduce*, ed è il motivo per cui il
parallelismo dati regge bene su tante schede. Il limite arriva da un'altra
parola, «copia». Oltre alla lista da sommare, ciascuno tiene davanti a sé il
manuale intero con cui lavora, cioè il modello. Finché è un opuscolo va bene.
Quando diventa un elenco telefonico da mille pagine, non c'è tasca che tenga.

`````

`````{tab} Superiore

Con $K$ repliche, la libreria di *collettive* di NVIDIA, **NCCL** (*NVIDIA
Collective Communications Library*), sceglie fra più algoritmi in base a
taglia del messaggio e topologia; quello classico è lo schema
**ring all-reduce**: le GPU formano un anello logico e ogni GPU comunica
soltanto con i due vicini, in due fasi (*reduce-scatter* e poi *all-gather*).
Il volume di dati che ciascuna GPU trasmette è $2\frac{K-1}{K}$ volte la
dimensione del gradiente: al crescere di $K$ tende a una costante, cioè è
ottimale in banda (cresce solo la latenza, non il traffico per GPU). È
proprio quella latenza, proporzionale a $K$, il motivo per cui a molti nodi
NCCL abbandona l'anello per schemi ad albero (*double binary tree*), che la
contengono senza sacrificare la banda.

L'anello ha preso il posto di un altro schema, e il confronto spiega la sua
fortuna. Quello di prima era il **parameter server**
{cite}`li2014parameterserver`: uno o
più nodi dedicati custodiscono i pesi, tutti gli altri ci mandano i gradienti e
ne rileggono i pesi aggiornati. È semplice, sopporta bene i lavoratori lenti
(in versione asincrona non si aspetta nessuno) ed è tuttora sensato quando i
nodi sono eterogenei o inaffidabili. Ma il traffico che attraversa il gruppo
dei server cresce linearmente con $K$: per reggerlo bisogna far crescere anche
quel gruppo, cioè spendere macchine per il coordinamento invece che per il
calcolo.
L'anello non ha un centro: nessun nodo vede più traffico degli altri, e il
costo per GPU smette di dipendere da quante sono. È la stessa ragione per cui,
nei sistemi distribuiti, si preferisce un protocollo fra pari a uno che passa
da un coordinatore, ogni volta che il coordinatore non serve. Come
già ricordato nella sezione «Prestazioni e scala», in
`DistributedDataParallel` questo all-reduce è eseguito *durante* il
`backward`, a pacchetti (*bucket*), così che la comunicazione si sovrapponga
al calcolo e sparisca dietro di esso.

Il limite è strutturale, non implementativo: ogni replica deve contenere
l’intero modello, più i suoi gradienti, più gli stati dell'ottimizzatore.
Se questo pacchetto non entra nella memoria di una singola GPU, il
parallelismo dati (per quanto ben implementato), non serve a nulla. Da qui le
strategie che seguono, che invece di replicare spezzano.

`````

## Spezzare la matrice: il tensor parallelism

La prima idea è tagliare il modello dove è più grosso. Un modello, dentro, è
fatto in buona parte delle stesse matrici della sezione sul GEMM (in uno
strato di Transformer sono almeno sei: le tre proiezioni dell'attenzione,
quella d'uscita e le due della rete che segue), e farne lavorare una vuol dire
moltiplicare i numeri che entrano per i numeri della matrice. Invece di tenerne
una copia intera su ogni GPU, se ne mette un pezzo su ciascuna, e ognuna
calcola la propria fetta del risultato. È il secondo pannello di
{numref}`fig-parallelismo-strategie`, ed è l'idea alla base di **Megatron-LM**
{cite}`shoeybi2019megatron`.

`````{tab} Elementare

Due contabili devono sommare le colonne di un registro gigantesco. Copiare
l'intero registro a entrambi sarebbe uno spreco: meglio strapparlo a metà per
il lungo, le prime colonne a uno e le ultime all'altro. Le ricevute da
riportare restano le stesse per tutti e due, e ognuno le ha tutte sotto gli
occhi: quello che cambia è la metà di registro in cui ciascuno le scrive.
Ciascuno somma la sua metà, in parallelo, e poi i due si scambiano i risultati
parziali per rimetterli insieme. Nessuno dei due ha mai avuto l'intero registro
in mano: sta metà in una testa e metà nell'altra. È così che si fa girare uno
strato che, intero, non entrerebbe in una scheda sola.

Il verso dello strappo non si sceglie a caso. Prima di riportare un totale alla
pagina dopo si passa sulle voci una per una e si applica a ciascuna un ritocco,
uno sconto: e lo sconto su una voce si calcola guardando quella voce e nient'altro.
È quel «nient'altro» a decidere il verso. Col
registro strappato per il lungo, il ritocco ciascuno lo fa sulle proprie voci
senza chiedere niente all'altro, e per rimettere insieme i conti i due si
fermano una volta sola per pagina. Strappato per il verso sbagliato,
dovrebbero riunire i pezzi prima dello sconto e ridividerli dopo: due soste al
posto di una, con gli stessi conti da fare.

Quella sosta, poi, non arriva una volta sola alla fine. Il totale di ogni
pagina serve per cominciare la pagina dopo, quindi i due contabili devono
fermarsi, mettere insieme i pezzi e ripartire a ogni pagina: e un modello
di pagine ne ha decine, due per strato. E nessuno dei due, nel frattempo, può
portarsi avanti con altro lavoro: quello che deve scrivere dipende proprio dal
foglio in arrivo.

Finché i due sono seduti allo stesso tavolo, passarsi un foglio non costa quasi
niente. Se sono in due edifici diversi, quel fermarsi continuo diventa la voce
di spesa principale. Per questo tagliare i tabelloni conviene solo fra schede
vicine, unite da una linea diretta velocissima.

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
$\mathbf{A}$ per colonne (la GeLU è elemento-per-elemento, quindi ogni GPU
può applicarla alla propria fetta senza consultare le altre) e $\mathbf{B}$ per
righe, così che

$$
\mathbf{Z} = \mathrm{GeLU}(\mathbf{X}\mathbf{A}_1)\,\mathbf{B}_1
           + \mathrm{GeLU}(\mathbf{X}\mathbf{A}_2)\,\mathbf{B}_2,
$$

dove $\mathbf{A}_i$ e $\mathbf{B}_i$ sono le porzioni assegnate alla GPU $i$.
La somma dei due
addendi richiede una sola collettiva (un all-reduce) in avanti e una
all'indietro, per blocco. Nell'attenzione multi-testa il taglio è ancora più
naturale: teste diverse su GPU diverse. Il costo è la comunicazione: le
collettive sulle *attivazioni* si ripetono a ogni blocco, sono sincrone e
stanno sul cammino critico (il calcolo non può proseguire finché non
finiscono, quindi non si nascondono dietro di esso, come invece fa
l'all-reduce dei gradienti). Per questo il
tensor parallelism vive di norma dentro un singolo nodo, dove le GPU sono
collegate da NVLink a centinaia di GB/s, e non tra nodi diversi.

`````

## La catena di montaggio: il pipeline parallelism

Se il tensor parallelism taglia il modello *in larghezza* (dentro ogni
strato), il **pipeline parallelism** lo taglia *in profondità*: strati diversi
su GPU diverse. GPU 0 tiene gli strati 1–8, GPU 1 i 9–16, e così via: il terzo
pannello di {numref}`fig-parallelismo-strategie`. È l'idea di **GPipe**
{cite}`huang2019gpipe`.

`````{tab} Elementare

È una catena di montaggio. La prima postazione monta il telaio, la seconda
aggiunge il motore, la terza la carrozzeria, la quarta vernicia e collauda.
C'è però un problema evidente: se in fabbrica entra una sola automobile,
mentre la prima postazione lavora le altre tre stanno con le mani in mano, e
quando l'auto arriva in fondo la prima è ferma da un pezzo. Quattro operai, ma
a ogni istante ne lavora uno. Il rimedio è non mandare un'auto sola, ma un
flusso continuo: appena la prima postazione ha finito il telaio di un'auto e
l'ha passato avanti, comincia subito quello dell'auto successiva. Presto tutte
e quattro le postazioni lavorano insieme, ciascuna su un'auto diversa. Nel
modello le «auto» sono pezzetti del mazzetto di esempi, i **micro-batch**, che
si fanno scorrere lungo gli strati.

Il tempo iniziale in cui le postazioni si riempiono, e quello finale in cui si
svuotano, è tempo sprecato, e ha un nome: all'inizio e alla fine della
lavorazione restano due sacche vuote, e quelle sacche si chiamano **bolla**.
Quanto costa la bolla si conta: con quattro postazioni e una macchina sola se
ne va in fumo il 75% del tempo, con trentadue macchine in fila si scende sotto
il 9%. Più micro-batch si mandano di seguito, più la bolla si assottiglia.

Assottigliarla costa, e costa in due modi. Le auto non si possono
rimpicciolire all'infinito: sotto una certa taglia ogni postazione passa più
tempo ad attrezzarsi che a lavorare. E ogni auto in viaggio lascia dietro di sé
appunti da conservare, perché a fine corsa la fila si ripercorre all'indietro:
è il ripasso con cui il modello impara dai propri errori, e per farlo ciascuna
postazione deve rivedere il lavoro che ha fatto all'andata. Quegli appunti
hanno un nome, le attivazioni. Tenerli tutti, per tutte le auto in viaggio,
riempirebbe il magazzino. Se ne tiene allora uno solo per auto, il foglio che
passa da una postazione all'altra; il resto si butta, e al ritorno ogni
postazione rifà i propri conti da capo per ritrovare quel che le serve. Rifare
un conto costa tempo, tenerlo da parte costa spazio: quando a mancare è lo
spazio, si sceglie di rifarlo.

Fra una postazione e l'altra, comunque, passa solo l'auto a metà montaggio, non
il magazzino dei pezzi. È poca roba, e per questo le postazioni possono anche
stare in capannoni diversi, collegati da una strada normale.

`````

`````{tab} Superiore

Gli strati sono partizionati in $p$ **stadi**, uno per GPU, disposti in
sequenza: l'output di uno stadio è l'input del successivo. Con un solo batch,
l'utilizzo è disastroso: a ogni istante lavora un solo stadio su $p$. GPipe
spezza allora ogni mini-batch in $m$ micro-batch che entrano nella pipeline
uno dopo l'altro. La frazione di tempo sprecata nel riempimento e nello
svuotamento (la bolla) vale

$$
\frac{p-1}{m+p-1},
$$

dove $p$ è il numero di stadi e $m$ il numero di micro-batch. Il conto è
questo: con $p$ stadi il lavoro utile dura $m$ tempi di stadio e la fila ne
occupa $m+p-1$, e la differenza è la bolla. Vale a stadi *bilanciati*, cioè se
ciascuno impiega lo stesso tempo, e trascurando il passaggio da uno stadio
all'altro; l'articolo infatti la scrive come ordine di grandezza, e sui propri
modelli attribuisce a uno sbilanciamento fra stadi lo scarto dalla scala
ideale. Con $p=4$ stadi e $m=1$ la bolla è i tre quarti del tempo; con $m=32$
scende sotto il 9%. L'articolo dà anche una regola pratica, ma è una misura
sperimentale e non una lettura della formula: con $m \ge 4p$, cioè con almeno
quattro micro-batch per stadio, gli autori trovano la bolla trascurabile, e
osservano che il merito è in parte del ricalcolo, che si lascia programmare in
anticipo. La formula da sola, a $m = 4p$, dà il 16% con quattro stadi e sale
verso il 20% allungando la fila, cioè il doppio abbondante di quel 9%: la
regola vale per quel che è, una misura sul loro sistema.

Il compromesso è che micro-batch più piccoli usano peggio ogni singola GPU
(meno lavoro per lancio) e che vanno conservate, per ogni micro-batch in volo,
le attivazioni ai confini fra stadi. Le attivazioni *interne* a uno stadio,
invece, GPipe non le conserva affatto: le ricalcola nel `backward`, ed è
quello che abbassa il picco di memoria da «tutte le attivazioni di tutti gli
strati» a «quelle di uno stadio solo, per un micro-batch solo». Il ricalcolo
non è però un'invenzione di GPipe, che dichiara di *supportarlo* e lo attribuisce
al lavoro sul costo di memoria sublineare del 2016: il contributo dell'articolo
sono i micro-batch, e il resto è il *gradient checkpointing* di sempre, lo
stesso baratto calcolo-per-memoria della sezione precedente su FlashAttention.
I sistemi di oggi, infine, hanno affiancato allo scheduling di GPipe quello
1F1B (un `forward` e un `backward` alternati, da PipeDream e da Megatron), che
a parità di bolla tiene in volo $p$ micro-batch invece di $m$, e quindi ne
conserva anche meno.

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
invece di replicare, si spartisse? Ogni GPU custodisce solo la propria
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

Al tavolo di prima, adesso, ognuno corregge una fetta di compiti: sono
insegnanti, e ciascuno ha lo zaino suo. Dentro ciascuno porta
tre cose: la griglia di valutazione, cioè le regole con cui si corregge; il
foglio delle correzioni appena fatte; e un quadernetto in cui tiene nota di
com'è andata ogni domanda nelle ultime settimane, che serve a decidere quanto
pesare la correzione di oggi. Nel parallelismo dati ognuno teneva in tasca una
fotocopia *completa* di tutte e tre. Comodo, ma se la griglia è un tomo di
mille pagine, tenerne una copia intera a testa è uno spreco enorme di zaini.

Si strappa, allora, e si strappa in tre tempi. Prima il quadernetto: se gli
insegnanti sono otto se ne tiene un ottavo a testa, e nessuno se ne accorge,
perché quelle pagine ognuno le guarda soltanto per le domande che gli toccano.
Poi il foglio delle correzioni, stessa storia. Questi due strappi non
aggiungono un solo viaggio fra colleghi: quello che a fine turno girava attorno
al tavolo continua a girare uguale, e intanto lo zaino si è già alleggerito.
Non c'è ragione di rinunciarci.

Il terzo strappo è quello che si paga, ed è anche quello che serve davvero: si
strappa la griglia. Otto fascicoli, uno per insegnante. Quando arriva la
domanda la cui regola sta a pagina 700, l'insegnante che non ce l'ha la chiede
al collega che tiene quel fascicolo, se la fa fotocopiare *giusto per quella
correzione*, e appena finito butta la fotocopia. Adesso i viaggi aumentano:
dove prima se ne facevano due se ne fanno tre, una volta e mezza. In cambio lo
zaino, contando anche i due strappi di prima, pesa otto volte meno di quello di
partenza. E i fascicoli vanno tagliati sottili: farsi
fotocopiare mezzo tomo per una domanda sola tornerebbe a riempire il banco, e
il guadagno sparirebbe.

Quel viaggio in più quasi non si sente, perché la richiesta si fa in
anticipo: mentre si corregge la domanda di adesso si chiede già il fascicolo
della prossima, e la fotocopia arriva mentre la penna è ancora sul foglio. Il
conto cambia se il collega sta in un altro edificio: allora chiedere in
anticipo non basta a coprire il viaggio, si finisce di correggere e si resta lì
ad aspettare.

È così che modelli enormi riescono a girare su GPU «normali»: nessuna scheda
tiene mai il modello intero, solo la sua fetta, radunando i pezzi che servono
un attimo prima di usarli e liberandoli subito dopo.

`````

`````{tab} Superiore

ZeRO elimina la ridondanza del parallelismo dati in tre stadi cumulativi:
partiziona tra le GPU prima gli stati dell'ottimizzatore (stadio 1), poi
anche i gradienti (stadio 2), infine anche i parametri (stadio 3).
FSDP arriva allo stesso risultato dello stadio 3, con un disegno rifatto per
PyTorch: gli autori scrivono di essersene fatti ispirare, non di averlo
riprodotto, e la differenza sta nelle collettive scelte e nel modo di
distribuire il lavoro. A regime, ogni GPU
detiene solo $1/K$ dei parametri di ciascuna unità del modello. Prima di
eseguire il `forward` di quell'unità, un **all-gather** ricostruisce
temporaneamente i pesi completi; subito dopo l'uso, la GPU li *ri-spartisce*
(scarta i pezzi non suoi), liberando memoria; nel `backward` la stessa cosa
avviene per i gradienti, ridistribuiti con un *reduce-scatter*.

Sul prezzo in comunicazione la sorpresa sta in cima all'elenco: i primi due
stadi sono gratis. Spartire gli stati dell'ottimizzatore e i gradienti «incurs
no additional communication» rispetto al parallelismo dati puro, scrive
l'articolo, a fronte di un risparmio di memoria fino a otto volte. Quel «fino
a» conta, perché il fattore cresce con il numero di schede e otto è il suo
limite: sulle otto GPU di questa scena i due stadi insieme dividono per poco
più di quattro. Difficile trovare un motivo per non accenderli. A pagare è solo
il terzo stadio, quello di FSDP, e paga al massimo $1{,}5\times$ la
comunicazione del parallelismo dati, a patto che le collettive usino l'anello
ottimale in banda. Il conto sta in tre passaggi contro due: l'all-reduce dei
gradienti ne vale due (un reduce-scatter e un all-gather) e qui si riduce al
solo reduce-scatter, perché a ogni GPU serve soltanto la fetta di gradiente che
le compete; in cambio i parametri vanno radunati con un all-gather nel
`forward` e con un altro nel `backward`. Anche così è quasi sempre un buon
affare, perché quella comunicazione si sovrappone al calcolo. In codice, FSDP
somiglia molto a DDP: si lancia con `torchrun` e il training loop resta
identico.

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

Nella pratica queste strategie non si scelgono a esclusione: si combinano.
Addestrare un modello di frontiera significa quasi sempre impilarne tre
insieme, e siccome sono tre tagli in tre direzioni diverse (fra gli esempi,
dentro un tabellone, lungo la fila degli strati) l'uso ha chiamato la
combinazione **3D parallelism**, come i tre assi di uno spazio.

A decidere quale strategia va dove c'è una domanda sola: quanto spesso le
schede devono fermarsi a parlarsi. Quella domanda ha senso per la stessa
ragione dell'occupancy vista all'inizio del capitolo: mandare dati e fare conti
sono due attività distinte e possono avvenire *insieme*, quindi una
comunicazione che parte mentre la scheda sta ancora calcolando non si paga
quasi. Si paga invece quella che obbliga
tutti a incrociare le braccia finché non è finita.

Da qui la distribuzione. Il tensor parallelism sta dentro ogni nodo: a ogni
strato obbliga le schede a mettere insieme un risultato e ad aspettarsi a
vicenda, e quell'attesa è di quelle che nessuno può coprire con altro lavoro,
perché il calcolo dopo dipende proprio da lei. Serve quindi il collegamento più
veloce che esista, quello interno al nodo (si chiama **NVLink**). Il pipeline
parallelism spezza gli strati fra gruppi di nodi, e si scambia poca roba. Il
parallelismo dati sta fra i nodi, dove c'è la rete lenta: gli basta una media
sola a ogni passo, cioè a ogni mazzetto di esempi.

A questi se ne aggiungono altri due, più specialistici. Il **sequence
parallelism** {cite}`korthikanti2023activation` lavora accanto al tensor
parallelism e ne tappa il buco: i pezzi di strato che il taglio delle matrici
lascia replicati su tutte le schede (le normalizzazioni, il dropout) si possono
spartire lungo il testo, un tratto per scheda, perché lì ogni posizione è
indipendente dalle altre. Serve ad alleggerire la memoria che si mangiano le
attivazioni, cioè i risultati intermedi che ogni strato produce e che vanno
conservati fino al passaggio all'indietro, quello in cui il modello impara dai
propri errori.
L’**expert parallelism** riguarda i modelli *Mixture of Experts*, quelli in cui
il modello non è uno solo ma un mazzo di modelli specializzati fra cui un
selettore smista ogni parola in arrivo: lì si mettono esperti diversi su schede
diverse.

Dietro tutta questa ingegneria c'è una tensione di fondo da nominare: il
**memory wall**, il muro della memoria. La dimensione dei modelli è cresciuta
molto più in fretta della memoria che si riesce a mettere su una singola GPU:
è per questo che *spartire* lo stato, e non solo replicarlo, è diventato
inevitabile, e che FSDP è oggi la via pratica per addestrare modelli grandi su
un numero ragionevole di schede.

Le quattro strategie (spartire gli esempi, spartire i tabelloni, spartire gli
strati, spartire lo stato) sono la mappa che spiega *come* nascono i modelli di
cui leggiamo i nomi ogni settimana, e non folklore da datacenter. E l'ultima,
FSDP, è alla portata già di due schede infilate nello stesso computer.
Perché proprio lei: il parallelismo dati su due schede è facilissimo da
accendere ma non risolve il problema del modello che non ci sta, visto che ogni
scheda ne tiene una copia intera; spezzare i tabelloni o gli strati risolve, ma
vuole che si rimetta mano a come il modello è scritto. FSDP risolve *e* costa
una riga di codice al posto di un'altra. Se un giorno vi troverete con un
modello che in una scheda sola non entra, è da lì che si comincia.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Alcuni modelli non stanno in una GPU sola: per un modello da 175 miliardi di
  parametri, tenere durante l'addestramento i pesi, le correzioni e i conti di
  servizio dell'ottimizzatore richiede qualche migliaio di gigabyte, contro le
  decine di una singola scheda. Da qui il lavoro spartito fra più schede.
- Il parallelismo dati (già visto nella sezione «Prestazioni e scala») dà a
  ogni scheda una copia del modello e una fetta diversa degli esempi, poi le
  schede mediano le correzioni parlando solo con il vicino, in due giri
  attorno al tavolo (uno per completare i pezzi, uno per distribuirli), così che
  nessuna faccia da imbuto. Il limite sta nella parola «copia»: ogni scheda deve
  tenere in tasca l'elenco telefonico intero.
- Tagliare le matrici per il lungo (in inglese *tensor parallelism*, perché
  una traduzione italiana non ha mai preso piede)
  {cite}`shoeybi2019megatron`: è il registro strappato a metà, ogni
  scheda tiene mezzo tabellone di numeri, calcola la sua parte e poi si scambia i
  risultati parziali con le altre. Le schede però devono parlarsi a ogni strato e
  aspettarsi a vicenda, quindi conviene solo fra schede vicine, unite da una
  linea velocissima.
- Mettere strati diversi su schede diverse (in inglese *pipeline
  parallelism*) {cite}`huang2019gpipe`: è la catena di montaggio, con pezzetti
  del mini-batch (i micro-batch) che scorrono in fila. Il tempo in cui le
  postazioni si riempiono e si svuotano è sprecato (si chiama bolla) e si
  assottiglia mandando più micro-batch di seguito.
- Spartire invece di fotocopiare {cite}`rajbhandari2020zero`
  {cite}`zhao2023pytorchfsdp`: il tomo si strappa in fascicoli, uno per scheda, e
  il pezzo che manca si fa passare dal collega che lo tiene giusto un attimo
  prima di usarlo, per poi buttare la fotocopia. Nessuna scheda tiene mai il
  modello intero, solo la sua fetta, ed è così che oggi si addestrano i modelli
  grandi. I due nomi che si incontrano sono sigle inglesi e dicono esattamente
  questa cosa: ZeRO sta per «ottimizzatore senza copie ripetute», FSDP per
  «parallelismo dati con tutto spartito».
- Nella realtà le strategie si combinano (dati, tensor e pipeline insieme),
  più le varianti che spartiscono la lunghezza del testo o i vari «esperti» di
  un modello. La ragione di fondo: i modelli crescono più in fretta della
  memoria che si riesce a mettere su una scheda, e allora spartire non è più
  un'opzione.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Alcuni modelli non stanno in una GPU sola: lo stato di addestramento (pesi
  + gradienti + Adam) di un modello da 175 miliardi di parametri è dell'ordine
  dei terabyte, contro le decine di GB di una GPU. Da qui il parallelismo su più
  schede.
- Il parallelismo dati (già visto nella sezione «Prestazioni e scala»)
  replica il modello e media i gradienti con un all-reduce (via NCCL; lo
  schema classico è il ring, ottimale in banda); il suo limite è che ogni
  GPU deve contenere il modello *intero*, più i suoi gradienti, più gli stati
  dell'ottimizzatore.
- Il tensor parallelism {cite}`shoeybi2019megatron` taglia le singole
  matrici di pesi tra GPU (Megatron-LM), ricomponendo con una collettiva; le
  collettive sono frequenti e sul cammino critico, quindi vive dentro un nodo
  (NVLink).
- Il pipeline parallelism {cite}`huang2019gpipe` mette strati diversi su GPU
  diverse e fa scorrere micro-batch in catena di montaggio; la bolla vale
  $\frac{p-1}{m+p-1}$ a stadi bilanciati, e la regola pratica $m \ge 4p$ è una
  misura degli autori, non una lettura della formula (che a $m = 4p$ dà il
  16%). Conserva le attivazioni
  ai *confini* fra stadi, e quelle interne le ricalcola: stesso baratto
  calcolo-per-memoria di FlashAttention e del *gradient checkpointing*.
- ZeRO/FSDP {cite}`rajbhandari2020zero` {cite}`zhao2023pytorchfsdp` non
  replicano ma spartiscono parametri, gradienti e stati dell'ottimizzatore,
  ricomponendoli al volo (all-gather) solo quando servono: la via pratica per i
  modelli grandi. I primi due stadi non costano nulla in comunicazione; solo il
  terzo, quello di FSDP, paga fino a $1{,}5\times$, con le collettive ad anello.
  In PyTorch: `FullyShardedDataParallel`.
- Nella realtà si combinano (3D parallelism: dati × tensor × pipeline),
  più sequence ed expert parallelism. Il memory wall (modelli che crescono
  più in fretta della memoria per GPU) è la ragione per cui lo sharding conta.
```
`````

Da qui in avanti la macchina non è più una scatola chiusa: sappiamo perché le
piacciono certi conti e non altri, dove il tempo se ne va per davvero, e come
un modello che non entrerebbe in nessuna scheda venga fatto stare in mille. È
il motivo per cui oggi si possono impilare decine di strati senza aspettare
mesi. Finora, però, è la macchina che si è piegata al modello. Il
{doc}`capitolo sull'efficienza </Efficienza/overview>` fa la domanda opposta:
quanto si può restringere il modello, perché di macchina ne serva meno?
