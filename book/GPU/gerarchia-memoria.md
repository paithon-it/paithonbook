# La memoria: il vero collo di bottiglia

Nella sezione «Prestazioni e scala» del capitolo su PyTorch avevamo lasciato
cadere, quasi di sfuggita, un'osservazione scomoda: «il collo di bottiglia, più
spesso del calcolo, è il movimento dei dati». È il momento di prenderla sul
serio, perché è una delle verità meno intuitive di tutto l'hardware moderno.

L'immagine che viene spontanea è quella di una GPU come un mostro di calcolo
che divora numeri. La realtà, molto più spesso, è un mostro *affamato* che
aspetta di essere imboccato: *feeding the beast*, «sfamare la bestia», è il
modo in cui gli ingegneri chiamano il problema. Le migliaia di core di cui
abbiamo parlato nell'architettura macinano in un lampo i dati che hanno già
sotto mano, poi restano fermi ad aspettare i prossimi. Abbiamo visto che i
**warp** mascherano la latenza dei singoli accessi tenendo la GPU sempre
occupata con altro lavoro pronto; ma quel trucco *nasconde* l'attesa, non
fabbrica dati più in fretta. La **banda** (quanti byte al secondo la memoria
riesce davvero a consegnare) è finita, ed è lei, non la potenza di calcolo, a
decidere il destino di moltissimi programmi. È il «muro della banda»: puoi
anche raddoppiare i core, ma se i byte non arrivano, i core in
più restano a girarsi i pollici.

Per capire dove i byte si perdono bisogna conoscere la geografia della memoria
di una GPU. Non è un unico serbatoio: è una **piramide** di livelli, ognuno un
compromesso diverso tra quanto è veloce e quanto è capiente.

## La piramide della memoria

La regola, valida per ogni computer ma spietata su una GPU, è che *veloce* e
*capiente* non stanno mai nello stesso posto. La memoria vicina ai core è
velocissima ma minuscola; quella grande abbastanza da contenere un modello è
lontana e lenta. In mezzo, una scala di compromessi
({numref}`fig-gerarchia-memoria`).

```{figure} ../figures/gpu-gerarchia-memoria.svg
:name: fig-gerarchia-memoria
:alt: "Piramide a cinque livelli della gerarchia di memoria di una GPU. Dall'apice alla base: registri (per-thread, pochi kilobyte, immediati); shared memory (per-blocco, circa cento kilobyte, on-chip); cache L2 (condivisa, decine di megabyte); memoria globale HBM (decine di gigabyte, banda di qualche terabyte al secondo, latenza di centinaia di cicli); memoria host, oltre il bus PCIe a decine di gigabyte al secondo. Salendo crescono velocità e banda, scendendo cresce la capacità."
:width: 80%

La gerarchia di memoria di una GPU. Salendo verso l'apice la memoria è più
veloce ma più piccola; scendendo verso la base è più capiente ma più lontana
dai calcolatori, e più lenta. I livelli fino alla L2 compresa stanno *dentro*
il chip della GPU (*on-chip*); la HBM e la memoria del computer sono fuori dal
chip (*off-chip*), ed è per questo che raggiungerle costa tanto.
```

`````{tab} Elementare
Pensa a dove tieni le cose mentre lavori a una scrivania. Quello che hai
letteralmente in mano (la penna che stai usando) è a distanza zero: sono i
**registri**, i cassetti privati di ogni singolo lavoratore, velocissimi ma
grandi appena da tenere un pugno di numeri alla volta. Sul piano della
scrivania tieni i fogli del momento: è la **shared memory**, un ripiano
piccolo (qualche decina di pagine) ma condiviso da tutta la squadra che siede
a quel tavolo. Il cassetto grande della scrivania è la **cache L2**, più
capiente e in comune con gli altri tavoli. («Cache», parola inglese che si
pronuncia *cash*, vuol dire nascondiglio, riserva: è un ripostiglio vicino in
cui il computer tiene le cose che ha appena usato, sperando di doverle riusare
presto. La L sta per *level*, livello, e il numero dice quanto è vicino a chi
lavora: c'è anche una L1, ancora più vicina, che sul silicio è lo stesso pezzo
di memoria della shared memory.) Poi c'è l'armadio della stanza, la **memoria
globale**: ci sta *tutto* il progetto, ma ogni volta devi alzarti e
attraversare la stanza. È quella che i tecnici chiamano **HBM**, tre lettere
che stanno per «memoria a banda larga». E infine il magazzino in un altro
edificio, la **memoria del computer** di là dal cavo che collega CPU e GPU (il
cavo si chiama **PCIe**): enorme, ma raggiungerlo è una spedizione. La
differenza di tempo tra prendere la penna in
mano e mandare qualcuno al magazzino non è del doppio: è di migliaia di volte.
Per questo il mestiere di chi programma le GPU somiglia a quello di chi
organizza bene la scrivania: tenere vicino ciò che serve adesso, e
attraversare la stanza il meno possibile.
`````

`````{tab} Superiore
I livelli, dall'alto verso il basso, sono cinque e differiscono per ordini di
grandezza (le cifre esatte cambiano con la generazione: qui contano le
*proporzioni*).

- **Registri**, privati del singolo thread: ciascun thread ne ha appena un
  pugno, dell'ordine del kilobyte (il tetto architetturale è 255 registri da
  32 bit, cioè poco più di 1 KB), con latenza di fatto nulla. Sono la memoria
  più veloce che esista sul chip. Attenzione però al singolare: *per SM* il
  register file è il banco on-chip più **grande** di tutti, 256 KB su A100
  contro i 192 KB di L1 e shared messe insieme, e sull'intera GPU sono 27 MB
  di registri. È questa abbondanza, non il «pugno» del singolo thread, a
  rendere possibile il secondo livello di tiling di cui si parlerà nel GEMM,
  quello che vive nei registri.
- **Shared memory e cache L1**: on-chip, all'interno dell'SM. La shared memory
  è condivisa dai thread di uno stesso blocco, dell'ordine di un centinaio di
  KB per unità di calcolo, e la sua particolarità è che *non* è una cache
  automatica: la gestisci a mano, decidendo tu cosa metterci. La L1 sì, è
  automatica. Il punto che la piramide disegnata nasconde è che dal 2017 le due
  sono **lo stesso banco di SRAM**, ripartito fra le due funzioni da un pomello
  che il programmatore gira (192 KB combinati per SM su A100, di cui fino a 164
  configurabili come shared; 256 KB su H100). Tre conseguenze pratiche:
  chiedere tutta la shared possibile non è gratis, perché toglie cache; il
  riuso «sperato» che il GEMM ingenuo strappa alle cache lo recupera in primo
  luogo la L1, non la L2; e negli **spill** dei registri (quando un thread ne
  chiede più di quanti ne ha) finisce lì il traffico che rende improduttivo
  alzare ancora i registri per thread. Latenza di poche decine di cicli.
- **Cache L2**: condivisa da tutte le unità di calcolo, dell'ordine di decine
  di MB (40 MB su A100), con latenza di un paio di centinaia di cicli. È
  l'ultimo livello *dentro* il chip.
- **Memoria globale (HBM)**, la *High Bandwidth Memory* off-chip: decine di
  GB, banda dell'ordine di qualche TB/s, ma latenza di *centinaia* di cicli. È
  dove vivono tensori, pesi e attivazioni.
- **Memoria host**, la RAM di sistema, di là dal bus **PCIe** che separa CPU e
  GPU: capiente quanto vuoi, ma con banda di appena qualche decina di GB/s
  (uno o due ordini di grandezza sotto la HBM). È il motivo per cui, come
  ricordava la sezione «Prestazioni e scala», `.to(device)` va fatto *una
  volta per batch* e non tensore per tensore.

Due parametri descrivono ogni livello: la **latenza** (quanto aspetti il primo
byte) e la **banda** (quanti byte al secondo, a regime). I warp nascondono la
*latenza* (mentre un warp aspetta la HBM, l'hardware ne fa girare un altro) ma
non moltiplicano la *banda*. Salendo la piramide la banda cresce e la latenza
cala: la memoria on-chip (registri, shared) ha banda di un ordine di grandezza
superiore alla HBM, che a sua volta ne ha uno o due sul PCIe.

Sulla capienza, invece, la forma a piramide va presa con le pinze, perché
diminuisce *per unità che ne dispone* (per thread, per blocco, per SM, per
GPU), non in assoluto: come si è visto, il register file di un SM è più
capiente della sua L1 più shared, ed è la punta della piramide a essere il
banco on-chip più grande. Il triangolo dice bene la scarsità che si affaccia
al singolo lavoratore, non quanta memoria ci sia a ciascun piano.

Tenere il lavoro il più in alto possibile nella piramide è, in una frase,
l'intera arte dell'ottimizzazione su GPU.
`````

## Accessi coalescenti: leggere in fila

Sapere *dove* stanno i dati non basta: conta anche *come* li si chiede. Qui
entra in gioco un dettaglio che distingue un kernel efficiente da uno che
spreca metà della banda senza accorgersene: la **coalescenza** degli accessi.
(Un **kernel** è il programmino che gira sulla GPU, quello che i migliaia di
lavoratori eseguono tutti insieme ciascuno sul proprio pezzo di dato: la
prossima sezione è dedicata a lui, qui basta sapere che quando si dice
«kernel» si intende il programma che sta girando.)

`````{tab} Elementare
Immagina un fattorino che deve consegnare 32 pacchi, e che il suo furgone ne
carichi otto per volta. Se i 32 indirizzi sono tutti sulla stessa via, uno dopo
l'altro, gli bastano **quattro** giri, e a ogni giro scarica il furgone pieno:
efficientissimo. Se invece i 32 indirizzi sono sparsi ai quattro angoli della
città, deve fare **32** viaggi separati, ognuno con un pacco solo e sette posti
vuoti, per consegnare esattamente gli stessi 32 pacchi. Il lavoro utile è
identico, il tempo speso è otto volte tanto. La
memoria di una GPU funziona così: non consegna un byte alla volta, ma a
*blocchi* di indirizzi vicini. Se i 32 thread di un warp chiedono dati messi in
fila in memoria, l'hardware li serve in poche consegne piene; se li chiedono
sparsi, deve fare una consegna quasi vuota per ognuno, e la banda va in fumo.
La morale pratica: **sistema i dati in modo che thread vicini leggano indirizzi
vicini**. Il fattore otto di questo esempio è esattamente quello che il conto
del livello Superiore, e il riquadro in fondo alla pagina, chiamano
«$8\times$ di banda buttata via».
`````

`````{tab} Superiore
La memoria globale viene servita in **segmenti** di indirizzi contigui:
diciamo, per fissare le idee, da 32 byte l'uno. Consideriamo un warp di 32
thread che legge un vettore di `float32` (4 byte ciascuno).

- *Accesso coalescente*: i thread leggono 32 elementi consecutivi, cioè
  $32 \times 4 = 128$ byte contigui. Servono $128 / 32 = 4$ segmenti; 128 byte
  trasferiti, 128 utili → **efficienza 100%**.
- *Accesso sparso*: per uno stride tale che ogni thread cada in un segmento
  diverso, servono 32 segmenti da 32 byte, cioè $32 \times 32 = 1024$ byte
  trasferiti per consegnare gli stessi 128 byte utili → **efficienza 12,5%**,
  ovvero $8\times$ di banda buttata via.

L'efficienza è il rapporto $\text{byte utili} / \text{byte trasferiti}$. Su un
carico limitato dalla banda, un fattore 8 di traffico sprecato è un fattore 8 di
tempo: ecco perché il modo in cui un tensore è disposto in memoria (il suo
*layout*, l'ordine `row-major` di righe e colonne) e l'indice con cui ogni
thread vi accede non sono dettagli, ma spesso la differenza tra un kernel che
satura la GPU e uno che la lascia mezza spenta.
`````

## Caricare una volta, servire in tanti

C'è un secondo modo di risparmiare banda, complementare al primo: non
ri-leggere dalla HBM ciò che ti serve più volte. Se un blocco di dati verrà
usato da molti thread, conviene portarlo *una sola volta* nella shared memory
(il ripiano condiviso della scrivania) e da lì servirlo a tutti.

L'esempio più celebre di questo principio ha un nome che incontreremo ancora,
**FlashAttention**, ed è il modo in cui oggi si esegue l'attenzione dei
Transformer: una sezione più avanti lo riprende per esteso. Vale la pena
anticipare qui una cosa sola, perché è il principio di questa sezione allo
stato puro: quell'algoritmo non fa *meno* conti dell'attenzione normale, ne fa
altrettanti (e nel passaggio all'indietro, come vedremo, qualcuno in più), e va
molte volte più veloce soltanto perché muove molti meno byte.

`````{tab} Elementare
Immagina una squadra che deve consultare lo stesso manuale decine di volte. La
mossa sciocca è che ognuno, ogni volta, corra in magazzino a prendere una
copia, la legga e la riporti. La mossa intelligente è portare *una* copia sul
tavolo comune all'inizio, e lasciare che tutti la consultino lì, a portata di
mano, per tutto il tempo. Il viaggio in magazzino (la lettura dalla memoria
lontana) si paga una volta sola invece di decine. Questo «carica una volta,
riusa in tanti» è il segreto di quasi tutti i kernel veloci, e sarà il cuore
della sezione in cui vedremo come si moltiplicano due matrici sul serio.
`````

`````{tab} Superiore
La leva quantitativa è il **fattore di riuso**: quante volte un dato caricato
in shared memory viene poi letto dai thread del blocco prima di essere
scartato. Se lo carichi una volta e lo usi $r$ volte, hai diviso per $r$ il
traffico verso la HBM per quel dato, e, come vedremo tra poco con il roofline,
ridurre i byte spostati a parità di conti è esattamente ciò che sposta un
kernel dal regime *memory-bound* verso quello *compute-bound*. La shared
memory è programmer-managed proprio per questo: a differenza di una cache
automatica, sei tu a decidere quale tessera trattenere e per quanto, adattando
il riuso alla struttura del calcolo. È un potere che si paga in complessità,
ed è il motivo per cui i kernel di alte prestazioni si scrivono a mano (o li
genera un compilatore come Triton, che incontreremo).

Ha però un secondo prezzo, ed è l'esatto analogo della coalescenza un piano più
su. La shared memory è divisa in **32 banchi** da 32 bit, con le parole
consecutive assegnate a banchi consecutivi, e i banchi sono tanti quanti i
thread di un warp proprio perché nel caso buono ciascun thread ne colpisca uno
diverso e i 32 accessi vengano serviti insieme. Se invece più thread dello
stesso warp chiedono parole *diverse* dello stesso banco (un **bank conflict**)
l'hardware spezza la richiesta in tante richieste prive di conflitto quante
servono, e la banda effettiva si divide per quel numero: fino a $32\times$ nel
caso peggiore. Fa eccezione il caso in cui i thread chiedono la *stessa*
parola: quello è un broadcast, ed è gratis.

Nel tiling classico del GEMM la cosa non morde (la tessera di $\mathbf{A}$ si
legge in broadcast, quella di $\mathbf{B}$ per parole consecutive) e proprio
per questo la nota serve: il caso comodo è quello, mentre ogni variante che
percorre una tessera *per colonna* (una trasposta, il caricamento dei frammenti
per i tensor core, il tile di $K^\top$ in FlashAttention) cade nel caso
scomodo. Il rimedio canonico sta in una riga: si dichiara la tessera con una
colonna in più (`[32][33]` invece di `[32][32]`), così l'indirizzo di ogni riga
slitta di un banco e la colonna smette di ricadere sempre sullo stesso.
`````

Ripetuto tante volte, questo schema (caricare una *tessera* di dati in shared
memory e riusarla da tutti i thread del blocco prima di passare alla
successiva) prende il nome di **tiling**, ed è il motore della moltiplicazione
tra matrici efficiente. Lo vedremo per esteso nella sezione dedicata al
**GEMM**, che è il nome con cui la moltiplicazione fra matrici va sotto nelle
librerie di calcolo (*GEneral Matrix Multiply*); qui basti sapere che la shared
memory esiste proprio per rendere possibile questo riuso.

## Il modello roofline: limitati dai conti o dai byte?

Mettiamo ora insieme i due limiti (quanto sa calcolare la GPU e quanti byte le
arrivano) in un unico quadro. Lo strumento si chiama **roofline** e viene da
un lavoro del 2009 di Williams, Waterman e Patterson
{cite}`williams2009roofline`, che già nel titolo lo definisce «un modello
visuale perspicace delle prestazioni», e la promessa è mantenuta, perché
riassume in un solo grafico il perché un programma va veloce o lento
({numref}`fig-roofline`).

L'idea ruota attorno a una sola quantità, l'**intensità aritmetica**: quanti
conti fai per ogni byte che sposti dalla memoria. Si misura, appunto, in FLOP
per byte, cioè in conti elementari per scatoletta di dati. Poche operazioni per
tanti byte significa che passi la vita ad aspettare i dati; tante operazioni
per pochi byte significa che i dati ti bastano e sei limitato solo da quanto
calcoli. Le due situazioni hanno un nome, e sono due parole inglesi che
ricorreranno in ogni pagina che segue: nel primo caso si è **memory-bound**,
alla lettera «legati alla memoria», cioè bloccati dal magazzino; nel secondo
**compute-bound**, «legati al calcolo», cioè bloccati dai cuochi.

```{figure} ../figures/roofline.svg
:name: fig-roofline
:alt: Grafico roofline in scala logaritmica. L'asse orizzontale è l'intensità aritmetica (FLOP per byte), il verticale la prestazione raggiungibile (FLOP/s). Il tetto ha un tratto inclinato a sinistra, il cui pendio è la banda di memoria, e un tratto orizzontale a destra, il picco di calcolo; si incontrano nel ginocchio. A sinistra del ginocchio i kernel sono memory-bound, a destra compute-bound. Una somma vettoriale a intensità circa un dodicesimo cade in basso a sinistra; un GEMM grande cade sul tetto piatto a destra. Una freccia mostra che la fusione dei kernel alza l'intensità, spostando l'operazione verso destra.
:width: 90%

Il modello roofline. Il «tetto» inclinato è il limite di banda (più conti per
byte → più prestazione), il tetto piatto è il picco di calcolo. A sinistra del
*ginocchio* si è bloccati dalla memoria (*memory-bound*), a destra dal calcolo
(*compute-bound*). Fondere più operazioni in una sola alza i conti per byte e
sposta l'operazione verso destra, verso il tetto di calcolo.
```

`````{tab} Elementare
Pensa a una cucina. La velocità con cui servi i piatti dipende da due cose:
quanto sono bravi i cuochi (il calcolo) e quanto in fretta arrivano gli
ingredienti dal magazzino (la banda). Se una ricetta richiede pochissima
preparazione ma tantissimi ingredienti (tipo «apri mille scatolette e svuotale
in una ciotola»), i cuochi finiscono in un attimo e stanno fermi ad aspettare
il prossimo carico: sei limitato dal *magazzino*. Se invece la ricetta lavora
a lungo su pochi ingredienti (un brodo che sobbolle per ore), gli ingredienti
bastano e avanzano, e a contare è solo la bravura dei cuochi: sei limitato dai
*cuochi*. Sono le due parole del paragrafo qui sopra: limitato dal magazzino si
dice *memory-bound*, limitato dai cuochi *compute-bound*, e da qui in avanti il
capitolo le userà così. Il roofline è il grafico che dice, per ogni ricetta, da
quale delle due parti sei bloccato. E suggerisce la cura: se sei limitato dal
magazzino, cerca di fare *più conti con gli stessi ingredienti* prima di
rimandarli indietro, che è esattamente ciò che fa la fusione dei kernel vista
nella sezione «Prestazioni e scala».
`````

`````{tab} Superiore
Formalizziamo. Sia $I$ l'intensità aritmetica in FLOP/byte, $B$ la banda di
memoria in byte/s e $P_\text{picco}$ il picco di calcolo in FLOP/s. La
prestazione raggiungibile è

$$
P(I) = \min\big(P_\text{picco},\; B \cdot I\big),
$$

dove il primo termine è il **tetto di calcolo** (piatto: non puoi superare il
picco di FLOP dell'hardware) e il secondo è il **tetto di banda** (inclinato: con
banda $B$, se sposti tanti byte per pochi conti, non puoi andare più veloce di
$B \cdot I$). I due tetti si incontrano nel **ginocchio**

$$
I^\star = \frac{P_\text{picco}}{B},
$$

l'intensità di pareggio. A sinistra ($I < I^\star$) domina la banda: si è
**memory-bound**. A destra ($I > I^\star$) domina il calcolo: si è
**compute-bound**.

Conviene fissare subito dove cade quel ginocchio, perché è il metro con cui il
resto del capitolo giudicherà ogni tecnica, e perché **ce n'è più d'uno sulla
stessa scheda**: dipende da quali unità di calcolo si stanno usando. Su una
A100 da 80 GB (banda $1{,}935$ TB/s) i CUDA core in `float32` danno
$19{,}5/1{,}935 \approx 10$ FLOP/byte, ma i tensor core in `float16` danno
$312/1{,}935 \approx 161$; su una H100 SXM (banda $3{,}35$ TB/s) si passa da
$67/3{,}35 = 20$ a $989/3{,}35 \approx 295$. Un calcolo che sta a destra del
primo ginocchio può stare comodamente a sinistra del secondo, e il secondo è
quello che conta appena si accende la mezza precisione.

Due esempi concreti, con dati in `float32` (4 byte):

- **Somma elemento-per-elemento** $z = x + y$. Per ogni elemento di uscita:
  1 FLOP (la somma), e $3 \times 4 = 12$ byte spostati (leggo $x$, leggo $y$,
  scrivo $z$). Intensità $I = 1/12 \approx 0{,}08$ FLOP/byte: bassissima,
  profondamente memory-bound. Con un ginocchio a 10 FLOP/byte questa operazione
  usa meno dell'1 % del picco di calcolo.
- **GEMM grande** $\mathbf{C} = \mathbf{A}\mathbf{B}$ con matrici
  $n \times n$. Circa $2n^3$ FLOP e, *con riuso perfetto*,
  $3 n^2 \times 4 = 12 n^2$ byte, per un'intensità $I = n/6$
  FLOP/byte che cresce con $n$: per $n = 4096$ vale circa 680 FLOP/byte,
  saldamente compute-bound. Attenzione però a che cosa vuol dire «riuso
  perfetto»: leggere ogni elemento di $\mathbf{A}$ e $\mathbf{B}$ *una volta
  sola*, cioè tenerle intere in memoria veloce. Per $n = 4096$ in `float32`
  sarebbero 201 MB, contro gli 84 MB di memoria on-chip di una H100: quell'$n/6$
  è un tetto ideale, non un traguardo. Ci si torna nella sezione sul GEMM, dove
  si vede quanto ci si arriva davvero (e perché non serve arrivarci).

Ora è chiaro *perché* la **kernel fusion** paga: fondere tre operazioni
elemento-per-elemento in un solo kernel significa leggere gli input una volta
e scrivere l'output una volta invece di tre (meno byte a parità di FLOP, cioè
intensità $I$ più alta). Sul roofline l'operazione scivola verso destra, dal
tetto di banda verso il tetto di calcolo. Nota infine che i **tensor core**
alzano $P_\text{picco}$ di un ordine di grandezza: spostano il ginocchio a
destra, e rendono *ancora più* facile ritrovarsi memory-bound. Ecco perché,
nell'era dei tensor core, la partita si gioca sempre più sui byte e sempre
meno sui FLOP.
`````

Questo modello sarà la bussola delle prossime sezioni. La moltiplicazione tra
matrici tiled è l'arte di spingere il GEMM il più a destra possibile sul
roofline; **FlashAttention** {cite}`dao2022flashattention`, che incontreremo
più avanti, è la stessa idea applicata all'attenzione: riorganizzare il
calcolo per non sprecare banda sulla HBM. Sotto nomi diversi, la domanda è
sempre la stessa: sto tenendo la bestia sfamata?

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Il collo di bottiglia di una GPU è quasi sempre **portare i numeri**, non
  farci i conti. Tenere tanti gruppi al lavoro nasconde le attese, ma non fa
  arrivare i dati più in fretta: quanti ne consegna la memoria al secondo (la
  **banda**) è un tetto che non si alza.
- La memoria è una **scrivania**: la penna in mano (i *registri*, privatissimi
  e minuscoli), i fogli sul piano (la *shared memory*, il tavolo della
  squadra), il cassetto grande (la *cache L2*), l'armadio dall'altra parte
  della stanza (la memoria grande della scheda, la *HBM*) e il magazzino in un
  altro edificio (la memoria del computer). Fra la penna e il magazzino non c'è
  il doppio di distanza: ce n'è migliaia di volte.
- Conta anche **come** si chiedono i dati, non solo dove stanno. Trentadue
  pacchi sulla stessa via si consegnano in quattro giri di furgone pieno; gli
  stessi trentadue sparsi per la città vogliono trentadue viaggi quasi vuoti:
  otto volte il tempo per lo stesso lavoro. Perciò conviene disporre i dati in
  modo che lavoratori vicini leggano posti vicini.
- L'altra mossa che risparmia viaggi è **portare il manuale sul tavolo comune
  una volta sola** e lasciare che tutti lo consultino lì. Ripetuta su blocchetti
  di dati (le *tessere*), è la tecnica che rende veloce la moltiplicazione fra
  matrici, ed è il motivo per cui la shared memory esiste.
- Ogni calcolo è bloccato o dal **magazzino** o dai **cuochi**, e il grafico
  che lo dice si chiama *roofline*: si guarda quanti conti si fanno per ogni
  byte portato. Poche operazioni su tanti dati (sommare due liste di numeri)
  sono bloccate dal magazzino; tanti conti su pochi dati (una grande
  moltiplicazione fra matrici) sono bloccati dai cuochi.
- Più i «cuochi» diventano veloci di generazione in generazione, più è facile
  ritrovarsi bloccati dal magazzino: è la ragione per cui tutto il capitolo
  parla di byte e non di conti.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Su una GPU il collo di bottiglia è spesso il **movimento dei dati**, non il
  calcolo: i warp nascondono la *latenza*, ma la **banda** resta finita; è il
  «muro della banda».
- La memoria è una **piramide**: registri (per-thread, immediati) → shared
  memory e cache L1, lo stesso banco di SRAM ripartito da un pomello
  (per-blocco, on-chip) → cache L2 → memoria globale HBM (decine di GB,
  centinaia di cicli di latenza) → memoria host, oltre il PCIe. Salendo cresce
  la velocità; la capienza cala *per unità che ne dispone*, non in assoluto (il
  register file di un SM è più grande della sua L1+shared).
- La **coalescenza** conta: se i 32 thread di un warp leggono indirizzi
  contigui, l'hardware fonde gli accessi in poche transazioni piene; sparsi,
  spreca banda (fino a $8\times$ nell'esempio). L'analogo un piano più su sono
  i **bank conflict** della shared memory, divisa in 32 banchi: due parole
  diverse dello stesso banco si serializzano, fino a $32\times$.
- Caricare un blocco *una volta* in shared memory e riusarlo da tutti i thread
  (il **tiling**) risparmia letture dalla HBM: è il motore del GEMM efficiente.
- Il **roofline** {cite}`williams2009roofline` mette l'intensità aritmetica
  (FLOP/byte) contro la prestazione: a sinistra del ginocchio si è
  **memory-bound**, a destra **compute-bound**. La somma vettoriale ($1/12$) è
  memory-bound; un GEMM grande è compute-bound.
- La **kernel fusion** aiuta perché alza l'intensità aritmetica; i **tensor
  core** alzano il picco di calcolo e spostano il ginocchio a destra (da
  $\approx 10$ FLOP/byte con i CUDA core a $\approx 160$ su A100 in `float16`),
  rendendo la banda ancora più decisiva.
```
`````
