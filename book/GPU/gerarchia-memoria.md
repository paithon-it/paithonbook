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
dai core, e più lenta. I livelli sopra la L2 sono *on-chip*, dentro il
silicio; HBM e memoria host sono *off-chip*.
```

`````{tab} Elementare
Pensa a dove tieni le cose mentre lavori a una scrivania. Quello che hai
letteralmente in mano (la penna che stai usando) è a distanza zero: sono i
**registri**, i cassetti privati di ogni singolo lavoratore, velocissimi ma
grandi appena da tenere un pugno di numeri alla volta. Sul piano della
scrivania tieni i fogli del momento: è la **shared memory**, un ripiano
piccolo (qualche decina di pagine) ma condiviso da tutta la squadra che siede
a quel tavolo. Il cassetto grande della scrivania è la **cache L2**, più
capiente e in comune con gli altri tavoli. Poi c'è l'armadio della stanza, la
**memoria globale**: ci sta *tutto* il progetto, ma ogni volta devi alzarti e
attraversare la stanza. E infine il magazzino in un altro edificio, la
**memoria del computer** di là dal cavo che collega CPU e GPU: enorme, ma
raggiungerlo è una spedizione. La differenza di tempo tra prendere la penna in
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
  pugno, dell'ordine del kilobyte, con latenza di fatto nulla. Sono la memoria
  più veloce che esista sul chip.
- **Shared memory** (o *scratchpad*): on-chip, condivisa dai thread di uno
  stesso blocco, dell'ordine di un centinaio di KB per unità di calcolo. La
  sua particolarità è che *non* è una cache automatica: la gestisci a mano,
  decidendo tu cosa metterci. Latenza di poche decine di cicli.
- **Cache L2**: condivisa da tutte le unità di calcolo, dell'ordine di decine
  di MB, con latenza di un paio di centinaia di cicli.
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
cala, di pari passo con la capienza che diminuisce: la memoria on-chip
(registri, shared) ha banda di un ordine di grandezza superiore alla HBM, che
a sua volta ne ha uno o due sul PCIe. Tenere il lavoro il più in alto
possibile nella piramide è, in una frase, l'intera arte dell'ottimizzazione su
GPU.
`````

## Accessi coalescenti: leggere in fila

Sapere *dove* stanno i dati non basta: conta anche *come* li si chiede. Qui
entra in gioco un dettaglio che distingue un kernel efficiente da uno che
spreca metà della banda senza accorgersene: la **coalescenza** degli accessi.

`````{tab} Elementare
Immagina un fattorino che deve consegnare 32 pacchi. Se i 32 indirizzi sono
tutti sulla stessa via, uno dopo l'altro, fa un solo giro e li lascia in blocco:
efficientissimo. Se invece i 32 indirizzi sono sparsi ai quattro angoli della
città, deve fare 32 viaggi separati per consegnare esattamente gli stessi 32
pacchi. Il lavoro utile è identico, il tempo speso è enormemente diverso. La
memoria di una GPU funziona così: non consegna un byte alla volta, ma a
*blocchi* di indirizzi vicini. Se i 32 thread di un warp chiedono dati messi in
fila in memoria, l'hardware li serve in poche consegne; se li chiedono sparsi,
deve fare una consegna quasi vuota per ognuno, e la banda va in fumo. La morale
pratica: **sistema i dati in modo che thread vicini leggano indirizzi vicini**.
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
ri-leggere dalla HBM ciò che ti serve più volte.

```{figure} ../figures/flashattention-2022.svg
:name: fig-flashattention-blocchi
:alt: "Le matrici Q, K e V dell'attenzione sono divise in blocchi. Un blocco per volta viene caricato dalla memoria HBM nella shared memory, dove il calcolo dell'attenzione viene svolto per intero su quel blocco; il risultato parziale viene accumulato e il blocco successivo prende il suo posto. La grande matrice dei punteggi non viene mai scritta per intero in HBM."
:width: 100%

Il risparmio sta in ciò che non si scrive. La matrice dei punteggi esiste solo
a pezzi, dentro la shared memory, e non tocca mai la memoria grande.
```

{numref}`fig-flashattention-blocchi` è l'esempio più celebre del principio di
questa sezione, e la sezione dedicata più avanti lo riprende nei dettagli.
Vale la pena notare fin d'ora che l'algoritmo non fa *meno* conti: ne fa
esattamente gli stessi, muovendo molti meno byte. Se un blocco di dati verrà
usato da molti thread, conviene portarlo *una sola volta* nella shared memory
(il ripiano condiviso della scrivania) e da lì servirlo a tutti.

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
`````

Ripetuto tante volte, questo schema (caricare una *tessera* di dati in shared
memory e riusarla da tutti i thread del blocco prima di passare alla
successiva) prende il nome di **tiling**, ed è il motore della moltiplicazione
tra matrici efficiente. Lo vedremo per esteso nella sezione dedicata al GEMM:
qui basti sapere che la shared memory esiste proprio per rendere possibile
questo riuso.

## Il modello roofline: limitati dai conti o dai byte?

Mettiamo ora insieme i due limiti (quanto sa calcolare la GPU e quanti byte le
arrivano) in un unico quadro. Lo strumento si chiama **roofline** e viene da
un lavoro del 2009 di Williams, Waterman e Patterson
{cite}`williams2009roofline`, che già nel titolo lo definisce «un modello
visuale perspicace delle prestazioni», e la promessa è mantenuta, perché
riassume in un solo grafico il perché un programma va veloce o lento
({numref}`fig-roofline`).

L'idea ruota attorno a una sola quantità, l'**intensità aritmetica**: quanti
conti fai per ogni byte che sposti dalla memoria. Poche operazioni per tanti
byte significa che passi la vita ad aspettare i dati; tante operazioni per pochi
byte significa che i dati ti bastano e sei limitato solo da quanto calcoli.

```{figure} ../figures/roofline.svg
:name: fig-roofline
:alt: Grafico roofline in scala logaritmica. L'asse orizzontale è l'intensità aritmetica (FLOP per byte), il verticale la prestazione raggiungibile (FLOP/s). Il tetto ha un tratto inclinato a sinistra, il cui pendio è la banda di memoria, e un tratto orizzontale a destra, il picco di calcolo; si incontrano nel ginocchio. A sinistra del ginocchio i kernel sono memory-bound, a destra compute-bound. Una somma vettoriale a intensità circa un dodicesimo cade in basso a sinistra; un GEMM grande cade sul tetto piatto a destra. Una freccia mostra che la fusione dei kernel alza l'intensità, spostando l'operazione verso destra.
:width: 90%

Il modello roofline. Il «tetto» inclinato è il limite di banda (più intensità →
più prestazione), il tetto piatto è il picco di calcolo. A sinistra del
*ginocchio* si è *memory-bound*, a destra *compute-bound*. La fusione dei kernel
alza l'intensità e sposta l'operazione verso destra, verso il tetto di calcolo.
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
*cuochi*. Il roofline è il grafico che dice, per ogni ricetta, da quale delle
due parti sei bloccato. E suggerisce la cura: se sei limitato dal magazzino,
cerca di fare *più conti con gli stessi ingredienti* prima di rimandarli
indietro, che è esattamente ciò che fa la fusione dei kernel vista nella
sezione «Prestazioni e scala».
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
**compute-bound**. Due esempi concreti, con dati in `float32` (4 byte):

- **Somma elemento-per-elemento** $z = x + y$. Per ogni elemento di uscita:
  1 FLOP (la somma), e $3 \times 4 = 12$ byte spostati (leggo $x$, leggo $y$,
  scrivo $z$). Intensità $I = 1/12 \approx 0{,}08$ FLOP/byte: bassissima,
  profondamente memory-bound. Una GPU con ginocchio a qualche decina di FLOP/byte
  qui userebbe una frazione irrisoria del suo picco di calcolo.
- **GEMM grande** $C = A B$ con matrici $n \times n$. Circa $2n^3$ FLOP e, *con
  riuso perfetto*, $3 n^2 \times 4 = 12 n^2$ byte, per un'intensità $I = n/6$
  FLOP/byte che cresce con $n$: per $n = 4096$ vale circa 680 FLOP/byte,
  saldamente compute-bound. Ma quel $n/6$ è l'ideale, e si raggiunge *solo* con
  il riuso in shared memory di cui sopra; senza tiling il GEMM sposterebbe molti
  più byte e scivolerebbe a sinistra.

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

```{admonition} Da ricordare
:class: important
- Su una GPU il collo di bottiglia è spesso il **movimento dei dati**, non il
  calcolo: i warp nascondono la *latenza*, ma la **banda** resta finita; è il
  «muro della banda».
- La memoria è una **piramide**: registri (per-thread, immediati) → shared
  memory (per-blocco, on-chip, gestita a mano) → cache L2 → memoria globale HBM
  (decine di GB, centinaia di cicli di latenza) → memoria host, oltre il PCIe.
  Salendo cresce la velocità, scendendo la capienza.
- La **coalescenza** conta: se i 32 thread di un warp leggono indirizzi
  contigui, l'hardware fonde gli accessi; sparsi, spreca banda (fino a $8\times$
  nell'esempio).
- Caricare un blocco *una volta* in shared memory e riusarlo da tutti i thread
  (il **tiling**) risparmia letture dalla HBM: è il motore del GEMM efficiente.
- Il **roofline** {cite}`williams2009roofline` mette l'intensità aritmetica
  (FLOP/byte) contro la prestazione: a sinistra del ginocchio si è
  **memory-bound**, a destra **compute-bound**. La somma vettoriale ($1/12$) è
  memory-bound; un GEMM grande è compute-bound.
- La **kernel fusion** aiuta perché alza l'intensità aritmetica; i **tensor
  core** alzano il picco di calcolo e spostano il ginocchio a destra, rendendo
  la banda ancora più decisiva.
```
