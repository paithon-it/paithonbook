# La memoria: il vero collo di bottiglia

Nella sezione «Prestazioni e scala» del {doc}`capitolo su PyTorch </PyTorch/overview>` avevamo lasciato
cadere, quasi di sfuggita, un'osservazione scomoda: «il collo di bottiglia, più
spesso del calcolo, è il movimento dei dati». È il momento di prenderla sul
serio, perché è una delle verità meno intuitive di tutto l'hardware moderno.

L'immagine che viene spontanea è quella di una GPU come un mostro di calcolo
che divora numeri. La realtà, molto più spesso, è un mostro *affamato* che
aspetta di essere imboccato: *feeding the beast*, «sfamare la bestia», è il
modo in cui gli ingegneri chiamano il problema. Le migliaia di postazioni di
calcolo di cui abbiamo parlato nell'architettura macinano in un lampo i dati
che hanno già sotto mano, poi restano ferme ad aspettare i prossimi.

Nella sezione precedente abbiamo visto la mossa che salva la GPU da quelle
attese: mentre un plotone di trentadue (un **warp**) aspetta i suoi numeri, il
caposquadra ne manda avanti un altro, così l'officina non resta mai a mani
vuote. Quel trucco però *nasconde* l'attesa del singolo, non fabbrica dati più
in fretta. Le due cose vanno tenute distinte, e da qui in avanti le chiameremo
sempre con lo stesso nome: la **latenza** è quanto si aspetta perché arrivi la
prima consegna; la **banda** è quanti byte al secondo la memoria riesce davvero
a consegnare, a regime (è il *throughput* della sezione precedente, misurato in
byte invece che in compiti finiti). I warp coprono la prima. La seconda è finita,
non si nasconde, ed è lei, non la potenza di calcolo, a decidere il destino di
moltissimi programmi. È il «muro della banda»: puoi anche raddoppiare le
postazioni di calcolo, ma se i byte non arrivano, quelle in più restano a
girarsi i pollici.

Per capire dove i byte si perdono bisogna conoscere la geografia della memoria
di una GPU. È una **piramide** di livelli e non un unico serbatoio, ognuno un
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

I cinque piani della memoria di una GPU. Salendo verso l'apice si trova
memoria più veloce ma più piccola; scendendo verso la base memoria più
capiente ma più lenta, perché più lontana dalle unità che fanno i conti. I
primi tre piani stanno *dentro* il chip della GPU (in inglese *on-chip*); gli
ultimi due, la memoria grande della scheda e quella del computer, stanno fuori
dal chip (*off-chip*), ed è per questo che raggiungerli costa tanto. 
```

`````{tab} Elementare
La penna che hai in mano è a distanza zero: la usi senza pensarci. Sono i
**registri**, e ciascuno ha i suoi, che nessun altro può toccare: velocissimi,
e ci sta appena un pugno di numeri.

Sul piano della scrivania c'è un ripiano in comune con tutta la squadra che
siede al tavolo. Ci sta l'equivalente di qualche decina di pagine, e un
separatore lo divide in due. Da una parte ci metti tu i fogli del momento, e
scegli quali tenere e per quanto: è la **shared memory**, la memoria condivisa.
Dall'altra parte finisce da sé, senza che nessuno lo decida, quello che hai
usato di recente, nel caso serva ancora: è la **cache L1** (si pronuncia
*cash*, e in inglese vuol dire ripostiglio). Il separatore si sposta, lo spazio
no: la parte che scegli tu si allarga rubando all'altra.

Il cassetto grande sotto il tavolo è la **cache L2**: stesso mestiere un numero
più in là, più capiente e in comune con gli altri tavoli. In fondo alla stanza
c'è l'armadio, il magazzino della squadra: ci sta *tutto* il progetto, ma ogni volta ti tocca alzarti e
attraversare la stanza. È la **memoria globale**, che i tecnici chiamano
**HBM**, tre lettere per «memoria a banda larga», costruita apposta per
consegnare tantissimi byte al secondo. E in un altro edificio c'è il deposito,
la memoria del computer, di là dal cavo che collega CPU e GPU (il cavo si
chiama **PCIe**): enorme, e andarci è una spedizione, tanto che la roba di là
la si va a prendere una volta sola, all'inizio.

A quel tavolo lavorano più di mille persone. Le penne sono minuscole una per
una, ma tutte insieme sono più roba di quanta ne stia sul ripiano: il pugno di
numeri è quello che tocca al singolo, non quanto ce n'è in tutto. E chi ha
bisogno di più penne di quante gliene stiano in mano non le perde: quelle di
troppo finiscono sul ripiano, dove rubano posto alla squadra e vanno riprese
ogni volta. Caricarsi di penne oltre un certo punto peggiora le cose invece di
migliorarle.

Contano le proporzioni, più dei nomi. Se prendere la penna che hai già fra le
dita costa un secondo, cercare fra i fogli sul piano ne costa una ventina,
aprire il cassetto grande un paio di centinaia, attraversare la stanza fino
all'armadio cinquecento, e mandare qualcuno al deposito dell'altro edificio è
una gita che dura un'ora. Fra la penna e il deposito ci sono migliaia di volte,
non il doppio. Chi programma una GPU fa lo stesso mestiere di chi si tiene in
ordine la scrivania: vicino quello che serve adesso, e in giro per la stanza il
meno possibile.
`````

`````{tab} Superiore
I livelli, dall'alto verso il basso, sono cinque e differiscono per ordini di
grandezza (le cifre esatte cambiano con la generazione: qui contano le
*proporzioni*).

- **Registri**, privati del singolo thread: ciascun thread ne ha appena un
  pugno, dell'ordine del kilobyte (il tetto architetturale è 255 registri da
  32 bit, cioè 1020 byte, appena sotto il kilobyte), con latenza di fatto
  nulla. Sono la memoria più veloce che esista sul chip. Attenzione però al singolare: *per SM* il
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
entra in gioco un dettaglio che può far buttare via i sette ottavi della banda
senza che nessuno se ne accorga. Il fatto è che la memoria non consegna un byte
alla volta: consegna a pacchi di indirizzi vicini, e chiedere numeri che stanno
in fila costa molto meno che chiedere gli stessi numeri sparsi. Quando le
richieste di un plotone cadono in fila, l'hardware le **fonde** in poche
consegne piene, e quel fondersi ha dato il nome alla cosa: **coalescenza** degli
accessi, dal verbo *coalescere*, che si dice di due gocce quando diventano una
sola.

`````{tab} Elementare
Un fattorino ha 32 pacchi da consegnare e un furgone che ne carica otto per
volta. C'è però una regola del deposito, ed è la regola che rende questo
esempio vero: **il furgone può scaricare in una via sola**, e per cambiare via
deve tornare a caricare. Se i 32 indirizzi sono tutti sulla stessa via, uno
dopo l'altro, gli bastano dunque **quattro** giri, e a ogni giro scarica il
furgone pieno: otto pacchi, otto consegne. Se invece i 32 indirizzi sono sparsi
ai quattro angoli della città, ogni giro consegna un pacco solo e riporta
indietro sette posti vuoti: servono **32** giri per consegnare esattamente gli
stessi 32 pacchi. Il lavoro utile è identico, i viaggi sono 32 invece di 4:
**otto volte tanto**, e l'otto viene da qui, dai posti del furgone.

La memoria di una GPU funziona proprio così, e nessuno dei due numeri è
inventato. Il plotone è da 32 perché così è fatta la GPU, e il furgone porta
otto numeri perché la memoria consegna a blocchi da 32 byte, dentro i quali di
numeri da quattro byte ce ne stanno appunto otto. Se i 32 lavoratori di un
plotone chiedono dati messi in fila, l'hardware li serve in quattro consegne
piene; se li chiedono sparsi, deve fare una consegna quasi vuota per ognuno, e
la banda va in fumo. La regola del deposito, poi, dipende da come sono fatti i
collegamenti dentro il chip, e non si può cambiare. La morale
pratica: **sistema i dati in modo che lavoratori vicini leggano posizioni
vicine**.
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

Di questo principio c'è un esempio celebre, e conviene anticiparlo qui perché
è il principio di questa sezione allo stato puro. Si chiama
**FlashAttention**, ed è il modo in cui oggi si eseguono i confronti fra le
parole di un testo dentro un modello linguistico; una sezione più avanti lo
racconta per esteso. La cosa da sapere fin da adesso è una sola: quel metodo
non fa *meno* conti di prima. Ne fa altrettanti, e in un passaggio addirittura
qualcuno in più. Va molte volte più veloce soltanto perché muove molti meno
byte.

`````{tab} Elementare
Lo stesso manuale, consultato decine di volte da tutta la squadra: i modi di
farlo sono due. La mossa sciocca è che ognuno, ogni volta, corra in magazzino
a prendere una copia, la legga e la riporti. La mossa intelligente è portare
*una* copia sul tavolo comune all'inizio, e lasciare che tutti la consultino
lì, a portata di mano, per tutto il tempo. Il viaggio in magazzino (la lettura
dalla memoria lontana) si paga una volta sola invece di decine: trenta
consultazioni e un viaggio, cioè un trentesimo della strada. Il tavolo comune,
poi, non si riempie da sé, e qui sta il suo vantaggio: qualcuno sceglie che
cosa metterci e quando toglierlo per far posto al pezzo dopo. Questo «carica
una volta, riusa in tanti» è il segreto di quasi tutti i **kernel** veloci (un
kernel è il programmino che gira sulla GPU, quello che tutti i lavoratori
eseguono insieme ciascuno sul proprio pezzo di dato: gli è dedicata la
prossima sezione), e sarà il cuore di quella in cui vedremo come si
moltiplicano due tabelloni di numeri sul serio.

Il tavolo comune ha però una regola sua, e a ignorarla si perde per strada
quello che si era appena guadagnato. Il piano è uno scaffale con trentadue
caselle, e le pagine ci vanno a giro: la prima nella prima casella, la seconda
nella seconda, fino alla
trentaduesima, poi si ricomincia. Una casella la può aprire una persona alla
volta. Se i trentadue della squadra chiedono pagine che stanno in trentadue
caselle diverse, si servono tutti nello stesso istante. Se invece chiedono
pagine diverse che stanno nella stessa casella, devono fare la fila: in due,
due turni; in trentadue sulla stessa casella, trentadue turni, e del vantaggio
di avere il manuale sul tavolo resta poco (i tecnici la chiamano *bank
conflict*: le caselle, dentro una GPU, si chiamano banchi). Un caso non
costa niente: quando tutti vogliono la *stessa* pagina, uno la legge ad alta
voce e la sentono tutti insieme.

Se la squadra legge di seguito, riga per riga, la fila non si forma: pagine
vicine stanno in caselle vicine. Si forma quando si legge una tabella per
colonne, e ogni riga della tabella è larga esattamente trentadue pagine: allora
una colonna intera cade tutta nella stessa casella, e servono trentadue turni
per una lettura sola. Il rimedio sembra uno scherzo e funziona: si lascia una
casella vuota in fondo a ogni riga, larghezza trentatré invece di trentadue.
Ogni riga slitta di uno, la colonna si sparpaglia su tutte le caselle, la fila
sparisce. Si butta via un trentatreesimo dello scaffale e si guadagna un
fattore trentadue.
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
per i tensor core, il tile di $\mathbf{K}^\top$ in FlashAttention) cade nel caso
scomodo. Il rimedio canonico sta in una riga: si dichiara la tessera con una
colonna in più (`[32][33]` invece di `[32][32]`), così l'indirizzo di ogni riga
slitta di un banco e la colonna smette di ricadere sempre sullo stesso.
`````

Questo schema (portare sul tavolo comune un blocchetto di dati, farlo usare a
tutta la squadra, e solo allora passare al blocchetto successivo) si ripete
tante volte di seguito, e ha un nome inglese che ricorrerà fino alla fine del
capitolo: **tiling**, cioè «piastrellare», perché i dati si spezzano in
quadratini come un pavimento, e da qui in avanti chiameremo *tessera* ciascuno
di quei quadratini. È il motore della moltiplicazione fra tabelloni di numeri
(fra **matrici**, in matematica), che è l'operazione su cui una rete neurale
passa quasi tutto il suo tempo. Le è dedicata una sezione più avanti, quella
sul **GEMM**, che è la sigla sotto cui quella moltiplicazione va nelle librerie
di calcolo (*GEneral Matrix Multiply*). Qui basti sapere che la shared memory
esiste proprio per rendere possibile questo riuso.

## Il modello roofline: limitati dai conti o dai byte?

Mettiamo ora insieme i due limiti (quanto sa calcolare la GPU e quanti byte le
arrivano) in un unico quadro. Lo strumento si chiama **roofline**, cioè «linea
del tetto», e viene da un lavoro del 2009 di Williams, Waterman e Patterson
{cite}`williams2009roofline`. Il titolo lo presenta come «un modello visuale
illuminante delle prestazioni», e la promessa è mantenuta: riassume in un solo
grafico il perché un programma va veloce o lento
({numref}`fig-roofline`).

L'idea ruota attorno a una sola quantità, l’**intensità aritmetica**: quanti
conti fai per ogni byte che sposti dalla memoria. Si misura in FLOP per byte
(un FLOP, come si è detto nell'architettura, è un conto elementare: una
moltiplicazione o una somma). Poche operazioni per tanti byte significa che
passi la vita ad aspettare i dati; tante operazioni per pochi byte significa
che i dati ti bastano e sei limitato solo da quanto calcoli. Le due situazioni
hanno un nome, e sono due parole inglesi che ricorreranno in ogni pagina che
segue: nel primo caso si è **memory-bound**, alla lettera «legati alla
memoria», cioè bloccati dal magazzino; nel secondo **compute-bound**, «legati
al calcolo», cioè bloccati dai cuochi.

```{figure} ../figures/roofline.svg
:name: fig-roofline
:alt: Grafico roofline in scala logaritmica. L'asse orizzontale è l'intensità aritmetica (FLOP per byte), il verticale la prestazione raggiungibile (FLOP/s). Il tetto ha un tratto inclinato a sinistra, il cui pendio è la banda di memoria, e un tratto orizzontale a destra, il picco di calcolo; si incontrano nel ginocchio. A sinistra del ginocchio i kernel sono memory-bound, a destra compute-bound. Una somma vettoriale a intensità circa un dodicesimo cade in basso a sinistra; un GEMM grande cade sul tetto piatto a destra. Una freccia mostra che la fusione dei kernel alza l'intensità, spostando l'operazione verso destra.
:width: 90%

Come si legge il roofline. In orizzontale, quanti conti si fanno per ogni byte
portato dalla memoria: più a destra, più conti. In verticale, la velocità che
se ne ricava. Il «tetto» ha due falde: quella inclinata a sinistra è il limite
imposto dalla banda, quella piatta a destra è la velocità massima di calcolo
della scheda, che non si può superare in nessun modo. Il punto in cui le due
falde si incontrano, e il tetto cambia pendenza come una gamba che si piega, si
chiama **ginocchio**. Un calcolo che cade a sinistra del
ginocchio è bloccato dalla memoria (*memory-bound*), uno a destra dal calcolo
(*compute-bound*). Fondere più operazioni in una sola alza i conti per byte e
sposta l'operazione verso destra.
```

`````{tab} Elementare
In cucina la velocità con cui escono i piatti dipende da due cose: quanto sono
bravi i cuochi e quanto in fretta arrivano gli ingredienti dal magazzino. I
cuochi sono le unità che fanno i conti, il magazzino è la memoria grande della
scheda, e la velocità dei piatti è quella del più lento dei due.

Una ricetta che chiede pochissima preparazione e tantissimi ingredienti (apri
mille scatolette e svuotale in una ciotola) lascia i cuochi fermi ad aspettare
il carico dopo: sei limitato dal magazzino, e si dice *memory-bound*. Un brodo
che sobbolle per ore lavora a lungo su pochi ingredienti: la roba basta e
avanza, e a contare è solo la mano dei cuochi, cioè si è *compute-bound*.

Il roofline è il grafico che dice, per ogni ricetta, da quale delle due parti
sei bloccato, e per leggerlo basta una misura: quanti conti fai per ogni byte
che ti sei fatto portare. Prendi due lunghe liste di numeri da sommare. Per
ogni somma il fattorino porta i due addendi e riporta indietro il risultato:
tre numeri da quattro byte l'uno, dodici byte per *un* conto. Un dodicesimo di
conto per byte, e sei nel magazzino fino al collo.

Adesso due tabelloni da quattromila numeri di lato, da moltiplicare. Il
fattorino porta i tre tabelloni, 48 milioni di numeri, cioè 192 milioni di
byte. I cuochi, per ciascuna delle 16 milioni di caselle del risultato, fanno
quattromila moltiplicazioni e altrettante somme: 128 miliardi di conti.
Dividendo, quasi settecento conti per ogni byte portato, e il magazzino smette
di essere il problema. Quel settecento però suppone che ogni ingrediente entri
in cucina una volta sola, cioè che tutta la roba stia sul tavolo accanto ai
cuochi mentre lavorano. Per tabelloni di quella taglia sul tavolo non ci sta, e
qualche viaggio in più si fa comunque: settecento è il massimo sperabile, e la
cucina vera resta un po’ sotto.

In mezzo c'è il pareggio, che su una scheda di qualche anno fa sta intorno ai
dieci conti per byte: sotto comanda il magazzino, sopra comandano i cuochi. È
un numero che si sposta, e sempre nella stessa direzione. Accendi le unità
costruite apposta per moltiplicare due tabelloni, i *tensor core*, e il
pareggio sale oltre i centocinquanta: i cuochi sono diventati sedici volte più
svelti, mentre il magazzino consegna alla stessa velocità di prima. Da qui la
cura, per chi sta sotto: fare più conti con gli stessi ingredienti prima di
rimandarli indietro.
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
  sarebbero 201 MB, contro i poco meno di 120 MB che una H100 ha on-chip in tutto (50 di cache L2,
  34 di shared e L1, 34 di registri): quell’$n/6$
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

Questo grafico sarà la bussola delle prossime sezioni. Spezzare in tessere la
moltiplicazione fra due tabelloni di numeri è l'arte di spingerla il più a
destra possibile sul roofline; **FlashAttention** {cite}`dao2022flashattention`
è la stessa idea applicata ai confronti fra le parole di un testo, cioè
riorganizzare il calcolo per non sprecare banda. Sotto nomi diversi, la domanda
è sempre la stessa: sto tenendo la bestia sfamata?

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
  della stanza (la memoria grande della scheda, la *HBM*, quella che in tutto
  il capitolo si chiama «il magazzino») e il deposito in un altro edificio (la
  memoria del computer). Fra la penna e il deposito non c'è il doppio di
  distanza: ce n'è migliaia di volte.
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
