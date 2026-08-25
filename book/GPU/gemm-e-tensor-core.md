# GEMM: la moltiplicazione di matrici, spremuta

Prendi l'addestramento di una rete neurale qualunque e mettici sopra un
*profiler*: uno strumento che cronometra il programma pezzo per pezzo e dice
quanto tempo se ne va in ciascuno. Guarda poi dove il tempo è finito. In cima
alla lista trovi quasi sempre la stessa voce, ed è la moltiplicazione fra due
**matrici**, cioè fra
due tabelloni di numeri: una tabella per una tabella, e viene fuori una terza
tabella.

Quell'operazione, nelle **librerie** di calcolo (le raccolte di pezzi di
programma già scritti e collaudati, che chiunque richiama invece di
riscriverseli), porta da decenni una sigla: **GEMM**, *GEneral Matrix
Multiply*, moltiplicazione generale fra matrici. È
probabilmente il pezzo di codice più ottimizzato della storia
dell'informatica: a ogni generazione di hardware qualcuno lo riscrive da capo
per spremerne l'ultima goccia.

Non è un caso. Nella sezione «Prestazioni e scala» del {doc}`capitolo su PyTorch </PyTorch/overview>`
abbiamo detto che una rete neurale, vista dall'hardware, è quasi soltanto una
cosa: moltiplicazioni fra matrici. Da AlexNet {cite}`krizhevsky2012imagenet` in
poi, il grosso dei conti di qualunque rete si riduce a quella. E la sezione «La
memoria: il vero collo di bottiglia» ha lasciato in sospeso una promessa,
mostrare per esteso il trucco con cui la si rende veloce, il **tiling**: è il
momento di mantenerla.

## Il conto, e il problema della versione ingenua

Moltiplicare due matrici vuol dire riempire una tabella di risultati, e ogni
casella del risultato è una somma di prodotti: si prende una riga della prima
tabella e una colonna della seconda, le si moltiplica numero per numero e si
somma il tutto. In simboli, $\mathbf{C} = \mathbf{A}\,\mathbf{B}$, con
$\mathbf{A}$ di forma $(M, K)$ (cioè con $M$ righe e $K$ colonne) e
$\mathbf{B}$ di forma $(K, N)$, dà una matrice $\mathbf{C}$ di forma $(M, N)$.
Quanti conti servano si vede a occhio; il problema, come sempre su una GPU, non
è *quanti conti* fai, ma *quanti byte* muovi per farli.

`````{tab} Elementare
Una riga per una colonna, moltiplica e somma: facile. Ma immagina di farlo
davvero *una casella alla volta*,
andando ogni volta a ripescare la riga e la colonna dal magazzino lontano, cioè
dalla memoria globale della scheda. Due caselle vicine sulla stessa riga
usano la *stessa* riga della prima tabella: eppure, alla cieca, vai a
riprenderla da capo per ognuna. È come cucinare cento piatti identici correndo
in dispensa a prendere gli stessi ingredienti cento volte.

Attenzione a non fraintendere: i conti da fare sono tantissimi, ed è proprio
per questo che qui se ne va il tempo di un addestramento. Il punto è che i
conti fatti per ogni viaggio in dispensa sono pochissimi, e la sezione
sulla memoria ha stabilito che è quel rapporto, e non il totale, a decidere se
un calcolo va veloce: quanti conti si fanno per ogni byte che ci si è fatti
portare. Fatta una casella alla volta, la moltiplicazione fra matrici sta in
fondo a quella classifica: consuma tutti i byte al secondo che la memoria
riesce a consegnare, e tiene le unità di calcolo per lo più ferme. E
ingrandire le tabelle non le cambia posto in classifica: raddoppia i lati, e
conti e viaggi crescono insieme, nella stessa misura, mentre il rapporto fra i
due resta quello di prima.

Un aiuto arriva soltanto per caso. Fra la dispensa e le cucine c'è un
ripostiglio in comune fra tutte le squadre, dove resta per un po’ quello che è
appena passato di lì (i tecnici lo chiamano **cache L2**), e ogni tanto la riga
che ti serve è ancora lì: te la ritrovi a due passi invece che in fondo al
corridoio. Qualche viaggio lo risparmi davvero, e le cose vanno un po’ meglio
di così. Ma è un colpo di fortuna, e nessuno l'ha deciso: il tiling farà di
meglio, perché quel risparmio se lo prende per iscritto invece di sperarlo.
`````

`````{tab} Superiore
Il prodotto costa circa $2 M N K$ operazioni in virgola mobile: per ognuno
degli $M \times N$ elementi di $\mathbf{C}$, una somma di $K$ prodotti, cioè $K$
moltiplicazioni e $K$ addizioni. La versione ingenua assegna un thread a ogni
elemento di uscita e legge dalla memoria globale, per calcolare
$C_{ij} = \sum_{k} A_{ik} B_{kj}$, un'intera riga di $\mathbf{A}$ ($K$ valori)
e un'intera colonna di $\mathbf{B}$ ($K$ valori). Sommando su tutte le uscite sono
$2 M N K$ letture di elementi, a fronte di $2 M N K$ FLOP: in `float32`
(4 byte per elemento) l’**intensità aritmetica** vale

$$
I_\text{naive} = \frac{2 M N K}{4 \cdot 2 M N K} = \frac{1}{4} \ \text{FLOP/byte},
$$

*indipendente dalla taglia delle matrici*. Il conto assume il modello più
crudo: ogni lettura emessa viene servita dalla HBM, senza cache di mezzo.
Nella realtà la L2 e il broadcast dentro il warp recuperano una parte del
riuso, e il kernel ingenuo fa un po’ meglio di $\tfrac14$; ma è un riuso
*sperato*, affidato alla cache, mentre il tiling che segue lo rende
*garantito* dal programma. Sul roofline della sezione «La
memoria: il vero collo di bottiglia» è comunque un punto incollato in basso a
sinistra: profondamente memory-bound. La radice dello spreco è la ri-lettura:
la stessa riga di $\mathbf{A}$ torna dalla HBM per ognuna delle $N$ colonne di
$\mathbf{C}$, la stessa colonna di $\mathbf{B}$ per ognuna delle $M$ righe. Si
spostano montagne di byte per rileggere all'infinito gli stessi numeri.
`````

## Tiling: portare gli ingredienti sul tavolo una volta sola

La cura è quella già anticipata nella sezione sulla memoria: **caricare una
volta, riusare in tanti**. Invece di calcolare $\mathbf{C}$ una casella alla
volta, la si spezza in **tessere** (i *tile*); per ogni tessera si portano i
blocchi corrispondenti di $\mathbf{A}$ e di $\mathbf{B}$ nella shared memory
(il ripiano condiviso della scrivania) *una sola volta*, e da lì si riusano per
tutti i prodotti della tessera ({numref}`fig-gemm-tiling`).

```{figure} ../figures/gemm-tiling.svg
:name: fig-gemm-tiling
:alt: La matrice A sta a sinistra della matrice C e ne condivide le righe; la matrice B sta sopra C e ne condivide le colonne. Una tessera di C è evidenziata; nasce dal prodotto della banda-riga di A che le sta a sinistra per la banda-colonna di B che le sta sopra. Le bande si scorrono a blocchi lungo la dimensione K; un blocco di A e uno di B, in ocra, rappresentano i dati caricati una volta nella shared memory e riusati per tutta la tessera, con frecce che li collegano alla tessera di C.
:width: 90%

Il tiling del GEMM: un quadratino del risultato nasce dalla striscia di righe
che ha a sinistra per la striscia di colonne che ha sopra. Le strisce si
scorrono a blocchi; ogni blocco, portato una volta sul tavolo di lavoro vicino
ai calcolatori (in ocra), serve tutti i prodotti del quadratino prima di essere
scartato.
```

`````{tab} Elementare
Cento piatti identici, e la dispensa in fondo al corridoio. Chi va a prendere
gli ingredienti per ogni piatto passa la giornata nel corridoio; chi carica una
cassetta all'inizio e cucina con quella l'infornata intera paga il viaggio una
volta sola. La cassetta, qui, è un blocchetto della prima tabella e uno della
seconda, portati sul ripiano accanto ai fornelli (la shared memory) e tenuti lì
finché non hanno servito tutti i prodotti della tessera che la squadra sta
calcolando. Ogni numero arriva una volta e viene riusato molte volte prima di
essere buttato.

Più grande la cassetta, più piatti escono da ogni viaggio, e sul ripiano ci sta
una cassetta e poco più. Con una tessera da trentadue caselle di lato i viaggi
si dividono per trentadue, e nemmeno questo basta. Il pareggio fra magazzino e
cuochi, quello stabilito nella sezione sulla memoria, sta a dieci conti per
ogni byte che ci si fa portare, e una tessera così ne fa otto, con i cuochi che
restano un po’ fermi ad aspettare. Qualche casella in più colmerebbe il
divario, se la soglia stesse ferma. Ma le unità costruite apposta per
moltiplicare tabelloni la portano oltre il centocinquanta, e una cassetta
capace di tenere quel passo sul ripiano non ci sta. Nemmeno scegliendo meglio
la misura, perché la più grande che ci entra resta lontanissima. E un ripiano
più largo rende meno di quanto prometta, visto che per fare il doppio dei conti
su ogni byte portato deve diventare quattro volte più grande.

Come mai, allora, le moltiplicazioni vere volano? Perché in fondo al corridoio
non ci va quasi nessuno. Le squadre che lavorano fianco a fianco chiedono
cassette identiche, e la prima che la ordina la lascia nel ripostiglio comune,
dove le altre la trovano a due passi. Su una moltiplicazione grande la
differenza è tutta qui. Contando ogni cassetta come un viaggio fino in
dispensa, si passerebbe due volte e mezzo più tempo a trasportare che a
cucinare; contando quello che il ripostiglio serve da sé, il trasporto scende
sotto un decimo della cottura. Stesso lavoro, stessi piatti: nel primo conto
comanda la dispensa, nel secondo i cuochi.

Dentro quel viaggio i programmi veri ne infilano un secondo. Ogni cuoco prende
dal ripiano sedici numeri, se li tiene in mano e ne ricava sessantaquattro
prodotti senza tornare al ripiano nemmeno una volta. Quei sedici numeri non
risparmiano un solo viaggio in dispensa, che quelli li decide la cassetta
grande. Servono contro una coda diversa: al ripiano ci vanno tutte le mani
della squadra, e se ognuna ci torna per ogni singolo prodotto si fa la fila.
Stessa mossa, scala diversa.
`````

`````{tab} Superiore
Spezziamo $\mathbf{C}$ in tessere $T \times T$. Per calcolare una tessera si
scorre la dimensione $K$ a blocchi: a ogni passo si caricano *una volta* un
blocco $T \times T$ di $\mathbf{A}$ e uno di $\mathbf{B}$ nella shared memory,
e si accumulano nella tessera tutti i $T \times T$ prodotti che quei due
blocchi generano, prima di passare al blocco $K$ successivo. Ogni valore
caricato serve $T$ moltiplicazioni invece di una: il **fattore di riuso** (la
grandezza già incontrata nella sezione sulla memoria) è $T$. Le letture dalla
HBM scendono da $2 M N K$ a circa $2 M N K / T$ elementi, e l'intensità
aritmetica sale da $\tfrac14$ a

$$
I_\text{tiled} \approx \frac{T}{4} \ \text{FLOP/byte}.
$$

Con $T = 32$ sono $8$ FLOP/byte: trentadue volte l'intensità della versione
ingenua, e sul roofline la tessera scivola di parecchio verso destra.

Il confronto, però, va portato fino in fondo, perché la conclusione non è
quella che ci si aspetta: **otto FLOP/byte non bastano ancora**. Il ginocchio
del roofline sta a $\approx 10$
con i CUDA core in `float32` e a $\approx 161$ con i tensor core in `float16`
su A100: otto è a sinistra di entrambi, quindi questa tessera, da sola, è
ancora memory-bound. E il tiling in shared memory non può cavarsela da sé:
servirebbe $T \ge 41$ per superare il primo ginocchio e, in `float16` (dove
$I \approx T/2$), $T \ge 323$ per superare il secondo, cioè una tessera che
occuperebbe oltre 400 KB contro i 164 KB di shared memory configurabile di una
A100. E non è questione di scegliere meglio la taglia: la tessera più grande
che in quei 164 KB ci sta è $204 \times 204$ (due blocchi da $T^2$ elementi a
2 byte l'uno), e dà $I \approx 102$. **Nessuna** tessera in shared memory
arriva a 161, e quella $128 \times 128$ dei GEMM industriali si ferma a 64.

La domanda diventa allora chi li salvi davvero, quei GEMM, dato che veloci lo
sono. La risposta sta un piano più giù ed è la **cache L2**. Il modello usato
finora (ogni lettura che esce dall'SM arriva fino alla HBM) è lo stesso modello
crudo del kernel ingenuo, e sbaglia allo stesso modo: le tessere di
$\mathbf{A}$ e di $\mathbf{B}$ che blocchi diversi si portano sul tavolo sono
*le stesse*, e a servirle è la L2 senza disturbare la memoria. Il conto, su un
GEMM $8192 \times 8192 \times 8192$ in `float16` con tessere $128 \times 128$:
nel modello crudo dalla HBM escono $2MNK/T$ elementi, cioè circa **17 GB**, che
a $1{,}935$ TB/s valgono $8{,}9$ ms contro i $3{,}5$ ms di calcolo dei tensor
core; con il riuso in L2 dalla HBM $\mathbf{A}$ e $\mathbf{B}$ escono una volta
sola e $\mathbf{C}$ ci rientra una volta sola, cioè circa **400 MB** e
$0{,}21$ ms. Stesso kernel, stessi FLOP: nel primo conto è bloccato dalla
memoria, nel secondo dai tensor core.

E allora a che serve il **secondo livello di tessere nei registri** che i GEMM
veri impilano davvero sotto il primo? Non a spostare il punto sul roofline
della HBM: lì il traffico lo decide soltanto la tessera in shared memory, e la
micro-tessera nei registri non ne cambia un byte. Serve a un problema diverso e
altrettanto reale, un piano più su: la **banda della shared memory**. Ogni SM
la serve con 32 banchi da 4 byte per colpo di clock, cioè 128 byte per ciclo,
che su una A100 fanno circa 19 TB/s aggregati (la stessa cifra che il paper di
FlashAttention attribuisce alla SRAM on-chip). Per alimentare 312 TFLOP/s di
tensor core servono dunque **16 FLOP per ogni byte letto dalla shared**, e un
thread che vada a prendersi i due operandi di ogni singolo prodotto ne fa
$0{,}5$: due FLOP ogni quattro byte in `float16`. Con una micro-tessera
$R \times R$ tenuta nei registri, invece, ogni thread legge $2R$ valori e ne
ricava $R^2$ prodotti, cioè $R/2$ FLOP per byte. È la stessa aritmetica del
tiling, con la shared al posto della HBM e i registri al posto della shared: il
riuso si ricompra un piano più giù, ed è possibile per la ragione vista nella
sezione sulla memoria, che il register file di un SM è il banco on-chip più
capiente che ci sia.

Il fatto generale, più della gerarchia in sé, è questo: **c'è un roofline per ogni livello della piramide**, ciascuno con
la sua banda e il suo ginocchio, e ogni livello di tiling esiste per superare
il proprio. Quanto al tetto, l’$n/6$ del roofline è un *ideale* che
richiederebbe le tre matrici intere on-chip, e per fortuna non serve
raggiungerlo: l'intensità realmente raggiungibile non cresce con $n$, ma con la
**radice** della memoria veloce disponibile (è il risultato classico di Hong e
Kung sulla complessità di I/O {cite}`hongkung1981io`, che dà
$\Omega(n^3/\sqrt{M_\text{chip}})$ trasferimenti e quindi
$I = O(\sqrt{M_\text{chip}})$, dove $M_\text{chip}$ è la memoria veloce
disponibile e non va confusa con le $M$ righe di $\mathbf{A}$).
`````

Chi programma può vedere la struttura del tiling scritta per esteso, senza GPU,
in puro NumPy. Il codice qui sotto non è veloce (NumPy fa già i suoi prodotti
in modo ottimizzato) ma rende visibile *il nido di cicli*: si scorre il
risultato a tessere, e per ogni tessera si sommano i contributi dei blocchi
lungo $K$. Ogni `a` e `b` è un blocco «caricato in shared memory»; `a @ b` è il
lavoro che lo riusa, e alla fine il risultato coincide con la moltiplicazione
diretta `A @ B`. Chi non programma può guardarlo da lontano cogliendone la
forma: tre cicli uno dentro l'altro, e in mezzo la riga in cui il blocco
portato sul tavolo viene usato. Quella forma è tutto il messaggio, e il
paragrafo dopo il codice la riassume in una riga.

```python
import numpy as np

def matmul_a_blocchi(A, B, T=32):
    """Moltiplica A (M,K) per B (K,N) lavorando a tessere T×T.
    Stesso risultato di A @ B, ma esplicita il riuso: ogni blocco di A
    e di B, caricato una volta, serve tutti i prodotti della tessera."""
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N))
    for i in range(0, M, T):              # scorre le tessere di righe di C
        for j in range(0, N, T):          # scorre le tessere di colonne di C
            acc = np.zeros((min(T, M - i), min(T, N - j)))
            for k in range(0, K, T):      # somma sui blocchi lungo K
                a = A[i:i+T, k:k+T]       # blocco di A -> "shared memory"
                b = B[k:k+T, j:j+T]       # blocco di B -> "shared memory"
                acc += a @ b              # riuso: un blocco, molti prodotti
            C[i:i+T, j:j+T] = acc
    return C

A = np.random.randn(96, 80)
B = np.random.randn(80, 64)
print(np.allclose(matmul_a_blocchi(A, B), A @ B))
```

```text
True
```

Il triplo ciclo su tessere è lo scheletro di *ogni* moltiplicazione fra matrici
veloce, dalla CPU alla GPU. Quello che cambia da una macchina all'altra sono
solo due cose: chi esegue i conti di una tessera (su una GPU, i lavoratori di
una stessa squadra) e dove la tessera viene tenuta mentre la si usa (il ripiano
condiviso della squadra, la shared memory). La logica è questa.

## I tensor core: un intero prodotto in un colpo

Il tiling risolve il problema dei byte. Resta quello dei conti: anche
saturando la banda, ogni moltiplicazione la deve pur fare qualcuno. Dal 2017
quel qualcuno non è più il generico CUDA core, ma un'unità costruita apposta
per il prodotto tra matrici: il **tensor core**.

`````{tab} Elementare
Un ragioniere compila a mano una tabellina di quattro righe per quattro
colonne. Cella per cella è un lavoro noioso: sono sedici celle, e ognuna è una
somma di quattro prodotti, quindi in tutto sessantaquattro moltiplicazioni. Poi
gli arriva un timbro speciale, che stampa la tabellina intera in un colpo solo:
lo appoggia, preme, fatto. Il tensor core è quel timbro. Là dove una postazione
di calcolo normale fa una moltiplicazione per ogni battito del metronomo del
chip (il clock dell'inizio del capitolo, che batte più di un miliardo di volte
al secondo), il tensor core ne fa **sessantaquattro**.

Il guadagno sull'intera scheda però non è di sessantaquattro volte, ed è un
conto che si fa in testa. Sulla scheda che li ha introdotti i timbri erano uno
ogni otto postazioni normali: otto postazioni fanno otto conti a battito, il
timbro che sta al loro posto ne fa sessantaquattro, cioè **otto volte tanto**.
Sulle schede di oggi il rapporto è salito a una quindicina, perché i timbri
sono diventati più grandi e a ogni battito ne stampano di più.

C'è poi un secondo gesto, e spiega perché il timbro possa correre tanto. Per
fare le moltiplicazioni lavora con numeri «arrotondati», scritti con la metà
dello spazio (quelli della *mezza precisione* incontrata nella sezione
«Prestazioni e scala»: due byte invece di quattro, meno cifre e più velocità di
lettura); il totale che va accumulando, però, lo tiene nel formato lungo, per
non perdere per strada le cifre che contano. È il gesto di chi pesa gli
ingredienti a occhio, perché tanto un grammo non cambia il piatto, ma il conto
della spesa lo tiene all'ultimo centesimo, perché lì gli errori si sommano.

Un timbro così veloce, però, sposta il problema invece di chiuderlo. La
dispensa consegna alla velocità di sempre, e chi timbra sedici volte più in
fretta pretende sedici volte più conti da ogni cassetta che gli arriva. Più il
timbro corre, più conta portare sul tavolo poca roba e spremerla fino in fondo:
cassetta e timbro lavorano insieme, e nessuno dei due basta da solo. Il timbro,
poi, non lo prende in mano chi scrive il
programma: lo appoggiano da sé le librerie che PyTorch chiama ogni volta che
una rete gira, a patto che i numeri siano quelli corti.
`````

`````{tab} Superiore
Introdotti con l'architettura **Volta** (la GPU V100, 2017), i tensor core
calcolano in un solo colpo di clock un piccolo prodotto-matrice con accumulo,
della forma

$$
\mathbf{D} = \mathbf{A} \, \mathbf{B} + \mathbf{C},
$$

su tessere $4 \times 4$, cioè 64 moltiplicazioni-accumulo per colpo di clock
per unità (l'operazione è esposta al programmatore, a livello di warp, su
tessere $16 \times 16$). Le forme sono quelle di Volta: le generazioni
successive ne usano di più grandi, e da Hopper l'unità che emette l'istruzione
non è più il singolo warp ma un gruppo di quattro. Il cuore è la
**precisione mista** {cite}`micikevicius2018mixed`: gli ingressi $\mathbf{A}$ e
$\mathbf{B}$
sono a 16 bit (`float16` sulla V100; le architetture successive, da Ampere in
poi, aggiungono anche `bfloat16`), mentre l'accumulo di $\mathbf{C}$ e
$\mathbf{D}$ può restare
a `float32`: è la modalità dell'addestramento in precisione mista, così la
somma di molti prodotti non degrada (il silicio offre anche l'accumulo a 16
bit, usato talvolta in inferenza). È, non a caso, la
forma «generale» del GEMM delle BLAS
($\mathbf{C} \leftarrow \alpha \mathbf{A}\mathbf{B} + \beta \mathbf{C}$, dove
$\alpha$ e $\beta$ sono due numeri che pesano il prodotto nuovo e il valore già
accumulato: moltiplica *e* accumula) cablata nel silicio. Il guadagno è di
circa un ordine
di grandezza sul throughput di matmul rispetto ai CUDA core normali (un fattore
8 sulla V100, 16 sull'A100, 15 sulla H100): è
l'innalzamento di $P_\text{picco}$ che, come notava il roofline, sposta il
ginocchio verso destra e rende la banda ancora più decisiva. Non li programmi
tu direttamente: **cuBLAS** e **cuDNN** li usano dietro le quinte ogni volta
che una `nn.Linear` o una convoluzione girano su una GPU recente in mezza
precisione.
`````

## L'altra strada: far scorrere i dati invece dei conti

Il tiling e i tensor core sono la risposta della GPU a una domanda che si può
affrontare anche in un modo completamente diverso, e conviene vederlo perché
mette in prospettiva tutto il capitolo. La domanda è sempre quella: come si
moltiplicano due matrici muovendo il meno possibile.

`````{tab} Elementare
Un capannone con i banchi disposti a scacchiera, e su ogni banco un numero
solo, consegnato prima che il turno cominci: sono i numeri di una delle due
tabelle, uno per banco, e lì resteranno fermi fino alla fine. In una GPU sono i
dati a stare fermi in memoria e i calcolatori ad andarseli a prendere, e il
tiling accorcia quei viaggi senza toglierli. Qui è il contrario.

Da sinistra entra in ogni fila un carrello con i numeri dell'altra tabella, e
scorre lungo la fila da un capo all'altro. Dall'alto scende, di banco in banco,
un foglio con un totale ancora incompleto. Ogni postazione sa fare un gesto
solo: prende il numero che le passa davanti, lo moltiplica per il proprio,
aggiunge il risultato al foglio che sta scendendo e passa il foglio al banco
sotto. In fondo alla colonna il foglio esce compilato, ed è una casella del
risultato.

Che cosa resti fermo sul banco è una scelta di chi progetta, e cambia la
macchina che ne esce. In un altro capannone a restare fermo è il totale che si
va accumulando, e a passargli davanti sono i numeri di tutt'e due le tabelle.
Ha avuto più fortuna il primo, quello con il numero fermo sul banco, ed è
quello che gli acceleratori più noti hanno adottato.

Il guadagno sta tutto in quel passaggio da un vicino all'altro. Un numero
prelevato una volta sola dal magazzino attraversa un'intera fila di banchi e li
serve tutti, senza che nessuno debba andare a ripescarlo, e il totale si
costruisce camminando lungo la colonna come un pezzo su una catena di
montaggio. Una macchina fatta così si chiama **array sistolico**, dove *array*
è la schiera dei banchi; chi la inventò, alla Carnegie Mellon alla fine degli
anni Settanta, prese il resto del nome dal cuore, perché i dati attraversano la
scacchiera a ondate regolari come il sangue spinto dalla sistole.

Capannoni del genere si costruiscono davvero, ed è così che sono fatti i chip
pensati apposta per l'intelligenza artificiale: la **TPU** di Google ha una
scacchiera di 256 banchi per 256. Il guadagno più grosso si legge sul contatore
della luce, perché mandare un numero da un capo all'altro del chip costa più
corrente che moltiplicarlo, e qui i numeri camminano da un banco al vicino.

Il prezzo è la rigidità, e si vede appena il lavoro cambia. Se la tabella da
moltiplicare è più piccola della scacchiera, molti banchi passano il turno a
moltiplicare zeri. E se quello che c'è da fare non è una moltiplicazione fra
tabelle, il capannone non serve, e bisogna uscirne per farlo altrove. Una GPU è
più lenta di lei sul suo terreno e sa fare tutto il resto: è la stessa tensione
fra la lepre e il formicaio della prima sezione, spostata di un livello.
`````

`````{tab} Superiore

Un **array sistolico** {cite}`kung1982why` è una griglia di elementi di
elaborazione identici, ciascuno collegato soltanto ai vicini immediati e capace
di una sola operazione: moltiplicare due ingressi, sommare il prodotto a un
valore che gli arriva, e propagare ai vicini al ciclo successivo. Non c'è
memoria condivisa, non c'è arbitraggio, non c'è un file di registri da
indirizzare: il movimento dei dati è cablato nella topologia. Che cosa stia
fermo e che cosa scorra, però, non è unico: è la scelta di **dataflow**, e cambia la macchina. I nomi con cui queste
scelte si chiamano oggi vengono dalla tassonomia di Chen, Emer e Sze
{cite}`chen2016eyeriss`, non dagli array sistolici originali.

Nella variante *output stationary* è il totale a restare nell'elemento (un accumulatore interno) mentre entrambi gli operandi
scorrono. Nella variante *weight stationary*, che è quella adottata dagli
acceleratori più noti, l'elemento non tiene un totale: tiene un **peso**,
precaricato e fermo. Le attivazioni entrano da sinistra e attraversano le
righe; le **somme parziali** scendono di riga in riga raccogliendo un prodotto
per volta, e i totali completi escono in fondo all'array, in una memoria di
accumulatori posta sotto di esso. In entrambi i casi ogni valore letto una
volta dalla memoria esterna viene riusato lungo tutta una dimensione
dell'array, e il riuso non è ottenuto da una cache che *spera* di essere
colpita, ma dalla geometria.

La prima TPU di Google {cite}`jouppi2017datacenter` è la realizzazione più nota
del secondo schema: un array $256 \times 256$, cioè $65\,536$
moltiplicazioni-accumulo per ciclo di clock in una sola unità (le generazioni
successive usano più unità $128 \times 128$), con i pesi precaricati dall'alto,
i dati che entrano da sinistra e 4 MiB di accumulatori a 32 bit sotto la
matrice, che raccolgono una somma parziale da 256 elementi per ciclo. Il
confronto con la GPU non è «chi calcola di più»
ma «chi si muove di meno», ed è per questo che gli acceleratori dedicati
guadagnano soprattutto in **energia per operazione**, un tema che il capitolo
su MLOps riprende quando si tratta di pagare la bolletta.

Il prezzo della specializzazione è la rigidità. Un array sistolico è bravo
esattamente a una cosa. Se la matrice è più piccola dell'array, gran parte
degli elementi calcola zeri; se l'operazione non è un GEMM (una convoluzione
sparsa, un gather irregolare, un'operazione elemento per elemento), la
struttura non serve e bisogna uscire dall'array. La GPU, con la sua gerarchia
di memoria programmabile e i suoi CUDA core generici, perde in efficienza di
picco e guadagna in tutto il resto: è la stessa tensione fra lepre e formicaio
della prima sezione, spostata di un livello.

`````

## In pratica: forme «tonde» e mezza precisione

Quasi certamente non scriverai mai a mano una
moltiplicazione fra matrici: esistono librerie che la fanno meglio di quanto
potrebbe chiunque, sfruttando tessere a più livelli e tensor core in modi che
cambiano a ogni generazione di schede. Sono quelle che PyTorch chiama sotto
sotto ogni volta che una rete gira (**cuBLAS** per le matrici, **cuDNN** per le
convoluzioni, scritte da NVIDIA), più i due strumenti che quel codice lo
generano invece di averlo già scritto, **CUTLASS** e **Triton**
{cite}`tillet2019triton`.

Perché allora capire il tiling? Perché spiega due regole pratiche che spostano
davvero il cronometro, e che altrimenti sembrerebbero magia:

- **Dai alle matrici forme «tonde»**, cioè misure che siano multipli di numeri
  come 8, 16 o 64 invece di misure qualsiasi. Se le dimensioni sono multiple
  della tessera (e dei blocchetti che i tensor core divorano) le tessere si
  riempiono senza avanzi, e nessun lavoratore resta a lavorare su un bordo
  incompleto. È il motivo per cui conviene portare al multiplo di 8 più vicino
  la *dimensione nascosta* di un modello (quanti numeri usa per rappresentare
  al proprio interno una parola o un'immagine) o la *taglia del vocabolario*
  (quante parole diverse conosce): un guadagno spesso gratuito. Una forma
  «storta» lascia i tensor core mezzi vuoti.
- **Usa la mezza precisione**, cioè numeri scritti nella metà dello spazio, 16
  cifre binarie invece di 32 (una cifra binaria, un *bit*, è un sì o un no, e
  otto di fila fanno un byte: con 16 bit si scrivono numeri meno precisi, con
  32 più precisi). Occupano metà spazio e si leggono in metà tempo, e i tensor
  core esistono per loro: senza, girano a una frazione della propria potenza.
  Le quattro righe di `autocast` viste in «Prestazioni e scala» non sono un
  vezzo da datacenter, sono l'interruttore che accende il pezzo di silicio più
  veloce che hai.

Tutte e due queste regole promettono un guadagno quasi gratuito, e tutte e due
capita che non lo diano. Le ragioni sono due, e nessuna delle due riguarda la
tessera, che è la cosa a cui si dà la colpa.

`````{tab} Elementare
Succede di arrotondare le misure e di non vedere cambiare niente, e la prima
ragione è che non conta solo *quante* caselle ha una riga: conta **da che punto
della memoria la riga comincia**. Torniamo al furgone della sezione sulla
memoria, quello che consegna solo pacchi già ordinati per via: le consegne
piene partono solo dall'inizio di una via, mai da metà. Una tabella con le
misure giuste, ma il cui primo numero si trova a metà via, costringe a spezzare
ogni consegna in due, e la strada veloce si chiude lo stesso. Capita più spesso
di quanto si creda: per esempio quando si lavora su un ritaglio di una tabella
più grande invece che sulla tabella intera.

La seconda ragione non riguarda le tessere né le misure della tabella: riguarda
**quante officine restano ferme all'ultimo giro**, cioè quante ne restano
inutilizzate quando il lavoro non si divide in parti uguali (le officine sono
le unità in cui la GPU è divisa, quelle della prima sezione, ed è un
centinaio). Il lavoro si distribuisce a giri: una tessera a testa, e quando
hanno finito un'altra a testa. Se le tessere da calcolare sono, poniamo,
centodieci, le prime cento vanno in un giro pieno e le dieci rimaste ne
occupano un secondo tutto per loro, con novanta officine a guardare. Due giri
per fare poco più del lavoro di uno: il tempo quasi raddoppia. È il motivo per
cui certe misure «tonde» vanno peggio di misure vicine, e chi non conosce
questo secondo effetto dà la colpa alla tessera, che non c'entra.
`````

`````{tab} Superiore
La prima precisazione: il requisito vero non è sulle dimensioni logiche ma
sull’**allineamento in byte** (multipli di 16 byte sulla dimensione
principale), quindi una matrice con $M$, $N$ e $K$ multipli di 8 ma con un
*passo di riga* storto (lo *stride*, cioè la distanza in memoria fra l'inizio
di una riga e l'inizio della successiva, che in una vista o in una fetta non
coincide con la larghezza) esce comunque dalla strada veloce.

La seconda: esiste una quantizzazione gemella che non dipende dalla tessera ma
dal **numero di SM**, la *wave quantization*. Se il numero di tessere da
calcolare supera di poco un multiplo degli SM disponibili (108 su A100),
l'ultimo giro impegna pochissime officine e tutte le altre restano ferme: il
tempo raddoppia quasi. È per questo che certe taglie di batch «tonde» vanno
peggio di taglie vicine, e chi non conosce questo secondo effetto lo attribuisce
alla tessera, che non c'entra.
`````

Il tiling, insomma, tiene insieme i due limiti che il grafico della sezione
precedente (il roofline, quello che dice se sei bloccato dal magazzino o dai
cuochi) metteva uno di fronte all'altro: taglia i byte da spostare, perché
riusa quel che ha già sul tavolo, e in cambio dà da lavorare ai tensor core.
La stessa idea, cioè riorganizzare un calcolo per non tornare mai a rileggere
dalla memoria lontana ciò che si può tenere vicino, applicata ai confronti fra
le parole di un testo dà la **FlashAttention** {cite}`dao2022flashattention`
della prossima sezione. La moltiplicazione fra matrici è il primo, e più puro,
esempio di una lezione che tornerà a ogni pagina.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Una rete neurale, vista dall'hardware, è quasi soltanto **moltiplicazioni fra
  tabelle di numeri**. Quella routine ha un nome che si incontra ovunque,
  **GEMM**, ed è probabilmente il pezzo di codice più ottimizzato della storia
  dell'informatica.
- Farla nel modo ovvio, una casella del risultato alla volta, vuol dire correre
  in dispensa a riprendere gli stessi ingredienti centinaia di volte. I conti
  che si fanno a ogni viaggio sono pochi, i viaggi tantissimi: si finisce
  bloccati dal magazzino.
- La cura è il **tiling**: portare sul tavolo di lavoro un blocchetto di
  ciascuna tabella e usarlo per tutti i prodotti che può servire prima di
  buttarlo. Un viaggio invece di cento. Più grande il blocchetto, meglio è, ma
  sul tavolo ci sta poco. I programmi veri lo fanno allora **due volte**, a due
  scale: blocchetti grandi sul tavolo, che tagliano i viaggi in dispensa, e
  blocchetti piccolissimi in mano a ciascun lavoratore, che di viaggi non ne
  tolgono nemmeno uno ma sciolgono la fila al tavolo.
- I **tensor core** sono il timbro che stampa un pezzo intero di tabellina in
  un colpo solo, sessantaquattro moltiplicazioni per battito, con i numeri
  arrotondati ma il totale tenuto preciso. È il pezzo di silicio più veloce di
  una GPU, e si accende dicendo a PyTorch di usare la mezza precisione.
- Ci si può muovere anche dall'altro capo: invece di andare a prendere i dati,
  si può far **scorrere** i dati fra postazioni vicine, come su una catena di
  montaggio, dove il totale scende lungo la colonna raccogliendo un pezzo per
  postazione. Si chiama **array sistolico** {cite}`kung1982why` ed è la scelta
  della TPU di Google {cite}`jouppi2017datacenter`: bravissima a fare questa
  cosa, inadatta a tutto il resto.
- Le due regole pratiche che restano, e che valgono anche per chi non scriverà
  mai un programma per GPU: dare alle tabelle misure **tonde** (multipli di 8 o
  16) e usare la **mezza precisione**, i numeri scritti nella metà dello
  spazio. Sono guadagni quasi sempre gratuiti, e quando non arrivano la colpa
  non è della tessera: o la tabella comincia nel punto sbagliato della memoria,
  o l'ultimo giro di lavoro lascia quasi tutte le officine a guardare.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- **GEMM** (*GEneral Matrix Multiply*) è il cuore di calcolo di ogni rete: il
  prodotto $\mathbf{C} = \mathbf{A}\mathbf{B}$ con $\mathbf{A}$ di forma
  $(M,K)$ e $\mathbf{B}$ di forma $(K,N)$ costa circa
  $2 M N K$ FLOP.
- La versione **ingenua** rilegge dalla HBM le stesse righe e colonne
  all'infinito: nel modello senza cache l'intensità aritmetica resta ferma a
  $\tfrac14$ FLOP/byte, indipendente dalla taglia (nella realtà la L1 e la L2
  recuperano qualcosa, ma è riuso *sperato*, non garantito dal programma). In
  ogni caso, profondamente **memory-bound**.
- Il **tiling** spezza $\mathbf{C}$ in tessere e carica i blocchi di
  $\mathbf{A}$ e $\mathbf{B}$ in **shared memory** una volta sola, riusandoli:
  con tessere $T \times T$ l'intensità sale a circa $T/4$ FLOP/byte. Con
  $T = 32$ fa 8, che è ancora **a sinistra** di ogni ginocchio (10 con i CUDA
  core, 161 con i tensor core su A100), e nessuna tessera che stia nei 164 KB di
  shared di una A100 ci arriva: la più grande è $204 \times 204$, cioè
  $I \approx 102$. A tenere i GEMM veri lontani dal muro della HBM è la **cache
  L2**, che serve le tessere che i blocchi si ripassano (su un GEMM $8192^3$ in
  `float16` il traffico scende da 17 GB a circa 400 MB, e il tempo da $8{,}9$ a
  $0{,}21$ ms contro $3{,}5$ ms di calcolo). Il **secondo** livello di tessere,
  quello nei registri, risolve un problema diverso: la **banda della shared
  memory**, circa 19 TB/s su A100 contro i 16 FLOP/byte che i tensor core
  pretendono. C'è un roofline per ogni livello della piramide, e ogni tiling
  supera il proprio. L’$n/6$ è un tetto ideale; il raggiungibile cresce come
  $\sqrt{M_\text{chip}}$, non come $n$ {cite}`hongkung1981io`.
- I **tensor core** (dal 2017, Volta) eseguono un piccolo prodotto-matrice con
  accumulo $\mathbf{D} = \mathbf{A}\mathbf{B} + \mathbf{C}$ per colpo di clock
  (64 FMA per unità, su tessere $4\times4$ nella forma Volta), in **precisione
  mista** {cite}`micikevicius2018mixed` (ingressi 16 bit, accumulo 32 bit):
  circa un ordine di grandezza di throughput in più. `cuBLAS`/`cuDNN` li usano
  da soli.
- L’**array sistolico** {cite}`kung1982why` risolve lo stesso problema
  dall'altro capo: invece di andare a prendere i dati, li fa **scorrere** fra
  unità adiacenti, e il riuso è nella geometria invece che in una cache. La TPU
  {cite}`jouppi2017datacenter` usa la variante *weight stationary*: pesi
  precaricati e fermi, attivazioni da sinistra, somme parziali che scendono
  verso gli accumulatori sotto l'array. Massima efficienza sul GEMM, rigidità
  su tutto il resto.
- Raramente scriverai un GEMM a mano, ma capire il tiling spiega perché le
  forme «tonde» (allineamento a 16 byte, più che multipli logici di 8/16) e la
  mezza precisione vanno più veloci; e perché esiste una *wave quantization*
  legata al numero di SM, che le forme tonde non curano.
- Riuso in shared memory + tensor core: la stessa ricetta tornerà, applicata
  all'attenzione, in **FlashAttention** {cite}`dao2022flashattention`.
```
`````
