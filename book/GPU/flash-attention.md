# FlashAttention: l'attenzione che non spreca memoria

Chiedere a un modello di riassumere un romanzo intero, o di rispondere su un
contratto di cento pagine, fino a pochi anni fa era impensabile, e non per una
carenza di intelligenza. L'ostacolo era di portata: non mancava la capacità di
calcolare, mancava la possibilità di far arrivare abbastanza in fretta i numeri
fin sotto alle unità che li calcolano. Di mezzo c'è una tabella che cresce con
il *quadrato* della lunghezza del testo. Raddoppia le parole e quella tabella
quadruplica; moltiplicale per dieci e diventa cento volte più grande. A un
certo punto non ci sta più nella memoria della GPU, e anche quando ci sta,
spostarla avanti e indietro costa così tanto tempo da rendere tutto
insopportabilmente lento. Questa sezione racconta l'idea (sorprendentemente
semplice nella sostanza) che ha fatto saltare quel muro.

Serve prima sapere che cos'è l'attenzione, il meccanismo su cui i modelli
linguistici sono costruiti. Il capitolo sui **Transformer** le è dedicato per
intero e ne racconta il *perché*; qui basta il *che cosa*, in tre passi, perché
sono i tre passi che si tratta di eseguire in fretta. Primo: ogni parola del testo
viene confrontata con tutte le altre, e da ogni confronto esce un punteggio di
somiglianza. È la grande tabella. Secondo: i punteggi di ciascuna riga vengono
trasformati in percentuali che sommano a cento, e questa trasformazione ha un
nome che ricorrerà per tutta la sezione, la **softmax**. Terzo: quelle
percentuali dicono in che proporzione mescolare. Ogni parola si porta dietro
una manciata di numeri, che è il modo in cui il modello dice che cosa
significa lì dentro; se «salta» ha preso il 70% su «gatto» e il 20% su «muro»,
il suo risultato è fatto per sette decimi dei numeri di «gatto» e per due di
quelli di «muro». Quello che ne esce è, per ogni parola, un riassunto del resto
della frase pesato su quanto ciascuna le interessa.

Là il problema sarà *quale* informazione l'attenzione raccoglie; qui è un
altro, tutto hardware: *come* si eseguono quei tre passi senza affogare nel
traffico di memoria.

`````{tab} Elementare
Il punto da tenere a mente è uno solo, ed è una questione di conteggio: se le
parole sono mille e ognuna va confrontata con tutte, i confronti sono un
milione. Con diecimila parole diventano cento milioni. La tabella di quei
confronti è l'oggetto ingombrante di tutta questa sezione: nessuno la vuole,
serve solo di passaggio, e proprio per questo scriverla è uno spreco.
`````

`````{tab} Superiore
In simboli, il ripasso sta in una riga:

$$
\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\!\big(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}\big)\mathbf{V},
$$

dove le righe di $\mathbf{Q}$ sono le *query* (una per posizione), quelle di
$\mathbf{K}$ le *key* e quelle di $\mathbf{V}$ i *value*; $d_k$ è la
dimensione delle key, e la
divisione per $\sqrt{d_k}$ tiene i punteggi in una scala in cui la softmax non
satura. Le due matrici $N \times N$ che compaiono qui dentro, i punteggi e le
loro versioni normalizzate, sono ciò di cui parleremo per il resto della
sezione.
`````

## Il problema è la memoria, non i conti

Il primo istinto è pensare che l'attenzione sia lenta perché fa *tanti conti*.
È vero solo a metà. Il vero collo di bottiglia, come quasi sempre su una GPU,
non è il calcolo: è il **movimento dei dati** (la stessa lezione della
gerarchia di memoria e del roofline delle sezioni precedenti).

`````{tab} Elementare
Mille parole, ognuna confrontata con ogni altra, fanno un milione di confronti:
una tabella di mille righe per mille colonne. Fin qui, tanti conti ma niente di
drammatico. Il guaio è *dove* metti quella tabella. È troppo grande per il
tavolo di lavoro veloce, il ripiano accanto ai calcolatori, quindi la GPU la
scrive nel magazzino lontano e poi deve tornare a prenderla per fare il secondo
passo (le percentuali), riscrivere anche quelle, e tornare *di nuovo* per il
terzo (la media). Quattro viaggi al magazzino per una tabella enorme che, alla
fine, non serviva nemmeno tenere: era solo un passaggio intermedio.

È come dover tenere la sfoglia in un capannone lontano perché sul tavolo non ci
sta, e correre fin là a ogni operazione: una volta per portarcela, una per
andarla a riprendere e tagliarla, una per riportarci i pezzi, una per andarli
a riprendere e infornarli. Il tempo non se ne va nel taglio: se ne va nella
corsa.
`````

`````{tab} Superiore
Per una sequenza di lunghezza $N$ e teste di dimensione $d_k$, l'attenzione
materializza due matrici $N \times N$: i punteggi
$\mathbf{S} = \mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}$ e i
pesi $\mathbf{P} = \text{softmax}(\mathbf{S})$. Il calcolo è $O(N^2 d_k)$ FLOP,
ma il dato che uccide le prestazioni è la **memoria**: $\mathbf{S}$ e
$\mathbf{P}$ occupano $O(N^2)$ byte e
vengono scritte e rilette dalla HBM più volte (produci $\mathbf{S}$, la rileggi
per la softmax, rileggi $\mathbf{P}$ per il prodotto con $\mathbf{V}$). Un
numero concreto: con
$N = 8192$, una sola matrice $\mathbf{S}$ ha $N^2 \approx 67$ milioni di
elementi; in
`float16` (2 byte) sono circa $134$ MB (*per testa, per strato*). La memoria
cresce quadraticamente, e con essa il traffico verso la HBM.

Sul roofline questa è l'operazione tipicamente **memory-bound**, e vale la pena
mettere l'imputato giusto al banco, perché la spiegazione che si legge più
spesso (i due matmul sarebbero compute-bound, e a rovinare tutto sarebbe la
softmax in mezzo) è sbagliata di suo. Un matmul che produce un'uscita
$N \times N$ ha intensità limitata dalla propria **dimensione interna**, che
qui è $d_k$, cioè 64 o 128: al crescere di $N$ il traffico è dominato dalla
scrittura di $\mathbf{S}$ e l'intensità tende a $2N^2 d_k / (2N^2) = d_k$
esatti. Con
$N = 8192$, $d_k = 64$ in `float16` fa 63 FLOP/byte, contro un ginocchio di
161 su A100 e 295 su H100: **anche i due matmul, da soli, sono memory-bound**,
e userebbero al più il 39 % del picco. Con $d_k = 128$ si arriva a 124, e resta
sotto il ginocchio di entrambe le schede.

Le operazioni della softmax in mezzo (gli esponenziali, le riduzioni per riga,
le scritture e riletture della matrice $N \times N$) hanno intensità quasi
nulla e dimezzano ancora il conto. Vale la pena farlo per esteso, perché è un
bilancio che si rifà in due righe. Sempre con $N = 8192$ e $d_k = 64$ in
`float16`: i conti sono i $2N^2 d_k$ del primo matmul più gli altrettanti del
secondo, più una manciata di operazioni per elemento della softmax, in tutto
circa $17{,}5$ GFLOP; i byte sono quattro passaggi della matrice $N \times N$
(scrivi $\mathbf{S}$, la rileggi, scrivi $\mathbf{P}$, la rileggi) a 2 byte per
elemento, più le briciole di $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ e
dell'uscita, in tutto circa $541$ MB. Il rapporto fa **32 FLOP/byte**: è lì che
sta l'attenzione intera, non fusa. La conclusione onesta è che nella forma
standard, in attenzione,
**niente** è compute-bound; e quindi la cura non è fondere la softmax con i
matmul, è non far mai atterrare $\mathbf{S}$ in HBM. Non serve una GPU più
potente nei FLOP: serve *non spostare* quei byte.
`````

## L'idea: lavorare a tessere, mai scrivere la matrice

La svolta arriva nel 2022 da Tri Dao e colleghi con **FlashAttention**
{cite}`dao2022flashattention`. La loro osservazione è che la matrice
$N \times N$ è solo un *intermedio*: alla fine ci serve l'output, non la
tabella dei punteggi. E allora perché scriverla? L'algoritmo è **IO-aware**
(«IO» sta per *input/output*, l'entrata e l'uscita dei dati dalla memoria):
ottimizza il movimento dei dati, non i conti, e (dettaglio cruciale) dà il
**risultato esatto**, non un'approssimazione.

Due ingredienti lo rendono possibile ({numref}`fig-flash-attention`). Il primo
è il **tiling**, cioè lo stesso «carica una tessera, riusala» della sezione
precedente. Qui le tessere si ritagliano non nella tabella dei confronti, che
non esisterà mai, ma nell'elenco delle parole di partenza: si tiene ferma una
manciata di parole e si fa scorrere davanti a loro tutto il resto, un
blocchetto per volta. Il secondo ingrediente è la **online softmax**, che
permette di calcolare le percentuali *a pezzi* invece che tutte insieme.

```{figure} ../figures/flash-attention-tiling.svg
:name: fig-flash-attention
:alt: "A sinistra la matrice dei punteggi S uguale Q per K trasposto, N per N, disegnata come griglia e barrata da una grande X: la matrice che FlashAttention non scrive mai nella memoria HBM. A destra lo schema: una shared memory on-chip tiene un tile fisso di Q e un blocco corrente di K e V; sotto, i blocchi di K e V scorrono uno per volta dalla HBM verso la shared memory; un accumulatore aggiorna a ogni blocco l'output O e le due statistiche del softmax, il massimo corrente m e la somma corrente l; alla fine l'uscita è O diviso l, ed è esatta."
:width: 90%

La grande tabella dei confronti non viene mai scritta (a sinistra, sbarrata).
Sul tavolo di lavoro veloce resta ferma una manciata di parole da elaborare, e
il resto del testo le scorre davanti a blocchetti; a ogni blocchetto si
aggiorna il risultato e due soli numeri di riepilogo, che bastano a rifare le
percentuali alla fine. Il risultato è identico a quello del calcolo in un colpo
solo.
```

`````{tab} Elementare
Il trucco è non costruire mai la tabella gigante. Tieni ferma sul tavolo di
lavoro una manciata di parole, quelle di cui ti stai occupando adesso (una
*tessera*, come le tessere della sezione precedente), e fai scorrere davanti a
loro tutto il resto del testo a blocchetti: prendi le prime parole con cui
confrontarle, calcoli i punteggi, aggiorni il risultato; butti via quel
blocchetto, prendi il successivo, e così via fino alla fine. Sul tavolo, in
ogni istante, c'è solo un pezzetto piccolo. La tabella da un milione di caselle
non viene mai scritta per intero da nessuna parte: esiste un blocchetto alla
volta, e sparisce appena hai finito di usarlo. Meno viaggi al magazzino, stesso
identico risultato.

Il prezzo si paga più tardi, ed è giusto dirlo subito. Quando la rete impara,
dopo aver letto il testo in avanti rifà la strada all'indietro per capire quali
numeri correggere, e in quel secondo passaggio le servirebbero proprio i
punteggi che sono stati buttati: non avendoli, se li rifà. Qualche conto in più,
quindi, in cambio di molti viaggi in meno. È un baratto che conviene, perché i
conti sono la cosa che una GPU fa quasi gratis e i viaggi sono quella che le
costa.

Resta un'insidia da risolvere, ed è la più bella di questa sezione: come fai a
calcolare delle *percentuali* se non hai ancora visto tutti i punteggi? Per
fare una percentuale ti serve il totale, e il totale lo conosci solo alla fine.
La risposta è la *online softmax* del prossimo passaggio.
`````

`````{tab} Superiore
Formalmente, si spezzano $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ in blocchi di
righe. Fissato il blocco
di query $\mathbf{Q}_i$, si itera sui blocchi $(\mathbf{K}_j, \mathbf{V}_j)$:
si carica $\mathbf{K}_j, \mathbf{V}_j$ in
**shared memory**, si calcola il tile di punteggi
$\mathbf{S}_{ij} = \mathbf{Q}_i \mathbf{K}_j^\top/\sqrt{d_k}$, e si
aggiorna l'output *sul posto*, senza mai scrivere l'intera matrice $\mathbf{S}$
in HBM.
Qui $\mathbf{Q}_i$ è il blocco di query corrente (quello che resta fermo sul
tavolo) e $\mathbf{K}_j, \mathbf{V}_j$ il blocco di key e value in transito,
per cui $\mathbf{S}_{ij}$ è la tessera
di punteggi che nasce dal loro incontro: $B_r \times B_c$, cioè $B_r$ righe di
query per $B_c$ chiavi (le due misure del blocchetto, che il kernel sceglie in
base a quanta shared memory ha), e non l'intera riga di $\mathbf{S}$.
(Quest'ordine dei cicli, con il blocco di query fermo e $\mathbf{K},\mathbf{V}$
che scorrono, è
quello reso canonico dalla seconda versione dell'algoritmo, che incontreremo a
breve; l'articolo del 2022 li annidava al contrario, ma l'idea non cambia.) La
memoria on-chip trattiene solo i tile correnti; la HBM vede scorrere
$\mathbf{K},\mathbf{V}$ una
volta per ogni blocco di query, e la matrice $\mathbf{S}$ mai. La **memoria
extra**
scende così da $O(N^2)$ a $O(N)$: da scrivere restano solo l'output e le
statistiche di riga.

Sui FLOP, invece, va detta una cosa che si sente ripetere al contrario. In
avanti i conti sono quelli di prima, a meno del riscalamento dell'accumulatore
a ogni blocco (ed è proprio quel di più, non-matmul, che FlashAttention-2 andrà
a limitare). All'indietro no: non avendo salvato $\mathbf{S}$ e $\mathbf{P}$,
il `backward` deve
**ricalcolarle** da $\mathbf{Q}, \mathbf{K}, \mathbf{V}$, ed è esattamente il
motivo per cui bastava
salvare l'output e due statistiche per riga. Il conto: in avanti $4N^2 d_k$
FLOP, all'indietro $8N^2 d_k$ nella versione standard e $10 N^2 d_k$ qui, cioè
**un quarto in più sul passaggio all'indietro** e un sesto in più sul totale.
Il paper misura $+12{,}9\,\%$ di FLOP su GPT-2 medium (75,2 contro 66,6 GFLOP)
a fronte di un traffico verso la HBM che scende di circa nove volte (4,4 contro
40,3 GB) e di un tempo che scende di quasi sei (7,3 contro 41,7 ms), e lo scrive
senza giri di parole:
«even with the increased FLOPs due to recomputation».

Il baratto è dunque **calcolo in cambio di traffico**, ed è lo stesso baratto
del *gradient checkpointing*, quello con cui si ricalcolano le attivazioni
invece di conservarle. Conviene per una ragione precisa: i FLOP ricomprati sono
matmul, cioè la cosa che i tensor core fanno a costo quasi nullo, mentre i byte
risparmiati sono accessi alla HBM, cioè la risorsa scarsa. È la stessa mossa
che si ritroverà nel pipeline parallelism della prossima sezione, e che il
capitolo sui modelli a spazio di stati usa per Mamba: vale la pena riconoscerla
sotto i tre nomi diversi.

Anche il traffico verso la HBM crolla: il paper lo conta in
$\Theta(N^2 d_k^2 / M)$ accessi, dove $M$ è la taglia della memoria on-chip,
contro il $\Theta(N d_k + N^2)$ dell'attenzione standard. Resta quadratico in
$N$, ma diviso per un fattore $M/d_k^2$ che si può mettere in cifre, perché il
paper quantifica $M$: 192 KB di SRAM per SM su A100, cioè poco meno di
centomila elementi in `float16`. Il fattore vale allora una sestina con
$d_k = 128$ e una ventina con $d_k = 64$: su un carico memory-bound è tanto. È
l'idea del tiling in shared memory del GEMM, applicata
all'attenzione: caricare una volta, riusare in tanti, non tornare al
magazzino.

Il nodo tecnico è che la softmax *non* è elemento-per-elemento: normalizza per
righe, e la normalizzazione richiede in teoria di aver già visto tutti i
punteggi della riga. Scorrere $\mathbf{K}$ a blocchi significa vedere i
punteggi un pezzo per volta, e qui entra la online softmax.
`````

### La online softmax, con i numeri

Il perno di tutto è calcolare le percentuali vedendo i punteggi **a blocchi**,
senza mai averli tutti sotto gli occhi insieme, e ottenendo comunque il
risultato esatto. Bastano due numeri di riepilogo, che la scheda Elementare
racconta come due foglietti e il conto qui sotto chiama $m$ (il punteggio più
alto visto finora) e $l$ (il totale accumulato finora). In
{numref}`fig-flash-attention-blocchi` si vedono aggiornarsi blocco dopo blocco,
insieme alla cosa che conta di più: le celle fuori dalla finestra restano
vuote, perché quella tabella non viene mai scritta da nessuna parte.

```{figure} ../figures/flash-attention-blocchi.svg
:name: fig-flash-attention-blocchi
:alt: Una riga di otto punteggi divisa in quattro blocchi da due: una finestra scorre da sinistra a destra e in ogni istante mostra i numeri di un solo blocco, mentre fuori le celle restano vuote perché la matrice dei punteggi non viene mai scritta. Sotto, una tabella si riempie riga per riga con il massimo del blocco, il massimo corrente m, il fattore di riscalatura alfa, la somma corrente l e l'output accumulato O: quando arriva un massimo più grande alfa scende sotto 1 e l'accumulatore viene riscalato. Alla fine O diviso l coincide con la softmax calcolata in un colpo solo.
:width: 95%

Gli stessi due foglietti, al lavoro su una riga di otto punteggi letti a due a
due: sono $1, 3, 2, 4, 1, 0, 5, 2$, e i valori che si portano dietro sono
$1, 4, 2, 5, 3, 0, 6, 2$. In alto la finestra della memoria veloce: in ogni
istante contiene un solo blocchetto, e tutto il resto della riga resta vuoto,
perché quella tabella non viene mai scritta da nessuna parte. Sotto, la tabella
di marcia: a ogni blocchetto si aggiornano il record ($m$, il punteggio più
alto visto finora) e il totale ($l$), insieme al risultato che si sta
accumulando ($\mathbf{o}$). Quando arriva un punteggio più alto del record, cioè alla
seconda e alla quarta riga, il fattore $\alpha$ (qui $0{,}368$, perché il
record sale di un punto) riesprime rispetto al nuovo record quello che era già
stato messo da parte. Alla fine $\mathbf{o}$ diviso $l$ vale $5{,}257$, lo stesso numero
che darebbe il calcolo fatto in un colpo solo su tutti e otto.
```

`````{tab} Elementare
Partiamo dal gesto facile: fare un totale a rate. Devi dire quanto pesa ogni
sacco rispetto al totale di tutti, ma sulla bilancia ne stanno due per volta.
Tieni un foglietto con «totale finora». Primo mucchietto, 10 e 30: il foglietto
dice 40. Secondo mucchietto, 20 e 40: il foglietto dice 100. A quel punto
dividi ciascun sacco per 100 e hai le percentuali (10%, 30%, 20%, 40%), che
sono esattamente quelle che avresti ottenuto stendendo tutti i sacchi per terra
in una volta sola. Nessuna approssimazione: la stessa somma, fatta a rate.

I foglietti però sono **due**, non uno, e il secondo è la parte meno ovvia. Nel
calcolo vero i punteggi, prima di essere sommati, non vengono presi così come
sono: si passa prima per un'operazione che li ingigantisce, e che ha una regola
semplice, **ogni punto in più moltiplica per 2,7 circa**. Un punteggio di due punti più alto pesa quindi $2{,}7 \times 2{,}7$, più di
sette volte tanto; dieci
punti più alto pesa ventiduemila volte tanto. Con punteggi anche moderatamente
alti si arriva a numeri che il computer non riesce più a scrivere.

Il rimedio è quello delle classifiche: invece del punteggio assoluto si segna
la distanza dal primo, «a tre punti dal record». Serve dunque un secondo
foglietto con **il punteggio più alto visto finora**, e ogni cosa si misura
rispetto a lui.

Da qui l'unico momento in cui la faccenda si fa interessante. Se in un
mucchietto salta fuori un punteggio più alto del record, il totale accumulato
era espresso rispetto al vecchio record e va riespresso rispetto al nuovo. Ed è
qui che quel «2,7 a punto» torna utile: se il record sale di un punto, tutto
quello che si era già messo da parte va diviso per 2,7, cioè moltiplicato per
0,37. Una moltiplicazione sola, si aggiorna il foglietto e si tira avanti. Alla
fine si divide per il totale, e i pesi che vengono fuori sono *esattamente*
quelli del calcolo fatto in un colpo unico.
`````

`````{tab} Superiore
Facciamo il conto a mano su una riga di quattro punteggi
$\mathbf{s} = (1, 3, 2, 4)$ (i valori di
$\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}$ per una query contro quattro
key). Il calcolo *in un colpo solo*, con la solita stabilizzazione che sottrae il
massimo per non far esplodere gli esponenziali:

$$
m = \max(\mathbf{s}) = 4, \qquad
l = \sum_i e^{s_i - m} = e^{-3}+e^{-1}+e^{-2}+e^{0} \approx 1{,}553,
$$

da cui i pesi softmax $(0{,}032,\ 0{,}237,\ 0{,}087,\ 0{,}644)$.

Ora *a blocchi di due*, $[1,3]$ poi $[2,4]$, tenendo aggiornati $m$ e $l$:

- **Blocco 1** $[1,3]$:  $\ m_1 = 3$,  $\ l_1 = e^{1-3}+e^{3-3} = e^{-2}+1 \approx 1{,}135$.
- **Blocco 2** $[2,4]$:  il massimo del blocco è $4$, quindi $m_2 = \max(3,4) = 4$.
  Il vecchio totale va **ri-scalato** al nuovo massimo con il fattore di
  correzione $\alpha = e^{m_1 - m_2} = e^{-1} \approx 0{,}368$:

$$
l_2 = \alpha\, l_1 + \big(e^{2-4}+e^{4-4}\big)
    = 0{,}368 \cdot 1{,}135 + e^{-2}+1 \approx 1{,}553.
$$

Il totale $l_2 \approx 1{,}553$ coincide **esattamente** con la somma calcolata in
un colpo solo: la online softmax dà gli stessi pesi. In generale, arrivando un
nuovo blocco con massimo locale $\tilde m$, le regole di aggiornamento sono

$$
m^{\text{new}} = \max(m, \tilde m), \quad
l^{\text{new}} = e^{\,m - m^{\text{new}}}\, l + \!\sum_{i \in \text{blocco}}\! e^{\,s_i - m^{\text{new}}}, \quad
\mathbf{o}^{\text{new}} = e^{\,m - m^{\text{new}}}\, \mathbf{o} + \!\sum_{i \in \text{blocco}}\! e^{\,s_i - m^{\text{new}}}\, \mathbf{v}_i,
$$

dove $\mathbf{o}$ è la riga di uscita accumulata (la somma pesata dei
$\mathbf{v}_i$, cioè un vettore, e per questo minuscolo grassetto: la $O$
maiuscola in queste pagine è già presa dalla notazione della complessità) e il
fattore
$e^{\,m - m^{\text{new}}}$ corregge ciò che avevamo già sommato quando compare un
massimo nuovo; alla fine si divide, $\mathbf{o} \leftarrow \mathbf{o}/l$. Tutto
qui: due scalari di stato per riga, e la matrice $N \times N$ non viene mai
scritta.
`````

## Cosa si guadagna (e cosa costa)

Il risultato è netto: la memoria che l'attenzione richiede non cresce più con
il *quadrato* della lunghezza del testo, ma in proporzione a essa. A testi
corti il guadagno è modesto, ma cresce con la lunghezza, ed è proprio sui testi
lunghi, dove la vecchia attenzione esauriva la memoria della scheda o
rallentava fino a fermarsi, che FlashAttention cambia le carte in tavola. Una
versione successiva, **FlashAttention-2** {cite}`dao2023flashattention2`,
spreme ancora di più l'hardware: ripartisce meglio il lavoro fra i gruppi di
lavoratori e riduce le operazioni che i tensor core non sanno accelerare,
quelle diverse dalla moltiplicazione fra tabelle. Ne esce un tempo di
esecuzione grosso modo dimezzato rispetto alla prima versione.

Va però detto con precisione **che cosa** tutto questo risolve, perché è facile
attribuirgli un merito che è di un'altra tecnica. Ci sono due momenti in cui la
tabella dei confronti viene costruita per intero: mentre il modello impara, e
nella prima passata con cui legge la domanda che gli abbiamo fatto. In tutti e
due il vincolo è quella tabella, e FlashAttention lo toglie. Mentre il modello
*scrive* la risposta, invece, la tabella non esiste nemmeno: si procede una
parola per volta, e i confronti da fare sono una riga sola. Lì il
peso è un altro, ed è la **KV cache**, cioè il taccuino in cui il modello
conserva quello che ha già letto per non rileggerlo da capo a ogni parola
(le due lettere stanno per *key* e *value*, chiave e valore, i due ingredienti
del confronto che vanno conservati entrambi).

Quel taccuino cresce in proporzione alla lunghezza del testo, non al suo
quadrato, ma è pesante, e il conto conviene di farlo con i numeri di un
modello vero, uno da otto miliardi di numeri imparati. Ha trentadue strati e
ognuno tiene il proprio taccuino; dentro ogni strato ci sono otto «teste» che
leggono il testo in parallelo, e ognuna descrive una parola con 128 numeri; di
ogni parola vanno conservate chiave e valore, quindi due volte tanto; e ogni
numero occupa due byte. In tutto $2 \times 32 \times 8 \times 128 \times 2$
byte, cioè $131\,072$ byte, che sono esattamente **128 KB** (un KB è 1024
byte) **per ogni parola letta**. Su centomila parole di contesto fanno
**dodici gigabyte**, per una conversazione sola, su una scheda che di gigabyte
ne ha ottanta. FlashAttention non lo tocca: è un altro mestiere, e lo
raccontano la sezione sui modelli linguistici del {doc}`capitolo sui Transformer </Transformers/overview>` e
il {doc}`capitolo su MLOps </MLOps/overview>`, dove si trovano anche le tecniche che quel problema lo
affrontano davvero. Conviene dire, per onestà, che FlashAttention **non**
riduce il numero di conti da fare, che resta proporzionale al quadrato della
lunghezza: quello è il mestiere del capitolo sull'attenzione lineare.

Onestà anche sul codice: quello che in queste pagine sta in un'idea semplice,
nel codice
è un kernel notoriamente complicato (indici, gestione della shared memory, casi
limite della **maschera causale**, la regola che impedisce a ogni parola di
sbirciare quelle che vengono dopo di lei). Non è codice che si scrive a mano per
un progetto normale, ed è giusto così. In PyTorch lo usi senza nemmeno saperlo:
la funzione `scaled_dot_product_attention` sceglie da sé, fra le varie
implementazioni che ha in casa (in gergo i *backend*), quella più adatta alla
scheda che ha davanti, e su GPU recenti quella è proprio FlashAttention. Le
righe qui sotto si possono guardare da lontano: quella che conta è la
penultima, dove si chiama la funzione; tutto il resto è preparare i numeri.

```{code-block} python
:class: pt-non-eseguibile

import torch
import torch.nn.functional as F

# Q, K, V: (batch, teste, N, d_k)
Q = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float16)
K = torch.randn_like(Q)
V = torch.randn_like(Q)

# PyTorch sceglie da sé il kernel: su GPU recenti, il backend FlashAttention.
# is_causal=True applica la maschera causale senza materializzarla.
O = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
print(O.shape)  # torch.Size([2, 8, 4096, 64])
```

Una riga di libreria, e sotto gira il kernel che abbiamo appena raccontato. È il
modo giusto di usarlo: capirne l'idea per sapere *quando* e *perché* aiuta, e
lasciarne l'implementazione a chi la mantiene ottimizzata generazione dopo
generazione.

## La frontiera: nascondere il movimento dei dati

FlashAttention è l'esempio più limpido di un filo conduttore che attraversa
tutto questo capitolo, ed è il modo migliore per chiuderlo dal lato
dell'hardware: **la storia della velocità sulle GPU è la storia di come
nascondere il movimento dei dati**. Ogni tecnica incontrata fin qui è una
variazione sullo stesso tema: chiedere i dati in fila invece che sparsi,
portare una tessera sul tavolo e riusarla, fare tre conti in un viaggio invece
che in tre, e adesso non scrivere affatto una tabella che serviva solo di
passaggio. Sempre la stessa cosa: fare più conti per ogni byte spostato, e
tenere il byte il più vicino possibile a chi calcola.

I kernel più veloci di oggi portano questa idea ancora più in là. Restando al
livello concettuale (niente istruzioni di basso livello) le leve sono tre.

`````{tab} Elementare
Sono le tre mosse di una catena di montaggio ben organizzata, e le vediamo in
quest’ordine.

**La prima: andare a prendere i pezzi mentre si lavora.** Nelle GPU più
recenti, mentre un gruppo di operai lavora sui pezzi che ha già sul banco, un
*altro* gruppo è già andato a prendere i pezzi successivi dal magazzino: quando
i primi finiscono, il materiale nuovo è lì pronto, e nessuno resta mai fermo ad
aspettare. La copia dal magazzino e il lavoro sul banco avvengono *nello stesso
momento*, sovrapposti, invece che uno dopo l'altro.

**La seconda: macchine più potenti, e pezzi più piccoli.** Le macchine sono i
*tensor core*, i timbri della sezione sul GEMM, che a ogni generazione stampano
più tabelline per battito; i pezzi più piccoli sono i numeri scritti con ancora
meno cifre binarie (dopo i sedici della mezza precisione sono arrivati gli
otto), che occupano metà spazio e viaggiano in metà tempo. C'è però un
rovescio, ed è la morale di tutto il capitolo: più la macchina è veloce, più è
facile che a mancare siano i pezzi e non le braccia.

**La terza: dare a ciascuno un ruolo fisso.** Invece di far fare a ogni squadra
un po’ di tutto, alcune squadre fanno *solo* i portapacchi e altre *solo* il
montaggio, come in una catena vera: un operaio dedicato a un compito lo fa
meglio di uno che salta di continuo da un lavoro all'altro, e così le macchine
non restano mai senza materiale e nessuno dei due lavori si ferma ad aspettare
l'altro.

Tutte e tre servono alla stessa cosa: far arrivare i pezzi *mentre* si lavora,
così che il banco non si fermi mai.
`````

`````{tab} Superiore
Tre direzioni, tutte volte a *nascondere* la latenza del movimento dati dietro il
calcolo:

- **Movimento asincrono dei dati.** Le GPU recenti (dall'architettura Hopper
  in poi) hanno unità dedicate (come il *Tensor Memory Accelerator*, TMA) che
  copiano tessere dalla HBM alla shared memory *in parallelo* al calcolo sui
  tensor core, sovrapponendo trasferimento ed esecuzione. Il kernel non
  aspetta i dati: lavora sul tile corrente mentre il prossimo è già in
  viaggio.
- **Tensor core sempre più potenti e formati più stretti.** Le unità di matmul
  crescono in throughput di generazione in generazione (Hopper, poi Blackwell)
  e guadagnano formati numerici più compatti, fino a **FP8** (8 bit), che
  dimezzano ancora i byte da spostare, nella stessa logica della precisione
  mista vista nella sezione «Prestazioni e scala». Ma più i tensor core sono
  veloci, più è facile ritrovarsi memory-bound: il ginocchio del roofline si
  sposta a destra, e la partita torna a giocarsi sui byte.
- **Warp specialization.** Invece di far fare a ogni warp un po’ di tutto, gli si
  assegnano *ruoli*: alcuni warp fanno solo da *producer* (caricano i dati dalla
  HBM), altri da *consumer* (calcolano sui tensor core), coordinati come i
  reparti di una catena di montaggio. La specializzazione tiene le unità di
  calcolo sempre rifornite e i canali di memoria sempre occupati.

Sono le tecniche con cui sono scritte le versioni più recenti dei kernel di
attenzione, che portano l'idea di FlashAttention fin sul silicio più nuovo.
Chi vuole seguirle fino in fondo (dal TMA alla warp specialization, fino alle
generazioni più recenti di FlashAttention) trova una trattazione avanzata nel
corso *Modern GPU Programming for MLSys* di mlc.ai. Il messaggio, però, resta
quello con cui abbiamo aperto il capitolo: le migliaia di core semplici sono
la parte facile; l'ingegneria vera è tenerle sfamate.
`````

Con questo il capitolo ha finito di guardare *dentro* una scheda: dal modo in
cui esegue, alla memoria che la rifornisce, ai due calcoli su cui una rete
spende quasi tutto il suo tempo. Resta la domanda che si affaccia quando il
modello, semplicemente, in una scheda non ci sta: è la sezione che chiude il
capitolo.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Per confrontare ogni parola con ogni altra, l'attenzione costruisce una
  tabella grande quanto il testo per il testo: raddoppia le parole e la tabella
  quadruplica. Il tempo però non se ne va nei conti: se ne va nei **viaggi** fra
  il magazzino lento e il tavolo di lavoro veloce.
- **FlashAttention** {cite}`dao2022flashattention` quella tabella non la scrive
  mai: tiene ferma sul tavolo una manciata di parole e fa scorrere le altre a
  **blocchetti**, uno per volta, buttando via ogni blocchetto appena usato. Il
  risultato non è un'approssimazione: è lo stesso identico numero di prima.
- A rendere possibile il lavoro a blocchetti è la **online softmax**, il gesto
  di chi pesa i sacchi due per volta tenendo un foglietto con il totale finora:
  qui i foglietti sono due, il totale e il punteggio più alto visto fin lì, e
  alla fine danno le stesse percentuali del calcolo in un colpo unico.
- Non fa **meno** conti degli altri: ne fa altrettanti, e nel viaggio di
  ritorno (quello in cui la rete impara dai propri errori) qualcuno in più,
  perché avendo buttato i blocchetti se li deve rifare. È un baratto voluto:
  si spende un po’ di calcolo, che costa poco, per risparmiare tanti viaggi,
  che costano molto.
- Il guadagno: la memoria non cresce più con il quadrato della lunghezza del
  testo ma in proporzione ad essa, e sui testi lunghi (dove prima la GPU
  si fermava per memoria esaurita) il salto è grande. Attenzione però a non
  dargli meriti di altri: questo vale mentre il modello *impara* e mentre
  *legge*. Mentre **scrive** la risposta il peso è un altro, il taccuino di
  ciò che ha già letto, e quello resta. Una seconda
  versione, **FlashAttention-2** {cite}`dao2023flashattention2`, ripartisce
  ancora meglio il lavoro. In PyTorch basta chiamare
  `scaled_dot_product_attention`.
- I kernel più veloci di oggi (i dati che viaggiano dal magazzino *mentre* si
  lavora, tavoli di lavoro sempre più potenti che usano numeri più corti, operai
  con ruoli fissi fra chi porta i pezzi e chi li monta) girano tutti attorno
  alla stessa idea: **nascondere il movimento dei dati** dietro il calcolo, così
  che nessuno resti fermo ad aspettare.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- L'attenzione materializza due matrici $N \times N$
  ($\mathbf{S} = \mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}$ e
  $\mathbf{P} = \text{softmax}(\mathbf{S})$): $O(N^2)$ memoria e traffico HBM. Il collo di
  bottiglia è la **memoria**, non i FLOP, e lo è per intero: anche i due matmul,
  avendo dimensione interna $d_k$, stanno a $\approx 63$ FLOP/byte contro un
  ginocchio di 161 su A100. Nella forma standard **niente** è compute-bound.
- **FlashAttention** {cite}`dao2022flashattention` è **IO-aware**: con il
  **tiling** di $\mathbf{Q},\mathbf{K},\mathbf{V}$ in shared memory e la
  **online softmax** non scrive mai la matrice $N \times N$ in HBM. Il risultato
  è **esatto**, non approssimato.
- La **online softmax** normalizza i punteggi a blocchi tenendo due scalari di
  stato (massimo corrente $m$ e somma corrente $l$) e ri-scalando ciò che ha
  già sommato quando compare un massimo nuovo: dà gli stessi pesi del calcolo
  in un colpo solo.
- Non fa **meno** FLOP: in avanti altrettanti, e nel backward $10N^2 d_k$
  contro $8N^2 d_k$, perché $\mathbf{S}$ e $\mathbf{P}$ non sono salvate e vanno **ricalcolate**
  ($+25\,\%$ sul backward, $+12{,}9\,\%$ misurati sul totale nel paper). È il
  baratto calcolo-per-traffico del *gradient checkpointing*, e conviene perché
  i FLOP ricomprati sono matmul e i byte risparmiati sono HBM.
- Il guadagno: memoria da $O(N^2)$ a $O(N)$, grande accelerazione a sequenze
  lunghe **in addestramento e in prefill**. In decodifica il vincolo è un
  altro, la **KV cache**, lineare in $N$ ma pesante, e FlashAttention non la
  tocca (né rende l'attenzione sub-quadratica nei FLOP: quello è l'argomento
  del capitolo sull'attenzione lineare). **FlashAttention-2**
  {cite}`dao2023flashattention2` migliora ancora la ripartizione del lavoro. In
  PyTorch lo si usa via `scaled_dot_product_attention`.
- La frontiera dei kernel veloci (movimento asincrono dei dati con TMA, tensor
  core e formati come FP8, **warp specialization**) è tutta una variazione sullo
  stesso tema: **nascondere il movimento dei dati** dietro il calcolo.
```
`````
