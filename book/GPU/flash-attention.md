# FlashAttention: l'attenzione che non spreca memoria

Chiedere a un modello di riassumere un romanzo intero, o di rispondere su un
contratto di cento pagine, fino a pochi anni fa era fuori portata, e uno dei
motivi era la memoria. Per confrontare ogni parola del testo con tutte le
altre, l'attenzione costruisce una tabella che cresce con il *quadrato* della
lunghezza. Raddoppia le parole e quella tabella quadruplica; moltiplicale per
dieci e diventa cento volte più grande. A un certo punto non ci sta più nella
memoria della GPU, e anche quando ci sta, spostarla avanti e indietro costa più
tempo dei conti che servono a riempirla. L'idea che ha spostato quel muro,
FlashAttention, è semplice nella sostanza: quella tabella non scriverla mai.

Serve prima il *che cosa* dell'attenzione, il meccanismo su cui i modelli
linguistici sono costruiti: la {doc}`sezione sulla matematica dei modelli
linguistici </Matematica/matematica-llm>` l'ha costruita come una media pesata,
e il {doc}`capitolo sui Transformer </Transformers/overview>` ne racconterà il
*perché*. Qui bastano i tre passi da eseguire in fretta, su un testo di $N$
posizioni (i token, cioè le parole o i pezzi di parola), ciascuna con tre
vettori di $d_k$ numeri, la *query*, la *key* e il *value*, che sono le righe
di tre matrici $\mathbf{Q}$, $\mathbf{K}$ e $\mathbf{V}$. Primo: ogni posizione
viene confrontata con tutte le altre, e da ogni confronto esce un punteggio di
somiglianza; sono i punteggi $\mathbf{S} =
\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}$, una tabella $N \times N$, la grande
tabella. Secondo: i punteggi di ciascuna riga vengono trasformati in
percentuali che sommano a cento, con un'operazione che ricorrerà per tutta la
sezione, la softmax: sono i pesi $\mathbf{P} = \text{softmax}(\mathbf{S})$,
un'altra tabella $N \times N$. Terzo: quei pesi dicono in che proporzione
mescolare i value, $\mathbf{O} = \mathbf{P}\mathbf{V}$. Se «salta» ha preso il
70% su «gatto», il 20% su «muro» e il resto sulle altre parole, il suo
risultato è fatto per sette decimi del value di «gatto», per due di quello di
«muro» e per l'ultimo decimo degli altri. Quello che ne esce è, per ogni
parola, un riassunto del resto della frase pesato su quanto ciascuna le
interessa.

Là il problema sarà *quale* informazione l'attenzione raccoglie; qui è un
altro, tutto hardware: *come* si eseguono quei tre passi senza affogare nel
traffico di memoria.

`````{tab} Elementare
Il punto da tenere a mente è uno solo, ed è una questione di conteggio: se le
parole sono mille e ognuna va confrontata con tutte, i confronti sono un
milione. Con diecimila parole diventano cento milioni. La tabella di quei
confronti è l'oggetto ingombrante di tutta la storia: nessuno la vuole, serve
solo di passaggio, e proprio per questo scriverla è uno spreco.

Sui punteggi c'è una precauzione da prendere prima di trasformarli in
percentuali. Il punteggio di somiglianza fra due parole si costruisce mettendo
a confronto tutti i numeri che le descrivono, e quei numeri sono decine: più ce
ne sono, più i punteggi finiscono lontani gli uni dagli altri. Se la distanza
cresce troppo, le percentuali vanno tutte alla parola in testa e alle altre
resta zero, cioè il confronto smette di dire qualcosa. Per questo i punteggi si
rimpiccioliscono tutti nella stessa misura prima di distribuire le percentuali:
una divisione sola, uguale per tutti, che lascia intatta la graduatoria di chi
somiglia a chi e tiene i numeri in una scala maneggevole.
`````

`````{tab} Superiore
In simboli, i tre passi stanno in una riga:

$$
\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\!\big(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}\big)\mathbf{V},
$$

con la softmax applicata riga per riga. La divisione per $\sqrt{d_k}$ tiene i
punteggi in una scala in cui la softmax non satura: con componenti
indipendenti di media nulla e varianza unitaria, il prodotto scalare di due
vettori di dimensione $d_k$ ha varianza $d_k$, e dividerlo per $\sqrt{d_k}$ la
riporta a uno {cite}`vaswani2017attention`. Le due
matrici $N \times N$ che compaiono qui dentro, $\mathbf{S}$ e $\mathbf{P}$,
sono le responsabili del costo: due intermedi che nessuno vuole come risultato,
e che pure, nella forma standard, vanno scritti per intero.
`````

## Il problema è la memoria, non i conti

Il primo istinto è pensare che l'attenzione sia lenta perché fa *tanti conti*.
I conti sono davvero tanti, ma nella forma standard non sono loro a fissare il
tempo: lo fissa il movimento dei dati, come capita spesso su una GPU (è la
lezione del roofline della {doc}`sezione sulla memoria <gerarchia-memoria>`).

`````{tab} Elementare
Mille parole, ognuna confrontata con ogni altra, fanno un milione di confronti:
una tabella di mille righe per mille colonne. Fin qui, tanti conti ma niente di
drammatico. Il guaio è *dove* metti quella tabella. È troppo grande per il
tavolo di lavoro veloce, il ripiano accanto ai calcolatori, quindi la GPU la
scrive nel magazzino lontano e poi deve tornare a prenderla per fare il secondo
passo (le percentuali), riscrivere anche quelle, e tornare *di nuovo* per il
terzo (la media). Quattro viaggi al magazzino per una tabella enorme che, alla
fine, non serviva nemmeno tenere: era solo un passaggio intermedio.

È come dover tenere la sfoglia in fondo al magazzino perché sul tavolo non ci
sta, e correre fin là a ogni operazione: una volta per portarcela, una per
andarla a riprendere e tagliarla, una per riportarci i pezzi, una per andarli
a riprendere e infornarli. Il tempo non se ne va nel taglio: se ne va nella
corsa.

Quanto se ne va si può contare, ed è il conto che decide tutto il resto. Il
confronto fra due parole non è un colpo d'occhio: si fa numero per numero, e i
numeri sono una sessantina per parte, quindi ogni casella della tabella costa
poco più di cento conti. Quella casella, però, è un numero solo, due byte da
portare in magazzino: sono una sessantina di conti per ogni byte spostato,
mentre il pareggio (il punto in cui il lavoro al tavolo dura quanto la corsa)
con le macchine di oggi sta oltre i centocinquanta. Anche il confronto, che è
la parte laboriosa, tiene occupato chi lavora sì e no quattro decimi del tempo;
il resto lo passa ad aspettare. La mossa che verrebbe in mente, sbrigare due
lavorazioni in un viaggio solo portandosi dietro il mattarello insieme al
coltello, non basta: la sfoglia in magazzino ci va comunque, e comunque la
corsa dura più del lavoro. L'unica che paga è non portarcela mai.
`````

`````{tab} Superiore
Per una sequenza di lunghezza $N$ e teste di dimensione $d_k$, l'attenzione
materializza due matrici $N \times N$: i punteggi
$\mathbf{S} = \mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}$ e i
pesi $\mathbf{P} = \text{softmax}(\mathbf{S})$. Il calcolo è $O(N^2 d_k)$ FLOP,
ma il dato che uccide le prestazioni è la memoria: $\mathbf{S}$ e
$\mathbf{P}$ occupano $O(N^2)$ byte e
vengono scritte e rilette dalla HBM più volte (produci $\mathbf{S}$, la rileggi
per la softmax, scrivi $\mathbf{P}$, la rileggi per il prodotto con
$\mathbf{V}$). Un numero concreto: con
$N = 8192$, una sola matrice $\mathbf{S}$ ha $N^2 \approx 67$ milioni di
elementi; in
`float16` (2 byte) sono circa $134$ MB (*per testa, per strato*). La memoria
cresce quadraticamente, e con essa il traffico verso la HBM.

Sul roofline questa è l'operazione tipicamente memory-bound, e al banco va
messo l'imputato giusto, perché la spiegazione che si legge più
spesso (i due matmul sarebbero compute-bound, e a rovinare tutto sarebbe la
softmax in mezzo) è sbagliata di suo. Il prodotto che dà $\mathbf{S}$ ha
uscita $N \times N$ e **dimensione interna** $d_k$, cioè 64 o 128; quello che
usa $\mathbf{P}$ legge un operando $N \times N$ e produce $N \times d_k$. In
tutti e due il traffico è dominato dalla matrice $N \times N$, scritta nel
primo e letta nel secondo, e al crescere di $N$ l'intensità tende a
$2N^2 d_k / (2N^2) = d_k$ esatti. Con $N = 8192$ e $d_k = 64$ in `float16` fa
63 FLOP/byte, contro un ginocchio di 161 su A100 e 295 su H100: anche i due
matmul, da soli, sono memory-bound, e userebbero al più il 39 % del picco. Con
$d_k = 128$ si arriva a 124, e resta sotto il ginocchio di entrambe le schede.

Le operazioni della softmax in mezzo (gli esponenziali, le riduzioni per riga,
le scritture e riletture della matrice $N \times N$) hanno intensità quasi
nulla e dimezzano ancora il conto. Il bilancio si rifà in due righe. Sempre
con $N = 8192$ e $d_k = 64$ in
`float16`: i conti sono i $2N^2 d_k$ del primo matmul più gli altrettanti del
secondo, più circa 5 operazioni per elemento della softmax (il confronto per
il massimo, la sottrazione, l'esponenziale, la somma, la divisione), in tutto
circa $17{,}5$ GFLOP; i byte sono quattro passaggi della matrice $N \times N$
(scrivi $\mathbf{S}$, la rileggi, scrivi $\mathbf{P}$, la rileggi) a 2 byte per
elemento, più le briciole di $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ e
dell'uscita, in tutto circa $541$ MB. Il rapporto fa 32 FLOP/byte: è lì che
sta l'attenzione intera, non fusa. La conclusione onesta è che nella forma
standard niente, in attenzione, è compute-bound. Fondere la sola softmax con
un matmul non basta, perché resterebbe una matrice $N \times N$ scritta e
riletta: la cura è non far mai atterrare $\mathbf{S}$ in HBM, e per farlo
bisogna fondere in un kernel solo tutti e tre i passi, con il tiling e la
online softmax che vengono subito dopo. Non serve una GPU più potente nei
FLOP: serve *non spostare* quei byte.
`````

## L'idea: lavorare a tessere, mai scrivere la matrice

La svolta arriva nel 2022 da Tri Dao e colleghi con **FlashAttention**
{cite}`dao2022flashattention`. La loro osservazione è che la matrice
$N \times N$ è solo un *intermedio*: alla fine ci serve l'output, non la
tabella dei punteggi. E allora perché scriverla? L'algoritmo è **IO-aware**
(«IO» sta per *input/output*, l'entrata e l'uscita dei dati dalla memoria):
ottimizza il movimento dei dati, non i conti, e (dettaglio cruciale) dà il
risultato esatto, non un'approssimazione.

Due ingredienti lo rendono possibile ({numref}`fig-flash-attention`). Il primo è
il tiling, cioè lo stesso «carica una tessera, riusala» della sezione sul
GEMM. Qui le tessere si ritagliano non nella tabella dei confronti, che
non esisterà mai, ma nell'elenco delle parole di partenza: si tiene ferma una
manciata di parole e si fa scorrere davanti a loro tutto il resto, un blocchetto
per volta. Il secondo ingrediente è la **online softmax**, proposta nel 2018 da
Milakov e Gimelshein per calcolare in una passata sola il massimo e il totale
che la softmax chiede {cite}`milakov2018online`, e che permette di calcolare le
percentuali *a pezzi* invece che tutte insieme.

```{figure} ../figures/flash-attention-tiling.svg
:name: fig-flash-attention
:alt: "A sinistra la matrice dei punteggi S uguale Q per K trasposto, N per N, disegnata come griglia e barrata da una grande X: la matrice che FlashAttention non scrive mai nella memoria HBM. A destra lo schema: una shared memory on-chip tiene un tile fisso di Q e un blocco corrente di K e V; sotto, i blocchi di K e V scorrono uno per volta dalla HBM verso la shared memory; un accumulatore aggiorna a ogni blocco l'output O e le due statistiche del softmax, il massimo corrente m e la somma corrente l; alla fine l'uscita è O diviso l, ed è esatta."
:width: 90%

La grande tabella dei confronti, $\mathbf{S}$, non viene mai scritta (a
sinistra, sbarrata: la scritta $O(N^2)$ vuol dire memoria che cresce con il
quadrato della lunghezza). Nella memoria veloce resta ferma una manciata di
righe di $\mathbf{Q}$, e i blocchi di $\mathbf{K}$ e $\mathbf{V}$ le scorrono
davanti uno alla volta; a ogni blocco si aggiornano il risultato e due soli
numeri di riepilogo, il massimo $m$ e la somma $l$, che bastano a rifare le
percentuali alla fine. Il risultato è quello del calcolo in un colpo solo, a
meno dell'ultima cifra, perché le stesse somme si fanno in un altro ordine.
```

`````{tab} Elementare
Il trucco è non costruire mai la tabella gigante. Tieni ferma sul tavolo di
lavoro una manciata di parole, quelle di cui ti stai occupando adesso (una
*tessera*, come quelle della moltiplicazione fra matrici), e fai scorrere
davanti a loro tutto il resto del testo a blocchetti: prendi le prime parole
con cui confrontarle, calcoli i punteggi, aggiorni il risultato; butti via quel
blocchetto, prendi il successivo, e così via fino alla fine. Sul tavolo, in
ogni istante, c'è solo un pezzetto piccolo. La tabella da un milione di caselle
non viene mai scritta per intero da nessuna parte: esiste un blocchetto alla
volta, e sparisce appena hai finito di usarlo. Meno viaggi al magazzino, e lo
stesso risultato della tabella intera: le somme sono le stesse, fatte in un
altro ordine, e come ogni somma fatta a rate possono differire al più
nell'ultima cifra.

Quanto grande è un blocchetto? Quanto ci sta sul tavolo insieme alla manciata
di parole ferme, e non un dito di più: la misura la decide il tavolo, non il
testo. E i viaggi non spariscono. Per ogni nuova manciata di parole da
elaborare, tutto il resto del testo deve sfilare daccapo, quindi il viavai
continua a crescere con il quadrato della lunghezza, come prima. Quello che
cambia è che ogni carico, una volta arrivato, serve per tutte le parole ferme
sul tavolo invece che per una sola: con le taglie in uso oggi i viaggi si
dividono per un numero dell'ordine della decina (nelle misure di chi l'ha
inventata, circa nove), e su un lavoro che passava la vita ad aspettare è
tantissimo.

Il prezzo si paga più tardi. Quando la rete impara,
dopo aver letto il testo in avanti rifà la strada all'indietro per capire quali
numeri correggere, e in quel secondo passaggio le servirebbero proprio i
punteggi che sono stati buttati: non avendoli, se li rifà. Qualche conto in più,
quindi, in cambio di molti viaggi in meno. È un baratto che conviene, perché i
conti sono la cosa che una GPU fa quasi gratis e i viaggi sono quella che le
costa.

Resta un'insidia da risolvere, ed è la più bella di tutta la storia: come fai a
calcolare delle *percentuali* se non hai ancora visto tutti i punteggi? Per
fare una percentuale ti serve il totale, e il totale lo conosci solo alla fine.
La risposta si chiama *online softmax*, e sta nel modo in cui si tiene il conto
strada facendo.
`````

`````{tab} Superiore
Formalmente, si spezzano $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ in blocchi di
righe. Fissato il blocco
di query $\mathbf{Q}_i$, si itera sui blocchi $(\mathbf{K}_j, \mathbf{V}_j)$:
si carica $\mathbf{K}_j, \mathbf{V}_j$ in
shared memory, si calcola il tile di punteggi
$\mathbf{S}_{ij} = \mathbf{Q}_i \mathbf{K}_j^\top/\sqrt{d_k}$, e si
aggiorna l'output *sul posto*, senza mai scrivere l'intera matrice $\mathbf{S}$
in HBM.
Qui $\mathbf{Q}_i$ è il blocco di query corrente (quello che resta fermo in
shared memory) e $\mathbf{K}_j, \mathbf{V}_j$ il blocco di key e value in
transito, per cui $\mathbf{S}_{ij}$ è la tessera
di punteggi che nasce dal loro incontro: $B_r \times B_c$, cioè $B_r$ righe di
query per $B_c$ chiavi (le due misure del blocchetto, che il kernel sceglie in
base a quanta shared memory ha), e non l'intera riga di $\mathbf{S}$.
(Quest'ordine dei cicli, con il blocco di query fermo e $\mathbf{K},\mathbf{V}$
che scorrono, è
quello reso canonico dalla seconda versione dell'algoritmo, che incontreremo a
breve; l'articolo del 2022 li annidava al contrario, ma l'idea non cambia.) La
memoria on-chip trattiene solo i tile correnti; la HBM vede scorrere
$\mathbf{K},\mathbf{V}$ una
volta per ogni blocco di query, e la matrice $\mathbf{S}$ mai. La memoria
extra
scende così da $O(N^2)$ a $O(N)$: da scrivere restano solo l'output e le
statistiche di riga.

Sui FLOP, invece, si sente ripetere il contrario di quello che succede. In
avanti i conti sono quelli di prima, a meno del riscalamento dell'accumulatore
a ogni blocco (ed è proprio quel di più, non-matmul, che FlashAttention-2 andrà
a limitare). All'indietro no: non avendo salvato $\mathbf{S}$ e $\mathbf{P}$,
il `backward` deve ricalcolarle da $\mathbf{Q}, \mathbf{K}, \mathbf{V}$, ed è
esattamente il motivo per cui bastava salvare l'output e due statistiche per
riga. Il conto: in avanti $4N^2 d_k$ FLOP, all'indietro $8N^2 d_k$ nella
versione standard e $10 N^2 d_k$ qui, cioè un quarto in più sul passaggio
all'indietro e un sesto in più sul totale. Sull'addestramento di GPT-2 medium,
in avanti e all'indietro, il paper riporta 75,2 GFLOP contro 66,6, cioè
$+12{,}9\,\%$. Sono numeri riportati: la figura non dice che cosa contenga il
conteggio, e il rialzo, minore del sesto calcolato sui soli prodotti fra
matrici, fa pensare a operazioni uguali nei due casi che lo diluiscono. A
fronte di questo, il traffico verso la HBM scende di circa nove volte (4,4
contro 40,3 GB) e il tempo di quasi sei (7,3 contro 41,7 ms), e il paper lo
scrive senza giri di parole: «even with the increased FLOPs due to
recomputation».

Il baratto è dunque calcolo in cambio di traffico, ed è lo stesso baratto del
*gradient checkpointing*, quello con cui si ricalcolano le attivazioni invece
di conservarle. Conviene per una ragione precisa: i FLOP ricomprati sono
matmul, cioè la cosa che i tensor core fanno a costo quasi nullo, mentre i byte
risparmiati sono accessi alla HBM, cioè la risorsa scarsa. È la stessa mossa
che si ritroverà nel pipeline parallelism della sezione sul parallelismo
distribuito, e che la sezione su {doc}`Mamba </StateSpaceModel/mamba>` ritrova
a sua volta: tre nomi diversi per la stessa mossa.

Anche il traffico verso la HBM crolla: il paper lo conta in $\Theta(N^2 d_k^2 /
M_\text{chip})$ accessi, con lo stesso $M_\text{chip}$ del GEMM, la memoria
veloce disponibile, contro il $\Theta(N d_k + N^2)$ dell'attenzione standard.
Resta quadratico in $N$, ma diviso per un fattore dell'ordine di
$M_\text{chip}/d_k^2$, che dà l'ordine di grandezza del guadagno e non il
guadagno: le costanti nascoste nei $\Theta$ non valgono 1, e le tessere non
occupano tutta la memoria veloce. Con i 192 KB di SRAM per SM di una A100, poco
meno di centomila elementi in `float16`, il rapporto varrebbe 24 con $d_k = 64$
e 6 con $d_k = 128$; il paper, discutendo il teorema, prende un $M_\text{chip}$
«around 100KB», circa la metà, e su GPT-2 medium ($d_k = 64$) misura un
traffico diviso per circa nove: su un carico memory-bound è tanto. Il conto
vale per $d_k \le M_\text{chip} \le N d_k$, e in quel regime non si fa di
meglio: il paper dimostra che nessun algoritmo di attenzione esatta scende a
$o(N^2 d_k^2 / M_\text{chip})$ accessi per tutti i valori di $M_\text{chip}$ di
quell'intervallo. È un limite inferiore su un intervallo di taglie, non per
ogni singola scheda. È l'idea del tiling in shared memory del GEMM, applicata
all'attenzione: caricare una volta, riusare in tanti, non tornare a leggere
dalla HBM.

Il nodo tecnico è che la softmax *non* è elemento-per-elemento: normalizza per
righe, e la normalizzazione richiede in teoria di aver già visto tutti i
punteggi della riga. Scorrere $\mathbf{K}$ a blocchi significa vedere i
punteggi un pezzo per volta, e qui entra la online softmax.
`````

### La online softmax, con i numeri

Il perno di tutto è calcolare le percentuali vedendo i punteggi a blocchi,
senza mai averli tutti sotto gli occhi insieme, e ottenendo comunque il
risultato esatto. Bastano due numeri di riepilogo: $m$, il punteggio più alto
visto finora, e $l$, il totale accumulato finora. In
{numref}`fig-flash-attention-blocchi` si vedono aggiornarsi blocco dopo blocco,
insieme alla cosa che conta di più: le celle fuori dalla finestra restano
vuote, perché quella tabella non viene mai scritta da nessuna parte.

```{figure} ../figures/flash-attention-blocchi.svg
:name: fig-flash-attention-blocchi
:alt: Una riga di otto punteggi divisa in quattro blocchi da due: una finestra scorre da sinistra a destra e in ogni istante mostra i numeri di un solo blocco, mentre fuori le celle restano vuote perché la matrice dei punteggi non viene mai scritta. Sotto, una tabella si riempie riga per riga con il massimo del blocco, il massimo corrente m, il fattore di riscalatura alfa, la somma corrente l e l'output accumulato O: quando arriva un massimo più grande alfa scende sotto 1 e l'accumulatore viene riscalato. Alla fine O diviso l coincide con la softmax calcolata in un colpo solo.
:width: 95%

Le due statistiche al lavoro su una riga di otto punteggi letti a due a due:
sono $1, 3, 2, 4, 1, 0, 5, 2$, e i value che si portano dietro sono
$1, 4, 2, 5, 3, 0, 6, 2$. In alto la finestra della memoria veloce: in ogni
istante contiene un solo blocchetto, e tutto il resto della riga resta vuoto,
perché quella tabella non viene mai scritta da nessuna parte. Sotto, la tabella
di marcia: a ogni blocchetto si aggiornano il massimo ($m$, il punteggio più
alto visto finora) e la somma ($l$), insieme al risultato che si sta
accumulando ($\mathbf{o}$). Quando arriva un punteggio più alto del massimo,
cioè alla seconda e alla quarta riga, il fattore $\alpha$ (qui $0{,}368$,
perché il massimo sale di un punto) riesprime rispetto al nuovo massimo quello
che era già stato messo da parte. Alla fine $\mathbf{o}$ diviso $l$ vale
$5{,}257$, lo stesso numero che darebbe il calcolo fatto in un colpo solo su
tutti e otto.
```

`````{tab} Elementare
Partiamo dal gesto facile: fare un totale a rate. Devi dire quanto pesa ogni
sacco rispetto al totale di tutti, ma sulla bilancia ne stanno due per volta.
Tieni un foglietto con «totale finora». Primo mucchietto, 10 e 30: il foglietto
dice 40. Secondo mucchietto, 20 e 40: il foglietto dice 100. A quel punto
dividi ciascun sacco per 100 e hai le percentuali (10%, 30%, 20%, 40%), che
sono esattamente quelle che avresti ottenuto stendendo tutti i sacchi per terra
in una volta sola. Nessuna approssimazione: la stessa somma, fatta a rate.

I foglietti però sono due, non uno, e il secondo è la parte meno ovvia. Nel
calcolo vero i punteggi, prima di essere sommati, non vengono presi così come
sono: si passa prima per un'operazione che li ingigantisce, e che ha una regola
semplice, ogni punto in più moltiplica per 2,7 circa (è il numero $e$, che vale
$2{,}718\ldots$, lo stesso della softmax incontrata fra le {doc}`funzioni di
attivazione </RetiNeurali/funzioni-attivazione>`). Un punteggio di due punti
più alto pesa quindi $2{,}7 \times 2{,}7$, più di sette volte tanto; dieci
punti più alto pesa ventiduemila volte tanto. Con punteggi anche moderatamente
alti si arriva a numeri che il computer non riesce più a scrivere.

Il rimedio è quello delle classifiche: invece del punteggio assoluto si segna
la distanza dal primo, «a tre punti dal record». Serve dunque un secondo
foglietto con il punteggio più alto visto finora, e ogni cosa si misura
rispetto a lui.

Da qui l'unico momento in cui la faccenda si fa interessante. Se in un
mucchietto salta fuori un punteggio più alto del record, il totale accumulato
era espresso rispetto al vecchio record e va riespresso rispetto al nuovo. Ed è
qui che quel «2,7 a punto» torna utile: se il record sale di un punto, tutto
quello che si era già messo da parte va diviso per 2,7, cioè moltiplicato per
0,37. Una moltiplicazione sola, si aggiorna il foglietto e si tira avanti.

Con i numeri, su quattro punteggi, 1, 3, 2, 4, letti a due a due. Primo
mucchietto, record 3: l'1 sta due punti sotto e pesa 0,135 (cioè 1 diviso 2,7
due volte), il 3 pesa 1, e il totale dice 1,135. Secondo mucchietto: arriva il
4, il record sale di un punto, e il totale già messo da parte va moltiplicato
per 0,368, il «diviso 2,7» di prima: 1,135 diventa 0,418. Poi si aggiungono i
due nuovi, il 2 (due punti sotto il record, 0,135) e il 4 (che pesa 1): il
totale fa 1,553, esattamente quello che darebbe guardare i quattro punteggi
tutti insieme. Sono i primi quattro della riga che la
{numref}`fig-flash-attention-blocchi` fa scorrere.

Manca il pezzo che poi ci si porta a casa, perché i punteggi non servono per
sé: servono a dosare. Ogni sacco contiene una farina diversa, e quello che si
vuole alla fine è la miscela in cui ciascuna entra in proporzione al proprio
peso. Anche la miscela si compone a rate, un mucchietto per volta, con le dosi
annotate rispetto al record del momento; e quando il record sale, la
conversione non riguarda solo il totale sul foglietto, riguarda anche le dosi
già annotate, con la stessa moltiplicazione per 0,37. All'ultimo mucchietto la
miscela si divide per il totale, ed è *esattamente* quella che darebbe il
calcolo fatto in un colpo unico. E quando, al ritorno, la rete deve rifare i
conti per imparare, i due foglietti non servono più tutti e due: per ogni
parola basta conservare un numero solo, che riassume record e totale insieme.
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
  Il vecchio totale va ri-scalato al nuovo massimo con il fattore di
  correzione $\alpha = e^{m_1 - m_2} = e^{-1} \approx 0{,}368$:

$$
l_2 = \alpha\, l_1 + \big(e^{2-4}+e^{4-4}\big)
    = 0{,}368 \cdot 1{,}135 + e^{-2}+1 \approx 1{,}553.
$$

Il totale $l_2 \approx 1{,}553$ coincide esattamente con la somma calcolata in
un colpo solo: la online softmax dà gli stessi pesi. In generale, arrivando un
nuovo blocco con massimo locale $\tilde m$, le regole di aggiornamento sono

$$
\begin{aligned}
m^{\text{new}} &= \max(m, \tilde m), \\
l^{\text{new}} &= e^{\,m - m^{\text{new}}}\, l + \!\sum_{i \in \text{blocco}}\! e^{\,s_i - m^{\text{new}}}, \\
\mathbf{o}^{\text{new}} &= e^{\,m - m^{\text{new}}}\, \mathbf{o} + \!\sum_{i \in \text{blocco}}\! e^{\,s_i - m^{\text{new}}}\, \mathbf{v}_i,
\end{aligned}
$$

dove $\mathbf{o}$ è la riga di uscita accumulata, cioè la somma pesata dei
$\mathbf{v}_i$ (un vettore, quindi minuscolo grassetto), e il fattore
$e^{\,m - m^{\text{new}}}$ corregge ciò che avevamo già sommato quando compare un
massimo nuovo; si parte da $m = -\infty$, $l = 0$ e $\mathbf{o} = \mathbf{0}$,
e alla fine si divide, $\mathbf{o} \leftarrow \mathbf{o}/l$. Tutto qui: due
scalari di stato per riga, e la matrice $N \times N$ non viene mai scritta.

Per il passo all'indietro basta salvarne uno, $L = m + \log l$ (il logaritmo
della somma degli esponenziali, *logsumexp*): da lì i pesi si ricostruiscono
come $P_{ij} = e^{S_{ij} - L_i}$ senza rifare il massimo
{cite}`dao2023flashattention2`. Resta un caso da trattare a parte. Una riga che
finora ha visto solo punteggi mascherati, cioè posti a $-\infty$ (una maschera
di riempimento, una finestra locale), ha ancora $m = -\infty$, e
$m - m^{\text{new}}$ darebbe $-\infty - (-\infty)$, cioè NaN: i kernel in quel
caso sostituiscono 0 al massimo. Con la maschera causale (ogni posizione vede
solo quelle che la precedono) i blocchi interamente mascherati, invece, si
saltano e basta, e il lavoro quasi si dimezza.
`````

## Cosa si guadagna (e cosa costa)

Il risultato è netto: la memoria che l'attenzione richiede non cresce più con
il *quadrato* della lunghezza del testo, ma in proporzione a essa. A testi
corti il guadagno è modesto, ma cresce con la lunghezza: è massimo dove
l'attenzione standard esauriva la memoria della scheda o passava quasi tutto il
tempo a spostare la matrice $N \times N$. Una versione successiva,
**FlashAttention-2** {cite}`dao2023flashattention2`, avvicina il kernel
all'efficienza di un GEMM con tre modifiche. Rinvia alla fine del giro sui
blocchi la divisione per la somma $l$, riducendo le operazioni che i tensor
core non accelerano. Distribuisce su SM diversi anche i blocchi di query della
stessa testa (i modelli eseguono l'attenzione in più copie parallele, le
*teste*), così che un testo lungo con pochi esempi tenga occupata tutta la
scheda. E dentro il blocco divide il lavoro fra i warp per righe di query
invece che per colonne di key, eliminando lo scambio di risultati parziali in
shared memory. Ne esce un tempo grosso modo dimezzato rispetto alla prima
versione.

Va però detto con precisione che cosa tutto questo risolve, perché è facile
attribuirgli un merito che è di un'altra tecnica. Ci sono due momenti in cui la
tabella dei confronti viene costruita per intero: mentre il modello impara, e
nella prima passata con cui legge la domanda che gli abbiamo fatto, che si
chiama *prefill*. In tutti e due il vincolo è quella tabella, e FlashAttention
lo toglie. Mentre il modello *scrive* la risposta, invece, la tabella non
esiste nemmeno: si procede un token per volta (è la *decodifica*), e i
confronti da fare sono una riga sola. Lì il peso è un altro, ed è la
**KV cache**: le key e i value che l'attenzione ha già calcolato per i token
precedenti, conservati per non ricalcolarli a ogni token nuovo (vanno tenuti
entrambi, perché la key serve al confronto e il value alla miscela).

La KV cache cresce in proporzione alla lunghezza del testo, non al suo
quadrato, ma pesa, e il conto si fa con un modello vero, della taglia del più
piccolo di Llama 3, otto miliardi di parametri {cite}`grattafiori2024llama3`:

- 32 strati, ognuno con la propria cache;
- 32 teste di query ma solo 8 di key e value, perché gruppi di quattro teste
  condividono le stesse key e gli stessi value (*grouped-query attention*,
  GQA {cite}`ainslie2023gqa`), e la cache è già divisa per quattro;
- 128 numeri per testa, e 2 byte per numero.

Per ogni token si conservano key e value in ogni strato, cioè
$2 \times 32 \times 8 \times 128 \times 2 = 131\,072$ byte. Su centomila token
di contesto fanno 13 GB, per una conversazione sola, su una scheda che di GB ne
ha 80. FlashAttention la KV cache non la tocca: è un altro mestiere, e lo fanno
altre tecniche, che stanno nella sezione sui {doc}`grandi modelli linguistici
</Transformers/llm>` e in quella su {doc}`prefill e decodifica
</MLOps/metriche-di-servizio>`. E FlashAttention non riduce il numero di conti
da fare, che resta proporzionale al quadrato della lunghezza: quello è il
mestiere del {doc}`capitolo sull'attenzione lineare
</AttenzioneLineare/overview>`.

Onestà anche sul codice: l'idea è semplice, il kernel che la realizza è
notoriamente complicato (indici, gestione della shared memory, casi limite
della **maschera causale**, la regola che impedisce a ogni parola di sbirciare
quelle che vengono dopo di lei). Non è codice che si scrive a mano per
un progetto normale, ed è giusto così. In PyTorch lo usi senza nemmeno saperlo:
la funzione `scaled_dot_product_attention` sceglie da sé, fra le varie
implementazioni che ha in casa (in gergo i *backend*), quella più adatta alla
scheda che ha davanti, e su GPU recenti quella è proprio FlashAttention. Nel
codice la riga che conta è quella che chiama `scaled_dot_product_attention`;
tutto il resto è preparare i numeri.

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

## La frontiera dei kernel veloci

FlashAttention è l'esempio più limpido del filo che lega le tecniche viste per
le GPU. Chiedere i dati in fila invece che sparsi, portare una tessera in
shared memory e riusarla, fare tre conti in un viaggio invece che in tre, e
adesso non scrivere affatto una tabella che serviva solo di passaggio: sempre
la stessa cosa, fare più conti per ogni byte spostato, e tenere il byte il più
vicino possibile a chi calcola. Quando i byte non si possono ridurre più,
resta da farli viaggiare mentre le unità di calcolo lavorano.

I kernel più veloci di oggi portano queste idee ancora più in là, con tre
leve.

`````{tab} Elementare
Sono le tre mosse di una catena di montaggio ben organizzata, e le vediamo in
quest’ordine.

La prima: andare a prendere i pezzi mentre si lavora. Nelle GPU degli ultimi
anni, mentre un gruppo di operai lavora sui pezzi che ha già sul banco, un
*altro* gruppo è già andato a prendere i pezzi successivi dal magazzino: quando
i primi finiscono, il materiale nuovo è lì pronto, e nessuno resta mai fermo ad
aspettare. La copia dal magazzino e il lavoro sul banco avvengono *nello stesso
momento*, sovrapposti, invece che uno dopo l'altro.

La seconda: macchine più potenti, e pezzi più piccoli. Le macchine sono i
*tensor core*, i timbri della sezione sul GEMM, che a ogni generazione stampano
più tabelline per battito; i pezzi più piccoli sono i numeri scritti con ancora
meno cifre binarie (dopo i sedici della mezza precisione sono arrivati gli
otto, e sulle schede più recenti i quattro), che occupano meno spazio e
viaggiano più in fretta. C'è però un rovescio, ed è la morale di tutto il
capitolo: più la macchina è veloce, più è facile che a mancare siano i pezzi e
non le braccia.

La terza: dare a ciascuno un ruolo fisso. Invece di far fare a ogni squadra
un po’ di tutto, alcune squadre fanno *solo* i portapacchi e altre *solo* il
montaggio, come in una catena vera: un operaio dedicato a un compito lo fa
meglio di uno che salta di continuo da un lavoro all'altro, e così le macchine
non restano mai senza materiale e nessuno dei due lavori si ferma ad aspettare
l'altro.

Tutte e tre servono a non lasciare il banco senza pezzi: la prima e la terza
facendoli arrivare *mentre* si lavora, la seconda facendone stare di più in
ogni viaggio.
`````

`````{tab} Superiore
Tre direzioni: la prima e la terza *nascondono* il movimento dei dati dietro il
calcolo, la seconda alza il picco di calcolo e riduce i byte per elemento.

- **Movimento asincrono dei dati.** Da Ampere (2020) la copia di tessere dalla
  HBM alla shared memory può avvenire senza passare dai registri e *in
  parallelo* al calcolo (`cp.async`, che CUDA espone come `cuda::memcpy_async`)
  {cite}`luo2024hopper`; da Hopper se ne occupa un'unità dedicata, il *Tensor
  Memory Accelerator* (TMA), con cui un solo thread sposta un'intera tessera,
  fino a cinque dimensioni. Il kernel non aspetta i dati: lavora sul tile
  corrente mentre il prossimo è già in viaggio.
- **Tensor core sempre più potenti e formati più stretti.** Le unità di matmul
  crescono in throughput di generazione in generazione (Hopper, poi Blackwell)
  e guadagnano formati numerici più compatti, **FP8** (8 bit) su Hopper e FP4
  su Blackwell {cite}`shah2024flashattention3`, che dimezzano o riducono a un
  quarto i byte da spostare rispetto a `float16`, nella stessa logica della
  precisione mista vista nella sezione «Prestazioni e scala». Ma più i tensor
  core sono veloci, più è facile ritrovarsi memory-bound: il ginocchio del
  roofline si sposta a destra, e la partita torna a giocarsi sui byte.
- **Warp specialization.** Invece di far fare a ogni warp un po’ di tutto, gli si
  assegnano *ruoli*: alcuni warp fanno solo da *producer* (caricano i dati dalla
  HBM), altri da *consumer* (calcolano sui tensor core), coordinati come i
  reparti di una catena di montaggio. La specializzazione tiene le unità di
  calcolo sempre rifornite e i canali di memoria sempre occupati.

Sono le tre leve di FlashAttention-3 {cite}`shah2024flashattention3`, scritta
nel 2024 per le GPU Hopper, più due mosse sue. La prima alterna due gruppi di
warp (*ping-pong*): la softmax dell'uno, che i tensor core non accelerano, gira
mentre l'altro esegue i propri GEMM. La seconda, in FP8, precede la
quantizzazione a blocchi con una rotazione casuale (una trasformata di Hadamard
con segni casuali, detta *incoherent processing*) che sparpaglia i valori
anomali prima di arrotondare. Algoritmo, online softmax e conto degli accessi
alla HBM restano quelli del 2022: cambia quanta parte del movimento dei dati il
calcolo riesce a coprire. Chi vuole seguirle fino al codice trova una
trattazione avanzata nel corso *Modern GPU Programming for MLSys* di mlc.ai. Il
messaggio, però, resta quello del roofline: le unità di calcolo sono tante, e
la partita si gioca sul tenerle rifornite di dati.
`````

Fin qui lo sguardo è rimasto *dentro* una scheda: dal modo in cui esegue,
alla memoria che la rifornisce, ai due calcoli in cui si concentra quasi tutta
l'aritmetica di una rete, il prodotto fra matrici e l'attenzione. Resta la
domanda che si affaccia quando il modello, semplicemente, in una scheda non ci
sta, ed è il tema del {doc}`parallelismo distribuito
<parallelismo-distribuito>`.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Per confrontare ogni parola con ogni altra, l'attenzione costruisce una
  tabella grande quanto il testo per il testo: raddoppia le parole e la tabella
  quadruplica. Il tempo però non se ne va nei conti: se ne va nei viaggi fra
  il magazzino lento e il tavolo di lavoro veloce.
- FlashAttention {cite}`dao2022flashattention` quella tabella non la scrive
  mai: tiene ferma sul tavolo una manciata di parole e fa scorrere le altre a
  blocchetti, uno per volta, buttando via ogni blocchetto appena usato. Il
  risultato è lo stesso numero di prima, a meno dell'ultima cifra, e non
  un'approssimazione.
- A rendere possibile il lavoro a blocchetti è la online softmax, il gesto
  di chi pesa i sacchi due per volta tenendo un foglietto con il totale finora:
  qui i foglietti sono due, il totale e il punteggio più alto visto fin lì, e
  alla fine danno le stesse percentuali del calcolo in un colpo unico.
- Non fa meno conti degli altri: ne fa altrettanti, e nel viaggio di
  ritorno (quello in cui la rete impara dai propri errori) qualcuno in più,
  perché avendo buttato i blocchetti se li deve rifare. È un baratto voluto:
  si spende un po’ di calcolo, che costa poco, per risparmiare tanti viaggi,
  che costano molto.
- Il guadagno: la memoria non cresce più con il quadrato della lunghezza del
  testo ma in proporzione ad essa, e sui testi lunghi (dove prima la GPU
  si fermava per memoria esaurita) il salto è grande. Attenzione però a non
  dargli meriti di altri: questo vale mentre il modello *impara* e mentre
  *legge*. Mentre scrive la risposta il peso è un altro, la KV cache, il
  taccuino di ciò che ha già letto, e quello resta. Una seconda
  versione, FlashAttention-2 {cite}`dao2023flashattention2`, ripartisce
  ancora meglio il lavoro. In PyTorch basta chiamare
  `scaled_dot_product_attention`.
- I kernel più veloci di oggi (i dati che viaggiano dal magazzino *mentre* si
  lavora, tavoli di lavoro sempre più potenti che usano numeri più corti, operai
  con ruoli fissi fra chi porta i pezzi e chi li monta) servono tutti a non
  lasciare le unità di calcolo senza dati: o facendoli viaggiare mentre si
  lavora, o facendone stare di più in ogni viaggio.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- L'attenzione materializza due matrici $N \times N$ ($\mathbf{S} =
  \mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}$ e $\mathbf{P} =
  \text{softmax}(\mathbf{S})$): $O(N^2)$ memoria e traffico HBM. Il collo di
  bottiglia è la memoria, non i FLOP, e lo è per intero: anche i due matmul,
  dominati dal traffico della matrice $N \times N$ (scritta dal primo, letta
  dal secondo), stanno a $\approx 63$ FLOP/byte contro un ginocchio di 161 su
  A100. Nella forma standard niente è compute-bound.
- FlashAttention {cite}`dao2022flashattention` è IO-aware: con il
  tiling di $\mathbf{Q},\mathbf{K},\mathbf{V}$ in shared memory e la
  online softmax non scrive mai la matrice $N \times N$ in HBM. Il risultato
  è esatto, non approssimato.
- La online softmax normalizza i punteggi a blocchi tenendo due scalari di
  stato (massimo corrente $m$ e somma corrente $l$) e ri-scalando ciò che ha
  già sommato quando compare un massimo nuovo: dà gli stessi pesi del calcolo
  in un colpo solo. Per il backward basta salvare il logsumexp
  $L = m + \log l$.
- Non fa meno FLOP: in avanti altrettanti, e nel backward $10N^2 d_k$
  contro $8N^2 d_k$, perché $\mathbf{S}$ e $\mathbf{P}$ non sono salvate e vanno ricalcolate
  ($+25\,\%$ sul backward; il paper riporta $+12{,}9\,\%$ sul totale). È il
  baratto calcolo-per-traffico del *gradient checkpointing*, e conviene perché
  i FLOP ricomprati sono matmul e i byte risparmiati sono HBM.
- Il guadagno: memoria da $O(N^2)$ a $O(N)$, grande accelerazione a sequenze
  lunghe in addestramento e in prefill. In decodifica il vincolo è un
  altro, la KV cache, lineare in $N$ ma pesante, e FlashAttention non la
  tocca (né rende l'attenzione sub-quadratica nei FLOP: quello è l'argomento
  del capitolo sull'attenzione lineare). FlashAttention-2
  {cite}`dao2023flashattention2` migliora ancora la ripartizione del lavoro. In
  PyTorch lo si usa via `scaled_dot_product_attention`.
- La frontiera dei kernel veloci ha due mosse: nascondere il movimento dei
  dati dietro il calcolo (copie asincrone da Ampere, TMA da Hopper, warp
  specialization) e ridurre i byte per elemento con formati più corti (FP8,
  FP4).
```
`````
