# FlashAttention: l'attenzione che non spreca memoria

Chiedere a un modello di riassumere un romanzo intero, o di rispondere su un
contratto di cento pagine, fino a pochi anni fa era fuori portata, e uno dei
motivi era la memoria. Per confrontare ogni parola del testo con tutte le
altre, infatti, l'attenzione costruisce una tabella che cresce con il
*quadrato* della lunghezza. Raddoppia le parole e quella tabella quadruplica;
moltiplicale per dieci e diventa cento volte più grande.

A un certo punto la tabella non ci sta più nella memoria della GPU, e anche
quando ci sta, spostarla avanti e indietro prende più tempo dei calcoli che
servono a riempirla. FlashAttention, l'idea che ha spostato quel muro, si dice
in una riga: quella tabella non viene mai scritta.

Serve prima il *che cosa* dell'attenzione, il meccanismo su cui i modelli
linguistici sono costruiti. La {doc}`sezione sulla matematica dei modelli
linguistici </Matematica/matematica-llm>` l'ha costruita come una media pesata,
e il {doc}`capitolo sui Transformer </Transformers/overview>` ne racconterà il
*perché*. Qui bastano i tre passi da eseguire in fretta su un testo di $N$
posizioni (i token, cioè le parole o i pezzi di parola). Ciascuna posizione ha
tre vettori di $d_k$ numeri, la *query*, la *key* e il *value*, che formano le
righe di tre matrici $\mathbf{Q}$, $\mathbf{K}$ e $\mathbf{V}$.

Nel primo passo ogni posizione viene confrontata con tutte le altre, e da ogni
confronto esce un punteggio di somiglianza; sono i punteggi $\mathbf{S} =
\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}$, una tabella $N \times N$, la grande
tabella. Nel secondo passo i punteggi di ciascuna riga vengono trasformati in
percentuali che sommano a cento, con la softmax; sono i pesi $\mathbf{P} =
\text{softmax}(\mathbf{S})$, un'altra tabella $N \times N$. Nel terzo passo
quei pesi dicono in che proporzione mescolare i value, $\mathbf{O} =
\mathbf{P}\mathbf{V}$.

Ad esempio, se «salta» ha preso il 70% su «gatto», il 20% su «muro» e il resto
sulle altre parole, il suo risultato è fatto per sette decimi del value di
«gatto», per due di quello di «muro» e per l'ultimo decimo degli altri. In
generale, per ogni parola ne esce un riassunto del resto della frase, pesato su
quanto ciascuna le interessa.

Nel capitolo sui Transformer il problema sarà *quale* informazione l'attenzione
raccoglie; qui è un altro, tutto hardware: *come* si eseguono quei tre passi
senza affogare nel traffico di memoria.

`````{tab} Elementare
Per capire le dimensioni del problema basta un conteggio. Se le parole sono
mille e ognuna va confrontata con tutte, i confronti sono un milione; con
diecimila parole diventano cento milioni. La tabella di quei confronti è
l'oggetto ingombrante di tutta la storia: nessuno la vuole, serve solo di
passaggio, e scriverla è uno spreco.

Sui punteggi c'è una precauzione da prendere prima di trasformarli in
percentuali. Il punteggio di somiglianza fra due parole si costruisce mettendo
a confronto tutti i numeri che le descrivono, e quei numeri sono decine. Più ce
ne sono, più i punteggi finiscono lontani gli uni dagli altri; se la distanza
cresce troppo, le percentuali vanno tutte alla parola in testa e alle altre
resta zero, cioè il confronto smette di dire qualcosa.

Per questo i punteggi si rimpiccioliscono tutti nella stessa misura prima di
distribuire le percentuali. È una divisione sola, uguale per tutti,
che lascia intatta la graduatoria di chi somiglia a chi e tiene i numeri in una
scala maneggevole.
`````

`````{tab} Superiore
In simboli, i tre passi stanno in una riga:

$$
\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\!\big(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}\big)\mathbf{V},
$$

con la softmax applicata riga per riga. La divisione per $\sqrt{d_k}$ tiene i
punteggi in una scala in cui la softmax non satura. Infatti, se le componenti
sono indipendenti, di media nulla e varianza unitaria, il prodotto scalare di
due vettori di dimensione $d_k$ ha varianza $d_k$, e dividerlo per
$\sqrt{d_k}$ la riporta a uno {cite}`vaswani2017attention`.

Le due matrici $N \times N$ che compaiono nella formula, $\mathbf{S}$ e
$\mathbf{P}$, sono le responsabili del costo: due intermedi che nessuno vuole
come risultato ma che, nella forma standard, vanno scritti.
`````

## Il problema è la memoria, non i calcoli

Verrebbe da pensare che l'attenzione sia lenta perché fa *molti calcoli*. Sono
molti, ma nella forma standard non sono loro a fissare il tempo: lo fissa il
movimento dei dati, come accade spesso su una GPU (è il roofline della
{doc}`sezione sulla memoria <gerarchia-memoria>`).

`````{tab} Elementare
Mille parole, ognuna confrontata con ogni altra, fanno un milione di confronti,
cioè una tabella di mille righe per mille colonne. Fin qui, tanti conti ma
niente di drammatico. Il problema, però, è *dove* metti quella tabella. È
troppo grande per il tavolo di lavoro veloce, il ripiano accanto a chi calcola,
quindi la GPU la scrive nel magazzino lontano e poi deve tornare a prenderla per
fare il secondo passo (le percentuali), riscrivere anche quelle, e tornare *di
nuovo* per il terzo (la media). Quattro viaggi al magazzino per una tabella
enorme che, alla fine, non serviva nemmeno tenere, perché era solo un passaggio
intermedio.

È come dover tenere la sfoglia in fondo al magazzino perché sul tavolo non ci
sta, e correre fin là a ogni operazione: una volta per portarcela, una per
andarla a riprendere e tagliarla, una per riportarci i pezzi, una per andarli
a riprendere e infornarli. Il tempo non se ne va nel taglio: se ne va nella
corsa.

Quanto tempo se ne va si può calcolare, e il risultato decide tutto il resto.
Il confronto fra due parole non è un colpo d'occhio: si fa numero per numero, e
con una sessantina di numeri per parte servono poco più di cento operazioni per
ogni casella della tabella. Quella casella, però, è un numero solo, due byte da
portare in magazzino, quindi le operazioni sono una sessantina per ogni byte
spostato, mentre il pareggio (il punto in cui il lavoro al tavolo dura quanto la
corsa) con le macchine di oggi sta oltre i centocinquanta. Anche
il confronto, che è la parte laboriosa, tiene occupato chi lavora sì e no
quattro decimi del tempo; il resto lo passa ad aspettare.

Si potrebbero sbrigare due lavorazioni in un viaggio solo, portandosi dietro il
mattarello insieme al coltello, ma non basta, perché la sfoglia in magazzino ci
va comunque e la corsa dura più del lavoro. Serve evitare del tutto di
portarcela.
`````

`````{tab} Superiore
Per una sequenza di lunghezza $N$ e teste di dimensione $d_k$, l'attenzione
materializza due matrici $N \times N$, i punteggi
$\mathbf{S} = \mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}$ e i pesi
$\mathbf{P} = \text{softmax}(\mathbf{S})$. Il calcolo è $O(N^2 d_k)$ FLOP, ma a
limitare le prestazioni è la memoria, perché $\mathbf{S}$ e $\mathbf{P}$
occupano $O(N^2)$ byte e vengono scritte e rilette dalla HBM più volte
(produci $\mathbf{S}$, la rileggi per la softmax, scrivi $\mathbf{P}$, la
rileggi per il prodotto con $\mathbf{V}$).

Ad esempio, con $N = 8192$ una sola matrice $\mathbf{S}$ ha
$N^2 \approx 67$ milioni di elementi, che in `float16` (2 byte) sono circa
$134$ MB (*per testa, per strato*). La memoria cresce quadraticamente, e con
essa il traffico verso la HBM.

Sul roofline questa è l'operazione tipicamente memory-bound, ma la spiegazione
che si legge più spesso (i due matmul sarebbero compute-bound, e a rovinare
tutto sarebbe la softmax in mezzo) è sbagliata. Il prodotto che dà
$\mathbf{S}$, infatti, ha uscita $N \times N$ e **dimensione interna** $d_k$,
cioè 64 o 128; quello che usa $\mathbf{P}$ legge un operando $N \times N$ e
produce $N \times d_k$. In tutti e due il traffico è dominato dalla matrice
$N \times N$, scritta nel primo e letta nel secondo, e al crescere di $N$
l'intensità tende a $2N^2 d_k / (2N^2) = d_k$.

Con $N = 8192$ e $d_k = 64$ in `float16` l'intensità vale 63 FLOP/byte,
contro un ginocchio di 161 su A100 e 295 su H100. Quindi anche i due
matmul, da soli, sono memory-bound, e userebbero al più il 39 % del picco.
Con $d_k = 128$ si arriva a 124, che resta sotto il ginocchio di entrambe le
schede.

Le operazioni della softmax in mezzo (gli esponenziali, le riduzioni per riga,
le scritture e riletture della matrice $N \times N$) hanno intensità quasi
nulla e dimezzano ancora il rapporto. Lo mostra il bilancio dell'attenzione
intera, sempre con $N = 8192$ e $d_k = 64$ in `float16`. I calcoli sono i
$2N^2 d_k$ del primo matmul più gli altrettanti del secondo, più circa 5
operazioni per elemento della softmax (il confronto per il massimo, la
sottrazione, l'esponenziale, la somma, la divisione), in tutto circa
$17{,}5$ GFLOP. I byte sono quattro passaggi della matrice $N \times N$
(scrivi $\mathbf{S}$, la rileggi, scrivi $\mathbf{P}$, la rileggi) a 2 byte
per elemento, più le briciole di $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ e
dell'uscita, in tutto circa $541$ MB. Il rapporto è di 32 FLOP/byte per
l'attenzione intera, non fusa.

Nella forma standard, dunque, nessuna parte dell'attenzione è compute-bound, e
fondere la sola softmax con un matmul non basta, perché resterebbe una matrice
$N \times N$ scritta e riletta. La cura è non far mai atterrare $\mathbf{S}$
in HBM; per farlo bisogna fondere in un kernel solo tutti e tre i passi, con
il tiling e la online softmax.
`````

## L'idea: lavorare a tessere, mai scrivere la matrice

La svolta arriva nel 2022 da Tri Dao e colleghi con **FlashAttention**
{cite}`dao2022flashattention`. La loro osservazione è che la matrice
$N \times N$ è solo un *intermedio*: alla fine ci serve l'output, non la
tabella dei punteggi. Perché scriverla, allora? L'algoritmo è **IO-aware**
(«IO» sta per *input/output*, l'entrata e l'uscita dei dati dalla memoria),
cioè ottimizza il movimento dei dati, non i calcoli, e dà il risultato esatto,
non un'approssimazione.

Due ingredienti lo rendono possibile ({numref}`fig-flash-attention`). Il primo
è il tiling, cioè lo stesso «carica una tessera, riusala» della
{doc}`sezione sul GEMM <gemm-e-tensor-core>`. In questo caso le tessere si
ritagliano nell'elenco delle parole di partenza, perché la tabella dei
confronti non esisterà mai: si tiene ferma una manciata di parole e si fa
scorrere davanti a loro tutto il resto, un blocco per volta.

Il secondo ingrediente è la **online softmax**, proposta nel 2018 da Milakov e
Gimelshein per calcolare in una passata sola il massimo e il totale che la
softmax chiede {cite}`milakov2018online`, e che permette di calcolare le
percentuali *a pezzi* invece che tutte insieme.

```{figure} ../figures/flash-attention-tiling.svg
:name: fig-flash-attention
:alt: "A sinistra la matrice dei punteggi S uguale Q per K trasposto, N per N, disegnata come griglia e barrata da una grande X: la matrice che FlashAttention non scrive mai nella memoria HBM. A destra lo schema: una shared memory on-chip tiene un tile fisso di Q e un blocco corrente di K e V; sotto, i blocchi di K e V scorrono uno per volta dalla HBM verso la shared memory; un accumulatore aggiorna a ogni blocco l'output O e le due statistiche del softmax, il massimo corrente m e la somma corrente l; alla fine l'uscita è O diviso l, ed è esatta."
:width: 90%

La grande tabella dei confronti, $\mathbf{S}$, non viene mai scritta (a
sinistra, sbarrata: la scritta $O(N^2)$ indica la memoria, che cresce con il
quadrato della lunghezza). Nella memoria veloce resta ferma una manciata di
righe di $\mathbf{Q}$, e i blocchi di $\mathbf{K}$ e $\mathbf{V}$ le scorrono
davanti uno alla volta; a ogni blocco si aggiornano il risultato e due soli
numeri di riepilogo, il massimo $m$ e la somma $l$, che bastano a rifare le
percentuali alla fine. Il risultato è quello del calcolo in un colpo solo, a
meno dell'ultima cifra, perché le stesse somme si fanno in un altro ordine.
```

`````{tab} Elementare
La tabella gigante non si costruisce mai. Tieni ferma sul tavolo di
lavoro una manciata di parole, quelle di cui ti stai occupando adesso (una
*tessera*, come quelle della moltiplicazione fra matrici), e fai scorrere
davanti a loro tutto il resto del testo a blocchetti: prendi le prime parole
con cui confrontarle, calcoli i punteggi, aggiorni il risultato; butti via quel
blocchetto, prendi il successivo, e così via fino alla fine. Sul tavolo, in
ogni istante, c'è solo un pezzetto piccolo.

La tabella da un milione di caselle non viene mai scritta da nessuna parte:
esiste un blocchetto alla volta, e sparisce appena hai finito di usarlo.
In questo modo i viaggi al magazzino diminuiscono e il risultato resta quello
della tabella intera, perché le somme sono le stesse, fatte in un altro ordine;
come ogni somma fatta a rate, possono differire al più nell'ultima cifra.

Quanto grande è un blocchetto? Quanto ci sta sul tavolo insieme alla manciata
di parole ferme, e non un dito di più: la misura dipende dal tavolo, non dal
testo.

I viaggi, però, non spariscono. Per ogni nuova manciata di parole da
elaborare, tutto il resto del testo deve sfilare daccapo, quindi il viavai
continua a crescere con il quadrato della lunghezza, come prima. La differenza
è che ogni carico, una volta arrivato, serve per tutte le parole ferme sul
tavolo invece che per una sola. Con le taglie in uso oggi i viaggi si dividono
così per un numero dell'ordine della decina (nelle misure di chi l'ha
inventata, circa nove), e su un lavoro che passava la vita ad aspettare è
tantissimo.

Il costo arriva più tardi. Quando la rete impara, dopo aver letto il testo in
avanti rifà la strada all'indietro per capire quali numeri correggere, e in
quel secondo passaggio le servirebbero i punteggi che sono stati buttati; non
avendoli, se li rifà. Qualche conto in più, quindi, in cambio di molti viaggi
in meno. Il baratto conviene, perché una GPU esegue i calcoli quasi senza
fatica, mentre i viaggi la rallentano.

Resta un'insidia: come fai a calcolare delle *percentuali* se non hai ancora
visto tutti i punteggi? Per fare una percentuale ti serve il totale, e il
totale lo conosci solo alla fine. La risposta si chiama *online softmax*, e sta
nel modo in cui si aggiorna il totale strada facendo.
`````

`````{tab} Superiore
Formalmente, si spezzano $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ in blocchi di
righe. Fissato il blocco di query $\mathbf{Q}_i$, si itera sui blocchi
$(\mathbf{K}_j, \mathbf{V}_j)$: si carica $\mathbf{K}_j, \mathbf{V}_j$ in
shared memory, si calcola il tile di punteggi $\mathbf{S}_{ij} = \mathbf{Q}_i
\mathbf{K}_j^\top/\sqrt{d_k}$, e si aggiorna l'output *sul posto*, senza mai
scrivere l'intera matrice $\mathbf{S}$ in HBM.

Qui $\mathbf{Q}_i$ è il blocco di query corrente (il blocco che resta fermo in
shared memory) e $\mathbf{K}_j, \mathbf{V}_j$ il blocco di key e value in
transito, per cui $\mathbf{S}_{ij}$ è la tessera di punteggi che nasce dal
loro incontro: $B_r \times B_c$, cioè $B_r$ righe di query per $B_c$ chiavi
(le due misure del blocco, che il kernel sceglie in base a quanta shared memory
ha), e non l'intera riga di $\mathbf{S}$.

Quest'ordine dei cicli, con il blocco di query fermo e
$\mathbf{K},\mathbf{V}$ che scorrono, è quello reso canonico dalla seconda
versione dell'algoritmo, FlashAttention-2; l'articolo del 2022 li annidava al
contrario, ma l'idea non cambia. La memoria on-chip trattiene solo i tile
correnti; la HBM vede scorrere $\mathbf{K},\mathbf{V}$ una volta per ogni
blocco di query, e la matrice $\mathbf{S}$ mai. La memoria extra scende così da
$O(N^2)$ a $O(N)$, perché da scrivere restano solo l'output e le statistiche
di riga.

Sui FLOP, al contrario di quanto si sente ripetere, l'algoritmo non risparmia
nulla. In avanti i calcoli sono quelli di prima, a meno del riscalamento
dell'accumulatore a ogni blocco (l'extra non-matmul che FlashAttention-2
limita). All'indietro no, perché non avendo salvato $\mathbf{S}$ e
$\mathbf{P}$ il `backward` deve ricalcolarle da $\mathbf{Q}, \mathbf{K},
\mathbf{V}$, e per questo basta salvare l'output e le statistiche di riga. Il
bilancio è di $4N^2 d_k$ FLOP in avanti, mentre all'indietro è di $8N^2 d_k$
nella versione standard e di $10 N^2 d_k$ con FlashAttention, cioè un quarto in
più sul passaggio all'indietro e un sesto in più sul totale.

Sull'addestramento di GPT-2 medium, in avanti e all'indietro, l'articolo
riporta 75,2 GFLOP contro 66,6, cioè $+12{,}9\,\%$. Sono numeri riportati, e
la figura non dice che cosa contenga il conteggio; il rialzo, minore del
sesto calcolato sui soli prodotti fra matrici, fa pensare a operazioni
uguali nei due casi che lo diluiscono. A fronte di questo, il
traffico verso la HBM scende di circa nove volte (4,4 contro 40,3 GB) e il
tempo di quasi sei (7,3 contro 41,7 ms), e l'articolo lo dichiara: «even with
the increased FLOPs due to recomputation».

Il baratto è dunque calcolo in cambio di traffico, lo stesso del *gradient
checkpointing* con cui si ricalcolano le attivazioni invece di conservarle.
Conviene perché i FLOP in più sono prodotti fra matrici, che i tensor core
eseguono a un costo marginale quasi nullo, mentre i byte risparmiati sono
accessi alla HBM, la risorsa scarsa. Lo stesso scambio ricorre nel pipeline
parallelism della {doc}`sezione sul parallelismo distribuito
<parallelismo-distribuito>` e in {doc}`Mamba </StateSpaceModel/mamba>`.

Anche il traffico verso la HBM crolla. L'articolo lo conta in
$\Theta(N^2 d_k^2 / M_\text{chip})$ accessi, con lo stesso $M_\text{chip}$ del
GEMM, la memoria veloce disponibile, contro il $\Theta(N d_k + N^2)$
dell'attenzione standard. Resta quadratico in $N$, ma diviso per un fattore
dell'ordine di $M_\text{chip}/d_k^2$, che dà l'ordine di grandezza del
guadagno e non il guadagno, perché le costanti nascoste nei $\Theta$ non
valgono 1 e le tessere non occupano tutta la memoria veloce.

In concreto, con i 192 KB di SRAM per SM di una A100, poco meno di centomila
elementi in `float16`, il rapporto varrebbe 24 con $d_k = 64$ e 6 con
$d_k = 128$. L'articolo, discutendo il teorema, prende un $M_\text{chip}$
«around 100KB», circa la metà, e su GPT-2 medium ($d_k = 64$) misura un
traffico diviso per circa nove, che su un carico memory-bound è tanto.

Il risultato vale per $d_k \le M_\text{chip} \le N d_k$, e in quel regime non
si fa di meglio. L'articolo dimostra infatti che nessun algoritmo di attenzione
esatta scende a $o(N^2 d_k^2 / M_\text{chip})$ accessi per tutti i valori di
$M_\text{chip}$ di quell'intervallo. È un limite inferiore su un
intervallo di taglie, non per ogni singola scheda. Il meccanismo è il tiling in
shared memory del GEMM applicato all'attenzione.

La difficoltà tecnica sta nella softmax, che *non* è elemento-per-elemento:
normalizza per righe, e la normalizzazione richiede in teoria di aver già visto
tutti i punteggi della riga. Scorrere $\mathbf{K}$ a blocchi, invece, significa
vedere i punteggi un pezzo per volta, e qui entra in gioco la online softmax.
`````

### La online softmax, con i numeri

La online softmax calcola le percentuali vedendo i punteggi a blocchi, senza
averli mai tutti sotto gli occhi insieme, e dà comunque il risultato esatto.
Bastano due numeri di riepilogo: $m$, il punteggio più alto visto finora, e
$l$, il totale accumulato finora. In {numref}`fig-flash-attention-blocchi` si
vedono aggiornarsi blocco dopo blocco, mentre le celle fuori dalla finestra
restano vuote.

```{figure} ../figures/flash-attention-blocchi.svg
:name: fig-flash-attention-blocchi
:alt: Una riga di otto punteggi divisa in quattro blocchi da due: una finestra scorre da sinistra a destra e in ogni istante mostra i numeri di un solo blocco, mentre fuori le celle restano vuote perché la matrice dei punteggi non viene mai scritta. Sotto, una tabella si riempie riga per riga con il massimo del blocco, il massimo corrente m, il fattore di riscalatura alfa, la somma corrente l e l'output accumulato O: quando arriva un massimo più grande alfa scende sotto 1 e l'accumulatore viene riscalato. Alla fine O diviso l coincide con la softmax calcolata in un colpo solo.
:width: 95%

Le due statistiche al lavoro su una riga di otto punteggi letti a due a due.
I punteggi sono $1, 3, 2, 4, 1, 0, 5, 2$, e i value che si portano dietro sono
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
Si comincia dal caso facile, il totale a rate. Devi dire quanto pesa ogni
sacco rispetto al totale di tutti, ma sulla bilancia ne stanno due per volta.
Tieni allora un foglietto con «totale finora». Primo mucchietto, 10 e 30: il
foglietto dice 40. Secondo mucchietto, 20 e 40: il foglietto dice 100. A quel
punto dividi ciascun sacco per 100 e hai le percentuali (10%, 30%, 20%, 40%),
che sono le stesse che avresti ottenuto stendendo tutti i sacchi per terra in
una volta sola. Nessuna approssimazione: la stessa somma, fatta a rate.

I foglietti però sono due, non uno, e il secondo è la parte meno ovvia. Nel
calcolo vero i punteggi, prima di essere sommati, non vengono presi così come
sono, ma passano per un'operazione che li ingigantisce. La regola è semplice:
ogni punto in più moltiplica per 2,7 circa (è il numero $e$, che vale
$2{,}718\ldots$, lo stesso della softmax incontrata fra le {doc}`funzioni di
attivazione </RetiNeurali/funzioni-attivazione>`). Un punteggio di due punti
più alto pesa quindi $2{,}7 \times 2{,}7$, più di sette volte tanto; dieci
punti più alto pesa ventiduemila volte tanto. Con punteggi
anche moderatamente alti si arriva a numeri che il computer non riesce più a
scrivere.

Il rimedio è quello delle classifiche: invece del punteggio assoluto si segna
la distanza dal primo, «a tre punti dal record». Serve dunque un secondo
foglietto con il punteggio più alto visto finora, e ogni cosa si misura
rispetto a lui.

Se in un mucchietto salta fuori un punteggio più alto del record, il totale
accumulato era espresso rispetto al vecchio record e va riespresso rispetto al
nuovo. A questo punto torna utile quel «2,7 a punto»: se il record sale di un
punto, tutto il materiale già messo da parte va diviso per 2,7, cioè
moltiplicato per 0,37. Una moltiplicazione sola, si aggiorna il foglietto e si
tira avanti.

Ad esempio, con quattro punteggi, 1, 3, 2, 4, letti a due a due, si procede
così. Primo mucchietto, record 3: l'1 sta due punti sotto e pesa 0,135 (cioè 1
diviso 2,7 due volte), il 3 pesa 1, e il totale dice 1,135.

Secondo mucchietto: arriva il 4, il record sale di un punto, e il totale già
messo da parte va moltiplicato per 0,368, il «diviso 2,7» di prima, e 1,135
diventa 0,418. Poi si aggiungono i due nuovi, il 2 (due punti sotto il
record, 0,135) e il 4 (che pesa 1): il totale fa 1,553, uguale al totale che
darebbe guardare i quattro punteggi tutti insieme. Sono i primi quattro della
riga che la {numref}`fig-flash-attention-blocchi` fa scorrere.

Resta la miscela. I punteggi non servono per sé, servono a dosare: ogni sacco
contiene una farina diversa, e alla fine si vuole una miscela in cui ciascuna
entra in proporzione al proprio peso. Anche la miscela si compone a rate, un
mucchietto per volta, con le dosi annotate rispetto al record del momento; e
quando il record sale, la conversione non riguarda solo il totale sul
foglietto, riguarda anche le dosi già annotate, con la stessa moltiplicazione
per 0,37. All'ultimo mucchietto la miscela si divide per il totale e coincide
con quella che darebbe il calcolo fatto in un colpo unico.

Quando poi, al ritorno, la rete deve rifare i calcoli per imparare, dei due
foglietti ne basta uno: per ogni parola si conserva un numero solo, che
riassume record e totale insieme.
`````

`````{tab} Superiore
Rifacciamo a mano il calcolo su una riga di quattro punteggi
$\mathbf{s} = (1, 3, 2, 4)$ (i valori di
$\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}$ per una query contro quattro key). Il
calcolo *in un colpo solo*, con la solita stabilizzazione che sottrae il
massimo per non far esplodere gli esponenziali, dà

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

Il totale $l_2 \approx 1{,}553$ coincide con la somma calcolata in un colpo
solo, per cui la online softmax dà gli stessi pesi.

In generale, quando arriva un nuovo blocco con massimo locale $\tilde m$, le
regole di aggiornamento sono

$$
\begin{aligned}
m^{\text{new}} &= \max(m, \tilde m), \\
l^{\text{new}} &= e^{\,m - m^{\text{new}}}\, l + \!\sum_{i \in \text{blocco}}\! e^{\,s_i - m^{\text{new}}}, \\
\mathbf{o}^{\text{new}} &= e^{\,m - m^{\text{new}}}\, \mathbf{o} + \!\sum_{i \in \text{blocco}}\! e^{\,s_i - m^{\text{new}}}\, \mathbf{v}_i,
\end{aligned}
$$

dove $\mathbf{o}$ è la riga di uscita accumulata, cioè la somma pesata dei
vettori $\mathbf{v}_i$, e il fattore $e^{\,m - m^{\text{new}}}$ corregge ciò
che avevamo già sommato quando compare un massimo nuovo. Si parte da
$m = -\infty$, $l = 0$ e $\mathbf{o} = \mathbf{0}$, e alla fine si divide,
$\mathbf{o} \leftarrow \mathbf{o}/l$. Oltre all'accumulatore $\mathbf{o}$, lo
stato è di due scalari per riga, e la matrice $N \times N$ non viene mai
scritta.

Per il passo all'indietro basta salvare uno scalare solo, che li combina,
$L = m + \log l$ (il logaritmo della somma degli esponenziali, *logsumexp*), e
da lì i pesi si ricostruiscono come $P_{ij} = e^{S_{ij} - L_i}$ senza
rifare il massimo {cite}`dao2023flashattention2`.

Una riga che finora ha visto solo punteggi mascherati, cioè posti a $-\infty$
(una maschera di riempimento, una finestra locale), ha ancora $m = -\infty$, e
$m - m^{\text{new}}$ darebbe $-\infty - (-\infty)$, cioè NaN; per questo i
kernel in quel caso sostituiscono 0 al massimo. Con la maschera causale (ogni
posizione vede solo quelle che la precedono) i blocchi interamente mascherati,
invece, si saltano, e il lavoro quasi si dimezza.
`````

## Il guadagno e i suoi limiti

La memoria richiesta dall'attenzione cresce ora in proporzione alla lunghezza
del testo, invece che con il suo *quadrato*. A testi corti il guadagno è
modesto, ma cresce con la lunghezza, ed è massimo dove l'attenzione standard
esauriva la memoria della scheda o passava quasi tutto il tempo a spostare la
matrice $N \times N$.

Una versione successiva, **FlashAttention-2** {cite}`dao2023flashattention2`,
avvicina il kernel all'efficienza di un GEMM con tre modifiche. La prima rinvia
alla fine del ciclo sui blocchi la divisione per la somma $l$, riducendo le
operazioni che i tensor core non accelerano. La seconda distribuisce su SM
diversi anche i blocchi di query della stessa testa, così che un testo lungo
con pochi esempi tenga occupata tutta la scheda (i modelli eseguono
l'attenzione in più copie parallele, le *teste*). La terza, dentro il blocco,
divide il lavoro fra i warp per righe di query invece che per colonne di key,
eliminando lo scambio di risultati parziali in shared memory. Ne esce un tempo
grosso modo dimezzato rispetto alla prima versione.

Tutto questo risolve un problema ben preciso, ma è facile
attribuirgli i meriti di un'altra tecnica. Ci sono due momenti in cui la
tabella dei confronti viene costruita tutta: mentre il modello impara, e
nella prima passata con cui legge la domanda che gli abbiamo fatto, che si
chiama *prefill*. In tutti e due il vincolo è quella tabella, e FlashAttention
lo toglie.

Mentre il modello *scrive* la risposta, invece, la tabella non esiste
nemmeno, perché si procede un token per volta (è la *decodifica*) e i
confronti da fare sono una riga sola. In questa fase il peso è un altro, la
**KV cache**: le key e i value che l'attenzione ha già calcolato per i token
precedenti, conservati per non ricalcolarli a ogni token nuovo (vanno tenuti
entrambi, perché la key serve al confronto e il value alla media pesata).

La KV cache cresce in proporzione alla lunghezza del testo, non al suo
quadrato, ma pesa. Lo mostra un modello vero, il più piccolo di
Llama 3, con otto miliardi di parametri {cite}`grattafiori2024llama3`:

- 32 strati, ognuno con la propria cache;
- 32 teste di query ma solo 8 di key e value, perché gruppi di quattro teste
  condividono le stesse key e gli stessi value (*grouped-query attention*,
  GQA {cite}`ainslie2023gqa`), e la cache è già divisa per quattro;
- 128 numeri per testa, e 2 byte per numero.

Per ogni token si conservano key e value in ogni strato, cioè
$2 \times 32 \times 8 \times 128 \times 2 = 131\,072$ byte. Su centomila token
di contesto fanno 13 GB, per una conversazione sola, su una scheda che di GB ne
ha 80.

FlashAttention non agisce sulla KV cache; se ne occupano altre tecniche, che
stanno nella sezione sui
{doc}`grandi modelli linguistici </Transformers/llm>` e in quella su
{doc}`prefill e decodifica </MLOps/metriche-di-servizio>`. Non riduce nemmeno
il numero di calcoli da fare, che resta proporzionale al quadrato della
lunghezza, e a quello pensa il
{doc}`capitolo sull'attenzione lineare </AttenzioneLineare/overview>`.

Il kernel che realizza l'idea è complicato: indici, gestione della shared
memory, casi limite della **maschera causale**, la regola che impedisce a ogni
parola di sbirciare quelle che vengono dopo di lei. Per questo, di norma, non
si scrive a mano. In PyTorch la funzione `scaled_dot_product_attention`
sceglie fra le implementazioni disponibili (i *backend*) quella adatta alla
scheda, e su GPU recenti, con dati in `float16` o `bfloat16`, è in genere
FlashAttention. Nel codice la riga che conta è quella che la chiama; il resto
prepara i numeri.

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

Sotto quella riga gira in genere il kernel di FlashAttention. Capirne l'idea
serve a sapere *quando* e *perché* aiuta; l'implementazione resta a chi la
mantiene ottimizzata generazione dopo generazione.

## La frontiera dei kernel veloci

FlashAttention porta all'estremo il principio delle tecniche viste per le GPU:
chiedere i dati contigui invece che sparsi, portare una tessera in shared
memory e riusarla, fare tre operazioni in un viaggio invece che in tre, e ora
non scrivere affatto una tabella che serviva solo di passaggio. Ogni volta si
fanno più operazioni per ogni byte spostato e si tiene il byte vicino a chi
calcola. Quando i byte non si possono ridurre più, resta da farli viaggiare
mentre le unità di calcolo lavorano.

I kernel più veloci di oggi portano queste idee ancora più in là, con tre
leve.

`````{tab} Elementare
Sono le tre regole di una catena di montaggio ben organizzata.

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
viaggiano più in fretta. C'è però un rovescio: più la macchina è veloce, più è
facile che a mancare siano i pezzi e non le braccia.

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
  e guadagnano formati numerici più compatti, **FP8** (8 bit) su Hopper
  e FP4 su Blackwell {cite}`shah2024flashattention3`, che dimezzano o riducono a
  un quarto i byte da spostare rispetto a `float16`, nella stessa logica della
  precisione mista della sezione {doc}`Prestazioni e scala
  </PyTorch/prestazioni>`. Più i tensor core sono veloci, però, più è facile
  ritrovarsi memory-bound: il ginocchio del roofline si sposta a destra, e la
  partita torna a giocarsi sui byte.
- **Warp specialization.** Invece di far fare a ogni warp un po’ di tutto, gli si
  assegnano *ruoli*: alcuni warp fanno solo da *producer* (caricano i dati dalla
  HBM), altri da *consumer* (calcolano sui tensor core), coordinati come i
  reparti di una catena di montaggio. La specializzazione tiene le unità di
  calcolo sempre rifornite e i canali di memoria sempre occupati.

FlashAttention-3 {cite}`shah2024flashattention3`, scritta nel 2024 per le GPU
Hopper, applica tutte e tre le leve e vi aggiunge due tecniche proprie. La
prima alterna due gruppi di warp (*ping-pong*), così la softmax dell'uno, che
i tensor core non accelerano, gira mentre l'altro esegue i propri GEMM. La
seconda, in FP8, precede la quantizzazione a blocchi con una rotazione casuale
(una trasformata di Hadamard con segni casuali, detta *incoherent processing*)
che sparpaglia i valori anomali prima di arrotondare.

Algoritmo, online softmax e accessi alla HBM restano quelli del 2022; quello
che cambia è quanta parte del movimento dei dati il calcolo riesce a coprire.
Chi vuole seguire queste tecniche fino al codice trova una trattazione avanzata
nel corso *Modern GPU Programming for MLSys* di mlc.ai.
`````

Tutto questo avviene dentro una singola scheda: nel modo in cui esegue, nella
memoria che la rifornisce, nei due calcoli (il prodotto fra matrici e
l'attenzione) in cui si concentra quasi tutta l'aritmetica di una rete. Quando
il modello non entra in una scheda sola, serve il {doc}`parallelismo
distribuito <parallelismo-distribuito>`.

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
- A rendere possibile il lavoro a blocchetti è la online softmax, la pesata dei
  sacchi due per volta con un foglietto per il totale finora: qui i foglietti
  sono due, il totale e il punteggio più alto visto fin lì, e alla fine danno le
  stesse percentuali del calcolo in un colpo unico.
- Non fa meno conti degli altri: ne fa altrettanti, e nel viaggio di ritorno
  (quello in cui la rete impara dai propri errori) qualcuno in più, perché
  avendo buttato i blocchetti se li deve rifare. È un baratto voluto: si
  aggiunge un po’ di calcolo, che è a buon mercato, per risparmiare tanti
  viaggi, che sono cari.
- Il guadagno: la memoria non cresce più con il quadrato della lunghezza del
  testo ma in proporzione ad essa, e sui testi lunghi (dove prima la GPU si
  fermava per memoria esaurita) il salto è grande. Questo vale però solo mentre
  il modello *impara* e mentre *legge*. Mentre scrive la risposta il peso è un
  altro, la KV cache, il taccuino di ciò che ha già letto, e quello resta. Una
  seconda versione, FlashAttention-2 {cite}`dao2023flashattention2`, ripartisce
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
  bottiglia è la memoria, non i FLOP, e lo è senza eccezioni: anche i due matmul,
  dominati dal traffico della matrice $N \times N$ (scritta dal primo, letta
  dal secondo), stanno a $\approx 63$ FLOP/byte contro un ginocchio di 161 su
  A100. Nella forma standard niente è compute-bound.
- FlashAttention {cite}`dao2022flashattention` è IO-aware: con il
  tiling di $\mathbf{Q},\mathbf{K},\mathbf{V}$ in shared memory e la
  online softmax non scrive mai la matrice $N \times N$ in HBM. Il risultato
  è esatto, non approssimato.
- La online softmax normalizza i punteggi a blocchi tenendo,
  oltre all'accumulatore, due scalari di stato (massimo corrente $m$ e somma
  corrente $l$) e ri-scalando ciò che ha già sommato quando compare un massimo
  nuovo: dà gli stessi pesi del calcolo in un colpo solo. Per il backward basta
  salvare il logsumexp $L = m + \log l$.
- Non fa meno FLOP: in avanti altrettanti, e nel backward $10N^2 d_k$ contro
  $8N^2 d_k$, perché $\mathbf{S}$ e $\mathbf{P}$ non sono salvate e vanno
  ricalcolate ($+25\,\%$ sul backward; l'articolo riporta $+12{,}9\,\%$ sul
  totale). È il baratto calcolo-per-traffico del *gradient checkpointing*, e
  conviene perché i FLOP in più sono matmul e i byte risparmiati sono HBM.
- Il guadagno: memoria da $O(N^2)$ a $O(N)$, grande accelerazione a sequenze
  lunghe in addestramento e in prefill. In decodifica il vincolo è un
  altro, la KV cache, lineare in $N$ ma pesante, e FlashAttention non la
  tocca (né rende l'attenzione sub-quadratica nei FLOP: quello è l'argomento
  del capitolo sull'attenzione lineare). FlashAttention-2
  {cite}`dao2023flashattention2` migliora ancora la ripartizione del lavoro. In
  PyTorch lo si usa via `scaled_dot_product_attention`.
- La frontiera dei kernel veloci ha due direzioni: nascondere il movimento dei
  dati dietro il calcolo (copie asincrone da Ampere, TMA da Hopper, warp
  specialization) e ridurre i byte per elemento con formati più corti (FP8,
  FP4).
```
`````
