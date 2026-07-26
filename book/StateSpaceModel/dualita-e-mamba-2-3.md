# La dualità: Mamba-2 e Mamba-3

Nella sezione precedente abbiamo visto Mamba pagare un prezzo per la sua
stessa forza. Rendendo $B$, $C$ e il passo $\Delta$ dipendenti dall'ingresso,
il sistema è diventato **selettivo** (capace di scegliere cosa ricordare in
base al contenuto) ma ha perso la tempo-invarianza, e con essa il kernel di
convoluzione fisso che permetteva l'addestramento parallelo. Al suo posto è
rimasto lo **scan**: un algoritmo ricorrente reso parallelo sfruttando
l'associatività, veloce, ma con un difetto nascosto. Lo scan lavora a suon di
somme e prodotti elemento per elemento, e le GPU moderne non danno il meglio
su quelle operazioni: il grosso della loro potenza sta nei **tensor core**,
unità dedicate a una cosa sola, moltiplicare matrici. Lo scan di Mamba li
lascia quasi spenti.

L'intuizione di Mamba-2, di Tri Dao e Albert Gu {cite}`dao2024mamba2`, è insieme
teorica e pratica, ed è la ragione per cui questa sezione chiude il cerchio del
capitolo. Teorica: gli State Space Model e l'attenzione non sono due famiglie
distinte, ma **due viste della stessa cosa**. Pratica: da quella equivalenza
discende un algoritmo che si scrive come una sequenza di moltiplicazioni di
matrici, riaccende i tensor core, e rende il modello parecchie volte più veloce.

## State Space Duality: un SSM è un'attenzione

Il titolo del paper è programmatico (*Transformers are SSMs*) ed è il rovescio
della medaglia di quello che avevamo incontrato nel capitolo sull'attenzione
lineare, *Transformers are RNNs* {cite}`katharopoulos2020transformers`. Lì
avevamo tolto la softmax dall'attenzione e scoperto sotto una rete ricorrente
a stato fisso. Qui si parte dall'altro capo (un SSM, cioè un sistema dinamico
discretizzato) e si scopre che, con la giusta restrizione, è **esattamente**
un'attenzione mascherata. La chiamano **State Space Duality** (SSD): la
dualità tra spazio degli stati e attenzione.

`````{tab} Elementare

Abbiamo già visto una dualità, in questo capitolo e nel precedente: la stessa
funzione calcolabile «passo dopo passo» (ricorrente) oppure «tutta insieme»
(convoluzione o attenzione). Qui la dualità è più profonda e riguarda due
oggetti che avevamo trattato come parenti lontani.

Immagina due dialetti che si sono sviluppati in valli diverse e che, messi a
confronto, risultano essere la stessa lingua. Da una parte gli State Space
Model, nati dalla teoria del controllo, con il loro stato che evolve nel tempo.
Dall'altra l'attenzione, nata dalla traduzione automatica, con la sua matrice
che confronta ogni parola con ogni altra. Mamba-2 dimostra che, appena si
sceglie la versione più semplice dello stato di un SSM, i due dialetti
coincidono: le stesse frasi, dette in due modi. Non «si assomigliano»: sono la
stessa identica operazione, scritta con simboli diversi.

E se sono la stessa operazione, si può calcolare in due modi: passo dopo passo,
come un SSM, oppure formando una grande tabella di confronti, come l'attenzione.
Il secondo modo è quello che le GPU adorano.

`````

`````{tab} Superiore

Riprendiamo la convenzione del capitolo: lo stato $S_t$ è una memoria
chiave→valore, aggiornata per prodotto esterno e letta con la query. Nel
capitolo sull'attenzione lineare avevamo messo in fila lo «zoo» delle
ricorrenze, e la riga di **Mamba-2** era il decadimento scalare

$$
S_t = \alpha_t\, S_{t-1} + v_t\, k_t^\top, \qquad o_t = S_t\, q_t,
$$

con transizione $\alpha_t I$ (uno scalare per l'identità). La SSD mostra che
questa è *precisamente* la forma cui si riduce un SSM quando si impone
$A = a_t I$: una matrice di stato scalare per l'identità. Basta identificare i
ruoli. Lo stato dell'SSM per una testa a dimensione $P$ è la matrice
$S_t \in \mathbb{R}^{P\times N}$; la matrice d'ingresso $B_t\in\mathbb{R}^{N}$
fa da **chiave** $k_t$, l'ingresso $x_t\in\mathbb{R}^{P}$ fa da **valore** $v_t$,
la matrice d'uscita $C_t\in\mathbb{R}^{N}$ fa da **query** $q_t$, e lo scalare
$a_t$ è il gate $\alpha_t$. La ricorrenza dell'SSM,

$$
S_t = a_t\, S_{t-1} + x_t\, B_t^\top, \qquad y_t = S_t\, C_t,
$$

è la stessa riga della tabella. Srotolandola dallo stato iniziale nullo, l'uscita
al passo $i$ è

$$
y_i = \sum_{j=1}^{i} \Big(\underbrace{\textstyle\prod_{k=j+1}^{i} a_k}_{\text{decadimento}}\Big)\,
      \big(C_i^\top B_j\big)\, x_j ,
$$

dove il fattore $\prod_{k} a_k$ è quanto è sopravvissuto, dal passo $j$ al passo
$i$, di ciò che era stato scritto. Raccogliamo tutti i passi in una sola
matrice. Impilando le query $C_i$, le chiavi $B_j$ e i valori $x_j$ nelle
righe di $C$, $B$, $X$, l'intera sequenza di uscite si scrive

$$
Y = \big(L \circ C B^\top\big)\, X, \qquad
L_{ij} = \begin{cases} \prod_{k=j+1}^{i} a_k & i \ge j \\ 0 & i < j \end{cases}
$$

dove $C B^\top$ è la matrice $L\times L$ di tutte le affinità query–chiave
(esattamente $QK^\top$ dell'attenzione), $\circ$ è il prodotto elemento per
elemento, e $L$ è una **maschera causale con decadimento**: azzera il futuro
(triangolo superiore) e pesa il passato con i prodotti degli scalari $a_t$.
Questa è, alla lettera, un'attenzione mascherata: la stessa
$\mathrm{softmax}(QK^\top)V$ dei Transformer {cite}`vaswani2017attention`, con
la softmax rimpiazzata dalla maschera $L$. La matrice $L$ ha una struttura
particolare, detta **1-semiseparabile**: ogni suo blocco interamente contenuto
nel triangolo inferiore ha rango uno, perché ogni elemento si fattorizza nei
prodotti cumulati degli $a_t$. È questa struttura a fare da ponte: i sistemi a
spazio di stati con transizione scalare *sono* le attenzioni con maschera
semiseparabile.

`````

Il risultato ha una lettura che vale la pena esplicitare, perché è il perno di
due capitoli. Nel capitolo sull'attenzione lineare avevamo costruito, un
gradino alla volta, uno «zoo» di ricorrenze (accumulo, decadimento scalare,
diagonale, delta rule) tutte con la stessa struttura: stato di dimensione
fissa, addestramento parallelo, inferenza ricorrente, e a cambiare solo la
**transizione di stato**. Mamba-2 occupa il gradino del decadimento scalare
$\alpha_t I$. Arrivando dai sistemi dinamici invece che dalla softmax, ci
ritroviamo esattamente lì: è la prova che le due strade (quella partita
dall'attenzione e quella partita da Kálmán) portavano alla stessa città. La
stessa funzione ha una forma **lineare e ricorrente**, di costo $O(L)$ nella
lunghezza (la vista «SSM»), e una forma **quadratica e attention-like**, con
la matrice $L\times L$ mascherata (la vista «attenzione»). Non è un'analogia:
è un'uguaglianza.

## Perché conviene: i tensor core

La dualità non sarebbe che un'eleganza teorica se non pagasse in velocità.
Paga, e la chiave è una restrizione apparentemente minima. Mamba-1 usava una
matrice di stato $A$ diagonale con $N$ valori **distinti** per canale; Mamba-2
impone che siano tutti **uguali**: $A = a_t I$, uno scalare per l'identità.
Sembra una perdita di espressività, ed è ciò che rende l'algoritmo esprimibile
come pura moltiplicazione di matrici.

`````{tab} Elementare

Immagina un'officina con un attrezzo formidabile ma specializzato: una pressa
che stampa lastre intere in un colpo solo, purché il pezzo abbia una certa
forma. Finché lavori a mano, pezzo per pezzo, la pressa resta ferma e tu vai
lentissimo. Se accetti di dare ai pezzi quella forma standard, puoi usarla, e
vai molto più veloce.

I tensor core della GPU sono quella pressa: sanno fare una cosa sola,
moltiplicare matrici, e la fanno a velocità impressionante. Lo scan di
Mamba-1, fatto di operazioni una-alla-volta, li teneva spenti. La piccola
rinuncia di Mamba-2 (imporre che quello scalare dello stato sia lo stesso per
tutti i canali di una testa) dà ai conti la «forma standard» che la pressa
accetta: tutto il calcolo diventa una sequenza di moltiplicazioni di matrici.
Il risultato è un livello che gira dalle due alle otto volte più in fretta di
quello di Mamba-1, a parità di lavoro. In più, potendo permettersi una memoria
più capiente, Mamba-2 allarga lo stato (da una manciata di numeri per canale a
diverse decine o centinaia) e lo organizza a **teste multiple**, esattamente
come l'attenzione.

`````

`````{tab} Superiore

Il motivo per cui il matmul batte lo scan è nell'hardware. Un tensor core esegue
un piccolo prodotto matrice–matrice per ciclo: su una GPU moderna è lì che
risiede la stragrande maggioranza dei FLOP disponibili. Un *selective scan* come
quello di Mamba-1 è invece una ricorrenza associativa fatta di moltiplicazioni
elemento per elemento e somme: parallelizzabile in $O(\log L)$ passi, ma su unità
generiche, molto meno dense di FLOP. Sta usando la frazione lenta della GPU.

Con $A = a_t I$ l'algoritmo pratico non forma davvero l'intera matrice
$L\times L$ (sarebbe $O(L^2)$ in memoria). Si adotta una **decomposizione a
blocchi** (*chunked scan*): la sequenza si spezza in blocchi di lunghezza $C$;
dentro ciascun blocco si calcola la forma quadratica, attention-like, come un
prodotto di matrici sui tensor core; tra un blocco e il successivo si passa
solo lo **stato** riassuntivo, con un termine di rango basso, in forma
ricorrente. Si interpola così tra le due viste della dualità (quadratica
dentro il blocco, lineare tra i blocchi) e il costo scende a $O(LC)$ tenendo
il grosso del lavoro su moltiplicazioni di matrici. Ne seguono tre vantaggi
concreti: il livello SSD è $2$–$8\times$ più veloce del selective scan di
Mamba-1; la struttura è **multi-head** come l'attenzione (dimensione di testa
$P$ tipicamente $64$ o $128$); e lo stato può crescere di un ordine di
grandezza (da $N=16$ in Mamba-1 a $N$ dell'ordine di $64$–$256$ e oltre in
Mamba-2), perché una memoria più grande, ora, non costa in velocità. Uno stato
più capiente è direttamente più memoria associativa: meno *crosstalk*,
richiamo più preciso.

`````

Vale la pena fermarsi sul senso di questa mossa. Mamba-1 aveva reso l'SSM
selettivo pagando con lo scan; Mamba-2 recupera il parallelismo pieno delle
matrici accettando una transizione di stato più semplice, e lo fa proprio
perché la dualità gli garantisce che quella forma più semplice è ancora
un'attenzione: non un modello impoverito, ma lo stesso oggetto scritto in modo
che l'hardware lo digerisca. È il compromesso tipico di questa famiglia:
qualche grado di libertà in meno sulla transizione, in cambio di forme
parallele che sfruttano le GPU.

## Mamba-3

L'ultimo anello, al momento in cui scriviamo, è **Mamba-3**, di Lahoti, Li e
colleghi con Dao e Gu, presentato come *Oral* a ICLR 2026
{cite}`lahoti2026mamba3`. È un lavoro molto recente, e conviene leggerlo per ciò
che aggiunge di qualitativo più che per le cifre puntuali, ancora da assestare.
Le novità rispetto a Mamba-2 sono tre, e tutte lavorano sul *come* lo stato
evolve, non sulla struttura generale.

La prima riguarda la **discretizzazione**, cioè il modo di trasformare il sistema
continuo in una ricorrenza, che avevamo introdotto all'inizio del capitolo.

`````{tab} Elementare

Ricordiamo il problema: un sistema che scorre nel tempo va «campionato» a
intervalli, e bisogna indovinare cosa succede *tra* un campione e l'altro.
Mamba usava la regola più semplice, che tiene l'ingresso costante
nell'intervallo: un gradino. È come approssimare l'area sotto una curva con
dei rettangoli: rapido, ma con un errore che si accumula.

Mamba-3 usa una regola più accurata, che approssima quel tratto con un
**trapezio** invece che con un rettangolo: la stessa idea della regola del
trapezio che si incontra in analisi numerica. L'errore a ogni passo è più
piccolo, e la conseguenza pratica è curiosa: Mamba-1 e Mamba-2 avevano
bisogno, prima dell'SSM, di una **piccola convoluzione causale** (un
mini-filtro che mescola qualche token vicino) per funzionare bene. Con la
discretizzazione più precisa, quel pezzo aggiuntivo diventa **opzionale**: il
modello lavora bene anche senza. Una regola migliore per fare i conti, e una
stampella in meno.

`````

`````{tab} Superiore

Mamba e Mamba-2 discretizzano con lo **zero-order hold**, che equivale a
un'integrazione del prim'ordine (Eulero): l'errore locale su un passo
$\Delta t$ è dell'ordine di $O(\Delta t^2)$. Mamba-3 adotta una
discretizzazione **esponenziale-trapezoidale**, un'integrazione del
second'ordine che stima il tratto con la media dei valori agli estremi (la
regola del trapezio applicata alla dinamica): l'errore locale scende a
$O(\Delta t^3)$. La conseguenza riportata nel paper è che la **short causal
convolution** posta prima dell'SSM (presente in tutti i blocchi Mamba
precedenti come stabilizzatore) diventa **opzionale** senza perdita di
qualità: la discretizzazione più fine recupera da sola l'effetto di
mescolamento locale che quel filtro forniva.

`````

La seconda novità è la più concettuale, ed è quella che riaggancia questo
capitolo ai Transformer.

`````{tab} Elementare

Fino a qui lo stato di un SSM è stato una collezione di numeri che possono
solo crescere o sbiadire: salire di volume e poi spegnersi, come l'eco nella
valle. Mamba-3 permette allo stato di **ruotare**, non solo di affievolirsi. È
come passare da una manopola del volume a una lancetta che può girare su un
quadrante: oltre a «quanto forte», ora c'è un «dove sto puntando».

Perché serve? Ci sono compiti in cui la risposta dipende dal *contare* o dal
*tenere il segno*: capire se il numero di parentesi aperte è pari o dispari,
seguire un'aritmetica a modulo, tracciare uno stato che si alterna. Una memoria
che sa solo sbiadire fatica; una che sa ruotare può, letteralmente, «girare la
lancetta» a ogni passo e ricordare a che punto del ciclo si trova. Gli
ingegneri lo chiamano *state tracking*, tenere traccia dello stato, ed è
storicamente un punto debole delle ricorrenze lineari.

`````

`````{tab} Superiore

Mamba-3 introduce **transizioni a valori complessi**: la dinamica dello stato
non è più un semplice decadimento reale, ma una moltiplicazione per un numero
complesso, che ha un modulo (il decadimento, come prima) e una **fase** (una
rotazione). Nel piano complesso, moltiplicare per $e^{i\theta}$ è ruotare di
un angolo $\theta$; ripetendo il passo, lo stato percorre un cerchio. È
esattamente ciò che serve per rappresentare fenomeni periodici (parità,
aritmetica modulare) che un decadimento puramente reale non può codificare, e
il paper documenta un netto miglioramento sui compiti di **state tracking**.

Il legame con i Transformer è preciso. Il paper mostra che l'SSM complesso con
discretizzazione trapezoidale equivale a un **RoPE data-dipendente** applicato
alle matrici $B$ e $C$. RoPE (la *Rotary Position Embedding* che avevamo solo
nominato nel capitolo sui Transformer, tra le codifiche posizionali relative
più recenti) inietta la posizione ruotando query e key di un angolo
proporzionale all'indice del token. Qui accade lo stesso, con due differenze:
le rotazioni si applicano alle controparti SSM di key e query ($B$ e $C$), e
l'angolo non dipende solo dalla posizione ma dai **dati**, perché il passo
$\Delta$ è selettivo. È l'ennesimo ponte tra le due famiglie: la codifica
posizionale rotazionale dei Transformer riemerge, spontaneamente, come la fase
di una dinamica di stato complessa.

`````

La terza novità è più ingegneristica. Mamba-2 e i suoi predecessori sono,
nella loro forma base, sistemi a **singolo ingresso e singola uscita** (SISO):
ogni canale evolve con un proprio stato scalare, indipendente. Mamba-3 propone
una formulazione **MIMO** (*multi-input multi-output*), in cui più ingressi e
più uscite condividono lo stesso stato attraverso matrici $B$ e $C$ non più
vettoriali ma di rango maggiore. L'effetto tecnico è aumentare l'**intensità
aritmetica** (il numero di operazioni per ogni byte letto dalla memoria) che è
proprio ciò che tiene occupati i tensor core: si fa più lavoro utile per ogni
accesso in memoria. Il guadagno pratico riportato è qualità migliore **senza**
aumentare la latenza di decodifica, cioè senza rallentare la generazione token
per token. In coerenza con la cautela dovuta a un lavoro appena uscito, ci
fermiamo alle novità qualitative: la direzione è chiara (stato più accurato,
più espressivo e meglio calibrato sull'hardware), mentre i numeri esatti
andranno confermati man mano che il modello viene ripreso e riprodotto.

## Da S4 a Mamba-3: cosa è cambiato

Conviene, a questo punto, riavvolgere l'intero arco degli State Space Model,
perché ogni tappa ha smontato un pezzo diverso del problema e la somma racconta
una storia pulita.

Si parte da **S4**: un sistema lineare *tempo-invariante*, inizializzato con
HiPPO per avere memoria a lungo raggio e reso efficiente da una struttura
diagonale-più-basso-rango. Il suo pregio (e insieme il suo limite) è la
tempo-invarianza: tratta ogni token con la stessa regola fissa, il che gli dà
la doppia forma ricorrente/convoluzionale ma gli impedisce di *scegliere* cosa
ricordare in base al contenuto.

Arriva **Mamba**, che rompe la tempo-invarianza rendendo $B$, $C$ e $\Delta$
funzione dell'ingresso: il sistema diventa **selettivo**, sa filtrare il
rilevante dall'irrilevante, ma perde il kernel di convoluzione e deve
affidarsi allo scan; veloce, però lontano dai tensor core.

Poi **Mamba-2**, che riconcilia l'SSM con l'attenzione. La dualità dello
spazio degli stati mostra che la versione a transizione scalare è
un'attenzione mascherata semiseparabile; la restrizione $A = a_t I$ trasforma
il calcolo in moltiplicazioni di matrici, riaccende i tensor core, allarga lo
stato e lo organizza a teste. È il punto in cui le due strade di questo libro
(attenzione lineare e sistemi dinamici) si rivelano una sola.

Infine **Mamba-3**, che non cambia l'impianto ma ne raffina la dinamica:
discretizzazione trapezoidale più accurata (che rende opzionale la
convoluzione causale), stato **complesso** capace di ruotare (che riporta a
galla RoPE e migliora lo state tracking), formulazione MIMO (che spreme meglio
l'hardware). Da un sistema lineare invariante e a lungo raggio a uno
selettivo, poi riconciliato con l'attenzione e reso veloce, poi affinato nel
modo in cui lo stato evolve: è la parabola di una singola, ostinata idea
(comprimere il passato in uno stato di dimensione fissa) che a ogni passo si
avvicina un po' di più al meglio dei Transformer senza rinunciare al costo
lineare.

```{admonition} Da ricordare
:class: important
- **Mamba-2** nasce da un problema pratico: lo scan di Mamba-1 non usa i
  **tensor core** della GPU, l'hardware dedicato a moltiplicare matrici, e lascia
  gran parte della potenza inutilizzata.
- La **State Space Duality** (Dao e Gu, ICML 2024) è il ponte esplicito con il
  capitolo sull'attenzione: un SSM con matrice di stato $A = a_t I$ (scalare per
  l'identità) è **esattamente** un'attenzione mascherata, con una maschera
  causale $1$-semiseparabile. La stessa funzione ha una forma lineare/ricorrente
  $O(L)$ e una quadratica/attention-like.
- È lo stesso gradino (il decadimento scalare $\alpha_t I$) che occupava la
  riga «Mamba-2» nello «zoo» delle ricorrenze lineari: le due strade,
  dall'attenzione e dai sistemi dinamici, arrivano allo stesso posto.
- La restrizione $A = a_t I$ (diagonale tutta uguale, mentre Mamba-1 aveva valori
  distinti) rende il calcolo pura **moltiplicazione di matrici**: livello
  $2$–$8\times$ più veloce, struttura **multi-head**, stato molto più grande (da
  $N=16$ a $64$–$256$ e oltre).
- **Mamba-3** (Lahoti et al., ICLR 2026, Oral) raffina la dinamica con tre mosse:
  discretizzazione **trapezoidale** (secondo ordine, che rende opzionale la
  convoluzione causale), stato **complesso** con aggiornamenti **rotazionali**
  (migliore *state tracking*, con un legame formale al **RoPE** data-dipendente
  su $B$ e $C$), e formulazione **MIMO** (più qualità senza aumentare la latenza
  di decodifica). Lavoro recente: novità qualitative solide, cifre da confermare.
- L'arco **S4 → Mamba → Mamba-2 → Mamba-3**: da tempo-invariante a lungo
  raggio, a selettivo, a riconciliato con l'attenzione e veloce, a raffinato
  nella dinamica (sempre la stessa idea di comprimere il passato in uno stato
  di dimensione fissa).
```
