# La dualità: Mamba-2 e Mamba-3

Nella sezione precedente abbiamo visto Mamba (che da qui in avanti chiameremo
**Mamba-1**, per distinguerlo dai suoi successori) pagare un prezzo per la sua
stessa forza. Lasciando decidere all'ingresso quanto scrivere e quanto
dimenticare, il sistema è diventato **selettivo**, ma ha perso la regola fissa,
e con essa il filtro unico che permetteva di addestrarlo tutto in una volta. Al
suo posto è rimasto lo **scan**: la catena svolta a gruppi invece che in fila,
veloce, ma con un difetto nascosto.

Il difetto sta nel tipo di conti che lo scan fa fare. Una scheda grafica sa
fare due cose, e non le fa affatto alla stessa velocità. La prima è prendere
due numeri, moltiplicarli, e ripetere: operazioni minute, una per volta,
ciascuna con i suoi due numeri da andare a prendere in memoria. La seconda è
moltiplicare fra loro due **tabelle** di numeri (in matematica una tabella di
numeri si chiama **matrice**, e da qui in poi le due parole vogliono dire la
stessa cosa), che è un'operazione sola con dentro migliaia di moltiplicazioni
tutte uguali, disposte in un ordine noto in anticipo: si caricano i numeri una
volta e si fa tutto il lavoro sul posto. Per
questa seconda operazione, e solo per questa, le schede moderne hanno un
reparto dedicato, i **tensor core**, ed è lì che sta la stragrande maggioranza
della loro potenza. Lo scan di Mamba-1 fa conti del primo tipo, e quel reparto
lo lascia quasi spento.

L'intuizione di Mamba-2, di Tri Dao e Albert Gu {cite}`dao2024mamba2`, è insieme
teorica e pratica, ed è la ragione per cui questa sezione chiude il cerchio del
capitolo. Teorica: gli State Space Model e l'attenzione non sono due famiglie
distinte, ma **due viste della stessa cosa**. Pratica: da quella equivalenza
discende un algoritmo che si scrive come una sequenza di moltiplicazioni di
matrici, cioè che riporta il calcolo proprio sul reparto dove sta quasi tutta
la potenza della scheda.

## State Space Duality: un SSM è un'attenzione

Il risultato, detto in una riga, è questo: una macchina a spazio degli stati,
purché si accetti di semplificarne un pezzo, non somiglia all'attenzione, *è*
l'attenzione. Più precisamente un’**attenzione mascherata**, cioè
un'attenzione a cui è vietato guardare avanti, che confronta ogni parola solo
con quelle che l'hanno preceduta. Gli autori chiamano questo fatto **State
Space Duality** (SSD), la dualità fra spazio degli stati e attenzione.

Il titolo del loro articolo è programmatico, *Transformers are SSMs*, ed è il
rovescio della medaglia di quello che avevamo incontrato nel capitolo
sull'attenzione lineare, *Transformers are RNNs*
{cite}`katharopoulos2020transformers`. Lì avevamo tolto dall'attenzione il
pezzo che costava di più e trovato sotto una rete ricorrente a stato fisso; qui
si parte dall'altro capo, da un sistema dinamico misurato a intervalli, e si
arriva all'attenzione.

`````{tab} Elementare

Abbiamo già visto una dualità, in questo capitolo e nel precedente: la stessa
funzione calcolabile «passo dopo passo» (ricorrente) oppure «tutta insieme»
(convoluzione o attenzione). Qui la dualità è più profonda e riguarda due
oggetti che avevamo trattato come parenti lontani.

Immagina due dialetti che si sono sviluppati in valli diverse e che, messi a
confronto, risultano essere la stessa lingua. Da una parte gli State Space
Model, nati dalla teoria del controllo, con il loro stato che evolve nel tempo.
Dall'altra l'attenzione, nata dalla traduzione automatica, con la sua tabella
che confronta ogni parola con ogni altra. Mamba-2 dimostra che i due dialetti
coincidono, a una condizione: che si prenda la versione più semplice del
riassunto, quella in cui tutte le sue caselle sbiadiscono alla stessa velocità
invece che ognuna alla propria. Allora sono le stesse frasi, dette in due modi.
Non «si assomigliano»: sono la stessa identica operazione, scritta con simboli
diversi.

E se sono la stessa operazione, si può calcolare in due modi: passo dopo passo,
come un SSM, oppure formando una grande tabella di confronti, come l'attenzione.
Il secondo modo è quello che le GPU adorano.

`````

`````{tab} Superiore

Riprendiamo la convenzione del capitolo: lo stato $\mathbf{S}_t$ è una memoria
chiave→valore, aggiornata per prodotto esterno e letta con la query. Nel
capitolo sull'attenzione lineare avevamo messo in fila lo «zoo» delle
ricorrenze, e la riga di **Mamba-2** era il decadimento scalare

$$
\mathbf{S}_t = \alpha_t\, \mathbf{S}_{t-1} + \mathbf{v}_t\, \mathbf{k}_t^\top, \qquad \mathbf{o}_t = \mathbf{S}_t\, \mathbf{q}_t,
$$

con transizione $\alpha_t \mathbf{I}$ (uno scalare per l'identità). Qui, e solo
qui, il lato da cui la transizione moltiplica lo stato non conta: uno scalare
commuta, mentre un fattore diagonale o di rango uno andrebbe scritto a destra,
come vedremo nell'ultima sezione. La SSD mostra che
questa è *precisamente* la forma cui si riduce un SSM quando si impone
$\mathbf{A} = a\mathbf{I}$, con $a$ scalare fisso (uno per testa): la discretizzazione fa il
resto, perché la transizione discreta diventa $\bar{\mathbf{A}}_t = a_t \mathbf{I}$ con
$a_t = e^{\Delta_t a}$, data-dipendente attraverso $\Delta_t$. Basta
identificare i ruoli. Lo stato dell'SSM per una testa a dimensione $P$ è la
matrice $\mathbf{S}_t \in \mathbb{R}^{P\times N}$; la matrice d'ingresso
$\mathbf{B}_t\in\mathbb{R}^{N}$ fa da **chiave** $\mathbf{k}_t$, l'ingresso
$\mathbf{x}_t\in\mathbb{R}^{P}$ fa da **valore** $\mathbf{v}_t$, la matrice d'uscita
$\mathbf{C}_t\in\mathbb{R}^{N}$ fa da **query** $\mathbf{q}_t$, e lo scalare $a_t$ è il gate
$\alpha_t$. La ricorrenza dell'SSM,

$$
\mathbf{S}_t = a_t\, \mathbf{S}_{t-1} + \mathbf{x}_t\, \mathbf{B}_t^\top, \qquad \mathbf{y}_t = \mathbf{S}_t\, \mathbf{C}_t,
$$

è la stessa riga della tabella. Un avvertimento sulla scrittura, perché
altrimenti stona con il resto del capitolo: qui $\mathbf{B}_t$ è già la matrice
**discretizzata**, quella che altrove scriviamo $\bar{\mathbf{B}}_t$, e il passo
$\Delta_t$ sta dentro, non davanti. È la convenzione del paper SSD, che
dichiara in una nota di aver dato ai parametri discreti le lettere dei
continui per alleggerire la notazione; più avanti, quando ricomparirà
$\bar{\mathbf{B}}_t = \Delta_t \mathbf{B}_t$, saremo tornati alle lettere del
capitolo.

Srotolando la ricorrenza dallo stato iniziale nullo, l'uscita
al passo $i$ è

$$
\mathbf{y}_i = \sum_{j=1}^{i} \Big(\underbrace{\textstyle\prod_{k=j+1}^{i} a_k}_{\text{decadimento}}\Big)\,
      \big(\mathbf{C}_i^\top \mathbf{B}_j\big)\, \mathbf{x}_j ,
$$

dove il fattore $\prod_{k} a_k$ è quanto è sopravvissuto, dal passo $j$ al passo
$i$, di ciò che era stato scritto. Raccogliamo tutti i passi in una sola
matrice. Impilando le query $\mathbf{C}_i$, le chiavi $\mathbf{B}_j$ e i valori $\mathbf{x}_j$ nelle
righe di $\mathbf{C}$, $\mathbf{B}$, $\mathbf{X}$, l'intera sequenza di uscite si scrive

$$
\mathbf{Y} = \big(\mathbf{M} \odot \mathbf{C} \mathbf{B}^\top\big)\, \mathbf{X}, \qquad
M_{ij} = \begin{cases} \prod_{k=j+1}^{i} a_k & i \ge j \\ 0 & i < j \end{cases}
$$

dove $\mathbf{C} \mathbf{B}^\top$ è la matrice $L\times L$ (lunghezza per lunghezza) di tutte le
affinità query–chiave, esattamente $\mathbf{Q}\mathbf{K}^\top$ dell'attenzione; $\odot$ è il
prodotto elemento per elemento; e $\mathbf{M}$ è una **maschera causale con
decadimento**: azzera il futuro (triangolo superiore) e pesa il passato con i
prodotti degli scalari $a_t$. Il paper di Mamba-2 chiama $\mathbf{L}$ questa maschera;
qui la chiamiamo $\mathbf{M}$ perché in tutto il libro $L$ è la lunghezza della
sequenza, e le due cose comparirebbero nella stessa formula. Questa è, alla
lettera, un'attenzione mascherata: la stessa
$\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ dei Transformer {cite}`vaswani2017attention`, con
la softmax rimpiazzata dalla maschera $\mathbf{M}$. La matrice $\mathbf{M}$ ha una struttura
particolare, detta **1-semiseparabile**: ogni sua sottomatrice interamente
contenuta nel triangolo inferiore ha rango al più uno, perché ogni elemento si
fattorizza nei prodotti cumulati degli $a_t$. È questa struttura a fare da
ponte: i sistemi a spazio di stati con transizione scalare *sono* le attenzioni
con maschera semiseparabile.

`````

Il risultato ha una lettura da esplicitare, perché è il perno di due capitoli.
Nel capitolo sull'attenzione lineare avevamo messo in fila una piccola
collezione di architetture (uno «zoo», lo avevamo chiamato) che avevano tutte
lo stesso corpo (una memoria di taglia fissa, addestrata in parallelo e usata
passo dopo passo) e differivano in una cosa sola: **come il passato
sbiadisce** quando arriva il presente. C'era chi non dimentica niente, chi
sbiadisce tutta la memoria della stessa quantità, chi la sbiadisce casella per casella, e chi cancella di mira la vecchia voce che sta per essere
riscritta. Mamba-2 occupa il secondo gradino, quello che sbiadisce tutto in
blocco. Arrivando dai sistemi dinamici invece che dall'attenzione, ci
ritroviamo esattamente lì: è la prova che le due strade (quella partita
dall'attenzione e quella partita dai sistemi dinamici di Kálmán) portavano
alla stessa città. La stessa funzione ha una forma **ricorrente**, che costa
quanto la lunghezza del testo (la vista «SSM»), e una forma **a tabella**, la
grande griglia dei confronti fra tutte le coppie di parole, mascherata perché
ciascuna guardi solo all'indietro (la vista «attenzione»). Non è un'analogia:
è un'uguaglianza.

## Perché conviene: i tensor core

La dualità sarebbe solo un'eleganza teorica se non pagasse in velocità. Paga, e
la chiave è una rinuncia apparentemente minima. In Mamba-1 ogni casella della
memoria sbiadiva a velocità propria; Mamba-2 impone che sbiadiscano tutti alla
stessa. Sembra una perdita di espressività, cioè di cose che il modello sa
distinguere, ed è invece ciò che rende l'algoritmo esprimibile come pura
moltiplicazione di matrici.

`````{tab} Elementare

Immagina un'officina con un attrezzo formidabile ma specializzato: una pressa
che stampa lastre intere in un colpo solo, purché il pezzo abbia una certa
forma. Finché lavori a mano, pezzo per pezzo, la pressa resta ferma e tu vai
lentissimo. Se accetti di dare ai pezzi quella forma standard, puoi usarla, e
vai molto più veloce.

I tensor core della GPU sono quella pressa: sanno fare una cosa sola,
moltiplicare tabelle di numeri, e la fanno a velocità impressionante. Lo scan
di Mamba-1, fatto di operazioni una-alla-volta, li teneva spenti. La piccola
rinuncia di Mamba-2 dà ai conti la «forma standard» che la pressa accetta: il
modello lavora a corsie (ogni corsia ha il suo pezzo di memoria), e invece di
lasciare che ogni corsia dimentichi a modo suo, si chiede a un intero gruppo di
corsie di dimenticare tutte alla stessa velocità. Basta questo, e il calcolo di
tutto il gruppo diventa una sola moltiplicazione di tabelle invece di tante
operazioni minute: la pressa si accende. In più, potendo permettersi una
memoria più capiente senza pagarla in velocità, Mamba-2 allarga il foglio di
ogni corsia (da una manciata di numeri a diverse decine o centinaia) e
organizza le corsie in gruppi, che chiama **teste**, esattamente come
l'attenzione.

Un'avvertenza, la stessa del capitolo precedente: la grande tabella dei
confronti non si forma mai per intero, perché su un testo lungo sarebbe di
nuovo la tabella da cui eravamo scappati. Si lavora **a blocchi**: dentro un
blocco di poche centinaia di parole la tabella è piccola e si fa tutta insieme,
e da un blocco al successivo passa soltanto il riassunto. Tabella dentro il
blocco, riassunto da un blocco all'altro: è così che il lavoro resta
proporzionale alla lunghezza e la pressa lavora lo stesso.

`````

`````{tab} Superiore

La rinuncia, in formule: la matrice di stato $\mathbf{A}$, diagonale, non ha più
$N$ valori distinti per canale ma un solo scalare ripetuto, $\mathbf{A} = a\mathbf{I}$,
da cui una transizione discreta $\bar{\mathbf{A}}_t = a_t \mathbf{I}$.

Il motivo per cui il matmul batte lo scan è nell'hardware. Un tensor core esegue
un piccolo prodotto matrice–matrice per ciclo: su una GPU moderna è lì che
risiede la stragrande maggioranza dei FLOP disponibili. Un *selective scan* come
quello di Mamba-1 è invece una ricorrenza associativa fatta di moltiplicazioni
elemento per elemento e somme: parallelizzabile in $O(\log L)$ passi, ma su unità
generiche, molto meno dense di FLOP. Sta usando la frazione lenta della GPU.

Con la transizione $\bar{\mathbf{A}}_t = a_t \mathbf{I}$ l'algoritmo pratico non forma davvero
l'intera matrice lunghezza per lunghezza (sarebbe $O(L^2)$ in memoria). Si
adotta una **decomposizione a blocchi** (*chunked scan*): la sequenza si spezza
in blocchi di lunghezza $Q$ (la lettera è quella del paper, e non c'entra con
le query $\mathbf{Q}$, che essendo una matrice restano in grassetto); dentro
ciascun blocco si calcola la forma quadratica, attention-like, come un prodotto
di matrici sui tensor core; tra un
blocco e il successivo si passa solo lo **stato** riassuntivo, con un termine
di rango basso, in forma ricorrente. Si interpola così tra le due viste della
dualità: quadratica dentro il blocco, lineare tra i blocchi.

Sul costo conviene essere precisi, perché è qui che si annida il malinteso.
Il conto torna **lineare nella lunghezza**: $O\big(L\,Q\,(N+P) + L\,N\,P\big)$
con blocchi di lunghezza $Q$, stato $N$ e dimensione di testa $P$, che nel caso
del Teorema 6.1 del paper ($P = N$, blocchi dell'ordine di $N$) diventa
$O(L\,N^2)$. Rispetto alla forma quadratica, che di operazioni ne fa
$O(L^2 N)$, è un
guadagno enorme; rispetto alla ricorrenza pura non si risparmia nulla, anzi con
blocchi lunghi si fanno più operazioni. Il punto non è farne meno, è **farne
di un tipo diverso**: tutte moltiplicazioni di matrici, cioè lavoro che la
pressa accetta. Ne seguono due conseguenze di progetto: la struttura è
**multi-head** come l'attenzione (dimensione di testa $P$ tipicamente $64$ o
$128$), e lo stato può crescere di un ordine di grandezza (da $N=16$ in
Mamba-1 a $N$ dell'ordine di $64$–$256$ e oltre in Mamba-2), perché una memoria
più grande, ora, non costa in velocità. Uno stato più capiente è direttamente
più memoria associativa: meno *crosstalk*, richiamo più preciso.

`````

Vale la pena fermarsi sul senso di questa mossa. Mamba-1 aveva reso l'SSM
selettivo pagando con lo scan. Mamba-2 recupera il parallelismo pieno delle
matrici accettando una transizione di stato più semplice, e può permetterselo
proprio perché la dualità gli garantisce che quella forma più semplice è ancora
un'attenzione. Non un modello impoverito, quindi: lo stesso oggetto, scritto in
modo che l'hardware lo digerisca. È il compromesso tipico di questa famiglia:
qualche grado di libertà in meno sulla transizione, in cambio di forme
parallele che sfruttano le GPU.

## Mamba-3

L'ultimo anello di questa catena è **Mamba-3**, di Lahoti, Li e colleghi con
Dao e Gu, che a ICLR 2026 è finito fra i pochi lavori esposti dal palco invece
che a un poster, in gergo un *Oral* {cite}`lahoti2026mamba3`. Trattandosi di un
lavoro recente conviene leggerlo per ciò che aggiunge di qualitativo più che
per le cifre puntuali, ancora da assestare. Le novità rispetto a Mamba-2 sono
tre, e tutte lavorano sul *come* lo stato evolve, non sulla struttura generale.

La prima riguarda la **discretizzazione**, cioè il modo di trasformare il sistema
continuo in una ricorrenza, che avevamo introdotto all'inizio del capitolo.

`````{tab} Elementare

Ricordiamo il problema: un sistema che scorre nel tempo va «campionato» a
intervalli, e bisogna indovinare cosa succede *tra* un campione e l'altro. Il
punto delicato è quanta parte di ciò che entra in quel tratto finisce nella
memoria. Torniamo al rubinetto: più a lungo lo tieni aperto e più forte lo
apri, più acqua entra, e la quantità è la superficie della figura che ha per
base la durata del tratto e per altezza l'apertura. Mamba-1 usa una sola
altezza, l'apertura del campione che sta leggendo: quella figura è un
**rettangolo**, ed è il conto sbrigativo, il valore di adesso moltiplicato per
la durata, come se fosse stato quello per tutto il tratto. Rapido, ma con un
errore che a ogni passo si accumula.

Mamba-3 rifà lo stesso conto a **trapezi**, cioè guardando tutte e due le
aperture, quella di adesso e quella del campione precedente: si tira un
segmento fra i due valori e si misura la superficie che gli sta sotto. E c'è una
furbizia in più, tipica di questo capitolo: quanto contano i due estremi non è
deciso una
volta per tutte a metà e metà, lo decide il modello a ogni passo, in base a ciò
che legge (il trapezio della geometria, quello che fa la media, è il caso
particolare in cui i due estremi pesano uguale). Le due cose, però, tirano da
parti opposte: il conto è davvero più preciso solo se i due estremi pesano
quasi uguale, e lasciato libero il modello preferisce sbilanciarli. Gli autori
hanno provato a obbligarlo a stare vicino alla metà, e i risultati sono
peggiorati: quello che si guadagna non è tanto un errore più piccolo, è una
regola più ricca, che il modello dosa come gli conviene. La conseguenza pratica
è curiosa: Mamba-1 e Mamba-2 avevano
bisogno, prima del cuore selettivo, di una **piccola convoluzione causale** (un
mini-filtro che mescola qualche parola vicina) per funzionare bene. Con il
conto più fine, e con un ritocco in più (un numero fisso aggiunto ai due pezzi
che scrivono nella memoria e la rileggono), quel filtro diventa **opzionale**:
il modello lavora bene anche senza. Una regola migliore per fare i conti, e una
stampella in meno.

`````

`````{tab} Superiore

Mamba e Mamba-2 discretizzano la transizione con lo **zero-order hold**, che
per la parte di stato è esatto ($\bar{\mathbf{A}}_t = \exp(\Delta_t \mathbf{A})$, come visto a
inizio capitolo); il termine d'ingresso, però, viene semplificato al
prim'ordine (Eulero): $\bar{\mathbf{B}}_t = \Delta_t \mathbf{B}_t$, con un errore locale
dell'ordine di $O(\Delta_t^2)$ sul passo. È su questo pezzo che interviene
Mamba-3, con una discretizzazione **esponenziale-trapezoidale**:
un'integrazione del second'ordine che stima il contributo dell'ingresso con una
**combinazione convessa** dei valori agli estremi dell'intervallo,

$$
\mathbf{h}_t = e^{\Delta_t \mathbf{A}_t} \mathbf{h}_{t-1}
    + (1-\lambda_t)\,\Delta_t\, e^{\Delta_t \mathbf{A}_t} \mathbf{B}_{t-1}x_{t-1}
    + \lambda_t\,\Delta_t\, \mathbf{B}_t x_t ,
$$

dove il peso $\lambda_t \in [0,1]$ è uno scalare **deciso dai dati**, token per
token, esattamente come $\Delta_t$ (e $\mathbf{A}_t$ è la transizione al passo $t$, che
il paper indicizza per generalità: in Mamba-2 era $a\mathbf{I}$ con $a$ fisso, e la
dipendenza dal token passava tutta per $\Delta_t$). La regola classica del
trapezio (la media dei due estremi) è il caso $\lambda_t = 1/2$ e la regola di
Eulero di Mamba-2 è il caso $\lambda_t = 1$: sono due casi particolari di una
famiglia, non l'alternativa secca fra due metodi. L'errore locale scende a
$O(\Delta_t^3)$ a condizione che
$\lambda_t$ resti vicino a $1/2$ (precisamente $\lambda_t = 1/2 + O(\Delta_t)$),
e il paper riporta che imporre quella condizione **peggiora** i risultati
empirici: il modello preferisce dosare il peso a modo suo, e quello che si
guadagna in accuratezza formale si perde in qualità. Non è la trasformazione
bilineare di S4, che approssima l'esponenziale di $\mathbf{A}$: qui il trapezio agisce
sul termine d'ingresso data-dipendente, mentre la transizione resta
esponenziale. La conseguenza riportata nel paper è che la **short causal
convolution** posta prima dell'SSM (presente in tutti i blocchi Mamba
precedenti come stabilizzatore) diventa **opzionale**: insieme a un termine di
**bias esplicito** su $\mathbf{B}$ e $\mathbf{C}$, la discretizzazione più fine recupera
l'effetto di mescolamento locale che quel filtro forniva. Vanno insieme, i due
ingredienti: togliere la convoluzione corta senza aggiungere i bias non è ciò
che il paper ha misurato.

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
tenere il conto di qualcosa che si ripete a cicli (come le ore su un
quadrante, dove dopo il dodici si ricomincia), seguire uno stato che si
alterna. Una memoria che sa solo sbiadire fatica; una che sa ruotare può,
letteralmente, «girare la lancetta» a ogni passo e ricordare a che punto del
ciclo si trova. Sulle parentesi si vede bene: basta che a ogni parentesi la
lancetta faccia mezzo giro, e dopo un numero pari di parentesi è tornata
esattamente al punto di partenza, dopo un numero dispari è dalla parte opposta
del quadrante. Le due situazioni si distinguono a colpo d'occhio, mentre una
memoria che può solo affievolirsi non ha modo di tenerle separate. Gli
ingegneri lo chiamano *state tracking*, tenere traccia dello stato, ed è
storicamente un punto debole delle ricorrenze lineari.

E qui torna il filo con i Transformer, che è la ragione per cui questa è la più
concettuale delle tre novità. Dentro un modello ogni parola è una fila di numeri, che si
può guardare come una freccia; e anche i Transformer, per dire a che punto
della frase sta una parola, fanno ruotare la sua freccia di un angolo che
cresce con la posizione.
La differenza è che lì la rotazione viene aggiunta apposta da fuori, mentre qui
nasce da sola dal modo in cui la memoria evolve; e l'angolo, invece di
dipendere solo da quanto si è andati avanti, dipende da ciò che si sta
leggendo.

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
alle matrici $\mathbf{B}$ e $\mathbf{C}$. RoPE (la *Rotary Position Embedding* che avevamo solo
nominato nel capitolo sui Transformer, tra le codifiche posizionali relative
più recenti) inietta la posizione ruotando query e key di un angolo
proporzionale all'indice del token. Qui accade lo stesso, con due differenze:
le rotazioni si applicano alle controparti SSM di key e query ($\mathbf{B}$ e $\mathbf{C}$), e
l'angolo non dipende solo dalla posizione ma dai **dati**, perché il passo
$\Delta$ è selettivo. È l'ennesimo ponte tra le due famiglie: la codifica
posizionale rotazionale dei Transformer riemerge, spontaneamente, come la fase
di una dinamica di stato complessa.

`````

La terza novità è più ingegneristica.

`````{tab} Elementare

Finora ogni corsia del modello teneva un foglio tutto suo, e i fogli non
si parlavano. Mamba-3 fa condividere a più corsie un
foglio comune, più capiente. Il vantaggio sta nel modo di lavorare delle
schede grafiche: andare a prendere i dati in memoria costa più che farci i
conti sopra, quindi conviene, a ogni viaggio, portare a casa più lavoro utile.
Con il foglio condiviso ogni lettura serve più corsie in un colpo solo, e il
risultato pratico è una qualità un po’ migliore **senza** rallentare la
generazione: l'attesa tra una parola prodotta e la successiva resta la stessa.

`````

`````{tab} Superiore

Mamba-2 e i suoi predecessori sono, nella loro forma base, sistemi a **singolo
ingresso e singola uscita** (SISO): ogni canale evolve con un proprio stato,
indipendente, e $\mathbf{B}_t$ e $\mathbf{C}_t$ sono vettori. Mamba-3 propone una formulazione
**MIMO** (*multi-input multi-output*, come per S5), in cui più ingressi e più
uscite condividono lo stesso stato attraverso matrici $\mathbf{B}$ e $\mathbf{C}$ non più
vettoriali ma di rango maggiore. L'effetto tecnico è aumentare l’**intensità
aritmetica** (il numero di operazioni per ogni byte letto dalla memoria) che è
proprio ciò che tiene occupati i tensor core: si fa più lavoro utile per ogni
accesso in memoria. Il guadagno pratico riportato è qualità migliore **senza**
aumentare la latenza di decodifica, cioè senza rallentare la generazione token
per token.

`````

Trattandosi di un lavoro recente ci fermiamo alle novità qualitative: la
direzione è chiara (stato più accurato, più espressivo e meglio calibrato
sull'hardware), mentre i numeri esatti andranno confermati man mano che il
modello viene ripreso e riprodotto.

## Da S4 a Mamba-3: cosa è cambiato

Conviene, a questo punto, riavvolgere l'intero arco, perché ogni tappa ha
smontato un pezzo diverso del problema e la somma racconta una storia pulita.

Si parte da **S4**: una macchina che tratta ogni parola con la **stessa
regola**, con i numeri di partenza scelti bene (è la ricetta di HiPPO) perché
la memoria sia lunga, e con una forma regolare che rende i conti veloci. Il suo
pregio è anche il suo limite: siccome la regola non cambia mai, lo stesso
calcolo si può fare in due modi (passo dopo passo oppure tutto insieme), ma la
macchina non può *scegliere* cosa ricordare in base a quel che legge.

Arriva **Mamba-1**, che rompe la regola fissa: quanto scrivere e quanto
dimenticare lo decide la parola in arrivo. Il sistema diventa **selettivo** e sa
separare il rilevante dal riempimento, ma perde il filtro unico e deve
affidarsi allo scan: veloce, e però lontano dal reparto della scheda grafica
dove sta quasi tutta la potenza.

Poi **Mamba-2**, che riconcilia le due famiglie del libro. Nella sua versione
più semplice (tutte le corsie di un gruppo dimenticano alla stessa velocità)
questa macchina *è* un'attenzione che guarda solo all'indietro, e scritta così
il suo calcolo diventa una moltiplicazione di tabelle: la pressa si accende,
la memoria può crescere, e le corsie si organizzano in gruppi come le teste
dell'attenzione. È il punto in cui le due strade di questi due capitoli
(attenzione lineare e sistemi dinamici) si rivelano una sola.

Infine **Mamba-3**, che non cambia l'impianto ma ne raffina il funzionamento
interno: i conti sull'intervallo rifatti a trapezi invece che a rettangoli (e
il mini-filtro che stava prima del cuore selettivo diventa opzionale), una
memoria che oltre a sbiadire sa ruotare come una lancetta su un quadrante
(utile per contare e tenere il segno, ed è la stessa idea con cui i Transformer
codificano la posizione), e un foglio condiviso fra più corsie, che spreme
meglio l'hardware. Da una macchina che tratta tutti allo stesso modo e ricorda
a lungo, a una che sceglie, poi riconciliata con l'attenzione e resa veloce,
poi affinata nel modo in cui la memoria evolve: è la parabola di una singola,
ostinata idea (comprimere il passato in un riassunto che non cresce mai) che a
ogni passo si avvicina un po’ di più al meglio dei Transformer senza rinunciare
al costo lineare.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- **Mamba-2** nasce da un problema pratico: dentro la scheda grafica c'è una
  pressa specializzata che sa fare una cosa sola, moltiplicare matrici, ed è lì
  che sta quasi tutta la potenza disponibile. Il calcolo passo dopo passo di
  Mamba-1, fatto di operazioni minute, la lasciava spenta.
- La **dualità stato-attenzione** (Dao e Gu, 2024) è il ponte esplicito con il
  capitolo sull'attenzione: appena si sceglie la versione più semplice dello
  stato (tutte le corsie di una testa sbiadiscono allo stesso ritmo), un SSM
  **è** un'attenzione che guarda solo all'indietro, dove ogni confronto fra due
  parole è pesato da quanto è sopravvissuto nel frattempo. Due dialetti della
  stessa lingua: lo stesso conto si fa passo dopo passo, oppure formando la
  grande tabella dei confronti.
- È lo stesso gradino che nello «zoo» delle ricorrenze del capitolo precedente
  portava già il nome di Mamba-2: le due strade, dall'attenzione e dai sistemi
  dinamici, arrivano allo stesso posto.
- Quella piccola rinuncia (le corsie di uno stesso gruppo dimenticano tutte
  alla stessa velocità) dà ai conti la forma che la pressa accetta: tutto
  diventa moltiplicazione di tabelle. Non si fanno **meno** operazioni, se ne
  fanno di un tipo che la macchina digerisce meglio. In più la memoria si
  organizza a gruppi (le **teste**, come nell'attenzione) e il foglio di ogni
  corsia può diventare molto più capiente.
- **Mamba-3** (Lahoti et al., 2026) non cambia l'impianto, ne raffina la
  dinamica con tre mosse: i conti sull'intervallo rifatti a **trapezi** invece
  che a rettangoli, con il peso dei due estremi deciso volta per volta dal
  modello (e il mini-filtro che stava prima del cuore selettivo diventa
  **opzionale**, purché si aggiunga il numero fisso che l'accompagna); uno
  stato che oltre a sbiadire sa **ruotare**, come una lancetta su un quadrante,
  utile per contare e tenere il segno; e un foglio di memoria **condiviso**
  fra più corsie, che dà più qualità senza rallentare la generazione. Lavoro
  recente: la direzione è solida, le cifre da confermare.
- L'arco **S4 → Mamba → Mamba-2 → Mamba-3**: da un sistema che tratta ogni
  token con la stessa regola e ricorda a lungo, a uno che sceglie cosa
  ricordare, a uno riconciliato con l'attenzione e veloce, a uno raffinato nel
  modo in cui la memoria evolve (sempre la stessa idea: comprimere il passato in
  un riassunto che non cresce mai).
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- **Mamba-2** nasce da un problema pratico: lo scan di Mamba-1 non usa i
  **tensor core** della GPU, l'hardware dedicato a moltiplicare matrici, e
  lascia gran parte della potenza inutilizzata.
- La **State Space Duality** (Dao e Gu, ICML 2024) è il ponte esplicito con il
  capitolo sull'attenzione: un SSM con $\mathbf{A} = a\mathbf{I}$, cioè con
  transizione discreta $\bar{\mathbf{A}}_t = a_t \mathbf{I}$ (uno scalare per
  l'identità), è **esattamente** un'attenzione mascherata,
  $\mathbf{Y} = (\mathbf{M} \odot \mathbf{C}\mathbf{B}^\top)\mathbf{X}$ con maschera causale $\mathbf{M}$ $1$-semiseparabile (il
  paper la chiama $\mathbf{L}$; qui $L$ è la lunghezza). La stessa funzione ha una forma
  lineare/ricorrente $O(L)$ e una quadratica/attention-like.
- È lo stesso gradino (il decadimento scalare $\alpha_t \mathbf{I}$) che
  occupava la riga «Mamba-2» nello «zoo» delle ricorrenze lineari: le due
  strade, dall'attenzione e dai sistemi dinamici, arrivano allo stesso posto.
- La restrizione $\mathbf{A} = a\mathbf{I}$ (diagonale tutta uguale, mentre
  Mamba-1 aveva valori distinti) rende il calcolo pura **moltiplicazione di
  matrici**, con costo $O(L\,N\,(Q+P))$ a blocchi di lunghezza $Q$: non meno
  operazioni della ricorrenza pura, ma operazioni che stanno sui tensor core.
  Ne seguono la struttura **multi-head** e uno stato molto più grande (da
  $N=16$ a $64$–$256$ e oltre).
- **Mamba-3** (Lahoti et al., ICLR 2026, Oral) raffina la dinamica con tre mosse:
  discretizzazione **esponenziale-trapezoidale** (combinazione convessa degli
  estremi con peso $\lambda_t$ data-dipendente; il trapezio classico è
  $\lambda_t=1/2$, Eulero è $\lambda_t=1$), che insieme a un bias esplicito su
  $\mathbf{B}$ e $\mathbf{C}$ rende opzionale la convoluzione causale corta; stato **complesso**
  con aggiornamenti **rotazionali** (migliore *state tracking*, con un legame
  formale al **RoPE** data-dipendente su $\mathbf{B}$ e $\mathbf{C}$); e formulazione **MIMO**
  (più qualità senza aumentare la latenza di decodifica). Lavoro recente:
  novità qualitative solide, cifre da confermare.
- L'arco **S4 → Mamba → Mamba-2 → Mamba-3**: da tempo-invariante a lungo
  raggio, a selettivo, a riconciliato con l'attenzione e veloce, a raffinato
  nella dinamica (sempre la stessa idea di comprimere il passato in uno stato
  di dimensione fissa).
```

`````
