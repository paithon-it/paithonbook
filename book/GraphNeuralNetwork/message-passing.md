# Message passing: il cuore delle GNN

C'è un gioco che tutti conosciamo: il *telefono senza fili*. Una persona
sussurra una frase all'orecchio del vicino, quello la ripete al suo, e così
via. Cambiamo le regole. Invece di ascoltare *un* vicino, ognuno ascolta
*tutti* i propri vicini in una volta sola, riassume ciò che ha sentito e si fa
un'idea aggiornata; poi si ricomincia. Dopo un giro, ogni persona sa qualcosa
dei suoi amici diretti. Dopo due giri, anche degli amici degli amici. Dopo
$K$ giri, la voce partita da un capo della rete è arrivata a chi sta a $K$
strette di mano di distanza.

Questo passaparola a giri è, alla lettera, il modo in cui una rete neurale su
grafo elabora l'informazione. La sezione «Il mondo come grafo» ha messo il dato
in forma di tabelle, e per seguire questa sezione basta ricordare che cosa
dicono: **chi è collegato a chi** e **che cosa c'è scritto su ogni nodo**. Se i
nomi propri sono scivolati via non fa niente; eccoli comunque, da lasciar
passare senza fermarsi: la matrice di adiacenza $\mathbf{A}$ (chi è collegato a
chi), la matrice delle feature dei nodi $\mathbf{X}$ (le file di numeri dei
nodi), la matrice diagonale dei gradi $\mathbf{D}$ (quanti vicini ha ciascuno) e
la versione con i cappi $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ (la
prima, con ogni nodo dichiarato vicino di sé stesso).

L'introduzione al capitolo ha dato un nome al meccanismo: **message passing**,
«scambio di messaggi». Qui lo apriamo: prima nella sua forma generale, poi nella
sua incarnazione più usata, la *Graph Convolutional Network*. Di quella
vedremo, in quest'ordine, la formula, i conti che ne escono su un grafo
minuscolo, la ragione per cui è fatta così e infine da dove salta fuori.

## Un nodo, i suoi vicini, tre mosse

L'idea si regge su un'operazione sola, ripetuta a ogni giro e uguale per ogni
nodo: **guarda i vicini, riassumili, aggiornati**. La
{numref}`fig-message-passing` la mostra tutta in un colpo d'occhio.

```{figure} ../figures/message-passing.svg
:name: fig-message-passing
:alt: Un nodo centrale riceve messaggi dai vicini; i messaggi confluiscono in un blocco «aggrega» e poi in un blocco «aggiorna» che, insieme allo stato precedente del nodo, produce il nuovo stato. In basso, due pannelli mostrano il campo recettivo che passa da un salto (uno strato) a due salti (due strati).
:width: 100%

Un passo di message passing su un nodo $v$: i **messaggi** dei vicini (le loro
file di numeri) si **aggregano** in un unico riassunto che non dipende
dall'ordine in cui arrivano, poi la mossa di **aggiornamento** fonde quel
riassunto con quello che il nodo sapeva già di sé. In basso: impilando due
strati, il pezzo di grafo che un nodo riesce a sentire (il suo **campo
recettivo**) cresce dai vicini diretti ai vicini dei vicini.
```

`````{tab} Elementare

Ogni nodo è una persona con una scheda su cui scrive «chi sono».
Sulla scheda non c'è una frase: c'è una **fila di numeri**, uno per ogni
caratteristica, come le tre caselline accanto a ogni nodo nella figura della
sezione «Il mondo come grafo» (14 anni, 2 sport, 300 messaggi al giorno). Vale
la pena fissarlo adesso, perché è quello che viaggia lungo gli archi per tutto
il capitolo: il messaggio non è un testo, è una fila di numeri.

A ogni giro un nodo fa tre cose, sempre nell'ordine. **Primo**, ascolta: ogni
amico gli passa un bigliettino con sopra la propria fila di numeri (sono i
*messaggi*). **Secondo**, mette insieme i bigliettini in un unico riassunto, e
siccome sono file di numeri «mettere insieme» vuol dire fare i conti casella
per casella: gli anni con gli anni, gli sport con gli sport. E qui c'è un
dettaglio importante: il riassunto non deve dipendere dall'ordine in cui
arrivano i bigliettini, perché tra amici non c'è un «primo» e un «ultimo». Un
riassunto che va bene è la **somma**, o la **media**: cambi l'ordine degli
addendi e il totale non cambia. **Terzo**, aggiorna la propria scheda mettendo
insieme il riassunto degli amici e quello che già sapeva di sé.

Fatto questo per tutti i nodi, il giro è finito e se ne può fare un altro. È
lo stesso identico meccanismo per ogni persona della rete: nessuno ha una
regola speciale. Proprio come nella convoluzione delle immagini, dove lo
stesso piccolo filtro scorre su tutti i pixel: solo che qui i «vicini» non
sono i quattro pixel accanto, ma gli amici sul grafo, che possono essere due o
dieci.

`````

`````{tab} Superiore

Il quadro generale è la *Message Passing Neural Network* (MPNN) di Gilmer e
colleghi {cite}`gilmer2017neural`, che unifica sotto un'unica notazione quasi
tutte le GNN. Sia $\mathbf{h}_v^{(k)}$ il vettore di stato del nodo $v$ dopo
$k$ giri, con $\mathbf{h}_v^{(0)} = \mathbf{x}_v$ (la sua feature iniziale). Un
passo si scrive in due mosse:

$$
\mathbf{m}_v^{(k)} = \bigoplus_{u \in \mathcal{N}(v)}
   M_k\!\big(\mathbf{h}_v^{(k-1)},\, \mathbf{h}_u^{(k-1)},\, \mathbf{e}_{vu}\big),
\qquad
\mathbf{h}_v^{(k)} = U_k\!\big(\mathbf{h}_v^{(k-1)},\, \mathbf{m}_v^{(k)}\big).
$$

Qui $\mathcal{N}(v)$ è l'insieme dei vicini di $v$; $M_k$ è la **funzione
messaggio** (una rete, che può usare anche la feature dell'arco
$\mathbf{e}_{vu}$); il simbolo $\bigoplus$ è l’**aggregazione**, un'operazione
*invariante alla permutazione* dei vicini (tipicamente $\sum$, la media o il
massimo) che produce il messaggio aggregato $\mathbf{m}_v^{(k)}$; e $U_k$ è la
**funzione di aggiornamento** che fonde lo stato precedente con
$\mathbf{m}_v^{(k)}$.

Vale la pena dire di che oggetti si parla, perché è il punto su cui la formula
si legge o non si legge: sono tutti **vettori**. Lo stato è
$\mathbf{h}_v^{(k)} \in \mathbb{R}^{d_k}$, il messaggio è
$\mathbf{m}_v^{(k)} \in \mathbb{R}^{d_m}$, e quindi
$M_k \colon \mathbb{R}^{d_{k-1}} \times \mathbb{R}^{d_{k-1}} \times
\mathbb{R}^{d_e} \to \mathbb{R}^{d_m}$ e
$U_k \colon \mathbb{R}^{d_{k-1}} \times \mathbb{R}^{d_m} \to \mathbb{R}^{d_k}$.
Il $\bigoplus$ opera **componente per componente** su un numero variabile di
vettori tutti della stessa lunghezza $d_m$ e ne restituisce uno solo, sempre di
lunghezza $d_m$: è per questo che l'ordine dei vicini non conta e che il grado
variabile non rompe le dimensioni. (Attenzione al simbolo: qui $\bigoplus$ è
l'aggregazione, mentre nel resto del libro $\oplus$ indica la concatenazione,
che in questo capitolo si scrive $\|$.) Dopo $K$ passi,
per un compito sull'intero grafo si applica una funzione di lettura
($\mathrm{READOUT}$), anch'essa invariante alla permutazione,
$\hat{y}_G = R\big(\{\, \mathbf{h}_v^{(K)} : v \in V \,\}\big)$.

L'invarianza di $\bigoplus$ è ciò che garantisce l’**equivarianza alla
permutazione** anticipata nell'introduzione: rinumerare i nodi non cambia i
messaggi, perché una somma non ha un primo addendo. Ed è la stessa forma
astratta («aggrega dai vicini, poi aggiorna») dello schema
$\mathrm{AGGREGATE}$/$\mathrm{UPDATE}$ visto nell'overview del capitolo, qui
resa esplicita nelle sue tre componenti apprendibili.

`````

## Dalla formula alla matrice: la GCN

Lo schema delle tre mosse è un telaio, non un modello: per avere qualcosa che
gira bisogna decidere *come* si scrive il bigliettino, *come* si riassumono e
*come* si riscrive la scheda. La scelta più celebre (semplice, veloce, e ancora
oggi il primo modello che si prova su un grafo) è la
**Graph Convolutional Network** (GCN),
presentata nel 2017 da Thomas Kipf e Max Welling {cite}`kipf2017semi`.

Una parola sul vocabolario, perché da qui in avanti le due si alternano: ogni
**giro** di passaparola è uno **strato** della rete. Sono la stessa cosa detta
dai due lati, dal lato di chi ascolta e dal lato di chi la rete la costruisce.

La regola di propagazione della GCN, da uno strato al successivo, sta in una
riga:

$$
\mathbf{H}^{(l+1)} = \sigma\!\left( \hat{\mathbf{A}}\, \mathbf{H}^{(l)}\, \mathbf{W}^{(l)} \right),
\qquad
\hat{\mathbf{A}} = \tilde{\mathbf{D}}^{-1/2}\, \tilde{\mathbf{A}}\, \tilde{\mathbf{D}}^{-1/2}.
$$

Ogni simbolo ha un ruolo preciso, e la riga si può leggere a due profondità.

`````{tab} Elementare

È la scena dei bigliettini scritta in forma abbreviata, tutta in una riga e per
tutti i nodi insieme.

$\mathbf{H}^{(l)}$ è la pila delle schede al giro $l$: una riga per nodo e, al
giro zero, quello che ogni nodo sa di sé.

La $\mathbf{A}$ col cappello è la rubrica di chi è collegato a chi, ritoccata
in due punti: ogni nodo vi figura anche come vicino di sé stesso (sono i cappi
della sezione «Il mondo come grafo»: chi ascolta gli altri non deve
dimenticare la propria scheda) e ogni collegamento porta un peso, calcolato in
modo che chi ha tanti
vicini non copra la voce degli altri. Moltiplicare la pila delle schede per
questa rubrica è il giro di raccolta dei bigliettini.

$\mathbf{W}^{(l)}$ è la ricetta con cui ogni nodo riscrive la propria scheda
dopo la raccolta, la stessa per tutti, come il filtro che scorre identico su
tutta l'immagine in una rete convoluzionale; ed è qui che stanno i numeri che
la rete impara. Infine $\sigma$ è il solito ritocco finale, una funzione come
la ReLU che ci accompagna fin dalle prime reti neurali. Un giro intero di
passaparola, per l'intera rete, in una riga.

Resta la coda della formula, quella con l'ondina e gli esponenti: è soltanto il
modo compatto di scrivere «i pesi da mettere sui collegamenti», e dice quel che
si è appena detto a parole. Non c'è niente da leggerci dentro. Adesso quei pesi
li vediamo all'opera su un grafo di quattro nodi, e subito dopo si dirà perché
sono fatti così.

`````

`````{tab} Superiore

- $\mathbf{H}^{(l)} \in \mathbb{R}^{N \times d_l}$ raccoglie, riga per riga,
  gli stati di tutti i nodi allo strato $l$; si parte da
  $\mathbf{H}^{(0)} = \mathbf{X}$, le feature d'ingresso.
- $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ è l'adiacenza con i **cappi**
  (*self-loop*): aggiungere la matrice identità $\mathbf{I}$ mette ogni nodo
  tra i propri vicini, così che nell'aggregazione un nodo tenga conto anche di
  sé stesso e non dimentichi la propria feature.
- $\tilde{\mathbf{D}}$ è la matrice diagonale dei gradi di
  $\tilde{\mathbf{A}}$, cioè $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$ (il
  numero di vicini del nodo $i$, più uno per il cappio).
- $\hat{\mathbf{A}} = \tilde{\mathbf{D}}^{-1/2}\,\tilde{\mathbf{A}}\,\tilde{\mathbf{D}}^{-1/2}$
  è l'adiacenza **normalizzata in modo simmetrico**: il pezzo che pesa i
  messaggi.
- $\mathbf{W}^{(l)} \in \mathbb{R}^{d_l \times d_{l+1}}$ è la matrice dei pesi
  appresi dello strato (la stessa per tutti i nodi, come il filtro di una CNN)
  e $\sigma$ una non-linearità (di solito la ReLU). È lei a decidere la
  larghezza del passo successivo: $\mathbf{H}^{(l)}\mathbf{W}^{(l)}$ manda
  $N \times d_l$ in $N \times d_{l+1}$.

Letta nodo per nodo, la riga matriciale dice esattamente «aggrega, poi
aggiorna»:

$$
\mathbf{h}_v^{(l+1)} = \sigma\!\left(
   \sum_{u \in \mathcal{N}(v)\cup\{v\}}
   \frac{1}{\sqrt{\tilde{d}_v\,\tilde{d}_u}}\; \mathbf{W}^{(l)\top} \mathbf{h}_u^{(l)}
\right),
$$

dove $\tilde{d}_v$ è il grado di $v$ in $\tilde{\mathbf{A}}$. Il messaggio del
vicino $u$ è la sua feature trasformata da $\mathbf{W}^{(l)}$; l'aggregazione è
una **somma pesata**, con pesi fissi $1/\sqrt{\tilde{d}_v\,\tilde{d}_u}$;
l'aggiornamento è la non-linearità $\sigma$. È un caso particolare della MPNN
in cui la funzione messaggio è lineare e l'aggregazione è la somma
normalizzata.

`````

### Il conto, coi numeri

Vale più di mille formule vedere i conti tornare, e conviene farlo subito, su
un grafo piccolissimo: quattro nodi in fila
($1 - 2 - 3 - 4$), ciascuno con **un solo numero** sulla scheda invece di una
fila, cioè $\mathbf{X} = (1,\, 2,\, 3,\, 4)^\top$. (La $\top$ in alto vuol dire
solo che quei quattro numeri vanno letti in colonna, uno per nodo, invece che
in riga: è una convenzione di scrittura e non cambia niente.)

Delle tre mosse ne teniamo una sola. La ricetta di riscrittura e il ritocco
finale li mettiamo a riposo (in formule, $\mathbf{W} = \mathbf{I}$ e
$\sigma$ uguale all'identità: due modi di dire «per stavolta, lascia le cose
come stanno»), così quello che si vede è l'effetto della sola raccolta dei
bigliettini.

Prima di leggere le tabelle conviene sapere che cosa si sta per vedere, perché
in due righe si dice tutto. Ogni collegamento porta un **peso**, e il peso è
tanto più piccolo quanti più vicini hanno i due nodi che collega, contando
anche il cappio che ciascuno ha verso sé stesso. Qui i pesi sono tre: $0{,}500$
sui due cappi dei nodi di bordo, che di vicini ne hanno uno solo; $0{,}408$ sui
due archi che uniscono un nodo di bordo a uno interno; $0{,}333$ sull'arco fra
i due nodi interni e sui loro cappi. Il nuovo valore di un nodo è la somma dei
valori dei vicini (e del proprio), ciascuno moltiplicato per il peso del
collegamento: da lì in poi è una moltiplicazione e un'addizione. Le tabelle qui
sotto sono il conto esatto di quei pesi; chi non ha voglia di rifarlo può
saltare alla riga dei quattro risultati e non perde niente, perché la morale
sta lì.

La matrice di adiacenza e quella con i cappi
($\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$) sono

$$
\mathbf{A} = \begin{bmatrix}
0 & 1 & 0 & 0 \\
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1 \\
0 & 0 & 1 & 0
\end{bmatrix},
\qquad
\tilde{\mathbf{A}} = \begin{bmatrix}
1 & 1 & 0 & 0 \\
1 & 1 & 1 & 0 \\
0 & 1 & 1 & 1 \\
0 & 0 & 1 & 1
\end{bmatrix}.
$$

Sommando le righe di $\tilde{\mathbf{A}}$ si contano i vicini di ciascun nodo,
cappio compreso: sono i gradi $\tilde{\mathbf{d}} = (2,\, 3,\, 3,\, 2)$, perché
i due nodi di bordo hanno un vicino e i due interni ne hanno due, più in tutti
e quattro i casi sé stessi. Dunque

$$
\tilde{\mathbf{D}} = \mathrm{diag}(2,3,3,2),
\qquad
\tilde{\mathbf{D}}^{-1/2} = \mathrm{diag}\!\left(
\tfrac{1}{\sqrt{2}},\, \tfrac{1}{\sqrt{3}},\,
\tfrac{1}{\sqrt{3}},\, \tfrac{1}{\sqrt{2}} \right)
\approx \mathrm{diag}(0{,}707,\ 0{,}577,\ 0{,}577,\ 0{,}707),
$$

dove l'esponente $-1/2$ vuol dire soltanto «uno diviso la radice quadrata»:
$2^{-1/2} = 1/\sqrt{2} \approx 0{,}707$. Quei quattro numeri, uno per nodo,
sono la porzione di peso che ciascuno mette in ogni suo collegamento, e il peso
dell'arco è il prodotto delle porzioni delle due estremità: $0{,}707 \cdot
0{,}707 = 0{,}5$ fra due nodi di bordo, $0{,}707 \cdot 0{,}577 \approx 0{,}408$
fra un bordo e un interno, $0{,}577 \cdot 0{,}577 \approx 0{,}333$ fra due
interni. La tabella dei pesi così ottenuta si chiama **adiacenza
normalizzata**, e si scrive $\hat{\mathbf{A}}$, la $\mathbf{A}$ col cappello:
«normalizzare» vuol dire appunto questo, dividere per rimettere tutti sulla
stessa scala, e il perché lo si vede appena finito il conto. In una riga sola:
$\hat{A}_{vu} = \tilde{A}_{vu} / \sqrt{\tilde{d}_v\,\tilde{d}_u}$, cioè
$\hat{A}_{12} = 1/\sqrt{2\cdot 3} = 1/\sqrt{6} \approx 0{,}408$ e
$\hat{A}_{22} = 1/\sqrt{3\cdot 3} = 1/3 \approx 0{,}333$. La matrice completa è

$$
\hat{\mathbf{A}} = \tilde{\mathbf{D}}^{-1/2}\,\tilde{\mathbf{A}}\,\tilde{\mathbf{D}}^{-1/2} \approx
\begin{bmatrix}
0{,}500 & 0{,}408 & 0 & 0 \\
0{,}408 & 0{,}333 & 0{,}333 & 0 \\
0 & 0{,}333 & 0{,}333 & 0{,}408 \\
0 & 0 & 0{,}408 & 0{,}500
\end{bmatrix}.
$$

La tabella è simmetrica, e non poteva essere altrimenti: gli archi qui non
hanno un verso, quindi se il 2 è vicino dell'1 anche l'1 è vicino del 2, e il
peso del collegamento è lo stesso letto nei due sensi. Il passo di propagazione
è $\mathbf{H}' = \hat{\mathbf{A}}\,\mathbf{X}$, cioè per ogni nodo la somma
pesata di sé e dei suoi vicini:

$$
\begin{aligned}
h'_1 &= \tfrac{1}{2}\cdot 1 + \tfrac{1}{\sqrt6}\cdot 2 = 0{,}500 + 0{,}816 = 1{,}316, \\
h'_2 &= \tfrac{1}{\sqrt6}\cdot 1 + \tfrac{1}{3}\cdot 2 + \tfrac{1}{3}\cdot 3 = 0{,}408 + 0{,}667 + 1{,}000 = 2{,}075, \\
h'_3 &= \tfrac{1}{3}\cdot 2 + \tfrac{1}{3}\cdot 3 + \tfrac{1}{\sqrt6}\cdot 4 = 0{,}667 + 1{,}000 + 1{,}633 = 3{,}300, \\
h'_4 &= \tfrac{1}{\sqrt6}\cdot 3 + \tfrac{1}{2}\cdot 4 = 1{,}225 + 2{,}000 = 3{,}225.
\end{aligned}
$$

Le moltiplicazioni sono fatte con i pesi esatti ($\tfrac{1}{3}$ e
$\tfrac{1}{\sqrt6}$) e arrotondate solo alla fine: chi le rifà con i valori
tondi della tabella, $0{,}333$ e $0{,}408$, trova le ultime cifre diverse (per
esempio $0{,}333 \cdot 3 = 0{,}999$ e non $1{,}000$), e non ha sbagliato niente.

Il risultato è
$\mathbf{H}' \approx (1{,}316,\, 2{,}075,\, 3{,}300,\, 3{,}225)^\top$, e
racconta bene cosa fa la GCN: **i quattro valori si stringono**. Partivano da
$1$ e arrivavano a $4$, tre punti fra il più basso e il più alto; adesso vanno
da $1{,}32$ a $3{,}30$, due punti scarsi. Il nodo 1, che valeva $1$, sale a
$1{,}316$ perché è tirato in alto dal vicino 2; il nodo 4, che valeva $4$,
scende a $3{,}225$ perché è tirato in basso dal 3.

I due nodi di mezzo salgono invece tutti e due, e conviene non nasconderlo,
perché smonta una scorciatoia che verrebbe naturale: **questa non è la media
dei vicini**. Se lo fosse, il nodo 3, che vale $3$ e sta fra un $2$ e un $4$,
resterebbe a $3$; invece sale a $3{,}300$. La ragione è che i pesi di una riga
non sommano a uno (in quella del nodo 3 fanno $1{,}07$), quindi ogni giro non
è una media ma una somma pesata, che può alzare il livello generale. Quello
che la GCN garantisce non è che ciascuno vada verso i suoi vicini a ogni
singolo passo: è che, ripetendo, le differenze di partenza si consumino. Poco
più avanti in questa sezione si vede succedere, giro dopo giro, su questi
stessi quattro numeri.

### Perché normalizzare così

Nel conto appena fatto ogni collegamento portava un suo peso, e i pesi erano
tutti più piccoli di uno: qualcosa è stato diviso. Perché? Perché non sommare e
basta i bigliettini dei vicini?

`````{tab} Elementare

Immagina un riassunto fatto sommando e basta, senza dividere niente. Un nodo
con dieci amici riceve dieci bigliettini e li somma: un numerone. Un nodo con
due amici ottiene un numero piccolo. Dopo qualche giro, i nodi «popolari» hanno
valori enormi e quelli isolati valori minuscoli: non perché contino di più, ma
solo perché hanno più connessioni. La rete finirebbe per confondere «essere
importante» con «avere tanti amici».

La divisione rimette tutti sulla stessa scala, e lo spirito è quello di una
**media** invece di una somma: dieci opinioni o due, quello che conta è il
tenore, non il numero. Non è una media esatta, per la ragione appena vista (i
pesi di una riga non fanno precisamente uno), ma il mestiere che svolge è
quello. In più, il messaggio di un amico molto popolare pesa un po’ meno,
perché la sua attenzione è «spalmata» su tanti: proprio come il consiglio di
chi conosce mezzo mondo vale un filo meno di quello dell'amico che hai solo tu.

`````

`````{tab} Superiore

Ci sono due normalizzazioni naturali. Quella per righe,
$\tilde{\mathbf{D}}^{-1}\tilde{\mathbf{A}}$, fa la **media** dei vicini: ogni
riga somma a $1$, è la matrice di transizione di una passeggiata aleatoria. La
GCN usa invece quella **simmetrica**,
$\hat{\mathbf{A}} = \tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}$,
in cui il peso dell'arco $(v,u)$ è $1/\sqrt{\tilde{d}_v\,\tilde{d}_u}$: si
sconta il grado di *entrambi* gli estremi.

Va detto subito che cosa **non** distingue le due, perché è l'argomento che
viene spontaneo e non regge. Non è la **scala**: le due matrici sono simili,

$$
\tilde{\mathbf{D}}^{-1}\tilde{\mathbf{A}} =
\tilde{\mathbf{D}}^{-1/2}\big(\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\big)\tilde{\mathbf{D}}^{1/2},
$$

quindi hanno **esattamente lo stesso spettro**, contenuto in $[-1,1]$ con il
massimo pari a $1$. Tenere le attivazioni e i gradienti su una scala stabile,
strato dopo strato, è il guadagno della normalizzazione *in quanto tale*
rispetto a $\tilde{\mathbf{A}}$ nuda (che sui nodi ad alto grado amplifica i
valori in modo incontrollato): è il collegamento diretto con il problema dei
gradienti nelle reti profonde, discusso nella sezione sulla backpropagation, ed
è la sola cosa che Kipf e Welling rivendicano quando chiamano *renormalization
trick* il passaggio da
$\mathbf{I}_N + \mathbf{D}^{-1/2}\mathbf{A}\,\mathbf{D}^{-1/2}$, che ha
autovalori in $[0,2]$, alla forma
$\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}$ con
$\tilde{\mathbf{D}}$ calcolata su $\tilde{\mathbf{A}}$, cappi inclusi.
Attenzione a non leggerci più di quanto ci sia: scala stabile non vuol dire
informazione conservata, e poco più avanti in questa stessa pagina vedremo che
tutto ciò che non giace lungo l'autovettore dominante svanisce comunque.

Quel che distingue davvero la forma simmetrica è la **simmetria** stessa:
$\hat{\mathbf{A}}$ è autoaggiunta, quindi ha autovalori reali e una base di
autovettori **ortonormale**, mentre $\tilde{\mathbf{D}}^{-1}\tilde{\mathbf{A}}$
non è simmetrica e i suoi autovettori non sono ortogonali. È questa proprietà,
e non il controllo delle scale, a rendere lecito tutto ciò che segue:
decomporre un segnale sui nodi nelle sue «frequenze», moltiplicarle una per una
e ricomporre con $\mathbf{U}^\top$ presuppone una base ortonormale, e senza di
essa la lettura spettrale non sta in piedi.

Ed è anche la ragione per cui quella forma non è stata scelta: è **caduta** dal
conto. Qui sta il secondo punto, l’**origine spettrale**. La GCN nasce come
approssimazione al prim'ordine di una convoluzione definita nel dominio
spettrale del grafo: i filtri polinomiali di Čebyšëv di Defferrard, Bresson e
Vandergheynst {cite}`defferrard2016convolutional`. Troncare quel polinomio al
primo grado e riordinare i termini restituisce esattamente $\hat{\mathbf{A}}$:
è da qui che la normalizzazione simmetrica «cade» naturalmente, non è una
scelta arbitraria. Rispetto al modello originale di Scarselli e colleghi
{cite}`scarselli2009graph`, che iterava fino a un punto fisso, la GCN fissa un
numero piccolo di strati e si addestra come una qualunque rete profonda.

`````

### Da dove viene la formula: le frequenze di un grafo

Arrivati qui la domanda viene da sé: quella formula da dove esce? Nessuno l'ha
inventata a tavolino: è quel che **resta** di un conto più grande, e conviene
raccontare che conto sia. Rifarlo per intero vorrebbe strumenti che qui non
servono; ma l'idea si dice a parole in mezza pagina, ed è un buon affare,
perché in fondo c'è un premio: spiega da sola il difetto più famoso delle GNN.

`````{tab} Elementare

Su un'immagine c'è una parola che descrive quanto in fretta le cose cambiano da
un punto al punto accanto, e quella parola è **frequenza**: bassa vuol dire
zone di colore che cambiano piano, alta vuol dire dettagli fitti e bordi netti.
Un filtro che «sfoca» toglie le alte e tiene le basse.

Su un grafo la stessa parola ha un senso preciso, e basta cambiare che cosa si
guarda: una configurazione di numeri sui nodi è a **bassa frequenza** se nodi
collegati portano valori simili, ad **alta frequenza** se lungo ogni arco il
valore salta. Il caso estremo di bassa frequenza è «tutti lo stesso numero»;
quello di alta frequenza è una scacchiera, dove ogni vicino ha il segno
opposto.

Queste configurazioni, dalla più liscia alla più a scacchiera, hanno un nome
proprio: si chiamano gli **autovettori del laplaciano** del grafo. È il nome che si trova scritto ovunque, e vale la pena registrarlo perché
nell'ultima sezione del capitolo torna a fare un mestiere che nessuno si
aspetta: dire a ogni nodo dove sta nel grafo, come i Transformer dicono a ogni
parola dove sta nella frase.

Una volta stabilito questo si può copiare, di sana pianta, il mestiere di chi
lavora sui suoni e sulle immagini: si prendono i numeri sui nodi e si scrivono
come somma di quelle configurazioni, dalla più liscia alla più a scacchiera; si
decide quanto tenere di ciascuna, alzando le une e abbassando le altre; e si
rimette tutto insieme. È esattamente ciò che i primi lavori sulle reti
convoluzionali su grafo hanno fatto, e il computer ci metteva un tempo
proibitivo: per scrivere quelle configurazioni bisogna prima calcolarle, e su
un grafo grande è un lavoro immane.

La GCN è quello che resta dopo aver tagliato tutto il superfluo, e quel che
resta è un filtro che **attenua le alte frequenze**, cioè che smussa le
differenze fra vicini.

Tieni a mente questa frase, perché nei prossimi paragrafi torna con un'aria
molto meno amichevole.

`````

`````{tab} Superiore

Si parte dal **laplaciano normalizzato** del grafo,

$$
\mathbf{L} = \mathbf{I}_N - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top ,
$$

simmetrico e semidefinito positivo, quindi diagonalizzabile con autovettori
ortonormali $\mathbf{U}$ e autovalori reali $\lambda_i \in [0, 2]$.

Che gli autovalori siano «frequenze» non è un'analogia vaga: si legge dalla
**forma quadratica** del laplaciano normalizzato, che vale

$$
\mathbf{x}^\top \mathbf{L} \mathbf{x} = \sum_{(u,v) \in E}
\left( \frac{x_u}{\sqrt{d_u}} - \frac{x_v}{\sqrt{d_v}} \right)^{\!2} ,
$$

dove la somma percorre ogni arco non diretto una volta sola. È una somma di
quadrati di **differenze lungo gli archi**: un autovettore con $\lambda$
piccolo varia poco fra nodi collegati, uno con $\lambda$ grande alterna. Su un
autovettore di norma unitaria questa quantità *è* l'autovalore stesso, perché
coincide con il **quoziente di Rayleigh**
$R(\mathbf{x}) = \mathbf{x}^\top \mathbf{L} \mathbf{x} / \mathbf{x}^\top \mathbf{x}$:
la forma quadratica e il quoziente sono due oggetti distinti, e vale la pena
non confonderli, ma su $\lVert \mathbf{x} \rVert = 1$ dicono la stessa cosa, ed
è il quoziente (con il principio di minimax) a caratterizzare gli autovalori
come minimi della variazione. Su una griglia regolare gli autovettori del
laplaciano sono seni e coseni, e questa costruzione si riduce alla trasformata
di Fourier di sempre.

Definita la trasformata come $\hat{\mathbf{x}} = \mathbf{U}^\top \mathbf{x}$,
un filtro è una moltiplicazione punto per punto nello spettro e un ritorno
indietro:

$$
g_\theta \star \mathbf{x} = \mathbf{U}\, g_\theta(\boldsymbol{\Lambda})\, \mathbf{U}^\top \mathbf{x} ,
$$

che è la rete spettrale di Bruna e colleghi {cite}`bruna2014spectral`. Ha due
difetti fatali: richiede la diagonalizzazione di $\mathbf{L}$, cioè $O(N^3)$, e
i filtri appresi non sono **localizzati**, perché un
$g_\theta(\boldsymbol{\Lambda})$ arbitrario mescola nodi a distanza qualunque.

Entrambi si curano con lo stesso trucco: approssimare $g_\theta$ con un
**polinomio** di grado $K$, e in particolare con i polinomi di Čebyšëv
{cite}`hammond2011wavelets`,

$$
g_\theta(\boldsymbol{\Lambda}) \approx \sum_{k=0}^{K} \theta_k\, T_k(\tilde{\boldsymbol{\Lambda}}),
\qquad \tilde{\boldsymbol{\Lambda}} = \frac{2}{\lambda_{\max}}\boldsymbol{\Lambda} - \mathbf{I}_N ,
$$

con $T_k(x) = 2x\,T_{k-1}(x) - T_{k-2}(x)$, $T_0 = 1$, $T_1 = x$. Il guadagno è
doppio e vale la pena vederlo bene. Primo, poiché
$\mathbf{U} f(\boldsymbol{\Lambda}) \mathbf{U}^\top = f(\mathbf{L})$ per
qualunque polinomio $f$, gli autovettori spariscono dal conto: restano prodotti
fra la matrice sparsa $\mathbf{L}$ e un vettore, cioè $O(|E|)$ invece di
$O(N^3)$. Secondo, una potenza $\mathbf{L}^k$ è non nulla in $(u,v)$ solo se
esiste un cammino di lunghezza $\le k$ fra $u$ e $v$: un polinomio di grado $K$
è quindi automaticamente **$K$-localizzato**, tocca soltanto i vicini entro $K$
salti. È ChebNet {cite}`defferrard2016convolutional`, ed è già una GNN: la
localizzazione, che nella lettura spaziale era il punto di partenza, qui
*emerge* dal troncamento.

L'ultimo passo è di Kipf e Welling {cite}`kipf2017semi`, e consiste nel
rinunciare a quasi tutto. Si pone $K = 1$ (un solo salto per strato, la
profondità la darà lo stack) e si approssima $\lambda_{\max} \approx 2$, il che
manda $\tilde{\mathbf{L}} = \frac{2}{\lambda_{\max}} \mathbf{L} - \mathbf{I}_N$
in $-\mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$. Restano due parametri
liberi:

$$
g_{\theta'} \star \mathbf{x} \approx \theta'_0\, \mathbf{x} + \theta'_1 \big(- \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}\big) \mathbf{x} .
$$

Legandoli con $\theta = \theta'_0 = -\theta'_1$, per ridurre l'overfitting e i
parametri a uno solo per canale, si ottiene

$$
g_{\theta} \star \mathbf{x} \approx \theta \big( \mathbf{I}_N + \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2} \big) \mathbf{x} .
$$

La matrice fra parentesi ha autovalori in $[0, 2]$ e applicarla ripetutamente
fa esplodere i valori: da qui il *renormalization trick*, cioè sostituirla con
$\hat{\mathbf{A}} = \tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}$
dove $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}_N$ e $\tilde{\mathbf{D}}$ è
la matrice dei gradi di $\tilde{\mathbf{A}}$. È la formula della GCN, e il
conto rimasto in sospeso è saldato: la normalizzazione simmetrica non era una
scelta di comodo, è quel che resta di una convoluzione spettrale dopo due
approssimazioni.

I cappi, per inciso, non riscalano soltanto: rendono il grafo non bipartito e
staccano il fondo dello spettro da $-1$ (su una catena di otto nodi il minimo
passa da $-1$ a $-0{,}30$, su un ciclo di otto da $-1$ a $-0{,}33$), cioè
**smorzano la componente a frequenza più alta**
{cite}`wu2019simplifying`. È già il filtro passa-basso della prossima
sottosezione, comparso mentre credevamo di stare solo mettendo in sicurezza i
numeri.

`````

Il premio annunciato arriva adesso.

`````{tab} Elementare

Riprendi la frase da tenere a mente: la GCN è un filtro che smussa le
differenze fra vicini. Ogni giro di bigliettini ne cancella un po’; e se i giri
sono tanti? Le differenze finiscono.

Si vede sulla catena di quattro nodi di poco fa, quella che partiva da 1, 2, 3
e 4. Rifacendo il giro più volte, e sempre senza la ricetta di riscrittura né
il ritocco finale (che qui restano a riposo, come nel conto di prima), i
quattro valori vanno così:

| dopo | nodo 1 | nodo 2 | nodo 3 | nodo 4 |
|---|---|---|---|---|
| $0$ giri | $1{,}00$ | $2{,}00$ | $3{,}00$ | $4{,}00$ |
| $1$ giro | $1{,}32$ | $2{,}07$ | $3{,}30$ | $3{,}22$ |
| $5$ giri | $1{,}95$ | $2{,}57$ | $2{,}88$ | $2{,}50$ |
| $20$ giri | $2{,}22$ | $2{,}72$ | $2{,}73$ | $2{,}23$ |

Guarda l'ultima riga. Il nodo che partiva da $1$ e quello che partiva da $4$
sono finiti praticamente sullo stesso numero, e così i due di mezzo. Delle
differenze di partenza non è rimasto niente, e i due valori diversi che si
vedono ancora ($2{,}2$ e $2{,}7$) non dicono chi era il nodo: dicono soltanto
quanti vicini ha, uno i due di bordo e due i due interni.

Non serve nessun conto sofisticato per capire il perché. Se a ogni giro ognuno
si rimescola con i vicini, e i vicini fanno lo stesso con i loro, dopo un po’
nessuno ha più niente di suo: è la classe in cui tutti copiano un po’ dal
compagno di banco, e dopo un'ora i compiti si somigliano tutti e non si capisce
più chi la lezione la sapeva davvero.

`````

`````{tab} Superiore

Chiamiamo $\lambda_i(\hat{\mathbf{A}})$ gli autovalori dell'operatore appena
ottenuto: stanno in $[-1, 1]$, il più grande vale esattamente $1$, e il suo
autovettore è $\tilde{\mathbf{D}}^{1/2}\mathbf{1}$, cioè la radice dei gradi.
Uno strato GCN (a meno di $\mathbf{W}$ e della non linearità) è la
moltiplicazione per $\hat{\mathbf{A}}$; $K$ strati sono $\hat{\mathbf{A}}^K$.
Ma elevare alla $K$ una matrice eleva alla $K$ i suoi autovalori, e ogni
autovalore di modulo minore di $1$ **svanisce**: dopo abbastanza strati
sopravvive solo la componente lungo l'autovettore dominante, che è la stessa
per tutti i nodi a meno del loro grado. (Vale su un grafo **connesso**: se le
componenti connesse sono più d'una, l'autovalore $1$ ha la loro molteplicità e
il collasso avviene dentro ciascuna componente separatamente. Non è un caso di
scuola, perché un batch di molecole in PyTorch Geometric *è* un unico grafo
sconnesso, una componente per molecola, e l'oversmoothing non le mescola fra
loro.)

Sulla catena di quattro nodi del conto qui sopra, con
$\mathbf{X} = (1,2,3,4)^\top$, i quattro autovalori di $\hat{\mathbf{A}}$
valgono $1$, $0{,}729$, $0{,}167$ e $-0{,}229$. Applicando $\hat{\mathbf{A}}$
venti volte a $\mathbf{X}$, il rapporto fra il valore di ogni nodo e la radice
del suo grado con cappio vale $1{,}5714$, $1{,}5724$, $1{,}5739$, $1{,}5748$:
i quattro nodi, che partivano da valori distinti, coincidono ormai nelle prime
due cifre decimali. Il divario fra il più alto e il più basso è $3{,}4 \cdot
10^{-3}$ e si stringe come la potenza $K$-esima del secondo autovalore,
$0{,}729^K$: a cinquanta applicazioni vale $2{,}6 \cdot 10^{-7}$ e a cento
$3{,}5 \cdot 10^{-14}$. Anche lì i quattro numeri restano diversi fra loro (in doppia precisione li separano ancora un centinaio scarso di passi
elementari, il gradino minimo fra due numeri rappresentabili, e quanti
esattamente dipende dall'ordine in cui si fanno le moltiplicazioni), ma è una
differenza che nessun modello può più usare: al passo successivo della
rete, moltiplicata per pesi dell'ordine dell'unità, resta quello che era.

A rigore l'argomento vale per l'operatore lineare $\hat{\mathbf{A}}^K$, cioè
per la GCN privata di $\mathbf{W}$ e della non linearità. Nella rete completa i
pesi possono contrastare il collasso, e Oono e Suzuki {cite}`oono2020graph`
dimostrano che avviene comunque quando le norme dei pesi restano sotto una
soglia legata allo spettro di $\hat{\mathbf{A}}$: per l'operatore è algebra,
per la rete intera è un teorema con le sue condizioni.

`````

Questo appiattimento ha un nome, **oversmoothing**, cioè «levigatura
eccessiva», e l'ultima sezione del capitolo lo elencherà fra i limiti delle
GNN. Adesso però sappiamo che non è una sfortuna capitata in laboratorio: è
quello che fa, per costruzione, un filtro che smussa le differenze (in gergo un
**filtro passa-basso**) quando lo si applica molte volte di fila. Non c'è
nessun errore di programmazione da andare a cercare. C'è da decidere quanti
strati mettere, oppure da cambiare filtro.

## Impilare gli strati: il campo recettivo a $K$ salti

Come vada a finire se si esagera lo sappiamo già. Resta da dire perché,
fino a un certo punto, impilare gli strati conviene, e conviene molto: un solo
strato di GCN fa vedere a ogni nodo i suoi vicini diretti, e il bello comincia
appunto quando gli strati sono più d'uno.

`````{tab} Elementare

Torniamo alla catena $1 - 2 - 3 - 4$ e mettiamoci nei panni del nodo 1. Al
primo giro parla col nodo 2, il suo unico vicino. Ma attenzione: nello stesso
giro, anche il nodo 2 ha parlato col nodo 3. Così, al **secondo** giro, quando
il nodo 1 riascolta il nodo 2, dentro il nodo 2 c'è già un pezzo di nodo 3.
Senza essersi mai «visti» direttamente, l'informazione del nodo 3 è arrivata al
nodo 1 in due passi. Al terzo giro arriverebbe anche quella del nodo 4.

È esattamente ciò che succede in una rete convoluzionale, dove impilando i
livelli ogni neurone «vede» una porzione via via più grande dell'immagine: il
suo *campo recettivo* cresce con la profondità. Sul grafo vale la stessa legge,
contata in **salti**: con $K$ strati, ogni nodo raccoglie informazione da tutto
ciò che sta entro $K$ passi da lui. La striscia in basso nella
{numref}`fig-message-passing` mostra proprio questo salto da uno a due.

`````

`````{tab} Superiore

Impilare $K$ strati di GCN corrisponde ad applicare $K$ volte l'operatore di
propagazione: lo stato finale $\mathbf{h}_v^{(K)}$ dipende da tutti i nodi $u$
per cui esiste un cammino di lunghezza $\le K$ fino a $v$ (il **campo
recettivo** a $K$ salti, l'analogo esatto del campo recettivo che cresce con la
profondità nelle CNN). Da qui due indicazioni pratiche. Primo, la profondità va
scelta in base a **quanti salti di distanza** sta l'informazione che serve: due
o tre strati bastano quasi sempre, perché il numero di nodi raggiunti cresce in
fretta col grado.
Secondo, andare troppo profondi è controproducente: applicando molte volte
$\hat{\mathbf{A}}$ le rappresentazioni dei nodi convergono verso un unico punto
e diventano indistinguibili; il fenomeno dell’*oversmoothing*, per cui in
pratica le GCN molto profonde rendono peggio di quelle a due strati.

`````

```{figure} ../figures/message-passing.gif
:name: fig-message-passing-animato
:alt: Animazione di un grafo con un nodo centrale v, quattro vicini diretti e quattro nodi a due salti. Al primo giro i messaggi viaggiano dai vicini diretti verso v; al secondo giro partono prima dai nodi esterni verso i vicini, poi di nuovo verso v.
:width: 90%

Il campo recettivo che si allarga: al giro $k=1$ arrivano a $v$ solo i vicini
diretti; al giro $k=2$ i messaggi partono dai nodi a due salti, passano *per* i
vicini e arrivano anch'essi.
```

La {numref}`fig-message-passing-animato` rende evidente il punto che rende
delicata la profondità: l'informazione lontana non salta, **transita**. Ogni
strato in più la fa passare per un altro nodo, che la mescola con la propria,
ed è proprio questa mescolanza ripetuta a produrre, alla lunga,
l'oversmoothing.

## Addestrare: classificare i nodi con poche etichette

Con lo schema in mano, addestrare una GCN non richiede niente di nuovo: si
misura quanto la rete sbaglia, si calcola in che direzione muovere i pesi per
sbagliare meno e ci si muove di un passo, esattamente come nella sezione
sull'addestramento delle reti (sono la *loss*, il *gradiente* e la
*backpropagation* di quel capitolo). Cambia solo la forma del dato.

Il banco di prova classico è **Cora**, ed è un grafo di articoli scientifici.
I nodi sono circa 2700 articoli e gli archi le citazioni: c'è un arco ogni
volta che un articolo ne cita un altro, e in tutto sono circa 5400. Su ogni
nodo c'è una fila di 1433 numeri, che dice quali parole compaiono
nell'articolo. Il compito è assegnare a ciascun articolo una di 7 categorie
tematiche.

`````{tab} Elementare

La particolarità è che conosciamo l'argomento di **pochissimi** articoli (nella
versione standard di Cora appena 20 per categoria, 140 nodi in tutto su 2700) e
vogliamo indovinare quello di tutti gli altri. Come si fa con così poche
risposte in mano? Sfruttando i collegamenti: un articolo tende a citare
articoli del suo stesso campo. Il message passing fa scorrere le poche
etichette note lungo le citazioni, contagiando i vicini. È come indovinare gli
hobby di una comitiva conoscendone solo alcuni: chi frequenta i patiti di
scacchi, probabilmente gioca a scacchi anche lui.

Il trucco è che, pur pagando solo gli errori sui 140 articoli di cui sappiamo
la risposta, per rispondere su quei 140 la rete ha dovuto far girare
l'informazione su **tutto** il grafo. Aggiustandosi per i 140, quindi, migliora
la fila di numeri di tutti: anche quella degli altri articoli, più di
duemilacinquecento, su cui non le abbiamo mai detto se aveva ragione. Un
apprendimento che parte da poche
risposte e tanta struttura si chiama **semi-supervisionato**.

`````

`````{tab} Superiore

Formalmente è *node classification* semi-supervisionata in regime
**transduttivo**: il grafo intero, feature comprese, è visibile in
addestramento, ma solo un piccolo insieme $\mathcal{V}_{\text{train}}$ di nodi
è etichettato. Una GCN a due strati produce i logit per tutti i nodi,

$$
\mathbf{Z} = \hat{\mathbf{A}}\,\sigma\!\big(\hat{\mathbf{A}}\,\mathbf{X}\,\mathbf{W}^{(0)}\big)\,\mathbf{W}^{(1)},
$$

e la loss è la cross-entropia calcolata **solo** sui nodi etichettati:

$$
\mathcal{L} = -\sum_{v \in \mathcal{V}_{\text{train}}}
   \sum_{c=1}^{C} y_{vc} \, \log \hat{y}_{vc},
\qquad
\hat{\mathbf{y}}_{v} = \mathrm{softmax}(\mathbf{z}_v),
$$

dove $y_{vc}$ è l'etichetta one-hot del nodo $v$ per la classe $c$ e $C$ il
numero di classi. Il punto sottile è che $\mathbf{z}_v$ dipende, tramite
$\hat{\mathbf{A}}$, dalle feature dell'intero vicinato a due salti: il
gradiente di $\mathcal{L}$ fluisce quindi indietro anche attraverso nodi
**non** etichettati, che partecipano all'addestramento pur senza comparire
nella somma. Nel loro articolo Kipf e Welling riportano su Cora l’$81{,}5\%$ di
accuratezza contro il $75{,}7\%$ del miglior metodo che confrontano: quel salto,
ottenuto con appena due strati e $140$ nodi etichettati, è la ragione per cui la
GCN si è imposta.

`````

## Uno strato GCN in PyTorch

Quel che resta di questa sezione è per chi programma e vuole vedere la regola
tradotta in istruzioni. Non c'è nessuna idea nuova, solo la stessa scritta in
un'altra lingua: chi non programma può saltare i tre riquadri di codice e
riprendere dall'ultimo paragrafo, che è il ponte verso la sezione seguente.

Tradurre la regola
$\mathbf{H}^{(l+1)} = \sigma(\hat{\mathbf{A}}\,\mathbf{H}^{(l)}\,\mathbf{W}^{(l)})$
in codice è sorprendentemente breve. Uno strato è una trasformazione lineare
seguita dal prodotto con l'adiacenza normalizzata, precalcolata una volta sola:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)  # la matrice W

    def forward(self, H, A_hat):
        # H: (N, in_dim) stati dei nodi; A_hat: (N, N) adiacenza normalizzata
        return A_hat @ self.lin(H)  # Â (H W)

class GCN(nn.Module):
    def __init__(self, in_dim, hid, n_classi):
        super().__init__()
        self.gc1 = GCNLayer(in_dim, hid)
        self.gc2 = GCNLayer(hid, n_classi)

    def forward(self, H, A_hat):
        H = F.relu(self.gc1(H, A_hat))  # primo strato + ReLU
        H = self.gc2(H, A_hat)          # secondo strato: logit per nodo
        return H
```

L'addestramento è un normale ciclo di discesa del gradiente, con l'unico
accorgimento di mascherare la loss sui soli nodi etichettati:

```{code-block} python
:class: pt-non-eseguibile

model = GCN(in_dim=1433, hid=16, n_classi=7)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoca in range(200):
    model.train()
    opt.zero_grad()
    logit = model(H, A_hat)                                # tutti i nodi
    loss = F.cross_entropy(logit[mask_train], y[mask_train])  # solo etichettati
    loss.backward()                                        # backprop su tutto il grafo
    opt.step()
```

In pratica non serve scrivere lo strato a mano: la libreria **PyTorch
Geometric** offre `GCNConv`, che aggiunge i cappi e applica la normalizzazione
simmetrica al volo. Al posto della matrice $\hat{\mathbf{A}}$ intera prende il
grafo in un formato compatto, `edge_index`, che è la sola lista degli archi (di
forma `(2, num_archi)`). È l'unica strada praticabile sui grafi grandi, dove
$\hat{\mathbf{A}}$ per intero non entrerebbe in memoria:

```python
from torch_geometric.nn import GCNConv

conv = GCNConv(in_channels=1433, out_channels=16)
# forward: conv(x, edge_index), con x di forma (N, 1433)
```

Da qui in avanti le domande diventano: e se i vicini fossero troppi per
guardarli tutti? E se alcuni contassero più di altri? Sono le questioni
dell'ultima sezione del capitolo, «Oltre la GCN», dove incontreremo il
campionamento dei vicini di GraphSAGE e i pesi di attenzione delle Graph
Attention Network. Prima però c'è una sezione che allarga il campo in un'altra
direzione: che cosa succede quando gli archi non sono tutti uguali e ciascuno
porta scritto sopra un verbo.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il **passaparola** (*message passing*) aggiorna ogni nodo in tre mosse,
  ripetute a ogni giro: ogni vicino passa un bigliettino con quello che sa, i
  bigliettini si **riassumono** in un modo che non guarda l'ordine (somma,
  media, massimo) e il nodo **riscrive la propria scheda** unendo il riassunto
  a quello che già sapeva di sé. È lo schema generale, valido per quasi tutte
  le reti su grafo {cite}`gilmer2017neural`.
- La **GCN** {cite}`kipf2017semi` è la versione più usata: la raccolta dei
  bigliettini segue la rubrica di chi è collegato a chi, in cui ogni nodo
  figura anche come vicino di sé stesso per non dimenticare la propria scheda;
  poi una ricetta di riscrittura uguale per tutti i nodi (sono i numeri che la
  rete impara) e un ritocco finale non lineare.
- I bigliettini si **pesano** invece di sommarli e basta: chi ha tanti vicini
  non deve coprire la voce degli altri, un po’ come fare una media invece di un
  totale. Serve anche a tenere i valori sulla stessa scala giro dopo giro.
- Anche su un grafo si può parlare di **frequenze**: bassa se nodi collegati
  portano valori simili, alta se lungo ogni collegamento il valore salta. Un
  giro di GCN è un filtro che attenua le alte, cioè smussa le differenze fra
  vicini; e la sua formula è ciò che resta, dopo aver tagliato il superfluo,
  dei filtri che si usano sui suoni e sulle immagini
  {cite}`defferrard2016convolutional`.
- Ogni strato in più allarga l'orecchio di un salto: con due giri arrivano gli
  amici degli amici, con tre quelli ancora dopo, come in una rete per immagini
  il campo visivo di un neurone cresce con la profondità. Ma smussando a ogni
  giro, troppi giri cancellano le differenze e i nodi diventano
  indistinguibili (è l’*oversmoothing*, la classe in cui tutti copiano dal
  compagno di banco finché i compiti si somigliano tutti): non è un errore di
  programmazione, è quello che il metodo fa per costruzione.
- L'addestramento tipico è indovinare la categoria di tutti i nodi
  conoscendola per pochissimi (Cora): si pagano solo gli errori su quei pochi,
  ma per rispondere la rete ha dovuto far girare l'informazione su tutto il
  grafo, e così impara a rappresentare bene anche gli altri.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il **message passing** aggiorna ogni nodo in tre mosse ripetute a ogni
  strato: calcola i **messaggi** dei vicini, li **aggrega** con una funzione
  invariante all'ordine (somma, media, max) e **aggiorna** lo stato del nodo.
  È il telaio generale delle MPNN {cite}`gilmer2017neural`.
- La **GCN** {cite}`kipf2017semi` è l'istanza più usata:
  $\mathbf{H}^{(l+1)} = \sigma(\hat{\mathbf{A}}\,\mathbf{H}^{(l)}\,\mathbf{W}^{(l)})$
  con
  $\hat{\mathbf{A}} = \tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}$
  e $\tilde{\mathbf{A}} = \mathbf{A}+\mathbf{I}$.
- **Normalizzare** impedisce ai nodi ad alto grado di dominare e tiene gli
  autovalori in $[-1,1]$, stabilizzando le scale attraverso gli strati: vale
  però per entrambe le forme, quella per righe e quella simmetrica, che sono
  matrici **simili** e hanno lo stesso spettro. La GCN sceglie la
  **simmetrica** perché $\hat{\mathbf{A}}$ è autoaggiunta, quindi ha
  autovettori ortonormali, ed è questo a rendere lecita la lettura spettrale da
  cui la formula discende (filtri di Čebyšëv,
  {cite}`defferrard2016convolutional`).
- Sul grafo le **frequenze** hanno un senso preciso: un segnale è a bassa
  frequenza se nodi collegati portano valori simili, e gli autovalori del
  laplaciano le misurano. Una convoluzione spettrale costa $O(N^3)$; troncarla
  a un polinomio di Čebyšëv di grado $K$ la rende sparsa e **$K$-localizzata**,
  e il caso $K=1$ con $\lambda_{\max}\approx 2$ **è** la GCN.
- Impilare $K$ strati dà a ogni nodo un **campo recettivo a $K$ salti**,
  l'esatto analogo della profondità nelle CNN. Ma uno strato GCN è un **filtro
  passa-basso**, e applicarlo molte volte lascia sopravvivere solo
  l'autovettore dominante di $\hat{\mathbf{A}}$: è l’*oversmoothing*, e non è
  un incidente ma una conseguenza algebrica.
- L'addestramento tipico è la **classificazione dei nodi semi-supervisionata**
  (Cora): cross-entropia sui soli nodi etichettati, ma gradienti che fluiscono
  su tutto il grafo, con la solita discesa del gradiente.
```

`````
