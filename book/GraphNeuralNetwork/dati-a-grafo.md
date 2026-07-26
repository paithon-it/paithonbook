# Il mondo come grafo

Fermati un attimo a guardare i dati che ti circondano. I tuoi contatti sul
telefono non sono un elenco: sono una rete di persone che, a loro volta, si
conoscono tra loro. Una molecola non è una lista di atomi, ma un intreccio di
legami. Il web non è una pila di pagine, bensì un tessuto di link. La rete
stradale di una città, i pagamenti tra conti bancari, le citazioni tra articoli
scientifici, le proteine che interagiscono in una cellula: in tutti questi
casi l'informazione più preziosa non sta *dentro* le singole entità, ma nelle
**relazioni** tra loro. Il gatto nero che abbiamo seguito per tutto il libro,
qui, non ci basta più da solo: ci interessa *chi* gli sta accanto.

Le reti che abbiamo studiato finora (quelle convoluzionali per le immagini,
quelle ricorrenti e i Transformer per il testo) sono nate per dati con una
forma regolare: una griglia di pixel, una sequenza di parole. Ma un grafo non
ha né una griglia né un ordine. È una struttura più libera e, proprio per
questo, più difficile da dare in pasto a una rete neurale. Questo capitolo
racconta come si fa; questa prima sezione mette in fila il vocabolario e i
problemi di fondo, per capire *perché* servono strumenti nuovi.

## Nodi e archi: l'anatomia di un grafo

Un **grafo** è la cosa più semplice del mondo: un insieme di puntini e un
insieme di linee che li collegano. I puntini si chiamano **nodi** (o vertici),
le linee **archi** (o spigoli). Tutto il resto è dettaglio: importante, ma
dettaglio. La {numref}`fig-grafo-anatomia` mostra un grafo minuscolo, cinque
nodi, che useremo come filo conduttore per l'intera sezione.

```{figure} ../figures/grafo-anatomia.svg
:name: fig-grafo-anatomia
:alt: Un grafo non diretto di cinque nodi con i suoi archi; accanto a ogni nodo un vettore di tre feature; a destra la matrice di adiacenza A e la matrice delle feature X.
:width: 100%

Un grafo di 5 nodi. A ogni nodo è associato un piccolo **vettore di feature**
(le tre caselle). La struttura è riassunta dalla **matrice di adiacenza** $A$
(chi è collegato a chi) e le feature dalla **matrice** $X$ (una riga per nodo).
Evidenziato in terracotta il **vicinato** del nodo 3, che ha grado 3.
```

Prima di formalizzare, sistemiamo tre distinzioni che tornano di continuo.

`````{tab} Elementare

Gli archi possono avere o non avere un verso. In una rete di amicizie
l'amicizia è reciproca, se sono tuo amico, sei mio amico: gli archi non hanno
freccia, il grafo è **non diretto**. In una rete di «chi segue chi» sui
social, invece, posso seguire una persona che non mi segue: ogni arco ha un
verso, il grafo è **diretto**.

Gli archi possono anche avere un **peso**, cioè un numero che dice quanto è
forte quel collegamento: tra due città il peso può essere la distanza in
chilometri, tra due persone il numero di messaggi che si scambiano. Se non
mettiamo pesi, è come dire che ogni arco vale 1: il collegamento o c'è o non
c'è.

Infine, sia i nodi sia (a volte) gli archi portano con sé delle
**caratteristiche**, in gergo *feature*: per una persona l'età e gli interessi,
per un atomo il tipo di elemento. Nella figura sono le tre caselline accanto a
ogni nodo. Un grafo, insomma, è fatto di due cose insieme: una **struttura**
(chi è connesso a chi) e dei **contenuti** (cosa sono i nodi).

`````

`````{tab} Superiore

Un grafo è una coppia $G = (V, E)$, dove $V$ è l'insieme dei nodi ($N = |V|$) ed
$E \subseteq V \times V$ l'insieme degli archi. La struttura si codifica nella
**matrice di adiacenza** $A \in \mathbb{R}^{N \times N}$:

$$
A_{ij} =
\begin{cases}
1 & \text{se esiste l'arco } (i, j),\\
0 & \text{altrimenti.}
\end{cases}
$$

Se il grafo è **non diretto** allora $A_{ij} = A_{ji}$, cioè $A$ è simmetrica
($A = A^\top$); se è **diretto** non lo è in generale. In un grafo **pesato** lo
0/1 è sostituito dal peso $w_{ij} \in \mathbb{R}$ dell'arco. Le **feature dei
nodi** si impilano nella matrice $X \in \mathbb{R}^{N \times F}$, la cui riga
$i$-esima è il vettore $x_i \in \mathbb{R}^F$ delle $F$ caratteristiche del nodo
$i$; eventuali feature di arco si raccolgono in un tensore analogo indicizzato
dalle coppie.

Il **grado** di un nodo è il numero dei suoi vicini, cioè la somma della sua
riga: $\deg(i) = \sum_{j} A_{ij}$. I gradi si radunano nella **matrice dei
gradi** $D \in \mathbb{R}^{N \times N}$, diagonale, con

$$
D_{ii} = \deg(i) = \sum_{j} A_{ij}, \qquad D_{ij} = 0 \ \text{ per } i \neq j,
$$

dove $D_{ii}$ è quante linee escono dal nodo $i$. Le tre matrici $A$, $X$ e $D$
sono tutto ciò che serve per descrivere un grafo con feature: le ritroveremo in
ogni formula del capitolo.

`````

## Perché i grafi mettono in crisi le reti classiche

Verrebbe la tentazione di prendere la matrice di adiacenza $A$, allungarla in un
vettore e darla in pasto a una rete densa come faremmo con un'immagine
appiattita. Non funziona, e capire *perché non funziona* è la chiave di tutto il
capitolo. Il problema nasce da una libertà che griglie e sequenze non hanno: in
un grafo **i nodi non hanno un ordine**.

Un'immagine è una griglia: il pixel in alto a sinistra è *sempre* in alto a
sinistra, e la convoluzione sfrutta proprio questa regolarità. Una frase è una
sequenza: la prima parola viene sempre prima della seconda, e la ricorrenza (o
la posizione nei Transformer) conta su quell'ordine. Un grafo no: se rinumero
i suoi nodi (chiamo «1» quello che prima chiamavo «3» e viceversa) è
esattamente lo *stesso* grafo, ma la matrice $A$ cambia completamente aspetto.

`````{tab} Elementare

Immagina di fotografare le persone a una festa e le loro amicizie. Se poi
rifai l'elenco degli invitati in ordine diverso (prima per cognome, poi per
età), la festa non è cambiata di una virgola: sono le stesse persone, le
stesse amicizie. Ma se avessi scritto le amicizie come una tabella «riga per
invitato», riordinando l'elenco la tabella si stravolge, pur descrivendo la
stessa realtà.

Una rete che analizza i grafi deve capire questa cosa ovvia per noi: **l'ordine
in cui elenco i nodi non conta**. Se do lo stesso grafo con i nodi numerati in
due modi diversi, la risposta deve essere identica (per una proprietà
dell'intero grafo) oppure semplicemente rietichettata allo stesso modo (per una
risposta nodo per nodo). Le reti per immagini e testo, che invece si aspettano
un ordine fisso, qui inciamperebbero: vedrebbero due grafi diversi dove ce n'è
uno solo.

E c'è un secondo scoglio: nelle immagini ogni pixel ha sempre lo stesso numero
di vicini, in una frase ogni parola ha una prima e una dopo. In un grafo no: un
nodo può avere 2 vicini, un altro 100. Non esiste una «finestra» di dimensione
fissa da far scorrere.

`````

`````{tab} Superiore

Chiamiamo $P$ una **matrice di permutazione** $N \times N$ (una sola $1$ per
riga e colonna). Rinumerare i nodi significa trasformare $A \mapsto P A P^\top$
e $X \mapsto P X$. Poiché questi descrivono lo stesso grafo, un modello sensato
deve rispettare una di due proprietà. Per un compito che produce **una risposta
per l'intero grafo**, serve l'**invarianza a permutazione**:

$$
f(P A P^\top,\, P X) = f(A, X),
$$

la predizione non cambia comunque si rinumerino i nodi. Per un compito che
produce **una risposta per nodo** (un vettore per ciascuno), serve invece
l'**equivarianza a permutazione**:

$$
f(P A P^\top,\, P X) = P\, f(A, X),
$$

le uscite si permutano *insieme* agli ingressi. Progettare architetture che
incorporano questa simmetria come *bias induttivo* (anziché sperare che la
rete la impari a forza di esempi) è il cuore del programma della *geometric
deep learning* {cite}`bronstein2021geometric`, che legge sotto un'unica lente
CNN (invarianza a traslazione sulla griglia) e GNN (invarianza a permutazione
sul grafo). A ciò si aggiungono due irregolarità: il **grado variabile** (ogni
nodo ha un numero diverso di vicini, quindi niente kernel di dimensione fissa)
e la **taglia variabile** ($N$ cambia da grafo a grafo, mentre una rete densa
vuole un input di dimensione fissata). Sono esattamente i vincoli che il
*message passing* della prossima sezione risolverà con un'operazione locale,
condivisa e simmetrica.

`````

## Tre modi di fare una domanda a un grafo

Non tutti i problemi su grafo hanno la stessa forma. Conviene distinguere tre
**livelli** di compito, a seconda di *cosa* vogliamo prevedere: un singolo
nodo, una coppia di nodi, o l'intero grafo. La {numref}`fig-grafo-tre-compiti`
li mette in fila sullo stesso grafo.

```{figure} ../figures/grafo-tre-compiti.svg
:name: fig-grafo-tre-compiti
:alt: "Lo stesso grafo di cinque nodi in tre pannelli: nel primo un nodo è colorato (classificazione di nodo); nel secondo un arco tratteggiato con un punto interrogativo tra due nodi non collegati (link prediction); nel terzo l'intero grafo dentro un riquadro con un'unica etichetta (proprietà del grafo)."
:width: 100%

I tre livelli di compito. **Nodo**: prevedere un'etichetta per ciascun nodo.
**Arco**: prevedere se due nodi sono (o saranno) collegati (la *link
prediction*). **Grafo**: prevedere una proprietà dell'intero grafo, per
esempio se una molecola è tossica.
```

- **Livello-nodo.** A ogni nodo si assegna un'etichetta o un valore. È il caso
  di un social network in cui vogliamo classificare gli utenti (per esempio
  distinguere account autentici e bot), o di una rete di citazioni in cui
  prevediamo l'argomento di ciascun articolo.
- **Livello-arco.** Si prevede se un arco esiste, o esisterà, tra due nodi: la
  **link prediction**. È il motore del «forse conosci…» di un social e del «chi
  ha comprato questo…» di un negozio online. Non a caso il grafo bipartito
  utente–prodotto e la previsione di collegamento sono esattamente il modo in
  cui abbiamo descritto la raccomandazione nel capitolo sui sistemi di
  raccomandazione: consigliare un film è prevedere un arco mancante.
- **Livello-grafo.** Si prevede una proprietà dell'*intero* grafo, riassunto in
  un solo verdetto. L'esempio principe è la chimica: una molecola è un grafo di
  atomi e legami, e vogliamo prevedere se è solubile, tossica, o efficace come
  farmaco.

C'è poi una distinzione ortogonale, che riguarda *quali* nodi vediamo in fase di
addestramento, ed è quella che più di ogni altra separa i metodi antichi da
quelli moderni.

`````{tab} Elementare

Immagina di dover indovinare l'argomento di ogni articolo in una biblioteca
collegata da citazioni. In un caso hai già davanti *tutti* gli articoli, con le
loro citazioni: di alcuni conosci l'argomento, di altri no, e devi solo
riempire i buchi. Questo si dice modo **transduttivo**: il grafo è uno solo,
fissato, e non arriverà mai nessun articolo nuovo.

In un altro caso, invece, vuoi imparare una regola che funzioni anche su
articoli che *non hai ancora visto*, o addirittura su un'altra biblioteca. È
il modo **induttivo**: si impara qualcosa di generale, che si applica a nodi e
grafi mai incontrati durante l'addestramento. È la differenza tra imparare a
memoria la mappa di una città e imparare a leggere le mappe: la seconda
abilità funziona anche in una città nuova.

`````

`````{tab} Superiore

Nell'impostazione **transduttiva** l'addestramento e la predizione avvengono
sullo stesso grafo fisso $G$: tutti i nodi (etichettati e non) sono noti fin
dall'inizio, e l'obiettivo è propagare le etichette dai nodi noti a quelli
ignoti; è la classificazione di nodo *semi-supervisionata*. Nell'impostazione
**induttiva** si impara invece una funzione $f$ che generalizza a nodi o
interi grafi *mai visti* in addestramento: indispensabile quando il grafo
evolve nel tempo (un social in cui si iscrivono nuovi utenti) o quando ogni
esempio è un grafo distinto (un dataset di molecole). Come vedremo tra poco, i
primi metodi di rappresentazione (quelli basati sui cammini casuali) sono
intrinsecamente transduttivi, e sarà proprio questo limite a spingere verso le
reti neurali su grafo, induttive per costruzione {cite}`hamilton2020graph`.

`````

## Un primo assaggio senza reti neurali: camminare a caso sul grafo

Prima di tirare in ballo le reti neurali, vale la pena vedere un'idea più
semplice che, per qualche anno, è stata lo stato dell'arte per rappresentare i
nodi. L'obiettivo è familiare: come per le parole nel capitolo di NLP, vogliamo
dare a ogni nodo un **embedding**, un vettore denso di poche decine di numeri,
tale che nodi «vicini» nel grafo finiscano vicini anche nello spazio dei
vettori. E il trucco per ottenerlo è sorprendente: riusare, quasi senza
modifiche, lo *skip-gram* di word2vec.

`````{tab} Elementare

Ricorda l'idea degli word embedding: una parola si conosce dalla compagnia che
frequenta, cioè dalle parole che le compaiono accanto nelle frasi. Ma un grafo
non ha frasi. E allora **fabbrichiamocele**: partiamo da un nodo e facciamo
una passeggiata a caso, saltando ogni volta a un vicino scelto a sorte;
annotiamo i nodi che tocchiamo, in ordine. Otteniamo una sequenza («nodo 3,
nodo 1, nodo 2, nodo 3, nodo 4…») che possiamo trattare esattamente come una
frase, in cui ogni nodo è una «parola».

Ripetiamo migliaia di volte, da tutti i nodi, e ci ritroviamo con un enorme
«testo» fatto di passeggiate. A quel punto diamo in pasto queste finte frasi
allo stesso algoritmo che imparava gli embedding delle parole: i nodi che
capitano spesso vicini nelle passeggiate (cioè quelli ben connessi tra loro)
riceveranno vettori simili. È l'idea di **DeepWalk**. Una variante di poco
successiva, **node2vec**, aggiunge due manopole per decidere *che tipo* di
passeggiata fare: più «esploratrice», che si allontana, oppure più «pigra»,
che gironzola attorno al punto di partenza; così si può scegliere se catturare
comunità larghe o ruoli locali.

`````

`````{tab} Superiore

**DeepWalk** {cite}`perozzi2014deepwalk` genera, da ogni nodo, un certo numero
di cammini casuali di lunghezza fissa: da $v$ si passa a un vicino scelto
uniformemente, e si itera. Ogni cammino $(v_1, v_2, \dots, v_\ell)$ è trattato
come una «frase» e dato a **skip-gram**: si massimizza la probabilità dei nodi
del contesto (entro una finestra) dato il nodo centrale,

$$
\max_{\theta} \ \sum_{i} \sum_{-c \le k \le c,\, k \neq 0}
\log P_\theta\big(v_{i+k} \mid v_i\big),
$$

dove $c$ è la mezza-ampiezza della finestra, $\theta$ raccoglie gli embedding
appresi e $P_\theta$ è la solita softmax (in pratica approssimata con
*negative sampling* o softmax gerarchica, per non normalizzare su tutti i
nodi). **node2vec** {cite}`grover2016node2vec` rende il cammino *distorto*
(*biased*): due iperparametri $p$ e $q$ controllano la probabilità, a ogni
passo, di tornare indietro, restare nei paraggi o allontanarsi. Regolando $p$
e $q$ si interpola con continuità tra un'esplorazione «in ampiezza» (di tipo
BFS, che tende a cogliere l'*equivalenza strutturale*, nodi con ruoli simili,
per esempio due «hub») e una «in profondità» (di tipo DFS, che esplora regioni
più ampie e coglie l'*omofilia*, nodi della stessa comunità). In entrambi i
casi l'embedding di un nodo è appreso come una riga di una tabella,
esattamente come per le parole.

`````

Questi metodi funzionano e sono tuttora un'ottima *baseline*. Ma portano scritti
in fronte tre limiti, ed è illuminante metterli a fuoco, perché sono
esattamente i punti che le reti neurali su grafo verranno a risolvere.

- Sono **transduttivi**: l'embedding è una riga di tabella imparata per *quel*
  nodo. Arriva un nodo nuovo? Non ha nessuna riga, e bisogna riaddestrare. Non
  c'è modo di generalizzare a un grafo mai visto.
- **Ignorano le feature dei nodi**: guardano solo la struttura (chi è connesso a
  chi), buttando via la matrice $X$. Due nodi con le stesse connessioni ma
  contenuti diversissimi ricevono embedding identici.
- **Non condividono parametri**: ogni nodo ha il suo vettore indipendente, senza
  una funzione riusabile che, dati struttura e feature, *calcoli* la
  rappresentazione. È l'opposto della condivisione dei pesi che rende potenti le
  CNN.

## Verso le reti neurali su grafo

Il conto della serva è presto fatto. Da una parte abbiamo la struttura del
grafo, la matrice $A$; dall'altra le feature dei nodi, la matrice $X$. I cammini
casuali usano solo la prima. Le reti dense saprebbero usare la seconda, ma non
sanno gestire l'assenza di ordine e il grado variabile. Ci serve un modello che
usi **struttura e feature insieme**, che sia **induttivo** (una funzione
riusabile, non una tabella), che **condivida i parametri** su tutti i nodi come
la convoluzione li condivide su tutti i pixel, e che rispetti l'invarianza a
permutazione discussa sopra.

L'idea che concilia tutte queste richieste è tanto semplice quanto feconda: far
sì che ogni nodo **aggiorni la propria rappresentazione ascoltando i vicini**, e
ripetere l'operazione a strati, con gli stessi pesi ovunque. Ogni nodo, a ogni
strato, raccoglie messaggi da chi gli sta intorno e li fonde con ciò che già sa.
È il **message passing**, il meccanismo che dà il nome e la sostanza alle *Graph
Neural Network* e a cui è dedicata la prossima sezione.

## Un esempio numerico: dalla figura alla matrice

Chiudiamo mettendo le mani nei numeri, sul grafo di cinque nodi della
{numref}`fig-grafo-anatomia`. I suoi archi sono: 1–2, 1–3, 2–3, 3–4 e 4–5. È un
grafo non diretto e non pesato, quindi la matrice di adiacenza $A$ è simmetrica
e fatta di soli 0 e 1. Riga per riga, mettiamo un 1 dove due nodi sono collegati:

$$
A =
\begin{pmatrix}
0 & 1 & 1 & 0 & 0\\
1 & 0 & 1 & 0 & 0\\
1 & 1 & 0 & 1 & 0\\
0 & 0 & 1 & 0 & 1\\
0 & 0 & 0 & 1 & 0
\end{pmatrix}.
$$

Il **grado** di ogni nodo è la somma della sua riga: quante linee ne escono.
Contando i vicini: il nodo 1 ne ha 2 (il 2 e il 3), il nodo 2 ne ha 2, il nodo
3 ne ha 3 (l'unico «snodo», evidenziato in figura), il nodo 4 ne ha 2, il nodo
5 ne ha 1 (solo il 4). Quindi i gradi sono $(2, 2, 3, 2, 1)$, e la loro somma
vale $2+2+3+2+1 = 10$: esattamente il doppio dei 5 archi, come dev'essere
(ogni arco conta per i due nodi che collega). La **matrice dei gradi** $D$ è
diagonale e porta questi valori:

$$
D =
\begin{pmatrix}
2 & 0 & 0 & 0 & 0\\
0 & 2 & 0 & 0 & 0\\
0 & 0 & 3 & 0 & 0\\
0 & 0 & 0 & 2 & 0\\
0 & 0 & 0 & 0 & 1
\end{pmatrix}.
$$

C'è un'ultima mossa che ritroveremo di continuo nella prossima sezione:
aggiungere a ogni nodo un **cappio** (*self-loop*), cioè un arco che lo collega a
sé stesso. Serve a far sì che, quando un nodo «ascolta» i vicini, non dimentichi
sé stesso. In matrice significa mettere degli 1 sulla diagonale, cioè sommare la
matrice identità $I$:

$$
\tilde{A} = A + I =
\begin{pmatrix}
1 & 1 & 1 & 0 & 0\\
1 & 1 & 1 & 0 & 0\\
1 & 1 & 1 & 1 & 0\\
0 & 0 & 1 & 1 & 1\\
0 & 0 & 0 & 1 & 1
\end{pmatrix},
$$

dove $I$ è la matrice identità $5 \times 5$ e $\tilde{A}$ è l'adiacenza «con i
cappi». Ogni nodo guadagna così un vicino in più (sé stesso) e i gradi salgono
di uno: da $(2,2,3,2,1)$ a $(3,3,4,3,2)$.

`````{tab} Elementare

Tutto qui: costruire $A$, $D$ e $\tilde{A}$ è solo contare vicini e riportare i
conti in una tabella quadrata. Con `numpy` diventano tre righe, e i numeri sono
esattamente quelli scritti sopra:

```python
import numpy as np

A = np.array([          # matrice di adiacenza (righe/colonne = nodi 1..5)
    [0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0],
])

gradi = A.sum(axis=1)          # somma di ogni riga -> [2 2 3 2 1]
D = np.diag(gradi)             # matrice dei gradi (diagonale)
A_tilde = A + np.eye(5, dtype=int)   # aggiunge i self-loop: A + I

print(gradi)                   # [2 2 3 2 1]
print(A_tilde.sum(axis=1))     # gradi con i cappi: [3 3 4 3 2]
```

`````

`````{tab} Superiore

Queste tre matrici non sono un esercizio di contabilità: sono i mattoni con cui
si costruisce la convoluzione su grafo. Il *message passing* della prossima
sezione, nella sua forma più nota, non usa direttamente $\tilde{A}$ ma la sua
versione **normalizzata simmetricamente**

$$
\hat{A} = \tilde{D}^{-1/2}\, \tilde{A}\, \tilde{D}^{-1/2},
$$

dove $\tilde{D}$ è la matrice dei gradi di $\tilde{A}$ (quella con i cappi,
qui $\mathrm{diag}(3,3,4,3,2)$). La normalizzazione serve a evitare che i nodi
con tanti vicini dominino la somma dei messaggi, riscalando ogni contributo
secondo i gradi delle due estremità dell'arco. I dettagli (perché proprio
$-1/2$ da entrambi i lati, e cosa c'entra il Laplaciano del grafo) sono il
cuore della prossima sezione; qui basti aver visto da dove partono: da $A$, da
$D$ e dal gesto elementare di aggiungere $I$.

`````

```{admonition} Da ricordare
:class: important
- Un **grafo** $G=(V,E)$ è fatto di **nodi** e **archi** (diretti o no, pesati o
  no). La struttura sta nella **matrice di adiacenza** $A$, le caratteristiche
  dei nodi nella **matrice delle feature** $X$; il **grado** di un nodo è la
  somma della sua riga, raccolto nella **matrice diagonale dei gradi** $D$.
- I grafi sfidano le reti classiche perché **i nodi non hanno ordine** (serve
  **invarianza/equivarianza a permutazione**), hanno **grado variabile** e
  **taglia variabile**: né la griglia della CNN né la sequenza della RNN vanno
  bene.
- Tre livelli di compito: **nodo** (classificare gli utenti), **arco** (*link
  prediction*: suggerire un'amicizia o un prodotto), **grafo** (una proprietà
  della molecola). E due regimi: **transduttivo** (grafo fisso) vs
  **induttivo** (generalizzare a nodi/grafi nuovi).
- I **cammini casuali** (**DeepWalk**, **node2vec**) trattano le passeggiate sul
  grafo come «frasi» e riusano *skip-gram*: buoni embedding, ma
  **transduttivi**, ciechi alle feature e senza condivisione di parametri.
- Le **GNN** nascono per superare questi limiti: combinare **struttura e
  feature** in modo **induttivo** e con **pesi condivisi**, facendo aggiornare
  ogni nodo tramite i suoi vicini. Il come è il **message passing** della
  prossima sezione.
```
