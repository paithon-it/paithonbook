# Fatti come archi: i knowledge graph

Nel maggio del 2012 Google annuncia una modifica al motore di ricerca con uno
slogan che vale più della modifica: *things, not strings*, cose e non stringhe.
Fino a quel momento cercare «Torino» significava chiedere le pagine che
contengono quella sequenza di sei caratteri. Da lì in avanti il motore prova a
sapere che Torino è una **città**, che sta in Italia, che ha un fiume, un
sindaco e una squadra di calcio, e che «Torino» può anche essere quella
squadra.

L'idea non era nuova. La stessa struttura era già stata proposta tre volte, e
vale la pena elencarle. Negli anni Sessanta con le **reti semantiche**, schemi
in cui i concetti sono puntini e le linee fra loro dicono «è un», «ha un», «si
trova in». Dagli anni Ottanta con quei progetti in cui squadre di persone
scrivevano a mano, un fatto per volta, le ovvietà che tutti sanno e nessuno
scrive («la pioggia bagna», «chi dorme ha gli occhi chiusi»). E infine con il
**web semantico**, che voleva pagine leggibili non solo dalle persone ma anche
dai programmi. Nuovo era che, per la prima volta, un grafo di fatti abbastanza
grande da servire a qualcosa si poteva costruire in modo automatico.

Fin qui il capitolo ha trattato grafi in cui tutti i nodi sono la stessa specie
di cosa e tutti gli archi vogliono dire la stessa cosa: utenti, atomi,
articoli. Questa sezione toglie quella comodità.

## Una tripla è un fatto

`````{tab} Elementare

Un **knowledge graph** è un grafo in cui ogni arco porta scritto sopra che cosa
lega le due cose che collega: «si trova in», «è capitale di», «ha diretto».
L'unità elementare è una frasetta di tre parole:

> (Torino, si-trova-in, Piemonte) · (Torino, attraversata-da, Po) ·
> (Po, sfocia-in, Adriatico)

Soggetto, relazione, oggetto. Si chiama **tripla**, e migliaia di triple messe
insieme formano un grafo in cui i nodi sono cose (persone, luoghi, film,
proteine, prodotti) e gli archi sono fatti. Le cose, d'ora in poi, si chiamano
**entità**: è la parola che userà il resto della sezione, e vuol dire soltanto
questo. Nodi di specie diverse, archi di specie diverse: un grafo
**eterogeneo**.

C'è poi una differenza che sembra filosofica e invece decide come si progetta
tutto il resto. In un knowledge graph, un arco che non c'è vuol dire **«non lo
so»**, non «è falso». Nessuno ha scritto tutti i fatti veri del mondo, e
nessuno mai lo farà: l'assenza di (Torino, gemellata-con, Salt Lake City) non è
una smentita, è un silenzio.

Sembra un dettaglio da logici, e invece è il motivo per cui, poco più avanti,
non si potrà addestrare un modello nel modo consueto. Per imparare a distinguere
il vero dal falso a un modello servono esempi delle due specie; qui gli esempi
di fatti veri abbondano, e di fatti falsi non ce n'è nemmeno uno, perché
nessuno si mette a scrivere le cose che non sono successe.

`````

`````{tab} Superiore

Un knowledge graph è un insieme di triple
$\mathcal{G} \subseteq \mathcal{E} \times \mathcal{R} \times \mathcal{E}$, dove
$\mathcal{E}$ sono le entità e $\mathcal{R}$ i tipi di relazione. È un
**multigrafo diretto etichettato**: fra due entità possono correre più archi con
relazioni diverse, e la direzione conta ($r$ e la sua inversa sono relazioni
distinte).

Sopra le triple sta di solito uno **schema** (o ontologia): una gerarchia di
tipi (`Città` è un `LuogoAbitato` è un `Luogo`) e i vincoli di dominio e
codominio di ogni relazione (`sindaco-di` va da una `Persona` a un
`LuogoAbitato`). Lo schema serve a due cose molto pratiche: validare ciò che
entra e permettere l'inferenza per ereditarietà, cioè dedurre triple non scritte
da quelle scritte.

La proprietà semantica decisiva è l'**assunzione di mondo aperto**: la
mancanza di una tripla non è la sua negazione. Ne discende che il problema
naturale su un knowledge graph, la **link prediction**, non è una
classificazione binaria ordinaria: dispone di soli esempi positivi, e gli
insiemi di addestramento e di valutazione vanno costruiti di conseguenza.

`````

## Costruirlo è il lavoro

Il grafo non arriva già fatto, e va detto con chiarezza che questa è la parte
grossa: rispetto a costruirlo, i modelli che ci girano sopra sono la parte
facile e divertente. Il percorso da un mucchio di testi a un grafo di fatti
passa per quattro gradini, e il libro ha già affrontato il primo.

Il primo gradino è **trovare i nomi**: individuare nel testo i pezzi che
nominano una cosa. Si chiama **riconoscimento delle entità nominate**, ed è la
sezione «POS tagging ed entità» del capitolo sul linguaggio.

Il secondo è capire *quale* cosa. Trovato «Torino» in una frase non si sa
ancora se sia la città, la squadra, il comune omonimo in un altro paese o la
persona con quel cognome; a deciderlo è il resto della frase, perché «Torino ha
battuto la Juve» e «Torino è bagnata dal Po» parlano di due nodi diversi.
Agganciare un nome al nodo giusto si chiama **collegamento delle entità**.

Il terzo, e in pratica il più costoso, è il gemello del secondo: capire che
«F.C. Juventus», «Juventus Football Club» e «la Juve» sono un nodo solo, e che
due schede prodotto con nomi diversi descrivono lo stesso oggetto. Si chiama
**risoluzione delle entità**, è il problema di togliere i doppioni su una scala
enorme, ed è la voce su cui chi mantiene un knowledge graph spende gran parte
del proprio lavoro.

L'ultimo gradino è l'**estrazione di relazioni**: dedurre dal testo che fra due
entità esiste un certo legame. Oggi si fa in larga parte chiedendolo a un
grande modello di linguaggio, con tutti i problemi di verifica che il capitolo
sui Transformer ha già discusso: un modello che inventa una tripla plausibile e
falsa la inserisce nel grafo con la stessa faccia con cui inserisce quelle
vere.

## Entità come punti, relazioni come frecce

Un grafo di fatti si può interrogare come un database, e per molte domande è la
cosa giusta. Ma per prevedere i fatti **mancanti** serve trasformarlo in numeri,
e qui succede una cosa che al lettore di questo libro suonerà familiare.

`````{tab} Elementare

Nel capitolo sul linguaggio si è visto che a ogni parola si può assegnare una
fila di numeri, e che quella fila si può immaginare come un **punto**, come una
città su una mappa: solo che invece di due coordinate ne ha qualche decina.
Parole di significato simile finiscono vicine.

E lì era emersa una proprietà curiosa: hanno un senso anche gli *spostamenti*
da un punto all'altro. Lo spostamento che separa «uomo» da «donna» è più o meno
lo stesso che separa «re» da «regina», tanto che partendo da «re», togliendo lo
spostamento «uomo» e aggiungendo lo spostamento «donna», si atterra vicino a
«regina». Le relazioni di significato, insomma, diventano **frecce** sulla
mappa.

L'idea di base per i knowledge graph è la stessa, presa sul serio e fatta
diventare l'obiettivo dell'addestramento invece di un effetto collaterale.
Ogni entità è un punto nello spazio. Ogni **relazione è una freccia**, sempre
la stessa per tutte le coppie che quella relazione lega. Si chiede che, per
ogni fatto vero, partire dal soggetto e seguire la freccia della relazione
faccia arrivare vicino all'oggetto: dal punto «Roma», seguendo la freccia
«capitale-di», si deve atterrare vicino al punto «Italia»; la stessa freccia,
da «Parigi», deve portare vicino a «Francia».

Fatto questo, prevedere un fatto mancante diventa un calcolo: prendi
«Lisbona», applica la freccia «capitale-di», guarda quale entità è più vicina
al punto in cui sei arrivato.

Qui si paga il debito lasciato in sospeso all'inizio, quello degli esempi
falsi che non esistono. Per sistemare punti e frecce bisogna poter dire al
modello «questo sì e quest'altro no», e i «no» non ce li ha nessuno. La
soluzione è tanto sfacciata quanto efficace: **ce li fabbrichiamo guastando i
fatti veri**. Si prende (Roma, capitale-di, Italia), si sostituisce una delle
due estremità con un'entità pescata a caso, e viene fuori (Roma, capitale-di,
Portogallo), che quasi certamente è falsa. Chiediamo allora che il fatto vero
finisca più vicino del suo gemello guastato, e tanto basta.

Restano però due problemi, e sono geometrici prima ancora che informatici,
perché nascono da com'è fatta una freccia. Il primo: una freccia porta da un
punto a **un solo** punto. Ma «ha-recitato-in» lega un attore a decine di film:
la stessa freccia dovrebbe arrivare in decine di posti diversi, e non può.

Il secondo è più insidioso, e riguarda le relazioni che si ereditano lungo la
catena. Se sei antenato di mio nonno, sei antenato anche mio; se un pezzo è
parte di un motore e il motore è parte di un'automobile, quel pezzo è parte
dell'automobile. Qui la stessa freccia deve valere tanto per un passo quanto
per due, e nessuna freccia lo fa, tranne una: prova con una freccia che sposta
di tre, e farla due volte sposta di sei, che non è tre. L'unico numero per cui
due passi e un passo portano nello stesso posto è **zero**, cioè la freccia che
non sposta niente. La relazione si annulla, e con lei ogni possibilità di
prevederla.

Dal primo problema è nata una lunga discendenza di modelli, che sostituiscono
la freccia con qualcosa di più flessibile (una moltiplicazione, una rotazione),
e ognuno rimedia a un caso e ne rompe un altro. Il secondo problema non lo
risolve nessuno di loro. Per le relazioni che si ereditano servono modelli in
cui un'entità non è un punto ma una regione, capace di **contenerne** un'altra,
il che è un modo molto più naturale di dire «è un caso particolare di».

C'è infine una via del tutto diversa, ed è quella che questo capitolo ha
costruito per intero: portare il **passaparola** delle sezioni precedenti su
questo grafo. La difficoltà nuova è che qui gli archi non sono tutti uguali, e
un bigliettino che arriva lungo un «è nato a» non va letto come uno che arriva
lungo un «ha diretto». La risposta è semplice: una ricetta di riscrittura
diversa per ogni tipo di arco. Il vantaggio rispetto ai punti e alle frecce è
lo stesso di tutto il capitolo, cioè che la fila di numeri di un'entità non è
più imparata a memoria, si **calcola** da quel che le sta intorno.

`````

`````{tab} Superiore

Un avviso di notazione, perché qui il capitolo cambia alfabeto: in questa
sezione $\mathbf{h}$, $\mathbf{r}$ e $\mathbf{t}$ sono la **testa**, la
**relazione** e la **coda** di una tripla, non gli stati nascosti
$\mathbf{h}_v^{(k)}$ delle sezioni sul message passing.

**TransE** {cite}`bordes2013translating` rappresenta ogni entità con un vettore
$\mathbf{e} \in \mathbb{R}^d$ e ogni relazione con un vettore
$\mathbf{r} \in \mathbb{R}^d$ interpretato come traslazione, e chiede che per
ogni tripla vera $(h, r, t)$ valga

$$
\mathbf{h} + \mathbf{r} \approx \mathbf{t} .
$$

La funzione di punteggio è la distanza, $f(h,r,t) = -\lVert \mathbf{h} +
\mathbf{r} - \mathbf{t} \rVert$, e si addestra con una *margin ranking loss*
che chiede alle triple vere di stare a distanza minore delle triple false di
almeno un margine $\gamma$:

$$
\mathcal{L} = \sum_{(h,r,t) \in \mathcal{G}} \; \sum_{(h',r,t') \in S'_{(h,r,t)}}
\big[\, \gamma - f(h,r,t) + f(h',r,t') \,\big]_+ .
$$

Le triple false non esistono in natura, per l'assunzione di mondo aperto: si
**fabbricano corrompendo** quelle vere, ed è per questo che l'insieme dei
negativi $S'_{(h,r,t)}$ porta in pedice la tripla positiva da cui nasce, invece
di essere un unico insieme globale: si sostituisce **una sola** delle due
estremità con un'entità pescata a caso, mai tutt'e due. È l'equivalente, per i
grafi, del *negative sampling* di word2vec {cite}`mikolov2013distributed`, e la
parentela non è casuale: entrambi trasformano un problema con soli positivi in
un problema di discriminazione.

C'è poi un vincolo che sembra implementativo e non lo è: gli embedding delle
entità vanno rinormalizzati a $\lVert \mathbf{e} \rVert_2 = 1$, e nell'algoritmo
del paper la riga sta all'inizio di **ogni iterazione**, non a fine
addestramento. Senza, il modello ha una scappatoia banale, cioè far crescere le
norme finché la loss scende senza che nessuna relazione sia stata imparata.

I limiti di una traslazione sono espressivi, non implementativi. Le relazioni
**uno-a-molti** e **molti-a-uno** non sono rappresentabili: se $(h, r, t_1)$ e
$(h, r, t_2)$ sono entrambe vere, TransE forza
$\mathbf{t}_1 \approx \mathbf{t}_2$, cioè fa collassare entità distinte. Le
relazioni **simmetriche** ($r(a,b) \Leftrightarrow r(b,a)$) richiedono
$\mathbf{r} \approx -\mathbf{r}$, cioè $\mathbf{r} \approx \mathbf{0}$, e con
la relazione annullata collassano anche le due entità. Le relazioni
**riflessive** finiscono allo stesso modo.

Il caso che manca chiede una precisazione, perché è il punto in cui il racconto
corrente sbaglia bersaglio. Una traslazione **compone** benissimo: se dalla
coppia $r_1(a,b)$ e $r_2(b,c)$ deve seguire $r_3(a,c)$, basta porre
$\mathbf{r}_3 = \mathbf{r}_1 + \mathbf{r}_2$, ed è per questo che nella
tassonomia diventata standard con RotatE (Sun e colleghi, 2019) TransE è dato
capace di composizione, di inversione e di antisimmetria, e incapace della sola
simmetria. Quel che non regge è il caso particolare in cui le tre relazioni
sono **la stessa**, cioè la **transitività** (`antenato-di`, `parte-di`,
`sottoclasse-di`: la norma in qualunque grafo con un'ontologia). Lì servirebbero
insieme $\mathbf{a} + 2\mathbf{r} \approx \mathbf{c}$ e
$\mathbf{a} + \mathbf{r} \approx \mathbf{c}$, cioè ancora una volta
$\mathbf{r} \approx \mathbf{0}$.

Il seguito della famiglia sistema altre caselle, e conviene dire quali, perché
non è la transitività. I modelli **bilineari** come DistMult sono simmetrici
per costruzione, e quindi perdono l'antisimmetria che TransE aveva; la loro
estensione ai numeri complessi, **ComplEx**, la recupera ma perde la
composizione; **RotatE** sostituisce la traslazione con una rotazione nel piano
complesso e le tiene insieme tutte e quattro. La transitività però resta fuori
anche di lì, per lo stesso motivo algebrico: se una rotazione applicata due
volte deve dare sé stessa, e ha modulo uno, allora è l'identità, e la relazione
torna a non spostare niente. A reggere le gerarchie servono famiglie di altro
tipo, che rappresentano un'entità non come un punto ma come un oggetto capace
di **contenerne** un altro (ordini parziali, scatole, spazi iperbolici).

Poi c'è la via che questo capitolo ha costruito. **R-GCN**
{cite}`schlichtkrull2018modeling` porta il message passing sui grafi
eterogenei con una mossa diretta: una matrice di pesi **per ogni tipo di
relazione**,

$$
\mathbf{h}_v^{(l+1)} = \sigma\!\Big( \mathbf{W}_0^{(l)} \mathbf{h}_v^{(l)} +
\sum_{r \in \mathcal{R}} \sum_{u \in \mathcal{N}_v^{r}}
\frac{1}{c_{v,r}} \mathbf{W}_r^{(l)} \mathbf{h}_u^{(l)} \Big),
$$

dove $\mathcal{N}_v^{r}$ sono i vicini di $v$ raggiunti da archi di tipo $r$ e
$c_{v,r}$ è una normalizzazione (tipicamente $|\mathcal{N}_v^{r}|$). Il
problema evidente è il numero di parametri, che cresce con il numero di
relazioni: un grafo con mille tipi di arco vorrebbe mille matrici. Si controlla
imponendo che le $\mathbf{W}_r$ siano combinazioni di poche matrici di base
condivise, il che è una forma di condivisione dei pesi fra relazioni simili. La
differenza rispetto a TransE è che qui l'embedding di un'entità **si calcola**
dal suo vicinato invece di essere una riga di tabella: è la stessa differenza
fra DeepWalk e le GNN vista all'inizio del capitolo. Il vantaggio
dell'induttività, però, arriva solo se i nodi portano feature proprie da cui
partire: nel paper originale le entità non ne hanno, lo stato iniziale è a sua
volta un embedding appreso per ciascuna entità, e senza quella riga di tabella
un'entità mai vista resta fuori, esattamente come in TransE.

`````

## Rispondere navigando

Riempire da sé i buchi che ha, come si è appena visto con Lisbona, è già
qualcosa. Ma a che cosa serve, un grafo di fatti, quando la domanda arriva da
fuori? A tre cose, e sono tre cose che un archivio di documenti non sa fare.

La prima è **comporre**. Se il grafo contiene «il regista di questo film è X» e
«X è nato in questa città», la domanda «in che città è nato il regista di
questo film» si risponde percorrendo due archi. Il modo usuale di rispondere a
una domanda su un archivio di testi è invece cercare i **brani più somiglianti
alla domanda** e darli in pasto a un modello di linguaggio: è il **retrieval
denso** del capitolo sui Transformer, dove i brani non si confrontano parola
per parola, ma trasformando ciascuno in una fila di numeri e cercando le file
più vicine. Se nessun documento contiene entrambi i fatti nella stessa frase,
quel sistema non li mette insieme: non gli è stato chiesto di ragionare, gli è
stato chiesto di somigliare.

Il secondo vantaggio è che **il cammino è la spiegazione**. Una ricerca per
somiglianza restituisce tre paragrafi e una risposta, e per verificarla bisogna
leggere i paragrafi. Una risposta ottenuta navigando restituisce la catena di
fatti che l'ha prodotta, e ogni anello si può controllare da solo. In un
dominio dove sbagliare costa (clinico, legale, finanziario) è una differenza di
natura.

Il terzo vantaggio si dimentica spesso ed è forse il più pratico: le domande
che chiedono di **contare**. «Quanti registi italiani hanno girato almeno tre
film ambientati a Napoli» non è una domanda a cui un modello di linguaggio
possa rispondere in modo affidabile, e non è nemmeno una domanda di
somiglianza: è un'interrogazione da fare a un archivio ordinato, e vuole una
struttura su cui contare davvero.

Messi insieme, i tre vantaggi hanno prodotto un'idea che gira parecchio.
Dare a un modello di linguaggio dei documenti pescati sul momento, invece di
fidarsi di quel che ricorda, è la tecnica che i capitoli sui Transformer e
sugli agenti chiamano **RAG**, dalle iniziali di *Retrieval-Augmented
Generation*, «generazione con recupero». Da qui l'idea di fare la stessa cosa
con un grafo, e il nome che ne è venuto fuori è **GraphRAG**: invece di andare
a prendere dei brani di testo si va a prendere un **sottografo**, cioè il
pezzetto di grafo attorno alle cose nominate nella domanda, e si mette quello
davanti al modello insieme alla domanda. Le varianti differiscono per come si
sceglie il pezzetto e per come lo si riscrive in frasi (un grafo va disteso in
una fila di parole prima di poterlo dare a un modello che legge testo), ma il
principio è quello.

## Quando conviene, e quando no

L'onestà dovuta, perché su questo tema si sente molto entusiasmo.

Un knowledge graph è **caro da costruire e caro da tenere aggiornato**. Ogni
fatto del mondo che cambia è un arco da correggere, e un grafo non manutenuto
invecchia peggio di un archivio di documenti, perché sembra ancora autorevole
mentre è già falso. La domanda da farsi prima di cominciare non è se sarebbe
utile, ma chi lo aggiornerà fra due anni.

I grandi modelli di linguaggio, inoltre, hanno assorbito buona parte del
mestiere che si affidava ai grafi di fatti: molte domande fattuali ricevono
oggi una risposta corretta senza che nessun grafo sia stato consultato. Quello
che i modelli non danno, e che resta la ragione durevole di questa struttura, è
di altro tipo. Il cammino che si può **verificare** anello per anello. La
possibilità di **contare**. E la possibilità di dichiarare delle regole che il
sistema non ha il permesso di violare: che il sindaco di un posto debba essere
una persona e non un'altra città, che nessuno possa essere nato dopo essere
morto. Sono garanzie, non conoscenza, ed è per le garanzie che si paga il
prezzo di costruirlo.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un **knowledge graph** è un grafo di **fatti**: i nodi sono cose (persone,
  luoghi, film, prodotti) e ogni arco porta un'etichetta che è un verbo. L'unità
  minima è una frasetta di tre parole, la **tripla**: soggetto, relazione,
  oggetto. Nodi di specie diverse e archi di specie diverse: è un grafo
  **eterogeneo**, mentre quelli visti finora avevano nodi tutti della stessa
  specie.
- Un arco che manca vuol dire **«non lo so»**, non «è falso»: nessuno ha mai
  scritto tutti i fatti veri del mondo. Ne segue una conseguenza pratica
  fastidiosa: di esempi sbagliati non ce ne sono, e per addestrare un modello
  bisogna fabbricarseli **guastando** i fatti veri, cioè sostituendo una delle
  due estremità con una cosa pescata a caso.
- Il lavoro vero è **costruirlo**: trovare i nomi nel testo, capire di quale
  Torino si parla, accorgersi che «la Juve» e «Juventus Football Club» sono lo
  stesso nodo, ed estrarre dalle frasi i legami fra le cose. I modelli che ci
  girano sopra sono la parte facile.
- Il modo più semplice di metterlo in numeri è fare di ogni cosa un **punto** e
  di ogni relazione una **freccia** sempre uguale: da «Roma», seguendo la
  freccia «capitale-di», si atterra vicino a «Italia». Funziona, ma una freccia
  porta in **un solo** punto, e quindi non può legare un attore a dieci film né
  reggere le relazioni che si ereditano lungo la catena (se sei antenato di mio
  nonno sei antenato mio): là la freccia dovrebbe valere sia un passo sia due,
  e l'unica freccia che lo fa è quella lunga zero. Da lì una lunga discendenza
  di modelli che sostituiscono la freccia con qualcosa di più flessibile, senza
  però che nessuno di loro risolva proprio quest'ultimo caso.
- L'altra via è portare il **passaparola** delle sezioni precedenti su questo
  grafo, usando una ricetta di riscrittura diversa per ogni tipo di arco: così
  la fila di numeri di un'entità non si impara a memoria, si calcola da quel
  che le sta intorno.
- Il vantaggio che resta, e per cui vale la pena pagare la manutenzione, non è
  sapere i fatti (per quello ci sono i modelli di linguaggio): è **mettere
  insieme** più fatti in catena, mostrare il **percorso** che ha prodotto la
  risposta perché sia verificabile, rispondere a domande che chiedono di
  **contare**, e poter dichiarare **regole** che il sistema non può violare.
  E un grafo non aggiornato è peggio di nessun grafo, perché sembra ancora
  autorevole quando è già falso.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un **knowledge graph** è un multigrafo diretto etichettato di **triple**
  (soggetto, relazione, oggetto): nodi ed archi di tipi diversi, cioè un grafo
  **eterogeneo**, a differenza di tutti quelli visti finora nel capitolo.
- Vale l'**assunzione di mondo aperto**: un arco che manca vuol dire «non lo
  so», non «è falso». Da qui il fatto che gli esempi negativi non esistano e si
  debbano fabbricare **corrompendo** le triple vere.
- Costruirlo è il lavoro: riconoscimento delle entità, **collegamento** (quale
  Torino?), **risoluzione** (Juventus e la Juve sono un nodo solo), estrazione
  di relazioni. La parte modellistica viene dopo, ed è la più facile.
- **TransE** {cite}`bordes2013translating` fa delle relazioni delle
  **traslazioni** ($\mathbf{h}+\mathbf{r}\approx\mathbf{t}$), cioè prende sul
  serio l'aritmetica delle analogie del capitolo sul linguaggio. Non regge le
  relazioni uno-a-molti, le simmetriche e le riflessive; la **composizione**
  invece la regge ($\mathbf{r}_3 = \mathbf{r}_1 + \mathbf{r}_2$), ed è il suo caso
  particolare, la **transitività**, a forzare $\mathbf{r} \approx \mathbf{0}$.
  Da lì la discendenza (DistMult perde l'antisimmetria, ComplEx la recupera e
  perde la composizione, RotatE le tiene tutte), che però la transitività non
  la risolve: per le gerarchie servono rappresentazioni che contengono invece
  di spostare (ordini, scatole, spazi iperbolici).
- **R-GCN** {cite}`schlichtkrull2018modeling` porta il message passing sul
  grafo eterogeneo con una matrice di pesi per tipo di relazione, controllata
  con matrici di base condivise per non esplodere in parametri.
- Il vantaggio durevole non è sapere i fatti (per quello ci sono gli LLM), è
  **comporre** più fatti, esibire il **cammino** come spiegazione e rispondere
  a domande **aggregate**. Il prezzo è la manutenzione, e un grafo non
  aggiornato è peggio di nessun grafo perché sembra ancora autorevole.
```

`````
