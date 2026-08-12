# Fatti come archi: i knowledge graph

Nel maggio del 2012 Google annuncia una modifica al motore di ricerca con uno
slogan che vale più della modifica: *things, not strings*, cose e non stringhe.
Fino a quel momento cercare «Torino» significava chiedere le pagine che
contengono quella sequenza di sei caratteri. Da lì in avanti il motore prova a
sapere che Torino è una **città**, che sta in Italia, che ha un fiume, un
sindaco e una squadra di calcio, e che «Torino» può anche essere quella
squadra.

L'idea non era nuova. Le **reti semantiche** degli anni Sessanta (schemi in cui
i concetti sono puntini e le linee fra loro dicono «è un», «ha un», «si trova
in»), gli enormi progetti di senso comune scritti a mano a partire dagli anni
Ottanta e tutta la tradizione del **web semantico**, che voleva pagine
leggibili non solo dalle persone ma anche dai programmi, avevano già proposto
la stessa struttura. Nuovo era che, per la prima volta, un grafo di fatti
abbastanza grande da servire a qualcosa si poteva costruire in modo
automatico.

Fin qui il capitolo ha trattato grafi in cui tutti i nodi sono la stessa specie
di cosa e tutti gli archi vogliono dire la stessa cosa: utenti, atomi,
articoli. Questa sezione toglie quella comodità.

## Una tripla è un fatto

`````{tab} Elementare

Un **knowledge graph** è un grafo in cui gli archi hanno un'etichetta, e
l'etichetta è un verbo. L'unità elementare è una frasetta di tre parole:

> (Torino, si-trova-in, Piemonte) · (Torino, attraversata-da, Po) ·
> (Po, sfocia-in, Adriatico)

Soggetto, relazione, oggetto. Si chiama **tripla**, e migliaia di triple messe
insieme formano un grafo in cui i nodi sono cose (persone, luoghi, film,
proteine, prodotti) e gli archi sono fatti. Nodi di specie diverse, archi di
specie diverse: un grafo **eterogeneo**.

C'è una differenza che sembra filosofica e invece decide come si progetta tutto
il resto. In una tabella, una casella vuota di solito vuol dire «no». In un
knowledge graph, un arco che non c'è vuol dire **«non lo so»**. Nessuno ha
scritto tutti i fatti veri del mondo, e nessuno mai lo farà: l'assenza di
(Torino, gemellata-con, Salt Lake City) non è una smentita, è un silenzio.

Sembra un dettaglio da logici, e invece è il motivo per cui, poco più avanti,
non si potrà addestrare un modello nel modo ovvio: non ci sono esempi negativi
da nessuna parte.

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
passa per tre gradini, e il libro ha già affrontato il primo.

Il **riconoscimento delle entità nominate** individua nel testo i pezzi che
nominano una cosa, ed è la sezione sull'etichettatura di sequenze del capitolo
di NLP. Trovato «Torino» in una frase, però, non si sa ancora *quale* Torino
sia: la città, la squadra, il comune omonimo in un altro paese, la persona con
quel cognome. Risolverlo si chiama **collegamento delle entità**, e comporta
decidere a quale nodo del grafo un nome si riferisce, usando il contesto.

Il gradino gemello, e in pratica il più costoso, è la **risoluzione delle
entità**: capire che «F.C. Juventus», «Juventus Football Club» e «la Juve»
sono un nodo solo, e che due schede prodotto con nomi diversi descrivono lo
stesso oggetto. È il problema di togliere i doppioni, su una scala enorme, e le
aziende che mantengono knowledge graph ci spendono la maggior parte dello
sforzo.

L'ultimo gradino è l'**estrazione di relazioni**: dedurre dal testo che fra due
entità esiste un certo legame. Oggi si fa in larga parte chiedendolo a un
grande modello di linguaggio, con tutti i problemi di verifica che il capitolo
dedicato ha già discusso: un modello che inventa una tripla plausibile e falsa
la inserisce nel grafo con la stessa faccia con cui inserisce quelle vere.

## Entità come punti, relazioni come frecce

Un grafo di fatti si può interrogare come un database, e per molte domande è la
cosa giusta. Ma per prevedere i fatti **mancanti** serve trasformarlo in numeri,
e qui succede una cosa che al lettore di questo libro suonerà familiare.

`````{tab} Elementare

Nel capitolo sul linguaggio si è visto che a ogni parola si può assegnare una
fila di numeri, e che quella fila si può immaginare come un **punto**, come una
città su una mappa: solo che invece di due coordinate ne ha qualche decina.
Parole di significato simile finiscono vicine. E lì era emersa una proprietà
curiosa: anche gli spostamenti da un punto all'altro hanno un senso. Lo
spostamento che porta da «re» a «regina» è più o meno lo stesso che porta da
«attore» ad «attrice», tanto che partendo da «re», annullando lo spostamento
del maschile e applicando quello del femminile, si arriva vicino a «regina».
Le relazioni di significato, insomma, diventano **frecce** sulla mappa.

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

Resta un problema, ed è geometrico prima ancora che informatico. Una freccia
porta da un punto a **un solo** punto. Ma «ha-recitato-in» lega un attore a
decine di film: la stessa freccia dovrebbe arrivare in decine di posti diversi,
e non può. È il difetto che ha generato una lunga discendenza di modelli.

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
entità vanno rinormalizzati a $\lVert \mathbf{e} \rVert_2 = 1$ a ogni epoca.
Senza, il modello ha una scappatoia banale, cioè far crescere le norme finché
la loss scende senza che nessuna relazione sia stata imparata.

I limiti di una traslazione sono espressivi, non implementativi, e si elencano
in quattro righe. Le relazioni **uno-a-molti** e **molti-a-uno** non sono
rappresentabili: se $(h, r, t_1)$ e $(h, r, t_2)$ sono entrambe vere, TransE
forza $\mathbf{t}_1 \approx \mathbf{t}_2$, cioè fa collassare entità distinte.
Le relazioni **simmetriche** ($r(a,b) \Leftrightarrow r(b,a)$) richiedono
$\mathbf{r} \approx -\mathbf{r}$, cioè $\mathbf{r} \approx \mathbf{0}$. Le
relazioni **riflessive** collassano tutto allo stesso modo.

La quarta è quella che fallisce più duramente, e va detta perché riguarda
relazioni ordinarissime. Le relazioni **transitive** (`antenato-di`,
`parte-di`, `sottoclasse-di`, cioè la norma in qualunque grafo con
un'ontologia) chiedono che da $(a,r,b)$ e $(b,r,c)$ segua $(a,r,c)$: la
traslazione dovrebbe soddisfare insieme $\mathbf{a} + 2\mathbf{r} \approx
\mathbf{c}$ e $\mathbf{a} + \mathbf{r} \approx \mathbf{c}$, cioè di nuovo
$\mathbf{r} \approx \mathbf{0}$, e stavolta senza nemmeno una via d'uscita
degenere che tenga in piedi la loss. Le prime tre si lasciano approssimare a
un prezzo (l'una-a-molti facendo coincidere due entità, la simmetrica
annullando la relazione); la transitiva no. Ed è per questo che la
**composizione** (che è la transitività vista come «applicare $r$ due volte»)
compare fra i meriti dei modelli venuti dopo. Da qui la discendenza: modelli
bilineari come DistMult, la sua estensione ai numeri complessi ComplEx (che
recupera l'antisimmetria), e le rotazioni di RotatE, che sostituiscono la
traslazione con una rotazione nel piano complesso e catturano simmetria,
antisimmetria e composizione.

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

A che serve, in concreto, oltre a completarsi da sé.

La cosa che un grafo di fatti fa e un archivio di testi non fa è **comporre**.
Se il grafo contiene «il regista di questo film è X» e «X è nato in questa
città», la domanda «in che città è nato il regista di questo film» si risponde
percorrendo due archi. Il modo usuale di rispondere a una domanda su un archivio
di testi è invece cercare i **brani più somiglianti alla domanda** e darli in
pasto a un modello di linguaggio (è il *recupero denso* del capitolo sui
Transformer). Se nessun documento contiene entrambi i fatti nella stessa frase,
quel sistema non li mette insieme: non gli è stato chiesto di ragionare, gli è
stato chiesto di somigliare.

Il secondo vantaggio è che **il cammino è la spiegazione**. Una ricerca per
somiglianza restituisce tre paragrafi e una risposta, e per verificarla bisogna
leggere i paragrafi. Una risposta ottenuta navigando restituisce la catena di fatti che
l'ha prodotta, e ogni anello si può controllare da solo. In un dominio dove
sbagliare costa (clinico, legale, finanziario) è una differenza di natura.

È da qui che nasce l'idea di combinare le due cose, che va sotto il nome
generico di **GraphRAG**: invece di andare a prendere dei brani di testo, si va
a prendere un **sottografo**, cioè il pezzetto di grafo attorno alle cose
nominate nella domanda, e si mette quello davanti al modello di linguaggio
insieme alla domanda. Le varianti differiscono per come si sceglie il pezzetto
e per come lo si riscrive in frasi (un grafo va disteso in una fila di parole
prima di poterlo dare a un modello che legge testo), ma il principio è quello,
e si innesta esattamente sulla tecnica, discussa nel capitolo sugli agenti, di
dare al modello dei documenti pescati sul momento invece di fidarsi di ciò che
ricorda.

Il terzo vantaggio si dimentica spesso ed è forse il più pratico: le domande
**aggregate**. «Quanti registi italiani hanno girato almeno tre film
ambientati a Napoli» non è una domanda a cui un modello di linguaggio possa
rispondere in modo affidabile, e non è una domanda di somiglianza. È una query,
e vuole una struttura su cui contare.

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
di altro tipo: la **verificabilità** del cammino, la possibilità di
**contare** e aggregare, e la capacità di dichiarare **vincoli** che il sistema
non può violare. Sono garanzie, non conoscenza, ed è per le garanzie che si
paga il prezzo di costruirlo.

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
  reggere le relazioni che si concatenano (se sei antenato di mio nonno sei
  antenato mio). Da lì una lunga discendenza di modelli che sostituiscono la
  freccia con qualcosa di più flessibile.
- L'altra via è portare il **passaparola** dei capitoli precedenti su questo
  grafo, usando una ricetta di riscrittura diversa per ogni tipo di arco.
- Il vantaggio che resta, e per cui vale la pena pagare la manutenzione, non è
  sapere i fatti (per quello ci sono i modelli di linguaggio): è **mettere
  insieme** più fatti in catena, mostrare il **percorso** che ha prodotto la
  risposta perché sia verificabile, e rispondere a domande che chiedono di
  **contare**. E un grafo non aggiornato è peggio di nessun grafo, perché
  sembra ancora autorevole quando è già falso.
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
  serio l'aritmetica delle analogie del capitolo di NLP. Non regge le relazioni
  uno-a-molti, le simmetriche e le riflessive, e soprattutto le **transitive**,
  che sono l'unico caso senza nemmeno una soluzione degenere; da lì la
  discendenza (DistMult, ComplEx, RotatE, che recupera la composizione).
- **R-GCN** {cite}`schlichtkrull2018modeling` porta il message passing sul
  grafo eterogeneo con una matrice di pesi per tipo di relazione, controllata
  con matrici di base condivise per non esplodere in parametri.
- Il vantaggio durevole non è sapere i fatti (per quello ci sono gli LLM), è
  **comporre** più fatti, esibire il **cammino** come spiegazione e rispondere
  a domande **aggregate**. Il prezzo è la manutenzione, e un grafo non
  aggiornato è peggio di nessun grafo perché sembra ancora autorevole.
```

`````
