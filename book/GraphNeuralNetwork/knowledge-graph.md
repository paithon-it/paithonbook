# Fatti come archi: i knowledge graph

Nel maggio del 2012 Google annuncia una modifica al motore di ricerca con uno
slogan che vale più della modifica: *things, not strings*, cose e non stringhe.
Fino a quel momento cercare «Torino» significava chiedere le pagine che
contengono quella sequenza di sei caratteri. Da lì in avanti il motore prova a
sapere che Torino è una **città**, che sta in Italia, che ha un fiume, un
sindaco e una squadra di calcio, e che «Torino» può anche essere quella
squadra.

L'idea non era nuova. Le reti semantiche degli anni Sessanta, gli enormi
progetti di senso comune scritti a mano a partire dagli anni Ottanta e tutta la
tradizione del web semantico avevano già proposto la stessa struttura. Nuovo
era che, per la prima volta, un grafo di fatti abbastanza grande da servire a
qualcosa si poteva costruire in modo automatico.

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
facile e divertente. Il percorso da un corpus di testo a un grafo di fatti
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
stesso oggetto. È un problema di deduplicazione su scala, e le aziende che
mantengono knowledge graph ci spendono la maggior parte dello sforzo.

L'ultimo gradino è l'**estrazione di relazioni**: dedurre dal testo che fra due
entità esiste un certo legame. Oggi si fa in larga parte chiedendolo a un
modello di linguaggio, con tutti i problemi di verifica che il capitolo sugli
LLM ha già discusso: un modello che inventa una tripla plausibile e falsa la
inserisce nel grafo con la stessa faccia con cui inserisce quelle vere.

## Entità come vettori, relazioni come traslazioni

Un grafo di fatti si può interrogare come un database, e per molte domande è la
cosa giusta. Ma per prevedere i fatti **mancanti** serve trasformarlo in numeri,
e qui succede una cosa che al lettore di questo libro suonerà familiare.

`````{tab} Elementare

Nel capitolo sul linguaggio abbiamo visto una proprietà curiosa degli
embedding di parole: le relazioni di significato diventano **direzioni** nello
spazio, e si sommano come frecce. «Re meno uomo più donna» finiva vicino a
«regina», perché la freccia che porta dal maschile al femminile è più o meno la
stessa in tutte le coppie.

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
\mathcal{L} = \sum_{(h,r,t) \in \mathcal{G}} \; \sum_{(h',r,t') \in \mathcal{G}^-}
\big[\, \gamma - f(h,r,t) + f(h',r,t') \,\big]_+ .
$$

Le triple false $\mathcal{G}^-$ non esistono in natura, per l'assunzione di
mondo aperto: si **fabbricano corrompendo** quelle vere, cioè sostituendo la
testa o la coda con un'entità pescata a caso. È l'equivalente, per i grafi, del
*negative sampling* di word2vec {cite}`mikolov2013efficient`, e la parentela
non è casuale: entrambi trasformano un problema con soli positivi in un
problema di discriminazione.

I limiti di una traslazione sono espressivi, non implementativi, e si elencano
in tre righe. Le relazioni **uno-a-molti** e **molti-a-uno** non sono
rappresentabili: se $(h, r, t_1)$ e $(h, r, t_2)$ sono entrambe vere, TransE
forza $\mathbf{t}_1 \approx \mathbf{t}_2$, cioè fa collassare entità distinte.
Le relazioni **simmetriche** ($r(a,b) \Leftrightarrow r(b,a)$) richiedono
$\mathbf{r} \approx -\mathbf{r}$, cioè $\mathbf{r} \approx \mathbf{0}$. Le
relazioni **riflessive** collassano tutto allo stesso modo. Da qui la
discendenza: modelli bilineari come DistMult, la sua estensione ai numeri
complessi ComplEx (che recupera l'antisimmetria), e le rotazioni di RotatE, che
sostituiscono la traslazione con una rotazione nel piano complesso e catturano
simmetria, antisimmetria e composizione.

Poi c'è la via che questo capitolo ha costruito. **R-GCN**
{cite}`schlichtkrull2018modeling` porta il message passing sui grafi
eterogenei con una mossa diretta: una matrice di pesi **per ogni tipo di
relazione**,

$$
h_v^{(l+1)} = \sigma\!\Big( W_0^{(l)} h_v^{(l)} +
\sum_{r \in \mathcal{R}} \sum_{u \in \mathcal{N}_v^{r}}
\frac{1}{c_{v,r}} W_r^{(l)} h_u^{(l)} \Big),
$$

dove $\mathcal{N}_v^{r}$ sono i vicini di $v$ raggiunti da archi di tipo $r$ e
$c_{v,r}$ è una normalizzazione (tipicamente $|\mathcal{N}_v^{r}|$). Il
problema evidente è il numero di parametri, che cresce con il numero di
relazioni: un grafo con mille tipi di arco vorrebbe mille matrici. Si controlla
imponendo che le $W_r$ siano combinazioni di poche matrici di base condivise, il
che è una forma di condivisione dei pesi fra relazioni simili. La differenza
rispetto a TransE è che qui l'embedding di un'entità **si calcola** dal suo
vicinato invece di essere una riga di tabella: è la stessa differenza fra
DeepWalk e le GNN vista all'inizio del capitolo, e porta con sé lo stesso
vantaggio, cioè l'induttività.

`````

## Rispondere navigando

A che serve, in concreto, oltre a completarsi da sé.

La cosa che un grafo di fatti fa e un archivio di testi non fa è **comporre**.
Se il grafo contiene «il regista di questo film è X» e «X è nato in questa
città», la domanda «in che città è nato il regista di questo film» si risponde
percorrendo due archi. Un sistema di recupero denso, come quello del capitolo
sui Transformer, cerca passaggi simili alla domanda: se nessun documento
contiene entrambi i fatti nella stessa frase, non li mette insieme, perché non
gli è stato chiesto di ragionare ma di somigliare.

Il secondo vantaggio è che **il cammino è la spiegazione**. Un recupero denso
restituisce tre paragrafi e una risposta, e per verificarla bisogna leggere i
paragrafi. Una risposta ottenuta navigando restituisce la catena di fatti che
l'ha prodotta, e ogni anello si può controllare da solo. In un dominio dove
sbagliare costa (clinico, legale, finanziario) è una differenza di natura.

È da qui che nasce l'idea di combinare le due cose, che va sotto il nome
generico di **GraphRAG**: invece di recuperare passaggi, si recupera un
**sottografo** attorno alle entità nominate nella domanda, e lo si passa al
modello come contesto. Le varianti differiscono per come si sceglie il
sottografo e per come lo si linearizza in testo, ma il principio è quello, e
si innesta esattamente sul RAG avanzato del capitolo sugli agenti.

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
  uno-a-molti né quelle simmetriche, e da lì la discendenza (DistMult, ComplEx,
  RotatE).
- **R-GCN** {cite}`schlichtkrull2018modeling` porta il message passing sul
  grafo eterogeneo con una matrice di pesi per tipo di relazione, controllata
  con matrici di base condivise per non esplodere in parametri.
- Il vantaggio durevole non è sapere i fatti (per quello ci sono gli LLM), è
  **comporre** più fatti, esibire il **cammino** come spiegazione e rispondere
  a domande **aggregate**. Il prezzo è la manutenzione, e un grafo non
  aggiornato è peggio di nessun grafo perché sembra ancora autorevole.
```
