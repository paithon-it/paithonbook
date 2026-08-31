# Context engineering: la finestra come sistema

Se potessimo aprire una di quelle applicazioni che le aziende costruiscono
attorno a un modello, e guardare che cosa gli arriva davvero a ogni richiesta,
troveremmo che quasi nulla di quel testo lo ha battuto una persona sulla
tastiera. C'è il *system prompt*, cioè le istruzioni
di fondo che il programma antepone sempre, quelle del regista visto nella
sezione scorsa. Ci sono alcuni esempi già svolti, tenuti da parte in un
archivio e ripescati perché somigliano al caso di adesso. C'è un pezzo di
manuale, la pagina che serve a questa domanda e non le altre mille, andata a
prendere in quel momento perché il manuale intero nella finestra non ci
starebbe mai. C'è il riassunto degli scambi
precedenti, perché il modello non ricorda nulla da solo. C'è il risultato
dell'ultima operazione che il modello ha chiesto al programma di eseguire per
lui: una ricerca sul web, un calcolo, una domanda a un archivio. E in fondo,
ultima, la breve frase dell'utente.

Quella frase è la punta dell'iceberg; sotto c'è tutto il carico montato dal
programma, il payload di cui si diceva aprendo il capitolo.
Quando Andrej Karpathy, nel 2025, ha dato credito al nome *context
engineering* per il mestiere di montarlo {cite}`karpathy2025context`, ha spostato
l'oggetto del lavoro: non più la singola frase («trovare l'incantesimo
giusto») ma il **governo dell'intero contesto** che riempie la finestra a ogni
passo.

È il tema di questa sezione. Quella precedente ne ha aperto la porta: là
abbiamo lavorato sul singolo messaggio, sulla frase scritta bene; qui saliamo
di un livello e trattiamo la finestra come un **sistema** da progettare nel
suo insieme. Cole Medin, uno che di mestiere costruisce queste applicazioni,
lo dice con uno slogan efficace {cite}`medin2025contextintro`: il context
engineering è un sistema completo (regole, esempi, documentazione, prove di
collaudo) e non «parole magiche», e sta al prompt engineering come una
**sceneggiatura** sta a un *post-it*. Un post-it dice cosa fare in una riga;
una sceneggiatura dà a ogni scena il contesto per recitarla bene. È
un'affermazione di chi costruisce, non un risultato misurato, e la riportiamo
per quello che è.

La meccanica del contesto, cioè come si sceglie che cosa entra in una finestra
che non basta per tutto, come il programma rimonta quel carico a ogni giro e
dove si tengono i ricordi che nella finestra non stanno, è quella del
{doc}`context engineering degli agenti </Agenti/context-engineering>`. Qui si
guarda l'altra metà: **come si pensa** il contesto (un modello mentale a scale
di complessità), **quali mosse** lo governano, **come si guasta**, e **come lo
si rende una procedura ripetibile**.

## Una scala di complessità: dagli atomi agli organi

Il primo passo è avere un'immagine di *quanto* è complesso il contesto che
stiamo montando. David Kim, che di questa materia tiene una raccolta di
appunti molto seguita {cite}`kim2025contextengineering`, propone una metafora
presa in prestito dalla biologia: come la materia vivente sale di
complessità dagli atomi agli organismi, così il contesto sale da una singola
istruzione fino a interi ecosistemi di componenti che collaborano. È una
metafora didattica, un modo di ordinare le idee e non una classificazione
scientifica, e come tale la usiamo.

`````{tab} Elementare

Qualcosa di vivo si costruisce a strati. Alla base ci sono gli **atomi**, i
pezzi più piccoli: una singola regola, «rispondi in italiano». Metti insieme
più atomi e ottieni una **molecola**: l'istruzione più due o tre esempi che
mostrano cosa intendi. Un gradino sopra ci sono le **cellule**, e quello che
si aggiunge è la memoria: il sistema si ricorda chi sei da un messaggio
all'altro. Le cellule si organizzano in **organi**, cioè in lavori a più
passi, dove il modello può chiedere al programma di fare qualcosa per lui
(cercare, calcolare, aprire un archivio) e poi usare il risultato. Ancora più
in su c'è chi propone di trattare tutto il contesto come una cosa sola,
continua, invece che come pezzi distinti: sono idee di ricerca, non attrezzi
da usare oggi, e le nominiamo solo perché nessuno te le venda per tali.

Della biologia non ti serve altro. Quello che serve è l'idea che il contesto
non è tutto uguale, che ce n'è di semplice come un atomo e di complesso come
un corpo, e che sapere a che altezza della scala stai lavorando ti dice quanta
cura serve.

`````

`````{tab} Superiore

La scala, dal basso verso l'alto, si legge come una progressione di ciò che il
contesto deve contenere e coordinare:

| Livello | Metafora | Cosa mette nel contesto |
|---|---|---|
| Atomi | singola istruzione | un vincolo, una direttiva isolata |
| Molecole | *few-shot* | istruzione + esempi svolti (condizionamento) |
| Cellule | memoria / stato | informazione che persiste tra i turni |
| Organi | workflow, strumenti | più passi coordinati, *tool use*, template di ragionamento |
| Sistemi / campi | frontiera | rappresentazioni «a campo» del contesto |

I primi quattro livelli corrispondono a pratiche consolidate: gli esempi
*few-shot* sono lo stesso condizionamento visto nella sezione sul prompt
engineering; la memoria persistente e gli strumenti sono il pane degli agenti.
L'estremità alta della scala è un'altra cosa, e va detto con chiarezza: le
proposte di modellare il contesto come **neural field** o di parlare di
«semantica quantistica» sono **frontiera speculativa** (analogie suggestive,
non risultati consolidati né tecniche con evidenza empirica robusta). Le
includiamo per onestà verso la fonte, ma chi costruisce oggi lavora tra gli
atomi e gli organi; il resto è ricerca aperta, da maneggiare con lo
scetticismo che merita ogni cosa non ancora misurata.

`````

La scala non è solo ordine mentale: dice anche **dove va speso lo sforzo**. Un
compito semplice vive negli atomi e nelle molecole, e lì un buon prompt basta.
Un **agente**, cioè il programma che usa il modello a più riprese e a ogni
ripresa può fargli usare uno strumento, vive negli organi: lì il collo di
bottiglia diventa amministrare quello che entra ed esce dalla finestra a ogni
giro, invece della frase.

## Quattro mosse: scrivere, selezionare, comprimere, isolare

Più si sale di scala, più il contesto va amministrato invece che scritto e
basta. Chi costruisce agenti ha finito per raccogliere tutte le tattiche in
quattro mosse sole, ed è una divisione comoda che si è imposta a partire dalle
note di LangChain, una delle cassette di attrezzi già pronti con cui questi
sistemi si costruiscono {cite}`langchain2025context`. Conviene tenerle a mente
come un piccolo repertorio.

`````{tab} Elementare

La scrivania è minuscola, la stessa vista con gli agenti: sul tavolo ci sta
poca roba, e accanto hai uno schedario grande quanto vuoi. Da qui i quattro
gesti.

**Scrivere**: quello che adesso non ti serve lo metti nello schedario, così
libera il tavolo e non è perso. **Selezionare**: quando ti serve qualcosa, vai
a prendere *solo quella cosa*, non svuoti il cassetto sul tavolo.
**Comprimere**: una pila di appunti lunga la riscrivi in tre righe di sunto,
che occupano molto meno spazio. **Isolare**: se il compito è grosso, lo spezzi
e ne affidi un pezzo a un collega che ha la *sua* scrivania, così la tua non
si intasa. Il collega, qui, è un'altra copia dello stesso modello e non una
persona: con una finestra sua, a cui il programma dà un pezzo di lavoro e da
cui riprende solo il risultato.

Quattro gesti semplici, ripetuti a ogni passo, che tengono la finestra pulita.

`````

`````{tab} Superiore

Le quattro operazioni, in termini di ingegneria del contesto:

- **Write**, persistere informazione *fuori* dalla finestra per riusarla dopo:
  uno *scratchpad* per gli stati intermedi, una memoria esterna per i fatti a
  lungo termine. Esempio: l'agente salva su un file il piano che sta seguendo,
  invece di riportarlo in ogni prompt.
- **Select**, recuperare *dentro* la finestra soltanto ciò che serve al passo
  corrente: per i documenti è il recupero in-context, discendente del **RAG**
  di Lewis e colleghi {cite}`lewis2020retrieval` (che però addestrava insieme
  il **lato query** del retriever e il generatore, tenendo fissi l'encoder dei
  documenti e l'indice, perché riaddestrarlo avrebbe imposto di ricostruire
  l'indice durante il training; qui invece tutti i pesi restano congelati);
  ma è anche il recupero della memoria giusta o della descrizione dello
  strumento giusto.
  Esempio: su una domanda di fatturazione, si iniettano le tre pagine di
  policy pertinenti, non l'intero manuale.
- **Compress**, ridurre i token di ciò che *deve* restare: riassunto
  progressivo della cronologia, potatura delle osservazioni verbose degli
  strumenti. Esempio: dopo venti scambi, la conversazione diventa un sunto di
  poche righe.
- **Isolate**, partizionare il contesto tra ambienti separati: sotto-agenti
  con finestre proprie, sandbox, contesti dedicati. Esempio: un agente
  «ricercatore» e uno «scrittore», ciascuno con il suo contesto, che si
  scambiano solo il risultato.

Le prime tre sono le operazioni che il *context builder* del capitolo sugli
Agenti già esegue sotto il cofano: sono la stessa aritmetica del budget di
token, vista dal lato delle tattiche invece che dal lato del codice. La
quarta, *isolate*, apre verso la progettazione multi-agente, e chiama in causa
il **loop engineering** che vedremo nella prossima sezione: decidere *quando*
delegare a un sotto-contesto è già una scelta sul ciclo, non sul singolo
messaggio.

`````

```{figure} ../figures/context-quattro-mosse.svg
:name: fig-context-quattro-mosse
:alt: "Al centro la finestra di contesto, un riquadro che contiene tre blocchi impilati (istruzioni, cronologia, documenti recuperati) e in fondo lo spazio tratteggiato per la risposta. Quattro frecce numerate: la prima esce verso sinistra, verso un riquadro «memoria esterna», ed è scrivere; la seconda rientra da lì, ed è selezionare; la terza è un arco che esce dal bordo alto e vi rientra, cioè resta dentro la finestra, ed è comprimere; la quarta esce verso destra, verso un secondo riquadro «un'altra finestra», ed è isolare."
:width: 96%

Le quattro mosse, disegnate rispetto al bordo della finestra. Tre spostano
roba oltre quel bordo, una la rimpicciolisce restando dentro, e nessuna delle
quattro aggiunge spazio.
```

Messe una accanto all'altra come in {numref}`fig-context-quattro-mosse`, le
quattro mosse rivelano di essere quattro risposte alla stessa domanda: dove
sta la roba rispetto al bordo della finestra. Fuori e recuperabile (scrivere),
fuori e da riportare dentro un pezzo alla volta (selezionare), dentro ma più
corta (comprimere), fuori e affidata a qualcun altro che ha un bordo suo
(isolare). La finestra resta grande quanto era: quello che cambia è la
disciplina con cui la si riempie.

## Come si guasta un contesto

Un contesto più lungo non è un contesto migliore. Anzi: quasi tutti i modi in
cui le risposte peggiorano via via che si va avanti hanno a che fare con un
contesto che si sporca. Drew Breunig, che di queste applicazioni scrive da
anni, ne ha proposto un catalogo utile {cite}`breunig2025contexts`, che qui
riprendiamo con parole nostre. Quattro guasti ricorrenti:

- **L'avvelenamento** (*context poisoning*): un errore, o una cosa che il
  modello si è inventato di sana pianta (un’**allucinazione**), entra nel
  contesto e ci resta. Da lì in poi il modello la tratta come un fatto
  acquisito e ci costruisce sopra. È il guasto peggiore, perché si alimenta da
  sé.
- **La distrazione** (*context distraction*): il contesto cresce tanto che il
  modello si fissa su ciò che ci legge dentro e trascura quello che sapeva già
  da prima, cioè quello che aveva imparato durante l'addestramento. Si perde
  fra i propri passi passati invece di guardare avanti, e nei casi osservati
  arriva a rifare azioni che aveva già fatto.
- **La confusione** (*context confusion*): informazione inutile ma presente,
  che il modello usa soltanto perché è lì, e che tira la risposta fuori fuoco.
  L'esempio di Breunig riguarda l'elenco degli strumenti fra cui il modello
  deve scegliere per rispondere: un modello piccolo, di quelli che girano su un
  computer normale, messo davanti a quarantasei strumenti prende quello
  sbagliato e la richiesta fallisce; con lo stesso compito e diciannove
  strumenti in elenco sceglie giusto.
- **Il litigio** (*context clash*): pezzi di contesto che si contraddicono fra
  loro, due documenti che si smentiscono, il regolamento vecchio accanto a
  quello nuovo. Il modello non sa a chi credere, e la risposta ne risente.

`````{tab} Elementare

Il più insidioso è il primo, l'avvelenamento, e funziona come una diceria. Basta
che in un gruppo entri una voce falsa («il negozio chiude alle 18») e da quel
momento tutti la ripetono come vera: chi arriva dopo la sente già «confermata»
da tre persone e non la mette in dubbio. Nel contesto succede lo stesso: se al
passo tre il modello «decide» per sbaglio che l'utente si chiama Marco, ai
passi quattro, cinque, sei quel Marco è ormai lì, scritto, e il modello ci
parla come se fosse sempre stato vero. L'errore non resta un errore: diventa
una premessa. È per questo che, con gli agenti, conviene ripulire il contesto
invece di lasciarlo crescere all'infinito.

Da qui il gesto che costa un secondo: quando una conversazione comincia a dire
cose sbagliate, aprine una nuova invece di insistere. Correggere il modello
dentro la stessa chat lascia l'errore dov'è, in mezzo a tutto quello che si è
detto prima, e lui continua a rileggerlo. Una chat nuova parte dal foglio
bianco, ed è l'unico modo che hai, da fuori, di togliere la diceria dal gruppo.

Che «più lungo» non voglia dire «migliore» è stato misurato. In una prova di
Liu e colleghi diventata famosa si dava al modello una domanda e un mucchio di
documenti in cui cercare la risposta, spostando quello giusto ora in cima, ora
in mezzo, ora in fondo. Con venti o trenta documenti, e quello giusto nel
mezzo, il modello rispondeva **peggio** di quando non gliene davano nessuno e
doveva rispondere a memoria. Una finestra più capiente non bastava: gli stessi
modelli, nella versione che ne teneva molto di più, non usavano meglio quello
che ci trovavano dentro. Lo spazio dichiarato non è lo spazio che il modello
sa sfruttare.

`````

`````{tab} Superiore

Per *distraction* e *confusion* una lettura possibile (interpretativa: le
mette insieme chi scrive, non la letteratura) è la **diluizione
dell'attenzione**: man mano che il contesto si allunga, il segnale rilevante
si distribuisce su più token e la capacità del modello di isolarlo cala. Le
evidenze si sovrappongono più di quanto la distinzione dei nomi suggerisca.
Liu e colleghi {cite}`liu2024lost` misurano la **posizione** (la curva di
accuratezza in funzione di dove sta l'informazione ha la forma a U, come visto
nel capitolo sugli Agenti) e insieme la **lunghezza**: sul QA multi-documento
fanno variare il numero di documenti in finestra e trovano che, su
GPT-3.5-Turbo, **nel caso peggiore** (cioè quando il documento rilevante
capita in mezzo) con venti o trenta documenti l'accuratezza scende **sotto**
quella a libro chiuso, cioè sotto il 56,1 per cento che lo stesso modello
ottiene senza alcun documento. Il numero è di un modello solo e di quel
momento, e non va portato in giro come una soglia universale; quello che si
porta in giro è il fatto che la curva, a un certo punto, gira verso il basso.
Aggiungere contesto, oltre una certa soglia, costa più di quanto renda. Dallo
stesso lavoro viene un secondo punto che conviene tenere: i modelli a contesto
esteso **non usano il proprio contesto meglio** di quelli da cui derivano, e
quindi la finestra dichiarata non è la finestra utile. La *distraction* del
catalogo è quest'ultimo effetto visto dal lato pratico, con soglie osservate
attorno alle decine di migliaia di token; *confusion* è invece il caso in cui
token irrilevanti ma presenti attirano indebitamente
l'attenzione. Il *poisoning* è di natura diversa (è un problema di
**veridicità** dello stato, non di posizione) e il *clash* è un problema di
**coerenza** dell'insieme. La lezione operativa è simmetrica alle quattro
mosse: *compress* combatte distraction, *select* combatte confusion, l'igiene
dello stato (rimuovere ciò che si è rivelato falso) combatte poisoning, e la
deduplicazione delle fonti combatte clash.

`````

## Scrivere il contesto una volta sola: il PRP

Un contesto montato bene costa fatica, e quella fatica si può **fare una volta
sola**: invece di rimettere insieme tutto da capo a ogni lavoro, si prepara in
anticipo un foglio con dentro quello che serve, e si consegna quello. L'idea è
di Cole Medin, e nasce per gli assistenti che scrivono codice, ma il gesto
vale anche per chi usa soltanto la chat: un foglio di istruzioni preparato
bene, da incollare all'inizio, fa la stessa cosa. Il nome che gli ha dato è
**PRP**, *Product Requirements Prompt*, cioè il prompt che descrive per intero
che cosa si vuole ottenere.

`````{tab} Elementare

A un falegname puoi dire «fammi un tavolo», oppure puoi consegnargli un
**progetto completo**: le misure, il tipo di legno, la foto di un tavolo che ti
piace, perché il tuo venga di quello stile, e le due pagine del catalogo della
ferramenta con le viti giuste, non il catalogo intero. In fondo al foglio c'è
il dettaglio decisivo, la regola con cui si stabilisce se il tavolo è venuto
bene, «deve stare in piano e reggere 40 chili». Con il progetto in mano
il falegname lavora quasi da solo e sbaglia di meno, perché ha davanti tutto
*prima* di iniziare. E quella regola finale non serve soltanto a te. Quando il
tavolo è pronto, lui lo appoggia, ci carica sopra il peso, e se traballa
ripialla la gamba corta prima di consegnartelo. Il PRP è quel foglio, scritto
per un assistente che programma: regole, esempi, le pagine di manuale che
servono e la prova finale, tutto insieme, pronto da riusare al prossimo
lavoro.

`````

`````{tab} Superiore

Un PRP tipico raccoglie quattro ingredienti: (1) le **regole di progetto** in
un file versionato accanto al codice (`CLAUDE.md`, `AGENTS.md` a seconda
dell'assistente), con convenzioni, vincoli, cosa evitare; (2)
**esempi di codice** del repository, che condizionano l'assistente sullo stile
reale invece che su uno generico; (3) la **documentazione** pertinente (API,
riferimenti), selezionata e non l'intera libreria; (4) un **validation gate**,
cioè il criterio oggettivo (i test da far passare, il *linter*, il comando che
deve tornare a zero) con cui verificare che il lavoro sia effettivamente
finito. I primi tre ingredienti sono context engineering allo stato puro: sono
le mosse *select* e *write* rese esplicite in un artefatto versionabile. Il
quarto anticipa la prossima sezione: il *validation gate* è il seme del **loop
engineering**, perché trasforma un colpo solo in un **ciclo** (genera,
verifica contro il gate, e se fallisce reitera con l'esito in contesto). Non è
un caso che il repo riassuma la propria tesi con uno slogan volutamente
iperbolico («10x meglio del prompt engineering, 100x meglio del *vibe
coding*») che riportiamo come rivendicazione di chi propone il metodo, non
come misura verificata: l'ordine di grandezza è retorica, l'intuizione (dare
più contesto strutturato riduce gli errori) è sensata.

`````

C'è una variante «leggera» della stessa idea che merita una riga. Al posto del
progetto per un lavoro solo, si mette nel contesto un metodo di lavoro: schemi
di ragionamento riusabili, «scomponi il problema», «verifica il risultato»,
che fanno da impalcatura a come il modello procede. Ebouky, Bartezzaghi e
Rigotti li hanno studiati sotto il nome di **cognitive tools**, strumenti
cognitivi, e mostrano che con questi nel contesto il modello ragiona meglio
senza che dentro gli si sia toccato niente {cite}`ebouky2025cognitive`. È lo
stesso spirito del PRP applicato non al codice ma al pensiero, e sta anch'esso
al livello degli «organi» della scala di prima: non un singolo prompt, ma
qualcosa che mette in fila più passi.

La direzione è chiara, ed è anche il senso di un lavoro del 2025 che ha
setacciato oltre mille e quattrocento articoli scientifici sul tema per
metterli in fila {cite}`mei2025context`: il context engineering sta diventando
una **disciplina**, con le sue immagini mentali, le sue tattiche e i suoi modi
di fallire. È la stessa strada che ha percorso, decenni fa, il mestiere di
scrivere programmi, quando ha smesso di essere un'arte individuale e si è dato
delle regole. La finestra è un sistema e non una casella di testo, e va
progettata come tale.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Quello che arriva al modello è **tutto un pacco** e non la tua frase: lo
  monta il programma ogni volta rimettendoci dentro le istruzioni di fondo, gli
  esempi, i pezzi di documento che servono e il riassunto di quanto già detto.
  Progettare quel pacco è il mestiere; la tua frase ne è l'ultima riga.
- Il contesto non è tutto uguale: c'è quello semplice come un atomo (una
  regola sola) e quello complesso come un corpo (più passi coordinati). Sapere
  a che altezza si sta lavorando dice quanta cura serve. Le proposte in cima a
  quella scala sono ancora **ricerca**, non tecniche pronte, e va detto.
- Sulla scrivania piccola ci sono **quattro gesti**: appuntare fuori quel che
  non serve adesso, andare a prendere quel che serve e nient'altro, riassumere
  in poche righe una pila di appunti, e passare un pezzo di lavoro a un collega
  con la sua scrivania.
- Un contesto si guasta in **quattro modi**: un errore che ci entra e da lì in
  poi viene ripetuto come se fosse vero; un contesto così lungo che il modello
  si fissa su quello che c'è scritto dentro e dimentica quello che sa; dettagli
  inutili che tirano la risposta fuori strada; pezzi che si contraddicono a
  vicenda. Più lungo non vuol dire migliore: nella prova di Liu e colleghi, con
  venti o trenta documenti in finestra e quello giusto nel mezzo, le risposte
  erano peggiori di quelle date senza alcun documento; e una finestra più
  capiente non è una finestra usata meglio.
- Il contesto si può preparare una volta e riusare: un **foglio di progetto**
  con dentro le regole, gli esempi, i pezzi di manuale che servono e,
  decisivo, la **prova con cui si stabilisce se il lavoro è finito**.
  Quest'ultima è il ponte verso la prossima sezione.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il context engineering {cite}`karpathy2025context` sposta l'unità del
  mestiere dal **singolo messaggio** (il prompt) all’**intero payload** che
  riempie la finestra a ogni passo: un **sistema** (regole, esempi,
  documentazione, validazione) non una «frase magica».
- Una scala di complessità utile (metafora biologica): **atomi** (istruzioni) →
  **molecole** (*few-shot*) → **cellule** (memoria) → **organi** (workflow,
  strumenti). L'estremità «campi / neural fields» è **frontiera speculativa**, non
  risultato consolidato: va detto.
- Quattro mosse per governare il contesto: **write** (fuori dalla finestra),
  **select** (andare a prendere solo ciò che serve al passo corrente, il gesto
  che sta anche dietro ai sistemi che recuperano documenti prima di rispondere
  {cite}`lewis2020retrieval`), **compress** (riassumere/potare), **isolate**
  (partizionare tra sotto-agenti).
- Quattro guasti (catalogo di Breunig {cite}`breunig2025contexts`):
  **poisoning** (un errore che si sedimenta e si autoalimenta), **distraction**
  (il contesto lungo che fa prevalere ciò che vi si legge su ciò che il modello
  ha appreso), **confusion** (token irrilevanti usati perché presenti) e
  **clash** (contesto contraddittorio). Il *lost in the middle*
  {cite}`liu2024lost` misura sia la **posizione** (curva a U) sia la
  **lunghezza**: sul modello lì misurato, con venti o trenta documenti e nel
  caso peggiore (rilevante in mezzo) si scende sotto il risultato a libro
  chiuso, e la finestra dichiarata non è la finestra utile.
- Il **PRP** rende il context engineering una **procedura ripetibile**: regole,
  esempi, documentazione e **validation gate** (il criterio oggettivo che dice
  se il lavoro è finito). Quest'ultimo anticipa il **loop
  engineering** della prossima sezione: verificare l'esito e reiterare.
- La meccanica (il budget di token come problema dello zaino, il *context
  builder*, la memoria) è già nel capitolo
  sugli **Agenti**: qui abbiamo aggiunto i modelli mentali, le tattiche e i modi
  di fallire.
```

`````
