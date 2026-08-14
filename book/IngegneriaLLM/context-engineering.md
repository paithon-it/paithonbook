# Context engineering: la finestra come sistema

Apri un'applicazione LLM seria e guarda cosa arriva davvero al modello a ogni
richiesta. Quasi nulla di quel testo lo ha battuto una persona sulla tastiera.
C'è il *system prompt*, cioè le istruzioni di fondo che il programma antepone
sempre, quelle del regista visto nella sezione scorsa. Ci sono alcuni esempi
già svolti, tenuti da parte in un archivio e ripescati perché somigliano al
caso di adesso. C'è un pezzo di documentazione andato a prendere in quel
momento, in un manuale che nella finestra non ci starebbe mai per intero. C'è
il riassunto degli scambi precedenti, perché il modello non ricorda nulla da
solo. C'è il risultato dell'ultima operazione che il modello ha chiesto al
programma di eseguire per lui (una ricerca, una query, un calcolo). E in
fondo, ultima, la breve frase dell'utente. La domanda dell'utente è la punta
dell'iceberg; sotto c'è un intero **payload** (il carico) assemblato dal
codice. Quando
Andrej Karpathy, nel 2025, ha proposto di chiamare tutto questo *context
engineering* {cite}`karpathy2025context`, ha spostato l'unità di misura del
mestiere: non più la singola frase («trovare l'incantesimo giusto») ma il
**governo dell'intero contesto** che riempie la finestra a ogni passo.

È il tema di questa sezione, e quella precedente sul prompt engineering
ne ha aperto la porta: là abbiamo lavorato sul *singolo messaggio*, sulla
frase scritta bene. Qui saliamo di un livello e trattiamo la finestra come un
**sistema** da progettare nel suo insieme. Il *repository* (un archivio di
codice pubblico, «repo» per brevità) *context-engineering-intro*
di Cole Medin {cite}`medin2025contextintro` lo dice con uno slogan efficace,
che vale la pena riportare come
affermazione di chi costruisce, non come teorema: il context engineering non
sono «parole magiche», è un sistema completo (regole, esempi, documentazione,
validazione) e sta al prompt engineering come una **sceneggiatura** sta a un
*post-it*. Un post-it dice cosa fare in una riga; una sceneggiatura dà a ogni
scena il contesto per recitarla bene.

Una precisazione di perimetro, per non ripeterci. La **meccanica** di questo
sistema, il budget di token come problema di zaino (scegliere cosa mettere in
uno spazio che non basta per tutto), il *context builder* che
assembla il prompt a ogni passo, il *lost in the middle* di Liu e colleghi
{cite}`liu2024lost`, le forme di memoria dentro e fuori la finestra: l'abbiamo
già smontata pezzo per pezzo nel capitolo sugli **Agenti**, nella sua sezione
di context engineering. Non la riscriviamo. Qui prendiamo un'angolazione
diversa e complementare: **come si pensa** il contesto (un modello mentale a
scale di complessità), **quali mosse** lo governano, **come si guasta**, e
**come lo si rende una procedura ripetibile**.

## Una scala di complessità: dagli atomi ai campi

Il primo passo è avere un'immagine mentale di *quanto* è complesso il contesto
che stiamo montando. Il repo *Context-Engineering* di David Kim
{cite}`kim2025contextengineering` propone una
metafora presa in prestito dalla biologia: come la materia vivente sale di
complessità dagli atomi agli organismi, così il contesto sale da una singola
istruzione fino a interi ecosistemi di componenti che collaborano. È una
metafora didattica (un modo di ordinare le idee, non una tassonomia
scientifica) e come tale la usiamo.

`````{tab} Elementare

Pensa a come si costruisce qualcosa di vivo. Parti dagli **atomi**: i mattoni più
piccoli, una singola regola («rispondi in italiano»). Metti insieme più atomi e
ottieni una **molecola**: l'istruzione più due o tre esempi che mostrano cosa
intendi. Le molecole formano **cellule**, che hanno una memoria: il sistema
ricorda chi sei tra un messaggio e l'altro. Le cellule si organizzano in
**organi** che svolgono funzioni complesse: un flusso a più passi, in cui il
modello può chiedere al programma di fare qualcosa per lui (cercare, calcolare,
aprire un archivio) e poi usare il risultato. E in cima ci sono proposte che
guardano al contesto come a un unico ambiente continuo, ma sono ricerca
aperta, non attrezzi da usare oggi. Non serve
imparare la biologia: serve l'idea che il contesto non è tutto uguale: c'è quello
semplice come un mattone e quello complesso come un corpo, e sapere a che
altezza della scala stai lavorando ti dice quanta cura serve.

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
task semplice vive negli atomi e nelle molecole: un buon prompt basta. Un
agente vive negli organi, e lì il collo di bottiglia non è più la frase, è la
*gestione* di ciò che entra ed esce dalla finestra a ogni giro.

## Quattro mosse: scrivere, selezionare, comprimere, isolare

Salendo di scala, il contesto diventa qualcosa da **gestire attivamente**,
come si gestisce la memoria di un programma. Tra chi costruisce agenti si è
imposto un framing comune (reso popolare dalle note di LangChain
{cite}`langchain2025context`) che
raccoglie tutte le tattiche in quattro mosse. Vale la pena tenerle a mente
come un piccolo repertorio.

`````{tab} Elementare

Immagina di lavorare su una scrivania minuscola, come già negli agenti: ci sta
poca roba, e va tenuta in ordine. Hai quattro gesti a disposizione.
**Scrivere** fuori:
quello che non ti serve *adesso* lo appunti su un foglio di lato, così libera il
tavolo ma resta recuperabile. **Selezionare**: quando ti serve qualcosa, vai a
prendere *solo quella cosa* dal foglio o dallo schedario, non svuoti il cassetto
sul tavolo. **Comprimere**: una pila di appunti lunga la riscrivi in tre righe di
sunto, che occupano molto meno spazio. **Isolare**: se il compito è grosso, lo
spezzi e ne dai un pezzo a un collega, con la *sua* scrivania, così la tua non si
intasa. Quattro gesti semplici, ripetuti a ogni passo, che tengono la finestra
pulita.

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

Un contesto più lungo non è un contesto migliore. Anzi: quasi tutti i modi in cui
un'applicazione LLM peggiora con l'uso hanno a che fare con un contesto che si
sporca. Drew Breunig ne ha proposto un catalogo utile
{cite}`breunig2025contexts`, che qui riprendiamo con
parole nostre. Quattro guasti ricorrenti:

- **Context poisoning**, un errore o un'allucinazione entra nel contesto e vi
  si **sedimenta**: da lì in poi il modello lo tratta come un fatto acquisito
  e ci costruisce sopra, avvelenando ogni passo successivo. È il più insidioso
  perché si **autoalimenta**.
- **Context distraction**, il contesto cresce tanto che il modello si
  **fissa** su ciò che ci legge dentro e trascura quello che aveva imparato in
  addestramento: si perde tra i propri passi passati invece di guardare avanti,
  e nei casi osservati arriva a ripetere azioni che ha già fatto.
- **Context confusion**, informazione superflua ma presente **confonde**:
  dettagli irrilevanti spingono verso risposte fuori fuoco, perché il modello
  «li usa» solo perché ci sono. L'esempio di Breunig è il catalogo degli
  strumenti: un modello piccolo che sbaglia con quarantasei strumenti a
  disposizione e ci riesce con diciannove.
- **Context clash**, pezzi di contesto in **contraddizione** tra loro (due
  documenti che si smentiscono, una policy vecchia accanto a una nuova): il
  modello non sa a chi credere e la risposta ne risente.

`````{tab} Elementare

Il più subdolo è il primo, il *poisoning*, e funziona come una diceria. Basta
che in un gruppo entri una voce falsa («il negozio chiude alle 18») e da quel
momento tutti la ripetono come vera: chi arriva dopo la sente già «confermata»
da tre persone e non la mette in dubbio. Nel contesto succede lo stesso: se al
passo tre il modello «decide» per sbaglio che l'utente si chiama Marco, ai
passi quattro, cinque, sei quel Marco è ormai lì, scritto, e il modello ci
parla come se fosse sempre stato vero. L'errore non resta un errore: diventa
una premessa. È per questo che, con gli agenti, conviene ripulire il contesto
invece di lasciarlo crescere all'infinito.

E c'è una conseguenza pratica che vale la pena dire subito, perché è la cosa
più utile di questa pagina e si fa in un secondo: **quando una conversazione
comincia a dire cose sbagliate, non insistere: aprine una nuova.** Correggere
il modello dentro la stessa chat lascia l'errore dov'è, in mezzo a tutto quello
che si è detto prima, e lui continuerà a rileggerlo. Una chat nuova parte dal
foglio bianco, ed è l'unico modo che hai, da fuori, di togliere una diceria dal
gruppo.

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
GPT-3.5-Turbo, con venti o trenta documenti l'accuratezza scende **sotto**
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

## Il PRP: context engineering come procedura ripetibile

Se il context engineering è un sistema, allora dovrebbe potersi **scrivere una
volta e riusare**, non reinventare a ogni task. È l'idea del **Product
Requirements Prompt (PRP)**, proposta nel repo di Cole Medin per gli assistenti
di *coding*. Un PRP è un *blueprint*: un documento che impacchetta in modo
sistematico tutto ciò che serve a portare a termine un compito, così che
l'assistente non parta da un contesto vuoto.

`````{tab} Elementare

È la differenza tra dire a un artigiano «fammi un tavolo» e consegnargli un
**progetto completo**: le misure, il tipo di legno, un disegno di com'è fatto
un tavolo che ti piace, e (dettaglio decisivo) la regola con cui alla fine
controllerete insieme se il tavolo è venuto bene («deve stare in piano e
reggere 40 chili»). Con il progetto in mano, l'artigiano lavora quasi da solo
e sbaglia di meno, perché ha davanti tutto il contesto *prima* di iniziare, e
sa già come si misura il successo. Il PRP è quel progetto, scritto per un
assistente che programma: regole, esempi, documentazione e il test finale,
tutto in un foglio solo, pronto da riusare al prossimo lavoro.

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

C'è una variante «leggera» di questa stessa idea che merita una riga. Ebouky,
Bartezzaghi e Rigotti hanno studiato i **cognitive tools**: schemi di
ragionamento riusabili («scomponi il problema», «verifica il risultato») che
si mettono nel contesto per **dare un'impalcatura** al modo in cui il modello
procede, tirandone fuori un ragionamento senza riaddestrare nulla
{cite}`ebouky2025cognitive`.
È lo stesso spirito del PRP applicato non al codice ma al pensiero: invece di
sperare che il modello ragioni bene, gli si fornisce l'impalcatura del
ragionamento come parte del contesto. Rientra nella scala di prima al livello
degli «organi»: non un singolo prompt, ma uno strumento cognitivo che
orchestra più passi.

La direzione è chiara, ed è anche il senso della rassegna del 2025 che ha
setacciato oltre mille e quattrocento articoli scientifici sul tema
{cite}`mei2025context`: il
context engineering sta diventando una **disciplina** con i suoi modelli
mentali, le sue tattiche e i suoi modi di fallire; proprio come, a suo tempo,
lo è diventata l'ingegneria del software. La finestra non è una casella di
testo: è un sistema, e va progettata come tale.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Quello che arriva al modello non è la tua frase: è **tutto un pacco**, che il
  programma monta ogni volta rimettendoci dentro le istruzioni di fondo, gli
  esempi, i pezzi di documento che servono e il riassunto di quanto già detto.
  Progettare quel pacco è il mestiere; la tua frase ne è l'ultima riga.
- Il contesto non è tutto uguale: c'è quello semplice come un mattone (una
  regola sola) e quello complesso come un corpo (più passi coordinati). Sapere
  a che altezza si sta lavorando dice quanta cura serve. Le proposte in cima a
  quella scala sono ancora **ricerca**, non tecniche pronte, e va detto.
- Sulla scrivania piccola ci sono **quattro gesti**: appuntare fuori quel che
  non serve adesso, andare a prendere **solo** quel che serve, riassumere in
  poche righe una pila di appunti, e passare un pezzo di lavoro a un collega
  con la sua scrivania.
- Un contesto si guasta in **quattro modi**: un errore che ci entra e da lì in
  poi viene ripetuto come se fosse vero; un contesto così lungo che il modello
  si fissa su quello che c'è scritto dentro e dimentica quello che sa; dettagli
  inutili che tirano la risposta fuori strada; pezzi che si contraddicono a
  vicenda. Più lungo **non** vuol dire migliore: in una misura fatta su un
  modello di qualche anno fa, oltre una ventina di documenti in finestra le
  risposte erano peggiori di quelle date senza alcun documento.
- La cosa si può scrivere una volta e riusare: un **progetto** che contiene
  regole, esempi, documentazione e, decisivo, la **prova con cui si stabilisce
  se il lavoro è finito**. Quest'ultima è il ponte verso la prossima sezione.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il context engineering {cite}`karpathy2025context` sposta l'unità del
  mestiere dal **singolo messaggio** (il prompt) all'**intero payload** che
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
  **lunghezza**: sul modello lì misurato, oltre venti documenti si scende sotto
  il risultato a libro chiuso, e la finestra dichiarata non è la finestra
  utile.
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
