# Context engineering: la finestra come sistema

Apri un'applicazione LLM seria e guarda cosa arriva davvero al modello a ogni
richiesta. Quasi nulla di quel testo lo ha battuto una persona sulla tastiera:
c'è il *system prompt* montato dal programma, ci sono alcuni esempi pescati da
una libreria, un pezzo di documentazione recuperato al volo, il riassunto
degli scambi precedenti, l'esito dell'ultima chiamata a uno strumento, e, in
fondo, la breve frase dell'utente. La domanda dell'utente è la punta
dell'iceberg; sotto c'è un intero **payload** assemblato a codice. Quando
Andrej Karpathy, nel 2025, ha proposto di chiamare tutto questo *context
engineering* {cite}`karpathy2025context`, ha spostato l'unità di misura del
mestiere: non più la singola frase («trovare l'incantesimo giusto») ma il
**governo dell'intero contesto** che riempie la finestra a ogni passo.

È il tema di questo capitolo, e la sezione precedente sul prompt engineering
ne ha aperto la porta: là abbiamo lavorato sul *singolo messaggio*, sulla
frase scritta bene. Qui saliamo di un livello e trattiamo la finestra come un
**sistema** da progettare nel suo insieme. Il repo *context-engineering-intro*
di Cole Medin lo dice con uno slogan efficace, che vale la pena riportare come
affermazione di chi costruisce, non come teorema: il context engineering non
sono «parole magiche», è un sistema completo (regole, esempi, documentazione,
validazione) e sta al prompt engineering come una **sceneggiatura** sta a un
*post-it*. Un post-it dice cosa fare in una riga; una sceneggiatura dà a ogni
scena il contesto per recitarla bene.

Una precisazione di perimetro, per non ripeterci. La **meccanica** di questo
sistema, il budget di token come problema di zaino, il *context builder* che
assembla il prompt a ogni passo, il *lost in the middle* di Liu e colleghi
{cite}`liu2024lost`, le forme di memoria dentro e fuori la finestra: l'abbiamo
già smontata pezzo per pezzo nel capitolo sugli **Agenti**, nella sua sezione
di context engineering. Non la riscriviamo. Qui prendiamo un'angolazione
diversa e complementare: **come si pensa** il contesto (un modello mentale a
scale di complessità), **quali mosse** lo governano, **come si guasta**, e
**come lo si rende una procedura ripetibile**.

## Una scala di complessità: dagli atomi ai campi

Il primo passo è avere un'immagine mentale di *quanto* è complesso il contesto
che stiamo montando. Il repo *Context-Engineering* di David Kim propone una
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
**organi** che svolgono funzioni complesse: un flusso a più passi, con strumenti
che il modello può usare. E in cima, gli **organismi** interi. Non serve
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
imposto un framing comune (reso popolare dalle note di LangChain) che
raccoglie tutte le tattiche in quattro mosse. Vale la pena tenerle a mente
come un piccolo repertorio.

`````{tab} Elementare

Immagina la scrivania minuscola di cui parlavamo negli agenti: ci sta poca roba,
e va tenuta in ordine. Hai quattro gesti a disposizione. **Scrivere** fuori:
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
  corrente: è il **RAG** {cite}`lewis2020retrieval` per i documenti, ma anche
  il recupero della memoria giusta o della descrizione dello strumento giusto.
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

## Come si guasta un contesto

Un contesto più lungo non è un contesto migliore. Anzi: quasi tutti i modi in cui
un'applicazione LLM peggiora con l'uso hanno a che fare con un contesto che si
sporca. Drew Breunig ne ha proposto un catalogo utile, che qui riprendiamo con
parole nostre. Quattro guasti ricorrenti:

- **Context poisoning**, un errore o un'allucinazione entra nel contesto e vi
  si **sedimenta**: da lì in poi il modello lo tratta come un fatto acquisito
  e ci costruisce sopra, avvelenando ogni passo successivo. È il più insidioso
  perché si **autoalimenta**.
- **Context distraction**, troppa cronologia accumulata **distrae** dal
  compito: il modello si perde tra i suoi stessi passi passati invece di
  guardare avanti.
- **Context confusion**, informazione superflua ma presente **confonde**:
  dettagli irrilevanti spingono verso risposte fuori fuoco, perché il modello
  «li usa» solo perché ci sono.
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

`````

`````{tab} Superiore

*Distraction* e *confusion* hanno una radice comune che conosciamo già: il
**lost in the middle** {cite}`liu2024lost` e, più in generale, la **diluizione
dell'attenzione**. Man mano che il contesto si allunga, il segnale rilevante
si distribuisce su più token e la capacità del modello di isolarlo cala:
l'abbiamo misurato nel capitolo sugli Agenti, dove la curva di accuratezza in
funzione della posizione dell'informazione ha la forma a U. *Distraction* è il
caso in cui il segnale utile è annegato nella cronologia; *confusion* quello
in cui token irrilevanti ma «vicini» al compito attirano indebitamente
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

Un PRP tipico raccoglie quattro ingredienti: (1) le **regole di progetto**,
una sorta di file `CLAUDE.md`, con convenzioni, vincoli, cosa evitare; (2)
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

C'è una variante «leggera» di questa stessa idea che merita una riga. Un
gruppo di IBM Research ha studiato i **cognitive tools**: template di
ragionamento riusabili («scomponi il problema», «verifica il risultato») che
si iniettano nel contesto per **strutturare** il modo in cui il modello pensa,
elicitando ragionamento senza riaddestrare nulla {cite}`ebouky2025cognitive`.
È lo stesso spirito del PRP applicato non al codice ma al pensiero: invece di
sperare che il modello ragioni bene, gli si fornisce l'impalcatura del
ragionamento come parte del contesto. Rientra nella scala di prima al livello
degli «organi»: non un singolo prompt, ma uno strumento cognitivo che
orchestra più passi.

La direzione è chiara, ed è anche il senso della survey del 2025 che ha
censito oltre un migliaio di lavori sul tema {cite}`mei2025context`: il
context engineering sta diventando una **disciplina** con i suoi modelli
mentali, le sue tattiche e i suoi modi di fallire; proprio come, a suo tempo,
lo è diventata l'ingegneria del software. La finestra non è una casella di
testo: è un sistema, e va progettata come tale.

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
  **select** (recuperare il pertinente, RAG {cite}`lewis2020retrieval`),
  **compress** (riassumere/potare), **isolate** (partizionare tra sotto-agenti).
- Quattro guasti (catalogo di Breunig): **poisoning** (un errore che si sedimenta
  e si autoalimenta), **distraction** e **confusion** (imparentati col *lost in
  the middle* {cite}`liu2024lost`), **clash** (contesto contraddittorio).
- Il **PRP** rende il context engineering una **procedura ripetibile**: regole,
  esempi, documentazione e **validation gate**. Quest'ultimo anticipa il **loop
  engineering** della prossima sezione: verificare l'esito e reiterare.
- La meccanica (budget/knapsack, *context builder*, memoria) è già nel capitolo
  sugli **Agenti**: qui abbiamo aggiunto i modelli mentali, le tattiche e i modi
  di fallire.
```
