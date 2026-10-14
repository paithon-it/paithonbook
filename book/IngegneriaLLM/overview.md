# Prompt, contesto e loop: programmare gli LLM

```{image} ../figures/aperture/ingegneria-llm.png
:class: pt-apertura only-light
:width: 100%
:alt: Una mano posa una carta dentro una cornice vuota; accanto, altre carte messe da parte.
```

```{image} ../figures/aperture/ingegneria-llm-scura.png
:class: pt-apertura only-dark
:width: 100%
:alt: Una mano posa una carta dentro una cornice vuota; accanto, altre carte messe da parte.
```

Nel giugno 2025 Andrej Karpathy (tra i fondatori di OpenAI, per anni a capo
dell'intelligenza artificiale in Tesla) ha dato il suo appoggio a un nome
appena proposto per un lavoro che esisteva già. Su X, il social network che
fino al 2023 si chiamava Twitter, ha scritto di preferire il termine context
engineering a «prompt engineering», e lo ha definito così: l'arte e insieme la
scienza, delicata, di riempire la finestra di contesto con *la giusta
informazione per il passo successivo* {cite}`karpathy2025context`.

Chi arriva dal capitolo sugli agenti queste parole le ha già incontrate nella
{doc}`sezione sul contesto come interfaccia </Agenti/context-engineering>`, e
da lì si riparte. Il *prompt*, in senso stretto, è il messaggio che si scrive
al modello, la richiesta vera e propria; tutto il testo che il programma le
monta attorno è il *contesto*. Nell'uso corrente, e nei nomi del codice,
«prompt» si allarga spesso a tutto il testo montato; qui i due oggetti vanno
tenuti distinti, e nella prosa «prompt» vale in senso stretto. In un assistente
per i clienti di un negozio, «il mio ordine non è arrivato» è il prompt; le
istruzioni di servizio che il programma antepone sempre, i dati dell'ordine
presi dall'archivio e i messaggi scambiati prima sono il contesto. Gli esempi
già svolti stanno dall'una o dall'altra parte secondo chi ce li mette: nel
prompt se li scrive chi fa la richiesta, nel contesto se li aggiunge il
programma.

La finestra di contesto è il tetto di testo che un modello riesce a leggere in
una volta sola: la richiesta, e insieme tutto ciò che vogliamo che il modello
sappia prima di rispondere, deve starci dentro, e quando è piena qualcosa va
tolto per far posto. Il limite lo fissa chi ha costruito il modello, e non è
una metafora: per questo riempire bene la finestra è un lavoro a sé. Il «passo
successivo» della definizione di Karpathy è la richiesta che segue, in un
lavoro che procede a riprese: si tratta di *riempire bene una finestra*, e di
rifarlo a ogni passo.

Karpathy aveva preparato il terreno da tempo. Anni prima aveva parlato di
**Software 2.0**, una numerazione che presuppone un Software 1.0: il software
di sempre, che qualcuno scrive riga per riga in un linguaggio di
programmazione. Nel 2.0, cioè nei sistemi di apprendimento automatico, il
programma non lo scrive più una persona riga per riga, lo si *addestra*, cioè
gli si mostrano montagne di esempi e lo si lascia regolare, un pochino alla
volta, i numeri che ha dentro. Quei numeri sono i pesi della rete, milioni o
miliardi, e contengono tutto l'apprendimento del modello: sono il suo codice.
Nel 2025 ha aggiunto un terzo capitolo, il **Software 3.0**, osservando che
oggi, con i grandi modelli linguistici, «si programma in inglese»: il prompt
*è* il programma, scritto in lingua naturale invece che in Python (con
«inglese» si intende la lingua naturale, e qui useremo l'italiano per gli
esempi). È un'immagine forte, da prendere con prudenza; ma coglie qualcosa di
vero, e da lì parte l'ingegneria dei prompt.

## Programmare a parole

Per programmare un modello di linguaggio bisogna sapere che cosa fa quando
risponde. Un LLM, il grande modello di linguaggio (*large language model*) che
il capitolo sugli agenti ha messo al centro di ogni ciclo, compie una sola
operazione, ripetuta moltissime volte: guarda il testo che ha davanti e stima
quale pezzo di testo verrà dopo.

```{figure} ../figures/cos-e-davvero-un-llm.svg
:name: fig-cos-e-un-llm
:alt: "La frase «Il gatto nero salta sul» entra nel modello, seguita da una casella vuota con un punto interrogativo. Il modello restituisce una graduatoria di parole con la loro probabilità: muro 41 per cento, tetto 20, ramo 11, divano 3, tavolo 2, e un 23 per cento distribuito su tutto il resto del vocabolario. Il modello non produce una risposta ma questa graduatoria: la risposta nasce dopo, pescandone un elemento."
:width: 92%

Cosa restituisce un LLM. Non una frase: una classifica di parole
possibili, ciascuna con la sua probabilità. Tutto ciò che sembra dialogo è
questo passaggio, ripetuto.
```

Tenere presente {numref}`fig-cos-e-un-llm` cambia il modo di leggere tutto il
capitolo. Dal modello esce una classifica di parole candidate, ciascuna con la
sua percentuale: dopo «il gatto nero salta sul» potrebbe dire muro 41%, tetto
20%, ramo 11%, e via calando fino alle ultime parole del vocabolario. La
risposta nasce estraendo a sorte da quella classifica, con le percentuali come
pesi: «muro» esce circa quattro volte su dieci, «tetto» due. Estrarre così si
dice *campionare*, e per questo la stessa domanda può avere risposte diverse
(la statistica chiama questa classifica «distribuzione»). Nella figura le voci
si chiamano *token*, che è il pezzo di parola con cui il modello lavora, come
abbiamo visto nella {doc}`sezione sui tokenizzatori
</NaturalLanguageProcessing/tokenizzatori>`. Diremo «parola» dove la differenza
non conta, e «token» dove conta: cioè quando si tratta di contarli, perché a
token si misura quanto testo entra nella finestra e a token si paga. In
simboli, un modello di parametri $\theta$ calcola $P_\theta(x_t \mid x_{<t})$,
la probabilità di ciascun token $x_t$ dati i token $x_{<t}$ che lo precedono;
la risposta si costruisce un token alla volta, estraendolo e rimettendolo in
coda al testo.

Questa classifica dipende dal testo che il modello ha davanti: cambia il testo
e i numeri si spostano. Ecco perché «programmare a parole» non è un modo di
dire. Le parole che gli scriviamo sono la leva che sposta la classifica, e
finché non mettiamo le mani dentro il modello è l'unica che abbiamo.

Ne segue che, con un LLM già addestrato, cioè uno di quelli che si trovano
pronti e che sanno già leggere e scrivere, non si programma più toccando i
pesi: quelli sono congelati, li ha fissati l'addestramento. Il congelamento non
è definitivo (riaprirli e proseguire l'addestramento sui propri dati si può, si
chiama fine-tuning ed è la strada più costosa, come mostra la
{numref}`fig-quale-leva`), ma i pesi restano fermi mentre si lavora sul testo.
Programmiamo con le parole, cioè con il testo che gli mettiamo davanti prima di
chiedergli una risposta. Cambiare quel testo cambia il comportamento del
sistema tanto quanto, nel software tradizionale, cambierebbe riscrivere una
funzione.

`````{tab} Elementare

È il collega nuovo del capitolo sugli agenti, quello a cui lasciavi il briefing
sulla scrivania: bravissimo e velocissimo, ha letto mezza biblioteca, è appena
arrivato e non sa nulla del *tuo* lavoro. Puoi anche mandarlo a un corso di
formazione, ma è una faccenda lunga e costosa. Subito, ogni giorno e senza
spedirlo da nessuna parte, puoi parlargli bene. Se gli dici «occupati dei
clienti» otterrai un risultato; se gli lasci un foglio con il ruolo, tre esempi
di risposte giuste e il tono da tenere, ne otterrai un'altra, molto migliore:
stesso collaboratore, stesso cervello, solo parole diverse. Nota il dettaglio
del foglio, perché tornerà per tutto il capitolo: quei tre esempi già svolti
sono l'aiuto più grande, e lui non ha imparato niente, li ha soltanto letti.
Programmare a parole è questo: la persona resta la stessa e cambia ciò che le
si dice. Una cosa il foglio non te la dà: la stessa risposta due volte.
Ridaglielo domani e avrai qualcosa di simile, non la copia di oggi, come
succede con chiunque. Siccome parole diverse danno risultati diversi, a volte
molto diversi, sceglierle bene richiede metodo.

`````

`````{tab} Superiore

È utile leggere questa tesi lungo la scala che Karpathy chiama Software 1.0 →
2.0 → 3.0. Nel Software 1.0 il comportamento è codice imperativo scritto a
mano. Nel Software 2.0 è un vettore di parametri $\theta$ appreso
minimizzando una loss $\mathcal{L}$ su dei dati: il programma emerge
dall'ottimizzazione, non dalla penna del programmatore. Il Software 3.0
sposta di nuovo il piano: $\theta$ resta *fisso* (il modello pre-addestrato non
si tocca) e ciò che varia è il contesto $C$ passato in ingresso. Il sistema
genera

$$
\hat{y} \sim P_{\theta}(\,\cdot \mid C),
$$

dove $P_{\theta}$ è la distribuzione condizionata calcolata dal modello
congelato, $C$ è tutto il testo che gli forniamo (istruzioni, esempi,
documenti, cronologia) e $\hat{y}$ la risposta, *campionata* da quella
distribuzione, eventualmente riscalata e troncata dai parametri di decoding
(temperatura, top_p) della {doc}`sezione sul prompt engineering
<prompt-engineering>`: lo stesso $C$ può dare risposte diverse a temperatura
non nulla e, come mostra quella sezione, perfino a temperatura zero.
Programmare significa progettare $C$. Il meccanismo che rende possibile tutto
questo è l’**in-context learning**, documentato su larga scala da Brown e
colleghi nel lavoro su GPT-3 {cite}`brown2020language`: bastano poche coppie
richiesta → risposta nel contesto (il *few-shot*), perché il modello esegua un
compito nuovo *senza alcun aggiornamento dei pesi*. Gli esempi non addestrano:
condizionano. La scoperta è raccontata nella {doc}`sezione sui grandi modelli
linguistici </Transformers/llm>`, e la sua forma probabilistica è formalizzata
nella {doc}`sezione sul prompt engineering <prompt-engineering>`; qui ci basta
la conseguenza: la programmazione avviene nel testo.

`````

Detto così, sembra che il tutto si riduca a scrivere una buona frase. È un
equivoco da togliere subito, e per questo la terminologia è cambiata. Fra noi e
il modello, in un'applicazione vera, c'è sempre un programma: il sito,
l'assistente, le righe di codice che raccolgono la nostra richiesta e la
spediscono. Il testo che arriva al modello lo scrive quel programma, che lo
monta con parti dai ruoli diversi poco prima di partire; non è una frase. Ogni
spedizione è una *chiamata* al modello, e in gergo il carico che viaggia con
lei si chiama **payload**: dentro ci stanno anche le impostazioni della
chiamata (per esempio quanto può essere lunga la risposta), e il contesto ne è
la parte testuale. Il carico va costruito, misurato e ricostruito a ogni passo.
Da qui i tre livelli del capitolo.

## Tre cerchi concentrici

Programmare a parole si fa a tre livelli, uno dentro l'altro come cerchi
concentrici ({numref}`fig-ingegneria-cerchi`), dal più piccolo al più grande.

```{figure} ../figures/ingegneria-llm-cerchi.svg
:name: fig-ingegneria-cerchi
:alt: "Tre cerchi concentrici. Al centro, il più piccolo, porta la scritta \"prompt: il singolo messaggio\". Il cerchio mediano che lo racchiude porta la scritta \"contesto: la finestra come sistema\". Il cerchio esterno che racchiude entrambi porta la scritta \"loop: il processo iterativo\". A destra, una legenda ripete i tre nomi per esteso: prompt engineering, context engineering, loop engineering. In fondo, la riga che spiega il disegno: ogni cerchio contiene il precedente, il prompt nel contesto e il contesto nel loop."
:width: 70%

I tre livelli del «programmare a parole», l'uno dentro l'altro: il prompt (il
singolo messaggio) sta dentro il contesto (l'intera finestra come sistema),
che a sua volta sta dentro il loop (il processo che la ri-riempie a ogni
passo).
```

Il primo cerchio, quello interno, è il **prompt engineering**: il singolo
messaggio. Come si formula un'istruzione perché il modello faccia ciò che
vogliamo: chiarezza, esempi, formato richiesto, il modo di chiedere il
ragionamento passo passo. È il livello a cui si pensa per istinto, e da lì si
comincia.

Il secondo cerchio, che racchiude il primo, è il context engineering: l'intera
finestra come sistema. Qui il prompt dell'utente è solo un pezzo. Ci sono le
istruzioni di fondo, gli esempi, la memoria di ciò che si è detto prima, i
documenti recuperati da un archivio, le descrizioni degli strumenti a
disposizione: le operazioni che il modello può chiedere al programma di
eseguire per lui (cercare sul web, interrogare un archivio, mandare una mail).
Gli strumenti sono il cuore del capitolo sugli **agenti**. Il lavoro non è più
«scrivere una frase» ma decidere cosa mettere nella finestra, in quale ordine,
entro quale budget: perché la finestra è finita, e perché il testo si paga a
quantità. Chi chiama un modello da un programma riceve una fattura
proporzionale al testo che gli manda e al testo che ne riceve: nella chat che
si usa gratis quel conto lo paga qualcun altro, ma esiste, e attorno a questo
vincolo gira tutto il secondo cerchio.

Il terzo cerchio, il più esterno, è il **loop engineering**: il processo che
sta attorno alle chiamate. Dentro c'è il ciclo dell'agente, *osserva → ragiona
→ agisci*, che a ogni giro ri-riempie e ripulisce la finestra; attorno c'è il
programma che decide quando far partire quel ciclo. È lui a stabilire quali
informazioni portare da un giro all'altro e dove memorizzarle quando la
finestra è piena, come validare il risultato e quando fermarsi. Progettare
questo processo è il livello più esterno e più difficile.

`````{tab} Elementare

Sei al ristorante e chiedi al sommelier, l'esperto di vini: «mi
consigli un vino per il pesce?». Quella domanda è il prompt. Il contesto è
tutto ciò che lui ha davanti mentre risponde: il menù, la lista dei vini in
cantina, quanto gli hai detto di voler spendere, la bottiglia che hai
apprezzato la volta scorsa. Il loop è la serata intera: lui propone, tu
assaggi, storci il naso, lui riparte da lì, e così via fino alla bottiglia
giusta.

Il tavolo però è piccolo, come la finestra del modello, e dopo tre giri è pieno
di bicchieri. Il sommelier bravo sparecchia, e prima di sparecchiare segna sul
taccuino la riga che conta: «quel bianco no, troppo secco». Al quarto giro
rilegge il taccuino, non il tavolo. Chi invece si limita a limare il modo di
porre la domanda, e non sparecchia né annota niente, alla quarta bottiglia ti
riporta quel bianco: la domanda era perfetta, la serata no. Domanda dentro il
tavolo apparecchiato, il tavolo dentro la serata: ogni cerchio contiene quello
prima.

`````

`````{tab} Superiore

L'annidamento è letterale, non metaforico. Il prompt, la stringa d'istruzione,
è una delle componenti del contesto $C$, cioè della parte testuale del carico
montato prima della chiamata, $C = [\,\text{system}, \text{esempi},
\text{memoria}, \text{documenti}, \text{strumenti}, \text{prompt}\,]$, dove
$\text{system}$ sono le istruzioni di fondo, quelle che fissano il ruolo. Il
contesto, a sua volta, è ciò che il loop produce e consuma a ogni iterazione:
detto $C_t$ il contesto al passo $t$, $M_t$ la memoria esterna e $o_t$
l'osservazione di ritorno (l'output del modello su $C_t$, il risultato di uno
strumento), il ciclo è

$$
\left(C_{t+1},\; M_{t+1}\right) = g\!\left(C_t,\; M_t,\; o_t\right),
$$

dove $g$ è la politica che aggiorna finestra e memoria: aggiunge
l'osservazione utile, comprime o scarta la cronologia superflua, scrive nella
memoria esterna ciò che va conservato e ne reinietta ciò che serve al passo
dopo. Il loop engineering progetta $g$ e, insieme, il criterio che stabilisce
quando il compito è finito. Ottimizzare il singolo prompt senza
governare $g$ significa curare un fotogramma e ignorare il film: il prompt
vive un istante, il loop dura quanto il compito. Ecco perché i tre livelli non
sono alternativi ma annidati, e perché conviene affrontarli dal piccolo al
grande.

`````

## Perché «ingegneria» e non «magia»

Chiamarlo *engineering*, ingegneria, non serve a darsi un tono. Serve a
prendere le distanze da come veniva trattato agli inizi, cioè come una
stregoneria: la caccia alla formula segreta, alla parola che «sblocca» il
modello. Chiamarla ingegneria è ammettere tre cose poco romantiche ma vere: che
le strade a disposizione costano, e in modo diverso; che quello che si fa si
misura, invece di fidarsi a occhio; e che di ogni modifica si tiene traccia,
come si fa col codice.

Il costo si vede meglio in una figura. Quando un modello non risponde come
vorremmo, abbiamo davanti tre strade, con costi crescenti: riscrivere il
messaggio; recuperare i documenti che al modello mancano e fornirglieli insieme
alla domanda (è il RAG, sigla di *retrieval augmented generation*, «generazione
con recupero»); oppure riaprire i pesi e riaddestrarlo in parte sui nostri
esempi (il fine-tuning). Le prime due lasciano il modello intatto, la terza lo
altera ed è la più costosa da mettere in piedi.

```{figure} ../figures/fine-tuning-rag-o-prompt.svg
:name: fig-quale-leva
:alt: "Schema a domande, un albero di decisione, che parte dal problema e passa subito dal prompt, indicato come la strada più economica e la prima da provare. Solo se il prompt non basta si scende alle due domande successive: se al modello mancano fatti o documenti propri la risposta è il RAG; se invece serve un comportamento costante che il prompt non riesce a tenere, è il fine-tuning; altrimenti si torna a riscrivere il prompt e a misurare."
:width: 88%

Tre strade in ordine di costo, che è anche l'ordine in cui si provano. La prima
domanda non è «quale tecnica è migliore» ma «il prompt ha già fallito?»; solo
dopo viene «cosa manca davvero».
```

L'ordine delle domande in {numref}`fig-quale-leva` dice già come ragiona un
ingegnere. Si comincia dalla strada che costa meno e si sale solo se serve: il
fine-tuning è più caro del prompt, non più avanzato (vuole esempi raccolti a
mano, macchine per addestrare, e va rifatto ogni volta che si cambia modello),
e va giustificato da qualcosa che il prompt non poteva dare. La domanda che fa
scegliere, quando il prompt non basta, è che cosa manchi davvero. Se mancano
dei *fatti* (un manuale, l'archivio degli ordini, i dati di casa nostra) la
risposta è il RAG. Se invece manca un *comportamento*, un modo di rispondere
che nessuna istruzione riesce a tenere fermo su migliaia di richieste (il
gergo di un settore, il tono di un marchio, il modo di rispondere di un
servizio clienti), allora è il caso del fine-tuning. Le due strade hanno già la
loro casa nel libro, il {doc}`recupero e RAG </Transformers/rag>` per la prima
e il {doc}`post-training </Transformers/post-training>` per la seconda: qui ci
basta sapere quando si imboccano.

`````{tab} Elementare

Il foglio del collaboratore si riscrive ogni giorno, e in questo si vede la
differenza fra un incantesimo e un mestiere. L'incantesimo lo pronunci e speri.

Il mestiere lavora dentro dei vincoli: sul foglio ci sta solo una certa
quantità di roba, e ogni riga in più è tempo che lui impiega a leggerla, cioè
denaro. Poi c'è la misura. Tieni da parte venti richieste vere, di quelle che
arrivano davvero; prima di adottare il foglio nuovo lo provi su quelle venti, e
sulle stesse venti riprovi anche il vecchio. Vince il foglio che sbaglia meno,
e a pari errori il più corto. Venti diverse a ogni prova non si potrebbero
confrontare, e una sola non direbbe niente. Restano le versioni: i fogli vecchi
si datano e si tengono nel cassetto, così quando il giovedì le risposte
peggiorano sai che il colpevole è il foglio di mercoledì.

È noioso come tutta l'ingegneria, e funziona per questo. Con
un'avvertenza: è una disciplina nata ieri. Non ci sono leggi, ci sono regole
pratiche che spesso funzionano e ogni tanto no. Chi ti promette la frase che
funziona sempre ti sta vendendo qualcosa.

`````

`````{tab} Superiore

Tre proprietà rendono l'attività ingegneristica e non magica. Primo, i vincoli
sono reali e quantificabili: la finestra ha un tetto di token, e ogni token
pesa su latenza, memoria (la KV cache vista nel capitolo sui Transformer) e
denaro, e del costo per token si occupa la {doc}`sezione su capacità e costo
</MLOps/capacita-e-costo>`. Non si ottimizza «la qualità» in astratto ma la
qualità *sotto vincolo di budget*. Secondo, si misura: una versione del prompt
o della politica di contesto si valuta su una batteria di casi e si confronta
sugli stessi casi con la precedente prima di sostituirla; senza misura non c'è
miglioramento, solo opinioni. Terzo, si versiona: *il prompt è codice*, va
messo sotto controllo di versione e trattato come un artefatto del software,
tema che riprende il {doc}`capitolo su MLOps </MLOps/overview>`. È
un'ingegneria giovane, fatta oggi più di euristiche che di garanzie, e la
terminologia è ancora in assestamento: per un buon tratto si è chiamato tutto
«prompt engineering», finché non si è visto che il problema stava un cerchio
più in fuori.

`````

## Tre mestieri intorno allo stesso modello

Sul primo cerchio la {doc}`sezione sul prompt engineering <prompt-engineering>`
mette le tecniche di fronte alle prove che le sostengono e alle loro fragilità,
compreso il ragionamento passo passo che si chiede esplicitamente, la
*chain-of-thought* {cite}`wei2022chain`. Del secondo cerchio il capitolo sugli
agenti ha già trattato la parte che tocca un agente (quanto spazio occupano gli
strumenti e la traccia nella finestra, come memorizzare i ricordi in eccesso);
la {doc}`sezione sul context engineering <context-engineering>` lo prende per
intero, dal montaggio dentro un budget ai modi in cui si guasta. Il terzo
cerchio, il più esterno e il più difficile, separa un modello che risponde da
un sistema che porta a termine un compito, e lo tratta la {doc}`sezione sul
loop engineering <loop-engineering>`.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un modello già addestrato di norma non si tocca, perché riaprirlo è la strada
  più cara: si cambia il testo che gli si mette davanti da leggere. Programmare
  a parole è questo, e Karpathy lo chiama «programmare in inglese» (cioè in
  lingua umana: l'italiano va uguale).
- Nel foglio che gli si lascia pesano di più gli esempi: due o tre casi già
  risolti dentro il messaggio lo orientano verso il compito, senza che abbia
  imparato niente di nuovo. Alla fine della risposta è com'era prima.
- Ci sono tre cerchi, uno dentro l'altro: il singolo messaggio che scrivi,
  tutto il testo che il modello ha davanti mentre risponde (il contesto), e la
  conversazione o il processo che ripete il giro più volte correggendo il tiro
  (il loop).
- Si chiama ingegneria e non magia per tre ragioni concrete: c'è un limite
  (nella finestra ci sta solo una certa quantità di testo, e il testo si paga),
  si prova e si confronta invece di fidarsi a occhio, e si tiene traccia
  di cosa si è cambiato, così quando le risposte peggiorano si sa perché.
- Quando il modello non risponde come vuoi, le strade si provano in ordine di
  costo: prima si riscrive il messaggio; se gli mancano dei fatti, glieli si
  va a prendere (il RAG); solo se manca un modo di rispondere che nessuna
  istruzione tiene fermo si riapre il modello (il fine-tuning).
- È un campo giovane: più regole pratiche che leggi. Nessuna frase
  garantisce un risultato; le pratiche buone spostano le probabilità, non
  comandano la macchina. Chi promette certezze sta vendendo qualcosa.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Con un LLM istruito non si programma coi pesi (congelati dall'addestramento)
  ma con le parole: il testo che gli mettiamo davanti. Il congelamento non è
  definitivo, perché riaprirli e proseguire l'addestramento sui propri dati si
  può (è il fine-tuning), ma è la strada più cara da mettere in piedi. Karpathy
  lo chiama Software 3.0 («si programma in inglese») e nel 2025 ha dato
  credito, per questa pratica, al nome context engineering
  {cite}`karpathy2025context`.
- Il meccanismo che lo rende possibile è l’in-context learning: pochi esempi
  nel contesto (*few-shot*) orientano il modello senza aggiornarne i pesi
  {cite}`brown2020language`. Gli esempi non addestrano, condizionano.
- Si programma a parole su tre livelli concentrici: il prompt (il singolo
  messaggio) dentro il contesto (l'intera finestra come sistema) dentro il
  loop (il processo che la ri-riempie a ogni passo). Ogni cerchio contiene il
  precedente.
- Si dice ingegneria e non magia per tre motivi: i vincoli sono reali
  (finestra finita, costo per token), i risultati si misurano e si
  confrontano, e il prompt si versiona; «il prompt è codice», come in
  MLOps.
- Le tre strade si provano in ordine di costo: il prompt per primo; il RAG se
  mancano fatti; il fine-tuning se manca un comportamento che nessuna
  istruzione tiene fermo; è il più caro, non il più avanzato.
- È una disciplina giovane: più euristiche che garanzie, terminologia ancora in
  assestamento. Niente formule magiche: si spostano le probabilità, non si
  comanda l'output.
- Quanto costa lo spazio della finestra di un agente e dove si tiene il
  materiale che non ci entra sta nella {doc}`sezione sul contesto come
  interfaccia </Agenti/context-engineering>`, nel capitolo sugli agenti.
```

`````
