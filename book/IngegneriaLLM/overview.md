# Prompt, contesto e loop: programmare gli LLM

Nel giugno 2025 Andrej Karpathy (tra i fondatori di OpenAI, per anni a capo
dell'intelligenza artificiale in Tesla) ha reso popolare un nome per un
mestiere che esisteva già ma non si sapeva ancora come chiamare. Su X ha
scritto di preferire il termine **context engineering** a «prompt
engineering», e lo ha definito così: l'arte e insieme la scienza,
delicata, di riempire la finestra di contesto con *la giusta informazione per
il passo successivo* {cite}`karpathy2025context`.

Due parole di quella frase vanno sciolte subito, perché torneranno in ogni
pagina del capitolo. Il **prompt** è il messaggio che scriviamo al modello: la
richiesta, più tutto quello che le mettiamo attorno. La **finestra di
contesto** è il tetto di testo che un modello riesce a leggere in una volta
sola: tutto ciò che vogliamo che sappia, prima di rispondere, deve starci
dentro, e quando è piena qualcosa va tolto per far posto. Non è una metafora,
è un limite di progetto del modello, ed è la ragione per cui riempirla bene è
un mestiere.

Anche «il passo successivo» merita una riga: è la mossa dopo, in un lavoro che
va avanti a riprese, e vedremo che è lì che la faccenda si fa difficile. Detto
questo, la definizione di Karpathy si legge da sé: poche parole, ma pesano.
Non dicono «trova la frase magica»: dicono che il lavoro è *riempire bene una
finestra*, e farlo passo dopo passo.

Karpathy aveva preparato il terreno da tempo. Anni prima aveva parlato di
**Software 2.0**: nei sistemi di apprendimento automatico il programma non lo
scrive più una persona riga per riga, lo si *addestra*, cioè gli si mostrano
montagne di esempi e lo si lascia aggiustare da sé, un pochino alla volta, i
numeri che ha dentro. Quei numeri, in una rete neurale, si chiamano **pesi**,
sono milioni, e sono in tutto e per tutto quello che il modello ha imparato:
il suo codice. Nel 2025 ha aggiunto un terzo capitolo, il **Software 3.0**,
osservando che oggi, con i grandi modelli linguistici, «si programma in
inglese»: il prompt *è* il programma, scritto in lingua naturale invece che in
Python. («In inglese» sta per «in lingua umana»: l'italiano va altrettanto
bene, e infatti tutti gli esempi di questo capitolo sono in italiano.) È
un'immagine forte, e come tutte le immagini forti va presa con prudenza; ma
coglie qualcosa di vero, ed è il punto di partenza di questo capitolo.

## Programmare a parole

Prima di vedere come si programma un modello di linguaggio, conviene guardare
che cosa fa davvero quando risponde. Un **modello linguistico di grandi
dimensioni** (in inglese *large language model*, da cui la sigla **LLM**, che
useremo d'ora in poi perché è quella che si incontra ovunque) fa una cosa
sola, e la fa moltissime volte di fila: guarda il testo che ha davanti e stima
quale pezzo di testo verrà dopo.

```{figure} ../figures/cos-e-davvero-un-llm.svg
:name: fig-cos-e-un-llm
:alt: "La frase «Il gatto nero salta sul» entra nel modello, seguita da una casella vuota con un punto interrogativo. Il modello restituisce una graduatoria di parole con la loro probabilità: muro 41 per cento, tetto 20, ramo 11, divano 3, tavolo 2, e un 23 per cento distribuito su tutto il resto del vocabolario. Il modello non produce una risposta ma questa graduatoria: la risposta nasce dopo, pescandone un elemento."
:width: 92%

Cosa restituisce davvero un LLM. Non una frase: una classifica di parole
possibili, ciascuna con la sua probabilità. Tutto ciò che sembra dialogo è
questo passaggio, ripetuto.
```

Tenere presente {numref}`fig-cos-e-un-llm` cambia il modo di leggere tutto il
capitolo. Quello che esce dal modello è una classifica di parole candidate,
ciascuna con la sua percentuale: dopo «il gatto nero salta sul» potrebbe dire
muro 41%, tetto 20%, ramo 11%, e via calando fino alle ultime parole del
vocabolario. La risposta nasce pescando da quella classifica. (Quando la
sentirai chiamare «distribuzione», da qui o da altri, è di questa classifica
che si sta parlando: è la parola che usano gli statistici.) Nella figura le
voci si chiamano
*token*, che è il pezzo di parola con cui il modello lavora davvero, come
abbiamo visto nel capitolo sul linguaggio naturale. Diremo «parola» dove la
differenza non conta, e «token» dove conta: cioè quando si tratta di contarli,
perché è a token che si misura quanto testo entra nella finestra, ed è a token
che si paga.

Questa classifica dipende dal testo che il modello ha davanti: cambia il testo
e i numeri si spostano. Ecco perché «programmare a parole» non è un modo di
dire. Le parole che gli scriviamo sono la leva che sposta la classifica, ed è
l'unica che abbiamo finché non mettiamo le mani dentro al modello.

Da qui una tesi semplice da enunciare e ricca di conseguenze. Con un LLM già
addestrato, cioè uno di quelli che si trovano pronti e che sanno già leggere e
scrivere, noi non programmiamo più toccando i **pesi**: quelli sono congelati,
li ha fissati l'addestramento. «Congelati» non vuol dire immutabili per
sempre: riaprirli e proseguire l'addestramento sui propri dati si può, si
chiama **fine-tuning**, e fra poco vedremo perché è la più cara delle strade
che abbiamo. Vuol dire che nel modo di lavorare di cui parla questo capitolo
restano fermi. Programmiamo con le
**parole**, cioè con il testo che gli mettiamo davanti prima di chiedergli una
risposta. Cambiare quel testo cambia il comportamento del sistema tanto
quanto, nel software tradizionale, cambierebbe riscrivere una funzione.

`````{tab} Elementare

Pensa a un collaboratore bravissimo e velocissimo, che ha letto mezza
biblioteca, ma che è appena arrivato e non sa nulla del *tuo* lavoro. Puoi
anche mandarlo a un corso di formazione, ma è una faccenda lunga e costosa, e
la vedremo fra poco. Quello che puoi fare subito, ogni giorno, gratis, è
**parlargli bene**. Se gli dici «occupati dei clienti» otterrai una cosa; se
gli lasci un foglio con il ruolo, tre esempi di risposte giuste e il tono da
tenere, ne otterrai un'altra, molto migliore: stesso collaboratore, stesso
cervello, solo parole diverse. E nota il dettaglio del foglio, perché tornerà
per tutto il capitolo: quei tre esempi già svolti sono la cosa che lo aiuta di
più, e lui non ha imparato niente, li ha soltanto letti. Ecco cosa vuol dire
«programmare a parole»: non si cambia la persona, si cambia ciò che le si
dice. E siccome le parole giuste fanno un lavoro giusto e quelle sbagliate un
disastro, sceglierle diventa un mestiere.

`````

`````{tab} Superiore

È utile leggere questa tesi lungo la scala che Karpathy chiama Software 1.0 →
2.0 → 3.0. Nel **Software 1.0** il comportamento è codice imperativo scritto a
mano. Nel **Software 2.0** è un vettore di parametri $\theta$ appreso
minimizzando una loss $\mathcal{L}$ su dei dati: il programma emerge
dall'ottimizzazione, non dalla penna del programmatore. Il **Software 3.0**
sposta di nuovo il piano: $\theta$ resta *fisso* (il modello pre-addestrato non
si tocca) e ciò che varia è il **contesto** $C$ passato in ingresso. Il sistema
genera

$$
\hat{y} \sim P_{\theta}(\,\cdot \mid C),
$$

dove $P_{\theta}$ è la distribuzione condizionata calcolata dal modello
congelato, $C$ è tutto il testo che gli forniamo (istruzioni, esempi,
documenti, cronologia) e $\hat{y}$ la risposta, *campionata* da quella
distribuzione, eventualmente riscalata e troncata dai parametri di decoding
(temperatura, top_p) che vedremo nella sezione sul prompt: a temperatura non
nulla, lo stesso $C$ può dare risposte diverse (e non solo a temperatura non
nulla, come vedremo lì). Programmare significa progettare $C$. Il meccanismo che rende
possibile tutto questo è l’**in-context learning**, documentato su larga scala
da Brown e colleghi nel lavoro su GPT-3 {cite}`brown2020language`: bastano poche
coppie richiesta → risposta nel contesto (il *few-shot*), perché il modello
esegua un compito nuovo *senza alcun aggiornamento dei pesi*. Gli esempi non
addestrano: **condizionano**. La scoperta è raccontata nel capitolo sui
Transformer e ne abbiamo scritto la forma probabilistica nella sezione di
context engineering del capitolo sugli **Agenti**; qui ci basta la
conseguenza: la programmazione avviene nel testo.

`````

Detto così, sembra che il tutto si riduca a scrivere una buona frase. È
l'equivoco da cui bisogna liberarsi subito, ed è la ragione per cui la
terminologia è cambiata. Fra noi e il modello, in un'applicazione vera, c'è
sempre un **programma**: il sito, l'assistente, le righe di codice che
raccolgono la nostra richiesta e la spediscono. Il testo che arriva al modello
lo scrive quel programma, e non è una frase: è un carico fatto di parti con
ruoli diversi, montate poco prima di partire (in gergo si chiama **payload**,
appunto il carico). E questo carico va costruito, misurato e ricostruito a
ogni passo. Da qui i tre livelli del capitolo.

## Tre cerchi concentrici

Programmare a parole si fa a tre livelli, uno dentro l'altro come cerchi
concentrici ({numref}`fig-ingegneria-cerchi`), dal più piccolo al più grande.

```{figure} ../figures/ingegneria-llm-cerchi.svg
:name: fig-ingegneria-cerchi
:alt: "Tre cerchi concentrici. Al centro, il più piccolo, porta la scritta \"prompt: il singolo messaggio\". Il cerchio mediano che lo racchiude porta la scritta \"contesto: la finestra come sistema\". Il cerchio esterno che racchiude entrambi porta la scritta \"loop: il processo iterativo\". A destra, una legenda ripete i tre nomi per esteso: prompt engineering, context engineering, loop engineering. In fondo, la riga che spiega il disegno: ogni cerchio contiene il precedente, il prompt nel contesto e il contesto nel loop."
:width: 70%

I tre livelli del «programmare a parole», l'uno dentro l'altro: il **prompt** (il
singolo messaggio) sta dentro il **contesto** (l'intera finestra come sistema),
che a sua volta sta dentro il **loop** (il processo che la ri-riempie a ogni
passo).
```

Il primo cerchio, quello interno, è il **prompt engineering**: il singolo
messaggio. Come si formula un'istruzione perché il modello faccia ciò che
vogliamo: chiarezza, esempi, formato richiesto, il modo di chiedere il
ragionamento passo passo. È il livello a cui si pensa istintivamente, ed è
dove si comincia.

Il secondo cerchio, che racchiude il primo, è il **context engineering**:
l'intera finestra come **sistema**. Qui il prompt dell'utente è solo un pezzo.
Ci sono le istruzioni di fondo, gli esempi, la memoria di ciò che si è detto
prima, i documenti recuperati da un archivio, le descrizioni degli
**strumenti** a disposizione: le operazioni che il modello può chiedere al
programma di eseguire per lui (cercare sul web, interrogare un archivio,
mandare una mail). Gli strumenti sono il cuore del capitolo precedente, quello
sugli **agenti**: un agente è un programma che si serve del modello a più
riprese e che, a ogni ripresa, può lasciargli usare uno di questi strumenti,
finché il compito non è finito. Il lavoro non è più
«scrivere una frase» ma **decidere cosa mettere nella finestra, in quale
ordine, entro quale budget**: perché la finestra è finita, e perché il testo
si paga a quantità. Chi chiama un modello da un programma riceve una fattura
proporzionale al testo che gli manda e a quello che riceve indietro: nella
chat che si usa gratis quel conto lo paga qualcun altro, ma esiste, ed è il
vincolo attorno a cui gira tutto il secondo cerchio.

Il terzo cerchio, il più esterno, è il **loop engineering**: il **processo**.
Un agente non fa una sola chiamata al modello: gira in un ciclo *osserva →
ragiona → agisci*, e a ogni giro la finestra va **ri-riempita e ripulita**. Cosa
portarsi dietro del giro precedente, cosa buttare, come verificare che il
passo sia andato a buon fine, dove tenere quello che il ciclo deve ricordare
e che nella finestra non ci sta. Progettare questo ciclo è il livello più esterno e più difficile.

`````{tab} Elementare

Un'immagine per tenerli insieme. Il **prompt** è la singola domanda che fai a un
esperto: «mi consigli un vino per il pesce?». Il **contesto** è tutto ciò che
l'esperto ha davanti mentre risponde: il menù, la lista dei vini in cantina, il
fatto che gli hai detto di avere un budget, la bottiglia che hai apprezzato la
volta scorsa. Il **loop** è la conversazione intera che ne segue: lui propone,
tu assaggi, storci il naso, lui riparte tenendo conto di com'è andata, e così
via fino a trovare la bottiglia giusta. Domanda dentro il tavolo apparecchiato,
il tavolo dentro la serata: ogni cerchio contiene quello prima.

`````

`````{tab} Superiore

L'inclusione è propria, non metaforica. Il **prompt** è la stringa
d'istruzione; è un *sottoinsieme* del **contesto** $C$, che è l'intero payload
$C = [\,\text{system}, \text{esempi}, \text{memoria}, \text{documenti},
\text{strumenti}, \text{prompt}\,]$ montato prima della chiamata. Il contesto, a
sua volta, è ciò che il **loop** produce e consuma a ogni iterazione: detto
$C_t$ il contesto al passo $t$, $M_t$ la memoria esterna e $o_t$
l'osservazione di ritorno (l'output del modello su $C_t$, il risultato di uno
strumento), il ciclo è

$$
\left(C_{t+1},\; M_{t+1}\right) = g\!\left(C_t,\; M_t,\; o_t\right),
$$

dove $g$ è la politica che aggiorna finestra e memoria: aggiunge
l'osservazione utile, comprime o scarta la cronologia superflua, scrive nella
memoria esterna ciò che va conservato e ne reinietta ciò che serve al passo
dopo. Ottimizzare il singolo prompt senza
governare $g$ significa curare un fotogramma e ignorare il film: il prompt
vive un istante, il loop dura quanto il compito. Ecco perché i tre livelli non
sono alternativi ma **annidati**, e perché conviene affrontarli dal piccolo al
grande.

`````

## Perché «ingegneria» e non «magia»

Chiamarlo *engineering*, ingegneria, non serve a darsi un tono. Serve a
prendere le distanze da come la cosa veniva trattata agli inizi, cioè come una
stregoneria: la caccia alla formula segreta, alla parola che «sblocca» il
modello. Chiamarla ingegneria vuol dire ammettere tre cose poco romantiche ma
vere: che le strade a disposizione **costano**, e in modo diverso; che quello
che si fa **si misura**, invece di fidarsi a occhio; e che di ogni modifica
**si tiene traccia**, come si fa col codice.

La prima si vede meglio in una figura. Quando un modello non fa quello che vogliamo,
davanti abbiamo tre strade, e non costano uguale: riscrivere il messaggio;
andare a prendere i documenti che al
modello mancano e metterglieli davanti insieme alla domanda (è il **RAG**);
oppure riaprire i pesi e riaddestrarlo un po’ sui nostri esempi (il
**fine-tuning**). Le prime due lasciano il modello com'è, la terza lo cambia,
ed è la più cara di tutte.

```{figure} ../figures/fine-tuning-rag-o-prompt.svg
:name: fig-quale-leva
:alt: "Schema a domande, un albero di decisione, che parte dal problema e passa subito dal prompt, indicato come la leva più economica e la prima da provare. Solo se il prompt non basta si scende alle due domande successive: se al modello mancano fatti o documenti propri la risposta è il RAG; se invece serve un comportamento costante che il prompt non riesce a tenere, è il fine-tuning; altrimenti si torna a riscrivere il prompt e a misurare."
:width: 88%

Tre leve, in ordine di costo, ed è l'ordine in cui si provano. La prima
domanda non è «quale tecnica è migliore» ma «il prompt ha già fallito?»; solo
dopo viene «cosa manca davvero».
```

L'ordine delle domande in {numref}`fig-quale-leva` dice già come ragiona un
ingegnere. Si comincia dalla leva che costa meno e si sale solo se serve: il
fine-tuning non è più avanzato del prompt, è più caro (vuole esempi raccolti a
mano, macchine per addestrare, e va rifatto ogni volta che si cambia modello),
e va giustificato da qualcosa che il prompt non poteva dare. La domanda che fa
scegliere, quando il prompt non basta, è che cosa manchi davvero. Se mancano
dei *fatti* (un manuale, l'archivio degli ordini, i dati di casa nostra) la
risposta è il RAG, sigla di
*retrieval augmented generation*, cioè «generazione con recupero». Se invece
manca un *comportamento*, un modo di rispondere che nessuna istruzione riesce
a tenere fermo, allora è il caso del fine-tuning. Sono due strade che il libro
ha già percorso altrove, e qui ci basta sapere quando si imboccano.

`````{tab} Elementare

La differenza tra un incantesimo e un mestiere sta in tre parole:
**vincoli**, **misura**, **versioni**. Un incantesimo lo pronunci e speri; un
mestiere fa i conti con dei limiti (nella finestra ci sta solo una certa
quantità di testo, e più ne metti più paghi), controlla se ha funzionato (provi,
guardi i risultati, tieni quello che va meglio) e tiene traccia di cosa ha
cambiato (così, se oggi le risposte peggiorano, sai che è stata la modifica di
ieri). Chi lavora bene con gli LLM non «indovina la frase»: prova, misura,
corregge. È noioso come tutta l'ingegneria, ed è per questo che funziona.

Una cosa però va detta subito, ed è un'avvertenza che vale per tutto il
capitolo: è un mestiere nato ieri. Non ci sono leggi, ci sono regole del
pollice che spesso funzionano e ogni tanto no. Chi ti promette la frase che
funziona sempre ti sta vendendo qualcosa.

`````

`````{tab} Superiore

Tre proprietà rendono l'attività ingegneristica e non magica. **Primo, i
vincoli sono reali e quantificabili**: la finestra ha un tetto di token, e
ogni token pesa su latenza, memoria (la KV cache vista nel capitolo sui
Transformer) e denaro, e del **costo per token** si occupa la sezione su
LLMOps, nel capitolo su MLOps. Non si ottimizza «la qualità» in astratto ma la
qualità *sotto vincolo
di budget*. **Secondo, si misura**: una versione del prompt o della politica
di contesto si valuta su una batteria di casi e si confronta (A/B) con la
precedente prima di sostituirla, senza misura non c'è miglioramento, solo
opinioni. **Terzo, si versiona**: come abbiamo anticipato nel capitolo sugli
agenti, *il prompt è codice*, va messo sotto controllo di versione e trattato
come un artefatto del software, tema che ritroveremo in MLOps. Con una
avvertenza di onestà intellettuale: è un'ingegneria **giovane**, fatta oggi
più di euristiche che di garanzie. La terminologia stessa è in assestamento,
per un buon tratto si è chiamato tutto «prompt engineering», finché non si è
capito che il problema vero stava un cerchio più in fuori. Chi promette leggi
certe, in questo campo, sta vendendo qualcosa.

`````

Conviene ripetere l'avvertenza, perché il tono di questo capitolo dipende da
lei. Non esistono ricette che garantiscano la risposta; esistono pratiche che
*spostano le probabilità* nella direzione giusta, e si riconoscono perché sono
misurabili e ripetibili. Tratteremo prompt, contesto e loop con questo
spirito: niente formule magiche, molte euristiche oneste, e la costante
consapevolezza dei limiti (la finestra finita, il costo, l'incertezza di fondo
di un modello che *stima* la parola successiva e non la *sa*).

## Tre mestieri intorno allo stesso modello

Il capitolo segue i tre cerchi, dal centro verso l'esterno.

- **Prompt engineering: il singolo messaggio**, come si scrive un prompt che
  funziona: struttura, esempi, richiesta esplicita del ragionamento passo
  passo (la *chain-of-thought* {cite}`wei2022chain`), forma della risposta, e
  le fragilità da conoscere.
- **Context engineering: la finestra come sistema**, cioè tutto il carico come
  oggetto da progettare: cosa entra nella finestra, in quale ordine, entro
  quale budget. Di questa materia il capitolo sugli **agenti** ha già smontato
  la meccanica: come si sceglie cosa mettere in uno spazio che non basta per
  tutto, perché un testo lungo si fa trascurare proprio nel mezzo, dove tenere
  i ricordi che nella finestra non stanno. Qui la riprendiamo dal lato del
  progetto, e quella costruzione non la rifacciamo.
- **Loop engineering: progettare il ciclo**, il processo iterativo che
  ri-riempie e ripulisce la finestra a ogni passo, con verifica dei risultati
  e stato tenuto fuori dalla finestra: il cerchio più esterno, quello che fa
  la differenza tra un LLM che risponde e un sistema che porta a termine un
  compito.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un modello già addestrato non si cambia più: quello che si cambia è **ciò che
  gli si mette davanti da leggere**. Programmare a parole vuol dire questo, e
  Karpathy lo ha chiamato «programmare in inglese» (cioè in lingua umana:
  l'italiano va uguale).
- Nel foglio che gli si lascia, la parte che pesa di più sono **gli esempi**:
  due o tre casi già risolti dentro il messaggio lo orientano verso il compito,
  senza che abbia imparato niente di nuovo. Alla fine della risposta è
  esattamente com'era prima.
- Ci sono **tre cerchi**, uno dentro l'altro: il singolo **messaggio** che
  scrivi, tutto quello che il modello ha davanti mentre risponde (il
  **contesto**), e la **conversazione** o il processo che ripete la cosa più
  volte correggendo il tiro (il **loop**).
- Si chiama ingegneria e non magia per tre ragioni concrete: c'è un **limite**
  (nella finestra ci sta solo una certa quantità di testo, e il testo si paga),
  si **prova e si confronta** invece di fidarsi a occhio, e si **tiene traccia**
  di cosa si è cambiato, così quando le risposte peggiorano si sa perché.
- È un mestiere **giovane**: più regole pratiche che leggi. Nessuna frase
  garantisce un risultato; le pratiche buone spostano le probabilità, non
  comandano la macchina. Chi promette certezze sta vendendo qualcosa.
- Il capitolo va **dal centro verso l'esterno**: prima il messaggio, poi la
  finestra, poi il ciclo.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Con un LLM istruito non si programma coi **pesi** (congelati
  dall'addestramento) ma con le **parole**: il testo che gli mettiamo davanti.
  Karpathy lo chiama **Software 3.0** («si programma in inglese») e nel 2025 ha
  reso popolare per quel mestiere il nome **context engineering**
  {cite}`karpathy2025context`.
- Il meccanismo che lo rende possibile è l’**in-context learning**: pochi esempi
  nel contesto (*few-shot*) orientano il modello senza aggiornarne i pesi
  {cite}`brown2020language`. Gli esempi non addestrano, **condizionano**.
- Si programma a parole su **tre livelli concentrici**: il **prompt** (il singolo
  messaggio) dentro il **contesto** (l'intera finestra come sistema) dentro il
  **loop** (il processo che la ri-riempie a ogni passo). Ogni cerchio contiene il
  precedente.
- Si dice **ingegneria** e non magia per tre motivi: i **vincoli** sono reali
  (finestra finita, costo per token), i risultati si **misurano** e si
  confrontano, e il prompt si **versiona**; «il prompt è codice», come in
  MLOps.
- È un mestiere **giovane**: più euristiche che garanzie, terminologia ancora in
  assestamento. Onestà sui limiti, zero formule magiche: si spostano le
  probabilità, non si comanda l'output.
- Il capitolo procede dal centro verso l'esterno (prompt, contesto, loop)
  rimandando alla sezione «context engineering» del capitolo sugli **agenti**
  per la meccanica del budget, che qui non si ripete.
```

`````
