# Prompt, contesto e loop: programmare gli LLM

Nel giugno 2025 Andrej Karpathy (tra i fondatori di OpenAI, per anni a capo
dell'intelligenza artificiale in Tesla) ha reso popolare un nome per un
mestiere che esisteva già ma non si sapeva ancora come chiamare. Su X ha
scritto che preferisce di gran lunga il termine **context engineering** a
«prompt engineering», e lo ha definito così: l'arte e insieme la scienza,
delicata, di riempire la finestra di contesto con *la giusta informazione per
il passo successivo* {cite}`karpathy2025context`. Poche parole, ma pesano. Non
dicono «trova la frase magica»: dicono che il lavoro è *riempire bene una
finestra*, e farlo passo dopo passo.

Karpathy aveva preparato il terreno da tempo. Anni prima aveva parlato di
**Software 2.0**: nei sistemi di apprendimento automatico il programma non lo
scrive più una persona riga per riga, lo si *addestra*, e il codice sono i
pesi della rete. Nel 2025 ha aggiunto un terzo capitolo, il **Software 3.0**,
osservando che oggi, con i grandi modelli linguistici, «si programma in
inglese»: il prompt *è* il programma, scritto in lingua naturale invece che in
Python. È un'immagine forte, e come tutte le immagini forti va presa con
prudenza; ma coglie qualcosa di vero, ed è il punto di partenza di questo
capitolo.

## Programmare a parole

```{figure} ../figures/cos-e-davvero-un-llm.svg
:name: fig-cos-e-un-llm
:alt: "Una sequenza di testo entra nel modello, che restituisce una distribuzione di probabilità su tutte le possibili parole successive. Il modello non produce una risposta ma una graduatoria: la risposta nasce dopo, scegliendo un elemento da quella graduatoria."
:width: 92%

Cosa restituisce davvero un LLM. Non una frase: una distribuzione sulla parola
dopo. Tutto ciò che sembra dialogo è questo passaggio, ripetuto.
```

Tenere presente {numref}`fig-cos-e-un-llm` cambia il modo di leggere tutto il
capitolo. «Distribuzione» è una parola da statistici per una cosa semplice:
una classifica di parole candidate, ciascuna con la sua percentuale (dopo «il
gatto nero salta sul» potrebbe dire: muro 41%, tetto 20%, divano 3%), e la
risposta nasce pescando da quella classifica. Se l'oggetto è questa
classifica, condizionata dal testo che precede, allora «programmare a parole»
non è una metafora: è il modo letterale di spostarla, ed è l'unico che si
abbia senza toccare i pesi.

La tesi è semplice da enunciare e ricca di conseguenze. Con un LLM già
addestrato (un modello «istruito», che sa già leggere e scrivere), noi non
programmiamo più toccando i **pesi**: quelli sono congelati, li ha fissati
l'addestramento. Programmiamo con le **parole**, cioè con il testo che gli
mettiamo davanti prima di chiedergli una risposta. Cambiare quel testo cambia
il comportamento del sistema tanto quanto, nel software tradizionale,
cambierebbe riscrivere una funzione.

`````{tab} Elementare

Pensa a un collaboratore bravissimo e velocissimo, che ha letto mezza
biblioteca, ma che è appena arrivato e non sa nulla del *tuo* lavoro. Non puoi
mandarlo a scuola: è già «finito», non impara più di così. Quello che
puoi fare è **parlargli bene**. Se gli dici «occupati dei clienti» otterrai
una cosa; se gli lasci un foglio con il ruolo, tre esempi di risposte giuste e
il tono da tenere, ne otterrai un'altra, molto migliore: stesso collaboratore,
stesso cervello, solo parole diverse. Ecco cosa vuol dire «programmare a
parole»: non si cambia la persona, si cambia ciò che le si dice. E siccome le
parole giuste fanno un lavoro giusto e quelle sbagliate un disastro,
sceglierle diventa un mestiere.

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
nulla, lo stesso $C$ può dare risposte diverse. Programmare significa progettare $C$. Il meccanismo che rende
possibile tutto questo è l'**in-context learning**, isolato su larga scala da
Brown e colleghi nel lavoro su GPT-3 {cite}`brown2020language`: bastano poche
coppie richiesta → risposta nel contesto (il *few-shot*), perché il modello
esegua un compito nuovo *senza alcun aggiornamento dei pesi*. Gli esempi non
addestrano: **condizionano**. Ne abbiamo dato la forma probabilistica nel
capitolo sui Transformer; qui ci basta la conseguenza: la programmazione
avviene nel testo.

`````

Detto così, sembra che il tutto si riduca a scrivere una buona frase. È
l'equivoco da cui bisogna liberarsi subito, ed è la ragione per cui la
terminologia è cambiata. Nelle applicazioni serie il testo che diamo al modello
non è una frase: è un **payload** montato dal programma, fatto di parti con
ruoli diversi. E questo payload va costruito, misurato e ricostruito a ogni
passo. Da qui i tre livelli del capitolo.

## Tre cerchi concentrici

Programmare a parole si fa a tre livelli, uno dentro l'altro come cerchi
concentrici ({numref}`fig-ingegneria-cerchi`), dal più piccolo al più grande.

```{figure} ../figures/ingegneria-llm-cerchi.svg
:name: fig-ingegneria-cerchi
:alt: "Tre cerchi concentrici. Al centro, il cerchio più piccolo, è etichettato \"prompt\": il singolo messaggio inviato al modello. Il cerchio mediano che lo racchiude è etichettato \"contesto\": l'intera finestra passata al modello, comprendente istruzioni di sistema, esempi, memoria, documenti recuperati e descrizioni degli strumenti, con il prompt come sua parte. Il cerchio esterno che racchiude entrambi è etichettato \"loop\": il processo iterativo che a ogni passo ricostruisce e ripulisce il contesto, con verifica dei risultati e stato conservato all'esterno della finestra. Le frecce lungo il cerchio esterno indicano la ciclicità del processo."
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
prima, i documenti recuperati da un archivio, le descrizioni degli strumenti a
disposizione. Il lavoro non è più «scrivere una frase» ma **decidere cosa
mettere nella finestra, in quale ordine, entro quale budget**: perché la
finestra è finita e ogni parola costa.

Il terzo cerchio, il più esterno, è il **loop engineering**: il **processo**.
Un LLM che lavora davvero (un agente, come abbiamo visto nel capitolo
dedicato) non fa una sola chiamata: gira in un ciclo *osserva → ragiona →
agisci*, e a ogni giro la finestra va **ri-riempita e ripulita**. Cosa
portarsi dietro del giro precedente, cosa buttare, come verificare che il
passo sia andato a buon fine, dove tenere lo stato che non entra nella
finestra. Progettare questo ciclo è il livello più esterno e più difficile.

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

La parola *engineering* non è un vezzo. Segnala un cambio di atteggiamento
verso qualcosa che, agli inizi, veniva trattato come stregoneria: la caccia
alla formula segreta, alla parola che «sblocca» il modello. Chiamarla
ingegneria vuol dire ammettere tre cose poco romantiche ma vere.

```{figure} ../figures/fine-tuning-rag-o-prompt.svg
:name: fig-quale-leva
:alt: "Albero di decisione che parte dal problema. Se al modello mancano fatti o documenti propri, la risposta è il RAG. Altrimenti, se serve un formato o uno stile costante, è il fine-tuning. Altrimenti si resta sul prompt, che è la leva più economica e la prima da provare."
:width: 92%

Tre leve, in ordine di costo. La prima domanda non è «quale tecnica è
migliore» ma «cosa manca davvero», e le tre risposte portano a strumenti
diversi.
```

L'ordine delle domande in {numref}`fig-quale-leva` è già una posizione
ingegneristica, ed è la prima delle tre cose poco romantiche. Si comincia
dalla leva che costa meno e si sale solo se serve: il fine-tuning non è più
avanzato del prompt, è più caro, e va giustificato da qualcosa che il prompt
non poteva dare.

`````{tab} Elementare

La differenza tra un incantesimo e un mestiere sta in tre parole:
**vincoli**, **misura**, **versioni**. Un incantesimo lo pronunci e speri; un
mestiere fa i conti con dei limiti (nella finestra ci sta solo una certa
quantità di testo, e più ne metti più paghi), controlla se ha funzionato (provi,
guardi i risultati, tieni quello che va meglio) e tiene traccia di cosa ha
cambiato (così, se oggi le risposte peggiorano, sai che è stata la modifica di
ieri). Chi lavora bene con gli LLM non «indovina la frase»: prova, misura,
corregge. È noioso come tutta l'ingegneria, ed è per questo che funziona.

`````

`````{tab} Superiore

Tre proprietà rendono l'attività ingegneristica e non magica. **Primo, i
vincoli sono reali e quantificabili**: la finestra ha un tetto di token, e
ogni token pesa su latenza, memoria (la KV cache vista nel capitolo sui
Transformer) e denaro; il **costo per token** che è materia del capitolo su
LLMOps. Non si ottimizza «la qualità» in astratto ma la qualità *sotto vincolo
di budget*. **Secondo, si misura**: una versione del prompt o della politica
di contesto si valuta su una batteria di casi e si confronta (A/B) con la
precedente prima di sostituirla, senza misura non c'è miglioramento, solo
opinioni. **Terzo, si versiona**: come abbiamo anticipato nel capitolo sugli
agenti, *il prompt è codice*, va messo sotto controllo di versione e trattato
come un artefatto del software, tema che ritroveremo in LLMOps. Con una
avvertenza di onestà intellettuale: è un'ingegneria **giovane**, fatta oggi
più di euristiche che di garanzie. La terminologia stessa è in assestamento,
per un buon tratto si è chiamato tutto «prompt engineering», finché non si è
capito che il problema vero stava un cerchio più in fuori. Chi promette leggi
certe, in questo campo, sta vendendo qualcosa.

`````

Vale la pena ripetere l'avvertenza, perché il tono di questo capitolo dipende
da lei. Non esistono ricette che garantiscano l'output; esistono pratiche che
*spostano le probabilità* nella direzione giusta, e si riconoscono perché sono
misurabili e ripetibili. Tratteremo prompt, contesto e loop con questo
spirito: niente formule magiche, molte euristiche oneste, e la costante
consapevolezza dei limiti (la finestra finita, il costo, l'incertezza di fondo
di un modello che *stima* la parola successiva e non la *sa*).

## Come è organizzato il capitolo

Il capitolo segue i tre cerchi, dal centro verso l'esterno.

- **Prompt engineering: il singolo messaggio**, come si scrive un prompt che
  funziona: struttura, esempi, richiesta esplicita del ragionamento passo
  passo (la *chain-of-thought* {cite}`wei2022chain`), formato dell'output, e
  le fragilità da conoscere.
- **Context engineering: la finestra come sistema**, l'intero payload come
  oggetto da progettare: cosa entra nella finestra, in quale ordine, entro
  quale budget. Ne abbiamo già visto la **meccanica** (il *context builder*
  come problema di zaino, il *lost in the middle* {cite}`liu2024lost`, le
  forme di memoria) nella sezione «context engineering» del capitolo sugli
  **agenti**; qui la riprendiamo dall'angolazione del *progetto*, non del
  ciclo agentico, senza ripeterne la costruzione.
- **Loop engineering: progettare il ciclo**, il processo iterativo che
  ri-riempie e ripulisce la finestra a ogni passo, con verifica dei risultati
  e stato tenuto fuori dalla finestra: il cerchio più esterno, quello che fa
  la differenza tra un LLM che risponde e un sistema che porta a termine un
  compito.

```{admonition} Da ricordare
:class: important
- Con un LLM istruito non si programma coi **pesi** (congelati
  dall'addestramento) ma con le **parole**: il testo che gli mettiamo davanti.
  Karpathy lo chiama **Software 3.0** («si programma in inglese») e nel 2025
  ha ribattezzato il mestiere **context engineering**
  {cite}`karpathy2025context`.
- Il meccanismo che lo rende possibile è l'**in-context learning**: pochi esempi
  nel contesto (*few-shot*) orientano il modello senza aggiornarne i pesi
  {cite}`brown2020language`. Gli esempi non addestrano, **condizionano**.
- Si programma a parole su **tre livelli concentrici**: il **prompt** (il singolo
  messaggio) dentro il **contesto** (l'intera finestra come sistema) dentro il
  **loop** (il processo che la ri-riempie a ogni passo). Ogni cerchio contiene il
  precedente.
- Si dice **ingegneria** e non magia per tre motivi: i **vincoli** sono reali
  (finestra finita, costo per token), i risultati si **misurano** e si
  confrontano, e il prompt si **versiona**; «il prompt è codice», come in
  LLMOps.
- È un mestiere **giovane**: più euristiche che garanzie, terminologia ancora in
  assestamento. Onestà sui limiti, zero formule magiche: si spostano le
  probabilità, non si comanda l'output.
- Il capitolo procede dal centro verso l'esterno (prompt, contesto, loop)
  rimandando alla sezione «context engineering» del capitolo sugli **agenti**
  per la meccanica del budget, che qui non si ripete.
```
