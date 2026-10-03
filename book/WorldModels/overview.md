# World Model

```{image} ../figures/aperture/world-models.png
:class: pt-apertura only-light
:width: 100%
:alt: Un gatto acquattato su un tavolino sta per saltare su una mensola, e un arco a puntini traccia il salto che ha già immaginato.
```

```{image} ../figures/aperture/world-models-scura.png
:class: pt-apertura only-dark
:width: 100%
:alt: Un gatto acquattato su un tavolino sta per saltare su una mensola, e un arco a puntini traccia il salto che ha già immaginato.
```

Un gatto di casa ha molto più senso comune e comprensione del mondo di qualunque
modello di linguaggio. A ripeterlo da anni, con poche variazioni, in conferenze
e interviste, non è uno scettico qualsiasi ma Yann LeCun, premio Turing 2018 e
uno dei padri del deep learning. Mentre mezzo mondo si stupiva di ciò che sanno
scrivere i grandi modelli di linguaggio, uno dei loro nonni intellettuali
indicava un gatto. Provocazione calcolata, certo. Ma proviamo a prenderla sul
serio: che cosa sa fare, un gatto? Non risolve integrali e non scrive sonetti;
però salta sul mobile calibrando la traiettoria al primo colpo, prevede da che
parte sbucherà il gomitolo rotolato sotto il divano, e se una mossa è finita
male non la ripete tale e quale.

E prima del gatto, il bambino. A pochi mesi di vita un neonato si stupisce
quando un giocattolo nascosto da uno schermo, una volta abbassato lo schermo,
non c'è più; entro il primo anno si stupisce se un oggetto resta sospeso a
mezz'aria invece di cadere. Lo stupore si misura da quanto a lungo guarda la
scena impossibile rispetto a una possibile (è il metodo della *violazione
dell'aspettativa*), e quello che misura è l'attesa, non la sua origine. Nessuno
gli ha spiegato la permanenza degli oggetti o la gravità, e nessuno gli ha
mostrato milioni di esempi etichettati; quanto di quell'attesa venga dal
guardare e quanto sia già lì alla nascita, però, gli psicologi dello sviluppo
lo discutono, e c'è chi sostiene che i primi sistemi con cui il bambino si
rappresenta gli oggetti siano innati {cite}`spelke2007core`. LeCun scommette
sulla prima lettura: guardando, il bambino si costruisce dentro qualcosa che
gli permette di *aspettarsi* il mondo, cioè un modello.

Col metro del {doc}`dibattito sul rinforzo </AutoSupervisione/dibattito-rl>`,
quanta informazione porta il segnale, il bambino che guarda ne ha uno
ricchissimo e gratuito: a ogni istante il mondo gli dice se aveva previsto
bene. Dare alle macchine qualcosa di simile, un **world model** (modello del
mondo), vuol dire dar loro lo stesso maestro: prevedere che cosa succede dopo
e, per chi deve agire, che cosa succede dopo un'azione. È un tentativo in
corso, e resta aperto il dibattito su quanto sia davvero il pezzo mancante
dell'intelligenza artificiale.

## Un modello in scala ridotta della realtà

L'idea è molto più vecchia del deep learning. Kenneth Craik, filosofo e
psicologo scozzese, la mette nero su bianco nel 1943 in un libro breve e
fulminante, *The Nature of Explanation* {cite}`craik1943nature`: se un
organismo porta nella testa «un "modello in scala ridotta" della realtà
esterna e delle proprie possibili azioni», scrive, allora può «provare diverse
alternative, concludere quale sia la migliore e reagire alle situazioni future
prima che si presentino». Craik morì due anni dopo, a trentun anni, ma quella
pagina è considerata l'atto di nascita dei *modelli mentali* nelle scienze
cognitive, ed è ancora oggi la definizione più limpida di world model: un
simulatore interno che serve a prevedere («cosa succede se lascio il
bicchiere?») e quindi a pianificare, senza dover provare tutto per
davvero.

Su quella parola, «simulatore», bisogna intendersi, perché torna spesso. Il
simulatore di volo su cui si esercitano i piloti l'hanno scritto degli
ingegneri, un'equazione dell'aria alla volta, ed è anch'esso un modello del
mondo nel senso di Craik. Nella letteratura che segue, però, «world model»
indica un simulatore *appreso*: le regole le ricava dai dati, e resta sempre
una copia approssimata. Serve alla stessa cosa (esercitarsi senza conseguenze),
ma sbaglia per ragioni diverse: un simulatore scritto a mano sbaglia dove le
sue equazioni semplificano, uno appreso dove i dati non l'hanno mai portato.
Che forma prendano quegli sbagli lo racconta la {doc}`sezione sui mondi in
miniatura </WorldModels/mondi-in-miniatura>`.

`````{tab} Elementare

Lo usi già, questo simulatore. Quando giochi a scacchi, prima di toccare il
pezzo ragioni così: «se sposto la torre lì, lui la mangia con l'alfiere...
allora no», e la mossa cattiva muore nella tua testa, senza costarti la
partita. Quando parcheggi, giri il volante e *vedi già* l'arco che il paraurti
disegnerà: se l'auto immaginata finisce sul marciapiede, correggi prima che ci
finisca quella vera.

Guarda però che cosa ti serve, per prevedere quell'arco. Del cortile ti arriva
un'immagine, quella che passa dal finestrino, e di quell'immagine tieni tre
cose (dove finisce il muro, quanto spazio resta, dov'è appoggiata la bici); il
colore delle persiane lo butti via, che per la manovra non conta niente. E da
un'occhiata sola non sapresti dire se quella bici è ferma o ti sta arrivando
addosso: lo sai perché la stai seguendo da qualche secondo, e quel filo che
tieni mentre guardi conta quanto l'immagine.

Il modello del mondo è questo cinema interiore in cui il futuro si prova a
costo zero. Non è perfetto (la manovra immaginata a volte finisce comunque con
una strisciata) ma ogni volta che la realtà ti smentisce, il cinema interiore
si aggiorna e la prossima previsione è un po’ migliore.

`````

`````{tab} Superiore

Nel linguaggio della {doc}`sezione sui processi decisionali di
Markov </ReinforcementLearning/mdp-valore>`: l'ambiente è un MDP
con dinamica $P(s' \mid s, a)$, che gli algoritmi di pianificazione (value
iteration, policy iteration) assumevano nota e che Q-learning e DQN
aggiravano imparando direttamente i valori dall'esperienza. Un world model è
la terza via: una stima appresa della dinamica,

$$
p_\theta(s_{t+1} \mid s_t, a_t),
$$

dove $s_t$ e $a_t$ sono lo stato e l'azione al tempo $t$, $s_{t+1}$ lo stato
successivo e $\theta$ i parametri (tipicamente di una rete neurale) stimati
dalle transizioni osservate; spesso si apprende anche un modello della
ricompensa $r_\theta(s_t, a_t)$. Un modello così abilita tre operazioni:
predizione (srotolare traiettorie future senza toccare l'ambiente),
pianificazione (cercare, tra le traiettorie immaginate, quella con il ritorno
più alto) e simulazione di alternative («e se agissi diversamente?»). Su
quest'ultima una precisazione da manuale di causalità: ri-simulare da $s_t$ con
un'altra azione è, nel lessico di Pearl, un *intervento* nel modello; il
controfattuale in senso stretto («cosa *sarebbe* successo in *quella*
traiettoria») chiederebbe invece di tenere fisso il caso già uscito (di riusare
cioè la stessa realizzazione del rumore esogeno di quella traiettoria, non di
ri-estrarlo) e di ri-simulare cambiando la sola azione; le due cose coincidono
quando la dinamica è deterministica. C'è però un dettaglio che occuperà mezzo
capitolo: nel mondo reale lo stato non si osserva. Si osservano pixel, suoni,
letture di sensori: un'osservazione $\mathbf{x}_t$ ad alta dimensione e piena
di dettagli irrilevanti. I world model moderni imparano perciò due oggetti
distinti: un codice compatto della singola osservazione, $\mathbf{z}_t =
f_\phi(\mathbf{x}_t)$ con $f_\phi$ un encoder appreso, e una memoria
$\mathbf{h}_t$ che riassume la storia precedente. Da un solo fotogramma
mancherebbero, per dire, le velocità: è $\mathbf{h}_t$ a portarle, ed è la
memoria della prossima sezione. Lo stato del modello è quindi la coppia
$(\mathbf{z}_t, \mathbf{h}_t)$ e la dinamica si scrive
$p_\theta(\mathbf{z}_{t+1} \mid \mathbf{z}_t, a_t, \mathbf{h}_t)$; nell'RSSM
(*recurrent state-space model*, il modello ricorrente a spazio di stati che i
Dreamer adottano) i due oggetti sopravvivono con gli stessi nomi,
$\mathbf{h}_t$ deterministico e $\mathbf{z}_t$ stocastico condizionato su di
esso. Che cosa debba finire in $\mathbf{z}_t$ (e che cosa sia giusto lasciar
fuori) è una delle domande centrali del capitolo.

`````

## Immaginare costa meno che provare

La prima ragione per volere un world model si chiama efficienza nei campioni
(*sample efficiency*): quante interazioni con l'ambiente servono per imparare un
compito. Il {doc}`capitolo sul Deep Reinforcement
Learning </DeepReinforcementLearning/overview>` ne ha mostrato il prezzo. Il DQN
arrivava al livello di un collaudatore umano professionista su molti dei 49
giochi Atari del confronto, ma dopo decine di milioni di fotogrammi per titolo,
più di un mese di gioco senza mai staccare; il collaudatore con cui era
misurato si era esercitato circa due ore per gioco {cite}`mnih2015human`. Il
vocabolario per questa differenza esiste da decenni
{cite}`sutton2018reinforcement`. I metodi *model-free* imparano valori o
strategia direttamente dall'esperienza, e ogni aggiornamento consuma
interazione vera; i metodi *model-based* imparano anche un modello di come
l'ambiente risponde alle azioni, e lo usano per pianificare o per generare
esperienza simulata, che costa soltanto calcolo.

`````{tab} Elementare

Nessuna compagnia aerea fa esercitare le emergenze (un motore in fiamme, una
raffica in atterraggio) su un aereo vero: si usa il simulatore, dove un errore
non costa niente e la stessa situazione si può ripetere cento volte in un
pomeriggio. Il DQN di quel capitolo è un allievo senza simulatore: ogni cosa
che impara la impara schiantandosi per davvero, e per questo gli servono quelle
decine di milioni di fotogrammi, partite su partite per settimane. Tu no: dopo
qualche pallina persa a
*Breakout* hai già in testa un piccolo *Breakout* tascabile
(«se la racchetta è qui e la pallina scende lì, la manco») e le mosse le
ripassi lì dentro, gratis. Chi possiede un simulatore interno spreme da ogni
esperienza vera decine di esperienze immaginate.

Il prezzo si paga quando il simulatore è impreciso, e si paga a rate. Il
*Breakout* tascabile sbaglia di poco: dopo un rimbalzo la pallina immaginata è
quasi dove sarà davvero, dopo cinque rimbalzi quel «quasi» è mezzo schermo, e
la racchetta che avevi preparato aspetta nel posto sbagliato. Non tutti gli
sbagli si allargano così: l'aereo del simulatore, se lo lasci andare, torna in
assetto da solo, e lì lo scarto si riassorbe invece di crescere. Il guaio
peggiore è un altro: se nel tuo *Breakout* mentale il muro ha un buco che nel
gioco vero non c'è, ti alleni a infilarci la pallina e diventi bravissimo a un
gioco che non esiste.

`````

`````{tab} Superiore

Un metodo model-free (Q-learning, DQN, policy gradient) apprende
direttamente $Q(s, a; \theta)$ oppure $\pi_\theta(a \mid s)$: ogni
aggiornamento consuma interazione reale, e la *sample efficiency* è
notoriamente il suo tallone d'Achille. Un metodo model-based apprende
prima $p_\theta(s_{t+1} \mid s_t, a_t)$ e poi lo usa in due modi: per
pianificare (cercare azioni buone dentro il modello, come la value
iteration faceva sul modello vero) o per generare esperienza sintetica su
cui allenare valori e policy (l'architettura Dyna di Sutton, che già nel 1990
alternava passi vissuti e passi immaginati {cite}`sutton1990integrated`, e che
la {doc}`sezione sul reinforcement learning basato su modello
</DeepReinforcementLearning/model-based>` tratta per esteso). Il prezzo è il
*model bias*. Se la dinamica vera è $L$-Lipschitz nello stato e il modello
sbaglia al più di $\epsilon$ in ogni stato raggiunto, a parità di azioni lo
scarto $e_k$ fra stato immaginato e stato vero soddisfa
$e_k \le \epsilon + L\,e_{k-1}$ con $e_0 = 0$, quindi
$e_k \le \epsilon \sum_{i=0}^{k-1} L^{i}$: cresce esponenzialmente per $L > 1$,
linearmente per $L = 1$, e resta sotto $\epsilon/(1-L)$ per $L < 1$ (la
derivazione, con i suoi limiti, è nella sezione appena citata). Peggio, una
policy ottimizzata dentro il modello impara a sfruttarne i difetti (*model
exploitation*), ottenendo ritorni immaginari che l'ambiente vero non paga. Gran
parte del capitolo è il racconto di come la ricerca ha negoziato questo
compromesso: quanta fiducia concedere al sogno, e per quanti passi.

`````

C'è poi una seconda ragione, meno contabile e più profonda. Il **senso
comune** che LeCun rivendica al gatto non è un elenco di fatti, ma un
repertorio di previsioni: le cose non sostenute cadono, ciò che è nascosto
continua a esistere, i liquidi si versano, gli oggetti spinti si muovono. È la
*fisica intuitiva* che il neonato dell'incipit costruisce guardando, senza
etichette, e il segnale di cui si nutre è la sorpresa, nel senso che
l’{doc}`apertura del capitolo sull'auto-supervisione
</AutoSupervisione/overview>` le ha dato: quanto era improbabile, per il
modello che il bambino si è fatto, quello che poi accade. È una lezione
auto-supervisionata, perché il bersaglio non lo scrive nessuno: è il futuro
stesso che arriva. Se il senso comune è fatto così, inseguirlo significa
costruire macchine che imparano a prevedere il mondo, non a memorizzarlo.

## La scommessa di LeCun (e chi non è d'accordo)

Nel 2022 LeCun mette online, aperto ai commenti di chiunque, un documento di
una sessantina di pagine: *A Path Towards Autonomous Machine Intelligence*
{cite}`lecun2022path`. È un programma di ricerca e non un articolo di
risultati, cioè il disegno di come andrebbe costruita una macchina che si
arrangia da sola nel mondo. Il disegno è fatto di pezzi che si passano il
lavoro: uno guarda, uno ricorda, uno propone la mossa, uno pianifica provando
le alternative. Al centro c'è un modello del mondo, imparato guardando e senza
etichette. La tesi
ha una faccia costruttiva (come *dovrebbe* essere fatta un'intelligenza
artificiale che capisce il mondo) e una polemica: i modelli di linguaggio
autoregressivi, addestrati solo a indovinare la parola successiva, per quanto
grandi non basteranno. LeCun ha legato a questa tesi anche la propria carriera:
alla fine del 2025 ha lasciato Meta, dove nel 2013 aveva fondato il laboratorio
di ricerca FAIR, per fondare una società dedicata ai world model.

Dentro quel programma c'è anche la retrocessione dell'apprendimento per
rinforzo a «ciliegina sulla torta» {cite}`lecun2016cake`, che il dibattito sul
rinforzo ha appena pesato con il suo contraddittorio. L'argomento è il criterio
da cui il capitolo è partito: chi impara per tentativi riceve una correzione
poverissima, un «bravo» a fine giornata, mentre chi prevede il mondo ne riceve
una a ogni istante. Da lì viene la proposta di sostituire i tentativi con la
pianificazione dentro un modello del mondo, che è l'oggetto delle pagine che
seguono.

`````{tab} Elementare

L'accusa di LeCun, in soldoni: un LLM scrive come chi detta una storia una
parola alla volta senza poter mai rileggere. Ogni parola è una scommessa
basata sulle precedenti; se una scommessa introduce uno sbaglio (un
personaggio che cambia nome, un bicchiere che cade verso l'alto) le parole
dopo costruiscono sopra lo sbaglio, e più la storia è lunga più è probabile
che deragli. Mettiamo che vada storta una parola su cento: dopo cinquecento
parole, di cento racconti così ne resta in piedi meno di uno.
Soprattutto, dice LeCun, a un sistema simile manca il cinema
interiore del gatto: non immagina la scena, non prova le alternative nella
testa, non ha mai visto un bicchiere cadere; ha solo letto miliardi di frasi e
sceglie la parola più plausibile dopo le altre. Attenzione, però: questa è una
*posizione* nel dibattito scientifico, non una verità assodata. Altri
ricercatori rispondono che per indovinare bene la parola successiva in tutti i
testi del mondo bisogna, in qualche misura, aver imparato molto del mondo che
quei testi descrivono, e fanno notare che intanto i modelli continuano a
migliorare. E che rileggere, un po’, quei programmi lo fanno: capita che si
accorgano dello sbaglio e lo aggiustino nella frase dopo, e allora la catena
non si spezza. Su questo c'è perfino un esperimento pensato per decidere la
questione con i dati invece che con gli slogan, condotto su un gioco da tavolo:
lo racconta per intero la {doc}`sezione sui simulatori e il dibattito
</WorldModels/simulatori-e-dibattito>`, perché è la prova più pulita che il
dibattito abbia prodotto. Chi abbia ragione è ancora da vedere.

`````

`````{tab} Superiore

Un LLM autoregressivo {cite}`brown2020language` fattorizza la probabilità di
una sequenza come $P(w_1, \dots, w_n) = \prod_{t=1}^{n} P(w_t \mid w_{<t})$ e
genera campionando un token alla volta. L'argomento che LeCun ripete nei
seminari è di natura moltiplicativa: se a ogni token la probabilità di uscire
dall'insieme delle continuazioni accettabili è $\epsilon$, sempre la stessa e
indipendente da quanto è già stato scritto, e se l'errore non è
recuperabile, la probabilità che una sequenza di $n$ token resti accettabile
decade come $(1-\epsilon)^n$: con $\epsilon = 0{,}01$ e $n = 500$ ne resta
appena $0{,}99^{500} \approx 0{,}007$, meno dell'1%. Le obiezioni colpiscono
proprio le ipotesi: gli errori non sono né indipendenti né irrecuperabili (i
modelli, empiricamente, si correggono), e nulla fissa $\epsilon$ costante al
crescere di scala e addestramento {cite}`kaplan2020scaling`. Esperimenti di
*probing*, inoltre, indicano che un GPT a 8 strati addestrato soltanto su
sequenze di mosse dell'Otello (20 milioni di partite sintetiche) sviluppa una
rappresentazione interna dello stato della scacchiera, e che intervenire su di
essa cambia le mosse proposte {cite}`li2023emergent`: un world model implicito,
per quanto rudimentale, emerso dalla sola predizione del token successivo, in
un mondo di 64 caselle con regole fisse. Misure e limiti sono nella
{doc}`sezione sui simulatori e il dibattito
</WorldModels/simulatori-e-dibattito>`. La proposta alternativa di
{cite}`lecun2022path` (predire non nello spazio dei token o dei pixel ma in
uno spazio di rappresentazioni astratte, con architetture *joint-embedding*
addestrate a energia) è ciò che studia la {doc}`sezione sulla JEPA
</WorldModels/jepa>`, nel linguaggio del {doc}`capitolo sui modelli a energia
</ModelliEnergia/overview>`.

`````

## Quattro risposte alla stessa domanda

Come si dà a una macchina un modello del mondo, e che cosa ci deve stare
dentro? Le risposte che il capitolo segue vengono da strade diverse.

La prima sono i mondi in miniatura. Nel 2018 David Ha e Jürgen Schmidhuber
addestrano un agente a schivare palle di fuoco in un livello di *Doom*, e lo
addestrano *dentro il suo stesso sogno*: è il nome che danno alla simulazione
del gioco che il programma si è costruito da sé, guardando partite giocate a
caso. A far proseguire la partita sognata è una rete ricorrente, che legge un
fotogramma compresso alla volta portandosi dietro un riassunto di quel che ha
visto, e la policy dell'agente si allena lì dentro senza toccare il gioco vero.
Dal 2020 quella linea di ricerca arriva ai Dreamer di Danijar Hafner e
colleghi. DreamerV3, uscito su *Nature* nel 2025 {cite}`hafner2023mastering`,
ottiene un diamante in *Minecraft* senza dimostrazioni umane: il primo
algoritmo a riuscirci partendo da zero, dichiarano gli autori, a condizioni che
la sezione dedicata racconta. Dreamer 4, nel settembre dello stesso anno, ci
arriva senza mai giocare durante l'addestramento: il modello del mondo lo
impara da registrazioni di partite giocate da persone, la strategia soltanto
dentro il modello {cite}`hafner2025training`.

Seconda risposta, la via di LeCun: prevedere nello spazio delle
rappresentazioni invece che in quello dei pixel, cioè prevedere il riassunto di
quel che verrà e non ogni puntino dello schermo. Le architetture che lo fanno si
chiamano JEPA (*Joint-Embedding Predictive Architecture*, architettura
predittiva a rappresentazioni congiunte): un encoder riassume la parte vista, un
secondo encoder la parte da prevedere, e un predittore stima il secondo
riassunto a partire dal primo, nello stesso spazio. Le generazioni costruite
finora sono tre: I-JEPA per le immagini, V-JEPA per i video e V-JEPA 2, che con
lo stesso schema guida un braccio robotico. Una JEPA è un modello a energia nel
senso del {doc}`capitolo dedicato </ModelliEnergia/overview>`, e ne eredita il
pericolo principale, il collasso: le due reti possono mettersi d'accordo per
dare a ogni cosa lo stesso riassunto. Come lo si eviti lo racconta la
{doc}`sezione sulle JEPA </WorldModels/jepa>`.

Terza risposta, l’inferenza attiva, che cambia disciplina: viene dalle
neuroscienze teoriche. La tesi è che percezione, azione e apprendimento
minimizzino la stessa quantità, l’energia libera variazionale, che fa da tetto
alla sorpresa di ciò che si osserva: la percezione aggiorna le credenze,
l'azione cambia le osservazioni, l'apprendimento cambia il modello, più
lentamente. Ne esce un sistema senza una ricompensa scritta a parte, perché le
preferenze dell'organismo stanno nei priori del modello, accanto a quello che
si aspetta.

Ultima risposta, i simulatori generativi di video: Sora di OpenAI, presentato
nel febbraio 2024 come passo verso «simulatori di mondo»
{cite}`brooks2024video`, e la famiglia Genie di Google DeepMind, che dal 2024
genera ambienti in cui si può giocare {cite}`bruce2024genie`. Con loro arriva la
domanda con cui il capitolo si chiude, ed è onestamente aperta: generare video
plausibili significa aver capito la fisica, o soltanto saperla imitare?

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un world model è il cinema interiore di cui parlavamo: un simulatore
  dell'ambiente che la macchina si costruisce da sé, e che le serve per
  prevedere cosa succede, scegliere la mossa e provare alternative senza
  pagarle davvero. L'idea del «modello in scala ridotta della realtà» è di
  Kenneth Craik (1943).
- Chi non ha il simulatore impara schiantandosi nel mondo vero; chi ce l'ha si
  allena nella propria immaginazione, come i piloti prima di salire su un
  aereo. La seconda strada costa molta meno esperienza (al programma che
  impara i giochi Atari senza simulatore serve più di un mese di gioco per
  titolo, alla persona con cui è confrontato bastavano un paio d'ore di
  pratica), ma ha un prezzo: se il simulatore è impreciso ci si allena a
  vincere un gioco che non esiste, e in un mondo che non si rimette in assetto
  da sé l'imprecisione si somma quanto più lontano si prova a guardare.
- Il senso comune è un repertorio di previsioni e non un elenco di fatti
  (le cose non sostenute cadono, quel che è nascosto continua a esistere) che
  i bambini costruiscono guardando, anche se quanto ne abbiano già alla
  nascita è discusso. Nessuno etichetta niente: il maestro è il futuro, e la
  lezione arriva quando il mondo smentisce la previsione.
- Per LeCun un modello che indovina una parola alla volta non basta: serve un
  sistema che immagini il mondo, non solo il racconto del mondo. È una
  posizione autorevole dentro un dibattito aperto, non un verdetto: altri
  ricercatori sostengono che, per azzeccare le parole, quei modelli un modello
  del mondo se lo siano già costruito dentro, per quanto rudimentale.
- Nella stessa proposta c'è una retrocessione: imparare per tentativi e
  premi, dice LeCun, è «la ciliegina sulla torta», e al suo posto va la
  pianificazione dentro un modello del mondo. Il motivo è un conto e non il
  disprezzo: chi impara per tentativi riceve una correzione pochissimo
  informativa, una specie di «bravo» a fine giornata, mentre chi prevede il
  mondo viene corretto a ogni istante.
- La domanda con cui il capitolo si chiude: chi sa girare il filmato giusto ha
  capito come funziona il mondo, o è solo bravissimo a imitarlo?
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un world model è un simulatore interno *appreso* dell'ambiente,
  $p_\theta(s_{t+1} \mid s_t, a_t)$: serve a prevedere, pianificare e provare
  azioni diverse senza farle per davvero. L'idea del «modello in scala
  ridotta della realtà» risale a Kenneth Craik (1943).
- Model-free prova nel mondo, model-based prova nell'immaginazione: il
  secondo promette di ridurre le interazioni necessarie (il DQN ne usa decine
  di milioni di fotogrammi per gioco, il collaudatore umano con cui è
  confrontato circa due ore di pratica), al prezzo del *model bias*. Con
  dinamica $L$-Lipschitz ed errore del modello al più $\epsilon$, lo scarto
  dopo $k$ passi è maggiorato da $\epsilon \sum_{i<k} L^{i}$: esponenziale per
  $L > 1$, limitato da $\epsilon/(1-L)$ per $L < 1$.
- Il senso comune è un repertorio di previsioni (fisica intuitiva) che i
  bambini costruiscono guardando, senza etichette: apprendimento
  auto-supervisionato, dove il bersaglio è il futuro stesso. Gli esperimenti
  di violazione dell'aspettativa misurano l'attesa, non la sua origine, e la
  quota innata è discussa.
- Per LeCun {cite}`lecun2022path` gli LLM autoregressivi non bastano: serve
  un world model che predica in uno spazio di rappresentazioni. È una
  posizione autorevole dentro un dibattito aperto, non un consenso: altri
  ricercatori vedono negli LLM world model impliciti già in formazione.
- La stessa proposta retrocede il reinforcement learning a «ciliegina sulla
  torta» {cite}`lecun2016cake`, in favore del controllo predittivo su modello.
  L'argomento è l'informazione del bersaglio (uno scalare per episodio contro
  ordini di grandezza in più nel pre-addestramento) e l'assegnazione del credito
  lungo la traiettoria: la {doc}`sezione sul dibattito attorno al rinforzo
  </AutoSupervisione/dibattito-rl>` lo quantifica, con il contraddittorio.
- Il percorso del capitolo: mondi in miniatura (Ha & Schmidhuber, la linea
  Dreamer fino a Dreamer 4) → JEPA (I-JEPA, V-JEPA, V-JEPA 2) → inferenza
  attiva (percezione, azione e apprendimento come minimizzazioni della stessa
  energia libera, con le preferenze nei priori invece che in una ricompensa) →
  simulatori video generativi e dibattito. Il linguaggio dell'energia, su cui
  poggia la JEPA, è quello del capitolo sui modelli a energia, e non è la
  stessa «energia» dell'inferenza attiva: la sezione sull'inferenza attiva lo
  dice apertamente.
```

`````
