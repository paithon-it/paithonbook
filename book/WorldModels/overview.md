# World Model

Un gatto di casa ha molto più senso comune e comprensione del mondo di
qualunque modello di linguaggio. A ripeterlo da anni, con poche variazioni, in
conferenze e interviste, non è uno scettico qualsiasi ma Yann LeCun, premio
Turing 2018 (il riconoscimento che in informatica vale quanto un Nobel) e uno
dei padri del deep learning. Mentre mezzo mondo si stupiva di ciò che sanno
scrivere i programmi che compongono un testo indovinando una parola dopo
l'altra, i grandi modelli di linguaggio (gli **LLM**, *Large Language Model*),
uno dei loro nonni intellettuali indicava un gatto. Provocazione calcolata,
certo. Ma proviamo a prenderla sul serio: che cosa sa fare, un gatto? Non
risolve integrali e non scrive sonetti; però salta sul mobile calibrando la
traiettoria al primo colpo, prevede da che parte sbucherà il gomitolo rotolato
sotto il divano, e se una mossa è finita male non la ripete tale e quale.

E prima del gatto, il bambino. A pochi mesi di vita un neonato si stupisce (lo
si misura da quanto a lungo guarda), quando un giocattolo nascosto da uno
schermo, una volta abbassato lo schermo, non c'è più: ha già capito che gli
oggetti non svaniscono. Entro il primo anno si stupisce se un oggetto resta
sospeso a mezz'aria invece di cadere. Nessuno gli ha spiegato la permanenza
degli oggetti o la gravità; nessuno gli ha mostrato milioni di esempi con la
risposta giusta scritta accanto (in gergo: esempi *etichettati*, come le foto
con sotto il nome dell'animale che ritraggono). Ha guardato, e guardando si è
costruito dentro qualcosa che gli
permette di *aspettarsi* il mondo: un modello. Questo capitolo racconta il
tentativo di dare alle macchine qualcosa di simile (un **world model**, un
modello del mondo) e il dibattito, tuttora aperto, su quanto sia davvero il
pezzo mancante dell'intelligenza artificiale.

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
**simulatore interno** che serve a **prevedere** («cosa succede se lascio il
bicchiere?») e quindi a **pianificare**, senza dover provare tutto per
davvero.

Su quella parola, «simulatore», conviene intendersi subito, perché il capitolo
la userà spesso e i mestieri non sono lo stesso. Il simulatore di volo su cui
si esercitano i piloti l'hanno scritto degli ingegneri che le equazioni
dell'aria le conoscevano già: la fisica, lì dentro, ce l'ha messa qualcuno, una
regola alla volta. Un modello del mondo no: nessuno gliel'ha scritto, se lo
costruisce guardando, e resta per sempre una copia approssimata. Quando qui un
world model viene chiamato «simulatore» si intende questo, un simulatore
*imparato*: serve alla stessa cosa (esercitarsi senza conseguenze), ma nessuno
ne ha scritto le regole. E siccome le ha indovinate da sé, può sbagliare in
modi che un simulatore scritto a mano non sbaglierebbe. Mezzo capitolo parla
proprio di quegli sbagli.

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

Nel linguaggio del capitolo sul Reinforcement Learning: l'ambiente è un MDP
con dinamica $P(s' \mid s, a)$, che gli algoritmi di pianificazione (value
iteration, policy iteration) assumevano **nota** e che Q-learning e DQN
aggiravano imparando direttamente i valori dall'esperienza. Un world model è
la terza via: una stima **appresa** della dinamica,

$$
p_\theta(s_{t+1} \mid s_t, a_t),
$$

dove $s_t$ e $a_t$ sono lo stato e l'azione al tempo $t$, $s_{t+1}$ lo stato
successivo e $\theta$ i parametri (tipicamente di una rete neurale) stimati
dalle transizioni osservate; spesso si apprende anche un modello della
ricompensa $r_\theta(s_t, a_t)$. Un modello così abilita tre operazioni:
**predizione** (srotolare traiettorie future senza toccare l'ambiente),
**pianificazione** (cercare, tra le traiettorie immaginate, quella con il
ritorno più alto) e **simulazione di alternative** («e se agissi
diversamente?»). Su quest'ultima una precisazione da manuale di causalità:
ri-simulare da $s_t$ con un'altra azione è, nel lessico di Pearl, un
*intervento* nel modello; il contrafattuale in senso stretto («cosa *sarebbe*
successo in *quella* traiettoria») chiederebbe invece di tenere fisso il caso
già uscito (di riusare cioè la stessa realizzazione del rumore esogeno di
quella traiettoria, non di ri-estrarlo) e di ri-simulare cambiando la sola
azione; le due cose coincidono solo se la dinamica è deterministica. C'è però
un dettaglio che occuperà mezzo capitolo: nel mondo reale lo stato non si
osserva. Si osservano pixel, suoni, letture di
sensori: un'osservazione $\mathbf{x}_t$ ad alta dimensione e piena di dettagli
irrilevanti. I world model moderni imparano perciò **due** oggetti distinti: un
**codice** compatto della singola osservazione, $\mathbf{z}_t = f_\phi(\mathbf{x}_t)$ con
$f_\phi$ un encoder appreso, e una **memoria** $\mathbf{h}_t$ che riassume la storia
precedente. Da un solo fotogramma mancherebbero, per dire, le velocità: è $\mathbf{h}_t$
a portarle, ed è la memoria della prossima sezione. Lo stato del modello è
quindi la coppia $(\mathbf{z}_t, \mathbf{h}_t)$ e la dinamica si scrive
$p_\theta(\mathbf{z}_{t+1} \mid \mathbf{z}_t, a_t, \mathbf{h}_t)$; nell'RSSM dei Dreamer i due oggetti
sopravvivono con gli stessi nomi, $\mathbf{h}_t$ deterministico e $\mathbf{z}_t$ stocastico
condizionato su di esso. Che cosa debba finire in $\mathbf{z}_t$ (e che cosa sia giusto
lasciar fuori) è una delle domande centrali del capitolo.

`````

## Immaginare costa meno che provare

La prima ragione per volere un world model è il conto della spesa, e il
{doc}`capitolo sul Deep Reinforcement Learning </DeepReinforcementLearning/overview>` lo ha già pagato. Là il **DQN**
(*Deep Q-Network*, la rete che impara da sé quanto vale ogni mossa) arrivava al
livello di un giocatore umano sui vecchi videogiochi Atari. Ci arrivava però
dopo decine di milioni di fotogrammi per titolo, cioè settimane di gioco senza
mai staccare {cite}`mnih2015human`. A una persona, per capire *Breakout* (la
pallina che rimbalza su una racchetta e sbriciola un muro di mattoni), bastano
pochi minuti. Il vocabolario tecnico per questa differenza esiste da decenni
{cite}`sutton2018reinforcement`: gli algoritmi **model-free** provano tutto
per davvero, quelli **model-based** provano nella propria immaginazione.

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

Un metodo **model-free** (Q-learning, DQN, policy gradient) apprende
direttamente $Q(s, a; \theta)$ oppure $\pi_\theta(a \mid s)$: ogni
aggiornamento consuma interazione reale, e la *sample efficiency* è
notoriamente il suo tallone d'Achille. Un metodo **model-based** apprende
prima $p_\theta(s_{t+1} \mid s_t, a_t)$ e poi lo usa in due modi: per
**pianificare** (cercare azioni buone dentro il modello, come la value
iteration faceva sul modello vero) o per **generare esperienza sintetica** su
cui allenare valori e policy (l'idea dell'architettura Dyna di Sutton, che già
negli anni Novanta alternava passi vissuti e passi immaginati
{cite}`sutton2018reinforcement`). Il prezzo è il **model bias**: l'errore di
predizione si compone lungo l'orizzonte, quindi una predizione appena
imprecisa a un passo può essere pessima a $k$ passi; peggio, una policy
ottimizzata dentro il modello impara a sfruttarne i difetti (*model
exploitation*), ottenendo ritorni immaginari che l'ambiente vero non paga.
Gran parte del capitolo è il racconto di come la ricerca ha negoziato questo
compromesso: quanta fiducia concedere al sogno. Con una precisazione che il
capitolo sul Deep Reinforcement Learning quantifica e che qui conviene non
promettere di più: *quanto in fretta* l'errore si componga dipende da quanto la
dinamica amplifica le perturbazioni, e su sistemi contrattivi lo scarto, invece
di esplodere, si assesta.

`````

C'è poi una seconda ragione, meno contabile e più profonda. Il **senso
comune** che LeCun rivendica al gatto non è un elenco di fatti, ma un
repertorio di previsioni: le cose non sostenute cadono, ciò che è nascosto
continua a esistere, i liquidi si versano, gli oggetti spinti si muovono. È la
*fisica intuitiva* che il neonato dell'incipit costruisce guardando, senza
etichette: il segnale di apprendimento è la sorpresa, lo scarto tra ciò che il
suo modello prevedeva e ciò che accade. Nel lessico di questo libro quella è
una lezione **auto-supervisionata**: la risposta giusta su cui correggersi (in
gergo il **bersaglio**) non la scrive nessuno, è il futuro stesso che arriva. Il
maestro non è un annotatore umano, è il mondo un istante dopo. Se il senso
comune è fatto così, inseguirlo significa costruire macchine che imparano a
prevedere il mondo, non a memorizzarlo.

## La scommessa di LeCun (e chi non è d'accordo)

Nel 2022 LeCun mette online, aperto ai commenti di chiunque, un documento di
una sessantina di pagine: *A Path Towards Autonomous Machine Intelligence*
{cite}`lecun2022path`. Non è un articolo di risultati, è un programma di
ricerca, cioè il disegno di come andrebbe costruita una macchina che si
arrangia da sola nel mondo. Il disegno è fatto di pezzi che si passano il
lavoro: uno guarda, uno ricorda, uno propone la mossa, uno pianifica provando
le alternative. Al centro c'è un modello del mondo, imparato guardando e senza
etichette. La tesi
ha una faccia costruttiva (come *dovrebbe* essere fatta un'intelligenza
artificiale che capisce il mondo) e una polemica: i modelli di linguaggio
autoregressivi, addestrati solo a indovinare la parola successiva, per quanto
grandi non basteranno. Che LeCun ci creda davvero lo dice la biografia: a fine
2025 ha lasciato Meta (dove nel 2013 aveva fondato il laboratorio di ricerca
FAIR) per avviare una startup dedicata proprio ai world model.

Dentro quel programma c'è anche una **retrocessione**, ed è quella che di solito
si ricorda per prima. In una conferenza del 2016 LeCun disse che se
l'intelligenza è una torta, il grosso della torta è l'apprendimento senza
etichette, la glassa è l'apprendimento dalle etichette e la ciliegina è
l'apprendimento per rinforzo {cite}`lecun2016cake`. Due capitoli di questo libro
sono dedicati a quella ciliegina, e conviene dire subito che la battuta ha un
argomento sotto, non è uno sfottò: riguarda **quanta informazione** porta la
correzione con cui un sistema impara, e chi impara per tentativi ne riceve
pochissima. Il capitolo sull'auto-supervisione lo misura e riporta anche chi la
pensa diversamente; qui basta sapere che è da lì che viene la proposta di sostituire i
tentativi con la **pianificazione dentro un modello del mondo**, che è
esattamente l'oggetto delle pagine che seguono.

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
migliorare. E che rileggere, un po', quei programmi lo fanno: capita che si
accorgano dello sbaglio e lo aggiustino nella frase dopo, e allora la catena
non si spezza. Su questo c'è perfino un esperimento pensato per decidere la
questione con i dati invece che con gli slogan, condotto su un gioco da tavolo:
lo raccontiamo per intero nell'ultima sezione del capitolo, perché è la prova
più pulita che il dibattito abbia prodotto. Chi ha ragione si vedrà; questo
capitolo serve ad avere gli strumenti per seguire la partita.

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
crescere di scala e addestramento {cite}`kaplan2020scaling`. Esperimenti di *probing*,
inoltre, indicano che reti addestrate solo su sequenze di mosse (l'esempio
celebre è il gioco dell'Otello) sviluppano rappresentazioni interne dello
stato della scacchiera: un world model implicito, per quanto rudimentale,
emerso dalla sola predizione del token successivo. La proposta alternativa di
{cite}`lecun2022path` (predire non nello spazio dei token o dei pixel ma in
uno spazio di rappresentazioni astratte, con architetture *joint-embedding*
addestrate a energia) è esattamente ciò che studieremo nella sezione sulla
JEPA, nel linguaggio del capitolo sui modelli a energia.

`````

## Mondi in miniatura, e chi li abita

Quattro tappe.

La prima sono i **mondi in miniatura**. Nel 2018 David Ha e Jürgen Schmidhuber
addestrano un **agente**, cioè un programma che guarda e sceglie le mosse, a
giocare a un vecchio sparatutto: schivare palle di fuoco in *Doom*. E lo
addestrano *dentro il suo stesso sogno*, che è il nome che gli autori danno
alla simulazione del gioco che il programma si è costruito da sé. A raccontargli
come prosegue la partita è una rete **ricorrente**, cioè una rete che legge una
cosa alla volta portandosi dietro un riassunto di quel che ha già visto: le
basta quel riassunto, e la strategia dell'agente (in gergo la sua *policy*) si
allena lì dentro senza mai toccare il gioco vero.
Quella linea di ricerca arriva ai **Dreamer** di Danijar Hafner e colleghi
(dal 2020, con l'ultima versione uscita su *Nature* nel 2025), che imparano
quasi soltanto immaginando, fino a ottenere (primo
algoritmo al mondo) un diamante in *Minecraft* senza dimostrazioni umane.

Seconda tappa, la **via di LeCun**. Invece di immaginare il mondo puntino per
puntino, lo immagina per idee: prevede a grandi linee che cosa ci sarà, non
ogni singolo dettaglio dello schermo. Le architetture che lo fanno si chiamano
**JEPA** (*Joint-Embedding Predictive Architecture*, «architettura che predice
fra due riassunti»): una rete riassume il presente, un'altra riassume il
futuro, e la previsione avviene fra i due riassunti. Ce ne sono due versioni,
I-JEPA per le immagini e V-JEPA per i video, e tutte e due lavorano nello
**spazio delle rappresentazioni**, che è poi lo «spazio delle idee» del titolo
di quella sezione: il posto in cui una scena è già diventata un riassunto e non
è più un mosaico di puntini colorati. Qui il {doc}`capitolo sui modelli a energia </ModelliEnergia/overview>`
torna utile
per intero, perché una JEPA è un modello a energia: la stessa idea del
buttafuori che assegna un voto di compatibilità, e lo stesso pericolo, che le
due reti si mettano d'accordo per dare a ogni cosa lo stesso riassunto (è il
**collasso**, e lo vedremo da vicino).

Terza tappa, l’**inferenza attiva**, e qui si cambia disciplina: è la risposta
che alla stessa domanda danno le neuroscienze teoriche. La tesi è che percepire
e agire non siano due mestieri ma lo stesso, in due direzioni: davanti a uno
scarto fra quel che ti aspettavi e quel che trovi, o cambi idea o cambi il
mondo. E che imparare sia ancora la stessa cosa, più lenta. Ne esce un sistema
in cui non c'è nessun premio scritto a parte, perché quello che l'organismo
desidera sta nello stesso posto in cui sta quello che si aspetta.

Ultima tappa, i **simulatori generativi di video** (Sora di OpenAI, presentato
nel 2024 come passo verso «simulatori di mondo», e Genie di Google DeepMind,
che genera ambienti interattivi giocabili) e la domanda con cui il capitolo
chiude, onestamente aperta: generare video plausibili significa aver capito la
fisica, o soltanto saperla imitare? È anche la sezione in cui si racconta per
intero l'esperimento sul gioco da tavolo promesso poco fa.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un **world model** è il cinema interiore di cui parlavamo: un simulatore
  dell'ambiente che la macchina si costruisce da sé, e che le serve per
  prevedere cosa succede, scegliere la mossa e provare alternative senza
  pagarle davvero. L'idea del «modello in scala ridotta della realtà» è di
  Kenneth Craik (1943).
- Chi non ha il simulatore impara schiantandosi nel mondo vero; chi ce l'ha si
  allena nella propria immaginazione, come i piloti prima di salire su un
  aereo. La seconda strada costa incomparabilmente meno esperienza (ai
  programmi che imparano a giocare senza simulatore servono settimane di
  *Breakout* giocato senza mai fermarsi, a te bastano pochi minuti), ma ha un
  prezzo: se il simulatore è impreciso ci si allena a vincere un gioco che non
  esiste, e in un mondo che non si rimette in assetto da sé l'imprecisione si
  somma quanto più lontano si prova a guardare.
- Il **senso comune** non è un elenco di fatti, è un repertorio di previsioni
  (le cose non sostenute cadono, quel che è nascosto continua a esistere) che
  i bambini costruiscono guardando. Nessuno etichetta niente: il maestro è il
  futuro, e la lezione arriva quando il mondo smentisce la previsione.
- Per LeCun un modello che indovina una parola alla volta non basta: serve un
  sistema che immagini il mondo, non solo il racconto del mondo. È una
  **posizione** autorevole dentro un dibattito aperto, non un verdetto: altri
  ricercatori sostengono che, per azzeccare le parole, quei modelli un modello
  del mondo se lo siano già costruito dentro, per quanto rudimentale.
- Nella stessa proposta c'è una **retrocessione**: imparare per tentativi e
  premi, dice LeCun, è «la ciliegina sulla torta», e al suo posto va la
  pianificazione dentro un modello del mondo. Il motivo non è il disprezzo, è un
  conto: chi impara per tentativi riceve una correzione pochissimo informativa,
  una specie di «bravo» a fine giornata. Il capitolo sull'auto-supervisione lo
  fa, quel conto, e riporta anche le obiezioni.
- Il percorso del capitolo, in quattro tappe. Prima i **mondi in miniatura**: il
  programma che impara a giocare allenandosi dentro il proprio sogno, e i suoi
  eredi (i *Dreamer*, che di sogno vivono quasi soltanto). Poi la strada di
  LeCun, che invece di immaginare il mondo immagine per immagine lo immagina
  **per idee**, cioè prevede a grandi linee cosa ci sarà, non ogni singolo
  puntino dello schermo (la sigla è **JEPA**, *Joint-Embedding Predictive
  Architecture*: «architettura che predice fra due riassunti»). Poi
  l’**inferenza attiva**, che è la risposta delle neuroscienze alla stessa
  domanda: percepire e agire sono la stessa cosa in due direzioni, e un pesce si
  muove per non trovarsi all'asciutto, non per incassare un premio. Infine i
  programmi che sanno
  generare video, e la domanda con cui il capitolo si chiude: chi sa girare il
  filmato giusto ha capito come funziona il mondo, o è solo bravissimo a
  imitarlo?
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un **world model** è un simulatore interno *appreso* dell'ambiente,
  $p_\theta(s_{t+1} \mid s_t, a_t)$: serve a prevedere, pianificare e provare
  azioni diverse senza farle per davvero. L'idea del «modello in scala
  ridotta della realtà» risale a Kenneth Craik (1943).
- **Model-free** prova nel mondo, **model-based** prova nell'immaginazione: il
  secondo promette enorme efficienza nei campioni (DQN: decine di milioni di
  fotogrammi; un umano: minuti), al prezzo del *model bias*; l'errore del
  modello si accumula lungo l'orizzonte, tanto più in fretta quanto più la
  dinamica amplifica le perturbazioni.
- Il **senso comune** è un repertorio di previsioni (fisica intuitiva) che i
  bambini costruiscono guardando, senza etichette: apprendimento
  auto-supervisionato, dove il bersaglio è il futuro stesso.
- Per LeCun {cite}`lecun2022path` gli LLM autoregressivi non bastano: serve
  un world model che predica in uno spazio di rappresentazioni. È una
  **posizione** autorevole dentro un dibattito aperto, non un consenso: altri
  ricercatori vedono negli LLM world model impliciti già in formazione.
- La stessa proposta **retrocede il reinforcement learning** a «ciliegina sulla
  torta» {cite}`lecun2016cake`, in favore del controllo predittivo su modello.
  L'argomento è l'informazione del bersaglio (uno scalare per episodio contro
  ordini di grandezza in più nel pre-addestramento) e l'assegnazione del credito
  lungo la traiettoria: il capitolo sull'auto-supervisione lo quantifica, con il
  contraddittorio.
- Il percorso del capitolo: mondi in miniatura (Ha & Schmidhuber, Dreamer) →
  JEPA → **inferenza attiva** (percezione, azione e apprendimento come
  minimizzazioni della stessa energia libera, con le preferenze nei priori
  invece che in una ricompensa) → simulatori video generativi e dibattito. Il
  linguaggio dell'energia, su cui poggia la JEPA, è quello del capitolo sui
  modelli a energia, e non è la stessa «energia» dell'inferenza attiva: la
  sezione lo dice apertamente.
```

`````
