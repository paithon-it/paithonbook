# World Model

«Un gatto di casa ha molto più senso comune e comprensione del mondo di
qualunque LLM.» A scriverlo, all'inizio del 2023, non è uno scettico qualsiasi
ma Yann LeCun (premio Turing 2018, uno dei padri del deep learning) che da
allora lo ha ripetuto, con poche variazioni, in conferenze e interviste.
Mentre mezzo mondo si stupiva di ciò che i grandi modelli di linguaggio (gli
LLM della citazione, *Large Language Model*), sanno scrivere, uno dei loro
nonni intellettuali indicava un gatto. Provocazione calcolata, certo. Ma
proviamo a prenderla sul serio: che cosa sa fare, un gatto? Non risolve
integrali e non scrive sonetti; però salta sul mobile calibrando la
traiettoria al primo colpo, prevede da che parte sbucherà il gomitolo rotolato
sotto il divano, e se una mossa è finita male non la ripete tale e quale.

E prima del gatto, il bambino. A pochi mesi di vita un neonato si stupisce (lo
si misura da quanto a lungo guarda), quando un giocattolo nascosto da uno
schermo, una volta abbassato lo schermo, non c'è più: ha già capito che gli
oggetti non svaniscono. Entro il primo anno si stupisce se un oggetto resta
sospeso a mezz'aria invece di cadere. Nessuno gli ha spiegato la permanenza
degli oggetti o la gravità; nessuno gli ha mostrato milioni di esempi
etichettati. Ha guardato, e guardando si è costruito dentro qualcosa che gli
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

`````{tab} Elementare

Lo usi già, questo simulatore. Quando giochi a scacchi, prima di toccare il
pezzo ragioni così: «se sposto la torre lì, lui la mangia con l'alfiere...
allora no», e la mossa cattiva muore nella tua testa, senza costarti la
partita. Quando parcheggi, giri il volante e *vedi già* l'arco che il paraurti
disegnerà: se l'auto immaginata finisce sul marciapiede, correggi prima che ci
finisca quella vera. E alla domanda «cosa succede se lascio il bicchiere?»
rispondi senza bisogno di lasciarlo: nella tua testa il bicchiere è già caduto
e già in mille pezzi. Il modello del mondo è questo cinema interiore in cui il
futuro si prova a costo zero. Non è perfetto (la manovra immaginata a volte
finisce comunque con una strisciata) ma ogni volta che la realtà ti smentisce,
il cinema interiore si aggiorna e la prossima previsione è un po' migliore.

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
sensori: un'osservazione $x_t$ ad alta dimensione e piena di dettagli
irrilevanti. I world model moderni imparano perciò uno **stato latente**
compatto $z_t = f_\phi(x_{1:t})$, con $f_\phi$ un encoder appreso, e simulano
lì: $p_\theta(z_{t+1} \mid z_t, a_t)$. Il pedice $1{:}t$ non è un dettaglio:
da un solo fotogramma mancherebbero, per dire, le velocità, e nessuna
transizione su $z_t$ sarebbe ben definita; per questo i sistemi che
incontreremo aggregano la storia con uno stato ricorrente (la memoria $h_t$
della prossima sezione). Che cosa debba finire in $z_t$ (e che cosa sia
giusto lasciar fuori) è una delle domande centrali del capitolo.

`````

## Immaginare costa meno che provare

La prima ragione per volere un world model è un conto della spesa, e lo
abbiamo già pagato nel capitolo sul Deep Reinforcement Learning: DQN ha
raggiunto il livello umano sui giochi Atari consumando decine di milioni di
fotogrammi per titolo (settimane di gioco ininterrotto {cite}`mnih2015human`),
dove a una persona bastano pochi minuti per capire *Breakout*. Il vocabolario
tecnico per questa differenza esiste da decenni
{cite}`sutton2018reinforcement`: gli algoritmi **model-free** provano tutto
per davvero, quelli **model-based** provano nella propria immaginazione.

`````{tab} Elementare

Pensa a come si formano i piloti di linea. Nessuna compagnia fa esercitare le
emergenze (un motore in fiamme, una raffica in atterraggio) su un aereo vero:
si usa il simulatore, dove un errore non costa niente e la stessa situazione
si può ripetere cento volte in un pomeriggio. Il DQN di quel capitolo è un
allievo senza simulatore: ogni cosa che impara la impara schiantandosi per
davvero, e per questo gli servono milioni di partite. Tu no: dopo qualche
pallina persa a *Breakout* hai già in testa un piccolo *Breakout* tascabile
(«se la racchetta è qui e la pallina scende lì, la manco») e le mosse le
ripassi lì dentro, gratis. Chi possiede un simulatore interno spreme da ogni
esperienza vera decine di esperienze immaginate. L'unico rischio è fidarsi di
un simulatore sbagliato: se il tuo *Breakout* mentale è impreciso, ti alleni a
vincere un gioco che non esiste.

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
compromesso: quanta fiducia concedere al sogno.

`````

C'è poi una seconda ragione, meno contabile e più profonda. Il **senso
comune** che LeCun rivendica al gatto non è un elenco di fatti, ma un
repertorio di previsioni: le cose non sostenute cadono, ciò che è nascosto
continua a esistere, i liquidi si versano, gli oggetti spinti si muovono. È la
*fisica intuitiva* che il neonato dell'incipit costruisce guardando, senza
etichette: il segnale di apprendimento è la sorpresa, lo scarto tra ciò che il
suo modello prevedeva e ciò che accade. Tradotto nel lessico di questo libro:
è apprendimento **auto-supervisionato**, dove il bersaglio non lo fornisce un
annotatore umano ma il futuro stesso. Se il senso comune è fatto così,
inseguirlo significa costruire macchine che imparano a prevedere il mondo: non
a memorizzarlo.

## La scommessa di LeCun (e chi non è d'accordo)

Nel 2022 LeCun deposita su OpenReview un documento di posizione di una
sessantina di pagine, *A Path Towards Autonomous Machine Intelligence*
{cite}`lecun2022path`. Non è un articolo di risultati: è un programma di
ricerca. Al centro c'è un'architettura modulare per agenti autonomi il cui
cuore è precisamente un world model appreso in modo auto-supervisionato,
affiancato da percezione, memoria e un modulo che pianifica simulando. La tesi
ha una faccia costruttiva (come *dovrebbe* essere fatta un'intelligenza
artificiale che capisce il mondo) e una polemica: i modelli di linguaggio
autoregressivi, addestrati solo a indovinare la parola successiva, per quanto
grandi non basteranno. Che LeCun ci creda davvero lo dice la biografia: a fine
2025 ha lasciato Meta (dove nel 2013 aveva fondato il laboratorio di ricerca
FAIR) per avviare una startup dedicata proprio ai world model.

`````{tab} Elementare

L'accusa di LeCun, in soldoni: un LLM scrive come chi detta una storia una
parola alla volta senza poter mai rileggere. Ogni parola è una scommessa
basata sulle precedenti; se una scommessa introduce uno sbaglio (un
personaggio che cambia nome, un bicchiere che cade verso l'alto) le parole
dopo costruiscono sopra lo sbaglio, e più la storia è lunga più è probabile
che deragli. Soprattutto, dice LeCun, a un sistema così manca il cinema
interiore del gatto: non immagina la scena, non prova le alternative nella
testa, non ha mai visto un bicchiere cadere; ha solo letto miliardi di frasi e
sceglie la parola più plausibile dopo le altre. Attenzione, però: questa è una
*posizione* nel dibattito scientifico, non una verità assodata. Altri
ricercatori rispondono che per indovinare bene la parola successiva in tutti i
testi del mondo bisogna, in qualche misura, aver imparato molto del mondo che
quei testi descrivono, e fanno notare che intanto i modelli continuano a
migliorare. Chi ha ragione si vedrà; questo capitolo serve ad avere gli
strumenti per seguire la partita.

`````

`````{tab} Superiore

Un LLM autoregressivo {cite}`brown2020language` fattorizza la probabilità di
una sequenza come $P(w_1, \dots, w_n) = \prod_{t=1}^{n} P(w_t \mid w_{<t})$ e
genera campionando un token alla volta. L'argomento che LeCun ripete nei
seminari è di natura moltiplicativa: se a ogni token la probabilità di uscire
dall'insieme delle continuazioni accettabili è $\epsilon$, e l'errore non è
recuperabile, la probabilità che una sequenza di $n$ token resti accettabile
decade come $(1-\epsilon)^n$, con $\epsilon = 0{,}01$ e $n = 500$ resta appena
$0{,}99^{500} \approx 0{,}007$, meno dell'1%. Le obiezioni colpiscono proprio
le ipotesi: gli errori non sono né indipendenti né irrecuperabili (i modelli,
empiricamente, si correggono), e nulla fissa $\epsilon$ costante al crescere
di scala e addestramento {cite}`kaplan2020scaling`. Esperimenti di *probing*,
inoltre, indicano che reti addestrate solo su sequenze di mosse (l'esempio
celebre è il gioco dell'Otello) sviluppano rappresentazioni interne dello
stato della scacchiera: un world model implicito, per quanto rudimentale,
emerso dalla sola predizione del token successivo. La proposta alternativa di
{cite}`lecun2022path` (predire non nello spazio dei token o dei pixel ma in
uno spazio di rappresentazioni astratte, con architetture *joint-embedding*
addestrate a energia) è esattamente ciò che studieremo nella sezione sulla
JEPA, nel linguaggio del capitolo precedente.

`````

## Come è organizzato il capitolo

Tre tappe. Si parte dai **mondi in miniatura**: nel 2018 David Ha e Jürgen
Schmidhuber addestrano un agente che impara a giocare a un vecchio sparatutto
(schivare palle di fuoco in *Doom*) esercitandosi *dentro il proprio sogno*:
un world model ricorrente in cui la policy si allena senza toccare il gioco
vero. Quella linea di ricerca arriva ai **Dreamer** di Danijar Hafner e
colleghi (2020–2023), che imparano interamente nell'immaginazione latente,
fino a ottenere (primo algoritmo al mondo) un diamante in *Minecraft* senza
dimostrazioni umane. Seconda tappa, la **via di LeCun**: le architetture
**JEPA** (I-JEPA per le immagini, V-JEPA per i video), che predicono nello
spazio delle rappresentazioni e non dei pixel. Qui il capitolo precedente
torna utile per intero: una JEPA è un modello a energia non normalizzato, e
senza quel linguaggio (energia come compatibilità, collasso, metodi
regolarizzati) la proposta di LeCun resta illeggibile. Ultima tappa, i
**simulatori generativi di video** (Sora di OpenAI, presentato nel 2024 come
passo verso «simulatori di mondo», e Genie di Google DeepMind, che genera
ambienti interattivi giocabili) e la domanda con cui il capitolo chiude,
onestamente aperta: generare video plausibili significa aver capito la fisica,
o soltanto saperla imitare?

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
  programmi che imparano a giocare senza simulatore servono milioni di partite
  a *Breakout*, a te bastano pochi minuti), ma ha un prezzo: se il simulatore
  è impreciso ci si allena a vincere un gioco che non esiste, e l'imprecisione
  si somma quanto più lontano si prova a guardare.
- Il **senso comune** non è un elenco di fatti, è un repertorio di previsioni
  (le cose non sostenute cadono, quel che è nascosto continua a esistere) che
  i bambini costruiscono guardando. Nessuno etichetta niente: il maestro è il
  futuro, e la lezione arriva quando il mondo smentisce la previsione.
- Per LeCun un modello che indovina una parola alla volta non basta: serve un
  sistema che immagini il mondo, non solo il racconto del mondo. È una
  **posizione** autorevole dentro un dibattito aperto, non un verdetto: altri
  ricercatori sostengono che, per azzeccare le parole, quei modelli un modello
  del mondo se lo siano già costruito dentro, per quanto rudimentale.
- Il percorso del capitolo, in tre tappe. Prima i **mondi in miniatura**: il
  programma che impara a giocare allenandosi dentro il proprio sogno, e i suoi
  eredi (i *Dreamer*, che di sogno vivono quasi soltanto). Poi la strada di
  LeCun, che invece di immaginare il mondo immagine per immagine lo immagina
  **per idee**, cioè prevede a grandi linee cosa ci sarà, non ogni singolo
  puntino dello schermo (la sigla è **JEPA**). Infine i programmi che sanno
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
  modello si accumula lungo l'orizzonte.
- Il **senso comune** è un repertorio di previsioni (fisica intuitiva) che i
  bambini costruiscono guardando, senza etichette: apprendimento
  auto-supervisionato, dove il bersaglio è il futuro stesso.
- Per LeCun {cite}`lecun2022path` gli LLM autoregressivi non bastano: serve
  un world model che predica in uno spazio di rappresentazioni. È una
  **posizione** autorevole dentro un dibattito aperto, non un consenso: altri
  ricercatori vedono negli LLM world model impliciti già in formazione.
- Il percorso del capitolo: mondi in miniatura (Ha & Schmidhuber, Dreamer) →
  JEPA → simulatori video generativi e dibattito. Il linguaggio dell'energia,
  su cui poggia la JEPA, è quello del capitolo precedente.
```

`````
