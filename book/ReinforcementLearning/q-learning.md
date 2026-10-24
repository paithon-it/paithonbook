# Q-learning e l'apprendimento per differenze temporali

Nel videogioco dell’{doc}`apertura del capitolo <overview>` premi tasti, il
personaggio si muove, ogni tanto il punteggio sale, e nessuno ti dice quale
delle decine di mosse fatte abbia meritato quei punti. Eppure a ogni mossa il
gioco risponde con due informazioni: i punti guadagnati in quel momento e una
schermata nuova, di cui ti sei già fatto un'idea di quanto prometta. Messe
insieme bastano a correggere una stima prima che la partita finisca.

## Imparare senza aspettare la fine: le differenze temporali

L'apprendimento per differenze temporali (TD) aggiorna le stime *durante* la
partita, a ogni passo, usando la ricompensa appena incassata più la stima che
già si ha della situazione in cui si è finiti. Il nome e la teoria sono di
Richard Sutton, 1988 {cite}`sutton1988learning`. L'idea è più vecchia: il
programma di dama di Samuel con cui si apre il {doc}`capitolo sul machine
learning </MachineLearning/overview>` spostava già la valutazione di una
posizione verso quella della posizione vista più avanti {cite}`samuel1959some`.

`````{tab} Elementare

Stai stimando quanto dura un viaggio in auto. Parti dicendo "due ore". Dopo
mezz'ora sei più avanti del previsto, e il navigatore dice che ne manca "un'ora
e dieci". Non hai aspettato di arrivare per correggere la tua previsione: hai
usato una stima più recente per aggiustare quella vecchia.

Questa aritmetica è tutto il metodo, quindi la facciamo per bene. La previsione
vecchia diceva due ore, cioè centoventi minuti. La stima nuova ha due pezzi, la
strada già fatta più la strada che resta: trenta minuti percorsi, più settanta
che restano, fanno cento minuti.

La differenza è di venti minuti, e quei venti minuti sono l'errore: il viaggio
andrà meglio di come lo avevi previsto, e la previsione va tirata giù. (Qui si
stima un tempo, e una bella notizia lo fa *scendere*. Nel reinforcement
learning, invece, si stima un punteggio, e una bella notizia lo fa salire. Il
meccanismo è lo stesso, cambia solo che cosa si conta.)

Il TD learning fa proprio questo, correggendo un pezzetto alla volta. Il primo
dei due pezzi, la strada già fatta, è l'unico dato vero della faccenda e tiene
la correzione ancorata alla realtà invece che a un'altra opinione.

L'altro pezzo, i settanta minuti che mancano, resta un'opinione, e se è
sbagliata la correzione tira dalla parte sbagliata. Ad esempio, col cantiere
trenta chilometri più avanti, che il navigatore non conosce, la previsione
scende mentre dovrebbe salire. Su una strada mai fatta succede spesso, e in
generale [ogni stima appoggiata a un'altra stima rischia di sbagliare
così]{.pt-matita}.

`````

`````{tab} Superiore

Sia $V(s)$ la stima del valore di uno stato $s$ sotto la politica che si sta
seguendo, cioè il ritorno scontato che ci aspettiamo partendo da lì e
continuando a seguire quella politica. Dopo aver osservato la transizione $s
\to s'$ con ricompensa osservata $r$, l'aggiornamento TD(0) è

$$
V(s) \leftarrow V(s) + \alpha\,\underbrace{\big[\,r + \gamma\,V(s') -
V(s)\,\big]}_{\text{errore TD}\;\delta} .
$$

Qui $\alpha \in (0,1]$ è il learning rate, $\gamma \in [0,1]$ il fattore di
sconto e $\delta$ l’**errore TD**. Il termine $r + \gamma V(s')$ è una stima
aggiornata di $V(s)$ costruita *usando la stima successiva* $V(s')$. Questa
dipendenza da una stima per aggiornarne un'altra si chiama *bootstrapping*, e i
metodi Monte Carlo non ce l'hanno.

Il bersaglio non è più il ritorno osservato $G_t$ ma una sua approssimazione a
un passo. Di conseguenza la varianza diminuisce (un solo termine casuale invece
di una somma lunga) e [compare una distorsione, perché $V(s')$ è a sua volta
una stima, e all'inizio è sbagliata]{.pt-matita}. La distorsione, però, non
resta. Per una policy fissa e una tabella, TD(0) converge a $V^\pi$ con
probabilità $1$ se ogni stato è visitato infinite volte, se $\gamma < 1$ (o gli
episodi terminano quasi certamente) e se i passi, presi sulle visite di
ciascuno stato, soddisfano le condizioni di Robbins-Monro viste sui bandit
{cite}`dayan1992convergence`.

A parità di dati le due famiglie non convergono alla stessa stima. In
particolare, se un insieme finito di episodi viene ripassato fino a
convergenza (aggiornamento *batch*, con $\alpha$ abbastanza piccolo), Monte
Carlo si ferma sui valori che minimizzano l'errore quadratico sui ritorni
osservati, mentre TD(0) si ferma sul valore esatto dell'MDP di massima
verosimiglianza stimato da quegli episodi, la stima di *certainty equivalence*
{cite}`sutton2018reinforcement`. Se il processo è markoviano ci si aspetta che
questa sbagli meno sui dati futuri, il che può spiegare in parte, oltre alla
varianza, perché TD impara spesso più in fretta.

`````

## Q-learning: stimare il valore delle azioni

Conoscere il valore di uno stato non basta per decidere senza il modello, come
si è visto con i {doc}`metodi Monte Carlo <monte-carlo>`: serve il valore di
ogni azione in quello stato. A stimarlo è il **Q-learning**, formulato da Chris
Watkins nella tesi di dottorato del 1989 {cite}`watkins1989learning` e rimasto
uno degli algoritmi più noti del campo.

`````{tab} Elementare

Una grande tabella: una riga per ogni situazione in cui l'agente può trovarsi,
una colonna per ogni mossa possibile, e all'incrocio un voto, che dice quanto
conviene fare quella mossa in quella situazione. Quel voto, nelle formule, si
chiama $Q$, e da lì viene il nome dell'algoritmo. La lettera viene dalla tesi
di Watkins ed è rimasta; torna comoda da ricordare come l'iniziale di
*quality*, la qualità di una mossa, anche se il nome ufficiale della cosa è
«funzione azione-valore».

All'inizio i voti sono tutti a zero, cioè l'agente non sa nulla. Giocando e
ricevendo ricompense, corregge i voti. Alla fine, per agire bene, gli basta
guardare la riga della situazione corrente e scegliere la mossa col voto più
alto.

La parte sorprendente è che l'agente può muoversi anche a casaccio, sbagliando
di proposito per esplorare, e imparare comunque quali sarebbero le mosse
migliori. In altre parole, [impara una cosa mentre ne fa un'altra]{.pt-matita}.
Per questo si dice *off-policy*, cioè "fuori dalla propria strategia".

Il trucco sta nel massimo: quando l'agente corregge il voto di una mossa,
guarda dove è finito e prende il voto della migliore fra le mosse possibili da
lì, non di quella che poi farà. Anche se subito dopo tira un dado e sbaglia
apposta, la correzione appena scritta riguarda un giocatore che non sbaglia.

Giocando abbastanza a lungo i voti finiscono al posto giusto, e c'è una
dimostrazione che lo garantisce. Infatti ogni correzione è un giro della
ricetta della value iteration, fatto con una partita sola invece che con la
mappa, e mediando tante partite il caso si cancella.

La dimostrazione, però, pone delle condizioni. Ogni mossa deve essere provata
tante volte in ogni situazione, non una o due, perché un voto corretto due
volte vale quanto un sorteggio. La tabella deve potersi scrivere per intero,
riga per riga. E la frazione di sorpresa a cui si dà retta va calata man mano
che si gioca. Il programma del labirinto la tiene ferma, e se lo può permettere
perché in quel labirinto ogni mossa ha sempre lo stesso esito e il premio
è fisso.

Prendere sempre il voto più alto ha un difetto. Un voto si costruisce
sommando una frazione di ogni sorpresa incassata fin lì, e dove il premio o
l'esito di una mossa cambiano da una partita all'altra, le sorprese cambiano
con loro. Per questo, anche in una riga in cui tutte e quattro le mosse valgono
zero, i voti scritti ballano un po’ attorno allo zero, per esempio $-0{,}2$,
$+0{,}1$, $-0{,}3$, $+0{,}3$. Il più alto è $+0{,}3$, più di quanto la riga
valga.

In una riga singola può andare diversamente (se tutti e quattro ballano verso
il basso, il massimo resta sotto), ma in media il massimo di stime rumorose
cade sopra il massimo dei valori veri: fra quattro numeri sballati, il più alto
tende a essere un numero che ha sbagliato per eccesso. L'agente si fa così
un'idea un po’ troppo rosea delle proprie mosse.

Per rimediare si tengono due tabelle, e a ogni correzione una sceglie e l'altra
giudica: la prima dice qual è la mossa migliore dalla situazione d'arrivo, la
seconda dice quanto vale quella mossa. Se un voto è uscito troppo alto nella
prima per puro caso, la fa scegliere, ma il numero che entra nella correzione
lo legge la seconda, che quel caso non l'ha avuto, e la gonfiatura non passa.

A ogni correzione si tira a sorte quale delle due tabelle aggiornare, così che
imparino tutte e due e ciascuna faccia a turno da giudice dell'altra. Si chiama
**Double Q-learning**, il Q-learning a due tabelle.

`````

`````{tab} Superiore

Definiamo la **funzione azione-valore ottima** $Q^*(s,a)$ come la ricompensa
scontata attesa se in $s$ eseguiamo $a$ e poi seguiamo la politica ottima.
Soddisfa l'equazione di ottimalità di Bellman

$$
Q^*(s,a) = \mathbb{E}\big[\,R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1},a')
\;\big|\; S_t = s,\ A_t = a\,\big],
$$

dove l'attesa è sulla ricompensa e sullo stato d'arrivo,
$S_{t+1} \sim P(\cdot \mid s, a)$.

Il Q-learning è off-policy perché il suo *target* usa $\max_{a'} Q(s',a')$
(il valore dell'azione *migliore* nello stato successivo) indipendentemente da
quale azione l'agente abbia poi effettivamente scelto. In questo modo [impara la
politica ottima anche mentre ne segue una esplorativa]{.pt-matita}.

Watkins e Dayan {cite}`watkins1992q` ne dimostrarono la convergenza sotto
ipotesi precise, e la più restrittiva, la tabella, è il punto da cui parte il
{doc}`deep reinforcement learning </DeepReinforcementLearning/overview>`. In un
MDP finito, con $Q$ tabellare, ricompense limitate e sconto $\gamma < 1$, se
ogni coppia $(s,a)$ è visitata infinite volte e i passi soddisfano le
condizioni di Robbins-Monro già viste sui bandit, $Q$ converge a $Q^*$ con
probabilità $1$.

Si noti che le due somme vanno prese sui passaggi di ciascuna coppia e non sul
tempo globale: detta $n^i(s,a)$ la $i$-esima volta che in $s$ si prova $a$,
servono $\sum_i \alpha_{n^i(s,a)} = \infty$ e $\sum_i \alpha^2_{n^i(s,a)} <
\infty$ per ogni $(s,a)$. La differenza ha conseguenze. Consideriamo
$\alpha_t = 1/t$ e una coppia visitata ai tempi $t = 2, 4, 8, \dots$: la somma
globale diverge, mentre la somma presa sui soli passaggi, che è quella
richiesta dal teorema, vale $\sum_{k \ge 1} 2^{-k} = 1$, e l'ipotesi cade.

La convergenza ha una ragione precisa: il Q-learning è la value iteration con
l'attesa sostituita da un campione. Infatti l'operatore di ottimalità

$$
(\mathcal{B}Q)(s,a) = r(s,a)
+ \gamma \sum_{s'} P(s'\mid s,a)\, \max_{a'} Q(s',a')
$$

è una contrazione di fattore $\gamma$ nella norma del massimo, per la stessa
disuguaglianza sul massimo della value iteration, e il suo punto fisso è $Q^*$.
Il bersaglio $r + \gamma \max_{a'} Q(S_{t+1},a')$ ne è un campione non
distorto, e l'aggiornamento è un passo di Robbins-Monro verso di esso. È
l'approssimazione stocastica del punto fisso di una contrazione, la strada
delle dimostrazioni di Jaakkola, Jordan e Singh
{cite}`jaakkola1994convergence` e di Tsitsiklis
{cite}`tsitsiklis1994asynchronous`, accanto a quella di Watkins e Dayan.

La garanzia vale per il caso **tabellare**. Sostituita la tabella con una
funzione approssimata, la dimostrazione non si trasporta, e approssimazione di
funzione, bootstrapping e apprendimento off-policy insieme possono far
divergere i valori anche con un approssimatore lineare. Il controesempio di
Baird, ripreso nella {doc}`sezione sul DQN </DeepReinforcementLearning/dqn>`, è
il caso classico {cite}`sutton2018reinforcement`.

Il $\max$ nel target introduce inoltre una distorsione. Applicato a stime
rumorose, il massimo è uno stimatore distorto verso l'alto del massimo dei
valori veri. Ad esempio, se in uno stato tutte le azioni valgono zero ma le
stime oscillano attorno allo zero, $\max_{a'}Q(s',a')$ ha attesa positiva. Lo
dice la disuguaglianza di Jensen applicata al $\max$, che è convesso:
$\mathbb{E}[\max_a \hat{Q}(s,a)] \ge \max_a \mathbb{E}[\hat{Q}(s,a)]$, con
disuguaglianza stretta appena nessuna azione è quasi certamente la migliore. È
il **bias di massimizzazione**.

Il bias si evita tenendo due stime indipendenti $Q_A$ e $Q_B$ e usandone una
per scegliere l'azione e l'altra per valutarla, con i ruoli estratti a caso a
ogni aggiornamento (*Double Q-learning*, di Hado van Hasselt
{cite}`vanhasselt2010double`). Il bersaglio $Q_B(s', \arg\max_{a'}
Q_A(s',a'))$ ha un'attesa che non supera il massimo dei valori veri e può
restare sotto, per cui la sovrastima lascia il posto a una possibile
sottostima. Da questa idea nasce il Double DQN.

`````

## La formula di aggiornamento

L'aggiornamento è una sola riga. Il valore stimato $Q(s,a)$ dell'azione $a$
nello stato $s$, il suo voto, si sposta di una frazione $\alpha$ verso un
bersaglio: la ricompensa $r$ appena incassata più il valore dell'azione
migliore nello stato $s'$ in cui si è finiti, scontato di $\gamma$. La
differenza fra bersaglio e stima è l'errore TD, la sorpresa.

`````{tab} Elementare

Scritta a parole, la riga è questa:

> voto nuovo = voto vecchio + tasso × (bersaglio − voto vecchio)

ed è la stessa forma della regola del {doc}`quaderno delle leve <banditi>`: una
stima vecchia, più una frazione della sorpresa. Dentro ci sono quattro pezzi, e
ognuno ha un nome che torna in ogni metodo che impara qualcosa.

- Il **voto vecchio**, il numero scritto nella tabella per quella situazione e
  quella mossa.
- Il **bersaglio**: quanto quella mossa sembra valere adesso, cioè il premio
  appena incassato più il miglior voto della riga in cui si è finiti, ridotto
  dallo sconto. È una stima migliore della precedente perché contiene un pezzo
  di realtà, il premio appena visto. In inglese si chiama *target*, la parola
  che si incontra nel codice.
- La **sorpresa**, cioè bersaglio meno voto vecchio: positiva se è andata
  meglio del previsto, negativa se peggio.
- Il **tasso di apprendimento** (*learning rate*), nelle formule la lettera
  greca $\alpha$ (alfa): un numero fra zero e uno che dice a quanta parte della
  sorpresa dare retta. Sulle leve lo stesso numero si chiamava il passo. Vicino
  a zero l'agente corregge poco per volta ed è cauto; vicino a uno butta via il
  voto vecchio a ogni sorpresa.

Poi c'è lo sconto, la $\gamma$ (gamma) della {doc}`sezione sui processi
decisionali di Markov <mdp-valore>` (in sigla MDP, il modo di descrivere un
ambiente come stati, mosse e ricompense), che decide quanto pesano le
prospettive future rispetto al premio incassato subito.

`````

`````{tab} Superiore

In simboli:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\,\Big[\,r + \gamma \max_{a'} Q(s',a') - Q(s,a)\,\Big].
$$

Il termine $r + \gamma \max_{a'} Q(s',a')$ è il **target TD**, la stima
aggiornata del valore di $(s,a)$. Nel target, $r$ è la ricompensa *osservata* in
questa transizione (non la ricompensa attesa $r(s,a)$ del modello, che qui non
conosciamo), e il termine $\gamma \max_{a'} Q(s',a')$ si pone a zero quando
$s'$ è terminale, perché dopo uno stato terminale non arrivano altre
ricompense. Sottraendo la stima
corrente $Q(s,a)$ otteniamo l'errore TD. Il learning rate $\alpha \in (0,1]$
decide quanto fidarci della correzione (piccolo = passi cauti), e il fattore di
sconto $\gamma \in [0,1]$ pesa il futuro (vicino a $1$ = lungimirante, vicino a
$0$ = miope).

`````

Il Q-learning tabellare non usa altro: nessuna rete neurale, nessun gradiente,
solo la tabella dei valori $Q$ aggiornata a ogni passo.

## Esplorare o sfruttare: la strategia $\varepsilon$-greedy

Se l'agente scegliesse sempre la mossa di valore $Q$ più alto, potrebbe
fermarsi su una strategia peggiore dell'ottima senza mai provare le mosse che
la migliorerebbero. Serve quindi un modo di esplorare, e il più semplice è
quello già usato con le leve, $\varepsilon$-greedy.

`````{tab} Elementare

È di nuovo il dilemma del ristorante, e la ricetta è quella delle leve: quasi
sempre la mossa che il quaderno dà per migliore, e con una piccola probabilità
$\varepsilon$ una a caso. All'inizio si esplora molto, e $\varepsilon$ si riduce
man mano che si impara.

Il ritmo della riduzione decide che cosa si scoprirà. Trenta cene al mese, e
parti con una novità su dieci: tre ristoranti nuovi il primo mese. Se poi
dimezzi la quota ogni mese passi a una e mezza, poi a tre quarti, e sommando
tutti i mesi da qui all'eternità arrivi a sei. Sei ristoranti nuovi in tutta la
vita: i ristoranti che non hai provato entro allora non li proverai mai, e se
uno era il migliore della città non lo saprai.

Se invece dividi la quota per il numero dei mesi passati (metà il secondo mese,
un terzo il terzo, un decimo il decimo), la somma diventa tre più uno e mezzo
più uno più tre quarti e così via, cioè sedici novità in dieci anni e ventitré
in un secolo. Sembrano poche anche queste, ma [la somma della prima ricetta si
ferma a sei, quella della seconda non si ferma mai]{.pt-matita}, e qualunque
numero si fissi prima o poi lo si supera. Rallenti senza mai fermarti, e
aspettando abbastanza ogni ristorante prima o poi ti capita.

La seconda ricetta tiene in piedi la promessa dei voti che si assestano al
posto giusto, perché quella promessa chiede che ogni mossa venga provata tante
volte. Qui la situazione è una sola, la città, e i mesi sono anche
le volte in cui ci si è trovati lì; con più situazioni si divide per le volte
in cui si è stati in quella. La prima ricetta si usa lo stesso, perché porta
prima a risultati decenti, sapendo che cosa si lascia indietro.

`````

`````{tab} Superiore

Data la tabella corrente, la politica $\varepsilon$-greedy sceglie

$$
a =
\begin{cases}
\text{azione casuale uniforme} & \text{con probabilità } \varepsilon,\\[4pt]
\arg\max_{a'} Q(s,a') & \text{con probabilità } 1-\varepsilon.
\end{cases}
$$

Un $\varepsilon$ costante garantisce esplorazione perpetua; in pratica, però,
si usa un *decay*, spesso esponenziale, per avvicinarsi gradualmente allo
sfruttamento puro. [Il decadimento esponenziale sacrifica la garanzia
teorica]{.pt-matita}. Infatti la convergenza di Watkins e Dayan vuole ogni
coppia $(s,a)$ visitata infinite volte, e a questo basta che la somma degli
$\varepsilon$ diverga su ogni stato. Accade, ad esempio, con $\varepsilon
\propto 1/n(s)$, dove $n(s)$ conta le visite a quello stato, mentre con il
tempo globale al denominatore la somma può restare finita: uno stato visitato
ai tempi $t = 2^k$ raccoglie $\sum_k 2^{-k}$.

Le due richieste, visite infinite a ogni coppia e policy greedy nel limite, si
chiamano **GLIE** (*greedy in the limit with infinite exploration*). Con GLIE e
le condizioni sul passo richieste dal teorema di Watkins e Dayan, SARSA
converge alla policy ottima {cite}`singh2000convergence`. Si noti che GLIE è la
coppia di requisiti, e $\varepsilon \propto 1/n(s)$ ne è soltanto un modo
comodo di soddisfarli. Con un $\varepsilon$ che si spegne esponenzialmente,
invece, le azioni esplorative sono quasi certamente in numero finito, perché
una serie geometrica di ragione minore di uno ha somma finita, e in pratica lo
scambio si accetta.

`````

## SARSA: la variante on-policy

**SARSA** differisce dal Q-learning per un solo termine dell'aggiornamento: al
posto del valore dell'azione migliore nello stato d'arrivo, $\max_{a'}
Q(s',a')$, usa il valore $Q(s',a')$ dell'azione $a'$ che l'agente esegue subito
dopo. Il nome è l'elenco dei cinque elementi dell'aggiornamento: *state,
action, reward, state, action*.

L'algoritmo è quindi *on-policy*: [invece di imparare quanto varrebbero le
mosse di un giocatore perfetto, impara quanto valgono le proprie, esplorazione
compresa]{.pt-matita}. Lo proposero Rummery e Niranjan
{cite}`rummery1994online`, che lo chiamavano *modified connectionist
Q-learning*; Sutton gli diede nel 1996 il nome con cui lo conosciamo.

`````{tab} Elementare

Il Q-learning è un ottimista spericolato: valuta ogni mossa immaginando di
comportarsi *perfettamente* subito dopo. SARSA è più prudente: valuta le mosse
tenendo conto che, di tanto in tanto, esplorerà e potrebbe sbagliare. Impara il
valore della politica che segue, esplorazione compresa.

Il cambio è di un pezzo solo. Quando corregge il voto di una mossa, SARSA
guarda dove è finito e prende il voto della mossa che farà, dado compreso, al
posto del voto della migliore di quella riga. C'è anche una via di mezzo:
invece di aspettare il dado, fare la media dei voti della riga, ciascuno pesato
per quanto spesso la strategia lo sceglierebbe. In questo modo toglie dal
calcolo il caso del dado, e se la strategia sceglie sempre la mossa migliore
coincide con il Q-learning.

Il risultato tipico si vede su un esperimento classico, che si chiama *cammino
sul precipizio*: una griglia di quattro righe per dodici colonne, in cui la
partenza e l'arrivo occupano le due caselle agli estremi della riga in basso e
tutte le dieci caselle in mezzo a loro sono un burrone. La strada più corta
passa nella riga subito sopra, cioè sull'orlo, e un passo storto fa cadere di
sotto: chi cade paga cento punti e si ritrova alla partenza, con la partita che
continua e la strada tutta da rifare.

SARSA impara a salire fin sopra, lontano dal bordo, e ci arriva in qualche
passo in più senza quasi mai finire di sotto; il Q-learning impara a camminare
sull'orlo, perché "in teoria" non sbaglierebbe mai un passo, e ogni tanto ci
casca davvero.

Il bilancio dei punti spiega perché la prudenza paga. Ogni passo fa perdere un
punto e la caduta cento; la strada alta aggiunge quattro passi a partita,
quindi la strada corta conviene soltanto se fa cadere meno di una partita su
venticinque (cento punti ogni venticinque partite fanno quattro punti a
partita).

Il dado si tira una volta ogni dieci mosse (è la quota di esplorazione, qui
tenuta ferma a un decimo) e sceglie fra quattro direzioni, di cui una sola
porta di sotto. Quindi ogni casella sul bordo fa cadere due volte e
mezza su cento, e di caselle affacciate sul vuoto ce n'è una decina da
attraversare. La probabilità di scamparle tutte e dieci è $0{,}975$
moltiplicato per sé stesso dieci volte, cioè poco più di tre quarti: di sotto
ci si finisce quasi in una partita su quattro.

Se il dado sparisce, sparisce anche la differenza: un agente che non esplora
più non fa il passo storto, l'orlo torna a essere la strada migliore, e i due
imparano la stessa cosa. La prudenza di SARSA diventa allora qualche passo
buttato via.

`````

`````{tab} Superiore

SARSA è on-policy: nel target compare il valore dell'azione $a'$ scelta nello
stato $s'$ dalla stessa politica (ad esempio $\varepsilon$-greedy), al posto
del massimo:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\,\big[\,r + \gamma\,Q(s',a') - Q(s,a)\,\big].
$$

Valuta dunque la politica di comportamento anziché quella greedy. Consideriamo
il classico esempio del *cliff walking*: il compito è episodico e non scontato
($\gamma = 1$), con $-1$ su ogni transizione e $-100$ per una caduta, che
rispedisce alla partenza senza chiudere l'episodio. Con $\varepsilon$ costante
(negli esperimenti classici $0{,}1$) SARSA converge ai valori della migliore
politica $\varepsilon$-greedy, che passa lontano dal precipizio; il Q-learning
converge ai valori della politica ottima, che corre lungo il bordo, ma la sua
prestazione online, eseguita con $\varepsilon$-greedy, è peggiore perché ogni
tanto cade. Le due soluzioni coincidono solo quando $\varepsilon \to 0$
{cite}`sutton2018reinforcement`.

Fra i due sta una terza regola, **Expected SARSA**, che al posto del campione
$a'$ usa l'attesa sotto la policy:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\,\Big[\,r
+ \gamma \sum_{a'} \pi(a'\mid s')\, Q(s',a') - Q(s,a)\,\Big].
$$

A parità di policy ha la stessa media di SARSA e varianza minore, perché
elimina la casualità della scelta di $a'$. In particolare, se $\pi$ è la policy
greedy, coincide con il Q-learning. Poiché l'attesa si calcola su tutte le
azioni, il comportamento può differire da $\pi$ senza pesi di importanza: la
regola è on-policy se i due coincidono, off-policy altrimenti
{cite}`sutton2018reinforcement`.

`````

Le due strade, disegnate sulla stessa griglia, si confrontano a colpo d'occhio
({numref}`fig-cammino-sul-precipizio`): quella corta cammina sull'orlo per
dieci caselle, quella alta se ne tiene lontana e paga quattro passi.

```{figure} ../figures/cammino-sul-precipizio.svg
:name: fig-cammino-sul-precipizio
:alt: "Griglia di quattro righe per dodici colonne. Nella riga in basso, la partenza a sinistra e la meta a destra; fra le due, dieci caselle nere sono il burrone. Due cammini uniscono la partenza alla meta. Quello del Q-learning, in terracotta, sale di una casella e corre lungo il bordo del burrone: è il più corto. Quello del SARSA, in teal, sale fino alla riga più alta, corre di là e ridiscende: quattro passi in più, e nessuna casella affacciata sul vuoto."
:width: 92%

Il cammino sul precipizio. La via corta corre sull'orlo del burrone, la via
alta lo evita salendo fino in cima alla griglia. Il Q-learning impara la prima
perché valuta un giocatore che non sbaglia mai; SARSA impara la seconda perché
mette in conto i passi storti che farà esplorando.
```

## Un labirinto concreto

Prendiamo una griglia di tre righe per quattro colonne. La partenza (S) è in
basso a sinistra; la meta, che paga $+1$, è in alto a destra; la trappola, che
fa perdere un punto e chiude comunque la partita, è subito sotto la meta; e c'è
un muro nella seconda casella della riga di mezzo, contro cui si sbatte
restando fermi. A ogni passo si sceglie fra su, giù, sinistra, destra.

Le regole non sono quelle del robot della {doc}`sezione sugli MDP
<mdp-valore>`, dove ogni passo faceva perdere un punto, mentre qui girovagare
non fa perdere punti. Di conseguenza, a spingere l'agente verso l'uscita c'è
soltanto lo sconto, che toglie valore al premio quanto più lo si fa aspettare e
rende così conveniente la strada corta. Sono invece le regole della griglia su
cui la stessa sezione mostra l'iterazione dei valori (premio $+1$, passi
gratis, sconto $0{,}9$), con in più la trappola.

```{figure} ../figures/labirinto-qlearning.svg
:name: fig-labirinto
:alt: Griglia 3x4 con cella di partenza in basso a sinistra, meta con ricompensa +1 in alto a destra, trappola -1 sotto la meta, un muro al centro e frecce che indicano la politica appresa in ogni cella.
:width: 85%

La strategia imparata dal Q-learning sulla griglia: in ogni cella la freccia
indica la mossa che, finite tutte le partite di allenamento, ha il voto più
alto. È il risultato che stampa il programma del labirinto. Il disegno mostra il
punto d'arrivo, non la strada per arrivarci: il valore della meta
che retrocede una casella per volta si vede nella
{numref}`fig-voto-che-retrocede`.
```

All'inizio la tabella dei voti è tutta a zero. Fissiamo un tasso di
apprendimento di $0{,}5$ (si dà retta a metà della sorpresa) e uno sconto di
$0{,}9$.

`````{tab} Elementare

La prima volta che l'agente calpesta la meta incassa $1$ e la partita finisce
lì, quindi dalla casella d'arrivo non c'è più niente da aspettarsi. La sorpresa
è tutta lì: si aspettava $0$, ha incassato $1$. Dandole retta a metà, il voto
dell'ultima mossa passa da $0$ a $0{,}5$.

Alla casella che veniva prima tocca più tardi, alla prima partita in cui
l'agente ci ripassa. Quando ci era passato, i voti della casella in cui era
finito erano ancora tutti zero, e correggere verso zero non muove niente.
Adesso invece il $0{,}5$ c'è, e servirà a chi ripasserà di lì.

Da quella casella la mossa non paga niente e porta nella casella accanto alla
meta, che nella sua riga di tabella ha ormai una mossa da $0{,}5$ (i voti
stanno sulle mosse, e qui conta il migliore della riga): scontato, vale $0{,}9
\times 0{,}5 = 0{,}45$. La sorpresa è di nuovo positiva (si aspettava $0$, la
prospettiva vale $0{,}45$) e dandole retta a metà il voto diventa $0{,}225$.

Tutti e due i risultati dipendono dal fatto che qui muoversi non costa nulla:
nel labirinto del robot, dove ogni passo faceva perdere un punto, la seconda
mossa avrebbe reso $-1 + 0{,}45 = -0{,}55$, e il voto sarebbe diventato
$-0{,}275$ invece di $0{,}225$.

`````

`````{tab} Superiore

Poniamo $\alpha=0{,}5$ e $\gamma=0{,}9$. La prima volta che l'agente calpesta
la meta ($r=+1$, stato successivo terminale con valore $0$), la casella
dell'ultima mossa diventa

$$
Q(s,\rightarrow) \leftarrow 0 + 0{,}5\,\big[\,1 + 0{,}9\cdot 0 - 0\,\big] = 0{,}5 .
$$

Al primo episodio successivo che ripassa da lì, la cella precedente $s^-$, da
cui si arriva a $s$, "vede" $\max_{a'} Q(s,a')=0{,}5$ e si aggiorna:

$$
Q(s^-,\rightarrow) \leftarrow 0 + 0{,}5\,\big[\,0 + 0{,}9\cdot 0{,}5 -
0\,\big] = 0{,}225 .
$$

Entrambi gli aggiornamenti usano $r$ come ricompensa osservata, e in questo
mondo la transizione non ha costo di passo: in un mondo che penalizzasse ogni
mossa con $-1$, il secondo aggiornamento darebbe $0{,}5\,[-1 + 0{,}45] =
-0{,}275$.

`````

È il meccanismo TD in azione: [la ricompensa non si propaga a tutta la griglia
in una volta, ma retrocede verso la partenza di un passo per
episodio]{.pt-matita} (di più solo se l'episodio ripassa dalla stessa casella),
come una macchia che si allarga all'indietro dalla meta. Il codice, ambiente
compreso, è breve:

```python
import numpy as np

# Griglia 3x4: 12 stati, 4 azioni (0=su 1=giù 2=sinistra 3=destra)
RIGHE, COLONNE = 3, 4
MURO, META, TRAPPOLA = (1, 1), (0, 3), (1, 3)
MOSSE = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LIBERE = [(i, j) for i in range(RIGHE) for j in range(COLONNE)
          if (i, j) not in (MURO, META, TRAPPOLA)]

n_stati, n_azioni = RIGHE * COLONNE, 4
Q = np.zeros((n_stati, n_azioni))       # tabella dei voti, tutta a zero
alpha, gamma, epsilon = 0.5, 0.9, 0.1
rng = np.random.default_rng(20260807)

def indice(cella):
    return cella[0] * COLONNE + cella[1]

def ambiente(cella, a):
    """Dove si finisce, quanto si incassa, se la partita è finita."""
    i, j = cella[0] + MOSSE[a][0], cella[1] + MOSSE[a][1]
    if not (0 <= i < RIGHE and 0 <= j < COLONNE) or (i, j) == MURO:
        i, j = cella                          # contro un muro si resta fermi
    if (i, j) == META:     return (i, j), 1.0, True
    if (i, j) == TRAPPOLA: return (i, j), -1.0, True
    return (i, j), 0.0, False

def epsilon_greedy(s):
    if rng.random() < epsilon:
        return int(rng.integers(n_azioni))    # esplora: mossa a caso
    return int(np.argmax(Q[s]))               # sfrutta: mossa col voto più alto

def aggiorna(s, a, r, s_next, fine):
    # target TD: usa la stima migliore dello stato successivo (off-policy)
    td_target = r if fine else r + gamma * np.max(Q[s_next])
    Q[s, a] += alpha * (td_target - Q[s, a])  # correggi verso il target

# Inizi esplorativi: ogni episodio comincia da una casella sorteggiata.
for _ in range(5000):
    cella = LIBERE[rng.integers(len(LIBERE))]
    for _ in range(100):
        s = indice(cella)
        a = epsilon_greedy(s)
        cella_dopo, r, fine = ambiente(cella, a)
        aggiorna(s, a, r, indice(cella_dopo), fine)
        if fine:
            break
        cella = cella_dopo

FRECCE = "^v<>"
def voto(cella):
    if cella == MURO:     return "  muro"
    if cella == META:     return "    +1"
    if cella == TRAPPOLA: return "    -1"
    s = indice(cella)
    return f"{FRECCE[int(np.argmax(Q[s]))]}{Q[s].max():5.2f}"

for i in range(RIGHE):
    print(" | ".join(voto((i, j)) for j in range(COLONNE)))
```

```text
> 0.81 | > 0.90 | > 1.00 |     +1
^ 0.73 |   muro | ^ 0.90 |     -1
^ 0.66 | > 0.73 | ^ 0.81 | < 0.73
```

Le stesse tre righe, con le frecce al posto di `^v<>` e i due traguardi
chiamati per nome:

| colonna 1 | colonna 2 | colonna 3 | colonna 4 |
|:--|:--|:--|:--|
| → $0{,}81$ | → $0{,}90$ | → $1{,}00$ | **meta**, $+1$ |
| ↑ $0{,}73$ | muro | ↑ $0{,}90$ | **trappola**, $-1$ |
| ↑ $0{,}66$ | → $0{,}73$ | ↑ $0{,}81$ | ← $0{,}73$ |

Le frecce sono, una per una, quelle della {numref}`fig-labirinto`, e i numeri
accanto si controllano a mano. La casella da cui basta una mossa per arrivare
vale il premio pieno, $1{,}00$; ogni passo indietro lo moltiplica per lo
sconto, e viene $0{,}90$, poi $0{,}81$, poi $0{,}729$ e poi $0{,}6561$, che il
programma stampa arrotondati a $0{,}73$ e $0{,}66$. Si ritrovano tutti:
l'angolo in basso a sinistra, la casella più lontana dalla meta (cinque passi),
porta infatti $0{,}66$.

Di ogni casella la griglia stampa un numero solo, il migliore dei
quattro; la tabella per intero è quella della {numref}`fig-tabella-dei-voti`,
dove ogni quadretto del labirinto è una riga e ogni mossa una colonna.

```{figure} ../figures/tabella-dei-voti.svg
:name: fig-tabella-dei-voti
:alt: "A sinistra il labirinto di tre righe per quattro colonne: in alto a destra la meta che paga più uno, sotto di lei la trappola che ne toglie uno, un muro al centro, e quattro quadretti liberi marcati con le lettere A, B, C e D. A destra la tabella dei voti: una riga per quadretto, una colonna per ciascuna delle quattro mosse, su, giù, sinistra e destra. La riga A porta 0,90 0,81 0,81 1,00; la riga B, evidenziata come il suo quadretto, porta 0,90 0,73 0,81 e meno 1,00, e il voto più alto della riga, lo 0,90 della mossa su, è cerchiato; la riga C porta 0,81 0,73 0,66 0,66; la riga D porta meno 1,00 0,66 0,73 0,66. Sotto, la riga che spiega: si guarda la riga del quadretto in cui si è, e si prende il voto più alto."
:width: 96%

Quattro quadretti del labirinto e le loro righe di tabella, con i voti che il
programma ha scritto dopo cinquemila partite. Il quadretto B ha la trappola a
destra, e infatti quella mossa vale $-1$; la freccia che si legge nella griglia
è il voto cerchiato, il più alto della riga. Il quadretto D, che sta sotto la
trappola, ha lo stesso $-1$ sulla mossa in su.
```

La casella da cui basta una mossa per arrivare vale qui $1{,}00$, e nel primo
calcolo a mano valeva $0{,}5$. Sono due istantanee della stessa storia. Il
$0{,}5$ era il voto dopo il primo passaggio, quando alla sorpresa si dava retta
a metà partendo da zero. Per quella mossa il bersaglio resta sempre $1$, perché
incassa il premio e la partita finisce lì. A ogni passaggio successivo, quindi,
il voto recupera metà della distanza che lo separa da $1$, e diventa $0{,}75$,
poi $0{,}875$, poi $0{,}9375$, e così via. Dopo cinquemila partite ci è
arrivato così vicino che, alla seconda cifra, vale $1{,}00$.

Per le altre caselle vale lo stesso, con una complicazione che ha già un nome,
il *bootstrapping*: il bersaglio contiene la stima della casella successiva, e
quindi non sta fermo. Ad esempio, la casella accanto era partita rincorrendo
$0{,}45$, cioè lo sconto per il $0{,}5$ che c'era allora; mentre ci correva
dietro, quel $0{,}5$ è salito verso $1$, e il bersaglio è salito verso
$0{,}90$. In generale, ogni casella insegue un numero che a sua volta sta
salendo, e la fila si assesta dall'ultima all'indietro. I calcoli a mano dicono
da dove parte ciascun voto, la griglia stampata dice dove arriva, e la
{numref}`fig-voto-che-retrocede` mostra il tragitto.

```{figure} ../figures/voto-che-retrocede.svg
:name: fig-voto-che-retrocede
:alt: Griglia di tre righe per quattro colonne: la meta che paga più uno in alto a destra, la trappola che ne toglie uno subito sotto, un muro al centro, e nove caselle libere in ciascuna delle quali si legge il voto della mossa migliore, con la casella tinta tanto più intensamente quanto più quel voto è alto. Si parte con tutte e nove le caselle a zero dopo 7 partite; alla 8ª la sola casella accanto alla meta prende 0,50; alla 11ª quella sale a 0,75 e la casella prima prende 0,23; alla 13ª le caselle con un voto sono quattro, alla 21ª sei, alla 60ª tutte e nove. L'ultima a prendere un voto è la casella in basso a destra, sotto la trappola, e non l'angolo più lontano dalla meta, che ce l'ha già alla 21ª. Dopo 5000 partite i voti sono 0,81, 0,90, 1,00; 0,73, 0,90; 0,66, 0,73, 0,81, 0,73, riga per riga dall'alto in basso.
:width: 90%

Sette istantanee della stessa esecuzione, quella del programma del labirinto con
il suo seme. Per sette partite le nove caselle restano bianche, e non perché la
tabella sia vuota: alla quarta l'agente finisce nella trappola e quella mossa
incassa il suo $-0{,}50$. Qui però si legge il voto della mossa migliore, e
finché nessuna partita ha toccato la meta la mossa migliore di ogni casella vale
ancora zero. All'ottava la casella accanto alla meta prende $0{,}50$, che è il
numero del primo calcolo a mano; all'undicesima quella è già a $0{,}75$ e la
casella prima prende $0{,}23$, che è il secondo. Da lì la macchia arretra una
casella alla volta, e non a ogni partita: perché una casella riceva il suo primo
voto bisogna che l'agente ci passi sopra quando la casella dopo un voto ce l'ha
già. Per questo l'ultima a illuminarsi non è quella più lontana dalla meta, che
ce l'ha alla ventunesima partita, ma l'angolo in basso a destra, sotto la
trappola: a decidere è il passaggio, non la distanza. L'ultima istantanea è la
griglia che il programma stampa.
```

Nel programma, la funzione `aggiorna` è il Q-learning per intero. Per ottenere
SARSA basterebbe passarle l'azione che l'agente sceglierà al passo successivo e
usare il voto di *quella* al posto del voto della mossa migliore.

Il tasso di apprendimento (`alpha`) resta fermo a metà per tutte e cinquemila
le partite. Un tasso che non si accorcia mai continua a rincorrere le ultime
sorprese, per cui in un mondo con premi casuali non si posa su un numero; nel
labirinto è tutto fisso, la sorpresa alla fine è zero e nessuno se ne accorge.

Fra le condizioni della dimostrazione c'è un tasso che si accorci come il passo
del quaderno delle leve: abbastanza da posarsi, non tanto da fermarsi prima di
arrivare. Il programma del labirinto lascia cadere la condizione, in cambio di
un algoritmo che reagisce in fretta, e in pratica è la scelta più comune.

Anche la quota di esplorazione resta ferma a un decimo invece di calare. Al
Q-learning, che impara la mossa migliore mentre ne fa un'altra, basta che ogni
coppia continui a essere provata. SARSA chiede invece che l'esplorazione alla
lunga si spenga, perché valuta le mosse che fa, e con un decimo di mosse a caso
impara il valore di chi sbaglia un decimo delle volte.

Ogni partita, poi, comincia da una casella estratta a caso invece che dalla
partenza. Sono gli inizi esplorativi dei metodi Monte Carlo in versione più
debole: là si estraeva a caso la coppia casella-mossa, qui solo la casella, e
a provare tutte le mosse ci pensa l’$\varepsilon$-greedy. Dentro un
simulatore, che possiamo far ripartire dove vogliamo, si ottengono con una
riga.

Senza gli inizi esplorativi, le caselle lontane dal cammino migliore verrebbero
visitate di rado, la loro riga resterebbe quasi vuota e la loro freccia sarebbe
quasi casuale. Per vederlo si rifà l'addestramento con tutte le partite che
cominciano dalla partenza, in basso a sinistra, e si conta quante volte
l'agente passa dall'angolo in basso a destra, che sul cammino migliore non sta:

```python
def addestra(dalla_partenza, seme=20260807):
    """Rifà le 5000 partite; dà visite e voto dell'angolo in basso a destra."""
    global Q, rng
    Q, rng = np.zeros((n_stati, n_azioni)), np.random.default_rng(seme)
    visite = np.zeros(n_stati, dtype=int)
    for _ in range(5000):
        cella = (2, 0) if dalla_partenza else LIBERE[rng.integers(len(LIBERE))]
        for _ in range(100):
            s = indice(cella)
            visite[s] += 1
            a = epsilon_greedy(s)
            cella_dopo, r, fine = ambiente(cella, a)
            aggiorna(s, a, r, indice(cella_dopo), fine)
            if fine:
                break
            cella = cella_dopo
    angolo = indice((2, 3))
    return visite[angolo], Q[angolo].max()

for nome, dalla_partenza in (("partenze sorteggiate", False),
                             ("sempre dalla partenza", True)):
    visite, voto_angolo = addestra(dalla_partenza)
    print(f"{nome:22} visite all'angolo: {visite:4d}   voto: {voto_angolo:.2f}")
altri = [addestra(True, seme)[0] for seme in (1, 2, 3, 4, 5)]
print("sempre dalla partenza, altri cinque semi, visite:", *altri)
```

```text
partenze sorteggiate   visite all'angolo:  750   voto: 0.73
sempre dalla partenza  visite all'angolo:   22   voto: 0.55
sempre dalla partenza, altri cinque semi, visite: 0 0 0 1 0
```

Con il seme del primo programma l'agente passa dall'angolo settecentocinquanta
volte con le partenze sorteggiate e ventidue partendo sempre dal basso a
sinistra, e il voto dell'angolo si ferma a $0{,}55$ invece che a $0{,}73$.

Con altri semi può non passarci mai, o passarci una volta sola, e una volta non
basta. Infatti, con la riga tutta a zero, la prima mossa che si prova è quella
in su, che porta nella trappola, e il $-0{,}50$ che ne esce non tocca il
massimo della riga. La freccia di quella riga dipende allora da un pareggio fra
voti a zero, e non contiene informazione.

## Fra un passo e la fine: quanti passi guardare avanti

Nel Q-learning la macchia si allarga all'indietro dalla meta, una casella per
partita, perché il bersaglio guarda avanti di un passo solo. I metodi Monte
Carlo, all'estremo opposto, usano il totale della partita e portano la notizia
a tutte le caselle attraversate in una partita sola: una notizia sola, però, e
rumorosa. Fra i due estremi c'è una famiglia di metodi, in cui il bersaglio può
guardare avanti $n$ passi, da $n = 1$, le differenze temporali, fino alla fine
della partita, Monte Carlo.

`````{tab} Elementare

La domanda è: quanti passi guardare prima di fidarsi della propria stima? Uno
solo, e allora è il TD; tutti quelli che restano fino alla fine, e allora è
Monte Carlo; oppure tre, o dieci.

Guardare pochi passi dà una correzione stabile ma un po’ sbagliata, perché si
appoggia a una stima che, quando si è appena cominciato a giocare, non vale
niente. Guardare fino in fondo dà una correzione onesta ma ballerina. [Una
manciata di passi, nei casi studiati, in genere batte entrambi gli
estremi]{.pt-matita}.

C'è anche un modo elegante di non scegliere: fare la media di tutte le
lunghezze, dando più peso a quelle corte e via via meno a quelle lunghe. Il
peso cala di una frazione fissa a ogni passo in più, come un'eco che si spegne,
e la manopola che decide quanto in fretta si spenga si chiama $\lambda$
(lambda), un numero fra zero e uno.

I due estremi si capiscono guardando come si spartisce il peso. La lunghezza
più corta, un passo, si prende quanto manca a $\lambda$ per arrivare a uno;
quel che avanza se lo spartiscono le lunghezze successive, con la stessa
regola. Con $\lambda = 0$ la prima si prende tutto, e siamo tornati alle
differenze temporali. Con $\lambda = 0{,}5$ il passo singolo prende metà del
peso, quello doppio un quarto, quello triplo un ottavo, e così via dimezzando.

Più $\lambda$ si avvicina a uno, più il peso scivola verso le lunghezze lunghe.
Con $\lambda = 0{,}9$, ad esempio, alle prime dieci lunghezze messe insieme ne
toccano poco meno di due terzi (uno meno $0{,}9$ moltiplicato per sé stesso
dieci volte, cioè $1 - 0{,}35$), e il resto va più in là. Nelle partite che
finiscono, però, questo «più in là» ha un fondo: oltre la fine, guardare
avanti dieci passi o cento è la stessa cosa, cioè guardare fino in fondo, e
tutto il peso che avanza va al totale della partita. A $\lambda = 1$ ci va
tutto, e siamo tornati a Monte Carlo.

Detta così sembra impossibile da fare mentre si gioca, perché quella media
guarda avanti, e il futuro non lo si conosce. Il trucco è guardare dall'altra
parte: invece di chiedersi "che cosa succederà dopo questa casella", si tiene
un elenco delle caselle appena attraversate, ciascuna con un ricordo che sfuma
a ogni passo. A farlo sfumare sono $\lambda$ e lo sconto, che pesa meno ciò che
sta lontano nel tempo. Quel ricordo si chiama **traccia**, e quando arriva una
sorpresa la si distribuisce a tutta la scia, tanto più forte quanto più recente
è il passaggio.

Che le due viste diano lo stesso risultato non è ovvio e richiede una
dimostrazione; l'idea è che dare a ogni casella un pezzetto di correzione alla
volta, per tutta la partita, produca alla fine quanto si otterrebbe in un colpo
solo guardando avanti. Il totale coincide se si aspetta la fine della partita
per applicare le correzioni, e coincide solo in modo approssimato se le si
applicano strada facendo, che è il modo in cui il metodo si usa.

Il guadagno è che non si aspetta mai la fine: la correzione si scrive a ogni
passo, e una manopola sola copre tutte le lunghezze insieme. Di memoria, però,
non se ne risparmia, perché il ricordo che sfuma va tenuto per ogni casella
della griglia, non solo per le ultime dieci attraversate, e a ogni passo si
aggiornano tutti i ricordi.

`````

`````{tab} Superiore

Consideriamo prima il valore di uno stato, $V(s)$, che è la forma in cui le
equazioni si leggono meglio; per il controllo il calcolo si estende poi
direttamente ai valori delle azioni, diventando SARSA a $n$ passi e
SARSA($\lambda$).

Il **ritorno a $n$ passi** tronca la somma dopo $n$ ricompense vere e chiude
con la stima corrente:

$$
G_{t:t+n} = R_{t+1} + \gamma\, R_{t+2} + \cdots + \gamma^{n-1} R_{t+n}
+ \gamma^{n} V(S_{t+n}),
$$

e l'aggiornamento è il solito $V(S_t) \leftarrow V(S_t) +
\alpha\,[\,G_{t:t+n} - V(S_t)\,]$. Per $n=1$ si ritrova TD(0), mentre per $n$
pari o superiore ai passi che restano fino alla fine dell'episodio il termine
con $V$ sparisce e resta il ritorno Monte Carlo.

Il compromesso è quello classico fra distorsione e varianza: [con $n$ piccolo si
ha poca varianza e molta distorsione, con $n$ grande il contrario]{.pt-matita}. La
distorsione ha un limite preciso: il valore atteso del ritorno a $n$ passi
dista da $v_\pi$ al più $\gamma^n$ volte la distanza della stima da cui parte,
in norma del massimo (la proprietà di riduzione dell'errore). Nei banchi di
prova di Sutton e Barto si impara più in fretta a valori intermedi che agli
estremi {cite}`sutton2018reinforcement`.

Il **$\lambda$-return** evita di dover scegliere $n$: è la media pesata di
*tutti* i ritorni a $n$ passi, con pesi che decadono geometricamente,

$$
G_t^{\lambda} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{\,n-1}\, G_{t:t+n},
\qquad \lambda \in [0,1],
$$

con la convenzione $G_{t:t+n} = G_t$ se $t+n \ge T$, perché il ritorno non
guarda oltre la fine dell'episodio. Per $\lambda = 0$ si ha TD(0). Per $\lambda
= 1$, in un episodio che termina al passo $T$ tutti i ritorni con $n \ge T-t$
coincidono con il ritorno intero $G_t$; raccogliendone i pesi, la coda della
somma si compatta in un termine $\lambda^{\,T-t-1}\, G_t$, che a $\lambda = 1$
è l'unico a sopravvivere: si ritrova Monte Carlo.

Scritta così la formula è impraticabile, perché richiede di conoscere il futuro:
è la **vista in avanti**. Le **tracce di eleggibilità** danno la stessa cosa
dalla **vista all'indietro**, calcolabile passo per passo mentre si gioca: si
tiene un vettore $\mathbf{z}$ che segna quali stati sono «in attesa di credito»,

$$
z_t(s) = \gamma\lambda\, z_{t-1}(s) + \mathbb{1}[S_t = s],
$$

cioè la traccia di uno stato sale di $1$ quando lo si visita e sfuma di
$\gamma\lambda$ a ogni passo successivo.

A ogni istante si calcola un solo errore TD $\delta_t$ e lo si distribuisce a
tutti gli stati in proporzione alla loro traccia: $V(s) \leftarrow V(s) +
\alpha\,\delta_t\, z_t(s)$. Una ricompensa inattesa corregge così in
un colpo tutta la scia di stati che l'hanno preceduta, i più recenti di più.

La memoria è una traccia per stato, dello stesso ordine delle stime di valore
che si tengono già: $O(|\mathcal{S}|)$ nel caso tabellare,
$O(|\mathcal{S}||\mathcal{A}|)$ quando le tracce stanno sulle coppie, $O(d)$
con l'approssimazione lineare, dove $d$ è il numero di caratteristiche
dell'approssimatore. Non cresce né con la lunghezza dell'episodio né con
$\lambda$. Anche il calcolo per passo aumenta: nella forma tabellare ingenua
ogni passo aggiorna tutte le tracce e tutti i valori, $O(|\mathcal{S}|)$
invece di $O(1)$ (in pratica si aggiornano gli stati con traccia non
trascurabile).

Con una tabella il guadagno non sta nella memoria, perché una traccia per stato
occupa più dei soli $n$ stati che il ritorno a $n$ passi tiene da parte; il
risparmio arriva con l'approssimazione, dove un vettore di tracce sta al posto
degli ultimi $n$ vettori di caratteristiche. Il guadagno vero è
che la correzione si scrive a ogni passo invece di arrivare con $n$ passi di
ritardo, e che un iperparametro solo sostituisce la scelta di $n$.

Le due viste danno lo stesso risultato: coincidono esattamente se i pesi
restano fermi per tutto l'episodio, mentre online l'uguaglianza esatta
richiede la *dutch trace* del true online TD($\lambda$), e la traccia
accumulante vista qui la approssima {cite}`sutton2018reinforcement`.

Per il controllo off-policy la traccia così com'è non basta. Infatti il
Q-learning valuta la policy greedy, e una scia che continua dopo un'azione
esplorativa darebbe credito a un cammino che quella policy scarta. Per questo
il Q($\lambda$) di Watkins azzera tutte le tracce alla prima azione non
greedy, e a decidere il taglio è l'azione scelta, non l'estrazione
dell’$\varepsilon$: un'esplorazione che pesca comunque l'azione greedy non
interrompe niente {cite}`sutton2018reinforcement`.

`````

Sul labirinto la differenza si vede nella {numref}`fig-tracce-illuminano`: a
parità di episodio, un passo alla volta rinforza la sola mossa che incassa il
premio, mentre con le tracce si rinforzano tutte, in dissolvenza all'indietro.
Per questo i metodi che guardano più passi avanti imparano più in fretta quando
le ricompense sono rare.

```{figure} ../figures/tracce-illuminano.svg
:name: fig-tracce-illuminano
:alt: Due griglie identiche di tre righe per quattro colonne, con la partenza in basso a sinistra, la meta che paga più uno in alto a destra, la trappola sotto di lei e un muro al centro. In tutte e due, cinque frecce disegnano lo stesso cammino dalla partenza alla meta. Nella griglia di sinistra, «un passo alla volta», una sola freccia è colorata, l'ultima, quella che entra nella meta, con peso 1,00; le altre quattro sono grigie. Nella griglia di destra, «con le tracce», sono colorate tutte e cinque, e il loro peso sfuma all'indietro: 1,00, 0,81, 0,66, 0,53 e 0,43 per la prima mossa del cammino, che resta comunque accesa.
:width: 92%

Lo stesso episodio, il cammino più corto dalla partenza alla meta, con la
tabella dei voti tutta a zero. A un passo alla volta il bersaglio è zero
ovunque tranne sull'ultima mossa, che incassa il premio: si rinforza quella e
basta. Con le tracce si rinforzano tutte, ciascuna per quanto vale la sua
traccia nell'istante della sorpresa: lo sconto per $\lambda$, cioè
$\gamma\lambda$, moltiplicato per sé stesso tante volte quanti sono i passi che
la separano dalla fine. Con lo sconto di $0{,}9$ e $\lambda = 0{,}9$ quel
prodotto vale $0{,}81$, e alla mossa più vecchia, quattro passi prima della
fine, arriva ancora $0{,}81^4 \approx 0{,}43$, il $43\%$ della correzione.
```

La scia intera del disegno vale quando si giudica la strategia che si sta
seguendo, come fa SARSA. Il Q-learning giudica invece un giocatore che non
sbaglia mai, e allora la scia va tagliata alla prima mossa esplorativa che quel
giocatore non avrebbe fatto: come nel Monte Carlo fuori policy, dopo una mossa
del genere la partita non è più la sua. La variante che taglia così si chiama
Q($\lambda$) di Watkins.

Lo stesso $\lambda$ ritorna nella {doc}`sezione sul gradiente di policy
</DeepReinforcementLearning/policy-gradient>`. Là il segnale che guida
l'apprendimento è il *vantaggio* di una mossa, cioè quanto quella mossa è
migliore della media delle mosse possibili in quella situazione, e l'errore TD
di un passo ne è già una stima, la più semplice. La
*generalized advantage estimation* {cite}`schulman2016high` somma gli errori TD
dei passi successivi con pesi che calano di $\gamma\lambda$ a ogni passo, come
le tracce, e $\lambda$ regola di nuovo lo stesso scambio fra distorsione e
varianza.

## Quando la tabella non basta più

Ogni algoritmo visto finora presuppone che le situazioni si possano elencare,
una riga di tabella ciascuna: un'ipotesi che è comparsa solo fra le condizioni
della dimostrazione di Watkins e Dayan. Nel labirinto le
situazioni sono dodici. A scacchi sono più delle molecole d'aria di una
stanza. Sullo schermo di un videogioco, dove ogni fotogramma diverso è una
situazione diversa, i conti della {doc}`sezione sugli MDP <mdp-valore>` danno
un numero che nessuna tabella può contenere.

La tabella cede per due motivi. Il primo è la memoria: servirebbe un voto per
ogni coppia situazione-mossa. Il secondo, più grave, sono i dati: anche avendo
la tabella, quasi ogni situazione che l'agente incontra è nuova, la sua riga è
vuota, e riempire per esperienza diretta ogni riga richiederebbe più partite di
quante se ne possano giocare. Al posto della tabella serve allora una funzione
che, davanti a una situazione mai vista, ne indovini i voti accostandola a
quelle già viste, come fanno le reti neurali.

Ma la convergenza del Q-learning è un teorema, dimostrato nel 1992 da Watkins e
Peter Dayan, e parla di una tabella soltanto. [Senza la tabella la garanzia
cade]{.pt-matita}, e non per una dimostrazione che manchi: si conoscono esempi
costruiti apposta in cui i voti divergono. La combinazione responsabile si
chiama {doc}`triade fatale </DeepReinforcementLearning/dqn>`
{cite}`sutton2018reinforcement`: approssimazione di funzione (valori calcolati
da una formula invece che scritti uno per uno), bootstrapping e apprendimento
off-policy, messi insieme, possono far divergere i valori. Un Q-learning con
una rete al posto della tabella le ha tutte e tre.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Le differenze temporali correggono la stima durante il viaggio, non
  all'arrivo: come il navigatore che dopo mezz'ora rivede il tempo che manca,
  si usa la stima più recente per aggiustare quella vecchia, un pezzetto alla
  volta.
- Il Q-learning tiene una tabella di voti (una riga per situazione, una colonna
  per mossa) e la corregge giocando: nuovo voto uguale vecchio voto più un po’
  della sorpresa. Impara quali sarebbero le mosse migliori anche mentre si
  muove a casaccio per esplorare, cioè impara una strategia mentre ne segue
  un'altra.
- Nella correzione ci sono due manopole: una decide quanto dare retta alla
  sorpresa dell'ultimo passo (piccola: passi cauti), l'altra quanto pesa il
  futuro rispetto al premio immediato. Per il dilemma del ristorante c'è la
  ricetta $\varepsilon$-greedy: quasi sempre la mossa col voto più alto, ogni
  tanto una a caso per scoprire di meglio. Se quell'«ogni tanto» cala troppo in
  fretta, le novità finiscono, e con loro la promessa che i voti si assestino
  al posto giusto.
- Prendere sempre il voto più alto gonfia in media le stime, perché fra quattro
  numeri che ballano il massimo tende a essere un numero che ha sbagliato per
  eccesso. La contromisura sono due tabelle, una che sceglie la mossa e una che
  le dà il voto.
- SARSA valuta le mosse mettendo in conto che ogni tanto esplorerà e sbaglierà:
  sul bordo del burrone si tiene a distanza di sicurezza, mentre il Q-learning
  cammina sull'orlo perché in teoria non cadrebbe mai. Tolto il dado, la
  differenza sparisce, e la prudenza di SARSA diventa strada in più.
- Guardare avanti un passo solo o fino alla fine della partita sono i due
  estremi di una famiglia: una manciata di passi in genere batte entrambi, e si
  può anche fare la media di tutte le lunghezze, con più peso alle corte.
  Quella media si tiene aggiornata mentre si gioca ricordando le situazioni
  appena attraversate, con un ricordo che sfuma a ogni passo: così una
  ricompensa a sorpresa corregge in un colpo tutta la scia alle spalle, le più
  recenti di più.
- Tutto questo funziona finché le situazioni si possono elencare una per una.
  Per dodici caselle la tabella si scrive; per un videogioco in cui quasi ogni
  schermata è nuova, no, e nemmeno basterebbero le partite per riempirla. Al
  suo posto serve qualcosa che sappia indovinare il voto di una situazione
  mai vista accostandola a quelle già viste: sono le reti neurali del Deep
  Reinforcement Learning.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il TD aggiorna le stime a ogni passo usando la stima dello stato successivo
  (*bootstrapping*), senza attendere la fine dell'episodio: meno varianza e più
  distorsione rispetto a Monte Carlo.
- Il Q-learning impara una tabella $Q(s,a)$ ed è *off-policy*: il suo
  target usa $\max_{a'} Q(s',a')$, quindi apprende la politica ottima anche
  mentre esplora. È la value iteration con l'attesa sostituita da un campione,
  e la convergenza di Watkins e Dayan vale per MDP finiti, $Q$ tabellare,
  ricompense limitate, sconto $\gamma < 1$, visite infinite e passi che
  soddisfano Robbins-Monro sui passaggi di ciascuna coppia.
- Nella formula, $\alpha$ dosa la correzione e $\gamma$ pesa il futuro; la
  strategia $\varepsilon$-greedy bilancia esplorazione e sfruttamento. Delle due
  metà di GLIE (visite infinite, e policy greedy nel limite) al Q-learning,
  che è off-policy, serve solo la prima; al SARSA servono tutte e due, perché
  con $\varepsilon$ fisso si posa sul valore della politica
  $\varepsilon$-greedy invece che su quello ottimo.
- SARSA è la variante *on-policy* ($Q(s',a')$ al posto del massimo): più
  prudente, valuta la politica che segue; Expected SARSA ne prende l'attesa
  sotto $\pi$, e con $\pi$ greedy ridà il Q-learning. Il $\max$ del Q-learning
  porta invece con sé il bias di massimizzazione, che il Double Q-learning
  elimina con il rischio di una sottostima.
- TD e Monte Carlo sono i due estremi di un continuo: il ritorno a $n$
  passi sta in mezzo, e il $\lambda$-return li media tutti (vista in
  avanti). Le tracce di eleggibilità sono la vista all'indietro, che
  distribuisce un solo errore TD su tutta la scia degli stati appena
  visitati: equivalente esatta a pesi fermi per l'episodio, approssimazione
  quando i pesi si muovono a ogni passo.
- Tutto l'impianto presuppone $\mathcal{S}$ finito e abbastanza piccolo da
  stare in memoria. La tabella cede due volte, per memoria (una casella per
  coppia stato-azione) e per dati (quasi ogni stato incontrato è nuovo, e la
  sua riga è vuota). Con una funzione approssimata la garanzia di convergenza
  non vale più, e approssimazione, bootstrapping e off-policy insieme (la
  triade fatale) possono far divergere i valori.
```

`````

Il {doc}`Deep Reinforcement Learning </DeepReinforcementLearning/overview>`
conserva la regola di aggiornamento e il suo bersaglio,
$r + \gamma \max_{a'} Q(s',a')$, e cambia soltanto dove vive $Q$: in una rete
$Q(s,a;\theta)$ con parametri $\theta$, invece che in una riga per ogni coppia
stato-azione. Da lì in poi correggere un voto ne sposta anche altri, e il
bersaglio dipende dagli stessi parametri che si stanno correggendo.
