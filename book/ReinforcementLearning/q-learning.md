# Q-learning e l'apprendimento per differenze temporali

Torniamo per un momento al videogioco della panoramica: premi tasti, il
personaggio si muove, ogni tanto il punteggio sale, e nessuno ti dice quale
delle decine di mosse fatte abbia meritato quei punti. La sezione precedente ha
affrontato il problema nel modo più diretto che ci sia: giocare la partita
intera, guardare quanti punti si sono fatti e usare quel totale per giudicare
tutte le mosse. E si è fermata su una promessa, che questa sezione mantiene: si
può correggere la stima strada facendo, senza aspettare la fine. Ne uscirà
l'algoritmo più celebre del campo.

## Imparare senza aspettare la fine: le differenze temporali

Aspettare la fine funziona, ed è corretto in media, ma costa: il totale di una
singola partita è un numero ballerino, e finché la partita non finisce non si
scrive niente. L'apprendimento per **differenze temporali**
(*temporal-difference*, TD), introdotto da Richard Sutton nel 1988
{cite}`sutton1988learning`, propone invece di aggiornare le stime *durante*
la partita, a ogni singolo passo, usando la ricompensa appena incassata più la
stima che già si ha della situazione in cui si è finiti.

`````{tab} Elementare

Stai stimando quanto dura un viaggio in auto. Parti dicendo "due ore". Dopo
mezz'ora sei più avanti del previsto, e il navigatore dice che ne manca "un'ora
e dieci". Non hai aspettato di arrivare per correggere la tua previsione: hai
usato una **stima più recente** per aggiustare quella vecchia.

Il conto vale la pena farlo per bene, perché è tutto il metodo. La previsione
vecchia diceva due ore, cioè centoventi minuti. La stima nuova è fatta di due
pezzi, quello che è già successo più quello che ancora manca: trenta minuti
percorsi, più settanta che restano, fa cento minuti. La differenza è di venti
minuti, e quei venti minuti sono l'errore: il viaggio andrà meglio di come lo
avevi previsto, e la previsione va tirata giù. (Attenzione a un dettaglio che
inganna: qui il numero che si stima è un tempo, e una bella notizia lo fa
*scendere*. Nel resto del capitolo il numero è un punteggio, e una bella
notizia lo fa salire. Il meccanismo è lo stesso, cambia solo che cosa si
conta.) Il TD learning fa esattamente
questo, correggendo un pezzetto alla volta. E si noti che il primo dei due
pezzi, quello che è già successo, è l'unico dato vero della faccenda: è quello
che tiene la correzione ancorata alla realtà invece che a un'altra opinione.

`````

`````{tab} Superiore

Sia $V(s)$ la stima del valore atteso di uno stato $s$, cioè la ricompensa
totale futura che ci aspettiamo partendo da lì. Dopo aver osservato la
transizione $s \to s'$ con ricompensa **osservata** $r$, l'aggiornamento TD(0)
è

$$
V(s) \leftarrow V(s) + \alpha\,\underbrace{\big[\,r + \gamma\,V(s') -
V(s)\,\big]}_{\text{errore TD}\;\delta} .
$$

Qui $\alpha \in (0,1]$ è il **learning rate**, $\gamma \in [0,1]$ il **fattore
di sconto** e $\delta$ l’**errore TD**. Il termine $r + \gamma V(s')$ è una
stima aggiornata di $V(s)$ costruita *usando la stima successiva* $V(s')$:
questa dipendenza da una stima per aggiornarne un'altra si chiama
*bootstrapping*, ed è esattamente ciò che i metodi Monte Carlo della sezione
precedente non fanno. Il bersaglio non è più il ritorno osservato $G_t$ ma una
sua approssimazione a un passo: si guadagna in varianza (un solo termine
casuale invece di una somma lunga) e si perde in correttezza, perché $V(s')$ è
a sua volta una stima, e all'inizio è sbagliata.

`````

## Q-learning: stimare il valore delle azioni

Conoscere il valore di uno stato non basta per decidere: serve sapere quanto
vale ogni **azione** in quello stato. È il salto del **Q-learning**, formulato
da Chris Watkins nel 1989, forse l'algoritmo più celebre del campo.

`````{tab} Elementare

Immagina una grande tabella: una riga per ogni situazione in cui puoi
trovarti, una colonna per ogni mossa possibile. In ciascuna casella scrivi un
voto che dice "quanto conviene fare questa mossa in questa situazione". Quel
voto, nelle formule, si chiama $Q$, e da lì viene il nome dell'algoritmo.
Perché proprio quella lettera non lo dice nessuno con certezza: la usò Watkins
nella sua tesi ed è rimasta; torna comoda da ricordare come l'iniziale di
*quality*, la qualità di una mossa, anche se il nome ufficiale della cosa è
«funzione azione-valore». All'inizio i voti sono tutti a zero: l'agente non sa
nulla.
Giocando e ricevendo ricompense, corregge i voti. Alla fine, per agire bene,
gli basta guardare la riga della situazione corrente e scegliere la mossa col
voto più alto.

La parte sorprendente: l'agente può muoversi anche a casaccio, sbagliando di
proposito per esplorare, e **imparare comunque quali sarebbero le mosse
migliori**. Impara una cosa mentre ne fa un'altra. Per questo si dice
*off-policy*, cioè "fuori dalla propria strategia".

Il trucco sta in una parola sola della formula, e la si vedrà fra poco: quando
l'agente corregge il voto di una mossa, guarda dove è finito e prende il voto
della **migliore** fra le mosse possibili da lì, non di quella che poi farà
davvero. Quindi anche se subito dopo tira un dado e sbaglia apposta, il conto
che ha appena scritto parlava di un giocatore che non sbaglia.

`````

`````{tab} Superiore

Definiamo la funzione azione-valore ottima $Q^*(s,a)$ come la ricompensa
scontata attesa se in $s$ eseguiamo $a$ e poi seguiamo la politica ottima.
Soddisfa l'equazione di ottimalità di Bellman

$$
Q^*(s,a) = \mathbb{E}\big[\,R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1},a')
\;\big|\; S_t = s,\ A_t = a\,\big],
$$

dove l'attesa è sulla ricompensa e sullo stato d'arrivo,
$S_{t+1} \sim P(\cdot \mid s, a)$.

Il Q-learning è **off-policy** perché il suo *target* usa $\max_{a'} Q(s',a')$
(il valore dell'azione *migliore* nello stato successivo) indipendentemente da
quale azione l'agente abbia poi effettivamente scelto. Impara così la politica
ottima anche mentre ne segue una esplorativa.

Watkins e Dayan {cite}`watkins1992q` ne dimostrarono la convergenza, e le
ipotesi vanno enunciate per intero perché una di esse è la cerniera di tutto il
capitolo seguente: in un MDP **finito**, con $Q$ **tabellare** e ricompense
limitate, purché ogni coppia $(s,a)$ sia visitata infinite volte e i passi
soddisfino le condizioni di Robbins-Monro già viste sui bandit
($\sum_t \alpha_t = \infty$ e $\sum_t \alpha_t^2 < \infty$), $Q$ converge a
$Q^*$ con probabilità $1$. La parola da segnare è **tabellare**: sostituita la
tabella con una funzione approssimata, la dimostrazione non si trasporta, e il
deep reinforcement learning nasce per far funzionare in pratica qualcosa che in
generale non è garantito convergere.

Il $\max$ nel target ha poi un costo che vale la pena nominare, perché il
capitolo successivo introdurrà un algoritmo apposta per correggerlo. Applicato a
stime rumorose, il massimo è uno stimatore distorto **verso l'alto** del massimo
dei valori veri: se in uno stato tutte le azioni valgono davvero zero ma le
stime oscillano attorno allo zero, $\max_{a'}Q(s',a')$ è sistematicamente
positivo. È il **bias di massimizzazione**, e si corregge tenendo due stime
indipendenti e usandone una per scegliere l'azione e l'altra per valutarla
(*Double Q-learning*): è l'idea che nel capitolo seguente diventerà il Double
DQN.

`````

## La formula di aggiornamento

Il cuore dell'algoritmo è una sola riga. In parole suona così: *nuovo voto =
vecchio voto + un po’ della sorpresa*, dove la sorpresa è la differenza tra
com'è andata davvero (premio incassato più prospettive dalla nuova casella) e
come pensavi andasse.

`````{tab} Elementare

Scritta a parole, la riga è questa:

> voto nuovo = voto vecchio + tasso × (bersaglio − voto vecchio)

ed è la stessa forma della regola del quaderno delle leve, in apertura di
capitolo: una stima vecchia, più una frazione della sorpresa. Dentro ci sono
quattro pezzi, e conviene chiamarli per nome perché tornano in tutto il resto
del libro.

- Il **voto vecchio**, quello che c'era scritto nella casella della tabella.
- Il **bersaglio**: quanto quella mossa sembra valere adesso, cioè il premio
  appena incassato più il miglior voto della riga in cui si è finiti, ridotto
  dallo sconto. È una stima migliore della precedente perché contiene un pezzo
  di realtà, il premio appena visto. In inglese si chiama *target*, ed è la
  parola che si incontra nel codice.
- La **sorpresa**, cioè bersaglio meno voto vecchio: positiva se è andata
  meglio del previsto, negativa se peggio.
- Quanta parte della sorpresa dare retta: un numero fra zero e uno che si chiama
  **tasso di apprendimento** (*learning rate*) e nelle formule è la lettera
  greca $\alpha$ (alfa). Vicino a zero l'agente corregge poco per volta ed è
  cauto; vicino a uno butta via il voto vecchio a ogni sorpresa.

E c'è lo sconto, la $\gamma$ (gamma) della sezione sugli MDP, che decide quanto
pesano le prospettive future rispetto al premio incassato subito.

`````

`````{tab} Superiore

In simboli:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\,\Big[\,r + \gamma \max_{a'} Q(s',a') - Q(s,a)\,\Big].
$$

Leggiamola da destra: $r + \gamma \max_{a'} Q(s',a')$ è il **target TD**, la
stima aggiornata del valore di $(s,a)$, dove $r$ è la ricompensa *osservata* in
questa transizione (non la ricompensa attesa $r(s,a)$ del modello, che qui non
conosciamo); sottraendo la stima corrente $Q(s,a)$ otteniamo l'errore; il
learning rate $\alpha$ decide quanto fidarci della correzione (piccolo = passi
cauti); il fattore di sconto $\gamma$ pesa il futuro (vicino a $1$ =
lungimirante, vicino a $0$ = miope).

`````

Nient'altro. Nessuna rete neurale, nessuna delle macchinerie dei capitoli
precedenti: solo una tabella di numeri che si aggiusta a ogni mossa.

## Esplorare o sfruttare: la strategia $\varepsilon$-greedy

Se l'agente scegliesse sempre la mossa col voto più alto, resterebbe
intrappolato nella prima strategia decente che trova, senza mai scoprire
scorciatoie migliori. Deve ogni tanto **esplorare**.

`````{tab} Elementare

È il dilemma del ristorante: torni sempre da quello che conosci e ti piace
(**sfruttare**), o provi il nuovo che ha appena aperto e potrebbe essere una
scoperta (**esplorare**)? La ricetta $\varepsilon$-greedy è semplice: nella grande
maggioranza dei casi vai sul sicuro, ma con una piccola probabilità (chiamata
$\varepsilon$, epsilon, ad esempio il 10%) tiri un dado e provi una mossa a
caso. All'inizio esplori molto; man mano che impari, riduci $\varepsilon$ e ti
affidi sempre più a ciò che sai.

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

Un $\varepsilon$ costante garantisce esplorazione perpetua; in pratica si usa un
*decay*, spesso esponenziale, per convergere gradualmente allo sfruttamento
puro. Un decadimento esponenziale, però, sacrifica la garanzia teorica: la
convergenza appena citata vuole ogni coppia $(s,a)$ visitata infinite volte, e
per assicurarlo serve $\sum_t \varepsilon_t = \infty$, che con
$\varepsilon_t \propto 1/t$ si ottiene. Questa condizione, unita al fatto che la
policy diventi greedy nel limite, è ciò che si chiama **GLIE** (*greedy in the
limit with infinite exploration*), sotto cui anche il SARSA di qui a poco
converge alla policy ottima {cite}`singh2000convergence`: si noti che
GLIE è la coppia di requisiti, non la ricetta $\varepsilon_t = 1/t$, che ne è
soltanto un modo comodo di soddisfarli. Con un $\varepsilon$ che si spegne
esponenzialmente le mosse esplorative sono invece quasi certamente in numero
finito, perché una serie geometrica di ragione minore di uno ha somma finita.
In pratica lo scambio
si accetta. La scelta di $\varepsilon$ regola il compromesso
*exploration–exploitation*, uno dei nodi teorici centrali del reinforcement
learning.

`````

## SARSA: la variante on-policy

C'è un cugino stretto del Q-learning che cambia un solo simbolo nella formula,
con conseguenze interessanti. Si chiama **SARSA**, dalle iniziali dei cinque
ingredienti del suo aggiornamento: stato, azione, ricompensa, nuovo stato,
nuova azione: sono le iniziali inglesi, *state, action, reward, state,
action*, che a differenza di quelle italiane compongono una parola
pronunciabile. E si dice **on-policy**, che è il contrario di off-policy:
invece di imparare quanto varrebbero le mosse di un giocatore perfetto, impara
quanto valgono le proprie, esplorazione compresa. L'algoritmo è di Rummery e Niranjan
{cite}`rummery1994online`, che però lo chiamavano *modified connectionist
Q-learning*; il nome con cui lo conosciamo oggi arriva da Sutton qualche anno
dopo, nel 1996.

`````{tab} Elementare

Il Q-learning è un ottimista spericolato: valuta ogni mossa immaginando di
comportarsi *perfettamente* subito dopo. SARSA è più prudente: valuta le mosse
tenendo conto che, di tanto in tanto, esplorerà davvero e potrebbe sbagliare.
Impara il valore della politica che **effettivamente segue**, esplorazione
compresa.

Il risultato tipico si vede su un esperimento classico, che si chiama
*cammino sul precipizio*: un corridoio di caselle il cui bordo inferiore è un
burrone, con la partenza e l'arrivo alle due estremità. La strada più corta
passa proprio sull'orlo, e un passo storto fa cadere di sotto. SARSA impara a salire fin sopra, lontano dal bordo, e ci arriva in qualche
passo in più senza cadere mai; il
Q-learning impara a camminare sull'orlo, perché "in teoria" non sbaglierebbe
mai un passo, e ogni tanto ci casca davvero.

`````

`````{tab} Superiore

SARSA è **on-policy**: nel target non compare il massimo, ma il valore
dell'azione $a'$ realmente scelta nello stato $s'$ dalla stessa politica (ad
esempio $\varepsilon$-greedy):

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\,\big[\,r + \gamma\,Q(s',a') - Q(s,a)\,\big].
$$

Valuta dunque la politica di comportamento anziché quella greedy. Nel classico
esempio del *cliff walking* (Sutton e Barto), che è un compito episodico
**non scontato** ($\gamma = 1$, con $-1$ su ogni transizione e una caduta che
costa $-100$), SARSA converge a un cammino più sicuro e lontano dal precipizio,
il Q-learning al cammino ottimo ma rischioso lungo il bordo: differenza che
sparisce solo quando $\varepsilon \to 0$.

`````

## Un labirinto concreto

Rendiamo tutto tangibile con una griglia di tre righe per quattro colonne.
L'agente parte in basso a sinistra (**S**); la meta, che paga $+1$, è in alto a
destra; la trappola, che fa perdere un punto e chiude comunque la partita, è
subito sotto la meta; e c'è un muro nella casella centrale della riga di mezzo,
contro cui si sbatte restando fermi. A ogni passo si sceglie fra su, giù,
sinistra, destra.

Attenzione alle regole, perché non sono quelle del labirinto della sezione
sugli MDP. Qui i passi **non costano nulla**: girovagare non fa perdere punti.
A spingere l'agente verso l'uscita c'è soltanto lo sconto, che rende il premio
meno appetitoso quanto più lo si fa aspettare, ed è quindi lui, da solo, a
rendere conveniente la strada corta.

```{figure} ../figures/labirinto-qlearning.svg
:name: fig-labirinto
:alt: Griglia 3x4 con cella di partenza in basso a sinistra, meta con ricompensa +1 in alto a destra, trappola -1 sotto la meta, un muro al centro e frecce che indicano la politica appresa in ogni cella.
:width: 85%

La strategia imparata dal Q-learning sulla griglia: in ogni cella la freccia
indica la mossa che, finite tutte le partite di allenamento, ha il voto più
alto. È il risultato che stampa il codice di qui a poco. Il disegno mostra il
punto d'arrivo, non la strada per arrivarci: quella, cioè il valore della meta
che retrocede una casella per volta, si vede solo guardando i voti cambiare
partita dopo partita.
```

All'inizio la tabella dei voti è tutta a zero. Fissiamo un tasso di
apprendimento di $0{,}5$ (si dà retta a metà della sorpresa) e uno sconto di
$0{,}9$.

`````{tab} Elementare

La prima volta che l'agente calpesta la meta incassa $1$ e la partita finisce
lì, quindi dalla casella d'arrivo non c'è più niente da aspettarsi. La sorpresa
è tutta lì: si aspettava $0$, ha incassato $1$. Dandole retta a metà, il voto
dell'ultima mossa passa da $0$ a $0{,}5$.

Alla casella che veniva prima tocca solo una partita dopo, e conviene capire
perché: quando l'agente ci è passato, in questa partita, il voto della casella
d'arrivo era ancora zero, e correggere verso zero non muove niente. Adesso
invece il $0{,}5$ c'è, e alla prossima partita servirà. Da lì la mossa non paga
niente, ma porta in una casella la cui riga contiene ormai una mossa da $0{,}5$
(i voti stanno sulle mosse, e quello che conta qui è il migliore della riga):
scontato, vale $0{,}9 \times 0{,}5 = 0{,}45$. La sorpresa è di nuovo positiva
(si aspettava $0$, la prospettiva vale $0{,}45$) e dandole retta a metà il voto
diventa $0{,}225$.

Nota che tutti e due i conti funzionano perché qui muoversi non costa nulla:
nel labirinto della sezione sugli MDP, dove ogni passo faceva perdere un punto,
la seconda mossa avrebbe reso $-1 + 0{,}45 = -0{,}55$, e il voto sarebbe
diventato $-0{,}275$ invece di $0{,}225$.

`````

`````{tab} Superiore

Poniamo $\alpha=0{,}5$ e $\gamma=0{,}9$. La prima volta che l'agente calpesta
la meta ($r=+1$, stato successivo terminale con valore $0$), la casella
dell'ultima mossa diventa

$$
Q(s,\rightarrow) \leftarrow 0 + 0{,}5\,\big[\,1 + 0{,}9\cdot 0 - 0\,\big] = 0{,}5 .
$$

Un episodio dopo, la cella precedente $s^-$, da cui si arriva a $s$, "vede"
$\max_{a'} Q(s,a')=0{,}5$ e si aggiorna:

$$
Q(s^-,\rightarrow) \leftarrow 0 + 0{,}5\,\big[\,0 + 0{,}9\cdot 0{,}5 -
0\,\big] = 0{,}225 .
$$

Entrambi i conti usano $r$ come ricompensa **osservata** e sfruttano il fatto
che in questo mondo la transizione non paga alcun costo di passo: in un mondo
che penalizzasse ogni mossa con $-1$, il secondo aggiornamento darebbe
$0{,}5\,[-1 + 0{,}45] = -0{,}275$.

`````

Ecco il meccanismo TD in azione: la ricompensa non salta dappertutto in una
volta, ma **retrocede** verso la partenza un passo per episodio, come una
macchia che si allarga all'indietro dalla meta. In codice sta tutto in una
pagina, ambiente compreso:

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
    """Dove si finisce, quanto si incassa, se la partita e' finita."""
    i, j = cella[0] + MOSSE[a][0], cella[1] + MOSSE[a][1]
    if not (0 <= i < RIGHE and 0 <= j < COLONNE) or (i, j) == MURO:
        i, j = cella                          # contro un muro si resta fermi
    if (i, j) == META:     return (i, j), 1.0, True
    if (i, j) == TRAPPOLA: return (i, j), -1.0, True
    return (i, j), 0.0, False

def epsilon_greedy(s):
    if rng.random() < epsilon:
        return int(rng.integers(n_azioni))    # esplora: mossa a caso
    return int(np.argmax(Q[s]))               # sfrutta: mossa col voto piu alto

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

# > 0.81 | > 0.90 | > 1.00 |     +1
# ^ 0.73 |   muro | ^ 0.90 |     -1
# ^ 0.66 | > 0.73 | ^ 0.81 | < 0.73
```

Ecco che cosa stampa, disposto come la griglia:

| | | | |
|:--|:--|:--|:--|
| → $0{,}81$ | → $0{,}90$ | → $1{,}00$ | **meta**, $+1$ |
| ↑ $0{,}73$ | muro | ↑ $0{,}90$ | **trappola**, $-1$ |
| ↑ $0{,}66$ | → $0{,}73$ | ↑ $0{,}81$ | ← $0{,}73$ |

Le frecce sono, una per una, quelle della {numref}`fig-labirinto`, e i numeri
accanto si controllano a mano. La casella da cui basta una mossa per arrivare
vale il premio pieno, $1{,}00$; ogni passo indietro lo moltiplica per lo
sconto, e viene $0{,}90$, poi $0{,}81$, poi $0{,}729$ e poi $0{,}6561$, che il
programma stampa arrotondati a $0{,}73$ e $0{,}66$. Ci sono tutti: l'angolo in
basso a sinistra, che dalla meta dista cinque passi ed è la casella più
lontana, porta appunto il $0{,}66$.

Un momento, però: la casella da cui basta una mossa per arrivare, quella che
qui vale $1{,}00$, poco fa valeva $0{,}5$. Non è una
contraddizione, sono due istantanee della stessa storia. Il $0{,}5$ era il voto
dopo il **primo** passaggio, quando alla sorpresa si dava retta a metà
partendo da zero. Per quella mossa il bersaglio resta sempre $1$, perché
incassa il premio e la partita finisce lì: a ogni passaggio successivo, quindi,
il voto recupera metà della distanza che lo separa da $1$, e diventa $0{,}75$,
poi $0{,}875$, poi $0{,}9375$, e così via. Dopo
cinquemila partite ci è arrivato così vicino che, alla seconda cifra, si legge
$1{,}00$.

Per le altre caselle succede la stessa cosa, con una complicazione in più da
nominare perché è tutto il capitolo in miniatura: il loro bersaglio non sta
fermo. La casella accanto era partita rincorrendo $0{,}45$, cioè lo sconto per
il $0{,}5$ che c'era allora; ma mentre lei ci correva dietro, quel $0{,}5$ è
salito verso $1$, e quindi il bersaglio è salito verso $0{,}90$. Ogni casella
insegue un numero che a sua volta sta salendo, e la fila si assesta
dall'ultima all'indietro. I conti a mano di poco fa dicono da dove parte
ciascun voto, la tabella qui sopra dice dove arriva.

Tre note sul codice.

La funzione `aggiorna` è il Q-learning per intero, tutto qui. Per ottenere
invece SARSA basterebbe passarle l'azione che l'agente sceglierà davvero al
passo successivo, e usare il voto di *quella* al posto del voto della mossa
migliore.

Il tasso di apprendimento (`alpha`) qui resta fermo a metà per tutte e
cinquemila le partite. Un tasso che non si accorcia mai continua per sempre a
rincorrere le ultime sorprese, invece di posarsi su un numero, ed è una scelta
che ha un prezzo. Esiste infatti un teorema, di cui la fine della sezione dice
qualcosa in più, che promette che i voti finiscono prima o poi al posto giusto;
e fra le cose che chiede c'è proprio un tasso che si riduca col tempo, come il
passo del quaderno delle leve all'inizio del capitolo. Qui ci si rinuncia, in
cambio di un algoritmo che
reagisce in fretta, che nella pratica conviene quasi sempre.

Ogni partita comincia da una casella sorteggiata invece che dalla partenza. Si
chiamano **inizi esplorativi**, e dentro un simulatore, che possiamo far
ripartire dove vogliamo, costano una riga. Senza, le caselle fuori dal cammino
migliore verrebbero visitate troppo di rado, la loro riga resterebbe quasi
vuota e la loro freccia sarebbe poco più di un sorteggio. Si provi a far
cominciare tutte le partite dalla partenza, in basso a sinistra, e a guardare
l'angolo in basso a destra, che sul cammino migliore non sta. Con i sorteggi
scritti nel codice (il numerone accanto a `default_rng` è il **seme**: fissa la
sequenza dei numeri a caso, così che rilanciando il programma esca la stessa
identica storia) l'agente ci capita ventidue volte invece di
settecentocinquanta, e il suo voto si ferma a $0{,}55$ invece che a $0{,}73$.
Cambiando seme, il più delle volte non ci capita mai: quella riga resta tutta a
zero, e la freccia è quella che viene.

## Fra un passo e la fine: quanti passi guardare avanti

Torniamo un momento alla macchia che si allarga all'indietro dalla meta. Il
Q-learning la fa retrocedere **di una casella per partita**, perché il suo
bersaglio guarda avanti di un passo solo. Monte Carlo, all'estremo opposto,
usa il totale della partita, e in una partita sola porta la notizia a tutte le
caselle attraversate: una notizia sola, però, e rumorosa. Detta così, la
scelta sembra fra due
poli. Non lo è: fra i due c'è un continuo, e si attraversa con una manopola.

`````{tab} Elementare

La domanda è: **quanti passi guardo prima di fidarmi della mia stima?** Uno
solo (e allora ho il TD), tutti fino alla fine (e allora ho Monte Carlo),
oppure tre, o dieci.

Guardare pochi passi dà una correzione stabile ma quasi sempre un po’
sbagliata, perché si appoggia a una stima che, quando si è appena cominciato a
giocare, non vale niente. Guardare fino in fondo dà una correzione sempre
onesta ma ballerina.
Guardarne una manciata, in pratica, batte quasi sempre entrambi gli estremi.

C'è anche un modo elegante di non scegliere: fare la **media di tutte le
lunghezze**, dando più peso a quelle corte e via via meno a quelle lunghe. Il
peso cala di una frazione fissa a ogni passo in più, come un'eco che si spegne,
e la manopola che decide quanto in fretta si spenga si chiama $\lambda$
(lambda), un numero fra zero e uno.

I due estremi si capiscono guardando come si spartisce il peso. La lunghezza
più corta, guardare avanti un passo, si prende quello che l'eco lascia fuori
subito, cioè quanto manca a $\lambda$ per arrivare a uno: con
$\lambda = 0{,}5$ si prende metà di tutto il peso, con $\lambda = 0{,}9$ soltanto
un decimo. Quel che avanza se lo spartiscono le lunghezze successive, sempre
con la stessa regola.

Con $\lambda = 0$ la prima si prende tutto, e siamo tornati alle differenze
temporali. Alzando $\lambda$ il peso scivola sempre più in fondo alla fila; e in
fondo alla fila c'è la fine della partita, dove «guardare avanti dieci passi» e
«guardarne cento» sono ormai la stessa identica cosa, cioè guardare fino in
fondo. A $\lambda = 1$ è tutto lì che va a finire: resta il totale della
partita, e siamo tornati a Monte Carlo. In mezzo c'è tutto il resto.

Detta così sembra impossibile da fare mentre si gioca, perché quella media
guarda avanti, e il futuro non lo si conosce. Il trucco è guardare
dall'altra parte: invece di chiedersi "che cosa succederà dopo questa casella",
si tiene un elenco delle caselle appena attraversate, ciascuna con un ricordo
che sfuma a ogni passo. Quel ricordo si chiama **traccia**, e quando arriva una
sorpresa la si distribuisce a tutta la scia, tanto più forte quanto più recente
è il passaggio. Che venga davvero lo stesso risultato non è ovvio ed è un conto da fare; l'idea è che dare a ogni casella un
pezzetto di correzione alla volta, per tutta la partita, alla fine somma
esattamente quanto le si sarebbe dato in un colpo solo guardando avanti. Il
guadagno è che così non si aspetta mai la fine.

`````

`````{tab} Superiore

Il **ritorno a $n$ passi** tronca la somma dopo $n$ ricompense vere e chiude
con la stima corrente:

$$
G_{t:t+n} = R_{t+1} + \gamma\, R_{t+2} + \cdots + \gamma^{n-1} R_{t+n}
+ \gamma^{n} V(S_{t+n}),
$$

e l'aggiornamento è il solito
$V(S_t) \leftarrow V(S_t) + \alpha\,[\,G_{t:t+n} - V(S_t)\,]$.
Per $n=1$ si ritrova TD(0); per $n$ pari o superiore alla durata
dell'episodio il termine con $V$ sparisce e resta il ritorno Monte Carlo.
Il compromesso è quello classico fra distorsione e varianza: $n$ piccolo poca
varianza e molta distorsione, $n$ grande il contrario. Nei banchi di prova di
Sutton e Barto l'ottimo sta quasi sempre a valori intermedi, non agli estremi
{cite}`sutton2018reinforcement`.

Il **$\lambda$-return** evita di dover scegliere $n$: è la media pesata di
*tutti* i ritorni a $n$ passi, con pesi che decadono geometricamente,

$$
G_t^{\lambda} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{\,n-1}\, G_{t:t+n},
\qquad \lambda \in [0,1],
$$

con $\lambda = 0$ che restituisce TD(0). E $\lambda = 1$? In un episodio che
termina al passo $T$ tutti i ritorni con $n \ge T-t$ coincidono con il ritorno
intero $G_t$; raccogliendone i pesi, la coda della somma si compatta in un
termine $\lambda^{\,T-t-1}\, G_t$, che a $\lambda = 1$ è l'unico a
sopravvivere: si ritrova Monte Carlo.

Scritta così la formula è impraticabile, perché richiede di conoscere il futuro:
è la **vista in avanti**. Le **tracce di eleggibilità** danno la stessa cosa
dalla **vista all'indietro**, calcolabile passo per passo mentre si gioca: si
tiene un vettore $\mathbf{z}$ che segna quali stati sono «in attesa di credito»,

$$
z_t(s) = \gamma\lambda\, z_{t-1}(s) + \mathbb{1}[S_t = s],
$$

cioè la traccia di uno stato sale di $1$ quando lo si visita e sfuma di
$\gamma\lambda$ a ogni passo successivo. A ogni istante si calcola **un solo**
errore TD $\delta_t$ e lo si distribuisce a tutti gli stati in proporzione
alla loro traccia: $V(s) \leftarrow V(s) + \alpha\,\delta_t\, z_t(s)$. Una
ricompensa inattesa corregge così in un colpo tutta la scia di stati che
l'hanno preceduta, i più recenti di più. L'occupazione di memoria è una traccia
per stato, cioè lo stesso ordine delle stime di valore che si tengono già
($O(|\mathcal{S}|)$ nel caso tabellare, $O(d)$ con approssimazione lineare): il
guadagno è che non cresce né con la lunghezza dell'episodio né con $\lambda$,
mentre la vista in avanti a $n$ passi dovrebbe tenere in memoria gli ultimi $n$
stati. Le due viste danno lo stesso risultato, in modo esatto sotto opportune
varianti {cite}`sutton2018reinforcement`.

`````

Sul labirinto la differenza si vede a occhio: con le tracce, il primo episodio
che tocca la meta non illumina solo l'ultima casella, illumina tutta la strada
percorsa, in dissolvenza. Il che, detto in modo meno pittoresco, è il motivo
per cui i metodi multi-passo imparano più in fretta quando le ricompense sono
rare.

Questa manopola non è un residuo storico, e la si ritroverà identica nel
capitolo successivo. Là il segnale che guida l'apprendimento si chiama
*vantaggio* di una mossa, ed è quanto quella mossa è migliore della media delle
mosse possibili in quella situazione. E la sorpresa di un passo di cui si è
appena detto, guardata da vicino, è già una misura di quello: dice di quanto la
mossa fatta è andata meglio di come ci si aspettava. Nella sua forma più
semplice il vantaggio è proprio lei; e il modo
standard di calcolarlo è questa identica media pesata, con questo identico
$\lambda$. Si sceglie di nuovo la stessa cosa: quanto accettare che il
bersaglio sia storto, in cambio di quanto farlo ballare di meno.

## Quando la tabella non basta più

Tutto quello che si è letto in questo capitolo poggia su un'ipotesi che non
abbiamo mai dovuto nominare, perché nei nostri esempi era ovvia: che le
situazioni si possano **elencare**, una riga di tabella ciascuna. Nel labirinto
sono dodici. In un gioco da tavolo sono più delle molecole d'aria di questa
stanza. Sullo schermo di un videogioco, dove ogni fotogramma diverso è una
situazione diversa, la tabella non si può scrivere in nessun universo, come
dicono i conti della sezione sugli MDP.

E si rompe due volte, non una. Si rompe per **memoria**, perché servirebbe una
casella per ogni coppia situazione-mossa. E si rompe per **dati**, che è il
guasto peggiore: anche avendo la tabella, quasi ogni situazione che l'agente
incontra non l'ha mai vista prima, quindi la sua riga è ancora vuota, e riempire
per esperienza diretta ogni riga di una tabella così grande richiederebbe più
partite di quante se ne possano giocare.

La via d'uscita non è un algoritmo diverso: le idee di questo capitolo (la
sorpresa che corregge, il bersaglio a un passo, l'esplorazione dosata) restano
tutte. È una **rappresentazione** diversa. Al posto della tabella serve
qualcosa che, vista una situazione mai incontrata, sappia indovinarne i voti
somigliandola a quelle che ha già visto, e quel qualcosa sono le reti neurali
dei capitoli precedenti. È il capitolo che comincia alla pagina dopo questa, ed
è la ragione per cui esiste.

Con un'avvertenza che conviene portarsi dietro. Che il Q-learning arrivi prima
o poi ai voti giusti non è una speranza, è un teorema, dimostrato nel 1992 da
Watkins insieme a Peter Dayan; ma quel teorema parla di una tabella, e di una
tabella soltanto. Buttata via la tabella, la promessa non c'è più. Il deep
reinforcement learning è in buona parte il mestiere di far funzionare lo stesso
qualcosa che nessuno ha dimostrato che funzioni.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Le **differenze temporali** correggono la stima durante il viaggio, non
  all'arrivo: come il navigatore che dopo mezz'ora rivede il tempo che manca,
  si usa la stima più recente per aggiustare quella vecchia, un pezzetto alla
  volta.
- Il **Q-learning** tiene una tabella di voti (una riga per situazione, una
  colonna per mossa) e la corregge giocando: nuovo voto uguale vecchio voto più
  un po’ della sorpresa. Impara quali sarebbero le mosse migliori anche mentre
  si muove a casaccio per esplorare, cioè impara una cosa mentre ne fa
  un'altra.
- Nella correzione ci sono due manopole: una decide quanto dare retta alla
  sorpresa dell'ultimo passo (piccola vuol dire passi cauti), l'altra quanto
  pesa il futuro rispetto al premio immediato. E c'è la ricetta $\varepsilon$-greedy per il
  dilemma del ristorante: quasi sempre la mossa col voto più alto, ogni tanto
  una a caso per scoprire di meglio.
- **SARSA** valuta le mosse mettendo in conto che ogni tanto esplorerà davvero
  e sbaglierà: sul bordo del burrone si tiene a distanza di sicurezza, mentre
  il Q-learning cammina sull'orlo perché in teoria non cadrebbe mai.
- Guardare avanti un passo solo o fino alla fine della partita sono i due
  estremi di un continuo: una manciata di passi in genere batte entrambi, e si
  può anche non scegliere, facendo la media di tutte le lunghezze con più peso
  alle corte. Quella media si tiene aggiornata **mentre si gioca**, senza
  aspettare la fine: basta ricordare quali situazioni si sono appena
  attraversate, con un ricordo che sfuma a ogni passo. Così una ricompensa a
  sorpresa corregge in un colpo tutta la scia alle spalle, le più recenti di
  più: nel labirinto, il primo episodio che tocca la meta non illumina solo
  l'ultima casella, illumina tutta la strada percorsa, in dissolvenza.
- Tutto questo funziona finché le situazioni si possono elencare una per una.
  Per dodici caselle la tabella si scrive; per un videogioco in cui quasi ogni
  schermata è nuova, no, e nemmeno basterebbero le partite per riempirla. Al
  suo posto serve qualcosa che sappia **indovinare** il voto di una situazione
  mai vista somigliandola a quelle già viste: sono le reti neurali del Deep
  Reinforcement Learning.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il **temporal-difference** aggiorna le stime a ogni passo usando la stima
  successiva (*bootstrapping*), senza attendere la fine dell'episodio.
- Il **Q-learning** impara una tabella $Q(s,a)$ ed è *off-policy*: il suo
  target usa $\max_{a'} Q(s',a')$, quindi apprende la politica ottima anche
  mentre esplora. La convergenza di Watkins e Dayan vale per MDP **finiti**,
  $Q$ **tabellare**, ricompense limitate, visite infinite e passi che
  soddisfano Robbins-Monro.
- Nella formula, $\alpha$ dosa la correzione e $\gamma$ pesa il futuro; la
  strategia **$\varepsilon$-greedy** bilancia esplorazione e sfruttamento, e la condizione
  **GLIE** (visite infinite più policy greedy nel limite) è ciò che serve alla
  garanzia.
- **SARSA** è la variante *on-policy* ($\gamma\,Q(s',a')$ al posto del massimo):
  più prudente, valuta la politica che segue davvero. Il $\max$ del Q-learning
  porta invece con sé il **bias di massimizzazione**, che il Double Q-learning
  corregge.
- TD e Monte Carlo sono i due estremi di un continuo: il ritorno a **$n$
  passi** sta in mezzo, e il **$\lambda$-return** li media tutti (vista in
  avanti). Le **tracce di eleggibilità** sono la vista all'indietro
  equivalente, calcolabile online distribuendo un solo errore TD su tutta la
  scia degli stati appena visitati.
- Tutto l'impianto presuppone $\mathcal{S}$ enumerabile. Cade due volte, per
  **memoria** (una casella per coppia stato-azione) e per **dati** (quasi ogni
  stato incontrato è nuovo, e la sua riga è vuota): la via d'uscita non è un
  algoritmo diverso ma una **rappresentazione** diversa, e con essa se ne va la
  garanzia di convergenza.
```

`````

Il capitolo si chiude con una tabella in mano, ed è proprio lei a rompersi
appena il mondo diventa grande. Tutto il resto regge: correggere una stima con
la stima successiva, dosare quella correzione, decidere quanto pesa il futuro,
tentare ogni tanto una strada nuova per non affezionarsi alla prima trovata. In
**Deep Reinforcement Learning** la tabella lascia il posto a una rete, che le
situazioni non le elenca ma le riconosce; e insieme alla tabella se ne va la
certezza che il metodo converga.
