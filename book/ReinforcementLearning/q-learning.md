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
l'episodio, a ogni singolo passo, usando la ricompensa appena incassata più la
stima che già si ha della situazione in cui si è finiti.

`````{tab} Elementare

Pensa a stimare quanto manca all'arrivo di un viaggio in auto. Parti dicendo
"due ore". Dopo mezz'ora sei più avanti del previsto e il navigatore aggiorna:
"un'ora e dieci". Non hai aspettato di arrivare per correggere la tua
previsione: hai usato una **stima più recente** per aggiustare quella vecchia.
Il TD learning fa esattamente questo. La differenza tra la stima nuova (più
informata) e quella vecchia è l'"errore" che usiamo per correggere, un pezzetto
alla volta.

`````

`````{tab} Superiore

Sia $V(s)$ la stima del valore atteso di uno stato $s$, cioè la ricompensa
totale futura che ci aspettiamo partendo da lì. Dopo aver osservato la
transizione $s \to s'$ con ricompensa **osservata** $r$, l'aggiornamento TD(0)
è

$$
V(s) \leftarrow V(s) + \alpha\,\underbrace{\big[\,r + \gamma\,V(s') - V(s)\,\big]}_{\text{errore TD}\;\delta} .
$$

Qui $\alpha \in (0,1]$ è il **learning rate**, $\gamma \in [0,1]$ il **fattore
di sconto** e $\delta$ l'**errore TD**. Il termine $r + \gamma V(s')$ è una
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
voto (la $Q$) che dice "quanto conviene fare questa mossa in questa
situazione". All'inizio i voti sono tutti a zero: l'agente non sa nulla.
Giocando e ricevendo ricompense, corregge i voti. Alla fine, per agire bene,
gli basta guardare la riga della situazione corrente e scegliere la mossa col
voto più alto.

La parte sorprendente: l'agente può muoversi anche a casaccio, sbagliando di
proposito per esplorare, e **imparare comunque quali sarebbero le mosse
migliori**. Impara una cosa mentre ne fa un'altra. Per questo si dice
*off-policy*, cioè "fuori dalla propria strategia".

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
vecchio voto + un po' della sorpresa*, dove la sorpresa è la differenza tra
com'è andata davvero (premio incassato più prospettive dalla nuova casella) e
come pensavi andasse.

`````{tab} Elementare

Dentro quella riga ci sono quattro pezzi, e conviene chiamarli per nome perché
tornano in tutto il resto del libro.

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

E c'è la vecchia conoscenza $\gamma$ (gamma), lo sconto, che decide quanto
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

Nient'altro: nessun gradiente, nessuna rete neurale. Solo una tabella che si
aggiusta.

## Esplorare o sfruttare: la strategia ε-greedy

Se l'agente scegliesse sempre la mossa col voto più alto, resterebbe
intrappolato nella prima strategia decente che trova, senza mai scoprire
scorciatoie migliori. Deve ogni tanto **esplorare**.

`````{tab} Elementare

È il dilemma del ristorante: torni sempre da quello che conosci e ti piace
(**sfruttare**), o provi il nuovo che ha appena aperto e potrebbe essere una
scoperta (**esplorare**)? La ricetta ε-greedy è semplice: nella grande
maggioranza dei casi vai sul sicuro, ma con una piccola probabilità (chiamata
$\varepsilon$, epsilon, ad esempio il 10%) tiri un dado e provi una mossa a
caso. All'inizio esplori molto; man mano che impari, riduci $\varepsilon$ e ti
affidi sempre più a ciò che sai.

`````

`````{tab} Superiore

Data la tabella corrente, la politica ε-greedy sceglie

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
nuova azione. L'algoritmo è di Rummery e Niranjan
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
passa proprio sull'orlo, e un passo storto fa cadere di sotto. SARSA impara a
tenersi una fila più in su, perdendo qualche passo ma non cadendo mai; il
Q-learning impara a camminare sull'orlo, perché "in teoria" non sbaglierebbe
mai un passo, e ogni tanto ci casca davvero.

`````

`````{tab} Superiore

SARSA è **on-policy**: nel target non compare il massimo, ma il valore
dell'azione $a'$ realmente scelta nello stato $s'$ dalla stessa politica (ad
esempio ε-greedy):

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

Rendiamo tutto tangibile con una griglia. L'agente parte in basso a sinistra
(**S**), deve raggiungere la meta (**+1**) evitando la trappola (**−1**) e il
muro. A ogni passo sceglie tra su, giù, sinistra, destra.

Attenzione alle regole, che non sono quelle del labirinto della sezione sugli
MDP: qui i passi **non costano nulla**, e a spingere l'agente verso l'uscita
invece di farlo girovagare non è una penalità per ogni mossa ma soltanto lo
sconto, che rende il premio meno appetitoso quanto più lo si fa aspettare.

```{figure} ../figures/labirinto-qlearning.svg
:name: fig-labirinto
:alt: Griglia 3x4 con cella di partenza in basso a sinistra, meta con ricompensa +1 in alto a destra, trappola -1 sotto la meta, un muro al centro e frecce che indicano la politica appresa in ogni cella.
:width: 85%

La politica appresa dal Q-learning sulla griglia: da ogni cella la freccia
indica l'azione con $Q$ più alta a fine addestramento. È il risultato che
stampa il codice di qui a poco; la strada per arrivarci, cioè il valore della
meta che retrocede una casella per volta, il disegno non la mostra.
```

All'inizio la tabella dei voti è tutta a zero. Fissiamo un tasso di
apprendimento di $0{,}5$ (si dà retta a metà della sorpresa) e uno sconto di
$0{,}9$.

`````{tab} Elementare

La prima volta che l'agente calpesta la meta incassa $1$ e la partita finisce
lì, quindi dalla casella d'arrivo non c'è più niente da aspettarsi. La sorpresa
è tutta lì: si aspettava $0$, ha incassato $1$. Dandole retta a metà, il voto
dell'ultima mossa passa da $0$ a $0{,}5$.

Alla casella che veniva prima tocca solo un episodio dopo, e conviene capire
perché: quando l'agente ci è passato, in questa partita, il voto della casella
d'arrivo era ancora zero, e correggere verso zero non muove niente. Adesso
invece il $0{,}5$ c'è, e alla prossima partita servirà. Da lì la mossa non paga
niente, ma porta in una casella dove ormai c'è scritto un voto di $0{,}5$:
scontato, vale $0{,}9 \times 0{,}5 = 0{,}45$. La sorpresa è di nuovo positiva
(si aspettava $0$, la prospettiva vale $0{,}45$) e dandole retta a metà il voto
diventa $0{,}225$.

Nota che tutti e due i conti funzionano perché qui muoversi non costa nulla: nel
labirinto della sezione sugli MDP, dove ogni passo costava $-1$, il secondo
numero sarebbe stato negativo.

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
Q(s^-,\rightarrow) \leftarrow 0 + 0{,}5\,\big[\,0 + 0{,}9\cdot 0{,}5 - 0\,\big] = 0{,}225 .
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

Quelle frecce sono, una per una, quelle della {numref}`fig-labirinto`, e i
numeri accanto si controllano a mano: la casella da cui basta una mossa per
arrivare vale il premio pieno, $1{,}00$, e ogni passo indietro lo moltiplica per
lo sconto, $0{,}90$, poi $0{,}81$, poi $0{,}73$.

Tre note sul codice. La funzione `aggiorna` è il Q-learning per intero; per
ottenere SARSA basterebbe passare l'azione realmente scelta `a_next` e
sostituire `np.max(Q[s_next])` con `Q[s_next, a_next]`. `alpha` qui è costante,
e un passo che non si accorcia mai insegue per sempre le ultime sorprese invece
di assestarsi: si rinuncia alla garanzia teorica di convergenza, che chiede un
passo che rimpicciolisca col tempo come sui bandit, in cambio di una reattività
che in pratica conviene quasi sempre. E ogni episodio comincia da una casella
sorteggiata invece che dalla partenza: si chiamano **inizi esplorativi**, e
dentro un simulatore, che possiamo far ripartire dove vogliamo, costano una
riga. Senza, le caselle fuori dal cammino migliore verrebbero visitate troppo di
rado, la loro riga resterebbe quasi vuota e la loro freccia sarebbe un
sorteggio: si provi a far cominciare tutti gli episodi dalla partenza, in basso
a sinistra, e a guardare che cosa succede all'ultima riga della griglia.

## Fra un passo e la fine: quanti passi guardare avanti

Torniamo un momento alla macchia che si allarga all'indietro dalla meta. Il
Q-learning la fa retrocedere **di una casella per episodio**, perché il suo
bersaglio guarda avanti di un passo solo. Monte Carlo, all'estremo opposto,
usa il ritorno intero e in un solo episodio informa tutte le caselle
attraversate, ma con un numero rumoroso. Detta così, la scelta sembra fra due
poli. Non lo è: fra i due c'è un continuo, e si attraversa con una manopola.

`````{tab} Elementare

La domanda è: **quanti passi guardo prima di fidarmi della mia stima?** Uno
solo (e allora ho il TD), tutti fino alla fine (e allora ho Monte Carlo),
oppure tre, o dieci.

Guardare pochi passi dà una correzione stabile ma quasi sempre un po'
sbagliata, perché si appoggia a una stima che a inizio addestramento non vale
niente. Guardare fino in fondo dà una correzione sempre onesta ma ballerina.
Guardarne una manciata, in pratica, batte quasi sempre entrambi gli estremi.

C'è anche un modo elegante di non scegliere: fare la **media di tutte le
lunghezze**, dando più peso a quelle corte e via via meno a quelle lunghe. Il
peso cala di una frazione fissa a ogni passo in più, come un'eco che si spegne,
e la manopola che decide quanto in fretta si spenga si chiama $\lambda$
(lambda). Con $\lambda = 0$ resta solo il passo singolo, con $\lambda = 1$ resta
il ritorno intero, in mezzo c'è tutto il resto.

Detta così sembra impossibile da fare mentre si gioca, perché quella media
guarda avanti, e il futuro non lo si conosce. Il trucco è guardare
dall'altra parte: invece di chiedersi "che cosa succederà dopo questa casella",
si tiene un elenco delle caselle appena attraversate, ciascuna con un ricordo
che sfuma a ogni passo. Quel ricordo si chiama **traccia**, e quando arriva una
sorpresa la si distribuisce a tutta la scia, tanto più forte quanto più recente
è il passaggio. Il risultato è lo stesso, e si ottiene senza mai aspettare la
fine.

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
capitolo successivo. Là il segnale che guiderà l'apprendimento (il *vantaggio*
di una mossa, cioè quanto è migliore della media delle mosse in quella
situazione) nella sua forma più semplice è proprio la sorpresa di un passo di
cui si è appena detto; e il modo standard di calcolarlo è questa identica media
pesata, con questo identico $\lambda$: si sceglie quanta distorsione accettare
in cambio di quanta varianza risparmiare.

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
sorpresa che corregge, il bersaglio a un passo, l'esplorazione dosata)
restano tutte. È una **rappresentazione** diversa. Al posto della tabella serve
qualcosa che, vista una situazione mai incontrata, sappia indovinare i suoi voti
somigliandola a quelle che ha già visto, e quel qualcosa sono le reti neurali.
È il capitolo che comincia alla pagina dopo questa, ed è la ragione per cui
esiste; con un'avvertenza che conviene portarsi dietro. La dimostrazione che il
Q-learning arriva prima o poi ai voti giusti, quella di Watkins e Dayan, vale
per una tabella e per una tabella soltanto: buttata via la tabella, quella
promessa non c'è più, e il deep reinforcement learning è in buona parte il
mestiere di far funzionare lo stesso qualcosa che nessuno ha dimostrato che
funzioni.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Le **differenze temporali** correggono la stima durante il viaggio, non
  all'arrivo: come il navigatore che dopo mezz'ora rivede il tempo che manca,
  si usa la stima più recente per aggiustare quella vecchia, un pezzetto alla
  volta.
- Il **Q-learning** tiene una tabella di voti (una riga per situazione, una
  colonna per mossa) e la corregge giocando: nuovo voto uguale vecchio voto più
  un po' della sorpresa. Impara quali sarebbero le mosse migliori anche mentre
  si muove a casaccio per esplorare, cioè impara una cosa mentre ne fa
  un'altra.
- Nella correzione ci sono due manopole: una decide quanto dare retta alla
  sorpresa dell'ultimo passo (piccola vuol dire passi cauti), l'altra quanto
  pesa il futuro rispetto al premio immediato. E c'è la ricetta ε-greedy per il
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
  mai vista somigliandola a quelle già viste: sono le reti neurali del capitolo
  successivo.
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
  strategia **ε-greedy** bilancia esplorazione e sfruttamento, e la condizione
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
