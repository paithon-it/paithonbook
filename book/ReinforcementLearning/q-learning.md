# Q-learning e l'apprendimento per differenze temporali

Immagina di imparare un videogioco senza manuale. Premi tasti, il personaggio si muove, ogni tanto il punteggio sale. Nessuno ti dice *quale* mossa, tra le decine che hai fatto, ha meritato quei punti. È il problema centrale del *reinforcement learning* (apprendimento per rinforzo): un agente agisce in un ambiente, riceve **ricompense** sporadiche e deve capire, per tentativi, quali comportamenti convengano nel lungo periodo. Non c'è un insegnante che fornisce la risposta giusta a ogni passo, solo un segnale numerico che arriva in ritardo e va distribuito all'indietro sulle scelte che l'hanno prodotto.

## Imparare senza aspettare la fine: le differenze temporali

Una prima idea, ingenua, sarebbe: gioca una partita intera, guarda quanti punti hai totalizzato, e usa quel totale per giudicare tutte le mosse fatte. Funziona, ma è lento e spreca informazione. L'apprendimento per **differenze temporali** (*temporal-difference*, TD), introdotto da Richard Sutton nel 1988, propone invece di aggiornare le stime *durante* l'episodio, a ogni singolo passo.

`````{tab} Elementare

Pensa a stimare quanto manca all'arrivo di un viaggio in auto. Parti dicendo "due ore". Dopo mezz'ora sei più avanti del previsto e il navigatore aggiorna: "un'ora e dieci". Non hai aspettato di arrivare per correggere la tua previsione: hai usato una **stima più recente** per aggiustare quella vecchia. Il TD learning fa esattamente questo. La differenza tra la stima nuova (più informata) e quella vecchia è l'"errore" che usiamo per correggere, un pezzetto alla volta.

`````

`````{tab} Superiore

Sia $V(s)$ la stima del valore atteso di uno stato $s$, cioè la ricompensa totale futura che ci aspettiamo partendo da lì. Dopo aver osservato la transizione $s \to s'$ con ricompensa $r$, l'aggiornamento TD(0) è

$$
V(s) \leftarrow V(s) + \alpha\,\underbrace{\big[\,r + \gamma\,V(s') - V(s)\,\big]}_{\text{errore TD}\;\delta} .
$$

Qui $\alpha \in (0,1]$ è il **learning rate**, $\gamma \in [0,1)$ il **fattore di sconto** e $\delta$ l'**errore TD**. Il termine $r + \gamma V(s')$ è una stima aggiornata di $V(s)$ costruita *usando la stima successiva* $V(s')$: questa dipendenza da una stima per aggiornarne un'altra si chiama *bootstrapping*, ed è ciò che distingue il TD dai metodi Monte Carlo, che attendono la fine dell'episodio.

`````

## Q-learning: stimare il valore delle azioni

Conoscere il valore di uno stato non basta per decidere: serve sapere quanto vale ogni **azione** in quello stato. È il salto del **Q-learning**, formulato da Chris Watkins nel 1989, forse l'algoritmo più celebre del campo.

`````{tab} Elementare

Immagina una grande tabella: una riga per ogni situazione in cui puoi
trovarti, una colonna per ogni mossa possibile. In ciascuna casella scrivi un
voto (la $Q$) che dice "quanto conviene fare questa mossa in questa
situazione". All'inizio i voti sono tutti a zero: l'agente non sa nulla.
Giocando e ricevendo ricompense, corregge i voti. Alla fine, per agire bene,
gli basta guardare la riga della situazione corrente e scegliere la mossa col
voto più alto.

La parte sorprendente: l'agente può muoversi anche a casaccio, sbagliando di proposito per esplorare, e **imparare comunque quali sarebbero le mosse migliori**. Impara una cosa mentre ne fa un'altra. Per questo si dice *off-policy*.

`````

`````{tab} Superiore

Definiamo la funzione azione-valore ottima $Q^*(s,a)$ come la ricompensa scontata attesa se in $s$ eseguiamo $a$ e poi seguiamo la politica ottima. Soddisfa l'equazione di ottimalità di Bellman

$$
Q^*(s,a) = \mathbb{E}\big[\,r + \gamma \max_{a'} Q^*(s',a')\,\big].
$$

Il Q-learning è **off-policy** perché il suo *target* usa $\max_{a'} Q(s',a')$
(il valore dell'azione *migliore* nello stato successivo) indipendentemente da
quale azione l'agente abbia poi effettivamente scelto. Impara così la politica
ottima anche mentre ne segue una esplorativa. Watkins e Dayan
{cite}`watkins1992q` dimostrarono che, sotto ipotesi ragionevoli (ogni coppia
$(s,a)$ visitata infinite volte, $\alpha$ decrescente in modo opportuno), $Q$
converge a $Q^*$.

`````

## La formula di aggiornamento

Il cuore dell'algoritmo è una sola riga, che unisce l'idea TD alla tabella delle azioni. In parole suona così: *nuovo voto = vecchio voto + un po' della sorpresa*, dove la sorpresa è la differenza tra com'è andata davvero (premio incassato più prospettive dalla nuova casella) e come pensavi andasse. In simboli:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\,\Big[\,r + \gamma \max_{a'} Q(s',a') - Q(s,a)\,\Big].
$$

Leggiamola da destra: $r + \gamma \max_{a'} Q(s',a')$ è il **target TD**, la stima aggiornata del valore di $(s,a)$; sottraendo la stima corrente $Q(s,a)$ otteniamo l'errore; il learning rate $\alpha$ decide quanto fidarci della correzione (piccolo = passi cauti); il fattore di sconto $\gamma$ pesa il futuro (vicino a $1$ = lungimirante, vicino a $0$ = miope). Nient'altro: nessun gradiente, nessuna rete neurale. Solo una tabella che si aggiusta.

## Esplorare o sfruttare: la strategia ε-greedy

Se l'agente scegliesse sempre la mossa col voto più alto, resterebbe intrappolato nella prima strategia decente che trova, senza mai scoprire scorciatoie migliori. Deve ogni tanto **esplorare**.

`````{tab} Elementare

È il dilemma del ristorante: torni sempre da quello che conosci e ti piace
(**sfruttare**), o provi il nuovo che ha appena aperto e potrebbe essere una
scoperta (**esplorare**)? La ricetta ε-greedy è semplice: nella grande
maggioranza dei casi vai sul sicuro, ma con una piccola probabilità (chiamata
$\varepsilon$, ad esempio il 10%) tiri un dado e provi una mossa a caso.
All'inizio esplori molto; man mano che impari, riduci $\varepsilon$ e ti
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

Un $\varepsilon$ costante garantisce esplorazione perpetua; in pratica si usa un *decay*, ad esempio $\varepsilon_t = \varepsilon_0\,\lambda^t$ con $\lambda<1$, per convergere gradualmente allo sfruttamento puro. La scelta di $\varepsilon$ regola il compromesso *exploration–exploitation*, uno dei nodi teorici centrali del reinforcement learning.

`````

## SARSA: la variante on-policy

C'è un cugino stretto del Q-learning che cambia un solo simbolo nella formula, con conseguenze interessanti. Si chiama **SARSA** {cite}`rummery1994online`, dal quintetto di ingredienti del suo aggiornamento: $(s, a, r, s', a')$.

`````{tab} Elementare

Il Q-learning è un ottimista spericolato: valuta ogni mossa immaginando di comportarsi *perfettamente* subito dopo. SARSA è più prudente: valuta le mosse tenendo conto che, di tanto in tanto, esplorerà davvero e potrebbe sbagliare. Impara il valore della politica che **effettivamente segue**, esplorazione compresa. Il risultato tipico: sul bordo di un burrone, SARSA impara a tenersi a distanza di sicurezza, mentre il Q-learning cammina sull'orlo perché "in teoria" non cadrebbe mai.

`````

`````{tab} Superiore

SARSA è **on-policy**: nel target non compare il massimo, ma il valore dell'azione $a'$ realmente scelta nello stato $s'$ dalla stessa politica (ad esempio ε-greedy):

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\,\big[\,r + \gamma\,Q(s',a') - Q(s,a)\,\big].
$$

Valuta dunque la politica di comportamento anziché quella greedy. Nel classico esempio del *cliff walking* (Sutton e Barto), SARSA converge a un cammino più sicuro e lontano dal precipizio, il Q-learning al cammino ottimo ma rischioso lungo il bordo: differenza che sparisce solo quando $\varepsilon \to 0$.

`````

## Un labirinto concreto

Rendiamo tutto tangibile con una griglia. L'agente parte in basso a sinistra (**S**), deve raggiungere la meta (**+1**) evitando la trappola (**−1**) e il muro. A ogni passo sceglie tra su, giù, sinistra, destra.

```{figure} ../figures/labirinto-qlearning.svg
:name: fig-labirinto
:alt: Griglia 3x4 con cella di partenza in basso a sinistra, meta con ricompensa +1 in alto a destra, trappola -1 sotto la meta, un muro al centro e frecce che indicano la politica appresa in ogni cella.
:width: 85%

La politica appresa dal Q-learning sulla griglia: da ogni cella la freccia indica l'azione con $Q$ più alta. Il valore della meta "retrocede" fin dalla partenza, un passo per episodio.
```

All'inizio la tabella $Q$ è tutta a zero. Poniamo $\alpha=0{,}5$ e $\gamma=0{,}9$. La prima volta che l'agente calpesta la meta ($r=+1$, stato successivo terminale con valore $0$), la casella dell'ultima mossa diventa

$$
Q(s,\rightarrow) \leftarrow 0 + 0{,}5\,\big[\,1 + 0{,}9\cdot 0 - 0\,\big] = 0{,}5 .
$$

Un episodio dopo, la cella precedente $s^-$, da cui si arriva a $s$, "vede" $\max_{a'} Q(s,a')=0{,}5$ e si aggiorna:

$$
Q(s^-,\rightarrow) \leftarrow 0 + 0{,}5\,\big[\,0 + 0{,}9\cdot 0{,}5 - 0\,\big] = 0{,}225 .
$$

Ecco il meccanismo TD in azione: la ricompensa non salta dappertutto in una volta, ma **retrocede** verso la partenza un passo per episodio, come una macchia che si allarga all'indietro dalla meta. In codice, il nucleo dell'algoritmo è compatto:

```python
import numpy as np

# Griglia 3x4: 12 stati, 4 azioni (0=su 1=giù 2=sinistra 3=destra)
n_stati, n_azioni = 12, 4
Q = np.zeros((n_stati, n_azioni))       # tabella dei voti, tutta a zero

alpha, gamma, epsilon = 0.5, 0.9, 0.1

def epsilon_greedy(s):
    if np.random.rand() < epsilon:
        return np.random.randint(n_azioni)   # esplora: mossa a caso
    return int(np.argmax(Q[s]))              # sfrutta: mossa col voto piu alto

def aggiorna(s, a, r, s_next):
    # target TD: usa la stima migliore dello stato successivo (off-policy)
    td_target = r + gamma * np.max(Q[s_next])
    Q[s, a] += alpha * (td_target - Q[s, a])   # correggi verso il target
```

La funzione `aggiorna` è il Q-learning per intero; per ottenere SARSA
basterebbe passare l'azione realmente scelta `a_next` e sostituire
`np.max(Q[s_next])` con `Q[s_next, a_next]`. L'ambiente vero e proprio, che
dato $(s,a)$ restituisce $r$ e $s'$, è omesso qui, ma è ciò che, ripetuto per
migliaia di episodi, riempie la tabella e fa emergere le frecce della
{numref}`fig-labirinto`.

```{admonition} Da ricordare
:class: important
- Il **temporal-difference** aggiorna le stime a ogni passo usando la stima
  successiva (*bootstrapping*), senza attendere la fine dell'episodio.
- Il **Q-learning** impara una tabella $Q(s,a)$ ed è *off-policy*: il suo
  target usa $\max_{a'} Q(s',a')$, quindi apprende la politica ottima anche
  mentre esplora.
- Nella formula, $\alpha$ dosa la correzione e $\gamma$ pesa il futuro; la
  strategia **ε-greedy** bilancia esplorazione e sfruttamento.
- **SARSA** è la variante *on-policy* ($\gamma\,Q(s',a')$ al posto del massimo):
  più prudente, valuta la politica che segue davvero.
```
