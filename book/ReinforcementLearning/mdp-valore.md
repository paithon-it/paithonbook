# Processi decisionali di Markov e funzioni valore

Nessuno consegna a un bambino che impara ad andare in bicicletta un manuale di
fisica. Prova, oscilla, cade; una pedalata storta finisce a terra, una decisa
lo porta avanti. Impara dalle *conseguenze*. Il **reinforcement learning**
(apprendimento per rinforzo) costruisce agenti che imparano nello stesso modo:
agiscono, osservano cosa succede, incassano una ricompensa e correggono il
tiro. Non c'è un insegnante che sussurra la risposta giusta a ogni passo; c'è
solo un ambiente che reagisce.

La sezione precedente aveva tolto di mezzo proprio questo pezzo: nel bandit
ogni tiro è a sé, e la leva scelta adesso non cambia in nulla la situazione del
tiro successivo. Qui invece la pedalata storta *sposta* la bicicletta, e ciò
che si potrà fare fra un istante dipende da cosa si è fatto ora. Rimettere lo
stato al suo posto costa una notazione in più e regala il problema vero.

Per trasformare questa intuizione in matematica serve un'impalcatura precisa.
Quell'impalcatura, formalizzata da Richard Bellman nel 1957 e diventata la
spina dorsale del testo di riferimento di Sutton e Barto
{cite}`sutton2018reinforcement`, si chiama **processo
decisionale di Markov** (*Markov Decision Process*, MDP).

## Il ciclo: stati, azioni, ricompense

A ogni istante l'agente si trova in uno **stato**, sceglie un'**azione**,
l'ambiente lo trasporta in un nuovo stato e gli consegna una **ricompensa**
numerica. Poi il ciclo riparte. Tutto il reinforcement learning abita dentro
questo giro.

`````{tab} Elementare

Pensa a un piccolo robot in un labirinto a caselle. Lo *stato* è la casella in
cui si trova; le *azioni* sono i movimenti possibili (su, giù, destra,
sinistra); la *transizione* è dove finisce dopo la mossa; la *ricompensa* è il
punteggio che riceve: diciamo $+10$ quando raggiunge l'uscita e $-1$ per ogni
passo, così impara a uscire *in fretta*. Il robot non conosce la mappa: la
scopre muovendosi.

`````

`````{tab} Superiore

Un MDP è la quintupla

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma).
$$

$\mathcal{S}$ è l'insieme degli stati, $\mathcal{A}$ quello delle azioni. La
dinamica dell'ambiente è la **funzione di transizione**

$$
P(s' \mid s, a) = \Pr(S_{t+1} = s' \mid S_t = s,\ A_t = a),
$$

cioè la probabilità di finire in $s'$ eseguendo l'azione $a$ nello stato $s$.
La **ricompensa attesa** è

$$
r(s,a) = \mathbb{E}[\,R_{t+1} \mid S_t = s,\ A_t = a\,],
$$

minuscola perché, a differenza della ricompensa aleatoria $R_{t+1}$ che
l'ambiente estrae a ogni passo, è una funzione deterministica di stato e
azione: è la convenzione annunciata nella panoramica del capitolo. Infine
$\gamma \in [0,1)$ è il fattore di sconto (fra poco). Le transizioni sono
*stocastiche*: la stessa azione può condurre in stati diversi.

`````

## La proprietà di Markov

Il nome non è un vezzo: rende onore ad Andrej Markov e a una richiesta molto
precisa. Lo stato deve essere *sufficiente*, deve cioè riassumere tutto ciò che
serve per decidere il futuro: **il futuro dipende solo dal presente, non
dall'intera storia passata**.

`````{tab} Elementare

Fotografa una partita a scacchi a metà. A un bravo giocatore, per decidere la
prossima mossa, basta la foto: non gli serve sapere in che ordine i pezzi sono
arrivati lì. La posizione attuale racconta già tutto ciò che conta. Uno stato
fatto così si dice *markoviano*. E se la foto non bastasse? Si allarga
l'inquadratura finché basta: negli scacchi veri, per esempio, alla foto va
aggiunta una nota ("il re non ha ancora mosso"), perché da essa dipende una
mossa speciale come l'arrocco. L'importante è che tutto il necessario stia
nello stato, e niente resti nascosto nella storia.

`````

`````{tab} Superiore

Formalmente si richiede che

$$
P(S_{t+1} \mid S_t, A_t) = P(S_{t+1} \mid S_0, A_0, \dots, S_t, A_t).
$$

La distribuzione dello stato successivo, condizionata a stato e azione
correnti, non cambia aggiungendo l'intera traiettoria passata. Se
l'osservazione disponibile non soddisfa questa proprietà, si *arricchisce* lo
stato (aggiungendo variabili, o una finestra di osservazioni recenti), finché
la proprietà vale: è esattamente ciò che farà il DQN impilando quattro frame
consecutivi di un videogioco per catturare le velocità.

`````

## La policy: la strategia dell'agente

Sapere in quali stati ci si può trovare non dice ancora *cosa fare*. La regola
di comportamento dell'agente si chiama **policy**.

`````{tab} Elementare

La policy è l'abitudine dell'agente: per ogni stato, quale azione scegliere.
"In questa stanza vado sempre a destra" è una policy. Può anche assomigliare a
un dado truccato ("qui vado a destra 8 volte su 10"), quando conviene
esplorare invece di ripetere sempre la stessa mossa. Imparare, nel RL,
significa migliorare la policy.

`````

`````{tab} Superiore

Una policy $\pi$ è una distribuzione sulle azioni condizionata allo stato:

$$
\pi(a \mid s) = P(A_t = a \mid S_t = s).
$$

È *deterministica* se concentra tutta la probabilità su una sola azione,
$a = \pi(s)$; *stocastica* altrimenti. L'obiettivo dell'apprendimento è trovare
la policy che massimizza la ricompensa accumulata nel tempo.

`````

## Quanto vale il futuro: il ritorno scontato

Una ricompensa da sola dice poco: conta la *somma* delle ricompense lungo tutto
il percorso. Ma un premio subito vale più dello stesso premio fra dieci mosse.

`````{tab} Elementare

Dieci euro oggi valgono più di dieci euro l'anno prossimo. Il **fattore di
sconto** $\gamma$ (gamma), un numero tra 0 e 1, misura questa impazienza. Con
$\gamma = 0{,}9$ la prossima ricompensa conta per intero, ma ogni passo di attesa
in più la moltiplica per $0{,}9$: un $+10$ che arriva un passo più tardi vale
$0{,}9 \times 10 = 9$, due passi più tardi $0{,}9^2 \times 10 = 8{,}1$. Più è
lontana, meno pesa. Con $\gamma$ vicino a 0 l'agente è miope (guarda solo al
premio immediato), vicino a 1 è lungimirante.

`````

`````{tab} Superiore

Il **ritorno** al tempo $t$ è la somma scontata delle ricompense future:

$$
G_t = R_{t+1} + \gamma\, R_{t+2} + \gamma^2 R_{t+3} + \cdots
= \sum_{k=0}^{\infty} \gamma^k\, R_{t+k+1}.
$$

Con $0 \le \gamma < 1$ la serie converge anche su orizzonti infiniti (le
ricompense sono limitate), il che rende il problema ben posto. $\gamma$ non è
un semplice trucco matematico: codifica *quanto lontano* nel futuro all'agente
conviene guardare.

`````

## Un MDP in miniatura

Vale la pena vedere tutti i pezzi in un solo disegno. La {numref}`fig-mdp`
ritrae un mondo minuscolo con tre stati: da $s_0$ l'agente può salire verso
$s_1$ oppure restare fermo, da $s_1$ può scendere verso l'obiettivo $s_2$
(terminale) o tornare indietro. Ogni freccia è un'azione ed è annotata con la
ricompensa che paga: salire non costa nulla, restare o tornare costa $-1$,
raggiungere l'obiettivo frutta $+10$. Con $\gamma$ vicino a 1 la policy
ottimale è intuibile a colpo d'occhio ($s_0 \to s_1 \to s_2$) ed è proprio
quel "colpo d'occhio" che le funzioni valore rendono calcolabile in modo
sistematico.

```{figure} ../figures/mdp-grafo.svg
:name: fig-mdp
:alt: Grafo con tre stati s0, s1 e s2 (obiettivo, terminale) collegati da frecce che rappresentano le azioni, ciascuna annotata con la ricompensa.
:width: 85%

Un MDP in miniatura: gli stati sono cerchi, le azioni frecce, e ogni freccia
riporta la ricompensa. L'obiettivo $s_2$ è terminale.
```

## Le funzioni valore: $V$ e $Q$

Il ritorno di un singolo episodio è rumoroso: dipende dal caso e dalle scelte.
Ciò che ci serve è il ritorno *atteso*: quanto promette, in media, trovarsi in
una certa situazione seguendo una data policy. È il compito delle **funzioni
valore**.

`````{tab} Elementare

Due domande, due funzioni. **Quanto è buono trovarsi qui?** è il *valore di
stato* $V$: la ricompensa totale che mi aspetto di raccogliere partendo da
questo stato. **Quanto è buono fare questa mossa qui?** è il *valore di
stato-azione* $Q$: come sopra, ma fissando anche l'azione. $Q$ è spesso più
utile in pratica, perché confrontando le azioni in uno stato mi dice
direttamente quale conviene.

`````

`````{tab} Superiore

Data una policy $\pi$, la **funzione valore di stato** e la **funzione valore
di stato-azione** sono

$$
V^\pi(s) = \mathbb{E}_\pi[\,G_t \mid S_t = s\,],
\qquad
Q^\pi(s,a) = \mathbb{E}_\pi[\,G_t \mid S_t = s,\ A_t = a\,].
$$

$V^\pi(s)$ è il ritorno atteso partendo da $s$ e seguendo poi $\pi$;
$Q^\pi(s,a)$ fissa la prima azione ad $a$ e da lì prosegue con $\pi$. Le due
sono legate da $V^\pi(s) = \sum_a \pi(a\mid s)\, Q^\pi(s,a)$.

`````

## L'equazione di Bellman: il valore è ricorsivo

Qui arriva l'idea che tiene in piedi tutto. Il valore di uno stato non va
calcolato da zero sommando infinite ricompense: si spezza in due pezzi,
*l'adesso* e *il dopo*.

`````{tab} Elementare

Il valore di dove sei = la ricompensa che incassi al prossimo passo **più** il
valore (scontato) di dove finisci. È una scala a pioli: ogni gradino è definito
in funzione del successivo. Nel labirinto, il valore della casella accanto
all'uscita è alto perché *l'uscita* vale molto; quel valore poi
"retropropaga" alle caselle precedenti, gradino dopo gradino.

`````

`````{tab} Superiore

Spezzando il ritorno come $G_t = R_{t+1} + \gamma\, G_{t+1}$ e prendendo
l'attesa si ottiene l'**equazione di Bellman** per $V^\pi$:

$$
V^\pi(s) = \mathbb{E}_\pi\!\left[\, R_{t+1} + \gamma\, V^\pi(S_{t+1})
\;\middle|\; S_t = s \,\right],
$$

che, esplicitando policy e transizioni, diventa

$$
V^\pi(s) = \sum_{a} \pi(a\mid s) \sum_{s'} P(s'\mid s,a)
\big[\,r(s,a) + \gamma\, V^\pi(s')\,\big].
$$

È un sistema di equazioni lineari: una relazione di consistenza fra il valore
di uno stato e quello dei suoi successori. Da qui partono tutti gli algoritmi
che incontreremo: a cominciare dalla *value iteration* e dalla *policy
iteration* qui sotto, fino al *Q-learning* della prossima sezione.

`````

## Value iteration: l'equazione diventa algoritmo

L'equazione di Bellman, da sola, è una fotografia: descrive come devono stare
i valori quando sono *giusti*, ma non spiega come trovarli. Il primo modo per
trovarli è di una semplicità disarmante, ed è l'idea con cui Bellman inaugurò
la **programmazione dinamica** {cite}`bellman1957dynamic`: usare l'equazione
non come descrizione ma come *regola di aggiornamento*, da ripetere finché i
valori non si assestano. Si chiama **value iteration**.

`````{tab} Elementare

La ricetta, nel labirinto: scrivi $0$ su ogni casella. Poi, casella per
casella, guarda tutte le mosse possibili e chiediti: "quanto rende ciascuna,
contando la ricompensa immediata più il valore (scontato) della casella dove
finirei?". Scrivi sulla casella il risultato della mossa migliore. Finito il
giro, ricomincia da capo con i numeri nuovi, e poi ancora, finché i numeri
smettono di muoversi. A quel punto ogni casella dice quanto vale *davvero*, e
la strategia migliore è in omaggio: da ogni casella, scegli la mossa che rende
di più.

`````

`````{tab} Superiore

Partendo da una stima arbitraria $V_0$ (tipicamente nulla), si itera

$$
V_{k+1}(s) = \max_{a} \sum_{s'} P(s'\mid s,a)
\big[\,r(s,a) + \gamma\, V_k(s')\,\big],
$$

dove $V_k$ è la stima dei valori al passo $k$: è l'equazione di Bellman con un
$\max$ sulle azioni al posto della media pesata dalla policy. Il punto fisso è
l'**equazione di ottimalità di Bellman**,
$V^*(s) = \max_a \sum_{s'} P(s'\mid s,a)\big[r(s,a) + \gamma\, V^*(s')\big]$,
dove $V^*$ è il valore della migliore policy possibile. La convergenza è
garantita: l'operatore di aggiornamento è una **contrazione** di fattore
$\gamma$ nella norma del massimo (a ogni passo la distanza da $V^*$ si riduce
almeno di un fattore $\gamma$) quindi il punto fisso è unico e l'iterazione vi
arriva da qualunque inizializzazione {cite}`bellman1957dynamic`
{cite}`sutton2018reinforcement`. Estratto $V^*$, la policy ottima è quella
*greedy*: in ogni stato, l'azione che realizza il massimo.

`````

## La value iteration all'opera

Facciamo davvero i conti, sull'MDP in miniatura della {numref}`fig-mdp` e con
$\gamma = 0{,}9$. Le transizioni lì sono deterministiche, quindi la somma
sugli stati d'arrivo ha un solo termine e l'aggiornamento si legge:
"ricompensa della mossa più $0{,}9$ volte il valore dello stato d'arrivo,
tenendo la mossa migliore". Lo stato terminale $s_2$ vale sempre $0$ (lì il
gioco è finito) e partiamo da $V_0(s_0) = V_0(s_1) = 0$.

**Prima iterazione.** In $s_1$: scendere rende $10 + 0{,}9 \times 0 = 10$,
tornare rende $-1 + 0{,}9 \times 0 = -1$; vince scendere, quindi
$V_1(s_1) = 10$. In $s_0$: salire rende $0 + 0{,}9 \times 0 = 0$, restare
$-1 + 0{,}9 \times 0 = -1$; quindi $V_1(s_0) = 0$. Il $+10$ dell'obiettivo è
"entrato" in $s_1$, ma non ha ancora raggiunto $s_0$.

**Seconda iterazione.** In $s_1$ non cambia nulla: $V_2(s_1) = 10$. In $s_0$,
però, salire ora rende $0 + 0{,}9 \times 10 = 9$ contro il $-1$ di restare:
quindi $V_2(s_0) = 9$. Il valore dell'obiettivo è retrocesso di un altro
passo verso l'inizio.

**Terza iterazione.** Rifacendo i conti non si muove più niente: salire da
$s_0$ rende ancora $9$, restare renderebbe $-1 + 0{,}9 \times 9 = 7{,}1$;
scendere da $s_1$ rende ancora $10$, tornare $-1 + 0{,}9 \times 9 = 7{,}1$.
Quindi $V_3(s_0) = 9$ e $V_3(s_1) = 10$: l'algoritmo si è fermato.

|        | $V(s_0)$ | $V(s_1)$ | $V(s_2)$ |
|:-------|:--------:|:--------:|:--------:|
| $k=0$  | $0$      | $0$      | $0$      |
| $k=1$  | $0$      | $10$     | $0$      |
| $k=2$  | $9$      | $10$     | $0$      |
| $k=3$  | $9$      | $10$     | $0$      |

I numeri hanno smesso di muoversi: quello è $V^*$. E la policy ottima si legge
dalle mosse vincenti (da $s_0$ salire, da $s_1$ scendere) esattamente il
"colpo d'occhio" di prima, solo che adesso è un calcolo che un computer può
ripetere identico su un milione di stati.

Su tre stati l'onda si esaurisce in due passi. Su una griglia si vede meglio.

```{figure} ../figures/iterazione-valore.gif
:name: fig-iterazione-valore
:alt: Animazione di un mondo a griglia 4x4 con due muri e una casella obiettivo contrassegnata da una stella in alto a destra. A ogni iterazione k i valori delle caselle si aggiornano e la colorazione, che parte dall'obiettivo, si propaga verso le caselle sempre più lontane fino a riempire la griglia.
:width: 90%

La stessa formula su un mondo a griglia $4\times4$ con $\gamma = 0{,}9$: il
valore parte dalla casella con la ricompensa e **risale di una casella per
iterazione**, aggirando i muri, finché i numeri smettono di muoversi.
```

Guardando la {numref}`fig-iterazione-valore` si capisce anche quante
iterazioni servono: tante quante il numero di passi che separano lo stato più
lontano dall'obiettivo. Il valore non "si diffonde" ovunque insieme: cammina.

## Policy iteration: valutare e migliorare, a turni

La value iteration fonde due gesti in un unico aggiornamento: stimare quanto
rendono gli stati e scegliere le azioni migliori. La **policy iteration** li
separa e li alterna: prima *valuta* fino in fondo la policy corrente, poi la
*migliora*, e ricomincia.

`````{tab} Elementare

È il metodo dell'allenatore. Primo tempo, la pagella: con la squadra che gioca
così com'è, misuro quanto rende ogni situazione (*valutazione*). Secondo
tempo, la correzione: scorro la pagella e, situazione per situazione, cambio
la mossa dove una diversa renderebbe di più (*miglioramento*). Poi rifaccio la
pagella per la nuova strategia, la correggo ancora, e avanti così. Quando un
giro di correzioni non cambia più nulla, la strategia è la migliore possibile.
Rispetto alla value iteration i giri sono pochi (spesso una manciata) ma ogni
pagella completa è un lavoro lungo.

`````

`````{tab} Superiore

Si alternano due passi. **Valutazione**: data la policy $\pi$, si calcola
$V^\pi$ risolvendo il sistema lineare dell'equazione di Bellman (o iterandola,
stavolta senza $\max$, fino a convergenza). **Miglioramento**: si rende la
policy *greedy* rispetto ai valori appena calcolati,

$$
\pi'(s) = \arg\max_{a} \sum_{s'} P(s'\mid s,a)
\big[\,r(s,a) + \gamma\, V^\pi(s')\,\big].
$$

Il *policy improvement theorem* garantisce $V^{\pi'}(s) \ge V^\pi(s)$ in ogni
stato, con miglioramento stretto da qualche parte finché $\pi$ non è ottima; e
poiché in un MDP finito le policy deterministiche sono in numero finito,
l'alternanza termina sulla policy ottima in un numero finito di iterazioni
{cite}`sutton2018reinforcement`. Il confronto con la value iteration è un
compromesso classico: la policy iteration converge in *meno* iterazioni, ma
ciascuna contiene una valutazione completa (costosa: un sistema di
$|\mathcal{S}|$ equazioni, o molte passate); la value iteration fa iterazioni
molto più economiche (una sola passata con il $\max$) ma ne richiede di più.
Si può anzi leggere la value iteration come una policy iteration impaziente,
che tronca la valutazione dopo un solo passo.

`````

Due strade per la stessa vetta, insomma: pochi passi pesanti o molti passi
leggeri. Nei problemi reali si sceglie in base alle dimensioni del problema, o
si mescolano le due, troncando la valutazione dopo poche passate.

## Quando manca la mappa

C'è però un dettaglio che finora abbiamo dato per scontato, ed è enorme. Per
fare quei conti ("ricompensa della mossa più valore dello stato d'arrivo"),
bisogna *sapere in anticipo* dove porta ogni mossa e quanto paga: value
iteration e policy iteration richiedono di conoscere il modello dell'ambiente,
cioè le funzioni $P$ e $r$. È pianificare un viaggio con la mappa già in mano.
Ma il robot del nostro labirinto la mappa non ce l'ha, e il mondo reale quasi
mai la consegna: nessuno fornisce a un agente le probabilità di transizione
del traffico o di una partita a Go.

Quando il modello manca resta una sola strada: stimare i valori
*dall'esperienza*. E ci sono due modi di percorrerla, che le prossime due
sezioni prendono in ordine. Il primo è il più diretto che si possa immaginare:
giocare partite intere e fare la media di com'è andata, senza chiedere niente
a nessuno. Sono i **metodi Monte Carlo**. Il secondo non aspetta la fine della
partita e corregge la stima a ogni passo, usando la stima successiva come
bersaglio provvisorio: è l'apprendimento per **differenze temporali**, e il suo
esemplare più famoso è il **Q-learning**.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Tutto il reinforcement learning sta dentro un giro solo: l'agente si trova in
  una situazione (il robot in una casella del labirinto), sceglie una mossa,
  finisce da qualche parte e incassa un punteggio. Poi si ricomincia.
- La situazione deve **bastare da sola**: come la foto di una partita a
  scacchi, deve dire tutto ciò che serve per decidere, senza che occorra sapere
  come ci si è arrivati. Se non basta, si allarga l'inquadratura.
- La **strategia** è l'abitudine dell'agente (in questa casella vado a destra),
  eventualmente truccata come un dado quando conviene provare altro. E il
  futuro pesa meno del presente: dieci euro oggi valgono più di dieci euro
  l'anno prossimo, e il fattore di sconto misura questa impazienza.
- Il **valore** di una casella è il punteggio che ci si aspetta di raccogliere
  da lì in avanti; il valore di una mossa fa lo stesso fissando anche la prima
  mossa. Ogni valore si appoggia al successivo come i pioli di una scala: è
  così che il premio dell'uscita risale il labirinto, una casella per volta.
- Se la mappa è nota (dove porta ogni mossa e quanto paga), ci sono due
  ricette: aggiornare i numeri di tutte le caselle finché smettono di muoversi,
  oppure alternare pagella e correzione come un allenatore. Quando la mappa
  manca bisogna imparare giocando: partite intere (Monte Carlo) o correzioni a
  ogni passo (differenze temporali).
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un **MDP** $(\mathcal{S},\mathcal{A},P,r,\gamma)$ formalizza un agente che
  sceglie azioni, transita fra stati e raccoglie ricompense.
- La **proprietà di Markov**: il futuro dipende solo dallo stato presente, non
  dall'intera storia.
- La **policy** $\pi(a\mid s)$ è la strategia; il **ritorno scontato** $G_t$
  pesa il futuro con $\gamma$.
- $V^\pi$ e $Q^\pi$ misurano il ritorno *atteso*; l'**equazione di Bellman** li
  definisce in modo ricorsivo, ed è la base di ogni algoritmo di RL.
- Con il modello ($P$ e $r$) noto, **value iteration** e **policy iteration**
  calcolano valori e policy ottimi iterando Bellman; quando il modello manca
  bisogna imparare dall'esperienza, coi metodi Monte Carlo o con le differenze
  temporali.
```

`````
