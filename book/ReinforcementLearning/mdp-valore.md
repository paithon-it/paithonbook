# Processi decisionali di Markov e funzioni valore

La sezione precedente ha lavorato su un mondo che non si muove: nel bandit ogni
tiro è a sé, e la leva scelta adesso non cambia in nulla la situazione del tiro
successivo. Il mondo vero non è così. Un bambino che impara ad andare in
bicicletta prova, oscilla, cade; e la pedalata storta non gli costa soltanto un
brutto voto, gli **sposta la bicicletta**: quello che potrà fare fra un istante
dipende da quello che ha fatto ora, e si ritrova in una situazione che si è
creato da sé. Rimettere al suo posto la situazione che cambia costa qualche
simbolo in più, e in cambio consegna il problema vero.

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

Quei due numeri, come si diceva nella panoramica, li scegliamo noi: sono il modo
di dire al robot che cosa vogliamo. Tienilo a mente, perché in questo capitolo
di labirinti ne incontrerai tre, con regole diverse: questo, la griglia
dell'animazione più avanti (dove l'uscita paga $+1$ e i passi non costano
nulla) e la griglia del Q-learning nella sezione finale (uscita $+1$, una
trappola $-1$, di nuovo passi gratis). Ogni volta lo diremo: sono tre mondi
diversi, non lo stesso mondo che cambia idea.

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
l'ambiente estrae a ogni passo, è una funzione deterministica di stato e azione.
Attenzione al doppio uso, che è una trappola vera: $r(s,a)$ **con i suoi
argomenti** è sempre la ricompensa attesa, mentre la $r$ nuda che comparirà
nelle regole di aggiornamento di Monte Carlo e del Q-learning è la ricompensa
*osservata* in una singola transizione, cioè una realizzazione di $R_{t+1}$.
Confondere le due vuol dire credere che l'agente conosca una media che invece
deve stimare.

Infine $\gamma \in [0,1]$ è il fattore di sconto (fra poco). Le transizioni
**possono essere** stocastiche, cioè la stessa azione può condurre in stati
diversi; il caso deterministico, come l'MDP in miniatura di qualche riga più
avanti, è il caso particolare in cui $P(s'\mid s,a)$ vale $1$ su un solo stato.

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

Vale la pena dare un nome a quel caso, perché è la regola e non l'eccezione.
Quando l'agente non osserva lo stato ma solo una sua **funzione parziale e
rumorosa**, il modello si chiama **POMDP** (*Partially Observable MDP*): oltre
a stati, azioni e ricompense c'è un insieme di **osservazioni** e una
distribuzione $P(o \mid s)$ che dice cosa si riesce a vedere. Un robot con
sensori limitati, un sistema di raccomandazione che non conosce l'umore
dell'utente, un giocatore di poker che non vede le carte altrui: tutti POMDP.

Il fatto scomodo è che in un POMDP **la policy ottima non può dipendere solo
dall'osservazione corrente**. La soluzione teorica è ragionare su una
distribuzione di probabilità sugli stati possibili (il *belief state*), che
però vive in uno spazio continuo anche quando gli stati sono pochi, e rende il
problema molto più duro. In pratica si fa una delle due cose, ed entrambe
compaiono in questo libro: si **impila una finestra** di osservazioni recenti,
come il DQN con i quattro fotogrammi, oppure si dà all'agente una **memoria**,
cioè una rete ricorrente il cui stato nascosto fa da riassunto approssimato di
tutto ciò che si è visto finora. Quando nei capitoli successivi si vedrà una
policy con dentro una LSTM, la ragione è questa.

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
il percorso. Quella somma è il **ritorno** annunciato nella panoramica: non
quanto si incassa adesso, ma quanto si incasserà in tutto da qui alla fine. Ma
un premio subito vale più dello stesso premio fra dieci mosse, e quindi nella
somma i premi lontani entrano ridotti: da qui il nome **ritorno scontato**, che
è il numero che l'agente cerca di rendere più grande possibile.

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

Con $0 \le \gamma < 1$, e se le ricompense sono limitate, la serie converge
anche su orizzonti infiniti, il che rende il problema ben posto. È il motivo per
cui nei compiti **continui**, quelli che non finiscono mai, lo sconto è
obbligatorio. Nei compiti **episodici** la somma ha invece un numero finito di
termini, perché l'episodio termina, e $\gamma = 1$ è ammesso: è il caso di metà
degli esempi classici, compreso il *cliff walking* che incontreremo nella
sezione sul Q-learning. Nell'uno e nell'altro caso $\gamma$ non è un semplice
trucco matematico: codifica *quanto lontano* nel futuro all'agente conviene
guardare.

`````

## Un MDP in miniatura

Vale la pena vedere tutti i pezzi in un solo disegno. La {numref}`fig-mdp`
ritrae un mondo minuscolo con tre stati, che chiamiamo $s_0$, $s_1$ e $s_2$
(sono soltanto nomi di caselle, e il disegno le mostra). Da $s_0$ l'agente può
salire verso $s_1$ oppure restare fermo; da $s_1$ può scendere verso
l'obiettivo $s_2$ o tornare indietro. L'obiettivo si dice **terminale**, che
vuol dire semplicemente che lì la partita finisce: arrivati, non si fa più
niente e non si incassa più niente. Ogni freccia è un'azione ed è annotata con
la ricompensa che paga: salire non costa nulla, restare o tornare costa $-1$,
raggiungere l'obiettivo frutta $+10$. Con $\gamma$ vicino a 1 la strategia
migliore è intuibile a colpo d'occhio ($s_0 \to s_1 \to s_2$) ed è proprio quel
"colpo d'occhio" che le funzioni valore rendono calcolabile in modo
sistematico.

```{figure} ../figures/mdp-grafo.svg
:name: fig-mdp
:alt: Grafo con tre stati s0, s1 e s2 (obiettivo, terminale) collegati da frecce che rappresentano le azioni, ciascuna annotata con la ricompensa.
:width: 85%

Un MDP in miniatura: gli stati sono cerchi, le azioni frecce, e ogni freccia
riporta la ricompensa che si incassa facendola. L'obiettivo $s_2$ è terminale:
arrivati lì la partita finisce.
```

## Le funzioni valore: $V$ e $Q$

Una partita sola dice poco. Il ritorno raccolto in un singolo **episodio**,
cioè in una partita giocata dall'inizio alla fine, dipende dalla fortuna e dalle
scelte fatte quella volta: rigiocandola viene un numero diverso. Quello che
serve è il ritorno **medio**, quanto cioè promette, in media, trovarsi in una
certa situazione e comportarsi in un certo modo. Quella media si chiama, in
statistica, valore *atteso*, ed è il mestiere delle **funzioni valore**.

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
all'uscita è alto perché *l'uscita* vale molto; e poi quel valore fa un passo
all'indietro, dalla casella accanto all'uscita a quella prima ancora, e poi a
quella prima ancora, gradino dopo gradino, fino alla partenza. Il premio non si
sposta: si sposta la notizia che esiste.

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

La regola della scala, da sola, è una fotografia: dice come devono stare i
valori quando sono *giusti* (ogni gradino appoggiato al successivo), ma non
spiega come trovarli, e all'inizio non li conosciamo. Il primo modo per
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
uno sconto di $0{,}9$. In quel mondo ogni mossa porta sempre nella stessa
casella (si dice che le transizioni sono **deterministiche**: niente sorprese,
niente da mediare), quindi la ricetta si legge senza complicazioni: "quanto
paga la mossa, più $0{,}9$ volte il valore della casella dove si finisce", e si
tiene la mossa che rende di più. L'obiettivo $s_2$ vale sempre $0$, perché lì la
partita è finita e non c'è più niente da raccogliere; e si comincia scrivendo
$0$ anche sulle altre due caselle, tanto per avere un punto di partenza.

`````{tab} Elementare

**Primo giro.** Cominciamo da $s_1$, la casella accanto all'obiettivo. Scendere
paga $10$ subito e porta nell'obiettivo, che vale $0$: in tutto
$10 + 0{,}9 \times 0 = 10$. Tornare indietro costa $1$ e porta in $s_0$, che per
adesso vale $0$: in tutto $-1 + 0{,}9 \times 0 = -1$. Vince scendere, e su $s_1$
scriviamo $10$. Passiamo a $s_0$: salire non costa nulla e porta in $s_1$, che
in questo momento vale ancora $0$, quindi rende $0$; restare fermi costa $1$,
quindi rende $-1$. Vince salire, e su $s_0$ scriviamo $0$. Il premio è entrato
in $s_1$, ma in $s_0$ non è ancora arrivato.

**Secondo giro.** Su $s_1$ non cambia niente, resta $10$. Su $s_0$ invece salire
adesso porta in una casella che vale $10$, quindi rende
$0 + 0{,}9 \times 10 = 9$, contro il $-1$ di restare fermi: scriviamo $9$. Il
premio ha fatto un altro passo all'indietro.

**Terzo giro.** Rifacendo gli stessi conti non si muove più niente: da $s_0$
salire rende ancora $9$ (restare renderebbe $-1 + 0{,}9 \times 9 = 7{,}1$, che è
meno), da $s_1$ scendere rende ancora $10$ (tornare renderebbe $7{,}1$). I
numeri si sono fermati, e allora abbiamo finito.

| dopo il giro | $s_0$ vale | $s_1$ vale | $s_2$ vale |
|:-------------|:----------:|:----------:|:----------:|
| all'inizio   | $0$        | $0$        | $0$        |
| primo        | $0$        | $10$       | $0$        |
| secondo      | $9$        | $10$       | $0$        |
| terzo        | $9$        | $10$       | $0$        |

`````

`````{tab} Superiore

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
Quindi $V_3(s_0) = 9$ e $V_3(s_1) = 10$: l'algoritmo si è fermato, e quel punto
fisso è $V^*$, il valore della policy ottima.

|        | $V(s_0)$ | $V(s_1)$ | $V(s_2)$ |
|:-------|:--------:|:--------:|:--------:|
| $k=0$  | $0$      | $0$      | $0$      |
| $k=1$  | $0$      | $10$     | $0$      |
| $k=2$  | $9$      | $10$     | $0$      |
| $k=3$  | $9$      | $10$     | $0$      |

`````

I numeri hanno smesso di muoversi, e quelli sono i valori veri: dicono, da ogni
casella, quanto ci si può aspettare di raccogliere da lì in avanti giocando al
meglio. La strategia migliore arriva in omaggio, leggendo le mosse che hanno
vinto (da $s_0$ salire, da $s_1$ scendere): è lo stesso "colpo d'occhio" di
prima, solo che adesso è un calcolo che un computer ripete identico su un
milione di caselle.

Un milione, però, è un tetto, non un vanto, e conviene fissarlo qui perché è la
ragione per cui esiste il capitolo successivo. La tabella si può scrivere
finché le situazioni si possono elencare. Gli scacchi hanno circa
$4{,}8 \times 10^{44}$ posizioni legali, il Go su goban $19\times19$ circa
$2{,}08 \times 10^{170}$: due conti fatti sul serio dal matematico John Tromp e
dai suoi collaboratori, il Go nel 2016 (numero esatto, tutte e centosettantuno
le cifre) e gli scacchi nel 2021. E un solo fotogramma in scala di grigi di un
videogioco Atari, $84$ pixel per $84$ con $256$ livelli, dà $256^{7056}$
situazioni possibili: un numero di quasi diciassettemila cifre, mentre per
contare tutti gli atomi dell'universo osservabile ne basta un'ottantina. Non è
che su quei mondi la tabella sia lenta: non c'è nessun universo in cui la si
possa scrivere. Un milione di stati è poco.

Su tre stati l'onda si esaurisce in due passi. Su una griglia si vede meglio.

```{figure} ../figures/iterazione-valore.gif
:name: fig-iterazione-valore
:alt: Animazione di un mondo a griglia 4x4 con due muri e una casella obiettivo contrassegnata da una stella in alto a destra. A ogni iterazione k i valori delle caselle si aggiornano e la colorazione, che parte dall'obiettivo, si propaga verso le caselle sempre più lontane fino a riempire la griglia.
:width: 90%

La stessa ricetta su un mondo a griglia $4\times4$. Attenzione, è un labirinto
diverso dall'MDP a tre stati: qui l'obiettivo (la stella) paga $+1$ e i passi
non costano nulla, le caselle scure sono muri, e lo sconto vale sempre
$0{,}9$. Detto questo, i numeri dentro le caselle si leggono da soli: la casella
da cui basta una mossa per arrivare vale $1{,}00$, cioè il premio pieno, e ogni
passo indietro lo moltiplica per $0{,}9$, perché lo stesso premio arriva più
tardi: $0{,}90$, poi $0{,}81$, e così via. La casella dell'obiettivo non porta
numeri perché lì la partita è finita. Il valore parte dall'obiettivo e risale
di una casella per
giro, aggirando i muri, finché i numeri smettono di muoversi: qui bastano sei
giri, quanti sono i passi della casella più lontana. A destra della griglia, il
contatore dei giri e la formula dell'aggiornamento, che è quella della scheda
Superiore qui sopra.
```

Su un mondo come questo, dove ogni mossa porta sempre nella stessa casella e il
premio sta tutto sul traguardo, si capisce anche quanti giri servono: tanti
quanti i passi che separano la casella più lontana dall'obiettivo, e infatti la
{numref}`fig-iterazione-valore` si ferma a sei. Quando invece le mosse hanno
esito incerto la cosa cambia: il calcolo non finisce mai del tutto, perché a
ogni giro l'errore residuo viene moltiplicato per lo sconto (con $0{,}9$, dopo
dieci giri è circa un terzo di quello di partenza) e si avvicina a zero senza
mai toccarlo; si smette quando è abbastanza piccolo. In tutti e due i casi
resta vero il punto: il valore non "si diffonde" ovunque insieme, cammina.

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
bisogna *sapere in anticipo* dove porta ogni mossa e quanto paga. Value
iteration e policy iteration richiedono cioè di conoscere il **modello
dell'ambiente**: la mappa, con dentro tutte le mosse e i loro esiti (in simboli,
le due funzioni $P$ e $r$ della scheda qui sopra). È pianificare un viaggio con
la mappa già in mano.
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
- Tutto questo si tiene in una tabella con una casella per situazione, e la
  tabella si scrive finché le situazioni si possono elencare: un milione va
  benissimo, ma gli scacchi ne hanno $10^{44}$ e il Go $10^{170}$. È il muro
  contro cui va a sbattere il capitolo successivo.
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
- Tutto l'impianto presuppone $\mathcal{S}$ **enumerabile**, una casella di
  tabella per stato: $10^6$ stati si trattano, $10^{44}$ (scacchi) o $10^{170}$
  (Go) no. È l'ipotesi che il capitolo seguente dovrà abbandonare.
```

`````
