# Processi decisionali di Markov e funzioni valore

La sezione precedente ha lavorato su un mondo che non si muove: davanti
all'agente c'era una fila di leve sempre uguale, e tirarne una non cambiava in
nulla quello che si sarebbe trovato davanti al tiro dopo. Il mondo vero non è
così. Un bambino che impara ad andare in bicicletta prova, oscilla, cade; e la
pedalata storta non gli costa soltanto un brutto voto, gli **sposta la
bicicletta**: quello che potrà fare fra un istante dipende da quello che ha
fatto ora, e si ritrova in una situazione che si è creato da sé.

Rimettere al suo posto la situazione che cambia è il passo che resta da fare, e
il resto del capitolo lo dà per fatto. Costa qualche simbolo in più, e in cambio
restituisce il problema per intero.

Per trasformare questa intuizione in matematica serve un'impalcatura precisa.
Quell'impalcatura, formalizzata da Richard Bellman nel 1957 e diventata la
spina dorsale del testo di riferimento di Sutton e Barto
{cite}`sutton2018reinforcement`, si chiama **processo
decisionale di Markov**: in inglese *Markov Decision Process*, che tutti
abbreviano in **MDP**, ed è la sigla che d'ora in poi si incontra dappertutto.

## Il ciclo: stati, azioni, ricompense

A ogni istante l'agente si trova in uno **stato**, sceglie un’**azione**,
l'ambiente lo trasporta in un nuovo stato e gli consegna una **ricompensa**
numerica. Poi il ciclo riparte. Tutto il reinforcement learning abita dentro
questo giro.

`````{tab} Elementare

Un piccolo robot si muove in un labirinto a caselle. Lo *stato* è la casella in
cui si trova; le *azioni* sono i movimenti possibili (su, giù, destra,
sinistra); la *transizione* è dove finisce dopo la mossa; la *ricompensa* è il
punteggio che riceve: diciamo $+10$ quando raggiunge l'uscita e $-1$ per ogni
passo, così impara a uscire *in fretta*. Il robot non conosce la mappa: la
scopre muovendosi.

Quei due numeri, il $+10$ dell'uscita e il $-1$ del passo, li scegliamo noi:
sono il modo di dire al robot che cosa vogliamo. Cambiandoli si cambia il
problema, quindi ogni volta che comparirà un labirinto nuovo diremo che regole
ha.

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

Il nome non è un vezzo: rende onore ad Andrej Markov, il matematico russo che
fra Otto e Novecento studiò le sequenze di eventi in cui ciò che viene dopo
dipende soltanto da ciò che c'è adesso. E dice una richiesta molto precisa. Lo
stato deve bastare da solo, deve cioè riassumere tutto quello che serve per
decidere il futuro: **il futuro dipende solo dal presente, non dall'intera
storia passata**.

`````{tab} Elementare

Fotografa una partita a scacchi a metà. A un bravo giocatore, per decidere la
prossima mossa, basta la foto: non gli serve sapere in che ordine i pezzi sono
arrivati lì. La posizione attuale racconta già tutto ciò che conta. Uno stato
fatto così si dice *markoviano*. E se la foto non bastasse? Si allarga
l'inquadratura finché basta: negli scacchi veri, per esempio, alla foto va
aggiunta una nota ("il re non ha ancora mosso"), perché da essa dipende una
mossa speciale, l'arrocco, in cui il re e la torre si scambiano di posto e che
è permessa solo se nessuno dei due si è mai mosso prima. L'importante è che
tutto il necessario stia nella foto, e niente resti nascosto nella storia.

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
tutto ciò che si è visto finora. È la ragione per cui, nel capitolo sui *world
model*, l'agente sceglie l'azione leggendo due cose e non una: ciò che vede in
questo istante, e lo stato nascosto di una rete ricorrente che ha visto tutto
il resto.

`````

## La policy: la strategia dell'agente

Sapere in quali stati ci si può trovare non dice ancora *cosa fare*. La regola
di comportamento dell'agente si chiama **policy**, che è la parola inglese per
«politica» e in questo libro si alterna con «strategia»: le tre parole indicano
la stessa cosa.

`````{tab} Elementare

La policy è l'abitudine dell'agente: per ogni stato, quale azione scegliere.
"In questa stanza vado sempre a destra" è una policy. Può anche assomigliare a
un dado truccato: "qui vado a destra 8 volte su 10". Sembra uno spreco lasciare
due volte su dieci alla sorte, ma è il dilemma del ristorante della panoramica:
se vado sempre a destra, che cosa ci fosse a sinistra non lo scoprirò mai, e la
mia abitudine potrebbe essere sbagliata senza che io possa accorgermene.
Imparare, nel reinforcement learning, significa migliorare la policy.

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

Conviene vedere tutti i pezzi in un solo disegno. La {numref}`fig-mdp` ritrae
un mondo minuscolo con tre stati, che chiamiamo $s_0$, $s_1$ e $s_2$ (sono
soltanto nomi di caselle, e il disegno le mostra). Da $s_0$ l'agente può
salire verso $s_1$ oppure restare fermo; da $s_1$ può scendere verso
l'obiettivo $s_2$ o tornare indietro. L'obiettivo si dice **terminale**, che
vuol dire semplicemente che lì la partita finisce: arrivati, non si fa più
niente e non si incassa più niente. Ogni freccia è un'azione ed è annotata con
la ricompensa che paga: salire non costa nulla, restare o tornare fanno
perdere un punto (nel disegno, $r = -1$), raggiungere l'obiettivo ne fa
guadagnare dieci ($r = +10$). Con $\gamma$ vicino a 1 la strategia migliore è
intuibile a colpo d'occhio ($s_0 \to s_1 \to s_2$) ed è proprio quel "colpo
d'occhio" che le funzioni valore rendono calcolabile in modo sistematico.

```{figure} ../figures/mdp-grafo.svg
:name: fig-mdp
:alt: Grafo con tre stati s0, s1 e s2 (obiettivo, terminale) collegati da frecce che rappresentano le azioni, ciascuna annotata con la ricompensa.
:width: 85%

Un MDP in miniatura: gli stati sono cerchi, le azioni frecce, e ogni freccia
riporta il nome della mossa e la ricompensa che si incassa facendola (la $r$ del
disegno sta per ricompensa). L'obiettivo $s_2$ è terminale: arrivati lì la
partita finisce.
```

## Le funzioni valore: quanto vale trovarsi qui

Una partita sola dice poco. Il ritorno raccolto in un singolo **episodio**,
cioè in una partita giocata dall'inizio alla fine, dipende da come è andata
quella volta: dalle mosse scelte, che possono essere state tirate a sorte, e
dal mondo, che alla stessa mossa può rispondere in modi diversi (una pedalata
non fa sempre lo stesso effetto). Rigiocando viene un numero diverso. Quello
che serve è il ritorno **medio**: quanto promette, in media, trovarsi in una
certa situazione e comportarsi in un certo modo. È il mestiere delle **funzioni
valore**.

`````{tab} Elementare

Due domande, due funzioni. **Quanto è buono trovarsi qui?** è il *valore di
stato* $V$: la ricompensa totale che mi aspetto di raccogliere partendo da
questo stato. **Quanto è buono fare questa mossa qui?** è il *valore di
stato-azione* $Q$: come sopra, ma fissando anche l'azione. $Q$ è spesso più
utile in pratica, perché confrontando le azioni in uno stato mi dice
direttamente quale conviene.

Una precisazione che serve subito, perché altrimenti le ricette qui sotto
sembrano tirare fuori un'idea dal nulla. «Quanto mi aspetto di raccogliere»
dipende da **come gioco**: la stessa casella vale poco per chi si muove a caso
e molto per chi si muove bene, quindi non c'è un valore solo, ce n'è uno per
ogni strategia. In questo capitolo, quando non si dice niente, si intende il
valore **giocando al meglio**, ed è quello che le due ricette qui sotto
calcolano; dove invece interessa il valore di una strategia particolare, lo
diremo.

I conti di questa sezione li faremo tutti sul primo dei due, il valore di una
casella, che è il più corto da scrivere. Il secondo, il valore di una mossa,
torna nell'ultima sezione del capitolo, tanto centrale da dare il nome
all'algoritmo che se ne occupa: il *Q-learning*, che è appunto imparare quella
$Q$ lì.

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

## L'equazione di Bellman: ogni valore si appoggia al successivo

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
l'attesa si ottiene l’**equazione di Bellman** per $V^\pi$:

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
iteration* qui sotto, fino al *Q-learning* con cui il capitolo si chiude.

`````

## Value iteration: l'equazione diventa algoritmo

La regola della scala, da sola, è una fotografia: dice come devono stare i
valori quando sono *giusti* (ogni gradino appoggiato al successivo), ma non
spiega come trovarli, e all'inizio non li conosciamo. Il primo modo per
trovarli è di una semplicità disarmante: usare la regola non come descrizione
ma come *istruzione*, cioè scrivere su ogni casella quello che la regola dice
che dovrebbe esserci, e ripetere finché i numeri non si assestano. In inglese
si chiama **value iteration**, «iterazione dei valori», ed è il nome con cui si
incontra ovunque.

È anche l'idea con cui Bellman inaugurò la **programmazione dinamica**
{cite}`bellman1957dynamic`: due parole che non spiegano niente (e lo sapeva lui
per primo, che le scelse anche perché suonavano innocue a chi doveva
finanziarlo), ma che ancora oggi indicano questo, risolvere un problema grande
riusando le risposte già trovate ai suoi pezzi piccoli.

`````{tab} Elementare

La ricetta, nel labirinto: scrivi $0$ su ogni casella. Poi, casella per
casella, guarda tutte le mosse possibili e chiediti: "quanto rende ciascuna,
contando la ricompensa immediata più il valore (scontato) della casella dove
finirei?". Scrivi sulla casella il risultato della mossa **migliore**, perché
il valore che stiamo calcolando è quello di chi gioca al meglio. Finito il
giro, ricomincia da capo con i numeri nuovi, e poi ancora, finché i numeri
smettono di muoversi. A quel punto ogni casella dice quanto vale *davvero*, e
la strategia migliore è in omaggio: da ogni casella, scegli la mossa che rende
di più.

Un dettaglio del "giro" va fissato adesso, perché senza di quello i conti qui
sotto sembrano sbagliati. Si compila una scheda nuova guardando la vecchia, non
si corregge la vecchia mentre la si legge. Quindi, dentro un giro, i numeri che
si leggono sono sempre quelli con cui il giro è cominciato: anche quelli di una
casella che nel frattempo si è già riscritta.

`````

`````{tab} Superiore

Partendo da una stima arbitraria $V_0$ (tipicamente nulla), si itera

$$
V_{k+1}(s) = \max_{a} \sum_{s'} P(s'\mid s,a)
\big[\,r(s,a) + \gamma\, V_k(s')\,\big],
$$

dove $V_k$ è la stima dei valori al passo $k$: è l'equazione di Bellman con un
$\max$ sulle azioni al posto della media pesata dalla policy. Il punto fisso è
l’**equazione di ottimalità di Bellman**,
$V^*(s) = \max_a \sum_{s'} P(s'\mid s,a)\big[r(s,a) + \gamma\, V^*(s')\big]$,
dove $V^*$ è il valore della migliore policy possibile. Con $\gamma < 1$ la convergenza è garantita: l'operatore di aggiornamento è
una **contrazione** di fattore $\gamma$ nella norma del massimo (nei compiti
episodici con $\gamma = 1$ il fattore di contrazione sparisce, e la garanzia
va ricomprata altrove: serve che ogni policy raggiunga con probabilità $1$ uno
stato terminale) (a ogni passo la distanza da $V^*$ si riduce
almeno di un fattore $\gamma$) quindi il punto fisso è unico e l'iterazione vi
arriva da qualunque inizializzazione {cite}`bellman1957dynamic`
{cite}`sutton2018reinforcement`. Estratto $V^*$, la policy ottima è quella
*greedy*: in ogni stato, l'azione che realizza il massimo.

`````

## La value iteration all'opera

Facciamo davvero i conti, sull'MDP in miniatura della {numref}`fig-mdp` e con
uno sconto di $0{,}9$. In quel mondo ogni mossa porta sempre nella stessa
casella (si dice che le transizioni sono **deterministiche**: la stessa mossa fa
sempre la stessa cosa, e non c'è nessuna media da fare fra esiti diversi),
quindi la ricetta si legge senza complicazioni: "quanto
paga la mossa, più $0{,}9$ volte il valore della casella dove si finisce", e si
tiene la mossa che rende di più. L'obiettivo $s_2$ vale sempre $0$, perché lì la
partita è finita e non c'è più niente da raccogliere; e si comincia scrivendo
$0$ anche sulle altre due caselle, tanto per avere un punto di partenza.

`````{tab} Elementare

Vale la regola di prima: dentro un giro si leggono i numeri con cui il giro è
cominciato.

**Primo giro.** Cominciamo da $s_1$, la casella accanto all'obiettivo. L'ordine
non conta, dato che i numeri li leggiamo tutti dalla scheda vecchia: comincia da
$s_0$ e i risultati sono gli stessi. Scendere paga $10$ subito e porta
nell'obiettivo, che vale $0$: in tutto $10 + 0{,}9 \times 0 = 10$. Tornare
indietro costa $1$ e porta in $s_0$, che per adesso vale $0$: in tutto
$-1 + 0{,}9 \times 0 = -1$. Vince scendere, e su $s_1$ scriviamo $10$. Passiamo
a $s_0$: salire non costa nulla e porta in $s_1$, che sulla scheda vecchia vale
ancora $0$ (il $10$ l'abbiamo appena scritto su quella nuova, e in questo giro
non si legge), quindi rende $0$; restare fermi costa $1$ e lascia dove si è,
cioè in $s_0$, che vale $0$, quindi rende $-1$. Vince salire, e su $s_0$
scriviamo $0$. Il premio è entrato in $s_1$, ma in $s_0$ non è ancora arrivato.

**Secondo giro.** Su $s_1$ non cambia niente, resta $10$. Su $s_0$ invece salire
adesso porta in una casella che vale $10$, quindi rende
$0 + 0{,}9 \times 10 = 9$, contro il $-1$ di restare fermi: scriviamo $9$. Il
premio ha fatto un altro passo all'indietro.

**Terzo giro.** Rifacendo gli stessi conti non si muove più niente. Da $s_0$
salire rende ancora $9$, mentre restare fermi adesso costa $1$ e lascia in una
casella che vale $9$, cioè $-1 + 0{,}9 \times 9 = 7{,}1$: meno di $9$, quindi
si sale ancora. Da $s_1$ scendere rende ancora $10$, mentre tornare costa $1$ e
porta in $s_0$, che vale $9$: $-1 + 0{,}9 \times 9 = 7{,}1$ di nuovo. Che venga
lo stesso numero non è un mistero: le due mosse costano tutte e due $1$ punto e
finiscono tutte e due in una casella che vale $9$, quindi il conto è
letteralmente lo stesso. I numeri si sono fermati, e allora abbiamo finito: è
questo che si intende con «finché non si assestano», e infatti nella tabella
qui sotto le ultime due righe sono uguali.

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
finché le situazioni si possono elencare, e ci sono giochi comunissimi in cui
non si possono.

Gli scacchi hanno circa $4{,}8 \times 10^{44}$ posizioni legali, dove
$10^{44}$ è la scrittura breve di «uno seguito da quarantaquattro zeri». Il Go,
che si gioca sugli incroci di una griglia di diciannove righe per diciannove,
ne ha circa $2{,}08 \times 10^{170}$. Sono due conti fatti sul serio dal
matematico John Tromp e dai suoi collaboratori: il Go nel 2016, dove il numero
è esatto e ha tutte e centosettantuno le sue cifre, e gli scacchi nel 2021,
dove è una stima.

E poi c'è il caso che chiude il discorso. Prendiamo un solo fotogramma di un
videogioco Atari, ridotto come lo riducevano quegli agenti: niente colori, solo
sfumature di grigio, e una griglia di $84$ punti per $84$ (una misura scelta da
loro, abbastanza piccola da essere maneggiabile e abbastanza grande da vederci
ancora qualcosa). Sono $7056$ punti, e ognuno può essere in uno di $256$
grigi, dal nero al bianco. Le combinazioni si contano
moltiplicando: due punti da $256$ grigi danno $256 \times 256$ immagini
diverse, tre punti $256 \times 256 \times 256$, e settemila punti danno $256$
moltiplicato per sé stesso settemila volte, che si scrive $256^{7056}$. È un
numero di quasi diciassettemila cifre; per contare tutti gli atomi
dell'universo osservabile ne bastano un'ottantina. Non è che su quei mondi la
tabella sia lenta: non c'è nessun universo in cui la si possa scrivere. Un
milione di stati è poco.

Il premio, si è visto, non resta fermo dov'è: risale il mondo una casella per
giro, come un'onda che parte dal traguardo e va all'indietro. Su tre stati
quell'onda si esaurisce in due passi, e non c'è granché da guardare. Su una
griglia si vede meglio, ed è quello che mostra la
{numref}`fig-iterazione-valore`.

Attenzione, è un labirinto diverso da quello a tre caselle di prima: qui
l'obiettivo (la stella) paga $+1$, i passi non costano nulla, le due caselle
scure sono muri, e lo sconto vale sempre $0{,}9$. Con queste regole i numeri
dentro le caselle si leggono da soli: la casella da cui basta una mossa per
arrivare vale $1{,}00$, cioè il premio pieno, e ogni passo indietro lo
moltiplica per $0{,}9$, perché lo stesso premio arriva più tardi: $0{,}90$, poi
$0{,}81$, e così via. Dopo sei giri i numeri si fermano, e sei sono esattamente
i passi che separano dall'obiettivo la casella più lontana, quella in basso a
sinistra: si contino sul disegno, aggirando i muri, e tornano.

```{figure} ../figures/iterazione-valore.gif
:name: fig-iterazione-valore
:alt: Animazione di un mondo a griglia 4x4 con due muri e una casella obiettivo contrassegnata da una stella in alto a destra. A ogni iterazione k i valori delle caselle si aggiornano e la colorazione, che parte dall'obiettivo, si propaga verso le caselle sempre più lontane fino a riempire la griglia.
:width: 90%

La stessa ricetta su un mondo a griglia $4\times4$. Il valore parte
dall'obiettivo e risale la griglia di una casella per giro, aggirando i muri,
finché i numeri smettono di muoversi; la casella dell'obiettivo non porta
numeri perché lì la partita è finita. A destra, il contatore dei giri e la
stessa ricetta scritta in simboli: il valore nuovo di una casella è, fra tutte
le mosse possibili, la migliore fra «quanto paga la mossa, più lo sconto per il
valore vecchio della casella dove si finisce».
```

Che i giri siano esattamente sei vale però solo in un mondo come questo, dove
ogni mossa porta sempre nella stessa casella e il premio sta tutto sul
traguardo. Quando le mosse hanno esito incerto, cioè quando la stessa mossa a
volte riesce e a volte no, il calcolo non finisce mai del tutto. A ogni giro,
però, quello che ancora manca ai numeri veri si riduce, e si riduce
moltiplicandosi per lo sconto: il motivo è che l'errore di una casella entra
nel conto della casella prima soltanto dopo essere passato per una
moltiplicazione per $0{,}9$, ed è quella moltiplicazione che se lo mangia, un
giro alla volta. Ridursi però non è azzerarsi: con uno sconto di $0{,}9$, dopo
dieci giri manca circa un terzo di quello che mancava all'inizio ($0{,}9$
moltiplicato per sé stesso dieci volte fa circa $0{,}35$), dopo altri dieci un
terzo di quel terzo, e così via. Si smette quando è abbastanza piccolo. In
tutti e due i casi resta vero il punto: il valore non "si diffonde" ovunque
insieme, cammina.

## Policy iteration: valutare e migliorare, a turni

La value iteration fonde due gesti in un unico aggiornamento: stimare quanto
rendono gli stati e scegliere le azioni migliori. La **policy iteration**, che si deve a Ronald Howard
{cite}`howard1960dynamic`, li separa e li alterna: prima *valuta* fino in
fondo la policy corrente, poi la *migliora*, e ricomincia.

`````{tab} Elementare

Restiamo nel labirinto, che è più concreto. Immagina di avere già in mano una
strategia, anche stupida: in ogni casella una freccia che dice dove andare.
Primo tempo, la **pagella**: tenendo quelle frecce ferme, si calcola con
pazienza quanto rende partire da ogni casella, e si ricalcola finché i numeri
non si assestano. Secondo tempo, la **correzione**: con la pagella davanti si
scorrono le caselle una per una, e dove una freccia diversa porterebbe in un
posto che vale di più, si gira la freccia. Poi si rifà la pagella per le frecce
nuove, si corregge ancora, e avanti così. Quando un giro di correzioni non gira
più nessuna freccia, quella è la strategia migliore possibile.

Detta così sembra troppo bella: e se mi fossi incastrato in una strategia
mediocre che da sola non riesce a migliorarsi? Non succede, ed è un teorema: in
questo tipo di problema, se non esiste **nemmeno una** casella in cui una
freccia diversa renda di più, allora non esiste nemmeno un cambio di molte
frecce insieme che renda di più. Il controllo casella per casella, che sembra
miope, basta.

La differenza con il metodo di prima è il ritmo. Là ogni giro era leggero
(un'occhiata sola per casella) e i giri erano tanti; qui i giri sono pochi,
spesso una manciata, ma ognuno contiene una pagella completa, che è un lavoro
lungo.

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
si mescolano le due, fermando la pagella dopo poche riletture invece di
portarla fino in fondo.

## Quando manca la mappa

C'è però un dettaglio che finora abbiamo dato per scontato, ed è enorme. Per
fare quei conti ("ricompensa della mossa più valore della casella d'arrivo"),
bisogna *sapere in anticipo* dove porta ogni mossa e quanto paga. Le due
ricette appena viste richiedono cioè di avere in mano la mappa: per ogni mossa,
dove si finisce (e con quali probabilità, quando l'esito è incerto) e quanto si
incassa a farla. È pianificare un viaggio con la cartina già aperta sul tavolo.

Quella mappa ha un nome tecnico, ed è **modello dell'ambiente**. Attenzione a
non confonderlo con il «modello» di cui si parla altrove, la rete neurale
addestrata: qui modello vuol dire una descrizione di come funziona il mondo,
niente di più. Il robot del nostro labirinto quella descrizione non ce l'ha, e
il mondo reale quasi mai la consegna: nessuno può dire a un agente, per ogni
mossa e in anticipo, con che probabilità troverà traffico o come risponderà
l'avversario a Go.

Quando la mappa manca resta una sola strada: stimare i valori
*dall'esperienza*, cioè giocando. E ci sono due modi di percorrerla, che le
prossime due sezioni prendono in ordine.

Il primo è il più diretto che si possa immaginare. Si gioca una partita intera,
si guarda quanti punti si sono fatti, e si usa quel totale per dare un voto a
tutte le caselle attraversate. Poi un'altra partita, e un'altra ancora, e si fa
la media. Sono i **metodi Monte Carlo**, dal nome del casinò, perché tutto si
regge sul ripetere molte volte una cosa che ogni volta va a finire
diversamente.

Il secondo non aspetta nemmeno la fine della partita. Dopo ogni singola mossa
guarda dov'è finito, legge il numero che era già scritto su quella casella (un
numero provvisorio, magari sbagliato, ma è quello che si ha) e con quello
corregge subito il numero della casella da cui era partito. Correggere una
propria stima appoggiandosi a un'altra propria stima sembra un trucco da
illusionisti, e per certi versi lo è; funziona, e si chiama apprendimento per
**differenze temporali**, perché la correzione nasce dalla differenza fra
quello che si credeva un istante fa e quello che si crede adesso. Il suo
esemplare più famoso è il **Q-learning**, con cui il capitolo si chiude.

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
  benissimo, ma gli scacchi ne hanno uno seguito da quarantaquattro zeri e il
  Go uno seguito da centosettanta zeri. È il muro che il capitolo successivo esiste
  per aggirare.
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
- $V^\pi$ e $Q^\pi$ misurano il ritorno *atteso*; l’**equazione di Bellman** li
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
