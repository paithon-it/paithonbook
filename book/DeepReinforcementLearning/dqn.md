---
titolo_ricerca: "DQN: la rete che gioca ad Atari dai pixel"
descrizione: "Come funziona la Deep Q-Network: la rete al posto della tabella,
  perché poteva divergere, gli accorgimenti per stabilizzarla e il limite della
  sovrastima."
---
# Deep Q-Network (DQN)

Il risultato del 2013 su *Breakout* veniva da un programma solo, provato su
sette videogiochi Atari, da *Pong* a *Space Invaders*, e in tre di quei sette
(*Breakout*, *Enduro* e *Pong*) superava un giocatore umano esperto
{cite}`mnih2013playing`. Due anni dopo la versione definitiva finì sulla
copertina di *Nature* {cite}`mnih2015human`. Quel programma si chiama Deep
Q-Network, DQN.

Il {doc}`Q-learning </ReinforcementLearning/q-learning>` impara una funzione
$Q(s,a)$ che stima quanto è conveniente, nel lungo periodo, compiere l'azione
$a$ trovandosi nello stato $s$, e la tiene in una tabella: una riga per ogni
stato, una colonna per ogni azione. Questo funziona con pochi stati, ma non con
le schermate di un videogioco, come mostra il calcolo nell’{doc}`apertura del
capitolo <overview>`: già i fotogrammi ridotti a $84\times84$ pixel in scala
di grigi sono tanti che il loro numero ha quasi diciassettemila cifre, e a una
tabella così non basterebbe l'universo. Al posto della tabella serve quindi
una funzione che generalizzi, cioè che dia valori simili a stati simili anche
quando non li ha mai visti.

## Dalla tabella alla rete

DQN è semplice quanto radicale: [butta via la tabella e mette al suo posto una
rete neurale]{.pt-matita}.

`````{tab} Elementare

Uno schedario, un cartellino per ogni possibile schermata di gioco, e su ciascun
cartellino quanto vale ciascuna mossa: è la tabella di prima, e compilarlo, si è
visto, non si può. Al posto dello schedario si mette allora un *esperto* che
guarda la schermata e dice il valore di tutte le mosse insieme, in un colpo
solo, senza passarle in rassegna una per volta. E lo fa anche per schermate che
non ha mai visto prima, perché ha imparato a riconoscere le somiglianze.
Quell'esperto è la rete neurale.

Dentro la rete ci sono dei numeri, nella rete di DQN più di un milione e mezzo,
che decidono come una schermata si trasforma in un voto. Si chiamano pesi, e
sono le uniche cose che cambiano mentre la rete impara. Addestrare la rete vuol
dire ritoccarli, un pochino alla volta, finché i voti non diventano sensati.

`````

`````{tab} Superiore

Approssimiamo la funzione azione-valore ottima con una rete parametrizzata da
$\theta$:

$$
Q(s, a; \theta) \approx Q^{*}(s, a).
$$

La rete prende in ingresso lo stato $s$ (i pixel) e restituisce in uscita un
vettore con un valore $Q$ per ciascuna azione ammissibile, quindi non serve
una passata per azione. È un *function approximator*, che
generalizza a stati mai visti sfruttando la struttura condivisa delle
immagini invece di memorizzare ogni caso singolarmente.

`````

Lo stato entra da un lato, una rete convoluzionale ne estrae le
caratteristiche visive, e dall'altro lato escono i valori delle azioni
({numref}`fig-dqn`).

```{figure} ../figures/schema-dqn.svg
:name: fig-dqn
:alt: Lo schermo di un gioco stile Breakout entra in una rete convoluzionale con volumi decrescenti, seguita da uno strato denso, e produce una barra di valore Q per ciascuna delle quattro azioni possibili.
:width: 95%

Lo schema di DQN. Lo schermo di gioco attraversa gli strati convoluzionali
e uno strato denso; l'uscita è un valore $Q$ per ogni azione, cioè un voto per
ogni mossa. L'agente sceglie l'azione con il valore più alto.
```

## Perché poteva divergere: la triade fatale

Mettere una rete al posto della tabella non era un'idea nuova. TD-Gammon lo
faceva dal 1992, con una rete addestrata su partite di backgammon giocate
contro sé stessa; nella versione descritta tre anni dopo giocava quasi alla
pari dei più forti giocatori del mondo {cite}`tesauro1995temporal`.

Il Q-learning con una rete, invece, era noto per la sua instabilità: i valori
stimati potevano crescere senza fermarsi, invece di assestarsi. In casi
particolari, però, funzionava: il replay delle esperienze risale a Long-Ji
Lin, nel 1992 {cite}`lin1992self`, e con aggiornamenti a lotti funzionava
anche il *Neural Fitted Q-iteration* di Martin Riedmiller
{cite}`riedmiller2005neural`. In modo affidabile, su problemi grandi e
partendo dai pixel, no.

Per la stabilità, TD-Gammon e il Q-learning differiscono in un punto.
TD-Gammon aggiornava i valori della policy che stava giocando (apprendimento
*on-policy*), mentre il Q-learning gioca in un modo e ne impara un altro. Ogni
tanto, apposta, sceglie un'azione a caso per esplorare, ma il bersaglio verso
cui corregge la stima non usa il valore di quell'azione: usa quello della
*migliore* fra le azioni disponibili nella situazione successiva.

La policy che raccoglie i dati ($\varepsilon$-greedy) e quella di cui si
stimano i valori (greedy) sono quindi due policy diverse, e si parla di
apprendimento *off-policy*. È comodo, perché permette di imparare da qualunque
partita, anche da una giocata male o da una giocata da altri molto tempo
prima.

L'instabilità non dipendeva dalla sfortuna né da un passo di apprendimento
regolato male (il *learning rate*, cioè di quanto si spostano i pesi a ogni
correzione). I due accorgimenti che hanno reso praticabile il metodo vengono
dopo; il difetto da cui nasce l'instabilità, intanto, ha una descrizione
sorprendentemente pulita.

`````{tab} Elementare

Al cabinato c'è un tale col taccuino: guarda giocare e per ogni schermata
segna quanto promette bene. Ha tre abitudini.

Non tiene un foglio per ogni schermata, giudica a somiglianza: ci rimette in
precisione sulla singola, e in cambio ha un voto anche per quelle che non ha
mai visto. È l’approssimazione.

Non aspetta la fine della partita. Ad esempio, una mossa gli frutta $1$ punto,
valuta $7$ la schermata in cui si ritrova, e scrive subito i due numeri messi
insieme, il secondo contato per nove decimi: $7{,}3$, il bersaglio verso cui
correggerà il voto di partenza. Aggiustare un voto con un altro voto si chiama
bootstrapping, e separa le {doc}`differenze
temporali </ReinforcementLearning/q-learning>`, il TD, dai metodi Monte Carlo,
che aspettano il fischio finale per tirare le somme.

Guarda partite giocate a casaccio e scrive i voti come se al posto di quel
giocatore ci fosse un campione: si gioca da esploratore e si prendono appunti
da campione. È l'off-policy.

Ciascuna abitudine è utile, e a coppie il taccuino si tiene in ordine. [Tutte e
tre insieme, però, espongono a un rischio: i voti possono crescere senza
fermarsi]{.pt-matita}. Non succede per forza, e ci sono taccuini più
complicati, costruiti apposta, che lo evitano; ma il pericolo c'è, e Richard
Sutton e Andrew Barto, autori del manuale classico della materia, chiamano
**triade fatale** la combinazione dei tre.

Per vederla basta il gioco più facile possibile. Lo costruì Baird, e si chiama
**controesempio di Baird**: sette schermate, e non si guadagna mai un punto.
Zero dappertutto è la risposta giusta, e il taccuino potrebbe scriverla alla
perfezione: basterebbe mettere a zero la manciata di numeri con cui calcola i
voti. Quei numeri, invece di posarsi sullo zero, crescono e non smettono più.

Il tale ritocca il voto di una schermata, e per somiglianza si spostano anche
quelli delle vicine, e di solito è un vantaggio. Il bersaglio da cui era
partito, però, è il voto di una vicina, uno di quelli che ha appena mosso.
Così ogni ritocco sposta il bersaglio che l'aveva deciso, e il ritocco
seguente insegue un bersaglio già spostato: i voti si rimpallano fra loro.

Di solito quel rimpallo si spegne, e il motivo sta in quanto spesso il tale
vede ciascuna schermata. Se le partite che guarda sono giocate come le
giocherebbe lui, le schermate frequenti gli passano davanti spesso e quelle
rare di rado, nelle proporzioni in cui contano davvero; e sono le frequenti a
decidere come va a finire, così il rimpallo si smorza.

Due studiosi, John Tsitsiklis e Benjamin Van Roy, hanno dimostrato nel 1997
che va così, ma solo per il taccuino più semplice, quello in cui il voto di
una schermata è una somma di pezzi, uno per ogni cosa che si vede (tanto per
la pallina lì, tanto per la racchetta là). Con una rete a molti strati non
c'è riuscito nessuno, e si conoscono casi in cui i voti scappano anche così.

La terza abitudine toglie questa protezione. Le partite del tale sono
giocate in un altro modo, quindi certe schermate gli passano davanti molto più
spesso di quanto capiterebbero nel suo gioco, altre quasi mai. Il tale
corregge con forza dove non serve, il rimpallo non si smorza più, e i voti
salgono invece di posarsi.

`````

`````{tab} Superiore

La **triade fatale** {cite}`sutton2018reinforcement` è la coesistenza di:

1. approssimazione di funzione, cioè una rappresentazione parametrica che
   generalizza fra stati invece di trattarli come voci indipendenti;
2. bootstrapping, cioè bersagli che contengono stime correnti (TD,
   programmazione dinamica) invece dei soli ritorni osservati;
3. addestramento off-policy, cioè una distribuzione degli aggiornamenti
   diversa da quella indotta dalla policy che si sta valutando.

Con due soli dei tre l'instabilità si può evitare. [Con tutti e tre il pericolo
c'è]{.pt-matita}, e la divergenza si osserva già nel caso della sola
predizione, senza controllo né miglioramento della policy. Non dipende nemmeno
dall'incertezza sull'ambiente, poiché si manifesta identica nella
programmazione dinamica, dove il modello è noto per intero.

La divergenza, tuttavia, non è inevitabile. I metodi *Gradient-TD*, che
scendono lungo il gradiente dell'errore di Bellman proiettato a costo di un
secondo vettore di parametri con il suo passo, e gli *Emphatic-TD*, che
ripesano gli aggiornamenti, combinano tutti e tre gli elementi con garanzie di
convergenza nel caso dell'approssimazione lineare
{cite}`sutton2018reinforcement`.

Il controesempio di Baird esibisce la divergenza in forma minima: sette
stati, due azioni, ricompensa sempre nulla, $\gamma = 0{,}99$, e una policy di
comportamento che visita gli stati in modo uniforme mentre la policy bersaglio
ne concentra tutta la massa su uno solo. La funzione valore vera è
identicamente zero ed è
esattamente rappresentabile dai parametri disponibili; il TD semi-gradiente,
ciononostante, fa divergere i pesi. Il fattore decisivo è la distribuzione
degli aggiornamenti, che è uniforme sugli stati, mentre la policy bersaglio li
visiterebbe in proporzioni tutte diverse.

Per spiegare la divergenza non basta osservare che l'aggiornamento
semi-gradiente non è il gradiente di nessuna funzione obiettivo (si deriva
rispetto alla stima ma non rispetto al bersaglio, che pure dipende dai
parametri). Lo stesso aggiornamento, infatti, con approssimatore lineare e
sotto la distribuzione on-policy converge, come dimostrarono Tsitsiklis e Van
Roy nel 1997 {cite}`tsitsiklis1997analysis`. Le garanzie di convergenza note
si fermano però a quel caso, lineare e on-policy; con approssimatori non
lineari come le reti si conoscono controesempi di divergenza perfino
on-policy.

Sutton e Barto passano poi in rassegna i tre elementi per vedere a quale si
possa rinunciare. All’approssimazione no, perché senza di essa il metodo non
scala. Al bootstrapping si può, usando Monte Carlo, a scapito dell'efficienza
computazionale (bisogna conservare tutto fino alla fine dell'episodio) e
dell'efficienza nei dati. All’off-policy si può, sostituendo il Q-learning con
Sarsa, e si perde la possibilità di imparare da un archivio di esperienze
altrui, che è però la premessa della memoria di replay.

`````

## Due accorgimenti per non far esplodere l'addestramento

[DQN tiene tutti e tre gli elementi della triade, senza rinunciare a
nessuno]{.pt-matita}, e ne rende praticabile la convivenza con due
accorgimenti.

### Experience replay

`````{tab} Elementare

Un agente che impara sui fotogrammi *nell'ordine in cui li vive* è come uno
studente che rilegge cento volte la stessa pagina di seguito. I fotogrammi
consecutivi, infatti, si somigliano troppo, e la rete finisce per "fissarsi".

La **memoria di replay** è un grande quaderno degli appunti: ogni esperienza
vissuta viene annotata e, per allenarsi, l'agente pesca a caso vecchie
esperienze dal quaderno. Così [mescola situazioni lontane nel tempo e impara in
modo più equilibrato]{.pt-matita}, e riutilizza ogni esperienza molte volte,
non una sola.

`````

`````{tab} Superiore

Ogni transizione $(s, a, r, s')$ viene salvata in un buffer $\mathcal{D}$
(tipicamente un milione di transizioni). L'aggiornamento dei pesi avviene su
*minibatch* campionati uniformemente da $\mathcal{D}$, e non sull'ultima
transizione. In questo modo [si rompe la correlazione temporale tra campioni
consecutivi]{.pt-matita} (che violerebbe l'ipotesi di indipendenza nella discesa
stocastica del gradiente) e si aumenta enormemente l'efficienza nell'uso dei
dati, riutilizzando ogni transizione in molti aggiornamenti.

`````

### Rete-target

`````{tab} Elementare

C'è un secondo problema: la rete deve inseguire un bersaglio che lei stessa
sposta a ogni passo, come cercare di colpire la propria ombra. La soluzione è
tenere due copie della rete: una che impara di continuo e una "congelata"
che fornisce il bersaglio e viene aggiornata solo ogni tanto. [Il bersaglio
resta fermo abbastanza a lungo perché la rete che apprende riesca a
raggiungerlo]{.pt-matita}.

`````

`````{tab} Superiore

Si mantiene una **rete-target** con parametri $\theta^{-}$, copia periodica
dei parametri $\theta$ ogni $C$ passi. La rete con parametri $\theta$, detta
**rete online**, sceglie le azioni e si aggiorna a ogni passo, minimizzando
l'errore quadratico sull'equazione di Bellman:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s',d)\sim U(\mathcal{D})}
\left[\big(\, r + \gamma\,(1-d) \max_{a'} Q(s', a'; \theta^{-}) - Q(s, a;
\theta)\,\big)^2\right].
$$

Qui $r$ è la ricompensa immediata, $\gamma\in[0,1)$ il fattore di sconto,
$d\in\{0,1\}$ vale $1$ quando $s'$ è terminale (e allora del futuro non resta
niente), e il termine $y = r + \gamma\,(1-d)\max_{a'} Q(s', a'; \theta^{-})$ è
il bersaglio, calcolato con i pesi congelati $\theta^{-}$. [Congelarli attenua
il *feedback* instabile in cui il bersaglio si muove insieme alla
stima]{.pt-matita}; con le parole del lavoro, divergenza e oscillazioni
diventano molto meno probabili.

Due dettagli del lavoro su *Nature* {cite}`mnih2015human` cambiano questa
perdita. Nel gradiente l'errore $\delta = y - Q(s,a;\theta)$ subisce il clipping in
$[-1,1]$, il che equivale alla perdita di Huber, quadratica per $|\delta| \le
1$ e lineare oltre; in questo modo un errore di cento contribuisce al
gradiente quanto un errore di uno. E le ricompense subiscono il clipping a $-1$,
$0$, $+1$, perché lo stesso passo valga su giochi con scale di punteggio
diversissime, il che impedisce però di distinguere una ricompensa piccola da
una grande.

Completano la procedura un $\varepsilon$ portato da $1$ a $0{,}1$ nel primo
milione di fotogrammi e ogni azione ripetuta per quattro fotogrammi. Nel
lavoro, quindi, un «fotogramma» è un passo dell'agente: i $50$ milioni
dell'addestramento sono $200$ milioni di fotogrammi dell'emulatore, circa $38$
giorni di gioco a sessanta al secondo.

`````

Sono due modi di rompere lo stesso legame, quello che tiene attaccate fra loro
cose che si susseguono nel tempo. La memoria di replay lo spezza fra
un'esperienza e la successiva, estraendo a caso invece che nell'ordine in cui
le cose sono state vissute; la rete-target lo spezza fra la stima e il bersaglio
che la guida, tenendo fermo il secondo mentre la prima si muove
({numref}`fig-dqn-stabilita`).

```{figure} ../figures/dqn-stabilita.svg
:name: fig-dqn-stabilita
:alt: "Animazione: a sinistra la memoria di replay, ventiquattro celle disposte in ordine di arrivo, di cui a ogni passo se ne accendono quattro pescate a caso, sparpagliate nella memoria invece che una di fila all'altra: sono il minibatch. A destra il valore Q di una stessa coppia stato-azione: la curva della rete che impara sale a ogni passo, mentre la scaletta della rete-target resta ferma per tre passi e poi scatta a raggiungerla."
:width: 95%

I due accorgimenti al lavoro sullo stesso addestramento. A sinistra la memoria
di replay: le esperienze entrano in ordine, e la manciata su cui si studia (il
*minibatch*) le pesca a caso. A destra lo stesso valore $Q$ calcolato dalla rete
che impara, che si muove a ogni passo, e dalla rete-target, che resta ferma
per un numero fisso di passi (nel disegno tre) e poi scatta a raggiungerla. La
figura è in scala ridotta: in un addestramento vero le esperienze in memoria
sono un milione e quelle pescate a ogni aggiornamento sono trentadue.
```

Vediamo ora, per ciascuno dei due accorgimenti, quale dei tre elementi della
triade attenua; la risposta è meno simmetrica di quanto sembri. La rete-target
attenua il bootstrapping, cioè il correggere una stima guardandone un'altra,
perché quell'altra adesso sta ferma per un po’ e si fa raggiungere.

La memoria di replay, invece, non ne attenua nessuno, e stabilizza per
un'altra via: spezza il legame fra un'esperienza e la successiva, che non è
uno dei tre elementi. Sull'off-policy, semmai, tira dall'altra parte, perché
estrarre da un milione di esperienze memorizzate vuol dire allenarsi su
partite giocate da versioni vecchie della policy, più lontane da quella di
adesso. Anzi, la memoria di replay presuppone l'apprendimento off-policy:
senza un bersaglio che valuti una policy diversa da quella che ha raccolto i
dati, un'esperienza vecchia non si potrebbe riusare affatto.

Il primo elemento, la rete al posto della tabella, resta intatto, ed è quello
per cui si è fatto tutto il resto.

La rete, in PyTorch, si costruisce in poche righe. I fotogrammi arrivano
ridotti a $84\times84$ pixel in scala di grigi e impilati a quattro a quattro,
perché da una sola immagine ferma non si capisce dove stia andando la pallina.

I tre strati convoluzionali usano filtri $8\times8$ con stride $4$, $4\times4$
con stride $2$ e $3\times3$ con stride $1$, senza bordo aggiunto. Un filtro
$k\times k$ con stride $s$ porta un lato di $n$ pixel a $\lfloor
(n-k)/s\rfloor + 1$, come nella {doc}`sezione sulle reti convoluzionali
</DeepLearning/reti-convoluzionali>`, quindi il lato passa da $84$ a $20$, poi
a $9$, poi a $7$. Restano così $7\times7$ posizioni per ciascuno dei $64$
filtri dell'ultimo strato, ognuno un rivelatore di un tratto dell'immagine (un
bordo verticale, la pallina, e così via). Da qui viene il `64 * 7 * 7` del
primo strato denso, quello in cui ogni numero in entrata è collegato a ogni
numero in uscita.

```python
from torch import nn

def crea_q_network(n_azioni):
    # ingresso (4, 84, 84): 4 fotogrammi impilati (per cogliere il movimento)
    return nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 512),
        nn.ReLU(),
        nn.Linear(512, n_azioni),  # un valore Q per azione, nessuna attivazione
    )

rete_q = crea_q_network(n_azioni=4)            # Breakout ha quattro azioni
print(f"pesi da imparare: {sum(p.numel() for p in rete_q.parameters())}")
```

```text
pesi da imparare: 1686180
```

Il bersaglio si calcola con la rete-target, e quando la partita finisce lì (uno
stato *terminale*, senza seguito e quindi senza ricompense future da
aggiungere) si tiene la sola ricompensa:

```{code-block} python
:class: pt-non-eseguibile

import torch

# minibatch: 32 esperienze pescate a caso dalla memoria di replay
s, a, r, s_next, fine = replay.campiona(batch=32)

with torch.no_grad():                              # il bersaglio non si corregge
    # il voto migliore nella situazione seguente, secondo la COPIA CONGELATA
    q_next = target_net(s_next).max(dim=1).values
# gamma (fra 0 e 1) dice quanto conta il futuro rispetto al presente
bersaglio = r + gamma * q_next * (1 - fine)        # se finisce qui, resta solo r
```

## Atari: giocare partendo dai pixel

[Il dettaglio storicamente rilevante è l'ingresso della rete: l'immagine, cioè
i quattro fotogrammi impilati, e nient'altro]{.pt-matita}. Nessuna informazione
sulle regole, e nessuna misura scelta e calcolata a mano da un programmatore
(in gergo, nessuna *feature*, ad esempio la «distanza fra pallina e racchetta»
o il «numero di mattoni rimasti»).

Inoltre, da un gioco all'altro non cambiava niente. Lo stesso algoritmo ha
imparato i 49 titoli del confronto con il collaudatore umano raccontato
nell’{doc}`apertura del capitolo <overview>`, con gli stessi *iperparametri*
(i valori che si decidono prima e non si imparano), scelti una volta per tutte
con qualche prova su cinque di quei giochi. Il lavoro lo presenta come il
primo agente capace di eccellere in una gamma così ampia di compiti partendo
da input sensoriali grezzi.

## Il massimo sovrastima i valori

Nel bersaglio di DQN c'è un'operazione che sembra innocua e non lo è: prendere
il valore più alto. Capire perché il massimo porta a sovrastimare i valori
serve per due ragioni. Il fenomeno è controintuitivo (prendere il massimo è
l'operazione che si vuole fare), e lo stesso difetto, con la stessa cura,
torna identico nella {doc}`sezione sul controllo continuo
<controllo-continuo>`.

`````{tab} Elementare

Otto mosse da cui scegliere, e per ciascuna un voto approssimativo: giusto *in
media*, ma sporcato ogni volta da un errore in più o in meno. Tu prendi sempre
il voto più alto. Il più alto degli otto, però, è di solito quello della mossa
a cui l'errore ha dato la spinta verso l'alto più grande, non quello della
mossa migliore. Fra otto misure sbagliate a caso, la più alta è quasi sempre
una misura fortunata.

[Prendere il massimo di stime rumorose, insomma, non restituisce il massimo dei
valori veri: restituisce qualcosa di sistematicamente più grande]{.pt-matita}.

Il conto si può anche fare. Supponiamo che le otto mosse valgano tutte $5$, e
che ogni voto sbagli di una quantità qualsiasi fra $-1$ e $+1$, in su come in
giù, senza preferenze. Fra otto errori pescati così, il più grande sta quasi
sempre vicino al bordo alto, e un ragionamento col righello ne dà la misura.

Otto punti buttati a caso su un segmento lo tagliano, in media, in nove pezzi
uguali, e il più alto lascia sopra di sé un pezzo solo, un nono del segmento.
Qui il segmento va da $-1$ a $+1$ ed è lungo $2$, quindi il più alto degli
otto errori sta in media a $2/9$ dal bordo, cioè a $1 - 2/9 = 7/9$, circa
$+0{,}78$, invece che a $0$. È la formula $(8-1)/(8+1)$. Il voto più alto
degli otto, in media, non vale dunque $5$: vale $5{,}78$.

In generale, quanto si gonfia dipende da due cose: da quante sono le mosse fra
cui si sceglie, e da quanto sono sballati i voti. Con due mosse sole, e gli
stessi errori di prima, la gonfiatura scende a $1/3$; con otto mosse ma errori
larghi il doppio, tutto raddoppia e sale a $1{,}56$. Il voto gonfiato diventa
il bersaglio dell'aggiornamento successivo, quindi la gonfiatura non resta
dov'era: si tramanda.

Il rimedio si chiama **Double DQN**, e divide in due un lavoro che prima faceva
una rete sola. Prima: la stessa rete decide qual è la mossa migliore *e* dice
quanto vale. Dopo: la rete che sta imparando dice quale mossa, e la rete-target
dice quanto vale quella mossa. Che i due ruoli stiano su reti
diverse è tutta la sostanza: l'errore che ha fatto *sembrare* buona quella
mossa non è lo stesso errore che poi ne *misura* il valore, così la fortuna non
viene contata due volte. I ruoli non sono intercambiabili, perché la rete-target
deve dire *quanto vale*, altrimenti si perde il bersaglio fermo che
serviva a non far esplodere l'addestramento.

Attenua, però, non guarisce. Le due reti non sono estranee fra loro: una è la
copia dell'altra di qualche passo prima, e quella parentela lascia passare buona
parte della gonfiatura. Quando le mosse non valgono tutte uguali, il
correttivo tende a esagerare dalla parte opposta, e i voti escono un filo
bassi invece che alti.

`````

`````{tab} Superiore

Le singole stime non sono distorte: a produrre la sovrastima del bersaglio è
l'incontro fra il rumore e la convessità del massimo, che la disuguaglianza di
Jensen descrive. Per stime $\hat Q$ non distorte,

$$
\mathbb{E}\big[\max_a \hat Q(s,a)\big] \;\ge\; \max_a \mathbb{E}\big[\hat Q(s,a)\big]
= \max_a Q(s,a),
$$

e il divario cresce con il numero di azioni e con la varianza dell'errore. In
particolare, la disuguaglianza è stretta ogni volta che il rumore può cambiare
quale azione risulti la migliore, e con stime esatte si ridurrebbe a
un'uguaglianza. [Basta quindi un errore di stima a media nulla perché il
bersaglio sia sistematicamente sovrastimato]{.pt-matita}, e il bootstrapping
propaga la sovrastima all'indietro.

Consideriamo il caso più semplice, che si calcola per intero. Se gli $m$
valori veri sono uguali
e gli errori $\epsilon_a$ sono indipendenti e uniformi in $[-1,1]$, il loro
massimo ha funzione di ripartizione $F(x) = \big((x+1)/2\big)^m$ su $[-1,1]$, e
quindi

$$
\mathbb{E}\Big[\max_a \epsilon_a\Big] = 1 - \int_{-1}^{1}
\Big(\frac{x+1}{2}\Big)^{m} dx = 1 - \frac{2}{m+1} = \frac{m-1}{m+1},
$$

che vale $1/3$ con due azioni e $7/9 \approx 0{,}78$ con otto, e raddoppia se
gli errori sono larghi il doppio. È il Teorema 2 di van Hasselt e colleghi,
dimostrato nell'appendice della versione estesa del lavoro
{cite}`vanhasselt2016deep`: la sovrastima cresce con il numero di azioni anche
a errori fissati.

Il **Double DQN** {cite}`vanhasselt2016deep` disaccoppia i due ruoli. Il
bersaglio di DQN usa $\theta^{-}$ sia per scegliere sia per valutare; quello di
Double DQN fa scegliere alla rete online $\theta$ e valutare alla rete
target $\theta^{-}$:

$$
y^{\text{Double}} = r + \gamma\, Q\Big(s',\;
\arg\max_{a'} Q(s', a'; \theta);\ \theta^{-}\Big).
$$

Rispetto a $y = r + \gamma \max_{a'} Q(s',a';\theta^{-})$, la differenza sta
tutta in quali parametri compaiono dentro l’$\arg\max$, e nel codice del
bersaglio di DQN cambia soltanto il calcolo di `q_next`:

```{code-block} python
:class: pt-non-eseguibile

with torch.no_grad():
    # la rete online sceglie l'azione, la rete target ne dice il valore
    a_star = online_net(s_next).argmax(dim=1, keepdim=True)
    q_next = target_net(s_next).gather(1, a_star).squeeze(1)
bersaglio = r + gamma * q_next * (1 - fine)
```

L'ordine dei due ruoli non è però scambiabile a piacere: far scegliere
a $\theta^{-}$ e valutare a $\theta$ conserverebbe il disaccoppiamento, e quindi
una parte della correzione, ma rimetterebbe i pesi in aggiornamento dentro il
bersaglio, buttando via il congelamento che era servito a stabilizzarlo.

Gli autori sono prudenti sul risultato: l'algoritmo «riduce le sovrastime
osservate», non le elimina. Il disaccoppiamento annullerebbe il bias solo con
due stimatori indipendenti, e qui il secondo non lo è: la rete-target è una
copia ritardata del primo, scelta perché è il candidato che c'era già, e il
lavoro stesso avverte che il disaccoppiamento non è completo.

Inoltre il correttivo non è neutro: quando i valori veri delle azioni non sono
tutti uguali, lo stimatore doppio tende a sostituire la sovrastima con una
lieve sottostima.

`````

Il calcolo delle otto mosse si rifà estraendo gli errori a caso, un milione di
volte per ciascun caso, e facendo la media del massimo:

```python
import numpy as np

rng = np.random.default_rng(0)
for mosse, ampiezza in ((8, 1), (2, 1), (8, 2)):
    # valori veri tutti uguali: cio' che il massimo trova in piu' e' errore
    errori = rng.uniform(-ampiezza, ampiezza, size=(1_000_000, mosse))
    gonfiatura = errori.max(axis=1).mean()
    formula = ampiezza * (mosse - 1) / (mosse + 1)
    print(f"{mosse} mosse, errori fra -{ampiezza} e +{ampiezza}: "
          f"il massimo gonfia di {gonfiatura:.2f} (formula {formula:.2f})")
```

```text
8 mosse, errori fra -1 e +1: il massimo gonfia di 0.78 (formula 0.78)
2 mosse, errori fra -1 e +1: il massimo gonfia di 0.33 (formula 0.33)
8 mosse, errori fra -2 e +2: il massimo gonfia di 1.56 (formula 1.56)
```

## Replay con priorità e dueling network

A partire da DQN la ricerca ha sviluppato un'intera famiglia di migliorie.
Due sono entrate nella pratica quasi quanto il Double DQN, e portano un'idea
ciascuna: una cambia quali esperienze si estraggono più spesso dalla memoria
di replay, l'altra come si scompone il valore di un'azione.

`````{tab} Elementare

Il quaderno degli appunti di prima si ripassa pescando a caso, e a caso vuol
dire che un'esperienza banale vale quanto una sorprendente. Uno studente vero
non fa così: ripassa più spesso le pagine dove l'ultimo compito è andato
peggio. È il **replay con priorità**: ogni esperienza porta un segnalibro
grande quanto l'errore che la rete ci ha fatto sopra l'ultima volta, e la pesca
premia i segnalibri grandi. Una manopola regola quanto premiarli: portata a
zero, si torna alla pesca a caso di prima. Le esperienze appena vissute entrano
col segnalibro al massimo, così nessuna finisce nel dimenticatoio prima di un
primo ripasso.

Chi ripassa quasi soltanto le pagine dove sbaglia, però, si fa un'idea storta
del libro intero, e il rimedio è contare i ripassi pescati apposta un po’ meno
di quelli che sarebbero usciti a caso.

L'altra idea spezza il voto in due domande: quanto è buona la situazione, e
quanto aggiunge ciascuna mossa. Su un rettilineo vuoto, ad esempio, guidare
bene non dipende dalla piccola correzione che dai al volante: conta che il
rettilineo è tranquillo, e lo sarà per chiunque. La rete **a due rami** impara
le due cose separatamente e le rimette insieme alla fine, sommandole.

Sommare, però, lascia una libertà di troppo: "la strada vale 10 e la sterzata
non aggiunge niente" e "la strada vale 7 e la sterzata aggiunge 3" fanno lo
stesso voto, e niente dice quale delle due divisioni sia quella buona. Serve
quindi un patto, e glielo si impone: i contributi delle mosse devono
compensarsi fra loro, tanto in su quanto in giù, e quello che avanza è il
giudizio sulla situazione.

Con i due rami separati, [quel giudizio si affina a ogni passaggio, anche
quando sulle singole mosse non c'è niente da imparare]{.pt-matita}. Così è già
pronto quando arriva la curva in cui le mosse tornano a contare.

`````

`````{tab} Superiore

Il **prioritized experience replay** {cite}`schaul2016prioritized` sostituisce
il campionamento uniforme dal buffer con

$$
P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}},
\qquad p_i = |\delta_i| + \epsilon,
$$

dove $\delta_i$ è l'ultimo errore TD misurato sulla transizione $i$, $\epsilon$
è una costante piccola che impedisce a un errore sceso a zero di escludere per
sempre quella transizione, e $\alpha \ge 0$ dosa l'effetto della priorità
($\alpha = 0$ riporta all'uniforme); le transizioni nuove entrano con priorità
massima.

Il campionamento non uniforme distorce però la distribuzione degli
aggiornamenti, e la correzione è un peso di *importance sampling* $w_i =
\big(N\, P(i)\big)^{-\beta}$, dove $N$ è il numero di transizioni in memoria;
il peso si normalizza sul massimo del minibatch, e $\beta$ viene portato verso
$1$ nel corso dell'addestramento, quando la correzione conta di più.

La **dueling network** {cite}`wang2016dueling` spezza la testa della rete in
due rami, il valore dello stato $V(s)$ e il vantaggio delle azioni $A(s,a)$,
e li ricompone in

$$
Q(s,a) \;=\; V(s) + \Big(A(s,a) - \tfrac{1}{|\mathcal{A}|}
\sum_{a'} A(s,a')\Big),
$$

dove $|\mathcal{A}|$ è il numero di azioni e la sottrazione della media rende
identificabile la scomposizione, perché senza di essa una costante potrebbe
passare da $V$ ad $A$ lasciando $Q$ identica.

Il guadagno è che [ogni aggiornamento allena $V$, qualunque azione contenga il
minibatch]{.pt-matita}. In particolare, negli stati in cui le azioni più o meno si
equivalgono, e in molti giochi sono tanti, la rete impara comunque qualcosa
che servirà altrove.

Con il Double DQN e altre tre migliorie (i ritorni a più passi, la stima di
un'intera distribuzione di ritorni al posto della media, l'esplorazione
tramite rumore nei pesi), questi due pezzi confluiscono in **Rainbow**
{cite}`hessel2018rainbow`; l'ablazione di quel lavoro indica il replay con
priorità e i ritorni a più passi come i componenti la cui rimozione penalizza di
più l'insieme.

`````

## I limiti

Oltre alla sovrastima del massimo, che il Double DQN attenua senza eliminarla,
restano tre limiti principali, che hanno guidato la ricerca successiva.

- Fame di dati. La versione di *Nature* usa $50$ milioni di passi dell'agente
  per gioco, circa $38$ giorni di gioco ininterrotto; il collaudatore umano con
  cui è confrontata si era allenato circa due ore per gioco.
- Le azioni devono essere poche e distinte (in gergo *discrete*, cioè
  contabili una per una, come le voci di un menu). Prendere il massimo vuol
  dire scorrerle tutte, e questo funziona con le poche scelte di un joystick,
  ma non quando lo spazio delle azioni diventa continuo. Lì servono i metodi a
  {doc}`gradiente di policy <policy-gradient>`, estesi poi al {doc}`controllo
  continuo <controllo-continuo>`.
- Ricompense rade. In certi giochi il punteggio arriva solo dopo lunghe
  sequenze di mosse esatte, e lì DQN sostanzialmente fallisce, perché senza
  segnale non c'è nulla da inseguire. Il caso più celebre, *Montezuma's
  Revenge*, apre la
  {doc}`sezione sull'esplorazione <esplorazione-e-ricompensa>`.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- DQN butta via lo schedario che aveva un cartellino per ogni schermata e
  ci mette una rete neurale: guarda i pixel e dice a colpo d'occhio quanto
  vale ciascuna mossa, anche su schermate mai viste prima.
- Per anni un'idea così è stata instabile, e per una ragione precisa, la triade
  fatale: una rete al posto della tabella, stime aggiornate a partire da altre
  stime, e una strategia imparata mentre se ne gioca un'altra. Con due
  qualunque dei tre l'instabilità si può evitare; con tutti e tre i valori
  possono crescere senza fermarsi. Non succede per
  forza (DQN li tiene tutti e tre), ma il pericolo c'è: il controesempio di
  Baird lo mostra su sette stati in cui non si guadagna mai nulla; il sistema
  saprebbe rappresentare alla perfezione la risposta giusta ("tutto vale
  zero"), e i numeri crescono lo stesso.
- Due accorgimenti lo rendono stabile: il quaderno degli appunti, che si
  chiama memoria di replay (ogni esperienza viene annotata e ripescata a
  caso, così l'agente mescola situazioni lontane invece di rileggere cento volte
  la stessa pagina) e la
  rete-target, che tiene fermo il bersaglio abbastanza a
  lungo perché lo si possa raggiungere. Nessuno dei tre ingredienti sparisce:
  la rete-target addolcisce il bootstrapping (correggere una stima
  guardandone un'altra), mentre il quaderno toglie un difetto che nella triade
  non c'era, il rileggere la stessa pagina; sull'off-policy,
  semmai, tira dall'altra parte, perché i ricordi vengono da partite giocate
  in un altro modo e più vecchie.
- Prendere sempre il voto più alto gonfia i voti: fra tante stime sporcate
  da un errore, la più alta è quasi sempre una stima fortunata, non la mossa
  migliore. Il Double DQN attenua il difetto facendo dire *quale mossa*
  alla rete che impara e *quanto vale* alla rete-target; non lo elimina,
  perché le due reti sono parenti strette.
- Due migliorie con un'idea ciascuna: si ripassa più spesso ciò che ha
  sorpreso (contando un po’ meno i ripassi pescati apposta, per non farsi
  un'idea storta), e si giudica la situazione separatamente dalle mosse,
  così si impara anche dove le mosse non contano.
- Il risultato storico del 2015: un solo programma, con le stesse manopole di
  regolazione, arriva al livello di un collaudatore umano professionista su
  molti giochi Atari partendo dai soli pixel. Restano i limiti: servono
  quantità enormi di partite, le mosse devono essere poche e distinte, e dove
  il punteggio arriva di rado l'agente resta senza nulla da inseguire.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- DQN sostituisce la tabella $Q$ con una rete neurale $Q(s,a;\theta)$ che
  mappa i pixel dello stato ai valori delle azioni.
- Era instabile per una ragione precisa, la triade fatale: approssimazione,
  bootstrapping e off-policy insieme possono far esplodere i valori. Con due
  soli dei tre l'instabilità si può evitare; con tutti e tre il pericolo c'è,
  e il controesempio di Baird lo mostra su sette stati con ricompense tutte
  nulle, dove la soluzione esatta è rappresentabile e i pesi divergono lo
  stesso. Non è una condanna: Gradient-TD ed Emphatic-TD li combinano con
  garanzie nel caso lineare.
- Due accorgimenti lo rendono stabile: l’experience replay (memoria di
  transizioni campionate a caso) e la rete-target (bersaglio congelato).
  Nessuno dei tre ingredienti sparisce, e uno solo viene attenuato: la
  rete-target smorza il bootstrapping; il replay decorrela i campioni, e presuppone
  l'off-policy invece di attenuarlo.
- Il $\max$ nel bersaglio sovrastima perché il rumore incontra una funzione
  convessa, non perché le stime siano distorte (Jensen; stretta solo se il
  rumore può cambiare quale azione risulta la migliore, non su tutte le stime
  aleatorie):
  $\mathbb{E}[\max_a \hat Q] \ge \max_a \mathbb{E}[\hat Q]$. Il Double DQN
  fa scegliere l'azione a $\theta$ e valutarla a $\theta^{-}$: *riduce* il bias,
  non lo annulla, perché i due stimatori non sono indipendenti.
- Il prioritized replay campiona con
  $P(i)\propto(|\delta_i|+\epsilon)^{\alpha}$ e corregge il bias con pesi di
  importance sampling; la dueling network ricompone $Q = V + (A - \bar A)$ e
  allena $V$ a ogni aggiornamento. Insieme a Double DQN e ad altre tre
  migliorie, queste due soluzioni confluiscono in Rainbow.
- Il risultato storico (Mnih et al., 2015): livello umano su molti giochi
  Atari partendo dai soli pixel. Restano limiti di efficienza, azioni discrete
  e ricompense rade.
```
`````
