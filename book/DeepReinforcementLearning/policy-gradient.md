# Metodi a gradiente di policy

Seul, marzo 2016. Alla trentasettesima mossa della seconda partita contro il
campione Lee Sedol, il programma AlphaGo appoggia una pietra in un punto che
nessun giocatore professionista avrebbe scelto: i commentatori pensano a un
errore. (Il Go si gioca appoggiando pietre bianche e nere sugli incroci di una
griglia, e vince chi circonda più territorio.) Era invece una mossa che, secondo
le stime del programma stesso, un umano avrebbe giocato circa una volta su
diecimila. Lee Sedol si alza dal tavolo per un quarto d'ora. Quella mossa non
era stata copiata da nessun archivio di partite: era il frutto di una
*strategia* appresa giocando milioni di volte contro se stesso.

Come si insegna a una macchina una strategia? Nei metodi basati sul valore,
che abbiamo incontrato con il Q-learning, impariamo a stimare *quanto vale una
situazione* e *quanto vale una mossa in quella situazione* (nel capitolo
precedente le due stime si chiamavano rispettivamente $V$ e $Q$), e poi ne
ricaviamo l'azione migliore scegliendo di volta in volta quella col valore più
alto. I metodi a **gradiente di policy** ribaltano la prospettiva: invece di
valutare e poi decidere, imparano *direttamente a decidere*. Il «gradiente» del
nome è il modo in cui in matematica si chiama una pendenza: la direzione lungo
cui una quantità cresce più in fretta, e quindi la direzione in cui conviene
fare un passo. Qui la quantità che si vuole far crescere è la ricompensa.

E a fare il passo sono i **pesi** di una rete neurale: i numeri, dentro la rete,
che decidono come una situazione si trasforma in una decisione. In tutta questa
sezione, quando si dice che «la strategia cambia» o che «si fa un passo», si
intende sempre questo: qualche milione di numeri che si sposta un pochino.

## Imparare la policy, non il valore

Perché conviene ribaltare così la prospettiva? Per due motivi, e il primo è
quello che regge mezzo capitolo: dare un voto a ogni mossa e poi prendere la
migliore funziona finché le mosse si possono contare, e smette di funzionare
quando la mossa è una quantità da dosare, come di quanto girare uno sterzo. Lì
non c'è più un elenco da scorrere. Il secondo motivo è di tutt'altro genere: una
strategia imparata così può *tirare i dadi*, cioè nella stessa situazione fare
a volte una cosa e a volte un'altra, e contro un avversario che ti studia essere
prevedibili è una condanna.

`````{tab} Elementare

Un cane si può allenare in due modi. Uno è compilare una tabella mentale che
assegna a ogni situazione un "punteggio di bontà" per ciascun comportamento
possibile, e poi far scegliere al cane il comportamento col punteggio più alto.
L'altro approccio, più diretto, è modellare la *tendenza* del cane: rendere più
probabili i comportamenti che in passato hanno fruttato un premio, meno
probabili quelli finiti male. Non calcoliamo un punteggio per poi decidere:
regoliamo direttamente le probabilità con cui l'animale sceglie.

Una **policy** è esattamente questo: data una situazione, dice con quale
probabilità compiere ciascuna azione. In italiano si direbbe «strategia», ed è
quello che vuol dire; il libro usa le due parole come sinonimi, perché *policy*
è il termine che si legge dappertutto e conviene averlo in mano.

`````

`````{tab} Superiore

Una policy parametrica $\pi_\theta(a \mid s)$ è una distribuzione di
probabilità sulle azioni $a$ condizionata allo stato $s$, controllata dai
parametri $\theta$ (i pesi di una rete neurale). L'obiettivo è massimizzare il
**ritorno atteso**:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[ R(\tau) \big],
\qquad R(\tau) = \sum_{t=0}^{T} \gamma^{t}\, r_t ,
$$

dove $\tau=(s_0,a_0,r_0,s_1,\dots)$ è una *traiettoria* generata seguendo la
policy, $r_t$ è la ricompensa al passo $t$ e $\gamma\in[0,1)$ è il fattore di
sconto, che pesa meno il futuro lontano. Rispetto ai metodi basati sul valore,
ottimizzare $\pi_\theta$ direttamente gestisce con naturalezza gli spazi di
azioni continui e le policy stocastiche.

Un avviso sulla notazione, perché da qui in avanti cambia. Questa è la
convenzione dei lavori di deep RL: $r_t$ è la ricompensa che **segue** l'azione
$a_t$, cioè esattamente ciò che il capitolo precedente indicava con $R_{t+1}$;
e stati e azioni si scrivono in minuscolo, perché la maiuscola $A_t$ qui serve
al *vantaggio*, come annunciato nella sezione sui bandit. Il pedice si sposta
di uno, la sostanza no: $G_t = \sum_{k\ge t}\gamma^{\,k-t} r_k$ e
$G_t = \sum_{k\ge 0}\gamma^{k} R_{t+1+k}$ sono lo stesso oggetto scritto in due
modi.

`````

## REINFORCE: premiare ciò che ha funzionato

Il primo algoritmo di questa famiglia, dovuto a Ronald Williams
{cite}`williams1992simple`, ha
un'idea tanto semplice da sembrare ingenua: gioca un'intera partita, guarda
com'è andata, e poi *aumenta la probabilità delle azioni che hanno portato a
ricompense alte*, diminuendo quella delle azioni seguite da ricompense basse.

`````{tab} Elementare

È il metodo "prova e ricorda". Il giocatore fa una partita intera, dall'inizio
alla fine. Se ha vinto, si dice: "qualunque cosa io abbia fatto, rendila più
probabile la prossima volta". Se ha perso, il contrario. Ripetuto migliaia di
volte, questo semplice riflesso spinge il comportamento verso le mosse buone
senza che nessuno debba mai spiegare *perché* siano buone.

Il difetto salta subito all'occhio, ed è il motivo per cui esiste tutto il
resto della sezione: il giudizio arriva **solo alla fine**, ed è uno solo per
tutta la partita. Se hai vinto, l'algoritmo rende più probabili anche le due o
tre mosse pessime che avevi fatto per strada; se hai perso, rende meno probabili
anche quelle buone. E due partite giocate con la stessa identica strategia
possono finire in modi opposti per puro caso, con la correzione che cambia
segno di conseguenza. Il risultato è un apprendimento che **balla**: va nella
direzione giusta in media, ma a strattoni, e ci mette moltissimo.

`````

`````{tab} Superiore

REINFORCE stima il gradiente di $J(\theta)$ tramite il *policy gradient
theorem* {cite}`sutton2000policy`:

$$
\nabla_\theta J(\theta) =
\mathbb{E}\Big[ \sum_{t=0}^{T} \gamma^{\,t}\,\nabla_\theta \log \pi_\theta(a_t
\mid s_t)\, G_t \Big],
$$

dove $G_t=\sum_{k\ge t}\gamma^{\,k-t} r_k$ è il ritorno osservato a partire dal
passo $t$.

La derivazione sta in tre passaggi, e vederli toglie al risultato l'aria di
magia: un gradiente che dipende dall'ambiente, calcolato senza conoscere
l'ambiente. Il primo è l'identità della log-derivata,
$\nabla_\theta p_\theta = p_\theta\, \nabla_\theta \log p_\theta$, con cui

$$
\nabla_\theta\, \mathbb{E}_{\tau \sim p_\theta}\big[R(\tau)\big]
= \mathbb{E}_{\tau \sim p_\theta}\big[\nabla_\theta \log p_\theta(\tau)\,
R(\tau)\big],
$$

dove $\tau$ è la traiettoria intera e $R(\tau)$ il suo ritorno. Il secondo è
che la probabilità della traiettoria si fattorizza,
$p_\theta(\tau) = p(s_0) \prod_t \pi_\theta(a_t \mid s_t)\,
P(s_{t+1} \mid s_t, a_t)$, e nel logaritmo i fattori dell'ambiente diventano
addendi che di $\theta$ non sanno niente: derivando spariscono, ed è il
passaggio in cui l'ambiente esce di scena,
$\nabla_\theta \log p_\theta(\tau) = \sum_t \nabla_\theta \log
\pi_\theta(a_t \mid s_t)$. Il terzo è la causalità: l'azione al passo $t$ non
può cambiare le ricompense già incassate, i prodotti con le ricompense passate
hanno aspettazione nulla, e a ciascun addendo resta agganciato solo il ritorno
da lì in avanti, $G_t$.

L'enunciato vale sotto le ipotesi consuete ($\pi_\theta$ differenziabile in
$\theta$, ritorni limitati, distribuzione stazionaria degli stati ben definita)
e sotto una in più, che conviene tenere a mente perché tornerà fra poche pagine:
il teorema è **on-policy**. Nella forma originale al posto di $G_t$ compare
$Q^\pi(s_t,a_t)$, e sostituirvi il ritorno osservato è lecito solo perché
$\mathbb{E}[G_t \mid s_t, a_t] = Q^\pi(s_t,a_t)$, il che richiede che il seguito
della traiettoria sia stato generato **dalla stessa policy che stiamo
derivando**. È esattamente l'ipotesi che PPO dovrà rattoppare con un rapporto di
importance sampling per riusare i dati della policy vecchia.

Il termine $\nabla_\theta \log \pi_\theta(a_t\mid s_t)$ indica *come*
ritoccare i parametri per rendere più probabile l'azione $a_t$; moltiplicandolo
per $G_t$, quel ritocco viene amplificato quando l'esito è stato buono e
invertito quando è stato cattivo. L'aggiornamento è quindi

$$
\theta \leftarrow \theta + \alpha\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t ,
$$

con $\alpha$ il passo di apprendimento. E qui c'è un'avvertenza di rigore, perché
l'aggiornamento appena scritto **non** è la formula del teorema: il $\gamma^{\,t}$
davanti a ciascun addendo è sparito. Ometterlo è la prassi, ed è la prassi che
seguiamo anche noi, ma va detto che cosa costa: la direzione che si ottiene è
leggermente distorta rispetto a $\nabla_\theta J(\theta)$, in cambio di non
soffocare il segnale dei passi lontani nel tempo, che con lo sconto esatto
peserebbero quasi nulla. Chi ha letto la sezione
sui bandit
riconosce la struttura: il *bandit a gradiente* era esattamente questo, in un
mondo con un solo stato, dove la softmax sulle preferenze $H(a)$ faceva le
veci di $\pi_\theta(a\mid s)$. Anche il rimedio che segue è già comparso là.

Il punto debole è la **varianza**:
$G_t$ dipende dall'intero seguito casuale della partita, e le stime risultano
rumorose e lente a convergere.

`````

## Actor-Critic: chi agisce e chi giudica

Come si smorza quell'altalena? L'idea è affiancare al giocatore un giudice
che commenta le mosse una per una, senza aspettare la fine
({numref}`fig-actor-critic`).

```{figure} ../figures/actor-critic.svg
:name: fig-actor-critic
:alt: Il riquadro Agente contiene Attore e Critico; il Critico passa il vantaggio all'Attore, che invia un'azione all'Ambiente, il quale restituisce stato successivo e ricompensa.
:width: 85%

L'architettura Actor-Critic. L'attore decide la mossa, il critico la giudica e
gli restituisce un segnale che dice di quanto è andata meglio (o peggio) del
previsto: nel disegno quel segnale si chiama *vantaggio*. L'ambiente risponde
con la situazione successiva e la ricompensa.
```

`````{tab} Elementare

L’**attore** è chi gioca: decide le mosse. Il **critico** è un allenatore a
bordo campo che, mossa dopo mossa, mormora "meglio del previsto" oppure "peggio
del previsto". L'attore non deve più aspettare la fine della partita per sapere
com'è andata: riceve un giudizio immediato a ogni passo e corregge subito la
rotta. Impara più in fretta e in modo più stabile.

Attore e critico non sono due persone, naturalmente: sono due reti neurali, che
si allenano insieme sulla stessa partita. Una impara a decidere, l'altra impara
a prevedere come andrà a finire.

`````

`````{tab} Superiore

Nell'architettura **Actor-Critic** convivono due reti. L’*attore* è la policy
$\pi_\theta(a\mid s)$; il *critico* stima la funzione valore $V_\phi(s)$. Al
ritorno grezzo $G_t$ si sostituisce il **vantaggio** (*advantage*), che è per
definizione

$$
A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s),
$$

cioè quanto quella singola azione vale più (o meno) della media delle azioni che
$\pi$ giocherebbe in quello stato. Nessuno dei due termini è noto, quindi in
pratica lo si *stima*, e la stima più economica è l'errore di differenza
temporale:

$$
\hat A_t = r_t + \gamma\, V_\phi(s_{t+1}) - V_\phi(s_t) .
$$

Da qui in avanti scriveremo $A_t$ per questa stima, com'è d'uso in deep RL, ma
vale la pena ricordare che è uno stimatore e non la definizione: coincide con
$A^\pi$ solo se il critico ha ragione, cioè se $V_\phi = V^\pi$.

$A_t$ misura di quanto l'azione compiuta ha superato le *aspettative*
codificate dal critico: è positivo se l'esito è stato migliore del previsto. La
regola diventa $\nabla_\theta \log\pi_\theta(a_t\mid s_t)\,A_t$, e la varianza
scende per due vie che conviene distinguere. Sottrarre la *baseline*
$V_\phi(s_t)$ **non distorce** il gradiente, e il conto sta in due righe:

$$
\mathbb{E}_{a\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a\mid s)\, b(s)\big]
= b(s)\sum_a \nabla_\theta \pi_\theta(a\mid s)
= b(s)\,\nabla_\theta\!\!\sum_a \pi_\theta(a\mid s)
= b(s)\,\nabla_\theta 1 = 0 .
$$

Il primo passaggio usa l’**identità della log-derivata**,
$\pi_\theta\,\nabla_\theta\log\pi_\theta = \nabla_\theta \pi_\theta$, che è la
stessa che fa comparire il $\log$ nel teorema di poco fa; il secondo scambia la
derivata con la somma. È scritto per azioni discrete, e nel continuo la somma
diventa un integrale e lo scambio va giustificato, ma la conclusione non cambia.
L'unica cosa che serve è che $b$ **non dipenda dall'azione**: qualunque
funzione del solo stato si può sottrarre gratis. Sostituire il ritorno $G_t$ con
$r_t+\gamma V_\phi(s_{t+1})$ è invece *bootstrapping*, e rende la stima
distorta finché il critico è impreciso. Si scambia varianza con *bias*: è il
compromesso al cuore dell'actor-critic.

Il vantaggio scritto sopra è la scelta più economica del compromesso, cioè
l'errore TD a **un passo**: poca varianza e parecchio bias. All'altro estremo
c'è il ritorno completo di REINFORCE, che è non distorto e ballerino. Fra i due
non c'è un salto ma una famiglia continua, governata da un parametro $\lambda$
che dice quanti passi guardare avanti prima di affidarsi al critico: detto
$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ l'errore TD di un
passo, la stima del vantaggio è

$$
\hat{A}_t \;=\; \sum_{l \ge 0} (\gamma\lambda)^l\, \delta_{t+l},
$$

che con $\lambda = 0$ si riduce al solo $\delta_t$ e con $\lambda = 1$ torna
al ritorno completo meno la baseline. È la
*generalized advantage estimation* {cite}`schulman2016high`, ed è quella che si
usa in pratica dentro PPO. Attore e critico si addestrano insieme: il critico
affina le sue stime, l'attore le usa come segnale.

`````

## A3C e PPO: gli algoritmi che funzionano davvero

Su REINFORCE e sull'attore-critico, messi insieme, poggiano i due algoritmi che
si usano davvero.

**A3C** {cite}`mnih2016asynchronous` fa giocare molte copie dell'agente
insieme, ciascuna la propria partita in una copia sua del gioco. Il guadagno è
lo stesso della memoria delle esperienze in DQN, ottenuto per un'altra strada.
Lì si rimescolavano esperienze pescate da momenti lontani fra loro; qui le
esperienze arrivano già diverse l'una dall'altra, perché nello stesso istante
ogni copia si trova in un punto diverso della sua partita. La rete non si
ritrova mai a correggersi dieci volte di fila su
situazioni quasi identiche, e l'addestramento balla di meno. Come effetto
collaterale, si usano tutti i processori della macchina invece di uno. La sigla
sta per *Asynchronous Advantage Actor-Critic*: attore-critico, con il
vantaggio, e in parallelo, cioè le tre cose appena dette.

**PPO** (*Proximal Policy Optimization* {cite}`schulman2017proximal`) è
l'algoritmo che oggi si prova per primo, e la ragione non è che sia il più
potente: è che **perdona la taratura**, cioè funziona ragionevolmente su una
gamma larga di problemi senza che qualcuno passi giorni a regolarne le manopole.

```{figure} ../figures/ppo-2017.svg
:name: fig-ppo-clipping
:alt: "Dalla policy vecchia, al centro, si dipartono due frecce. La prima è un passo breve che resta dentro una fascia consentita disegnata attorno al punto di partenza, e viene accettata. La seconda è un salto lungo che esce dalla fascia: l'aggiornamento viene tagliato al bordo, e oltre quel bordo non porta più alcun vantaggio."
:width: 84%

Il guinzaglio di PPO. Non impedisce di migliorare: toglie il premio a chi prova
a migliorare troppo in una volta, per tenere l'aggiornamento vicino alla
strategia che ha generato le partite. È un incentivo e non un divieto, e la
differenza conta: niente impedisce all'aggiornamento di uscire dalla fascia,
semplicemente uscirne non frutta più. La sezione ci torna sopra, perché il nome
promette più di quanto mantenga.
```

Il guinzaglio di {numref}`fig-ppo-clipping` esiste perché le esperienze
invecchiano in fretta. Le partite da cui l'agente sta imparando le ha giocate
con la strategia di prima, non con quella che sta diventando: dicono in che
direzione conviene muoversi, ma solo finché le due si somigliano ancora. Con un
passo troppo lungo quei ricordi finiscono per descrivere il comportamento di
qualcun altro.

E quanto si somigliano si può misurare, mossa per mossa. Si guarda che
probabilità le dava la strategia vecchia e che probabilità le dà la nuova, e si
fa il rapporto fra le due: se la mossa aveva il $10\%$ e adesso ha il $12\%$ il
rapporto vale $1{,}2$, se non è cambiato niente vale $1$. La fascia disegnata
nella figura va appunto da $0{,}8$ a $1{,}2$, cioè un quinto in meno e un quinto
in più.

E che succede fuori da lì? Ricordiamo che tutto questo mestiere consiste nel far
salire un numero, la ricompensa attesa, spostando i pesi della rete. Ecco: fuori
dalla fascia quel numero smette di salire per quanto ci si sposti. Il passo
successivo non è vietato, semplicemente non frutta più niente, e una rete che
cerca il guadagno non ha nessun motivo di farlo.

`````{tab} Elementare

Il rischio, quando aggiorni una strategia, è esagerare: un solo passo troppo
lungo può rovinare l'apprendimento di ore intere. PPO fa esattamente ciò che
suggerisce il nome (*proximal*, «vicino»): fa in modo che allontanarsi molto
dalla strategia attuale non convenga, così i passi vengono piccoli e prudenti.
Piccoli, ma tanti.

`````

`````{tab} Superiore

PPO massimizza un obiettivo "tosato" (*clipped*):

$$
L^{\text{CLIP}}(\theta) =
\mathbb{E}\big[\min\!\big(\rho_t A_t,\
\operatorname{clip}(\rho_t,\,1-\epsilon,\,1+\epsilon)\,A_t\big)\big],
$$

dove $\rho_t = \dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$
è il rapporto tra la nuova e la vecchia policy, e $\epsilon$ (tipicamente
$0{,}2$) fissa quanto le è concesso spostarsi. Quel rapporto non è un
espediente inventato qui: è il **rapporto di importance sampling** incontrato
nel capitolo precedente, quello che permette di valutare una policy con dati
generati da un'altra, troncato a un passo solo. E il suo difetto è lo stesso
già visto là: può assumere valori enormi e mandare in aria la stima. Il
*clipping* **toglie il premio**, campione per campione, a chi spinge $\rho_t$
fuori dall'intervallo $[1-\epsilon,1+\epsilon]$.

Ma lo toglie **da un lato solo**, e quale lato dipende dal segno del vantaggio:
è il dettaglio che il nome nasconde, e conviene fare il conto invece di fidarsi.
Se l'azione è andata meglio del previsto ($A_t>0$) il $\min$ morde a destra:
oltre $1+\epsilon$ l'obiettivo si appiattisce e il gradiente si annulla, mentre
sotto $1-\epsilon$ il gradiente resta pieno, perché rendere *meno* probabile
un'azione buona non è il rischio da cui ci si vuole difendere. Se l'azione è
andata peggio del previsto ($A_t<0$) i due lati si scambiano: il taglio scatta
sotto $1-\epsilon$, e sopra $1+\epsilon$ non scatta affatto. Con $\epsilon=0{,}2$,
$A_t=-1$ e $\rho_t=5$, cioè un rapporto più di quattro volte l'estremo
superiore della fascia ($1{,}2$), l'obiettivo vale $-5$ e la sua derivata rispetto a
$\rho_t$ vale $-1$: gradiente intero, niente tosatura. Metà dei campioni fuori
banda, insomma, non viene tosata affatto.

Conviene dire con precisione che cosa questo garantisce, perché è meno di quanto
il nome suggerisca. È un'euristica **del primo ordine e per campione**, non un
vincolo: niente impedisce a $\rho_t$ di finire fuori dall'intervallo, il
gradiente si annulla solo dove il campione è già stato tosato (e si è appena
visto quanto poco spesso sia), e PPO fa più
epoche di minibatch sugli stessi dati. Ancora prima: al primo passo di ogni
aggiornamento la nuova policy coincide con la vecchia, tutti i rapporti valgono
$1$ e **nulla è tosato**, quindi quel passo è esattamente quello dell'obiettivo
non vincolato. E misurato sul serio il recinto non tiene: Engstrom e colleghi
{cite}`engstrom2020implementation` verificano che **nessuno** dei tre algoritmi
che confrontano riesce a tenere i rapporti dentro l'intervallo
$[1-\epsilon,1+\epsilon]$, PPO compreso, che pure è addestrato con un obiettivo
che quei rapporti li tosa. Attenzione però a quale recinto si sta misurando:
questo è il vincolo sui rapporti, non la regione di fiducia in divergenza di
Kullback-Leibler della formula qui sotto, che TRPO impone davvero, quasi per
costruzione, e che gli stessi autori misurano e trovano rispettata.

Detto così il tosaggio sembra un trucco, e invece è l'approssimazione
economica di un'idea precisa che lo precede. **TRPO** (*Trust Region Policy Optimization*) {cite}`schulman2015trust` pone
il problema come massimizzazione **vincolata**: si
massimizza lo stesso obiettivo con importance sampling, ma imponendo che la
nuova policy resti vicina alla vecchia in **divergenza di Kullback-Leibler**,

$$
\max_\theta\ \mathbb{E}\big[\rho_t A_t\big]
\quad \text{soggetto a} \quad
\mathbb{E}\big[D_{\mathrm{KL}}(\pi_{\theta_{\text{old}}} \,\|\, \pi_\theta)\big]
\le \delta .
$$

dove $\delta$ è il raggio ammesso, cioè quanto la policy nuova può differire
dalla vecchia: è la manopola che in PPO fa l'$\epsilon$ del tosaggio, con la
differenza che qui è un vincolo e là un incentivo.

Il vincolo definisce una **regione di fiducia**, cioè l'intorno entro il quale
l'approssimazione lineare dell'obiettivo è ancora credibile. La ragione per cui
serve è quella che la sezione ha già raccontato con la metafora del
guinzaglio: una policy non è un modello supervisionato qualunque, perché
determina i dati che raccoglierà, e un passo troppo lungo non produce un errore
recuperabile ma una policy che smette di visitare gli stati utili.

Il prezzo di TRPO è computazionale: risolvere quel vincolo richiede
un'approssimazione del secondo ordine con la matrice di informazione di Fisher,
gestita con gradiente coniugato e ricerca di linea. Funziona, ed è pesante e
scomodo da implementare. PPO osserva che l'effetto che si vuole (non
allontanarsi troppo) si ottiene quasi tutto con un `min` e un `clip` dentro un
normale ottimizzatore del primo ordine.

Va aggiunto, per completezza, che a rigore l'obiettivo che si implementa non è
solo $L^{\text{CLIP}}$: gli si sommano la perdita del critico e un piccolo
**bonus di entropia**,

$$
L^{\text{CLIP}} - c_1 L^{\text{VF}} + c_2\, \mathcal{H}[\pi_\theta],
$$

dove $L^{\text{VF}}$ è l'errore quadratico del critico, $\mathcal{H}[\pi_\theta]$
è l'entropia media della policy (la stessa $\mathcal{H}$ che la sezione sul
controllo continuo definirà come $-\mathbb{E}_{a\sim\pi}[\log\pi(a\mid s)]$) e
$c_1, c_2$ sono due pesi fissi. Il terzo termine tiene la policy dal collassare
troppo presto su un'unica azione: è lo stesso ingrediente che diventerà il
segno distintivo di SAC, qui con un peso molto più piccolo e uno scopo più
modesto.

Sarebbe comodo chiudere dicendo che PPO ha vinto perché il tosaggio è
«abbastanza corretto», ma è una spiegazione data a posteriori che la letteratura
non regge, e vale la pena guardarla in faccia per due ragioni.

La prima è che il confronto con TRPO non è fra un'euristica e un teorema. Il
teorema di miglioramento monotono chiede il **massimo** della KL su tutti gli
stati e un coefficiente di penalità; la formula scritta qui sopra, quella che si
implementa, è già il suo rilassamento con la KL **media**. Anche TRPO, così
com'è usato, ha già rinunciato alla garanzia.

La seconda è che il vantaggio empirico di PPO su TRPO, misurato, viene in
larghissima parte da altro. Engstrom e colleghi
{cite}`engstrom2020implementation` isolano nove ottimizzazioni di
implementazione della versione di riferimento (normalizzazione e clipping delle
osservazioni, scalatura e clipping della ricompensa, clipping della value
function, inizializzazione ortogonale, annealing del passo di Adam, attivazioni
tanh, clipping globale del gradiente) e trovano che sono **loro** a rendere
conto della maggior parte del guadagno, non l'obiettivo tosato. Lo studio
indipendente di Andrychowicz e colleghi {cite}`andrychowicz2021what`, su più di
250.000 agenti addestrati, parte dalla stessa constatazione. È una lezione più
utile della prima: in deep RL la distanza fra l'algoritmo pubblicato e il codice
che lo esegue è spesso più larga della distanza fra due algoritmi.

`````

## Pensare prima di agire: la ricerca ad albero Monte Carlo

Finora la strategia ha sempre risposto d'istinto: la situazione entra da un lato
della rete, la mossa esce dall'altro, e in mezzo non c'è nessuna riflessione. Ma
un giocatore forte, prima di muovere, **pensa**: prova mentalmente qualche
continuazione, valuta dove porta, sceglie.

Quel pensare ha un algoritmo, e si chiama **ricerca ad albero Monte Carlo**
(MCTS, dalle iniziali inglesi; e «Monte Carlo», come al casinò, è il nome che
i matematici danno ai metodi che fanno i conti tirando a sorte). Torna in AlphaGo, in AlphaZero, in MuZero, e nei modelli linguistici che
esplorano più ragionamenti prima di rispondere: AlphaGo, AlphaZero, MuZero, e i modelli
linguistici (i programmi che scrivono testo, come quelli dietro agli
assistenti conversazionali) quando esplorano più ragionamenti prima di
rispondere. Conviene vederlo una volta per bene, anche perché è un vecchio
amico travestito.

Il capitolo sulla ricerca aveva lasciato la faccenda esattamente qui: la
ricerca classica, per fermarsi a metà albero, ha bisogno di una formula che
dia un voto alla posizione, e nel Go quella formula nessuno è mai riuscito a
scriverla. La via d'uscita era smettere di giudicare e mettersi a contare, cioè
giocare da lì un mucchio di partite a caso e guardare come finiscono. Quello
che segue è il seguito di quella frase.

`````{tab} Elementare

Il problema è che le continuazioni sono troppe. Agli scacchi, dopo tre mosse a
testa, i seguiti sono milioni; nel Go, molti di più. Esaminarle tutte è
impossibile, quindi bisogna guardare a fondo **solo dove conviene**. Ma per
sapere dove conviene bisognerebbe aver già guardato. È lo stesso dilemma delle
slot machine del capitolo precedente (i «bandit a più braccia», che si chiamano
così perché una slot machine è un bandito con una leva sola, e lì di leve ce ne
sono tante): tirare quella che finora ha pagato meglio, o provarne una di cui si
sa poco?

MCTS lo risolve costruendo un albero delle possibilità **a poco a poco**,
ripetendo migliaia di volte lo stesso giro di quattro mosse:

1. **Selezione.** Si scende dall'inizio seguendo, a ogni bivio, la mossa che
   ha il punteggio migliore, dove «migliore» mette insieme quanto ha reso
   finora e quanto poco è stata provata.
2. **Espansione.** Quando si arriva a un bivio con una mossa mai tentata, si
   aggiunge quel ramo all'albero.
3. **Simulazione.** Da lì si tira dritto fino alla fine della partita, in fretta
   e alla buona (nella versione originale, a caso), solo per farsi un'idea
   grezza di come va a finire.
4. **Risalita.** Il risultato torna indietro lungo la strada percorsa, e ogni
   nodo attraversato aggiorna la propria media e il proprio conteggio.

Il bello è che l'albero cresce **storto, e di proposito**: profondissimo sulle
linee promettenti, largo appena un dito su quelle che non convincono. Nessuno
gli ha detto quali fossero: lo ha scoperto giocandoci.

E la mossa da fare, alla fine, non è quella con la media migliore: è quella
**più visitata**. Sembra strano, ed è più solido: una media alta può venire da
due prove fortunate, mentre un ramo visitato mille volte ha resistito a mille
occasioni di essere abbandonato.

`````

`````{tab} Superiore

La formulazione standard è **UCT** (*Upper Confidence bounds applied to
Trees*), di Kocsis e Szepesvári {cite}`kocsis2006bandit`, costruita sopra il
framework di ricerca di Coulom {cite}`coulom2006efficient`. L'idea è di
trattare **ogni nodo come un bandit indipendente** sulle sue mosse, e in fase
di selezione scegliere

$$
a^\star = \arg\max_a \left[\, Q(s,a) + c \sqrt{\frac{\ln N(s)}{N(s,a)}}
\,\right],
$$

dove $N(s)$ è il numero di visite al nodo, $N(s,a)$ quelle al figlio, e
$Q(s,a) = W(s,a)/N(s,a)$ la media dei ritorni osservati passando di lì. È
**letteralmente UCB1**, la formula della sezione sui bandit, applicata a ogni
bivio: stesso ottimismo di fronte all'incertezza, stesso decadimento
logaritmico. Il contributo di UCT è mostrare che applicandola ricorsivamente
la stima alla radice converge a quella minimax, con garanzie **asintotiche**
sull'errore di campionamento. Asintotiche va preso alla lettera: sono garanzie
sul limite, non sul caso peggiore. Coquelin e Munos
{cite}`coquelin2007bandit` mostrarono l'anno dopo che l'ottimismo di UCT può
costare, in alberi profondi e ostili, un numero di simulazioni proibitivo prima
che la ricerca trovi il ramo buono, e proposero una variante con intervallo di
confidenza che cresce esponenzialmente con la profondità: il prezzo di una
garanzia vera. In pratica funziona; in teoria funziona alla lunga.

La risalita aggiorna $N$ e $W$ lungo il cammino; nei giochi a due giocatori il
ritorno si alterna di segno a ogni livello, perché ciò che è buono per me è
cattivo per l'avversario.

**AlphaGo e i suoi successori cambiano due dei quattro passi**, ed è lì che
entrano le reti. Il termine di esplorazione diventa **PUCT**, pesato da una
probabilità a priori fornita dalla rete di policy,

$$
U(s,a) = c_{\text{puct}}\, P(s,a)\,
\frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)},
$$

così che la ricerca guardi per prime le mosse che la rete considera plausibili
invece di trattarle tutte alla pari.

Il secondo cambiamento riguarda la valutazione della foglia, e avviene in **due
tappe**, che vale la pena non confondere. AlphaGo (2016) non butta via la
simulazione casuale: le **affianca** la rete di valore e media i due giudizi in
parti uguali,

$$
V(s_L) = (1-\lambda)\, v_\theta(s_L) + \lambda\, z_L ,
$$

dove $z_L$ è l'esito di un rollout giocato fino in fondo con una policy veloce e
$\lambda = 0{,}5$ (è il simbolo del paper, e non ha niente a che vedere con il
$\lambda$ della *generalized advantage estimation* di poche pagine fa: qui è
soltanto il peso con cui si mescolano due giudizi). Rete di valore
e partita giocata a caso pesano quindi identico, che è un modo educato per dire
che nel 2016 della rete non ci si fidava ancora abbastanza. La simulazione
casuale sparisce del tutto solo con **AlphaGo Zero** (2017), dove la rete di
valore basta da sola: è la stessa tappa in cui spariscono le partite umane, e
non è una coincidenza, perché entrambe le cose diventano superflue quando la
rete è abbastanza buona da giudicare da sé.

Il risultato è quello che rende possibile il ciclo di *self-play*: **la ricerca
gioca meglio delle reti che la guidano**. La distribuzione delle visite alla
radice, normalizzata, è una policy migliorata rispetto a $P(s,\cdot)$, e
diventa il bersaglio su cui la rete si addestra. MCTS, in questa lettura, è un
**operatore di miglioramento della policy**: lo stesso ruolo che nella
programmazione dinamica ha il passo di *policy improvement*, ottenuto con la
ricerca invece che con un massimo esatto.

`````

L'idea è più generale del gioco da tavolo, ed è il motivo per cui conviene
averla in tasca: **quando si può simulare, si può pensare**. Il programma
MuZero, per esempio, la usa senza nemmeno conoscere le regole del gioco: se le
costruisce da solo, guardando le partite. E il modello che si costruisce non
ridisegna la scacchiera pezzo per pezzo, ne tiene solo un riassunto interno, il
minimo che serve per pianificarci dentro. La sezione sul RL basato su modello ci
torna sopra. Anche i modelli linguistici che esplorano più catene di
ragionamento prima di rispondere fanno, con altri nomi, la stessa cosa.

### In pratica: le visite si concentrano

Che l'albero cresca storto non è un modo di dire, ed è la cosa più facile da
verificare. Prendiamo un albero giocattolo: due strade a ogni bivio e quattro
bivi in fila, cioè $2\times2\times2\times2 = 16$ finali possibili, ognuno con il
suo valore. Il migliore lo mettiamo noi, nascosto in mezzo agli altri: la
risposta giusta la sappiamo, l'algoritmo no, e il gioco è vedere se ci arriva.
Due parole di gergo, che tornano nel codice e nei risultati: la **radice** è il
punto di partenza dell'albero, le **foglie** sono le sue punte, cioè i sedici
finali.

Una precauzione, prima di leggere i numeri, e vale per tutto il resto del
capitolo. Se lancio un dado una volta e fa sei, non posso dire che quel dado fa
sempre sei: ho misurato quel lancio, non il dado. Lo stesso vale per un
algoritmo che a ogni passo tira a sorte. Quindi la ricerca qui sotto si lancia
**sessanta volte**, cambiando ogni volta il *seme*, cioè il numero da cui parte
il sorteggio (dentro un computer il caso non è vero caso: è una sequenza
calcolata, che dipende tutta da quel numero iniziale, e cambiarlo è il modo di
rifare l'esperimento daccapo). Di ciò che ne esce non si guarda un risultato: si
guardano il valore di mezzo (la **mediana**: la metà delle sessanta prove sta
sotto, l'altra metà sopra) e gli estremi.

```python
import math
import numpy as np

# Un albero giocattolo: profondità 4, due mosse per nodo, 16 foglie.
# I valori delle foglie li conosciamo, così sappiamo qual è la risposta giusta.
PROFONDITA, RAMI = 4, 2
PRIME = (RAMI ** PROFONDITA - 1) // (RAMI - 1)   # indice della prima foglia

def figli(nodo):
    return [nodo * RAMI + 1 + k for k in range(RAMI)]

def foglia(nodo):
    return nodo >= PRIME

def cerca(seme, giri=2000):
    """Una partita di MCTS completa, con il proprio seme casuale."""
    rng = np.random.default_rng(seme)
    valori_foglie = rng.uniform(0, 1, RAMI ** PROFONDITA)
    valori_foglie[6] = 0.98                  # la foglia buona, nascosta in mezzo
    migliore = int(valori_foglie.argmax())
    N, W = {0: 0}, {0: 0}                    # visite e somma dei ritorni

    def uct(nodo, c=1.4):
        """UCB1 su un bivio dell'albero: è la formula della sezione bandit."""
        padre = N[nodo]
        def punteggio(f):
            if N.get(f, 0) == 0:
                return float("inf")          # mai provato: massimamente urgente
            return W[f] / N[f] + c * math.sqrt(math.log(padre) / N[f])
        return max(figli(nodo), key=punteggio)

    def simula(nodo):
        """Discesa a caso fino a una foglia: la stima grezza di questo nodo."""
        while not foglia(nodo):
            nodo = int(rng.choice(figli(nodo)))
        return valori_foglie[nodo - PRIME]

    for _ in range(giri):
        nodo, cammino = 0, [0]
        while not foglia(nodo) and all(N.get(f, 0) > 0 for f in figli(nodo)):
            nodo = uct(nodo)                                      # 1. SELEZIONE
            cammino.append(nodo)
        if not foglia(nodo):
            nodo = next(f for f in figli(nodo) if N.get(f, 0) == 0)  # 2. ESPANSIONE
            cammino.append(nodo)
            N[nodo] = W[nodo] = 0
        ritorno = simula(nodo)                                    # 3. SIMULAZIONE
        for n in cammino:                                         # 4. RISALITA
            N[n] += 1
            W[n] += ritorno

    visite = np.array([N.get(PRIME + i, 0) for i in range(RAMI ** PROFONDITA)])
    n = PRIME + migliore                     # risalgo dalla foglia buona
    while (n - 1) // RAMI != 0:              # fino al ramo che parte dalla radice
        n = (n - 1) // RAMI
    return {
        "rami": [N[f] for f in figli(0)],
        "quota": visite[migliore] / visite.sum(),
        "ramo_buono": N[n],
        "azzecca": N[n] == max(N[f] for f in figli(0)),   # la mossa più visitata
    }

e = cerca(seme=7)
print(f"seme 7 | visite ai due rami dalla radice: {e['rami']}")
print(f"seme 7 | quota delle visite sulla foglia migliore: {e['quota']:.1%}")
print(f"         tirando a caso sarebbe stata: {1 / RAMI ** PROFONDITA:.1%}")

# Lo stesso esperimento su sessanta semi: quanto è rappresentativo quel numero?
prove = [cerca(s) for s in range(60)]
quote = np.array([p["quota"] for p in prove])
buoni = np.array([p["ramo_buono"] for p in prove])
print(f"\n60 semi | quota sulla foglia migliore: mediana {np.median(quote):.1%}, "
      f"da {quote.min():.1%} a {quote.max():.1%}")
print(f"          metà centrale fra {np.percentile(quote, 25):.1%} "
      f"e {np.percentile(quote, 75):.1%}")
print(f"          visite al ramo giusto: mediana {np.median(buoni):.0f}, "
      f"da {buoni.min()} a {buoni.max()}")
print(f"          la mossa più visitata è il ramo sbagliato in "
      f"{sum(not p['azzecca'] for p in prove)} semi su 60")
```

Sul seme $7$, quello dell'esempio, il ramo che porta alla foglia buona riceve
**1922 visite su 2000** e l'altro $78$: dopo poche decine di prove la ricerca ha
smesso di sprecare tempo di là. In fondo all'albero, il $58\%$ di tutte le
visite finisce sulla foglia migliore, contro il $6{,}2\%$ che le toccherebbe
tirando a caso, cioè una foglia su sedici.

Su sessanta semi, però, quella quota ha **mediana $50{,}5\%$** e oscilla fra il
$16\%$ e l’$89\%$; scartando le quindici prove più basse e le quindici più alte,
le trenta di mezzo stanno fra il $33\%$ e il $60\%$. Le visite al ramo giusto
hanno mediana $1582$ e vanno da $593$ a $1965$, cioè il $1922$ del seme $7$ è
vicino al massimo osservato. Peggio, in **nove semi su sessanta** il ramo più
visitato alla radice non è quello che contiene la foglia migliore: la regola
«si gioca la mossa più visitata», che poco fa abbiamo presentato come la scelta
più solida, in quei casi sbaglia. Nove su sessanta è quasi una volta su sei:
abbastanza da ricordarsi che è una regola pratica e non un teorema, e che qui i
giri di ricerca sono duemila mentre in una partita vera sono molti di più.

Nessuna di queste cifre smentisce il punto della sezione, e proprio per questo
si possono riportare senza imbarazzo. La quota oscilla, la forma no: l'albero
cresce storto su tutti e sessanta i semi, e mai una volta le visite si
distribuiscono uniformemente. Ma se avessimo tenuto il solo numero del seme $7$,
con la sua bella cifra decimale, l'algoritmo sarebbe sembrato più preciso di
quanto sia, e la regola della mossa più visitata più sicura di quanto sia. È il
motivo per cui in questo campo i risultati si riportano su molte ripetizioni, e
non su una.

## Da AlphaGo ad AlphaZero

Torniamo alla mossa 37. AlphaGo {cite}`silver2016mastering` non era un solo
algoritmo, ma una sintesi: una rete di policy che proponeva mosse promettenti,
una rete di valore che stimava chi fosse in vantaggio, e la ricerca ad albero
Monte Carlo appena vista, che usava entrambe per esplorare in profondità solo
le linee più sensate.

Con una prudenza che oggi fa sorridere. Per giudicare una posizione raggiunta in
fondo alla ricerca, AlphaGo non si affidava soltanto alla rete di valore: ne
faceva la media, mezzo e mezzo, con l'esito di una partita tirata avanti alla
svelta e quasi a caso fino alla fine. Nel 2016 della rete non ci si fidava
ancora abbastanza; un anno dopo basterà da sola.

```{figure} ../figures/alphago-2016.svg
:name: fig-alphago
:alt: "In alto, staccato e in tratteggio, il punto di partenza del 2016: le partite umane su cui la rete di policy viene addestrata all'inizio, con l'annotazione che è il passo che AlphaGo Zero eliminerà. Sotto, il ciclo chiuso: il self-play genera partite che il sistema gioca contro sé stesso; dalle partite si affinano due reti, quella di policy che propone le mosse e quella di valore che stima chi sta vincendo; le due reti guidano a loro volta una ricerca ad albero Monte Carlo, che gioca meglio di entrambe e produce le partite del giro successivo."
:width: 92%

Il giro che si alimenta da solo, e il gradino da cui il giro parte. Nel 2016
quel gradino sono ancora le partite umane; è il pezzo tratteggiato, ed è il
primo che i successori toglieranno.
```

Il ciclo di {numref}`fig-alphago` è il motivo per cui i successori di AlphaGo
poterono fare a meno delle partite umane, e poggia su un fatto da enunciare da
solo: **la ricerca gioca meglio delle due reti che la guidano**. Se ci si
pensa è quasi ovvio. La rete propone di getto, guardando la posizione; la
ricerca, prima di decidere, prova per davvero migliaia di continuazioni.
Quindi la mossa che esce dalla ricerca è quasi sempre migliore di quella che
la rete avrebbe scelto da sola, ed è un esempio su cui la rete può allenarsi.

Ecco la fonte di supervisione interna: non serve un maestro, basta giocare
contro sé stessi e imparare da dove la ricerca ha portato. Nel 2016 AlphaGo
questo giro lo faceva solo a metà: le sue due reti erano state prima addestrate
su partite umane, e solo dopo affinate giocando contro se stesse. Quel gradino
umano è il pezzo tratteggiato della figura, ed è il primo che cadrà.

Un anno dopo, **AlphaGo Zero** {cite}`silver2017mastering` elimina persino le
partite umane: parte dalle sole regole del Go e impara *tabula rasa*, dal
nulla, soltanto affrontando copie di sé, fino a battere nettamente la versione
che aveva sconfitto Lee Sedol. Nel 2018 **AlphaZero** {cite}`silver2018general`
generalizza la ricetta: lo stesso programma, senza ritocchi per gioco,
padroneggia Go, scacchi e shogi (gli scacchi giapponesi), e in tutti e tre batte
il programma più forte del momento; nel Go, quel programma è il suo stesso
predecessore. È la dimostrazione più limpida di cosa nasce dall'unione di
apprendimento per rinforzo, ricerca ad albero e reti profonde.

## Un ultimo salto: allineare i modelli linguistici

Lo stesso meccanismo (aumentare la probabilità di ciò che riceve un giudizio
positivo) è oggi al cuore dell'addestramento dei modelli linguistici, i
programmi che stanno dietro agli assistenti conversazionali.

**Allineare** un modello vuol dire portarlo a fare ciò che chi lo interroga
intende davvero. Non è scontato, perché un modello linguistico nasce sapendo
fare una cosa sola: indovinare come prosegue un testo. A «spiegami perché il
cielo è azzurro» un continuatore di testi può rispondere benissimo con un'altra
domanda, o con l'indice di un libro di fisica: sono continuazioni plausibili, e
non sono la risposta che si voleva. L'allineamento serve a chiudere quella
distanza.

```{figure} ../figures/instructgpt-2022.svg
:name: fig-instructgpt
:alt: "Il giro dell'RLHF, in tre riquadri: alcune persone ordinano per preferenza più risposte allo stesso prompt; da questi ordinamenti si addestra un modello di ricompensa (reward model) che impara ad assegnare punteggi; il modello di ricompensa guida infine l'ottimizzazione della policy del modello linguistico, che genera, viene valutata e aggiornata."
:width: 100%

Il giudizio umano che diventa un numero. Le persone non danno voti: mettono in
fila delle risposte, ed è il *reward model* (il modello di ricompensa) a
tradurre quell'ordine in un punteggio che l'ottimizzazione sa usare.
```

Il dettaglio di {numref}`fig-instructgpt` da notare è il primo riquadro: alle
persone si chiede di **ordinare**, non di valutare. Confrontare due risposte è
un giudizio che gli esseri umani danno con buona coerenza fra loro; assegnare
un voto da uno a dieci molto meno, e su scale diverse. Nell’**RLHF**
(*Reinforcement Learning from Human Feedback*, cioè apprendimento per rinforzo
dal giudizio umano; {cite}`christiano2017deep`, {cite}`ouyang2022training`) le
risposte del modello sono l’"azione", dei valutatori umani indicano quali
preferiscono, e le loro preferenze addestrano un *modello di ricompensa* che
fa da critico. Con PPO si ritocca poi la policy del modello (la sua tendenza a
produrre certe risposte), verso ciò che gli umani apprezzano. La stessa idea
che ha portato una macchina a giocare la mossa 37 aiuta oggi un assistente a
rispondere in modo utile e onesto.

Il disegno comincia dagli ordinamenti, ma prima c'è un passo che non si vede:
il modello viene addestrato a imitare risposte scritte da persone, cioè a
copiare quello che avrebbe fatto qualcuno di bravo. Si chiama **clonazione
comportamentale**, ed è la scorciatoia più ovvia di tutte: la sezione
sull'imitazione ci torna sopra per esteso, e spiega perché da sola non basta.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- I metodi a gradiente di policy imparano **direttamente a decidere**: invece
  di dare un voto a ogni mossa e poi scegliere la migliore, regolano la
  *tendenza* dell'agente, come si allena un cane rendendo più probabili i
  comportamenti che hanno fruttato un premio. È la via naturale quando le
  mosse possibili non sono un menu di poche voci.
- **REINFORCE** è il "prova e ricorda": si gioca una partita intera e, se è
  andata bene, si rende più probabile tutto ciò che si è fatto. Semplice, ma
  lento e altalenante, perché il giudizio arriva solo alla fine.
- **Actor-Critic** affianca al giocatore un allenatore a bordo campo che
  commenta ogni mossa ("meglio del previsto", "peggio del previsto"):
  l'apprendimento diventa più rapido e più stabile. **A3C** fa giocare molti
  attori in parallelo; **PPO** cambia la strategia solo di poco per volta
  (passi piccoli e prudenti, ma tanti) ed è quello che si prova per primo,
  perché perdona gli errori di taratura più degli altri.
- La **ricerca ad albero Monte Carlo** è il "pensare prima di muovere":
  migliaia di volte si scende nell'albero delle possibilità scegliendo dove
  conviene, si prova un ramo nuovo, si tira fino alla fine e si riporta
  indietro il risultato. L'albero cresce **storto di proposito**, profondo
  dove promette e appena accennato altrove, e la mossa scelta è la **più
  visitata**, non quella con la media più alta. È però una regola pratica, non un
  teorema: su sessanta ripetizioni dell'esperimento l'albero cresce storto
  sempre, ma *quanto* storto cambia parecchio, e in nove casi su sessanta il
  ramo più visitato è quello sbagliato. È il motivo per cui i risultati si
  contano su molte prove e non su una.
- **AlphaGo** e **AlphaZero** uniscono la strategia, la stima di chi sta
  vincendo e quella esplorazione ad albero. Nel 2016 la ricerca si fidava a
  metà della rete e a metà delle partite tirate a caso; solo con AlphaGo Zero,
  l'anno dopo, la rete basta da sola. Con l’**RLHF** lo stesso meccanismo,
  guidato dalle preferenze delle persone, allinea i modelli linguistici.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- I metodi a gradiente di policy apprendono **direttamente**
  $\pi_\theta(a\mid s)$: adatti ad azioni continue e strategie stocastiche.
- **REINFORCE** aumenta la probabilità delle azioni seguite da ritorni alti,
  ma soffre di varianza elevata.
- **Actor-Critic** aggiunge un critico $V_\phi(s)$ che fornisce il *vantaggio*
  $A_t$, riducendo la varianza: la baseline non distorce (basta che non dipenda
  dall'azione), il bootstrapping sì. **A3C** parallelizza gli attori; **PPO**
  scoraggia i passi lunghi con il *clipping*, che è un'euristica del primo
  ordine e **non** un vincolo di trust region; il suo vantaggio misurato su
  TRPO viene in larga parte dalle ottimizzazioni di implementazione
  {cite}`engstrom2020implementation`.
- **MCTS/UCT** applica UCB1 a ogni nodo dell'albero,
  $Q(s,a) + c\sqrt{\ln N(s)/N(s,a)}$, e alterna selezione, espansione,
  simulazione e risalita; le garanzie sono asintotiche, non sul caso peggiore.
  AlphaGo sostituisce il termine di esplorazione con **PUCT**, pesato dalla
  policy a priori, e *media* rete di valore e rollout ($\lambda=0{,}5$); la
  simulazione casuale sparisce solo con AlphaGo Zero. La distribuzione delle
  visite alla radice è una **policy migliorata**: è l'operatore che rende
  possibile il *self-play*.
- **AlphaGo/AlphaZero** uniscono policy, valore e ricerca ad albero; **RLHF**
  applica PPO all'allineamento degli LLM.
```
`````
