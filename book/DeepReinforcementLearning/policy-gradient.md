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
quello che vuol dire; *policy* è la parola che si legge dappertutto, e le due
valgono l'una per l'altra.

Il premio, però, quasi mai arriva subito. In una seduta di addestramento il cane
fa una decina di cose di fila e il biscotto compare in fondo: quello che si
vuole far crescere è il bottino di tutta la seduta. Un biscotto che arriva fra
dieci mosse conta meno di uno che arriva adesso, perché nel frattempo può
succedere di tutto, e quanto meno conta lo decidiamo noi. Poi una seduta va bene
e la successiva male, con lo stesso cane e la stessa tendenza: il conto che
interessa è la media su tante sedute, mai su una sola.

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

Il difetto salta subito all'occhio, ed è quello da cui nasce tutto il seguito:
il giudizio arriva **solo alla fine**, ed è uno solo per tutta la partita. Se
hai vinto, l'algoritmo rende più probabili anche le due o
tre mosse pessime che avevi fatto per strada; se hai perso, rende meno probabili
anche quelle buone. E due partite giocate con la stessa identica strategia
possono finire in modi opposti per puro caso, con la correzione che cambia
segno di conseguenza. Il risultato è un apprendimento che **balla**: va nella
direzione giusta in media, ma a strattoni, e ci mette moltissimo.

Un primo ritocco costa niente. Quando i punti si segnano per strada, e non solo
alla fine, una mossa non può cambiare quelli già segnati prima di lei: la si
giudica soltanto su quello che viene dopo. La mossa dell'ultimo minuto risponde
dell'ultimo minuto, e la spinta che riceve è tanto più forte quanto più alto è
il bottino che l'ha seguita.

C'è poi una condizione che regge tutto il resto: le partite da cui si impara
devono essere state giocate con la strategia che si sta correggendo. Chi studia
le registrazioni di come giocava l'anno scorso sta correggendo il giocatore
dell'anno scorso, e più cambia, meno quelle registrazioni parlano di lui. Finché
si gioca una partita, si guarda com'è andata e si corregge, la condizione è
rispettata senza doverci pensare; diventa un problema appena si vuole spremere
la stessa partita più volte.

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
l'aggiornamento appena scritto non è la formula del teorema: il $\gamma^{\,t}$
davanti a ciascun addendo è sparito. Ometterlo è la prassi, ed è la prassi che
seguiamo anche noi, e costa: la direzione che si ottiene è
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

Quel «previsto» l'allenatore lo fissa guardando la situazione, prima che la
mossa sia giocata: «da qui, di solito, si porta a casa un pareggio». È questo
che rende onesto il commento che verrà dopo. Un allenatore che decidesse quanto era
difficile *dopo* aver visto la mossa, e alzasse l'asticella solo davanti alle
mosse che non gli piacciono, insegnerebbe al giocatore i propri gusti invece del
gioco.

E l'allenatore si può sbagliare. Il suo «meglio del previsto» vale quanto
valgono le sue previsioni, e all'inizio non ne sa più del giocatore: sono
giudizi affrettati presi per buoni. Il verdetto di fine partita, quello, non
sbagliava mai; era soltanto rumoroso, perché una partita sola dice poco. Si
scambia una cosa con l'altra, commenti immediati in cambio del rischio che siano
storti, ed è lo scambio su cui si regge il metodo.

Fra i due estremi c'è una manopola, e la si gira dove si vuole: quanto aspetta
l'allenatore prima di parlare. Una mossa sola, e parla subito fidandosi tutto
del proprio fiuto. Dieci mosse, e si fida un po’ meno, perché nel frattempo ha
visto succedere delle cose. Fino alla fine della partita, che è tornare al prova
e ricorda.

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
resta uno stimatore e non la definizione: coincide con
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
lungo può rovinare l'apprendimento di ore intere. Il nome di PPO promette la
cura (*proximal* vuol dire «vicino»): tenere la strategia nuova a poca distanza
da quella che ha giocato le partite.

Il freno, però, agisce da un lato solo, e quale lato dipende da com'è andata la
mossa. Su una mossa che era andata bene, il guadagno si ferma quando la si è
resa molto più probabile di prima: è lì che si rischia di strafare. Se invece
quella stessa mossa buona è diventata molto meno probabile, il premio a
rimetterla su resta intero, perché quello non è il pericolo. Sulle mosse andate
male i due lati si scambiano. In due di quei quattro casi, insomma, il
freno non frena niente, e «vicino» resta il proposito: quanto le due strategie
si somiglino davvero, va misurato.

Vietare sul serio i passi lunghi si può, ed è la strada più vecchia, quella che
PPO ha semplificato: si misura di quanto la strategia nuova differisce dalla
vecchia e, se la differenza supera una soglia, il passo si accorcia finché
rientra. Quel modo la promessa la mantiene, e costa: la misura e l'accorciamento
vanno rifatti a ogni passo, ed è parecchia macchina in più da costruire e da far
girare. PPO ottiene quasi lo stesso effetto con due righe dentro un allenamento
normale.

Al numero che si fa salire si aggiunge poi un piccolo premio per chi non si
riduce a giocare sempre la stessa mossa: serve a non far irrigidire la strategia
su un'unica risposta prima di aver visto abbastanza.

L'ultima cosa è la meno elegante. Quanto PPO vada meglio del suo predecessore,
misurato, dipende poco dal guinzaglio: viene soprattutto da nove accorgimenti su
come i numeri vengono scalati, tagliati e avviati dentro il programma. Fra
l'algoritmo raccontato su una pagina e il programma che lo esegue c'è spesso più
distanza che fra due algoritmi diversi.

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
$0{,}2$) fissa la larghezza della fascia entro cui lo spostamento continua a
fruttare. Quel rapporto non è un
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
$\rho_t$ vale $-1$: gradiente intero, niente tosatura. Dei quattro casi fuori
banda (vantaggio positivo o negativo, rapporto sopra o sotto la fascia), in due
la tosatura non interviene affatto.

Conviene dire con precisione che cosa questo garantisce, perché è meno di quanto
il nome suggerisca. È un'euristica **del primo ordine e per campione**, non un
vincolo: niente impedisce a $\rho_t$ di finire fuori dall'intervallo, il
gradiente si annulla solo dove il campione è già stato tosato (e si è appena
visto in quali casi non lo sia), e PPO fa più
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
dalla vecchia: è la manopola che in PPO fa l’$\epsilon$ del tosaggio, con la
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

A rigore l'obiettivo che si implementa non è
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
non regge, e conviene guardarla in faccia per due ragioni.

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

Con REINFORCE, l'attore-critico e PPO la cassetta degli attrezzi del gradiente
di policy è completa: una strategia che si aggiorna a piccoli passi, un
critico che tiene bassa la varianza, e un guinzaglio che impedisce di buttare
via quello che funzionava.

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
  l'apprendimento diventa più rapido e più stabile, al prezzo che finché
  l'allenatore è inesperto i suoi commenti sono storti. **A3C** fa giocare
  molti attori in parallelo; **PPO** scoraggia i passi lunghi invece di
  vietarli: a chi si allontana troppo dalla strategia che ha giocato le
  partite toglie il premio, non la possibilità. È quello che si prova per
  primo, perché perdona gli errori di taratura più degli altri.
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
  ordine e non un vincolo di trust region; il suo vantaggio misurato su
  TRPO viene in larga parte dalle ottimizzazioni di implementazione
  {cite}`engstrom2020implementation`.
```
`````

Resta il gesto che i giocatori forti fanno prima di muovere: pensare. È la
[ricerca ad albero Monte Carlo](mcts-alphago.md), dove la strategia appena
costruita smette di rispondere d'istinto, ed è la strada che porta ad AlphaGo
e, di lì, all'allineamento dei modelli linguistici.
