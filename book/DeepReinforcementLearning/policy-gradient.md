# Metodi a gradiente di policy

Seul, marzo 2016. Alla trentasettesima mossa della seconda partita contro il
campione Lee Sedol, il programma AlphaGo appoggia una pietra in un punto che
nessun giocatore professionista avrebbe scelto, tanto che i commentatori
pensano a un errore. (Il Go si gioca appoggiando pietre bianche e nere sugli
incroci di una griglia, e vince chi circonda più territorio.) Era invece una
mossa che, secondo le stime del programma stesso, un umano avrebbe giocato
circa una volta su diecimila. Lee Sedol si alza dal tavolo per un quarto d'ora.

AlphaGo aveva due reti e una ricerca. La prima rete proponeva le mosse da
esaminare, e aveva imparato imitando partite umane; era lei a stimare
quell'una su diecimila. La seconda, la rete di valore, giudicava chi fosse in
vantaggio in una posizione. La ricerca provava le continuazioni prima di
muovere, guidata da tutte e due, e fu lei a trovare la mossa. Il giudizio della
rete di valore si era formato su partite che il programma giocava contro sé
stesso, e nella versione descritta pochi mesi prima a giocare quelle partite era
una *strategia* affinata proprio con il gradiente di policy.

Come si insegna a una macchina una strategia? Nei metodi basati sul valore, che
abbiamo incontrato con il Q-learning, impariamo a stimare *quanto vale una
situazione* e *quanto vale una mossa in quella situazione* (nella {doc}`sezione
su MDP e funzioni valore </ReinforcementLearning/mdp-valore>` le due stime si
chiamavano rispettivamente $V$ e $Q$), e poi ne ricaviamo l'azione migliore
scegliendo di volta in volta quella col valore più alto. I metodi a **gradiente
di policy** ribaltano la prospettiva: invece di valutare e poi decidere,
imparano *direttamente a decidere*.

La policy è una rete neurale con pesi $\theta$, e la quantità da far crescere
è il ritorno atteso $J(\theta)$, cioè la somma dei premi che quella policy
raccoglie in media. Il «gradiente» del nome è $\nabla_\theta J(\theta)$, il
vettore delle derivate di $J$ rispetto ai pesi, che indica la direzione in cui
$J$ cresce più in fretta; e l'aggiornamento fa un passo in quella
direzione, $\theta \leftarrow \theta + \alpha\,\nabla_\theta
J(\theta)$, con $\alpha$ il passo di apprendimento.

In altri termini, quando si dice che «la strategia cambia» o che «si fa un
passo», si intende sempre questo: i pesi della rete, i numeri che decidono come
una situazione si trasforma in una decisione, si spostano un pochino nella
direzione del gradiente.

## Imparare la policy, non il valore

Ribaltare così la prospettiva conviene per due motivi. Il primo è che
scegliere l'azione con $\arg\max_a Q(s,a)$ richiede di scorrere le azioni una
per una, e non si può quando l'azione è una quantità da dosare, come l'angolo
di uno sterzo. Il secondo è che la policy migliore può essere *stocastica*,
cioè fare nella stessa situazione a volte una cosa e a volte un'altra, con
probabilità precise. Ad esempio, solo una policy stocastica non si lascia
sfruttare da un avversario che coglie ogni regolarità (a sasso-carta-forbici,
ciascuna mossa con probabilità $1/3$), e lo stesso vale quando la situazione
si vede solo in parte, come con le carte coperte del poker. Un metodo che
sceglie sempre l'azione di valore massimo, però, non ha un modo naturale di
trovarla.

Si aggiunge un vantaggio teorico. Le probabilità della policy cambiano con
continuità al variare dei pesi, mentre l'azione greedy può saltare da una
mossa all'altra per una variazione minima dei valori; anche per questo i
metodi a gradiente di policy hanno garanzie di convergenza più forti
{cite}`sutton2018reinforcement`.

`````{tab} Elementare

Un cane si può allenare in due modi. Uno è compilare una tabella mentale che
assegna a ogni situazione un "punteggio di bontà" per ciascun comportamento
possibile, e poi far scegliere al cane il comportamento col punteggio più alto.
L'altro approccio, più diretto, è modellare la *tendenza* del cane: rendere più
probabili i comportamenti che in passato hanno fruttato un premio, meno
probabili quelli finiti male. In altre parole, non calcoliamo un punteggio per
poi decidere, ma regoliamo direttamente le probabilità con cui l'animale
sceglie.

Una policy è proprio questo: data una situazione, dice con quale
probabilità compiere ciascuna azione. In italiano si direbbe «strategia»;
*policy* è la parola che si legge dappertutto, e le due si equivalgono.

Il premio, però, quasi mai arriva subito. In una seduta di addestramento il
cane fa una decina di cose, una dopo l'altra, e il biscotto compare in fondo.
Per questo a dover crescere è il bottino di tutta la seduta, cioè la somma dei
premi, che nella sezione su MDP e funzioni valore si chiamava *ritorno*. Un
biscotto che arriva fra dieci mosse conta meno di uno che arriva adesso,
perché nel frattempo può succedere di tutto, e quanto meno conta lo decidiamo
noi.

Una seduta, poi, va bene e la successiva male, con lo stesso cane e la stessa
tendenza. Interessa quindi la media su tante sedute, mai una seduta sola.

`````

`````{tab} Superiore

Una **policy parametrica** $\pi_\theta(a \mid s)$ è una distribuzione di
probabilità sulle azioni $a$ condizionata allo stato $s$, controllata dai
parametri $\theta$ (i pesi di una rete neurale). L'obiettivo è massimizzare il
ritorno atteso:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[ R(\tau) \big],
\qquad R(\tau) = \sum_{t=0}^{T} \gamma^{t}\, r_t ,
$$

dove $\tau=(s_0,a_0,r_0,s_1,\dots)$ è una *traiettoria* generata seguendo la
policy, $r_t$ è la ricompensa al passo $t$ e $\gamma\in[0,1)$ è il fattore di
sconto, che pesa meno il futuro lontano.

Rispetto ai metodi basati sul valore, ottimizzare $\pi_\theta$ direttamente
gestisce con naturalezza gli spazi di azioni continui e le policy stocastiche.
In particolare, per azioni continue basta una distribuzione continua, per
esempio una gaussiana
$\pi_\theta(\mathbf{a}\mid s)=\mathcal{N}\big(\boldsymbol{\mu}_\theta(s),\,
\sigma^2\mathbf{I}\big)$ con la media data dalla rete e $\sigma$ fissato o
appreso. Il termine che comparirà nel gradiente vale allora
$\nabla_\theta\log\pi_\theta(\mathbf{a}\mid s)=
\sigma^{-2}\,\mathbf{J}_\theta(s)^{\top}
\big(\mathbf{a}-\boldsymbol{\mu}_\theta(s)\big)$,
con $\mathbf{J}_\theta=\partial\boldsymbol{\mu}_\theta/\partial\theta$ lo
jacobiano della media. Moltiplicato per l'esito, questo termine sposta la
media verso le azioni che hanno reso di più {cite}`sutton2018reinforcement`.

Si noti che la notazione segue ora la convenzione dei lavori di deep RL: $r_t$
è la ricompensa che segue l'azione $a_t$, cioè ciò che la sezione su MDP e
funzioni valore indicava con $R_{t+1}$; e stati e azioni si scrivono in
minuscolo, perché la maiuscola $A_t$ qui serve al *vantaggio*, come annunciato
nella {doc}`sezione sui bandit </ReinforcementLearning/banditi>`. Il pedice si
sposta di uno, la sostanza no: $G_t = \sum_{k\ge t}\gamma^{\,k-t} r_k$ e
$G_t = \sum_{k\ge 0}\gamma^{k} R_{t+1+k}$ sono lo stesso oggetto scritto in due
modi.

`````

## REINFORCE: premiare ciò che ha funzionato

Il primo algoritmo di questa famiglia è REINFORCE, dovuto a Ronald
Williams {cite}`williams1992simple`. Ha un'idea tanto semplice da sembrare
ingenua: gioca un'intera partita, guarda com'è andata, e poi *aumenta la
probabilità delle azioni che hanno portato a ricompense alte*, diminuendo
quella delle azioni seguite da ricompense basse.

`````{tab} Elementare

È il metodo "prova e ricorda". Il giocatore fa una partita intera, dall'inizio
alla fine. Se ha vinto, si dice: "qualunque cosa io abbia fatto, rendila più
probabile la prossima volta". Se ha perso, il contrario. Ripetuto migliaia di
volte, questo semplice riflesso spinge il comportamento verso le mosse buone
senza che nessuno debba mai spiegare *perché* siano buone.

Il difetto salta subito all'occhio, e da qui nasce tutto il seguito: il
giudizio arriva solo alla fine, uno solo per tutta la partita. Se hai vinto,
l'algoritmo rende più probabili anche le due o tre mosse pessime che avevi
fatto per strada; se hai perso, rende meno probabili anche quelle buone.

E due partite giocate con la stessa identica strategia possono finire in
modi opposti per puro caso, con la correzione che cambia segno di conseguenza.
Il risultato è un apprendimento che balla: va nella direzione giusta in media,
ma a strattoni, e ci mette moltissimo. Come nella
{doc}`sezione sui metodi Monte Carlo </ReinforcementLearning/monte-carlo>`, la
varianza misura quanto balla.

Un primo ritocco è gratuito. Quando i punti si segnano per strada, e non solo
alla fine, una mossa non può cambiare quelli già segnati prima di lei, e quindi
la si giudica soltanto sul seguito. La mossa dell'ultimo minuto risponde
dell'ultimo minuto, e la spinta che riceve è tanto più forte quanto più alto è
il bottino che l'ha seguita.

C'è poi una condizione da cui dipende tutto il resto: le partite da cui si
impara devono essere state giocate con la strategia che si sta correggendo. Chi
studia le registrazioni di come giocava l'anno scorso sta
correggendo il giocatore dell'anno scorso, e più cambia, meno quelle
registrazioni parlano di lui. Finché si gioca una partita, si guarda com'è
andata e si corregge, la condizione è rispettata senza doverci pensare; diventa
un problema appena si vuole spremere la stessa partita più volte.

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
l'ambiente. Il primo è l’**identità della log-derivata**,
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
addendi che non dipendono da $\theta$. Derivando spariscono, e con loro
l'ambiente esce di scena,
$\nabla_\theta \log p_\theta(\tau) = \sum_t \nabla_\theta \log
\pi_\theta(a_t \mid s_t)$. Il terzo è la causalità: poiché l'azione al passo
$t$ non può cambiare le ricompense già incassate, i prodotti con le ricompense
passate hanno valore atteso nullo, e a ciascun addendo resta agganciato solo il
ritorno da lì in avanti, $G_t$.

L'enunciato vale sotto le ipotesi consuete ($\pi_\theta$ differenziabile in
$\theta$, ricompense limitate, orizzonte finito oppure $\gamma<1$, quanto basta
per scambiare derivata e valore atteso; la distribuzione stazionaria degli
stati serve solo nella formulazione a ricompensa media, che qui non si usa) e
sotto un'altra: il teorema è on-policy.

Infatti, nella forma originale al posto di $G_t$ compare $Q^\pi(s_t,a_t)$, e
sostituirvi il ritorno osservato è lecito solo perché
$\mathbb{E}[G_t \mid s_t, a_t] = Q^\pi(s_t,a_t)$, il che richiede che il seguito
della traiettoria sia stato generato dalla stessa policy che stiamo
derivando. È l'ipotesi che PPO dovrà correggere con un rapporto di
importance sampling per riusare i dati della policy vecchia.

Il termine $\nabla_\theta \log \pi_\theta(a_t\mid s_t)$ indica *come*
ritoccare i parametri per rendere più probabile l'azione $a_t$; moltiplicandolo
per $G_t$, quel ritocco viene amplificato quando l'esito è stato buono e
invertito quando è stato cattivo. L'aggiornamento è quindi

$$
\theta \leftarrow \theta + \alpha\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t ,
$$

con $\alpha$ il passo di apprendimento. Si noti, però, che questo
aggiornamento non è la formula del teorema: il $\gamma^{\,t}$ davanti a
ciascun addendo è sparito. Ometterlo è la prassi, e anche qui la si segue, ma
la direzione che si ottiene non è più $\nabla_\theta J(\theta)$. Nota e Thomas
{cite}`nota2020policy` hanno anzi mostrato che non è il gradiente di nessuna
funzione, per cui le garanzie della salita del gradiente non la coprono (esiste
un esempio costruito in cui converge a una policy pessima). In pratica la si
usa perché non soffoca il segnale dei passi lontani nel tempo, che con lo
sconto esatto peserebbero quasi nulla.

La struttura è quella del *bandit a gradiente* della
{doc}`sezione sui bandit </ReinforcementLearning/banditi>`, lo stesso
aggiornamento in un mondo con un solo stato, dove la softmax sulle preferenze
$H(a)$ faceva le veci di $\pi_\theta(a\mid s)$. Anche il rimedio che segue, la
baseline, è già comparso là.

Il punto debole è la varianza. Poiché $G_t$ dipende dall'intero seguito
casuale della partita, le stime risultano rumorose e lente a convergere.

`````

## Actor-Critic: chi agisce e chi giudica

La varianza di REINFORCE si abbassa affiancando alla policy, che qui prende il
nome di **attore**, una stima del valore, il **critico**, che giudica le mosse
una per una senza aspettare la fine dell'episodio
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

L'attore è chi gioca, cioè chi decide le mosse. Il critico è un allenatore a
bordo campo che, mossa dopo mossa, mormora "meglio del previsto" oppure "peggio
del previsto". L'attore non deve più aspettare la fine della partita per sapere
com'è andata, perché riceve un giudizio immediato a ogni passo e corregge
subito la rotta. Così impara più in fretta e in modo più stabile.

Attore e critico non sono due persone, naturalmente: sono due reti neurali, che
si allenano insieme sulla stessa partita. Una impara a decidere, l'altra impara
a prevedere come andrà a finire.

Quel «previsto» l'allenatore lo fissa guardando la situazione, prima che la
mossa sia giocata: «da qui, di solito, si porta a casa un pareggio». Questo
rende onesto il commento che verrà dopo. Un allenatore che decidesse quanto era
difficile *dopo* aver visto la mossa, e alzasse l'asticella solo davanti alle
mosse che non gli piacciono, insegnerebbe al giocatore i propri gusti invece
del gioco.

L'allenatore, però, può sbagliare. Il suo «meglio del previsto» vale quanto
valgono le sue previsioni, e all'inizio non ne sa più del giocatore: sono
giudizi affrettati presi per buoni. Il verdetto di fine partita, quello, non
sbagliava mai; ballava soltanto, perché una partita sola dice poco. Si
scambia una cosa con l'altra, commenti immediati in cambio del
rischio che siano storti, e su questo scambio si basa il metodo.

Fra i due estremi c'è una manopola, e la si gira dove si vuole: quanto aspetta
l'allenatore prima di parlare. Una mossa sola, e parla subito fidandosi in tutto
del proprio fiuto. Dieci mosse, e si fida un po’ meno, perché nel frattempo ha
visto succedere delle cose. Fino alla fine della partita, che è tornare al prova
e ricorda.

`````

`````{tab} Superiore

Nell'architettura Actor-Critic convivono due reti. L’*attore* è la policy
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

Come d'uso in deep RL, questa stima si indica con $A_t$. Resta tuttavia uno
stimatore e non la definizione, poiché coincide con $A^\pi$ solo se il critico
ha ragione, cioè se $V_\phi = V^\pi$.

$A_t$ misura di quanto l'azione compiuta ha superato le *aspettative*
codificate dal critico, ed è positivo se l'esito è stato migliore del
previsto. La regola diventa $\nabla_\theta \log\pi_\theta(a_t\mid s_t)\,A_t$, e
la varianza scende in due modi distinti. Il primo è sottrarre la *baseline*
$V_\phi(s_t)$, che non distorce il gradiente; la verifica è immediata:

$$
\mathbb{E}_{a\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a\mid s)\, b(s)\big]
= b(s)\sum_a \nabla_\theta \pi_\theta(a\mid s)
= b(s)\,\nabla_\theta\!\!\sum_a \pi_\theta(a\mid s)
= b(s)\,\nabla_\theta 1 = 0 .
$$

Il primo passaggio usa l'identità della log-derivata,
$\pi_\theta\,\nabla_\theta\log\pi_\theta = \nabla_\theta \pi_\theta$, la stessa
che fa comparire il $\log$ nel teorema precedente; il secondo scambia la
derivata con la somma. È scritto per azioni discrete, e nel continuo la somma
diventa un integrale e lo scambio va giustificato, ma la conclusione non
cambia. È sufficiente che $b$ non dipenda dall'azione: qualunque funzione del
solo stato si può sottrarre senza introdurre distorsione.

Il secondo modo è il *bootstrapping*, cioè sostituire il ritorno $G_t$ con
$r_t+\gamma V_\phi(s_{t+1})$, che rende la stima distorta finché il critico è
impreciso. Si scambia quindi varianza con *bias*, ed è il compromesso alla
base dell'actor-critic.

Il vantaggio stimato con l'errore TD a un passo è la scelta più economica del
compromesso, con poca varianza e parecchio bias. All'altro estremo c'è il
ritorno completo di REINFORCE, privo di distorsione ma ad alta varianza. Fra i
due non c'è un salto ma una famiglia continua, governata da un parametro
$\lambda$ che regola quanto lontano si guarda avanti prima di affidarsi al
critico. Detto $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ l'errore
TD di un passo, la stima del vantaggio è

$$
\hat{A}_t \;=\; \sum_{l \ge 0} (\gamma\lambda)^l\, \delta_{t+l},
$$

che con $\lambda = 0$ si riduce al solo $\delta_t$ e con $\lambda = 1$ torna al
ritorno completo meno la baseline. È la *generalized advantage estimation*
{cite}`schulman2016high`, usata in pratica dentro PPO. Attore e critico si
addestrano insieme: il critico affina le sue stime, l'attore le usa come
segnale.

`````

## A3C e PPO

Dall'attore-critico discendono due algoritmi che hanno fatto scuola.

**A3C** fa giocare molte copie dell'agente insieme, ciascuna la propria
partita in una copia sua del gioco {cite}`mnih2016asynchronous`. Il guadagno è
lo stesso della memoria delle esperienze in DQN, ottenuto per un'altra strada.
In DQN si rimescolavano esperienze pescate da momenti lontani fra loro; in A3C
le esperienze arrivano già diverse l'una dall'altra, perché nello stesso
istante ogni copia si trova in un punto diverso della sua partita. Di
conseguenza la rete non si ritrova mai a correggersi dieci volte consecutive su
situazioni quasi identiche, e l'addestramento oscilla di meno. Come effetto
collaterale, si usano tutti i processori della macchina invece di uno.

La sigla sta per *Asynchronous Advantage Actor-Critic*: attore-critico, con il
vantaggio, e in parallelo senza aspettarsi a vicenda. Si usa più spesso la
variante sincrona, **A2C**, che aspetta tutte le copie e aggiorna i pesi una
volta sola; rende altrettanto, e l'asincronia non si è rivelata un
vantaggio {cite}`wu2017baselines`.

**PPO** (*Proximal Policy Optimization* {cite}`schulman2017proximal`) è oggi di
uso molto diffuso. Gli autori lo presentano come una versione più semplice di
un metodo precedente, TRPO: un normale ottimizzatore del primo ordine al posto
di un vincolo da risolvere a ogni passo, e nei loro esperimenti un'efficienza
campionaria almeno pari. Quanto renda in pratica, però, dipende parecchio dai
dettagli con cui lo si programma.

```{figure} ../figures/ppo-2017.svg
:name: fig-ppo-clipping
:alt: "Dalla policy vecchia, al centro, si dipartono due frecce. La prima è un passo breve che resta dentro una fascia consentita disegnata attorno al punto di partenza, e viene accettata. La seconda è un salto lungo che esce dalla fascia: l'aggiornamento viene tagliato al bordo, e oltre quel bordo non porta più alcun vantaggio."
:width: 84%

Il clipping di PPO. Non impedisce di migliorare: toglie il guadagno a chi prova
a cambiare troppo in una volta la probabilità di una mossa, per tenere
l'aggiornamento vicino alla policy che ha generato le partite. È un incentivo e
non un divieto: niente impedisce all'aggiornamento di uscire dalla fascia,
semplicemente uscirne non frutta più.
```

Il **clipping** di {numref}`fig-ppo-clipping` esiste perché le esperienze
invecchiano in fretta. L'agente ha giocato le partite da cui sta imparando con
la strategia precedente, non con quella che sta sviluppando. Quei dati
indicano in quale direzione conviene muoversi, ma solo finché le due strategie
si somigliano ancora; con un passo troppo lungo finiscono per descrivere il
comportamento di qualcun altro.

Quanto le due policy si somigliano su una mossa si misura con il rapporto fra
la probabilità della nuova e quella della vecchia,
$\rho_t = \pi_\theta(a_t\mid s_t)/\pi_{\theta_{\text{old}}}(a_t\mid s_t)$. Ad
esempio, se la mossa aveva il $10\%$ e adesso ha il $12\%$, il rapporto vale
$1{,}2$; se non è cambiato niente, vale $1$. La fascia disegnata nella figura
va appunto da $0{,}8$ a $1{,}2$, cioè un quinto in meno e un quinto in più.

L'addestramento serve a far salire la ricompensa attesa, spostando i pesi
della rete. Appena si esce dalla fascia dalla parte pericolosa, però, quel
numero smette di salire per quanto ci si sposti ancora. Il passo resta
consentito ma non frutta più niente, e una rete che cerca il guadagno non ha
ragione di farlo.

`````{tab} Elementare

Il rischio, quando aggiorni una strategia, è esagerare: un solo passo troppo
lungo può rovinare l'apprendimento di ore intere. Il nome di PPO promette la
cura (*proximal* vuol dire «vicino»): tenere la strategia nuova a poca distanza
da quella che ha giocato le partite.

Il freno, però, agisce da un lato solo, e quale lato dipende da com'è andata la
mossa. Prendi una mossa andata bene, che la strategia vecchia
sceglieva il $10\%$ delle volte. Portarla al $12\%$ frutta; portarla al $15\%$
non frutta niente di più, perché oltre il bordo della fascia il guadagno si
ferma, ed è lì che si rischia di strafare.

Se invece l'aggiornamento l'ha fatta scendere al $5\%$, rimetterla su frutta
per intero, perché perdere una mossa buona non è il pericolo da cui ci si
difende. Per una mossa andata male i lati si scambiano: farla scendere frutta
solo fino all’$8\%$, mentre se è salita, riportarla giù frutta sempre.
«Vicino», insomma, resta un proposito: quanto le due strategie si somiglino va
misurato.

Vietare sul serio i passi lunghi si può. È la strada più vecchia, quella che
PPO ha semplificato: si misura di quanto la strategia nuova differisce dalla
vecchia e, se la differenza supera una soglia, il passo si accorcia finché
rientra. Quella strada mantiene la promessa, ma è un lavoro in più a ogni
passo: la misura e l'accorciamento vanno rifatti ogni volta, con parecchia
macchina da costruire e da far girare. PPO ottiene quasi lo stesso effetto con
due righe dentro un allenamento normale.

Al numero che si fa salire si aggiunge poi un piccolo premio per chi non si
riduce a giocare sempre la stessa mossa: serve a non far irrigidire la strategia
su un'unica risposta prima di aver visto abbastanza.

L'ultima cosa è la meno elegante. Quanto PPO vada meglio della strada più
vecchia, se si va a misurarlo, dipende poco dal freno: viene soprattutto da nove
accorgimenti di programmazione, del genere di rimettere i numeri sulla stessa
scala, tagliare quelli fuori misura, scegliere da quali valori far partire la
rete. Fra l'algoritmo raccontato su una pagina e il programma che lo esegue c'è
spesso più distanza che fra due algoritmi diversi.

`````

`````{tab} Superiore

PPO massimizza un obiettivo con clipping (*clipped*):

$$
L^{\text{CLIP}}(\theta) =
\mathbb{E}\big[\min\!\big(\rho_t A_t,\
\operatorname{clip}(\rho_t,\,1-\epsilon,\,1+\epsilon)\,A_t\big)\big],
$$

dove $\rho_t = \dfrac{\pi_\theta(a_t\mid
s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$ è il rapporto tra la nuova e la
vecchia policy, e $\epsilon$ (tipicamente $0{,}2$) fissa la larghezza della
fascia entro cui lo spostamento continua a fruttare. Si tratta del rapporto di
importance sampling incontrato nella sezione sui metodi Monte Carlo, che
permette di valutare una policy con dati generati da un'altra, qui troncato a
un singolo passo. Il limite resta lo stesso: il rapporto può assumere valori
molto grandi e destabilizzare la stima. Per questo motivo il *clipping* toglie
il premio, campione per campione, a chi spinge $\rho_t$ fuori dall'intervallo
$[1-\epsilon,1+\epsilon]$.

Lo toglie però da un lato solo, e quale lato dipende dal segno del vantaggio.
Il nome nasconde questo dettaglio, che si verifica con il calcolo invece di
affidarsi all'intuizione.

Se l'azione è andata meglio del previsto ($A_t>0$), il $\min$ interviene a
destra: oltre $1+\epsilon$ l'obiettivo si appiattisce e il gradiente si
annulla, mentre sotto $1-\epsilon$ il gradiente resta pieno, perché rendere
*meno* probabile un'azione buona non è il rischio da cui ci si vuole
difendere. Se l'azione è andata peggio del previsto ($A_t<0$) i due lati si
scambiano: il taglio scatta sotto $1-\epsilon$, e sopra $1+\epsilon$ non
scatta affatto.

Ad esempio, con $\epsilon=0{,}2$, $A_t=-1$ e $\rho_t=5$, cioè un rapporto più
di quattro volte l'estremo superiore della fascia ($1{,}2$), l'obiettivo vale
$-5$ e la sua derivata rispetto a $\rho_t$ vale $-1$: gradiente intero, niente
clipping. In generale, dei quattro casi fuori banda (vantaggio positivo o
negativo, rapporto sopra o sotto la fascia), in due il clipping non interviene
affatto, e la derivazione automatica lo conferma caso per caso:

```python
import torch

eps = 0.2
print("vantaggio  rapporto  obiettivo  derivata")
for vantaggio in (+1.0, -1.0):
    for valore in (0.5, 0.9, 1.1, 1.5, 5.0):
        rapporto = torch.tensor(valore, requires_grad=True)
        tosato = torch.clamp(rapporto, 1 - eps, 1 + eps)
        obiettivo = torch.min(rapporto * vantaggio, tosato * vantaggio)
        obiettivo.backward()                 # derivata rispetto al rapporto
        print(f"{vantaggio:+9.1f} {valore:9.1f} {obiettivo.item():10.2f} "
              f"{rapporto.grad.item():9.1f}")
```

```text
vantaggio  rapporto  obiettivo  derivata
     +1.0       0.5       0.50       1.0
     +1.0       0.9       0.90       1.0
     +1.0       1.1       1.10       1.0
     +1.0       1.5       1.20       0.0
     +1.0       5.0       1.20       0.0
     -1.0       0.5      -0.80       0.0
     -1.0       0.9      -0.90      -1.0
     -1.0       1.1      -1.10      -1.0
     -1.0       1.5      -1.50      -1.0
     -1.0       5.0      -5.00      -1.0
```

La derivata è nulla solo dove il $\min$ sceglie il termine limitato dal
clipping, cioè sopra la fascia con vantaggio positivo e sotto con vantaggio
negativo.

Il clipping garantisce meno di quanto il nome suggerisca. È un'euristica del
primo ordine calcolata per singolo campione, non un vincolo: niente impedisce a
$\rho_t$ di uscire dall'intervallo, il gradiente si annulla solo dove il
campione è già stato limitato dal clipping, e PPO esegue più epoche di
minibatch sugli stessi dati. In particolare, al primo passo di ogni
aggiornamento la nuova policy coincide con la vecchia: tutti i rapporti valgono
$1$ e il clipping non agisce su nulla, quindi quel passo è esattamente quello
dell'obiettivo non vincolato.

Alla prova empirica il clipping non basta. Engstrom e colleghi
{cite}`engstrom2020implementation` verificano infatti che nessuno dei tre
algoritmi a confronto riesce a mantenere i rapporti dentro l'intervallo
$[1-\epsilon,1+\epsilon]$, compreso PPO, il cui obiettivo è progettato per
contenere quei rapporti. Bisogna però distinguere questo intervallo dalla regione
di fiducia in divergenza di Kullback-Leibler di TRPO: quest'ultima viene
imposta in modo rigoroso, quasi per costruzione; ma con gli iperparametri
migliori la divergenza media resta contenuta anche per PPO, e secondo gli
autori a tenerla è il modo in cui l'obiettivo viene ottimizzato (passo di
apprendimento, momento, ottimizzazioni di contorno) più che il clipping.

Detto così il clipping sembra un trucco; nasce invece come sostituto economico
di un'idea precisa che lo precede. **TRPO** (*Trust Region Policy
Optimization*) {cite}`schulman2015trust` pone il problema come massimizzazione
vincolata: si massimizza lo stesso obiettivo con importance sampling, ma
imponendo che la nuova policy resti vicina alla vecchia in divergenza di
Kullback-Leibler,

$$
\max_\theta\ \mathbb{E}\big[\rho_t A_t\big]
\quad \text{soggetto a} \quad
\mathbb{E}\big[D_{\mathrm{KL}}(\pi_{\theta_{\text{old}}} \,\|\, \pi_\theta)\big]
\le \delta ,
$$

dove $\delta$ è il raggio ammesso, ossia la massima divergenza consentita tra
la nuova policy e la vecchia. Questo parametro ha il ruolo dell’$\epsilon$ di
PPO, ma si misura in divergenza e non in rapporto fra probabilità, e in TRPO
agisce come vincolo rigido, in PPO come incentivo. (Il simbolo $\delta$ indica
qui una soglia scalare e non va confuso con l'errore TD $\delta_t$ della stima
del vantaggio.)

Il vincolo definisce una **regione di fiducia**, cioè l'intorno entro il quale
l'approssimazione lineare dell'obiettivo è ancora credibile. Serve per la
stessa ragione del clipping di PPO: una policy non è un modello
supervisionato qualunque, perché determina i dati che raccoglierà, e un passo
troppo lungo non produce un errore recuperabile ma una policy che smette di
visitare gli stati utili.

Il costo di TRPO è computazionale. Linearizzando l'obiettivo attorno a
$\theta_{\text{old}}$, con gradiente $\mathbf{g}=\nabla_\theta\,
\mathbb{E}[\rho_tA_t]$, e approssimando la divergenza al secondo ordine,
$\tfrac12\,\Delta\theta^{\top}\mathbf{F}\,\Delta\theta$ con $\mathbf{F}$ la
matrice di informazione di Fisher e $\Delta\theta=\theta-\theta_{\text{old}}$,
il passo vincolato ha forma chiusa,

$$
\Delta\theta =
\sqrt{\frac{2\delta}{\mathbf{g}^{\top}\mathbf{F}^{-1}\mathbf{g}}}\;
\mathbf{F}^{-1}\mathbf{g},
$$

cioè il gradiente naturale, con il passo scelto per arrivare al bordo del
raggio $\delta$. Il prodotto $\mathbf{F}^{-1}\mathbf{g}$ si ottiene con il
gradiente coniugato, senza mai invertire $\mathbf{F}$, e una ricerca di linea
controlla che il vincolo vero, non approssimato, sia rispettato. Funziona, ma è
pesante e laborioso da implementare. PPO propone di ottenere l'effetto che si
vuole (non allontanarsi troppo) con un `min` e un `clip` dentro un normale
ottimizzatore del primo ordine.

A rigore l'obiettivo che si implementa non è solo $L^{\text{CLIP}}$, perché
gli si sommano la perdita del critico e un piccolo bonus di entropia,

$$
L^{\text{CLIP}} - c_1 L^{\text{VF}} + c_2\, \mathcal{H}[\pi_\theta],
$$

dove $L^{\text{VF}}$ è l'errore quadratico del critico,
$\mathcal{H}[\pi_\theta]$ è l'entropia media della policy (la stessa
$\mathcal{H}$ che la
{doc}`sezione sul controllo continuo <controllo-continuo>` definirà come
$-\mathbb{E}_{a\sim\pi}[\log\pi(a\mid s)]$) e $c_1, c_2$ sono due pesi fissi. Il
terzo termine impedisce alla policy di collassare troppo presto su un'unica
azione, ed è lo stesso ingrediente che diventerà il segno distintivo di SAC,
qui con un peso molto più piccolo e uno scopo più modesto.

Si potrebbe concludere che PPO si è affermato perché il clipping è «abbastanza
corretto», ma la letteratura smentisce questa spiegazione data a posteriori, per
due ragioni.

La prima è che il confronto con TRPO non è fra un'euristica e un teorema.
Infatti il teorema di miglioramento monotono chiede il massimo della KL su
tutti gli stati e un coefficiente di penalità, mentre la formula di TRPO,
quella che si implementa, è già il suo rilassamento con la KL media. Anche
TRPO, così com'è usato, ha già rinunciato alla garanzia.

La seconda è che il vantaggio empirico di PPO rispetto a TRPO deriva in
gran parte da altri fattori. Engstrom e colleghi
{cite}`engstrom2020implementation` isolano nove ottimizzazioni di
implementazione della versione di riferimento (normalizzazione e clipping delle
osservazioni, scalatura e clipping della ricompensa, clipping della value
function, inizializzazione ortogonale, annealing del passo di Adam, attivazioni
tanh, clipping globale del gradiente) e trovano che sono loro a spiegare la
maggior parte del guadagno, non l'obiettivo con clipping. Lo studio indipendente di
Andrychowicz e colleghi {cite}`andrychowicz2021what`, su più di 250.000 agenti
addestrati, parte dalla stessa constatazione. È una lezione più utile della
prima: in deep RL la distanza fra l'algoritmo pubblicato e il codice che lo
esegue è spesso maggiore di quella che separa due algoritmi diversi.

`````

In sintesi, REINFORCE, l'attore-critico e PPO formano l'impianto del gradiente
di policy: una policy che si aggiorna per salita del gradiente, un critico che
abbassa la varianza della stima, e un obiettivo con clipping che scoraggia i passi
lunghi senza vietarli.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- I metodi a gradiente di policy imparano direttamente a decidere: invece
  di dare un voto a ogni mossa e poi scegliere la migliore, regolano la
  *tendenza* dell'agente, come si allena un cane rendendo più probabili i
  comportamenti che hanno fruttato un premio. È la via naturale quando le
  mosse possibili non sono un menu di poche voci.
- REINFORCE è il "prova e ricorda": si gioca una partita intera e, se è
  andata bene, si rende più probabile tutto ciò che si è fatto. Semplice, ma
  lento, e balla parecchio, perché il giudizio arriva solo alla fine.
- Actor-Critic affianca al giocatore un allenatore a bordo campo che commenta
  ogni mossa ("meglio del previsto", "peggio del previsto"): l'apprendimento
  diventa più rapido e più stabile, anche se finché l'allenatore è inesperto i
  suoi commenti sono storti. A3C fa giocare molti attori in parallelo; PPO
  scoraggia i passi lunghi invece di vietarli: a chi si allontana troppo dalla
  strategia che ha giocato le partite toglie il premio, non la possibilità. Si
  usa molto, perché è semplice da programmare; ma quanto rende dipende
  parecchio dai dettagli di come lo si programma.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- I metodi a gradiente di policy apprendono direttamente
  $\pi_\theta(a\mid s)$: adatti ad azioni continue e strategie stocastiche.
- REINFORCE aumenta la probabilità delle azioni seguite da ritorni alti,
  ma soffre di varianza elevata.
- Actor-Critic aggiunge un critico $V_\phi(s)$ che fornisce il *vantaggio*
  $A_t$, riducendo la varianza: la baseline non distorce (basta che non dipenda
  dall'azione), il bootstrapping sì. A3C parallelizza gli attori; PPO
  scoraggia i passi lunghi con il *clipping*, che è un'euristica del primo
  ordine e non un vincolo di trust region; il suo vantaggio misurato su
  TRPO viene in larga parte dalle ottimizzazioni di implementazione
  {cite}`engstrom2020implementation`.
```
`````

Ma una policy non è costretta a decidere da sola: può fare da guida alla
{doc}`ricerca ad albero Monte Carlo <mcts-alphago>`, la combinazione che porta
ad AlphaGo. PPO tornerà poi in un contesto diverso, l'allineamento dei modelli
linguistici trattato nella {doc}`sezione sul post-addestramento
</Transformers/post-training>`.
