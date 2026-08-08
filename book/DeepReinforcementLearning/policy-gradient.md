# Metodi a gradiente di policy

Seul, marzo 2016. Alla trentasettesima mossa della seconda partita contro il
campione Lee Sedol, il programma AlphaGo appoggia una pietra in un punto che
nessun giocatore professionista avrebbe scelto: i commentatori pensano a un
errore. Era invece una mossa che, secondo le stime del programma stesso, un
umano avrebbe giocato circa una volta su diecimila. Lee Sedol si alza dal
tavolo per un quarto d'ora. Quella mossa non era stata copiata da nessun
archivio di partite: era il frutto di una *strategia* appresa giocando milioni
di volte contro se stesso.

Come si insegna a una macchina una strategia? Nei metodi basati sul valore,
che abbiamo incontrato con il Q-learning, impariamo a stimare *quanto vale*
uno stato o un'azione, la funzione $V(s)$ e la funzione $Q(s,a)$, e ne
ricaviamo l'azione migliore scegliendo di volta in volta quella con valore più
alto. I metodi a **gradiente di policy** ribaltano la prospettiva: invece di
valutare e poi decidere, imparano *direttamente a decidere*.

## Imparare la policy, non il valore

`````{tab} Elementare

Immagina di allenare un cane. Un approccio è compilare una tabella mentale che
assegna a ogni situazione un "punteggio di bontà" per ciascun comportamento
possibile, e poi far scegliere al cane il comportamento col punteggio più alto.
L'altro approccio, più diretto, è modellare la *tendenza* del cane: rendere più
probabili i comportamenti che in passato hanno fruttato un premio, meno
probabili quelli finiti male. Non calcoliamo un punteggio per poi decidere:
regoliamo direttamente le probabilità con cui l'animale sceglie.

Una **policy** è esattamente questo: una funzione che, data una situazione,
dice con quale probabilità compiere ciascuna azione.

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

`````

`````{tab} Superiore

REINFORCE stima il gradiente di $J(\theta)$ tramite il *policy gradient
theorem* {cite}`sutton2000policy`:

$$
\nabla_\theta J(\theta) =
\mathbb{E}\Big[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t \Big],
$$

dove $G_t=\sum_{k\ge t}\gamma^{\,k-t} r_k$ è il ritorno osservato a partire dal
passo $t$. Il termine $\nabla_\theta \log \pi_\theta(a_t\mid s_t)$ indica *come*
ritoccare i parametri per rendere più probabile l'azione $a_t$; moltiplicandolo
per $G_t$, quel ritocco viene amplificato quando l'esito è stato buono e
invertito quando è stato cattivo. L'aggiornamento è quindi

$$
\theta \leftarrow \theta + \alpha\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t ,
$$

con $\alpha$ il passo di apprendimento. Un'avvertenza di rigore: il gradiente
esatto dell'obiettivo scontato conterrebbe un fattore $\gamma^{\,t}$ davanti a
ciascun addendo della somma; la prassi, che seguiamo qui, lo omette, ottenendo
una direzione leggermente distorta rispetto a $\nabla_\theta J(\theta)$ ma che
non soffoca il segnale dei passi lontani nel tempo. Chi ha letto la sezione
sui bandit
riconosce la struttura: il *bandit a gradiente* era esattamente questo, in un
mondo con un solo stato, dove la softmax sulle preferenze $H(a)$ faceva le
veci di $\pi_\theta(a\mid s)$. Anche il rimedio che segue è già comparso là.

Il punto debole è la **varianza**:
$G_t$ dipende dall'intero seguito casuale della partita, e le stime risultano
rumorose e lente a convergere.

`````

## Actor-Critic: chi agisce e chi giudica

Come si abbatte quella varianza? L'idea è affiancare al giocatore un giudice
che commenta le mosse una per una, senza aspettare la fine
({numref}`fig-actor-critic`).

```{figure} ../figures/actor-critic.svg
:name: fig-actor-critic
:alt: Il riquadro Agente contiene Attore e Critico; il Critico passa il vantaggio all'Attore, che invia un'azione all'Ambiente, il quale restituisce stato successivo e ricompensa.
:width: 85%

L'architettura Actor-Critic. L'attore decide l'azione, il critico la valuta e
gli restituisce un segnale (il *vantaggio*); l'ambiente risponde con lo stato
successivo e la ricompensa.
```

`````{tab} Elementare

Immagina due ruoli. L'**attore** è chi gioca: decide le mosse. Il **critico**
è un allenatore a bordo campo che, mossa dopo mossa, mormora "meglio del
previsto" oppure "peggio del previsto". L'attore non deve più aspettare la fine
della partita per sapere com'è andata: riceve un giudizio immediato a ogni
passo e corregge subito la rotta. Impara più in fretta e in modo più stabile.

`````

`````{tab} Superiore

Nell'architettura **Actor-Critic** convivono due reti. L'*attore* è la policy
$\pi_\theta(a\mid s)$; il *critico* stima la funzione valore $V_\phi(s)$. Al
ritorno grezzo $G_t$ si sostituisce il **vantaggio** (*advantage*), di solito
l'errore di differenza temporale:

$$
A_t = r_t + \gamma\, V_\phi(s_{t+1}) - V_\phi(s_t) .
$$

$A_t$ misura di quanto l'azione compiuta ha superato le *aspettative*
codificate dal critico: è positivo se l'esito è stato migliore del previsto. La
regola diventa $\nabla_\theta \log\pi_\theta(a_t\mid s_t)\,A_t$, e la varianza
scende per due vie che conviene distinguere. Sottrarre la *baseline*
$V_\phi(s_t)$ non distorce il gradiente; sostituire il ritorno $G_t$ con
$r_t+\gamma V_\phi(s_{t+1})$ è invece *bootstrapping*, e rende la stima
distorta finché il critico è impreciso. Si scambia varianza con *bias*: è il
compromesso al cuore dell'actor-critic. Attore e critico si addestrano
insieme: il critico affina le sue stime, l'attore le usa come segnale.

`````

## A3C e PPO: gli algoritmi che funzionano davvero

Su queste fondamenta poggiano gli algoritmi usati in pratica. **A3C**
(*Asynchronous Advantage Actor-Critic* {cite}`mnih2016asynchronous`) fa
girare molti
attori in parallelo su copie dell'ambiente: le loro esperienze scorrelate
rendono l'addestramento più stabile e sfruttano le CPU multi-core. **PPO**
(*Proximal Policy Optimization* {cite}`schulman2017proximal`) è oggi lo
standard di fatto, per robustezza e semplicità.

```{figure} ../figures/ppo-2017.svg
:name: fig-ppo-clipping
:alt: "Dalla policy vecchia, al centro, si dipartono due frecce. La prima è un passo breve che resta dentro una fascia consentita disegnata attorno al punto di partenza, e viene accettata. La seconda è un salto lungo che esce dalla fascia: l'aggiornamento viene tagliato al bordo, e oltre quel bordo non porta più alcun vantaggio."
:width: 84%

Il guinzaglio di PPO. Non impedisce di migliorare: toglie il premio a chi
prova a migliorare troppo in una volta, e così l'aggiornamento resta vicino
alla policy che ha generato i dati.
```

Il taglio disegnato in {numref}`fig-ppo-clipping` esiste perché i dati
invecchiano in fretta. Le esperienze raccolte sono state generate dalla policy
vecchia, e valgono a stimare il gradiente solo finché la nuova le somiglia; un
passo troppo lungo userebbe dati che non descrivono più il comportamento
dell'agente.

`````{tab} Elementare

Il rischio, quando aggiorni una strategia, è esagerare: un solo passo troppo
lungo può rovinare l'apprendimento di ore intere. PPO fa esattamente ciò che
suggerisce il nome (*proximal*, "vicino"): cambia la policy solo di poco per
volta, senza mai allontanarsi troppo da quella attuale. Piccoli passi prudenti,
ma tanti.

`````

`````{tab} Superiore

PPO massimizza un obiettivo "tosato" (*clipped*):

$$
L^{\text{CLIP}}(\theta) =
\mathbb{E}\big[\min\!\big(\rho_t A_t,\ \operatorname{clip}(\rho_t,\,1-\epsilon,\,1+\epsilon)\,A_t\big)\big],
$$

dove $\rho_t = \dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$
è il rapporto tra la nuova e la vecchia policy, e $\epsilon$ (tipicamente
$0{,}2$) fissa quanto le è concesso spostarsi. Quel rapporto non è un
espediente inventato qui: è il **rapporto di importance sampling** incontrato
nel capitolo precedente, quello che permette di valutare una policy con dati
generati da un'altra, troncato a un passo solo. E il suo difetto è lo stesso
già visto là: può assumere valori enormi e mandare in aria la stima. Il
*clipping* annulla
l'incentivo a spingere $\rho_t$ fuori dall'intervallo $[1-\epsilon,1+\epsilon]$:
la policy migliora a piccoli passi controllati, senza gli aggiornamenti
distruttivi che affliggevano i primi metodi.

`````

## Da AlphaGo ad AlphaZero

Torniamo alla mossa 37. AlphaGo {cite}`silver2016mastering` non era un solo
algoritmo, ma una sintesi: una rete di policy che proponeva mosse promettenti,
una rete di valore che stimava chi fosse in vantaggio, e una **ricerca ad
albero Monte Carlo** (MCTS) che usava entrambe per esplorare in profondità solo
le linee più sensate.

```{figure} ../figures/alphago-2016.svg
:name: fig-alphago
:alt: "Ciclo chiuso: il self-play genera partite che il sistema gioca contro sé stesso; dalle partite si addestrano due reti, quella di policy che propone le mosse e quella di valore che stima chi sta vincendo; le due reti guidano a loro volta una ricerca ad albero Monte Carlo, che gioca meglio di entrambe e produce le partite del giro successivo."
:width: 92%

Il giro che si alimenta da solo. La ricerca ad albero gioca meglio delle reti
che la guidano, e le partite che ne escono diventano il materiale con cui
quelle reti migliorano.
```

Il ciclo di {numref}`fig-alphago` è il motivo per cui i suoi successori
poterono fare a meno delle partite umane. Se la ricerca produce mosse migliori di quelle che
le reti sanno proporre, allora il sistema ha una fonte di supervisione interna:
non gli serve un maestro, gli basta giocare contro sé stesso e imparare da
dove la ricerca lo ha portato. Le reti erano state addestrate prima su partite umane,
poi affinate con RL giocando contro se stesse.

Un anno dopo, **AlphaGo Zero** {cite}`silver2017mastering` elimina persino le
partite umane: parte dalle sole regole del Go e impara *tabula rasa*, dal
nulla, soltanto affrontando copie di sé, fino a battere nettamente la versione
che aveva sconfitto Lee Sedol. Nel 2018 **AlphaZero** {cite}`silver2018general`
generalizza la ricetta: la stessa architettura, senza ritocchi per gioco,
padroneggia Go, scacchi e shogi, superando i migliori programmi
specializzati. È la dimostrazione più limpida di cosa nasce dall'unione di RL,
ricerca ad albero e reti profonde.

## Un ultimo salto: allineare i modelli linguistici

Lo stesso meccanismo (aumentare la probabilità di ciò che riceve un giudizio
positivo) è oggi al cuore dell'addestramento dei modelli linguistici.

```{figure} ../figures/instructgpt-2022.svg
:name: fig-instructgpt
:alt: "Schema in tre stadi: alcune persone ordinano per preferenza più risposte allo stesso prompt; da questi ordinamenti si addestra un reward model che impara ad assegnare punteggi; il reward model guida infine l'ottimizzazione della policy del modello linguistico, che genera, viene valutata e aggiornata."
:width: 100%

Il giudizio umano che diventa un numero. Le persone non danno voti: mettono in
fila delle risposte, ed è il reward model a tradurre quell'ordine in un
punteggio che l'ottimizzazione sa usare.
```

Il dettaglio di {numref}`fig-instructgpt` che vale la pena notare è il primo
stadio: alle persone si chiede di **ordinare**, non di valutare. Confrontare
due risposte è un giudizio che gli esseri umani danno con buona coerenza fra
loro; assegnare un voto da uno a dieci molto meno, e su scale diverse.
Nell'**RLHF** (*Reinforcement Learning from Human Feedback*; Christiano et
al., 2017; {cite}`ouyang2022training`) le risposte del modello sono
l'"azione", dei valutatori umani indicano quali preferiscono, e le loro
preferenze addestrano un *modello di ricompensa* che fa da critico. Con PPO si
ritocca poi la policy del modello (la sua tendenza a produrre certe risposte),
verso ciò che gli umani apprezzano. La stessa idea che ha portato una macchina
a giocare la mossa 37 aiuta oggi un assistente a rispondere in modo utile e
onesto.

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
  (passi piccoli e prudenti, ma tanti) ed è oggi lo standard.
- **AlphaGo** e **AlphaZero** uniscono la strategia, la stima di chi sta
  vincendo e l'esplorazione ad albero delle mosse; con l'**RLHF** lo stesso
  meccanismo, guidato dalle preferenze delle persone, allinea i modelli
  linguistici.
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
  $A_t$, riducendo la varianza. **A3C** parallelizza gli attori; **PPO** limita
  i passi con il *clipping* ed è lo standard attuale.
- **AlphaGo/AlphaZero** uniscono policy, valore e ricerca ad albero; **RLHF**
  applica PPO all'allineamento degli LLM.
```
`````
