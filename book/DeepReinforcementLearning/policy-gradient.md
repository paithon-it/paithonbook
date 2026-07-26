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
policy, $r_t$ è la ricompensa al passo $t$ e $\gamma\in[0,1]$ è il fattore di
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

con $\alpha$ il passo di apprendimento. Il punto debole è la **varianza**:
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
regola diventa $\nabla_\theta \log\pi_\theta(a_t\mid s_t)\,A_t$, con varianza
molto più bassa perché $V_\phi(s_t)$ funge da *baseline*. Attore e critico si
addestrano insieme: il critico affina le sue stime, l'attore le usa come
segnale.

`````

## A3C e PPO: gli algoritmi che funzionano davvero

Su queste fondamenta poggiano gli algoritmi usati in pratica. **A3C**
(*Asynchronous Advantage Actor-Critic* {cite}`mnih2016asynchronous`) fa
girare molti
attori in parallelo su copie dell'ambiente: le loro esperienze scorrelate
rendono l'addestramento più stabile e sfruttano le CPU multi-core. **PPO**
(*Proximal Policy Optimization* {cite}`schulman2017proximal`) è oggi lo
standard di fatto, per robustezza e semplicità.

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
$0{,}2$) fissa quanto le è concesso spostarsi. Il *clipping* annulla
l'incentivo a spingere $\rho_t$ fuori dall'intervallo $[1-\epsilon,1+\epsilon]$:
la policy migliora a piccoli passi controllati, senza gli aggiornamenti
distruttivi che affliggevano i primi metodi.

`````

## Da AlphaGo ad AlphaZero

Torniamo alla mossa 37. AlphaGo {cite}`silver2016mastering` non era un solo
algoritmo, ma una sintesi: una rete di policy che proponeva mosse promettenti,
una rete di valore che stimava chi fosse in vantaggio, e una **ricerca ad
albero Monte Carlo** (MCTS) che usava entrambe per esplorare in profondità solo
le linee più sensate. Le reti erano state addestrate prima su partite umane,
poi affinate con RL giocando contro se stesse.

Un anno dopo, **AlphaZero** {cite}`silver2017mastering` elimina persino le
partite
umane: parte dalle sole regole del gioco e impara *tabula rasa*, dal nulla,
soltanto affrontando copie di sé. La stessa architettura padroneggia Go,
scacchi e shogi, superando i migliori programmi specializzati. È la
dimostrazione più limpida di cosa nasce dall'unione di RL, ricerca ad albero e
reti profonde.

## Un ultimo salto: allineare i modelli linguistici

Lo stesso meccanismo (aumentare la probabilità di ciò che riceve un giudizio
positivo) è oggi al cuore dell'addestramento dei modelli linguistici.
Nell'**RLHF** (*Reinforcement Learning from Human Feedback*; Christiano et
al., 2017; {cite}`ouyang2022training`) le risposte del modello sono
l'"azione", dei valutatori umani indicano quali preferiscono, e le loro
preferenze addestrano un *modello di ricompensa* che fa da critico. Con PPO si
ritocca poi la policy del modello (la sua tendenza a produrre certe risposte),
verso ciò che gli umani apprezzano. La stessa idea che ha portato una macchina
a giocare la mossa 37 aiuta oggi un assistente a rispondere in modo utile e
onesto.

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
