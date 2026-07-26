# Controllo continuo: DDPG, TD3, SAC

Un joystick Atari ha nove posizioni: su, giù, le diagonali, il centro. Il Deep
Q-Network sceglie fra queste con un `argmax`, confrontando i valori di un
pugno di azioni. Ma prova a immaginare un braccio robotico con sette
articolazioni, o un robot a quattro zampe che deve imparare a camminare. A
ogni istante il controllore non decide "sinistra o destra": decide *quanta*
coppia applicare a ciascun motore, un numero reale (magari negativo per
frenare, magari 3,4 newton-metro, magari 3,41). Non c'è un menu di mosse da
scorrere: c'è un continuo di forze da dosare.

È lo scoglio con cui si chiude il capitolo su DQN. L'operatore $\max_a$, cuore
del Q-learning, richiede di *enumerare* le azioni per trovare la migliore. Con
un joystick funziona; con uno sterzo, un acceleratore o sette giunti che si
muovono insieme, le combinazioni sono infinite e l'`argmax` diventa
intrattabile. Dall'altra parte, i metodi a gradiente di policy che abbiamo
visto (REINFORCE, actor-critic, A3C, PPO) gestiscono nativamente le azioni
continue, ma nella loro forma *on-policy* buttano via ogni esperienza dopo
averla usata una volta e soffrono di varianza elevata: per un robot vero, dove
ogni tentativo costa secondi di usura reale, è un lusso che non ci si può
permettere. Servono metodi che uniscano le due virtù: *azioni continue* e
*riuso dei dati* alla maniera off-policy di DQN.

## Il problema del controllo continuo

`````{tab} Elementare

Con poche azioni possibili, decidere è come scegliere da un menu: leggi il
"voto" di ogni piatto e prendi il migliore. Con le azioni continue il menu ha
infinite righe. Non puoi più leggerle tutte per trovare la riga con il voto più
alto: dovresti scorrere all'infinito.

La via d'uscita è smettere di cercare il massimo confrontando le opzioni e
tenere invece, accanto al "giudice" che assegna i voti, un secondo personaggio:
un *attore* che, guardando la situazione, propone direttamente la forza da
applicare. Il giudice non deve più scandagliare infinite possibilità: deve solo
dire all'attore se la mossa proposta è buona e in che direzione ritoccarla.

`````

`````{tab} Superiore

Nel controllo continuo lo spazio delle azioni è $\mathcal{A}\subseteq
\mathbb{R}^{d}$: un vettore di $d$ comandi reali (le coppie ai giunti, lo
sterzo, l'accelerazione). Il Q-learning sceglie l'azione con

$$
a^\star = \arg\max_{a\in\mathcal{A}} Q(s,a),
$$

un problema di ottimizzazione da risolvere *a ogni passo* e per ogni stato. Con
$\mathcal{A}$ discreto e piccolo è una scansione; con $\mathcal{A}$ continuo è
un'ottimizzazione non convessa in $\mathbb{R}^d$, impraticabile *online*. La
soluzione è approssimare quel massimo con una policy parametrica $\mu_\theta(s)$
che restituisce direttamente l'azione, addestrata in modo che $\mu_\theta(s)
\approx \arg\max_a Q(s,a)$. Vogliamo inoltre un metodo *off-policy*, che riusi
un buffer di esperienze passate come DQN {cite}`mnih2015human`, per essere
campione-efficiente {cite}`sutton2018reinforcement`.

`````

## DDPG: un attore deterministico guidato dal critico

Il primo algoritmo a chiudere il cerchio è **DDPG**, *Deep Deterministic Policy
Gradient*, presentato da Lillicrap e colleghi di DeepMind nel 2016
{cite}`lillicrap2016continuous`. Il nome dice quasi tutto: un attore
*deterministico* addestrato con un *gradiente di policy*, dentro l'impianto
off-policy di DQN.

L'idea è tenere due reti che collaborano. L'**attore** $\mu_\theta(s)$ prende
lo stato e sputa fuori un'azione precisa: non una distribuzione, ma
esattamente la forza da applicare. Il **critico** $Q_\phi(s,a)$ è il vecchio
Q-network, ma ora prende in ingresso *anche* l'azione (un vettore continuo) e
restituisce un solo numero: quanto vale quella coppia stato-azione. Il critico
impara esattamente come in DQN, minimizzando l'errore di Bellman contro un
bersaglio calcolato con reti *target* congelate; l'attore impara a proporre le
azioni che il critico premia di più.

`````{tab} Elementare

Come fa l'attore a "sapere" in che direzione muovere la forza? Immagina il
critico come un paesaggio di colline: per ogni azione possibile c'è
un'altezza, il suo valore. L'attore sta in un punto e vuole salire. Il
critico, oltre a dirgli l'altezza, gli indica la *pendenza*: "da qui,
aumentando un filo la coppia sul secondo giunto, sali". L'attore fa un
passettino in quella direzione. Ripetuto tante volte, l'attore scivola verso
la cima (cioè verso l'azione di valore massimo) senza mai dover provare tutte
le azioni una per una. È la differenza tra cercare la vetta a tentoni e
seguire la bussola della pendenza.

Per non restare fermo su ciò che già conosce, l'attore aggiunge alle sue azioni
un po' di **rumore** casuale: piccole spinte imprevedibili che lo fanno provare
varianti nuove. È l'equivalente continuo del "tirare a caso ogni tanto" che in
DQN serviva a esplorare.

`````

`````{tab} Superiore

L'attore massimizza il ritorno atteso $J(\theta)=\mathbb{E}_{s}[Q_\phi(s,
\mu_\theta(s))]$. Il suo gradiente è il **deterministic policy gradient**
(Silver et al., 2014), che si ottiene per regola della catena:

$$
\nabla_\theta J(\theta) =
\mathbb{E}_{s\sim D}\Big[\,
\nabla_a Q_\phi(s,a)\big|_{a=\mu_\theta(s)}\;
\nabla_\theta \mu_\theta(s)
\,\Big].
$$

Il primo fattore, $\nabla_a Q_\phi$, è la pendenza del critico *rispetto
all'azione*: dice come cambiare $a$ per aumentare il valore. Il secondo,
$\nabla_\theta \mu_\theta$, propaga quella direzione ai parametri dell'attore.
L'aspettazione è su stati campionati dal replay buffer $D$: DDPG è quindi
*off-policy*. Il critico si addestra sul bersaglio di Bellman

$$
y = r + \gamma\, Q_{\phi'}\!\big(s', \mu_{\theta'}(s')\big),
$$

dove $\phi'$ e $\theta'$ sono i parametri delle reti target, aggiornate con
uno scorrimento lento (*Polyak averaging*) $\phi' \leftarrow \tau\phi +
(1-\tau)\phi'$, con $\tau\ll 1$. L'esplorazione avviene aggiungendo rumore
all'azione in fase di raccolta, $a = \mu_\theta(s) + \mathcal{N}$: nel paper
originale un processo di Ornstein-Uhlenbeck (rumore temporalmente correlato,
utile in sistemi con inerzia), ma nella pratica un semplice rumore gaussiano
indipendente funziona altrettanto bene.

`````

## Perché DDPG è fragile

DDPG funziona, ma chi lo ha usato sul serio lo descrive come nervoso. Due
problemi ne minano la stabilità.

Il primo è la **sovrastima del valore**, lo stesso male che affliggeva DQN. Il
critico ha errori di stima in ogni direzione; l'attore, addestrato a cercare le
azioni che il critico valuta di più, si infila proprio dove il critico ha
sbagliato *per eccesso*. Quegli errori ottimistici vengono così selezionati,
amplificati e reimmessi nel bersaglio di Bellman, dove tendono ad accumularsi.
Il secondo è l'**ipersensibilità agli iperparametri**: piccole variazioni nei
tassi di apprendimento, nella scala del rumore o nella dimensione delle reti
possono fare la differenza tra un agente che impara a camminare e uno che crolla
a terra. Riprodurre gli stessi risultati, da un seme casuale all'altro, è
notoriamente difficile.

## TD3: tre correzioni chirurgiche

Nel 2018 Scott Fujimoto, Herke van Hoof e David Meger analizzano queste
patologie e propongono **TD3**, *Twin Delayed DDPG* {cite}`fujimoto2018addressing`.
Non è un algoritmo nuovo: è DDPG con tre accorgimenti mirati, ognuno rivolto a
un difetto preciso.

`````{tab} Elementare

**Due giudici, non uno.** Il primo trucco combatte l'ottimismo del critico
tenendo *due* critici indipendenti invece di uno, e fidandosi sempre del più
prudente: per calcolare il valore di riferimento si prende il **minimo** dei
due voti. Se un giudice si è illuso e ha dato un voto troppo alto, l'altro fa
da freno. È come chiedere un preventivo a due meccanici e regolarsi sul più
cauto: si sbaglia meno per eccesso.

**L'attore parla di meno.** Il secondo trucco è rallentare l'attore: i critici
si aggiornano a ogni passo, l'attore solo una volta ogni due. Prima di cambiare
strategia, conviene che i giudici abbiano le idee chiare; un attore che insegue
critici ancora confusi rincorre bersagli sbagliati.

**Bersagli sfumati.** Il terzo trucco aggiunge un pizzico di rumore all'azione
usata nel calcolo del bersaglio, così che azioni quasi identiche ricevano voti
quasi identici. Impedisce all'attore di aggrapparsi a un picco stretto e
probabilmente illusorio del critico.

`````

`````{tab} Superiore

**(a) Clipped double-Q.** Si mantengono due critici $Q_{\phi_1}, Q_{\phi_2}$
addestrati sullo stesso bersaglio, costruito con il *minimo* delle due reti
target:

$$
y = r + \gamma \min_{i=1,2} Q_{\phi'_i}\!\big(s', \tilde a'\big).
$$

Prendere il minimo introduce un bias *pessimista* che compensa la sovrastima:
poiché l'errore che si propaga è il più piccolo dei due, il valore non si
gonfia. **(b) Delayed policy updates.** L'attore e le reti target si aggiornano
ogni $d$ passi del critico (tipicamente $d=2$): riducendo la frequenza degli
aggiornamenti dell'attore si abbassa la varianza e si evita che insegua stime
ancora immature. **(c) Target policy smoothing.** L'azione target è
"regolarizzata" da rumore troncato,

$$
\tilde a' = \mu_{\theta'}(s') + \epsilon,
\qquad \epsilon \sim \operatorname{clip}\big(\mathcal{N}(0,\sigma),\,-c,\,c\big),
$$

così che il bersaglio sia liscio rispetto all'azione: previene lo
sfruttamento, da parte dell'attore, di picchi acuti ed erronei nella superficie
del critico. Con questi tre interventi TD3 supera nettamente DDPG sui benchmark
di controllo continuo, restando concettualmente lo stesso algoritmo.

`````

## SAC: esplorare restando il più imprevedibile possibile

Quasi in contemporanea con TD3, Tuomas Haarnoja e colleghi propongono una
filosofia diversa: **SAC**, *Soft Actor-Critic* {cite}`haarnoja2018soft`. Qui
l'attore torna **stocastico** (restituisce una distribuzione di probabilità
sulle azioni, non un singolo valore) e cambia l'obiettivo stesso
dell'apprendimento.

`````{tab} Elementare

Un agente che punta solo al premio tende a incaponirsi presto su un'unica
strategia: la prima che sembra funzionare. Se quella strategia era solo
mediocre, non se ne accorgerà più, perché ha smesso di provare alternative.

SAC cambia la regola del gioco. All'agente non chiede soltanto "massimizza il
premio", ma "massimizza il premio *restando il più imprevedibile possibile*".
A parità di ricompensa attesa, preferisce la condotta più varia, quella che
mantiene aperte più opzioni. Immagina di andare al lavoro sempre per la stessa
strada perché "funziona": non scoprirai mai la scorciatoia. Un pendolare che
ogni tanto cambia percorso, senza perdere troppo tempo, resta pronto a cogliere
la via migliore quando si presenta. Questa preferizione per la varietà si
regola con una manopola, la "temperatura": alta, l'agente esplora molto; bassa,
si concentra sul premio. SAC di solito gira quella manopola da solo, adattandola
durante l'addestramento.

`````

`````{tab} Superiore

SAC ottimizza l'obiettivo di **massima entropia**: al ritorno somma l'entropia
della policy in ogni stato,

$$
J(\pi) = \sum_{t} \mathbb{E}\Big[\, r_t + \alpha\, \mathcal{H}\big(\pi(\cdot\mid s_t)\big)\Big],
\qquad
\mathcal{H}\big(\pi(\cdot\mid s)\big) = -\,\mathbb{E}_{a\sim\pi}\big[\log \pi(a\mid s)\big].
$$

Il coefficiente $\alpha>0$ è la **temperatura**, che pesa quanto conta esplorare
rispetto allo sfruttare. L'entropia $\mathcal{H}$ è massima quando la policy è
il più possibile casuale: massimizzarla spinge l'agente a non collassare
prematuramente su un'unica azione, migliorando l'esplorazione e la robustezza.
Il critico impara un *soft* Q-value con bersaglio

$$
y = r + \gamma\Big(\min_{i=1,2} Q_{\phi'_i}(s', a') - \alpha \log \pi_\theta(a'\mid s')\Big),
\qquad a' \sim \pi_\theta(\cdot\mid s'),
$$

che usa, come TD3, il minimo dei due critici e aggiunge il termine di entropia
$-\alpha\log\pi_\theta$. La temperatura $\alpha$ non va fissata a mano: nella
versione matura di SAC è **auto-regolata**, aggiustata per mantenere l'entropia
della policy attorno a un valore-obiettivo. Il risultato è un algoritmo robusto
e campione-efficiente, oggi tra i più usati nel controllo continuo.

`````

## Lo scheletro dell'aggiornamento, in PyTorch

I tre algoritmi condividono lo stesso ciclo off-policy: si pesca un minibatch
dal replay buffer, si aggiorna il critico verso il bersaglio di Bellman e
l'attore verso l'azione che il critico premia. Ecco il cuore nella variante
DDPG, senza gli orpelli; TD3 aggiunge il secondo critico e il rumore sul
bersaglio, SAC il termine di entropia.

```{code-block} python
:class: pt-non-eseguibile

import torch
import torch.nn.functional as F

# reti gia definite: attore mu(s), critico q_net(s, a) e le loro copie target
# ottimizzatori: opt_critico (parametri di q_net), opt_attore (parametri di mu)
# minibatch dal replay buffer, come in DQN: tensori s, a, r, s_next, fine

# --- bersaglio di Bellman: non si deriva, usa le reti target ---
with torch.no_grad():
    a_next = mu_target(s_next)                     # azione greedy dell'attore target
    q_next = q_target(s_next, a_next)              # Q^-(s', mu^-(s'))
    y = r + gamma * q_next * (1 - fine)            # se terminale resta solo r

# --- aggiornamento del critico: avvicina Q(s, a) al bersaglio ---
q = q_net(s, a)                                    # Q sulle azioni realmente eseguite
perdita_critico = F.mse_loss(q, y)
opt_critico.zero_grad()
perdita_critico.backward()
opt_critico.step()

# --- aggiornamento dell'attore: sali lungo il gradiente del critico ---
perdita_attore = -q_net(s, mu(s)).mean()           # massimizza Q(s, mu(s))
opt_attore.zero_grad()
perdita_attore.backward()                          # il gradiente scorre da Q dentro mu
opt_attore.step()

# --- aggiornamento morbido (Polyak) delle reti target ---
with torch.no_grad():
    for p, p_t in zip(q_net.parameters(), q_target.parameters()):
        p_t.mul_(1 - tau).add_(tau * p)
    for p, p_t in zip(mu.parameters(), mu_target.parameters()):
        p_t.mul_(1 - tau).add_(tau * p)
```

Il segno meno nella `perdita_attore` è tutto ciò che serve: minimizzare
$-Q(s,\mu(s))$ equivale a *massimizzare* il valore, e la retropropagazione fa
scorrere la pendenza del critico $\nabla_a Q$ dentro i parametri dell'attore
(esattamente il deterministic policy gradient scritto sopra).

## Onestà sui limiti

Questi metodi off-policy sono molto più **campione-efficienti** di PPO e A3C:
riusando ogni transizione molte volte dal replay buffer, imparano da meno
interazioni con l'ambiente (decisivo quando ogni tentativo consuma un robot
vero). Il prezzo è la **stabilità**. DDPG, in particolare, è fragile e
capriccioso; TD3 e SAC lo domano, ma restano più delicati da mettere a punto
di un PPO ben tarato, che spesso si preferisce proprio perché "perdona" di
più. Non esiste il vincitore assoluto: la scelta dipende da quanto costano i
campioni e da quanta cura si può dedicare alla taratura.

C'è poi un limite che nessuno di questi algoritmi risolve da sé, il
**sim-to-real gap**. Addestrare un robot direttamente nel mondo fisico è lento
e rischioso, così quasi sempre si impara in simulazione, dove i campioni sono
abbondanti e le cadute non rompono nulla. Ma il simulatore non è la realtà:
attriti, ritardi dei motori, giochi meccanici e rumore dei sensori non
coincidono mai del tutto. Una policy perfetta nel simulatore può inciampare al
primo passo reale. Colmare quello scarto (con randomizzazione dei parametri
fisici, calibrazione, adattamento sul campo) è un problema di ricerca ancora
aperto, e ci ricorda che l'algoritmo di controllo è solo un pezzo del percorso
che porta un robot a muoversi nel mondo.

```{admonition} Da ricordare
:class: important
- Nel **controllo continuo** l'azione è un vettore reale: l'`argmax` di DQN è
  intrattabile. La soluzione è un **attore** $\mu_\theta(s)$ che propone
  l'azione e un **critico** $Q_\phi(s,a)$ che la valuta, in impianto off-policy.
- **DDPG** addestra l'attore deterministico con il *deterministic policy
  gradient* (il gradiente del critico rispetto all'azione), riusando replay
  buffer e reti target ereditati da DQN; esplora aggiungendo rumore all'azione.
- **TD3** corregge la sovrastima e l'instabilità di DDPG con tre accorgimenti:
  *twin critics* (minimo dei due $Q$), *delayed policy updates* e *target policy
  smoothing*.
- **SAC** adotta un attore stocastico e l'obiettivo di **massima entropia**
  (premio + entropia, con temperatura $\alpha$ spesso auto-regolata): esplora
  meglio ed è robusto e campione-efficiente.
- Off-policy significa **efficienza nei campioni** ma minore **stabilità** di
  PPO; e resta il **sim-to-real gap**, lo scarto tra simulazione e mondo fisico.
```
