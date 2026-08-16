# Controllo continuo: DDPG, TD3, SAC

Un joystick Atari ha nove posizioni: su, giù, sinistra, destra, le quattro
diagonali, il centro. Il Deep Q-Network sceglie fra queste guardando i voti di
tutte e nove e tenendo il più alto: nel codice l'operazione si chiama `argmax`,
e restituisce *quale* voce ha il voto massimo, non il voto. Ma prova a
immaginare un braccio robotico con sette articolazioni, o un robot a quattro
zampe che deve imparare a camminare. A ogni istante il controllore non decide
"sinistra o destra": decide *quanta spinta* dare a ciascun motore, un numero
con la virgola, magari negativo per frenare, magari $3{,}4$, magari $3{,}41$.
Non c'è un menu di mosse da scorrere: c'è un continuo di forze da dosare.

È lo scoglio annunciato in fondo alla sezione su DQN. Prendere il voto più alto
vuol dire scorrere le mosse una per una: con un joystick si può, con uno sterzo,
un acceleratore o sette giunti che si muovono insieme le combinazioni sono
infinite e non si scorre più niente.

I metodi a gradiente di policy della sezione precedente (REINFORCE,
attore-critico, A3C, PPO) su questo non hanno problemi: imparano a decidere,
non a votare, e una quantità da dosare la sanno produrre. Ma hanno due difetti
loro. Il primo: imparano soltanto dalla strategia che stanno giocando in quel
momento (in gergo sono *on-policy*, il contrario dell’*off-policy* di DQN), e
quindi ogni esperienza si usa una volta e poi si butta. Il secondo: il loro
apprendimento
**balla**, cioè la stessa strategia, rigiocata, dà correzioni molto diverse fra
loro. Per un robot vero, dove ogni tentativo costa secondi di usura reale, sono
due lussi che non ci si può permettere.

Servono metodi che uniscano le due virtù: mosse da dosare, come nei gradienti di
policy, e riuso delle esperienze passate, come nel quaderno di DQN.

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
\mathbb{R}^{n}$: un vettore di $n$ comandi reali (le coppie ai giunti, lo
sterzo, l'accelerazione). Il Q-learning sceglie l'azione con

$$
a^\star = \arg\max_{a\in\mathcal{A}} Q(s,a),
$$

un problema di ottimizzazione da risolvere *a ogni passo* e per ogni stato. Con
$\mathcal{A}$ discreto e piccolo è una scansione; con $\mathcal{A}$ continuo è
un'ottimizzazione non convessa in $\mathbb{R}^n$, impraticabile *online*. La
soluzione è approssimare quel massimo con una policy parametrica $\mu_\theta(s)$
che restituisce direttamente l'azione, addestrata in modo che $\mu_\theta(s)
\approx \arg\max_a Q(s,a)$. Vogliamo inoltre un metodo *off-policy*, che riusi
un buffer di esperienze passate come DQN {cite}`mnih2015human`, per essere
campione-efficiente {cite}`sutton2018reinforcement`.

`````

## DDPG: un attore deterministico guidato dal critico

Il primo algoritmo a tenere insieme le due virtù appena chieste è **DDPG**,
*Deep Deterministic Policy Gradient*, presentato da Lillicrap e colleghi di
DeepMind nel 2016 {cite}`lillicrap2016continuous`.

L'idea è tenere due reti che collaborano. L’**attore** guarda la situazione e
propone un'azione precisa: non un ventaglio di possibilità con le loro
probabilità, come faceva la strategia della sezione precedente, ma esattamente
la spinta da dare, un numero per ciascun motore. È **deterministica**, cioè
nella stessa situazione risponde sempre la stessa cosa, senza tirare dadi: da lì
la seconda D del nome. Il **critico** è la vecchia rete dei voti di DQN con una
modifica: oltre alla situazione riceve in ingresso *anche* l'azione proposta, e
restituisce un numero solo, quanto vale fare quella mossa lì.

Il critico impara come in DQN, inseguendo un **bersaglio**, cioè il voto che
quella mossa dovrebbe avere secondo i conti del momento; e come in DQN quel
bersaglio si calcola con delle copie congelate delle due reti, per la stessa
ragione di allora, cioè perché un bersaglio che si sposta insieme a chi lo
insegue non si raggiunge mai. L'attore, dal canto suo, impara a proporre le
azioni che il critico premia di più.

`````{tab} Elementare

Come fa l'attore a "sapere" in che direzione muovere la forza? Immagina il
critico come un paesaggio di colline: per ogni azione possibile c'è
un'altezza, il suo valore. L'attore sta in un punto e vuole salire. Il
critico, oltre a dirgli l'altezza, gli indica la *pendenza*: "da qui,
spingendo un filo di più sul secondo giunto, sali". L'attore fa un
passettino in quella direzione. Ripetuto tante volte, l'attore scivola verso
la cima (cioè verso l'azione di valore massimo) senza mai dover provare tutte
le azioni una per una. È la differenza tra cercare la vetta a tentoni e
seguire la bussola della pendenza.

Per non restare fermo su ciò che già conosce, l'attore aggiunge alle sue azioni
un po’ di **rumore** casuale: piccole spinte imprevedibili che lo fanno provare
varianti nuove. È l'equivalente continuo del "ogni tanto tira a caso invece di
prendere la mossa migliore" con cui il Q-learning del capitolo precedente
esplorava.

`````

`````{tab} Superiore

In simboli: l'attore è una policy **deterministica** $\mu_\theta(s)$, che
restituisce direttamente il vettore delle azioni invece di una distribuzione su
di esse; il critico è $Q_\phi(s,a)$, con l'azione fra gli ingressi. L'attore
massimizza il ritorno atteso $J(\theta)=\mathbb{E}_{s}[Q_\phi(s,
\mu_\theta(s))]$, e il suo gradiente è il **deterministic policy gradient**
{cite}`silver2014deterministic`, che si ottiene per regola della catena:

$$
\nabla_\theta J(\theta) =
\mathbb{E}_{s\sim \mathcal{D}}\Big[\,
\nabla_a Q_\phi(s,a)\big|_{a=\mu_\theta(s)}\;
\nabla_\theta \mu_\theta(s)
\,\Big].
$$

Il primo fattore, $\nabla_a Q_\phi$, è la pendenza del critico *rispetto
all'azione*: dice come cambiare $a$ per aumentare il valore. Il secondo,
$\nabla_\theta \mu_\theta$, propaga quella direzione ai parametri dell'attore.

Un passaggio, qui, è un'approssimazione e non un'uguaglianza, e conviene non
farselo scivolare addosso. Il teorema vale per stati distribuiti secondo
$\rho^\mu$, cioè secondo la policy **corrente**; scriverci sotto $s\sim\mathcal{D}$,
gli stati del replay buffer raccolti da policy vecchie, è la mossa che rende DDPG
*off-policy* e costa un termine che si butta via. Funziona, ma non discende dalla
regola della catena: è una scelta, ed è la stessa che si fa in tutti i metodi
attore-critico off-policy. Il critico si addestra sul bersaglio di Bellman

$$
y = r + \gamma\, Q_{\phi'}\!\big(s', \mu_{\theta'}(s')\big),
$$

dove $\phi'$ e $\theta'$ sono i parametri delle reti target, aggiornate con
uno scorrimento lento (*Polyak averaging*) $\phi’ \leftarrow \tau\phi +
(1-\tau)\phi’$, con $\tau\ll 1$. L'esplorazione avviene aggiungendo rumore
all'azione in fase di raccolta, $a = \mu_\theta(s) + \epsilon$: nel paper
originale $\epsilon$ è un processo di Ornstein-Uhlenbeck (rumore temporalmente
correlato, utile in sistemi con inerzia), ma nella pratica un semplice rumore
gaussiano indipendente funziona altrettanto bene.

`````

## Perché DDPG è fragile

DDPG funziona, ma chi lo ha usato sul serio lo descrive come nervoso. Due
problemi ne minano la stabilità.

Il primo è la **sovrastima del valore**, lo stesso male che affliggeva DQN. Il
critico ha errori di stima in ogni direzione; l'attore, addestrato a cercare le
azioni che il critico valuta di più, si infila proprio dove il critico ha
sbagliato *per eccesso*. Quegli errori ottimistici vengono così selezionati,
amplificati e reimmessi nel bersaglio che il critico insegue, dove tendono ad
accumularsi. Il secondo è l’**ipersensibilità agli iperparametri**, cioè alle
manopole che si decidono prima di cominciare e non si imparano: la velocità con
cui le reti si correggono, quanto rumore aggiungere, quanto farle grandi.
Ritoccarne una di poco può fare la differenza fra un agente che impara a
camminare e uno che crolla a terra. E non serve nemmeno ritoccarla: basta
rilanciare lo stesso identico addestramento cambiando il seme, cioè il numero da
cui parte il sorteggio interno, e i risultati possono essere molto diversi. È la
ragione per cui in questo campo un risultato si riporta su molte ripetizioni,
come abbiamo appena fatto con la ricerca ad albero: una prova sola non dice
quasi niente.

## TD3: tre correzioni chirurgiche

Nel 2018 Scott Fujimoto, Herke van Hoof e David Meger analizzano queste
patologie e propongono **TD3**, *Twin Delayed DDPG* {cite}`fujimoto2018addressing`.
Quel «TD» non ha niente a che vedere con le differenze temporali del capitolo
precedente: sta per *Twin Delayed*, «gemello e ritardato», e i due aggettivi
dicono già due dei tre accorgimenti. Non è un algoritmo nuovo: è DDPG con tre
correzioni mirate, ognuna rivolta a un difetto preciso.

`````{tab} Elementare

**Due giudici, non uno.** Il primo trucco combatte l'ottimismo del critico
tenendo *due* critici invece di uno, e fidandosi sempre del più prudente: per
calcolare il valore di riferimento si prende il **minimo** dei due voti. Se un
giudice si è illuso e ha dato un voto troppo alto, l'altro fa da freno. È come
chiedere un preventivo a due meccanici e regolarsi sul più cauto: si sbaglia
meno per eccesso.

E vale l'avvertenza già vista per il Double DQN, perché è la stessa: i due
giudici non sono estranei fra loro, hanno studiato sugli stessi dati e inseguito
lo stesso bersaglio, quindi tendono a illudersi insieme. Il minimo attenua, non
guarisce, e semmai sposta il difetto: al posto di un voto un po’ troppo alto se
ne prende uno un po’ troppo basso.

**L'attore parla di meno.** Il secondo trucco è rallentare l'attore: i critici
si aggiornano a ogni passo, l'attore solo una volta ogni due. Prima di cambiare
strategia, conviene che i giudici abbiano le idee chiare; un attore che insegue
critici ancora confusi rincorre bersagli sbagliati.

**Bersagli sfumati.** Il terzo trucco aggiunge un pizzico di rumore all'azione
usata nel calcolo del voto di riferimento, così che azioni quasi identiche
ricevano voti quasi identici. Impedisce all'attore di aggrapparsi a un picco
stretto e probabilmente illusorio del critico.

Va detto anche su che cosa questi tre trucchi non promettono niente. Dei due
difetti elencati poco fa attaccano il primo, l'ottimismo dei voti, e lo
attaccano in due: i due giudici e i bersagli sfumati. L'attore che parla di meno
cura invece un difetto in più, che nell'elenco non c'era, cioè l'attore che
insegue giudizi ancora acerbi. Sul secondo difetto, la sensibilità alle manopole,
TD3 non dice nulla: l'addestramento è meno nervoso e quindi se ne soffre meno,
ma il problema è ancora tutto lì.

`````

`````{tab} Superiore

**(a) Clipped double-Q.** Si mantengono due critici $Q_{\phi_1}, Q_{\phi_2}$
addestrati sullo stesso bersaglio, costruito con il *minimo* delle due reti
target:

$$
y = r + \gamma \min_{i=1,2} Q_{\phi'_i}\!\big(s', \tilde a'\big).
$$

Prendere il minimo introduce un bias *pessimista* che compensa la sovrastima:
poiché l'errore che si propaga è il più piccolo dei due, il valore tende a non
gonfiarsi. Vale però lo stesso caveat visto per il Double DQN, ed è la stessa
ragione: i due critici sono addestrati sullo **stesso** bersaglio e sugli
**stessi** dati, quindi i loro errori sono correlati, e il minimo di due stime
correlate non elimina il bias, lo sposta, scambiando tipicamente una sovrastima
con una moderata sottostima. È un correttivo che funziona in pratica, non una
cura. **(b) Delayed policy updates.** L'attore e le reti target si aggiornano
ogni $d$ passi del critico (tipicamente $d=2$): riducendo la frequenza degli
aggiornamenti dell'attore si abbassa la varianza e si evita che insegua stime
ancora immature. **(c) Target policy smoothing.** L'azione target è
"regolarizzata" da rumore troncato,

$$
\tilde a' = \mu_{\theta'}(s') + \epsilon,
\qquad \epsilon \sim \operatorname{clip}\big(\mathcal{N}(0,\sigma^2),\,-c,\,c\big),
$$

così che il bersaglio sia liscio rispetto all'azione: previene lo
sfruttamento, da parte dell'attore, di picchi acuti ed erronei nella superficie
del critico. Dei due difetti elencati sopra, TD3 attacca frontalmente **il
primo**, la sovrastima, con il clipped double-Q e il target smoothing; il *delayed
policy update* cura un difetto che sopra non era in elenco e va aggiunto, cioè
l'attore che insegue stime ancora immature. Sull'ipersensibilità agli
iperparametri, invece, TD3 non promette nulla: ne attenua i sintomi perché
l'addestramento è meno nervoso, non perché il problema sia risolto. Il valore
dell'algoritmo sta tutto lì: resta concettualmente DDPG, e dove DDPG è nervoso in
genere non lo è.

`````

## SAC: esplorare restando il più imprevedibile possibile

Quasi in contemporanea con TD3, Tuomas Haarnoja e colleghi propongono una
filosofia diversa: **SAC**, *Soft Actor-Critic* {cite}`haarnoja2018soft`. Qui
l'attore torna **stocastico**, cioè l'opposto di deterministico: invece di una
sola spinta restituisce un ventaglio di spinte possibili con le loro
probabilità, e la mossa vera la si estrae da lì. E cambia l'obiettivo stesso
dell'apprendimento. Il *soft* del nome vuol dire «morbido», ed è un'allusione
proprio a questo: dove prima l'agente puntava tutto sulla mossa migliore, adesso
tiene aperto un ventaglio.

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
la via migliore quando si presenta. Questa preferenza per la varietà si
regola con una manopola, la "temperatura": alta, l'agente esplora molto; bassa,
si concentra sul premio. Il nome viene dalla fisica, e l'immagine è quella
giusta: più la temperatura è alta, più le cose si agitano e si mescolano; più è
bassa, più tutto si posa in un'unica configurazione. SAC di solito gira quella
manopola da solo, adattandola durante l'addestramento.

`````

`````{tab} Superiore

SAC ottimizza l'obiettivo di **massima entropia**: al ritorno somma l'entropia
della policy in ogni stato,

$$
J(\pi) = \sum_{t=0}^{T} \mathbb{E}\Big[\, r_t + \alpha\, \mathcal{H}\big(\pi(\cdot\mid s_t)\big)\Big],
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
versione matura di SAC è **auto-regolata**, ricavata risolvendo un problema
vincolato che chiede all'entropia media della policy di non scendere sotto un
valore-obiettivo $\mathcal{H}_0$. Il risultato è un algoritmo robusto e
campione-efficiente, e la ragione della sua fortuna è precisamente questa:
l'esplorazione smette di essere un parametro da indovinare a mano e diventa una
conseguenza dell'obiettivo.

`````

## Lo scheletro dell'aggiornamento, in PyTorch

I tre algoritmi condividono lo stesso ciclo off-policy: si pesca un pugno di
esperienze passate dal quaderno del replay, si aggiorna il critico verso il
bersaglio che insegue e l'attore verso l'azione che il critico premia. Ecco il
cuore nella variante DDPG, senza gli orpelli; TD3 aggiunge il secondo critico e
il rumore sul bersaglio, SAC il premio alla varietà, che in termini tecnici è
l’**entropia** della policy, cioè quanto le sue scelte restano imprevedibili: è
esattamente ciò che la manopola della temperatura dosa.

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
# ATTENZIONE alla forma: q e y devono essere entrambi (B,). Se q_net restituisce
# (B, 1) e y e' (B,), mse_loss non solleva niente: stampa un UserWarning, fa
# broadcasting a (B, B) e minimizza la loss sbagliata. E chi non legge i warning
# non se ne accorge: e' l'errore piu' comune nelle implementazioni di DDPG.
perdita_critico = F.mse_loss(q.squeeze(-1), y)
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

Il segno meno nella `perdita_attore` è tutto ciò che serve. L'ottimizzatore (il
pezzo di libreria che ritocca i pesi a ogni passo, qui `opt_attore`) sa fare una
cosa sola, *far scendere* il numero che gli si dà: quindi per far salire il voto
del critico gli si dà da far scendere quel voto cambiato di segno. Il resto lo fa la
retropropagazione, cioè il meccanismo con cui una rete si corregge partendo
dall'errore in uscita e risalendo verso i pesi: qui parte dal voto, attraversa
il critico, arriva all'azione, e da lì entra nei parametri dell'attore. È
esattamente il meccanismo raccontato all'inizio della sezione, quello per cui il
critico non dice solo *quanto vale* l'azione ma anche *da che parte* spostarla
per farla valere di più.

## Onestà sui limiti

Questi tre metodi hanno un pregio grosso: riusano ogni esperienza molte volte,
pescandola dal quaderno, e quindi imparano da molte meno prove nel mondo. È
decisivo quando ogni tentativo consuma un robot vero. Il prezzo è la
**stabilità**. DDPG, in particolare, è fragile e capriccioso; TD3 e SAC lo
domano, ma restano più delicati da mettere a punto di un PPO ben tarato (PPO è
l'algoritmo della sezione precedente, quello che «perdona» gli errori di
taratura), e per questo spesso si preferisce lui. Non esiste il vincitore
assoluto: la scelta dipende da quanto costa una prova e da quanta cura si può
dedicare alla messa a punto.

C'è poi un limite che nessuno di questi algoritmi risolve da sé, il
**sim-to-real gap**, lo scarto fra simulazione e mondo fisico. Addestrare un
robot direttamente nel mondo fisico è lento e rischioso, così quasi sempre si
impara in simulazione, dove le prove sono infinite e le cadute non rompono
nulla. Ma il simulatore non è la realtà: attriti, ritardi dei motori, giochi
meccanici e rumore dei sensori non coincidono mai del tutto. Una strategia
perfetta nel simulatore può inciampare al primo passo reale. Colmare quello scarto (con randomizzazione dei parametri
fisici, calibrazione, adattamento sul campo) è un problema di ricerca ancora
aperto, e ci ricorda che l'algoritmo di controllo è solo un pezzo del percorso
che porta un robot a muoversi nel mondo.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Nel **controllo continuo** l'azione non è una voce da scegliere in un menu,
  è una quantità da dosare (quanta forza a ciascun motore) e il menu ha
  infinite righe: scorrerle tutte, come fa DQN, non si può. La via d'uscita è
  affiancare al giudice che assegna i voti un **attore** che propone
  direttamente la mossa; al giudice resta da dire se è buona e da che parte
  ritoccarla.
- **DDPG** insegna all'attore a seguire la *pendenza* indicata dal critico,
  come chi sale una collina con la bussola invece che a tentoni; riusa il
  quaderno delle esperienze passate e le copie congelate delle reti ereditate
  da DQN, ed esplora aggiungendo un po’ di rumore casuale alle proprie mosse.
- DDPG è nervoso e si lascia illudere dai voti troppo alti. **TD3** lo
  corregge con tre accorgimenti: due giudici invece di uno, e ci si regola sul
  più prudente; l'attore cambia strategia una volta ogni due aggiornamenti dei
  giudici; il voto di riferimento viene sfumato con un pizzico di rumore, così
  l'attore non si aggrappa a un picco stretto e probabilmente illusorio.
- **SAC** cambia l'obiettivo del gioco: non solo il massimo premio, ma il
  massimo premio *restando il più imprevedibile possibile*, come il pendolare
  che ogni tanto cambia strada e per questo scopre la scorciatoia. Quanto
  contare la varietà è una manopola, che di solito l'algoritmo gira da sé.
- Riusare le esperienze già vissute fa imparare con molti meno tentativi
  (decisivo quando ogni prova consuma un robot vero), ma rende l'addestramento
  più delicato da tarare rispetto a PPO. E resta lo scarto fra simulatore e
  mondo fisico: una strategia perfetta in simulazione può inciampare al primo
  passo reale.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Nel **controllo continuo** l'azione è un vettore reale: l’`argmax` di DQN è
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
`````
