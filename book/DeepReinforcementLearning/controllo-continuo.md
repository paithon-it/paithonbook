# Controllo continuo: DDPG, TD3, SAC

Dopo la ricerca, che pensa prima di muovere, si torna ai metodi che decidono
d'istinto. Un joystick Atari ha nove posizioni (su, giù, sinistra, destra, le
quattro diagonali, il centro), che con il pulsante diventano diciotto azioni, e
molti giochi ne usano meno; *Breakout*, ad esempio, ne usa quattro. Il Deep
Q-Network sceglie fra le azioni del gioco guardando il valore di ciascuna e
tenendo il più alto. Nel codice l'operazione si chiama `argmax`, e restituisce
*quale* voce ha il valore massimo, non il valore.

Considera invece un braccio robotico con sette giunti o un robot a quattro
zampe che deve imparare a camminare. A ogni istante chi lo comanda non decide
"sinistra o destra", ma *quanta spinta* dare a ciascun motore: un numero con la
virgola, magari negativo per frenare, magari $3{,}4$, magari $3{,}41$. In altri
termini, non c'è un menu di azioni da scorrere, ma un continuo di forze da
dosare.

È lo scoglio annunciato in fondo alla {doc}`sezione su DQN <dqn>`. Per prendere
il voto più alto, infatti, bisogna scorrere le azioni una per una. Con un
joystick si può, ma con uno sterzo, un acceleratore o sette giunti che si
muovono insieme le combinazioni sono infinite, e non si scorre più niente.

I metodi a {doc}`gradiente di policy <policy-gradient>` (REINFORCE,
attore-critico, A3C, PPO) su questo non hanno problemi, perché imparano
direttamente la policy e sanno produrre una quantità da dosare.

Hanno però due limiti. Il primo è che sono *on-policy*, il contrario
dell’*off-policy* di DQN: ogni esperienza serve finché la policy che l'ha
prodotta è ancora quella di adesso, cioè per pochi aggiornamenti (PPO la riusa
per qualche passata, non di più), e poi si butta. Il secondo è che le loro
stime hanno varianza alta: la stessa policy, rigiocata, dà gradienti molto
diversi fra loro. Su un robot vero, dove ogni tentativo costa tempo e usura,
sono due svantaggi pesanti.

Servono quindi metodi che uniscano le due virtù: azioni da dosare, come nei
gradienti di policy, e riuso delle esperienze passate con la memoria di replay,
come in DQN.

## Il problema del controllo continuo

`````{tab} Elementare

Con poche azioni possibili, decidere è come scegliere da un menu: leggi il
"voto" di ogni piatto e prendi il migliore. Con le azioni continue il menu ha
infinite righe. Non puoi più leggerle tutte per trovare la riga con il voto più
alto, perché dovresti scorrere all'infinito.

La via d'uscita è smettere di cercare il massimo confrontando le opzioni e
tenere invece, accanto al "giudice" che assegna i voti, un secondo personaggio:
un *attore* che, guardando la situazione, propone direttamente la forza da
applicare. In questo modo il giudice non deve più scandagliare infinite
possibilità, ma solo dire all'attore se l'azione proposta è buona e in che
direzione ritoccarla.

C'è poi una seconda pretesa, che con il menu non c'entra: le prove già fatte non
si buttano. Ogni azione tentata, con il suo esito, finisce su un
quaderno, e da lì la si ripesca per imparare ancora, molto tempo dopo, quando
l'attore ha già cambiato modo di giocare. Per un robot vero, dove ogni
tentativo consuma cinghie e ingranaggi, il quaderno rende
praticabile l'apprendimento.

`````

`````{tab} Superiore

Nel controllo continuo lo spazio delle azioni è $\mathcal{A}\subseteq
\mathbb{R}^{n}$, e ogni azione è un vettore di $n$ comandi reali (le coppie ai
giunti, lo sterzo, l'accelerazione). Il Q-learning sceglie l'azione con

$$
\mathbf{a}^\star = \arg\max_{\mathbf{a}\in\mathcal{A}} Q(s,\mathbf{a}),
$$

un problema di ottimizzazione da risolvere *a ogni passo* e per ogni stato. Con
$\mathcal{A}$ discreto e piccolo è una scansione; con $\mathcal{A}$ continuo è
un'ottimizzazione non convessa in $\mathbb{R}^n$, impraticabile *online*.

Le alternative che conservano l’$\arg\max$ lo rendono trattabile in due modi:
restringendo $Q$ a una forma quadratica nell'azione, dove il massimo si scrive
in forma chiusa (NAF {cite}`gu2016continuous`), oppure cercandolo per
campionamento con il *cross-entropy method* (QT-Opt
{cite}`kalashnikov2018qtopt`). Discretizzare ciascuna delle $n$ componenti in
$k$ valori, invece, dà $k^n$ azioni da scorrere.

La via più seguita è un'altra: approssimare quel massimo con una policy
parametrica $\boldsymbol{\mu}_\theta(s)$ che restituisce direttamente l'azione,
addestrata in modo che $\boldsymbol{\mu}_\theta(s) \approx
\arg\max_{\mathbf{a}} Q(s,\mathbf{a})$. Vogliamo inoltre un metodo
*off-policy*, che riusi un buffer di esperienze passate come DQN
{cite}`mnih2015human`, per garantire un uso efficiente dei campioni
{cite}`sutton2018reinforcement`.

`````

## DDPG: un attore deterministico guidato dal critico

Il primo algoritmo a unire queste due proprietà con reti
profonde, su compiti di controllo con molti gradi di libertà, è **DDPG**,
*Deep Deterministic Policy Gradient*, presentato da Lillicrap e colleghi di
DeepMind nel 2016 {cite}`lillicrap2016continuous`. L'attore-critico con
policy deterministica esisteva già, anche con reti neurali piccole; il
contributo di DDPG è averlo reso stabile su reti profonde portandoci gli
accorgimenti di DQN.

L'idea è tenere due reti che collaborano. L’attore guarda la situazione e
propone un'azione precisa: non un ventaglio di possibilità con le loro
probabilità, come facevano le policy del gradiente di policy, ma la spinta da
dare, un numero per ciascun motore. La policy, in altre parole, è
deterministica: nella stessa situazione produce sempre la stessa azione, senza
tirare dadi; da qui la seconda D della sigla.

Il critico è la vecchia rete dei voti di DQN con una modifica: oltre alla
situazione riceve in ingresso *anche* l'azione proposta, e restituisce un
numero solo, quanto vale quell'azione.

Il critico impara come in DQN, inseguendo un bersaglio, cioè il voto che
quell'azione dovrebbe avere secondo le stime correnti. Anche qui il bersaglio
si calcola con copie congelate delle due reti, per la stessa ragione di allora:
un bersaglio che si sposta insieme a chi lo insegue non si raggiunge mai.
L'attore, dal canto suo, impara a proporre le azioni che il critico premia di
più.

`````{tab} Elementare

Come fa l'attore a "sapere" in che direzione muovere la forza? Il critico è un
paesaggio di colline: per ogni azione possibile c'è un'altezza, il suo valore.
L'attore sta in un punto e vuole salire. Il critico, oltre a dirgli l'altezza,
gli indica la *pendenza*: "da qui, spingendo un filo di più sul secondo giunto,
sali".

L'attore fa un passettino in quella direzione e, passo dopo passo, arriva in
cima alla collina su cui si trova, senza aver mai dovuto provare tutte le
azioni una per una. È la differenza tra cercare la vetta a tentoni e seguire la
bussola della pendenza.

La bussola, però, dice da che parte si sale da qui, non dove sta la vetta più
alta del paesaggio. Se una collina più alta comincia dall'altra parte della
valle, salendo non ci si arriva, perché per raggiungerla bisognerebbe prima
scendere, e la pendenza dice sempre di salire.

Per non restare fermo su ciò che già conosce, l'attore aggiunge alle sue azioni
un po’ di rumore casuale: piccole spinte imprevedibili che lo fanno provare
varianti nuove. È l'equivalente continuo del «ogni tanto tira a caso invece di
prendere l'azione migliore» con cui esplorava il
{doc}`Q-learning </ReinforcementLearning/q-learning>`.

L'attore, poi, misura le pendenze nei punti
pescati dal quaderno, cioè dove era finito quando giocava in un altro modo, e
non dove passa adesso. Il
conto fatto fino in fondo vorrebbe le pendenze dei posti di oggi; si usano
quelle di ieri perché sono già scritte, e in cambio ci si accontenta di una
direzione buona invece che esatta. Funziona bene, ma nessun conto lo
garantisce: è una scelta pratica.

`````

`````{tab} Superiore

In simboli, l'attore è una policy deterministica $\boldsymbol{\mu}_\theta(s)$,
che restituisce direttamente il vettore delle azioni invece di una distribuzione
su di esse; il critico è $Q_\phi(s,\mathbf{a})$, con l'azione fra gli ingressi.
L'obiettivo vero è il ritorno atteso di $\boldsymbol{\mu}_\theta$, e il
**deterministic policy gradient theorem** {cite}`silver2014deterministic` ne dà
il gradiente come
$\mathbb{E}_{s\sim\rho^{\mu}}\big[\nabla_{\mathbf{a}} Q^{\mu}(s,\mathbf{a})
\big|_{\mathbf{a}=\boldsymbol{\mu}_\theta(s)}
\nabla_\theta\boldsymbol{\mu}_\theta(s)\big]$,
con $\rho^\mu$ la distribuzione scontata degli stati che
$\boldsymbol{\mu}_\theta$ visita e $Q^\mu$ il suo valore vero. Come
nel teorema del gradiente di policy stocastico, la dipendenza degli stati
visitati da $\theta$ non va derivata.

DDPG sostituisce $Q^\mu$ con il critico e $\rho^\mu$
con il buffer, e sul surrogato
$\hat J(\theta)=
\mathbb{E}_{s\sim\mathcal{D}}[Q_\phi(s,\boldsymbol{\mu}_\theta(s))]$
la regola della catena dà:

$$
\nabla_\theta \hat J(\theta) =
\mathbb{E}_{s\sim \mathcal{D}}\Big[\,
\nabla_{\mathbf{a}} Q_\phi(s,\mathbf{a})
\big|_{\mathbf{a}=\boldsymbol{\mu}_\theta(s)}\;
\nabla_\theta \boldsymbol{\mu}_\theta(s)
\,\Big].
$$

Il primo fattore, $\nabla_{\mathbf{a}} Q_\phi$, è la pendenza del critico
*rispetto all'azione*, e dice come cambiare $\mathbf{a}$ per aumentare il
valore. Il secondo, $\nabla_\theta \boldsymbol{\mu}_\theta$, propaga quella
direzione ai parametri dell'attore.

Osserviamo però che un passaggio è un'approssimazione e non un'uguaglianza. Il
teorema, infatti, vale per stati distribuiti secondo $\rho^\mu$, cioè secondo
la policy corrente; sostituirvi gli stati del replay buffer,
$s\sim\mathcal{D}$, raccolti da policy precedenti, rende DDPG *off-policy*
omettendo un termine del gradiente. Funziona, ma non discende dalla regola
della catena: è una scelta, la stessa di tutti i metodi attore-critico
off-policy.

Il critico si addestra sul bersaglio di Bellman

$$
y = r + \gamma\, Q_{\phi'}\!\big(s', \boldsymbol{\mu}_{\theta'}(s')\big),
$$

dove $\phi'$ e $\theta'$ sono i parametri delle reti target, aggiornate con uno
scorrimento lento (*Polyak averaging*)
$\phi' \leftarrow \tau\phi + (1-\tau)\phi'$, con $\tau\ll 1$ (questo $\tau$ è un
numero, il peso dello scorrimento, e non ha niente a che vedere con la
traiettoria $\tau$ del gradiente di policy).

L'esplorazione avviene aggiungendo rumore all'azione in fase di raccolta,
$\mathbf{a} = \boldsymbol{\mu}_\theta(s) + \boldsymbol{\epsilon}$.
Nell'articolo originale $\boldsymbol{\epsilon}$ è un processo di
Ornstein-Uhlenbeck (rumore temporalmente correlato, utile in sistemi con
inerzia), ma Fujimoto e colleghi non trovano alcun vantaggio nel rumore
correlato, e un semplice rumore gaussiano indipendente basta
{cite}`fujimoto2018addressing`.

`````

## Perché DDPG è fragile

DDPG funziona, ma è fragile, e due problemi ne minano la stabilità. Il primo è
la sovrastima del valore, lo stesso male che affliggeva DQN. Il critico,
infatti, ha errori di stima in ogni direzione, e l'attore, addestrato a cercare
le azioni che il critico valuta di più, si infila dove il critico ha sbagliato
*per eccesso*. Quegli errori ottimistici vengono così selezionati, amplificati
e reimmessi nel bersaglio che il critico insegue, dove tendono ad accumularsi.

Il secondo è l’ipersensibilità agli iperparametri, cioè ai valori che si
decidono prima di cominciare e non si imparano: la velocità con cui le reti si
correggono, quanto rumore aggiungere, quanto farle grandi. Ritoccarne uno di
poco può fare la differenza fra un agente che impara a camminare e uno che
crolla a terra. Non serve nemmeno ritoccarlo; basta rilanciare lo stesso
addestramento cambiando soltanto il seme, e i risultati possono essere
molto diversi. Lo hanno misurato Henderson e colleghi mettendo a confronto
DDPG, TRPO, PPO e ACKTR: a parità di tutto il resto, gruppi di semi diversi
danno curve di apprendimento statisticamente diverse {cite}`henderson2018deep`.

## TD3: tre correzioni chirurgiche

Nel 2018 Scott Fujimoto, Herke van Hoof e David Meger analizzano queste
patologie e propongono **TD3**, *Twin Delayed DDPG*
{cite}`fujimoto2018addressing`. Si noti che quel «TD» non ha niente a che
vedere con le differenze temporali (*temporal difference*) del Q-learning; sta
per *Twin Delayed*, «gemello e ritardato», e i due aggettivi dicono già due dei
tre accorgimenti. È DDPG con tre correzioni mirate, ciascuna rivolta a un
difetto preciso, e non un algoritmo nuovo.

`````{tab} Elementare

Il primo trucco combatte l'ottimismo del critico tenendo *due* critici invece
di uno, e fidandosi sempre del più prudente: per calcolare il valore di
riferimento si prende il minimo dei due voti. Così, se un giudice si è illuso e
ha dato un voto troppo alto, l'altro fa da freno. È come far stimare la propria
casa da due agenti immobiliari e fidarsi di chi dice la cifra più bassa: ci si
illude di meno.

Vale però l'avvertenza già vista per il Double DQN. I due
giudici non sono estranei fra loro, hanno studiato sugli stessi dati e inseguito
lo stesso bersaglio, quindi tendono a illudersi insieme. Il minimo attenua, non
guarisce, e semmai sposta il difetto: al posto di un voto un po’ troppo alto se
ne prende uno un po’ troppo basso.

Il secondo trucco è rallentare l'attore: i critici si aggiornano a ogni passo,
l'attore solo una volta ogni due. Prima di cambiare strategia, conviene che i
giudici abbiano le idee chiare, perché un attore che insegue critici ancora
confusi rincorre bersagli sbagliati.

Il terzo trucco aggiunge un pizzico di rumore all'azione usata nel calcolo del
voto di riferimento, così che azioni quasi identiche ricevano voti quasi
identici. In questo modo impedisce all'attore di aggrapparsi a un picco stretto
e probabilmente illusorio del critico.

In sintesi, dei due problemi di DDPG i tre trucchi attaccano l'ottimismo dei
voti, e lo attaccano in due (i due giudici e i bersagli sfumati). L'attore che
parla di meno cura un problema che nell'elenco non c'era, cioè l'attore che
insegue giudizi ancora acerbi. Sulla sensibilità agli iperparametri, invece,
TD3 non promette niente. L'addestramento è meno nervoso e quindi se ne soffre
meno, ma il problema resta.

`````

`````{tab} Superiore

**(a) Clipped double-Q.** Si mantengono due critici $Q_{\phi_1}, Q_{\phi_2}$
addestrati sullo stesso bersaglio, costruito con il *minimo* delle due reti
target:

$$
y = r + \gamma \min_{i=1,2} Q_{\phi'_i}\!\big(s', \tilde{\mathbf{a}}'\big),
$$

dove $\tilde{\mathbf{a}}'$ è l'azione dell'attore target *sfumata dal rumore*,
definita dal *target policy smoothing*.

Prendere il minimo introduce un bias *pessimista* che compensa la sovrastima:
poiché l'errore che si propaga è il più piccolo dei due, il valore tende a non
gonfiarsi. Vale però l'avvertenza già vista per il Double DQN, perché i due
critici sono addestrati sullo stesso bersaglio e sugli stessi dati, e i loro
errori sono correlati. Il minimo di due stime correlate non elimina quindi il
bias, ma lo sposta, scambiando in genere una sovrastima con una moderata
sottostima. È un correttivo che funziona in pratica, non una cura.

**(b) Delayed policy updates.** L'attore e le reti target si aggiornano ogni $d$
passi del critico (tipicamente $d=2$). Riducendo la frequenza degli
aggiornamenti dell'attore si abbassa la varianza e si evita che insegua stime
ancora immature. L'attore, poi, sale lungo il gradiente di $Q_{\phi_1}$
soltanto, non del minimo dei due critici.

**(c) Target policy smoothing.**
L'azione target è "regolarizzata" da rumore troncato,

$$
\tilde{\mathbf{a}}' = \boldsymbol{\mu}_{\theta'}(s') + \boldsymbol{\epsilon},
\qquad \boldsymbol{\epsilon} \sim
\operatorname{clip}\big(\mathcal{N}(\mathbf{0},\sigma^2\mathbf{I}),
\,-c,\,c\big),
$$

dove $\sigma$ è l'ampiezza del rumore e $c$ la soglia oltre la quale viene
troncato (nell'articolo $\sigma = 0{,}2$ e $c = 0{,}5$; nell'implementazione
degli autori l'azione così ottenuta viene anche riportata dentro l'intervallo
delle azioni ammesse), così che il bersaglio sia liscio rispetto all'azione;
questo previene lo sfruttamento, da parte dell'attore, di picchi acuti ed
erronei nella superficie del critico.

In definitiva, dei due problemi di DDPG, TD3 attacca frontalmente la
sovrastima con il clipped double-Q e il target smoothing, mentre il *delayed
policy update* corregge l'instabilità dovuta a stime del critico ancora
immature. Sull'ipersensibilità agli iperparametri, invece, TD3 non offre
garanzie. Ne attenua i sintomi perché l'addestramento procede in modo più
regolare, non perché il problema sia risolto alla radice.

`````

## SAC: esplorare restando il più imprevedibile possibile

Quasi in contemporanea con TD3, Tuomas Haarnoja e colleghi propongono una
filosofia diversa: **SAC**, *Soft Actor-Critic* {cite}`haarnoja2018soft`. Qui
l'attore torna stocastico, e invece di una sola spinta restituisce una
distribuzione di probabilità sulle azioni possibili, da cui campiona l'azione
da eseguire. Cambia anche l'obiettivo stesso dell'apprendimento. Il *soft*
del nome allude a questo: dove prima l'agente puntava tutto sull'azione
migliore, adesso mantiene aperte più alternative.

`````{tab} Elementare

Chi va al lavoro sempre per la stessa strada, perché quella "funziona", non
scopre la scorciatoia: si è incaponito sulla prima strada che andava bene, e se
era solo mediocre non lo saprà mai.

SAC cambia la regola del gioco. All'agente non chiede soltanto "massimizza il
premio", ma "massimizza il premio *restando il più imprevedibile possibile*". A
parità di ricompensa attesa, preferisce la condotta più varia, quella che
mantiene aperte più opzioni. Un pendolare che ogni tanto cambia percorso, senza
perdere troppo tempo, resta pronto a cogliere la via migliore quando si
presenta.

Questa preferenza per la varietà si regola con una manopola, la
"temperatura": alta, l'agente esplora molto; bassa, si concentra sul premio. Il
nome viene dalla fisica, e l'immagine è quella giusta: più la temperatura è
alta, più le cose si agitano e si mescolano; più è bassa, più tutto si posa in
un'unica configurazione.

La manopola, di solito, non la gira una persona. Si fissa all'inizio quanta
varietà si pretende, un minimo sotto il quale non si vuole scendere, e poi è
l'algoritmo a muovere la manopola per tenere la promessa: se le scelte
dell'agente si stanno restringendo, sale; se l'agente sta girovagando più di
quanto la promessa chieda, scende, e il premio torna a contare di più.

Resta un intoppo pratico, e riguarda il modo in cui l'attore viene corretto.
Quello di DDPG si faceva indicare la direzione e spostava di un passettino
l'azione; il pendolare, invece, tira a sorte la strada del giorno, e due
giornate identiche gli danno percorsi diversi. Chiedersi "sarebbe andata meglio
allungando un po’ la deviazione?" non ha risposta, perché fra una giornata e
l'altra è cambiata anche la sorte.

Il rimedio è tenere il caso fuori dalla regola: il pendolare pesca prima un
numero da un sacchetto, poi applica la sua regola a quel numero e ne ricava il
percorso del giorno. Con il numero tenuto fermo la domanda ha una risposta, e
la regola si può ritoccare nella direzione giusta.

C'è poi un tetto alla deviazione, perché oltre un certo giro si arriva tardi.
La regola schiaccia sotto il tetto qualunque numero le venga passato, e i
numeri grandi finiscono tutti a ridosso del limite. Due numeri molto diversi,
se sono tutti e due grandi, danno quasi lo stesso giro largo. Il pendolare
sembra allora più vario di quanto sia: i numeri pescati dal sacchetto sono
vari, i percorsi no.

Di conseguenza, chi misura la varietà sui numeri, invece che sui percorsi,
gira la manopola della temperatura leggendo un valore falso, e compromette lo
scopo per cui SAC è stato inventato. Il rimedio è una correzione in più: si
toglie dalla misura la parte di varietà che il tetto ha schiacciato, e si torna
a contare quella dei percorsi.

`````

`````{tab} Superiore

SAC ottimizza l'obiettivo di **massima entropia**: al ritorno somma l'entropia
della policy in ogni stato,

$$
J(\pi) = \sum_{t=0}^{T} \mathbb{E}\Big[\, r_t + \alpha\,
\mathcal{H}\big(\pi(\cdot\mid s_t)\big)\Big],
\qquad
\mathcal{H}\big(\pi(\cdot\mid s)\big) = -\,\mathbb{E}_{a\sim\pi}\big[\log
\pi(a\mid s)\big].
$$

Il coefficiente $\alpha>0$ è la temperatura, che pesa quanto conta esplorare
rispetto allo sfruttare. L'entropia $\mathcal{H}$ è massima quando la policy è
il più possibile casuale, quindi massimizzarla spinge l'agente a non collassare
prematuramente su un'unica azione, migliorando l'esplorazione e la robustezza.

Il critico impara un *soft* Q-value con bersaglio

$$
y = r + \gamma\Big(\min_{i=1,2} Q_{\phi'_i}(s', \mathbf{a}') - \alpha \log
\pi_\theta(\mathbf{a}'\mid s')\Big),
\qquad \mathbf{a}' \sim \pi_\theta(\cdot\mid s'),
$$

che usa, come TD3, il minimo dei due critici e aggiunge il termine di entropia
$-\alpha\log\pi_\theta$.

L'attore, a sua volta, ha bisogno di far passare il gradiente attraverso il
campionamento dell'azione, operazione che altrimenti ne bloccherebbe il flusso.
La via è la **riparametrizzazione**: l'azione si scrive come funzione
deterministica di un rumore esterno,

$$
\mathbf{u} = \boldsymbol{\mu}_\theta(s) +
\boldsymbol{\sigma}_\theta(s) \odot \boldsymbol{\epsilon},
\qquad \mathbf{a} = \tanh(\mathbf{u}),
\qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),
$$

così il caso sta tutto in $\boldsymbol{\epsilon}$ e il gradiente scorre lungo
$\boldsymbol{\mu}_\theta$ e $\boldsymbol{\sigma}_\theta$, che qui sono media e
deviazione standard della gaussiana e non più, come in DDPG, l'intera policy. È
lo stesso accorgimento con cui si addestra il VAE, che si ricava nella
{doc}`sezione sul salto probabilistico
</ModelliLatenti/il-salto-probabilistico>`.

La $\tanh$ schiaccia l'azione nell'intervallo ammesso, ma modifica la densità
di probabilità, per cui nel $\log\pi_\theta$ va sottratto il termine di
correzione $\sum_j \log\big(1-\tanh^2(u_j)\big)$, dove $\mathbf{u}$ è l'azione
prima dello schiacciamento. Dimenticarlo è il tipico errore d'implementazione
in SAC, e falsa l'entropia, cioè il termine che l'algoritmo deve dosare.

La temperatura $\alpha$ non va fissata a mano, perché nella versione matura di
SAC {cite}`haarnoja2018applications` è auto-regolata dal problema vincolato che
chiede all'entropia media della policy di non scendere sotto un
valore-obiettivo $\mathcal{H}_0$ (di solito $-n$, l'opposto della dimensione
dell'azione). Il duale dà una discesa su $\alpha$ con perdita
$\mathcal{L}(\alpha) = \mathbb{E}_{s\sim\mathcal{D},\,\mathbf{a}\sim\pi_\theta}
\big[-\alpha\,(\log\pi_\theta(\mathbf{a}\mid s) + \mathcal{H}_0)\big]$, che
alza $\alpha$ quando l'entropia scende sotto la soglia e la abbassa quando la
supera; l'attore intanto minimizza
$\mathbb{E}_{s\sim\mathcal{D},\,\boldsymbol{\epsilon}}\big[\alpha\log
\pi_\theta(\mathbf{a}_\theta\mid s) - \min_{i=1,2}Q_{\phi_i}(s,
\mathbf{a}_\theta)\big]$, con $\mathbf{a}_\theta$ l'azione riparametrizzata.

Le garanzie sono poche. Nel caso tabellare, con $|\mathcal{A}|$ finito,
alternare valutazione e miglioramento *soft* della policy converge alla policy
ottima di massima entropia {cite}`haarnoja2018soft`; con azioni continue e reti
non c'è garanzia, e il vantaggio di SAC su DDPG è empirico: gli autori lo
trovano più stabile, con risultati simili fra semi diversi.

La prima versione richiedeva di calibrare soprattutto la scala delle ricompense,
che però equivale a scegliere la temperatura; regolarla in automatico rende
superfluo questo passaggio. In altri termini, l'esplorazione smette di essere
un parametro da indovinare a mano e diventa una conseguenza dell'obiettivo, e
questo spiega la fortuna del metodo.

`````

## Lo scheletro dell'aggiornamento, in PyTorch

I tre algoritmi condividono lo stesso ciclo off-policy: si pesca un minibatch di
esperienze passate dalla memoria di replay, si aggiorna il critico verso il
bersaglio che insegue e l'attore verso l'azione che il critico premia. Ecco il
cuore nella variante DDPG, senza gli orpelli. TD3 aggiunge il secondo critico e
il rumore sul bersaglio, SAC il premio alla varietà, che in termini tecnici è
l’entropia della policy, cioè quanto le sue scelte restano imprevedibili. La
temperatura ne dosa il peso.

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
    q_next = q_target(s_next, a_next).squeeze(-1)  # Q^-(s', mu^-(s')): (B,)
    y = r + gamma * q_next * (1 - fine)            # (B,): se finisce, solo r

# --- aggiornamento del critico: avvicina Q(s, a) al bersaglio ---
q = q_net(s, a).squeeze(-1)                   # Q delle azioni eseguite: (B,)
# ATTENZIONE alla forma: q e y devono essere entrambi (B,). Se si schiaccia solo
# uno dei due, il broadcasting porta il confronto a (B, B): mse_loss non solleva
# niente, stampa un UserWarning e minimizza la loss sbagliata. Chi non legge i
# warning non se ne accorge: e' l'errore piu' comune nelle implementazioni di
# DDPG.
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

Il segno meno nella `perdita_attore` è tutto ciò che serve, perché
l'ottimizzatore minimizza, e minimizzare $-Q_\phi(s,\boldsymbol{\mu}_\theta(s))$
equivale a far salire il valore che il critico dà all'azione dell'attore. La
retropropagazione parte da quel valore, attraversa il critico, raggiunge
l'azione ed entra nei parametri dell'attore. È la regola della catena del
gradiente deterministico di policy calcolata dalla libreria; per questo il
critico non dice solo *quanto vale* l'azione, ma anche *da che parte* spostarla
per farla valere di più.

## Onestà sui limiti

Questi tre metodi riusano ogni esperienza più volte, pescandola dalla memoria
di replay, e quindi imparano con molte meno prove nel mondo, e questo è
decisivo quando ogni tentativo consuma un robot vero. In cambio l'addestramento
è più delicato da mettere a punto, perché oltre agli iperparametri della rete
contano il rapporto fra aggiornamenti e passi di raccolta, la scala delle
ricompense e il rumore di esplorazione; DDPG è il più fragile dei
tre, TD3 e SAC lo sono meno.

Dove i campioni sono facili da raccogliere, invece, il vantaggio diminuisce:
con migliaia di robot simulati in parallelo si usa spesso PPO, che consuma
subito l'intera esperienza raccolta. Ad esempio, Rudin e colleghi hanno
insegnato così a camminare a un robot a quattro zampe in meno di quattro minuti
di addestramento sul piano, e in venti su un terreno accidentato
{cite}`rudin2022learning`. Non esiste quindi il vincitore assoluto: la scelta
dipende da quanto costa una prova e da quanta cura si può dedicare alla messa a
punto.

C'è poi un limite che nessuno di questi algoritmi risolve, il **sim-to-real
gap**, lo scarto fra simulazione e mondo fisico. Addestrare un robot
direttamente nel mondo fisico è lento e rischioso, così quasi sempre si impara
in simulazione, dove le prove sono infinite e le cadute non rompono nulla. Il
simulatore, però, non è la realtà: attriti, ritardi dei motori, giochi
meccanici e rumore dei sensori non coincidono mai del tutto. Di conseguenza,
una strategia perfetta nel simulatore può inciampare al primo passo reale.

Colmare quello scarto (con randomizzazione dei parametri fisici, calibrazione,
adattamento sul campo) è un problema di ricerca ancora aperto, e ricorda che
l'algoritmo di controllo è solo una parte del percorso che porta un robot a
muoversi nel mondo.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Nel controllo continuo l'azione è una quantità da dosare (quanta forza a
  ciascun motore) e non una voce da scegliere in un menu, che qui avrebbe
  infinite righe: scorrerle tutte, come fa DQN, non si può. La via d'uscita è
  affiancare al giudice che assegna i voti un attore che propone
  direttamente l'azione; al giudice resta da dire se è buona e da che parte
  ritoccarla.
- DDPG insegna all'attore a seguire la *pendenza* indicata dal critico,
  come chi sale una collina con la bussola invece che a tentoni; riusa il
  quaderno delle esperienze passate e le copie congelate delle reti ereditate
  da DQN, ed esplora aggiungendo un po’ di rumore casuale alle proprie azioni.
- DDPG è nervoso e si lascia illudere dai voti troppo alti. TD3 aggiunge
  tre accorgimenti: due giudici invece di uno, e ci si regola sul
  più prudente; l'attore cambia strategia una volta ogni due aggiornamenti dei
  giudici; il voto di riferimento viene sfumato con un pizzico di rumore, così
  l'attore non si aggrappa a un picco stretto e probabilmente illusorio. I
  voti si gonfiano meno; il nervosismo resta.
- SAC cambia l'obiettivo del gioco: cerca il massimo premio *restando il più
  imprevedibile possibile*, come il pendolare che ogni tanto cambia strada e
  per questo scopre la scorciatoia. Quanto contare la varietà è una manopola,
  che di solito l'algoritmo gira automaticamente per tenere una promessa di
  varietà minima fissata all'inizio.
- Riusare le esperienze già vissute fa imparare con molti meno tentativi
  (decisivo quando ogni prova consuma un robot vero), ma rende l'addestramento
  più delicato da tarare rispetto a PPO. Resta poi lo scarto fra simulatore e
  mondo fisico: una strategia perfetta in simulazione può inciampare al primo
  passo reale.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Nel controllo continuo l'azione è un vettore reale: l’`argmax` di DQN è
  intrattabile. La soluzione è un attore $\boldsymbol{\mu}_\theta(s)$ che
  propone l'azione e un critico $Q_\phi(s,\mathbf{a})$ che la valuta, in
  impianto off-policy.
- DDPG addestra l'attore deterministico con il *deterministic policy
  gradient* (il gradiente del critico rispetto all'azione), riusando replay
  buffer e reti target ereditati da DQN; esplora aggiungendo rumore all'azione.
- TD3 attenua la sovrastima di DDPG con due accorgimenti, *twin critics*
  (minimo dei due $Q$) e *target policy smoothing*, e con il terzo, i *delayed
  policy updates*, evita che l'attore insegua stime ancora immature;
  sull'ipersensibilità agli iperparametri non promette nulla.
- SAC adotta un attore stocastico e l'obiettivo di massima entropia
  (premio + entropia, con temperatura $\alpha$ spesso auto-regolata su un
  vincolo di entropia media minima): esplora meglio, è campione-efficiente ed
  empiricamente più stabile di DDPG; la convergenza è garantita solo nel caso
  tabellare.
- Off-policy significa efficienza nei campioni ma minore stabilità rispetto a
  PPO; e resta il sim-to-real gap, lo scarto tra simulazione e mondo fisico.
```
`````
