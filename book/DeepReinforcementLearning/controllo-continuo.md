# Controllo continuo: DDPG, TD3, SAC

Messa da parte la ricerca che pensa prima di muovere, si torna alle mosse
decise d'istinto. Un joystick Atari ha nove posizioni (su, giù, sinistra,
destra, le quattro diagonali, il centro), che con il pulsante diventano
diciotto azioni, e molti giochi ne usano meno: *Breakout* quattro. Il Deep
Q-Network sceglie fra le azioni del gioco guardando il valore di ciascuna e
tenendo il più alto: nel codice l'operazione si chiama `argmax`, e restituisce
*quale* voce ha il valore massimo, non il valore. Ma prova a immaginare un
braccio robotico con sette giunti, le sue articolazioni, o un robot a quattro
zampe che deve imparare a camminare. A ogni istante chi lo comanda non decide
"sinistra o destra": decide *quanta spinta* dare a ciascun motore, un numero
con la virgola, magari negativo per frenare, magari $3{,}4$, magari $3{,}41$.
Non c'è un menu di mosse da scorrere, ma un continuo di forze da dosare.

È lo scoglio annunciato in fondo alla {doc}`sezione su DQN <dqn>`. Prendere il
voto più alto vuol dire scorrere le mosse una per una: con un joystick si può,
con uno sterzo, un acceleratore o sette giunti che si muovono insieme le
combinazioni sono infinite e non si scorre più niente.

I metodi a {doc}`gradiente di policy <policy-gradient>` (REINFORCE,
attore-critico, A3C, PPO) su questo non hanno problemi: imparano direttamente
la policy, e una quantità da dosare la sanno produrre. Hanno però due limiti.
Sono *on-policy*, il contrario dell’*off-policy* di DQN: ogni esperienza serve
finché la policy che l'ha prodotta è ancora quella di adesso, cioè per pochi
aggiornamenti (PPO la riusa per qualche passata, non di più), e poi si butta.
E le loro stime hanno varianza alta: la stessa policy, rigiocata, dà gradienti
molto diversi fra loro. Su un robot vero, dove ogni tentativo costa tempo e
usura, sono due costi pesanti.

Servono metodi che uniscano le due virtù: azioni da dosare, come nei gradienti
di policy, e riuso delle esperienze passate con la memoria di replay, come in
DQN.

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

E c'è una seconda pretesa, che con il menu non c'entra: le prove già fatte non
si buttano. Ogni mossa tentata, con quello che ne è venuto, finisce su un
quaderno, e da lì la si ripesca per imparare ancora, molto tempo dopo, quando
l'attore ha già cambiato modo di giocare. Per un robot vero, dove ogni
tentativo consuma cinghie e ingranaggi, è quello che rende la faccenda
praticabile.

`````

`````{tab} Superiore

Nel controllo continuo lo spazio delle azioni è $\mathcal{A}\subseteq
\mathbb{R}^{n}$: un vettore di $n$ comandi reali (le coppie ai giunti, lo
sterzo, l'accelerazione). Il Q-learning sceglie l'azione con

$$
\mathbf{a}^\star = \arg\max_{\mathbf{a}\in\mathcal{A}} Q(s,\mathbf{a}),
$$

un problema di ottimizzazione da risolvere *a ogni passo* e per ogni stato. Con
$\mathcal{A}$ discreto e piccolo è una scansione; con $\mathcal{A}$ continuo è
un'ottimizzazione non convessa in $\mathbb{R}^n$, impraticabile *online*. Le
alternative che conservano l’$\arg\max$ lo rendono trattabile in due modi:
restringendo $Q$ a una forma quadratica nell'azione, dove il massimo si scrive
in forma chiusa (NAF {cite}`gu2016continuous`), oppure cercandolo per
campionamento con il *cross-entropy method* (QT-Opt
{cite}`kalashnikov2018qtopt`); discretizzare ciascuna delle $n$ componenti in
$k$ valori, invece, dà $k^n$ azioni da scorrere. La via più seguita è un'altra:
approssimare quel massimo con una policy parametrica
$\boldsymbol{\mu}_\theta(s)$ che restituisce direttamente l'azione, addestrata
in modo che
$\boldsymbol{\mu}_\theta(s) \approx \arg\max_{\mathbf{a}} Q(s,\mathbf{a})$.
Vogliamo inoltre un metodo *off-policy*, che riusi un buffer di esperienze
passate come DQN {cite}`mnih2015human`, per essere campione-efficiente
{cite}`sutton2018reinforcement`.

`````

## DDPG: un attore deterministico guidato dal critico

Il primo algoritmo a tenere insieme le due virtù appena chieste con reti
profonde, su compiti di controllo con molti gradi di libertà, è **DDPG**,
*Deep Deterministic Policy Gradient*, presentato da Lillicrap e colleghi di
DeepMind nel 2016 {cite}`lillicrap2016continuous`. L'attore-critico con
policy deterministica esisteva già, anche con reti neurali piccole; il
contributo di DDPG è averlo reso stabile su reti profonde portandoci gli
accorgimenti di DQN.

L'idea è tenere due reti che collaborano. L’attore guarda la situazione e
propone un'azione precisa: non un ventaglio di possibilità con le loro
probabilità, come facevano le policy del gradiente di policy, ma esattamente
la spinta da dare, un numero per ciascun motore. È deterministica, cioè
nella stessa situazione risponde sempre la stessa cosa, senza tirare dadi: da lì
la seconda D del nome. Il critico è la vecchia rete dei voti di DQN con una
modifica: oltre alla situazione riceve in ingresso *anche* l'azione proposta, e
restituisce un numero solo, quanto vale fare quella mossa lì.

Il critico impara come in DQN, inseguendo un bersaglio, cioè il voto che
quella mossa dovrebbe avere secondo i conti del momento; e come in DQN quel
bersaglio si calcola con delle copie congelate delle due reti, per la stessa
ragione di allora, cioè perché un bersaglio che si sposta insieme a chi lo
insegue non si raggiunge mai. L'attore, dal canto suo, impara a proporre le
azioni che il critico premia di più.

`````{tab} Elementare

Come fa l'attore a "sapere" in che direzione muovere la forza? Il critico è un
paesaggio di colline: per ogni azione possibile c'è un'altezza, il suo valore.
L'attore sta in un punto e vuole salire. Il critico, oltre a dirgli l'altezza,
gli indica la *pendenza*: "da qui, spingendo un filo di più sul secondo giunto,
sali". L'attore fa un passettino in quella direzione. Ripetuto tante volte,
arriva in cima alla collina su cui si trova, e ci arriva senza aver mai dovuto
provare tutte le azioni una per una. È la differenza tra cercare la vetta a
tentoni e seguire la bussola della pendenza.

La bussola, però, dice da che parte si sale da qui, non dove sta la vetta più
alta del paesaggio. Se una collina più alta comincia dall'altra parte della
valle, salendo non ci si arriva: per raggiungerla bisognerebbe prima scendere,
e la pendenza dice sempre di salire.

Per non restare fermo su ciò che già conosce, l'attore aggiunge alle sue azioni
un po’ di rumore casuale: piccole spinte imprevedibili che lo fanno provare
varianti nuove. È l'equivalente continuo del "ogni tanto tira a caso invece di
prendere la mossa migliore" con cui il Q-learning del capitolo precedente
esplorava.

Le pendenze, poi, l'attore non le misura dove passa adesso: le misura nei punti
pescati dal quaderno, cioè dove era finito quando giocava in un altro modo. Il
conto fatto fino in fondo vorrebbe le pendenze dei posti di oggi; si usano
quelle di ieri perché sono già scritte, e in cambio ci si accontenta di una
direzione buona invece che esatta. Funziona bene, ma nessun conto lo
garantisce: è una scelta pratica.

`````

`````{tab} Superiore

In simboli: l'attore è una policy deterministica $\boldsymbol{\mu}_\theta(s)$,
che restituisce direttamente il vettore delle azioni invece di una distribuzione
su di esse; il critico è $Q_\phi(s,\mathbf{a})$, con l'azione fra gli ingressi.
L'obiettivo vero è il ritorno atteso di $\boldsymbol{\mu}_\theta$, e il
**deterministic policy gradient theorem** {cite}`silver2014deterministic` ne dà
il gradiente come
$\mathbb{E}_{s\sim\rho^{\mu}}\big[\nabla_{\mathbf{a}} Q^{\mu}(s,\mathbf{a})
\big|_{\mathbf{a}=\boldsymbol{\mu}_\theta(s)}
\nabla_\theta\boldsymbol{\mu}_\theta(s)\big]$,
con $\rho^\mu$ la distribuzione scontata degli stati che
$\boldsymbol{\mu}_\theta$ visita e $Q^\mu$ il suo valore vero: come nel teorema
del gradiente di policy stocastico, la dipendenza degli stati visitati da
$\theta$ non va derivata. DDPG sostituisce $Q^\mu$ con il critico e $\rho^\mu$
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
*rispetto all'azione*: dice come cambiare $\mathbf{a}$ per aumentare il valore.
Il secondo, $\nabla_\theta \boldsymbol{\mu}_\theta$, propaga quella direzione ai
parametri dell'attore.

Un passaggio, qui, è un'approssimazione e non un'uguaglianza, e conviene non
farselo scivolare addosso. Il teorema vale per stati distribuiti secondo
$\rho^\mu$, cioè secondo la policy corrente; scriverci sotto $s\sim\mathcal{D}$,
gli stati del replay buffer raccolti da policy vecchie, è la mossa che rende DDPG
*off-policy* e costa un termine che si butta via. Funziona, ma non discende dalla
regola della catena: è una scelta, ed è la stessa che si fa in tutti i metodi
attore-critico off-policy. Il critico si addestra sul bersaglio di Bellman

$$
y = r + \gamma\, Q_{\phi'}\!\big(s', \boldsymbol{\mu}_{\theta'}(s')\big),
$$

dove $\phi'$ e $\theta'$ sono i parametri delle reti target, aggiornate con uno
scorrimento lento (*Polyak averaging*)
$\phi' \leftarrow \tau\phi + (1-\tau)\phi'$, con $\tau\ll 1$ (questo $\tau$ è un
numero, il peso dello scorrimento, e non ha niente a che vedere con la
traiettoria $\tau$ del gradiente di policy). L'esplorazione avviene aggiungendo
rumore all'azione in fase di raccolta,
$\mathbf{a} = \boldsymbol{\mu}_\theta(s) + \boldsymbol{\epsilon}$: nel paper
originale $\boldsymbol{\epsilon}$ è un processo di Ornstein-Uhlenbeck (rumore
temporalmente correlato, utile in sistemi con inerzia), ma nella pratica un
semplice rumore gaussiano indipendente funziona altrettanto bene.

`````

## Perché DDPG è fragile

DDPG funziona, ma è fragile, e due problemi ne minano la stabilità.

Il primo è la sovrastima del valore, lo stesso male che affliggeva DQN. Il
critico ha errori di stima in ogni direzione; l'attore, addestrato a cercare le
azioni che il critico valuta di più, si infila proprio dove il critico ha
sbagliato *per eccesso*. Quegli errori ottimistici vengono così selezionati,
amplificati e reimmessi nel bersaglio che il critico insegue, dove tendono ad
accumularsi. Il secondo è l’ipersensibilità agli iperparametri, cioè alle
manopole che si decidono prima di cominciare e non si imparano: la velocità con
cui le reti si correggono, quanto rumore aggiungere, quanto farle grandi.
Ritoccarne una di poco può fare la differenza fra un agente che impara a
camminare e uno che crolla a terra. E non serve nemmeno ritoccarla: basta
rilanciare lo stesso identico addestramento cambiando soltanto il seme, e i
risultati possono essere molto diversi. Lo hanno misurato Henderson e colleghi
mettendo a confronto DDPG, TRPO, PPO e ACKTR: a parità di tutto il resto, gruppi
di semi diversi danno curve di apprendimento statisticamente diverse
{cite}`henderson2018deep`.

## TD3: tre correzioni chirurgiche

Nel 2018 Scott Fujimoto, Herke van Hoof e David Meger analizzano queste
patologie e propongono **TD3**, *Twin Delayed DDPG* {cite}`fujimoto2018addressing`.
Quel «TD» non ha niente a che vedere con le differenze temporali del capitolo
precedente: sta per *Twin Delayed*, «gemello e ritardato», e i due aggettivi
dicono già due dei tre accorgimenti. È DDPG con tre correzioni mirate e non un
algoritmo nuovo, ognuna rivolta a un difetto preciso.

`````{tab} Elementare

Due giudici, non uno. Il primo trucco combatte l'ottimismo del critico
tenendo *due* critici invece di uno, e fidandosi sempre del più prudente: per
calcolare il valore di riferimento si prende il minimo dei due voti. Se un
giudice si è illuso e ha dato un voto troppo alto, l'altro fa da freno. È come
far stimare la propria casa da due agenti immobiliari e fidarsi di quello che
dice la cifra più bassa: ci si illude di meno.

E vale l'avvertenza già vista per il Double DQN, perché è la stessa: i due
giudici non sono estranei fra loro, hanno studiato sugli stessi dati e inseguito
lo stesso bersaglio, quindi tendono a illudersi insieme. Il minimo attenua, non
guarisce, e semmai sposta il difetto: al posto di un voto un po’ troppo alto se
ne prende uno un po’ troppo basso.

L'attore parla di meno. Il secondo trucco è rallentare l'attore: i critici
si aggiornano a ogni passo, l'attore solo una volta ogni due. Prima di cambiare
strategia, conviene che i giudici abbiano le idee chiare; un attore che insegue
critici ancora confusi rincorre bersagli sbagliati.

Bersagli sfumati. Il terzo trucco aggiunge un pizzico di rumore all'azione
usata nel calcolo del voto di riferimento, così che azioni quasi identiche
ricevano voti quasi identici. Impedisce all'attore di aggrapparsi a un picco
stretto e probabilmente illusorio del critico.

Dei due problemi di DDPG, i tre trucchi attaccano l'ottimismo dei voti, e lo
attaccano in due: i due giudici e i bersagli sfumati. L'attore che parla di
meno cura un problema che nell'elenco non c'era, cioè l'attore che insegue
giudizi ancora acerbi. Sulla sensibilità alle manopole, invece, TD3 non
promette niente: l'addestramento è meno nervoso e quindi se ne soffre meno, ma
il guaio è ancora tutto lì.

`````

`````{tab} Superiore

**(a) Clipped double-Q.** Si mantengono due critici $Q_{\phi_1}, Q_{\phi_2}$
addestrati sullo stesso bersaglio, costruito con il *minimo* delle due reti
target:

$$
y = r + \gamma \min_{i=1,2} Q_{\phi'_i}\!\big(s', \tilde{\mathbf{a}}'\big),
$$

dove $\tilde{\mathbf{a}}'$ è l'azione dell'attore target *sfumata dal rumore*,
definita poco più sotto dal *target policy smoothing*.

Prendere il minimo introduce un bias *pessimista* che compensa la sovrastima:
poiché l'errore che si propaga è il più piccolo dei due, il valore tende a non
gonfiarsi. Vale però lo stesso caveat visto per il Double DQN, ed è la stessa
ragione: i due critici sono addestrati sullo stesso bersaglio e sugli stessi
dati, quindi i loro errori sono correlati, e il minimo di due stime correlate
non elimina il bias, lo sposta, scambiando tipicamente una sovrastima con una
moderata sottostima. È un correttivo che funziona in pratica, non una cura.
**(b) Delayed policy updates.** L'attore e le reti target si aggiornano ogni $d$
passi del critico (tipicamente $d=2$): riducendo la frequenza degli
aggiornamenti dell'attore si abbassa la varianza e si evita che insegua stime
ancora immature. L'attore, poi, sale lungo il gradiente di $Q_{\phi_1}$
soltanto, non del minimo dei due critici. **(c) Target policy smoothing.**
L'azione target è "regolarizzata" da rumore troncato,

$$
\tilde{\mathbf{a}}' = \boldsymbol{\mu}_{\theta'}(s') + \boldsymbol{\epsilon},
\qquad \boldsymbol{\epsilon} \sim
\operatorname{clip}\big(\mathcal{N}(\mathbf{0},\sigma^2\mathbf{I}),
\,-c,\,c\big),
$$

dove $\sigma$ è l'ampiezza del rumore e $c$ la soglia oltre la quale viene
troncato (nel paper $\sigma = 0{,}2$ e $c = 0{,}5$; nell'implementazione degli
autori l'azione così ottenuta viene anche riportata dentro l'intervallo delle
azioni ammesse), così che il bersaglio sia liscio rispetto all'azione: previene
lo sfruttamento, da parte dell'attore, di picchi acuti ed erronei nella
superficie del critico. Dei due problemi di DDPG, TD3 attacca frontalmente la
sovrastima, con il clipped double-Q e il target smoothing; il *delayed policy
update* cura un problema che nell'elenco non c'era e va aggiunto, cioè l'attore
che insegue stime ancora immature. Sull'ipersensibilità agli iperparametri,
invece, TD3 non promette nulla: ne attenua i sintomi perché l'addestramento è
meno nervoso, non perché il problema sia risolto. Il valore dell'algoritmo sta
tutto lì: resta concettualmente DDPG, e dove DDPG è nervoso in genere non lo è.

`````

## SAC: esplorare restando il più imprevedibile possibile

Quasi in contemporanea con TD3, Tuomas Haarnoja e colleghi propongono una
filosofia diversa: **SAC**, *Soft Actor-Critic* {cite}`haarnoja2018soft`. Qui
l'attore torna stocastico, cioè l'opposto di deterministico: invece di una
sola spinta restituisce un ventaglio di spinte possibili con le loro
probabilità, e la mossa vera la si estrae da lì. E cambia l'obiettivo stesso
dell'apprendimento. Il *soft* del nome vuol dire «morbido», ed è un'allusione
proprio a questo: dove prima l'agente puntava tutto sulla mossa migliore, adesso
tiene aperto un ventaglio.

`````{tab} Elementare

Chi va al lavoro sempre per la stessa strada, perché quella "funziona", la
scorciatoia non la scopre: si è incaponito sulla prima strada che andava bene,
e se era solo mediocre non lo saprà mai.

SAC cambia la regola del gioco. All'agente non chiede soltanto "massimizza il
premio", ma "massimizza il premio *restando il più imprevedibile possibile*". A
parità di ricompensa attesa, preferisce la condotta più varia, quella che
mantiene aperte più opzioni. Un pendolare che ogni tanto cambia percorso, senza
perdere troppo tempo, resta pronto a cogliere la via migliore quando si
presenta. Questa preferenza per la varietà si regola con una manopola, la
"temperatura": alta, l'agente esplora molto; bassa, si concentra sul premio. Il
nome viene dalla fisica, e l'immagine è quella giusta: più la temperatura è
alta, più le cose si agitano e si mescolano; più è bassa, più tutto si posa in
un'unica configurazione.

La manopola, di solito, non la gira una persona. Si fissa all'inizio quanta
varietà si pretende, un minimo sotto il quale non si vuole scendere, e poi la
manopola si muove da sé per tenere la promessa: se le scelte dell'agente si
stanno restringendo, sale; se l'agente sta girovagando più di quanto la
promessa chieda, scende, e il premio torna a contare di più.

Resta un intoppo pratico, e riguarda il modo in cui l'attore viene corretto.
Quello di DDPG si faceva indicare la direzione e spostava di un passettino la
mossa; il pendolare, invece, la strada del giorno la tira a sorte, e due
giornate identiche gli danno percorsi diversi. Chiedersi "sarebbe andata meglio
allungando un po’ la deviazione?" non ha risposta, perché fra una giornata e
l'altra è cambiata anche la sorte. Il rimedio è tenere il caso fuori dalla regola: il pendolare
pesca prima un numero da un sacchetto, poi applica la sua regola a quel numero
e ne ricava il percorso del giorno. Con il numero tenuto fermo la domanda ha
una risposta, e la regola si può ritoccare nella direzione giusta.

C'è poi un tetto alla deviazione, perché oltre un certo giro si arriva tardi: la
regola schiaccia sotto il tetto qualunque numero le venga passato, e i numeri
grandi finiscono tutti a ridosso del limite. Due numeri molto diversi, se sono
tutti e due grandi, danno quasi lo stesso giro largo. Il pendolare sembra allora
più vario di quanto sia: i numeri pescati dal sacchetto sono vari, i percorsi
no. Chi misura la varietà sui numeri, invece che sui percorsi, gira la manopola
della temperatura leggendo un valore falso, e guasta proprio la cosa per cui SAC
è stato inventato. Il rimedio è un conto in più: si toglie dalla misura la parte
di varietà che il tetto ha schiacciato, e si torna a contare quella dei
percorsi.

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
il più possibile casuale: massimizzarla spinge l'agente a non collassare
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
campionamento dell'azione, che da solo non lo lascia passare. La via è la
**riparametrizzazione**: l'azione si scrive come funzione deterministica di un
rumore esterno,

$$
\mathbf{u} = \boldsymbol{\mu}_\theta(s) +
\boldsymbol{\sigma}_\theta(s) \odot \boldsymbol{\epsilon},
\qquad \mathbf{a} = \tanh(\mathbf{u}),
\qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),
$$

così il caso sta tutto in $\boldsymbol{\epsilon}$ e il gradiente scorre lungo
$\boldsymbol{\mu}_\theta$ e $\boldsymbol{\sigma}_\theta$, che qui sono media e
deviazione della gaussiana e non più, come in DDPG, l'intera policy. È lo stesso
trucco con cui si addestra il VAE, che il capitolo sui modelli latenti deriverà
nella {doc}`sezione sul salto probabilistico
</ModelliLatenti/il-salto-probabilistico>`. La $\tanh$ schiaccia l'azione
nell'intervallo ammesso, e non è gratis: cambia la densità, e nel
$\log\pi_\theta$ va sottratto il termine di correzione
$\sum_j \log\big(1-\tanh^2(u_j)\big)$, dove $\mathbf{u}$ è l'azione prima dello
schiacciamento. Dimenticarlo è l'errore d'implementazione classico di SAC, e
cade nel punto peggiore: falsa l'entropia, cioè proprio il termine che
l'algoritmo esiste per dosare.

La temperatura $\alpha$ non va fissata a mano: nella versione matura di SAC
{cite}`haarnoja2018applications` è auto-regolata dal problema vincolato che
chiede all'entropia media della policy di non scendere sotto un
valore-obiettivo $\mathcal{H}_0$ (di solito $-n$, meno la dimensione
dell'azione). Il duale dà una discesa su $\alpha$ con perdita
$\mathcal{L}(\alpha) = \mathbb{E}_{s\sim\mathcal{D},\,\mathbf{a}\sim\pi_\theta}
\big[-\alpha\,(\log\pi_\theta(\mathbf{a}\mid s) + \mathcal{H}_0)\big]$, che
alza $\alpha$ quando l'entropia scende sotto la soglia e la abbassa quando la
supera; l'attore intanto minimizza
$\mathbb{E}_{s\sim\mathcal{D},\,\boldsymbol{\epsilon}}\big[\alpha\log
\pi_\theta(\mathbf{a}_\theta\mid s) - \min_{i=1,2}Q_{\phi_i}(s,
\mathbf{a}_\theta)\big]$, con $\mathbf{a}_\theta$ l'azione riparametrizzata.

Che cosa si sa garantire? Nel caso tabellare, con $|\mathcal{A}|$ finito,
alternare valutazione e miglioramento *soft* della policy converge alla policy
ottima di massima entropia {cite}`haarnoja2018soft`; con azioni continue e reti
non c'è garanzia, e il vantaggio di SAC su DDPG è empirico: gli autori lo
trovano più stabile, con risultati simili fra semi diversi. La prima versione
chiedeva di tarare soprattutto la scala delle ricompense, che però equivale a
una scelta della temperatura, e la temperatura regolata in automatico la rende
superflua. La ragione della sua fortuna è questa: l'esplorazione smette di
essere un parametro da indovinare a mano e diventa una conseguenza
dell'obiettivo.

`````

## Lo scheletro dell'aggiornamento, in PyTorch

I tre algoritmi condividono lo stesso ciclo off-policy: si pesca un pugno di
esperienze passate dalla memoria di replay, si aggiorna il critico verso il
bersaglio che insegue e l'attore verso l'azione che il critico premia. Ecco il
cuore nella variante DDPG, senza gli orpelli; TD3 aggiunge il secondo critico e
il rumore sul bersaglio, SAC il premio alla varietà, che in termini tecnici è
l’entropia della policy, cioè quanto le sue scelte restano imprevedibili: è
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

Il segno meno nella `perdita_attore` è tutto ciò che serve: l'ottimizzatore
minimizza, e minimizzare $-Q_\phi(s,\boldsymbol{\mu}_\theta(s))$ vuol dire far
salire il valore che il critico dà all'azione dell'attore. La retropropagazione
parte da quel valore, attraversa il critico, arriva all'azione e da lì entra nei
parametri dell'attore: è la regola della catena del gradiente deterministico di
policy, calcolata dalla libreria, ed è il motivo per cui il critico non dice
solo *quanto vale* l'azione ma anche *da che parte* spostarla per farla valere
di più.

## Onestà sui limiti

Questi tre metodi riusano ogni esperienza molte volte, pescandola dalla memoria
di replay, e quindi imparano da molte meno prove nel mondo: è decisivo quando
ogni tentativo consuma un robot vero. Il prezzo è un addestramento più delicato
da mettere a punto, perché oltre agli iperparametri della rete contano il
rapporto fra aggiornamenti e passi di raccolta, la scala delle ricompense e il
rumore di esplorazione; DDPG è il più fragile dei tre, TD3 e SAC lo sono meno.
Dove i campioni costano poco, invece, il vantaggio conta meno: con migliaia di
robot simulati in parallelo si usa spesso PPO, che quell'esperienza fresca la
consuma tutta, e Rudin e colleghi insegnano così a camminare a un robot a
quattro zampe in meno di venti minuti di addestramento
{cite}`rudin2022learning`. Non esiste il vincitore assoluto: la scelta dipende
da quanto costa una prova e da quanta cura si può dedicare alla messa a punto.

C'è poi un limite che nessuno di questi algoritmi risolve da sé, il
**sim-to-real gap**, lo scarto fra simulazione e mondo fisico. Addestrare un
robot direttamente nel mondo fisico è lento e rischioso, così quasi sempre si
impara in simulazione, dove le prove sono infinite e le cadute non rompono
nulla. Ma il simulatore non è la realtà: attriti, ritardi dei motori, giochi
meccanici e rumore dei sensori non coincidono mai del tutto. Una strategia
perfetta nel simulatore può inciampare al primo passo reale. Colmare quello
scarto (con randomizzazione dei parametri
fisici, calibrazione, adattamento sul campo) è un problema di ricerca ancora
aperto, e ci ricorda che l'algoritmo di controllo è solo un pezzo del percorso
che porta un robot a muoversi nel mondo.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Nel controllo continuo l'azione è una quantità da dosare (quanta forza a
  ciascun motore) e non una voce da scegliere in un menu, e il menu ha
  infinite righe: scorrerle tutte, come fa DQN, non si può. La via d'uscita è
  affiancare al giudice che assegna i voti un attore che propone
  direttamente la mossa; al giudice resta da dire se è buona e da che parte
  ritoccarla.
- DDPG insegna all'attore a seguire la *pendenza* indicata dal critico,
  come chi sale una collina con la bussola invece che a tentoni; riusa il
  quaderno delle esperienze passate e le copie congelate delle reti ereditate
  da DQN, ed esplora aggiungendo un po’ di rumore casuale alle proprie mosse.
- DDPG è nervoso e si lascia illudere dai voti troppo alti. TD3 aggiunge
  tre accorgimenti: due giudici invece di uno, e ci si regola sul
  più prudente; l'attore cambia strategia una volta ogni due aggiornamenti dei
  giudici; il voto di riferimento viene sfumato con un pizzico di rumore, così
  l'attore non si aggrappa a un picco stretto e probabilmente illusorio. I
  voti si gonfiano meno; il nervosismo resta.
- SAC cambia l'obiettivo del gioco: non solo il massimo premio, ma il
  massimo premio *restando il più imprevedibile possibile*, come il pendolare
  che ogni tanto cambia strada e per questo scopre la scorciatoia. Quanto
  contare la varietà è una manopola, che di solito l'algoritmo gira da sé per
  tenere una promessa di varietà minima fissata all'inizio.
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
- Off-policy significa efficienza nei campioni ma minore stabilità di
  PPO; e resta il sim-to-real gap, lo scarto tra simulazione e mondo fisico.
```
`````
