# Deep Q-Network (DQN)

Torniamo al risultato del 2013 da cui si è aperto il capitolo, stavolta per
guardarci dentro. Quel piccolo gruppo di ricercatori londinesi di una startup
chiamata DeepMind aveva mostrato un unico programma che imparava a giocare a
diversi videogiochi Atari (*Breakout*, *Pong*, *Space
Invaders*) senza che nessuno gli avesse spiegato le regole. L'algoritmo
riceveva solo ciò che vedrebbe un ragazzino davanti al cabinato: i pixel dello
schermo e il punteggio. Da lì, per tentativi, su tre di quei giochi
arrivava a superare un umano esperto {cite}`mnih2013playing`. Due anni dopo il
risultato finì sulla
copertina di *Nature* {cite}`mnih2015human`. Quel programma si chiama **Deep
Q-Network**, DQN.

Nel capitolo precedente abbiamo incontrato il *Q-learning*: un agente impara una
funzione $Q(s,a)$ che stima quanto è conveniente, nel lungo periodo, compiere
l'azione $a$ trovandosi nello stato $s$. Lì la $Q$ viveva in una tabella: una
riga per ogni stato, una colonna per ogni azione. Funziona con pochi stati.

Ma quanti stati ha una schermata Atari? Lo schermo è di $210\times160$ punti,
cioè 33.600 in tutto, e ciascuno può avere un colore fra molti. Le possibilità
non si sommano fra loro, si moltiplicano: il numero che ne esce ha decine di
migliaia di cifre, mentre per contare tutti gli atomi dell'universo osservabile
ne bastano ottantuno. La tabella non basta più: non basterebbe nemmeno se una
casella pesasse un atomo.

## Dalla tabella alla rete

La mossa di DQN è tanto semplice quanto radicale: **buttiamo via la tabella e
mettiamo al suo posto una rete neurale**.

`````{tab} Elementare

Immagina un enorme schedario in cui, per ogni possibile schermata di gioco,
c'è un cartellino con scritto quanto vale ciascuna mossa. Impossibile
compilarlo: le schermate sono infinite. Allora sostituisci lo schedario con un
*esperto* che guarda la schermata e, a colpo d'occhio, ti dice il valore di
ogni mossa: anche per schermate che non ha mai visto prima, perché ha imparato
a riconoscere le somiglianze. Quell'esperto è la rete neurale.

Vale la pena fissare una parola che tornerà spesso. Dentro la rete ci sono dei
numeri, qualche milione, che decidono come una schermata si trasforma in un
voto: si chiamano **pesi**, e sono le uniche cose che cambiano mentre la rete
impara. Addestrare la rete vuol dire ritoccarli, un pochino alla volta, finché i
voti non diventano sensati.

`````

`````{tab} Superiore

Approssimiamo la funzione azione-valore ottima con una rete parametrizzata da
$\theta$:

$$
Q(s, a; \theta) \approx Q^{*}(s, a).
$$

La rete prende in ingresso lo stato $s$ (i pixel) e restituisce in uscita un
vettore con un valore $Q$ per **ciascuna** azione ammissibile: non serve una
passata per azione. È un *function approximator*: generalizza a stati mai
visti, sfruttando la struttura condivisa delle immagini invece di memorizzare
ogni caso singolarmente.

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

## Perché divergeva: la triade fatale

Mettere una rete al posto della tabella non era, di per sé, un'idea nuova.
**TD-Gammon** lo faceva dal 1992, con una rete addestrata a suon di partite di
backgammon giocate contro se stessa; nella versione descritta tre anni dopo
giocava quasi come i più forti campioni del mondo {cite}`tesauro1995temporal`.
Eppure il Q-learning con una rete, per anni, **divergeva**: i valori stimati
crescevano senza fermarsi, invece di assestarsi.

Fra i due c'è una differenza sola, e conviene guardarla da vicino perché torna
in tutto il capitolo. TD-Gammon si allenava sulle partite che stava giocando
davvero. Il Q-learning fa una cosa più furba, e più rischiosa: gioca in un modo
e impara un altro modo. Ogni tanto, apposta, tira una mossa a caso per vedere
che succede; ma quando poi si segna il voto di quella situazione non ci scrive
quanto vale la mossa a caso, ci scrive quanto vale la mossa *migliore* fra
quelle disponibili lì. Gioca da esploratore e prende appunti da campione. Si
chiama **off-policy**, ed è comodissimo, perché permette di imparare da
qualunque partita: anche da una giocata male, anche da una giocata da un altro
molto tempo prima.

E non divergeva per sfortuna, né perché qualcuno avesse sbagliato a tarare il
passo di apprendimento (il *learning rate*: di quanto si spostano i pesi a
ogni correzione). Prima dei due trucchi che lo hanno reso praticabile conviene
capire da che cosa lo hanno salvato, perché è un risultato preciso e
sorprendentemente pulito.

`````{tab} Elementare

Ci sono tre ingredienti in gioco, ognuno dei quali, preso da solo, è
innocuo e anzi utile.

Il primo è l’**approssimazione**: una rete al posto della tabella, cioè
sacrificare la precisione su ogni singolo stato in cambio della capacità di
generalizzare. Il secondo è il **bootstrapping**: aggiornare una stima usando
un'altra stima invece di aspettare la fine della partita, ed è la mossa che
distingue le differenze temporali (il **TD** del capitolo precedente, che
aggiornano subito) dai metodi **Monte Carlo**, che aspettano il fischio finale e
solo allora tirano le somme (si chiamano così, come il casinò, perché è il nome
che i matematici danno ai metodi che fanno i conti lasciando andare le cose a
sorte e guardando com'è finita). Il terzo è l’**off-policy**, quello di poco fa:
giocare in un modo e imparare un altro modo.

La sintesi, che si deve a Richard Sutton e Andrew Barto (i due autori del
manuale classico di questa materia), è che con due qualunque di questi tre
l'instabilità si può evitare. **Tutti e tre insieme no**: la combinazione può
divergere, cioè i valori possono crescere senza limite invece di assestarsi. La
chiamano **triade fatale**.

La parte inquietante è che per vederla non serve un ambiente difficile: serve
il contrario, e questo è tutto il punto. L'esempio classico si chiama
**controesempio di Baird**, dal nome di chi lo costruì, e più facile di così un
compito non si può fare: sette situazioni, e in nessuna si guadagna mai niente.
La risposta giusta è «tutto vale zero», il sistema saprebbe scriverla alla
perfezione (e non serve nemmeno una rete profonda: bastano una manciata di pesi
e la più semplice delle reti), e ciononostante quei pesi, invece di posarsi
sullo zero, cominciano a
crescere e non smettono più. Se il metodo sbaglia il problema più semplice del
mondo, il guasto non è nel problema.

Perché succede, in una frase. Ogni correzione punta verso un numero, il
**bersaglio**: il voto che, secondo i conti del momento, quella mossa dovrebbe
avere. Si ottiene sommando due cose, la ricompensa appena incassata e il
giudizio sulla situazione in cui si è finiti. Se ho fatto una mossa che mi ha
fruttato $1$ punto e la situazione in cui mi trovo adesso la valuto $7$, il
bersaglio è quei due numeri messi assieme, cioè circa $8$: ed è lì che vorrò
spostare il voto di quella mossa.

Ora, quando la rete corregge il proprio
giudizio su una situazione, la correzione si allarga da sé a tutte le
situazioni che le somigliano: è il prezzo della generalizzazione, e in
condizioni normali è un vantaggio. Ma il bersaglio verso cui la correzione
punta è a sua volta il giudizio su una situazione vicina, cioè una di quelle che
la correzione ha appena spostato. Ogni ritocco muove il bersaglio che serviva a
deciderlo, e il ritocco successivo parte da un bersaglio già mosso.

L'off-policy toglie l'ultima protezione, e conviene vedere quale. Una rete non
si corregge una volta sola: si corregge su un mucchio di esempi, e quanto una
situazione compare spesso in quel mucchio, tanto peso ha nel risultato.

Immagina allora un agente che si allena solo sulle situazioni in cui capita
davvero, giocando la strategia che sta imparando. Le situazioni frequenti
peseranno molto e quelle rare poco, che è giusto, perché sono le frequenti a
decidere come andrà a finire. Con quel bilanciamento il rimpallo di poco fa si
smorza da sé, ed è un risultato dimostrato.

Ma l'off-policy fa proprio saltare quel bilanciamento: l'agente si allena su
partite giocate in un altro modo, quindi certe situazioni le vede molto più
spesso di quanto le incontrerebbe davvero, e altre quasi mai. Si corregge con
forza dove non gli serve, e quel che ne esce può crescere invece di posarsi.

`````

`````{tab} Superiore

La **triade fatale** {cite}`sutton2018reinforcement` è la coesistenza di:

1. **approssimazione di funzione**, cioè una rappresentazione parametrica che
   generalizza fra stati invece di trattarli come voci indipendenti;
2. **bootstrapping**, cioè bersagli che contengono stime correnti (TD,
   programmazione dinamica) invece dei soli ritorni osservati;
3. **addestramento off-policy**, cioè una distribuzione degli aggiornamenti
   diversa da quella indotta dalla policy che si sta valutando.

Con due soli dei tre l'instabilità si può evitare; con tutti e tre no,
e la divergenza si osserva già nel caso della sola **predizione**, senza
controllo né miglioramento della policy. Non dipende nemmeno
dall'incertezza sull'ambiente: si manifesta identica nella programmazione
dinamica, dove il modello è noto per intero.

Il **controesempio di Baird** lo esibisce in forma minima: sette stati, due
azioni, ricompensa sempre nulla, $\gamma = 0{,}99$, e una policy di
comportamento che visita gli stati in modo uniforme mentre la policy bersaglio
ne concentra tutta la massa su uno solo. La funzione valore vera è
identicamente zero ed è **esattamente rappresentabile** dai parametri
disponibili; il TD semi-gradiente, ciononostante, fa divergere i pesi. Il
fattore decisivo è la **distribuzione degli aggiornamenti**: uniforme sugli
stati, mentre la policy bersaglio li visiterebbe in proporzioni tutte diverse.
Non basta osservare che l'aggiornamento semi-gradiente non è il gradiente di
nessuna funzione obiettivo (si deriva rispetto alla stima ma non rispetto al
bersaglio, che pure dipende dai parametri): lo stesso aggiornamento, con
approssimatore lineare e sotto la distribuzione on-policy, converge, come
dimostrarono Tsitsiklis e Van Roy nel 1997 {cite}`tsitsiklis1997analysis`. Ed
è a quel caso, lineare e
on-policy, che si fermano le garanzie di convergenza note: con approssimatori
non lineari come le reti si conoscono controesempi di divergenza perfino
on-policy.

Sutton e Barto passano poi in rassegna i tre elementi chiedendosi a quale si
possa rinunciare, ed è la lettura più utile per chi progetta. All’**approssimazione**
no: senza, non si scala. Al **bootstrapping** si può, usando Monte Carlo, e si
paga in efficienza computazionale (bisogna conservare tutto fino alla fine
dell'episodio) e in efficienza di dati. All’**off-policy** si può, sostituendo
il Q-learning con Sarsa, e si perde la possibilità di imparare da un archivio
di esperienze altrui, che è però proprio la premessa del replay buffer.

DQN non rinuncia a nessuno dei tre. Fa un'altra cosa: **rende gli ultimi due
meno velenosi**, il che spiega perché i due accorgimenti che seguono siano
esattamente due e non uno o tre.

`````

## Due accorgimenti per non far esplodere l'addestramento

DQN non rinuncia a nessuno dei tre ingredienti. Ne addolcisce due, uno per
accorgimento, e sono gli accorgimenti che seguono.

### Experience replay

`````{tab} Elementare

Un agente che impara sui fotogrammi *nell'ordine in cui li vive* è come uno
studente che rilegge cento volte la stessa pagina di seguito: fotogrammi
consecutivi si somigliano troppo e la rete finisce per "fissarsi". La memoria
di replay è un grande quaderno degli appunti: ogni esperienza vissuta viene
annotata e, per allenarsi, l'agente pesca **a caso** vecchie esperienze dal
quaderno. Così mescola situazioni lontane nel tempo e impara in modo più
equilibrato, e riutilizza ogni esperienza molte volte, non una sola.

`````

`````{tab} Superiore

Ogni transizione $(s, a, r, s')$ viene salvata in un buffer $\mathcal{D}$
(tipicamente un milione di transizioni). L'aggiornamento dei pesi avviene su
*minibatch* campionati uniformemente da $\mathcal{D}$, e non sull'ultima
transizione. Questo rompe la correlazione temporale tra campioni consecutivi
(che violerebbe l'ipotesi di indipendenza della discesa del gradiente
stocastica) e aumenta enormemente l'efficienza nell'uso dei dati, riutilizzando
ogni transizione in molti aggiornamenti.

`````

### Rete-target

`````{tab} Elementare

C'è un secondo problema: la rete deve inseguire un bersaglio che lei stessa
sposta a ogni passo, come cercare di colpire la propria ombra. La soluzione è
tenere **due copie** della rete: una che impara di continuo e una "congelata"
che fornisce il bersaglio e viene aggiornata solo ogni tanto. Il bersaglio
resta fermo abbastanza a lungo perché la rete che apprende riesca a
raggiungerlo.

`````

`````{tab} Superiore

Si mantiene una rete-target con parametri $\theta^{-}$, copia periodica dei
$\theta$ ogni $C$ passi. La rete si allena minimizzando l'errore quadratico
sull'equazione di Bellman:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')\sim U(\mathcal{D})}
\left[\big(\, r + \gamma \max_{a'} Q(s', a'; \theta^{-}) - Q(s, a;
\theta)\,\big)^2\right].
$$

Qui $r$ è la ricompensa immediata, $\gamma\in[0,1)$ il fattore di sconto, e il
termine $r + \gamma \max_{a'} Q(s', a'; \theta^{-})$ è il **bersaglio**,
calcolato con i pesi congelati $\theta^{-}$. Congelarli evita il *feedback*
instabile in cui il bersaglio si muove insieme alla stima.

`````

Sono due modi di rompere lo stesso legame, quello che tiene attaccate fra loro
cose che si susseguono nel tempo. La memoria di replay lo spezza fra
un'esperienza e la successiva, pescando a caso invece che nell'ordine in cui le
cose sono state vissute; la rete-target lo spezza fra la stima e il bersaglio
che la guida, tenendo fermo il secondo mentre la prima si muove
({numref}`fig-dqn-stabilita`).

```{figure} ../figures/dqn-stabilita.svg
:name: fig-dqn-stabilita
:alt: "Animazione: a sinistra un buffer di ventiquattro celle disposte in ordine di arrivo, di cui a ogni passo se ne accendono quattro pescate a caso, sparpagliate nel buffer invece che una di fila all'altra: sono il minibatch. A destra il valore Q di una stessa coppia stato-azione: la curva della rete che impara sale a ogni passo, mentre la scaletta della copia congelata resta ferma per tre passi e poi scatta a raggiungerla."
:width: 95%

I due accorgimenti al lavoro sullo stesso addestramento. A sinistra la memoria
di replay: le esperienze entrano in ordine, e la manciata su cui si studia (il
*minibatch*) le pesca a caso. A destra lo stesso valore $Q$ calcolato dalla rete
che impara, che si muove a ogni passo, e dalla copia congelata, che resta ferma
per un numero fisso di passi (nel disegno tre) e poi scatta a raggiungerla. La
figura è in scala ridotta: in un addestramento vero le esperienze in memoria
sono un milione e quelle pescate a ogni giro sono trentadue.
```

Si vede anche quale dei tre ingredienti ciascun accorgimento addolcisce. La
copia congelata addolcisce il secondo, cioè il correggere una stima guardandone
un'altra: quell'altra adesso sta ferma per un po’ e si fa raggiungere. La
memoria di replay addolcisce il terzo, cioè l'imparare da partite giocate in un
altro modo: pescando a caso da un milione di ricordi l'agente si allena su un
miscuglio largo, invece che sulla manciata di situazioni che sta attraversando
in questo momento. Il primo ingrediente, la rete al posto della tabella, resta
intatto: è quello per cui si è fatto tutto il resto.

La rete, in PyTorch, si costruisce in poche righe. Un paio di numeri prima di
leggerla. I fotogrammi arrivano ridotti a $84\times84$ punti in scala di grigi
e impilati a quattro a quattro, perché da una sola immagine ferma non si capisce
dove stia andando la pallina. Poi ciascuno dei tre strati convoluzionali passa
sull'immagine con una finestrella che avanza a salti, e più lungo è il salto più
piccolo è ciò che restituisce: il primo strato salta di quattro punti alla volta
e riduce $84$ a $20$, il secondo salta di due e porta $20$ a $9$, il terzo salta
di uno e lascia $7$. Alla fine restano $7\times7$ caselle per ciascuno dei $64$
**filtri**, cioè dei rivelatori che quello strato ha imparato (uno reagisce ai
bordi verticali, un altro alla pallina, e così via). Da lì esce il `64 * 7 * 7`
del primo **strato denso**, quello in cui ogni numero in entrata parla con ogni
numero in uscita, senza più finestrelle.

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
```

Il bersaglio si calcola con la rete-target, e quando la partita finisce lì (uno
stato *terminale*: nessun seguito, quindi niente futuro da scontare) si tiene la
sola ricompensa:

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

Il dettaglio storicamente rilevante è cosa vede la rete: nient'altro che
l'immagine. DeepMind impilava quattro fotogrammi consecutivi in scala di
grigi, ridotti a $84\times84$, per dare alla rete un senso del movimento (dove
va la pallina?). Nessuna informazione sulle regole, e nessuna misura scelta e
calcolata a mano da un programmatore (in gergo, nessuna *feature*: niente
«distanza fra pallina e racchetta», niente «numero di mattoni rimasti»). Lo
**stesso** algoritmo, con le **stesse** manopole di regolazione (gli
*iperparametri*: quelli che si decidono prima e non si imparano), fu addestrato
su 49 giochi diversi: raggiunse un livello comparabile a quello di
un tester umano professionista, ottenendo almeno il 75% del suo punteggio in
29 giochi su 49. In *Breakout* scoprì da solo la strategia del "tunnel"
(scavare un varco laterale per far rimbalzare la pallina dietro il muro) che
nessuno gli aveva insegnato. Era la prima volta che un singolo sistema
imparava una gamma così ampia di compiti partendo da input sensoriali grezzi.

## Il difetto che il massimo si porta dietro

Nel bersaglio di DQN c'è un'operazione che sembra innocua e non lo è:
**prendere il valore più alto**. Conviene capire perché gonfia le stime, sia
perché è controintuitivo (prendere il massimo è proprio quello che si vuole
fare), sia perché lo stesso difetto e la stessa cura torneranno, identici,
nella sezione sul controllo continuo.

`````{tab} Elementare

Immagina di dover scegliere fra otto mosse, e di avere per ciascuna un voto
approssimativo: giusto *in media*, ma sporcato ogni volta da un errore in più o
in meno. Tu prendi sempre il voto più alto. Ora, il voto più alto degli otto non
è quasi mai quello della mossa davvero migliore: è quello della mossa a cui
l'errore ha dato la spinta verso l'alto più grande. Fra otto misure sbagliate a
caso, la più alta è quasi sempre una misura fortunata.

Prendere il massimo di stime rumorose, insomma, non restituisce il massimo dei
valori veri: restituisce qualcosa di sistematicamente più grande. Il conto si
può anche fare. Immagina che le otto mosse valgano tutte esattamente $5$, e che
ogni voto sbagli di una quantità qualsiasi fra $-1$ e $+1$, in su come in giù,
senza preferenze. Fra otto errori pescati così, il più grande sta quasi sempre
vicino al bordo alto: in media vale $+0{,}78$, non $0$. Quindi il voto più alto
degli otto, in media, non vale $5$: vale $5{,}78$. (Il conto esatto si fa con un
po’ di probabilità, ma si può anche solo simulare, e viene lo stesso.) E il
guaio è che quel numero gonfiato diventa il bersaglio
dell'aggiornamento successivo, quindi la gonfiatura non resta dov'era: si
tramanda.

Il rimedio si chiama **Double DQN**, e divide in due un lavoro che prima faceva
una rete sola. Prima: la stessa rete decide qual è la mossa migliore *e* dice
quanto vale. Dopo: **la rete che sta imparando dice quale mossa**, e la **copia
congelata dice quanto vale quella mossa lì**. Che i due ruoli stiano su reti
**diverse** è tutta la sostanza: l'errore che ha fatto *sembrare* buona quella
mossa non è lo stesso errore che poi ne *misura* il valore, così la fortuna non
viene contata due volte. E non si assegnano a caso: a dire *quanto vale* deve
essere la copia congelata, altrimenti si perde il bersaglio fermo che serviva a
non far esplodere l'addestramento.

Attenua, però, non guarisce. Le due reti non sono estranee fra loro: una è la
copia dell'altra di qualche passo prima, e quella parentela lascia passare buona
parte della gonfiatura.

`````

`````{tab} Superiore

La causa non è che le singole stime siano distorte, perché non lo sono: è
l'incontro fra il **rumore** e la **convessità del massimo**, che la
disuguaglianza di Jensen mette in conto. Per stime $\hat Q$ non distorte,

$$
\mathbb{E}\big[\max_a \hat Q(s,a)\big] \;\ge\; \max_a \mathbb{E}\big[\hat Q(s,a)\big]
= \max_a Q(s,a),
$$

e il divario cresce con il numero di azioni e con la varianza dell'errore: la
disuguaglianza è **stretta** ogni volta che il rumore può cambiare quale azione
risulti la migliore, e con stime esatte si ridurrebbe a un'uguaglianza. Basta
quindi un errore di stima a media nulla
perché il bersaglio sia sistematicamente gonfio, e il bootstrapping lo propaga
all'indietro.

Il **Double DQN** {cite}`vanhasselt2016deep` disaccoppia i due ruoli. Il
bersaglio di DQN usa $\theta^{-}$ sia per scegliere sia per valutare; quello di
Double DQN fa scegliere alla rete **online** $\theta$ e valutare alla rete
**target** $\theta^{-}$:

$$
y^{\text{Double}} = r + \gamma\, Q\Big(s',\;
\arg\max_{a'} Q(s', a'; \theta);\ \theta^{-}\Big).
$$

Da confrontare con $y = r + \gamma \max_{a'} Q(s',a';\theta^{-})$: la differenza
sta tutta in quali parametri compaiono dentro l’$\arg\max$, ed è una riga sola
di codice. L'ordine dei due ruoli non è però scambiabile a piacere: far scegliere
a $\theta^{-}$ e valutare a $\theta$ conserverebbe il disaccoppiamento, e quindi
una parte della correzione, ma rimetterebbe i pesi in aggiornamento dentro il
bersaglio, buttando via il congelamento che era servito a stabilizzarlo.

Gli autori sono prudenti sul risultato, e conviene esserlo con loro: l'algoritmo
«riduce le sovrastime osservate», non le elimina. Il disaccoppiamento
annullerebbe il bias solo con due stimatori **indipendenti**, e qui il secondo
non lo è: la rete target è una copia ritardata del primo, scelta perché è il
candidato che c'era già, e il paper stesso avverte che il disaccoppiamento non è
completo. Resta inoltre che il correttivo non è neutro: quando i valori veri
delle azioni non sono tutti uguali, lo stimatore doppio tende a sostituire la
sovrastima con una lieve **sottostima**.

`````

## I limiti

Molti confini di questo approccio hanno guidato la ricerca successiva, e
conviene metterli in fila. Oltre alla sovrastima appena vista, ne restano tre.

- **Fame di dati.** Servono decine di milioni di fotogrammi per gioco:
  l'equivalente di settimane di gioco ininterrotto. Un umano impara in pochi
  minuti. DQN è potente ma spaventosamente inefficiente.
- **Le mosse devono essere poche e distinte** (in gergo *discrete*, cioè
  contabili una per una, come le voci di un menu). Prendere il valore più alto
  vuol dire scorrerle tutte: va bene per un joystick a poche direzioni, non per
  uno sterzo o un braccio robotico, dove la mossa è una quantità da dosare e le
  possibilità sono infinite. Da lì nascono gli algoritmi **attore-critico**
  (*actor-critic*), dove uno propone la mossa e l'altro la giudica, che
  incontreremo nelle prossime due sezioni.
- **Ricompense rade.** In certi giochi il punteggio arriva solo dopo lunghe
  sequenze di mosse esatte: in *Montezuma's Revenge*, per esempio, bisogna
  scendere una scala, saltare una fune e schivare un teschio prima di prendere
  la chiave che vale il primo punto. Lì DQN sostanzialmente fallisce: senza
  segnale, non c'è nulla da inseguire.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- **DQN** butta via lo schedario che aveva un cartellino per ogni schermata e
  ci mette una rete neurale: guarda i pixel e dice a colpo d'occhio quanto
  vale ciascuna mossa, anche su schermate mai viste prima.
- Per anni un'idea così divergeva, e per una ragione precisa, la **triade
  fatale**: una rete al posto della tabella, stime aggiornate a partire da
  altre stime, e una strategia imparata mentre se ne gioca un'altra. Due
  qualunque dei tre convivono senza danni, tutti e tre insieme no: i valori
  possono crescere senza fermarsi. Il **controesempio di Baird** lo mostra su
  sette stati in cui non si guadagna mai nulla e la risposta giusta ("tutto
  vale zero") il sistema saprebbe rappresentarla alla perfezione: i numeri
  crescono lo stesso, e non si fermano.
- Due accorgimenti lo rendono stabile: il **quaderno degli appunti**, che si
  chiama **memoria di replay** (ogni esperienza viene annotata e ripescata a
  caso, così l'agente mescola situazioni lontane invece di rileggere cento volte
  la stessa pagina) e la
  **copia congelata** della rete, che tiene fermo il bersaglio abbastanza a
  lungo perché lo si possa raggiungere. Nessuno dei tre ingredienti sparisce: il
  quaderno addolcisce il terzo (imparare da partite giocate in un altro modo) e
  la copia congelata il secondo (correggere una stima guardandone un'altra); il
  primo, la rete al posto della tabella, resta intatto.
- Prendere sempre il **voto più alto** gonfia i voti: fra tante stime sporcate
  da un errore, la più alta è quasi sempre una stima fortunata, non la mossa
  migliore. Il **Double DQN** attenua il difetto facendo dire *quale mossa*
  alla rete che impara e *quanto vale* alla copia congelata; non lo elimina,
  perché le due reti sono parenti strette.
- Il risultato storico del 2015: un solo programma, con le stesse manopole di
  regolazione, arriva al livello di un tester umano professionista su molti
  giochi Atari partendo dai soli pixel. Restano i limiti: servono quantità
  enormi di partite, le mosse devono essere poche e distinte, e dove il
  punteggio arriva di rado l'agente resta senza nulla da inseguire.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- **DQN** sostituisce la tabella $Q$ con una rete neurale $Q(s,a;\theta)$ che
  mappa i pixel dello stato ai valori delle azioni.
- Divergeva per una ragione precisa, la **triade fatale**: approssimazione,
  bootstrapping e off-policy insieme possono far esplodere i valori. Con due
  soli dei tre l'instabilità si può evitare, con tutti e tre no, e il **controesempio di
  Baird** lo mostra su sette stati con ricompense tutte nulle, dove la
  soluzione esatta è rappresentabile e i pesi divergono lo stesso.
- Due accorgimenti lo rendono stabile: l’**experience replay** (memoria di
  transizioni campionate a caso) e la **rete-target** (bersaglio congelato).
  Non rinunciano a nessuno dei tre anelli: ne attenuano due.
- Il $\max$ nel bersaglio **sovrastima** perché il rumore incontra una funzione
  convessa, non perché le stime siano distorte (Jensen; stretta solo se il
  rumore può cambiare quale azione risulta la migliore, non su tutte le stime
  aleatorie):
  $\mathbb{E}[\max_a \hat Q] \ge \max_a \mathbb{E}[\hat Q]$. Il **Double DQN**
  fa scegliere l'azione a $\theta$ e valutarla a $\theta^{-}$: *riduce* il bias,
  non lo annulla, perché i due stimatori non sono indipendenti.
- Il risultato storico (Mnih et al., 2015): livello umano su molti giochi
  Atari partendo dai soli pixel. Restano limiti di efficienza, azioni discrete
  e ricompense rade.
```
`````
