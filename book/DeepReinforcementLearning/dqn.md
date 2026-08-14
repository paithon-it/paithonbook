# Deep Q-Network (DQN)

Torniamo al video del 2013 da cui si è aperto il capitolo, stavolta per
guardarci dentro. Quel piccolo gruppo di ricercatori londinesi di una startup
chiamata DeepMind aveva mostrato un unico programma che imparava a giocare a
diversi videogiochi Atari (*Breakout*, *Pong*, *Space
Invaders*) senza che nessuno gli avesse spiegato le regole. L'algoritmo
riceveva solo ciò che vedrebbe un ragazzino davanti al cabinato: i pixel dello
schermo e il punteggio. Da lì, per tentativi, su alcuni di quei giochi
arrivava a superare un umano esperto {cite}`mnih2013playing`. Due anni dopo il risultato finì sulla
copertina di *Nature* {cite}`mnih2015human`. Quel programma si chiama **Deep
Q-Network**, DQN.

Nel capitolo precedente abbiamo incontrato il *Q-learning*: un agente impara una
funzione $Q(s,a)$ che stima quanto è conveniente, nel lungo periodo, compiere
l'azione $a$ trovandosi nello stato $s$. Lì la $Q$ viveva in una tabella: una
riga per ogni stato, una colonna per ogni azione. Funziona con pochi stati.
Ma quanti stati ha una schermata Atari? Uno schermo $210\times160$ a colori ha
più configurazioni possibili che atomi nell'universo osservabile. La tabella
non basta più.

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
:alt: Un fotogramma di gioco stile Breakout entra in una rete convoluzionale con volumi decrescenti, seguita da uno strato denso, e produce una barra di valore Q per ciascuna delle quattro azioni possibili.
:width: 95%

Lo schema di DQN. Il fotogramma di gioco attraversa gli strati convoluzionali
e uno strato denso; l'uscita è un valore $Q$ per ogni azione. L'agente sceglie
l'azione con il valore più alto.
```

## Perché divergeva: la triade fatale

Mettere una rete al posto della tabella non era, di per sé, un'idea nuova:
**TD-Gammon** lo faceva dal 1992, con una rete addestrata a suon di partite di
backgammon; nella versione descritta tre anni dopo giocava quasi come i più forti
campioni del mondo {cite}`tesauro1995temporal`. Ma TD-Gammon imparava **sulla
strategia che stava giocando**, e quel dettaglio, come stiamo per vedere, cambia
tutto. Il Q-learning con una rete, che impara una strategia mentre ne gioca
un'altra, per anni invece **divergeva**: i valori stimati crescevano senza
fermarsi, invece di assestarsi. E non divergeva per sfortuna, né perché qualcuno
avesse sbagliato a tarare il passo di apprendimento (il *learning rate*: di
quanto si sposta la rete a ogni correzione). Prima dei due trucchi che lo hanno
reso praticabile vale la pena capire da che cosa lo hanno salvato, perché è un
risultato preciso e sorprendentemente pulito.

`````{tab} Elementare

Ci sono tre ingredienti in gioco, ognuno dei quali, preso da solo, è
innocuo e anzi utile.

Il primo è l'**approssimazione**: una rete al posto della tabella, cioè
sacrificare la precisione su ogni singolo stato in cambio della capacità di
generalizzare. Il secondo è il **bootstrapping**: aggiornare una stima usando
un'altra stima invece di aspettare la fine della partita, ed è la mossa che
distingue le differenze temporali (il **TD** del capitolo precedente, che
aggiornano subito) dai metodi Monte Carlo (che aspettano il fischio finale e
solo allora tirano le somme). Il terzo è l'**off-policy**: imparare la strategia
migliore mentre se ne gioca un'altra, esplorativa, ed è quello che rende il
Q-learning così comodo.

La sintesi, che si deve a Sutton e Barto, è che con due qualunque di questi tre
l'instabilità si può evitare. **Tutti e tre insieme no**: la
combinazione può divergere, cioè i valori possono crescere senza limite invece
di assestarsi. La chiamano **triade fatale**, e la parte inquietante è che non
serve nemmeno un ambiente sconosciuto o rumoroso: si dimostra su un esempio
con sette stati, il **controesempio di Baird**, dove tutte le ricompense
valgono zero e la risposta giusta è «tutto vale zero». Quella risposta il
sistema saprebbe rappresentarla alla perfezione (e non serve nemmeno una rete
profonda: bastano pochi pesi messi in fila), e ciononostante quei pesi, invece
di posarsi sulla risposta giusta, cominciano a crescere e non smettono più.

Perché succede, in una frase. Quando la rete corregge il proprio giudizio su
uno stato, la correzione si allarga da sé a tutti gli stati che gli somigliano:
è il prezzo della generalizzazione, e in condizioni normali è un vantaggio. Ma
il bersaglio verso cui la correzione punta è a sua volta il giudizio su uno
stato vicino, cioè uno di quelli che la correzione ha appena spostato. Ogni
ritocco muove il bersaglio che serviva a deciderlo, e il ritocco successivo
parte da un bersaglio già mosso.

L'off-policy toglie l'ultima protezione, e conviene vedere quale. Se l'agente si
allenasse sulle situazioni che incontra davvero giocando la strategia che sta
imparando, ogni correzione peserebbe quanto quella situazione conta per lui, e
il rimpallo si smorzerebbe da sé: è un risultato che si dimostra. Ma imparando
da un archivio di partite giocate in un altro modo, l'agente vede certe
situazioni molto più spesso di quanto le incontrerebbe e altre quasi mai: si
corregge con forza dove non gli serve, e quel che ne esce può crescere invece di
posarsi.

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
dimostrarono Tsitsiklis e Van Roy nel 1997 {cite}`tsitsiklis1997analysis`. Ed è a quel caso, lineare e
on-policy, che si fermano le garanzie di convergenza note: con approssimatori
non lineari come le reti si conoscono controesempi di divergenza perfino
on-policy.

Sutton e Barto passano poi in rassegna i tre elementi chiedendosi a quale si
possa rinunciare, ed è la lettura più utile per chi progetta. All'**approssimazione**
no: senza, non si scala. Al **bootstrapping** si può, usando Monte Carlo, e si
paga in efficienza computazionale (bisogna conservare tutto fino alla fine
dell'episodio) e in efficienza di dati. All'**off-policy** si può, sostituendo
il Q-learning con Sarsa, e si perde la possibilità di imparare da un archivio
di esperienze altrui, che è però proprio la premessa del replay buffer.

DQN non rinuncia a nessuno dei tre. Fa un'altra cosa: **rende gli ultimi due
meno velenosi**, il che spiega perché i due accorgimenti che seguono siano
esattamente due e non uno o tre.

`````

## Due accorgimenti per non far esplodere l'addestramento

I trucchi di DQN si leggono allora come attacchi mirati a due degli anelli.

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
\left[\big(\, r + \gamma \max_{a'} Q(s', a'; \theta^{-}) - Q(s, a; \theta)\,\big)^2\right].
$$

Qui $r$ è la ricompensa immediata, $\gamma\in[0,1)$ il fattore di sconto, e il
termine $r + \gamma \max_{a'} Q(s', a'; \theta^{-})$ è il **bersaglio**,
calcolato con i pesi congelati $\theta^{-}$. Congelarli evita il *feedback*
instabile in cui il bersaglio si muove insieme alla stima.

`````

Sono due modi di rompere la stessa cosa, una correlazione nel tempo: la memoria
di replay la spezza fra un'esperienza e la successiva, pescando a caso invece
che nell'ordine in cui le cose sono state vissute; la rete-target la spezza fra
la stima e il bersaglio che la guida, tenendo fermo il secondo mentre la prima
si muove ({numref}`fig-dqn-stabilita`).

```{figure} ../figures/dqn-stabilita.svg
:name: fig-dqn-stabilita
:alt: "Animazione: a sinistra un buffer di ventiquattro celle disposte in ordine di arrivo, di cui a ogni passo se ne accendono quattro pescate a caso, sparpagliate nel buffer invece che una di fila all'altra: sono il minibatch. A destra il valore Q di una stessa coppia stato-azione: la curva della rete che impara sale a ogni passo, mentre la scaletta della copia congelata resta ferma per tre passi e poi scatta a raggiungerla."
:width: 95%

I due accorgimenti al lavoro sullo stesso addestramento. A sinistra la memoria
di replay: le esperienze entrano in ordine, e la manciata su cui si studia (il
*minibatch*) le pesca a caso. A destra lo stesso valore $Q$ calcolato dalla rete
che impara, che si muove a ogni passo, e dalla copia congelata, che resta ferma
per un numero fisso di passi (nel disegno tre) e poi scatta a raggiungerla. La
figura è in scala ridotta: nel testo le celle sono un milione, e le quattro
pescate a caso a ogni giro sono 32.
```

È una rete convoluzionale classica; in PyTorch la si costruisce in poche
righe. Un paio di numeri, prima di leggerla: i fotogrammi arrivano ridotti a
$84\times84$ punti in scala di grigi e impilati a quattro a quattro, perché da
una sola immagine ferma non si capisce dove stia andando la pallina. I tre
strati convoluzionali rimpiccioliscono l'immagine a ogni passaggio, e di quegli
$84\times84$ punti restano alla fine $7\times7$ caselle per ciascuno dei $64$
filtri: è da lì che esce il `64 * 7 * 7` dell'ultima riga.

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

Il bersaglio si calcola con la rete-target e si azzera negli stati terminali:

```{code-block} python
:class: pt-non-eseguibile

import torch

# minibatch pescato a caso dalla memoria di replay
s, a, r, s_next, fine = replay.campiona(batch=32)

with torch.no_grad():                              # il bersaglio non si deriva
    q_next = target_net(s_next).max(dim=1).values  # max_a' Q(s', a'; theta^-)
bersaglio = r + gamma * q_next * (1 - fine)        # se terminale, resta solo r
```

## Atari: giocare partendo dai pixel

Il dettaglio storicamente rilevante è cosa vede la rete: nient'altro che
l'immagine. DeepMind impilava quattro fotogrammi consecutivi in scala di
grigi, ridotti a $84\times84$, per dare alla rete un senso del movimento (dove
va la pallina?). Nessuna informazione sulle regole, nessuna feature costruita
a mano. Lo **stesso** algoritmo, con gli **stessi** iperparametri, fu
addestrato su 49 giochi diversi: raggiunse un livello comparabile a quello di
un tester umano professionista, ottenendo almeno il 75% del suo punteggio in
29 giochi su 49. In *Breakout* scoprì da solo la strategia del "tunnel"
(scavare un varco laterale per far rimbalzare la pallina dietro il muro) che
nessuno gli aveva insegnato. Era la prima volta che un singolo sistema
imparava una gamma così ampia di compiti partendo da input sensoriali grezzi.

## Il difetto che il massimo si porta dietro

Nel bersaglio di DQN c'è un'operazione che sembra innocua e non lo è: **prendere
il valore più alto**. Vale la pena capire perché gonfia le stime, sia perché è
controintuitivo (prendere il massimo è proprio quello che si vuole fare), sia
perché lo stesso difetto e la stessa cura torneranno, identici, nella sezione
sul controllo continuo.

`````{tab} Elementare

Immagina di dover scegliere fra otto mosse, e di avere per ciascuna un voto
approssimativo: giusto *in media*, ma sporcato ogni volta da un errore in più o
in meno. Tu prendi sempre il voto più alto. Ora, il voto più alto degli otto non
è quasi mai quello della mossa davvero migliore: è quello della mossa a cui
l'errore ha dato la spinta verso l'alto più grande. Fra otto misure sbagliate a
caso, la più alta è quasi sempre una misura fortunata.

Prendere il massimo di stime rumorose, insomma, non restituisce il massimo dei
valori veri: restituisce qualcosa di sistematicamente più grande. E il guaio è
che quel numero gonfiato diventa il bersaglio dell'aggiornamento successivo,
quindi la gonfiatura non resta dov'era: si tramanda.

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
disuguaglianza è **stretta** solo perché $\hat Q$ è aleatoria, e con stime esatte
si ridurrebbe a un'uguaglianza. Basta quindi un errore di stima a media nulla
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
sta tutta in quali parametri compaiono dentro l'$\arg\max$, ed è una riga sola
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

L'entusiasmo non deve nascondere i confini dell'approccio, molti dei quali
hanno guidato la ricerca successiva. Oltre alla sovrastima appena vista, ne
restano tre.

- **Fame di dati.** Servono decine di milioni di fotogrammi per gioco:
  l'equivalente di settimane di gioco ininterrotto. Un umano impara in pochi
  minuti. DQN è potente ma spaventosamente inefficiente.
- **Solo azioni discrete.** Prendere il valore più alto richiede di enumerare
  le azioni una per una: va bene per un joystick a poche direzioni, non per
  controllare uno sterzo o un braccio robotico continui, dove le mosse
  possibili sono infinite. Da lì nascono gli algoritmi **attore-critico**
  (*actor-critic*), dove uno propone la mossa e l'altro la giudica, che
  incontreremo nelle prossime due sezioni.
- **Ricompense rade.** Dove il punteggio arriva solo dopo lunghe sequenze
  (il famigerato *Montezuma's Revenge*), DQN sostanzialmente fallisce: senza
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
- Due accorgimenti lo rendono stabile: il **quaderno degli appunti** (ogni
  esperienza viene annotata e ripescata a caso, così l'agente mescola
  situazioni lontane invece di rileggere cento volte la stessa pagina) e la
  **copia congelata** della rete, che tiene fermo il bersaglio abbastanza a
  lungo perché lo si possa raggiungere. Nessuno dei tre ingredienti sparisce:
  due vengono addolciti.
- Prendere sempre il **voto più alto** gonfia i voti: fra tante stime sporcate
  da un errore, la più alta è quasi sempre una stima fortunata, non la mossa
  migliore. Il **Double DQN** attenua il difetto facendo dire *quale mossa*
  alla rete che impara e *quanto vale* alla copia congelata; non lo elimina,
  perché le due reti sono parenti strette.
- Il risultato storico del 2015: un solo programma, con gli stessi
  settaggi, arriva al livello di un tester umano professionista su molti
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
- Due accorgimenti lo rendono stabile: l'**experience replay** (memoria di
  transizioni campionate a caso) e la **rete-target** (bersaglio congelato).
  Non rinunciano a nessuno dei tre anelli: ne attenuano due.
- Il $\max$ nel bersaglio **sovrastima** perché il rumore incontra una funzione
  convessa, non perché le stime siano distorte (Jensen, stretta solo su stime
  aleatorie):
  $\mathbb{E}[\max_a \hat Q] \ge \max_a \mathbb{E}[\hat Q]$. Il **Double DQN**
  fa scegliere l'azione a $\theta$ e valutarla a $\theta^{-}$: *riduce* il bias,
  non lo annulla, perché i due stimatori non sono indipendenti.
- Il risultato storico (Mnih et al., 2015): livello umano su molti giochi
  Atari partendo dai soli pixel. Restano limiti di efficienza, azioni discrete
  e ricompense rade.
```
`````
