# Deep Q-Network (DQN)

Nel 2013 un piccolo gruppo di ricercatori londinesi di una startup chiamata
DeepMind mostrò un video destinato a diventare storico: un unico programma che
imparava a giocare a diversi videogiochi Atari — *Breakout*, *Pong*, *Space
Invaders* — senza che nessuno gli avesse spiegato le regole. L'algoritmo
riceveva solo ciò che vedrebbe un ragazzino davanti al cabinato: i pixel dello
schermo e il punteggio. Da lì, per tentativi, arrivava a giocare meglio di un
umano esperto {cite}`mnih2013playing`. Due anni dopo il risultato finì sulla
copertina di *Nature* {cite}`mnih2015human`.
Quel programma si chiama **Deep Q-Network**, DQN.

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
ogni mossa — anche per schermate che non ha mai visto prima, perché ha imparato
a riconoscere le somiglianze. Quell'esperto è la rete neurale.

`````

`````{tab} Superiore

Approssimiamo la funzione azione-valore ottima con una rete parametrizzata da
$\theta$:

$$
Q(s, a; \theta) \approx Q^{*}(s, a).
$$

La rete prende in ingresso lo stato $s$ (i pixel) e restituisce in uscita un
vettore con un valore $Q$ per **ciascuna** azione ammissibile — non serve una
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

## Due accorgimenti per non far esplodere l'addestramento

Sostituire la tabella con una rete sembra ovvio, ma per anni non aveva
funzionato: l'addestramento divergeva. Il merito di DQN è aver introdotto due
trucchi che rendono stabile ciò che prima era instabile.

### Experience replay

`````{tab} Elementare

Un agente che impara sui fotogrammi *nell'ordine in cui li vive* è come uno
studente che rilegge cento volte la stessa pagina di seguito: fotogrammi
consecutivi si somigliano troppo e la rete finisce per "fissarsi". La memoria
di replay è un grande quaderno degli appunti: ogni esperienza vissuta viene
annotata e, per allenarsi, l'agente pesca **a caso** vecchie esperienze dal
quaderno. Così mescola situazioni lontane nel tempo e impara in modo più
equilibrato — e riutilizza ogni esperienza molte volte, non una sola.

`````

`````{tab} Superiore

Ogni transizione $(s, a, r, s')$ viene salvata in un buffer $D$ (tipicamente
un milione di transizioni). L'aggiornamento dei pesi avviene su *minibatch*
campionati uniformemente da $D$, e non sull'ultima transizione. Questo rompe
la correlazione temporale tra campioni consecutivi — che violerebbe l'ipotesi
di indipendenza della discesa del gradiente stocastica — e aumenta enormemente
l'efficienza nell'uso dei dati, riutilizzando ogni transizione in molti
aggiornamenti.

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
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')\sim U(D)}
\left[\big(\, r + \gamma \max_{a'} Q(s', a'; \theta^{-}) - Q(s, a; \theta)\,\big)^2\right].
$$

Qui $r$ è la ricompensa immediata, $\gamma\in[0,1)$ il fattore di sconto, e il
termine $r + \gamma \max_{a'} Q(s', a'; \theta^{-})$ è il **bersaglio**,
calcolato con i pesi congelati $\theta^{-}$. Congelarli evita il *feedback*
instabile in cui il bersaglio si muove insieme alla stima.

`````

La rete è una CNN classica; in PyTorch la si costruisce in poche righe.

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
l'immagine. DeepMind impilava quattro fotogrammi consecutivi in scala di grigi,
ridotti a $84\times84$, per dare alla rete un senso del movimento (dove va la
pallina?). Nessuna informazione sulle regole, nessuna feature costruita a mano.
Lo **stesso** algoritmo, con gli **stessi** iperparametri, fu addestrato su 49
giochi diversi: raggiunse un livello comparabile a quello di un tester umano
professionista, ottenendo almeno il 75% del suo punteggio in 29 giochi su 49.
In *Breakout* scoprì da solo la strategia del "tunnel" — scavare un varco
laterale per far rimbalzare la pallina dietro il muro — che nessuno gli aveva
insegnato. Era la prima volta che un singolo sistema imparava una gamma così
ampia di compiti partendo da input sensoriali grezzi.

## I limiti

L'entusiasmo non deve nascondere i confini dell'approccio, molti dei quali
hanno guidato la ricerca successiva.

- **Sovrastima dei valori.** L'operatore $\max$ nel bersaglio tende a
  gonfiare sistematicamente le stime di $Q$. Il *Double DQN*
  {cite}`vanhasselt2016deep` corregge il difetto separando chi *sceglie*
  l'azione da chi ne *valuta* il valore.
- **Fame di dati.** Servono decine di milioni di fotogrammi per gioco:
  l'equivalente di settimane di gioco ininterrotto. Un umano impara in pochi
  minuti. DQN è potente ma spaventosamente inefficiente.
- **Solo azioni discrete.** Il $\max$ richiede di enumerare le azioni: va bene
  per un joystick a poche direzioni, non per controllare uno sterzo o un
  braccio robotico continui — da cui gli algoritmi *actor-critic* che vedremo.
- **Ricompense rade.** Dove il punteggio arriva solo dopo lunghe sequenze
  (il famigerato *Montezuma's Revenge*), DQN sostanzialmente fallisce: senza
  segnale, non c'è nulla da inseguire.

```{admonition} Da ricordare
:class: important
- **DQN** sostituisce la tabella $Q$ con una rete neurale $Q(s,a;\theta)$ che
  mappa i pixel dello stato ai valori delle azioni.
- Due accorgimenti lo rendono stabile: l'**experience replay** (memoria di
  transizioni campionate a caso) e la **rete-target** (bersaglio congelato).
- Il risultato storico (Mnih et al., 2015): livello umano su molti giochi
  Atari partendo dai soli pixel. Restano limiti di efficienza, azioni discrete
  e ricompense rade.
```
