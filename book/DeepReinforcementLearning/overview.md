# Deep Reinforcement Learning

Nel 2013 un laboratorio londinese ancora poco noto, DeepMind, pubblica un
risultato che sembra un giochino e invece è uno spartiacque: una rete neurale
impara a giocare a *Breakout*, il classico dei mattoncini dell'Atari 2600,
guardando **solo i pixel dello schermo** e il punteggio. Nessuno le ha spiegato
le regole, cosa sia la pallina, cosa sia la racchetta. Dopo qualche ora di
allenamento non solo gioca bene: scopre da sola la tattica di scavare un tunnel
sul lato del muro per far rimbalzare la pallina dietro i mattoni. Una strategia
che molti giocatori umani impiegano settimane a inventare.

Nel capitolo precedente abbiamo visto il *reinforcement learning* classico:
un agente, degli stati, delle azioni, delle ricompense, e algoritmi come il
Q-learning che imparano *quanto vale* ogni mossa. Ma quel Q-learning teneva i
suoi conti in una **tabella**: una casella per ogni coppia stato-azione. Ed è
proprio la tabella a rompersi non appena il mondo diventa grande. Il *deep*
reinforcement learning nasce per sostituirla.

## Quando la tabella diventa impossibile

Il Q-learning tabellare funziona benissimo in un labirinto con cento caselle.
Ma proviamo a giocare guardando lo schermo, come farebbe una persona.

`````{tab} Elementare

Immagina di dover scrivere su un quaderno un giudizio ("questa mossa è buona"
oppure "è pessima") per **ogni possibile immagine** che può comparire sullo
schermo. Ma le immagini possibili sono praticamente infinite: basta che la
pallina si sposti di un pixel e la figura è già diversa, e quindi ti servirebbe
una nuova riga sul quaderno. Non ci sono abbastanza quaderni al mondo. La
tabella, che nel labirinto bastava, qui è semplicemente impossibile da
riempire e da conservare.

`````

`````{tab} Superiore

Un fotogramma di gioco, ridotto come nell'esperimento originale a $84\times84$
pixel in $256$ livelli di grigio, ha

$$
|S| = 256^{\,84\times 84} = 256^{7056}
$$

stati possibili: un numero con migliaia di cifre, incommensurabilmente più
grande degli atomi dell'universo osservabile ($\sim 10^{80}$). Una $Q$-table
richiederebbe una cella per ciascuno stato $s$ e azione $a$: né la memoria né i
dati per visitarli tutti esisteranno mai. Il problema non è l'algoritmo, è la
**rappresentazione**: enumerare gli stati non scala.

`````

## L'idea: una rete al posto della tabella

La svolta concettuale è semplice da enunciare. Se non possiamo *elencare* il
valore di ogni stato, proviamo ad **approssimarlo** con una funzione che
*generalizza*: stati simili — schermate simili — dovrebbero ricevere giudizi
simili. E quale strumento sappiamo essere bravissimo a leggere immagini ed
estrarne una risposta? Una rete neurale, in particolare convoluzionale
({numref}`fig-drl-pixel-to-q`).

```{figure} ../figures/drl-pixel-to-q.svg
:name: fig-drl-pixel-to-q
:alt: I pixel dello schermo di gioco entrano in una rete neurale convoluzionale che produce un valore Q per ogni azione possibile; l'azione col valore più alto viene scelta.
:width: 90%

Dal pixel alla decisione. La rete riceve lo schermo grezzo e restituisce, in
un colpo solo, il valore stimato $Q(s,a)$ di ogni mossa disponibile: si sceglie
quella col valore più alto.
```

`````{tab} Elementare

Invece del quaderno con una riga per ogni immagine, addestriamo un "occhio
esperto" che guarda lo schermo e, sul momento, dà un voto a ciascuna mossa
possibile ("vai a sinistra: 3", "resta fermo: 1", "vai a destra: 8"). Poi si
sceglie la mossa col voto più alto. Il bello è che questo occhio, avendo visto
tante partite, sa dare un voto sensato anche a una schermata **mai vista
prima**, perché assomiglia ad altre che conosce. È la differenza fra imparare a
memoria e capire.

`````

`````{tab} Superiore

Si parametrizza la funzione valore con una rete di pesi $\theta$, scrivendo
$Q(s,a;\theta)$, e la si allena minimizzando lo scarto dal *target* di Bellman:

$$
\mathcal{L}(\theta) = \mathbb{E}\Big[\big(r + \gamma \max_{a'} Q(s',a';\theta^{-})
- Q(s,a;\theta)\big)^2\Big] ,
$$

dove $r$ è la ricompensa ottenuta, $\gamma \in [0,1)$ il fattore di sconto,
$s'$ lo stato successivo e $\theta^{-}$ i pesi di una copia "congelata" della
rete. È l'algoritmo **Deep Q-Network (DQN)**. In alternativa si può
parametrizzare direttamente la *policy* $\pi_\theta(a\mid s)$ e ottimizzarla per
salita del gradiente sulla ricompensa attesa (*policy gradient*): due famiglie
che il capitolo affronterà entrambe.

`````

## La svolta: da Atari a AlphaGo

Quel primo lavoro — *Playing Atari with Deep Reinforcement Learning* (Mnih e
colleghi, 2013) — diventa nel 2015 un articolo su *Nature*, *Human-level
control through deep reinforcement learning*: **un'unica architettura**, senza
ritocchi specifici per gioco, raggiunge o supera il livello di un giocatore
umano professionista in molti dei 49 titoli Atari testati. È la prova che pixel
grezzi e ricompensa scarna bastano.

L'anno dopo arriva il colpo che raggiunge il grande pubblico: **AlphaGo**
(Silver e colleghi, *Nature* 2016) batte per 4 a 1 il campione Lee Sedol nel
Go, un gioco con più configurazioni che atomi nell'universo, a lungo
considerato fuori portata per le macchine. Deep reinforcement learning e
ricerca ad albero, insieme. Il deep RL smette di essere una curiosità da
laboratorio.

## Il prezzo: instabilità e campioni costosi

L'entusiasmo non deve nascondere il conto da pagare. Il deep RL è notoriamente
capriccioso.

`````{tab} Elementare

Due difficoltà su tutte. La prima: l'allenamento è **instabile**, come inseguire
il proprio riflesso in uno specchio che si muove ogni volta che ci si avvicina;
piccole modifiche possono far crollare tutto. La seconda: serve **una quantità
enorme di partite**. L'agente impara per tentativi, e di tentativi ne vuole
milioni — settimane di gioco. In un videogioco simulato va bene; con un robot
vero che si può rompere, molto meno.

`````

`````{tab} Superiore

I campioni sono **fortemente correlati** (fotogrammi consecutivi) e il *target*
$r + \gamma \max_{a'} Q(s',a';\theta)$ **si muove** insieme ai pesi che stiamo
aggiornando: la combinazione di approssimazione, bootstrapping e
apprendimento off-policy è la celebre *deadly triad* che può divergere. DQN la
addomestica con due trucchi: l'**experience replay** (campionare a caso da un
buffer di transizioni passate, decorrelandole) e la **rete target** $\theta^{-}$
aggiornata di rado, che stabilizza il bersaglio. Resta il costo campionario:
la versione di *Nature* usava circa $50$ milioni di fotogrammi per titolo. La
*sample efficiency* è tuttora un problema di ricerca aperto.

`````

## Come è organizzato il capitolo

Partiremo dal **DQN** e dai suoi ingredienti stabilizzanti — experience replay e
rete target — costruendo un agente che gioca a un ambiente Atari con
`gymnasium` e PyTorch. Vedremo poi i miglioramenti più influenti (Double DQN,
Dueling, replay prioritizzato). Cambieremo quindi famiglia con i metodi a
**gradiente di policy** e gli approcci **actor-critic** (A2C, A3C, PPO). Da lì
apriremo la cassetta degli attrezzi del deep RL moderno: il **controllo
continuo** per la robotica (DDPG, TD3, SAC), l'apprendimento **basato su
modello** che impara a pianificare (da Dyna a MuZero e Dreamer), l'**offline
RL** che impara da dati già raccolti senza mai interagire (fino al Decision
Transformer), e il nodo dell'**esplorazione** con ricompense sparse — con
l'insidia del *reward hacking*, che ci accompagnerà fino al capitolo sull'AI
responsabile. Chiuderemo tornando al filo che unisce tutto: la ricerca dietro
ad AlphaGo e ai suoi successori. L'obiettivo non è collezionare acronimi, ma
capire *perché* ciascun pezzo esiste — quale fragilità dell'idea precedente è
venuto a curare.
